// Copyright 2026 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A correlated COUNT(*) over a file table counts its index candidates from
//! the hot store's versions while the table has no volumes, and stays exact
//! once it has some

#![cfg(feature = "test-failpoints")]

use std::path::Path;

use stoolap::core::Value;
use stoolap::storage::traits::{CappedEqual, Engine, ProbeScratch};
use stoolap::test_failpoints as fp;
use stoolap::Database;

const ROW_READ: &str = "seg.fetch_rows_by_ids_into";

fn open(dir: &Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    ))
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1), (2)", ()).unwrap();
    db.execute(
        "CREATE TABLE o (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX o_user ON o(user_id)", ()).unwrap();
    db.execute(
        "INSERT INTO o SELECT value, CASE WHEN value <= 300 THEN 1 ELSE 2 END FROM generate_series(1, 400)",
        (),
    )
    .unwrap();
    db
}

/// Each user's order count, and the read entries the query ran
fn counts(db: &Database) -> (Vec<i64>, Vec<&'static str>) {
    fp::take_read_paths();
    let counts = db
        .query(
            "SELECT u.id, (SELECT COUNT(*) FROM o WHERE o.user_id = u.id) FROM u ORDER BY u.id",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(1).unwrap())
        .collect();
    (counts, fp::take_read_paths())
}

#[test]
fn a_correlated_count_over_a_table_without_volumes_reads_no_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    assert!(db.engine().volume_stats().is_empty());
    let (counts, paths) = counts(&db);
    assert_eq!(counts, [300, 100]);
    assert!(!paths.contains(&ROW_READ), "rows read to count: {paths:?}");
}

#[test]
fn a_correlated_count_without_volumes_leaves_out_a_deleted_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("DELETE FROM o WHERE id = 5", ()).unwrap();
    db.execute("UPDATE o SET user_id = 2 WHERE id = 6", ())
        .unwrap();
    assert_eq!(counts(&db).0, [298, 101]);
}

#[test]
fn candidates_from_volumes_are_counted_with_the_hot_ones() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "INSERT INTO o SELECT value, value % 1000 + 10 FROM generate_series(1001, 6000)",
        (),
    )
    .unwrap();
    db.execute("UPDATE o SET user_id = 3 WHERE id > 10 AND id <= 400", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("DELETE FROM o WHERE id = 5", ()).unwrap();
    db.execute("INSERT INTO o VALUES (401, 1)", ()).unwrap();
    assert!(!db.engine().volume_stats().is_empty());
    let tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("o").unwrap();
    let epoch = table.index_view_epoch().expect("a quiet index view");
    let key = Value::Integer(1);
    let mut ids = Vec::new();
    let found = table
        .equality_candidates(
            "user_id",
            &key,
            usize::MAX,
            &mut ids,
            &mut ProbeScratch::default(),
        )
        .unwrap();
    assert!(matches!(found, Some(CappedEqual::Copied)), "{found:?}");
    let counted = table
        .count_equal_candidates(&ids, 1, &key, usize::MAX, Some(epoch))
        .unwrap();
    assert_eq!(counted, 10, "nine sealed rows and one hot");
}

#[test]
fn a_correlated_count_inside_a_transaction_sees_its_own_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO o VALUES (401, 1)", ()).unwrap();
    tx.execute("DELETE FROM o WHERE id = 400", ()).unwrap();
    let inside: Vec<i64> = tx
        .query(
            "SELECT u.id, (SELECT COUNT(*) FROM o WHERE o.user_id = u.id) FROM u ORDER BY u.id",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(1).unwrap())
        .collect();
    assert_eq!(inside, [301, 99]);
    tx.rollback().unwrap();
    assert_eq!(counts(&db).0, [300, 100]);
}
