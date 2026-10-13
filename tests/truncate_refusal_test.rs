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

//! A TRUNCATE that another transaction's claim refuses removes nothing,
//! hot or sealed, before or after a restart

use std::path::Path;

use stoolap::Database;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

fn counts(db: &Database) -> (i64, i64) {
    let counted: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    let scanned: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE v >= 0", ())
        .unwrap();
    (counted, scanned)
}

/// Twenty rows in two volumes and three hot ones; another transaction holds
/// a claim on row `claimed` while a TRUNCATE runs
fn refused_truncate(dir: &Path, claimed: i64) {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for volume in 0..2 {
        let values: Vec<String> = (volume * 10 + 1..=volume * 10 + 10)
            .map(|id| format!("({id}, {id})"))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("INSERT INTO t VALUES (21, 21), (22, 22), (23, 23)", ())
        .unwrap();

    let mut claiming = db.begin().unwrap();
    claiming
        .execute(&format!("UPDATE t SET v = 99 WHERE id = {claimed}"), ())
        .unwrap();
    assert!(
        db.execute("TRUNCATE TABLE t", ()).is_err(),
        "the claim refuses the truncate"
    );
    claiming.rollback().unwrap();
    assert_eq!(counts(&db), (23, 23), "the refused truncate removed rows");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();

    let reopened = Database::open(&dsn(dir)).unwrap();
    assert_eq!(counts(&reopened), (23, 23), "rows lost across a restart");
}

#[test]
fn a_table_that_cannot_check_first_runs_no_step_before_refusing() {
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut claiming = db.begin().unwrap();
    claiming
        .execute("UPDATE t SET v = 9 WHERE id = 1", ())
        .unwrap();

    let mut txn = db.engine().begin_transaction().unwrap();
    let mut table = txn.get_table("t").unwrap();
    let mut steps = 0;
    let refused = table.truncate_after_check(&mut || steps += 1);
    drop(table);
    txn.rollback().unwrap();
    claiming.rollback().unwrap();
    assert!(refused.is_err());
    assert_eq!(steps, 0, "the step ran before the refusal");
    assert_eq!(counts(&db), (2, 2));
}

#[test]
fn a_claim_on_a_hot_row_refuses_the_truncate_and_keeps_every_row() {
    let dir = tempfile::tempdir().unwrap();
    refused_truncate(dir.path(), 21);
}

#[test]
fn a_claim_on_a_sealed_row_refuses_the_truncate_and_keeps_every_row() {
    let dir = tempfile::tempdir().unwrap();
    refused_truncate(dir.path(), 5);
}
