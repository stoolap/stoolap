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

//! A primary key the transaction deleted from a volume may be inserted
//! again, and the row's other unique keys are still checked

use std::path::Path;

use stoolap::core::Error;
use stoolap::Database;

/// `t(id, k UNIQUE, v)` with (1, 10) and (2, 20) sealed
fn sealed(dir: &Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 0), (2, 20, 0)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}

#[test]
fn a_reinserted_key_is_checked_against_the_other_unique_keys() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let mut txn = db.begin().unwrap();
    txn.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    let inserted = txn.execute("INSERT INTO t VALUES (1, 20, 0)", ());
    assert!(
        matches!(inserted, Err(Error::UniqueConstraint { row_id: 2, .. })),
        "{inserted:?}"
    );
    txn.rollback().unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE k = 20"), 1);
}

#[test]
fn a_reinserted_key_in_a_batch_is_checked_against_the_other_unique_keys() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let mut txn = db.begin().unwrap();
    txn.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    let inserted = txn.execute("INSERT INTO t VALUES (5, 50, 0), (1, 20, 0)", ());
    assert!(
        matches!(inserted, Err(Error::UniqueConstraint { row_id: 2, .. })),
        "{inserted:?}"
    );
    txn.rollback().unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE k = 20"), 1);
}

#[test]
fn a_deleted_key_is_inserted_again_with_a_free_unique_value() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let mut txn = db.begin().unwrap();
    txn.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    txn.execute("INSERT INTO t VALUES (1, 99, 0)", ()).unwrap();
    txn.commit().unwrap();
    assert_eq!(count(&db, "SELECT k FROM t WHERE id = 1"), 99);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 2);
}
