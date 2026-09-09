// Copyright 2025 Stoolap Contributors
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

use std::sync::Arc;

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::mvcc::{
    get_fast_timestamp, MVCCTable, TransactionVersionStore, VersionStore,
};
use stoolap::storage::traits::Table;
use stoolap::storage::volume::manifest::SegmentManager;
use stoolap::storage::volume::table::SegmentedTable;
use stoolap::Database;

fn values(db: &Database) -> Vec<i64> {
    db.query("SELECT v FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn failed_insert_preserves_prior_statement_and_savepoint() {
    let db =
        Database::open("memory://failed_insert_preserves_prior_statement_and_savepoint").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v > 0))",
        (),
    )
    .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("SAVEPOINT saved", ()).unwrap();
    assert!(db
        .execute("INSERT INTO t VALUES (2, 20), (3, -1)", ())
        .is_err());
    assert_eq!(values(&db), [10]);
    db.execute("INSERT INTO t VALUES (4, 40)", ()).unwrap();
    db.execute("ROLLBACK TO saved", ()).unwrap();
    assert_eq!(values(&db), [10]);
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(values(&db), [10]);
}

#[test]
fn failed_prepared_insert_preserves_transaction_overlay() {
    let db =
        Database::open("memory://failed_prepared_insert_preserves_transaction_overlay").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v > 0))",
        (),
    )
    .unwrap();
    let insert = db
        .prepare("INSERT INTO t VALUES ($1, $2), ($3, $4)")
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    for first_id in [2, 4] {
        assert!(insert.execute((first_id, 20, first_id + 1, -1)).is_err());
        assert_eq!(values(&db), [10]);
    }
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(values(&db), [10]);
}

#[test]
fn failed_insert_select_rolls_back_destination_writes() {
    let db = Database::open("memory://failed_insert_select_rolls_back_destination_writes").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v > 0))",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO source VALUES (2, 20), (3, -1)", ())
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    assert!(db
        .execute("INSERT INTO t SELECT * FROM source ORDER BY id", ())
        .is_err());
    assert_eq!(values(&db), [10]);
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(values(&db), [10]);
}

#[test]
fn failed_cold_update_releases_claim_without_a_local_version() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}/claims", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    assert!(db.execute("UPDATE t SET v = 20 WHERE id = 1", ()).is_err());
    let other = db.clone();
    assert_eq!(
        other
            .execute("UPDATE t SET v = 11 WHERE id = 1", ())
            .unwrap(),
        1
    );
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(values(&db), [11, 20]);
}

#[test]
fn repeated_pending_tombstone_keeps_its_original_boundary() {
    let manager = SegmentManager::new(
        "repeated_pending_tombstone_keeps_its_original_boundary",
        None,
    );
    manager.add_pending_tombstone(1, 1);
    let boundary = get_fast_timestamp();
    manager.add_pending_tombstone(1, 1);
    manager.add_pending_tombstone(1, 2);
    manager.rollback_pending_tombstones_after(1, boundary);
    assert!(manager.is_pending_tombstone(1, 1));
    assert!(!manager.is_pending_tombstone(1, 2));
}

#[test]
fn table_rollback_keeps_the_claim_for_an_earlier_cold_delete() {
    let schema = SchemaBuilder::new("table_rollback_keeps_the_claim_for_an_earlier_cold_delete")
        .column("id", DataType::Integer, false, true)
        .build();
    let store = Arc::new(VersionStore::new(schema.table_name.clone(), schema));
    let mut local = TransactionVersionStore::new(Arc::clone(&store), 1);
    let manager = Arc::new(SegmentManager::new("retained_claim", None));
    store.try_claim_row(1, 1).unwrap();
    local.track_external_claim(1);
    manager.add_pending_tombstone(1, 1);
    let boundary = get_fast_timestamp();
    local
        .put(1, Row::from_values(vec![Value::Integer(1)]), false)
        .unwrap();
    let hot = MVCCTable::new(1, Arc::clone(&store), local);
    let table = SegmentedTable::new(Box::new(hot), Arc::clone(&manager));
    table.rollback_to_timestamp(boundary);
    assert!(manager.is_pending_tombstone(1, 1));
    assert!(store.try_claim_row(1, 2).is_err());
    table.rollback_to_timestamp(0);
    assert!(!manager.is_pending_tombstone(1, 1));
    store.try_claim_row(1, 2).unwrap();
}

#[test]
fn returning_error_undoes_each_dml_path_in_transaction_api() {
    let db =
        Database::open("memory://returning_error_undoes_each_dml_path_in_transaction_api").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    for sql in [
        "INSERT INTO t VALUES (2, 20) RETURNING REGEXP_LIKE('x', '[')",
        "INSERT INTO t SELECT 2, 20 RETURNING REGEXP_LIKE('x', '[')",
        "INSERT INTO t VALUES (1, 20) ON CONFLICT (id) DO UPDATE SET v = 20 RETURNING REGEXP_LIKE('x', '[')",
        "UPDATE t SET v = 20 WHERE id = 1 RETURNING REGEXP_LIKE('x', '[')",
        "DELETE FROM t WHERE id = 1 RETURNING REGEXP_LIKE('x', '[')",
    ] {
        assert!(tx.execute(sql, ()).is_err(), "{sql}");
        assert_eq!(tx.query_one::<i64, _>("SELECT COUNT(*) FROM t", ()).unwrap(), 1, "{sql}");
        assert_eq!(tx.query_one::<i64, _>("SELECT v FROM t WHERE id = 1", ()).unwrap(), 10, "{sql}");
    }
    tx.commit().unwrap();
    assert_eq!(values(&db), [10]);
}

#[test]
fn failed_cascade_restores_every_touched_table() {
    let db = Database::open("memory://failed_cascade_restores_every_touched_table").unwrap();
    db.execute("CREATE TABLE parents (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE CASCADE, v INTEGER)", ()).unwrap();
    db.execute("INSERT INTO parents VALUES (1), (2)", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 10), (2, 2, 20)", ())
        .unwrap();
    let other = db.clone();
    let mut blocker = other.begin().unwrap();
    blocker
        .execute("UPDATE children SET v = 21 WHERE id = 2", ())
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO parents VALUES (3)", ()).unwrap();
    assert!(db
        .execute("DELETE FROM parents WHERE id IN (1, 2)", ())
        .is_err());
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM parents", ())
            .unwrap(),
        3
    );
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM children", ())
            .unwrap(),
        2
    );
    assert_eq!(
        other
            .execute("UPDATE children SET v = 11 WHERE id = 1", ())
            .unwrap(),
        1
    );
    blocker.rollback().unwrap();
    db.execute("COMMIT", ()).unwrap();
}

#[test]
fn prepared_returning_error_undoes_its_insert() {
    let db = Database::open("memory://prepared_returning_error_undoes_its_insert").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let insert = db
        .prepare("INSERT INTO t VALUES ($1, $2) RETURNING REGEXP_LIKE('x', $3)")
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    for id in [2, 3] {
        assert!(insert.execute((id, 20, "[")).is_err());
        assert_eq!(values(&db), [10]);
    }
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(values(&db), [10]);
}
