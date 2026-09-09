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

use std::sync::Arc;
use stoolap::core::{Row, Value};
use stoolap::executor::semantic_cache::{CacheLookupResult, SemanticCache};
use stoolap::executor::{ExecutionContext, Executor};
use stoolap::storage::mvcc::{RowVersion, TransactionVersionStore, RECOVERY_TRANSACTION_ID};
use stoolap::storage::traits::{Engine, QueryResult};
use stoolap::{Database, IsolationLevel};

const QUERY: &str = "SELECT * FROM t WHERE id > 0";

fn drain(mut result: Box<dyn QueryResult>) -> Vec<Row> {
    let mut rows = Vec::new();
    while result.next() {
        rows.push(result.row().clone());
    }
    assert!(result.last_error().is_none());
    result.close().unwrap();
    rows
}
fn values(executor: &Executor) -> Vec<(i64, i64)> {
    let mut rows: Vec<_> = drain(executor.execute(QUERY).unwrap())
        .iter()
        .map(|row| {
            (
                row.get(0).and_then(Value::as_int64).unwrap(),
                row.get(1).and_then(Value::as_int64).unwrap(),
            )
        })
        .collect();
    rows.sort_unstable();
    rows
}
fn fixture() -> (Database, Executor, Executor) {
    fixture_from(Database::open_in_memory().unwrap())
}
fn fixture_from(db: Database) -> (Database, Executor, Executor) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let reader = Executor::new(Arc::clone(db.engine()));
    let writer = Executor::new(Arc::clone(db.engine()));
    (db, reader, writer)
}

#[test]
fn cross_executor_truncate_invalidates_before_explicit_transaction_finishes() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (db, reader, writer) = if cold {
            fixture_from(
                Database::open(&format!(
                    "file://{}?checkpoint_interval=3600",
                    dir.path().display()
                ))
                .unwrap(),
            )
        } else {
            fixture()
        };
        if cold {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            assert_eq!(db.engine().get_version_store("t").unwrap().row_count(), 0);
        }
        assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
        assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
        writer.execute("BEGIN").unwrap();
        writer.execute("TRUNCATE TABLE t").unwrap();
        assert!(
            values(&reader).is_empty(),
            "TRUNCATE is visible before COMMIT"
        );
        writer.execute("ROLLBACK").unwrap();
        assert!(values(&reader).is_empty());
        writer.execute("INSERT INTO t VALUES (1, 77)").unwrap();
        assert_eq!(values(&reader), vec![(1, 77)]);
    }
}

#[test]
fn cold_default_mapping_and_same_name_recreation_change_cache_identity() {
    let dir = tempfile::tempdir().unwrap();
    let (db, reader, writer) = fixture_from(
        Database::open(&format!(
            "file://{}?checkpoint_interval=3600",
            dir.path().display()
        ))
        .unwrap(),
    );
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().get_version_store("t").unwrap().row_count(), 0);
    assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
    assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
    writer.execute("ALTER TABLE t DROP COLUMN v").unwrap();
    writer
        .execute("ALTER TABLE t ADD COLUMN v INTEGER DEFAULT 99")
        .unwrap();
    assert_eq!(values(&reader), vec![(1, 99), (2, 99)]);
    writer.execute("DROP TABLE t").unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
        .unwrap();
    writer.execute("INSERT INTO t VALUES (1, 77)").unwrap();
    assert_eq!(values(&reader), vec![(1, 77)]);
}

#[test]
fn a_finished_old_scan_inserted_after_invalidation_keeps_its_original_proof() {
    let (db, reader, writer) = fixture();
    let epoch = db.engine().capture_read_epoch().unwrap().unwrap();
    let ctx = ExecutionContext::new().with_read_epoch(epoch.clone());
    let old_rows = drain(reader.execute_with_context(QUERY, &ctx).unwrap());
    let proof = db.engine().cache_provenance(&epoch).unwrap();
    let cache = SemanticCache::new();
    let columns = vec!["id".to_owned(), "v".to_owned()];
    writer.execute("INSERT INTO t VALUES (3,30)").unwrap();
    cache.invalidate_table("t");
    // The producer resumes after invalidation. Its rows still belong to epoch.
    cache.insert_with_provenance("t", columns.clone(), old_rows, None, proof);
    let fresh = db.engine().capture_read_epoch().unwrap().unwrap();
    let fresh_proof = db.engine().cache_provenance(&fresh).unwrap();
    assert!(matches!(
        cache.lookup_with_provenance("t", &columns, None, fresh_proof),
        CacheLookupResult::Miss
    ));
    assert!(matches!(
        cache.lookup("t", &columns, None),
        CacheLookupResult::Miss
    ));
    assert!(db.engine().cache_provenance(&epoch).is_none());
    assert_eq!(
        drain(reader.execute_with_context(QUERY, &ctx).unwrap()).len(),
        2
    );
    assert_eq!(values(&reader).len(), 3);
}

#[test]
fn supplied_snapshot_and_private_transaction_reads_do_not_use_shared_results() {
    let (db, reader, writer) = fixture();
    assert_eq!(values(&reader).len(), 2);
    assert_eq!(values(&reader).len(), 2);
    let mut snapshot = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let epoch = snapshot.capture_read_epoch().unwrap().unwrap();
    let ctx = ExecutionContext::new().with_read_epoch(epoch);
    writer.execute("INSERT INTO t VALUES (3,30)").unwrap();
    assert_eq!(
        drain(reader.execute_with_context(QUERY, &ctx).unwrap()).len(),
        2
    );
    assert_eq!(reader.semantic_cache_stats().exact_hits, 1);
    snapshot.rollback().unwrap();
    writer.execute("BEGIN").unwrap();
    writer.execute("INSERT INTO t VALUES (4,40)").unwrap();
    assert_eq!(values(&writer).len(), 4);
    writer.execute("ROLLBACK").unwrap();
    assert_eq!(values(&reader).len(), 3);
}

#[test]
fn direct_cold_table_commit_invalidates_before_registry_terminal_outcome() {
    let dir = tempfile::tempdir().unwrap();
    let (db, reader, _writer) = fixture_from(
        Database::open(&format!(
            "file://{}?checkpoint_interval=3600",
            dir.path().display()
        ))
        .unwrap(),
    );
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().get_version_store("t").unwrap().row_count(), 0);
    assert_eq!(values(&reader).len(), 2);
    assert_eq!(values(&reader).len(), 2);
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut table = tx.get_table("t").unwrap();
    assert_eq!(table.delete_by_row_ids(&[1]).unwrap(), 1);
    table.commit().unwrap(); // The public compatibility path publishes source 0.
    assert_eq!(values(&reader), vec![(2, 20)]);
    tx.commit().unwrap();
    assert_eq!(values(&reader), vec![(2, 20)]);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn failed_multi_table_publication_does_not_cache_partial_cold_tombstones() {
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let (db, reader, writer) = fixture_from(
        Database::open(&format!(
            "file://{}?checkpoint_interval=3600",
            dir.path().display()
        ))
        .unwrap(),
    );
    db.execute(
        "CREATE TABLE other_t (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO other_t VALUES (1,10)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().get_version_store("t").unwrap().row_count(), 0);
    assert_eq!(values(&reader).len(), 2);
    assert_eq!(values(&reader).len(), 2);
    writer.execute("BEGIN").unwrap();
    writer.execute("DELETE FROM t WHERE id=1").unwrap();
    writer
        .execute("UPDATE other_t SET v=90 WHERE id=1")
        .unwrap();
    stoolap::test_failpoints::fail_table_publish_on(2);
    assert!(writer.execute("COMMIT").is_err());
    assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
    assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
    writer.execute("INSERT INTO t VALUES (3,30)").unwrap();
    assert_eq!(values(&reader).len(), 3);
}

#[test]
fn public_already_visible_version_ingress_invalidates_cached_results() {
    for operation in 0..4 {
        let (db, reader, _writer) = fixture();
        assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
        assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
        let epoch = db.engine().capture_read_epoch().unwrap().unwrap();
        assert!(db.engine().cache_provenance(&epoch).is_some());
        let store = db.engine().get_version_store("t").unwrap();
        let version = RowVersion::new(
            RECOVERY_TRANSACTION_ID,
            Row::from_values(vec![Value::Integer(3), Value::Integer(30)]),
        );
        match operation {
            0 => store.add_version(3, version),
            1 => store.add_version_single(3, version),
            2 => store.add_versions_batch(vec![(3, version)]),
            3 => store.apply_recovered_version(3, version),
            _ => unreachable!(),
        }
        .unwrap();
        assert!(db.engine().cache_provenance(&epoch).is_none());
        let expected = vec![(1, 10), (2, 20), (3, 30)];
        assert_eq!(values(&reader), expected, "operation {operation}");
        assert_eq!(values(&Executor::new(Arc::clone(db.engine()))), expected);
        assert_eq!(reader.semantic_cache_stats().exact_hits, 1);
    }
}

#[test]
fn public_recovery_delete_and_physical_removal_invalidate_cached_results() {
    for physical_remove in [false, true] {
        let (db, reader, _writer) = fixture();
        assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
        assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
        let epoch = db.engine().capture_read_epoch().unwrap().unwrap();
        assert!(db.engine().cache_provenance(&epoch).is_some());
        let store = db.engine().get_version_store("t").unwrap();
        if physical_remove {
            let (_, snapshot) = store.extract_for_seal(RECOVERY_TRANSACTION_ID);
            let (removed, _, skipped) = store.remove_sealed_rows(&[1], &snapshot);
            assert_eq!(removed, 1);
            assert!(skipped.is_empty());
            store.subtract_committed_row_count(removed);
        } else {
            store.mark_deleted(1, RECOVERY_TRANSACTION_ID).unwrap();
        }
        assert!(db.engine().cache_provenance(&epoch).is_none());
        assert_eq!(values(&reader), vec![(2, 20)]);
        assert_eq!(
            values(&Executor::new(Arc::clone(db.engine()))),
            vec![(2, 20)]
        );
    }
}

#[test]
fn public_transaction_publication_and_undo_invalidate_already_visible_creator() {
    let (db, reader, _writer) = fixture();
    assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
    assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
    let epoch = db.engine().capture_read_epoch().unwrap().unwrap();
    let store = db.engine().get_version_store("t").unwrap();
    // Public low-level TVS callers can reuse a committed creator; no future
    // registry terminal transition will invalidate a previously captured proof.
    let creator = store.get_latest_version_id(1).unwrap();
    let mut local = TransactionVersionStore::new(store, creator);
    local
        .put(
            3,
            Row::from_values(vec![Value::Integer(3), Value::Integer(30)]),
            false,
        )
        .unwrap();
    local.prepare_publication().unwrap();
    local.apply_prepared_publication().unwrap();
    assert!(db.engine().cache_provenance(&epoch).is_none());
    assert_eq!(values(&reader), vec![(1, 10), (2, 20), (3, 30)]);
    assert_eq!(values(&reader), vec![(1, 10), (2, 20), (3, 30)]);
    let after_apply = db.engine().capture_read_epoch().unwrap().unwrap();
    assert!(db.engine().cache_provenance(&after_apply).is_some());
    local.finish_publication(false).unwrap();
    assert!(db.engine().cache_provenance(&after_apply).is_none());
    assert_eq!(values(&reader), vec![(1, 10), (2, 20)]);
}
