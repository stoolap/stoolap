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

use std::sync::{Arc, RwLock};
use stoolap::core::{DataType, Row, Schema, SchemaBuilder, Value};
use stoolap::storage::expression::{ComparisonExpr, Expression};
use stoolap::storage::mvcc::{
    table::MVCCTable, RowVersion, TransactionRegistry, TransactionVersionStore, VersionStore,
};
use stoolap::storage::traits::Table;
use stoolap::storage::volume::{
    manifest::{SegmentManager, SegmentMeta},
    table::SegmentedTable,
    writer::VolumeBuilder,
};

fn row(id: i64, g: i64) -> Row {
    Row::from(vec![id.into(), g.into()])
}
fn meta(id: u64, rows: &[(i64, i64)]) -> SegmentMeta {
    SegmentMeta {
        segment_id: id,
        file_path: format!("{id}.vol").into(),
        row_count: rows.len(),
        min_row_id: rows.first().unwrap().0,
        max_row_id: rows.last().unwrap().0,
        creation_lsn: 0,
        seal_seq: 0,
        schema_version: 0,
    }
}
fn volume(
    schema: &Schema,
    rows: &[(i64, i64)],
) -> Arc<stoolap::storage::volume::writer::FrozenVolume> {
    let mut builder = VolumeBuilder::new(schema);
    for &(id, g) in rows {
        builder.add_row(id, &row(id, g));
    }
    Arc::new(builder.finish())
}
fn register(manager: &SegmentManager, schema: &Schema, id: u64, rows: &[(i64, i64)]) {
    manager.register_segment(id, volume(schema, rows), meta(id, rows), Some(schema));
}
struct Fixture {
    registry: Arc<TransactionRegistry>,
    store: Arc<VersionStore>,
    manager: Arc<SegmentManager>,
    local: Arc<RwLock<TransactionVersionStore>>,
    table: SegmentedTable,
    txn: i64,
}
fn fixture() -> Fixture {
    let registry = Arc::new(TransactionRegistry::new());
    let schema = SchemaBuilder::new("t")
        .add_primary_key("id", DataType::Integer)
        .add("g", DataType::Integer)
        .build();
    let store = Arc::new(VersionStore::with_visibility_checker(
        "t",
        schema.clone(),
        registry.clone(),
    ));
    let manager = Arc::new(SegmentManager::new("t", None));
    register(&manager, &schema, 1, &[(1, 10), (2, 20)]);
    let epoch = registry.capture_read_epoch();
    let (txn, _) = registry.begin_transaction();
    let local = Arc::new(RwLock::new(TransactionVersionStore::new(
        store.clone(),
        txn,
    )));
    let hot = MVCCTable::new_with_shared_store(txn, store.clone(), local.clone());
    let mut table = SegmentedTable::new(Box::new(hot), manager.clone());
    table.set_read_epoch(epoch).unwrap();
    Fixture {
        registry,
        store,
        manager,
        local,
        table,
        txn,
    }
}
fn condition(table: &SegmentedTable, by_id: bool) -> Box<dyn Expression> {
    let mut expression = if by_id {
        ComparisonExpr::eq("id", 1.into())
    } else {
        ComparisonExpr::eq("g", 10.into())
    };
    expression.prepare_for_schema(table.schema());
    Box::new(expression)
}
fn change(
    f: &mut Fixture,
    mode: usize,
    delete: bool,
    seen: &mut Vec<Value>,
) -> stoolap::core::Result<i32> {
    let expression = condition(&f.table, mode == 0);
    if delete {
        if mode == 2 {
            f.table.delete_by_row_ids(&[1])
        } else {
            f.table.delete(Some(expression.as_ref()))
        }
    } else {
        let mut setter = |mut value: Row| {
            seen.push(value.get(1).unwrap().clone());
            value.set(1, 11.into())?;
            Ok((value, true))
        };
        if mode == 2 {
            f.table.update_by_row_ids(&[1], &mut setter)
        } else {
            f.table.update(Some(expression.as_ref()), &mut setter)
        }
    }
}

#[test]
fn captured_cold_dml_excludes_later_segments_and_keeps_own_changes() {
    for delete in [false, true] {
        for mode in 0..3 {
            let mut f = fixture();
            register(&f.manager, &f.store.schema(), 2, &[(3, 10)]);
            let mut seen = Vec::new();
            assert_eq!(change(&mut f, mode, delete, &mut seen).unwrap(), 1);
            assert!(f.manager.is_pending_tombstone(f.txn, 1));
            assert!(!f.manager.is_pending_tombstone(f.txn, 3));
            if !delete {
                assert_eq!(seen, vec![Value::Integer(10)]);
                let mut observed = Vec::new();
                assert_eq!(
                    f.table
                        .update_by_row_ids(&[1], &mut |value| {
                            observed.push(value.get(1).unwrap().clone());
                            Ok((value, false))
                        })
                        .unwrap(),
                    0
                );
                assert_eq!(observed, vec![Value::Integer(11)]);
            } else {
                assert_eq!(f.table.delete_by_row_ids(&[1]).unwrap(), 0);
            }
            assert!(f.local.read().unwrap().get_local_version(3).is_none());
        }
    }
}

#[test]
fn captured_cold_dml_rejects_new_hot_authority_before_mirroring() {
    for delete in [false, true] {
        for mode in 0..3 {
            let mut f = fixture();
            let (writer, _) = f.registry.begin_transaction();
            f.registry.start_commit(writer);
            f.store.add_version(1, RowVersion::new(writer, row(1, 99)));
            f.registry.complete_commit(writer);
            let mut seen = Vec::new();
            let error = change(&mut f, mode, delete, &mut seen).unwrap_err();
            assert!(error.to_string().contains("write conflict"), "{error}");
            if !delete {
                assert_eq!(seen, vec![Value::Integer(10)]);
            }
            assert!(!f.local.read().unwrap().has_local_changes());
            assert!(!f.manager.is_pending_tombstone(f.txn, 1));
        }
    }
}

#[test]
fn captured_cold_dml_rejects_later_tombstone_and_replaced_source() {
    for replace in [false, true] {
        for delete in [false, true] {
            for mode in 0..3 {
                let mut f = fixture();
                if replace {
                    let rows = [(1, 99), (2, 20)];
                    f.manager.replace_segments_atomic(
                        2,
                        volume(&f.store.schema(), &rows),
                        meta(2, &rows),
                        &[1],
                    );
                } else {
                    let (writer, _) = f.registry.begin_transaction();
                    let sequence = f.registry.start_commit(writer);
                    f.registry.complete_commit(writer);
                    f.manager.add_tombstones(&[1], sequence as u64);
                }
                let mut seen = Vec::new();
                let error = change(&mut f, mode, delete, &mut seen).unwrap_err();
                assert!(error.to_string().contains("write conflict"), "{error}");
                if !delete {
                    assert_eq!(seen, vec![Value::Integer(10)]);
                }
                assert!(!f.local.read().unwrap().has_local_changes());
                assert!(!f.manager.is_pending_tombstone(f.txn, 1));
            }
        }
    }
}

#[test]
fn captured_delete_of_unknown_ids_does_not_create_phantom_writes() {
    let mut f = fixture();
    assert_eq!(f.table.delete_by_row_ids(&[999]).unwrap(), 0);
    assert!(!f.local.read().unwrap().has_local_changes());
    assert!(f.manager.get_pending_tombstones(f.txn).is_empty());
}

#[test]
fn cold_dml_setter_runs_outside_transfer_fence_and_keeps_earliest_generation() {
    let mut f = fixture();
    let manager = f.manager.clone();
    let schema = f.store.schema();
    let old_generation = manager.seal_generation();
    assert_eq!(
        f.table
            .update_by_row_ids(&[1], &mut |mut value| {
                // This would deadlock if user code were invoked under seal_read. A
                // separate new row leaves the selected source identity authoritative.
                let _fence = manager.acquire_seal_write();
                register(&manager, &schema, 2, &[(3, 30)]);
                value.set(1, 11.into())?;
                Ok((value, true))
            })
            .unwrap(),
        1
    );
    assert!(manager.seal_generation() > old_generation);
    assert_eq!(manager.get_txn_seal_generation(f.txn), Some(old_generation));
    manager.record_txn_seal_generation(f.txn);
    assert_eq!(manager.get_txn_seal_generation(f.txn), Some(old_generation));
}

fn unique_g(f: &Fixture) {
    use stoolap::storage::index::HashIndex;
    f.store
        .add_index(
            "u_g".into(),
            Arc::new(HashIndex::new(
                "u_g".into(),
                "t".into(),
                vec!["g".into()],
                vec![1],
                vec![DataType::Integer],
                true,
                0,
            )),
        )
        .unwrap();
}

#[test]
fn captured_cold_update_checks_current_unique_keys_and_revalidates_preflight() {
    for concurrent_segment in [false, true] {
        let mut f = fixture();
        unique_g(&f);
        let manager = f.manager.clone();
        let schema = f.store.schema();
        let error = f
            .table
            .update_by_row_ids(&[1], &mut |mut value| {
                if concurrent_segment {
                    let _fence = manager.acquire_seal_write();
                    register(&manager, &schema, 2, &[(3, 50)]);
                }
                value.set(
                    1,
                    if concurrent_segment {
                        50.into()
                    } else {
                        20.into()
                    },
                )?;
                Ok((value, true))
            })
            .unwrap_err();
        if concurrent_segment {
            assert!(error.to_string().contains("write conflict"), "{error}");
        } else {
            assert!(
                matches!(error, stoolap::core::Error::UniqueConstraint { .. }),
                "{error}"
            );
        }
        assert!(!f.local.read().unwrap().has_local_changes());
        assert!(!f.manager.is_pending_tombstone(f.txn, 1));
    }
}

#[derive(Debug, Clone)]
struct FailingPredicate;
impl Expression for FailingPredicate {
    fn evaluate(&self, _: &Row) -> stoolap::core::Result<bool> {
        Err(stoolap::core::Error::internal(
            "cold target expression failed",
        ))
    }
    fn evaluate_fast(&self, _: &Row) -> bool {
        false
    }
    fn with_aliases(&self, _: &rustc_hash::FxHashMap<String, String>) -> Box<dyn Expression> {
        Box::new(self.clone())
    }
    fn prepare_for_schema(&mut self, _: &Schema) {}
    fn is_prepared(&self) -> bool {
        true
    }
    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }
}

#[test]
fn captured_cold_predicate_errors_cannot_become_successful_partial_dml() {
    for delete in [false, true] {
        let mut f = fixture();
        let result = if delete {
            f.table.delete(Some(&FailingPredicate))
        } else {
            f.table
                .update(Some(&FailingPredicate), &mut |row| Ok((row, true)))
        };
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("cold target expression failed"));
        assert!(!f.local.read().unwrap().has_local_changes());
        assert!(f.manager.get_pending_tombstones(f.txn).is_empty());
    }
}

#[test]
fn cold_dml_pruning_uses_readded_columns_default_mapping() {
    for delete in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = stoolap::Database::open(&format!(
            "file://{}?checkpoint_interval=3600",
            dir.path().display()
        ))
        .unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("ALTER TABLE t DROP COLUMN g", ()).unwrap();
        db.execute("ALTER TABLE t ADD COLUMN g INTEGER DEFAULT 99", ())
            .unwrap();
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM t WHERE g = 99", ())
                .unwrap(),
            2
        );
        let affected = if delete {
            db.execute("DELETE FROM t WHERE g = 99", ()).unwrap()
        } else {
            db.execute("UPDATE t SET g = 100 WHERE g = 99", ()).unwrap()
        };
        assert_eq!(
            affected, 2,
            "old physical g zone maps must not prune current default g"
        );
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM t WHERE g = 99", ())
                .unwrap(),
            0
        );
    }
}

#[test]
fn cold_dml_bloom_pruning_preserves_numeric_equivalence() {
    for delete in [false, true] {
        for (data_type, stored, compared) in [
            (DataType::Integer, Value::Integer(10), Value::Float(10.0)),
            (DataType::Float, Value::Float(10.0), Value::Integer(10)),
            (DataType::Float, Value::Float(0.0), Value::Float(-0.0)),
            (DataType::Float, Value::Float(-0.0), Value::Float(0.0)),
        ] {
            let registry = Arc::new(TransactionRegistry::new());
            let schema = SchemaBuilder::new("t")
                .column("id", DataType::Integer, false, true)
                .column("g", data_type, false, false)
                .build();
            let manager = Arc::new(SegmentManager::new("t", None));
            let mut builder = VolumeBuilder::new(&schema);
            let expected = Row::from(vec![1.into(), stored]);
            builder.add_row(1, &expected);
            manager.register_segment(
                1,
                Arc::new(builder.finish()),
                meta(1, &[(1, 10)]),
                Some(&schema),
            );
            let store = Arc::new(VersionStore::with_visibility_checker(
                "t",
                schema,
                registry.clone(),
            ));
            let (txn, _) = registry.begin_transaction();
            let local = TransactionVersionStore::new(store.clone(), txn);
            let mut table =
                SegmentedTable::new(Box::new(MVCCTable::new(txn, store, local)), manager);
            table.set_read_epoch(registry.capture_read_epoch()).unwrap();
            let mut predicate = ComparisonExpr::eq("g", compared);
            predicate.prepare_for_schema(table.schema());
            assert!(predicate.evaluate(&expected).unwrap());
            let changed = if delete {
                table.delete(Some(&predicate)).unwrap()
            } else {
                table
                    .update(Some(&predicate), &mut |row| Ok((row, true)))
                    .unwrap()
            };
            assert_eq!(changed, 1, "delete={delete}, physical type {data_type:?}");
        }
    }
}
