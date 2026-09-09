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

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use stoolap::core::{DataType, IndexType, Row, SchemaBuilder, Value};
use stoolap::storage::index::HnswIndex;
use stoolap::storage::mvcc::{
    persistence::IndexMetadata, table::MVCCTable, RowVersion, TransactionRegistry,
    TransactionVersionStore, VersionStore,
};
use stoolap::storage::traits::Table;
use stoolap::{Database, IsolationLevel};

fn memory_db() -> Database {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    Database::open(&format!(
        "memory://canonical_index_{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
    .unwrap()
}

#[test]
fn sql_index_build_uses_current_rows_without_advancing_its_old_snapshot() {
    for (suffix, columns, keys) in [
        ("", "g", vec![20.into()]),
        (" USING HASH", "g", vec![20.into()]),
        (" USING BITMAP", "g", vec![20.into()]),
        (" USING BTREE", "g", vec![20.into()]),
        ("", "g, id", vec![20.into(), 1.into()]),
    ] {
        let db = memory_db();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 11)", ())
            .unwrap();
        let mut old = db
            .begin_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(
            old.query_one::<i64, _>("SELECT g FROM t WHERE id = 1", ())
                .unwrap(),
            10
        );
        db.execute("UPDATE t SET g = 20 WHERE id = 1", ()).unwrap();
        db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
        let table = db.engine().get_table_for_txn(old.id(), "t").unwrap();
        let index_type = match suffix {
            " USING HASH" => Some(IndexType::Hash),
            " USING BITMAP" => Some(IndexType::Bitmap),
            " USING BTREE" => Some(IndexType::BTree),
            _ => None,
        };
        let column_names: Vec<_> = columns.split(", ").collect();
        table
            .create_index_with_type("idx_g", &column_names, false, index_type)
            .unwrap();
        let index = db
            .engine()
            .get_version_store("t")
            .unwrap()
            .get_index("idx_g")
            .unwrap();
        assert_eq!(
            index.get_row_ids_equal(&keys).as_slice(),
            &[1],
            "{suffix} {columns}"
        );
        let mut stale = keys.clone();
        stale[0] = 10.into();
        assert!(index.get_row_ids_equal(&stale).is_empty());
        assert_eq!(
            old.query_one::<i64, _>("SELECT g FROM t WHERE id = 1", ())
                .unwrap(),
            10
        );
        assert_eq!(
            old.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
                .unwrap(),
            2
        );
        old.rollback().unwrap();
    }
}

#[test]
fn sql_default_columns_populate_all_native_index_builders() {
    for (suffix, columns, keys) in [
        ("", "g", vec![77.into()]),
        (" USING HASH", "g", vec![77.into()]),
        (" USING BITMAP", "g", vec![77.into()]),
        (" USING BTREE", "g", vec![77.into()]),
        ("", "g, id", vec![77.into(), 1.into()]),
    ] {
        let db = memory_db();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("ALTER TABLE t ADD COLUMN g INTEGER DEFAULT 77", ())
            .unwrap();
        db.execute(&format!("CREATE INDEX idx_g ON t({columns}){suffix}"), ())
            .unwrap();
        let index = db
            .engine()
            .get_version_store("t")
            .unwrap()
            .get_index("idx_g")
            .unwrap();
        assert_eq!(
            index.get_row_ids_equal(&keys).as_slice(),
            &[1],
            "{suffix} {columns}"
        );
        assert_eq!(
            db.query_one::<i64, _>("SELECT g FROM t WHERE id = 1", ())
                .unwrap(),
            77
        );
    }
}

fn publish(registry: &TransactionRegistry, store: &VersionStore, row_id: i64, row: Row) {
    let (id, _) = registry.begin_transaction();
    registry.start_commit(id);
    store.add_version(row_id, RowVersion::new(id, row)).unwrap();
    registry.complete_commit(id);
}

fn integer_store() -> (Arc<TransactionRegistry>, Arc<VersionStore>) {
    let registry = Arc::new(TransactionRegistry::new());
    let schema = SchemaBuilder::new("t")
        .add_primary_key("id", DataType::Integer)
        .add("g", DataType::Integer)
        .build();
    let store = Arc::new(VersionStore::with_visibility_checker(
        "t",
        schema,
        registry.clone(),
    ));
    (registry, store)
}

#[test]
fn public_btree_and_multi_helpers_use_current_defaulted_keys() {
    for multi in [false, true] {
        let (registry, store) = integer_store();
        publish(&registry, &store, 1, Row::from(vec![1.into(), 10.into()]));
        let (old, _) = registry.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
        let mut table = MVCCTable::new(
            old,
            store.clone(),
            TransactionVersionStore::new(store.clone(), old),
        );
        table
            .create_column_with_default_value(
                "d",
                DataType::Integer,
                false,
                Some("77".into()),
                Some(77.into()),
            )
            .unwrap();
        publish(&registry, &store, 1, Row::from(vec![1.into(), 20.into()]));
        let keys = if multi {
            table
                .create_multi_column_index("idx", &["d", "g"], false)
                .unwrap();
            vec![77.into(), 20.into()]
        } else {
            table.create_btree_index("d", false, Some("idx")).unwrap();
            vec![77.into()]
        };
        assert_eq!(
            store
                .get_index("idx")
                .unwrap()
                .get_row_ids_equal(&keys)
                .as_slice(),
            &[1]
        );
        assert_eq!(
            store.get_visible_version(1, old).unwrap().data.get(1),
            Some(&10.into())
        );
    }
}

#[test]
fn hnsw_build_uses_current_vector_and_defaulted_payload() {
    let db = memory_db();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '[1, 0]')", ())
        .unwrap();
    let mut old = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    db.execute("UPDATE t SET v = '[9, 0]' WHERE id = 1", ())
        .unwrap();
    db.engine()
        .get_table_for_txn(old.id(), "t")
        .unwrap()
        .create_hnsw_index(
            "idx_v",
            "v",
            false,
            16,
            64,
            32,
            stoolap::storage::index::HnswDistanceMetric::L2,
        )
        .unwrap();
    let index = db
        .engine()
        .get_version_store("t")
        .unwrap()
        .get_index("idx_v")
        .unwrap();
    let hnsw = index.as_any().downcast_ref::<HnswIndex>().unwrap();
    let query: Vec<u8> = [9.0_f32, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    assert_eq!(hnsw.search_nearest(&query, 1, 16), vec![(1, 0.0)]);
    old.rollback().unwrap();

    // Public generic HNSW construction shares the native builder, while SQL
    // WITH parameters uses the dedicated HNSW builder exercised above.
    let registry = Arc::new(TransactionRegistry::new());
    let mut schema = SchemaBuilder::new("v")
        .add_primary_key("id", DataType::Integer)
        .add("v", DataType::Vector)
        .build();
    schema.columns[1].vector_dimensions = 2;
    schema.columns[1].default_value = Some(Value::vector(vec![9.0, 0.0]));
    let store = Arc::new(VersionStore::with_visibility_checker(
        "v",
        schema,
        registry.clone(),
    ));
    publish(&registry, &store, 1, Row::from(vec![1.into()]));
    let (txn, _) = registry.begin_transaction();
    let table = MVCCTable::new(
        txn,
        store.clone(),
        TransactionVersionStore::new(store.clone(), txn),
    );
    table
        .create_index_with_type("idx", &["v"], false, Some(IndexType::Hnsw))
        .unwrap();
    let index = store.get_index("idx").unwrap();
    let hnsw = index.as_any().downcast_ref::<HnswIndex>().unwrap();
    assert_eq!(hnsw.search_nearest(&query, 1, 16), vec![(1, 0.0)]);
}

#[test]
fn unresolved_applied_publisher_excludes_builder_until_terminal_outcome() {
    for commit in [false, true] {
        let (registry, store) = integer_store();
        publish(&registry, &store, 1, Row::from(vec![1.into(), 10.into()]));
        let (writer_id, _) = registry.begin_transaction();
        let mut writer = TransactionVersionStore::new(store.clone(), writer_id);
        writer
            .put(1, Row::from(vec![1.into(), 20.into()]), false)
            .unwrap();
        registry.start_commit(writer_id);
        writer.prepare_publication().unwrap();
        writer.apply_prepared_publication().unwrap();
        let (builder_id, _) = registry.begin_transaction();
        let builder = MVCCTable::new(
            builder_id,
            store.clone(),
            TransactionVersionStore::new(store.clone(), builder_id),
        );
        assert!(builder.create_index("idx", &["g"], false).is_err());
        assert!(store.get_index("idx").is_none());
        if commit {
            registry.complete_commit(writer_id);
        } else {
            registry.abort_transaction(writer_id);
        }
        writer.finish_publication(commit).unwrap();
        builder.create_index("idx", &["g"], false).unwrap();
        let expected = if commit { 20 } else { 10 };
        assert_eq!(
            store
                .get_index("idx")
                .unwrap()
                .get_row_ids_equal(&[expected.into()])
                .as_slice(),
            &[1]
        );
    }
}

#[test]
fn uncommitted_insert_is_not_installed_in_shared_index_and_commit_adds_it() {
    for commit in [false, true] {
        let db = memory_db();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)", ())
            .unwrap();
        let mut txn = db.begin().unwrap();
        txn.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
        db.engine()
            .get_table_for_txn(txn.id(), "t")
            .unwrap()
            .create_index("idx", &["g"], false)
            .unwrap();
        let index = db
            .engine()
            .get_version_store("t")
            .unwrap()
            .get_index("idx")
            .unwrap();
        assert!(index.get_row_ids_equal(&[10.into()]).is_empty());
        if commit {
            txn.commit().unwrap();
        } else {
            txn.rollback().unwrap();
        }
        assert_eq!(
            index.get_row_ids_equal(&[10.into()]).as_slice(),
            if commit { &[1][..] } else { &[][..] }
        );
    }
}

fn metadata(multi: bool, unique: bool) -> IndexMetadata {
    IndexMetadata {
        name: "idx".into(),
        table_name: "t".into(),
        column_names: if multi {
            vec!["g".into(), "id".into()]
        } else {
            vec!["g".into()]
        },
        column_ids: if multi { vec![1, 0] } else { vec![1] },
        data_types: if multi {
            vec![DataType::Integer; 2]
        } else {
            vec![DataType::Integer]
        },
        is_unique: unique,
        index_type: IndexType::Hash,
        hnsw_m: None,
        hnsw_ef_construction: None,
        hnsw_ef_search: None,
        hnsw_distance_metric: None,
    }
}

#[test]
fn recovery_population_projects_defaults_and_propagates_unique_errors() {
    for deferred in [false, true] {
        for multi in [false, true] {
            let (registry, store) = integer_store();
            let (txn, _) = registry.begin_transaction();
            let mut table = MVCCTable::new(
                txn,
                store.clone(),
                TransactionVersionStore::new(store.clone(), txn),
            );
            table.drop_column("g").unwrap();
            table
                .create_column_with_default_value(
                    "g",
                    DataType::Integer,
                    false,
                    Some("77".into()),
                    Some(77.into()),
                )
                .unwrap();
            publish(&registry, &store, 1, Row::from(vec![1.into()]));
            store
                .create_index_from_metadata(&metadata(multi, false), deferred)
                .unwrap();
            if deferred {
                store.populate_all_indexes().unwrap();
            }
            let keys = if multi {
                vec![77.into(), 1.into()]
            } else {
                vec![77.into()]
            };
            assert_eq!(
                store
                    .get_index("idx")
                    .unwrap()
                    .get_row_ids_equal(&keys)
                    .as_slice(),
                &[1]
            );
        }
        let (registry, store) = integer_store();
        publish(&registry, &store, 1, Row::from(vec![1.into(), 10.into()]));
        publish(&registry, &store, 2, Row::from(vec![2.into(), 10.into()]));
        let build = store.create_index_from_metadata(&metadata(false, true), deferred);
        if deferred {
            build.unwrap();
            assert!(
                store.populate_all_indexes().is_err(),
                "startup must not accept incomplete unique index"
            );
        } else {
            assert!(build.is_err());
            assert!(store.get_index("idx").is_none());
        }
    }
}

#[test]
fn direct_local_commit_must_reach_registry_outcome_before_index_build() {
    for delete in [false, true] {
        for start_before_apply in [false, true] {
            let (registry, store) = integer_store();
            publish(&registry, &store, 0, Row::from(vec![0.into(), 99.into()]));
            publish(&registry, &store, 1, Row::from(vec![1.into(), 10.into()]));
            let (writer_id, _) = registry.begin_transaction();
            let mut writer = TransactionVersionStore::new(store.clone(), writer_id);
            writer
                .put(1, Row::from(vec![1.into(), 20.into()]), delete)
                .unwrap();
            if start_before_apply {
                registry.start_commit(writer_id);
            }
            // The public local-store API releases its catalog lease here, but
            // the caller has not yet published the registry outcome.
            writer.commit().unwrap();
            let (builder_id, _) = registry.begin_transaction();
            let table = MVCCTable::new(
                builder_id,
                store.clone(),
                TransactionVersionStore::new(store.clone(), builder_id),
            );
            let error = table.create_index("idx", &["g"], true).unwrap_err();
            assert!(error.to_string().contains("unresolved row mutation"));
            assert!(
                store.get_index("idx").is_none(),
                "a valid prefix must not install a partial index"
            );
            if !start_before_apply {
                registry.start_commit(writer_id);
            }
            registry.complete_commit(writer_id);
            // An unrelated in-flight commit does not block the per-table check.
            let (unrelated, _) = registry.begin_transaction();
            registry.start_commit(unrelated);
            table.create_index("idx", &["g"], true).unwrap();
            let index = store.get_index("idx").unwrap();
            assert_eq!(index.get_row_ids_equal(&[99.into()]).as_slice(), &[0]);
            assert!(index.get_row_ids_equal(&[10.into()]).is_empty());
            assert_eq!(
                index.get_row_ids_equal(&[20.into()]).as_slice(),
                if delete { &[][..] } else { &[1][..] }
            );
            registry.abort_transaction(unrelated);
        }
    }
}

#[test]
fn visible_creator_with_unresolved_distinct_deleter_blocks_index_build() {
    let (registry, store) = integer_store();
    publish(&registry, &store, 1, Row::from(vec![1.into(), 10.into()]));
    let epoch = registry.capture_read_epoch();
    let mut deleted = store
        .capture_hot_root()
        .visible_version(1, &epoch)
        .unwrap()
        .clone();
    let (deleter, _) = registry.begin_transaction();
    deleted.deleted_at_txn_id = deleter;
    store.add_version(1, deleted).unwrap();
    let (builder, _) = registry.begin_transaction();
    let table = MVCCTable::new(
        builder,
        store.clone(),
        TransactionVersionStore::new(store.clone(), builder),
    );
    assert!(table.create_index("idx", &["g"], false).is_err());
    registry.start_commit(deleter);
    assert!(table.create_index("idx", &["g"], false).is_err());
    registry.complete_commit(deleter);
    table.create_index("idx", &["g"], false).unwrap();
    assert!(store
        .get_index("idx")
        .unwrap()
        .get_row_ids_equal(&[10.into()])
        .is_empty());
}
