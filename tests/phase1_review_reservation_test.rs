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
use stoolap::core::{DataType, Row, SchemaBuilder};
use stoolap::storage::index::HashIndex;
use stoolap::storage::mvcc::{
    table::MVCCTable,
    version_store::{TransactionVersionStore, VersionStore},
};
use stoolap::storage::traits::Table;
use stoolap::storage::volume::{
    manifest::{SegmentManager, SegmentMeta},
    table::SegmentedTable,
    writer::VolumeBuilder,
};

#[test]
fn cold_deleted_unique_key_stays_reserved_until_terminal_outcome() {
    for evolved_schema in [false, true] {
        let schema = SchemaBuilder::new("reservation_review")
            .column("id", DataType::Integer, false, true)
            .column("u", DataType::Integer, false, false)
            .build();
        let parent = Arc::new(VersionStore::new("reservation_review", schema.clone()));
        parent
            .add_index(
                "idx_u".into(),
                Arc::new(HashIndex::new(
                    "idx_u".into(),
                    "reservation_review".into(),
                    vec!["u".into()],
                    vec![1],
                    vec![DataType::Integer],
                    true,
                    0,
                )),
            )
            .unwrap();
        let old_schema = SchemaBuilder::new("reservation_review")
            .column("id", DataType::Integer, false, true)
            .column("removed", DataType::Integer, false, false)
            .column("u", DataType::Integer, false, false)
            .build();
        let volume_schema = if evolved_schema { &old_schema } else { &schema };
        let mut builder = VolumeBuilder::new(volume_schema);
        builder.add_row(
            1,
            &Row::from(if evolved_schema {
                vec![1.into(), 99.into(), 10.into()]
            } else {
                vec![1.into(), 10.into()]
            }),
        );
        let manager = Arc::new(SegmentManager::new("reservation_review", None));
        manager.register_segment(
            1,
            Arc::new(builder.finish()),
            SegmentMeta {
                segment_id: 1,
                file_path: "review.vol".into(),
                row_count: 1,
                min_row_id: 1,
                max_row_id: 1,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            Some(volume_schema),
        );
        if evolved_schema {
            manager.invalidate_mappings(&schema);
        }
        let first_local = Arc::new(RwLock::new(TransactionVersionStore::new(
            parent.clone(),
            10,
        )));
        let mut first = SegmentedTable::new(
            Box::new(MVCCTable::new_with_shared_store(
                10,
                parent.clone(),
                first_local.clone(),
            )),
            manager.clone(),
        );
        assert_eq!(first.delete_by_row_ids(&[1]).unwrap(), 1);
        first_local.write().unwrap().prepare_publication().unwrap();
        for row_id in manager.get_pending_tombstones(10) {
            let row = manager
                .get_cold_row_normalized(row_id, &schema)
                .unwrap()
                .unwrap();
            assert_eq!(row.get(1), Some(&10.into()));
            first_local
                .write()
                .unwrap()
                .reserve_cold_unique_keys(&row)
                .unwrap();
        }
        manager.prepare_tombstone_publication(10, 20);
        first_local
            .write()
            .unwrap()
            .apply_prepared_publication()
            .unwrap();
        manager.commit_pending_tombstones(10, 20);
        // T1 is paused immediately before recording COMMIT, matching engine ordering.
        let second_local = Arc::new(RwLock::new(TransactionVersionStore::new(
            parent.clone(),
            11,
        )));
        let mut second = SegmentedTable::new(
            Box::new(MVCCTable::new_with_shared_store(
                11,
                parent.clone(),
                second_local.clone(),
            )),
            manager.clone(),
        );
        let inserted = second.insert_discard(Row::from(vec![2.into(), 10.into()]));
        let prepared = inserted.and_then(|()| second_local.write().unwrap().prepare_publication());
        assert!(
            prepared.is_err(),
            "cold UNIQUE key removed by an unresolved publisher was available to another publisher"
        );
        second_local
            .write()
            .unwrap()
            .finish_publication(false)
            .unwrap();
        manager.finish_tombstone_publication(10, false);
        first_local
            .write()
            .unwrap()
            .finish_publication(false)
            .unwrap();
        assert_eq!(
            manager
                .get_cold_row_normalized(1, &schema)
                .unwrap()
                .unwrap()
                .get(1),
            Some(&10.into())
        );
    }
}

#[test]
fn detached_index_build_excludes_publishers_and_errors_leave_no_catalog_entry() {
    let schema = SchemaBuilder::new("detached_build")
        .column("id", DataType::Integer, false, true)
        .column("u", DataType::Integer, false, false)
        .build();
    let registry = Arc::new(stoolap::storage::mvcc::registry::TransactionRegistry::new());
    let parent = Arc::new(VersionStore::with_visibility_checker(
        "detached_build",
        schema,
        registry.clone(),
    ));
    let (seed_id, _) = registry.begin_transaction();
    let mut seed = TransactionVersionStore::new(parent.clone(), seed_id);
    seed.put(1, Row::from(vec![1.into(), 10.into()]), false)
        .unwrap();
    seed.commit().unwrap();
    registry.commit_transaction(seed_id);
    let (builder_id, _) = registry.begin_transaction();
    let table = MVCCTable::new(
        builder_id,
        parent.clone(),
        TransactionVersionStore::new(parent.clone(), builder_id),
    );
    let failed = table.create_index_with_type_and_finalizer(
        "idx_u",
        &["u"],
        true,
        Some(stoolap::core::IndexType::Hash),
        &mut |detached, _install| {
            assert!(parent.get_index("idx_u").is_none());
            assert_eq!(detached.get_row_ids_equal(&[10.into()]).as_slice(), &[1]);
            Err(stoolap::core::Error::internal(
                "cold decode failed before install",
            ))
        },
    );
    assert!(failed.is_err());
    assert!(parent.get_index("idx_u").is_none());
    table
        .create_index_with_type_and_finalizer(
            "idx_u",
            &["u"],
            true,
            Some(stoolap::core::IndexType::Hash),
            &mut |_detached, install| {
                for before_install in [true, false] {
                    if !before_install {
                        install()?;
                    }
                    let parent = Arc::clone(&parent);
                    std::thread::spawn(move || {
                        let mut writer = TransactionVersionStore::new(parent, 3);
                        writer
                            .put(2, Row::from(vec![2.into(), 20.into()]), false)
                            .unwrap();
                        assert!(
                            writer.prepare_publication().is_err(),
                            "DDL lease must exclude publication during build and installation"
                        );
                        writer.finish_publication(false).unwrap();
                    })
                    .join()
                    .unwrap();
                }
                Ok(())
            },
        )
        .unwrap();
    let mut writer = TransactionVersionStore::new(parent.clone(), 4);
    writer
        .put(2, Row::from(vec![2.into(), 20.into()]), false)
        .unwrap();
    writer.commit().unwrap();
    assert_eq!(
        parent
            .get_index("idx_u")
            .unwrap()
            .get_row_ids_equal(&[20.into()])
            .as_slice(),
        &[2]
    );
}
