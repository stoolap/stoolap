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
use stoolap::common::CompactArc;
use stoolap::core::{DataType, Error, Result, Row, RowVec, Schema, SchemaBuilder, Value};
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

fn row(id: i64, value: i64) -> Row {
    Row::from(vec![id.into(), value.into()])
}
fn schema() -> Schema {
    SchemaBuilder::new("ordered")
        .column("id", DataType::Integer, false, true)
        .column("g", DataType::Integer, false, false)
        .build()
}
fn publish(registry: &TransactionRegistry, store: &VersionStore, values: &[(i64, i64)]) {
    let (id, _) = registry.begin_transaction();
    registry.start_commit(id);
    store
        .add_versions_batch(
            values
                .iter()
                .map(|&(key, value)| (key, RowVersion::new(id, row(key, value))))
                .collect(),
        )
        .unwrap();
    registry.complete_commit(id);
}
fn hot_fixture() -> MVCCTable {
    let registry = Arc::new(TransactionRegistry::new());
    let store = Arc::new(VersionStore::with_visibility_checker(
        "ordered",
        schema(),
        registry.clone(),
    ));
    publish(
        &registry,
        &store,
        &[(i64::MIN, 30), (-7, 10), (0, 20), (3, 40), (i64::MAX, 10)],
    );
    let epoch = registry.capture_read_epoch();
    let (txn, _) = registry.begin_transaction();
    let mut local = TransactionVersionStore::new(store.clone(), txn);
    local.put(0, row(0, 5), false).unwrap();
    local.put(3, row(3, 40), true).unwrap();
    local.put(5, row(5, 10), false).unwrap();
    // The root is deliberately bound only after these unrelated commits.
    publish(&registry, &store, &[(-7, 99), (99, 1)]);
    let mut table = MVCCTable::new(txn, store, local);
    table.set_read_epoch(epoch).unwrap();
    table
}
fn canonical_hot() -> Vec<(i64, i64)> {
    vec![(i64::MIN, 30), (-7, 10), (0, 5), (5, 10), (i64::MAX, 10)]
}
fn pairs(rows: RowVec) -> Vec<(i64, i64)> {
    rows.into_iter()
        .map(|(id, row)| {
            let Value::Integer(value) = row.get(1).unwrap() else {
                panic!("integer expected")
            };
            (id, *value)
        })
        .collect()
}
fn model(
    mut rows: Vec<(i64, i64)>,
    ascending: bool,
    limit: usize,
    offset: usize,
) -> Vec<(i64, i64)> {
    rows.sort_by_key(|&(id, value)| (value, id));
    if !ascending {
        rows.reverse();
    }
    rows.into_iter().skip(offset).take(limit).collect()
}

#[test]
fn captured_hot_ordered_and_top_k_match_frozen_canonical_rows() {
    let table = hot_fixture();
    let mut filter = ComparisonExpr::gte("g", 10.into());
    filter.prepare_for_schema(table.schema());
    for ascending in [true, false] {
        for limit in [0, 1, 3, 20] {
            for offset in [0, 1, 4, 9] {
                let expected = model(canonical_hot(), ascending, limit, offset);
                assert_eq!(
                    pairs(
                        table
                            .scan_top_k(None, "g", ascending, limit, offset)
                            .unwrap()
                            .unwrap()
                    ),
                    expected
                );
                assert_eq!(
                    pairs(
                        table
                            .collect_rows_ordered_by_index("g", ascending, limit, offset)
                            .unwrap()
                            .unwrap()
                    ),
                    expected
                );
                let sorted = table
                    .collect_rows_sorted_with_limit(1, ascending, limit, offset)
                    .unwrap();
                let expected_rows: Vec<_> = expected.iter().map(|&(id, g)| row(id, g)).collect();
                assert_eq!(sorted, expected_rows);
                let qualifying = canonical_hot()
                    .into_iter()
                    .filter(|&(_, g)| g >= 10)
                    .collect();
                assert_eq!(
                    pairs(
                        table
                            .scan_top_k(Some(&filter), "g", ascending, limit, offset)
                            .unwrap()
                            .unwrap()
                    ),
                    model(qualifying, ascending, limit, offset)
                );
            }
        }
    }
}

#[test]
fn captured_pk_keysets_handle_extreme_ids_own_rows_and_reverse_bounds() {
    let table = hot_fixture();
    for ascending in [true, false] {
        for bound in [
            None,
            Some(i64::MIN),
            Some(-7),
            Some(0),
            Some(5),
            Some(i64::MAX),
        ] {
            for inclusive in [false, true] {
                for limit in [0, 1, 3, 20] {
                    let mut expected: Vec<_> = canonical_hot()
                        .into_iter()
                        .filter(|&(id, _)| {
                            bound.is_none_or(|b| if inclusive { id >= b } else { id > b })
                        })
                        .collect();
                    if !ascending {
                        expected.reverse();
                    }
                    expected.truncate(limit);
                    let (after, from) = if inclusive {
                        (None, bound)
                    } else {
                        (bound, None)
                    };
                    assert_eq!(
                        pairs(
                            table
                                .collect_rows_pk_keyset(after, from, ascending, limit)
                                .unwrap()
                        ),
                        expected,
                        "ascending={ascending} bound={bound:?} inclusive={inclusive} limit={limit}"
                    );
                }
            }
        }
    }
}

fn cold_fixture() -> SegmentedTable {
    let registry = Arc::new(TransactionRegistry::new());
    let old = schema();
    let mut current = SchemaBuilder::new("ordered")
        .column("id", DataType::Integer, false, true)
        .column("g", DataType::Integer, false, false)
        .column("extra", DataType::Integer, false, false)
        .build();
    current.columns[1].default_value = Some(99.into());
    current.columns[2].default_value = Some(77.into());
    let manager = Arc::new(SegmentManager::new("ordered", None));
    let mut builder = VolumeBuilder::new(&old);
    for (id, g) in [(-10, 100), (2, -100), (8, 5)] {
        builder.add_row(id, &row(id, g));
    }
    manager.register_segment(
        1,
        Arc::new(builder.finish()),
        SegmentMeta {
            segment_id: 1,
            file_path: "captured-order.vol".into(),
            row_count: 3,
            min_row_id: -10,
            max_row_id: 8,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        },
        Some(&old),
    );
    manager.record_column_drop("g", 1);
    manager.invalidate_mappings(&current);
    let store = Arc::new(VersionStore::with_visibility_checker(
        "ordered",
        current,
        registry.clone(),
    ));
    publish(&registry, &store, &[(0, 10), (2, 20)]);
    let epoch = registry.capture_read_epoch();
    let (txn, _) = registry.begin_transaction();
    let local = CompactArc::new(RwLock::new(TransactionVersionStore::new(
        store.clone(),
        txn,
    )));
    local.write().unwrap().put(8, row(8, 30), false).unwrap();
    publish(&registry, &store, &[(0, 1000), (9, 1)]);
    let mut table = SegmentedTable::new(
        Box::new(MVCCTable::new_with_shared_store(txn, store, local)),
        manager,
    );
    table.set_read_epoch(epoch).unwrap();
    table
}

#[test]
fn captured_cold_hot_ordering_projection_defaults_and_stats_share_authority() {
    let table = cold_fixture();
    let expected = vec![(-10, 99), (0, 10), (2, 20), (8, 30)];
    for ascending in [true, false] {
        for limit in [0, 1, 3, 20] {
            for offset in [0, 1, 3, 8] {
                let rows = table
                    .scan_top_k(None, "g", ascending, limit, offset)
                    .unwrap()
                    .unwrap();
                for (_, row) in &rows {
                    assert_eq!(row.get(2), Some(&77.into()));
                }
                assert_eq!(
                    pairs(rows),
                    model(expected.clone(), ascending, limit, offset)
                );
            }
        }
    }
    let mut filter = ComparisonExpr::eq("g", 99.into());
    filter.prepare_for_schema(table.schema());
    let mut scanner = table.scan(&[2, 0], Some(&filter)).unwrap();
    let mut projected = Vec::new();
    while scanner.next() {
        projected.push(scanner.take_row());
    }
    assert!(scanner.err().is_none());
    assert_eq!(projected, vec![Row::from(vec![77.into(), (-10).into()])]);
    assert_eq!(table.sum_column(1).unwrap(), Some((159.0, 4)));
    assert_eq!(table.min_column(1).unwrap(), Some(Some(Value::Integer(10))));
    assert_eq!(table.max_column(1).unwrap(), Some(Some(Value::Integer(99))));
}

fn cold_only_fixture() -> SegmentedTable {
    cold_scalar_fixture(DataType::Integer, Value::Integer(10))
}

fn cold_scalar_fixture(data_type: DataType, value: Value) -> SegmentedTable {
    let registry = Arc::new(TransactionRegistry::new());
    let schema = SchemaBuilder::new("ordered")
        .column("id", DataType::Integer, false, true)
        .column("g", data_type, false, false)
        .build();
    let manager = Arc::new(SegmentManager::new("ordered", None));
    let mut builder = VolumeBuilder::new(&schema);
    builder.add_row(1, &Row::from(vec![1.into(), value]));
    manager.register_segment(
        1,
        Arc::new(builder.finish()),
        SegmentMeta {
            segment_id: 1,
            file_path: "cold-only.vol".into(),
            row_count: 1,
            min_row_id: 1,
            max_row_id: 1,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        },
        Some(&schema),
    );
    let store = Arc::new(VersionStore::with_visibility_checker(
        "ordered",
        schema,
        registry.clone(),
    ));
    let (txn, _) = registry.begin_transaction();
    let local = TransactionVersionStore::new(store.clone(), txn);
    let mut table = SegmentedTable::new(Box::new(MVCCTable::new(txn, store, local)), manager);
    table.set_read_epoch(registry.capture_read_epoch()).unwrap();
    table
}

#[derive(Clone, Debug)]
struct FailingPredicate;
impl Expression for FailingPredicate {
    fn evaluate(&self, _: &Row) -> Result<bool> {
        Err(Error::internal("captured predicate read failure"))
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
fn captured_scanner_and_top_k_preserve_fallible_predicate_errors() {
    let tables: Vec<Box<dyn Table>> = vec![
        Box::new(hot_fixture()),
        Box::new(cold_fixture()),
        Box::new(cold_only_fixture()),
    ];
    for (kind, table) in tables.into_iter().enumerate() {
        for (route, result) in [
            ("all", table.collect_all_rows(Some(&FailingPredicate))),
            (
                "limit",
                table.collect_rows_with_limit(Some(&FailingPredicate), 20, 0),
            ),
            (
                "unordered",
                table.collect_rows_with_limit_unordered(Some(&FailingPredicate), 20, 0),
            ),
            ("fetch", table.fetch_rows_by_ids(&[0, 1], &FailingPredicate)),
        ] {
            let error = match result {
                Err(error) => error,
                Ok(rows) => panic!("table={kind} route={route} returned {} rows", rows.len()),
            };
            assert!(error
                .to_string()
                .contains("captured predicate read failure"));
        }
        let error = table
            .scan_top_k(Some(&FailingPredicate), "g", true, 20, 0)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("captured predicate read failure"));
        let mut scanner = table.scan(&[0, 1], Some(&FailingPredicate)).unwrap();
        while scanner.next() {}
        assert!(scanner
            .err()
            .unwrap()
            .to_string()
            .contains("captured predicate read failure"));
        assert!(!scanner.next());
        assert!(
            scanner.err().is_some(),
            "end-of-stream must preserve the error"
        );
        scanner.close().unwrap();
        assert!(scanner.err().is_some(), "close must preserve the error");
    }
}

#[test]
fn cold_metadata_pruning_preserves_cross_numeric_and_signed_zero_equality() {
    for (data_type, stored, compared) in [
        (DataType::Integer, Value::Integer(10), Value::Float(10.0)),
        (DataType::Float, Value::Float(10.0), Value::Integer(10)),
        (DataType::Float, Value::Float(0.0), Value::Float(-0.0)),
        (DataType::Float, Value::Float(-0.0), Value::Float(0.0)),
    ] {
        let table = cold_scalar_fixture(data_type, stored.clone());
        let mut predicate = ComparisonExpr::eq("g", compared);
        predicate.prepare_for_schema(table.schema());
        let expected = Row::from(vec![1.into(), stored]);
        assert!(predicate.evaluate(&expected).unwrap());
        let rows = table.collect_all_rows(Some(&predicate)).unwrap();
        assert_eq!(rows.len(), 1, "physical type {data_type:?}");
        assert_eq!(rows[0].1, expected);
    }
}

#[test]
fn captured_dictionary_partitions_merge_nulls_and_authoritative_versions() {
    let registry = Arc::new(TransactionRegistry::new());
    let schema = SchemaBuilder::new("partitions")
        .column("id", DataType::Integer, false, true)
        .column("label", DataType::Text, true, false)
        .build();
    let make_row = |id: i64, text: Option<&str>| {
        Row::from(vec![
            id.into(),
            text.map(Value::text).unwrap_or(Value::Null(DataType::Text)),
        ])
    };
    let manager = Arc::new(SegmentManager::new("partitions", None));
    for (segment_id, mut values) in [
        (1, vec![(1, Some("a")), (2, None), (3, Some("b"))]),
        (2, vec![(4, Some("b")), (5, Some("a")), (6, None)]),
    ] {
        let first = segment_id as i64 * 100;
        values.extend((first..first + 32).map(|id| (id, Some("a"))));
        let mut builder = VolumeBuilder::new(&schema);
        for &(id, text) in &values {
            builder.add_row(id, &make_row(id, text));
        }
        let volume = Arc::new(builder.finish());
        assert!(
            matches!(
                volume.columns.get(1).unwrap(),
                stoolap::storage::volume::column::ColumnData::Dictionary { .. }
            ),
            "each file must exercise its own dictionary IDs"
        );
        manager.register_segment(
            segment_id,
            volume,
            SegmentMeta {
                segment_id,
                file_path: format!("dictionary-{segment_id}.vol").into(),
                row_count: values.len(),
                min_row_id: values[0].0,
                max_row_id: values[values.len() - 1].0,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            Some(&schema),
        );
    }
    let store = Arc::new(VersionStore::with_visibility_checker(
        "partitions",
        schema,
        registry.clone(),
    ));
    let (creator, _) = registry.begin_transaction();
    registry.start_commit(creator);
    store
        .add_versions_batch(vec![
            (1, RowVersion::new(creator, make_row(1, Some("c")))),
            (7, RowVersion::new(creator, make_row(7, None))),
        ])
        .unwrap();
    registry.complete_commit(creator);
    let epoch = registry.capture_read_epoch();
    let (txn, _) = registry.begin_transaction();
    let mut local = TransactionVersionStore::new(store.clone(), txn);
    local.put(3, make_row(3, Some("a")), false).unwrap();
    local.put(4, make_row(4, Some("b")), true).unwrap();
    local.put(8, make_row(8, Some("d")), false).unwrap();
    let (later, _) = registry.begin_transaction();
    registry.start_commit(later);
    store
        .add_versions_batch(vec![
            (5, RowVersion::new(later, make_row(5, Some("later")))),
            (9, RowVersion::new(later, make_row(9, Some("later")))),
        ])
        .unwrap();
    registry.complete_commit(later);
    let hot = MVCCTable::new(txn, store, local);
    let mut table = SegmentedTable::new(Box::new(hot), manager);
    table.set_read_epoch(epoch).unwrap();
    let partitions = table.get_partition_values("label").unwrap().unwrap();
    let mut logical: Vec<Option<String>> = partitions
        .into_iter()
        .map(|value| match value {
            Value::Null(_) => None,
            Value::Text(text) => Some(text.to_string()),
            value => panic!("unexpected partition {value:?}"),
        })
        .collect();
    logical.sort();
    assert_eq!(
        logical,
        vec![None, Some("a".into()), Some("c".into()), Some("d".into())]
    );
    assert_eq!(table.get_partition_count("label").unwrap(), Some(3));
    let mut distinct = table.compute_distinct_values(1).unwrap().unwrap();
    distinct.sort();
    assert_eq!(
        distinct,
        vec![Value::text("a"), Value::text("c"), Value::text("d")]
    );
}

#[test]
fn closing_captured_table_releases_its_lease_but_not_a_live_result() {
    let registry = Arc::new(TransactionRegistry::new());
    let store = Arc::new(VersionStore::with_visibility_checker(
        "ordered",
        schema(),
        registry.clone(),
    ));
    publish(&registry, &store, &[(1, 10), (2, 20)]);
    let (txn, _) = registry.begin_transaction();
    let local = TransactionVersionStore::new(store.clone(), txn);
    let mut table = SegmentedTable::hot_only(Box::new(MVCCTable::new(txn, store, local)));
    table.set_read_epoch(registry.capture_read_epoch()).unwrap();
    let mut scanner = table.scan(&[], None).unwrap();
    registry.abort_transaction(txn);
    registry.acknowledge_rollback(txn);
    table.close().unwrap();
    assert!(
        registry.oldest_retention_horizon().is_some(),
        "the independently owned scanner must retain its epoch"
    );
    let mut actual = Vec::new();
    while scanner.next() {
        actual.push(scanner.take_row());
    }
    assert!(scanner.err().is_none());
    assert_eq!(actual, vec![row(1, 10), row(2, 20)]);
    scanner.close().unwrap();
    assert_eq!(
        registry.oldest_retention_horizon(),
        None,
        "table and completed scanner must release their lease"
    );
}
