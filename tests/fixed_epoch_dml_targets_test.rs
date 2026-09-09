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
use stoolap::core::{DataType, Error, Row, SchemaBuilder, Value};
use stoolap::storage::expression::{AndExpr, ComparisonExpr, Expression};
use stoolap::storage::mvcc::registry::ReadEpoch;
use stoolap::storage::mvcc::table::MVCCTable;
use stoolap::storage::mvcc::{
    RowVersion, TransactionRegistry, TransactionVersionStore, VersionStore,
};
use stoolap::storage::traits::Table;

fn row(id: i64, g: i64) -> Row {
    Row::from(vec![Value::Integer(id), Value::Integer(g)])
}

fn fixture() -> (Arc<TransactionRegistry>, Arc<VersionStore>, ReadEpoch) {
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
    publish(&registry, &store, &[(1, 10), (2, 20)]);
    let epoch = registry.capture_read_epoch();
    (registry, store, epoch)
}

fn publish(registry: &TransactionRegistry, store: &VersionStore, rows: &[(i64, i64)]) {
    let (txn, _) = registry.begin_transaction();
    registry.start_commit(txn);
    store.add_versions_batch(
        rows.iter()
            .map(|&(id, g)| (id, RowVersion::new(txn, row(id, g))))
            .collect(),
    );
    registry.complete_commit(txn);
}

fn bind(registry: &TransactionRegistry, store: Arc<VersionStore>, epoch: ReadEpoch) -> MVCCTable {
    let (txn, _) = registry.begin_transaction();
    let local = Arc::new(RwLock::new(TransactionVersionStore::new(
        store.clone(),
        txn,
    )));
    let mut table = MVCCTable::new_with_shared_store(txn, store, local);
    table.set_read_epoch(epoch).unwrap();
    table
}

fn predicate(mode: usize, table: &MVCCTable) -> Option<Box<dyn Expression>> {
    let mut expression: Box<dyn Expression> = match mode {
        0 => Box::new(ComparisonExpr::eq("id", Value::Integer(1))),
        1 => Box::new(AndExpr::and(
            Box::new(ComparisonExpr::gte("id", Value::Integer(1))),
            Box::new(ComparisonExpr::lt("id", Value::Integer(2))),
        )),
        2 => return None,
        _ => Box::new(ComparisonExpr::eq("g", Value::Integer(10))),
    };
    expression.prepare_for_schema(&table.version_store().schema());
    Some(expression)
}

#[test]
fn update_pk_range_ids_and_scan_keep_captured_originals() {
    for mode in 0..4 {
        let (registry, store, epoch) = fixture();
        publish(&registry, &store, &[(1, 99), (2, 99)]);
        let mut table = bind(&registry, store, epoch);
        let expression = predicate(mode, &table);
        let mut seen = Vec::new();
        let mut setter = |mut value: Row| {
            seen.push(value.get(1).unwrap().clone());
            value.set(1, Value::Integer(11))?;
            Ok((value, true))
        };
        let count = if mode == 2 {
            table.update_by_row_ids(&[1], &mut setter).unwrap()
        } else {
            table.update(expression.as_deref(), &mut setter).unwrap()
        };
        assert_eq!(count, 1, "target mode {mode}");
        assert_eq!(seen, vec![Value::Integer(10)], "target mode {mode}");
        let local = table.txn_versions().read().unwrap();
        let (_, _, original) = local.iter_local_with_old().next().unwrap();
        assert_eq!(original, Some(&row(1, 10)));
        assert!(
            local.detect_conflicts_safe().is_err(),
            "captured creator must not be replaced with latest creator in mode {mode}"
        );
    }
}

#[test]
fn delete_pk_range_ids_and_scan_keep_captured_originals() {
    for mode in 0..4 {
        let (registry, store, epoch) = fixture();
        publish(&registry, &store, &[(1, 99), (2, 99)]);
        let mut table = bind(&registry, store, epoch);
        let expression = predicate(mode, &table);
        let count = if mode == 2 {
            table.delete_by_row_ids(&[1]).unwrap()
        } else {
            table.delete(expression.as_deref()).unwrap()
        };
        assert_eq!(count, 1, "target mode {mode}");
        let local = table.txn_versions().read().unwrap();
        let (_, deleted, original) = local.iter_local_with_old().next().unwrap();
        assert!(deleted.is_deleted());
        assert_eq!(original, Some(&row(1, 10)));
        assert!(
            local.detect_conflicts_safe().is_err(),
            "captured creator must survive mode {mode}"
        );
    }
}

#[test]
fn dml_full_scan_uses_latest_own_writes_before_global_predicate() {
    let (registry, store, epoch) = fixture();
    let mut table = bind(&registry, store, epoch);
    {
        let mut local = table.txn_versions().write().unwrap();
        local.put(1, row(1, 30), false).unwrap();
        local.put(2, row(2, 20), true).unwrap();
        local.put(3, row(3, 30), false).unwrap();
    }
    let mut g30 = ComparisonExpr::eq("g", Value::Integer(30));
    g30.prepare_for_schema(&table.version_store().schema());
    assert_eq!(
        table
            .update(Some(&g30), &mut |mut value| {
                value.set(1, Value::Integer(31))?;
                Ok((value, true))
            })
            .unwrap(),
        2
    );
    let mut g10 = ComparisonExpr::eq("g", Value::Integer(10));
    g10.prepare_for_schema(&table.version_store().schema());
    assert_eq!(table.delete(Some(&g10)).unwrap(), 0);
    let mut g31 = ComparisonExpr::eq("g", Value::Integer(31));
    g31.prepare_for_schema(&table.version_store().schema());
    assert_eq!(table.delete(Some(&g31)).unwrap(), 2);
    assert_eq!(table.delete_by_row_ids(&[1, 2, 3]).unwrap(), 0);
}

#[test]
fn captured_update_setter_failure_does_not_stage_a_prefix() {
    let (registry, store, epoch) = fixture();
    let mut table = bind(&registry, store, epoch);
    let mut calls = 0;
    let result = table.update(None, &mut |mut value| {
        calls += 1;
        if calls == 2 {
            return Err(Error::internal("setter failed"));
        }
        value.set(1, Value::Integer(99))?;
        Ok((value, true))
    });
    assert!(result.is_err());
    assert_eq!(calls, 2);
    assert!(!table.txn_versions().read().unwrap().has_local_changes());
}

#[test]
fn captured_value_rejects_a_later_distinct_committed_deleter() {
    for delete in [false, true] {
        let (registry, store, epoch) = fixture();
        let creator = store.get_latest_version_id(1).unwrap();
        let (deleter, _) = registry.begin_transaction();
        registry.start_commit(deleter);
        let mut deletion = RowVersion::new(creator, row(1, 10));
        deletion.deleted_at_txn_id = deleter;
        store.add_version(1, deletion);
        registry.complete_commit(deleter);
        let mut table = bind(&registry, store, epoch);
        let pk = ComparisonExpr::eq("id", Value::Integer(1));
        let selected = if delete {
            table.delete(Some(&pk)).unwrap()
        } else {
            table
                .update(Some(&pk), &mut |mut value| {
                    value.set(1, Value::Integer(11))?;
                    Ok((value, true))
                })
                .unwrap()
        };
        assert_eq!(
            selected, 1,
            "invisible deletion still exposes the old value"
        );
        assert!(
            table
                .txn_versions()
                .read()
                .unwrap()
                .detect_conflicts_safe()
                .is_err(),
            "a committed deletion must conflict even when its creator stayed unchanged"
        );
    }
}

#[test]
fn aborted_distinct_deleter_keeps_the_visible_original_for_index_undo() {
    let (registry, store, epoch) = fixture();
    let creator = store.get_latest_version_id(1).unwrap();
    let (deleter, _) = registry.begin_transaction();
    registry.start_commit(deleter);
    let mut deletion = RowVersion::new(creator, row(1, 10));
    deletion.deleted_at_txn_id = deleter;
    store.add_version(1, deletion);
    // The captured tree owns the excluded deletion head even after undo puts
    // the original value back into the live store.
    let mut table = bind(&registry, store.clone(), epoch);
    registry.abort_transaction(deleter);
    store.add_version(1, RowVersion::new(creator, row(1, 10)));
    registry.acknowledge_rollback(deleter);
    let pk = ComparisonExpr::eq("id", Value::Integer(1));
    assert_eq!(
        table
            .update(Some(&pk), &mut |mut value| {
                value.set(1, Value::Integer(11))?;
                Ok((value, true))
            })
            .unwrap(),
        1
    );
    let local = table.txn_versions().read().unwrap();
    assert!(local.detect_conflicts_safe().is_ok());
    let (_, _, original) = local.iter_local_with_old().next().unwrap();
    assert_eq!(
        original,
        Some(&row(1, 10)),
        "old index keys belong to the visible value, despite the raw captured marker"
    );
}
