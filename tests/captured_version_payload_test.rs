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

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use stoolap::core::{Row, RowVec, Value};
use stoolap::storage::expression::ComparisonExpr;
use stoolap::storage::mvcc::version_store::{
    AggregateOp, AggregateResult, RowVersion, VersionStore,
};
use stoolap::storage::Engine;
use stoolap::{test_failpoints, Database, IsolationLevel};

fn read_across_update<T>(name: &str, read: impl FnOnce(&VersionStore, i64) -> T) -> T {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER, note TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO accounts VALUES (1, 100, 'original shared payload')",
        (),
    )
    .unwrap();
    let store = db.engine().get_version_store("accounts").unwrap();
    let reader = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let writer = db.clone();
    let updated = Arc::new(AtomicBool::new(false));
    let updated_in_hook = Arc::clone(&updated);
    test_failpoints::after_version_root(move || {
        let count = std::thread::spawn(move || {
            writer.execute(
                "UPDATE accounts SET balance = 900, note = 'updated shared payload' WHERE id = 1",
                (),
            )
        })
        .join()
        .unwrap()
        .unwrap();
        assert_eq!(
            count, 1,
            "rendezvous writer must commit one ordinary UPDATE"
        );
        updated_in_hook.store(true, Ordering::Release);
    });
    let result = read(&store, reader.id());
    assert!(
        updated.load(Ordering::Acquire),
        "reader must hit rendezvous"
    );
    let current: i64 = db
        .query_one("SELECT balance FROM accounts WHERE id = 1", ())
        .unwrap();
    assert_eq!(current, 900, "UPDATE must be visible outside the snapshot");
    result
}

fn update_rows(rows: Vec<(i64, Row, RowVersion)>) -> RowVec {
    rows.into_iter()
        .map(|(id, row, version)| {
            assert_eq!(row, version.data, "payload must match its captured version");
            (id, row)
        })
        .collect()
}

fn balance_filter() -> ComparisonExpr {
    ComparisonExpr::eq("balance", Value::Integer(100))
}

macro_rules! captured_rows_test {
    ($name:ident, $read:expr) => {
        #[test]
        fn $name() {
            let rows = read_across_update(stringify!($name), $read);
            assert_eq!(rows.len(), 1, "snapshot must retain the matching row");
            assert_eq!(rows[0].0, 1);
            assert_eq!(
                rows[0].1.as_slice(),
                &[
                    Value::Integer(1),
                    Value::Integer(100),
                    Value::text("original shared payload"),
                ],
                "snapshot must not substitute the updated arena payload"
            );
        }
    };
}

captured_rows_test!(captured_all_rows, |store, id| store
    .get_all_visible_rows(id));
captured_rows_test!(captured_arena_rows, |store, id| store
    .get_all_visible_rows_arena(id));
captured_rows_test!(captured_cached_rows, |store, id| store
    .get_all_visible_rows_cached(id));
captured_rows_test!(captured_unsorted_rows, |store, id| store
    .get_all_visible_rows_unsorted(id));
captured_rows_test!(captured_update_versions, |store, id| update_rows(
    store.get_visible_versions_for_update(&[1], id)
));
captured_rows_test!(captured_update_rows, |store, id| update_rows(
    store.get_all_visible_rows_for_update(id)
));
captured_rows_test!(captured_filtered_update, |store, id| update_rows(
    store.get_all_visible_rows_for_update_filtered(id, &balance_filter())
));
captured_rows_test!(captured_limited_rows, |store, id| store
    .get_visible_rows_with_limit(id, 1, 0));
captured_rows_test!(captured_batch, |store, id| store
    .get_visible_rows_batch(id, 0, 1)
    .0);
captured_rows_test!(captured_batch_into, |store, id| {
    let mut rows = RowVec::new();
    assert!(!store.get_visible_rows_batch_into(id, 0, 1, &mut rows));
    rows
});
captured_rows_test!(captured_pk_ordered, |store, id| store
    .collect_rows_pk_ordered(id, true, 1, 0)
    .unwrap());
captured_rows_test!(captured_keyset, |store, id| store.collect_rows_keyset(
    id,
    Some(0),
    None,
    true,
    1
));
captured_rows_test!(captured_filtered_rows, |store, id| store
    .get_all_visible_rows_filtered(id, &balance_filter()));
captured_rows_test!(captured_filtered_callback, |store, id| {
    let mut rows = RowVec::new();
    store.for_each_visible_filtered(id, &balance_filter(), |row_id, row| {
        rows.push((row_id, row));
        true
    });
    rows
});
captured_rows_test!(captured_filtered_limit, |store, id| store
    .get_visible_rows_filtered_with_limit(id, &balance_filter(), 1, 0));
captured_rows_test!(captured_sorted_rows, |store, id| store
    .get_visible_rows_sorted_limit(id, 1, true, 1, 0));

#[test]
fn captured_aggregate() {
    let values = read_across_update("captured_aggregate", |store, id| {
        store.compute_aggregates(id, &[(AggregateOp::Sum, 1)])
    });
    assert!(
        matches!(values.as_slice(), [AggregateResult::Sum(100.0, 1)]),
        "snapshot must aggregate the captured value, got {values:?}"
    );
}
