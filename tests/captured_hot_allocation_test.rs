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

// A counting System allocator cannot coexist with the library's optional
// mimalloc global allocator. Run this regression with --no-default-features.
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use stoolap::common::CowBTree;
use stoolap::core::{Row, SchemaBuilder, Value};
use stoolap::storage::mvcc::version_store::{CapturedHotView, RowVersion, VersionStore};
use stoolap::storage::mvcc::TransactionRegistry;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static REQUESTED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count_call(bytes: usize) {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
        let _ = REQUESTED_BYTES.try_with(|requested| requested.set(requested.get() + bytes));
    }
}

// SAFETY: Allocation ownership, alignment, and layouts are delegated unchanged
// to System. Thread-local counters do not allocate or access the allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_call(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_call(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_call(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocation_calls<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    struct StopCounting;
    impl Drop for StopCounting {
        fn drop(&mut self) {
            COUNTING.with(|counting| counting.set(false));
        }
    }
    CALLS.with(|calls| calls.set(0));
    REQUESTED_BYTES.with(|bytes| bytes.set(0));
    COUNTING.with(|counting| counting.set(true));
    let stop = StopCounting;
    let value = operation();
    drop(stop);
    (value, CALLS.with(Cell::get))
}

#[test]
fn cold_point_visibility_allocates_only_a_small_range_mask() {
    use stoolap::core::DataType;
    use stoolap::storage::traits::Scanner;
    use stoolap::storage::volume::scanner::VolumeScanner;
    use stoolap::storage::volume::writer::VolumeBuilder;

    let schema = SchemaBuilder::new("mask_allocations")
        .add_primary_key("id", DataType::Integer)
        .build();
    let registry = Arc::new(TransactionRegistry::new());
    let store =
        VersionStore::with_visibility_checker("mask_allocations", schema.clone(), registry.clone());
    let view = Arc::new(CapturedHotView::new(
        store.capture_hot_root(),
        registry.capture_read_epoch(),
        None,
    ));
    let mut builder = VolumeBuilder::new(&schema);
    for id in 0..20_000 {
        builder.add_row(id, &Row::from(vec![Value::Integer(id)]));
    }
    let volume = Arc::new(builder.finish());
    for rows in [1, 64, 256, 257, 20_000] {
        let mut scanner = VolumeScanner::with_range(volume.clone(), vec![0], 0, rows, None);
        let pending = Arc::default();
        let (_, calls) =
            allocation_calls(|| scanner.set_captured_visibility(view.clone(), pending));
        let bytes = REQUESTED_BYTES.with(Cell::get);
        if rows <= 256 {
            assert_eq!(calls, 1, "short ranges have no separately allocated bitmap");
            assert!(bytes <= 256, "short range requested {bytes} bytes");
        } else {
            assert_eq!(calls, 2, "large scans allocate one reusable bitmap");
            assert!(bytes <= 8192 + 256, "group mask requested {bytes} bytes");
        }
        assert!(scanner.next());
        assert_eq!(scanner.current_row_id(), 0);
        scanner.close().unwrap();
    }
}

#[test]
fn multilevel_tree_clone_and_bidirectional_walks_do_not_allocate() {
    let mut tree = CowBTree::new();
    // Non-monotonic insertion and subsequent deletion exercise split, merge,
    // and range boundary paths across several levels, rather than a leaf root.
    for n in 0..30_000 {
        let id = (n * 7919) % 30_000 - 15_000;
        tree.insert(id, id);
    }
    for id in (-15_000..15_000).step_by(3) {
        tree.remove(id);
    }
    let expected: i64 = tree.values().copied().sum();
    let expected_range: i64 = tree.range(-10_000..=10_000).map(|(_, value)| *value).sum();
    let ((all, range), calls) = allocation_calls(|| {
        let snapshot = tree.clone();
        (
            snapshot.iter().map(|(_, value)| *value).sum::<i64>(),
            snapshot
                .range(-10_000..=10_000)
                .map(|(_, value)| *value)
                .sum::<i64>(),
        )
    });
    assert_eq!((all, range), (expected, expected_range));
    assert_eq!(calls, 0);
    let ((all, range), calls) = allocation_calls(|| {
        let snapshot = tree.clone();
        (
            snapshot.iter_rev().map(|(_, value)| *value).sum::<i64>(),
            snapshot
                .range_rev(-10_000..=10_000)
                .map(|(_, value)| *value)
                .sum::<i64>(),
        )
    });
    assert_eq!((all, range), (expected, expected_range));
    assert_eq!(calls, 0);
}

#[test]
fn captured_root_clone_and_visible_callbacks_do_not_allocate() {
    let registry = Arc::new(TransactionRegistry::new());
    let store = VersionStore::with_visibility_checker(
        "allocation_test",
        SchemaBuilder::new("allocation_test").build(),
        registry.clone(),
    );
    let (txn, _) = registry.begin_transaction();
    registry.start_commit(txn);
    store.add_versions_batch(
        (0..20_000)
            .map(|id| {
                (
                    id,
                    RowVersion::new(txn, Row::from(vec![Value::Integer(id)])),
                )
            })
            .collect(),
    );
    registry.complete_commit(txn);
    let epoch = registry.capture_read_epoch();
    let ((count, sum, stopped_count, version_count, authority), calls) = allocation_calls(|| {
        let root = store.capture_hot_root();
        let mut version_count = 0;
        root.for_each_visible_version(&epoch, |_, _| version_count += 1);
        let view = CapturedHotView::new(root, epoch.clone(), None);
        let mut authority = [u64::MAX];
        view.mark_authoritative(&[7, 9999, 20_000], &mut authority);
        let cloned = view.clone();
        let mut count = 0;
        cloned.for_each_visible(|_, _| count += 1);
        let mut sum = 0;
        view.for_each_visible_range(100..200, |id, _| sum += id);
        let mut stopped_count = 0;
        view.for_each_visible_until(|_, _| {
            stopped_count += 1;
            stopped_count < 7
        });
        (count, sum, stopped_count, version_count, authority)
    });
    assert_eq!(
        (count, sum, stopped_count),
        (20_000, (100..200).sum::<i64>(), 7)
    );
    assert_eq!(version_count, 20_000);
    assert_eq!(authority, [0b11]);
    assert_eq!(calls, 0);
    let mut reverse = [0i64; 7];
    let (_, calls) = allocation_calls(|| {
        let view = CapturedHotView::new(store.capture_hot_root(), epoch.clone(), None);
        let mut count = 0;
        view.for_each_visible_range_rev_until(100..200, |id, _| {
            reverse[count] = id;
            count += 1;
            count < reverse.len()
        });
    });
    assert_eq!(reverse, [199, 198, 197, 196, 195, 194, 193]);
    assert_eq!(calls, 0);
}

#[test]
fn point_update_does_not_copy_a_large_prior_write_set() {
    use stoolap::core::DataType;
    use stoolap::storage::expression::ComparisonExpr;
    use stoolap::storage::mvcc::table::MVCCTable;
    use stoolap::storage::mvcc::TransactionVersionStore;
    use stoolap::storage::traits::Table;

    let registry = Arc::new(TransactionRegistry::new());
    let schema = SchemaBuilder::new("point_update")
        .add_primary_key("id", DataType::Integer)
        .add("value", DataType::Integer)
        .build();
    let store = Arc::new(VersionStore::with_visibility_checker(
        "point_update",
        schema,
        registry.clone(),
    ));
    let (creator, _) = registry.begin_transaction();
    store.add_version(
        1,
        RowVersion::new(
            creator,
            Row::from(vec![Value::Integer(1), Value::Integer(10)]),
        ),
    );
    registry.commit_transaction(creator);
    let (txn, _) = registry.begin_transaction();
    let mut local = TransactionVersionStore::new(store.clone(), txn);
    for id in 2..20_002 {
        local
            .put(
                id,
                Row::from(vec![Value::Integer(id), Value::Integer(id)]),
                false,
            )
            .unwrap();
    }
    let mut table = MVCCTable::new(txn, store, local);
    table
        .set_read_epoch(registry.read_epoch_for_transaction(txn).unwrap())
        .unwrap();
    let pk = ComparisonExpr::eq("id", Value::Integer(1));
    let (result, calls) = allocation_calls(|| table.update(Some(&pk), &mut |row| Ok((row, false))));
    assert_eq!(result.unwrap(), 0);
    assert_eq!(
        calls, 0,
        "a point UPDATE must not freeze/copy 20,000 unrelated own writes"
    );
    let mut table = stoolap::storage::volume::table::SegmentedTable::hot_only(Box::new(table));
    table
        .set_read_epoch(registry.read_epoch_for_transaction(txn).unwrap())
        .unwrap();
    let (result, calls) = allocation_calls(|| table.update(Some(&pk), &mut |row| Ok((row, false))));
    assert_eq!(result.unwrap(), 0);
    assert_eq!(
        calls, 0,
        "the cold DML wrapper must preserve the allocation-free hot point path"
    );
}

#[test]
fn captured_aggregates_allocate_per_group_not_per_input_row() {
    use stoolap::core::DataType;
    use stoolap::storage::expression::ComparisonExpr;
    use stoolap::storage::mvcc::version_store::{AggregateOp, TransactionVersionStore};
    use stoolap::storage::mvcc::MVCCTable;
    use stoolap::storage::traits::Table;
    use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
    use stoolap::storage::volume::table::SegmentedTable;
    use stoolap::storage::volume::writer::VolumeBuilder;

    fn fixture(count: i64, cold: bool) -> Box<dyn Table> {
        let schema = SchemaBuilder::new("aggregate_allocations")
            .add_primary_key("id", DataType::Integer)
            .add("a", DataType::Text)
            .add("b", DataType::Text)
            .add("n", DataType::Integer)
            .add("unread_payload", DataType::Text)
            .build();
        let registry = Arc::new(TransactionRegistry::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "aggregate_allocations",
            schema.clone(),
            registry.clone(),
        ));
        let (writer, _) = registry.begin_transaction();
        let mut volume = VolumeBuilder::new(&schema);
        let text_a = Value::text("a grouping string that exceeds the inline string representation");
        let text_b =
            Value::text("another grouping string that exceeds the inline string representation");
        let payload = Value::text("z".repeat(4096));
        for id in 1..=count {
            let row = Row::from(vec![
                Value::Integer(id),
                text_a.clone(),
                text_b.clone(),
                Value::Integer(id),
                payload.clone(),
            ]);
            if cold {
                volume.add_row(id, &row);
            } else {
                store.add_version(id, RowVersion::new(writer, row));
            }
        }
        registry.commit_transaction(writer);
        let (reader, _) = registry.begin_transaction();
        let hot = MVCCTable::new(
            reader,
            store.clone(),
            TransactionVersionStore::new(store, reader),
        );
        let mut table: Box<dyn Table> = if cold {
            let manager = Arc::new(SegmentManager::new("aggregate_allocations", None));
            manager.register_segment(
                1,
                Arc::new(volume.finish()),
                SegmentMeta {
                    segment_id: 1,
                    file_path: "aggregate-allocations.vol".into(),
                    row_count: count as usize,
                    min_row_id: 1,
                    max_row_id: count,
                    schema_version: 0,
                    creation_lsn: 0,
                    seal_seq: 0,
                },
                Some(&schema),
            );
            Box::new(SegmentedTable::new(Box::new(hot), manager))
        } else {
            Box::new(hot)
        };
        table.set_read_epoch(registry.capture_read_epoch()).unwrap();
        table
    }

    for cold in [false, true] {
        let operations = [(AggregateOp::CountStar, 0), (AggregateOp::Sum, 3)];
        let filter = ComparisonExpr::eq(
            "a",
            Value::text("a grouping string that exceeds the inline string representation"),
        );
        let measure = |count| {
            let table = fixture(count, cold);
            // Bind the immutable view once; subsequent aggregates share it.
            table
                .compute_grouped_aggregates(&[1], &operations)
                .unwrap()
                .unwrap();
            let ((single, composite, filtered), calls) = allocation_calls(|| {
                (
                    table
                        .compute_grouped_aggregates(&[1], &operations)
                        .unwrap()
                        .unwrap(),
                    // The longer key also verifies that spilled scratch is reused.
                    table
                        .compute_grouped_aggregates(&[1, 2, 1, 2, 1], &operations)
                        .unwrap()
                        .unwrap(),
                    table
                        .compute_filtered_aggregates(&operations, &filter)
                        .unwrap()
                        .unwrap(),
                )
            });
            assert_eq!(single.len(), 1);
            assert_eq!(composite.len(), 1);
            assert_eq!(filtered[0], Value::Integer(count));
            calls
        };
        let small = measure(10);
        let large = measure(20_000);
        assert_eq!(
            large, small,
            "cold={cold}: row materialization or key allocation grew with input length"
        );
    }
}

#[test]
fn statement_result_close_and_exhaustion_release_epoch_without_allocating() {
    use stoolap::executor::Executor;
    let db = stoolap::Database::open("memory://").unwrap();
    let executor = Executor::new(db.engine().clone());
    for close in [true, false] {
        // Warm result caches and the thread's reusable row-buffer pool before
        // measuring cleanup of an otherwise identical real SQL result.
        for _ in 0..2 {
            let mut result = executor.execute("SELECT 1 AS value").unwrap();
            assert!(result.next());
            if close {
                result.close().unwrap();
            } else {
                assert!(!result.next());
            }
        }
        let mut result = executor.execute("SELECT 1 AS value").unwrap();
        assert!(result.next());
        assert!(db.engine().registry().has_retention_obligations());
        let (_, calls) = allocation_calls(|| {
            if close {
                result.close().unwrap();
            } else {
                assert!(!result.next());
            }
        });
        assert_eq!(
            calls, 0,
            "close={close}: terminal cleanup must not allocate a replacement result"
        );
        assert_eq!(db.engine().registry().oldest_retention_horizon(), None);
        assert_eq!(result.columns(), &["value"]);
        assert_eq!(result.exact_len(), Some(0));
        assert!(!result.next());
        assert!(result.last_error().is_none());
    }
}
