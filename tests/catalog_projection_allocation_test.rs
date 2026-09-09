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

#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, Schema, SchemaColumn, Value};
use stoolap::storage::catalog::{
    ColumnId, Incarnation, TableId, TableIdentity, TableSchemaHistory,
};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count() {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
    }
}

// SAFETY: Ownership, layouts and alignment are passed unchanged to System.
// The thread-local counters do not allocate or access allocation contents.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn borrowed_projection_and_incarnation_sharing_do_not_allocate() {
    let identity = TableIdentity::new(TableId::new(1).unwrap(), Incarnation::FIRST);
    let first_schema = Schema::new(
        "t",
        vec![
            SchemaColumn::new(0, "id", DataType::Integer, false, true),
            SchemaColumn::new(1, "payload", DataType::Text, true, false),
        ],
    );
    let history = TableSchemaHistory::new(
        identity,
        0,
        first_schema.clone(),
        vec![ColumnId::new(1).unwrap(), ColumnId::new(2).unwrap()],
    )
    .unwrap();
    let mut next_schema = first_schema;
    let mut added = SchemaColumn::new(2, "introduced", DataType::Text, false, false);
    added.default_value = Some(Value::text("d".repeat(65_536)));
    next_schema.add_column(added).unwrap();
    let history = history
        .with_revision(
            identity,
            1,
            next_schema,
            vec![
                ColumnId::new(1).unwrap(),
                ColumnId::new(2).unwrap(),
                ColumnId::new(3).unwrap(),
            ],
        )
        .unwrap();
    let plan = history.projection(identity, 0).unwrap();
    let input = [Value::Integer(7), Value::text("p".repeat(65_536))];
    let default_pointer =
        plan.project(identity, 0, &input).unwrap().get(2).unwrap() as *const Value;

    struct StopCounting;
    impl Drop for StopCounting {
        fn drop(&mut self) {
            COUNTING.with(|counting| counting.set(false));
        }
    }
    CALLS.with(|calls| calls.set(0));
    COUNTING.with(|counting| counting.set(true));
    let stop = StopCounting;
    let snapshot = history.clone();
    let next_incarnation = snapshot.checked_next_incarnation().unwrap();
    assert_eq!(next_incarnation.column_high_water_mark(), 3);
    for _ in 0..1000 {
        let projected = plan.project(identity, 0, &input).unwrap();
        assert_eq!(projected.len(), 3);
        assert!(std::ptr::eq(projected.get(1).unwrap(), &input[1]));
        assert_eq!(projected.get(2).unwrap() as *const Value, default_pointer);
        assert_eq!(projected.iter().filter_map(Value::as_int64).sum::<i64>(), 7);
        assert_eq!(projected.iter().rev().count(), 3);
    }
    drop(stop);
    assert_eq!(CALLS.with(Cell::get), 0);
}
