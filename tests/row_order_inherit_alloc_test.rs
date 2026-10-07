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

//! A volume that already has its row order takes none from another

// The mimalloc and heap-profile features set the library's own global allocator
#![cfg(not(any(feature = "mimalloc", feature = "heap-profile")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::volume::writer::{FrozenVolume, VolumeBuilder};

struct ThreadCounting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

/// A volume whose rows were added in descending id order, so it has a row
/// order permutation
fn unordered_volume(rows: i64) -> FrozenVolume {
    let schema = SchemaBuilder::new("t")
        .column("v", DataType::Integer, false, false)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    builder.allow_any_row_order();
    for id in (1..=rows).rev() {
        builder.add_row(id, &Row::from_values(vec![Value::Integer(id)]));
    }
    builder.finish().unwrap()
}

#[test]
fn an_inherited_row_order_is_not_copied_again() {
    let from = unordered_volume(100_000);
    let into = unordered_volume(100_000);
    assert!(from.row_order().is_some());
    into.inherit_row_order(&from);
    let before = ALLOCATIONS.with(Cell::get);
    into.inherit_row_order(&from);
    assert_eq!(
        ALLOCATIONS.with(Cell::get) - before,
        0,
        "a volume with its order copies none"
    );
}
