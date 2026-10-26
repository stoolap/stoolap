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

//! Binding an IN list of parameters allocates a few times more for ten
//! thousand items than for a hundred, not once per item

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::Value;
use stoolap::executor::expression::Op;
use stoolap::executor::ExprProgram;

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

/// The allocations binding `column IN ($1, ..., $items)` takes
fn allocations_binding(items: u32) -> usize {
    let mut ops = vec![Op::LoadColumn(0)];
    ops.extend((0..items).map(Op::LoadParam));
    ops.extend([Op::InList(items), Op::Return]);
    let program = ExprProgram::new(ops);
    let params: Vec<Value> = (0..i64::from(items)).map(Value::Integer).collect();
    let before = ALLOCATIONS.with(Cell::get);
    let bound = program.bind(&params);
    let allocations = ALLOCATIONS.with(Cell::get) - before;
    let listing = bound.map(|bound| format!("{:?}", bound.ops()));
    assert!(
        listing.as_ref().is_some_and(|ops| ops.contains("InSet")),
        "the list binds to one set test: {listing:?}"
    );
    allocations
}

#[test]
fn binding_a_long_in_list_allocates_as_a_short_one_does() {
    let short = allocations_binding(100);
    let long = allocations_binding(10_000);
    // Buffers that double grow a few times more; one per item would be 10000
    assert!(
        long < 4 * short,
        "{long} allocations binding 10000 items, {short} binding 100"
    );
}
