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

//! A join operator allocates the same over wide inputs as over narrow ones

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{Row, Value};
use stoolap::executor::{
    HashJoinOperator, JoinSide, JoinType, MaterializedOperator, NestedLoopJoinOperator, Operator,
};

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

/// One row of `width` integer columns
fn input(width: usize) -> Box<dyn Operator> {
    let row = Row::from_values(vec![Value::integer(1); width]);
    Box::new(MaterializedOperator::new(vec![row], width))
}

/// The allocations of building `join` over two inputs of `width` columns and
/// reading all its rows
fn run<'a, J: Operator + 'a>(
    width: usize,
    join: impl Fn(Box<dyn Operator>, Box<dyn Operator>) -> J,
) -> usize {
    let (left, right) = (input(width), input(width));
    let before = ALLOCATIONS.with(Cell::get);
    let mut join = join(left, right);
    join.open().unwrap();
    let mut rows = 0;
    while join.next().unwrap().is_some() {
        rows += 1;
    }
    join.close().unwrap();
    drop(join);
    assert_eq!(rows, 1);
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_hash_join_allocates_nothing_per_column() {
    let join = |left, right| {
        HashJoinOperator::new(
            left,
            right,
            JoinType::Inner,
            vec![0],
            vec![0],
            JoinSide::Right,
        )
    };
    let (narrow, wide) = (run(2, join), run(66, join));
    assert_eq!(
        wide, narrow,
        "{wide} allocations over 66 columns, {narrow} over 2"
    );
}

#[test]
fn a_nested_loop_join_allocates_nothing_per_column() {
    let join = |left, right| NestedLoopJoinOperator::new(left, right, JoinType::Cross, None);
    let (narrow, wide) = (run(2, join), run(66, join));
    assert_eq!(
        wide, narrow,
        "{wide} allocations over 66 columns, {narrow} over 2"
    );
}
