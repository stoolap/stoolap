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

//! Removing ids from a paged id list allocates nothing, in the list and in
//! a clone of it

// The mimalloc feature sets the library's own global allocator
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::storage::index::id_list::IdList;

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

fn allocations_removing(list: &mut IdList<512>, batches: &[Vec<i64>]) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    for ids in batches {
        list.remove_sorted(ids);
    }
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn removals_that_merge_pages_allocate_nothing_in_a_list_or_its_clone() {
    let mut list = IdList::<512>::from_sorted((0..102_400).collect());
    let evens: Vec<i64> = (0..102_400).step_by(2).collect();
    assert_eq!(allocations_removing(&mut list, &[evens]), 0);
    // Cloned with every page half full, then one id less per page: the
    // pages fall under half and merge in pairs
    let mut copy = list.clone();
    let batches: Vec<Vec<i64>> = vec![
        (1..102_400).step_by(512).collect(),
        (3..102_000).step_by(2).collect(),
    ];
    assert_eq!(allocations_removing(&mut list, &batches), 0);
    assert_eq!(allocations_removing(&mut copy, &batches), 0);
    assert_eq!(
        list.iter().collect::<Vec<_>>(),
        copy.iter().collect::<Vec<_>>()
    );
}
