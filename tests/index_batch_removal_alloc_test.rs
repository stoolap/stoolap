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

//! A seal's batch removal from an index allocates a fixed number of times,
//! however many rows and keys the batch holds

// The mimalloc feature sets the library's own global allocator
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, Value};
use stoolap::storage::index::{BTreeIndex, HashIndex, MultiColumnIndex};
use stoolap::storage::Index;

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

const ROWS: i64 = 20_000;
const KEYS: i64 = 10;

/// Rows 1..=ROWS over KEYS keys; for three columns, the key spread over them
fn rows(columns: usize) -> Vec<(i64, Vec<Value>)> {
    (1..=ROWS)
        .map(|id| {
            let values = (0..columns)
                .map(|c| Value::Integer((id % KEYS) / (c as i64 + 1)))
                .collect();
            (id, values)
        })
        .collect()
}

fn fill(index: &dyn Index, rows: &[(i64, Vec<Value>)]) {
    let entries: Vec<(i64, &[Value])> = rows.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    index.add_batch_slice(&entries).unwrap();
}

/// Allocations of two batch removals, 2,000 rows then 10,000, each given in
/// descending id order as a clustered seal can give them
fn allocations_of_removals(index: &dyn Index) -> (usize, usize) {
    let mut counts = [0; 2];
    for (i, range) in [(1..=2_000i64), (2_001..=12_000)].into_iter().enumerate() {
        let ids: Vec<i64> = range.rev().collect();
        let before = ALLOCATIONS.with(Cell::get);
        index.remove_batch_ids(&ids).unwrap().unwrap();
        counts[i] = ALLOCATIONS.with(Cell::get) - before;
    }
    (counts[0], counts[1])
}

fn assert_bounded(name: &str, (small, large): (usize, usize)) {
    assert!(
        small < 16 && large < 16,
        "{name}: removing 2,000 rows allocated {small} times, 10,000 rows {large} times"
    );
}

#[test]
fn a_btree_batch_removal_allocates_a_fixed_number_of_times() {
    let index = BTreeIndex::new(
        "idx".into(),
        "t".into(),
        1,
        "k".into(),
        DataType::Integer,
        false,
        0,
    );
    fill(&index, &rows(1));
    assert_bounded("btree", allocations_of_removals(&index));
    assert_eq!(
        index.get_row_ids_equal(&[Value::Integer(3)]).len(),
        800,
        "the rows past 12,000 stay"
    );
}

#[test]
fn a_hash_batch_removal_allocates_a_fixed_number_of_times() {
    let index = HashIndex::new(
        "idx".into(),
        "t".into(),
        vec!["k".into()],
        vec![1],
        vec![DataType::Integer],
        false,
        0,
    );
    fill(&index, &rows(1));
    assert_bounded("hash", allocations_of_removals(&index));
    assert_eq!(index.get_row_ids_equal(&[Value::Integer(3)]).len(), 800);
}

#[test]
fn a_multi_column_batch_removal_allocates_a_fixed_number_of_times() {
    let index = MultiColumnIndex::new(
        "idx".into(),
        "t".into(),
        vec!["a".into(), "b".into(), "c".into()],
        vec![1, 2, 3],
        vec![DataType::Integer; 3],
        false,
        0,
    );
    fill(&index, &rows(3));
    // Every structure built: the sorted keys and both prefixes
    let key = [Value::Integer(3), Value::Integer(1), Value::Integer(1)];
    assert_eq!(index.find(&key[..1]).unwrap().len(), 2_000);
    assert_eq!(index.find(&key[..2]).unwrap().len(), 2_000);
    assert_eq!(
        index.find_range(&key, &key, true, true).unwrap().len(),
        2_000
    );
    assert_bounded("multi-column", allocations_of_removals(&index));
    assert_eq!(index.find(&key).unwrap().len(), 800);
    assert_eq!(index.find(&key[..1]).unwrap().len(), 800);
    assert_eq!(index.find(&key[..2]).unwrap().len(), 800);
}
