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

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, Value};
use stoolap::storage::index::{BTreeIndex, BitmapIndex, HashIndex, MultiColumnIndex, PkIndex};
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
    let mut buffers = index.removal_buffers(10_000).unwrap();
    let mut counts = [0; 2];
    for (i, range) in [(1..=2_000i64), (2_001..=12_000)].into_iter().enumerate() {
        let ids: Vec<i64> = range.rev().collect();
        let before = ALLOCATIONS.with(Cell::get);
        index.remove_batch_ids(&ids, &mut buffers).unwrap().unwrap();
        counts[i] = ALLOCATIONS.with(Cell::get) - before;
    }
    (counts[0], counts[1])
}

/// Allocations of a seal's two 10,000-row cleanup calls over rows `first`
/// onwards, the second emptying the index, with the buffers made before
fn allocations_of_a_seal_cleanup(index: &dyn Index, first: i64) -> (usize, usize) {
    let mut buffers = index.removal_buffers(10_000).unwrap();
    let mut counts = [0; 2];
    for (i, start) in [first, first + 10_000].into_iter().enumerate() {
        let ids: Vec<i64> = (start..start + 10_000).collect();
        let before = ALLOCATIONS.with(Cell::get);
        index.remove_batch_ids(&ids, &mut buffers).unwrap().unwrap();
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

#[test]
fn a_seal_cleanup_that_empties_an_index_allocates_nothing_more() {
    let btree = BTreeIndex::new(
        "idx".into(),
        "t".into(),
        1,
        "k".into(),
        DataType::Integer,
        false,
        0,
    );
    let hash = HashIndex::new(
        "idx".into(),
        "t".into(),
        vec!["k".into()],
        vec![1],
        vec![DataType::Integer],
        false,
        0,
    );
    let multi = MultiColumnIndex::new(
        "idx".into(),
        "t".into(),
        vec!["a".into(), "b".into(), "c".into()],
        vec![1, 2, 3],
        vec![DataType::Integer; 3],
        false,
        0,
    );
    for (name, index, columns) in [
        ("btree", &btree as &dyn Index, 1),
        ("hash", &hash, 1),
        ("multi-column", &multi, 3),
    ] {
        fill(index, &rows(columns));
        let (first, emptying) = allocations_of_a_seal_cleanup(index, 1);
        assert_eq!(first, emptying, "{name}: the emptying call allocated more");
    }
}

#[test]
fn a_seal_cleanup_of_a_bitmap_or_pk_index_allocates_nothing() {
    let bitmap = BitmapIndex::new(
        "idx".into(),
        "t".into(),
        vec!["k".into()],
        vec![1],
        vec![DataType::Integer],
        false,
        0,
    );
    fill(&bitmap, &rows(1));
    assert_eq!(allocations_of_a_seal_cleanup(&bitmap, 1), (0, 0), "bitmap");

    let pk = PkIndex::new("pk".into(), "t".into(), 0, "id".into());
    // Past the bitset, in the overflow set
    let first = 10_000_000;
    for id in first..first + ROWS {
        pk.add(&[Value::Integer(id)], id, id).unwrap();
    }
    assert_eq!(allocations_of_a_seal_cleanup(&pk, first), (0, 0), "pk");
}
