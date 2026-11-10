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

//! A new key of a B-tree index with one row costs its share of the tree's
//! nodes and nothing of its own: no copy of the key, no id list

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, Value};
use stoolap::storage::index::BTreeIndex;
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

const KEYS: usize = 10_000;

fn index(data_type: DataType, unique: bool) -> BTreeIndex {
    BTreeIndex::new("idx".into(), "t".into(), 1, "v".into(), data_type, unique)
}

fn keys(data_type: DataType) -> Vec<Value> {
    (0..KEYS as i64)
        .map(|i| match data_type {
            DataType::Integer => Value::Integer(i),
            _ => Value::text(format!("a key longer than inline text {i:08}")),
        })
        .collect()
}

#[test]
fn a_new_btree_key_with_one_row_allocates_nothing_of_its_own() {
    let mut wrong = Vec::new();
    for data_type in [DataType::Integer, DataType::Text] {
        for unique in [false, true] {
            let keys = keys(data_type);
            let one_by_one = index(data_type, unique);
            let before = ALLOCATIONS.with(Cell::get);
            for (id, key) in keys.iter().enumerate() {
                one_by_one
                    .add(std::slice::from_ref(key), id as i64, id as i64)
                    .unwrap();
            }
            let added = ALLOCATIONS.with(Cell::get) - before;

            let batched = index(data_type, unique);
            let entries: Vec<(i64, &[Value])> = keys
                .iter()
                .enumerate()
                .map(|(id, key)| (id as i64, std::slice::from_ref(key)))
                .collect();
            let before = ALLOCATIONS.with(Cell::get);
            batched.add_batch_slice(&entries).unwrap();
            // A unique batch counts its keys in one map sized up front
            let batch_added = ALLOCATIONS.with(Cell::get) - before - usize::from(unique);

            for (path, count) in [("add", added), ("add_batch_slice", batch_added)] {
                // A node holds 64 keys: under one allocation per ten keys
                if count * 10 > KEYS {
                    wrong.push(format!(
                        "{data_type:?}, unique {unique}, {path}: {count} allocations for {KEYS} keys"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
