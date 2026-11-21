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

//! Encoding a WAL record that is compressed allocates the record only

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::storage::mvcc::wal_manager::{WALEntry, WALOperationType};

struct ThreadCounting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

const RECORDS: usize = 1_000;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

#[test]
fn a_compressed_record_allocates_its_bytes_only() {
    // A row of about 200 bytes that LZ4 shrinks
    let data: Vec<u8> = (0..200u32).map(|i| (i % 11) as u8).collect();
    let entry = WALEntry::new(1, "t".to_string(), 2, WALOperationType::Insert, data);
    let first = entry.encode();
    assert_eq!(first[5] & 1, 1, "the record is not compressed");
    let before = allocations();
    for _ in 0..RECORDS {
        drop(entry.encode());
    }
    let per_record = (allocations() - before) as f64 / RECORDS as f64;
    assert!(
        per_record <= 1.0,
        "{per_record} allocations per compressed record, at most 1"
    );
}
