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

//! A DISTINCT over a sealed text column read from its volume's dictionary
//! shares each entry: long values allocate no more than short ones

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

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

const VALUES: usize = 300;
const RUNS: usize = 4;

/// The allocations of `RUNS` runs of a DISTINCT over a sealed column of
/// `VALUES` distinct texts, each `width` bytes long
fn runs(width: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, c TEXT)", ())
        .unwrap();
    let rows: Vec<String> = (0..VALUES)
        .map(|i| format!("({i}, '{i:0width$}')"))
        .collect();
    db.execute(&format!("INSERT INTO t VALUES {}", rows.join(", ")), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let run = || {
        db.clear_semantic_cache().unwrap();
        let before = ALLOCATIONS.with(Cell::get);
        let count = db
            .query("SELECT DISTINCT c FROM t", ())
            .unwrap()
            .map(|row| row.unwrap())
            .count();
        assert_eq!(count, VALUES);
        ALLOCATIONS.with(Cell::get) - before
    };
    run();
    (0..RUNS).map(|_| run()).sum()
}

#[test]
fn a_dictionary_distinct_allocates_no_more_for_long_values() {
    let (short, long) = (runs(8), runs(40));
    assert!(
        long < short + VALUES,
        "{short} allocations over {RUNS} runs of 8-byte values, {long} of 40-byte values"
    );
}
