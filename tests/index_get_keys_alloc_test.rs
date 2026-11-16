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

//! A key read with other indexed equalities to take its rows by allocates
//! no more than one with a single key, whether the first key settles it or
//! the others are counted

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::Value;
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

const RUNS: usize = 20;

/// The allocations per read of `sql` with `params`
fn per_read(db: &Database, sql: &str, params: &[i64]) -> usize {
    let statement = db.prepare(sql).unwrap();
    let params: Vec<Value> = params.iter().map(|&p| Value::Integer(p)).collect();
    let read = || {
        let mut rows = 0;
        for row in statement.query(params.clone()).unwrap() {
            row.unwrap();
            rows += 1;
        }
        assert_eq!(rows, 1, "{sql}");
    };
    read();
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..RUNS {
        read();
    }
    (ALLOCATIONS.with(Cell::get) - before) / RUNS
}

#[test]
fn a_key_read_with_other_keys_allocates_as_one_with_a_single_key() {
    let db = Database::open("memory://index_get_keys_alloc").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER, d INTEGER)",
        "INSERT INTO t SELECT value, value % 2, value, value % 3, value % 5 FROM generate_series(1, 600)",
        "CREATE INDEX t_a ON t (a)",
        "CREATE INDEX t_b ON t (b)",
        "CREATE INDEX t_c ON t (c)",
        "CREATE INDEX t_d ON t (d)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // Row 301: a = 1, b = 301, c = 1, d = 1
    let single = per_read(&db, "SELECT id FROM t WHERE b = $1", &[301]);
    let settled = per_read(
        &db,
        "SELECT id FROM t WHERE b = $1 AND a = $2 AND c = $3 AND d = $4",
        &[301, 1, 1, 1],
    );
    let counted = per_read(
        &db,
        "SELECT id FROM t WHERE a = $2 AND c = $3 AND d = $4 AND b = $1",
        &[301, 1, 1, 1],
    );
    assert!(
        settled <= single && counted <= single,
        "allocations per read, one key {single}, first key settles {settled}, others counted {counted}"
    );
}
