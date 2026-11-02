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

//! A scan of a volume that hands each row it decodes to its reader by
//! reference writes every row into the same one: what a run allocates does
//! not grow with the rows, with or without a WHERE the scan judges, whether
//! the volume is held whole or read from its file a row group at a time

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

/// The allocations of one run of `sql`, prepared, over a sealed table of
/// `rows` rows, opened again from its files first when `reopened`
fn allocations(rows: i64, reopened: bool, sql: &str) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, a INTEGER, b INTEGER, c TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT value, value % 10, value % 97, value % 7, 'c' || value \
             FROM generate_series(1, {rows})"
        ),
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let db = match reopened {
        true => {
            db.close().unwrap();
            drop(db);
            Database::open(&dsn).unwrap()
        }
        false => db,
    };
    let query = db.prepare(sql).unwrap();
    let run = || query.query(()).unwrap().count();
    assert_eq!(run(), 70, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_volume_scan_allocates_nothing_for_each_row_it_hands_over() {
    for reopened in [false, true] {
        for sql in [
            "SELECT k, b, SUM(a) FROM t GROUP BY k, b",
            "SELECT k, b, SUM(a) FROM t WHERE a >= 0 GROUP BY k, b",
        ] {
            let few = allocations(1_000, reopened, sql);
            let many = allocations(4_000, reopened, sql);
            assert!(
                many <= few + 100,
                "{sql}, reopened {reopened}: {few} allocations over 1000 rows, {many} over 4000"
            );
        }
    }
}
