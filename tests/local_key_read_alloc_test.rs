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

//! A transaction's own rows read by key with no write between the reads
//! cost no allocation for the writes the keys catch up on

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

const RUNS: usize = 20;

/// The allocations seven further IN members cost a read: their rows and
/// the rows they yield
const SEVEN_MEMBERS: usize = 15;

/// The allocations per read of `sql` in a transaction holding 200 rows
fn per_read(db: &Database, sql: &str) -> usize {
    let statement = db.prepare(sql).unwrap();
    let read = || {
        for row in statement.query(()).unwrap() {
            row.unwrap();
        }
    };
    read();
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..RUNS {
        read();
    }
    (ALLOCATIONS.with(Cell::get) - before) / RUNS
}

#[test]
fn reads_by_key_without_writes_between_cost_no_more_per_member() {
    let db = Database::open("memory://local_key_read_alloc").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX t_k ON t (k)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value FROM generate_series(1, 200)",
        (),
    )
    .unwrap();
    let one = per_read(&db, "SELECT id FROM t WHERE k IN (1)");
    let eight = per_read(&db, "SELECT id FROM t WHERE k IN (1, 2, 3, 4, 5, 6, 7, 8)");
    db.execute("COMMIT", ()).unwrap();
    assert!(
        eight - one <= SEVEN_MEMBERS,
        "allocations per read of one member and of eight: {one}, {eight}"
    );
}
