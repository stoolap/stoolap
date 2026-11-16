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

//! A run of a statement its plan reordered allocates what a run of the
//! statement written in that order does: taking the held order, and finding
//! that a statement has none to take, allocate nothing

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use stoolap::Database;

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the counter is an atomic,
    // which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The allocations of one warm run of `sql`, prepared, after three runs
fn allocations(db: &Database, sql: &str) -> usize {
    let stmt = db.prepare(sql).unwrap();
    for _ in 0..3 {
        assert_eq!(stmt.query(()).unwrap().count(), 1, "{sql}");
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(stmt.query(()).unwrap().count(), 1, "{sql}");
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

#[test]
fn a_reordered_run_allocates_as_the_order_written() {
    let db = Database::open("memory://join_order_alloc").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 7 FROM generate_series(1, 5000) g",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER, sid INTEGER)",
        "INSERT INTO u SELECT g.value, g.value, g.value % 100 FROM generate_series(1, 5000) g",
        "CREATE INDEX u_tid ON u (tid)",
        "CREATE INDEX u_sid ON u (sid)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO s SELECT g.value - 1, CASE WHEN (g.value - 1) % 50 = 0 THEN 1 ELSE 0 END \
         FROM generate_series(1, 100) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let reordered = allocations(
        &db,
        "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid JOIN s ON u.sid = s.id WHERE s.x = 1",
    );
    let written = allocations(
        &db,
        "SELECT COUNT(*) FROM s JOIN u ON u.sid = s.id JOIN t ON t.id = u.tid WHERE s.x = 1",
    );
    assert_eq!(
        reordered, written,
        "the reordered run allocates {reordered}, the written order {written}"
    );
}
