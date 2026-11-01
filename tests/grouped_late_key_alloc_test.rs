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

//! A grouped key computed over the rows the run returns writes into each row
//! in place: what it allocates beyond the same statement sorting by the item
//! does not grow with the groups

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(not(feature = "jemalloc"), not(feature = "test-filedb")))]

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

/// The allocations of one run of `sql`, prepared, over `groups` groups
fn allocations(groups: i64, sql: &str) -> usize {
    let db = Database::open(&format!("memory://grouped_late_key_alloc_{groups}")).unwrap();
    for setup in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
    ] {
        db.execute(setup, ()).unwrap();
    }
    db.execute(
        &format!("INSERT INTO t SELECT g.value, g.value % {groups} FROM generate_series(1, 400) g"),
        (),
    )
    .unwrap();
    db.execute(
        &format!("INSERT INTO u SELECT g.value, g.value % {groups} FROM generate_series(1, 400) g"),
        (),
    )
    .unwrap();
    let query = db.prepare(sql).unwrap();
    let run = || query.query(()).unwrap().count();
    assert_eq!(run(), groups as usize, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_late_key_allocates_nothing_for_each_group() {
    let item = "(SELECT COUNT(*) FROM u WHERE u.k = t.a)";
    let late = format!("SELECT a, {item} AS m FROM t GROUP BY a ORDER BY (SELECT 0) + m, a");
    let sorted = format!("SELECT a, {item} AS m FROM t GROUP BY a ORDER BY m, a");
    let beyond = |groups| allocations(groups, &late) as i64 - allocations(groups, &sorted) as i64;
    let (few, many) = (beyond(10), beyond(80));
    assert_eq!(
        few, many,
        "the late key's allocations beyond the item's: {few} over 10 groups, {many} over 80"
    );
}
