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

//! A join that answers its ORDER BY with a LIMIT keeps its sort keys for
//! the rows its top-N holds, not one key vector per joined row

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

/// The rows of `sql`'s second run and the allocations it took
fn allocations(db: &Database, sql: &str) -> (Vec<i64>, usize) {
    let ids = |db: &Database| -> Vec<i64> {
        db.query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    };
    ids(db);
    let before = ALLOCATIONS.with(Cell::get);
    let found = ids(db);
    (found, ALLOCATIONS.with(Cell::get) - before)
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_join_order_by_limit_keeps_keys_for_its_top_rows() {
    let db = Database::open("memory://join_order_limit_alloc").unwrap();
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER, balance INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users SELECT value, value % 70, value % 1000 FROM generate_series(1, 10000)",
        (),
    )
    .unwrap();
    let sql = |floor: i64| {
        format!(
            "SELECT u.id, u.age FROM users u JOIN users v ON u.id = v.id \
             WHERE u.balance > {floor} ORDER BY u.age + 0, u.id LIMIT 10"
        )
    };
    // 490 and 4,990 joined rows, sorted by age and then id
    let (few, few_allocs) = allocations(&db, &sql(950));
    let (many, many_allocs) = allocations(&db, &sql(500));
    assert_eq!(
        few,
        [980, 1960, 3990, 4970, 7980, 8960, 981, 1961, 3991, 4971]
    );
    assert_eq!(many, [560, 630, 700, 770, 840, 910, 980, 1540, 1610, 1680]);
    // The joined rows are still built one each; a key vector each doubles it
    let more_rows = 4_990 - 490;
    assert!(
        many_allocs <= few_allocs + more_rows + more_rows / 4,
        "{many_allocs} allocations over about 4,990 joined rows, {few_allocs} over 490"
    );
}
