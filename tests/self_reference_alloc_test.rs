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

//! A key into the statement's own table that waits for a later row keeps
//! the key once, not once per waiting row

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

const ROWS: i64 = 2000;

/// A waiting row costs its second parent lookup and its share of the
/// list's growth; copying the key per row would add four
const LIMIT: f64 = 1.5;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

/// Allocations of one INSERT of `ROWS` rows from `first`, each naming `up(id)`
fn insert(db: &Database, first: i64, up: impl Fn(i64) -> i64) -> usize {
    let values: Vec<String> = (first..first + ROWS)
        .map(|id| format!("({id}, {})", up(id)))
        .collect();
    let sql = format!("INSERT INTO s VALUES {}", values.join(", "));
    let before = allocations();
    db.execute(&sql, ()).unwrap();
    allocations() - before
}

#[test]
fn a_waiting_key_into_the_statements_table_keeps_its_key_once() {
    let db = Database::open("memory://self_reference_alloc").unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, up INTEGER REFERENCES s(id))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO s VALUES (1, NULL)", ()).unwrap();
    let found = insert(&db, 100_000, |_| 1);
    let waiting = insert(&db, 200_000, |id| id);
    let per_row = waiting.saturating_sub(found) as f64 / ROWS as f64;
    assert!(
        per_row <= LIMIT,
        "{per_row} more allocations per row naming itself than per row naming an existing parent"
    );
}
