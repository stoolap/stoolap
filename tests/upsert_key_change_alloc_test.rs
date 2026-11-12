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

//! What an upsert pays per conflict for changing a value a foreign key
//! references, over one that changes an unreferenced column

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

const ROWS: i64 = 500;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

/// Allocations of one upsert meeting every row of `p`, setting `set`
fn upsert(db: &Database, set: &str, offset: i64) -> usize {
    let values: Vec<String> = (1..=ROWS)
        .map(|id| format!("({id}, {}, 0)", id + offset))
        .collect();
    let sql = format!(
        "INSERT INTO p VALUES {} ON CONFLICT (id) DO UPDATE SET {set}",
        values.join(", ")
    );
    let before = allocations();
    db.execute(&sql, ()).unwrap();
    allocations() - before
}

#[test]
fn changing_a_referenced_value_costs_an_upsert_a_bounded_number_of_allocations() {
    let db = Database::open("memory://upsert_key_change_alloc").unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, code INTEGER UNIQUE, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pc INTEGER REFERENCES p(code) ON UPDATE CASCADE)",
        (),
    )
    .unwrap();
    let rows: Vec<String> = (1..=ROWS).map(|id| format!("({id}, {id}, 0)")).collect();
    db.execute(&format!("INSERT INTO p VALUES {}", rows.join(", ")), ())
        .unwrap();
    upsert(&db, "n = excluded.n + 1", 0);
    let unreferenced = upsert(&db, "n = excluded.n + 2", 0);
    upsert(&db, "code = excluded.code", 10_000);
    let referenced = upsert(&db, "code = excluded.code", 20_000);
    let per_row = referenced.saturating_sub(unreferenced) as f64 / ROWS as f64;
    assert!(
        per_row <= LIMIT,
        "{per_row} more allocations per conflict for a referenced value than for n"
    );
}

/// 27.15 measured before the walk that orders the keys' actions, which
/// adds none for a graph of a few columns
const LIMIT: f64 = 27.5;
