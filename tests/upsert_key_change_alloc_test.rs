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
    let per_row = extra_per_conflict("upsert_key_change_alloc_one", 1);
    assert!(
        per_row <= ONE_CHILD,
        "{per_row} more allocations per conflict for a referenced value than for n"
    );
}

#[test]
fn seven_referencing_tables_are_planned_once_per_statement() {
    let per_row = extra_per_conflict("upsert_key_change_alloc_seven", 7);
    assert!(
        per_row <= SEVEN_CHILDREN,
        "{per_row} more allocations per conflict for a value seven tables reference"
    );
}

#[test]
fn a_key_into_a_column_the_upsert_changes_is_planned_once_per_statement() {
    let db = Database::open("memory://upsert_key_change_alloc_into_changed").unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, \
         b INTEGER UNIQUE REFERENCES s(a) ON UPDATE CASCADE, \
         c INTEGER REFERENCES s(b) ON UPDATE CASCADE, n INTEGER)",
        (),
    )
    .unwrap();
    let rows: Vec<String> = (1..=ROWS)
        .map(|id| format!("({id}, {id}, {id}, NULL, 0)"))
        .collect();
    db.execute(&format!("INSERT INTO s VALUES {}", rows.join(", ")), ())
        .unwrap();
    let upsert = |set: &str, offset: i64| {
        let values: Vec<String> = (1..=ROWS)
            .map(|id| format!("({id}, {key}, {key}, NULL, {id})", key = id + offset))
            .collect();
        let sql = format!(
            "INSERT INTO s VALUES {} ON CONFLICT (id) DO UPDATE SET {set}",
            values.join(", ")
        );
        let before = allocations();
        db.execute(&sql, ()).unwrap();
        allocations() - before
    };
    upsert("n = excluded.n + 1", 0);
    let unreferenced = upsert("n = excluded.n + 2", 0);
    upsert("a = excluded.a, b = excluded.b", ROWS);
    let referenced = upsert("a = excluded.a, b = excluded.b", 2 * ROWS);
    let per_row = referenced.saturating_sub(unreferenced) as f64 / ROWS as f64;
    assert!(
        per_row <= INTO_CHANGED,
        "{per_row} more allocations per conflict changing a and the b that references it"
    );
}

/// What a conflict changing `p.code`, which `children` empty tables
/// reference, allocates over one changing `p.n`
fn extra_per_conflict(name: &str, children: usize) -> f64 {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, code INTEGER UNIQUE, n INTEGER)",
        (),
    )
    .unwrap();
    for child in 0..children {
        db.execute(
            &format!(
                "CREATE TABLE c{child} (id INTEGER PRIMARY KEY, pc INTEGER REFERENCES p(code) ON UPDATE CASCADE)"
            ),
            (),
        )
        .unwrap();
    }
    let rows: Vec<String> = (1..=ROWS).map(|id| format!("({id}, {id}, 0)")).collect();
    db.execute(&format!("INSERT INTO p VALUES {}", rows.join(", ")), ())
        .unwrap();
    upsert(&db, "n = excluded.n + 1", 0);
    let unreferenced = upsert(&db, "n = excluded.n + 2", 0);
    upsert(&db, "code = excluded.code", 10_000);
    let referenced = upsert(&db, "code = excluded.code", 20_000);
    referenced.saturating_sub(unreferenced) as f64 / ROWS as f64
}

// Each bound is what the upsert paid with its keys' plan found once per
// statement; finding it for each conflict paid 27.15, 115.178 and 95.822

/// 13.18 measured, a graph of two columns
const ONE_CHILD: f64 = 13.5;

/// 39.332 measured, a graph of eight columns
const SEVEN_CHILDREN: f64 = 39.5;

/// 62.89 measured, a key into a column the statement changes
const INTO_CHANGED: f64 = 63.0;
