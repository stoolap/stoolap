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

//! A name outside ASCII costs a row no more allocations than an ASCII one

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

/// The two statements differ only in their names' bytes
const LIMIT: f64 = 0.05;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

/// Allocations of one INSERT of `ROWS` rows from `first` into `table`
fn insert(db: &Database, table: &str, first: i64) -> usize {
    let values: Vec<String> = (first..first + ROWS)
        .map(|id| format!("({id}, {})", id % 50 + 1))
        .collect();
    let sql = format!("INSERT INTO {table} VALUES {}", values.join(", "));
    let before = allocations();
    db.execute(&sql, ()).unwrap();
    allocations() - before
}

fn extra_per_row(ascii: usize, other: usize) -> f64 {
    other.saturating_sub(ascii) as f64 / ROWS as f64
}

#[test]
fn a_foreign_key_into_a_lowercase_non_ascii_table_holds_its_keys_without_copies() {
    let db = Database::open("memory://name_case_alloc_fk").unwrap();
    for parent in ["pa", "\"pä\""] {
        db.execute(
            &format!("CREATE TABLE {parent} (id INTEGER PRIMARY KEY)"),
            (),
        )
        .unwrap();
        let ids: Vec<String> = (1..=50).map(|id| format!("({id})")).collect();
        db.execute(
            &format!("INSERT INTO {parent} VALUES {}", ids.join(", ")),
            (),
        )
        .unwrap();
    }
    db.execute(
        "CREATE TABLE ca (id INTEGER PRIMARY KEY, p INTEGER REFERENCES pa(id))",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE cb (id INTEGER PRIMARY KEY, p INTEGER REFERENCES \"pä\"(id))",
        (),
    )
    .unwrap();
    insert(&db, "ca", 1);
    insert(&db, "cb", 1);
    let ascii = insert(&db, "ca", 100_000);
    let other = insert(&db, "cb", 100_000);
    let per_row = extra_per_row(ascii, other);
    assert!(
        per_row <= LIMIT,
        "{per_row} more allocations per row for the parent \"pä\" than for pa"
    );
}

#[test]
fn a_cold_unique_check_on_a_sigma_column_copies_no_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    for (table, column) in [("ua", "sx"), ("ub", "\"Σx\"")] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, {column} INTEGER UNIQUE)"),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO {table} VALUES (1, -1)"), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let unique = |table: &str, first: i64| {
        let values: Vec<String> = (first..first + ROWS)
            .map(|id| format!("({id}, {id})"))
            .collect();
        let sql = format!("INSERT INTO {table} VALUES {}", values.join(", "));
        let before = allocations();
        db.execute(&sql, ()).unwrap();
        allocations() - before
    };
    let ascii = unique("ua", 10);
    let other = unique("ub", 10);
    let per_row = extra_per_row(ascii, other);
    assert!(
        per_row <= LIMIT,
        "{per_row} more allocations per row for the UNIQUE column \"Σx\" than for sx"
    );
}

#[test]
fn a_cold_unique_check_beside_a_column_differing_only_in_sigma_copies_no_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    for (table, first, unique) in [("va", "xa", "xb"), ("vb", "\"xΣ\"", "\"xσ\"")] {
        db.execute(
            &format!(
                "CREATE TABLE {table} (id INTEGER PRIMARY KEY, {first} INTEGER, {unique} INTEGER UNIQUE)"
            ),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO {table} VALUES (1, 1, -1)"), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let unique = |table: &str| {
        let values: Vec<String> = (10..10 + ROWS)
            .map(|id| format!("({id}, {id}, {id})"))
            .collect();
        let sql = format!("INSERT INTO {table} VALUES {}", values.join(", "));
        let before = allocations();
        db.execute(&sql, ()).unwrap();
        allocations() - before
    };
    let ascii = unique("va");
    let other = unique("vb");
    let per_row = extra_per_row(ascii, other);
    assert!(
        per_row <= LIMIT,
        "{per_row} more allocations per row for the UNIQUE column \"xσ\" beside \"xΣ\" than for xb"
    );
}

#[test]
fn a_cold_insert_into_a_table_keyed_by_a_sigma_column_copies_no_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    for (table, key) in [("ka", "sx"), ("kb", "\"Σx\"")] {
        db.execute(
            &format!("CREATE TABLE {table} ({key} INTEGER PRIMARY KEY, v INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO {table} VALUES (1, 1)"), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let ascii = insert(&db, "ka", 10);
    let other = insert(&db, "kb", 10);
    let per_row = extra_per_row(ascii, other);
    assert!(
        per_row <= LIMIT,
        "{per_row} more allocations per row for the key \"Σx\" than for sx"
    );
}
