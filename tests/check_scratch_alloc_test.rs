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

//! A column CHECK costs a statement a fixed number of allocations, not one
//! per row it checks

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.with(|c| c.set(c.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.with(|c| c.set(c.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

fn allocations(db: &Database, sql: &str) -> usize {
    let before = CALLS.with(Cell::get);
    db.execute(sql, ()).unwrap();
    CALLS.with(Cell::get) - before
}

/// Allocations a CHECK on `v` adds to an INSERT and an UPDATE of `rows`
/// rows; the UPDATE compares against a CHECK on the column it leaves alone
fn check_cost(rows: usize) -> (isize, isize) {
    let values: Vec<String> = (1..=rows).map(|i| format!("({i}, {i}, {i})")).collect();
    let values = values.join(", ");
    let mut insert = [0isize; 3];
    let mut update = [0isize; 3];
    let columns = [
        "v INTEGER, w INTEGER",
        "v INTEGER CHECK (v > 0), w INTEGER",
        "v INTEGER, w INTEGER CHECK (w > 0)",
    ];
    for (slot, columns) in columns.iter().enumerate() {
        let db = Database::open(&format!("memory://check_scratch_alloc_{rows}_{slot}")).unwrap();
        db.execute(
            &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, {columns})"),
            (),
        )
        .unwrap();
        insert[slot] = allocations(&db, &format!("INSERT INTO t VALUES {values}")) as isize;
        update[slot] = allocations(&db, "UPDATE t SET v = v + 1") as isize;
    }
    (insert[1] - insert[0], update[1] - update[2])
}

/// Allocations of a transaction INSERT, a general UPDATE and a PK UPDATE
/// on a table of `width` columns without a CHECK
fn unchecked_cost(width: usize) -> [usize; 3] {
    let db = Database::open(&format!("memory://check_scratch_unchecked_{width}")).unwrap();
    let columns: Vec<String> = (0..width).map(|i| format!("c{i} INTEGER")).collect();
    db.execute(
        &format!(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, {})",
            columns.join(", ")
        ),
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t (id, c0) VALUES (1, 1)", ())
        .unwrap();
    let mut cost = [0; 3];
    for _ in 0..2 {
        let mut tx = db.begin().unwrap();
        let before = CALLS.with(Cell::get);
        tx.execute("INSERT INTO t (id, c0) VALUES (2, 2)", ())
            .unwrap();
        cost[0] = CALLS.with(Cell::get) - before;
        tx.rollback().unwrap();
        cost[1] = allocations(&db, "UPDATE t SET c0 = c0 + 1 WHERE c0 > 0");
        cost[2] = allocations(&db, "UPDATE t SET c0 = 5 WHERE id = 1");
    }
    cost
}

#[test]
fn a_table_without_a_check_pays_nothing_per_column_for_checks() {
    assert_eq!(
        unchecked_cost(2),
        unchecked_cost(42),
        "allocations of [tx INSERT, UPDATE, PK UPDATE] grew with the column count"
    );
}

#[test]
fn a_check_allocates_per_statement_not_per_row() {
    let (insert_few, update_few) = check_cost(10);
    let (insert_many, update_many) = check_cost(1010);
    assert!(
        insert_many - insert_few < 100,
        "INSERT: the CHECK cost {insert_few} allocations for 10 rows, {insert_many} for 1010"
    );
    assert!(
        update_many - update_few < 100,
        "UPDATE: the CHECK cost {update_few} allocations for 10 rows, {update_many} for 1010"
    );
}
