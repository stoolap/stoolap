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

//! An INSERT with conflict handling or CHECK constraints compiles its CHECK
//! programs and ON DUPLICATE KEY UPDATE clause once, not per row or per run

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

const ROWS: i64 = 200;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

fn table(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v >= 0), s TEXT)",
        (),
    )
    .unwrap();
    db
}

const UPSERT: &str = "INSERT INTO t VALUES ($1, $2, 'x') ON DUPLICATE KEY UPDATE v = $2";

#[test]
fn test_upsert_of_new_rows_compiles_once() {
    let db = table("insert_plan_alloc_new_rows");
    db.execute(UPSERT, (-1, 0)).unwrap();
    let before = allocations();
    for id in 0..ROWS {
        db.execute(UPSERT, (id, id)).unwrap();
    }
    let per_row = (allocations() - before) / ROWS as usize;
    assert!(per_row <= 26, "{per_row} allocations per upserted row");
}

#[test]
fn test_upsert_of_existing_rows_compiles_once() {
    let db = table("insert_plan_alloc_existing_rows");
    for id in 0..ROWS {
        db.execute("INSERT INTO t VALUES ($1, 0, 'x')", (id,))
            .unwrap();
    }
    db.execute(UPSERT, (0, 1)).unwrap();
    let before = allocations();
    for id in 0..ROWS {
        db.execute(UPSERT, (id, id + 1)).unwrap();
    }
    let per_row = (allocations() - before) / ROWS as usize;
    assert!(per_row <= 34, "{per_row} allocations per updated row");
}

#[test]
fn test_transaction_insert_checks_once_per_statement() {
    let db = table("insert_plan_alloc_tx_rows");
    let values: Vec<String> = (0..ROWS).map(|id| format!("({id}, {id}, 'x')")).collect();
    let sql = format!("INSERT INTO t VALUES {}", values.join(", "));
    let mut tx = db.begin().unwrap();
    let before = allocations();
    tx.execute(&sql, ()).unwrap();
    let per_row = (allocations() - before) / ROWS as usize;
    tx.commit().unwrap();
    assert!(per_row <= 9, "{per_row} allocations per inserted row");
}

#[test]
fn test_values_subquery_answers_cost_a_bounded_count_per_row() {
    let db = table("insert_plan_alloc_values_answers");
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
    let per_run = |rows: i64| {
        let values: Vec<String> = (0..rows)
            .map(|id| format!("({id}, (SELECT x FROM u WHERE id = 1), 'x')"))
            .collect();
        let insert = db
            .prepare(&format!("INSERT INTO t VALUES {}", values.join(", ")))
            .unwrap();
        insert.execute(()).unwrap();
        db.execute("DELETE FROM t", ()).unwrap();
        let before = allocations();
        insert.execute(()).unwrap();
        let spent = allocations() - before;
        db.execute("DELETE FROM t", ()).unwrap();
        spent
    };
    let per_row = (per_run(32) - per_run(8)) / 24;
    assert!(per_row <= 21, "{per_row} allocations per answered row");
}

#[test]
fn test_prepared_insert_subqueries_keep_the_compiled_insert() {
    let db = table("insert_plan_alloc_prepared_subqueries");
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
    let per_run = |sql: &str, first: i64| {
        let insert = db.prepare(sql).unwrap();
        let run = |id: i64| {
            for row in insert.query((id,)).unwrap() {
                row.unwrap();
            }
        };
        run(first);
        let before = allocations();
        for id in first + 1..=first + ROWS {
            run(id);
        }
        (allocations() - before) / ROWS as usize
    };
    let plain = per_run("INSERT INTO t VALUES ($1, 1, 'x')", 0);
    let values = per_run(
        "INSERT INTO t VALUES ($1, (SELECT x FROM u WHERE id = 1), 'x')",
        1_000,
    );
    let plain_returning = per_run("INSERT INTO t VALUES ($1, 1, 'x') RETURNING id, v", 2_000);
    let returning = per_run(
        "INSERT INTO t VALUES ($1, 1, 'x') RETURNING id, (SELECT x FROM u WHERE u.id = t.v)",
        3_000,
    );
    assert!(
        values <= plain + 3 && returning <= plain_returning + 1,
        "{values} allocations with a VALUES subquery, {plain} without; \
         {returning} with a RETURNING subquery, {plain_returning} without"
    );
}
