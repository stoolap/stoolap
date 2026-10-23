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

//! A statement over a derived table or a view whose WHERE holds a
//! subquery costs what the same statement over the table costs, plus a
//! bounded number of allocations per run, whatever the number of rows

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

const RUNS: usize = 50;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

/// The allocations of one run of `sql`, prepared, after a first run
fn per_run(db: &Database, sql: &str) -> usize {
    let statement = db.prepare(sql).unwrap();
    let run = || {
        for row in statement.query(()).unwrap() {
            row.unwrap();
        }
    };
    run();
    let before = allocations();
    for _ in 0..RUNS {
        run();
    }
    (allocations() - before) / RUNS
}

#[test]
fn a_derived_table_or_view_costs_what_its_table_does_plus_a_bounded_count() {
    let mut costs = Vec::new();
    for rows in [100, 400] {
        let db = Database::open(&format!("memory://derived_table_alloc_{rows}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
            .unwrap();
        db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
            .unwrap();
        db.execute(
            &format!("INSERT INTO t SELECT value, value % 7 FROM generate_series(1, {rows})"),
            (),
        )
        .unwrap();
        db.execute("INSERT INTO u VALUES (1, 1), (2, 2)", ())
            .unwrap();
        let table = per_run(
            &db,
            "SELECT s.id FROM t AS s WHERE s.a IN (SELECT x FROM u)",
        );
        let derived = per_run(
            &db,
            "SELECT s.id FROM (SELECT id, a FROM t) AS s WHERE s.a IN (SELECT x FROM u)",
        );
        db.execute("CREATE VIEW tv AS SELECT id, a FROM t", ())
            .unwrap();
        let view = per_run(
            &db,
            "SELECT s.id FROM tv AS s WHERE s.a IN (SELECT x FROM u)",
        );
        costs.push((rows, table, derived, view));
    }
    // The set conjunct's program, bound once per run to the members read
    assert!(
        costs
            .iter()
            .all(|&(_, table, derived, view)| derived <= table + 16 && view <= table + 16),
        "(rows, allocations over the table, the derived table, the view): {costs:?}"
    );
}
