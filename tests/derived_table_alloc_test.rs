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
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counters are const
    // thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        BYTES.with(|b| b.set(b.get() + layout.size()));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        BYTES.with(|b| b.set(b.get() + new_size));
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
    per_run_counted(db, sql, allocations)
}

/// What `count` counts over one run of `sql`, prepared, after a first run
fn per_run_counted(db: &Database, sql: &str, count: fn() -> usize) -> usize {
    let statement = db.prepare(sql).unwrap();
    let run = || {
        for row in statement.query(()).unwrap() {
            row.unwrap();
        }
    };
    run();
    let before = count();
    for _ in 0..RUNS {
        run();
    }
    (count() - before) / RUNS
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

#[test]
fn a_single_use_cte_costs_what_its_derived_table_does() {
    let mut costs = Vec::new();
    for rows in [100, 400] {
        let db = Database::open(&format!("memory://single_use_cte_alloc_{rows}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
            .unwrap();
        db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
            .unwrap();
        db.execute(
            &format!("INSERT INTO t SELECT value, value % 7 FROM generate_series(1, {rows})"),
            (),
        )
        .unwrap();
        db.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
        // Computed rows, every one passing the WHERE
        let derived = per_run(
            &db,
            "SELECT * FROM (SELECT id + 1 AS id, a FROM t) AS s WHERE EXISTS (SELECT x FROM u)",
        );
        let cte = per_run(
            &db,
            "WITH s AS (SELECT id + 1 AS id, a FROM t) SELECT * FROM s WHERE EXISTS (SELECT x FROM u)",
        );
        costs.push((rows, derived, cte));
    }
    // The statement the CTE inlines into, built again each run
    assert!(
        costs.iter().all(|&(_, derived, cte)| cte <= derived + 32),
        "(rows, allocations over the derived table, the CTE): {costs:?}"
    );
}

#[test]
fn a_set_the_rows_test_is_shared_not_copied_per_run() {
    let bytes = || BYTES.with(Cell::get);
    // A number meets the numbers the TEXT members read as, built with the
    // set; a TIMESTAMP the timestamps they read as, taken on its first test
    for (probe, member) in [
        ("1", "{i}"),
        ("CAST('1001-01-01' AS TIMESTAMP)", "{year}-01-01"),
    ] {
        let mut costs = Vec::new();
        for members in [200, 4000] {
            let db = Database::open(&format!(
                "memory://rows_set_bytes_{members}_{}",
                probe.len()
            ))
            .unwrap();
            db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, s TEXT)", ())
                .unwrap();
            let insert = db.prepare("INSERT INTO m VALUES ($1, $2)").unwrap();
            for i in 1..=members {
                let text = member
                    .replace("{i}", &i.to_string())
                    .replace("{year}", &(1000 + i).to_string());
                insert.execute((i, text)).unwrap();
            }
            let per_run = per_run_counted(
                &db,
                &format!(
                    "SELECT q.x FROM (VALUES ({probe})) AS q(x) WHERE q.x IN (SELECT s FROM m)"
                ),
                bytes,
            );
            costs.push((members, per_run));
        }
        assert!(
            costs[1].1 < costs[0].1 + 4096,
            "{probe}: (TEXT members, bytes allocated per run): {costs:?}"
        );
    }
}
