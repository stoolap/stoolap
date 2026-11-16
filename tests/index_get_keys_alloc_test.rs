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

//! A key read with other indexed equalities to take its rows by allocates
//! no more than one with a single key, whether the first key settles it or
//! the others are counted; a sealed read choosing among several indexes
//! allocates no more for each further volume than one asking a single index

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::Value;
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

const RUNS: usize = 20;

/// The allocations per read of `sql` with `params`
fn per_read(db: &Database, sql: &str, params: &[i64]) -> usize {
    let statement = db.prepare(sql).unwrap();
    let params: Vec<Value> = params.iter().map(|&p| Value::Integer(p)).collect();
    let read = || {
        let mut rows = 0;
        for row in statement.query(params.clone()).unwrap() {
            row.unwrap();
            rows += 1;
        }
        assert_eq!(rows, 1, "{sql}");
    };
    read();
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..RUNS {
        read();
    }
    (ALLOCATIONS.with(Cell::get) - before) / RUNS
}

#[test]
fn a_key_read_with_other_keys_allocates_as_one_with_a_single_key() {
    let db = Database::open("memory://index_get_keys_alloc").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER, d INTEGER)",
        "INSERT INTO t SELECT value, value % 2, value, value % 3, value % 5 FROM generate_series(1, 600)",
        "CREATE INDEX t_a ON t (a)",
        "CREATE INDEX t_b ON t (b)",
        "CREATE INDEX t_c ON t (c)",
        "CREATE INDEX t_d ON t (d)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // Row 301: a = 1, b = 301, c = 1, d = 1
    let single = per_read(&db, "SELECT id FROM t WHERE b = $1", &[301]);
    let settled = per_read(
        &db,
        "SELECT id FROM t WHERE b = $1 AND a = $2 AND c = $3 AND d = $4",
        &[301, 1, 1, 1],
    );
    let counted = per_read(
        &db,
        "SELECT id FROM t WHERE a = $2 AND c = $3 AND d = $4 AND b = $1",
        &[301, 1, 1, 1],
    );
    assert!(
        settled <= single && counted <= single,
        "allocations per read, one key {single}, first key settles {settled}, others counted {counted}"
    );
}

#[test]
fn a_sealed_choice_among_indexes_allocates_nothing_more_per_volume() {
    // Three indexes, and nine, more than a small inline buffer holds
    for indexes in [3, 9] {
        let dir = tempfile::tempdir().unwrap();
        // Each checkpoint seals the rows inserted since into a volume of its own
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=3600&compact_threshold=64",
            dir.path().display()
        ))
        .unwrap();
        // t indexes every column; u, the same rows, only two of them, the
        // fewest a choice among indexes asks
        let columns: Vec<String> = (1..=indexes).map(|c| format!("c{c} INTEGER")).collect();
        for table in ["t", "u"] {
            db.execute(
                &format!(
                    "CREATE TABLE {table} (id INTEGER PRIMARY KEY, {})",
                    columns.join(", ")
                ),
                (),
            )
            .unwrap();
        }
        for c in 1..=indexes {
            db.execute(&format!("CREATE INDEX t_c{c} ON t (c{c})"), ())
                .unwrap();
        }
        for c in 1..=2 {
            db.execute(&format!("CREATE INDEX u_c{c} ON u (c{c})"), ())
                .unwrap();
        }
        // Column cN holds value % (N + 1): every volume holds each key, so no
        // bloom filter or zone map leaves one out before its side file is asked
        let values: Vec<String> = (1..=indexes)
            .map(|c| format!("value % {}", c + 1))
            .collect();
        let seal = |batch: usize| {
            let (from, to) = (batch * 500 + 1, batch * 500 + 500);
            for table in ["t", "u"] {
                db.execute(
                    &format!(
                        "INSERT INTO {table} SELECT value, {} FROM generate_series({from}, {to})",
                        values.join(", ")
                    ),
                    (),
                )
                .unwrap();
            }
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        };
        let terms: Vec<String> = (1..=indexes).map(|c| format!("c{c} = ${c}")).collect();
        let terms = terms.join(" AND ");
        let ones = vec![1; indexes];
        let reads = |db: &Database| {
            (
                per_read(db, &format!("SELECT COUNT(*) FROM u WHERE {terms}"), &ones),
                per_read(db, &format!("SELECT COUNT(*) FROM t WHERE {terms}"), &ones),
            )
        };
        (0..3).for_each(seal);
        let (two_three, all_three) = reads(&db);
        (3..6).for_each(seal);
        let (two_six, all_six) = reads(&db);
        assert!(
            all_six.saturating_sub(all_three) <= two_six.saturating_sub(two_three),
            "allocations per read over three and six volumes, two indexes {two_three} and \
             {two_six}, {indexes} indexes {all_three} and {all_six}"
        );
    }
}
