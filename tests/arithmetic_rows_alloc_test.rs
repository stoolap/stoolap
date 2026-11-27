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

//! Arithmetic over number columns as an aggregate's argument, computed by
//! the table over hot rows, is read in place: what the grouping allocates
//! does not grow with the rows

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(not(feature = "jemalloc"), not(feature = "test-filedb")))]

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
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

/// The allocations of one run of `sql`, prepared, over `rows` rows of one
/// group
fn allocations(rows: i64, sql: &str) -> usize {
    let db = Database::open(&format!("memory://arithmetic_rows_alloc_{rows}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, k INTEGER, v FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT g.value, 1, g.value, g.value * 0.5 \
             FROM generate_series(1, {rows}) g"
        ),
        (),
    )
    .unwrap();
    let query = db.prepare(sql).unwrap();
    let run = || query.query(()).unwrap().count();
    assert_eq!(run(), 1, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

/// The allocations of one failing run of `sql`, prepared, over `rows` rows
/// whose `k` doubled overflows, hot or sealed in `dir`, in database `n`
fn failing_allocations(n: usize, rows: i64, sql: &str, dir: Option<&std::path::Path>) -> usize {
    let dsn = match dir {
        None => format!("memory://arithmetic_rows_alloc_failing_{n}_{rows}"),
        Some(dir) => format!(
            "file://{}/failing_{n}_{rows}?checkpoint_interval=0",
            dir.display()
        ),
    };
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, k INTEGER, v FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT g.value, 1, 9223372036854775807, 0.5 \
             FROM generate_series(1, {rows}) g"
        ),
        (),
    )
    .unwrap();
    if dir.is_some() {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let query = db.prepare(sql).unwrap();
    let run = || {
        query
            .query(())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .is_err()
    };
    assert!(run(), "{sql}: must fail");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn arithmetic_that_fails_stops_at_its_first_row() {
    let dir = tempfile::tempdir().unwrap();
    for sealed in [None, Some(dir.path())] {
        for (n, sql) in [
            "SELECT g, SUM(k * 2), SUM(v * 2) FROM t GROUP BY g",
            "SELECT SUM(k * 2), SUM(v * 2) FROM t",
        ]
        .into_iter()
        .enumerate()
        {
            let (few, many) = (
                failing_allocations(n, 100, sql, sealed),
                failing_allocations(n, 800, sql, sealed),
            );
            assert_eq!(
                few,
                many,
                "{sql}, sealed {}: {few} allocations over 100 rows, {many} over 800",
                sealed.is_some()
            );
        }
    }
}

#[test]
fn arithmetic_arguments_allocate_alike_for_few_and_many_rows() {
    for sql in [
        "SELECT g, SUM(v * k), AVG(k - 1), MAX(-k) FROM t GROUP BY g",
        "SELECT SUM(v * k), COUNT(k / 2) FROM t",
    ] {
        let (few, many) = (allocations(100, sql), allocations(800, sql));
        assert_eq!(
            few, many,
            "{sql}: {few} allocations over 100 rows, {many} over 800"
        );
    }
}
