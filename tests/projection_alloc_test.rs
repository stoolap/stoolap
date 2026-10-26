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

//! A SELECT item that reads only parameters is computed once per execution,
//! not once per row, and a short concatenation allocates nothing however
//! many parts it joins

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

const ROWS: usize = 400;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute(
        &format!("INSERT INTO t SELECT value, 'x' FROM generate_series(1, {ROWS})"),
        (),
    )
    .unwrap();
    db
}

/// The allocations one execution of `sql` with `params` takes, after one
/// that warms its plan
fn allocations(db: &Database, sql: &str, params: &[Value]) -> usize {
    let stmt = db.prepare(sql).unwrap();
    let run = || {
        let mut rows = 0;
        for row in stmt.query(params.to_vec()).unwrap() {
            row.unwrap();
            rows += 1;
        }
        assert_eq!(rows, ROWS);
    };
    run();
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_concatenation_of_parameters_is_joined_once_per_execution() {
    let db = open("projection_alloc_params");
    let long = |c: char| Value::text(c.to_string().repeat(40));
    let params = [long('a'), long('b')];
    // Streamed, and sorted, which projects by the materialized path
    for order in ["", " ORDER BY id"] {
        let joined = allocations(&db, &format!("SELECT id, $1 || $2 FROM t{order}"), &params);
        let constant = allocations(&db, &format!("SELECT id, 1 FROM t{order}"), &[]);
        assert!(
            joined < constant + ROWS / 2,
            "{joined} allocations joining two parameters{order}, {constant} for a constant"
        );
    }
}

#[test]
fn a_short_concatenation_of_many_parts_allocates_nothing_per_row() {
    let db = open("projection_alloc_parts");
    let parts = allocations(
        &db,
        "SELECT id, s || s || s || s || s || s || s || s || s FROM t",
        &[],
    );
    let one = allocations(&db, "SELECT id, s || s FROM t", &[]);
    assert!(
        parts < one + ROWS / 2,
        "{parts} allocations joining nine short parts, {one} joining two"
    );
}
