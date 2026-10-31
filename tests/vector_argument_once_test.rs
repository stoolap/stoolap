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

//! A TEXT vector a vector distance takes from a literal or a parameter is
//! read as a vector once per execution, not once per row

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;

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

const SHAPES: &[&str] = &[
    "SELECT COUNT(*) FROM t WHERE VEC_DISTANCE_L2(e, '[0.5, 0.5, 0.5]') < 1e9",
    "SELECT COUNT(*) FROM t WHERE VEC_DISTANCE_COSINE(e, $1) < 1e9",
    "SELECT COUNT(*) FROM t WHERE VEC_DISTANCE_IP($1, e) < 1e9",
    "SELECT COUNT(*) FROM t WHERE e <=> $1 < 1e9",
    "SELECT COUNT(*) FROM t WHERE e <=> '[0.5, 0.5, 0.5]' < 1e9",
    "SELECT COUNT(*) FROM t HAVING SUM(VEC_DISTANCE_L2(e, $1)) > 0",
    "SELECT COUNT(*) FROM t HAVING SUM(VEC_DISTANCE_IP('[0.5, 0.5, 0.5]', e)) < 0",
];

/// Arguments that read no row yet are evaluated per row: the function
/// reads them itself, at its own cost
const PER_ROW: &[&str] = &[
    "SELECT COUNT(*) FROM t HAVING SUM(VEC_DISTANCE_L2(e, \
     CASE WHEN $1 IS NOT NULL THEN '[0, 0, 0]' ELSE '[1, 1, 1]' END)) > 0",
    "SELECT COUNT(*) FROM t HAVING SUM(VEC_DISTANCE_L2(e, \
     CASE WHEN RANDOM() >= 0 THEN '[0.5, 0.5, 0.5]' ELSE '[1, 1, 1]' END)) > 0",
    "SELECT COUNT(*) FROM t WHERE e <=> \
     CASE WHEN $1 IS NOT NULL THEN '[0, 0, 0]' ELSE '[1, 1, 1]' END < 1e9",
];

/// Closing a database clears process-wide caches a counted run reads
static SERIAL: Mutex<()> = Mutex::new(());

/// The allocations one execution of `sql` takes over a table of `rows`,
/// in a database of the `test`'s own
fn allocations(test: &str, rows: i64, sql: &str) -> usize {
    let db = Database::open(&format!("memory://vector_argument_once_{test}_{rows}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, e VECTOR(3))", ())
        .unwrap();
    let values: Vec<String> = (1..=rows)
        .map(|id| format!("({id}, '[{id}.0, 1.0, 2.0]')"))
        .collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
    let stmt = db.prepare(sql).unwrap();
    let count = || -> i64 {
        match sql.contains("$1") {
            true => stmt.query_one(("[0.5, 0.5, 0.5]",)).unwrap(),
            false => stmt.query_one(()).unwrap(),
        }
    };
    assert_eq!(count(), rows, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    count();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_known_text_vector_is_read_once_per_execution() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut wrong = Vec::new();
    for sql in SHAPES {
        let short = allocations("once", 200, sql);
        let long = allocations("once", 2_000, sql);
        if long >= short + 200 {
            wrong.push(format!(
                "{sql}: {long} allocations over 2000 rows, {short} over 200"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_argument_read_per_row_costs_only_the_functions_own_read() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut wrong = Vec::new();
    for sql in PER_ROW {
        let short = allocations("per_row", 200, sql);
        let long = allocations("per_row", 2_000, sql);
        // Parsing the text takes a buffer of floats and one of bytes
        if long >= short + 2 * 1_800 + 200 {
            wrong.push(format!(
                "{sql}: {long} allocations over 2000 rows, {short} over 200"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
