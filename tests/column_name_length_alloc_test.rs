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

//! A prepared statement's run allocates the same whether the column it
//! compares has a name of fifteen bytes, held inline, or of sixteen

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

const FIFTEEN: &str = "abcdefghijklmno";
const SIXTEEN: &str = "abcdefghijklmnop";
const RUNS: usize = 8;

/// Closing a database clears the process-wide program and classification
/// caches, which another test's measured runs read
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The allocations of `RUNS` warm runs of `sql`, `{c}` standing for a
/// column named `column`, over t, u and f holding three rows each; f's
/// column is BOOLEAN, true on even ids
fn runs(test: &str, column: &str, sql: &str) -> usize {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open(&format!("memory://{test}_{}", column.len())).unwrap();
    db.execute(
        &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, {column} INTEGER)"),
        (),
    )
    .unwrap();
    db.execute(
        &format!("CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER, {column} INTEGER)"),
        (),
    )
    .unwrap();
    db.execute(
        &format!("CREATE TABLE f (id INTEGER PRIMARY KEY, {column} BOOLEAN)"),
        (),
    )
    .unwrap();
    for i in 1..=3 {
        db.execute(&format!("INSERT INTO t VALUES ({i}, {i})"), ())
            .unwrap();
        db.execute(&format!("INSERT INTO u VALUES ({i}, {i}, {i})"), ())
            .unwrap();
        db.execute(&format!("INSERT INTO f VALUES ({i}, {})", i % 2 == 0), ())
            .unwrap();
    }
    let statement = db.prepare(&sql.replace("{c}", column)).unwrap();
    let run = || {
        let rows = statement.query((2,)).unwrap().count();
        assert!(rows > 0, "{sql}");
    };
    run();
    run();
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..RUNS {
        run();
    }
    ALLOCATIONS.with(Cell::get) - before
}

fn assert_same_for_both_lengths(test: &str, sql: &str) {
    assert_eq!(
        runs(test, SIXTEEN, sql),
        runs(test, FIFTEEN, sql),
        "a sixteen-byte column name allocates as a fifteen-byte one: {sql}"
    );
}

#[test]
fn a_prepared_where_allocates_alike_for_both_lengths() {
    assert_same_for_both_lengths("name_len_where", "SELECT id FROM t WHERE {c} = $1");
    assert_same_for_both_lengths(
        "name_len_runner",
        "SELECT id, (SELECT 1) FROM t WHERE {c} = $1",
    );
    assert_same_for_both_lengths(
        "name_len_two",
        "SELECT id, (SELECT 1) FROM t WHERE {c} = $1 AND id > 0",
    );
}

#[test]
fn a_boolean_check_allocates_alike_for_both_lengths() {
    for check in ["IS TRUE", "IS FALSE", "IS NOT TRUE", "IS NOT FALSE"] {
        assert_same_for_both_lengths(
            &format!("name_len_{}", check.replace(' ', "_").to_lowercase()),
            &format!("SELECT id FROM f WHERE {{c}} {check} AND id >= $1"),
        );
    }
}

#[test]
fn a_block_pushdown_allocates_alike_for_both_lengths() {
    assert_same_for_both_lengths(
        "name_len_block",
        "SELECT id, (SELECT v FROM u WHERE u.{c} = t.id) FROM t WHERE id >= $1",
    );
}
