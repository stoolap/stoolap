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

//! Preparing a join reads its sides' column names from their schemas,
//! copying none of them for the columns it reads

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

use stoolap::Database;

// Dropping a database clears the process-wide classification cache a
// measured run reads, so the tests here run one at a time
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

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

/// t with `columns` more columns after id and a, and u
fn open(name: &str, columns: usize) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    let more: String = (0..columns).map(|i| format!(", c{i} INTEGER")).collect();
    db.execute(
        &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER{more})"),
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t (id, a) VALUES (1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 5)", ()).unwrap();
    db
}

/// The allocations of the first run of `sql`, which parses and prepares it,
/// after the same statement with another trailing conjunct warmed the database
fn first_run(db: &Database, sql: &str) -> usize {
    db.query(&format!("{sql} AND 0 = 0"), ()).unwrap().count();
    let before = ALLOCATIONS.with(Cell::get);
    let rows = db.query(sql, ()).unwrap().count();
    assert_eq!(rows, 1, "{sql}");
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_join_prepares_its_column_mask_without_copying_names() {
    let _guard = serial();
    let sql = "SELECT t.a, u.v FROM t JOIN u ON u.id = t.id WHERE t.a > 0";
    let narrow = first_run(&open("join_needed_prepare_narrow", 0), sql);
    let wide = first_run(&open("join_needed_prepare_wide", 64), sql);
    // The join's scope and blocks take twelve per column; the mask takes none
    assert!(
        wide <= narrow + 12 * 64 + 8,
        "{wide} allocations over 66 columns, {narrow} over 2"
    );
}
