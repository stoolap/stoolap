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

//! Preparing a subquery block allocates the same for a qualified name as
//! for a bare one

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

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
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
fn a_qualified_name_prepares_as_a_bare_one_does() {
    let db = open("block_prepare_qualified");
    let terms = |term: &str| vec![term; 8].join(" AND ");
    let qualified = first_run(
        &db,
        &format!(
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE {})",
            terms("u.v < t.a")
        ),
    );
    let bare = first_run(
        &db,
        &format!(
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE {})",
            terms("v < a")
        ),
    );
    assert_eq!(qualified, bare, "eight qualified names against eight bare");
}
