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

//! A LIMIT subquery read through its plan costs a few allocations more
//! than the same LIMIT written as a literal

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

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

#[test]
fn test_limit_subquery_allocates_a_few_more_than_its_literal() {
    let db = Database::open("memory://limit_subquery_alloc").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3), (4)", ())
        .unwrap();
    let per_run = |sql: &str| {
        let statement = db.prepare(sql).unwrap();
        statement.query(()).unwrap().count();
        let before = allocations();
        for _ in 0..50 {
            assert_eq!(statement.query(()).unwrap().count(), 2);
        }
        (allocations() - before) / 50
    };
    let literal = per_run("SELECT id FROM t ORDER BY id LIMIT 2");
    let read = per_run("SELECT id FROM t ORDER BY id LIMIT (SELECT 2)");
    assert!(
        read <= literal + 8,
        "{read} allocations per run, {literal} for the literal"
    );
}
