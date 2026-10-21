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

//! The projection that computes SELECT items beside an ORDER BY or DISTINCT
//! ON finds each item's source once per statement: per row it allocates only
//! the row it builds, as a projection of bare columns does

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

/// The allocations of `sql` over a table of `rows` rows, read to the end
fn allocations(sql: &str, rows: i64) -> usize {
    let db = Database::open(&format!("memory://projection_evaluator_alloc_{rows}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    let insert = db.prepare("INSERT INTO t VALUES ($1, $2, $3)").unwrap();
    db.execute("BEGIN", ()).unwrap();
    for id in 1..=rows {
        insert.execute((id, id % 7, rows - id)).unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    let _ = db.query(sql, ()).unwrap().count();
    let before = ALLOCATIONS.with(Cell::get);
    let _ = db.query(sql, ()).unwrap().count();
    ALLOCATIONS.with(Cell::get) - before
}

/// What 1000 more rows cost `sql`
fn per_thousand_rows(sql: &str) -> usize {
    allocations(sql, 2000) - allocations(sql, 1000)
}

#[test]
fn computed_items_allocate_per_row_as_bare_columns_do() {
    for (sql, bare_sql) in [
        (
            "SELECT a + 1 FROM t ORDER BY b",
            "SELECT id FROM t ORDER BY b",
        ),
        (
            "SELECT t.a, a * 2 AS d FROM t ORDER BY b",
            "SELECT id FROM t ORDER BY b",
        ),
        (
            "SELECT DISTINCT ON (a + 0) a + 1 FROM t ORDER BY a + 0, b",
            "SELECT DISTINCT ON (a) a FROM t ORDER BY a, b",
        ),
    ] {
        let bare = per_thousand_rows(bare_sql);
        let computed = per_thousand_rows(sql);
        assert!(
            computed <= bare,
            "{sql}: 1000 more rows cost {computed} allocations, a bare column projection {bare}"
        );
    }
}

#[test]
fn an_extra_order_by_column_costs_a_query_no_list_of_its_own() {
    let ordered = allocations("SELECT id FROM t ORDER BY b", 10);
    let plain = allocations("SELECT id, b FROM t ORDER BY b", 10);
    // 18 on main 24fbda2, before the projection was split
    assert!(
        ordered - plain <= 18,
        "keeping b for the sort costs {} allocations beyond selecting it",
        ordered - plain
    );
}
