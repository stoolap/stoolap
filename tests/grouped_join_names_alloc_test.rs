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

//! A grouped join whose output holds a subquery names its joined columns
//! without copying them for each run: what one run allocates does not grow
//! with the columns its sides hold

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

/// The allocations of one run of `sql`, prepared, over sides of `width`
/// columns each
fn allocations(width: usize, sql: &str) -> usize {
    let db = Database::open(&format!("memory://grouped_join_names_alloc_{width}")).unwrap();
    let extra: String = (0..width).map(|i| format!(", c{i} INTEGER")).collect();
    for table in ["l", "r"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER{extra})"),
            (),
        )
        .unwrap();
        db.execute(
            &format!("INSERT INTO {table} (id, k) SELECT g.value, g.value % 4 FROM generate_series(1, 40) g"),
            (),
        )
        .unwrap();
    }
    let query = db.prepare(sql).unwrap();
    let run = || query.query(()).unwrap().count();
    assert_eq!(run(), 4, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_grouped_join_copies_no_column_name_for_each_run() {
    // Each side gains this many columns; copying the names costs one more
    // allocation for each of them on top of what the run reads per column
    let added = 2 * (100 - 2);
    let sql = "SELECT l.k, COUNT(*), (SELECT 1) FROM l JOIN r ON r.id = l.id GROUP BY l.k";
    let narrow = allocations(2, sql);
    let wide = allocations(100, sql);
    assert!(
        wide - narrow < 5 * added,
        "{narrow} allocations over narrow sides, {wide} over wide ones"
    );
}
