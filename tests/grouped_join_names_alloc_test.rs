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

//! A grouped join, gathered for a subquery in its output or aggregated in
//! parts, names its joined columns without copying them for each run: what
//! one run allocates does not grow with the columns its sides hold

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
/// columns each, in a database named after `name`
fn allocations(name: &str, width: usize, sql: &str) -> usize {
    let db = Database::open(&format!("memory://grouped_join_names_alloc_{name}_{width}")).unwrap();
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
    let narrow = allocations("gathered", 2, sql);
    let wide = allocations("gathered", 100, sql);
    assert!(
        wide - narrow < 5 * added,
        "{narrow} allocations over narrow sides, {wide} over wide ones"
    );
}

#[test]
fn a_join_aggregated_in_parts_copies_no_column_name_for_each_run() {
    // The run reads about five allocations per column; a copy of the names
    // costs one more for each
    let added = 2 * (100 - 2);
    let sql = "SELECT l.k, COUNT(*) FROM l JOIN r ON r.k = l.k GROUP BY l.k";
    let narrow = allocations("parts", 2, sql);
    let wide = allocations("parts", 100, sql);
    assert!(
        wide - narrow < 11 * added / 2,
        "{narrow} allocations over narrow sides, {wide} over wide ones"
    );
}

/// The most allocations a run of the merged grouped join below makes once
/// its plan is held: 568 when its family's turn from the parts path was
/// first carried without one. Lower it when a change saves one
const MERGED_FAMILY_RUN: usize = 568;

#[test]
fn a_merged_grouped_join_turned_from_parts_allocates_nothing_to_say_so() {
    let db = Database::open("memory://grouped_join_names_alloc_merged_family").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO l SELECT g.value, g.value % 4 FROM generate_series(1, 40) g",
        "INSERT INTO r SELECT g.value, g.value % 4 FROM generate_series(1, 40) g",
        "INSERT INTO s SELECT g.value, g.value FROM generate_series(1, 5) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT k, COUNT(*), (SELECT MAX(s.a) FROM s) FROM l JOIN r USING (k) GROUP BY k";
    let query = db.prepare(sql).unwrap();
    assert_eq!(query.query(()).unwrap().count(), 4, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    query.query(()).unwrap().count();
    let run = ALLOCATIONS.with(Cell::get) - before;
    assert!(
        run <= MERGED_FAMILY_RUN,
        "{sql}: a held run allocated {run} times, at most {MERGED_FAMILY_RUN} before"
    );
}
