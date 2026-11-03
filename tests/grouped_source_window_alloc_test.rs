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

//! A grouped statement computing windows alone over a source's rows copies
//! the source's column names as the runner's window step does: once a run,
//! its window plan made with the statement

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(not(feature = "jemalloc"), not(feature = "test-filedb")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

/// The length of each column name the statements read only through the
/// source, which no other allocation of their runs has
const NAME_LENGTH: usize = 23;

struct NameCounting;

thread_local! {
    static NAMES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for NameCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == NAME_LENGTH {
            NAMES.with(|n| n.set(n.get() + 1));
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static COUNTING: NameCounting = NameCounting;

/// The names of `NAME_LENGTH` bytes one run of `sql`, prepared, allocates
fn names_copied(db: &Database, sql: &str) -> usize {
    let query = db.prepare(sql).unwrap();
    let run = || query.query(()).unwrap().count();
    assert_eq!(run(), 3, "{sql}");
    let before = NAMES.with(Cell::get);
    run();
    NAMES.with(Cell::get) - before
}

#[test]
fn a_window_over_a_source_copies_its_column_names_once() {
    let db = Database::open("memory://grouped_source_window_alloc").unwrap();
    let names: Vec<String> = (0..8)
        .map(|i| format!("unread_column_number_{i:02}"))
        .collect();
    assert!(names.iter().all(|name| name.len() == NAME_LENGTH));
    let row = |n: i64| {
        let values: Vec<String> = (0..8).map(|i| (n * 10 + i).to_string()).collect();
        format!("({n}, {})", values.join(", "))
    };
    let source = format!(
        "(VALUES {}, {}, {}) v(k, {})",
        row(1),
        row(2),
        row(3),
        names.join(", ")
    );
    let family = format!("SELECT k, ROW_NUMBER() OVER (ORDER BY k), (SELECT 5) FROM {source}");
    let runner = format!("SELECT k, ROW_NUMBER() OVER (ORDER BY k), 5 FROM {source}");
    let (blocks, plain) = (names_copied(&db, &family), names_copied(&db, &runner));
    assert_eq!(
        blocks, plain,
        "the names a run copies: {blocks} with a subquery, {plain} without"
    );
}
