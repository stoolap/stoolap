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

//! A DELETE on a file table with no sealed volume does not gather the hot
//! row ids to skip in volumes it will not read, counted in the bytes the
//! statement allocates on its thread

// The mimalloc feature sets the library's own global allocator
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + layout.size()));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + new_size));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

#[test]
fn a_delete_without_volumes_does_not_gather_the_hot_row_ids() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, age INTEGER, active BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_age ON t(age)", ()).unwrap();
    let insert = db.prepare("INSERT INTO t VALUES ($1, $2, true)").unwrap();
    db.execute("BEGIN", ()).unwrap();
    for id in 1..=10_000i64 {
        insert.execute((id, id % 60)).unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    let delete = db
        .prepare("DELETE FROM t WHERE age >= $1 AND age <= $2 AND active = true")
        .unwrap();
    assert!(delete.execute((25_i64, 26_i64)).unwrap() > 0);
    let before = ALLOCATED.with(Cell::get);
    let deleted = delete.execute((25_i64, 26_i64)).unwrap();
    let allocated = ALLOCATED.with(Cell::get) - before;
    assert_eq!(deleted, 0);
    assert!(
        allocated < 32 * 1024,
        "a DELETE of no rows over 10,000 hot rows allocated {allocated} bytes"
    );
}
