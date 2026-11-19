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

//! A scan of sealed rows that stops at its first row unpacks the ids it
//! reads, not its row group's, when a tombstone makes it test each id

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

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
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + new_size));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

/// Bytes the first run of an EXISTS over `table` allocates
fn exists_allocates(db: &Database, table: &str) -> usize {
    let before = ALLOCATED.with(Cell::get);
    let sql = format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE b + 1 > 0)");
    assert_eq!(db.query(&sql, ()).unwrap().count(), 1);
    ALLOCATED.with(Cell::get) - before
}

#[test]
fn an_exists_over_sealed_rows_with_a_tombstone_unpacks_few_ids() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    for table in ["kept", "deleted"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, b FLOAT)"),
            (),
        )
        .unwrap();
        let insert = db
            .prepare(&format!("INSERT INTO {table} VALUES ($1, $2)"))
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        for i in 0..70_000i64 {
            insert.execute((i * 3, i as f64)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("DELETE FROM deleted WHERE id = 30000", ())
        .unwrap();
    for table in ["kept", "deleted"] {
        let warm = format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE b + 2 > 0)");
        assert_eq!(db.query(&warm, ()).unwrap().count(), 1);
    }
    let kept = exists_allocates(&db, "kept");
    let deleted = exists_allocates(&db, "deleted");
    assert!(
        deleted < kept + 64 * 1024,
        "with a tombstone the EXISTS allocated {deleted} bytes, without {kept}"
    );
}
