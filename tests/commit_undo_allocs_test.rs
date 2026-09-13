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

//! Recording the index undo log at commit must not allocate for a
//! single-row statement; this pins the allocation count of a prepared
//! INSERT so the commit path cannot quietly grow again.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use stoolap::Database;

static ALLOCS: AtomicU64 = AtomicU64::new(0);

/// The counter is process-wide, so the tests in this file never overlap
static SERIAL: Mutex<()> = Mutex::new(());

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations per prepared single-row INSERT into a table with only its
/// primary key index, measured after a warm-up
fn allocs_per_insert(db: &Database) -> f64 {
    let stmt = db.prepare("INSERT INTO t VALUES (?, ?, ?)").unwrap();
    let mut next = 0i64;
    for _ in 0..200 {
        stmt.execute((next, "k", next)).unwrap();
        next += 1;
    }
    let reps = 1000i64;
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..reps {
        stmt.execute((next, "k", next)).unwrap();
        next += 1;
    }
    (ALLOCS.load(Ordering::Relaxed) - before) as f64 / reps as f64
}

#[test]
fn single_row_insert_commit_records_its_undo_without_allocating() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://commit_undo_allocs").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    let per_insert = allocs_per_insert(&db);
    eprintln!("allocations per prepared single-row INSERT: {per_insert:.2}");
    // 20 with the undo log recorded through fresh vectors, 13 with the
    // index keys moved into it; anything above 16 means the commit path
    // allocates per statement again
    assert!(
        per_insert <= 16.0,
        "a prepared single-row INSERT allocates {per_insert:.1} times"
    );
}
