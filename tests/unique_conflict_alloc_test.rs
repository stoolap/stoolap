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

//! An upsert whose conflict is resolved does not format the key values of
//! the unique violation it never shows

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

const KEYS: i64 = 200;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

/// Allocations per row of `sql` run once for each of `KEYS` keys, each of
/// which the table already holds, hot or sealed into a volume
fn per_row(cold: bool, sql: &str) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY AUTO_INCREMENT, ts INTEGER NOT NULL, \
         name TEXT NOT NULL, field TEXT NOT NULL, v FLOAT, UNIQUE(ts, name, field))",
        (),
    )
    .unwrap();
    let insert = db
        .prepare("INSERT INTO m (ts, name, field, v) VALUES ($1, 'cpu', 'usage', $2)")
        .unwrap();
    for ts in 0..KEYS {
        insert.execute((ts, 1.0)).unwrap();
    }
    if cold {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(db.engine().volume_stats().len(), 1);
    }
    let statement = db.prepare(sql).unwrap();
    statement.execute((KEYS + 1_000, 2.0)).unwrap();
    let before = allocations();
    for ts in 0..KEYS {
        statement.execute((ts, 2.0)).unwrap();
    }
    (allocations() - before) / KEYS as usize
}

const DO_UPDATE: &str = "INSERT INTO m (ts, name, field, v) VALUES ($1, 'cpu', 'usage', $2) \
     ON CONFLICT (ts, name, field) DO UPDATE SET v = excluded.v";
const DO_NOTHING: &str = "INSERT INTO m (ts, name, field, v) VALUES ($1, 'cpu', 'usage', $2) \
     ON CONFLICT DO NOTHING";
const FRESH: &str = "INSERT INTO m (ts, name, field, v) VALUES ($1 + 100000, 'cpu', 'usage', $2)";

#[test]
fn conflicts_resolved_by_an_upsert_allocate_no_message() {
    for (name, cold, sql, most) in [
        ("hot do update", false, DO_UPDATE, 128),
        ("hot do nothing", false, DO_NOTHING, 28),
        ("cold do update", true, DO_UPDATE, 161),
        ("cold do nothing", true, DO_NOTHING, 26),
    ] {
        let allocated = per_row(cold, sql);
        assert!(
            allocated <= most,
            "{name}: {allocated} per row, at most {most}"
        );
    }
}

#[test]
fn an_insert_without_a_conflict_allocates_as_before() {
    for (name, cold) in [("hot fresh insert", false), ("cold fresh insert", true)] {
        let allocated = per_row(cold, FRESH);
        assert!(allocated <= 40, "{name}: {allocated} per row, at most 40");
    }
}
