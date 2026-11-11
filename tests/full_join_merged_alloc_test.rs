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

//! A full join's merged column is held in room each joined row is made
//! with, so it costs no allocation per row over the ON join it stands for,
//! on the streaming join and on the parallel one

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use stoolap::Database;

// The parallel join allocates on its worker threads, so the count is the
// process's, and the tests here run one at a time
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the counter is an atomic,
    // which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The allocations of a run of `sql` after a first one warmed it
fn run(db: &Database, sql: &str, rows: usize) -> usize {
    db.query(sql, ()).unwrap().count();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(db.query(sql, ()).unwrap().count(), rows, "{sql}");
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

/// The allocations of the full join USING over `n` rows a side, half of
/// them matched, `t`'s key of type `key`, and of the ON join it stands for
fn both(name: &str, n: usize, key: &str) -> (usize, usize) {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        format!("CREATE TABLE t (k {key}, a INTEGER)"),
        "CREATE TABLE u (k INTEGER, b INTEGER)".to_string(),
        format!("INSERT INTO t SELECT g.value, g.value FROM generate_series(1, {n}) g"),
        format!(
            "INSERT INTO u SELECT g.value + {}, g.value FROM generate_series(1, {n}) g",
            n / 2
        ),
    ] {
        db.execute(&sql, ()).unwrap();
    }
    let rows = n + n / 2;
    let merged = run(&db, "SELECT k, a, b FROM t FULL JOIN u USING (k)", rows);
    let on = run(
        &db,
        "SELECT COALESCE(t.k, u.k), a, b FROM t FULL JOIN u ON u.k = t.k",
        rows,
    );
    (merged, on)
}

#[test]
fn a_full_join_holds_its_merged_column_without_an_allocation_per_row() {
    let _guard = serial();
    let (merged, on) = both("full_join_merged_alloc_streaming", 600, "INTEGER");
    assert!(
        merged <= on + 64,
        "{merged} allocations over USING, {on} over ON, for 900 rows"
    );
}

#[test]
fn a_parallel_full_join_holds_its_merged_column_without_an_allocation_per_row() {
    let _guard = serial();
    let (merged, on) = both("full_join_merged_alloc_parallel", 12_000, "INTEGER");
    assert!(
        merged <= on + 1_000,
        "{merged} allocations over USING, {on} over ON, for 18000 rows"
    );
}

#[test]
fn a_nested_loop_full_join_holds_its_merged_column_without_an_allocation_per_row() {
    let _guard = serial();
    // A TEXT key meeting an INTEGER one is compared in the residual
    let (merged, on) = both("full_join_merged_alloc_nested", 200, "TEXT");
    assert!(
        merged <= on + 64,
        "{merged} allocations over USING, {on} over ON, for 300 rows"
    );
}
