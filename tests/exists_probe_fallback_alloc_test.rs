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

//! Clearing every thread-local cache releases the semi-join set a refused
//! EXISTS probe was answered from

// The library's own allocator features set a global allocator of their own
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use stoolap::Database;

static LIVE: AtomicIsize = AtomicIsize::new(0);

struct LiveBytes;

unsafe impl GlobalAlloc for LiveBytes {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: LiveBytes = LiveBytes;

#[test]
fn clearing_all_caches_releases_the_probe_fallback_set() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t SELECT value, CASE WHEN value % 10 = 0 THEN -1 ELSE value END \
         FROM generate_series(1, 100000)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, -1)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    let found: Vec<i64> = db
        .query(
            "SELECT o.id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.k = o.k) LIMIT 10",
            (),
        )
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(found, [1]);
    stoolap::executor::clear_all_thread_local_caches();
    let cleared = LIVE.load(Ordering::Relaxed);
    db.query("SELECT 1", ())
        .unwrap()
        .for_each(|r| drop(r.unwrap()));
    let next_select = LIVE.load(Ordering::Relaxed);
    assert!(
        cleared <= next_select + 64 * 1024,
        "clear_all kept {} bytes the next statement released",
        cleared.saturating_sub(next_select)
    );
    db.execute("ROLLBACK", ()).unwrap();
}
