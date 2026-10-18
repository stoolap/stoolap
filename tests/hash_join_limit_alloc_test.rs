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

//! A hash join under a LIMIT stops when it has the rows the LIMIT needs,
//! however large the LIMIT, counted in the bytes every thread allocates

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use stoolap::Database;

struct Counting;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the counter is an atomic,
    // which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocated(db: &Database, sql: &str) -> usize {
    let before = ALLOCATED.load(Ordering::Relaxed);
    let rows = db.query(sql, ()).unwrap().count();
    assert!(rows > 0, "{sql} returned rows");
    ALLOCATED.load(Ordering::Relaxed) - before
}

#[test]
fn a_hash_join_under_a_large_limit_stops_at_the_limit() {
    let db = Database::open("memory://hash_join_limit_alloc").unwrap();
    // 10,000 rows a side, each key on 10 rows: the join holds 100,000 rows
    for table in ["a", "b"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER)"),
            (),
        )
        .unwrap();
        for start in (1..=10_000).step_by(1000) {
            let values: Vec<String> = (start..start + 1000)
                .map(|id| format!("({id}, {})", id % 1000))
                .collect();
            db.execute(
                &format!("INSERT INTO {table} VALUES {}", values.join(", ")),
                (),
            )
            .unwrap();
        }
    }
    let sql = |limit: u32| format!("SELECT a.id, b.id FROM a JOIN b ON a.k = b.k LIMIT {limit}");
    allocated(&db, &sql(10));
    let small = allocated(&db, &sql(10));
    let large = allocated(&db, &sql(1001));
    assert!(
        large < small + 1024 * 1024,
        "LIMIT 10 allocated {small} bytes, LIMIT 1001 allocated {large}"
    );
}
