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

//! A VALUES row its WHERE filters out is let go before the next row is
//! evaluated, counted in the peak live bytes of the whole process

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering};

use stoolap::Database;

struct Counting;

static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the counters are atomics
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size() as i64, Ordering::Relaxed) + layout.size() as i64;
        PEAK.fetch_max(live, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let grown = new_size as i64 - layout.size() as i64;
        let live = LIVE.fetch_add(grown, Ordering::Relaxed) + grown;
        PEAK.fetch_max(live, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

const ROWS: usize = 300;
const VALUE_BYTES: i64 = 64 * 1024;

#[test]
fn a_values_row_filtered_out_is_not_kept() {
    let db = Database::open("memory://values_source_memory").unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1)", ()).unwrap();
    let rows = vec!["(REPEAT('x', $1))"; ROWS].join(", ");
    let mut peaks = Vec::new();
    for (path, sql) in [
        (
            "general",
            format!("SELECT x FROM (VALUES {rows}) AS s(x) WHERE LENGTH(x) = 0"),
        ),
        (
            "prepared",
            format!(
                "SELECT x FROM (VALUES {rows}) AS s(x) \
                 WHERE LENGTH(x) = 0 AND EXISTS (SELECT 1 FROM u)"
            ),
        ),
    ] {
        let prepared = db
            .query(&format!("EXPLAIN {sql}"), (VALUE_BYTES,))
            .unwrap()
            .any(|line| line.unwrap().get::<String>(0).unwrap().trim() == "Prepared: yes");
        assert_eq!(prepared, path == "prepared", "{path}: the path taken");
        let statement = db.prepare(&sql).unwrap();
        let run = || statement.query((VALUE_BYTES,)).unwrap().count();
        assert_eq!(run(), 0, "{path}: every row is filtered out");
        let before = LIVE.load(Ordering::Relaxed);
        PEAK.store(before, Ordering::Relaxed);
        assert_eq!(run(), 0, "{path}: every row is filtered out");
        peaks.push((path, PEAK.load(Ordering::Relaxed) - before));
    }
    assert!(
        peaks.iter().all(|&(_, peak)| peak < 16 * VALUE_BYTES),
        "{ROWS} rows of {VALUE_BYTES} bytes, all filtered out, peaked at (path, bytes) {peaks:?}"
    );
}
