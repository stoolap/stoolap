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

//! A sealed DISTINCT over a volume that hides most of its rows holds memory
//! for the values its shown rows hold, not for every key of the volume

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

const ROWS: i64 = 100_000;

/// The bytes `sql` holds at its peak above what was live when it began
fn peak_of(db: &Database, sql: &str) -> i64 {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let rows = db.query(sql, ()).unwrap().count();
    assert_eq!(rows, 5, "{sql}");
    PEAK.load(Ordering::Relaxed) - before
}

#[test]
fn a_sealed_distinct_over_a_mostly_hidden_volume_holds_memory_for_its_shown_values() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, u INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_v ON t (v)", ()).unwrap();
    db.execute("CREATE INDEX t_u ON t (u)", ()).unwrap();
    for start in (1..=ROWS).step_by(10_000) {
        let values: Vec<String> = (start..start + 10_000)
            .map(|id| format!("({id}, {id}, {})", id % 10))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("DELETE FROM t WHERE id > 5", ()).unwrap();
    let (many, few) = ("SELECT DISTINCT v FROM t", "SELECT DISTINCT u FROM t");
    // The side files' pages are cached by a first run of each
    peak_of(&db, many);
    peak_of(&db, few);
    let (many, few) = (peak_of(&db, many), peak_of(&db, few));
    assert!(
        many - few < 256 * 1024,
        "DISTINCT over {ROWS} keys, 5 shown, held {many} bytes; over 10 keys {few}"
    );
}
