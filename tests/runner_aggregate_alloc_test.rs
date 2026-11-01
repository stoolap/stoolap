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

//! An aggregate of one table's rows into one row whose calls cannot be
//! split over the thread pool reads the rows it decodes from volumes one at
//! a time, however many: the bytes it holds at once do not grow with the
//! rows, measured as the peak of what the query's thread holds

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

struct ThreadPeak;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn held(change: isize) {
    let live = LIVE.with(|l| {
        l.set(l.get() + change);
        l.get()
    });
    PEAK.with(|p| p.set(p.get().max(live)));
}

unsafe impl GlobalAlloc for ThreadPeak {
    // SAFETY: forwards to the system allocator; the counters are const
    // thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        held(layout.size() as isize);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        held(-(layout.size() as isize));
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        held(new_size as isize - layout.size() as isize);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static PEAK_COUNTING: ThreadPeak = ThreadPeak;

const ROWS: usize = 120_000;

#[test]
fn an_aggregate_whose_calls_do_not_split_streams_its_decoded_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT value, value % 7, value % 3 FROM generate_series(1, {ROWS})"
        ),
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let sql = "SELECT COUNT(DISTINCT a), COUNT(*) FROM t WHERE b >= 0";
    let answer = |db: &Database| -> (i64, i64) {
        let row = db.query(sql, ()).unwrap().next().unwrap().unwrap();
        (row.get(0).unwrap(), row.get(1).unwrap())
    };
    answer(&db);
    let start = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(start));
    let got = answer(&db);
    let peak = PEAK.with(Cell::get) - start;
    assert_eq!(got, (7, ROWS as i64));
    // Each held row costs far more than eight bytes
    assert!(
        peak < (ROWS * 8) as isize,
        "{peak} bytes held at once for {ROWS} rows: the decoded rows were held"
    );
}
