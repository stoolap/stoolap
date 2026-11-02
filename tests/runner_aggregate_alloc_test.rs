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

//! An aggregate of one table's rows, into one row, into groups or into
//! grouping sets, whose calls cannot be split over the thread pool reads the
//! rows it decodes from volumes one at a time, however many: the bytes it
//! holds at once do not grow with the rows, measured as the peak of what the
//! query's thread holds

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

/// The sum of the first column of `sql`'s rows over a sealed table of
/// `ROWS` rows, a = id % 7 and b = id % 3, and the bytes its second run
/// held at once
fn summed_and_peak(sql: &str) -> (i64, isize) {
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
    let answer = |db: &Database| -> i64 {
        db.query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .sum()
    };
    answer(&db);
    let start = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(start));
    let got = answer(&db);
    (got, PEAK.with(Cell::get) - start)
}

#[test]
fn an_aggregate_whose_calls_do_not_split_streams_its_decoded_rows() {
    for (sql, want) in [
        (
            "SELECT COUNT(*), COUNT(DISTINCT a) FROM t WHERE b >= 0",
            ROWS,
        ),
        (
            "SELECT COUNT(DISTINCT b), a FROM t WHERE b >= 0 GROUP BY a",
            21,
        ),
        (
            "SELECT COUNT(*), a, b FROM t GROUP BY ROLLUP (a, b)",
            3 * ROWS,
        ),
    ] {
        let (got, peak) = summed_and_peak(sql);
        assert_eq!(got, want as i64, "{sql}");
        // Each held row costs far more than eight bytes
        assert!(
            peak < (ROWS * 8) as isize,
            "{sql}: {peak} bytes held at once for {ROWS} rows: the decoded rows were held"
        );
    }
}
