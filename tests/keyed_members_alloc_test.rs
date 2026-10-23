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

//! A correlated IN block's members grouped by key cost a fixed number of
//! allocations per key, counted on the query's thread

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

const RUNS: usize = 20;

/// The allocations per run of a correlated IN over `keys` keys, its values
/// of `kind`
fn per_run(kind: &str, keys: usize) -> usize {
    let db = Database::open(&format!("memory://keyed_members_alloc_{kind}_{keys}")).unwrap();
    db.execute(
        &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, x {kind})"),
        (),
    )
    .unwrap();
    db.execute(
        &format!("CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v {kind})"),
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT value, value, CAST(value AS {kind}) FROM generate_series(1, {keys})"
        ),
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO u SELECT value, value % {keys}, CAST(value AS {kind}) \
             FROM generate_series(1, {})",
            keys * 3
        ),
        (),
    )
    .unwrap();
    let statement = db
        .prepare("SELECT id FROM t WHERE x IN (SELECT v FROM u WHERE u.k = t.k)")
        .unwrap();
    // A transaction keeps no answer between runs: each one groups again
    db.execute("BEGIN", ()).unwrap();
    let run = || {
        for row in statement.query(()).unwrap() {
            row.unwrap();
        }
    };
    run();
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..RUNS {
        run();
    }
    let cost = (ALLOCATIONS.with(Cell::get) - before) / RUNS;
    db.execute("COMMIT", ()).unwrap();
    cost
}

#[test]
fn a_keyed_member_set_costs_no_more_per_key_than_its_members() {
    // The allocations each further key costs: an INTEGER set, and a TEXT set
    // with the numbers its members read as
    let costs: Vec<(&str, usize, [usize; 2])> = [("INTEGER", 4), ("TEXT", 5)]
        .into_iter()
        .map(|(kind, per_key)| (kind, per_key, [per_run(kind, 100), per_run(kind, 200)]))
        .collect();
    assert!(
        costs
            .iter()
            .all(|&(_, per_key, [few, many])| many - few < (2 * per_key + 1) * 100 / 2),
        "(kind, allocations per key, per run over 100 and 200 keys): {costs:?}"
    );
}
