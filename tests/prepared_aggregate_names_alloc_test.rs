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

//! A prepared aggregate finds its source columns by name and names its
//! calls once for its runs: what one run allocates does not grow with the
//! columns its source holds, nor rebuilds the names of its output

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(not(feature = "jemalloc"), not(feature = "test-filedb")))]

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

/// The allocations of one run of `sql`, prepared, over a table of `width`
/// columns past its key, in a database named after `name`
fn allocations(name: &str, width: usize, sql: &str) -> usize {
    let db = Database::open(&format!("memory://prepared_aggregate_names_{name}_{width}")).unwrap();
    let extra: String = (0..width).map(|i| format!(", c{i} INTEGER")).collect();
    db.execute(
        &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, a INTEGER{extra})"),
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t (id, k, a) SELECT g.value, g.value % 4, g.value FROM generate_series(1, 40) g",
        (),
    )
    .unwrap();
    let query = db.prepare(sql).unwrap();
    let run = || query.query((1,)).unwrap().count();
    assert_eq!(run(), 1, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_prepared_aggregate_finds_its_columns_once_for_its_runs() {
    let sql = "SELECT COUNT(*), SUM(a) FROM t WHERE k = $1";
    let narrow = allocations("global", 2, sql);
    let wide = allocations("global", 60, sql);
    assert!(
        wide <= narrow + 4,
        "{sql}: {narrow} allocations over 4 columns, {wide} over 62"
    );
}

/// The most allocations a held run of each aggregate below makes, from
/// when it first named its calls and found its columns once. Lower one
/// when a change saves an allocation
const HELD_RUNS: [(&str, &str, usize); 2] = [
    (
        "named",
        "SELECT COUNT(*), SUM(a), MAX(a) FROM t WHERE k = $1",
        21,
    ),
    (
        "derived",
        "SELECT COUNT(*), SUM(s.a) FROM (SELECT k, a FROM t WHERE k = $1) s",
        34,
    ),
];

/// The most allocations a held run of each aggregate below makes, from
/// when it first bound its SELECT list over its groups once. Lower one
/// when a change saves an allocation
const PROJECTED_RUNS: [(&str, &str, usize); 3] = [
    (
        "grouped",
        "SELECT k, COUNT(*), SUM(a) FROM t WHERE k = $1 GROUP BY k",
        28,
    ),
    (
        "computed",
        "SELECT k, CASE WHEN SUM(a) > 100 THEN 'big' ELSE 'small' END AS size FROM t WHERE k = $1 GROUP BY k",
        29,
    ),
    (
        "global",
        "SELECT SUM(a) + 1, COUNT(*) * 2 FROM t WHERE k = $1",
        28,
    ),
];

#[test]
fn a_prepared_aggregate_projects_its_list_once_for_its_runs() {
    let over: Vec<String> = PROJECTED_RUNS
        .iter()
        .filter_map(|&(name, sql, most)| {
            let run = allocations(name, 2, sql);
            (run > most).then(|| format!("{sql}: {run} allocations, at most {most} before"))
        })
        .collect();
    assert!(over.is_empty(), "held runs allocated more: {over:#?}");
}

#[test]
fn a_prepared_aggregate_names_its_calls_once_for_its_runs() {
    for (name, sql, most) in HELD_RUNS {
        let run = allocations(name, 2, sql);
        assert!(
            run <= most,
            "{sql}: a held run allocated {run} times, at most {most} before"
        );
    }
}
