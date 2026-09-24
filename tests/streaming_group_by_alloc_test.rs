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

//! The index walk aggregates a group a page of row ids at a time; the
//! allocations of the query do not grow with the pages of its groups

// The mimalloc feature sets the library's own global allocator. The file
// engine walks captured groups and leaves the walk past its capture bound,
// so the fixed count holds for the memory engine only
#![cfg(all(not(feature = "mimalloc"), not(feature = "test-filedb")))]

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

/// 16 groups of `size` rows; the allocations of one aggregate walk over them
fn allocations_for_groups_of(size: i64) -> usize {
    let db = Database::open(&format!("memory://streaming_group_by_alloc_{size}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let insert = db.prepare("INSERT INTO t VALUES ($1, $2, $1)").unwrap();
    db.execute("BEGIN", ()).unwrap();
    for id in 1..=16 * size {
        insert.execute((id, (id - 1) / size)).unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    // HAVING keeps the query on the index walk, off storage aggregation
    let sql =
        "SELECT k, SUM(v), AVG(v), MIN(v), MAX(v) FROM t GROUP BY k HAVING SUM(v) > 0 LIMIT 16";
    let query = db.prepare(sql).unwrap();
    let run = || query.query(()).unwrap().count();
    assert_eq!(run(), 16);
    let before = ALLOCATIONS.with(Cell::get);
    assert_eq!(run(), 16);
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn an_aggregate_walk_allocates_the_same_for_groups_of_one_page_or_twelve() {
    let counts: Vec<(i64, usize)> = [256, 512, 513, 1_536, 6_000]
        .into_iter()
        .map(|size| (size, allocations_for_groups_of(size)))
        .collect();
    let least = counts.iter().map(|(_, n)| *n).min().unwrap();
    assert!(
        counts.iter().all(|(_, n)| *n <= least + 8),
        "allocations per query by rows per group: {counts:?}"
    );
}
