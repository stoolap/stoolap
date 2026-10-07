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

//! A correlated COUNT(*) probes the index on its key, whatever the index's
//! kind, counted in the bytes the query allocates on its thread

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + layout.size()));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + new_size));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_correlated_count_probes_every_equality_index_kind() {
    for kind in ["pk", "hash", "bitmap", "btree"] {
        let db = Database::open(&format!("memory://correlated_count_alloc_{kind}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute(
            "INSERT INTO t SELECT value, value FROM generate_series(1, 100000)",
            (),
        )
        .unwrap();
        let column = if kind == "pk" {
            "id"
        } else {
            db.execute(&format!("CREATE INDEX t_k ON t(k) USING {kind}"), ())
                .unwrap();
            "k"
        };
        db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO o VALUES (1, 42)", ()).unwrap();
        let sql =
            format!("SELECT o.id, (SELECT COUNT(*) FROM t WHERE t.{column} = o.k) FROM o LIMIT 1");
        let count = |db: &Database| -> i64 {
            db.query(&sql, ())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .get(1)
                .unwrap()
        };
        assert_eq!(count(&db), 1);
        let before = ALLOCATED.with(Cell::get);
        assert_eq!(count(&db), 1);
        let bytes = ALLOCATED.with(Cell::get) - before;
        assert!(bytes < 64 * 1024, "{kind}: {bytes} bytes for one probe");
    }
}
