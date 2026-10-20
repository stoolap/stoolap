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

//! An UPDATE whose WHERE matches none of the rows its transaction wrote
//! allocates no more for many such rows than for few

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn update_allocs(written: usize) -> usize {
    let db = Database::open(&format!("memory://update_own_rows_alloc_{written}")).unwrap();
    db.execute("CREATE TABLE t (v INTEGER)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    for _ in 0..written {
        db.execute("INSERT INTO t VALUES (0)", ()).unwrap();
    }
    db.execute("UPDATE t SET v = 2 WHERE v = 1", ()).unwrap();
    let before = ALLOCS.with(Cell::get);
    let updated = db.execute("UPDATE t SET v = 2 WHERE v = 1", ()).unwrap();
    let allocs = ALLOCS.with(Cell::get) - before;
    assert_eq!(updated, 0);
    db.execute("ROLLBACK", ()).unwrap();
    allocs
}

#[test]
fn an_update_matching_no_written_row_allocates_as_for_few() {
    assert_eq!(update_allocs(1000), update_allocs(10));
}
