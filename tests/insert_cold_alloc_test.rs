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

//! An INSERT's cold check allocates nothing while it decides a round
//! under the seal fence

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(feature = "test-failpoints", not(feature = "jemalloc")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::rc::Rc;

use stoolap::core::Error;
use stoolap::storage::traits::Engine;
use stoolap::test_failpoints::{
    after_insert_cold_snapshot, in_recheck_fence_section, volume_loads, FailpointGuard,
};
use stoolap::Database;

struct Counting;

thread_local! {
    static UNDER_FENCE: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the flag and the counter
    // are const thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if in_recheck_fence_section() {
            let _ = UNDER_FENCE.try_with(|n| n.set(n.get() + 1));
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if in_recheck_fence_section() {
            let _ = UNDER_FENCE.try_with(|n| n.set(n.get() + 1));
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

#[test]
fn an_insert_allocates_nothing_deciding_its_cold_check_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 0), (3, 30, 0)", ())
        .unwrap();
    db.engine().seal_for_test().unwrap();
    db.engine().cold_volumes_for_test("t");
    let (loads, before) = (volume_loads().0, UNDER_FENCE.with(Cell::get));
    db.execute("INSERT INTO t VALUES (2, 20, 0)", ()).unwrap();
    assert!(volume_loads().0 > loads, "the check loaded no volume");
    assert_eq!(
        UNDER_FENCE.with(Cell::get) - before,
        0,
        "allocations while the INSERT decided its cold check under the seal fence"
    );
}

fn open_t(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
    db
}

/// Drops column `v` from another thread; true when it finished
fn drop_v_apart(db: &Database) -> bool {
    let db = db.clone();
    std::thread::spawn(move || db.execute("ALTER TABLE t DROP COLUMN v", ()).is_ok())
        .join()
        .unwrap()
}

#[test]
fn an_insert_refused_for_a_column_change_allocates_nothing_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = open_t(dir.path());
    db.execute("INSERT INTO t VALUES (1, 10, 0), (3, 30, 0)", ())
        .unwrap();
    db.engine().seal_for_test().unwrap();
    let ddl = db.clone();
    let dropped = Rc::new(Cell::new(false));
    let slot = Rc::clone(&dropped);
    after_insert_cold_snapshot(move || slot.set(drop_v_apart(&ddl)));
    let before = UNDER_FENCE.with(Cell::get);
    let inserted = db.execute("INSERT INTO t VALUES (2, 20, 0)", ());
    assert!(
        dropped.get(),
        "the column change did not finish in the check"
    );
    assert!(
        matches!(inserted, Err(Error::SchemaChanged { .. })),
        "{inserted:?}"
    );
    assert_eq!(
        UNDER_FENCE.with(Cell::get) - before,
        0,
        "allocations while the INSERT refused a column change under the seal fence"
    );
}
