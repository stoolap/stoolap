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

//! A DELETE allocates nothing while it decides a round under the seal
//! fence or reads its transaction's pending tombstones, and a DELETE of
//! hot rows allocates no more than before

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(feature = "test-failpoints", not(feature = "jemalloc")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::rc::Rc;

use stoolap::core::Error;
use stoolap::storage::traits::Engine;
use stoolap::test_failpoints::{
    after_cold_round_prepared, in_pending_set_lock, in_recheck_fence_section, FailpointGuard,
};
use stoolap::Database;

struct Counting;

thread_local! {
    static UNDER_FENCE: Cell<usize> = const { Cell::new(0) };
    static UNDER_PENDING_LOCK: Cell<usize> = const { Cell::new(0) };
    static ALL: Cell<usize> = const { Cell::new(0) };
}

impl Counting {
    fn count() {
        let _ = ALL.try_with(|n| n.set(n.get() + 1));
        if in_recheck_fence_section() {
            let _ = UNDER_FENCE.try_with(|n| n.set(n.get() + 1));
        }
        if in_pending_set_lock() {
            let _ = UNDER_PENDING_LOCK.try_with(|n| n.set(n.get() + 1));
        }
    }
}

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the flag and the counters
    // are const thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::count();
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::count();
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn under_fence() -> usize {
    UNDER_FENCE.with(Cell::get)
}

/// A file database with (1, 0, 10), (2, 0, 20) and (3, 0, 30) sealed
fn sealed_t(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 0, 10), (2, 0, 20), (3, 0, 30)",
        (),
    )
    .unwrap();
    db.engine().seal_for_test().unwrap();
    db
}

/// Drops column `x` from another thread; true when it finished
fn drop_x_apart(db: &Database) -> bool {
    let db = db.clone();
    std::thread::spawn(move || db.execute("ALTER TABLE t DROP COLUMN x", ()).is_ok())
        .join()
        .unwrap()
}

#[test]
fn a_delete_allocates_nothing_deciding_a_round_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_t(dir.path());
    db.engine().cold_volumes_for_test("t");
    let before = under_fence();
    assert_eq!(db.execute("DELETE FROM t WHERE v >= 20", ()).unwrap(), 2);
    assert_eq!(
        under_fence() - before,
        0,
        "allocations while the DELETE decided a round under the seal fence"
    );
}

#[test]
fn a_delete_refused_for_a_column_change_allocates_nothing_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_t(dir.path());
    let ddl = db.clone();
    let dropped = Rc::new(Cell::new(false));
    let slot = Rc::clone(&dropped);
    after_cold_round_prepared(move || slot.set(drop_x_apart(&ddl)));
    let before = under_fence();
    let deleted = db.execute("DELETE FROM t WHERE v >= 20", ());
    assert!(
        dropped.get(),
        "the column change did not finish in the round"
    );
    assert!(
        matches!(deleted, Err(Error::SchemaChanged { .. })),
        "{deleted:?}"
    );
    assert_eq!(
        under_fence() - before,
        0,
        "allocations while the DELETE refused a column change under the seal fence"
    );
}

#[test]
fn an_update_refused_for_a_column_change_allocates_nothing_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_t(dir.path());
    let ddl = db.clone();
    let dropped = Rc::new(Cell::new(false));
    let slot = Rc::clone(&dropped);
    after_cold_round_prepared(move || slot.set(drop_x_apart(&ddl)));
    let before = under_fence();
    let updated = db.execute("UPDATE t SET v = v + 1 WHERE v >= 20", ());
    assert!(
        dropped.get(),
        "the column change did not finish in the round"
    );
    assert!(
        matches!(updated, Err(Error::SchemaChanged { .. })),
        "{updated:?}"
    );
    assert_eq!(
        under_fence() - before,
        0,
        "allocations while the UPDATE refused a column change under the seal fence"
    );
}

#[test]
fn a_hot_delete_by_primary_key_allocates_no_more_than_before() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    let insert = db.prepare("INSERT INTO t VALUES (?, 0, 0)").unwrap();
    let delete = db.prepare("DELETE FROM t WHERE id = ?").unwrap();
    for id in 0..20 {
        insert.execute((id,)).unwrap();
    }
    for id in 0..10 {
        delete.execute((id,)).unwrap();
    }
    let before = ALL.with(Cell::get);
    for id in 10..20 {
        assert_eq!(delete.execute((id,)).unwrap(), 1);
    }
    let per_delete = (ALL.with(Cell::get) - before) / 10;
    assert!(
        per_delete <= HOT_PK_DELETE_ALLOCATIONS,
        "{per_delete} allocations per hot DELETE by primary key, {HOT_PK_DELETE_ALLOCATIONS} before"
    );
}

/// Measured on 897038e4 (main with #371), the shape above, plus the one
/// removal history record a commit that takes index entries out publishes
const HOT_PK_DELETE_ALLOCATIONS: usize = 30;

#[test]
fn a_delete_finds_the_rows_its_transaction_changed_with_no_allocation_under_the_lock() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4), (5, 0, 5), (6, 0, 6), (7, 0, 7), (8, 0, 8)",
        (),
    )
    .unwrap();
    db.engine().seal_for_test().unwrap();
    let mut txn = db.engine().begin_transaction().unwrap();
    let mut deleter = txn.get_table("t").unwrap();
    // A second handle of the transaction changes six of the prepared rows,
    // more than the found ids keep inline
    let mut other = txn.get_table("t").unwrap();
    let changed = Rc::new(Cell::new(None));
    let slot = Rc::clone(&changed);
    after_cold_round_prepared(move || {
        let mut set = |mut row: stoolap::core::Row| {
            row.set(1, stoolap::core::Value::Integer(9))?;
            Ok((row, true))
        };
        slot.set(other.update_by_row_ids(&[1, 2, 3, 4, 5, 6], &mut set).ok());
    });
    let before = UNDER_PENDING_LOCK.with(Cell::get);
    let deleted = deleter.delete(None);
    let under = UNDER_PENDING_LOCK.with(Cell::get) - before;
    drop(deleter);
    txn.commit().unwrap();
    assert_eq!(
        changed.get(),
        Some(6),
        "the other handle did not change six rows"
    );
    assert_eq!(deleted.ok(), Some(8));
    assert_eq!(
        under, 0,
        "allocations while the DELETE held the pending tombstone sets"
    );
}
