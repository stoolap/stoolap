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

//! UPDATE and DELETE by primary key on a table whose rows are sealed into
//! volumes must resolve the one row by id, not walk the volume.

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use stoolap::Database;

static ALLOCS: AtomicU64 = AtomicU64::new(0);

/// The counter is process-wide, so the tests in this file never overlap
static SERIAL: Mutex<()> = Mutex::new(());

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn sealed_table(name: &str, rows: i64) -> (Database, PathBuf) {
    let dir = std::env::temp_dir().join(format!("{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    let mut sql = String::new();
    for start in (0..rows).step_by(1000) {
        sql.clear();
        sql.push_str("INSERT INTO t VALUES ");
        for i in start..(start + 1000).min(rows) {
            if i != start {
                sql.push(',');
            }
            sql.push_str(&format!("({i}, 'k{}', {i})", i % 97));
        }
        db.execute(&sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    (db, dir)
}

/// Allocations per statement over `ids`, each id touching a still-sealed row
fn allocs_per_statement(db: &Database, sql: &str, ids: std::ops::Range<i64>) -> f64 {
    let stmt = db.prepare(sql).unwrap();
    for id in 0..5 {
        stmt.execute((id,)).unwrap();
    }
    let n = ids.end - ids.start;
    let before = ALLOCS.load(Ordering::Relaxed);
    for id in ids {
        stmt.execute((id,)).unwrap();
    }
    (ALLOCS.load(Ordering::Relaxed) - before) as f64 / n as f64
}

fn scalar(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}

#[test]
fn update_by_pk_cost_does_not_grow_with_sealed_rows() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (small, small_dir) = sealed_table("pk_upd_small", 2_000);
    let (big, big_dir) = sealed_table("pk_upd_big", 20_000);
    let sql = "UPDATE t SET v = v + 1 WHERE id = ?";
    let per_small = allocs_per_statement(&small, sql, 100..120);
    let per_big = allocs_per_statement(&big, sql, 100..120);
    assert!(
        per_big <= per_small * 2.0,
        "allocations per UPDATE by pk scale with the table: {per_small:.0} at 2k rows, {per_big:.0} at 20k"
    );
    drop((small, big));
    let _ = std::fs::remove_dir_all(small_dir);
    let _ = std::fs::remove_dir_all(big_dir);
}

#[test]
fn update_and_delete_by_pk_on_a_sealed_table_are_correct() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (db, dir) = sealed_table("pk_dml_correct", 3_000);
    let total = scalar(&db, "SELECT COUNT(*) FROM t");

    db.execute("UPDATE t SET v = 1000 WHERE id = 7", ())
        .unwrap();
    assert_eq!(scalar(&db, "SELECT v FROM t WHERE id = 7"), 1000);
    db.execute("UPDATE t SET v = v + 1 WHERE id = 7", ())
        .unwrap();
    assert_eq!(
        scalar(&db, "SELECT v FROM t WHERE id = 7"),
        1001,
        "a second update finds the row that the first one moved into the hot store"
    );
    assert_eq!(
        db.execute("UPDATE t SET v = 0 WHERE id = 1000000", ())
            .unwrap(),
        0,
        "no row for an unknown pk"
    );

    db.execute("UPDATE t SET v = 42 WHERE id = 9 AND v >= 0", ())
        .unwrap();
    assert_eq!(
        scalar(&db, "SELECT v FROM t WHERE id = 9"),
        42,
        "a compound predicate still updates through the general path"
    );

    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE t SET v = -1 WHERE id = 11", ()).unwrap();
    assert_eq!(scalar(&db, "SELECT v FROM t WHERE id = 11"), -1);
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(scalar(&db, "SELECT v FROM t WHERE id = 11"), 11);

    db.execute("DELETE FROM t WHERE id = 8", ()).unwrap();
    assert_eq!(scalar(&db, "SELECT COUNT(*) FROM t WHERE id = 8"), 0);
    assert_eq!(scalar(&db, "SELECT COUNT(*) FROM t"), total - 1);
    assert_eq!(
        db.execute("DELETE FROM t WHERE id = 8", ()).unwrap(),
        0,
        "deleting the same pk again affects nothing"
    );
    assert_eq!(
        db.execute("UPDATE t SET v = 5 WHERE id = 8", ()).unwrap(),
        0,
        "updating a deleted pk affects nothing"
    );
    assert_eq!(
        db.execute("DELETE FROM t WHERE id = 1000000", ()).unwrap(),
        0,
        "deleting an unknown pk affects nothing"
    );
    db.execute("DELETE FROM t WHERE id = 7", ()).unwrap();
    assert_eq!(
        scalar(&db, "SELECT COUNT(*) FROM t WHERE id = 7"),
        0,
        "a row that was updated into the hot store is deleted from there"
    );
    assert_eq!(scalar(&db, "SELECT COUNT(*) FROM t"), total - 2);

    drop(db);
    let _ = std::fs::remove_dir_all(dir);
}

/// A child pointing at `t` (the parent), its rows still sealed
fn add_child(db: &Database) {
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY AUTO_INCREMENT, t_id INTEGER REFERENCES t(id))",
        (),
    )
    .unwrap();
}

#[test]
fn foreign_key_check_cost_does_not_grow_with_sealed_parent_rows() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (small, small_dir) = sealed_table("pk_fk_small", 2_000);
    let (big, big_dir) = sealed_table("pk_fk_big", 20_000);
    add_child(&small);
    add_child(&big);
    // Keys at the end of the volume: a walk to the matching row reads them last
    let sql = "INSERT INTO c (t_id) VALUES (?)";
    let per_small = allocs_per_statement(&small, sql, 1_980..2_000);
    let per_big = allocs_per_statement(&big, sql, 19_980..20_000);
    assert!(
        per_big <= per_small * 2.0,
        "allocations per foreign key check scale with the parent: {per_small:.0} at 2k rows, {per_big:.0} at 20k"
    );
    drop((small, big));
    let _ = std::fs::remove_dir_all(small_dir);
    let _ = std::fs::remove_dir_all(big_dir);
}

#[test]
fn foreign_key_check_on_a_sealed_parent_is_correct() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (db, dir) = sealed_table("pk_fk_correct", 3_000);
    add_child(&db);
    let insert = |id: i64| db.execute(&format!("INSERT INTO c (t_id) VALUES ({id})"), ());

    insert(2_999).unwrap();
    assert!(insert(1_000_000).is_err(), "an unknown parent is refused");

    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (5000, 'new', 0)", ())
        .unwrap();
    insert(5_000).expect("a parent inserted earlier in the transaction is seen");
    db.execute("DELETE FROM t WHERE id = 2500", ()).unwrap();
    assert!(
        insert(2_500).is_err(),
        "a sealed parent deleted earlier in the transaction is refused"
    );
    db.execute("ROLLBACK", ()).unwrap();
    assert!(insert(5_000).is_err(), "a rolled back parent is refused");
    insert(2_500).expect("a rolled back delete leaves the parent");

    db.execute("UPDATE t SET v = -1 WHERE id = 2600", ())
        .unwrap();
    insert(2_600).expect("a parent updated into the hot store is found there");
    assert_eq!(scalar(&db, "SELECT COUNT(*) FROM c"), 3);

    drop(db);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn foreign_key_check_of_a_parent_written_in_the_transaction_keeps_its_single_row_read() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir().join(format!("pk_fk_local_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    add_child(&db);
    // The same insert with no key to check
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY AUTO_INCREMENT, t_id INTEGER)",
        (),
    )
    .unwrap();
    // No volume yet: the parent's rows are this transaction's own
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT value, 'k', value FROM GENERATE_SERIES(0, 199)",
        (),
    )
    .unwrap();
    let checked = allocs_per_statement(&db, "INSERT INTO c (t_id) VALUES (?)", 100..120);
    let plain = allocs_per_statement(&db, "INSERT INTO d (t_id) VALUES (?)", 100..120);
    db.execute("COMMIT", ()).unwrap();
    // The hot store's single-row read of the key: 2.2 allocations per check
    // here, one more when the check went through the batch fetch
    assert!(
        checked - plain <= 2.5,
        "a check of a parent with no volume costs {:.1} allocations over an unchecked insert",
        checked - plain
    );
    drop(db);
    let _ = std::fs::remove_dir_all(dir);
}
