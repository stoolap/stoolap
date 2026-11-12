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

use stoolap::Database;
use tempfile::tempdir;

fn next_id_after_insert(db: &Database, table: &str) -> i64 {
    db.execute(&format!("INSERT INTO {table} (v) VALUES ('new')"), ())
        .unwrap();
    db.query_one(&format!("SELECT id FROM {table} WHERE v = 'new'"), ())
        .unwrap()
}

#[test]
fn test_deleted_id_not_reused_after_reopen() {
    // The failpoint tests of this group fail WAL syncs process-wide
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().join("db").display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
            .unwrap();
        db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        next_id_after_insert(&db, "t"),
        4,
        "id 3 was handed out before the reopen"
    );
}

#[test]
fn test_deleted_id_not_reused_after_checkpoint_and_crash() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let dir = tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().join("db").display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
            .unwrap();
        db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        next_id_after_insert(&db, "t"),
        4,
        "id 3 was handed out before the checkpoint"
    );
}

#[test]
fn test_deleted_rowid_not_reused_after_reopen() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().join("db").display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE n (v TEXT)", ()).unwrap();
        db.execute("INSERT INTO n (v) VALUES ('a'), ('b'), ('c')", ())
            .unwrap();
        db.execute("DELETE FROM n WHERE v = 'c'", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("INSERT INTO n (v) VALUES ('d')", ()).unwrap();
    let rowid: i64 = db
        .query_one("SELECT _rowid_ FROM n WHERE v = 'd'", ())
        .unwrap();
    assert_eq!(rowid, 4, "row id 3 was handed out before the reopen");
}

#[test]
fn test_truncate_restarts_ids_after_reopen() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().join("db").display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
            .unwrap();
        db.execute("TRUNCATE TABLE t", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(next_id_after_insert(&db, "t"), 1);
}

/// `statements` on a new table, then a close and a reopen with `params`
fn reopened_after(statements: &[&str], params: &str) -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}{params}", dir.path().join("db").display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
            (),
        )
        .unwrap();
        for sql in statements {
            db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        }
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    (dir, db)
}

#[test]
fn test_alter_auto_increment_survives_a_reopen() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let (_dir, db) = reopened_after(&["ALTER TABLE t AUTO_INCREMENT = 1000"], "");
    assert_eq!(
        next_id_after_insert(&db, "t"),
        1000,
        "through the catalog copy"
    );
}

#[test]
fn test_alter_auto_increment_survives_a_crash_through_the_log() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let (_dir, db) = reopened_after(
        &["ALTER TABLE t AUTO_INCREMENT = 1000"],
        "?checkpoint_on_close=off",
    );
    assert_eq!(next_id_after_insert(&db, "t"), 1000, "through the log");
}

#[test]
fn test_truncate_after_alter_auto_increment_restarts_ids() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    for params in ["", "?checkpoint_on_close=off"] {
        let (_dir, db) = reopened_after(
            &[
                "ALTER TABLE t AUTO_INCREMENT = 1000",
                "INSERT INTO t (v) VALUES ('a')",
                "TRUNCATE TABLE t",
            ],
            params,
        );
        assert_eq!(next_id_after_insert(&db, "t"), 1, "with {params:?}");
    }
}

/// The database files as they stand, as a crash would leave them
#[cfg(feature = "test-failpoints")]
fn crash_image(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_truncate_during_catalog_copies_restarts_ids_after_crash() {
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let root = tempdir().unwrap();
    let (first, crashed) = (root.path().join("0"), root.path().join("1"));
    let dsn = |dir: &std::path::Path| format!("file://{}?checkpoint_interval=3600", dir.display());
    let db = Database::open(&dsn(&first)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
        .unwrap();
    let other = db.clone();
    let truncate = std::rc::Rc::new(std::cell::RefCell::new(None));
    let slot = std::rc::Rc::clone(&truncate);
    stoolap::test_failpoints::after_catalog_copies_taken(move || {
        // The TRUNCATE either waits for the checkpoint's guard or lands now
        let (reached, settled) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let waits = reached.clone();
            stoolap::test_failpoints::on_ddl_guard_waiting(move || {
                let _ = waits.send("waits");
            });
            other.execute("TRUNCATE TABLE t", ()).unwrap();
            let _ = reached.send("landed");
        });
        settled.recv().unwrap();
        *slot.borrow_mut() = Some(handle);
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    truncate.borrow_mut().take().unwrap().join().unwrap();
    crash_image(&first, &crashed);
    db.close().unwrap();

    let db = Database::open(&dsn(&crashed)).unwrap();
    assert_eq!(
        next_id_after_insert(&db, "t"),
        1,
        "the TRUNCATE's reset after the crash"
    );
}
