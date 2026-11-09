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

//! A dropped table leaves no files behind, and the log replays a rename
//! or a drop over the volumes an open finds without losing their rows

use std::path::Path;

use stoolap::Database;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

fn rows(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect()
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &Path, to: &Path) {
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

fn run(dir: &Path, statements: &[&str]) {
    let db = Database::open(&dsn(dir)).unwrap();
    for sql in statements {
        db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    db.close().unwrap();
}

#[test]
fn a_copying_alter_over_a_table_on_disk_keeps_its_rows_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    run(
        dir.path(),
        &[
            "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
            "INSERT INTO t VALUES (1, 1), (2, 2)",
        ],
    );
    run(
        dir.path(),
        &[
            "CREATE TABLE tmp (id INTEGER PRIMARY KEY, n INTEGER, w INTEGER)",
            "INSERT INTO tmp SELECT id, n * 10, 7 FROM t",
            "ALTER TABLE t RENAME TO bak",
            "ALTER TABLE tmp RENAME TO t",
            "DROP TABLE bak",
        ],
    );
    for open in 3..=4 {
        let db = Database::open(&dsn(dir.path())).unwrap();
        assert_eq!(
            rows(&db, "SELECT id, n FROM t ORDER BY id"),
            [(1, 10), (2, 20)],
            "open {open}"
        );
        db.close().unwrap();
    }
    assert!(
        !dir.path().join("volumes").join("bak").exists(),
        "the dropped table left its directory"
    );
}

#[test]
fn a_table_renamed_to_a_dropped_name_takes_it() {
    let dir = tempfile::tempdir().unwrap();
    run(
        dir.path(),
        &[
            "CREATE TABLE a (id INTEGER PRIMARY KEY, n INTEGER)",
            "CREATE TABLE b (id INTEGER PRIMARY KEY, n INTEGER)",
            "INSERT INTO a VALUES (1, 1)",
            "INSERT INTO b VALUES (2, 2)",
        ],
    );
    run(dir.path(), &["DROP TABLE a", "ALTER TABLE b RENAME TO a"]);
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(rows(&db, "SELECT id, n FROM a"), [(2, 2)]);
}

#[test]
fn a_rename_after_the_last_checkpoint_replays_over_the_moved_files() {
    let dir = tempfile::tempdir().unwrap();
    run(
        dir.path(),
        &[
            "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
            "INSERT INTO t VALUES (1, 1), (2, 2)",
        ],
    );
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("INSERT INTO t VALUES (3, 3)", ()).unwrap();
    db.execute("ALTER TABLE t RENAME TO u", ()).unwrap();
    let image = tempfile::tempdir().unwrap();
    crash_image(dir.path(), image.path());
    let recovered = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(
        rows(&recovered, "SELECT id, n FROM u ORDER BY id"),
        [(1, 1), (2, 2), (3, 3)]
    );
}

#[test]
fn a_delete_before_a_rename_after_the_last_checkpoint_holds_over_two_opens() {
    let dir = tempfile::tempdir().unwrap();
    run(
        dir.path(),
        &[
            "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
            "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)",
        ],
    );
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    db.execute("ALTER TABLE t RENAME TO u", ()).unwrap();
    let image = tempfile::tempdir().unwrap();
    crash_image(dir.path(), image.path());
    for open in 1..=2 {
        let recovered = Database::open(&dsn(image.path())).unwrap();
        assert_eq!(
            rows(&recovered, "SELECT id, n FROM u ORDER BY id"),
            [(2, 2), (3, 3)],
            "open {open}"
        );
        recovered.close().unwrap();
    }
}

#[test]
fn a_rename_to_a_dropped_name_after_the_last_checkpoint_keeps_the_rows_it_moved() {
    let dir = tempfile::tempdir().unwrap();
    run(
        dir.path(),
        &[
            "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
            "CREATE TABLE u (id INTEGER PRIMARY KEY, n INTEGER)",
            "INSERT INTO t VALUES (1, 1)",
            "INSERT INTO u VALUES (2, 2), (3, 3)",
        ],
    );
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("DROP TABLE t", ()).unwrap();
    db.execute("ALTER TABLE u RENAME TO t", ()).unwrap();
    let image = tempfile::tempdir().unwrap();
    crash_image(dir.path(), image.path());
    let recovered = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(
        rows(&recovered, "SELECT id, n FROM t ORDER BY id"),
        [(2, 2), (3, 3)]
    );
}

#[test]
fn a_manifest_a_dropped_table_left_behind_opens_with_the_rows_the_log_kept() {
    let dir = tempfile::tempdir().unwrap();
    let volumes = dir.path().join("volumes");
    run(
        dir.path(),
        &[
            "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
            "INSERT INTO t VALUES (1, 1), (2, 2)",
            "CREATE TABLE s (id INTEGER PRIMARY KEY)",
            "INSERT INTO s VALUES (1)",
        ],
    );
    run(dir.path(), &["TRUNCATE TABLE s"]);
    // An emptied manifest of an earlier checkpoint, as a drop wrote one
    // before drops removed it
    let stale = std::fs::read(volumes.join("s").join("manifest.bin")).unwrap();
    run(
        dir.path(),
        &[
            "CREATE TABLE tmp (id INTEGER PRIMARY KEY, n INTEGER, w INTEGER)",
            "INSERT INTO tmp SELECT id, n * 10, 7 FROM t",
            "ALTER TABLE t RENAME TO bak",
            "ALTER TABLE tmp RENAME TO t",
            "DROP TABLE bak",
        ],
    );
    std::fs::create_dir_all(volumes.join("bak")).unwrap();
    std::fs::write(volumes.join("bak").join("manifest.bin"), stale).unwrap();
    for open in 1..=2 {
        let db = Database::open(&dsn(dir.path())).unwrap();
        assert_eq!(
            rows(&db, "SELECT id, n FROM t ORDER BY id"),
            [(1, 10), (2, 20)],
            "open {open}"
        );
        db.close().unwrap();
    }
    assert!(
        !volumes.join("bak").exists(),
        "the dropped table left its directory"
    );
}

#[cfg(feature = "test-failpoints")]
mod failpoints {
    use super::*;
    use stoolap::test_failpoints::{self as fp, FailpointGuard};

    #[test]
    fn a_checkpoint_holding_a_dropped_table_writes_no_manifest_for_it() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        run(
            dir.path(),
            &[
                "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
                "INSERT INTO t VALUES (1, 1)",
            ],
        );
        let db = Database::open(&dsn(dir.path())).unwrap();
        let other = db.clone();
        fp::after_checkpoint_managers_taken(move || {
            other.execute("DROP TABLE t", ()).map(drop).unwrap()
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert!(
            !dir.path().join("volumes").join("t").exists(),
            "the checkpoint wrote the dropped table's manifest again"
        );
    }

    #[test]
    fn a_rename_between_the_checkpoint_cut_and_its_catalog_records_reopens() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        run(
            dir.path(),
            &[
                "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
                "INSERT INTO t VALUES (1, 1)",
            ],
        );
        let db = Database::open(&dsn(dir.path())).unwrap();
        let other = db.clone();
        fp::after_checkpoint_cut_taken(move || {
            other
                .execute("ALTER TABLE t RENAME TO u", ())
                .map(drop)
                .unwrap()
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let image = tempfile::tempdir().unwrap();
        crash_image(dir.path(), image.path());
        let recovered = Database::open(&dsn(image.path())).unwrap();
        assert_eq!(rows(&recovered, "SELECT id, n FROM u"), [(1, 1)]);
    }

    #[test]
    fn two_renames_between_the_checkpoint_cut_and_its_catalog_records_reopen() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        run(
            dir.path(),
            &[
                "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
                "INSERT INTO t VALUES (1, 1)",
            ],
        );
        let db = Database::open(&dsn(dir.path())).unwrap();
        let other = db.clone();
        fp::after_checkpoint_cut_taken(move || {
            for sql in ["ALTER TABLE t RENAME TO u", "ALTER TABLE u RENAME TO v"] {
                other.execute(sql, ()).map(drop).unwrap()
            }
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let image = tempfile::tempdir().unwrap();
        crash_image(dir.path(), image.path());
        let recovered = Database::open(&dsn(image.path())).unwrap();
        assert_eq!(rows(&recovered, "SELECT id, n FROM v"), [(1, 1)]);
    }

    #[test]
    fn a_rename_and_a_drop_between_the_checkpoint_cut_and_its_catalog_records_reopen() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        run(
            dir.path(),
            &[
                "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
                "CREATE TABLE s (id INTEGER PRIMARY KEY, n INTEGER)",
                "INSERT INTO t VALUES (1, 1)",
                "INSERT INTO s VALUES (2, 2)",
            ],
        );
        let db = Database::open(&dsn(dir.path())).unwrap();
        let other = db.clone();
        fp::after_checkpoint_cut_taken(move || {
            for sql in ["ALTER TABLE t RENAME TO u", "DROP TABLE u"] {
                other.execute(sql, ()).map(drop).unwrap()
            }
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let image = tempfile::tempdir().unwrap();
        crash_image(dir.path(), image.path());
        let recovered = Database::open(&dsn(image.path())).unwrap();
        assert_eq!(rows(&recovered, "SELECT id, n FROM s"), [(2, 2)]);
        assert!(
            recovered.query("SELECT id FROM u", ()).is_err(),
            "u is dropped"
        );
        assert!(
            recovered.query("SELECT id FROM t", ()).is_err(),
            "t is renamed"
        );
    }
}
