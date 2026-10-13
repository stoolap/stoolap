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

//! A column type change that cannot complete: refused before its record
//! with nothing changed, or durable and kept closed until a reopen that
//! finishes it, wherever it stops

#![cfg(feature = "test-failpoints")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use stoolap::storage::traits::Engine;
use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

const MODIFY: &str = "ALTER TABLE t MODIFY COLUMN k TEXT";

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// `k INTEGER UNIQUE` with rows (1, 7) and (2, 8) sealed
fn sealed(dir: &Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7), (2, 8)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| format!("{:?}", r.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

const OLD: [&str; 2] = ["Integer(1),Integer(7)", "Integer(2),Integer(8)"];
const NEW: [&str; 2] = ["Integer(1),Text(\"7\")", "Integer(2),Text(\"8\")"];

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

/// The volume files of table `t`
fn volume_files(dir: &Path) -> BTreeSet<PathBuf> {
    match std::fs::read_dir(dir.join("volumes/t")) {
        Ok(entries) => entries
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "vol"))
            .collect(),
        Err(_) => BTreeSet::new(),
    }
}

/// Every statement on `t` and every new handle of it is refused
fn assert_closed(db: &Database, case: &str) {
    for sql in [
        "SELECT id, k FROM t ORDER BY id",
        "SELECT k FROM t WHERE id = 1",
    ] {
        let answer = db.query(sql, ()).and_then(|mut rows| match rows.next() {
            Some(row) => row.map(|_| ()),
            None => Ok(()),
        });
        assert!(
            matches!(answer, Err(stoolap::Error::SchemaChanged { .. })),
            "{case}: {sql} answered {answer:?}"
        );
    }
    let mut tx = db.engine().begin_transaction().unwrap();
    assert!(
        tx.get_table("t")
            .and_then(|table| table.collect_all_rows(None).map(|_| ()))
            .is_err(),
        "{case}: a new handle read the table"
    );
    let _ = tx.rollback();
}

#[test]
fn a_hot_point_read_after_the_change_takes_the_arena_row_in_the_new_type() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7), (2, 8)", ())
        .unwrap();
    db.execute(MODIFY, ()).unwrap();
    fp::take_read_paths();
    assert_eq!(rows(&db, "SELECT k FROM t WHERE id = 1"), ["Text(\"7\")"]);
    let paths = fp::take_read_paths();
    assert!(
        paths.contains(&"vs.get_visible_versions_batch.arena_probe"),
        "the point read took another path: {paths:?}"
    );
}

#[test]
fn a_vector_of_another_dimension_refuses_the_change_wherever_it_is() {
    let _guard = FailpointGuard::new();
    for (case, seal, hot) in [
        ("hot", false, true),
        ("sealed", true, false),
        ("both", true, true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(dir.path())).unwrap();
        db.execute("CREATE TABLE v (id INTEGER PRIMARY KEY, e VECTOR(3))", ())
            .unwrap();
        if seal {
            db.execute("INSERT INTO v VALUES (1, '[1, 2, 3]')", ())
                .unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        if hot {
            db.execute("INSERT INTO v VALUES (2, '[4, 5, 6]')", ())
                .unwrap();
        }
        let changed = db.execute("ALTER TABLE v MODIFY COLUMN e VECTOR(4)", ());
        assert!(
            matches!(changed, Err(stoolap::Error::VectorDimensionMismatch { .. })),
            "{case}: {changed:?}"
        );
        assert!(
            db.execute("INSERT INTO v VALUES (3, '[1, 2, 3]')", ())
                .is_ok(),
            "{case}: the column still takes three"
        );
    }
}

#[test]
fn a_dimension_change_over_nulls_takes_the_new_dimension_and_the_index_answers() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE v (id INTEGER PRIMARY KEY, e VECTOR(3))", ())
        .unwrap();
    db.execute("CREATE INDEX v_e ON v(e) USING HNSW", ())
        .unwrap();
    db.execute("INSERT INTO v VALUES (1, NULL)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO v VALUES (2, NULL)", ()).unwrap();
    db.execute("ALTER TABLE v MODIFY COLUMN e VECTOR(4)", ())
        .unwrap();
    assert!(db
        .execute("INSERT INTO v VALUES (3, '[1, 2, 3]')", ())
        .is_err());
    db.execute("INSERT INTO v VALUES (4, '[1, 2, 3, 4]')", ())
        .unwrap();
    assert_eq!(
        rows(
            &db,
            "SELECT id FROM v ORDER BY VEC_DISTANCE_L2(e, '[1, 2, 3, 4]') LIMIT 1"
        ),
        ["Integer(4)"]
    );
}

#[test]
fn a_change_that_returned_survives_a_crash_whatever_the_sync_mode() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let unsynced = format!(
        "file://{}?sync_mode=none&checkpoint_interval=3600",
        dir.path().display()
    );
    let db = Database::open(&unsynced).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 7), (2, 8)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute(MODIFY, ()).unwrap();
    let image = tempfile::tempdir().unwrap();
    crash_image(dir.path(), image.path());
    let recovered = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(rows(&recovered, "SELECT id, k FROM t ORDER BY id"), NEW);
}

#[test]
fn a_change_whose_record_the_log_refuses_leaves_everything_as_it_was() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    db.execute("INSERT INTO t VALUES (3, 9)", ()).unwrap();
    let files = volume_files(dir.path());
    fp::WAL_WRITE_FAIL.store(true, Ordering::SeqCst);
    let changed = db.execute(MODIFY, ());
    fp::WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
    assert!(changed.is_err(), "the record was refused: {changed:?}");
    assert!(
        !matches!(changed, Err(stoolap::Error::WalOutcomeUnknown { .. })),
        "{changed:?}"
    );
    assert_eq!(
        rows(&db, "SELECT id, k FROM t ORDER BY id"),
        [
            "Integer(1),Integer(7)",
            "Integer(2),Integer(8)",
            "Integer(3),Integer(9)"
        ]
    );
    assert_eq!(volume_files(dir.path()), files, "the rewrites were removed");
}

#[test]
fn a_change_whose_record_may_or_may_not_be_in_the_log_keeps_the_table_closed() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    fp::WAL_WRITE_PARTIAL.store(true, Ordering::SeqCst);
    fp::WAL_CUT_FAIL.store(true, Ordering::SeqCst);
    let changed = db.execute(MODIFY, ());
    fp::WAL_CUT_FAIL.store(false, Ordering::SeqCst);
    fp::WAL_WRITE_PARTIAL.store(false, Ordering::SeqCst);
    assert!(
        matches!(changed, Err(stoolap::Error::WalOutcomeUnknown { .. })),
        "{changed:?}"
    );
    assert_closed(&db, "unknown outcome");
}

#[test]
fn a_change_whose_manifest_fails_stays_closed_and_completes_at_the_next_open() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    fp::MANIFEST_WRITE_FAIL.store(true, Ordering::SeqCst);
    let changed = db.execute(MODIFY, ());
    fp::MANIFEST_WRITE_FAIL.store(false, Ordering::SeqCst);
    let error = changed.expect_err("the manifest write failed").to_string();
    assert!(error.contains("durable"), "{error}");
    assert_closed(&db, "manifest failed");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(rows(&db, "SELECT id, k FROM t ORDER BY id"), NEW);
}

#[test]
fn a_checkpoint_holding_a_cut_across_a_failed_manifest_does_not_skip_the_change() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let other = db.clone();
    let failed = std::rc::Rc::new(std::cell::Cell::new(false));
    let seen = std::rc::Rc::clone(&failed);
    fp::after_checkpoint_cut_taken(move || {
        fp::MANIFEST_WRITE_FAIL.store(true, Ordering::SeqCst);
        seen.set(other.execute(MODIFY, ()).is_err());
        fp::MANIFEST_WRITE_FAIL.store(false, Ordering::SeqCst);
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert!(
        failed.get(),
        "the change ran inside the checkpoint and failed"
    );
    let image = tempfile::tempdir().unwrap();
    crash_image(dir.path(), image.path());
    let recovered = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(rows(&recovered, "SELECT id, k FROM t ORDER BY id"), NEW);
}

#[test]
fn a_crash_while_the_rewrites_are_written_reopens_the_old_table() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let before = volume_files(dir.path());
    let image = tempfile::tempdir().unwrap();
    let (from, to) = (dir.path().to_path_buf(), image.path().to_path_buf());
    fp::after_retype_outputs_written(move || crash_image(&from, &to));
    db.execute(MODIFY, ()).unwrap();
    assert!(
        volume_files(image.path()).len() > before.len(),
        "the image holds the unregistered rewrite"
    );
    let recovered = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(rows(&recovered, "SELECT id, k FROM t ORDER BY id"), OLD);
    let kept: BTreeSet<_> = before
        .iter()
        .map(|path| image.path().join(path.strip_prefix(dir.path()).unwrap()))
        .collect();
    assert_eq!(volume_files(image.path()), kept, "the rewrite was removed");
}

#[test]
fn a_crash_after_the_manifest_reopens_the_new_table_without_the_old_volumes() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let before = volume_files(dir.path());
    let image = tempfile::tempdir().unwrap();
    let (from, to) = (dir.path().to_path_buf(), image.path().to_path_buf());
    fp::after_retype_manifest_written(move || crash_image(&from, &to));
    db.execute(MODIFY, ()).unwrap();
    let recovered = Database::open(&dsn(image.path())).unwrap();
    assert_eq!(rows(&recovered, "SELECT id, k FROM t ORDER BY id"), NEW);
    let old: BTreeSet<_> = before
        .iter()
        .map(|path| image.path().join(path.strip_prefix(dir.path()).unwrap()))
        .collect();
    assert!(
        volume_files(image.path()).is_disjoint(&old),
        "the replaced volumes went at the reopen"
    );
}

/// `k TEXT` with row (1, '007') sealed
fn sealed_text(dir: &Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '007')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

#[test]
fn a_change_through_the_engine_or_a_transaction_survives_a_crash() {
    let _guard = FailpointGuard::new();
    for through_transaction in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = sealed_text(dir.path());
        if through_transaction {
            let mut tx = db.engine().begin_transaction().unwrap();
            tx.modify_table_column(
                "t",
                stoolap::core::SchemaColumn::new(
                    1,
                    "k",
                    stoolap::core::DataType::Integer,
                    true,
                    false,
                ),
            )
            .unwrap();
            tx.commit().unwrap();
        } else {
            db.engine()
                .modify_column("t", "k", stoolap::core::DataType::Integer, true)
                .unwrap();
        }
        let image = tempfile::tempdir().unwrap();
        crash_image(dir.path(), image.path());
        let recovered = Database::open(&dsn(image.path())).unwrap();
        assert_eq!(
            rows(&recovered, "SELECT id, k FROM t"),
            ["Integer(1),Integer(7)"],
            "through a transaction: {through_transaction}"
        );
    }
}

#[test]
fn a_default_the_volume_does_not_hold_is_cast_and_checked() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("ALTER TABLE t ADD COLUMN v TEXT DEFAULT 'bad'", ())
        .unwrap();
    db.execute("ALTER TABLE t ADD COLUMN w TEXT DEFAULT '007'", ())
        .unwrap();
    let refused = db.execute("ALTER TABLE t MODIFY COLUMN v INTEGER NOT NULL", ());
    assert!(
        matches!(refused, Err(stoolap::Error::NotNullConstraint { .. })),
        "{refused:?}"
    );
    db.execute("ALTER TABLE t MODIFY COLUMN w INTEGER", ())
        .unwrap();
    assert_eq!(
        rows(&db, "SELECT id, v, w FROM t"),
        ["Integer(1),Text(\"bad\"),Integer(7)"]
    );
}

#[test]
fn a_change_drops_the_zone_maps_analyze_took_in_the_old_type() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '007')", ()).unwrap();
    db.execute("ANALYZE t", ()).unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN k INTEGER", ())
        .unwrap();
    assert_eq!(rows(&db, "SELECT id FROM t WHERE k = 7"), ["Integer(1)"]);
}

#[test]
fn an_analyze_that_read_the_old_type_does_not_publish_its_zone_maps_after_the_change() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '007')", ()).unwrap();
    let other = db.clone();
    fp::after_analyze_rows_collected(move || {
        let changed =
            std::thread::spawn(move || other.execute("ALTER TABLE t MODIFY COLUMN k INTEGER", ()));
        changed.join().unwrap().unwrap();
    });
    let _ = db.execute("ANALYZE t", ());
    assert_eq!(rows(&db, "SELECT id FROM t WHERE k = 7"), ["Integer(1)"]);
}

#[test]
fn a_volume_that_cannot_be_read_leaves_no_rewrite_behind() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    db.execute("INSERT INTO t VALUES (3, 9)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = volume_files(dir.path());
    assert_eq!(before.len(), 2);
    fp::RETYPE_READS_BEFORE_FAIL.store(2, Ordering::SeqCst);
    let refused = db.execute(MODIFY, ());
    fp::RETYPE_READS_BEFORE_FAIL.store(0, Ordering::SeqCst);
    assert!(refused.is_err(), "the second volume failed to read");
    assert_eq!(
        volume_files(dir.path()),
        before,
        "the first rewrite was removed"
    );
    assert_eq!(
        rows(&db, "SELECT id, k FROM t ORDER BY id"),
        [
            "Integer(1),Integer(7)",
            "Integer(2),Integer(8)",
            "Integer(3),Integer(9)"
        ]
    );
}

#[test]
fn a_column_change_before_a_type_change_replays_over_the_rewritten_volume() {
    let _guard = FailpointGuard::new();
    for change in [
        "ALTER TABLE t ADD COLUMN y INTEGER",
        "ALTER TABLE t DROP COLUMN z",
        "ALTER TABLE t RENAME COLUMN z TO y",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(dir.path())).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT, z INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, '007', 3)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute(change, ()).unwrap();
        db.execute("ALTER TABLE t MODIFY COLUMN k INTEGER", ())
            .unwrap();
        let image = tempfile::tempdir().unwrap();
        crash_image(dir.path(), image.path());
        let recovered =
            Database::open(&dsn(image.path())).unwrap_or_else(|error| panic!("{change}: {error}"));
        assert_eq!(
            rows(&recovered, "SELECT id, k FROM t"),
            ["Integer(1),Integer(7)"],
            "{change}"
        );
    }
}
