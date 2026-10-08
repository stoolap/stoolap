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

//! DROP COLUMN changes the schema, the rows, the index bindings and the
//! foreign key positions together: an index after the dropped column
//! answers for its column at once, an index over it goes with it, a key
//! after it keeps its constraint, and a statement that overlaps the drop
//! reads and writes the whole old table or the whole new one.

use stoolap::core::Error;
use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

/// `(index, column)` pairs of `table`, as SHOW INDEXES lists them
fn indexes(db: &Database, table: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = db
        .query(&format!("SHOW INDEXES FROM {table}"), ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(1).unwrap(), r.get(2).unwrap())
        })
        .collect();
    out.sort();
    out
}

fn file_dsn(dir: &tempfile::TempDir, options: &str) -> String {
    format!("file://{}{options}", dir.path().display())
}

const REPLAY: &str = "?checkpoint_on_close=off&checkpoint_interval=0";

/// A test below fails the log process-wide, so every test here takes turns
#[cfg(feature = "test-failpoints")]
fn serial() -> stoolap::test_failpoints::FailpointGuard {
    stoolap::test_failpoints::FailpointGuard::new()
}

#[cfg(not(feature = "test-failpoints"))]
fn serial() {}

// --- Index bindings --------------------------------------------------------

fn both(db: &Database, statements: &[&str]) {
    for table in ["t", "c"] {
        for sql in statements {
            db.execute(&sql.replace("{t}", table), ()).unwrap();
        }
    }
}

fn drop_setup(db: &Database) {
    both(
        db,
        &[
            "CREATE TABLE {t} (x INTEGER, id INTEGER PRIMARY KEY, a INTEGER, u INTEGER)",
            "INSERT INTO {t} VALUES (7, 1, 10, 100), (8, 2, 20, 200), (9, 3, 10, 300)",
        ],
    );
    db.execute("CREATE INDEX ia ON t(a)", ()).unwrap();
    db.execute("CREATE UNIQUE INDEX iu ON t(u)", ()).unwrap();
    db.execute("CREATE INDEX ixa ON t(x, a)", ()).unwrap();
}

fn drop_and_write(db: &Database) {
    both(
        db,
        &[
            "ALTER TABLE {t} DROP COLUMN x",
            "INSERT INTO {t} VALUES (4, 10, 400)",
            "UPDATE {t} SET a = 30 WHERE id = 2",
            "UPDATE {t} SET u = 350 WHERE id = 3",
            "DELETE FROM {t} WHERE id = 1",
        ],
    );
}

fn assert_bindings_follow(db: &Database, stage: &str) {
    for filter in [
        "a = 10",
        "a = 20",
        "a = 30",
        "a + 0 = 10",
        "u = 350",
        "u = 300",
        "u = 400",
        "id = 4",
        "id BETWEEN 2 AND 4",
    ] {
        assert_eq!(
            ids(db, &format!("SELECT id FROM t WHERE {filter}")),
            ids(db, &format!("SELECT id FROM c WHERE {filter}")),
            "{stage}: WHERE {filter}"
        );
    }
    assert_eq!(
        indexes(db, "t"),
        [
            ("ia".to_string(), "a".to_string()),
            ("iu".to_string(), "u".to_string()),
        ],
        "{stage}: the index over the dropped column goes with it"
    );
    assert!(
        db.execute("INSERT INTO t VALUES (5, 1, 400)", ()).is_err(),
        "{stage}: u keeps its unique index"
    );
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn indexes_follow_their_columns_right_after_a_drop() {
    let _serial = serial();
    let db = Database::open("memory://drop_transition_bindings").unwrap();
    drop_setup(&db);
    drop_and_write(&db);
    assert_bindings_follow(&db, "memory");
}

#[test]
fn indexes_follow_their_columns_right_after_a_drop_on_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(&file_dsn(&dir, REPLAY)).unwrap();
        drop_setup(&db);
        drop_and_write(&db);
        assert_bindings_follow(&db, "hot rows");
        db.close().unwrap();
    }
    let db = Database::open(&file_dsn(&dir, REPLAY)).unwrap();
    assert_bindings_follow(&db, "after replay");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    let db = Database::open(&file_dsn(&dir, "")).unwrap();
    assert_bindings_follow(&db, "after a checkpoint and reopen");
}

// --- Foreign keys ----------------------------------------------------------

fn fk_setup(db: &Database) {
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "INSERT INTO p VALUES (1), (2)",
        "CREATE TABLE r (x INTEGER, id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id))",
        "INSERT INTO r VALUES (0, 1, 1)",
        "CREATE TABLE k (x INTEGER, id INTEGER PRIMARY KEY, \
         pid INTEGER REFERENCES p(id) ON DELETE CASCADE)",
        "INSERT INTO k VALUES (0, 1, 2), (0, 2, 2)",
        "ALTER TABLE r DROP COLUMN x",
        "ALTER TABLE k DROP COLUMN x",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn assert_keys_hold(db: &Database, stage: &str) {
    assert!(
        matches!(
            db.execute("INSERT INTO r VALUES (9, 99)", ()),
            Err(Error::ForeignKeyViolation { .. })
        ),
        "{stage}: an orphan child is refused"
    );
    assert!(
        matches!(
            db.execute("DELETE FROM p WHERE id = 1", ()),
            Err(Error::ForeignKeyViolation { .. })
        ),
        "{stage}: the parent of a restricted child stays"
    );
    db.execute("INSERT INTO r VALUES (3, 2)", ()).unwrap();
    db.execute("DELETE FROM r WHERE id = 3", ()).unwrap();
}

fn assert_cascade(db: &Database, stage: &str) {
    db.execute("DELETE FROM p WHERE id = 2", ()).unwrap();
    assert_eq!(
        ids(db, "SELECT id FROM k"),
        Vec::<i64>::new(),
        "{stage}: the cascade reached the children"
    );
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn foreign_keys_after_a_dropped_column_hold() {
    let _serial = serial();
    let db = Database::open("memory://drop_transition_fk").unwrap();
    fk_setup(&db);
    assert_keys_hold(&db, "memory");
    assert_cascade(&db, "memory");
}

#[test]
fn foreign_keys_after_a_dropped_column_hold_across_replay_and_reopen() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(&file_dsn(&dir, REPLAY)).unwrap();
        fk_setup(&db);
        assert_keys_hold(&db, "hot rows");
        db.close().unwrap();
    }
    let db = Database::open(&file_dsn(&dir, REPLAY)).unwrap();
    assert_keys_hold(&db, "after replay");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    let db = Database::open(&file_dsn(&dir, "")).unwrap();
    assert_keys_hold(&db, "after a checkpoint and reopen");
    assert_cascade(&db, "after a checkpoint and reopen");
}

// --- Statements that overlap the drop ---------------------------------------

#[cfg(feature = "test-failpoints")]
fn overlapping_statements(dsn: &str) {
    use std::sync::mpsc;
    use std::time::Duration;
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100), (2, 20, 200)", ())
        .unwrap();
    let other = db.clone();
    let (sent, received) = mpsc::channel();
    let (ran, finished) = mpsc::channel::<()>();
    stoolap::test_failpoints::after_drop_column_recorded(move || {
        std::thread::spawn(move || {
            let read = |sql: &str| -> Result<Vec<i64>, Error> {
                other
                    .query(sql, ())
                    .and_then(|rows| rows.map(|r| r.map(|r| r.get(0).unwrap())).collect())
            };
            let reads = [
                (
                    "SELECT a FROM t WHERE id = 2",
                    read("SELECT a FROM t WHERE id = 2"),
                    vec![200],
                ),
                (
                    "SELECT a FROM t WHERE a > 0 ORDER BY a",
                    read("SELECT a FROM t WHERE a > 0 ORDER BY a"),
                    vec![100, 200],
                ),
                (
                    "SELECT SUM(a) FROM t",
                    read("SELECT SUM(a) FROM t"),
                    vec![300],
                ),
            ];
            let update = other.execute("UPDATE t SET a = a + 1 WHERE id = 1", ());
            sent.send((reads, update)).unwrap();
            ran.send(()).unwrap();
        });
        // The drop waits here for the statements, unless they wait for it
        let _ = finished.recv_timeout(Duration::from_secs(2));
    });
    db.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
    let (reads, update) = received
        .recv_timeout(Duration::from_secs(10))
        .expect("the overlapping statements ran");
    for (sql, read, expected) in reads {
        match read {
            Err(Error::SchemaChanged { .. }) => {}
            read => assert_eq!(read.unwrap(), expected, "{sql}"),
        }
    }
    let a1 = match update {
        Err(Error::SchemaChanged { .. }) => 100,
        update => {
            update.unwrap();
            101
        }
    };
    assert_eq!(ids(&db, "SELECT a FROM t ORDER BY id"), [a1, 200]);
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
fn writes_while_closed(dsn: &str) {
    use std::sync::mpsc;
    use std::time::Duration;
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX ua ON t(a)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100)", ()).unwrap();
    let other = db.clone();
    let (sent, received) = mpsc::channel();
    stoolap::test_failpoints::after_drop_column_recorded(move || {
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let inserted = other.execute("INSERT INTO t VALUES (2, 20, 100)", ());
            let mut tx = other.begin().unwrap();
            let updated = tx.execute("UPDATE t SET a = 101 WHERE id = 1", ());
            let _ = tx.rollback();
            let _ = done.send((inserted, updated));
        });
        sent.send(finished.recv_timeout(Duration::from_secs(2)).ok())
            .unwrap();
    });
    db.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
    let (inserted, updated) = received
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .expect("the writes finished while the table was closed");
    assert!(
        matches!(inserted, Err(Error::SchemaChanged { .. })),
        "insert: {inserted:?}"
    );
    assert!(
        matches!(updated, Err(Error::SchemaChanged { .. })),
        "update in a transaction: {updated:?}"
    );
    assert_eq!(ids(&db, "SELECT a FROM t"), [100]);
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn writes_while_closed_are_refused_at_the_statement() {
    let _serial = serial();
    writes_while_closed("memory://drop_transition_writes_closed");
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn statements_overlapping_a_drop_see_one_table() {
    let _serial = serial();
    overlapping_statements("memory://drop_transition_overlap");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn statements_overlapping_a_drop_see_one_table_on_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    overlapping_statements(&file_dsn(&dir, ""));
}

/// A statement that starts after the hot rows are published but before the
/// cold mappings follow reads a sealed row in its own column
#[cfg(feature = "test-failpoints")]
#[test]
fn a_sealed_row_read_before_the_cold_mappings_follow_reads_its_column() {
    let _serial = serial();
    use std::sync::mpsc;
    use std::time::Duration;
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&file_dsn(&dir, "")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let other = db.clone();
    let (sent, received) = mpsc::channel();
    let (ran, finished) = mpsc::channel::<()>();
    stoolap::test_failpoints::after_drop_hot_published(move || {
        std::thread::spawn(move || {
            let read = |sql: &str| -> Result<Vec<i64>, Error> {
                other
                    .query(sql, ())
                    .and_then(|rows| rows.map(|r| r.map(|r| r.get(0).unwrap())).collect())
            };
            let pk = other
                .prepare("SELECT a FROM t WHERE id = $1")
                .and_then(|select| {
                    select
                        .query((1,))
                        .and_then(|rows| rows.map(|r| r.map(|r| r.get(0).unwrap())).collect())
                });
            let reads = [
                ("prepared PK SELECT", pk),
                ("PK SELECT", read("SELECT a FROM t WHERE id = 1")),
                ("scan", read("SELECT a FROM t WHERE a > 0")),
            ];
            sent.send(reads).unwrap();
            ran.send(()).unwrap();
        });
        let _ = finished.recv_timeout(Duration::from_secs(2));
    });
    db.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
    let reads = received
        .recv_timeout(Duration::from_secs(10))
        .expect("the reads ran");
    for (what, read) in reads {
        match read {
            Err(Error::SchemaChanged { .. }) => {}
            read => assert_eq!(read.unwrap(), [100], "{what}"),
        }
    }
}

/// A commit whose index updates are in when a drop completes has its
/// publication refused, and its undo leaves every index as it was
#[cfg(feature = "test-failpoints")]
fn refused_commit_undone_across_a_drop(dsn: &str) {
    let db = Database::open(dsn).unwrap();
    drop_setup(&db);
    let other = db.clone();
    stoolap::test_failpoints::after_indexes_published(move || {
        for table in ["t", "c"] {
            other
                .execute(&format!("ALTER TABLE {table} DROP COLUMN x"), ())
                .unwrap();
        }
    });
    let inserted = db.execute("INSERT INTO t VALUES (5, 4, 10, 500)", ());
    assert!(
        matches!(inserted, Err(Error::SchemaChanged { .. })),
        "{inserted:?}"
    );
    db.execute("INSERT INTO t VALUES (5, 10, 500)", ()).unwrap();
    db.execute("INSERT INTO c VALUES (5, 10, 500)", ()).unwrap();
    for filter in ["a = 10", "a = 20", "u = 500", "u = 300", "id = 5"] {
        assert_eq!(
            ids(&db, &format!("SELECT id FROM t WHERE {filter}")),
            ids(&db, &format!("SELECT id FROM c WHERE {filter}")),
            "WHERE {filter}"
        );
    }
    assert!(
        db.execute("INSERT INTO t VALUES (6, 1, 500)", ()).is_err(),
        "u keeps its unique index"
    );
}

#[cfg(feature = "test-failpoints")]
fn insert_across_a_drop_before_index_capture(dsn: &str) {
    let db = Database::open(dsn).unwrap();
    drop_setup(&db);
    let other = db.clone();
    stoolap::test_failpoints::before_commit_index_capture(move || {
        for table in ["t", "c"] {
            other
                .execute(&format!("ALTER TABLE {table} DROP COLUMN x"), ())
                .unwrap();
        }
    });
    // Through the bindings after the drop, the row's a lands on u's position
    let inserted = db.execute("INSERT INTO t VALUES (4, 5, 100, 500)", ());
    assert!(
        matches!(inserted, Err(Error::SchemaChanged { .. })),
        "{inserted:?}"
    );
    for filter in ["a = 5", "a = 100", "u = 100", "u = 500", "id = 5"] {
        assert_eq!(
            ids(&db, &format!("SELECT id FROM t WHERE {filter}")),
            ids(&db, &format!("SELECT id FROM c WHERE {filter}")),
            "WHERE {filter}"
        );
    }
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_commit_across_a_drop_before_its_index_capture_writes_no_key() {
    let _serial = serial();
    insert_across_a_drop_before_index_capture("memory://drop_transition_before_capture");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_commit_across_a_drop_before_its_index_capture_writes_no_key_on_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    insert_across_a_drop_before_index_capture(&file_dsn(&dir, ""));
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_refused_commit_across_a_drop_leaves_the_indexes_whole() {
    let _serial = serial();
    refused_commit_undone_across_a_drop("memory://drop_transition_refused_commit");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_refused_commit_across_a_drop_leaves_the_indexes_whole_on_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    refused_commit_undone_across_a_drop(&file_dsn(&dir, ""));
}

// --- Side files across a replayed drop --------------------------------------

fn side_files(dir: &tempfile::TempDir) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.path().join("volumes").join("t"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".sidx"))
        .collect();
    names.sort();
    names
}

#[test]
fn a_side_file_built_after_a_drop_survives_its_replay() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let k_rows = |db: &Database| ids(db, "SELECT id FROM t WHERE k = 7");
    let built = {
        let db = Database::open(&file_dsn(&dir, REPLAY)).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, padding INTEGER, k INTEGER)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX ip ON t(padding)", ()).unwrap();
        db.execute(
            "INSERT INTO t SELECT value, value % 50, value % 100 FROM generate_series(1, 2000)",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert!(!side_files(&dir).is_empty(), "the seal covered ip");
        db.execute("ALTER TABLE t DROP COLUMN padding", ()).unwrap();
        assert!(
            side_files(&dir).is_empty(),
            "the file goes with the index over the dropped column"
        );
        db.execute("CREATE INDEX ik ON t(k)", ()).unwrap();
        db.execute("PRAGMA INDEX_BACKFILL", ()).unwrap();
        assert_eq!(k_rows(&db).len(), 20);
        db.close().unwrap();
        side_files(&dir)
    };
    assert!(!built.is_empty(), "the backfill built a side file");
    let db = Database::open(&file_dsn(&dir, REPLAY)).unwrap();
    assert_eq!(
        side_files(&dir),
        built,
        "the replayed drop keeps the side file of the index created after it"
    );
    assert_eq!(k_rows(&db).len(), 20);
}

// --- A drop the log refuses -------------------------------------------------

#[cfg(feature = "test-failpoints")]
#[test]
fn a_drop_the_log_refuses_changes_nothing() {
    use std::sync::atomic::Ordering;
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let dsn = file_dsn(
        &dir,
        "?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
    );
    let db = Database::open(&dsn).unwrap();
    drop_setup(&db);
    let identities =
        |db: &Database| ["ia", "iu", "ixa"].map(|i| db.engine().index_identity("t", i));
    let before = identities(&db);
    stoolap::test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
    let dropped = db.execute("ALTER TABLE t DROP COLUMN x", ());
    stoolap::test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
    assert!(dropped.is_err());
    let unchanged = |db: &Database, stage: &str| {
        assert_eq!(
            indexes(db, "t"),
            [
                ("ia".to_string(), "a".to_string()),
                ("iu".to_string(), "u".to_string()),
                ("ixa".to_string(), "(x, a)".to_string()),
            ],
            "{stage}"
        );
        assert_eq!(identities(db), before, "{stage}");
        assert_eq!(ids(db, "SELECT id FROM t WHERE a = 10"), [1, 3], "{stage}");
        assert_eq!(ids(db, "SELECT id FROM t WHERE x = 8"), [2], "{stage}");
    };
    unchanged(&db, "after the refused drop");
    let _ = db.close();
    let db = Database::open(&dsn).unwrap();
    unchanged(&db, "after reopen");
    db.execute("INSERT INTO t VALUES (6, 4, 10, 600)", ())
        .unwrap();
    assert_eq!(ids(&db, "SELECT id FROM t WHERE a = 10"), [1, 3, 4]);
}

// --- The log order across the drop -------------------------------------------

#[cfg(feature = "test-failpoints")]
fn rows_and_drop_replayed(hook_marker: bool) -> Vec<(i64, i64)> {
    let dir = tempfile::tempdir().unwrap();
    let dsn = file_dsn(
        &dir,
        "?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10, 100)", ()).unwrap();
        let other = db.clone();
        let drop = move || {
            other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
        };
        if hook_marker {
            stoolap::test_failpoints::before_commit_marker(drop);
        } else {
            stoolap::test_failpoints::after_indexes_published(drop);
        }
        let _ = db.execute("INSERT INTO t VALUES (2, 20, 200)", ());
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.query("SELECT id, a FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap())
        })
        .collect()
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_commit_marker_after_the_drop_record_replays_its_rows_before_it() {
    let _serial = serial();
    assert_eq!(rows_and_drop_replayed(true), [(1, 100), (2, 200)]);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn rows_logged_before_a_drop_whose_publication_is_refused_do_not_replay() {
    let _serial = serial();
    assert_eq!(rows_and_drop_replayed(false), [(1, 100)]);
}
