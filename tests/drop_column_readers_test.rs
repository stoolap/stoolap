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

//! A reader that took its table before a column change and reads after a
//! step of it returns the table as it was, the table as it is, or
//! SchemaChanged.

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use stoolap::core::{DataType, Error, Value};
use stoolap::test_failpoints as fp;
use stoolap::Database;

type Rows = Result<Vec<Vec<Value>>, Error>;

const OPTS: &str = "?checkpoint_on_close=off&checkpoint_interval=0";

fn rows(db: &Database, sql: &str) -> Rows {
    db.query(sql, ())?
        .map(|r| {
            r.map(|r| {
                (0..r.len())
                    .map(|i| {
                        r.get_value(i)
                            .cloned()
                            .unwrap_or(Value::Null(DataType::Null))
                    })
                    .collect()
            })
        })
        .collect()
}

fn ints(values: &[&[i64]]) -> Vec<Vec<Value>> {
    values
        .iter()
        .map(|r| r.iter().map(|&v| Value::Integer(v)).collect())
        .collect()
}

/// The engines a family runs on: memory, file with hot rows, file with
/// sealed rows
fn engines(dir: &tempfile::TempDir, tag: &str) -> Vec<(&'static str, String)> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let base = format!("file://{}", dir.path().display());
    vec![
        ("memory", format!("memory://readers_{tag}_{n}")),
        ("file hot", format!("{base}/{tag}_{n}_hot{OPTS}")),
        ("file sealed", format!("{base}/{tag}_{n}_sealed{OPTS}")),
    ]
}

/// The in-memory engine; under test-filedb a memory DSN opens a file
fn in_memory(engine: &str) -> bool {
    engine == "memory" && !cfg!(feature = "test-filedb")
}

fn seal_if(db: &Database, dsn: &str) {
    if dsn.contains("_sealed") {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
}

/// One outcome line; failures start with WRONG, PANIC or NOT REACHED
fn outcome(
    label: &str,
    got: std::thread::Result<Rows>,
    expected: &[Vec<Value>],
    fired: bool,
    paths: &[&str],
) -> String {
    let got = match got {
        Err(_) => return format!("PANIC [{label}] {paths:?}"),
        Ok(got) => got,
    };
    match got {
        Err(Error::SchemaChanged { .. }) if fired => {
            format!("ok SchemaChanged [{label}] {paths:?}")
        }
        Err(Error::SchemaChanged { .. }) => format!("ok refused before the hook [{label}]"),
        _ if !fired => format!("NOT REACHED [{label}] {paths:?}"),
        Ok(got) if got == expected => format!("ok right rows [{label}] {paths:?}"),
        got => {
            let got = format!("{got:?}");
            format!("WRONG [{label}] {paths:?}: {}", &got[..got.len().min(100)])
        }
    }
}

fn finish(report: Vec<String>) {
    for line in &report {
        eprintln!("{line}");
    }
    let failed: Vec<&String> = report.iter().filter(|l| !l.starts_with("ok")).collect();
    assert!(
        failed.is_empty(),
        "{} of {} cases failed",
        failed.len(),
        report.len()
    );
}

/// Runs `read` on its own thread and returns what it read, whether `hook`
/// fired and the read entries it ran
fn on_thread(
    install: impl FnOnce(Box<dyn FnOnce() + Send>) + Send + 'static,
    hook: impl FnOnce() + Send + 'static,
    read: impl FnOnce() -> Rows + Send + 'static,
) -> (std::thread::Result<Rows>, bool, Vec<&'static str>) {
    let fired = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&fired);
    let (sent, received) = mpsc::channel();
    std::thread::spawn(move || {
        install(Box::new(move || {
            hook();
            seen.store(true, Ordering::SeqCst);
            fp::take_read_paths();
        }));
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(read));
        let paths = fp::take_read_paths();
        // A hook that never fired holds a database this thread must drop now
        fp::reset_all();
        let _ = sent.send((got, paths));
    });
    let (got, paths) = received
        .recv_timeout(Duration::from_secs(30))
        .expect("the reader finished");
    (got, fired.load(Ordering::SeqCst), paths)
}

// --- SELECT families ---------------------------------------------------------

fn select_setup(db: &Database, dsn: &str) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        "INSERT INTO t SELECT value, 1000 + value, value * 10, value % 3 FROM generate_series(1, 40)",
        "CREATE INDEX ia ON t(a)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    seal_if(db, dsn);
}

const SELECTS: &[&str] = &[
    "SELECT a, b FROM t ORDER BY id",
    "SELECT a FROM t WHERE b = 1 ORDER BY id",
    "SELECT id FROM t WHERE a = 100",
    "SELECT SUM(a), MIN(a), MAX(a) FROM t",
    "SELECT COUNT(*), SUM(a) FROM t WHERE b = 2",
    "SELECT b, SUM(a) FROM t GROUP BY b ORDER BY b",
    "SELECT a FROM t ORDER BY id LIMIT 3",
    "SELECT a FROM t ORDER BY a DESC LIMIT 3",
    "SELECT a FROM t WHERE id IN (7, 8) ORDER BY id",
    "SELECT a FROM t WHERE a > 380 ORDER BY id",
];

fn expected_select(sql: &str) -> Vec<Vec<Value>> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dsn = format!("memory://readers_expected_{n}");
    let db = Database::open(&dsn).unwrap();
    select_setup(&db, &dsn);
    rows(&db, sql).unwrap()
}

/// The statement opened its table before the drop and reads after it
#[test]
fn a_select_whose_table_was_opened_before_a_drop() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for sql in SELECTS {
        let expected = expected_select(sql);
        for (engine, dsn) in engines(&dir, "stale") {
            let db = Database::open(&dsn).unwrap();
            select_setup(&db, &dsn);
            let other = db.clone();
            let (reader, sql_owned) = (db.clone(), sql.to_string());
            let (got, fired, paths) = on_thread(
                fp::after_select_table_opened,
                move || {
                    other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
                },
                move || rows(&reader, &sql_owned),
            );
            report.push(outcome(
                &format!("{engine}: {sql}"),
                got,
                &expected,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

/// The statement opened its table while the drop held it closed, and
/// reads after the drop opened it
#[test]
fn a_select_whose_table_was_opened_while_a_drop_held_it_closed() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for sql in SELECTS {
        let expected = expected_select(sql);
        for step in ["L+1, rows unmoved", "L+3, hot rows moved"] {
            for (engine, dsn) in engines(&dir, "closed") {
                let db = Database::open(&dsn).unwrap();
                select_setup(&db, &dsn);
                let reader = db.clone();
                let sql_owned = sql.to_string();
                let (opened, wait_opened) = mpsc::channel::<()>();
                let (go, wait_go) = mpsc::channel::<()>();
                let (sent, received) = mpsc::channel();
                let install: fn(Box<dyn FnOnce()>) = if step.starts_with("L+1") {
                    |hook| fp::after_drop_column_recorded(hook)
                } else {
                    |hook| fp::after_drop_hot_published(hook)
                };
                install(Box::new(move || {
                    let at_end = opened.clone();
                    std::thread::spawn(move || {
                        let fired = Arc::new(AtomicBool::new(false));
                        let seen = Arc::clone(&fired);
                        fp::after_select_table_opened(move || {
                            seen.store(true, Ordering::SeqCst);
                            let _ = opened.send(());
                            let _ = wait_go.recv_timeout(Duration::from_secs(10));
                            fp::take_read_paths();
                        });
                        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            rows(&reader, &sql_owned)
                        }));
                        let _ = at_end.send(());
                        let _ =
                            sent.send((got, fired.load(Ordering::SeqCst), fp::take_read_paths()));
                    });
                    let _ = wait_opened.recv_timeout(Duration::from_secs(10));
                }));
                db.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
                let _ = go.send(());
                let (got, fired, paths) = received
                    .recv_timeout(Duration::from_secs(30))
                    .expect("the reader finished");
                report.push(outcome(
                    &format!("{engine}, {step}: {sql}"),
                    got,
                    &expected,
                    fired,
                    &paths,
                ));
            }
        }
    }
    finish(report);
}

// --- Transaction-local rows ----------------------------------------------------

/// A transaction's own row, written before the drop, read with the
/// committed rows after it
#[test]
fn a_transaction_reads_its_own_rows_across_a_drop() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let sql = "SELECT a, b FROM t WHERE id > 38 ORDER BY id";
    let expected = ints(&[&[390, 0], &[400, 1], &[410, 2]]);
    let mut report = Vec::new();
    for stale in [true, false] {
        for (engine, dsn) in engines(&dir, "local") {
            let db = Database::open(&dsn).unwrap();
            select_setup(&db, &dsn);
            let other = db.clone();
            let whole = db.clone();
            let reader = db.clone();
            let (got, fired, paths) = on_thread(
                move |hook| {
                    if stale {
                        fp::after_select_table_opened(hook)
                    } else {
                        hook()
                    }
                },
                move || {
                    if stale {
                        other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
                    }
                },
                move || {
                    let mut tx = reader.begin()?;
                    tx.execute("INSERT INTO t VALUES (41, 1041, 410, 2)", ())?;
                    if !stale {
                        whole.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
                    }
                    fp::take_read_paths();
                    let got: Rows = tx
                        .query(sql, ())?
                        .map(|r| {
                            r.map(|r| {
                                (0..r.len())
                                    .map(|i| r.get_value(i).cloned().unwrap())
                                    .collect()
                            })
                        })
                        .collect();
                    let _ = tx.rollback();
                    got
                },
            );
            let window = if stale {
                "opened before"
            } else {
                "after a whole drop"
            };
            report.push(outcome(
                &format!("{engine}, {window}: {sql}"),
                got,
                &expected,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

// --- Foreign keys ----------------------------------------------------------------

/// A parent DELETE checks its children with the key position it took
/// before a drop on the child table
#[test]
fn a_parent_delete_checks_children_across_a_child_drop() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for (shape, child) in [
        (
            "key after the dropped column",
            "CREATE TABLE c (x INTEGER, pid INTEGER REFERENCES p(id), id INTEGER PRIMARY KEY)",
        ),
        (
            "key last",
            "CREATE TABLE c (x INTEGER, id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id))",
        ),
    ] {
        for (engine, dsn) in engines(&dir, "fk") {
            let db = Database::open(&dsn).unwrap();
            for sql in [
                "CREATE TABLE p (id INTEGER PRIMARY KEY)",
                "INSERT INTO p VALUES (1), (2)",
                child,
            ] {
                db.execute(sql, ()).unwrap();
            }
            let insert = if shape == "key last" {
                "INSERT INTO c VALUES (100, 50, 1)"
            } else {
                "INSERT INTO c VALUES (100, 1, 50)"
            };
            db.execute(insert, ()).unwrap();
            seal_if(&db, &dsn);
            let other = db.clone();
            let deleter = db.clone();
            let (got, fired, paths) = on_thread(
                fp::before_fk_child_probe,
                move || {
                    other.execute("ALTER TABLE c DROP COLUMN x", ()).unwrap();
                },
                move || {
                    let deleted = deleter.execute("DELETE FROM p WHERE id = 1", ());
                    let left = rows(&deleter, "SELECT id FROM p ORDER BY id")?;
                    match deleted {
                        Err(Error::SchemaChanged { .. }) => {
                            Err(Error::SchemaChanged { table: "c".into() })
                        }
                        Err(Error::ForeignKeyViolation { .. }) => Ok(left),
                        Err(e) => Err(e),
                        Ok(_) => Ok(left),
                    }
                },
            );
            // Refused: the parent row the child names stays
            let expected = ints(&[&[1], &[2]]);
            report.push(outcome(
                &format!("{engine}, {shape}: DELETE FROM p WHERE id = 1"),
                got,
                &expected,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

// --- Probes that keep a table for the statement ----------------------------------

fn probe_setup(db: &Database, dsn: &str) {
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO p SELECT value, value FROM generate_series(1, 10)",
        "CREATE TABLE c (x INTEGER, id INTEGER PRIMARY KEY, pid INTEGER, a INTEGER)",
        "INSERT INTO c SELECT 1000 + value, value, value % 10 + 1, value FROM generate_series(1, 30)",
        "CREATE INDEX cp ON c(pid)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    seal_if(db, dsn);
}

fn expected_probe(sql: &str) -> Vec<Vec<Value>> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dsn = format!(
        "memory://readers_probe_expected_{}",
        N.fetch_add(1, Ordering::Relaxed)
    );
    let db = Database::open(&dsn).unwrap();
    probe_setup(&db, &dsn);
    rows(&db, sql).unwrap()
}

/// A correlated EXISTS keeps one inner table for the statement and probes
/// it after a drop on that table
#[test]
fn a_correlated_exists_probes_across_a_drop() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for sql in [
        "SELECT p.id FROM p WHERE EXISTS (SELECT 1 FROM c WHERE c.pid = p.v) ORDER BY p.id LIMIT 10",
        "SELECT p.id, (SELECT COUNT(*) FROM c WHERE c.pid = p.v) FROM p ORDER BY p.id LIMIT 10",
    ] {
    let expected = expected_probe(sql);
    for (engine, dsn) in engines(&dir, "exists") {
        let db = Database::open(&dsn).unwrap();
        probe_setup(&db, &dsn);
        let other = db.clone();
        let reader = db.clone();
        let (got, fired, paths) = on_thread(
            fp::before_correlated_fetch,
            move || {
                other.execute("ALTER TABLE c DROP COLUMN x", ()).unwrap();
            },
            move || rows(&reader, sql),
        );
        report.push(outcome(&format!("{engine}: {sql}"), got, &expected, fired, &paths));
    }
    }
    finish(report);
}

/// A join that opened its inner table probes it after a drop on it
#[test]
fn a_join_probes_its_inner_table_across_a_drop() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    // The second shape is an index join on memory only: the file engine
    // plans no secondary-index join
    for (sql, memory_only) in [
        (
            "SELECT p.id, c.a FROM p JOIN c ON c.pid = p.id WHERE c.a > 20 ORDER BY c.a LIMIT 5",
            false,
        ),
        (
            "SELECT p.id, c.a FROM p JOIN c ON c.pid = p.id ORDER BY p.id, c.a LIMIT 4",
            true,
        ),
    ] {
        let expected = expected_probe(sql);
        let planned = engines(&dir, "join")
            .into_iter()
            .filter(|(engine, _)| !memory_only || in_memory(engine));
        for (engine, dsn) in planned {
            let db = Database::open(&dsn).unwrap();
            probe_setup(&db, &dsn);
            let other = db.clone();
            let reader = db.clone();
            let (got, fired, paths) = on_thread(
                fp::after_join_inner_opened,
                move || {
                    other.execute("ALTER TABLE c DROP COLUMN x", ()).unwrap();
                },
                move || rows(&reader, sql),
            );
            report.push(outcome(
                &format!("{engine}: {sql}"),
                got,
                &expected,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

// --- ADD COLUMN --------------------------------------------------------------------

/// A statement that reads while ADD COLUMN has published the schema and
/// has not laid out the rows sees the column's default
#[test]
fn a_select_between_an_added_column_and_its_rows() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for sql in [
        "SELECT SUM(c), COUNT(c) FROM t",
        "SELECT c FROM t WHERE id = 3",
        "SELECT COUNT(*) FROM t WHERE c = 5",
        "SELECT b, SUM(c) FROM t GROUP BY b ORDER BY b",
    ] {
        for (engine, dsn) in engines(&dir, "add") {
            let db = Database::open(&dsn).unwrap();
            select_setup(&db, &dsn);
            let expected = {
                let dsn = format!("memory://readers_add_expected_{}_{engine}", sql.len());
                let e = Database::open(&dsn).unwrap();
                select_setup(&e, &dsn);
                e.execute("ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 5", ())
                    .unwrap();
                rows(&e, sql).unwrap()
            };
            let reader = db.clone();
            let (sent, received) = mpsc::channel();
            fp::after_add_column_published(move || {
                let (got, fired, paths) =
                    on_thread(|hook| hook(), || {}, move || rows(&reader, sql));
                let _ = sent.send((got, fired, paths));
            });
            db.execute("ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 5", ())
                .unwrap();
            let (got, fired, paths) = received
                .recv_timeout(Duration::from_secs(30))
                .expect("the reader ran");
            report.push(outcome(
                &format!("{engine}: {sql}"),
                got,
                &expected,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

/// A row written under the schema of an ADD COLUMN whose record the log
/// then refuses does not lend its extra cell to a later added column
#[test]
fn a_refused_add_column_leaves_no_cell_for_a_later_column() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}{OPTS}&sync_mode=full", dir.path().display());
    let mut report = Vec::new();
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
        let writer = db.clone();
        let (sent, received) = mpsc::channel();
        fp::after_add_column_published(move || {
            let (done, finished) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = done.send(writer.execute("INSERT INTO t VALUES (2, 20, 99)", ()));
            });
            let inserted = finished.recv_timeout(Duration::from_secs(5));
            fp::WAL_WRITE_FAIL.store(true, Ordering::SeqCst);
            let _ = sent.send(format!("{inserted:?}"));
        });
        let added = db.execute("ALTER TABLE t ADD COLUMN c INTEGER", ());
        fp::WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
        let inserted = received.recv_timeout(Duration::from_secs(10)).unwrap();
        report.push(format!(
            "ok note: insert in the window {inserted}; ADD {added:?}"
        ));
        let _ = db.close();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("ALTER TABLE t ADD COLUMN y INTEGER", ())
        .unwrap();
    let got = rows(&db, "SELECT id, y FROM t ORDER BY id");
    let expected = vec![
        vec![Value::Integer(1), Value::Null(DataType::Integer)],
        vec![Value::Integer(2), Value::Null(DataType::Integer)],
    ];
    let line = match &got {
        Ok(r) if r == &expected => "ok y is NULL after reopen".to_string(),
        Ok(r) if r.len() == 1 => "ok the window's insert did not survive".to_string(),
        other => format!("WRONG y after reopen: {other:?}"),
    };
    report.push(line);
    finish(report);
}

// --- Review counterexamples --------------------------------------------------

/// A backup snapshot wrote the schema before a drop and takes the rows
/// after its hot publish: a restore gives one table
#[test]
fn a_backup_snapshot_across_a_drop_restores_one_table() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}{OPTS}", dir.path().display())).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        "INSERT INTO t VALUES (1, 1001, 10, 1), (2, 1002, 20, 2), (3, 1003, 30, 3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let other = db.clone();
    let (trying, tried) = mpsc::channel();
    let (published, publish) = mpsc::channel();
    let publish = Arc::new(std::sync::Mutex::new(publish));
    let waited = Arc::clone(&publish);
    let attempted = Arc::new(AtomicBool::new(false));
    let published_inside = Arc::new(AtomicBool::new(false));
    let (seen_attempt, seen_publish) = (Arc::clone(&attempted), Arc::clone(&published_inside));
    let dropper = Arc::new(std::sync::Mutex::new(None));
    let handle = Arc::clone(&dropper);
    fp::after_snapshot_schema_written(move || {
        *handle.lock().unwrap() = Some(std::thread::spawn(move || {
            fp::before_alter_ddl_guard(move || {
                let _ = trying.send(());
            });
            fp::after_drop_hot_published(move || {
                let _ = published.send(());
            });
            other.execute("ALTER TABLE t DROP COLUMN x", ())
        }));
        seen_attempt.store(
            tried.recv_timeout(Duration::from_secs(5)).is_ok(),
            Ordering::SeqCst,
        );
        let inside = waited.lock().unwrap().recv_timeout(Duration::from_secs(1));
        seen_publish.store(inside.is_ok(), Ordering::SeqCst);
    });
    let snapshot = db.execute("PRAGMA SNAPSHOT", ());
    let dropped = dropper.lock().unwrap().take().map(|d| d.join());
    let published_after = published_inside.load(Ordering::SeqCst)
        || publish
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .is_ok();
    let mut report = vec![
        format!(
            "{} [the DROP reached its DDL guard during the snapshot]",
            if attempted.load(Ordering::SeqCst) {
                "ok"
            } else {
                "NOT REACHED"
            }
        ),
        format!(
            "{} [the DROP waited for the snapshot]",
            if published_inside.load(Ordering::SeqCst) {
                "WRONG published inside"
            } else {
                "ok"
            }
        ),
        match dropped {
            Some(Ok(Ok(_))) if published_after => "ok [the DROP completed afterwards]".into(),
            other => format!("WRONG [the DROP completed afterwards]: {other:?}"),
        },
    ];
    let got = match snapshot {
        Err(e) => Err(e),
        Ok(_) => db
            .execute("PRAGMA RESTORE", ())
            .and_then(|_| rows(&db, "SELECT id, a, b FROM t ORDER BY id")),
    };
    let expected = ints(&[&[1, 10, 1], &[2, 20, 2], &[3, 30, 3]]);
    report.push(outcome(
        "PRAGMA SNAPSHOT, then RESTORE",
        Ok(got),
        &expected,
        attempted.load(Ordering::SeqCst),
        &[],
    ));
    finish(report);
}

/// The key a foreign key checks is its column, through a rename and a new
/// column that takes the old name, and through a drop on the child
#[test]
fn a_foreign_key_checks_its_column_through_a_rename_a_reused_name_and_a_drop() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for window in [false, true] {
        for (engine, dsn) in engines(&dir, "fk_rename") {
            let db = Database::open(&dsn).unwrap();
            for sql in [
                "CREATE TABLE p (id INTEGER PRIMARY KEY)",
                "INSERT INTO p VALUES (1), (2)",
                "CREATE TABLE c (x INTEGER, pid INTEGER REFERENCES p(id), id INTEGER PRIMARY KEY)",
                "INSERT INTO c VALUES (0, 1, 10)",
                "ALTER TABLE c RENAME COLUMN pid TO actual_pid",
                "ALTER TABLE c ADD COLUMN pid INTEGER",
            ] {
                db.execute(sql, ()).unwrap();
            }
            seal_if(&db, &dsn);
            let other = db.clone();
            let deleter = db.clone();
            let (got, fired, paths) = on_thread(
                move |hook| {
                    if window {
                        fp::before_fk_child_probe(hook)
                    } else {
                        hook()
                    }
                },
                move || {
                    if window {
                        other.execute("ALTER TABLE c DROP COLUMN x", ()).unwrap();
                    }
                },
                move || {
                    let deleted = deleter.execute("DELETE FROM p WHERE id = 1", ());
                    let left = rows(&deleter, "SELECT id FROM p ORDER BY id")?;
                    match deleted {
                        Err(Error::SchemaChanged { .. }) => {
                            Err(Error::SchemaChanged { table: "c".into() })
                        }
                        Err(Error::ForeignKeyViolation { .. }) | Ok(_) => Ok(left),
                        Err(e) => Err(e),
                    }
                },
            );
            let label = if window {
                "drop before the probe"
            } else {
                "no drop"
            };
            report.push(outcome(
                &format!("{engine}, {label}: DELETE FROM p WHERE id = 1"),
                got,
                &ints(&[&[1], &[2]]),
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

/// Two foreign keys of one child reference the same parent: a CASCADE on a
/// and a RESTRICT on b; the RESTRICT refuses the delete
#[test]
fn every_foreign_key_to_one_parent_is_checked() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for window in [false, true] {
        for (engine, dsn) in engines(&dir, "fk_two") {
            let db = Database::open(&dsn).unwrap();
            for sql in [
                "CREATE TABLE p (id INTEGER PRIMARY KEY)",
                "INSERT INTO p VALUES (1), (2)",
                "CREATE TABLE c (x INTEGER, id INTEGER PRIMARY KEY, \
                 a INTEGER REFERENCES p(id) ON DELETE CASCADE, b INTEGER REFERENCES p(id))",
                "INSERT INTO c VALUES (0, 10, 2, 1)",
            ] {
                db.execute(sql, ()).unwrap();
            }
            seal_if(&db, &dsn);
            let other = db.clone();
            let deleter = db.clone();
            let (got, fired, paths) = on_thread(
                move |hook| {
                    if window {
                        fp::before_fk_child_probe(hook)
                    } else {
                        hook()
                    }
                },
                move || {
                    if window {
                        other.execute("ALTER TABLE c DROP COLUMN x", ()).unwrap();
                    }
                },
                move || {
                    let deleted = deleter.execute("DELETE FROM p WHERE id = 1", ());
                    let mut left = rows(&deleter, "SELECT id FROM p ORDER BY id")?;
                    left.extend(rows(&deleter, "SELECT id, a, b FROM c")?);
                    match deleted {
                        Err(Error::SchemaChanged { .. }) => {
                            Err(Error::SchemaChanged { table: "c".into() })
                        }
                        Err(Error::ForeignKeyViolation { .. }) | Ok(_) => Ok(left),
                        Err(e) => Err(e),
                    }
                },
            );
            let label = if window {
                "drop before the probe"
            } else {
                "no drop"
            };
            report.push(outcome(
                &format!("{engine}, {label}: DELETE FROM p WHERE id = 1"),
                got,
                &ints(&[&[1], &[2], &[10, 2, 1]]),
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

const JOIN_SECONDARY: &[&str] = &[
    "CREATE TABLE p (id INTEGER PRIMARY KEY, v INTEGER)",
    "INSERT INTO p SELECT value, value FROM generate_series(1, 10)",
    "CREATE TABLE c (x INTEGER, id INTEGER PRIMARY KEY, k INTEGER, a INTEGER)",
    "INSERT INTO c SELECT 0, value, value % 10 + 1, value FROM generate_series(1, 30)",
    "CREATE INDEX ck ON c(k)",
];

const JOIN_PK: &[&str] = &[
    "CREATE TABLE p (id INTEGER PRIMARY KEY, v INTEGER)",
    "INSERT INTO p SELECT value, value FROM generate_series(1, 10)",
    "CREATE TABLE c (k INTEGER PRIMARY KEY, a INTEGER)",
    "INSERT INTO c SELECT value, value * 10 FROM generate_series(1, 10)",
];

fn join_setup(db: &Database, dsn: &str, setup: &[&str]) {
    for sql in setup {
        db.execute(sql, ()).unwrap();
    }
    seal_if(db, dsn);
}

/// The join plan took c's index or key from its planning handle; before
/// the join opens c, the column it joins on is replaced
#[test]
fn a_join_planned_on_an_index_runs_on_the_table_it_opens() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let replaced: &[&str] = &[
        "ALTER TABLE c DROP COLUMN k",
        "ALTER TABLE c ADD COLUMN k INTEGER",
        "UPDATE c SET k = id % 5 + 1",
    ];
    let recreated: &[&str] = &[
        "DROP INDEX ck ON c",
        "UPDATE c SET k = id % 5 + 1",
        "CREATE INDEX ck ON c(k)",
    ];
    let renamed: &[&str] = &[
        "ALTER TABLE c RENAME COLUMN k TO id",
        "ALTER TABLE c ADD COLUMN k INTEGER DEFAULT 9",
    ];
    let streaming = "SELECT p.id, c.id FROM p JOIN c ON c.k = p.v WHERE c.a > 0 LIMIT 100";
    let batch = "SELECT p.id, c.id FROM p JOIN c ON c.k = p.v WHERE c.a > 0";
    let cases: [(&str, &[&str], &[&str], &str); 6] = [
        (
            "column replaced, LIMIT",
            JOIN_SECONDARY,
            replaced,
            streaming,
        ),
        ("column replaced, no LIMIT", JOIN_SECONDARY, replaced, batch),
        (
            "index recreated, LIMIT",
            JOIN_SECONDARY,
            recreated,
            streaming,
        ),
        (
            "index recreated, no LIMIT",
            JOIN_SECONDARY,
            recreated,
            batch,
        ),
        (
            "key renamed, LIMIT",
            JOIN_PK,
            renamed,
            "SELECT p.id, c.a FROM p JOIN c ON c.k = p.v LIMIT 100",
        ),
        (
            "key renamed, no LIMIT",
            JOIN_PK,
            renamed,
            "SELECT p.id, c.a FROM p JOIN c ON c.k = p.v",
        ),
    ];
    let sorted = |rows: Rows| {
        rows.map(|mut r| {
            r.sort_by_key(|row| format!("{row:?}"));
            r
        })
    };
    let mut report = Vec::new();
    for (n, (label, setup, change, sql)) in cases.into_iter().enumerate() {
        let expected = {
            let e = Database::open(&format!("memory://readers_join_index_expected_{n}")).unwrap();
            join_setup(&e, "memory", setup);
            for sql in change {
                e.execute(sql, ()).unwrap();
            }
            sorted(rows(&e, sql)).unwrap()
        };
        // The file engine plans no secondary-index join
        let planned = engines(&dir, "join_index")
            .into_iter()
            .filter(|(engine, _)| setup != JOIN_SECONDARY || in_memory(engine));
        for (engine, dsn) in planned {
            let db = Database::open(&dsn).unwrap();
            join_setup(&db, &dsn, setup);
            let other = db.clone();
            let reader = db.clone();
            let (got, fired, paths) = on_thread(
                fp::after_join_index_chosen,
                move || {
                    for sql in change {
                        other.execute(sql, ()).unwrap();
                    }
                },
                move || sorted(rows(&reader, sql)),
            );
            report.push(outcome(
                &format!("{engine}, {label}: {sql}"),
                got,
                &expected,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

/// `outcome` against the old view, or the new view when the rows are that
fn either(
    label: &str,
    got: std::thread::Result<Rows>,
    old: &[Vec<Value>],
    new: &[Vec<Value>],
    fired: bool,
    paths: &[&str],
) -> String {
    match got {
        Ok(Ok(r)) if r == new && r != old => {
            outcome(&format!("{label}, new view"), Ok(Ok(r)), new, fired, paths)
        }
        got => outcome(label, got, old, fired, paths),
    }
}

fn chunk_setup(db: &Database, dsn: &str) {
    for sql in [
        "CREATE TABLE p (x INTEGER, id INTEGER PRIMARY KEY, v INTEGER, w INTEGER)",
        "INSERT INTO p SELECT 0, value, value, value * 2 FROM generate_series(1, 1000)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER, a INTEGER)",
        "INSERT INTO c SELECT value, value * 8, value * 80 FROM generate_series(1, 125)",
        "CREATE INDEX cp ON c(pid)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    seal_if(db, dsn);
}

/// A streaming join that joined one outer chunk fetches the next after the
/// outer table changed; every chunk has matches and the limit takes them all
#[test]
fn a_streaming_join_reads_each_outer_chunk_by_its_own_layout() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dropped: &[&str] = &["ALTER TABLE p DROP COLUMN x"];
    let readded: &[&str] = &[
        "ALTER TABLE p DROP COLUMN w",
        "ALTER TABLE p ADD COLUMN w INTEGER DEFAULT 7",
    ];
    // Only the matched rows keep the new column's default, so the row a
    // chunk ended on reads the same before and after
    let readded_quiet: &[&str] = &[
        "ALTER TABLE p DROP COLUMN w",
        "ALTER TABLE p ADD COLUMN w INTEGER DEFAULT 7",
        "UPDATE p SET w = v * 2 WHERE v % 8 <> 0",
    ];
    let moved: &[&str] = &[
        "ALTER TABLE p DROP COLUMN x",
        "ALTER TABLE p ADD COLUMN x INTEGER DEFAULT 0",
    ];
    let viewed: &[&str] = &[
        "ALTER TABLE p RENAME TO p_old",
        "CREATE VIEW p AS SELECT x, id, v, 7 AS w FROM p_old",
    ];
    let streaming = "SELECT p.id, p.w, c.a FROM p JOIN c ON c.pid = p.v LIMIT 125";
    let grouped =
        "SELECT p.id, p.w, COUNT(*) FROM p JOIN c ON c.pid = p.v GROUP BY p.id, p.w LIMIT 125";
    let cases: [(&str, &[&str], &str); 9] = [
        ("column before the key dropped", dropped, streaming),
        ("last column dropped and added back", readded, streaming),
        (
            "last column added back, chunk end unchanged",
            readded_quiet,
            streaming,
        ),
        ("first column moved to the end", moved, streaming),
        ("table replaced by a same-named view", viewed, streaming),
        ("grouped, column before the key dropped", dropped, grouped),
        (
            "grouped, last column dropped and added back",
            readded,
            grouped,
        ),
        (
            "grouped, last column added back, chunk end unchanged",
            readded_quiet,
            grouped,
        ),
        ("grouped, first column moved to the end", moved, grouped),
    ];
    let sorted = |rows: Rows| {
        rows.map(|mut r| {
            r.sort_by_key(|row| format!("{row:?}"));
            r
        })
    };
    let mut report = Vec::new();
    for (n, (label, change, sql)) in cases.into_iter().enumerate() {
        let e = Database::open(&format!("memory://readers_join_chunk_expected_{n}")).unwrap();
        chunk_setup(&e, "memory");
        let old = sorted(rows(&e, sql)).unwrap();
        for sql in change {
            e.execute(sql, ()).unwrap();
        }
        let new = sorted(rows(&e, sql)).unwrap();
        for (engine, dsn) in engines(&dir, "join_chunk") {
            let db = Database::open(&dsn).unwrap();
            chunk_setup(&db, &dsn);
            let other = db.clone();
            let reader = db.clone();
            let (got, fired, paths) = on_thread(
                fp::before_join_next_chunk,
                move || {
                    for sql in change {
                        other.execute(sql, ()).unwrap();
                    }
                },
                move || sorted(rows(&reader, sql)),
            );
            report.push(either(
                &format!("{engine}, {label}: {sql}"),
                got,
                &old,
                &new,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

/// An expression read by old positions after a drop must not reach a value
/// that makes it panic: the row a reads is b's i64::MAX
#[test]
fn an_expression_read_across_a_drop_does_not_panic() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for (statement, install) in [
        (
            "UPDATE t SET a = a + 1 WHERE CAST(a AS TIMESTAMP) IS NOT NULL",
            fp::after_dml_table_opened as fn(_),
        ),
        (
            "SELECT id FROM t WHERE CAST(a AS TIMESTAMP) IS NOT NULL",
            fp::after_select_table_opened as fn(_),
        ),
    ] {
        for (engine, dsn) in engines(&dir, "cast") {
            let db = Database::open(&dsn).unwrap();
            for sql in [
                "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
                "INSERT INTO t VALUES (1, 0, 5, 9223372036854775807)",
            ] {
                db.execute(sql, ()).unwrap();
            }
            seal_if(&db, &dsn);
            let other = db.clone();
            let reader = db.clone();
            let expected = if statement.starts_with("UPDATE") {
                ints(&[&[6]])
            } else {
                ints(&[&[1]])
            };
            let (got, fired, paths) = on_thread(
                install,
                move || {
                    other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
                },
                move || {
                    if statement.starts_with("UPDATE") {
                        reader.execute(statement, ())?;
                        rows(&reader, "SELECT a FROM t")
                    } else {
                        rows(&reader, statement)
                    }
                },
            );
            report.push(outcome(
                &format!("{engine}: {statement}"),
                got,
                &expected,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}

/// A handle of a transaction, opened before a drop, reads or updates rows
/// the transaction wrote after the drop through a newer handle
#[test]
fn an_older_handle_meets_its_transactions_rows_written_after_a_drop() {
    let _guard = fp::FailpointGuard::new();
    use stoolap::core::{Operator, Row};
    use stoolap::storage::expression::{AndExpr, ComparisonExpr, ConstBoolExpr, Expression};
    use stoolap::storage::traits::{Engine, Table};
    type Op = fn(&mut dyn Table, &AtomicBool) -> Result<(), Error>;
    let ops: [(&str, Op); 5] = [
        ("fetch_rows_by_ids", |table, _| {
            table
                .fetch_rows_by_ids(&[5], &ConstBoolExpr::new(true))
                .map(drop)
        }),
        ("collect_all_rows", |table, _| {
            table.collect_all_rows(None).map(drop)
        }),
        ("update_by_row_ids", |table, called| {
            table
                .update_by_row_ids(&[5], &mut |row| {
                    called.store(true, Ordering::SeqCst);
                    Ok((row, true))
                })
                .map(drop)
        }),
        ("update by key range", |table, called| {
            let mut low = ComparisonExpr::new("id", Operator::Gte, Value::Integer(5));
            low.prepare_for_schema(table.schema());
            let mut high = ComparisonExpr::new("id", Operator::Lt, Value::Integer(6));
            high.prepare_for_schema(table.schema());
            let range = AndExpr::new(vec![Box::new(low), Box::new(high)]);
            table
                .update(Some(&range), &mut |row| {
                    called.store(true, Ordering::SeqCst);
                    Ok((row, true))
                })
                .map(drop)
        }),
        ("update by key", |table, called| {
            let mut key = ComparisonExpr::new("id", Operator::Eq, Value::Integer(5));
            key.prepare_for_schema(table.schema());
            table
                .update(Some(&key), &mut |row| {
                    called.store(true, Ordering::SeqCst);
                    Ok((row, true))
                })
                .map(drop)
        }),
    ];
    let dir = tempfile::tempdir().unwrap();
    let mut report = Vec::new();
    for (name, op) in ops {
        for (engine, dsn) in engines(&dir, "local_after_drop") {
            let db = Database::open(&dsn).unwrap();
            db.execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
                (),
            )
            .unwrap();
            let tx = db.engine().begin_transaction().unwrap();
            let mut older = tx.get_table("t").unwrap();
            db.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
            let mut newer = tx.get_table("t").unwrap();
            newer
                .insert(Row::from_values(vec![
                    Value::Integer(5),
                    Value::Integer(10),
                    Value::Integer(20),
                ]))
                .unwrap();
            let called = AtomicBool::new(false);
            let got = op(&mut *older, &called);
            let line = match got {
                Err(Error::SchemaChanged { .. }) if !called.load(Ordering::SeqCst) => {
                    format!("ok SchemaChanged [{engine}: {name}]")
                }
                Err(Error::SchemaChanged { .. }) => {
                    format!("WRONG [{engine}: {name}]: the setter ran before the refusal")
                }
                got => format!("WRONG [{engine}: {name}]: {got:?}"),
            };
            report.push(line);
        }
    }
    finish(report);
}

/// An index MIN or MAX over hot and sealed rows read its hot bound, then
/// two renames hand the column's name to another column
#[test]
fn an_index_bound_reads_one_column_across_renames() {
    use stoolap::IsolationLevel;
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let renames: &[&str] = &[
        "ALTER TABLE t RENAME COLUMN a TO old_a",
        "ALTER TABLE t RENAME COLUMN b TO a",
    ];
    // (statement, sealed row, hot row): the mixed answer is neither view
    type Pair = (i64, i64);
    let cases: [(&str, Pair, Pair); 2] = [
        ("SELECT MIN(a) FROM t LIMIT 1", (20, 1), (10, 0)),
        ("SELECT MAX(a) FROM t LIMIT 1", (20, 1), (10, 30)),
    ];
    let setup = |db: &Database, dsn: &str, sealed: (i64, i64), hot: (i64, i64)| {
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)".to_string(),
            "CREATE INDEX ia ON t(a)".into(),
            "CREATE INDEX ib ON t(b)".into(),
            format!("INSERT INTO t VALUES (1, {}, {})", sealed.0, sealed.1),
        ] {
            db.execute(&sql, ()).unwrap();
        }
        if dsn.starts_with("file") {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db.execute(
            &format!("INSERT INTO t VALUES (2, {}, {})", hot.0, hot.1),
            (),
        )
        .unwrap();
    };
    let mut report = Vec::new();
    for (n, (sql, sealed, hot)) in cases.into_iter().enumerate() {
        let e = Database::open(&format!("memory://readers_index_bound_expected_{n}")).unwrap();
        setup(&e, "memory", sealed, hot);
        let old = rows(&e, sql).unwrap();
        for sql in renames {
            e.execute(sql, ()).unwrap();
        }
        let new = rows(&e, sql).unwrap();
        // Only sealed rows take the volume bound
        for (engine, dsn) in engines(&dir, "index_bound") {
            if engine != "file sealed" {
                continue;
            }
            let db = Database::open(&dsn).unwrap();
            setup(&db, &dsn, sealed, hot);
            let other = db.clone();
            let reader = db.clone();
            let (got, fired, paths) = on_thread(
                fp::after_index_bound_hot,
                move || {
                    for sql in renames {
                        other.execute(sql, ()).unwrap();
                    }
                },
                move || {
                    let mut tx = reader.begin_with_isolation(IsolationLevel::SnapshotIsolation)?;
                    tx.query(sql, ())?
                        .map(|r| {
                            r.map(|r| {
                                vec![r
                                    .get_value(0)
                                    .cloned()
                                    .unwrap_or(Value::Null(DataType::Null))]
                            })
                        })
                        .collect()
                },
            );
            report.push(either(
                &format!("{engine}: {sql}"),
                got,
                &old,
                &new,
                fired,
                &paths,
            ));
        }
    }
    finish(report);
}
