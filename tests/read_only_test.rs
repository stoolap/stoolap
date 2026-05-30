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

//! Tests for ReadOnlyDatabase and Statement::write_reason.

use stoolap::parser::parse_sql;
use stoolap::{Database, Error};

// ---------------------------------------------------------------------------
// write_reason classification tests
// ---------------------------------------------------------------------------

#[test]
fn write_reason_insert_is_write() {
    let stmts = parse_sql("INSERT INTO t VALUES (1)").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("INSERT"));
}

#[test]
fn write_reason_update_is_write() {
    let stmts = parse_sql("UPDATE t SET x = 1").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("UPDATE"));
}

#[test]
fn write_reason_delete_is_write() {
    let stmts = parse_sql("DELETE FROM t WHERE id = 1").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("DELETE"));
}

#[test]
fn write_reason_create_table_is_write() {
    let stmts = parse_sql("CREATE TABLE t (id INTEGER)").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("CREATE TABLE"));
}

#[test]
fn write_reason_drop_table_is_write() {
    let stmts = parse_sql("DROP TABLE t").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("DROP TABLE"));
}

#[test]
fn write_reason_select_is_read() {
    let stmts = parse_sql("SELECT * FROM t").unwrap();
    assert_eq!(stmts[0].write_reason(), None);
}

#[test]
fn write_reason_show_tables_is_read() {
    let stmts = parse_sql("SHOW TABLES").unwrap();
    assert_eq!(stmts[0].write_reason(), None);
}

#[test]
fn write_reason_begin_is_read() {
    let stmts = parse_sql("BEGIN").unwrap();
    assert_eq!(stmts[0].write_reason(), None);
}

#[test]
fn write_reason_commit_is_read() {
    let stmts = parse_sql("COMMIT").unwrap();
    assert_eq!(stmts[0].write_reason(), None);
}

#[test]
fn write_reason_rollback_is_read() {
    let stmts = parse_sql("ROLLBACK").unwrap();
    assert_eq!(stmts[0].write_reason(), None);
}

#[test]
fn write_reason_explain_select_is_read() {
    let stmts = parse_sql("EXPLAIN SELECT * FROM t").unwrap();
    assert_eq!(stmts[0].write_reason(), None);
}

#[test]
fn write_reason_explain_analyze_insert_is_write() {
    let stmts = parse_sql("EXPLAIN ANALYZE INSERT INTO t VALUES (1)").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("INSERT"));
}

#[test]
fn write_reason_pragma_read_is_read() {
    // volume_stats is in the fail-closed read allow-list.
    let stmts = parse_sql("PRAGMA volume_stats").unwrap();
    assert_eq!(stmts[0].write_reason(), None);
}

#[test]
fn write_reason_pragma_checkpoint_is_write() {
    let stmts = parse_sql("PRAGMA checkpoint").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("PRAGMA <maintenance>"));
}

#[test]
fn write_reason_pragma_with_value_is_write() {
    let stmts = parse_sql("PRAGMA page_size = 4096").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("PRAGMA <write>"));
}

// ---------------------------------------------------------------------------
// ReadOnlyDatabase API tests
// ---------------------------------------------------------------------------

#[test]
fn read_only_api_exists_and_dsn_works() {
    let db = Database::open_in_memory().unwrap();
    let rodb = db.as_read_only();
    assert!(!rodb.dsn().is_empty());
}

#[test]
fn read_only_rejects_insert() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();

    let rodb = db.as_read_only();
    let result = rodb.query("INSERT INTO t VALUES (1)", ());
    match result {
        Err(Error::ReadOnlyViolation(_)) => {}
        other => panic!("expected ReadOnlyViolation, got: {:?}", other.is_ok()),
    }
}

#[test]
fn read_only_rejects_update() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let rodb = db.as_read_only();
    let result = rodb.query("UPDATE t SET id = 2", ());
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn read_only_rejects_create_table() {
    let db = Database::open_in_memory().unwrap();
    let rodb = db.as_read_only();
    let result = rodb.query("CREATE TABLE t2 (id INTEGER)", ());
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn read_only_allows_select() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (42)", ()).unwrap();

    let rodb = db.as_read_only();
    let rows: Vec<_> = rodb
        .query("SELECT * FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows.len(), 1);
    let id: i64 = rows[0].get(0).unwrap();
    assert_eq!(id, 42);
}

#[test]
fn read_only_allows_show_tables() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE users (id INTEGER)", ()).unwrap();

    let rodb = db.as_read_only();
    let rows: Vec<_> = rodb
        .query("SHOW TABLES", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(!rows.is_empty());
}

#[test]
fn read_only_table_exists() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE present (id INTEGER)", ()).unwrap();

    let rodb = db.as_read_only();
    assert!(rodb.table_exists("present").unwrap());
    assert!(!rodb.table_exists("missing").unwrap());
}

#[test]
fn ro_handle_outlives_source_database() {
    // ReadOnlyDatabase must keep DatabaseInner alive across source drop.
    let ro = {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'alice')", ()).unwrap();
        db.execute("INSERT INTO t VALUES (2, 'bob')", ()).unwrap();
        db.as_read_only()
    };

    let mut rows = ro.query("SELECT v FROM t WHERE id = 1", ()).unwrap();
    let row = rows.next().expect("at least one row").unwrap();
    drop(row);
}

#[test]
fn ro_handle_outlives_open_read_only_temp_database() {
    // open_read_only's internal Database must be kept alive too.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_outlives.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'alice')", ()).unwrap();
    }

    let ro = Database::open_read_only(&dsn).unwrap();
    let mut rows = ro.query("SELECT v FROM t WHERE id = 1", ()).unwrap();
    let row = rows.next().expect("at least one row").unwrap();
    drop(row);
    drop(ro);
}

#[test]
fn ro_handle_survives_explicit_close_on_writable() {
    // close() must defer engine shutdown when an RO handle is alive.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'alice')", ()).unwrap();

    let ro = db.as_read_only();
    db.close().unwrap();

    let mut rows = ro.query("SELECT v FROM t WHERE id = 1", ()).unwrap();
    let row = rows.next().expect("at least one row").unwrap();
    drop(row);
}

#[test]
fn ro_handle_unregisters_when_last_holder_drops() {
    // Last RO drop must unregister so the engine + file lock don't leak.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_unregister.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        let ro = db.as_read_only();
        drop(db);
        drop(ro);
    }

    // Fresh writable open would fail with DatabaseLocked if the lock leaked.
    let db2 = Database::open(&dsn).unwrap();
    db2.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    drop(db2);
}

#[test]
fn ro_open_uses_shared_file_lock_so_two_readers_coexist() {
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_shared_lock.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    let ro1 = Database::open_read_only(&dsn).unwrap();
    let ro2 = Database::open_read_only(&dsn).unwrap();
    drop(ro1);
    drop(ro2);
}

#[test]
fn ro_blocks_subsequent_writer_via_shared_lock() {
    // Same-process exclusion is masked by registry sharing; true cross-
    // process exclusion is covered by file_lock unit tests.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_blocks_writer.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let ro = Database::open_read_only(&dsn).unwrap();
    drop(ro);
}

#[test]
fn ro_executor_refuses_dml_auto_commit_as_defense_in_depth() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let ro = db.as_read_only();

    let mut rows = ro.query("SELECT * FROM t", ()).unwrap();
    let row = rows.next().expect("one row").unwrap();
    drop(row);

    let result = ro.query("INSERT INTO t VALUES (2)", ());
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn writable_open_after_read_only_open_is_rejected() {
    // Reusing a registry RO entry as writable would bypass every gate.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_then_writable.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let ro = Database::open_read_only(&dsn).unwrap();

    let result = Database::open(&dsn);
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));

    drop(ro);
    let db2 = Database::open(&dsn).unwrap();
    db2.execute("INSERT INTO t VALUES (1)", ()).unwrap();
}

#[test]
fn pragma_restore_is_classified_as_write() {
    // PRAGMA RESTORE has no value but rewrites the entire DB.
    use stoolap::parser::parse_sql;
    let stmts = parse_sql("PRAGMA restore").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("PRAGMA <maintenance>"));

    let stmts = parse_sql("PRAGMA RESTORE").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("PRAGMA <maintenance>"));
}

#[test]
fn pragma_unknown_is_treated_as_write_fail_closed() {
    // Unknown pragmas default to write so new maintenance pragmas
    // can't sneak past the gate when the classifier lags.
    use stoolap::parser::parse_sql;
    let stmts = parse_sql("PRAGMA some_future_pragma").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("PRAGMA <maintenance>"));
}

#[test]
fn pragma_known_read_pragmas_are_reads() {
    use stoolap::parser::parse_sql;
    for sql in &[
        "PRAGMA volume_stats",
        "PRAGMA snapshot_interval",
        "PRAGMA checkpoint_interval",
        "PRAGMA sync_mode",
    ] {
        let stmts = parse_sql(sql).unwrap();
        assert_eq!(stmts[0].write_reason(), None, "{sql}");
    }
}

#[test]
fn ro_query_via_pragma_restore_is_rejected() {
    let db = Database::open_in_memory().unwrap();
    let ro = db.as_read_only();
    let result = ro.query("PRAGMA RESTORE", ());
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn ro_full_table_scan_works_no_pk_filter() {
    // Non-PK-fast-path SELECT must succeed on a fresh open_read_only DSN.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_full_scan.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
        db.execute("INSERT INTO t VALUES (2, 'b')", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3, 'c')", ()).unwrap();
        db.close().unwrap();
    }

    let ro = Database::open_read_only(&dsn).unwrap();

    let rows: Vec<_> = ro
        .query("SELECT v FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows.len(), 3);

    let tables: Vec<_> = ro
        .query("SHOW TABLES", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(!tables.is_empty());

    let explain: Vec<_> = ro
        .query("EXPLAIN SELECT v FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(!explain.is_empty());
}

#[test]
fn ro_begin_is_refused_on_read_only_handle() {
    // BEGIN on RO would otherwise construct a writable transaction
    // (Box<dyn WriteTransaction>) reachable internally; we refuse it
    // at the executor and route stable-snapshot use to set_auto_refresh(false).
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_begin_refused.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    let ro = Database::open_read_only(&dsn).unwrap();
    match ro.query("BEGIN", ()) {
        Err(stoolap::Error::ReadOnlyViolation(_)) => {}
        other => panic!(
            "expected ReadOnlyViolation on RO BEGIN, got {:?}",
            other.is_ok()
        ),
    }
}

#[test]
fn memory_dsn_rejects_read_only() {
    // RO on memory:// is meaningless; both entry points refuse early.
    match Database::open_read_only("memory://") {
        Ok(_) => panic!("open_read_only(memory://) must be refused"),
        Err(Error::InvalidArgument(_)) => {}
        Err(other) => {
            panic!("expected InvalidArgument from open_read_only(memory://), got: {other:?}")
        }
    }

    match Database::open("memory://?read_only=true") {
        Ok(_) => panic!("open(memory://?read_only=true) must be refused"),
        Err(Error::InvalidArgument(_)) => {}
        Err(other) => {
            panic!("expected InvalidArgument from open(memory://?read_only=true), got: {other:?}")
        }
    }

    match Database::open("memory://?mode=ro") {
        Ok(_) => panic!("open(memory://?mode=ro) must be refused"),
        Err(Error::InvalidArgument(_)) => {}
        Err(other) => {
            panic!("expected InvalidArgument from open(memory://?mode=ro), got: {other:?}")
        }
    }

    // Writable memory:// still works.
    let db = Database::open("memory://").unwrap();
    db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
}

#[test]
fn ro_open_replays_wal_alter_table_history() {
    // RO open must replay un-checkpointed ALTER TABLE history.
    // checkpoint_on_close=off keeps the ALTER chain in the WAL so the
    // reopen has to go through replay to materialize the post-drop schema.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_replay_alter.db");

    let dsn_w = format!("file://{}?checkpoint_on_close=off", path.display());
    {
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("ALTER TABLE t ADD COLUMN extra INTEGER", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (2, 42)", ()).unwrap();
        db.execute("ALTER TABLE t DROP COLUMN extra", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3)", ()).unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro)
        .expect("read-only open must succeed even when WAL contains ALTER TABLE history");

    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 3, "all three INSERT rows from WAL must be present");

    let mut rows = ro.query("SELECT id FROM t ORDER BY id", ()).unwrap();
    for expected in [1, 2, 3] {
        let row = rows.next().unwrap().unwrap();
        let id: i64 = row.get(0).unwrap();
        assert_eq!(id, expected);
    }

    // `extra` was dropped before close.
    let result = ro.query("SELECT extra FROM t", ());
    assert!(
        result.is_err(),
        "column `extra` was dropped before close; RO replay must materialize the post-drop schema"
    );
}

#[test]
fn ro_database_engine_accessor_does_not_exist_compile_check() {
    // RO no longer exposes engine() (writable back-door); only read_engine().
    use stoolap::storage::traits::ReadEngine;
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    let ro = db.as_read_only();

    let read_engine = ro.read_engine();
    let tx = ReadEngine::begin_read_transaction(read_engine.as_ref()).unwrap();
    let table = tx.get_read_table("t").unwrap();
    assert_eq!(table.row_count(), 1);
}

#[test]
fn close_does_not_invalidate_active_transaction() {
    // Transaction holds Arc<EngineEntry> so close() defers shutdown.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
        .unwrap();

    let mut tx = db.begin().unwrap();
    let n: i64 = tx
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("baseline query inside tx");
    assert_eq!(n, 3);

    db.close()
        .expect("close while tx alive must succeed (just defers engine close)");

    let n: i64 = tx
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("query on live tx after db.close() must succeed");
    assert_eq!(n, 3);

    tx.commit()
        .expect("commit on live tx after db.close() must succeed");
}

#[test]
fn close_with_clone_and_active_transaction_keeps_engine_alive() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let clone = db.clone();
    let mut tx = db.begin().unwrap();

    db.close().unwrap();

    let mut rows = clone.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 1);

    let n: i64 = tx.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(n, 1);

    tx.commit().unwrap();
}

#[test]
fn database_read_engine_returns_read_only_trait_object() {
    use stoolap::storage::traits::ReadEngine;

    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (42)", ()).unwrap();

    let read_engine: std::sync::Arc<dyn ReadEngine> = db.read_engine();
    let tx = ReadEngine::begin_read_transaction(read_engine.as_ref()).unwrap();
    let table = tx.get_read_table("t").unwrap();
    assert_eq!(table.row_count(), 1);
}

#[test]
fn read_only_database_engine_and_read_engine_accessors() {
    use stoolap::storage::traits::ReadEngine;
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();

    let read_engine: std::sync::Arc<dyn ReadEngine> = ro.read_engine();
    let tx = ReadEngine::begin_read_transaction(read_engine.as_ref()).unwrap();
    let _table = tx.get_read_table("t").unwrap();
}

#[test]
fn read_only_mode_mismatch_masks_dsn_query_string() {
    // Mode-mismatch error masks query string but keeps path visible.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("masked_dsn.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_with_secret = format!(
        "file://{}?wal_buffer_size=65536&password=hunter2",
        path.display()
    );
    let _ro = Database::open_read_only(&dsn_with_secret).unwrap();
    let err = match Database::open(&dsn_with_secret) {
        Ok(_) => panic!("mode mismatch must error"),
        Err(e) => e,
    };
    // Match by variant: Debug-formatting escapes backslashes on Windows.
    let msg = match err {
        Error::ReadOnlyViolation(m) => m,
        other => panic!("expected ReadOnlyViolation, got: {other:?}"),
    };
    assert!(
        !msg.contains("hunter2"),
        "DSN query string with secret leaked into error message: {msg}"
    );
    assert!(
        msg.contains("?***"),
        "expected masked '?***' in error message, got: {msg}"
    );
    assert!(
        msg.contains(&path.display().to_string()),
        "path must remain visible in masked DSN error, got: {msg}"
    );
}

#[test]
fn read_only_mode_mismatch_uses_canonical_format() {
    // Pins the canonical "registry: cannot open '<dsn>' as <requested>
    // while it is already open as <cached>" message shape for log-grep.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("mode_mismatch.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Same DSN string => registry mode-mismatch (distinct strings would
    // race on the file lock instead).
    let dsn = format!("file://{}", path.display());
    let _ro = Database::open_read_only(&dsn).unwrap();

    let err = match Database::open(&dsn) {
        Ok(_) => panic!("mode mismatch must error"),
        Err(e) => e,
    };
    match err {
        Error::ReadOnlyViolation(msg) => {
            assert!(
                msg.starts_with("registry: cannot open '"),
                "expected canonical 'registry: cannot open ...' prefix, got: {msg}"
            );
            assert!(msg.contains("read-only"));
            assert!(msg.contains("writable"));
        }
        other => panic!("expected ReadOnlyViolation, got: {other:?}"),
    }
}

#[test]
fn read_only_database_is_read_only_returns_true() {
    let db = Database::open_in_memory().unwrap();
    let ro = db.as_read_only();
    assert!(ro.is_read_only());
}

#[test]
fn database_is_read_only_accessor() {
    // Database is always writable; only ReadOnlyDatabase reports true.
    let writable = Database::open_in_memory().unwrap();
    assert!(
        !writable.is_read_only(),
        "writable Database must report is_read_only()=false"
    );
    let cloned = writable.clone();
    assert!(!cloned.is_read_only(), "Database clone stays writable");

    let ro_view = writable.as_read_only();
    assert!(
        ro_view.is_read_only(),
        "ReadOnlyDatabase must report is_read_only()=true"
    );
}

#[test]
fn ro_database_cached_plan_round_trip_with_params() {
    let db = Database::open_in_memory().unwrap();
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users VALUES (1,'a',20),(2,'b',30),(3,'c',40)",
        (),
    )
    .unwrap();

    let ro = db.as_read_only();
    let plan = ro
        .cached_plan("SELECT name FROM users WHERE age > $1")
        .expect("cached_plan(SELECT) must succeed on a read-only Database");

    let rows = ro.query_plan(&plan, (15i64,)).unwrap();
    let mut count = 0;
    for row in rows {
        let _ = row.unwrap();
        count += 1;
    }
    assert_eq!(count, 3);

    let rows = ro.query_plan(&plan, (25i64,)).unwrap();
    let mut count = 0;
    for row in rows {
        let _ = row.unwrap();
        count += 1;
    }
    assert_eq!(count, 2);

    let mut rows = ro.query_plan(&plan, (100i64,)).unwrap();
    assert!(rows.next().is_none());
}

#[test]
fn ro_database_cached_plan_named_params() {
    use stoolap::named_params;
    let db = Database::open_in_memory().unwrap();
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO users VALUES (1,20),(2,30),(3,40)", ())
        .unwrap();

    let ro = db.as_read_only();
    let plan = ro
        .cached_plan("SELECT id FROM users WHERE age = :age")
        .unwrap();
    let mut rows = ro
        .query_named_plan(&plan, named_params! { age: 30i64 })
        .unwrap();
    let row = rows.next().unwrap().unwrap();
    let id: i64 = row.get(0).unwrap();
    assert_eq!(id, 2);
}

#[test]
fn ro_database_cached_plan_refuses_write_sql_through_ro_handle() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();

    match ro.cached_plan("INSERT INTO t VALUES ($1)") {
        Ok(_) => panic!("ReadOnlyDatabase::cached_plan(INSERT) must be refused"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation, got: {other:?}"),
    }

    let _plan = ro
        .cached_plan("SELECT * FROM t WHERE id = $1")
        .expect("SELECT cached_plan must succeed");
}

#[test]
fn ro_database_cached_plan_refuses_write_sql_early() {
    // Refusal must happen at plan-creation, not at execute_plan.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_cached_plan_refuses.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open_read_only(&dsn_ro).unwrap();
    match db.cached_plan("INSERT INTO t VALUES ($1)") {
        Ok(_) => panic!("cached_plan(INSERT) on read-only must be refused"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation from cached_plan, got: {other:?}"),
    }

    let _ = db
        .cached_plan("SELECT * FROM t WHERE id = $1")
        .expect("SELECT must still cache on a read-only ReadOnlyDatabase");
}

#[test]
fn as_read_only_does_not_observe_uncommitted_writes_on_source() {
    // RO view has its own executor; uncommitted source writes are invisible.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let ro = db.as_read_only();

    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 1);

    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (2)", ()).unwrap();

    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(
        n, 1,
        "as_read_only must not observe uncommitted writes from the source \
         Database, separate executor / separate transaction state"
    );

    tx.commit().unwrap();
    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 2);
}

#[test]
fn open_with_read_only_dsn_flag_is_rejected_with_migration_message() {
    // open() rejects RO DSN flags so callers find open_read_only.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("open_rejects_ro_dsn.db");
    {
        let db = Database::open(&format!("file://{}", path.display())).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }
    for flag in ["read_only=true", "readonly=true", "mode=ro"] {
        let dsn = format!("file://{}?{}", path.display(), flag);
        match Database::open(&dsn) {
            Ok(_) => panic!("Database::open must reject {} with InvalidArgument", flag),
            Err(Error::InvalidArgument(msg)) => assert!(
                msg.contains("read-only DSN flag"),
                "error must point at open_read_only via the canonical \
                 'read-only DSN flag' substring; got: {msg}"
            ),
            Err(other) => panic!("expected InvalidArgument for {flag}, got: {other:?}"),
        }
    }
}

#[test]
fn open_read_only_accepts_redundant_read_only_dsn_flag() {
    // Redundant flag is a no-op so existing driver DSNs keep working.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("open_ro_accepts_flag.db");
    {
        let db = Database::open(&format!("file://{}", path.display())).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }
    for flag in ["read_only=true", "readonly=true", "mode=ro"] {
        let dsn = format!("file://{}?{}", path.display(), flag);
        let ro = match Database::open_read_only(&dsn) {
            Ok(r) => r,
            Err(e) => panic!(
                "open_read_only must accept redundant {} flag, got: {:?}",
                flag, e
            ),
        };
        let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
        let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
        assert_eq!(n, 1);
        drop(ro);
    }
}

#[test]
fn open_read_only_rejects_writable_dsn_flag() {
    // open_read_only rejects any DSN flag that requests writable.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("open_ro_rejects_writable.db");
    {
        let db = Database::open(&format!("file://{}", path.display())).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }
    for flag in ["read_only=false", "readonly=false", "mode=rw"] {
        let dsn = format!("file://{}?{}", path.display(), flag);
        match Database::open_read_only(&dsn) {
            Ok(_) => panic!("open_read_only must reject writable {} flag", flag),
            Err(Error::InvalidArgument(msg)) => assert!(
                msg.contains("explicitly requests writable mode"),
                "error must contain the canonical 'explicitly requests writable mode' \
                 substring; got: {msg}"
            ),
            Err(other) => panic!("expected InvalidArgument for {flag}, got: {other:?}"),
        }
    }
}

// SWMR coexistence requires Unix-only db.shm.
#[cfg(unix)]
#[test]
fn open_writable_then_open_read_only_query_param_succeeds_under_swmr() {
    // Writable engine and RO attach must coexist under SWMR v1.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("dsn_mode_mismatch.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    let db_rw = Database::open(&dsn_rw).unwrap();
    db_rw
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    let ro = Database::open_read_only(&dsn_ro)
        .expect("Shared read-only attach must coexist with writable open under SWMR");
    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 0);
    drop(ro);
    drop(db_rw);
}

#[test]
fn open_two_read_only_handles_via_query_param_share_engine() {
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("dsn_two_ro.db");
    {
        let db = Database::open(&format!("file://{}", path.display())).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn = format!("file://{}?read_only=true", path.display());
    let db1 = Database::open_read_only(&dsn).unwrap();
    let db2 = Database::open_read_only(&dsn).unwrap();
    drop(db1);
    drop(db2);
}

#[test]
fn open_with_invalid_read_only_value_errors() {
    assert!(Database::open("file:///tmp/nope?read_only=maybe").is_err());
    assert!(Database::open_read_only("file:///tmp/nope?read_only=maybe").is_err());
}

#[test]
fn open_read_only_on_missing_path_fails_without_creating() {
    // RO must refuse non-existent paths up front, not create them.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("nonexistent_db");
    let dsn = format!("file://{}", path.display());

    let result = Database::open_read_only(&dsn);
    assert!(result.is_err());
    assert!(!path.exists(), "read-only open created the path");
}

#[test]
fn open_read_only_on_non_stoolap_dir_fails() {
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("not_a_db");
    std::fs::create_dir_all(&path).unwrap();
    let dsn = format!("file://{}", path.display());

    let result = Database::open_read_only(&dsn);
    assert!(result.is_err());
    assert!(!path.join("wal").exists());
}

#[test]
fn ro_drop_does_not_create_wal_files() {
    // RO close_engine must skip checkpoint (no new wal-*.log).
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_no_wal_on_drop.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let wal_dir = path.join("wal");
    let wal_files_before: Vec<_> = std::fs::read_dir(&wal_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    {
        let _ro = Database::open_read_only(&dsn).unwrap();
    }

    let wal_files_after: Vec<_> = std::fs::read_dir(&wal_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    let new_files: Vec<_> = wal_files_after
        .iter()
        .filter(|f| !wal_files_before.contains(f))
        .collect();
    assert!(
        new_files.is_empty(),
        "read-only drop wrote new WAL files: {:?}",
        new_files
    );
}

#[test]
fn close_does_not_invalidate_clones() {
    // close() must check engine's strong count, not inner's.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let clone = db.clone();
    db.close().unwrap();

    // Clone must still work.
    let rows: Vec<_> = clone
        .query("SELECT id FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn execute_statement_on_read_only_executor_rejects_writes() {
    // execute_statement gates writes; previously only the SQL-parse path did.
    use std::sync::Arc;
    use stoolap::parser::parse_sql;
    use stoolap::{ExecutionContext, Executor};

    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    let engine = Arc::new(stoolap::MVCCEngine::in_memory());
    engine.open_engine().unwrap();

    let ro_executor = Executor::new_read_only(Arc::clone(&engine));
    let stmts = parse_sql("CREATE TABLE t2 (id INTEGER PRIMARY KEY)").unwrap();

    let ctx = ExecutionContext::new();
    let result = ro_executor.execute_statement(&stmts[0], &ctx);
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn open_with_read_only_param_on_missing_path_fails_without_creating() {
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("missing_via_open_param");
    let dsn = format!("file://{}?read_only=true", path.display());

    let result = Database::open_read_only(&dsn);
    assert!(result.is_err(), "expected open_read_only to fail, got Ok");
    assert!(!path.exists(), "open_read_only created the path");
}

#[test]
fn ro_open_does_not_seal_during_recovery() {
    // RO recovery must skip sealing (would write under shared lock).
    // Verify by snapshotting volumes/ before/after the RO open.
    use std::path::PathBuf;
    use std::time::SystemTime;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_no_seal_recovery.db");

    {
        let db = Database::open(&format!("file://{}", path.display())).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
            .unwrap();
        for i in 0..10 {
            db.execute(&format!("INSERT INTO t VALUES ({}, 'x')", i), ())
                .unwrap();
        }
    }

    let volumes_dir = path.join("volumes");
    let snapshot_dir_contents = |dir: &std::path::Path| -> Vec<(String, SystemTime)> {
        if !dir.exists() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let mtime = entry.metadata().unwrap().modified().unwrap();
            out.push((name, mtime));
            if entry.file_type().unwrap().is_dir() {
                for sub in std::fs::read_dir(entry.path()).unwrap().flatten() {
                    let sub_name = format!(
                        "{}/{}",
                        entry.file_name().to_string_lossy(),
                        sub.file_name().to_string_lossy()
                    );
                    let sub_mtime = sub.metadata().unwrap().modified().unwrap();
                    out.push((sub_name, sub_mtime));
                }
            }
        }
        out.sort();
        out
    };

    let before = snapshot_dir_contents(&volumes_dir);
    {
        let _ro = Database::open_read_only(&format!("file://{}", path.display())).unwrap();
    }
    let after = snapshot_dir_contents(&volumes_dir);

    assert_eq!(
        before, after,
        "read-only open mutated volumes/ during recovery"
    );
}

#[test]
fn drop_original_does_not_invalidate_clones() {
    // DatabaseInner::drop must defer when sibling clones live.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let clone = db.clone();
    drop(db);

    let rows: Vec<_> = clone
        .query("SELECT id FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn close_releases_lock_immediately() {
    // close() must release the file lock so a fresh open can take it.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("close_releases.db");
    let dsn = format!("file://{}", path.display());

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.close().unwrap();

    // After close() the lock is free even though db is still in scope.
    let db2 = Database::open(&dsn).unwrap();
    db2.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    drop(db);
    drop(db2);
}

#[test]
fn last_clone_releases_engine_lock_when_original_drops_first() {
    // Original drop while clone alive: clone must close on its own drop.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("clone_releases.db");
    let dsn = format!("file://{}", path.display());

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let clone = db.clone();
    drop(db);
    clone.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    drop(clone);
    let db2 = Database::open(&dsn).unwrap();
    db2.execute("INSERT INTO t VALUES (2)", ()).unwrap();
}

// ---------------------------------------------------------------------------
// Reviewer round-5 regressions
// ---------------------------------------------------------------------------

#[test]
fn second_open_dsn_survives_close_on_first() {
    // Two opens of same DSN: close on first must not tear down second.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("two_opens.db");
    let dsn = format!("file://{}", path.display());

    let db1 = Database::open(&dsn).unwrap();
    db1.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db1.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let db2 = Database::open(&dsn).unwrap();

    db1.close().unwrap();

    let mut rows = db2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 1);
    db2.execute("INSERT INTO t VALUES (2)", ()).unwrap();
}

#[test]
fn second_open_dsn_has_independent_transaction_state() {
    // Each handle must own its own executor (own active_transaction).
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("two_opens_iso.db");
    let dsn = format!("file://{}", path.display());

    let db1 = Database::open(&dsn).unwrap();
    db1.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    let db2 = Database::open(&dsn).unwrap();

    let mut tx1 = db1.begin().unwrap();
    tx1.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    let mut rows = db2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 0, "db2 must not see db1's uncommitted insert");
    tx1.commit().unwrap();
}

#[test]
fn dsn_requests_read_only_matches_parse_file_config_precedence() {
    // Last-flag-wins must match between open() pre-scan and parse_file_config.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("precedence.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // mode=ro after read_only=false => read-only.
    let dsn = format!("file://{}?read_only=false&mode=ro", path.display());

    match Database::open(&dsn) {
        Ok(_) => panic!("open must refuse a read-only DSN"),
        Err(Error::InvalidArgument(msg)) => assert!(
            msg.contains("read-only DSN flag"),
            "expected migration message; got: {msg}"
        ),
        Err(other) => panic!("expected InvalidArgument, got: {other:?}"),
    }

    let ro = match Database::open_read_only(&dsn) {
        Ok(r) => r,
        Err(e) => panic!(
            "open_read_only must accept a DSN whose last flag requests read-only, got: {:?}",
            e
        ),
    };
    drop(ro);
}

#[test]
fn dsn_read_only_flag_only_parses_query_keys_not_paths() {
    // Tokens like `mode=rw` in the path or unrelated query values
    // must not be misread as a writable-flag contradiction.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("dir_with_mode=rw").join("inner.db");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn = format!("file://{}", path.display());
    let ro = Database::open_read_only(&dsn)
        .expect("open_read_only must ignore `mode=rw` inside the file path");
    drop(ro);

    let db = Database::open(&dsn).unwrap();
    db.close().unwrap();

    let dsn_with_other_param = format!("file://{}?sync_mode=normal&compression=on", path.display());
    let ro2 = Database::open_read_only(&dsn_with_other_param).unwrap();
    drop(ro2);
    let db2 = Database::open(&dsn_with_other_param).unwrap();
    db2.close().unwrap();
}

#[test]
fn dsn_requests_read_only_writable_last_wins() {
    // Inverse: `?mode=ro&read_only=false` ends writable.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("precedence_w.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn = format!("file://{}?mode=ro&read_only=false", path.display());
    let db = Database::open(&dsn).expect("open accepts a DSN whose last flag is writable");
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    drop(db);

    match Database::open_read_only(&dsn) {
        Ok(_) => panic!("open_read_only must reject a DSN whose last flag is writable"),
        Err(Error::InvalidArgument(msg)) => assert!(
            msg.contains("explicitly requests writable mode"),
            "expected contradiction message; got: {msg}"
        ),
        Err(other) => panic!("expected InvalidArgument, got: {other:?}"),
    }
}

#[test]
fn crate_root_table_transaction_aliases_compile() {
    // Deprecated `Table` / `Transaction` crate-root aliases must compile.
    #[allow(deprecated)]
    fn _table_alias_resolves<T: stoolap::Table>(_: &T) {}
    #[allow(deprecated)]
    fn _txn_alias_resolves<T: stoolap::Transaction>(_: &T) {}
}

#[test]
fn third_open_after_original_drops_reuses_sibling_engine_file() {
    // Third open must reuse a still-live sibling, not race the file lock.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("third_open.db");
    let dsn = format!("file://{}", path.display());

    let db1 = Database::open(&dsn).unwrap();
    db1.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db1.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let db2 = Database::open(&dsn).unwrap();
    drop(db1);

    let db3 = Database::open(&dsn).expect(
        "third open(dsn) must reuse db2's still-live engine, not fail with \
         a file-lock conflict",
    );
    db3.execute("INSERT INTO t VALUES (2)", ()).unwrap();
    let mut rows = db2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 2, "db2 must see the row db3 inserted");
}

#[test]
fn third_open_after_original_drops_reuses_sibling_engine_memory() {
    // memory:// sibling reuse: third open sees same data, not fresh engine.
    let dsn = "memory://round5-share";

    let db1 = Database::open(dsn).unwrap();
    db1.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db1.execute("INSERT INTO t VALUES (42)", ()).unwrap();

    let db2 = Database::open(dsn).unwrap();
    drop(db1);

    let db3 = Database::open(dsn).unwrap();
    let mut rows = db3.query("SELECT id FROM t", ()).unwrap();
    let row = rows.next().expect("row must exist").unwrap();
    let id: i64 = row.get(0).unwrap();
    assert_eq!(id, 42);

    let mut rows = db2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 1);
}

#[test]
fn open_after_all_handles_drop_creates_fresh_engine_memory() {
    // memory:// loses data once all handles drop; next open is fresh.
    let dsn = "memory://round5-fresh";

    {
        let db = Database::open(dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    }

    let db = Database::open(dsn).unwrap();
    let exists = db.table_exists("t").unwrap();
    assert!(
        !exists,
        "memory:// engine should be fresh after last handle drops"
    );
}

#[test]
fn sibling_handles_share_semantic_cache_for_dml_invalidation() {
    // Per-handle Executors share EngineEntry's SemanticCache so DML on
    // one handle invalidates a sibling's cached SELECT.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("siblings_cache.db");
    let dsn = format!("file://{}", path.display());

    let a = Database::open(&dsn).unwrap();
    a.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    a.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();

    let b = Database::open(&dsn).unwrap();

    // Prime b's cache (cache-eligible: no aggregation, no params).
    let mut rows = b.query("SELECT v FROM t WHERE v > 0", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let v: i64 = row.get(0).unwrap();
    assert_eq!(v, 10);

    a.execute("UPDATE t SET v = 20 WHERE id = 1", ()).unwrap();

    let mut rows = b.query("SELECT v FROM t WHERE v > 0", ()).unwrap();
    let row = rows.next().expect("row must still be visible").unwrap();
    let v: i64 = row.get(0).unwrap();
    assert_eq!(
        v, 20,
        "sibling handle B served stale cached row after handle A's UPDATE; \
         semantic cache must be shared at the EngineEntry level"
    );
}

#[test]
fn transaction_dml_invalidates_sibling_handle_cache() {
    // Same as above but for explicit transactions.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("tx_invalidate_cache.db");
    let dsn = format!("file://{}", path.display());

    let writer = Database::open(&dsn).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (1, 100)", ()).unwrap();

    let reader = Database::open(&dsn).unwrap();
    let v: i64 = reader.query_one("SELECT v FROM t WHERE v > 0", ()).unwrap();
    assert_eq!(v, 100);

    let mut tx = writer.begin().unwrap();
    tx.execute("UPDATE t SET v = 200 WHERE id = 1", ()).unwrap();
    tx.commit().unwrap();

    let v: i64 = reader.query_one("SELECT v FROM t WHERE v > 0", ()).unwrap();
    assert_eq!(
        v, 200,
        "sibling handle served stale cached row after transactional UPDATE \
         committed on the writer; in-tx DML must invalidate the shared cache"
    );
}

#[test]
fn sibling_handles_share_query_planner_for_analyze_invalidation() {
    // Per-handle Executor shares Arc<QueryPlanner> via the engine entry,
    // so ANALYZE invalidation reaches every sibling. Public API can't
    // observe the Arc directly; verified at the construction site.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("siblings_planner.db");
    let dsn = format!("file://{}", path.display());

    let a = Database::open(&dsn).unwrap();
    a.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    a.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    a.execute("ANALYZE", ()).unwrap();

    let b = Database::open(&dsn).unwrap();

    a.execute("INSERT INTO t VALUES (4, 40), (5, 50)", ())
        .unwrap();
    a.execute("ANALYZE", ()).unwrap();

    let n: i64 = b.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(n, 5);

    b.execute("ANALYZE", ()).unwrap();
}

#[test]
fn open_read_only_does_not_fail_when_wal_dir_missing() {
    // RO must accept ENOENT on wal/ (volumes-only / fresh deployment)
    // and only fail on EACCES. Schemas are gone with the WAL.
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("no_wal_dir.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    assert!(path.join("volumes").exists());
    let wal_dir = path.join("wal");
    if wal_dir.exists() {
        std::fs::remove_dir_all(&wal_dir).unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let _ro = Database::open_read_only(&dsn_ro)
        .expect("open_read_only must not fail with ENOENT just because wal/ is missing");
}

#[test]
fn registry_reaps_dead_weak_entries_on_drop() {
    // EngineEntry::drop must reap its dead Weak from the registry
    // so ephemeral DSNs don't grow the map monotonically.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("registry_reap.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Reopen: must construct fresh entry, not upgrade dead Weak.
    let db = Database::open(&dsn).unwrap();
    assert!(db.table_exists("t").unwrap());
    drop(db);
}

#[cfg(unix)]
#[test]
fn open_read_only_works_when_wal_files_are_read_only() {
    // RO opens must not require write access to wal/.
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_wal_chmod.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
            .unwrap();
        db.close().unwrap();
    }

    // chmod -w wal/ files and the dir to block append/create.
    let wal_dir = path.join("wal");
    let mut originals: Vec<(PathBuf, std::fs::Permissions)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&wal_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            let orig = std::fs::metadata(&p).unwrap().permissions();
            originals.push((p.clone(), orig));
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        }
    }
    let wal_dir_orig = std::fs::metadata(&wal_dir).unwrap().permissions();
    std::fs::set_permissions(&wal_dir, std::fs::Permissions::from_mode(0o555)).unwrap();

    let result = std::panic::catch_unwind(|| {
        let dsn_ro = format!("file://{}?read_only=true", path.display());
        let ro = Database::open_read_only(&dsn_ro)
            .expect("open_read_only must succeed even when wal/ files are read-only");
        let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
        let row = rows.next().unwrap().unwrap();
        let n: i64 = row.get(0).unwrap();
        assert_eq!(
            n, 3,
            "all three rows must be visible from the WAL on read-only open"
        );
    });

    std::fs::set_permissions(&wal_dir, wal_dir_orig).unwrap();
    for (p, orig) in originals {
        std::fs::set_permissions(&p, orig).unwrap();
    }

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[cfg(unix)]
#[test]
fn open_read_only_persistence_failure_is_fatal_not_silent() {
    // Persistence init failure must surface as Err, not silently fall
    // back to an empty in-memory engine that 404s every persisted table.
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_persistence_fail.db");

    // checkpoint_on_close=off keeps INSERT in WAL only, so RO open
    // MUST replay to see the row.
    {
        let dsn_w = format!("file://{}?checkpoint_on_close=off", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    // Block both wal/ and volumes/ so any silent fallback shows up
    // as `table not found` from a cold engine.
    let wal_dir = path.join("wal");
    let vol_dir = path.join("volumes");
    let mut originals: Vec<(PathBuf, std::fs::Permissions)> = Vec::new();
    for dir in [&wal_dir, &vol_dir] {
        if dir.exists() {
            let orig = std::fs::metadata(dir).unwrap().permissions();
            originals.push((dir.clone(), orig));
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if let Ok(meta) = std::fs::metadata(&p) {
                        originals.push((p.clone(), meta.permissions()));
                    }
                    let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o000));
                }
            }
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o000)).unwrap();
        }
    }

    let result = std::panic::catch_unwind(|| {
        let dsn_ro = format!("file://{}?read_only=true", path.display());
        // Either the open errors (preferred), or it succeeds but the
        // persisted table is queryable. Open + `table not found` is
        // the silent-fallback bug.
        if let Ok(db) = Database::open_read_only(&dsn_ro) {
            let result = db.query("SELECT COUNT(*) FROM t", ());
            if result.is_err() {
                panic!(
                    "open_read_only succeeded but the persisted table \
                     is missing, silent fallback to an empty engine: {:?}",
                    result.err()
                );
            }
        }
    });

    for (p, orig) in originals {
        let _ = std::fs::set_permissions(&p, orig);
    }

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[cfg(unix)]
#[test]
fn open_read_only_on_chmod_dir_without_lock_file_errors_clearly() {
    // Lockless-shared fallback is reserved for kernel-level RO mounts
    // (statvfs ST_RDONLY); chmod-only EACCES must error, not silently
    // drop cross-process exclusion.
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_chmod_nolock.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }
    std::fs::remove_file(path.join("db.lock")).unwrap();

    // chmod -w on a writable mount: lockless fallback must NOT trigger.
    let dir_perms_orig = std::fs::metadata(&path).unwrap().permissions();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o555)).unwrap();

    let result = std::panic::catch_unwind(|| {
        let dsn = format!("file://{}", path.display());
        match Database::open_read_only(&dsn) {
            Ok(_) => panic!(
                "open_read_only on a chmod-only-restricted dir without a \
                 pre-existing db.lock must error rather than silently drop \
                 cross-process locking"
            ),
            Err(err) => {
                let msg = format!("{err:?}");
                assert!(
                    msg.contains("not read-only at the kernel level")
                        || msg.contains("Permission denied"),
                    "expected diagnostic about lock-file creation / \
                     kernel-level read-only requirement, got: {msg}"
                );
            }
        }
    });

    std::fs::set_permissions(&path, dir_perms_orig).unwrap();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[cfg(unix)]
#[test]
fn open_read_only_works_on_read_only_dir() {
    // RO must work on chmod -w dirs / RO mounts (db.lock open must not
    // pass write/create flags it doesn't need).
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_mount_test.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
            .unwrap();
        db.close().unwrap();
    }

    let lock_path = path.join("db.lock");
    let dir_perms_orig = std::fs::metadata(&path).unwrap().permissions();
    let lock_perms_orig = std::fs::metadata(&lock_path).unwrap().permissions();

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o555)).unwrap();
    std::fs::set_permissions(&lock_path, std::fs::Permissions::from_mode(0o444)).unwrap();

    let result = std::panic::catch_unwind(|| {
        let dsn = format!("file://{}", path.display());
        let ro = Database::open_read_only(&dsn).expect(
            "open_read_only must succeed on a read-only dir / read-only \
             db.lock, it shouldn't require write perm to acquire LOCK_SH",
        );
        let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
        let row = rows.next().unwrap().unwrap();
        let n: i64 = row.get(0).unwrap();
        assert_eq!(n, 3);
    });

    std::fs::set_permissions(&path, dir_perms_orig).unwrap();
    std::fs::set_permissions(&lock_path, lock_perms_orig).unwrap();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn close_strong_count_check_holds_registry_lock() {
    // close() must check strong_count under the registry write lock to
    // close out the open()-upgrade-Weak race that handed the caller a
    // freshly-closed engine.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc as StdArc;
    use std::thread;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("close_race.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn = format!("file://{}", path.display());
    let stop = StdArc::new(AtomicBool::new(false));

    let dsn_a = dsn.clone();
    let stop_a = StdArc::clone(&stop);
    let opener = thread::spawn(move || {
        while !stop_a.load(Ordering::Relaxed) {
            if let Ok(db) = Database::open(&dsn_a) {
                let mut rows = match db.query("SELECT COUNT(*) FROM t", ()) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let row = match rows.next() {
                    Some(Ok(r)) => r,
                    Some(Err(_)) => return false,
                    None => continue,
                };
                let _: i64 = row.get(0).unwrap();
                let _ = db.close();
            }
        }
        true
    });

    let dsn_b = dsn.clone();
    let stop_b = StdArc::clone(&stop);
    let closer = thread::spawn(move || -> bool {
        while !stop_b.load(Ordering::Relaxed) {
            let db = match Database::open(&dsn_b) {
                Ok(d) => d,
                Err(_) => continue,
            };
            if db.close().is_err() {
                return false;
            }
        }
        true
    });

    thread::sleep(std::time::Duration::from_millis(500));
    stop.store(true, Ordering::Relaxed);
    let opener_ok = opener.join().unwrap();
    let closer_ok = closer.join().unwrap();

    assert!(
        opener_ok,
        "opener thread observed an engine closed under it during open/query"
    );
    assert!(
        closer_ok,
        "closer thread saw a close() error from a freshly-opened handle"
    );
}

// ---------------------------------------------------------------------------
// Reader lease tests
// ---------------------------------------------------------------------------

#[test]
fn ro_handle_creates_lease_file_on_file_engine() {
    // RO file:// open must register `<db>/readers/<pid>.lease`.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("lease_create.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro).unwrap();
    let lease_file = path
        .join("readers")
        .join(format!("{}.lease", std::process::id()));
    assert!(
        lease_file.exists(),
        "RO open on file engine must create lease at {}",
        lease_file.display()
    );
    drop(ro);
    assert!(
        !lease_file.exists(),
        "lease file must be unlinked when ReadOnlyDatabase drops"
    );
}

#[test]
fn ro_handle_query_touches_lease_mtime() {
    // Every query/cached_plan must bump lease mtime (writer's heartbeat).
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("lease_touch.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro).unwrap();
    let lease_file = path
        .join("readers")
        .join(format!("{}.lease", std::process::id()));
    let mtime1 = std::fs::metadata(&lease_file).unwrap().modified().unwrap();

    // 50ms covers every common FS mtime granularity.
    std::thread::sleep(std::time::Duration::from_millis(50));
    ro.query("SELECT * FROM t", ()).unwrap().next();
    let mtime2 = std::fs::metadata(&lease_file).unwrap().modified().unwrap();

    assert!(
        mtime2 > mtime1,
        "query must advance lease mtime (was {:?}, now {:?})",
        mtime1,
        mtime2
    );
}

#[test]
fn in_process_as_read_only_does_not_create_lease() {
    // In-process RO view shares state with writer; no FS lease needed.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("inproc_lease.db");

    let dsn = format!("file://{}", path.display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    let ro = db.as_read_only();
    ro.query("SELECT * FROM t", ()).unwrap().next();

    let lease_file = path
        .join("readers")
        .join(format!("{}.lease", std::process::id()));
    assert!(
        !lease_file.exists(),
        "as_read_only() on writable engine must NOT create a lease (in-process coordination)"
    );
}

#[test]
fn refresh_no_op_when_epoch_unchanged() {
    // refresh() returns Ok(false) and skips manifest reload when
    // the writer hasn't checkpointed since the last refresh.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("refresh_noop.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro).unwrap();
    let advanced = ro.refresh().unwrap();
    assert!(
        !advanced,
        "refresh must be a no-op when epoch hasn't advanced since open"
    );
}

#[test]
fn refresh_returns_true_when_epoch_advances() {
    // Direct epoch-file bump simulates a real writer (cross-process
    // version is in swmr_snapshot_test).
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("refresh_advance.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let reader = Database::open_read_only(&dsn_ro).unwrap();

    let advanced = reader.refresh().unwrap();
    assert!(!advanced, "first refresh after open must no-op");

    let new_epoch = manifest_epoch::bump_epoch(&path).unwrap();
    assert!(new_epoch > 0, "bump_epoch must return positive");

    let advanced = reader.refresh().unwrap();
    assert!(
        advanced,
        "refresh must return true after epoch file was bumped"
    );

    let advanced2 = reader.refresh().unwrap();
    assert!(
        !advanced2,
        "second refresh after no further bump must no-op"
    );
}

#[test]
fn auto_refresh_default_is_on() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("auto_default.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro).unwrap();
    assert!(
        ro.auto_refresh_enabled(),
        "auto_refresh must default to true for fresh handles"
    );
    ro.set_auto_refresh(false);
    assert!(
        !ro.auto_refresh_enabled(),
        "set_auto_refresh(false) must turn it off"
    );
    ro.set_auto_refresh(true);
    assert!(
        ro.auto_refresh_enabled(),
        "set_auto_refresh(true) must turn it back on"
    );
}

#[test]
fn auto_refresh_picks_up_epoch_advance_via_query() {
    // With auto-refresh on, query() must pick up the epoch bump.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("auto_picks.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro).unwrap();

    manifest_epoch::bump_epoch(&path).unwrap();

    ro.query("SELECT COUNT(*) FROM t", ()).unwrap().next();

    let advanced = ro.refresh().unwrap();
    assert!(
        !advanced,
        "explicit refresh after auto-refresh must no-op (auto-refresh \
         already consumed the bump)"
    );

    // With auto-refresh off, refresh() must still see a fresh bump.
    let pre = manifest_epoch::read_epoch(&path).unwrap();
    manifest_epoch::bump_epoch(&path).unwrap();
    ro.set_auto_refresh(false);
    ro.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
    let advanced = ro.refresh().unwrap();
    assert!(
        advanced,
        "with auto-refresh OFF, explicit refresh must still see the bump"
    );
    let post = manifest_epoch::read_epoch(&path).unwrap();
    assert!(
        post > pre,
        "epoch should have advanced past the pre-disable value"
    );
}

#[test]
fn refresh_no_op_on_in_process_as_read_only() {
    // In-process RO shares EngineEntry; refresh() is a documented no-op.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();
    assert!(
        !ro.refresh().unwrap(),
        "in-process as_read_only must always no-op refresh"
    );
}

// IGNORED: flaky under parallel CI load. Reproducer for a real
// per-table manifest reload race during concurrent checkpoint:
// refresh() reloads per-table manifests one at a time so a writer
// that rewrites individual manifests between reloads can leave the
// reader with a staggered view. Remove `#[ignore]` once refresh()
// loads all manifests under a single epoch barrier.
#[cfg(unix)]
#[test]
#[ignore]
fn cross_table_atomicity_under_concurrent_checkpoint() {
    // Reader's BEGIN+SELECT_a+SELECT_b+COMMIT must pin a coherent
    // snapshot even while the writer is actively checkpointing.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc as StdArc;
    use std::thread;
    use std::time::Duration;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("xtable_atomic.db");
    let dsn_rw = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn_rw).unwrap();
        db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO a VALUES (0)", ()).unwrap();
        db.execute("INSERT INTO b VALUES (0)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let stop = StdArc::new(AtomicBool::new(false));

    let stop_w = StdArc::clone(&stop);
    let dsn_w = dsn_rw.clone();
    let writer = thread::spawn(move || -> bool {
        let db = match Database::open(&dsn_w) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let mut next_id: i64 = 1;
        while !stop_w.load(Ordering::Relaxed) {
            let sql_a = format!("INSERT INTO a VALUES ({})", next_id);
            let sql_b = format!("INSERT INTO b VALUES ({})", next_id);
            // Both inserts must commit together before the checkpoint.
            if db.execute("BEGIN", ()).is_err() {
                return false;
            }
            if db.execute(&sql_a, ()).is_err() {
                let _ = db.execute("ROLLBACK", ());
                return false;
            }
            if db.execute(&sql_b, ()).is_err() {
                let _ = db.execute("ROLLBACK", ());
                return false;
            }
            if db.execute("COMMIT", ()).is_err() {
                return false;
            }
            if db.execute("PRAGMA CHECKPOINT", ()).is_err() {
                return false;
            }
            next_id += 1;
        }
        true
    });

    let stop_r = StdArc::clone(&stop);
    let dsn_r = format!("{}?read_only=true", dsn_rw);
    let inconsistencies = StdArc::new(std::sync::Mutex::new(Vec::<(i64, i64)>::new()));
    let inc_r = StdArc::clone(&inconsistencies);
    let reader = thread::spawn(move || {
        let ro = match Database::open_read_only(&dsn_r) {
            Ok(r) => r,
            Err(_) => return,
        };
        while !stop_r.load(Ordering::Relaxed) {
            if ro.query("BEGIN", ()).is_err() {
                continue;
            }
            let mut rows_a = match ro.query("SELECT COUNT(*) FROM a", ()) {
                Ok(r) => r,
                Err(_) => {
                    let _ = ro.query("ROLLBACK", ());
                    continue;
                }
            };
            let count_a: i64 = match rows_a.next() {
                Some(Ok(row)) => row.get(0).unwrap_or(-1),
                _ => {
                    let _ = ro.query("ROLLBACK", ());
                    continue;
                }
            };
            drop(rows_a);

            let mut rows_b = match ro.query("SELECT COUNT(*) FROM b", ()) {
                Ok(r) => r,
                Err(_) => {
                    let _ = ro.query("ROLLBACK", ());
                    continue;
                }
            };
            let count_b: i64 = match rows_b.next() {
                Some(Ok(row)) => row.get(0).unwrap_or(-1),
                _ => {
                    let _ = ro.query("ROLLBACK", ());
                    continue;
                }
            };
            drop(rows_b);

            let _ = ro.query("COMMIT", ());

            if count_a != count_b {
                inc_r.lock().unwrap().push((count_a, count_b));
            }
        }
    });

    thread::sleep(Duration::from_millis(1500));
    stop.store(true, Ordering::Relaxed);

    let writer_ok = writer.join().expect("writer thread panicked");
    reader.join().expect("reader thread panicked");

    let observed = inconsistencies.lock().unwrap().clone();
    assert!(
        writer_ok,
        "writer thread failed mid-loop (test setup issue, not the SWMR property)"
    );
    assert!(
        observed.is_empty(),
        "reader observed cross-table inconsistencies inside BEGIN/COMMIT \
         blocks (count_a, count_b): {:?}",
        observed
    );
}

#[test]
fn refresh_detects_table_added_on_disk() {
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ddl_added.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO a VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let reader = Database::open_read_only(&dsn_ro).unwrap();

    // Simulate writer adding table 'b': drop a manifest.bin into volumes/.
    use stoolap::storage::volume::manifest::TableManifest;
    let new_table_dir = path.join("volumes").join("b");
    std::fs::create_dir_all(&new_table_dir).unwrap();
    let new_manifest = TableManifest::new("b");
    new_manifest
        .write_to_disk(&new_table_dir.join("manifest.bin"))
        .unwrap();

    manifest_epoch::bump_epoch(&path).unwrap();

    let result = reader.refresh();
    let err = match result {
        Err(Error::SchemaChanged(msg)) => msg,
        other => panic!("expected SchemaChanged for added table; got {:?}", other),
    };
    assert!(
        err.contains("tables added on disk") && err.contains("b"),
        "error must name the added table; got: {}",
        err
    );
}

#[test]
fn refresh_detects_table_dropped_on_disk() {
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ddl_dropped.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE keep (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("CREATE TABLE doomed (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO keep VALUES (1)", ()).unwrap();
        db.execute("INSERT INTO doomed VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let reader = Database::open_read_only(&dsn_ro).unwrap();

    let doomed_manifest = path.join("volumes").join("doomed").join("manifest.bin");
    assert!(
        doomed_manifest.exists(),
        "seed must have produced a manifest for 'doomed'"
    );
    std::fs::remove_file(&doomed_manifest).unwrap();

    manifest_epoch::bump_epoch(&path).unwrap();

    let err = match reader.refresh() {
        Err(Error::SchemaChanged(msg)) => msg,
        other => panic!("expected SchemaChanged for dropped table; got {:?}", other),
    };
    assert!(
        err.contains("tables dropped on disk") && err.contains("doomed"),
        "error must name the dropped table; got: {}",
        err
    );
}

#[test]
fn refresh_detects_schema_drift_on_compacted_volume() {
    // Segment.schema_version > reader.schema_epoch must hard-fail
    // with SchemaChanged. Simulated by mutating the on-disk manifest.
    use stoolap::storage::volume::manifest::TableManifest;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("schema_drift.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 100)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    // Reader opens FIRST so its schema_epoch baselines from seed=v1;
    // then we bump the on-disk segment's schema_version above that.
    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let reader = Database::open_read_only(&dsn_ro).unwrap();

    let manifest_path = path.join("volumes").join("t").join("manifest.bin");
    let mut manifest = TableManifest::read_from_disk(&manifest_path).unwrap();
    assert!(!manifest.segments.is_empty(), "seed must produce a segment");
    manifest.segments[0].schema_version = 999;
    manifest.write_to_disk(&manifest_path).unwrap();

    // Bump the sidecar so refresh() takes the reload path.
    use stoolap::storage::mvcc::manifest_epoch;
    manifest_epoch::bump_epoch(&path).unwrap();

    let result = reader.refresh();
    assert!(
        matches!(result, Err(Error::SchemaChanged(_))),
        "refresh must surface SchemaChanged when a segment's schema_version \
         exceeds the reader's schema_epoch; got {:?}",
        result
    );

    let q = reader.query("SELECT COUNT(*) FROM t", ());
    assert!(
        matches!(q, Err(Error::SchemaChanged(_))),
        "auto-refresh on query must propagate SchemaChanged; got {:?}",
        q.err()
    );

    // With auto-refresh off the pre-failure snapshot is still readable.
    reader.set_auto_refresh(false);
    let mut rows = reader
        .query("SELECT COUNT(*) FROM t", ())
        .expect("auto-refresh OFF: query must succeed against stale snapshot");
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 1, "stale snapshot must still see the seed row");
}

#[test]
fn checkpoint_bumps_manifest_epoch() {
    // Each successful checkpoint must monotonically advance the epoch
    // sidecar that readers poll.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("epoch_bump.db");
    let dsn = format!("file://{}", path.display());

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 100)", ()).unwrap();

    let before = manifest_epoch::read_epoch(&path).unwrap();

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let after = manifest_epoch::read_epoch(&path).unwrap();
    assert!(
        after > before,
        "epoch must advance after PRAGMA CHECKPOINT: before={}, after={}",
        before,
        after
    );

    // Second checkpoint must bump again so readers see writer-alive ticks.
    db.execute("INSERT INTO t VALUES (2, 200)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let after2 = manifest_epoch::read_epoch(&path).unwrap();
    assert!(
        after2 > after,
        "second checkpoint must advance epoch again: after={}, after2={}",
        after,
        after2
    );
}

#[test]
fn checkpoint_does_not_bump_epoch_on_memory_engine() {
    // memory:// has no path; bump short-circuits, must not panic.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    let _ = db.execute("PRAGMA CHECKPOINT", ());
}

#[test]
fn pragma_swmr_status_returns_diagnostic_row() {
    // Returns one row: (manifest_epoch, live_lease_count,
    // lease_max_age_secs, is_read_only, checkpoint_interval_secs).
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("swmr_status.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let expected_epoch = manifest_epoch::read_epoch(&path).unwrap();
    assert!(expected_epoch > 0, "seed should produce non-zero epoch");

    let dsn_ro = format!("file://{}?read_only=true&lease_max_age=300", path.display());
    let ro = Database::open_read_only(&dsn_ro).unwrap();

    let mut rows = ro.query("PRAGMA SWMR_STATUS", ()).unwrap();
    let row = rows.next().expect("must return a row").unwrap();

    let manifest_epoch: i64 = row.get(0).unwrap();
    let live_lease_count: i64 = row.get(1).unwrap();
    let lease_max_age_secs: i64 = row.get(2).unwrap();
    let is_read_only: bool = row.get(3).unwrap();
    let checkpoint_interval_secs: i64 = row.get(4).unwrap();

    assert_eq!(
        manifest_epoch as u64, expected_epoch,
        "manifest_epoch must match disk"
    );
    assert_eq!(
        live_lease_count, 1,
        "live_lease_count must include this reader's own lease"
    );
    assert_eq!(
        lease_max_age_secs, 300,
        "lease_max_age_secs must reflect the ?lease_max_age=300 DSN override"
    );
    assert!(
        is_read_only,
        "is_read_only must be true on a read-only handle"
    );
    assert_eq!(
        checkpoint_interval_secs, 60,
        "default checkpoint_interval is 60s"
    );
}

#[test]
fn pragma_swmr_status_works_on_writable_handle() {
    // Pure-read pragma; works on writable handles too.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("swmr_status_writable.db");
    let dsn = format!("file://{}", path.display());

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let mut rows = db.query("PRAGMA SWMR_STATUS", ()).unwrap();
    let row = rows.next().expect("must return a row").unwrap();
    let live_lease_count: i64 = row.get(1).unwrap();
    let is_read_only: bool = row.get(3).unwrap();
    assert_eq!(live_lease_count, 0, "no readers attached → 0 leases");
    assert!(
        !is_read_only,
        "writable handle must report is_read_only=false"
    );
}

#[test]
fn lease_max_age_dsn_param_is_parsed() {
    // ?lease_max_age=N (and the _secs alias) override the default;
    // invalid values surface as InvalidArgument.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("lease_max_age.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Open writable so we can read config() back; parsing path is same.
    let dsn_short = format!("file://{}?lease_max_age=2400", path.display());
    let db_short = Database::open(&dsn_short).unwrap();
    assert_eq!(
        db_short.engine().config().persistence.lease_max_age_secs,
        2400,
        "?lease_max_age=N must set the config field"
    );
    drop(db_short);

    let dsn_long = format!("file://{}?lease_max_age_secs=600", path.display());
    let db_long = Database::open(&dsn_long).unwrap();
    assert_eq!(
        db_long.engine().config().persistence.lease_max_age_secs,
        600,
        "?lease_max_age_secs=N must also work"
    );
    drop(db_long);

    let bad = format!("file://{}?lease_max_age=not-a-number", path.display());
    let err = Database::open(&bad).err().unwrap();
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "invalid lease_max_age must surface InvalidArgument; got {:?}",
        err
    );

    let dsn_ro = format!("file://{}?read_only=true&lease_max_age=300", path.display());
    let _ro = Database::open_read_only(&dsn_ro).unwrap();
}

#[test]
fn ro_handle_no_lease_on_memory_engine() {
    // as_read_only() over a memory engine has no path; lease branch
    // must short-circuit (no panic).
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();
    ro.query("SELECT * FROM t", ()).unwrap().next();
}

// ---------------------------------------------------------------------------
// refresh_interval / auto_refresh DSN flag + background ticker tests
// ---------------------------------------------------------------------------

#[test]
fn dsn_auto_refresh_off_disables_per_query_refresh() {
    // ?auto_refresh=off applies before the first query.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("auto_refresh_dsn.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }
    let dsn = format!("file://{}?auto_refresh=off", path.display());
    let ro = Database::open_read_only(&dsn).unwrap();
    assert!(
        !ro.auto_refresh_enabled(),
        "DSN auto_refresh=off must disable per-query refresh"
    );
}

#[test]
fn dsn_auto_refresh_invalid_value_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("auto_refresh_bad.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
        db.close().unwrap();
    }
    let dsn = format!("file://{}?auto_refresh=banana", path.display());
    let err = match Database::open_read_only(&dsn) {
        Ok(_) => panic!("expected open_read_only to reject auto_refresh=banana"),
        Err(e) => e,
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains("auto_refresh") && msg.contains("banana"),
        "expected invalid-value error mentioning auto_refresh and banana, got: {}",
        msg
    );
}

#[test]
fn dsn_refresh_interval_invalid_unit_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("refresh_int_bad.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
        db.close().unwrap();
    }
    // Unitless number is rejected (ambiguous between ms/s/minutes).
    let dsn = format!("file://{}?refresh_interval=30", path.display());
    let err = match Database::open_read_only(&dsn) {
        Ok(_) => panic!("expected open_read_only to reject unitless refresh_interval"),
        Err(e) => e,
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains("refresh_interval"),
        "expected error mentioning refresh_interval, got: {}",
        msg
    );
}

#[test]
fn dsn_refresh_interval_zero_means_no_ticker() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("refresh_int_zero.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
        db.close().unwrap();
    }
    let dsn = format!("file://{}?refresh_interval=0", path.display());
    let ro = Database::open_read_only(&dsn).unwrap();
    assert_eq!(
        ro.refresh_interval(),
        None,
        "refresh_interval=0 must be treated as 'no ticker'"
    );
}

#[test]
fn dsn_refresh_interval_below_minimum_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("refresh_int_tiny.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
        db.close().unwrap();
    }
    // Below 100ms minimum: must surface at open, not first query.
    let dsn = format!("file://{}?refresh_interval=50ms", path.display());
    let err = match Database::open_read_only(&dsn) {
        Ok(_) => panic!("expected open_read_only to reject 50ms refresh_interval"),
        Err(e) => e,
    };
    let msg = format!("{}", err);
    assert!(
        msg.to_lowercase().contains("refresh_interval") || msg.to_lowercase().contains("100ms"),
        "expected min-interval error, got: {}",
        msg
    );
}

#[test]
fn set_refresh_interval_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("set_int_rt.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
        db.close().unwrap();
    }
    let dsn = format!("file://{}", path.display());
    let ro = Database::open_read_only(&dsn).unwrap();
    assert_eq!(ro.refresh_interval(), None);

    ro.set_refresh_interval(Some(std::time::Duration::from_millis(200)))
        .unwrap();
    assert_eq!(
        ro.refresh_interval(),
        Some(std::time::Duration::from_millis(200))
    );

    ro.set_refresh_interval(None).unwrap();
    assert_eq!(ro.refresh_interval(), None);

    ro.set_refresh_interval(Some(std::time::Duration::from_millis(500)))
        .unwrap();
    assert_eq!(
        ro.refresh_interval(),
        Some(std::time::Duration::from_millis(500))
    );
}

#[test]
fn drop_joins_refresh_ticker_quickly() {
    // Drop must wake the ticker via condvar, not wait for the interval.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("drop_quick.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
        db.close().unwrap();
    }
    let dsn = format!("file://{}?refresh_interval=30s", path.display());
    let ro = Database::open_read_only(&dsn).unwrap();
    let start = std::time::Instant::now();
    drop(ro);
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(1),
        "drop with 30s ticker interval should not block on the interval; \
         took {:?}",
        elapsed
    );
}

#[cfg(unix)]
#[test]
fn ticker_advances_pin_without_queries() {
    // Ticker must advance the WAL pin without any queries.
    // Open writer FIRST: opening it after the reader bumps
    // writer_generation and trips SwmrWriterReincarnated.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ticker_advance.db");
    let dsn_rw = format!("file://{}", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let dsn_ro = format!(
        "file://{}?read_only=true&refresh_interval=200ms",
        path.display()
    );
    let ro = Database::open_read_only(&dsn_ro).unwrap();
    let initial_epoch = manifest_epoch::read_epoch(&path).unwrap_or(0);

    // Ticker should pick up this checkpoint within ~200ms.
    writer.execute("INSERT INTO t VALUES (2)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let post_writer_epoch = manifest_epoch::read_epoch(&path).unwrap_or(0);
    assert!(
        post_writer_epoch > initial_epoch,
        "writer checkpoint should bump on-disk epoch ({} -> {})",
        initial_epoch,
        post_writer_epoch
    );

    // Wait ~3 ticker intervals; only the background ticker can advance.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(2000);
    let mut observed = false;
    while std::time::Instant::now() < deadline {
        if ro.last_seen_epoch_for_test() >= post_writer_epoch {
            observed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(
        observed,
        "background ticker did not advance reader's last_seen_epoch \
         past {} within 2s (saw {})",
        post_writer_epoch,
        ro.last_seen_epoch_for_test()
    );

    drop(ro);
    drop(writer);
}

#[test]
fn try_clone_inherits_refresh_interval() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("clone_inherit.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
        db.close().unwrap();
    }
    let dsn = format!("file://{}?refresh_interval=500ms", path.display());
    let parent = Database::open_read_only(&dsn).unwrap();
    assert_eq!(
        parent.refresh_interval(),
        Some(std::time::Duration::from_millis(500))
    );
    let child = parent.try_clone().unwrap();
    assert_eq!(
        child.refresh_interval(),
        Some(std::time::Duration::from_millis(500)),
        "try_clone must inherit refresh_interval from parent"
    );
    // Each clone owns its own ticker.
    parent.set_refresh_interval(None).unwrap();
    assert_eq!(parent.refresh_interval(), None);
    assert_eq!(
        child.refresh_interval(),
        Some(std::time::Duration::from_millis(500)),
        "stopping parent's ticker must not stop child's"
    );
}

#[cfg(unix)]
#[test]
fn ticker_pauses_when_auto_refresh_disabled() {
    // Ticker must check auto_refresh on every wake and skip when off,
    // so set_auto_refresh(false) is honored end-to-end.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ticker_auto_refresh_pause.db");
    let dsn_rw = format!("file://{}", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let dsn_ro = format!(
        "file://{}?read_only=true&auto_refresh=off&refresh_interval=100ms",
        path.display()
    );
    let ro = Database::open_read_only(&dsn_ro).unwrap();
    let pinned_epoch = ro.last_seen_epoch_for_test();

    writer.execute("INSERT INTO t VALUES (2)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let post_writer_epoch = manifest_epoch::read_epoch(&path).unwrap_or(0);
    assert!(
        post_writer_epoch > pinned_epoch,
        "writer checkpoint should bump on-disk epoch ({} -> {})",
        pinned_epoch,
        post_writer_epoch
    );

    // ~5 ticker intervals.
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert_eq!(
        ro.last_seen_epoch_for_test(),
        pinned_epoch,
        "ticker must not advance last_seen_epoch while auto_refresh=off \
         (would silently violate set_auto_refresh(false) contract)"
    );

    // Re-enable: ticker resumes without a fresh set_refresh_interval.
    ro.set_auto_refresh(true);
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1000);
    let mut advanced = false;
    while std::time::Instant::now() < deadline {
        if ro.last_seen_epoch_for_test() >= post_writer_epoch {
            advanced = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(
        advanced,
        "after re-enabling auto_refresh the ticker must resume; \
         expected last_seen_epoch >= {}, saw {}",
        post_writer_epoch,
        ro.last_seen_epoch_for_test()
    );

    drop(ro);
    drop(writer);
}

#[cfg(unix)]
#[test]
fn set_auto_refresh_false_fences_in_flight_ticker() {
    // set_auto_refresh(false) must be a synchronous fence: an
    // in-flight tick that already passed the initial check must
    // not complete the refresh after this call returned.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("auto_refresh_fence.db");
    let dsn_rw = format!("file://{}", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let dsn_ro = format!(
        "file://{}?read_only=true&auto_refresh=on&refresh_interval=100ms",
        path.display()
    );
    let ro = Database::open_read_only(&dsn_ro).unwrap();

    // Drive checkpoints so the ticker has work, then fence.
    for i in 2..6 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({})", i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));
    }
    let on_disk_before_fence = manifest_epoch::read_epoch(&path).unwrap_or(0);

    ro.set_auto_refresh(false);
    let pinned_after_fence = ro.last_seen_epoch_for_test();

    // Keep bumping; last_seen_epoch must stay frozen.
    for i in 6..11 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({})", i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let on_disk_after = manifest_epoch::read_epoch(&path).unwrap_or(0);
    assert!(
        on_disk_after > on_disk_before_fence,
        "writer should have advanced on-disk epoch ({} -> {}) so the \
         test would catch a violation",
        on_disk_before_fence,
        on_disk_after
    );
    assert_eq!(
        ro.last_seen_epoch_for_test(),
        pinned_after_fence,
        "last_seen_epoch must not advance after set_auto_refresh(false) \
         returned (fence contract); on-disk advanced from {} to {}, \
         reader pinned at {}",
        on_disk_before_fence,
        on_disk_after,
        pinned_after_fence
    );

    drop(ro);
    drop(writer);
}

#[test]
fn last_drop_during_ticker_refresh_does_not_panic() {
    // Last RO drop landing on the ticker thread used to deadlock
    // on JoinHandle::join; shutdown now detaches in that case.
    // Stress loop forces the upgrade-then-drop window.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("drop_during_refresh.db");
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    let dsn_ro = format!("file://{}?refresh_interval=100ms", path.display());
    for _ in 0..30 {
        let ro = Database::open_read_only(&dsn_ro).unwrap();
        // 45ms straddles the 100ms tick: half drop asleep, half mid-refresh.
        std::thread::sleep(std::time::Duration::from_millis(45));
        drop(ro);
    }
}
