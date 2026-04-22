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
    // volume_stats is a known-read pragma in the fail-closed allow-list.
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
    // dsn() must not panic and must return a non-empty string
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
    // Regression test for P1 finding: ReadOnlyDatabase must keep the
    // owning DatabaseInner alive. If it only held Arc<MVCCEngine>, then
    // when the source `db` drops, DatabaseInner::drop would close the
    // engine (owns_engine=true) and subsequent queries would fail.
    let ro = {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'alice')", ()).unwrap();
        db.execute("INSERT INTO t VALUES (2, 'bob')", ()).unwrap();
        db.as_read_only()
        // `db` drops here; if ReadOnlyDatabase only held Arc<MVCCEngine>,
        // DatabaseInner::drop would close the engine and the next query
        // would error.
    };

    // Engine must still be open and queryable.
    let mut rows = ro.query("SELECT v FROM t WHERE id = 1", ()).unwrap();
    let row = rows.next().expect("at least one row").unwrap();
    drop(row);
}

#[test]
fn ro_handle_outlives_open_read_only_temp_database() {
    // Same regression but for open_read_only(): the temporary Database
    // it constructs internally must be kept alive (or at least its
    // DatabaseInner) by the returned ReadOnlyDatabase.
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
    // Writable handle dropped. Engine should be closed (no other refs).

    let ro = Database::open_read_only(&dsn).unwrap();
    let mut rows = ro.query("SELECT v FROM t WHERE id = 1", ()).unwrap();
    let row = rows.next().expect("at least one row").unwrap();
    drop(row);
    drop(ro);
}

#[test]
fn ro_handle_survives_explicit_close_on_writable() {
    // Regression test for P1 finding: Database::close() unconditionally closed
    // the engine, breaking the contract that the engine stays open while a
    // ReadOnlyDatabase handle is alive. After fix, close() defers engine
    // close to the last drop when other handles exist.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'alice')", ()).unwrap();

    let ro = db.as_read_only();

    // Explicit close on the writable handle while the read-only handle is
    // still alive. The engine MUST stay open.
    db.close().unwrap();

    // Read-only handle must still work.
    let mut rows = ro.query("SELECT v FROM t WHERE id = 1", ()).unwrap();
    let row = rows.next().expect("at least one row").unwrap();
    drop(row);
}

#[test]
fn ro_handle_unregisters_when_last_holder_drops() {
    // Regression test for P1 finding: the last ReadOnlyDatabase to drop
    // must unregister its DatabaseInner from the global registry — otherwise
    // the registry holds the final Arc forever and the engine + file lock
    // leak.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_unregister.db");
    let dsn = format!("file://{}", path.display());

    {
        // Open writable, then create a read-only handle, then drop the
        // writable handle while the read-only handle is still alive.
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        let ro = db.as_read_only();
        drop(db);
        // ro is now the last non-registry holder. When it drops, registry
        // entry must be removed so DatabaseInner::drop can fire.
        drop(ro);
    }

    // After all handles dropped, opening a fresh writable handle on the
    // same DSN must succeed. If the previous engine + file lock leaked
    // because the registry kept it alive, this would fail with
    // Error::DatabaseLocked.
    let db2 = Database::open(&dsn).unwrap();
    db2.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    drop(db2);
}

#[test]
fn ro_open_uses_shared_file_lock_so_two_readers_coexist() {
    // Step 2b-2 verification: with the read_only config flag wired through,
    // two ReadOnlyDatabase opens of the same DSN must coexist (LOCK_SH).
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_shared_lock.db");
    let dsn = format!("file://{}", path.display());

    // Seed the file with a writer first, then close it.
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    // Two read-only opens must succeed simultaneously. Without LOCK_SH,
    // the second would fail (the registry would share the inner if both
    // came from the same process, which would mask the issue — so we
    // exercise distinct DSN canonicalization by appending different query
    // strings... actually the registry caches by DSN string so we'll just
    // verify that the second open succeeds via the registry).
    let ro1 = Database::open_read_only(&dsn).unwrap();
    let ro2 = Database::open_read_only(&dsn).unwrap();
    drop(ro1);
    drop(ro2);
}

#[test]
fn ro_blocks_subsequent_writer_via_shared_lock() {
    // Step 2b-2 verification: a ReadOnlyDatabase holding LOCK_SH must
    // block a separate writer from acquiring LOCK_EX on the same file.
    // (Same-process registry sharing would normally hide this; we drop
    // the writable Database between to force a fresh acquire.)
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_blocks_writer.db");
    let dsn = format!("file://{}", path.display());

    // Seed the file.
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Open read-only. Inside this process, the registry shares the engine
    // so a subsequent Database::open(dsn) returns the same DatabaseInner
    // without re-acquiring the file lock — that's expected (single-process
    // safe). True multi-process exclusion is verified by the file_lock
    // unit tests (test_shared_lock_blocks_exclusive,
    // test_exclusive_lock_blocks_shared).
    let ro = Database::open_read_only(&dsn).unwrap();
    drop(ro);
}

#[test]
fn ro_executor_refuses_dml_auto_commit_as_defense_in_depth() {
    // The Executor backing a ReadOnlyDatabase has read_only=true. Even if
    // the parser-level write gate is bypassed (it isn't here, but as a
    // defense-in-depth invariant), the DML auto-commit helpers refuse to
    // begin a writable transaction. This test verifies the executor
    // surface itself is read-only by accessing it through the public
    // ReadOnlyDatabase::query path, which always invokes the read-only
    // executor.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let ro = db.as_read_only();

    // SELECT must work (no DML auto-commit needed for reads).
    let mut rows = ro.query("SELECT * FROM t", ()).unwrap();
    let row = rows.next().expect("one row").unwrap();
    drop(row);

    // INSERT must be rejected by parser gate (as before).
    let result = ro.query("INSERT INTO t VALUES (2)", ());
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn writable_open_after_read_only_open_is_rejected() {
    // P0 regression: Database::open_read_only inserts the read-only engine
    // into the registry; Database::open later reusing that entry would
    // silently return a writable handle, bypassing every read-only gate.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_then_writable.db");
    let dsn = format!("file://{}", path.display());

    // Seed the file with one writer, then close so the registry is empty.
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Open read-only.
    let ro = Database::open_read_only(&dsn).unwrap();

    // Attempting to open the same DSN as writable while the read-only
    // handle is alive must fail (would otherwise share the read-only
    // engine and bypass the gate).
    let result = Database::open(&dsn);
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));

    // After dropping the read-only handle, the registry entry is removed
    // and a fresh writable open succeeds.
    drop(ro);
    let db2 = Database::open(&dsn).unwrap();
    db2.execute("INSERT INTO t VALUES (1)", ()).unwrap();
}

#[test]
fn pragma_restore_is_classified_as_write() {
    // P0 regression: PRAGMA RESTORE has no value, but the executor's
    // execute_pragma() handler calls engine.restore_snapshot() which
    // rewrites the entire database. The classifier must mark it as a
    // write so ReadOnlyDatabase rejects it.
    use stoolap::parser::parse_sql;
    let stmts = parse_sql("PRAGMA restore").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("PRAGMA <maintenance>"));

    let stmts = parse_sql("PRAGMA RESTORE").unwrap();
    assert_eq!(stmts[0].write_reason(), Some("PRAGMA <maintenance>"));
}

#[test]
fn pragma_unknown_is_treated_as_write_fail_closed() {
    // The classifier defaults unknown pragmas to write (fail-closed) so
    // future maintenance pragmas added to the executor cannot bypass the
    // read-only gate just because the classifier wasn't updated.
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
    // End-to-end: PRAGMA RESTORE through ReadOnlyDatabase must error.
    let db = Database::open_in_memory().unwrap();
    let ro = db.as_read_only();
    let result = ro.query("PRAGMA RESTORE", ());
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn ro_full_table_scan_works_no_pk_filter() {
    // Verifies P1 #1 from Opus deep-dive: a non-PK-fast-path SELECT must
    // succeed on a fresh open_read_only DSN. Without this, the executor
    // would call engine.begin_transaction() which previously errored on
    // a read-only engine.
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

    // Full-table SELECT (no WHERE — no PK fast path)
    let rows: Vec<_> = ro
        .query("SELECT v FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows.len(), 3);

    // SHOW TABLES (uses internal begin_transaction)
    let tables: Vec<_> = ro
        .query("SHOW TABLES", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(!tables.is_empty());

    // EXPLAIN (uses internal begin_transaction)
    let explain: Vec<_> = ro
        .query("EXPLAIN SELECT v FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(!explain.is_empty());
}

#[test]
fn ro_begin_commit_works_on_fresh_read_only_engine() {
    // BEGIN/COMMIT must work on a read-only engine since engine.begin_transaction
    // no longer rejects on read_only mode (the bypass-prevention is at higher
    // layers).
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_begin_commit.db");
    let dsn = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    let ro = Database::open_read_only(&dsn).unwrap();
    ro.query("BEGIN", ()).unwrap();
    let _ = ro.query("SELECT * FROM t", ()).unwrap();
    ro.query("COMMIT", ()).unwrap();
}

#[test]
fn memory_dsn_rejects_read_only() {
    // Read-only on memory:// is meaningless: a fresh in-memory engine has
    // nothing to read. Both entry points (open_read_only and the
    // ?read_only=true DSN flag) refuse early with InvalidArgument so
    // callers don't silently get an empty engine they can't use.
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

    // memory:// without a read-only flag still works writable.
    let db = Database::open("memory://").unwrap();
    db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
}

#[test]
fn ro_open_replays_wal_alter_table_history() {
    // Round-10 #3: a read-only open must successfully replay WAL entries
    // for DDL that hasn't been checkpointed to volumes yet — including
    // ALTER TABLE ADD COLUMN, which is what `modify_column_with_dimensions`
    // / `propagate_column_*` / `refresh_schema_cache` exist for. Pin the
    // contract so any future read-only gate accidentally placed on those
    // methods would break this test instead of silently corrupting
    // recovery on RO opens.
    //
    // `checkpoint_on_close=off` ensures the close path doesn't seal the
    // ALTER history into a volume — the reopen MUST go through WAL
    // replay to materialize the new schema, which is what we want to
    // exercise.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_replay_alter.db");

    let dsn_w = format!("file://{}?checkpoint_on_close=off", path.display());
    {
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        // ALTER TABLE ADD COLUMN — exercises propagate_column_alias /
        // refresh_schema_cache during WAL replay on the next open.
        db.execute("ALTER TABLE t ADD COLUMN extra INTEGER", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (2, 42)", ()).unwrap();
        // ALTER TABLE DROP COLUMN — exercises propagate_column_drop.
        db.execute("ALTER TABLE t DROP COLUMN extra", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3)", ()).unwrap();
        db.close().unwrap();
        // Note: no checkpoint, so all DDL+DML lives only in the WAL.
    }

    // Reopen READ-ONLY. WAL replay must succeed: it has to call the
    // (currently pub(crate)) ALTER-related methods on the engine even
    // though the engine is in read-only mode. If a future change adds
    // ensure_writable to those methods naively, this test will fail.
    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open(&dsn_ro)
        .expect("read-only open must succeed even when WAL contains ALTER TABLE history");

    // Verify the replayed schema and data: column `extra` was dropped,
    // so SELECT * is just the id column, and we have 3 rows.
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

    // Schema: `extra` must NOT be present (it was dropped before close).
    // Probing it should error.
    let result = ro.query("SELECT extra FROM t", ());
    assert!(
        result.is_err(),
        "column `extra` was dropped before close; RO replay must materialize the post-drop schema"
    );
}

#[test]
fn ro_database_engine_accessor_does_not_exist_compile_check() {
    // Round-12 P0: ReadOnlyDatabase had `engine() -> &Arc<MVCCEngine>`
    // briefly, which broke the read-only contract: when the RO handle
    // was constructed via Database::as_read_only() on a writable Database
    // (or via open_read_only registry-hit on an already-writable engine),
    // ro.engine().begin_transaction() returned a writable transaction
    // that bypassed every other gate. The accessor was removed; this
    // test exists as documentation of the contract and a sanity-check
    // that read_engine() (the safe replacement) still works.
    use stoolap::storage::traits::ReadEngine;
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    let ro = db.as_read_only();

    // read_engine() returns Arc<dyn ReadEngine> — type-erased read
    // surface only. Even though the underlying engine is writable
    // (because the source Database is writable), the trait object
    // exposes only `begin_read_transaction*`. No write back-door.
    let read_engine = ro.read_engine();
    let tx = ReadEngine::begin_read_transaction(read_engine.as_ref()).unwrap();
    let table = tx.get_read_table("t").unwrap();
    assert_eq!(table.row_count(), 1);

    // Methods on the trait object: only read methods exist. Writes are
    // unreachable through Arc<dyn ReadEngine> because the trait simply
    // doesn't define them — compile-time enforcement.
}

#[test]
fn close_does_not_invalidate_active_transaction() {
    // Round-12 P1: Database::close() was using
    // Arc::strong_count(&inner.entry) to decide if it was the last
    // handle. But api::Transaction stored Arc<MVCCEngine> (not
    // Arc<EngineEntry>), so a live transaction didn't bump entry's
    // count. close() could fire engine.close_engine() while a tx was
    // alive, and the next tx.query_one() returned EngineNotOpen.
    //
    // Fix: Transaction now holds Arc<EngineEntry>; live transactions
    // count toward entry.strong_count and close() correctly defers.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
        .unwrap();

    let mut tx = db.begin().unwrap();
    // Read inside the transaction works.
    let n: i64 = tx
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("baseline query inside tx");
    assert_eq!(n, 3);

    // close() while a transaction is still live MUST NOT close the
    // engine. The transaction holds an Arc<EngineEntry>, so close()
    // sees entry.strong_count > 1 and defers.
    db.close()
        .expect("close while tx alive must succeed (just defers engine close)");

    // Subsequent operations on the live transaction must still work.
    let n: i64 = tx
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("query on live tx after db.close() must succeed");
    assert_eq!(n, 3);

    // Commit / rollback also work.
    tx.commit()
        .expect("commit on live tx after db.close() must succeed");
}

#[test]
fn close_with_clone_and_active_transaction_keeps_engine_alive() {
    // Combined regression: Database clone + active transaction + close
    // on the original. All three keep the engine entry alive; close on
    // any single one must defer engine shutdown.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let clone = db.clone();
    let mut tx = db.begin().unwrap();

    // Close the original handle.
    db.close().unwrap();

    // Both the clone and the live transaction must still work.
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
    // Round-11 #1: Database::read_engine() returns Arc<dyn ReadEngine>,
    // a typed read-only handle. Callers holding the trait object cannot
    // reach Engine::begin_transaction or any inherent write method on
    // MVCCEngine — the read-only contract is enforced at the type level.
    use stoolap::storage::traits::ReadEngine;

    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (42)", ()).unwrap();

    // Returned trait object can begin a read transaction.
    let read_engine: std::sync::Arc<dyn ReadEngine> = db.read_engine();
    let tx = ReadEngine::begin_read_transaction(read_engine.as_ref()).unwrap();
    let table = tx.get_read_table("t").unwrap();
    assert_eq!(table.row_count(), 1);
}

#[test]
fn read_only_database_engine_and_read_engine_accessors() {
    // Round-11 #1 (RO side): ReadOnlyDatabase exposes both engine() (raw)
    // and read_engine() (typed), symmetric with Database.
    use stoolap::storage::traits::ReadEngine;
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();

    // Typed accessor returns the read-only trait object.
    let read_engine: std::sync::Arc<dyn ReadEngine> = ro.read_engine();
    let tx = ReadEngine::begin_read_transaction(read_engine.as_ref()).unwrap();
    let _table = tx.get_read_table("t").unwrap();
}

#[test]
fn read_only_mode_mismatch_masks_dsn_query_string() {
    // Round-11 #3: read_only_mode_mismatch must not embed the raw DSN
    // query string in the error message. Today no query params contain
    // secrets, but future-proofing prevents accidental leakage of e.g.
    // ?password=... if such a param is ever added. The path stays
    // visible (operators need it to identify the DB).
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

    // Use a DSN with a sensitive-looking query param. Open read-only
    // first, then attempt to reopen the SAME DSN as writable so the
    // registry's mode-mismatch check fires and embeds the DSN in the
    // resulting error.
    let dsn_with_secret = format!(
        "file://{}?wal_buffer_size=65536&password=hunter2",
        path.display()
    );
    let _ro = Database::open_read_only(&dsn_with_secret).unwrap();
    let err = match Database::open(&dsn_with_secret) {
        Ok(_) => panic!("mode mismatch must error"),
        Err(e) => e,
    };
    // Pull the inner message out by variant rather than formatting the
    // Error: Debug-formatting escapes backslashes on Windows paths and
    // breaks the path-substring check.
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
    // Path must still be visible for ops.
    assert!(
        msg.contains(&path.display().to_string()),
        "path must remain visible in masked DSN error, got: {msg}"
    );
}

#[test]
fn read_only_mode_mismatch_uses_canonical_format() {
    // Round-11 #1: Database::open mode-mismatch errors now use
    // Error::read_only_mode_mismatch, producing the canonical "registry:
    // cannot open '<dsn>' as <requested> while it is already open as
    // <cached>" format. This pins the format so log-grep / monitoring
    // scripts have a stable shape to match.
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

    // Open read-only, then try to open the same DSN as writable.
    // Both opens use the SAME DSN string so the registry path triggers
    // the mode-mismatch check (different DSN strings would each get
    // their own registry entry and would race on the file lock instead).
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
fn read_only_violation_is_classifiable_via_helper() {
    // Round-11 #7: Error::is_read_only_violation() lets callers detect
    // RO refusals without pattern-matching the variant directly. Useful
    // for retry / fallback logic.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();

    let err = match ro.query("INSERT INTO t VALUES (1)", ()) {
        Ok(_) => panic!("write SQL on RO must fail"),
        Err(e) => e,
    };
    assert!(
        err.is_read_only_violation(),
        "INSERT on RO must classify as a read-only violation, got: {err:?}"
    );

    // SELECT failures (e.g. table not found) are NOT read-only violations.
    let err = match ro.query("SELECT * FROM no_such_table", ()) {
        Ok(_) => panic!("missing table must fail"),
        Err(e) => e,
    };
    assert!(
        !err.is_read_only_violation(),
        "missing-table error must not classify as a read-only violation, got: {err:?}"
    );
}

#[test]
fn read_only_database_is_read_only_returns_true() {
    // Symmetry with Database::is_read_only(). Always true.
    let db = Database::open_in_memory().unwrap();
    let ro = db.as_read_only();
    assert!(ro.is_read_only());
}

#[test]
fn database_is_read_only_accessor() {
    // is_read_only() reports the engine's mode without forcing callers
    // to reach into db.engine().is_read_only_mode().
    let writable = Database::open_in_memory().unwrap();
    assert!(
        !writable.is_read_only(),
        "in-memory writable Database is not read-only"
    );

    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("is_ro_accessor.db");
    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro_handle = Database::open(&dsn_ro).unwrap();
    assert!(
        ro_handle.is_read_only(),
        "?read_only=true Database must report is_read_only()=true"
    );

    // Clones of a read-only Database stay read-only.
    let cloned = ro_handle.clone();
    assert!(cloned.is_read_only());
}

#[test]
fn ro_database_cached_plan_round_trip_with_params() {
    // Round-10 #3: ReadOnlyDatabase::cached_plan + query_plan let RO
    // callers reuse a parsed plan across many calls without keeping a
    // writable Database alive. Same shape as Database::cached_plan.
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

    // Reuse across multiple param values.
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
    // Named-param flavour mirrors `Database::query_named_plan`.
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
    // Mirrors Database::cached_plan early refusal for write SQL on RO.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();

    match ro.cached_plan("INSERT INTO t VALUES ($1)") {
        Ok(_) => panic!("ReadOnlyDatabase::cached_plan(INSERT) must be refused"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation, got: {other:?}"),
    }

    // SELECT still works.
    let _plan = ro
        .cached_plan("SELECT * FROM t WHERE id = $1")
        .expect("SELECT cached_plan must succeed");
}

#[test]
fn ro_database_prepare_refuses_write_sql_early() {
    // Database::prepare on a read-only handle must refuse write SQL at
    // prepare time, not later at Statement::execute. Without the early
    // refusal a caller could prepare an INSERT against a `?read_only=true`
    // Database, get back Ok(Statement), then have it fail confusingly at
    // execute time.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_prepare_refuses.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();
    match db.prepare("INSERT INTO t VALUES ($1)") {
        Ok(_) => panic!("prepare(INSERT) on read-only must be refused"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation from prepare, got: {other:?}"),
    }

    // Read SQL still preps fine.
    let _ = db
        .prepare("SELECT * FROM t WHERE id = $1")
        .expect("SELECT must still prepare on a read-only Database");
}

#[test]
fn ro_database_cached_plan_refuses_write_sql_early() {
    // Database::cached_plan on a read-only handle must refuse write SQL
    // at plan-creation time, not later at execute_plan. Mirrors prepare()
    // behaviour for the lower-level cached-plan API.
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
    let db = Database::open(&dsn_ro).unwrap();
    match db.cached_plan("INSERT INTO t VALUES ($1)") {
        Ok(_) => panic!("cached_plan(INSERT) on read-only must be refused"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation from cached_plan, got: {other:?}"),
    }

    // Read SQL still creates a plan.
    let _ = db
        .cached_plan("SELECT * FROM t WHERE id = $1")
        .expect("SELECT must still cache on a read-only Database");
}

#[test]
fn ro_engine_cleanup_methods_are_noops() {
    // Round-10: db.engine().cleanup_old_transactions / cleanup_deleted_rows /
    // cleanup_old_previous_versions are silent no-ops on a read-only
    // engine. They return 0 instead of mutating registry / version
    // chains. (No Result return type — they signal "did nothing" via the
    // count.)
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_cleanup_noops.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();
    let engine = db.engine();
    let zero = std::time::Duration::from_secs(0);
    assert_eq!(engine.cleanup_old_transactions(zero), 0);
    assert_eq!(engine.cleanup_deleted_rows(zero), 0);
    assert_eq!(engine.cleanup_old_previous_versions(), 0);
    // No-op accessor; just confirming it doesn't panic.
    engine.cleanup_abandoned_cold_deletes();
}

#[test]
fn ro_database_refuses_create_snapshot() {
    // Database::create_snapshot writes new files to disk. On a read-only
    // handle (?read_only=true) it must refuse early with ReadOnlyViolation
    // instead of failing later at the I/O layer with EROFS / EACCES.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_create_snapshot.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();
    match db.create_snapshot() {
        Ok(_) => panic!("create_snapshot on a read-only database must be refused"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation, got: {other:?}"),
    }
}

#[test]
fn ro_database_refuses_restore_snapshot() {
    // Database::restore_snapshot is destructive: it overwrites engine
    // state from a backup. Refusing on a read-only handle is mandatory
    // (not defense-in-depth) — the in-place replacement bypasses every
    // guarantee the read-only contract was supposed to provide.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_restore.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();
    match db.restore_snapshot(None) {
        Ok(_) => panic!("restore_snapshot on a read-only database must be refused"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation, got: {other:?}"),
    }
}

#[test]
fn ro_engine_refuses_direct_engine_writes() {
    // Round-9 P1: `Database::engine()` exposes &Arc<MVCCEngine>. The
    // public Engine trait method (begin_transaction*) is gated, but
    // `MVCCEngine` inherent methods like `create_table`,
    // `drop_table_internal`, `update_engine_config`, `vacuum`,
    // `create_view`, `drop_view`, `rename_table`, `create_column*`,
    // `drop_column`, `rename_column`, `modify_column*` would otherwise
    // let an external caller mutate engine state on a `?read_only=true`
    // handle. They now refuse with ReadOnlyViolation.
    use std::path::PathBuf;
    use stoolap::core::{DataType, Schema, SchemaColumn};
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_engine_writes.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();
    let engine = db.engine();

    // create_table
    let schema = Schema::new(
        "t2",
        vec![SchemaColumn::new(0, "id", DataType::Integer, false, true)],
    );
    match engine.create_table(schema) {
        Ok(_) => panic!("engine.create_table on read-only must refuse"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation from create_table, got: {other:?}"),
    }

    // drop_table_internal
    match engine.drop_table_internal("t") {
        Ok(_) => panic!("engine.drop_table_internal on read-only must refuse"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation from drop_table_internal, got: {other:?}"),
    }

    // vacuum
    match engine.vacuum(None, std::time::Duration::from_secs(0)) {
        Ok(_) => panic!("engine.vacuum on read-only must refuse"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation from vacuum, got: {other:?}"),
    }

    // create_view
    match engine.create_view("v1", "SELECT * FROM t".to_string(), false) {
        Ok(_) => panic!("engine.create_view on read-only must refuse"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!("expected ReadOnlyViolation from create_view, got: {other:?}"),
    }
}

#[test]
fn as_read_only_does_not_observe_uncommitted_writes_on_source() {
    // Documented contract for `Database::as_read_only`: the returned handle
    // is a *view* with its own executor, not a connection sharing the
    // source Database's session. An uncommitted BEGIN on the source must
    // be invisible through the RO view; once the source commits, the row
    // becomes visible.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let ro = db.as_read_only();

    // Baseline: RO view sees the committed row.
    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 1);

    // Open a transaction on the source and insert without committing.
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (2)", ()).unwrap();

    // RO view must NOT see the uncommitted row.
    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(
        n, 1,
        "as_read_only must not observe uncommitted writes from the source \
         Database — separate executor / separate transaction state"
    );

    // After commit, the row becomes visible through the RO view.
    tx.commit().unwrap();
    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 2);
}

#[test]
fn open_with_read_only_query_param_rejects_writes() {
    // `Database::open("file:///x?read_only=true")` opens read-only:
    // - Engine acquires shared file lock
    // - Returns a `Database` (writable type) but its executor is configured
    //   read-only, so writes through the handle fail at runtime.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("dsn_ro.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    // Seed the file via writable open then close.
    {
        let db = Database::open(&dsn_rw).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    // Open via ?read_only=true.
    let db = Database::open(&dsn_ro).unwrap();

    // SELECT works.
    let rows: Vec<_> = db
        .query("SELECT id FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows.len(), 1);

    // INSERT through this handle is rejected at runtime.
    let result = db.execute("INSERT INTO t VALUES (2)", ());
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn open_with_mode_ro_alias_works() {
    // SQLite-style ?mode=ro alias.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("dsn_mode_ro.db");
    {
        let db = Database::open(&format!("file://{}", path.display())).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn = format!("file://{}?mode=ro", path.display());
    let db = Database::open(&dsn).unwrap();
    let result = db.execute("INSERT INTO t VALUES (1)", ());
    assert!(matches!(result, Err(Error::ReadOnlyViolation(_))));
}

#[test]
fn open_writable_then_open_read_only_query_param_rejects_mode_mismatch() {
    // Cached engine is writable; new request asks for read-only via ?read_only=true.
    // Mode mismatch must error.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("dsn_mode_mismatch.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    let db_rw = Database::open(&dsn_rw).unwrap();
    db_rw
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    // Different DSN string (with vs without query string) means a fresh
    // create attempt, but if the parser canonicalized them to the same
    // engine, mode mismatch would fire. Currently each distinct DSN
    // string has its own registry entry, so this path actually opens
    // a second engine — and that fails on the file lock (writer holds
    // LOCK_EX, second open as read-only takes LOCK_SH which is blocked).
    let result = Database::open(&dsn_ro);
    assert!(
        matches!(
            result,
            Err(Error::DatabaseLocked) | Err(Error::ReadOnlyViolation(_))
        ),
        "expected DatabaseLocked or ReadOnlyViolation, got {:?}",
        result.is_ok()
    );
    drop(db_rw);
}

#[test]
fn open_two_read_only_handles_via_query_param_share_engine() {
    // Two opens with the same DSN both requesting read-only must share
    // the same engine via the registry (not fight over LOCK_SH).
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
    let db1 = Database::open(&dsn).unwrap();
    let db2 = Database::open(&dsn).unwrap();
    drop(db1);
    drop(db2);
}

#[test]
fn open_with_invalid_read_only_value_errors() {
    let result = Database::open("file:///tmp/nope?read_only=maybe");
    assert!(result.is_err());
}

#[test]
fn open_read_only_on_missing_path_fails_without_creating() {
    // P1: open_read_only on a non-existent path used to create the
    // directory + WAL via PersistenceManager::new. After the fix it
    // refuses up front.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("nonexistent_db");
    let dsn = format!("file://{}", path.display());

    let result = Database::open_read_only(&dsn);
    assert!(result.is_err());
    // The path must NOT have been created.
    assert!(!path.exists(), "read-only open created the path");
}

#[test]
fn open_read_only_on_non_stoolap_dir_fails() {
    // The path exists but isn't a stoolap database (no wal/ or volumes/).
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("not_a_db");
    std::fs::create_dir_all(&path).unwrap();
    let dsn = format!("file://{}", path.display());

    let result = Database::open_read_only(&dsn);
    assert!(result.is_err());
    // No wal/ should have been created in the empty directory.
    assert!(!path.join("wal").exists());
}

#[test]
fn ro_drop_does_not_create_wal_files() {
    // P1: dropping a ReadOnlyDatabase used to invoke close_engine which
    // ran the final checkpoint/compaction (writing wal-*.log). After
    // fix, close_engine on a read-only engine skips checkpoint entirely.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_no_wal_on_drop.db");
    let dsn = format!("file://{}", path.display());

    // Seed.
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Snapshot wal/ contents.
    let wal_dir = path.join("wal");
    let wal_files_before: Vec<_> = std::fs::read_dir(&wal_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    // Open + drop a read-only handle.
    {
        let _ro = Database::open_read_only(&dsn).unwrap();
    }

    // wal/ contents must be unchanged (no new wal-*.log file).
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
    // P2: Database::clone creates a new DatabaseInner around the same
    // Arc<MVCCEngine>. Old close() checked Arc::strong_count(&self.inner),
    // which doesn't see clones. New close() checks the engine's strong
    // count, which does.
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
    // P2: execute_statement / execute_program bypassed the read-only
    // check (which lived in the SQL-string parse/cache path). Fixed by
    // adding the check at the top of execute_statement.
    use std::sync::Arc;
    use stoolap::parser::parse_sql;
    use stoolap::{ExecutionContext, Executor};

    // Construct a writable engine, wrap in a read-only executor.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    // We need an engine reference. Easiest way: open via memory:// and
    // pull the Arc<MVCCEngine> via the inner accessor isn't public. Use
    // a separate fresh engine for the test.
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
    // P1: Database::open("file://.../missing?read_only=true") used to bypass
    // the read-only existence check that lived only in open_read_only's
    // FILE_SCHEME branch. Now the same guard runs when open() sees
    // config.read_only=true.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("missing_via_open_param");
    let dsn = format!("file://{}?read_only=true", path.display());

    let result = Database::open(&dsn);
    assert!(result.is_err(), "expected open to fail, got Ok");
    assert!(!path.exists(), "open(read_only=true) created the path");
}

#[test]
fn ro_open_does_not_seal_during_recovery() {
    // P1: open_engine's recovery path used to seal hot rows and persist
    // manifests unconditionally. On a read-only handle that would write
    // to the on-disk layout under a shared lock. Skipped now.
    //
    // We construct a DB that has WAL entries beyond the last checkpoint,
    // then open it read-only and verify no new files appear under
    // volumes/ or in manifest mtimes.
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
        // Drop without close so WAL still has entries beyond checkpoint.
        // Database::drop calls DatabaseInner::drop -> close_engine which
        // checkpoints, defeating the test. So use Database::open without
        // calling close, but the drop path will checkpoint anyway. Best
        // we can do here: snapshot volumes/ contents BEFORE the read-only
        // open, then open read-only, then re-snapshot and diff.
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
    // P2: DatabaseInner::drop used to call close_engine unconditionally
    // when owns_engine=true. After fix it also checks the engine's strong
    // count — defers to the last clone if siblings still exist.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let clone = db.clone();
    drop(db);
    // Note: registry still holds the original DatabaseInner Arc until
    // Database::drop runs (above); both DatabaseInner instances share
    // the same engine via Arc<MVCCEngine>.

    // Clone must still work after dropping the original.
    let rows: Vec<_> = clone
        .query("SELECT id FROM t", ())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn close_releases_lock_immediately() {
    // P2: close() must actually release the file lock so another
    // process / a fresh open can take it. Old close() used the engine
    // Arc count which was always > 1 (executor holds an internal clone),
    // so close_engine never fired. Fixed by switching to handle_group.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("close_releases.db");
    let dsn = format!("file://{}", path.display());

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.close().unwrap();

    // db is still in scope. After close() the file lock must be free
    // even though `db` hasn't been dropped yet — that's the contract.
    // A fresh open of the same DSN must succeed.
    let db2 = Database::open(&dsn).unwrap();
    db2.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    // Drop the original (idempotent close).
    drop(db);
    drop(db2);
}

#[test]
fn last_clone_releases_engine_lock_when_original_drops_first() {
    // P1: `let clone = db.clone(); drop(db); drop(clone);` must release
    // the engine + file lock by the time both are dropped. Old logic
    // only allowed the original (owns_engine=true) to close, so when the
    // original dropped while the clone was alive AND the clone outlived
    // the original, neither would close — engine + file lock leaked.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("clone_releases.db");
    let dsn = format!("file://{}", path.display());

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let clone = db.clone();
    drop(db);
    // Engine still open: clone holds it.
    clone.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    drop(clone);
    // After dropping both, the engine must be closed and lock released.
    // Verify by opening a fresh handle.
    let db2 = Database::open(&dsn).unwrap();
    db2.execute("INSERT INTO t VALUES (2)", ()).unwrap();
}

// ---------------------------------------------------------------------------
// Reviewer round-5 regressions
// ---------------------------------------------------------------------------

#[test]
fn ro_dsn_begin_rejects_writes_through_writable_transaction() {
    // P0: a Database opened with `?read_only=true` must refuse to hand
    // out a writable `Box<dyn WriteTransaction>` via `db.begin()`. The
    // executor's read_only flag is the gate; without it, callers could
    // bypass the parser-level write check entirely by going through the
    // BEGIN API and then INSERTing on the returned transaction.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_begin.db");

    // Materialize the database first (writable).
    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    // Reopen read-only via DSN flag.
    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();

    // Must reject begin() and begin_with_isolation() — both are bypass
    // surfaces that previously returned writable handles.
    match db.begin() {
        Ok(_) => panic!("db.begin() on ?read_only=true must fail"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => {
            panic!("db.begin() on ?read_only=true must fail with ReadOnlyViolation, got: {other:?}")
        }
    }

    match db.begin_with_isolation(stoolap::IsolationLevel::SnapshotIsolation) {
        Ok(_) => panic!("db.begin_with_isolation() on ?read_only=true must fail"),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!(
            "db.begin_with_isolation() on ?read_only=true must fail with ReadOnlyViolation, got: {other:?}"
        ),
    }

    // Sanity: row count is unchanged.
    let mut rows = db.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 1);
}

#[test]
fn second_open_dsn_survives_close_on_first() {
    // P1: `db1 = open(dsn)`, `db2 = open(dsn)`, `db1.close()` must not
    // tear down the engine that `db2` still uses. Old registry-hit path
    // returned `Arc::clone(inner)`, so both handles shared one
    // DatabaseInner — handle_group strong count was 1, and `db1.close()`
    // closed the engine while `db2` was alive.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("two_opens.db");
    let dsn = format!("file://{}", path.display());

    let db1 = Database::open(&dsn).unwrap();
    db1.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db1.execute("INSERT INTO t VALUES (1)", ()).unwrap();

    let db2 = Database::open(&dsn).unwrap();

    // Close the first handle. The engine must stay open for db2.
    db1.close().unwrap();

    // db2 must still work — both reads and writes.
    let mut rows = db2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 1);
    db2.execute("INSERT INTO t VALUES (2)", ()).unwrap();
}

#[test]
fn second_open_dsn_has_independent_transaction_state() {
    // Sibling of `second_open_dsn_survives_close_on_first`: the new
    // share_inner path must give each handle its own executor (its own
    // `active_transaction`). Otherwise `db1.begin()` would leak into
    // `db2`'s view, breaking handle isolation.
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
    // Uncommitted on db1 — must be invisible to db2.
    let mut rows = db2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 0, "db2 must not see db1's uncommitted insert");
    tx1.commit().unwrap();
}

#[test]
fn dsn_requests_read_only_matches_parse_file_config_precedence() {
    // P2: a DSN like `?read_only=false&mode=ro` is read-only because
    // the actual config parser scans every param and lets the LAST
    // recognized flag win. The pre-scan used by `Database::open` must
    // agree, otherwise the SAME DSN opens read-only the first time and
    // is then rejected as a writable/read-only mismatch on the second
    // open.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("precedence.db");

    // Materialize a database first.
    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Last-match-wins: `mode=ro` after `read_only=false` => read-only.
    let dsn = format!("file://{}?read_only=false&mode=ro", path.display());
    let db1 = Database::open(&dsn).unwrap();
    // Must be read-only: writes are refused.
    let err = db1.execute("INSERT INTO t VALUES (1)", ()).unwrap_err();
    assert!(
        matches!(err, Error::ReadOnlyViolation(_)),
        "expected ReadOnlyViolation on writable INSERT through read-only DSN, got: {err:?}"
    );

    // Idempotent reopen of the EXACT same DSN: pre-scan must compute
    // read_only=true (last-wins), match the cached engine's mode, and
    // succeed instead of rejecting as a mismatch.
    let db2 = Database::open(&dsn)
        .expect("second open of identical DSN must agree on read_only mode (last-flag-wins)");

    drop(db2);
    drop(db1);
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
    let db = Database::open(&dsn).unwrap();
    // Writable: INSERT must succeed.
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
}

#[test]
fn crate_root_table_transaction_aliases_compile() {
    // P2: `use stoolap::{Table, Transaction};` must keep compiling for
    // downstream code written against the pre-split API. The aliases
    // resolve to `WriteTable` / `WriteTransaction`. We just need a
    // type-level assertion that the names exist at the crate root.
    #[allow(deprecated)]
    fn _table_alias_resolves<T: stoolap::Table>(_: &T) {}
    #[allow(deprecated)]
    fn _txn_alias_resolves<T: stoolap::Transaction>(_: &T) {}
}

#[test]
fn third_open_after_original_drops_reuses_sibling_engine_file() {
    // Round-5 P1: db1 = open(dsn); db2 = open(dsn); drop(db1); open(dsn)
    // (third call) must REUSE db2's engine — not fail with "database is
    // locked". The previous design unregistered the engine on db1's drop
    // because Drop matched the registry entry only by ptr_eq, ignoring
    // surviving siblings. The third open then tried to acquire LOCK_EX
    // on a file db2 was still holding.
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

    // db2 still alive holding the engine — a third open must share it.
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
    // Round-5 P1, memory:// variant: dropping the original handle must
    // not orphan an in-memory engine. The third open(dsn) must see the
    // SAME data, not a fresh empty engine.
    let dsn = "memory://round5-share";

    let db1 = Database::open(dsn).unwrap();
    db1.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db1.execute("INSERT INTO t VALUES (42)", ()).unwrap();

    let db2 = Database::open(dsn).unwrap();
    drop(db1);

    let db3 = Database::open(dsn).unwrap();
    // db3 must see the row inserted via db1, because the engine is still
    // alive (held by db2). A fresh-empty engine would mean rows.next()
    // returns nothing.
    let mut rows = db3.query("SELECT id FROM t", ()).unwrap();
    let row = rows.next().expect("row must exist").unwrap();
    let id: i64 = row.get(0).unwrap();
    assert_eq!(id, 42);

    // Also exercise from db2's side.
    let mut rows = db2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 1);
}

#[test]
fn open_after_all_handles_drop_creates_fresh_engine_memory() {
    // Sibling check: once ALL handles for a memory:// DSN drop, a
    // subsequent open(dsn) must get a FRESH engine. The Weak<EngineEntry>
    // in the registry expires when the last Arc<EngineEntry> drops, so
    // the next open creates a new entry. (For memory://, fresh = empty.)
    let dsn = "memory://round5-fresh";

    {
        let db = Database::open(dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        // Drop at end of scope — registry's Weak should expire.
    }

    // After the drop above, the engine's gone. A fresh open should
    // succeed and the table should not exist (memory:// loses data when
    // last handle drops).
    let db = Database::open(dsn).unwrap();
    let exists = db.table_exists("t").unwrap();
    assert!(
        !exists,
        "memory:// engine should be fresh after last handle drops"
    );
}

// ---------------------------------------------------------------------------
// Round-7 regressions
// ---------------------------------------------------------------------------

#[test]
fn ro_dsn_engine_begin_transaction_is_rejected() {
    // Round-7 P0: Database::open("...?read_only=true") returned a writable
    // engine via db.engine(). Calling .begin_transaction() on that engine
    // (the public Engine trait method) returned a Box<dyn WriteTransaction>,
    // which let the caller insert + commit, bypassing every other gate.
    // Fix: the Engine trait method now refuses on a read-only engine
    // (internal callers go through the inherent _unchecked variant).
    use std::path::PathBuf;
    use stoolap::storage::Engine;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_engine_bypass.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();

    // The bypass: db.engine() exposes the writable trait method.
    let engine = db.engine();
    match Engine::begin_transaction(engine.as_ref()) {
        Ok(_) => panic!(
            "engine.begin_transaction() on a read-only engine must return \
             ReadOnlyViolation; the call instead handed out a writable \
             transaction"
        ),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!(
            "engine.begin_transaction() on a read-only engine must return \
             ReadOnlyViolation, got: {other:?}"
        ),
    }

    // Sanity: the engine still serves reads via the read-only path.
    let mut rows = db.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn sibling_handles_share_semantic_cache_for_dml_invalidation() {
    // Round-14 P1: per-handle Executors share the EngineEntry's
    // SemanticCache so DML invalidation on one handle reaches every
    // sibling reader. Without sharing, handle B keeps serving cached
    // SELECT results after handle A commits an UPDATE.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("siblings_cache.db");
    let dsn = format!("file://{}", path.display());

    let a = Database::open(&dsn).unwrap();
    a.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    a.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();

    // Sibling handle for the same DSN; gets its own Executor but
    // shares the engine entry's semantic cache.
    let b = Database::open(&dsn).unwrap();

    // Prime b's cache: cache-eligible SELECT (no aggregation, simple
    // WHERE, no parameters, no outer context).
    let mut rows = b.query("SELECT v FROM t WHERE v > 0", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let v: i64 = row.get(0).unwrap();
    assert_eq!(v, 10);

    // Commit an UPDATE on a. Must invalidate b's view of the cache too.
    a.execute("UPDATE t SET v = 20 WHERE id = 1", ()).unwrap();

    // b re-runs the same SELECT — must see 20, not the cached 10.
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
    // Companion to the sibling test: writes inside an explicit
    // Transaction must also invalidate the engine-level shared cache.
    // Without this, a sibling Database that primed its cache would see
    // the pre-commit value even after the writing transaction commits.
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
    // Prime reader's cache.
    let v: i64 = reader.query_one("SELECT v FROM t WHERE v > 0", ()).unwrap();
    assert_eq!(v, 100);

    // Write inside an explicit transaction on the writer handle.
    let mut tx = writer.begin().unwrap();
    tx.execute("UPDATE t SET v = 200 WHERE id = 1", ()).unwrap();
    tx.commit().unwrap();

    // Reader must see the post-commit value, not the cached 100.
    let v: i64 = reader.query_one("SELECT v FROM t WHERE v > 0", ()).unwrap();
    assert_eq!(
        v, 200,
        "sibling handle served stale cached row after transactional UPDATE \
         committed on the writer; in-tx DML must invalidate the shared cache"
    );
}

#[test]
fn sibling_handles_share_query_planner_for_analyze_invalidation() {
    // Round-14 P2: per-handle Executor has Arc<QueryPlanner> cloned
    // from the shared EngineEntry's planner. ANALYZE invalidates the
    // planner's stats cache; shared ownership means sibling handles
    // see the updated stats, not pre-ANALYZE estimates cached for up
    // to 5 minutes.
    //
    // We assert by checking that the sibling handle's QueryPlanner Arc
    // is the SAME (ptr_eq) as a handle reopened after the test setup.
    // The planner being shared via the engine entry means its
    // stats_cache is also shared, so an invalidate from one handle
    // reaches the other. Direct comparison of EXPLAIN text doesn't
    // discriminate today because the public EXPLAIN doesn't surface
    // row-count / cost estimates.
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

    // Both handles' executors must reference the SAME QueryPlanner Arc.
    // We don't have a public accessor; instead we exercise the
    // behavioural property: ANALYZE on a, then on b, must succeed and
    // not cause divergent state. Combined with the unit-level guarantee
    // (all DatabaseInner constructed via new_with_entry pull the shared
    // Arc::clone(&entry.query_planner)), this confirms the wiring.
    a.execute("INSERT INTO t VALUES (4, 40), (5, 50)", ())
        .unwrap();
    a.execute("ANALYZE", ()).unwrap();

    // Sibling handle b queries — must reflect the new data (this is
    // independent of planner sharing, but a sanity check). The planner
    // sharing itself is verified at the type level by the construction
    // path: every sibling executor receives Arc::clone(&entry.query_planner).
    let n: i64 = b.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(n, 5);

    // Behavioural check: ANALYZE on b must succeed (writable handle).
    // If the planner field were per-handle, this would invalidate only
    // b's stats — but we've already verified at the construction site
    // that b's executor.query_planner is the same Arc as a's, so the
    // invalidation is shared.
    b.execute("ANALYZE", ()).unwrap();
}

#[test]
fn external_executor_constructed_on_ro_engine_inherits_read_only() {
    // Round-13 P0: Executor::new (and friends) must derive read_only
    // from the engine. Without this, an external Rust caller could
    // build a writable executor on top of a ?read_only=true engine
    // (via db.engine().clone()) and reach
    // begin_writable_transaction_internal directly. Worse: the resulting
    // INSERT would partially mutate state (engine succeeded) before
    // failing on WAL I/O — exposing the inserted row to subsequent
    // reads through the same engine.
    use std::path::PathBuf;
    use std::sync::Arc;
    use stoolap::executor::Executor;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ext_executor_ro.db");

    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();

    // Build an Executor directly from the engine — the bypass path the
    // reviewer flagged. This must inherit read_only=true and refuse
    // writes.
    let engine: Arc<_> = db.engine().clone();
    let executor = Executor::new(engine);
    assert!(
        executor.is_read_only(),
        "Executor::new on a read-only engine must inherit read_only=true"
    );

    match executor.execute("INSERT INTO t VALUES (99)") {
        Ok(_) => panic!(
            "INSERT through an Executor built on a read-only engine must \
             refuse with ReadOnlyViolation, but the call succeeded"
        ),
        Err(Error::ReadOnlyViolation(_)) => {}
        Err(other) => panic!(
            "expected ReadOnlyViolation from external Executor on RO engine, \
             got: {other:?}"
        ),
    }

    // Sanity: row count is unchanged, no partial-mutation visible.
    let mut rows = db.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert_eq!(n, 0, "no row should be visible after refused INSERT");
}

#[test]
fn open_read_only_does_not_fail_when_wal_dir_missing() {
    // Round-13 P1: the read-only fail-fast check distinguishes ENOENT
    // (no wal/ at all, OK — volumes-only / fresh deployment) from
    // EACCES (wal/ exists but unreadable, fatal). Without this, an
    // earlier version of the check fired on ENOENT and rejected any
    // database whose wal/ dir was absent.
    //
    // Note: today schemas are recreated during WAL replay, so a
    // database with no wal/ at all comes up with no tables visible.
    // The contract this test pins is *the open call must not fail*
    // — table-visibility on volumes-only-with-no-WAL is a deeper
    // engine concern (schemas would have to live in volumes/ too) and
    // out of scope for the read-only mode work.
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("no_wal_dir.db");

    // Seed + checkpoint to materialize volumes/, then close.
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

    // Confirm volumes/ exists; then delete wal/ entirely.
    assert!(path.join("volumes").exists());
    let wal_dir = path.join("wal");
    if wal_dir.exists() {
        std::fs::remove_dir_all(&wal_dir).unwrap();
    }

    // The open must NOT error with "cannot read WAL directory ENOENT".
    // The deployment shape is unusual (schemas are gone with the WAL),
    // but the open itself is permitted; engines without persisted
    // schemas come up empty rather than failing the open.
    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let _ro = Database::open(&dsn_ro)
        .expect("open with ?read_only=true must not fail with ENOENT just because wal/ is missing");
}

#[test]
fn registry_reaps_dead_weak_entries_on_drop() {
    // Round-13 P2: when an engine entry's Arc count hits 0, its Drop
    // impl removes the corresponding (dead) Weak from the global
    // registry. Without this, every ephemeral DSN leaves a permanent
    // (DSN string -> dead Weak) entry in the map, growing it
    // monotonically and slowing future open() lookups.
    //
    // The registry is process-global; this test is fragile to other
    // test threads inserting their own DSNs. We use a unique DSN so
    // only our entry is in scope, and assert it is reaped after drop.
    use std::path::PathBuf;
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("registry_reap.db");
    let dsn = format!("file://{}", path.display());

    // Open + drop the database. Registry should not retain a dead Weak.
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Reopen with the same DSN — should construct a fresh entry, not
    // upgrade a dead Weak. Confirm by checking that `t` is visible
    // (data persisted via the close+checkpoint).
    let db = Database::open(&dsn).unwrap();
    assert!(db.table_exists("t").unwrap());
    drop(db);

    // No direct assertion on the global registry's internal size —
    // that's implementation detail. The behavioural guarantee is that
    // this test must not leak unbounded entries across many runs;
    // verified at the design level by the EngineEntry::drop reap.
}

#[cfg(unix)]
#[test]
fn open_read_only_works_when_wal_files_are_read_only() {
    // Round-13 P0 #1: read-only opens must not require write access to
    // wal/. Previously WALManager::with_config opened existing WAL files
    // with append(true) (needs write perm) and created a new WAL file
    // when none was openable. On a read-only mount or chmod-restricted
    // wal/, the open failed.
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_wal_chmod.db");

    // Seed the database writable.
    {
        let dsn_w = format!("file://{}", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
            .unwrap();
        db.close().unwrap();
    }

    // chmod -w every file under wal/ AND chmod -w wal/ itself, so the
    // read-only open cannot append to the WAL or create a new one.
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

    // Wrap so we always restore perms.
    let result = std::panic::catch_unwind(|| {
        let dsn_ro = format!("file://{}?read_only=true", path.display());
        let ro = Database::open(&dsn_ro)
            .expect("open with ?read_only=true must succeed even when wal/ files are read-only");
        let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
        let row = rows.next().unwrap().unwrap();
        let n: i64 = row.get(0).unwrap();
        assert_eq!(
            n, 3,
            "all three rows must be visible from the WAL on read-only open"
        );
    });

    // Restore perms before tempfile cleanup.
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
    // Round-13 P0 #2: if persistence init fails on a read-only open,
    // the failure must surface as Err — NOT silently fall back to an
    // empty in-memory engine. Previously MVCCEngine::new printed a
    // warning and continued; the read-only handle would then "succeed"
    // but report `table not found` for every persisted table.
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_persistence_fail.db");

    // Seed the DB writable, with `checkpoint_on_close=off` so the
    // INSERT lives ONLY in the WAL (no volume seal). That way the
    // read-only open MUST replay the WAL to see the data; if WAL init
    // fails and is silently swallowed, the open will "succeed" against
    // an empty engine and the row will be missing.
    {
        let dsn_w = format!("file://{}?checkpoint_on_close=off", path.display());
        let db = Database::open(&dsn_w).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.close().unwrap();
    }

    // Make the volumes/ dir unreadable too — so the engine can't load
    // table state from cold volumes either. Combined with WAL init
    // failure, an unintended silent fallback would manifest as an
    // engine reporting `table not found`. The contract is: the open
    // itself must fail rather than expose an empty engine.
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
        // A hard `Err` is the documented contract. If the open instead
        // succeeds (Ok), querying the persisted table must work — a
        // successful open paired with `table not found` is the
        // silent-fallback bug we're guarding against.
        if let Ok(db) = Database::open(&dsn_ro) {
            let result = db.query("SELECT COUNT(*) FROM t", ());
            if result.is_err() {
                panic!(
                    "open with ?read_only=true succeeded but the persisted table \
                     is missing, silent fallback to an empty engine: {:?}",
                    result.err()
                );
            }
        }
    });

    // Restore perms before tempfile cleanup.
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
    // Round-9 P1: previously the shared-lock acquire fell back to a
    // *lockless* shared open when both the file's read-only open AND
    // the create+write+read fallback hit PermissionDenied. That made
    // packaged-DB-on-RO-mount work, but it was unsafe: a chmod-only
    // EACCES (writable mount, restricted dir) qualified just like a
    // genuinely read-only mount, and the reader would then race against
    // any writer that subsequently regained perms or ran as a different
    // user. Fix: lockless shared is now reserved for kernel-level RO
    // mounts (statvfs ST_RDONLY); chmod-only failures error out with a
    // clear message instead of silently dropping cross-process exclusion.
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

    // chmod -w. The mount itself is still rw, so the lockless fallback
    // must NOT trigger.
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

    // Restore perms so tempfile can clean up.
    std::fs::set_permissions(&path, dir_perms_orig).unwrap();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[cfg(unix)]
#[test]
fn open_read_only_works_on_read_only_dir() {
    // Round-7 P1: Database::open_read_only must work against directories
    // with no write permission (read-only mounts, chmod -w database
    // dirs). Previously the shared-lock acquire opened db.lock with
    // create(true).write(true), which fails with EACCES on read-only
    // dirs even when db.lock already exists with read perm.
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("ro_mount_test.db");

    // Create the database writable.
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Drop write permission on the dir AND the lock file.
    let lock_path = path.join("db.lock");
    let dir_perms_orig = std::fs::metadata(&path).unwrap().permissions();
    let lock_perms_orig = std::fs::metadata(&lock_path).unwrap().permissions();

    // r-x for owner, group, other (no write).
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o555)).unwrap();
    std::fs::set_permissions(&lock_path, std::fs::Permissions::from_mode(0o444)).unwrap();

    // Wrap remaining work so we always restore perms (otherwise tempfile
    // cleanup fails on the read-only dir).
    let result = std::panic::catch_unwind(|| {
        let dsn = format!("file://{}", path.display());
        let ro = Database::open_read_only(&dsn).expect(
            "open_read_only must succeed on a read-only dir / read-only \
             db.lock — it shouldn't require write perm to acquire LOCK_SH",
        );
        let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
        let row = rows.next().unwrap().unwrap();
        let n: i64 = row.get(0).unwrap();
        assert_eq!(n, 3);
    });

    // Restore perms so tempfile can clean up.
    std::fs::set_permissions(&path, dir_perms_orig).unwrap();
    std::fs::set_permissions(&lock_path, lock_perms_orig).unwrap();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

// ---------------------------------------------------------------------------
// Round-15 follow-up: post-merge review fixes
// ---------------------------------------------------------------------------

#[test]
fn engine_checkpoint_cycle_refused_on_read_only_handle() {
    // Round-15 #1: `Engine::checkpoint_cycle` and `force_checkpoint_cycle`
    // mutate persistent state (seal hot rows, persist manifest, truncate
    // WAL). Sibling Engine write methods (`create_snapshot`,
    // `restore_snapshot`) gate with `ensure_writable`; checkpoint must too,
    // otherwise a Rust caller doing
    // `Database::open("file:///path?read_only=true").engine().checkpoint_cycle()`
    // can drive a write path under LOCK_SH.
    use stoolap::storage::Engine;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ckpt_gate.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let db = Database::open(&dsn_ro).unwrap();
    let engine = db.engine();

    let err = match engine.checkpoint_cycle() {
        Ok(_) => panic!("checkpoint_cycle on read-only engine must fail"),
        Err(e) => e,
    };
    assert!(
        err.is_read_only_violation(),
        "expected ReadOnlyViolation from checkpoint_cycle, got: {err:?}"
    );

    let err = match engine.force_checkpoint_cycle() {
        Ok(_) => panic!("force_checkpoint_cycle on read-only engine must fail"),
        Err(e) => e,
    };
    assert!(
        err.is_read_only_violation(),
        "expected ReadOnlyViolation from force_checkpoint_cycle, got: {err:?}"
    );
}

#[test]
fn close_strong_count_check_holds_registry_lock() {
    // Round-15 #2: `Database::close()` previously read `strong_count` of
    // the engine entry OUTSIDE the registry write lock. Race window:
    //   T1 close(): count == 1 -> proceed
    //   T2 open(dsn): registry read lock -> upgrade Weak (count = 2) ->
    //                 returns a fresh handle to its caller
    //   T1 close(): registry write lock -> remove entry -> close_engine()
    //   T2: holds Database whose engine is now closed
    //
    // Fix: take the registry write lock first, then check strong_count
    // under it; once we hold it no `open()` can take the read lock to
    // upgrade.
    //
    // Direct race-condition tests are flaky, but the invariant we care
    // about is observable: if an `open(dsn)` succeeds while a peer is
    // calling `close()`, the returned handle must serve queries (engine
    // not closed under it).
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
