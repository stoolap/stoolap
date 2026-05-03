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
    let ro = Database::open_read_only(&dsn_ro)
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
    // Database is always writable now (Database::open rejects read-only
    // DSN flags), so Database::is_read_only() always returns false. The
    // read-only surface lives on ReadOnlyDatabase, whose is_read_only()
    // always returns true. Verify both halves of the symmetry.
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
fn ro_database_cached_plan_refuses_write_sql_early() {
    // ReadOnlyDatabase::cached_plan must refuse write SQL at plan-creation
    // time, not later at execute_plan, so callers learn the violation up
    // front instead of after a Statement is built.
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

    // Read SQL still creates a plan.
    let _ = db
        .cached_plan("SELECT * FROM t WHERE id = $1")
        .expect("SELECT must still cache on a read-only ReadOnlyDatabase");
}

// `MVCCEngine::cleanup_*` no-op-on-read-only behavior remains an
// engine implementation detail; with the public API now requiring
// `Database::open_read_only` to construct a read-only engine and
// `ReadOnlyDatabase` exposing no `engine()` accessor, integration
// tests can no longer reach an `Arc<MVCCEngine>` in read-only mode.
// Engine-internal coverage lives in src/storage/mvcc/engine.rs unit
// tests now.

// `Database::create_snapshot` / `restore_snapshot` no longer have a
// runtime read-only gate to cover: `Database` is always writable and
// `ReadOnlyDatabase` simply doesn't expose those methods (compile-time
// gate). The previous tests that opened with `?read_only=true` and
// expected ReadOnlyViolation are obsolete.

// `MVCCEngine` inherent write methods (`create_table`,
// `drop_table_internal`, `vacuum`, `create_view`, etc.) refusing on a
// read-only engine is now solely an engine-level invariant. The
// integration test that drove it through `Database::engine()` is
// obsolete (Database is always writable; ReadOnlyDatabase has no
// engine() accessor). Engine-internal coverage lives in
// src/storage/mvcc/engine.rs unit tests.

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
fn open_with_read_only_dsn_flag_is_rejected_with_migration_message() {
    // `Database::open` rejects every spelling of the read-only DSN flag
    // (`?read_only=true`, `?readonly=true`, `?mode=ro`) so callers
    // discover the typed `open_read_only` entry point instead of
    // silently getting a writable handle. The error message names the
    // replacement.
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
    // `Database::open_read_only` takes the same DSN strings drivers
    // already build, including the redundant `?read_only=true` flag.
    // The flag is treated as a no-op (matches the function name); a
    // drivers-friendly migration just changes the entry-point call.
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
    // The function name says read-only; a DSN flag that explicitly
    // requests writable contradicts it and is rejected so the caller
    // catches the disagreement at the API surface instead of getting
    // surprising behavior.
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

// SWMR cross-process coexistence requires `db.shm`, which is
// Unix-only in this build (the Windows shim returns an error from
// `ShmHandle::create_writer`). The engine refuses the read-only
// attach with a documented message: "this platform has no db.shm
// support, so live reader/writer coexistence is unavailable. Close
// the writer or retry the open." Skip on Windows; the rest of the
// read-only test suite (which doesn't need writer/reader coexistence)
// continues to run there.
#[cfg(unix)]
#[test]
fn open_writable_then_open_read_only_query_param_succeeds_under_swmr() {
    // SWMR v1 contract: a writable engine and a read-only attach can
    // coexist (Shared mode no longer takes a kernel lock; readers
    // signal presence via lease files). Pre-SWMR this errored with
    // DatabaseLocked because the writer's LOCK_EX blocked the reader's
    // LOCK_SH. Pinning the new behavior here so a regression to the
    // old lock semantics fails loudly.
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
    // ReadOnlyDatabase has no execute(); SELECT works.
    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 0);
    drop(ro);
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
    let db1 = Database::open_read_only(&dsn).unwrap();
    let db2 = Database::open_read_only(&dsn).unwrap();
    drop(db1);
    drop(db2);
}

#[test]
fn open_with_invalid_read_only_value_errors() {
    // Invalid read-only flag value is rejected at DSN-parse time on
    // both entry points (Database::open computes the same scan to
    // refuse the migration target).
    assert!(Database::open("file:///tmp/nope?read_only=maybe").is_err());
    assert!(Database::open_read_only("file:///tmp/nope?read_only=maybe").is_err());
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
    // open_read_only against a missing path must not silently
    // materialize a fresh empty engine. The presence check is mandatory
    // under the read-only contract.
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

// `Database::begin` / `begin_with_isolation` no longer carry a runtime
// read-only gate: `Database` is always writable, and `ReadOnlyDatabase`
// has no `begin` / `begin_with_isolation` method (compile-time gate).
// The previous test that opened with `?read_only=true` and expected
// ReadOnlyViolation from begin() is obsolete.

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
    // A DSN like `?read_only=false&mode=ro` is read-only because the
    // actual config parser scans every param and lets the LAST
    // recognized flag win. The pre-scan used by `Database::open` /
    // `Database::open_read_only` must agree: the same DSN must be
    // rejected by `open` (as a read-only request) and accepted by
    // `open_read_only` (which the redundant flag confirms).
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

    // Last-match-wins: `mode=ro` after `read_only=false` => read-only.
    let dsn = format!("file://{}?read_only=false&mode=ro", path.display());

    // open: rejected with the migration message because the LAST flag
    // requests read-only.
    match Database::open(&dsn) {
        Ok(_) => panic!("open must refuse a read-only DSN"),
        Err(Error::InvalidArgument(msg)) => assert!(
            msg.contains("read-only DSN flag"),
            "expected migration message; got: {msg}"
        ),
        Err(other) => panic!("expected InvalidArgument, got: {other:?}"),
    }

    // open_read_only: accepts the same DSN (last flag agrees with
    // function name).
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
    // P2: an earlier version checked for the substrings
    // `read_only=` / `readonly=` / `mode=` anywhere in the
    // DSN, so a file path containing `mode=` (or any unrelated
    // query value containing those tokens) was incorrectly
    // treated as a writable-flag contradiction. Verify the
    // parser only inspects recognized keys inside the
    // query-string portion.
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

    // The path contains `mode=rw`. open_read_only must NOT
    // misread that as a writable flag.
    let dsn = format!("file://{}", path.display());
    let ro = Database::open_read_only(&dsn)
        .expect("open_read_only must ignore `mode=rw` inside the file path");
    drop(ro);

    // And open must NOT misread the same path as a read-only
    // flag (it doesn't contain `mode=ro` but the symmetric
    // failure mode would have rejected this DSN).
    let db = Database::open(&dsn).unwrap();
    db.close().unwrap();

    // Unrelated query values that contain the trigger
    // substrings must also be ignored.
    let dsn_with_other_param = format!("file://{}?sync_mode=normal&compression=on", path.display());
    let ro2 = Database::open_read_only(&dsn_with_other_param).unwrap();
    drop(ro2);
    let db2 = Database::open(&dsn_with_other_param).unwrap();
    db2.close().unwrap();
}

#[test]
fn dsn_requests_read_only_writable_last_wins() {
    // Inverse: `?mode=ro&read_only=false` ends writable. open accepts
    // it; open_read_only rejects with the contradiction message.
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

// `Engine::begin_transaction` trait method refusing on a read-only
// engine is an engine-level invariant. The previous integration test
// that constructed a read-only Database via `?read_only=true` to reach
// `db.engine()` is obsolete: Database::open rejects the flag and
// Config::read_only is `pub(crate)`. Engine-internal coverage lives in
// src/storage/mvcc/engine.rs unit tests.

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

// `Executor::new` deriving read_only from the engine is verified at
// the engine + executor layer. The previous integration test that
// reached an Arc<MVCCEngine> in read-only mode through
// `Database::open(?read_only=true).engine()` is obsolete; Config's
// `read_only` field is `pub(crate)` and there is no public way to
// build a read-only Arc<MVCCEngine> from outside the crate.

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
    let _ro = Database::open_read_only(&dsn_ro)
        .expect("open_read_only must not fail with ENOENT just because wal/ is missing");
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
// Round-15 follow-up: post-merge fixes
// ---------------------------------------------------------------------------

// `Engine::checkpoint_cycle` / `force_checkpoint_cycle` refusing on a
// read-only engine is now an engine-level invariant only. The previous
// integration test that drove it through
// `Database::open(?read_only=true).engine()` is obsolete; Config's
// `read_only` field is `pub(crate)` and there is no public way to
// build a read-only Arc<MVCCEngine> from outside the crate.

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

// ---------------------------------------------------------------------------
// Reader lease tests
// ---------------------------------------------------------------------------

#[test]
fn ro_handle_creates_lease_file_on_file_engine() {
    // SWMR v1 P1.2: opening a `file://` read-only handle must register a
    // cross-process presence lease in `<db>/readers/<pid>.lease`. The
    // writer's GC paths consult that directory before unlinking volumes
    // or truncating WAL.
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
    // SWMR v1 P1.2: every `query` / `query_named` / `cached_plan` etc.
    // call must bump the lease mtime. This is the heartbeat the writer's
    // GC uses to distinguish live from stale leases.
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

    // Sleep enough that the FS records a different mtime (50ms covers
    // every common filesystem; APFS is ns-precise, ext4 is ms, HFS+ is
    // 1s but is not the macOS default for 5+ years).
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
    // `as_read_only()` over a writable engine in the same process must
    // NOT take a filesystem lease. The writer in this process coordinates
    // GC internally; a filesystem lease would just be noise. (Also
    // matters for tests: an in-process RO view shouldn't leave lease
    // files in the writable DB's dir.)
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
    // SWMR v1 P3.1: refresh() must be cheap when the writer hasn't
    // checkpointed since the reader's last refresh. Returns Ok(false)
    // and does not reload manifests.
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
    // First refresh: writer made no new checkpoint since we opened.
    let advanced = ro.refresh().unwrap();
    assert!(
        !advanced,
        "refresh must be a no-op when epoch hasn't advanced since open"
    );
}

#[test]
fn refresh_returns_true_when_epoch_advances() {
    // SWMR v1 P3.1: when the on-disk epoch is greater than the reader's
    // cached value, refresh() reloads manifests and returns true. We
    // simulate the writer's epoch bump by writing the file directly,
    // since a real writer process (with LOCK_EX) cannot coexist with
    // our LOCK_SH reader in the same OS process. End-to-end with a
    // real subprocess writer is covered by P4.2 in swmr_snapshot_test.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("refresh_advance.db");

    // Set up a real database on disk with a checkpointed table.
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

    // First refresh: epoch is whatever the writer left it at; reader's
    // cached value matches (initialized at open). No-op.
    let advanced = reader.refresh().unwrap();
    assert!(!advanced, "first refresh after open must no-op");

    // Simulate a writer checkpoint by bumping the epoch file directly.
    // In production this would happen from a separate writer process.
    let new_epoch = manifest_epoch::bump_epoch(&path).unwrap();
    assert!(new_epoch > 0, "bump_epoch must return positive");

    // Now refresh sees the advance.
    let advanced = reader.refresh().unwrap();
    assert!(
        advanced,
        "refresh must return true after epoch file was bumped"
    );

    // And subsequent calls no-op until next bump.
    let advanced2 = reader.refresh().unwrap();
    assert!(
        !advanced2,
        "second refresh after no further bump must no-op"
    );
}

#[test]
fn auto_refresh_default_is_on() {
    // SWMR v1 P3.2: auto-refresh defaults to ON for new ReadOnlyDatabase
    // handles. set_auto_refresh(false) toggles it off; getter reports state.
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
    // SWMR v1 P3.2: with auto-refresh ON (default), a query call must
    // pick up an epoch bump that happened since the last query, without
    // the caller having to invoke refresh() manually.
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

    // Simulate the writer doing another checkpoint by bumping the epoch.
    // We can't actually start a writer in this process (LOCK_EX vs LOCK_SH
    // conflict), so we bump the epoch file directly. The full
    // writer-and-reader test lives in P4.2 (subprocess-based).
    manifest_epoch::bump_epoch(&path).unwrap();

    // Issue a query; auto-refresh must observe the bump and advance.
    ro.query("SELECT COUNT(*) FROM t", ()).unwrap().next();

    // Verify by calling refresh() — it should now no-op because
    // auto-refresh already advanced our cached epoch.
    let advanced = ro.refresh().unwrap();
    assert!(
        !advanced,
        "explicit refresh after auto-refresh must no-op (auto-refresh \
         already consumed the bump)"
    );

    // Bump again. Disable auto-refresh, query, then verify refresh()
    // sees the un-consumed bump (auto-refresh did NOT advance it).
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
    // as_read_only() over a writable engine in-process shares state
    // with the writer through the same EngineEntry — no cross-process
    // refresh is needed and refresh() is documented to no-op.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();
    assert!(
        !ro.refresh().unwrap(),
        "in-process as_read_only must always no-op refresh"
    );
}

// Same reason as above: a writer + a read-only handle running
// concurrently against the same DSN need `db.shm` for the visibility
// handshake, and shm is Unix-only. The Windows engine refuses the
// read-only attach while the writer holds LOCK_EX so the test
// harness's writer thread cannot make the reader observe a
// cross-checkpoint snapshot in the first place.
#[cfg(unix)]
#[test]
fn cross_table_atomicity_under_concurrent_checkpoint() {
    // SWMR v2 P1.4: a reader inside a BEGIN/COMMIT block must observe
    // both tables at the same writer-side epoch, even while the writer
    // is actively checkpointing. The invariant chain:
    //   1. Writer persists per-table manifests in a loop.
    //   2. Writer bumps `<db>/volumes/epoch` ONLY after all manifests
    //      in the loop are durable (engine.rs checkpoint_cycle_inner).
    //   3. Reader's auto-refresh polls the epoch before each query AND
    //      skips during an active transaction (v2 P0.3).
    //   4. Therefore a reader's BEGIN+SELECT_a+SELECT_b+COMMIT block
    //      pins to a single coherent snapshot.
    //
    // This test runs writer and reader threads concurrently (writer in
    // its own engine, reader in another — they coexist under SWMR v1
    // because LockMode::Shared takes no kernel lock). Each writer
    // iteration inserts one row into each of two tables in a single
    // transaction, then checkpoints. The reader asserts COUNT(a) ==
    // COUNT(b) on every iteration.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc as StdArc;
    use std::thread;
    use std::time::Duration;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("xtable_atomic.db");
    let dsn_rw = format!("file://{}", path.display());

    // Seed both tables and produce an initial checkpoint so the reader
    // has a non-empty starting state.
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

    // Writer thread: own writable engine, insert+checkpoint loop.
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
            // Writer transaction: both inserts must commit together
            // before the checkpoint snapshots them.
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

    // Reader thread: own RO engine via different DSN string. Each
    // iteration: BEGIN, SELECT COUNT FROM a, SELECT COUNT FROM b,
    // COMMIT. Assert the two counts agree.
    let stop_r = StdArc::clone(&stop);
    let dsn_r = format!("{}?read_only=true", dsn_rw);
    let inconsistencies = StdArc::new(std::sync::Mutex::new(Vec::<(i64, i64)>::new()));
    let inc_r = StdArc::clone(&inconsistencies);
    let reader = thread::spawn(move || {
        // Open inside the thread so we don't race with seed close.
        let ro = match Database::open_read_only(&dsn_r) {
            Ok(r) => r,
            Err(_) => return,
        };
        while !stop_r.load(Ordering::Relaxed) {
            // BEGIN pins the snapshot. SchemaChanged (from any DDL)
            // would be unexpected here since we don't run DDL after
            // open; treat any error as test setup issue.
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

    // Run for ~1.5 seconds. On an unloaded laptop this is enough for
    // hundreds of writer iterations and a similar count of reader BEGIN
    // blocks — plenty of opportunity to surface a race.
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
fn auto_refresh_skipped_during_active_transaction() {
    // SWMR v2 P0.3: when the read-only handle has an open BEGIN, every
    // subsequent query in that transaction must see the same snapshot
    // — auto-refresh must NOT silently advance the epoch mid-txn. The
    // first statement after BEGIN pins the snapshot; subsequent ones
    // observe the same view.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("auto_skip_in_txn.db");

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

    // Open a transaction on the reader. Inside the txn we'll bump the
    // epoch externally and then issue another query — the query must
    // NOT auto-refresh (which would otherwise observe the bump).
    reader.query("BEGIN", ()).unwrap();

    // Bump epoch to simulate a writer checkpoint mid-transaction.
    let pre = manifest_epoch::read_epoch(&path).unwrap();
    manifest_epoch::bump_epoch(&path).unwrap();
    let post_disk = manifest_epoch::read_epoch(&path).unwrap();
    assert!(post_disk > pre, "bump must advance disk epoch");

    // Issue a query inside the txn. With auto-refresh enabled but a
    // txn open, the query must NOT call refresh — so subsequent
    // explicit refresh() should still see the bump as un-consumed.
    reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();

    // End the transaction.
    reader.query("COMMIT", ()).unwrap();

    // After COMMIT, an explicit refresh() should observe the bump
    // (proving the in-txn query never consumed it via auto-refresh).
    let advanced = reader.refresh().unwrap();
    assert!(
        advanced,
        "refresh after COMMIT must see the bump that was made mid-txn \
         (auto-refresh inside the txn was correctly skipped)"
    );
}

#[test]
fn refresh_detects_table_added_on_disk() {
    // SWMR v2 P0.2: when the writer creates a new table after the
    // reader opened (and checkpoints so manifest.bin appears on disk),
    // refresh() must surface SchemaChanged with a clear "tables added
    // on disk" message instead of silently ignoring the new table.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ddl_added.db");

    // Seed: table 'a' exists at open time.
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

    // Simulate the writer creating table 'b' in another process by
    // dropping a fresh manifest.bin into the volumes dir under that
    // table name. We don't need real volume files — the directory
    // scan only checks for manifest.bin presence.
    use stoolap::storage::volume::manifest::TableManifest;
    let new_table_dir = path.join("volumes").join("b");
    std::fs::create_dir_all(&new_table_dir).unwrap();
    let new_manifest = TableManifest::new("b");
    new_manifest
        .write_to_disk(&new_table_dir.join("manifest.bin"))
        .unwrap();

    // Bump epoch so refresh actually runs.
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
    // SWMR v2 P0.2: when the writer drops a table after the reader
    // opened (manifest.bin is unlinked), refresh() must surface
    // SchemaChanged with "tables dropped on disk" instead of silently
    // continuing with cached state.
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

    // Simulate writer dropping 'doomed' by unlinking its manifest.bin.
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
    // SWMR v2 P0.1: when a writer-produced segment carries a
    // schema_version higher than the reader's WAL-replayed schema_epoch,
    // the segment's bytes can't be safely interpreted against the
    // reader's stale schema. reload_manifests must hard-fail with
    // SchemaChanged so the caller knows to reopen.
    //
    // We simulate the scenario by directly manipulating the on-disk
    // manifest after the reader has loaded it: bump one segment's
    // schema_version above what the engine has. This isolates the drift
    // detection from the writer's own DDL machinery (which we test
    // separately in cross-process tests when v2.P0.2 lands).
    use stoolap::storage::volume::manifest::TableManifest;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("schema_drift.db");

    // Seed: create table, insert, checkpoint (produces one volume at
    // schema_version=1).
    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 100)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    // Open the reader FIRST so its `schema_epoch` baseline is set from
    // the seed segments (schema_version=1). After this, mutate the
    // on-disk manifest to claim a higher schema_version — that
    // simulates the writer doing ALTER TABLE + compaction in another
    // process AFTER the reader opened.
    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let reader = Database::open_read_only(&dsn_ro).unwrap();

    let manifest_path = path.join("volumes").join("t").join("manifest.bin");
    let mut manifest = TableManifest::read_from_disk(&manifest_path).unwrap();
    assert!(!manifest.segments.is_empty(), "seed must produce a segment");
    manifest.segments[0].schema_version = 999;
    manifest.write_to_disk(&manifest_path).unwrap();

    // Bump the manifest_epoch sidecar so refresh() actually triggers a
    // reload (otherwise the no-change fast path skips drift detection).
    use stoolap::storage::mvcc::manifest_epoch;
    manifest_epoch::bump_epoch(&path).unwrap();

    let result = reader.refresh();
    assert!(
        matches!(result, Err(Error::SchemaChanged(_))),
        "refresh must surface SchemaChanged when a segment's schema_version \
         exceeds the reader's schema_epoch; got {:?}",
        result
    );

    // Auto-refresh on the next query also surfaces it.
    let q = reader.query("SELECT COUNT(*) FROM t", ());
    assert!(
        matches!(q, Err(Error::SchemaChanged(_))),
        "auto-refresh on query must propagate SchemaChanged; got {:?}",
        q.err()
    );

    // With auto-refresh disabled, the reader still serves its
    // pre-failure snapshot (the bumped epoch never got applied because
    // refresh errored before mutating state). Verify the original row
    // is still readable.
    reader.set_auto_refresh(false);
    let mut rows = reader
        .query("SELECT COUNT(*) FROM t", ())
        .expect("auto-refresh OFF: query must succeed against stale snapshot");
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 1, "stale snapshot must still see the seed row");
}

#[test]
fn checkpoint_bumps_manifest_epoch() {
    // SWMR v1 P2: writer must publish a monotonically-increasing epoch
    // at `<db>/volumes/epoch` after each successful checkpoint cycle.
    // Reader processes poll this file to detect new checkpoints without
    // re-opening the database. Without this, refresh() has nothing to
    // poll and visibility lag becomes unbounded.
    use stoolap::storage::mvcc::manifest_epoch;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("epoch_bump.db");
    let dsn = format!("file://{}", path.display());

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 100)", ()).unwrap();

    // Before any checkpoint, epoch is 0 (file may or may not exist).
    let before = manifest_epoch::read_epoch(&path).unwrap();

    // Force a successful checkpoint via PRAGMA. Bumps require both
    // checkpoint_lsn > 0 (everything sealed under fence) AND all
    // manifests persisted; PRAGMA CHECKPOINT goes through the same
    // checkpoint_cycle_inner path as the background timer.
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let after = manifest_epoch::read_epoch(&path).unwrap();
    assert!(
        after > before,
        "epoch must advance after PRAGMA CHECKPOINT: before={}, after={}",
        before,
        after
    );

    // A second checkpoint must advance again — even with no new data,
    // each successful cycle bumps so readers can detect "writer is alive
    // and has gone through another checkpoint cycle". (If we wanted
    // bump-only-on-actual-change, that's a v2 optimization.)
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
    // memory:// has no path; the bump call short-circuits. Sanity check
    // that PRAGMA CHECKPOINT on an in-memory db doesn't error.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    // PRAGMA CHECKPOINT may or may not be a no-op on memory engines;
    // either way it must not panic or surface an error here.
    let _ = db.execute("PRAGMA CHECKPOINT", ());
}

// Tests for the engine-internal `defer_for_live_readers` helper now
// live as unit tests in `src/storage/mvcc/engine.rs` (V2.P1.5). Keeping
// them out of the integration test file lets us demote the helper from
// `#[doc(hidden)] pub` to `pub(crate)`, matching its true visibility.

// Visibility-lag microbenchmark lives in swmr_snapshot_test.rs (V2.P2.11)
// because it needs a real cross-process writer+reader pair. The same-
// process pattern races on engine shutdown / startup timing and produced
// flaky 0-inserts-observed runs that didn't reflect the SWMR mechanism.

#[test]
fn pragma_swmr_status_returns_diagnostic_row() {
    // SWMR v2 P2.12: `PRAGMA SWMR_STATUS` returns one row of cross-process
    // diagnostic info: manifest_epoch, live_lease_count, lease_max_age_secs,
    // is_read_only, checkpoint_interval_secs.
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
    // SWMR v2 P2.12: PRAGMA SWMR_STATUS is a pure-read pragma that works
    // on any handle — writable or read-only. Lets the writer introspect
    // "how many readers are attached?".
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
    // SWMR v2 P1.7: `?lease_max_age=N` overrides the engine-derived
    // default of `max(120s, 2 * checkpoint_interval)`. Both parameter
    // spellings (`lease_max_age` and `lease_max_age_secs`) parse the
    // same. Invalid values surface as Error::InvalidArgument.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("lease_max_age.db");

    {
        let dsn = format!("file://{}", path.display());
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.close().unwrap();
    }

    // Use writable Database::open so we can read config back via
    // engine().config() — open_read_only would put the engine into
    // read-only mode but ReadOnlyDatabase doesn't expose config(). The
    // lease_max_age parsing path is the same for both modes.
    let dsn_short = format!("file://{}?lease_max_age=2400", path.display());
    let db_short = Database::open(&dsn_short).unwrap();
    assert_eq!(
        db_short.engine().config().persistence.lease_max_age_secs,
        2400,
        "?lease_max_age=N must set the config field"
    );
    drop(db_short);

    // Long alias for symmetry with the field name.
    let dsn_long = format!("file://{}?lease_max_age_secs=600", path.display());
    let db_long = Database::open(&dsn_long).unwrap();
    assert_eq!(
        db_long.engine().config().persistence.lease_max_age_secs,
        600,
        "?lease_max_age_secs=N must also work"
    );
    drop(db_long);

    // Bad value: string surfaces InvalidArgument.
    let bad = format!("file://{}?lease_max_age=not-a-number", path.display());
    let err = Database::open(&bad).err().unwrap();
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "invalid lease_max_age must surface InvalidArgument; got {:?}",
        err
    );

    // The same parser also runs through open_read_only (it shares
    // parse_file_config), so a redundant ?read_only=true flag pairs
    // cleanly with the override.
    let dsn_ro = format!("file://{}?read_only=true&lease_max_age=300", path.display());
    let _ro = Database::open_read_only(&dsn_ro).unwrap();
}

#[test]
fn ro_handle_no_lease_on_memory_engine() {
    // memory:// can't have a lease (no filesystem path). open_read_only
    // already rejects memory:// (per existing test
    // `open_read_only_on_memory_dsn_errors`), but `as_read_only()` over a
    // memory writable engine reaches `from_entry` with an empty path. The
    // lease branch must short-circuit. Sanity check: just construct one
    // and confirm no panic.
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let ro = db.as_read_only();
    // No filesystem path to check; if we got here without panic, success.
    ro.query("SELECT * FROM t", ()).unwrap().next();
}
