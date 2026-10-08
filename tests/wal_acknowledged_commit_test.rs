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

//! A commit the log acknowledged stays committed when a later write or
//! sync fails, a commit reported as failed does not replay, and a log
//! that failed does not close or checkpoint as a success.

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::Duration;

use stoolap::core::Error;
use stoolap::storage::config::PersistenceConfig;
use stoolap::storage::mvcc::wal_manager::{WALEntry, WALManager, WALOperationType};
use stoolap::storage::SyncMode;
use stoolap::test_failpoints as fp;
use stoolap::Database;

/// No interval sync during a test: Normal commits stay written, unsynced
const LONG_INTERVAL_MS: u32 = 3_600_000;

fn wal(dir: &std::path::Path, mode: SyncMode) -> WALManager {
    let config = PersistenceConfig {
        sync_interval_ms: LONG_INTERVAL_MS,
        ..PersistenceConfig::default()
    };
    WALManager::with_config(dir, mode, Some(&config)).unwrap()
}

fn insert(wal: &WALManager, id: i64) {
    wal.append_entry(WALEntry::new(
        id,
        "t".to_string(),
        id,
        WALOperationType::Insert,
        vec![1, 2, 3],
    ))
    .unwrap();
}

/// A transaction's row and its marker; the marker's result is the commit's
fn commit(wal: &WALManager, id: i64) -> bool {
    insert(wal, id);
    wal.write_commit_marker(id).is_ok()
}

fn commit_on_thread(wal: &Arc<WALManager>, id: i64) -> bool {
    let wal = Arc::clone(wal);
    std::thread::spawn(move || commit(&wal, id)).join().unwrap()
}

fn replayed(dir: &std::path::Path) -> Vec<i64> {
    let reopened = WALManager::new(dir, SyncMode::Full).unwrap();
    let mut rows = Vec::new();
    reopened
        .replay_two_phase(0, |entry| {
            if entry.operation == WALOperationType::Insert {
                rows.push(entry.row_id);
            }
            Ok(())
        })
        .unwrap();
    rows.sort_unstable();
    rows
}

/// Two acknowledged Normal commits, written and not synced
fn normal_wal_with_two_commits(dir: &std::path::Path) -> WALManager {
    let wal = wal(dir, SyncMode::Normal);
    assert!(commit(&wal, 1));
    assert!(commit(&wal, 2));
    wal
}

#[test]
fn a_refused_write_keeps_the_commits_acknowledged_before_it() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = normal_wal_with_two_commits(dir.path());
    fp::WAL_WRITE_FAIL.store(true, Ordering::SeqCst);
    let third = commit(&wal, 3);
    fp::WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
    assert!(!third, "the refused write fails its commit");
    drop(wal);
    assert_eq!(replayed(dir.path()), [1, 2]);
}

/// The failed batch holds A's complete marker before the torn bytes of B
#[test]
fn a_partial_write_removes_the_complete_marker_it_carried() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = Arc::new(normal_wal_with_two_commits(dir.path()));
    insert(&wal, 3);
    let other = Arc::clone(&wal);
    let (sent, received) = mpsc::channel();
    fp::after_record_buffered(move || {
        fp::WAL_WRITE_PARTIAL.store(true, Ordering::SeqCst);
        let b = commit_on_thread(&other, 4);
        fp::WAL_WRITE_PARTIAL.store(false, Ordering::SeqCst);
        sent.send(b).unwrap();
    });
    let a = wal.write_commit_marker(3).is_ok();
    let b = received.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(
        !a && !b,
        "both commits of the failed write fail (A {a}, B {b})"
    );
    drop(wal);
    assert_eq!(replayed(dir.path()), [1, 2]);
}

/// A's marker is in the buffer when B's flush writes it; a later write
/// fails the log before A flushes itself
#[test]
fn a_commit_whose_marker_another_flush_wrote_succeeds() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = Arc::new(wal(dir.path(), SyncMode::Normal));
    assert!(commit(&wal, 1));
    insert(&wal, 10);
    let others = Arc::clone(&wal);
    let (sent, received) = mpsc::channel();
    fp::after_record_buffered(move || {
        let b = commit_on_thread(&others, 20);
        fp::WAL_WRITE_FAIL.store(true, Ordering::SeqCst);
        let c = commit_on_thread(&others, 30);
        fp::WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
        sent.send((b, c)).unwrap();
    });
    let a = wal.write_commit_marker(10).is_ok();
    let (b, c) = received.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(b, "B's commit was acknowledged");
    assert!(!c, "C's write failed");
    assert!(a, "B's write carried A's marker: A succeeds");
    drop(wal);
    assert_eq!(replayed(dir.path()), [1, 10, 20]);
}

#[test]
fn a_normal_commit_acknowledged_before_a_failed_rotation_survives() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let config = PersistenceConfig {
        wal_max_size: 1,
        sync_interval_ms: LONG_INTERVAL_MS,
        ..PersistenceConfig::default()
    };
    let wal = WALManager::with_config(dir.path(), SyncMode::Normal, Some(&config)).unwrap();
    assert!(commit(&wal, 1));
    assert!(commit(&wal, 2));
    fp::RETIRED_WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
    let rotated = wal.maybe_rotate();
    fp::RETIRED_WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
    assert!(rotated.is_err(), "the retired file's sync failed");
    drop(wal);
    assert_eq!(replayed(dir.path()), [1, 2]);
}

/// Full mode keeps its cut at the synced floor: the synced commit stays,
/// the one only written fails and does not replay
#[test]
fn a_full_commit_written_by_a_failed_flush_fails_and_does_not_replay() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = Arc::new(wal(dir.path(), SyncMode::Full));
    assert!(commit(&wal, 1));
    insert(&wal, 10);
    let others = Arc::clone(&wal);
    let (sent, received) = mpsc::channel();
    fp::after_record_buffered(move || {
        fp::WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
        let b = commit_on_thread(&others, 20);
        fp::WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
        sent.send(b).unwrap();
    });
    let a = wal.write_commit_marker(10).is_ok();
    let b = received.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(!a && !b, "neither marker was synced (A {a}, B {b})");
    drop(wal);
    assert_eq!(replayed(dir.path()), [1]);
}

#[test]
fn a_failed_log_closes_with_an_error_and_stops() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = normal_wal_with_two_commits(dir.path());
    fp::WAL_WRITE_FAIL.store(true, Ordering::SeqCst);
    assert!(!commit(&wal, 3));
    fp::WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
    assert!(wal.close().is_err());
    assert!(
        matches!(wal.flush(), Err(Error::WalNotRunning)),
        "the log stopped"
    );
}

// --- Through the database --------------------------------------------------

fn dsn(dir: &tempfile::TempDir, options: &str) -> String {
    format!(
        "file://{}?checkpoint_interval=0&sync_interval_ms={LONG_INTERVAL_MS}{options}",
        dir.path().display()
    )
}

fn count(db: &Database) -> i64 {
    db.query_one("SELECT COUNT(*) FROM t", ()).unwrap()
}

fn columns(db: &Database) -> usize {
    db.query("SELECT * FROM t", ()).unwrap().columns().len()
}

fn two_rows(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
}

/// Two acknowledged rows, then `refused` with the log failing
fn refuse_after_two_rows(db: &Database, refused: &str) {
    two_rows(db);
    fp::WAL_WRITE_FAIL.store(true, Ordering::SeqCst);
    let refused = db.execute(refused, ());
    fp::WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
    assert!(refused.is_err());
}

fn reopen(dsn: &str, db: Database) -> Database {
    let _ = db.close();
    drop(db);
    Database::open(dsn).unwrap()
}

#[test]
fn acknowledged_rows_survive_a_refused_statement_and_a_reopen() {
    let _guard = fp::FailpointGuard::new();
    for close_checkpoint in ["", "&checkpoint_on_close=off"] {
        for refused in [
            "INSERT INTO t VALUES (3, 30)",
            "ALTER TABLE t ADD COLUMN c INTEGER",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let dsn = dsn(&dir, close_checkpoint);
            let db = Database::open(&dsn).unwrap();
            refuse_after_two_rows(&db, refused);
            let db = reopen(&dsn, db);
            let stage = format!("{refused} {close_checkpoint}");
            assert_eq!(count(&db), 2, "{stage}");
            assert_eq!(columns(&db), 2, "the refused ADD COLUMN stays out: {stage}");
        }
    }
}

/// A Normal DDL needs its sync. A failed sync, or a write that tore its
/// marker, fails the DDL, keeps the rows acknowledged before it, and
/// leaves no record an earlier DDL's marker id could commit at replay
#[test]
fn a_normal_ddl_that_fails_in_the_log_leaves_no_record_and_keeps_earlier_rows() {
    let _guard = fp::FailpointGuard::new();
    let mut seen = Vec::new();
    for (failure, flag) in [
        ("sync", &fp::WAL_SYNC_FAIL),
        ("partial write", &fp::WAL_WRITE_PARTIAL),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = dsn(&dir, "&checkpoint_on_close=off");
        let db = Database::open(&dsn).unwrap();
        two_rows(&db);
        flag.store(true, Ordering::SeqCst);
        let added = db.execute("ALTER TABLE t ADD COLUMN c INTEGER", ()).is_ok();
        flag.store(false, Ordering::SeqCst);
        let live = columns(&db);
        let db = reopen(&dsn, db);
        seen.push((failure, added, live, count(&db), columns(&db)));
    }
    assert_eq!(
        seen,
        [("sync", false, 2, 2, 2), ("partial write", false, 2, 2, 2)],
        "(failure, DDL succeeded, live columns, rows after reopen, columns after reopen)"
    );
}

/// Normal mode's interval sync is not a commit's requirement: its failure
/// keeps the written commit and stops the log
#[test]
fn a_normal_commit_whose_interval_sync_fails_stays_and_the_log_refuses_more() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let config = PersistenceConfig {
        sync_interval_ms: 0,
        ..PersistenceConfig::default()
    };
    let wal = WALManager::with_config(dir.path(), SyncMode::Normal, Some(&config)).unwrap();
    assert!(commit(&wal, 1));
    fp::WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
    let second = commit(&wal, 2);
    fp::WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
    assert!(second, "the written commit is acknowledged");
    let more = wal.append_entry(WALEntry::new(
        3,
        "t".to_string(),
        3,
        WALOperationType::Insert,
        vec![1],
    ));
    assert!(more.is_err(), "the failed log refuses further writes");
    drop(wal);
    assert_eq!(replayed(dir.path()), [1, 2]);
}

#[test]
fn a_failed_cut_reports_an_unknown_outcome() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = normal_wal_with_two_commits(dir.path());
    insert(&wal, 3);
    fp::WAL_WRITE_PARTIAL.store(true, Ordering::SeqCst);
    fp::WAL_CUT_FAIL.store(true, Ordering::SeqCst);
    let marked = wal.write_commit_marker(3);
    fp::WAL_CUT_FAIL.store(false, Ordering::SeqCst);
    fp::WAL_WRITE_PARTIAL.store(false, Ordering::SeqCst);
    let error = marked.expect_err("the torn write fails").to_string();
    assert!(error.contains("unknown"), "{error}");
}

fn rotating_full_wal(dir: &std::path::Path) -> Arc<WALManager> {
    let config = PersistenceConfig {
        wal_max_size: 1,
        sync_interval_ms: LONG_INTERVAL_MS,
        ..PersistenceConfig::default()
    };
    Arc::new(WALManager::with_config(dir, SyncMode::Full, Some(&config)).unwrap())
}

/// A's Full marker is drained into the old file by a rotation; a write
/// into the new file fails before the old file's sync starts
#[test]
fn a_full_marker_in_a_retired_file_is_cut_when_a_later_write_fails() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = rotating_full_wal(dir.path());
    assert!(commit(&wal, 1));
    insert(&wal, 10);
    let others = Arc::clone(&wal);
    let (sent, received) = mpsc::channel();
    fp::after_record_buffered(move || {
        let writer = Arc::clone(&others);
        let rotating = Arc::clone(&others);
        let b = std::thread::spawn(move || {
            let (failed, outcome) = mpsc::channel();
            fp::after_wal_swap(move || {
                fp::WAL_WRITE_FAIL.store(true, Ordering::SeqCst);
                let b = commit_on_thread(&writer, 20);
                fp::WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
                failed.send(b).unwrap();
            });
            let _ = rotating.maybe_rotate();
            outcome.recv().unwrap()
        })
        .join()
        .unwrap();
        sent.send(b).unwrap();
    });
    let a = wal.write_commit_marker(10).is_ok();
    let b = received.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(!a && !b, "neither marker was synced (A {a}, B {b})");
    drop(wal);
    assert_eq!(replayed(dir.path()), [1]);
}

/// The old file's sync is running when a write into the new file fails:
/// A's outcome waits for that sync, which succeeds
#[test]
fn a_running_settlement_decides_a_commit_waiting_on_a_failure() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = rotating_full_wal(dir.path());
    assert!(commit(&wal, 1));
    insert(&wal, 10);
    let others = Arc::clone(&wal);
    let (sent, received) = mpsc::channel();
    fp::after_record_buffered(move || {
        let writer = Arc::clone(&others);
        let rotating = Arc::clone(&others);
        let b = std::thread::spawn(move || {
            let (b_sent, b_received) = mpsc::channel();
            fp::before_retired_settle(move || {
                let (said, heard) = mpsc::channel();
                let waits = said.clone();
                std::thread::spawn(move || {
                    fp::before_wal_failure_cleanup(move || {
                        let _ = waits.send("cleanup waits for the settlement");
                    });
                    fp::WAL_WRITE_FAIL.store(true, Ordering::SeqCst);
                    let b = commit(&writer, 20);
                    fp::WAL_WRITE_FAIL.store(false, Ordering::SeqCst);
                    let _ = said.send("B returned");
                    b_sent.send(b).unwrap();
                });
                heard
                    .recv_timeout(Duration::from_secs(10))
                    .expect("B failed its write");
            });
            let _ = rotating.maybe_rotate();
            b_received.recv_timeout(Duration::from_secs(10)).unwrap()
        })
        .join()
        .unwrap();
        sent.send(b).unwrap();
    });
    let a = wal.write_commit_marker(10).is_ok();
    let b = received.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(!b, "B's write failed");
    assert!(a, "the settlement made A's marker durable: A succeeds");
    drop(wal);
    assert_eq!(replayed(dir.path()), [1, 10]);
}

#[test]
fn a_checkpoint_whose_sync_fails_reports_it_and_keeps_the_recovery_boundary() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = dsn(&dir, "&checkpoint_on_close=off");
    let db = Database::open(&dsn).unwrap();
    two_rows(&db);
    let meta = dir.path().join("wal").join("checkpoint.meta");
    let boundary = std::fs::read(&meta).ok();
    fp::WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
    let checkpointed = db.execute("PRAGMA CHECKPOINT", ());
    fp::WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
    assert!(checkpointed.is_err(), "the failed sync is reported");
    assert_eq!(
        std::fs::read(&meta).ok(),
        boundary,
        "the boundary did not move"
    );
    let db = reopen(&dsn, db);
    assert_eq!(count(&db), 2);
}

#[test]
fn a_checkpoint_over_a_failed_log_reports_it() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(&dir, "")).unwrap();
    refuse_after_two_rows(&db, "INSERT INTO t VALUES (3, 30)");
    assert!(db.execute("PRAGMA CHECKPOINT", ()).is_err());
}

/// The old handle stays alive: the open succeeds only if the close
/// released the file lock and the registry
#[test]
fn a_close_after_a_failed_log_reports_it_and_releases_the_database() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = dsn(&dir, "");
    let db = Database::open(&dsn).unwrap();
    refuse_after_two_rows(&db, "INSERT INTO t VALUES (3, 30)");
    assert!(db.close().is_err(), "the failed log is reported at close");
    let again = Database::open(&dsn).expect("the database opens again in this process");
    assert_eq!(count(&again), 2);
    drop(db);
}

// --- Schema records, explicit syncs and sync_mode=none ------------------------

fn ddl_record(op: WALOperationType) -> WALEntry {
    WALEntry::new(
        stoolap::storage::mvcc::persistence::DDL_TXN_ID,
        "t".to_string(),
        0,
        op,
        vec![1, 2, 3],
    )
}

fn replayed_operations(dir: &std::path::Path) -> Vec<WALOperationType> {
    let reopened = WALManager::new(dir, SyncMode::Full).unwrap();
    let mut operations = Vec::new();
    reopened
        .replay_two_phase(0, |entry| {
            operations.push(entry.operation);
            Ok(())
        })
        .unwrap();
    operations
}

/// Every schema change commits under one marker id, so a record or a
/// marker of it written alone could commit a change that failed
#[test]
fn a_schema_record_or_marker_outside_a_unit_is_refused() {
    use stoolap::storage::mvcc::persistence::DDL_TXN_ID;
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = wal(dir.path(), SyncMode::Normal);
    wal.append_unit(
        vec![
            ddl_record(WALOperationType::CreateTable),
            WALEntry::commit_marker(DDL_TXN_ID),
        ],
        false,
    )
    .unwrap();
    assert!(
        wal.write_commit_marker(DDL_TXN_ID).is_err(),
        "a schema marker alone is refused"
    );
    fp::WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
    let alone = wal.append_entry(ddl_record(WALOperationType::AlterTable));
    fp::WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
    assert!(alone.is_err(), "a schema record alone is refused");
    drop(wal);
    let operations = replayed_operations(dir.path());
    assert!(
        operations.contains(&WALOperationType::CreateTable)
            && !operations.contains(&WALOperationType::AlterTable),
        "{operations:?}"
    );
}

/// The commit exception never hides the failure of a sync the caller
/// asked for
#[test]
fn an_explicit_sync_that_fails_is_an_error_with_everything_synced() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = wal(dir.path(), SyncMode::Full);
    assert!(commit(&wal, 1));
    fp::WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
    let synced = wal.sync();
    fp::WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
    assert!(synced.is_err());
}

#[test]
fn a_checkpoint_whose_wal_sync_fails_publishes_no_boundary() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let wal = wal(dir.path(), SyncMode::Full);
    assert!(commit(&wal, 1));
    fp::WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
    let checkpointed = wal.create_checkpoint(vec![]);
    fp::WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
    assert!(checkpointed.is_err());
    assert!(!dir.path().join("checkpoint.meta").exists());
}

/// sync_mode=none syncs no schema change; a checkpoint still syncs
#[test]
fn schema_changes_under_sync_mode_none_do_not_sync() {
    let _guard = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(&dir, "&sync_mode=none")).unwrap();
    fp::WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
    let changed = [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "ALTER TABLE t ADD COLUMN c INTEGER",
        "CREATE INDEX ia ON t(a)",
    ]
    .map(|sql| (sql, db.execute(sql, ()).is_ok()));
    fp::WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
    assert!(changed.iter().all(|(_, ok)| *ok), "{changed:?}");
    db.execute("INSERT INTO t VALUES (1, 10, 100)", ()).unwrap();
    fp::WAL_SYNC_FAIL.store(true, Ordering::SeqCst);
    let checkpointed = db.execute("PRAGMA CHECKPOINT", ());
    fp::WAL_SYNC_FAIL.store(false, Ordering::SeqCst);
    assert!(
        checkpointed.is_err(),
        "the checkpoint's sync is not optional"
    );
}
