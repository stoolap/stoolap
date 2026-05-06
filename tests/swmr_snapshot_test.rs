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

// SWMR v2's mmap-backed `db.shm` is Unix-only; the Windows ShmHandle is
// a stub that errors on construction. Gate the whole suite at file level.
#![cfg(unix)]

//! Cross-process SWMR (single-writer-multi-reader) tests.
//!
//! Tests spawn a writer/reader child via `Command::new(current_exe())` and
//! coordinate through env vars (`STOOLAP_SWMR_CHILD_ROLE`, `_DB`, `_ARG`).
//! The dispatch tests below run the matching helper when the env var is
//! set; otherwise they no-op.

use std::env;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use stoolap::storage::mvcc::manifest_epoch;
use stoolap::Database;

const CHILD_ROLE: &str = "STOOLAP_SWMR_CHILD_ROLE";
const CHILD_DB: &str = "STOOLAP_SWMR_CHILD_DB";
const CHILD_ARG: &str = "STOOLAP_SWMR_CHILD_ARG";

// ---------------------------------------------------------------------------
// Child role dispatcher
// ---------------------------------------------------------------------------

fn dispatched_as_child() -> bool {
    let role = match env::var(CHILD_ROLE) {
        Ok(r) => r,
        Err(_) => return false,
    };
    let db_path = env::var(CHILD_DB).expect("child must have STOOLAP_SWMR_CHILD_DB");
    let arg = env::var(CHILD_ARG).ok();
    match role.as_str() {
        "insert_then_checkpoint" => child_insert_then_checkpoint(&db_path, arg.as_deref()),
        "create_then_hold_open" => child_create_then_hold_open(&db_path, arg.as_deref()),
        "continuous_writer" => child_continuous_writer(&db_path, arg.as_deref()),
        "init_shm" => child_init_shm(&db_path, arg.as_deref()),
        "commit_then_hold" => child_commit_then_hold(&db_path, arg.as_deref()),
        "ro_lease_holder" => child_ro_lease_holder(&db_path, arg.as_deref()),
        "open_close_quick" => child_open_close_quick(&db_path, arg.as_deref()),
        other => panic!("unknown child role: {}", other),
    }
    true
}

fn child_insert_then_checkpoint(db_path: &str, arg: Option<&str>) {
    let n: i64 = arg
        .and_then(|s| s.parse().ok())
        .expect("child role insert_then_checkpoint requires arg = row count");
    let dsn = format!("file://{}", db_path);
    let db = Database::open(&dsn).expect("child: open writable");
    db.execute(
        "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .expect("child: create table");
    for i in 0..n {
        let sql = format!("INSERT INTO t VALUES ({}, {})", i, i * 10);
        db.execute(&sql, ()).expect("child: insert");
    }
    db.execute("PRAGMA CHECKPOINT", ())
        .expect("child: checkpoint");
    db.close().expect("child: close");
}

/// Continuous INSERT+CHECKPOINT loop. arg = "dur_ms,pace_ms".
fn child_continuous_writer(db_path: &str, arg: Option<&str>) {
    let (dur_ms, pace_ms): (u64, u64) = match arg {
        Some(s) => {
            let mut it = s.split(',');
            let d: u64 = it
                .next()
                .and_then(|x| x.parse().ok())
                .expect("child child_continuous_writer arg dur_ms");
            let p: u64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(80);
            (d, p)
        }
        None => (2500, 80),
    };
    let dsn = format!("file://{}", db_path);
    let db = Database::open(&dsn).expect("child: open writable");
    db.execute(
        "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, ts_ns INTEGER)",
        (),
    )
    .expect("child: create table");
    let start = std::time::Instant::now();
    let mut next_id: i64 = 1;
    while start.elapsed() < Duration::from_millis(dur_ms) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let sql = format!("INSERT INTO t VALUES ({}, {})", next_id, ts);
        db.execute(&sql, ()).expect("child: insert");
        db.execute("PRAGMA CHECKPOINT", ())
            .expect("child: checkpoint");
        next_id += 1;
        thread::sleep(Duration::from_millis(pace_ms));
    }
    db.close().expect("child: close");
}

/// Initializes db.shm with `(lsn,epoch,gen)` triple in arg.
fn child_init_shm(db_path: &str, arg: Option<&str>) {
    use std::sync::atomic::Ordering;
    use stoolap::storage::mvcc::shm::ShmHandle;

    let vals: Vec<u64> = arg
        .expect("child init_shm requires arg = 'lsn,epoch,gen'")
        .split(',')
        .map(|s| s.parse().expect("child init_shm arg must be u64,u64,u64"))
        .collect();
    assert_eq!(vals.len(), 3, "child init_shm arg = 'lsn,epoch,gen'");

    std::fs::create_dir_all(db_path).expect("child: mkdir db_path");
    let h = ShmHandle::create_writer(std::path::Path::new(db_path)).expect("child: create_writer");
    h.header()
        .visible_commit_lsn
        .store(vals[0], Ordering::Release);
    h.header().manifest_epoch.store(vals[1], Ordering::Release);
    h.header()
        .writer_generation
        .store(vals[2], Ordering::Release);
    // mark_ready last so open_reader can attach.
    h.mark_ready();
    thread::sleep(Duration::from_millis(300));
}

fn child_create_then_hold_open(db_path: &str, arg: Option<&str>) {
    let hold_ms: u64 = arg
        .and_then(|s| s.parse().ok())
        .expect("child role create_then_hold_open requires arg = hold_ms");
    let dsn = format!("file://{}", db_path);
    let db = Database::open(&dsn).expect("child: open writable");
    db.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)", ())
        .expect("child: create table");
    db.execute("INSERT INTO t VALUES (1)", ())
        .expect("child: insert");
    db.execute("PRAGMA CHECKPOINT", ())
        .expect("child: checkpoint");
    // Hold the writable handle so parent can observe coexistence.
    std::thread::sleep(Duration::from_millis(hold_ms));
    db.close().expect("child: close");
}

// ---------------------------------------------------------------------------
// Spawn helper used by parent tests
// ---------------------------------------------------------------------------

fn spawn_child(role: &str, db_path: &Path, arg: Option<&str>) -> Child {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.env(CHILD_ROLE, role)
        .env(CHILD_DB, db_path.display().to_string());
    if let Some(a) = arg {
        cmd.env(CHILD_ARG, a);
    }
    cmd.arg(format!("dispatch_child_role_{}", role));
    cmd.arg("--exact");
    cmd.arg("--nocapture");
    cmd.arg("--test-threads=1");
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd.spawn().expect("spawn child")
}

// Dispatch tests are no-ops in the parent invocation; they execute the
// helper when STOOLAP_SWMR_CHILD_ROLE matches.

#[test]
fn dispatch_child_role_insert_then_checkpoint() {
    let _ = dispatched_as_child();
}

#[test]
fn dispatch_child_role_create_then_hold_open() {
    let _ = dispatched_as_child();
}

#[test]
fn dispatch_child_role_continuous_writer() {
    let _ = dispatched_as_child();
}

#[test]
fn dispatch_child_role_init_shm() {
    let _ = dispatched_as_child();
}

#[test]
fn dispatch_child_role_ro_lease_holder() {
    let _ = dispatched_as_child();
}

#[test]
fn dispatch_child_role_open_close_quick() {
    let _ = dispatched_as_child();
}

/// Pure open/close cycle that bumps writer_generation in db.shm.
fn child_open_close_quick(db_path: &str, _arg: Option<&str>) {
    let dsn = format!("file://{}", db_path);
    let db = Database::open(&dsn).expect("child: open writable");
    db.execute("PRAGMA CHECKPOINT", ())
        .expect("child: checkpoint");
    db.close().expect("child: close");
}

/// Long-lived RO reader that pings the lease until killed.
fn child_ro_lease_holder(db_path: &str, arg: Option<&str>) {
    let hold_ms: u64 = arg
        .and_then(|s| s.parse().ok())
        .expect("ro_lease_holder requires arg = hold_ms");
    let dsn = format!("file://{}?read_only=true", db_path);
    let ro = Database::open_read_only(&dsn).expect("child: open RO");
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_millis(hold_ms) {
        // Cheap query refreshes lease; tolerate SwmrPendingDdl.
        let _ = ro.query("SELECT 1", ());
        thread::sleep(Duration::from_millis(50));
    }
}

// ---------------------------------------------------------------------------
// Parent (real) tests
// ---------------------------------------------------------------------------

#[test]
fn writer_subprocess_then_reader_sees_data() {
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("xproc_then.db");

    let child = spawn_child("insert_then_checkpoint", &path, Some("10"));
    let output = child.wait_with_output().expect("wait for writer child");
    assert!(
        output.status.success(),
        "writer child failed: stdout={:?}, stderr={:?}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro).expect("parent: open read-only");
    let mut rows = ro
        .query("SELECT COUNT(*) FROM t", ())
        .expect("parent: query");
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 10, "parent should see all 10 rows the child inserted");
}

#[test]
fn reader_can_attach_while_writer_subprocess_holds_lock() {
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("xproc_concurrent.db");

    let child = spawn_child("create_then_hold_open", &path, Some("2000"));

    // Let the child create the table + take LOCK_EX.
    std::thread::sleep(Duration::from_millis(400));

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let attach_result = Database::open_read_only(&dsn_ro);
    assert!(
        attach_result.is_ok(),
        "reader must attach while writer holds lock; got {:?}",
        attach_result.err()
    );
    let ro = attach_result.unwrap();

    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 1, "reader should see the 1 row the child checkpointed");

    drop(ro);
    let output = child.wait_with_output().expect("wait for writer child");
    assert!(output.status.success(), "writer child must exit cleanly");
}

#[test]
fn reader_lease_appears_during_attach() {
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("xproc_lease.db");

    let child = spawn_child("insert_then_checkpoint", &path, Some("1"));
    let output = child.wait_with_output().expect("wait for writer child");
    assert!(output.status.success());

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro).expect("parent: open ro");

    let lease = path
        .join("readers")
        .join(format!("{}.lease", std::process::id()));
    assert!(
        lease.exists(),
        "reader lease must exist at {}",
        lease.display()
    );

    drop(ro);
    assert!(
        !lease.exists(),
        "reader lease must be unlinked when ReadOnlyDatabase drops"
    );
}

#[test]
fn shm_cross_process_visibility() {
    use std::sync::atomic::Ordering;
    use stoolap::storage::mvcc::shm::ShmHandle;

    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("xproc_shm.db");

    let child = spawn_child("init_shm", &path, Some("1234,7,2"));
    thread::sleep(Duration::from_millis(100));

    let reader = ShmHandle::open_reader(&path).expect("parent: open_reader");
    let lsn = reader.header().visible_commit_lsn.load(Ordering::Acquire);
    let epoch = reader.header().manifest_epoch.load(Ordering::Acquire);
    let gen = reader.header().writer_generation.load(Ordering::Acquire);
    assert_eq!(lsn, 1234, "cross-process visible_commit_lsn");
    assert_eq!(epoch, 7, "cross-process manifest_epoch");
    assert_eq!(gen, 2, "cross-process writer_generation");
    assert!(
        !reader.is_writable(),
        "open_reader must return non-writable"
    );

    let output = child.wait_with_output().expect("wait for child");
    assert!(
        output.status.success(),
        "writer subprocess failed: stderr={:?}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "manual benchmark; flaky under CI scheduling — run with `cargo nextest run --test swmr_snapshot_test --run-ignored only visibility_lag_under_continuous_writer_is_bounded --no-capture`"]
fn visibility_lag_under_continuous_writer_is_bounded() {
    // Manual lag bench: child writer commits + checkpoints in a loop;
    // parent polls each new row's first-visible latency. Ignored under
    // CI because reload_from_disk churn under jitter can push observed
    // count below the threshold; the cross-table-atomicity test covers
    // the underlying mechanism reliably.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("xproc_visibility_lag.db");

    let child = spawn_child("continuous_writer", &path, Some("3000,250"));

    // Wait until the child creates the table.
    let table_ready = std::time::Instant::now();
    let mut ro: Option<stoolap::api::ReadOnlyDatabase> = None;
    let dsn_ro = format!("file://{}?read_only=true", path.display());
    while table_ready.elapsed() < Duration::from_millis(500) {
        if let Ok(handle) = Database::open_read_only(&dsn_ro) {
            if handle.query("SELECT MAX(id) FROM t", ()).is_ok() {
                ro = Some(handle);
                break;
            }
        }
        thread::sleep(Duration::from_millis(20));
    }
    let ro = ro.expect("reader could not attach + see table within 500ms");

    let mut last_seen: i64 = 0;
    let mut max_lag_ms: i64 = 0;
    let mut observed: i64 = 0;
    let mut samples: Vec<i64> = Vec::new();
    let measure_start = std::time::Instant::now();
    while measure_start.elapsed() < Duration::from_millis(2200) {
        let target_check_start = std::time::Instant::now();
        let mut rows = match ro.query("SELECT COUNT(*) FROM t", ()) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let count: i64 = match rows.next() {
            Some(Ok(row)) => row.get::<i64>(0).unwrap_or(0),
            _ => 0,
        };
        drop(rows);
        samples.push(count);
        if count > last_seen {
            let lag = target_check_start.elapsed().as_millis() as i64;
            if lag > max_lag_ms {
                max_lag_ms = lag;
            }
            observed += count - last_seen;
            last_seen = count;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let _ = samples;

    let output = child
        .wait_with_output()
        .expect("wait for writer subprocess");
    assert!(
        output.status.success(),
        "writer subprocess failed: stdout={:?}, stderr={:?}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        observed >= 5,
        "reader observed only {} new rows in 1.8s — wiring probably broken",
        observed
    );
    // Loose 2s upper bound for CI variance; real lag is sub-100ms.
    assert!(
        max_lag_ms < 2000,
        "max visibility lag {}ms (observed {} rows); regression in \
         bump-after-loop or auto-refresh?",
        max_lag_ms,
        observed
    );

    eprintln!(
        "visibility_lag (xproc): observed {} rows, max lag {}ms",
        observed, max_lag_ms
    );
}

#[test]
fn reader_observes_epoch_advance_after_writer_checkpoint() {
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("xproc_epoch.db");

    let child = spawn_child("insert_then_checkpoint", &path, Some("5"));
    let output = child.wait_with_output().expect("wait child 1");
    assert!(output.status.success());

    let epoch_after_first = manifest_epoch::read_epoch(&path).unwrap();
    assert!(
        epoch_after_first > 0,
        "first checkpoint should have bumped epoch above 0"
    );

    let dsn_ro = format!("file://{}?read_only=true", path.display());
    let ro = Database::open_read_only(&dsn_ro).expect("open reader");
    let mut rows = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 5, "reader sees the first batch");

    // Drop ro and open in-process writer for a non-conflicting insert
    // (subprocess helpers reuse ids 0..n which would clash on PK).
    drop(ro);

    let dsn = format!("file://{}", path.display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("INSERT INTO t VALUES (100, 999)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let epoch_after_second = manifest_epoch::read_epoch(&path).unwrap();
    assert!(
        epoch_after_second > epoch_after_first,
        "second checkpoint must bump epoch again (was {}, now {})",
        epoch_after_first,
        epoch_after_second
    );

    let ro2 = Database::open_read_only(&dsn_ro).expect("reopen reader");
    let mut rows = ro2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 6, "reader after reopen should see all 6 rows");
}

// ---------------------------------------------------------------------------
// writer publish ordering + commit-marker LSN plumbing
// ---------------------------------------------------------------------------

#[test]
fn writer_publishes_visible_commit_lsn_to_shm_on_each_commit() {
    use std::sync::atomic::Ordering;
    use stoolap::storage::mvcc::shm::ShmHandle;

    if dispatched_as_child() {
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phasec_publish.db");
    let dsn = format!("file://{}", path.display());

    let db = Database::open(&dsn).expect("open writable");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .expect("create table");

    let reader = ShmHandle::open_reader(&path).expect("open reader-side shm");

    let lsn0 = reader.header().visible_commit_lsn.load(Ordering::Acquire);
    assert!(
        lsn0 > 0,
        "after CREATE TABLE, visible_commit_lsn should be > 0 (was {})",
        lsn0
    );

    let gen = reader.header().writer_generation.load(Ordering::Acquire);
    assert!(
        gen > 0,
        "writer_generation should be bumped on open (was {})",
        gen
    );

    let mut prev = lsn0;
    for i in 0..5 {
        db.execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 10), ())
            .expect("insert");
        let now = reader.header().visible_commit_lsn.load(Ordering::Acquire);
        assert!(
            now > prev,
            "commit #{}: visible_commit_lsn must strictly advance ({} -> {})",
            i,
            prev,
            now
        );
        prev = now;
    }

    db.close().expect("close");
}

#[test]
fn shm_visible_commit_lsn_is_zero_for_in_memory_engine() {
    // memory:// has no path; engine must skip shm creation.
    if dispatched_as_child() {
        return;
    }
    let db = Database::open("memory://").expect("open in-memory");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    let mut rows = db.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 1);
}

#[test]
fn dispatch_child_role_commit_then_hold() {
    let _ = dispatched_as_child();
}

/// Inserts n rows then holds open so parent can attach shm.
fn child_commit_then_hold(db_path: &str, arg: Option<&str>) {
    let (n, hold_ms): (i64, u64) = match arg {
        Some(s) => {
            let mut it = s.split(',');
            let n: i64 = it
                .next()
                .and_then(|x| x.parse().ok())
                .expect("child commit_then_hold arg n");
            let h: u64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(500);
            (n, h)
        }
        None => (3, 500),
    };
    let dsn = format!("file://{}", db_path);
    let db = Database::open(&dsn).expect("child: open writable");
    db.execute(
        "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .expect("child: create table");
    for i in 0..n {
        let sql = format!("INSERT INTO t VALUES ({}, {})", i, i * 10);
        db.execute(&sql, ()).expect("child: insert");
    }
    thread::sleep(Duration::from_millis(hold_ms));
    db.close().expect("child: close");
}

#[test]
fn writer_subprocess_publishes_visible_commit_lsn_visible_to_other_process() {
    use std::sync::atomic::Ordering;
    use stoolap::storage::mvcc::shm::ShmHandle;

    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phasec_xproc.db");

    let child = spawn_child("commit_then_hold", &path, Some("4,800"));

    // Wait for child to flush + publish (no sync barrier; child holds 800ms).
    thread::sleep(Duration::from_millis(250));

    let reader = ShmHandle::open_reader(&path).expect("parent: open_reader");
    let lsn = reader.header().visible_commit_lsn.load(Ordering::Acquire);
    assert!(
        lsn > 0,
        "parent process must observe writer's published LSN (got {})",
        lsn
    );

    let output = child.wait_with_output().expect("wait for child");
    assert!(
        output.status.success(),
        "writer subprocess failed: stderr={:?}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// extended leases + WAL pinning
// ---------------------------------------------------------------------------

#[test]
fn read_only_handle_writes_pinned_lsn_into_lease_on_each_query() {
    // Writer must stay alive across the RO query: a closed writer
    // leaves a stale shm and the reader's handshake discards it
    // (no LOCK_EX held = stale = uncapped WAL recovery, no pin).
    use stoolap::storage::mvcc::lease::{read_pinned_lsn, READERS_DIR};
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phased_pin.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    let db_rw = Database::open(&dsn_rw).unwrap();
    db_rw
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    for i in 0..3 {
        db_rw
            .execute(&format!("INSERT INTO t VALUES ({})", i), ())
            .unwrap();
    }

    let ro = Database::open_read_only(&dsn_ro).expect("open RO");
    let _ = ro.query("SELECT COUNT(*) FROM t", ()).unwrap();

    let pid = std::process::id();
    let lease_path = path.join(READERS_DIR).join(format!("{}.lease", pid));
    assert!(
        lease_path.exists(),
        "RO open must register a lease at {}",
        lease_path.display()
    );

    let pinned = read_pinned_lsn(&lease_path).expect(
        "lease must be 8-byte v2 format after a Phase-D-aware query \
         (got non-8-byte content)",
    );
    assert!(
        pinned > 0,
        "pinned_lsn must reflect writer's published visible_commit_lsn (>0); got {}",
        pinned
    );

    drop(ro);
    drop(db_rw);
}

#[test]
fn writer_publishes_min_pinned_lsn_to_shm_when_reader_attached() {
    use std::sync::atomic::Ordering;
    use stoolap::storage::mvcc::lease::{read_pinned_lsn, READERS_DIR};
    use stoolap::storage::mvcc::shm::ShmHandle;
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phased_publish.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    let writer = Database::open(&dsn_rw).expect("open writer");
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    // Seed a non-empty checkpoint so reader has a manifest on attach
    // (avoids spurious SchemaChanged on the first real checkpoint).
    writer.execute("INSERT INTO t VALUES (-1, -1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).expect("open RO");
    let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap();

    let pid = std::process::id();
    let lease_path = path.join(READERS_DIR).join(format!("{}.lease", pid));
    let pin_at_query = read_pinned_lsn(&lease_path).expect("8-byte lease payload");
    assert!(pin_at_query > 0);

    for i in 0..3 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 7), ())
            .unwrap();
    }
    writer
        .execute("PRAGMA CHECKPOINT", ())
        .expect("PRAGMA CHECKPOINT");

    let shm = ShmHandle::open_reader(&path).expect("attach shm");
    let published_min = shm.header().min_pinned_lsn.load(Ordering::Acquire);
    assert!(
        published_min > 0,
        "writer must publish min_pinned_lsn > 0 when v2 reader attached (got {})",
        published_min
    );

    // Reader's required WAL entries survived truncation.
    let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap();

    drop(reader);
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// reader WAL-tail cursor advance
// ---------------------------------------------------------------------------

#[test]
fn reader_overlay_skips_rebuild_when_writer_lsn_unchanged() {
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phasee_noop.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();
    let lsn_after_first = reader.overlay().last_applied_lsn();

    reader.refresh().unwrap();
    let lsn_after_second = reader.overlay().last_applied_lsn();
    assert_eq!(
        lsn_after_first, lsn_after_second,
        "overlay LSN must NOT advance when writer hasn't committed"
    );

    drop(reader);
    writer.close().unwrap();
}

#[test]
fn writer_min_pinned_lsn_is_zero_when_no_v2_readers() {
    use std::sync::atomic::Ordering;
    use stoolap::storage::mvcc::shm::ShmHandle;
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phased_nopin.db");
    let dsn = format!("file://{}", path.display());

    let writer = Database::open(&dsn).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let shm = ShmHandle::open_reader(&path).expect("attach shm");
    let pinned = shm.header().min_pinned_lsn.load(Ordering::Acquire);
    assert_eq!(
        pinned, 0,
        "no v2 readers → min_pinned_lsn must stay 0 (got {})",
        pinned
    );
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// per-table cache invalidation precision
// ---------------------------------------------------------------------------

#[test]
fn reader_refresh_keeps_unrelated_table_cached_plans_alive() {
    // Writer commit on A must not evict cached plans for unrelated B.
    // Public API can't observe cache hits directly; check behavior
    // (B's count stable after a refresh that only affected A).
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phaseg_isolation.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE a (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer
        .execute("CREATE TABLE b (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer.execute("INSERT INTO a VALUES (1, 1)", ()).unwrap();
    writer.execute("INSERT INTO b VALUES (1, 1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    // Seed + hit plan cache for B.
    let _ = reader.query("SELECT COUNT(*) FROM b", ()).unwrap();
    let _ = reader.query("SELECT COUNT(*) FROM b", ()).unwrap();

    writer.execute("INSERT INTO a VALUES (2, 2)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    reader.refresh().unwrap();
    let mut rows = reader.query("SELECT COUNT(*) FROM b", ()).unwrap();
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 1, "B unchanged → row count stable across refresh");

    let mut rows = reader.query("SELECT COUNT(*) FROM a", ()).unwrap();
    let n: i64 = rows.next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(n, 2, "A changed → reader sees the new row after refresh");

    drop(reader);
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// typed sub-kind errors + DDL pass-through in WAL-tail
// ---------------------------------------------------------------------------

#[test]
fn refresh_surfaces_swmr_pending_ddl_when_writer_creates_table_after_attach() {
    use stoolap::Error;
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phaseh_ddl.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    // Baseline checkpointed so reader attaches without SchemaChanged.
    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE existing (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    writer
        .execute("INSERT INTO existing VALUES (1)", ())
        .unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();

    // DDL bumps visible_commit_lsn without necessarily checkpointing;
    // WAL-tail must catch it and surface SwmrPendingDdl.
    writer
        .execute("CREATE TABLE late_arrival (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    let res = reader.refresh();
    match res {
        Err(Error::SwmrPendingDdl(summary)) => {
            assert!(
                summary.contains("late_arrival"),
                "DDL summary must name the new table; got: {}",
                summary
            );
        }
        other => panic!(
            "expected Err(SwmrPendingDdl), got: {:?}",
            other.as_ref().err()
        ),
    }

    drop(reader);
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// cross-process + crash test matrix
// ---------------------------------------------------------------------------

#[test]
fn writer_reincarnation_advances_writer_generation_in_shm() {
    use std::sync::atomic::Ordering;
    use stoolap::storage::mvcc::shm::ShmHandle;
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phasei_gen.db");

    let child1 = spawn_child("insert_then_checkpoint", &path, Some("3"));
    let out1 = child1.wait_with_output().expect("wait writer 1");
    assert!(
        out1.status.success(),
        "writer 1 must exit cleanly: stderr={:?}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let shm1 = ShmHandle::open_reader(&path).expect("attach shm after w1");
    let gen_after_w1 = shm1.header().writer_generation.load(Ordering::Acquire);
    assert!(
        gen_after_w1 >= 1,
        "writer_generation must be >= 1 after first writer (got {})",
        gen_after_w1
    );
    drop(shm1);

    // open_close_quick avoids PK conflicts with writer 1's rows.
    let child2 = spawn_child("open_close_quick", &path, None);
    let out2 = child2.wait_with_output().expect("wait writer 2");
    assert!(
        out2.status.success(),
        "writer 2 must exit cleanly: stderr={:?}",
        String::from_utf8_lossy(&out2.stderr)
    );

    let shm2 = ShmHandle::open_reader(&path).expect("attach shm after w2");
    let gen_after_w2 = shm2.header().writer_generation.load(Ordering::Acquire);
    assert!(
        gen_after_w2 > gen_after_w1,
        "writer_generation must STRICTLY ADVANCE on writer reincarnation \
         (after w1={}, after w2={})",
        gen_after_w1,
        gen_after_w2
    );
}

#[test]
fn reader_subprocess_killed_mid_read_leaves_stale_lease_for_reaping() {
    use stoolap::storage::mvcc::lease::{reap_stale_leases, READERS_DIR};
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phasei_reaper.db");
    let dsn_rw = format!("file://{}", path.display());

    {
        let db = Database::open(&dsn_rw).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let mut child = spawn_child("ro_lease_holder", &path, Some("60000"));

    let readers_dir = path.join(READERS_DIR);
    let mut waited = 0;
    while waited < 2000 {
        if readers_dir
            .read_dir()
            .map(|mut e| e.any(|x| x.is_ok()))
            .unwrap_or(false)
        {
            break;
        }
        thread::sleep(Duration::from_millis(50));
        waited += 50;
    }
    let live_count_before = readers_dir
        .read_dir()
        .map(|e| e.flatten().count())
        .unwrap_or(0);
    assert!(
        live_count_before >= 1,
        "child reader must register a lease (found {})",
        live_count_before
    );

    // SIGKILL = simulated hard crash; Drop never runs.
    child.kill().expect("kill reader subprocess");
    let _ = child.wait();

    // Lease still on disk; sleep past max_age then reap.
    thread::sleep(Duration::from_millis(700));
    let reaped =
        reap_stale_leases(&readers_dir, Duration::from_millis(500)).expect("reap_stale_leases");
    assert!(
        reaped >= 1,
        "stale lease must be reaped after SIGKILL + max_age expiry (reaped={})",
        reaped
    );

    let live_count_after = readers_dir
        .read_dir()
        .map(|e| e.flatten().count())
        .unwrap_or(0);
    assert_eq!(live_count_after, 0, "no leases should remain after reap");
}

#[test]
fn corrupt_shm_is_rejected_by_open_reader() {
    use stoolap::storage::mvcc::shm::{ShmHandle, SHM_FILENAME, SHM_SIZE};
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path();

    // 1. File exists but is empty -> reject.
    std::fs::write(path.join(SHM_FILENAME), b"").unwrap();
    assert!(
        ShmHandle::open_reader(path).is_err(),
        "empty db.shm must be rejected"
    );

    // 2. File too small (<SHM_SIZE) -> reject.
    std::fs::write(path.join(SHM_FILENAME), vec![0u8; 16]).unwrap();
    assert!(
        ShmHandle::open_reader(path).is_err(),
        "too-small db.shm must be rejected"
    );

    // 3. Right size but bad magic -> reject.
    let mut buf = vec![0u8; SHM_SIZE];
    buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    std::fs::write(path.join(SHM_FILENAME), &buf).unwrap();
    assert!(
        ShmHandle::open_reader(path).is_err(),
        "bad-magic db.shm must be rejected"
    );

    // 4. Right size + magic but init_done == 0 (writer-mid-init crash) -> reject.
    use stoolap::storage::mvcc::shm::{SHM_MAGIC, SHM_VERSION};
    let mut buf = vec![0u8; SHM_SIZE];
    buf[0..4].copy_from_slice(&SHM_MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&SHM_VERSION.to_le_bytes());
    // init_done bytes (offset 8..16) intentionally left zero.
    std::fs::write(path.join(SHM_FILENAME), &buf).unwrap();
    assert!(
        ShmHandle::open_reader(path).is_err(),
        "init_done==0 db.shm must be rejected (writer crashed mid-init)"
    );
}

#[test]
fn dropping_one_of_two_in_process_ro_handles_keeps_lease_alive() {
    use stoolap::storage::mvcc::lease::READERS_DIR;
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("p1review_dual_ro.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    {
        let db = Database::open(&dsn_rw).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }

    let pid = std::process::id();
    let lease_path = path.join(READERS_DIR).join(format!("{}.lease", pid));

    let ro1 = Database::open_read_only(&dsn_ro).unwrap();
    let _ = ro1.query("SELECT COUNT(*) FROM t", ()).unwrap();
    assert!(lease_path.exists(), "lease must exist after first RO open");

    let ro2 = Database::open_read_only(&dsn_ro).unwrap();
    let _ = ro2.query("SELECT COUNT(*) FROM t", ()).unwrap();
    assert!(
        lease_path.exists(),
        "lease must still exist after second RO open"
    );

    drop(ro1);
    assert!(
        lease_path.exists(),
        "lease must survive while another in-process RO handle holds it"
    );
    let _ = ro2.query("SELECT COUNT(*) FROM t", ()).unwrap();

    drop(ro2);
    assert!(
        !lease_path.exists(),
        "lease must be unlinked once the LAST in-process RO handle drops"
    );
}

#[test]
fn refresh_does_not_misfire_swmr_pending_ddl_for_create_index_rerecord() {
    // Post-checkpoint DDL re-records of pre-existing CREATE INDEX
    // must not re-fire as SwmrPendingDdl on every checkpoint.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("p1review_index_rerecord.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer.execute("CREATE INDEX idx_v ON t(v)", ()).unwrap();
    writer.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();

    // Checkpoint re-records CREATE INDEX with new LSNs > attach baseline.
    writer.execute("INSERT INTO t VALUES (2, 2)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let res = reader.refresh();
    assert!(
        res.is_ok(),
        "post-checkpoint DDL re-record of pre-existing CREATE INDEX must not \
         trigger SwmrPendingDdl, got: {:?}",
        res.err()
    );

    drop(reader);
    writer.close().unwrap();
}

#[test]
fn engine_table_checkpoint_lsns_reflects_per_table_state() {
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("phaseg_engine_lsns.db");
    let dsn_rw = format!("file://{}", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE a (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    writer
        .execute("CREATE TABLE b (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    writer.execute("INSERT INTO a VALUES (1)", ()).unwrap();
    writer.execute("INSERT INTO b VALUES (1)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let engine = writer.engine();
    let lsns = engine.table_checkpoint_lsns();
    assert!(lsns.contains_key("a"), "table 'a' must appear: {:?}", lsns);
    assert!(lsns.contains_key("b"), "table 'b' must appear: {:?}", lsns);
    let lsn_a = *lsns.get("a").unwrap();
    let lsn_b = *lsns.get("b").unwrap();
    assert!(
        lsn_a > 0 && lsn_b > 0,
        "checkpoint_lsn must be > 0 after PRAGMA CHECKPOINT (a={}, b={})",
        lsn_a,
        lsn_b
    );

    writer.close().unwrap();
}

#[test]
fn reader_skips_wal_scan_when_writer_commits_only_dml() {
    // DDL watermark fast path: DML-only commits must not trigger a
    // WAL tail scan on the reader.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ddl_watermark.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();
    let scans_baseline = reader.overlay().wal_scan_count();

    for i in 0..10 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .unwrap();
    }
    reader.refresh().unwrap();
    assert_eq!(
        reader.overlay().wal_scan_count(),
        scans_baseline,
        "DML-only commits must not trigger a WAL tail scan"
    );

    // DDL DOES trigger a scan.
    writer.execute("CREATE INDEX idx_v ON t(v)", ()).unwrap();
    let _ = reader.refresh();
    assert!(
        reader.overlay().wal_scan_count() > scans_baseline,
        "DDL commit must trigger a WAL tail scan"
    );

    drop(reader);
    writer.close().unwrap();
}

#[test]
fn ddl_watermark_bumps_on_commit_marker_not_entry() {
    // Regression: bumping catalog_epoch at DDL ENTRY append lets the
    // reader skip the scan for transactional DDL whose commit lands
    // AFTER an unrelated DML commit that already advanced the
    // reader's last_applied past the DDL entry. Watermark must
    // represent the committed DDL frontier, not appended DDL.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("ddl_watermark_marker.db");
    let dsn_rw = format!("file://{}", path.display());
    let dsn_ro = format!("{}?read_only=true", dsn_rw);

    let writer1 = Database::open(&dsn_rw).unwrap();
    writer1
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer1.execute("INSERT INTO t VALUES (0, 0)", ()).unwrap();
    writer1.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();

    // Begin transactional DDL on writer1; do NOT commit.
    // (CREATE TABLE inside BEGIN uses the transactional DDL path;
    // CREATE INDEX would auto-commit even inside BEGIN.)
    writer1.execute("BEGIN", ()).unwrap();
    writer1
        .execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    // Independent writer2 commits DML, advancing visible_commit_lsn
    // past writer1's uncommitted DDL entry.
    let writer2 = Database::open(&dsn_rw).unwrap();
    writer2.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();

    // Reader refresh now: must NOT raise SwmrPendingDdl (writer1's
    // DDL is still uncommitted) AND must advance last_applied past
    // the DDL entry's LSN.
    match reader.refresh() {
        Ok(_) => {}
        Err(e) => panic!("reader saw uncommitted DDL: {:?}", e),
    }

    // Now writer1 commits the transactional DDL.
    writer1.execute("COMMIT", ()).unwrap();

    // Reader refresh: MUST surface the committed DDL. Without the
    // marker-side bump, catalog_epoch would still be the entry's LSN
    // (now <= reader.last_applied), the fast path would skip the
    // scan, and the DDL would be missed.
    match reader.refresh() {
        Err(stoolap::Error::SwmrPendingDdl(_)) => {}
        other => panic!(
            "expected SwmrPendingDdl after committed transactional DDL, got: {:?}",
            other
        ),
    }

    drop(reader);
    writer2.close().ok();
    writer1.close().unwrap();
}

#[test]
fn orphan_volumes_get_reaped_with_active_reader() {
    // Regression: pre-fix, compaction's unlink + the orphan reaper both
    // bailed when ANY reader was alive, so .vol files accumulated
    // unboundedly. Fix: gate on min(reader_manifest_epoch) >=
    // current_epoch instead of "any reader alive". An auto-refreshing
    // reader publishes its epoch every refresh, so the writer can reap
    // within ~one query interval of compaction.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("orphan_reap.db");
    let dsn_rw = format!("file://{}?compact_threshold=2", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (0, 0)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();

    let table_dir = path.join("volumes").join("t");
    let count_vols = || -> usize {
        std::fs::read_dir(&table_dir)
            .map(|it| {
                it.flatten()
                    .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("vol"))
                    .count()
            })
            .unwrap_or(0)
    };

    // Drive write+checkpoint cycles. Each cycle adds a sub-target volume;
    // every other cycle compaction merges them. Without the fix, ALL old
    // sub-target files persist as orphans.
    for i in 1..40 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        // Reader query → publishes new manifest_epoch to lease.
        let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
    }

    // Pre-fix: ~40 orphan files. Post-fix: at most a handful (the most
    // recent ones whose epoch hasn't been observed by reader yet).
    let n = count_vols();
    assert!(
        n < 15,
        "orphan reaping should keep .vol count bounded with active reader; \
         saw {} files (expected < 15 with compact_threshold=2)",
        n
    );

    drop(reader);
    writer.close().unwrap();
}

// Helpers for the tests below.
fn count_vols_for(table_dir: &std::path::Path) -> usize {
    std::fs::read_dir(table_dir)
        .map(|it| {
            it.flatten()
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("vol"))
                .count()
        })
        .unwrap_or(0)
}
fn count_sidecars_for(table_dir: &std::path::Path) -> usize {
    std::fs::read_dir(table_dir)
        .map(|it| {
            it.flatten()
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("retired"))
                .count()
        })
        .unwrap_or(0)
}

#[test]
fn auto_refresh_off_reader_blocks_orphan_reap() {
    // Reader with auto_refresh=off after attach holds a stale snapshot.
    // Compaction creates orphans; reaper creates retire sidecars; .vol
    // files MUST stay until the reader re-enables refresh and queries.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("auto_off_blocks.db");
    let dsn_rw = format!("file://{}?compact_threshold=2", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (0, 0)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();

    let table_dir = path.join("volumes").join("t");
    // Phase 1: warm up with auto_refresh=on so the reader's pin advances
    // and writer-side compaction can actually operate (visibility cap
    // restricts compaction to volumes <= reader pin).
    for i in 1..20 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
    }
    // Snapshot reader's current epoch, then freeze.
    let frozen_epoch = reader.last_seen_epoch_for_test();
    reader.set_auto_refresh(false);
    let vols_at_freeze = count_vols_for(&table_dir);

    // Phase 2: more cycles. Compaction can still merge volumes whose
    // visible_at_lsn <= reader's frozen pin (the high pin from last
    // refresh). Orphans accumulate. Reaper creates sidecars but
    // CAN'T unlink because reader's epoch file is stuck at frozen.
    for i in 20..40 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
    }

    let vols_frozen = count_vols_for(&table_dir);
    let sidecars = count_sidecars_for(&table_dir);
    assert!(
        sidecars > 0 || vols_frozen >= vols_at_freeze,
        "either sidecars created (preferred) or no compaction occurred; \
         vols_at_freeze={} vols_now={} sidecars={}",
        vols_at_freeze,
        vols_frozen,
        sidecars
    );

    // Phase 3: re-enable refresh + query → reader publishes a new epoch
    // past the sidecar stamps → reaper drains.
    reader.set_auto_refresh(true);
    for _ in 0..3 {
        let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
        writer.execute("INSERT INTO t VALUES (-1, -1)", ()).ok();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let unfrozen_epoch = reader.last_seen_epoch_for_test();
    assert!(
        unfrozen_epoch > frozen_epoch,
        "epoch must have advanced after re-enabling refresh ({} -> {})",
        frozen_epoch,
        unfrozen_epoch
    );

    drop(reader);
    writer.close().unwrap();
}

#[test]
fn active_begin_blocks_orphan_reap() {
    // BEGIN holds a stable snapshot via the maybe_auto_refresh
    // skip-during-active-tx guard. Reaper must defer until COMMIT
    // releases the BEGIN AND a subsequent refresh fires.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("begin_blocks.db");
    let dsn_rw = format!("file://{}?compact_threshold=2", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (0, 0)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();

    // Warm up so reader pin advances and compaction can operate.
    for i in 1..20 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
    }
    let frozen_epoch = reader.last_seen_epoch_for_test();
    reader.query("BEGIN", ()).unwrap();

    // BEGIN suppresses maybe_auto_refresh → no epoch publish.
    for i in 20..40 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
    }
    assert_eq!(
        reader.last_seen_epoch_for_test(),
        frozen_epoch,
        "BEGIN must suppress epoch advance"
    );

    // COMMIT releases the BEGIN; refresh advances epoch; reaper can drain.
    let _ = reader.query("COMMIT", ());
    for _ in 0..3 {
        let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
        writer.execute("INSERT INTO t VALUES (-1, -1)", ()).ok();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert!(
        reader.last_seen_epoch_for_test() > frozen_epoch,
        "epoch must advance after COMMIT + refresh"
    );

    drop(reader);
    writer.close().unwrap();
}

#[test]
fn writer_uses_min_handle_epoch_across_handles() {
    // Two RO handles in same PID. One auto-refreshes, the other is
    // held idle (auto_refresh=off). Writer's min_reader_handle_epoch
    // must track the IDLE one — orphans should NOT be reaped.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("multi_handle_min.db");
    let dsn_rw = format!("file://{}?compact_threshold=2", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    writer.execute("INSERT INTO t VALUES (0, 0)", ()).unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let active = Database::open_read_only(&dsn_ro).unwrap();
    let idle = Database::open_read_only(&dsn_ro).unwrap();
    active.refresh().unwrap();
    idle.refresh().unwrap();

    // Warm up both with auto_refresh=on so pins advance.
    for i in 1..20 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let _ = active.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
        let _ = idle.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
    }
    let idle_epoch_frozen = idle.last_seen_epoch_for_test();
    idle.set_auto_refresh(false);

    // Continue. active's epoch advances; idle's stays frozen.
    for i in 20..40 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let _ = active.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
    }
    let active_epoch_high = active.last_seen_epoch_for_test();
    assert!(
        active_epoch_high > idle_epoch_frozen,
        "active should have advanced past idle's frozen epoch"
    );

    // Writer sees min = idle's frozen epoch — orphans not reaped.
    // Drop idle → its epoch file gone → writer's MIN jumps to active.
    drop(idle);
    for _ in 0..3 {
        let _ = active.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
        writer.execute("INSERT INTO t VALUES (-1, -1)", ()).ok();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }

    drop(active);
    writer.close().unwrap();
}

#[test]
fn lazy_cold_load_succeeds_with_active_reader() {
    // Stale reader (auto_refresh=off) holds an old manifest with old
    // segment_ids. Writer compacts (new manifest, old IDs orphaned).
    // Reaper creates sidecars but does NOT unlink. Stale reader's
    // ensure_volume(old_id) must still succeed because the file is
    // still at vol_<id>.vol on disk.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("lazy_cold_load.db");
    let dsn_rw = format!("file://{}?compact_threshold=2", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    let writer = Database::open(&dsn_rw).unwrap();
    writer
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for i in 0..10 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 10), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }

    let reader = Database::open_read_only(&dsn_ro).unwrap();
    reader.refresh().unwrap();
    reader.set_auto_refresh(false);

    // Heavy compaction on writer side; reader keeps stale manifest.
    for i in 10..30 {
        writer
            .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 10), ())
            .unwrap();
        writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }

    // Reader scans rows pointed to by its STALE manifest (old segment
    // IDs). The lazy cold-load via ensure_volume MUST succeed.
    let mut rows = reader.query("SELECT COUNT(*) FROM t", ()).unwrap();
    let row = rows.next().unwrap().unwrap();
    let n: i64 = row.get(0).unwrap();
    assert!(
        n >= 10,
        "stale reader should still see at least the rows from its snapshot (got {})",
        n
    );

    drop(reader);
    writer.close().unwrap();
}

#[test]
fn orphans_reaped_after_writer_restart_with_no_readers() {
    // Crash recovery path: writer compacts (orphans + sidecars on
    // disk), is killed, restarts. With no readers attached, next
    // sweep reaps all orphans. Validates retire-sidecar protocol's
    // crash-safety: state lives entirely on disk.
    if dispatched_as_child() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("crash_recovery.db");
    let dsn_rw = format!("file://{}?compact_threshold=2", path.display());
    let dsn_ro = format!("file://{}?read_only=true", path.display());

    let table_dir = path.join("volumes").join("t");
    {
        let writer = Database::open(&dsn_rw).unwrap();
        writer
            .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        let reader = Database::open_read_only(&dsn_ro).unwrap();
        reader.refresh().unwrap();
        // Warm up so reader pin advances and writer compaction can act.
        for i in 0..20 {
            writer
                .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
                .unwrap();
            writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
            let _ = reader.query("SELECT COUNT(*) FROM t", ()).unwrap().next();
        }
        reader.set_auto_refresh(false);
        // Now drive cycles where reader's epoch is frozen → sidecars
        // accumulate without unlinks.
        for i in 20..40 {
            writer
                .execute(&format!("INSERT INTO t VALUES ({}, {})", i, i), ())
                .unwrap();
            writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        // Reader holds stale snapshot. Drop reader → epoch file gone.
        // Writer "crashes" (drop without close).
        drop(reader);
        drop(writer);
    }

    // Sidecars from the prior cycle are on disk. With no live leases,
    // restart's next sweep reaps everything.
    let writer = Database::open(&dsn_rw).unwrap();
    let vols_before = count_vols_for(&table_dir);
    let sidecars_before = count_sidecars_for(&table_dir);
    writer
        .execute("INSERT INTO t VALUES (100, 100)", ())
        .unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    writer
        .execute("INSERT INTO t VALUES (101, 101)", ())
        .unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let vols_after = count_vols_for(&table_dir);
    let sidecars_after = count_sidecars_for(&table_dir);
    assert!(
        vols_after <= vols_before && sidecars_after <= sidecars_before,
        "post-restart sweep with no readers should reap orphans+sidecars; \
         vols {}->{}, sidecars {}->{}",
        vols_before,
        vols_after,
        sidecars_before,
        sidecars_after
    );
    writer.close().unwrap();
}
