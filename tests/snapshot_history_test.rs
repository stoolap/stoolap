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

//! An open snapshot keeps the version it reads past the history limit,
//! and the history goes once no snapshot reads it

use stoolap::core::IsolationLevel;
use stoolap::Database;

fn values(tx: &mut stoolap::ApiTransaction, sql: &str) -> Vec<i64> {
    tx.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
}

fn chain_entries(db: &Database) -> usize {
    db.engine().get_version_store("t").unwrap().chain_entries()
}

#[test]
fn a_snapshot_keeps_its_row_past_the_history_limit() {
    for updates in [1, 10, 11, 25] {
        let db = Database::open(&format!("memory://snapshot_history_{updates}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 0), (2, 0)", ())
            .unwrap();
        let mut reader = db
            .begin_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(values(&mut reader, "SELECT v FROM t WHERE id = 1"), vec![0]);
        for n in 1..=updates {
            db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
                .unwrap();
        }
        assert_eq!(
            values(&mut reader, "SELECT v FROM t WHERE id = 1"),
            vec![0],
            "{updates} updates: the key read"
        );
        assert_eq!(
            values(&mut reader, "SELECT v FROM t ORDER BY id"),
            vec![0, 0],
            "{updates} updates: the scan"
        );
        reader.rollback().unwrap();
        assert_eq!(
            db.query("SELECT v FROM t WHERE id = 1", ())
                .unwrap()
                .map(|r| r.unwrap().get::<i64>(0).unwrap())
                .collect::<Vec<_>>(),
            vec![updates]
        );
    }
}

/// Two snapshots opened between updates each read their own version
#[test]
fn snapshots_opened_between_updates_read_their_own_versions() {
    let db = Database::open("memory://snapshot_history_two").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut first = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(values(&mut first, "SELECT v FROM t WHERE id = 1"), vec![0]);
    for n in 1..=15 {
        db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
    let mut second = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(
        values(&mut second, "SELECT v FROM t WHERE id = 1"),
        vec![15]
    );
    for n in 16..=40 {
        db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
    assert_eq!(values(&mut first, "SELECT v FROM t WHERE id = 1"), vec![0]);
    assert_eq!(
        values(&mut second, "SELECT v FROM t WHERE id = 1"),
        vec![15]
    );
    first.rollback().unwrap();
    // With only the second snapshot open, the history it does not read goes
    for n in 41..=60 {
        db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
    assert_eq!(
        values(&mut second, "SELECT v FROM t WHERE id = 1"),
        vec![15]
    );
    second.rollback().unwrap();
}

/// Once no snapshot is open the history past the limit is dropped again
#[test]
fn the_history_goes_once_no_snapshot_reads_it() {
    let db = Database::open("memory://snapshot_history_released").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut reader = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(values(&mut reader, "SELECT v FROM t WHERE id = 1"), vec![0]);
    for n in 1..=30 {
        db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
    assert!(chain_entries(&db) >= 30, "the snapshot's history is kept");
    reader.rollback().unwrap();
    db.execute("UPDATE t SET v = 31 WHERE id = 1", ()).unwrap();
    assert!(
        chain_entries(&db) <= 10,
        "{} history entries after the snapshot closed",
        chain_entries(&db)
    );
}

/// A snapshot that reads the replaced head keeps that head alone: the
/// history older than what it reads goes past the limit
#[test]
fn a_snapshot_keeps_only_the_version_it_reads() {
    let db = Database::open("memory://snapshot_history_head_only").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    // The chain at its limit: the head and nine versions behind it
    for n in 1..=9 {
        db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
    assert_eq!(chain_entries(&db), 9);
    let mut reader = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(values(&mut reader, "SELECT v FROM t WHERE id = 1"), vec![9]);
    db.execute("UPDATE t SET v = 10 WHERE id = 1", ()).unwrap();
    assert_eq!(chain_entries(&db), 1, "only the version the snapshot reads");
    assert_eq!(values(&mut reader, "SELECT v FROM t WHERE id = 1"), vec![9]);
    reader.rollback().unwrap();
}

/// A snapshot held across many updates keeps a long chain, which the
/// caller's thread must be able to let go and to relay out
fn long_chain(name: &str) -> (Database, stoolap::ApiTransaction, stoolap::Statement) {
    let db = Database::open(&format!("memory://snapshot_history_long_{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut reader = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(values(&mut reader, "SELECT v FROM t WHERE id = 1"), vec![0]);
    let update = db.prepare("UPDATE t SET v = $1 WHERE id = 1").unwrap();
    for n in 1..=30_000i64 {
        update.execute((n,)).unwrap();
    }
    assert!(chain_entries(&db) >= 30_000, "{name}: the chain is long");
    assert_eq!(values(&mut reader, "SELECT v FROM t WHERE id = 1"), vec![0]);
    (db, reader, update)
}

#[test]
fn a_long_kept_chain_is_let_go_without_recursion() {
    let (db, mut reader, update) = long_chain("drop");
    reader.rollback().unwrap();
    update.execute((30_001i64,)).unwrap();
    assert!(chain_entries(&db) <= 10, "the chain was let go");
}

#[test]
fn a_long_kept_chain_is_relaid_without_recursion() {
    let (db, mut reader, _update) = long_chain("alter");
    db.execute("ALTER TABLE t ADD COLUMN w INTEGER DEFAULT 7", ())
        .unwrap();
    assert_eq!(values(&mut reader, "SELECT v FROM t WHERE id = 1"), vec![0]);
    reader.rollback().unwrap();
}

/// Short snapshots handed on from one to the next keep only what the one
/// still open reads, not the history its predecessors read
#[test]
fn snapshots_handed_on_keep_only_what_the_open_one_reads() {
    let db = Database::open("memory://snapshot_history_handoff").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut open = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let mut v = 0i64;
    for _ in 0..20 {
        for _ in 0..24 {
            v += 1;
            db.execute(&format!("UPDATE t SET v = {v} WHERE id = 1"), ())
                .unwrap();
        }
        let mut next = db
            .begin_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(values(&mut next, "SELECT v FROM t WHERE id = 1"), vec![v]);
        v += 1;
        db.execute(&format!("UPDATE t SET v = {v} WHERE id = 1"), ())
            .unwrap();
        open.rollback().unwrap();
        open = next;
    }
    let seen = v - 1;
    v += 1;
    db.execute(&format!("UPDATE t SET v = {v} WHERE id = 1"), ())
        .unwrap();
    assert_eq!(
        values(&mut open, "SELECT v FROM t WHERE id = 1"),
        vec![seen]
    );
    assert!(
        chain_entries(&db) <= 10,
        "{} versions kept for one snapshot reading the version two back",
        chain_entries(&db)
    );
    open.rollback().unwrap();
}

/// A snapshot whose version lies deeper than the limit keeps the whole
/// chain at a publication; the background trim cuts what no open snapshot
/// reads below it, however deep
#[test]
fn the_background_trim_cuts_below_a_deep_version() {
    let db = Database::open("memory://snapshot_history_deep_trim").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut older = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(values(&mut older, "SELECT v FROM t WHERE id = 1"), vec![0]);
    for n in 1..=30 {
        db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
    let mut newer = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(values(&mut newer, "SELECT v FROM t WHERE id = 1"), vec![30]);
    // Twelve updates while both are open: the newer one's version is
    // further back than the limit
    for n in 31..=42 {
        db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
    older.rollback().unwrap();
    db.execute("UPDATE t SET v = 43 WHERE id = 1", ()).unwrap();
    assert!(chain_entries(&db) > 40, "the publication kept the chain");
    let store = db.engine().get_version_store("t").unwrap();
    assert!(
        store.trim_history_past_limit() > 25,
        "the trim dropped the tail"
    );
    assert!(
        chain_entries(&db) <= 14,
        "{} versions kept for a snapshot reading 13 back",
        chain_entries(&db)
    );
    assert_eq!(values(&mut newer, "SELECT v FROM t WHERE id = 1"), vec![30]);
    newer.rollback().unwrap();
    // With no snapshot open the trim leaves the limit
    for n in 44..=80 {
        db.execute(&format!("UPDATE t SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
    store.trim_history_past_limit();
    assert!(chain_entries(&db) <= 10, "{}", chain_entries(&db));
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_snapshot_taken_while_the_trim_waits_keeps_its_row() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let db = Database::open("memory://snapshot_history_trim_waits").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    let mut old = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    for n in 1..=30 {
        db.execute("UPDATE t SET v = $1 WHERE id = 1", (n,))
            .unwrap();
    }
    old.rollback().unwrap();
    let store = db.engine().get_version_store("t").unwrap();
    let reader = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&reader);
    let writer = db.clone();
    stoolap::test_failpoints::before_trim_lock(move || {
        let mut tx = writer
            .begin_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(values(&mut tx, "SELECT v FROM t WHERE id = 1"), vec![30]);
        for n in 31..=50 {
            writer
                .execute("UPDATE t SET v = $1 WHERE id = 1", (n,))
                .unwrap();
        }
        *slot.borrow_mut() = Some(tx);
    });
    store.trim_history_past_limit();
    let mut tx = reader.borrow_mut().take().expect("the trim hook ran");
    assert_eq!(
        values(&mut tx, "SELECT v FROM t WHERE id = 1"),
        vec![30],
        "a snapshot taken before the trim locked the versions lost its row"
    );
    tx.rollback().unwrap();
}

#[cfg(feature = "test-failpoints")]
fn read_while_an_update_commits(
    name: &str,
    updates: i64,
    commit_sql: &'static str,
) -> (Vec<i64>, Vec<i64>) {
    use std::sync::mpsc;
    use std::time::Duration;

    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0), (2, 0)", ())
        .unwrap();
    for n in 1..=updates {
        db.execute("UPDATE t SET v = $1 WHERE id = 1", (n,))
            .unwrap();
    }
    let (published_tx, published_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel::<()>();
    let writer = db.clone();
    let commit = std::thread::spawn(move || {
        let mut tx = writer.begin().unwrap();
        tx.execute(commit_sql, ()).unwrap();
        stoolap::test_failpoints::before_commit_visible(move || {
            published_tx.send(()).unwrap();
            finish_rx.recv_timeout(Duration::from_secs(15)).unwrap();
        });
        tx.commit().unwrap();
    });
    published_rx.recv_timeout(Duration::from_secs(15)).unwrap();
    let autocommit: Vec<i64> = db
        .query("SELECT v FROM t WHERE id = 1", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    let mut tx = db.begin().unwrap();
    let in_tx = values(&mut tx, "SELECT v FROM t WHERE id = 1");
    tx.rollback().unwrap();
    finish_tx.send(()).unwrap();
    commit.join().unwrap();
    (autocommit, in_tx)
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_row_at_its_history_limit_stays_readable_while_an_update_commits() {
    for (shape, sql) in [
        ("one", "UPDATE t SET v = 100 WHERE id = 1"),
        ("two", "UPDATE t SET v = 100 WHERE id IN (1, 2)"),
    ] {
        for updates in [3, 9, 10, 25] {
            // Each database reads on its own thread: the committed-id cache is per thread
            let name = format!("snapshot_history_commit_window_{shape}_{updates}");
            let seen =
                std::thread::spawn(move || read_while_an_update_commits(&name, updates, sql))
                    .join()
                    .unwrap();
            assert_eq!(
                seen,
                (vec![updates], vec![updates]),
                "{updates} updates, {shape} row commit: the row vanished while it committed"
            );
        }
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_snapshot_begun_while_an_update_commits_keeps_its_row() {
    use std::sync::mpsc;
    use std::time::Duration;

    let db = Database::open("memory://snapshot_history_begun_while_committing").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    for n in 1..=9 {
        db.execute("UPDATE t SET v = $1 WHERE id = 1", (n,))
            .unwrap();
    }
    let (start_tx, start_rx) = mpsc::channel::<()>();
    let (published_tx, published_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel::<()>();
    let writer = db.clone();
    let commit = std::thread::spawn(move || {
        start_rx.recv_timeout(Duration::from_secs(15)).unwrap();
        let mut tx = writer.begin().unwrap();
        tx.execute("UPDATE t SET v = 10 WHERE id = 1", ()).unwrap();
        stoolap::test_failpoints::before_commit_visible(move || {
            published_tx.send(()).unwrap();
            finish_rx.recv_timeout(Duration::from_secs(15)).unwrap();
        });
        tx.commit().unwrap();
    });
    stoolap::test_failpoints::after_transaction_begun(move || {
        start_tx.send(()).unwrap();
        published_rx.recv_timeout(Duration::from_secs(15)).unwrap();
    });
    let mut reader = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let before = values(&mut reader, "SELECT v FROM t WHERE id = 1");
    finish_tx.send(()).unwrap();
    commit.join().unwrap();
    let after = values(&mut reader, "SELECT v FROM t WHERE id = 1");
    reader.rollback().unwrap();
    assert_eq!(
        (before, after),
        (vec![9], vec![9]),
        "the snapshot lost its row to an update committing as it began"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_snapshot_keeps_its_row_through_commits_as_it_begins() {
    use std::sync::mpsc;
    use std::time::Duration;

    let db = Database::open("memory://snapshot_history_begin_window").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    for n in 1..=9 {
        db.execute("UPDATE t SET v = $1 WHERE id = 1", (n,))
            .unwrap();
    }
    let (published_tx, published_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel::<()>();
    let (start_tx, start_rx) = mpsc::channel::<()>();
    let writer = db.clone();
    let commits = std::thread::spawn(move || {
        start_rx.recv_timeout(Duration::from_secs(15)).unwrap();
        writer
            .execute("UPDATE t SET v = 10 WHERE id = 1", ())
            .unwrap();
        let mut tx = writer.begin().unwrap();
        tx.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
        stoolap::test_failpoints::before_commit_visible(move || {
            published_tx.send(()).unwrap();
            finish_rx.recv_timeout(Duration::from_secs(15)).unwrap();
        });
        tx.commit().unwrap();
    });
    stoolap::test_failpoints::after_transaction_begun(move || {
        start_tx.send(()).unwrap();
        published_rx.recv_timeout(Duration::from_secs(15)).unwrap();
    });
    let mut reader = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let before = values(&mut reader, "SELECT v FROM t WHERE id = 1");
    finish_tx.send(()).unwrap();
    commits.join().unwrap();
    let after = values(&mut reader, "SELECT v FROM t WHERE id = 1");
    reader.rollback().unwrap();
    assert_eq!(
        (before, after),
        (vec![9], vec![9]),
        "commits between the snapshot's sequence and its protection cut its row"
    );
}
