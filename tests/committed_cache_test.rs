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

//! A transaction id one database committed says nothing about the same id
//! in another database on the same thread

use stoolap::storage::TransactionRegistry;

#[test]
fn another_registrys_committed_id_does_not_make_a_transaction_visible() {
    let first = TransactionRegistry::new();
    let (id, _) = first.begin_transaction();
    first.commit_transaction(id);
    assert!(first.is_directly_visible(id));

    let second = TransactionRegistry::new();
    let (active, _) = second.begin_transaction();
    assert_eq!(active, id);
    let active_seen = second.is_directly_visible(active);
    second.start_commit(active);
    let committing_seen = second.is_directly_visible(active);
    second.abort_transaction(active);
    let aborted_seen = second.is_directly_visible(active);
    assert_eq!(
        (active_seen, committing_seen, aborted_seen),
        (false, false, false),
        "the first registry's committed id was taken for the second's"
    );
}

#[cfg(feature = "test-failpoints")]
fn read_while_a_commit_waits(db: &stoolap::Database) -> Vec<i64> {
    use std::sync::mpsc;
    use std::time::Duration;

    let (published_tx, published_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel::<()>();
    let writer = db.clone();
    let commit = std::thread::spawn(move || {
        let mut tx = writer.begin().unwrap();
        tx.execute("UPDATE t SET v = 100 WHERE id = 1", ()).unwrap();
        stoolap::test_failpoints::before_commit_visible(move || {
            published_tx.send(()).unwrap();
            finish_rx.recv_timeout(Duration::from_secs(15)).unwrap();
        });
        tx.commit().unwrap();
    });
    published_rx.recv_timeout(Duration::from_secs(15)).unwrap();
    let seen = db
        .query("SELECT v FROM t WHERE id = 1", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    finish_tx.send(()).unwrap();
    commit.join().unwrap();
    seen
}

/// Commits ids 1 to 4 on this thread and reads them, so they are cached
#[cfg(feature = "test-failpoints")]
fn commit_and_read_ids(db: &stoolap::Database, table: &str) {
    db.execute(
        &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, v INTEGER)"),
        (),
    )
    .unwrap();
    db.execute(&format!("INSERT INTO {table} VALUES (1, 0), (2, 0)"), ())
        .unwrap();
    for n in 1..=3 {
        db.execute(&format!("UPDATE {table} SET v = {n} WHERE id = 1"), ())
            .unwrap();
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_commit_waiting_to_be_visible_is_not_read_through_another_database() {
    let other = stoolap::Database::open("memory://committed_cache_other").unwrap();
    commit_and_read_ids(&other, "u");
    let db = stoolap::Database::open("memory://committed_cache_this").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    assert_eq!(
        read_while_a_commit_waits(&db),
        vec![0],
        "the other database's committed id showed an uncommitted update"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_reopened_database_does_not_take_its_old_commits_for_new_ones() {
    let dsn = "memory://committed_cache_reopened";
    let first = stoolap::Database::open(dsn).unwrap();
    commit_and_read_ids(&first, "u");
    first.close().unwrap();
    drop(first);
    let db = stoolap::Database::open(dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    assert_eq!(
        read_while_a_commit_waits(&db),
        vec![0],
        "the closed database's committed id showed an uncommitted update"
    );
}
