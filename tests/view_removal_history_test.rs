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

//! The removal history behind index lookups under a statement view: the
//! entries a commit the view does not see took out are read back, by the
//! commit's sequence while it publishes and by its stamp once it ended

#![cfg(feature = "test-failpoints")]

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use stoolap::test_failpoints as fp;
use stoolap::Database;

const WAIT: Duration = Duration::from_secs(15);

fn memory(name: &str) -> Database {
    Database::open(&format!("memory://view_removal_history_{name}")).unwrap()
}

fn table(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t (k)", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 2, 30), (4, 3, 40), (5, 5, 50)",
        (),
    )
    .unwrap();
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// Index answers on this thread so far: scan lookups under a view, and a
/// file table's equality candidates
fn index_answers() -> usize {
    fp::view_index_lookups() + fp::equality_candidate_answers()
}

/// The ids `sql` gives while another handle runs `sqls` after its view,
/// and the index answers it took
fn overtaken(db: &Database, sql: &str, sqls: Vec<String>) -> (Vec<i64>, usize) {
    let ran = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&ran));
    fp::after_statement_view_taken(move || {
        for sql in &sqls {
            other.execute(sql, ()).unwrap();
        }
        flag.set(true);
    });
    let answers = index_answers();
    let seen = ids(db, sql);
    assert!(ran.get(), "{sql}: the change did not land after the view");
    (seen, index_answers() - answers)
}

fn strings(sqls: &[&str]) -> Vec<String> {
    sqls.iter().map(|sql| sql.to_string()).collect()
}

#[test]
fn a_view_whose_floor_a_slow_commit_holds_reads_only_the_removals_it_misses() {
    let db = memory("old_floor");
    table(&db);
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    let rows: Vec<String> = (100..400).map(|id| format!("({id}, 7, 0)")).collect();
    db.execute(&format!("INSERT INTO t VALUES {}", rows.join(",")), ())
        .unwrap();
    let (held_tx, held) = mpsc::channel();
    let (go_tx, go) = mpsc::channel::<()>();
    let slow = {
        let other = db.clone();
        std::thread::spawn(move || {
            fp::before_commit_visible(move || {
                held_tx.send(()).unwrap();
                go.recv_timeout(WAIT).unwrap();
            });
            other.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
        })
    };
    held.recv_timeout(WAIT).unwrap();
    // More removals than one probe reads, all complete before the view
    for id in 100..400 {
        db.execute(&format!("UPDATE t SET k = 8 WHERE id = {id}"), ())
            .unwrap();
    }
    let (seen, lookups) = overtaken(
        &db,
        "SELECT id FROM t WHERE k = 2",
        strings(&["DELETE FROM t WHERE id = 3"]),
    );
    go_tx.send(()).unwrap();
    slow.join().unwrap();
    assert_eq!((seen, lookups > 0), (vec![2, 3], true));
}

#[test]
fn a_commit_held_after_its_index_removals_is_read_while_it_publishes() {
    let db = memory("publishing");
    table(&db);
    let (held_tx, held) = mpsc::channel();
    let (go_tx, go) = mpsc::channel::<()>();
    let writer: Rc<RefCell<Option<std::thread::JoinHandle<()>>>> = Rc::default();
    {
        let (other, slot) = (db.clone(), Rc::clone(&writer));
        fp::after_statement_view_taken(move || {
            *slot.borrow_mut() = Some(std::thread::spawn(move || {
                fp::after_indexes_published(move || {
                    held_tx.send(()).unwrap();
                    go.recv_timeout(WAIT).unwrap();
                });
                other.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
            }));
            held.recv_timeout(WAIT).unwrap();
        });
    }
    let answers = index_answers();
    let seen = ids(&db, "SELECT id FROM t WHERE k = 2");
    let lookups = index_answers() - answers;
    go_tx.send(()).unwrap();
    writer.borrow_mut().take().unwrap().join().unwrap();
    assert_eq!((seen, lookups > 0), (vec![2, 3], true));
}

#[test]
fn a_rename_after_the_removal_keeps_the_key_readable_under_the_new_name() {
    let db = memory("rename");
    table(&db);
    let (seen, lookups) = overtaken(
        &db,
        "SELECT id FROM t WHERE kk = 2",
        strings(&[
            "UPDATE t SET k = 9 WHERE id = 3",
            "ALTER TABLE t RENAME COLUMN k TO kk",
        ]),
    );
    assert_eq!((seen, lookups > 0), (vec![2, 3], true));
}

#[test]
fn composite_and_intersected_lookups_read_back_removed_keys() {
    let mut wrong = Vec::new();
    let cases: &[(&str, &str)] = &[
        (
            "SELECT id FROM m WHERE a = 1 AND b = 2",
            "UPDATE m SET a = 9 WHERE id = 2",
        ),
        (
            "SELECT id FROM m WHERE a = 1",
            "UPDATE m SET a = 9 WHERE id = 2",
        ),
        (
            "SELECT id FROM m WHERE a = 1 AND b <= 2",
            "UPDATE m SET b = 9 WHERE id = 2",
        ),
        (
            "SELECT id FROM m WHERE a = 1 AND c = 5",
            "UPDATE m SET a = 9 WHERE id = 2",
        ),
        (
            "SELECT id FROM m WHERE a = 1 AND c = 5",
            "UPDATE m SET c = 7 WHERE id = 2",
        ),
        (
            "SELECT id FROM m WHERE c = 5 AND a >= 1 AND a <= 1",
            "DELETE FROM m WHERE id = 2",
        ),
        (
            "SELECT id FROM m WHERE a = 1 AND c = 6",
            "UPDATE m SET c = 7 WHERE id = 3",
        ),
        (
            "SELECT id FROM m WHERE b = 3 AND c = 6",
            "UPDATE m SET c = 7 WHERE id = 3",
        ),
        (
            "SELECT id FROM m WHERE c IN (5, 6)",
            "UPDATE m SET c = 7 WHERE id = 2",
        ),
        (
            "SELECT id FROM m WHERE c >= 5 AND c <= 6",
            "UPDATE m SET c = 7 WHERE id = 3",
        ),
        (
            "SELECT id FROM m WHERE c > 5",
            "UPDATE m SET c = 4 WHERE id = 3",
        ),
        (
            "SELECT id FROM m WHERE c BETWEEN 5 AND 6",
            "UPDATE m SET c = 7 WHERE id = 3",
        ),
        (
            "SELECT id FROM m WHERE a = 1 AND b = 2 AND c BETWEEN 5 AND 5",
            "UPDATE m SET c = 7 WHERE id = 2",
        ),
    ];
    for (n, (sql, change)) in cases.iter().enumerate() {
        let db = memory(&format!("composite_{n}"));
        db.execute(
            "CREATE TABLE m (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX m_ab ON m (a, b)", ()).unwrap();
        db.execute("CREATE INDEX m_c ON m (c)", ()).unwrap();
        db.execute(
            "INSERT INTO m VALUES (1, 1, 1, 5), (2, 1, 2, 5), (3, 1, 3, 6), (4, 2, 1, 5)",
            (),
        )
        .unwrap();
        let expected = ids(&db, sql);
        let (seen, lookups) = overtaken(&db, sql, strings(&[change]));
        if seen != expected || lookups == 0 {
            wrong.push(format!(
                "{sql} after {change}: {seen:?} with {lookups} lookups, expected {expected:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn keys_of_each_type_are_read_back_as_the_index_matches_them() {
    let mut wrong = Vec::new();
    let long = "x".repeat(40);
    let cases: Vec<(String, String)> = vec![
        (
            "SELECT id FROM w WHERE f = 0.0".into(),
            "UPDATE w SET f = 9.5 WHERE id = 2".into(),
        ),
        (
            "SELECT id FROM w WHERE f = 1".into(),
            "UPDATE w SET f = 9.5 WHERE id = 1".into(),
        ),
        (
            "SELECT id FROM w WHERE s = 'a'".into(),
            "UPDATE w SET s = 'z' WHERE id = 1".into(),
        ),
        (
            format!("SELECT id FROM w WHERE s = '{long}'"),
            "UPDATE w SET s = 'z' WHERE id = 3".into(),
        ),
        (
            "SELECT id FROM w WHERE f >= -0.0 AND f < 2".into(),
            "DELETE FROM w WHERE id = 1".into(),
        ),
        (
            "SELECT id FROM w WHERE p = 2 AND q = 3 AND r = 4".into(),
            "UPDATE w SET r = 9 WHERE id = 2".into(),
        ),
    ];
    for (n, (sql, change)) in cases.into_iter().enumerate() {
        let db = memory(&format!("types_{n}"));
        db.execute(
            "CREATE TABLE w (id INTEGER PRIMARY KEY, f FLOAT, s TEXT, p INTEGER, q INTEGER, r INTEGER)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX w_f ON w (f)", ()).unwrap();
        db.execute("CREATE INDEX w_s ON w (s)", ()).unwrap();
        db.execute("CREATE INDEX w_pqr ON w (p, q, r)", ()).unwrap();
        db.execute(
            &format!(
                "INSERT INTO w VALUES (1, 1.0, 'a', 1, 1, 1), (2, -0.0, 'ab', 2, 3, 4), \
                 (3, 2.5, '{long}', 2, 3, 4)"
            ),
            (),
        )
        .unwrap();
        let expected = ids(&db, &sql);
        let (seen, lookups) = overtaken(&db, &sql, vec![change.clone()]);
        if seen != expected || lookups == 0 {
            wrong.push(format!(
                "{sql} after {change}: {seen:?} with {lookups} lookups, expected {expected:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_view_past_the_probe_budget_or_the_kept_history_scans_with_the_right_rows() {
    let db = memory("budget");
    table(&db);
    let rows: Vec<String> = (100..400).map(|id| format!("({id}, 7, 0)")).collect();
    db.execute(&format!("INSERT INTO t VALUES {}", rows.join(",")), ())
        .unwrap();
    let reads = fp::removal_history_reads();
    let (seen, lookups) = overtaken(
        &db,
        "SELECT id FROM t WHERE k = 2",
        strings(&[
            "DELETE FROM t WHERE id >= 100",
            "DELETE FROM t WHERE id = 3",
        ]),
    );
    let capped = fp::removal_history_reads()[3] - reads[3];
    assert_eq!(
        (seen, lookups, capped > 0),
        (vec![2, 3], 0, true),
        "the budget"
    );

    let db = memory("evicted");
    db.execute(
        "CREATE TABLE e (id INTEGER PRIMARY KEY, s TEXT NOT NULL, k INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX e_s ON e (s)", ()).unwrap();
    db.execute("CREATE INDEX e_k ON e (k)", ()).unwrap();
    let text = "y".repeat(4096);
    let rows: Vec<String> = (1..=600)
        .map(|id| format!("({id}, '{text}{id}', {})", id % 2))
        .collect();
    db.execute(&format!("INSERT INTO e VALUES {}", rows.join(",")), ())
        .unwrap();
    let expected = ids(&db, "SELECT id FROM e WHERE k = 1");
    let reads = fp::removal_history_reads();
    // Six commits of 60 keys of 4 KB each: past the kept history's budget
    let deletes: Vec<String> = (0..6)
        .map(|n| {
            format!(
                "DELETE FROM e WHERE id > {} AND id <= {}",
                n * 100,
                n * 100 + 60
            )
        })
        .collect();
    let (seen, lookups) = overtaken(&db, "SELECT id FROM e WHERE k = 1", deletes);
    let evicted = fp::removal_history_reads()[2] - reads[2];
    assert_eq!(
        (seen, lookups, evicted > 0),
        (expected, 0, true),
        "the kept history"
    );
}

#[test]
fn large_json_keys_count_against_the_kept_history() {
    let db = memory("json_evicted");
    db.execute(
        "CREATE TABLE j (id INTEGER PRIMARY KEY, doc JSON NOT NULL, k INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX j_doc ON j(doc) USING HASH", ())
        .unwrap();
    db.execute("CREATE INDEX j_k ON j (k)", ()).unwrap();
    let text = "y".repeat(4096);
    let rows: Vec<String> = (1..=600)
        .map(|id| format!("({id}, '{{\"p\": \"{text}{id}\"}}', {})", id % 2))
        .collect();
    db.execute(&format!("INSERT INTO j VALUES {}", rows.join(",")), ())
        .unwrap();
    let expected = ids(&db, "SELECT id FROM j WHERE k = 1");
    let reads = fp::removal_history_reads();
    // Six commits of 60 documents of 4 KB each: past the kept history's budget
    let deletes: Vec<String> = (0..6)
        .map(|n| {
            format!(
                "DELETE FROM j WHERE id > {} AND id <= {}",
                n * 100,
                n * 100 + 60
            )
        })
        .collect();
    let (seen, lookups) = overtaken(&db, "SELECT id FROM j WHERE k = 1", deletes);
    let evicted = fp::removal_history_reads()[2] - reads[2];
    assert_eq!(
        (seen, lookups, evicted > 0),
        (expected, 0, true),
        "the documents' bytes evicted the oldest records"
    );
}

#[test]
fn a_commit_failing_on_a_unique_key_keeps_its_removed_key_readable_across_the_rollback() {
    let db = memory("rollback_index");
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, u INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX x_u ON x (u)", ()).unwrap();
    db.execute("INSERT INTO x VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let (ready_tx, ready) = mpsc::channel();
    let (commit_tx, commit) = mpsc::channel::<()>();
    let (removed_tx, removed) = mpsc::channel();
    let (go_tx, go) = mpsc::channel::<()>();
    let (done_tx, done) = mpsc::channel();
    let writer = {
        let other = db.clone();
        std::thread::spawn(move || {
            let mut b = other.begin().unwrap();
            b.execute("UPDATE x SET u = 30 WHERE id = 2", ()).unwrap();
            ready_tx.send(()).unwrap();
            commit.recv_timeout(WAIT).unwrap();
            fp::after_commit_index_removed(move || {
                removed_tx.send(()).unwrap();
                go.recv_timeout(WAIT).unwrap();
            });
            done_tx.send(b.commit().is_err()).unwrap();
        })
    };
    ready.recv_timeout(WAIT).unwrap();
    db.execute("UPDATE x SET u = 30 WHERE id = 1", ()).unwrap();
    fp::after_statement_view_taken(move || {
        commit_tx.send(()).unwrap();
        removed.recv_timeout(WAIT).unwrap();
        // The probe finds the key gone; the commit fails and puts it back
        // before the history is read
        fp::before_removal_history_read(move || {
            go_tx.send(()).unwrap();
            assert!(done.recv_timeout(WAIT).unwrap(), "the commit failed");
        });
    });
    let seen = ids(&db, "SELECT id FROM x WHERE u = 20");
    writer.join().unwrap();
    assert_eq!(seen, [2], "the row the view sees under its key");
}

#[test]
fn a_commit_failing_on_its_wal_marker_keeps_its_removed_key_readable_across_the_rollback() {
    let _serial = fp::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    table(&db);
    let (removed_tx, removed) = mpsc::channel();
    let (go_tx, go) = mpsc::channel::<()>();
    let (done_tx, done) = mpsc::channel();
    let writer: Rc<RefCell<Option<std::thread::JoinHandle<()>>>> = Rc::default();
    {
        let (other, slot) = (db.clone(), Rc::clone(&writer));
        fp::after_statement_view_taken(move || {
            *slot.borrow_mut() = Some(std::thread::spawn(move || {
                fp::after_indexes_published(move || {
                    removed_tx.send(()).unwrap();
                    go.recv_timeout(WAIT).unwrap();
                });
                let failed = other.execute("DELETE FROM t WHERE id = 3", ()).is_err();
                done_tx.send(failed).unwrap();
            }));
            removed.recv_timeout(WAIT).unwrap();
            fp::before_removal_history_read(move || {
                fp::WAL_SYNC_FAIL.store(true, Ordering::Release);
                go_tx.send(()).unwrap();
                let failed = done.recv_timeout(WAIT).unwrap();
                fp::WAL_SYNC_FAIL.store(false, Ordering::Release);
                assert!(failed, "the commit failed");
            });
        });
    }
    let seen = ids(&db, "SELECT id FROM t WHERE k = 2");
    writer.borrow_mut().take().unwrap().join().unwrap();
    assert_eq!(seen, [2, 3], "the row the view sees under its key");
}
