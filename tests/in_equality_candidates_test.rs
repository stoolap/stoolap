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

//! An IN on a B-tree column of a table that keeps rows outside its index
//! reads the members by the table's candidates under one capture, checks
//! each row against the members, and otherwise answers as the scan does

use std::sync::{Mutex, MutexGuard};

use stoolap::Database;

// The index page ledger and the failpoint counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const ROWS: i64 = 20_000;

/// Row id has k = 777 for every tenth id, else id % 1000, and is sealed
/// into a volume whose side file covers k
fn sealed(dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT g.value, \
             CASE WHEN g.value % 10 = 0 THEN 777 ELSE g.value % 1000 END, g.value \
             FROM generate_series(1, {ROWS}) g"
        ),
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

/// The same members by OR, which no IN path reads
fn by_or(db: &Database, keys: &[i64]) -> Vec<i64> {
    let filter: Vec<String> = keys.iter().map(|k| format!("k = {k}")).collect();
    ids(
        db,
        &format!("SELECT id FROM t WHERE {}", filter.join(" OR ")),
    )
}

#[test]
fn an_in_list_on_sealed_keys_answers_as_the_scan() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(&dir);
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE k IN (1, 2, 3)"),
        by_or(&db, &[1, 2, 3])
    );
    assert!(ids(&db, "SELECT id FROM t WHERE k IN (123456, 654321)").is_empty());
    let limited = ids(&db, "SELECT id FROM t WHERE k IN (1, 2) LIMIT 3");
    assert_eq!(limited.len(), 3);
    assert!(limited.iter().all(|id| by_or(&db, &[1, 2]).contains(id)));
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE k IN (1, 2) AND v > 10000"),
        ids(&db, "SELECT id FROM t WHERE (k = 1 OR k = 2) AND v > 10000")
    );
}

#[test]
fn a_key_changed_in_hot_since_the_seal_leaves_the_list() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(&dir);
    db.execute("UPDATE t SET k = 999999 WHERE id = 1001", ())
        .unwrap();
    let ones = ids(&db, "SELECT id FROM t WHERE k IN (1)");
    assert!(
        !ones.contains(&1001),
        "a row whose key moved answered its old key"
    );
    assert_eq!(ones, by_or(&db, &[1]));
    assert_eq!(ids(&db, "SELECT id FROM t WHERE k IN (999999)"), vec![1001]);
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE k IN (1) LIMIT 100"),
        by_or(&db, &[1])
    );
}

#[test]
fn a_transaction_reads_its_own_rows() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(&dir);
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (60001, 3, 0)", ())
        .unwrap();
    assert!(ids(&db, "SELECT id FROM t WHERE k IN (3)").contains(&60001));
    db.execute("ROLLBACK", ()).unwrap();
    assert!(!ids(&db, "SELECT id FROM t WHERE k IN (3)").contains(&60001));
}

#[test]
fn a_nested_conjunct_beside_an_in_subquery_is_kept() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(&dir);
    db.execute("CREATE TABLE s (k INTEGER)", ()).unwrap();
    db.execute("INSERT INTO s VALUES (1)", ()).unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE (k IN (SELECT k FROM s) AND v > 10000) AND id > 0"
        ),
        ids(&db, "SELECT id FROM t WHERE k = 1 AND v > 10000")
    );
}

/// Row 1 has the first timestamp and every other row the second, sealed
/// into a volume whose side file covers ts
fn sealed_timestamps(dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_ts ON t(ts)", ()).unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT g.value, CASE WHEN g.value = 1 \
             THEN TIMESTAMP '2024-01-01 00:00:00' ELSE TIMESTAMP '2024-01-02 00:00:00' END \
             FROM generate_series(1, {ROWS}) g"
        ),
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE s (ts TIMESTAMP)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

#[cfg(feature = "test-failpoints")]
mod probed {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use stoolap::test_failpoints::{
        after_equality_key_probed, in_member_probes, in_member_reads, in_subquery_runs,
    };

    /// The rows of `sql`, with the member reads it made by candidates and
    /// by a scan
    fn read(db: &Database, sql: &str) -> (Vec<i64>, (usize, usize)) {
        let before = in_member_reads();
        let found = ids(db, sql);
        let after = in_member_reads();
        (found, (after.0 - before.0, after.1 - before.1))
    }

    /// Runs `then` once the next multi-key probe has taken its first key,
    /// and says whether it ran
    fn after_first_key(then: impl FnOnce() + 'static) -> Arc<AtomicBool> {
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        after_equality_key_probed(move || {
            flag.store(true, Ordering::SeqCst);
            then();
        });
        ran
    }

    #[test]
    fn the_candidates_read_a_moved_key_by_its_current_value() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("UPDATE t SET k = 999999 WHERE id = 1001", ())
            .unwrap();
        let expected = by_or(&db, &[1, 2]);
        let (found, reads) = read(&db, "SELECT id FROM t WHERE k IN (1, 2)");
        assert_eq!(reads, (1, 0), "the list did not use its candidates");
        assert_eq!(found, expected);
        assert!(!found.contains(&1001));
    }

    #[test]
    fn a_commit_between_two_keys_drops_the_probe() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("INSERT INTO t VALUES (50001, 5000002, 0)", ())
            .unwrap();
        let other = db.clone();
        let probed = after_first_key(move || {
            other
                .execute("UPDATE t SET k = 5000001 WHERE id = 50001", ())
                .unwrap();
        });
        let found = ids(&db, "SELECT id FROM t WHERE k IN (5000001, 5000002)");
        assert!(
            probed.load(Ordering::SeqCst),
            "the list did not read candidates"
        );
        assert_eq!(
            found,
            vec![50001],
            "a row meeting the list at both probes was lost"
        );
    }

    #[test]
    fn a_refused_key_after_answered_keys_falls_back_whole() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        // Left hot, past the candidate cap, so only the reads refuse it
        db.execute(
            "INSERT INTO t SELECT 100000 + g.value, 4242, 0 FROM generate_series(1, 70000) g",
            (),
        )
        .unwrap();
        let expected = by_or(&db, &[1, 2, 4242]);
        let probed = after_first_key(|| {});
        let found = ids(&db, "SELECT id FROM t WHERE k IN (1, 2, 4242)");
        assert!(
            probed.load(Ordering::SeqCst),
            "the list did not read candidates"
        );
        assert_eq!(found, expected);
    }

    #[test]
    fn null_rows_leave_a_sparse_key_to_its_candidates() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
        // 600 rows of 777 are more than a twentieth of the 10,000 keyed rows,
        // not of the volume's 100,000
        db.execute(
            "INSERT INTO t SELECT g.value, CASE WHEN g.value <= 600 THEN 777 \
             WHEN g.value = 601 THEN 0 WHEN g.value <= 10000 THEN 1 + g.value % 776 \
             ELSE NULL END FROM generate_series(1, 100000) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let (found, reads) = read(&db, "SELECT id FROM t WHERE k IN (0, 777)");
        assert_eq!(found.len(), 601);
        assert_eq!(reads, (1, 0), "a sparse key of the volume was refused");
    }

    #[test]
    fn a_heavy_key_refuses_the_list_before_any_key_is_read() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        let expected = by_or(&db, &[1, 2, 777]);
        let probed = after_first_key(|| {});
        let (found, reads) = read(&db, "SELECT id FROM t WHERE k IN (1, 2, 777)");
        assert!(
            !probed.load(Ordering::SeqCst),
            "a key was read before the heavy key refused the list"
        );
        assert_eq!(reads, (0, 0), "the list did not fall back");
        assert_eq!(found, expected);
    }

    #[test]
    fn a_seal_between_two_keys_answers_as_the_scan() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("INSERT INTO t VALUES (50001, 1, 0), (50002, 2, 0)", ())
            .unwrap();
        let expected = by_or(&db, &[1, 2]);
        let other = db.clone();
        let probed = after_first_key(move || {
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        let found = ids(&db, "SELECT id FROM t WHERE k IN (1, 2)");
        assert!(
            probed.load(Ordering::SeqCst),
            "the list did not read candidates"
        );
        assert_eq!(found, expected);
    }

    #[test]
    fn an_in_subquery_that_falls_back_runs_once() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("CREATE TABLE s (k INTEGER)", ()).unwrap();
        db.execute("INSERT INTO s VALUES (1), (2), (777)", ())
            .unwrap();
        let expected = by_or(&db, &[1, 2, 777]);
        db.execute("BEGIN", ()).unwrap();
        let before = in_subquery_runs();
        let (found, reads) = read(&db, "SELECT id FROM t WHERE k IN (SELECT k FROM s)");
        let runs = in_subquery_runs() - before;
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(runs, 1, "the subquery ran again for the fallback");
        assert_eq!(found, expected);
        assert_eq!(reads, (0, 1), "the subquery's members were not scanned");
    }

    #[test]
    fn an_in_subquery_reads_its_members_by_candidates() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("CREATE TABLE s (k INTEGER)", ()).unwrap();
        db.execute("INSERT INTO s VALUES (1), (2), (5)", ())
            .unwrap();
        let expected = by_or(&db, &[1, 2, 5]);
        let before = in_subquery_runs();
        let (found, reads) = read(&db, "SELECT id FROM t WHERE k IN (SELECT k FROM s)");
        assert_eq!(found, expected);
        assert_eq!(
            reads,
            (1, 0),
            "the subquery's members did not use candidates"
        );
        assert_eq!(in_subquery_runs() - before, 1);
    }

    #[test]
    fn a_semi_join_set_reads_its_members_by_candidates() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO s VALUES (1, 1), (2, 2), (3, 5)", ())
            .unwrap();
        let expected = by_or(&db, &[1, 2, 5]);
        let (found, reads) = read(
            &db,
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM s WHERE s.k = t.k)",
        );
        assert_eq!(found, expected);
        assert_eq!(reads, (1, 0), "the semi-join set did not use candidates");
    }

    /// The rows of `sql`, with the IN members it asked of the candidates
    fn probed(db: &Database, sql: &str) -> (Vec<i64>, usize) {
        let before = in_member_probes();
        let found = ids(db, sql);
        (found, in_member_probes() - before)
    }

    #[test]
    fn an_in_list_under_a_limit_opens_no_probe() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        let all = by_or(&db, &[1, 2]);
        let (found, probes) = probed(&db, "SELECT id FROM t WHERE k IN (1, 2) LIMIT 3");
        assert_eq!(probes, 0, "a list under a LIMIT asked the candidates");
        assert_eq!(found.len(), 3);
        assert!(found.iter().all(|id| all.contains(id)));
        let (found, reads) = read(&db, "SELECT id FROM t WHERE k IN (1, 2)");
        assert_eq!(reads, (1, 0), "the list did not use its candidates");
        assert_eq!(found, all);
    }

    #[test]
    fn an_in_subquery_under_a_limit_opens_no_probe_and_runs_once() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("CREATE TABLE s (k INTEGER)", ()).unwrap();
        db.execute("INSERT INTO s VALUES (1), (2), (5)", ())
            .unwrap();
        let all = by_or(&db, &[1, 2, 5]);
        db.execute("BEGIN", ()).unwrap();
        let runs = in_subquery_runs();
        let (found, probes) = probed(&db, "SELECT id FROM t WHERE k IN (SELECT k FROM s) LIMIT 2");
        let runs = in_subquery_runs() - runs;
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(probes, 0, "a subquery under a LIMIT asked the candidates");
        assert_eq!(runs, 1, "the subquery ran again");
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|id| all.contains(id)));
    }

    #[test]
    fn a_semi_join_set_under_a_limit_opens_no_probe() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO s VALUES (1, 1), (2, 2), (3, 5)", ())
            .unwrap();
        let all = by_or(&db, &[1, 2, 5]);
        let (found, probes) = probed(
            &db,
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM s WHERE s.k = t.k) LIMIT 2",
        );
        assert_eq!(
            probes, 0,
            "a semi-join set under a LIMIT asked the candidates"
        );
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|id| all.contains(id)));
    }

    #[test]
    fn a_timestamp_member_meets_its_rows_by_candidates() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed_timestamps(&dir);
        db.execute("INSERT INTO s VALUES (TIMESTAMP '2024-01-01 00:00:00')", ())
            .unwrap();
        let (found, reads) = read(&db, "SELECT id FROM t WHERE ts IN (SELECT ts FROM s)");
        assert_eq!(
            reads,
            (1, 0),
            "the subquery's members did not use candidates"
        );
        assert_eq!(found, vec![1], "a timestamp member lost its row");
    }

    #[test]
    fn a_timestamp_member_meets_its_rows_by_the_scan() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed_timestamps(&dir);
        db.execute(
            "INSERT INTO s VALUES (TIMESTAMP '2024-01-01 00:00:00'), \
             (TIMESTAMP '2024-01-02 00:00:00')",
            (),
        )
        .unwrap();
        let (found, reads) = read(&db, "SELECT id FROM t WHERE ts IN (SELECT ts FROM s)");
        assert_eq!(reads, (0, 1), "the subquery's members were not scanned");
        assert_eq!(
            found.len(),
            ROWS as usize,
            "a timestamp member lost its rows"
        );
    }

    #[test]
    fn a_conjunct_subquery_runs_once_when_the_list_falls_back() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("CREATE TABLE s (v INTEGER)", ()).unwrap();
        db.execute("INSERT INTO s VALUES (1), (2), (10)", ())
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        let (before, asked) = (in_subquery_runs(), in_member_probes());
        let (found, reads) = read(
            &db,
            "SELECT id FROM t WHERE k IN (1, 2, 777) AND v IN (SELECT v FROM s)",
        );
        let (runs, asked) = (in_subquery_runs() - before, in_member_probes() - asked);
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(asked, 1, "the list did not ask the candidates");
        assert_eq!(reads, (0, 0), "the list did not fall back");
        assert_eq!(found, vec![1, 2, 10]);
        assert_eq!(
            runs, 1,
            "the conjunct's subquery ran again for the fallback"
        );
    }
}
