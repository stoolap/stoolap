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

use stoolap::Database;

fn values(db: &Database) -> Vec<i64> {
    db.query("SELECT v FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn failed_multirow_insert_preserves_prior_statements_and_savepoints() {
    let db = Database::open("memory://statement_insert_undo").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("SAVEPOINT earlier", ()).unwrap();
    assert!(db
        .execute("INSERT INTO t VALUES (2, 20), (1, 99)", ())
        .is_err());
    assert_eq!(values(&db), vec![10]);
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    db.execute("ROLLBACK TO earlier", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(values(&db), vec![10]);
}

#[test]
fn cached_insert_and_transaction_api_rollback_a_failed_prefix() {
    let db = Database::open("memory://statement_cached_insert_undo").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    let sql = "INSERT INTO t VALUES ($1, $2), ($3, $4)";
    tx.execute(sql, (1, 10, 2, 20)).unwrap();
    assert!(tx.execute(sql, (3, 30, 1, 99)).is_err());
    tx.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
    tx.commit().unwrap();
    assert_eq!(values(&db), vec![11, 20]);
}

#[test]
fn cascade_failure_keeps_all_tables_and_prior_statement() {
    let db = Database::open("memory://statement_cascade_undo").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON DELETE CASCADE)",
        "CREATE TABLE g (id INTEGER PRIMARY KEY, cid INTEGER REFERENCES c(id) ON DELETE RESTRICT)",
        "INSERT INTO p VALUES (1)",
        "INSERT INTO c VALUES (1, 1), (2, 1)",
        "INSERT INTO g VALUES (1, 2)",
        "BEGIN",
        "INSERT INTO p VALUES (99)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert!(db.execute("DELETE FROM p WHERE id = 1", ()).is_err());
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM p", ())
            .unwrap(),
        2
    );
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM c", ())
            .unwrap(),
        2
    );
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM g", ())
            .unwrap(),
        1
    );
}

#[cfg(feature = "test-failpoints")]
mod failpoints {
    use super::*;
    use std::sync::atomic::Ordering;
    use stoolap::test_failpoints;

    #[test]
    fn second_table_publication_failure_restores_hot_cold_and_indexes() {
        let _guard = test_failpoints::FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!("file://{}", dir.path().display());
        let db = Database::open(&dsn).unwrap();
        for table in ["a", "b"] {
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, v INTEGER UNIQUE)"),
                (),
            )
            .unwrap();
            db.execute(&format!("INSERT INTO {table} VALUES (1, 10), (2, 20)"), ())
                .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("BEGIN", ()).unwrap();
        for table in ["a", "b"] {
            db.execute(&format!("UPDATE {table} SET v = 11 WHERE id = 1"), ())
                .unwrap();
            db.execute(&format!("DELETE FROM {table} WHERE id = 2"), ())
                .unwrap();
            db.execute(&format!("INSERT INTO {table} VALUES (3, 30)"), ())
                .unwrap();
        }
        test_failpoints::fail_table_publish_on(2);
        assert!(db.execute("COMMIT", ()).is_err());
        for table in ["a", "b"] {
            assert_eq!(
                db.query_one::<i64, _>(&format!("SELECT SUM(v) FROM {table}"), ())
                    .unwrap(),
                30
            );
            db.execute(&format!("INSERT INTO {table} VALUES (3, 30)"), ())
                .unwrap();
            assert!(db
                .execute(&format!("INSERT INTO {table} VALUES (4, 10)"), ())
                .is_err());
        }
        db.close().unwrap();
        let db = Database::open(&dsn).unwrap();
        for table in ["a", "b"] {
            assert_eq!(
                db.query_one::<i64, _>(&format!("SELECT SUM(v) FROM {table}"), ())
                    .unwrap(),
                60
            );
        }
    }

    #[test]
    fn nth_cold_read_failure_discards_update_and_delete_prefixes() {
        let _guard = test_failpoints::FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER UNIQUE)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let mut failures = 0;
        for sql in [
            "UPDATE t SET v = v + 100 WHERE id IN (1, 2, 3)",
            "DELETE FROM t WHERE id IN (1, 2, 3)",
        ] {
            for nth in 1..=32 {
                let mut tx = db.begin().unwrap();
                tx.execute("INSERT INTO t VALUES (99, 99)", ()).unwrap();
                test_failpoints::fail_cold_read_on(nth);
                let result = tx.execute(sql, ());
                test_failpoints::fail_cold_read_on(0);
                if result.is_err() {
                    failures += 1;
                    assert_eq!(
                        tx.query_one::<i64, _>("SELECT SUM(v) FROM t WHERE id < 99", ())
                            .unwrap(),
                        60,
                        "failed statement left a prefix after read {nth}: {sql}"
                    );
                    tx.commit().unwrap();
                    assert_eq!(
                        db.query_one::<i64, _>("SELECT v FROM t WHERE id = 99", ())
                            .unwrap(),
                        99
                    );
                    db.execute("DELETE FROM t WHERE id = 99", ()).unwrap();
                } else {
                    tx.rollback().unwrap();
                }
                // A second transaction can reclaim every cold row after failure.
                let mut probe = db.begin().unwrap();
                probe
                    .execute("UPDATE t SET v = v + 1 WHERE id = 1", ())
                    .unwrap();
                probe.rollback().unwrap();
            }
        }
        assert!(
            failures > 2,
            "must exercise several cold reads, including later rows"
        );
    }

    #[test]
    fn failed_commit_marker_restores_all_published_tables() {
        let _guard = test_failpoints::FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db =
            Database::open(&format!("file://{}?sync_mode=full", dir.path().display())).unwrap();
        for table in ["a", "b"] {
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, v INTEGER UNIQUE)"),
                (),
            )
            .unwrap();
            db.execute(&format!("INSERT INTO {table} VALUES (1, 10)"), ())
                .unwrap();
        }
        db.execute("BEGIN", ()).unwrap();
        for table in ["a", "b"] {
            db.execute(&format!("UPDATE {table} SET v = 20 WHERE id = 1"), ())
                .unwrap();
        }
        test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);
        assert!(db.execute("COMMIT", ()).is_err());
        test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);
        // A definite WAL failure permits reads, but the poisoned WAL rejects writes.
        for table in ["a", "b"] {
            assert_eq!(
                db.query_one::<i64, _>(&format!("SELECT SUM(v) FROM {table}"), ())
                    .unwrap(),
                10
            );
        }
    }
}
