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

//! SQL a transaction runs a third time is answered from the statement it
//! parsed on the second run, with the same results as parsing it again

use stoolap::core::IsolationLevel;
use stoolap::{named_params, Database};

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db
}

fn ids(tx: &mut stoolap::ApiTransaction, sql: &str) -> Vec<i64> {
    tx.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
}

#[test]
fn repeated_sql_takes_each_call_s_positional_parameters() {
    let db = open("tx_parse_cache_positional");
    let mut tx = db.begin().unwrap();
    for (id, v) in [(1, 10), (3, 30), (2, 20), (1, 10)] {
        let got: i64 = tx
            .query_one("SELECT v FROM t WHERE id = $1", (id,))
            .unwrap();
        assert_eq!(got, v, "id {id}");
    }
    tx.commit().unwrap();
}

#[test]
fn repeated_sql_takes_each_call_s_named_parameters() {
    let db = open("tx_parse_cache_named");
    let mut tx = db.begin().unwrap();
    for id in [2i64, 3, 1] {
        let v: i64 = tx
            .query_named("SELECT v FROM t WHERE id = :id", named_params! { id: id })
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(v, id * 10, "id {id}");
    }
    for id in [1i64, 2, 1] {
        tx.execute_named(
            "UPDATE t SET v = v + 1 WHERE id = :id",
            named_params! { id: id },
        )
        .unwrap();
    }
    assert_eq!(
        ids(&mut tx, "SELECT v FROM t ORDER BY id"),
        vec![12, 21, 30]
    );
    tx.commit().unwrap();
}

#[test]
fn repeated_sql_sees_the_transaction_s_own_writes() {
    let db = open("tx_parse_cache_own_writes");
    let mut tx = db.begin().unwrap();
    let count = "SELECT COUNT(*) FROM t";
    for n in 4..=6 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (n, n * 10))
            .unwrap();
        let rows: i64 = tx.query_one(count, ()).unwrap();
        assert_eq!(rows, n);
    }
    tx.execute("DELETE FROM t WHERE id = $1", (5,)).unwrap();
    assert_eq!(tx.query_one::<i64, _>(count, ()).unwrap(), 5);
    tx.rollback().unwrap();
    assert_eq!(db.query_one::<i64, _>(count, ()).unwrap(), 3);
}

#[test]
fn repeated_sql_keeps_the_snapshot() {
    let db = open("tx_parse_cache_snapshot");
    let mut tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let sql = "SELECT id FROM t ORDER BY id";
    assert_eq!(ids(&mut tx, sql), vec![1, 2, 3]);
    assert_eq!(ids(&mut tx, sql), vec![1, 2, 3]);
    db.execute("INSERT INTO t VALUES (4, 40)", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(ids(&mut tx, sql), vec![1, 2, 3]);
    tx.commit().unwrap();
}

#[test]
fn repeated_sql_follows_savepoints() {
    let db = open("tx_parse_cache_savepoint");
    let mut tx = db.begin().unwrap();
    let bump = "UPDATE t SET v = v + 1 WHERE id = $1";
    let sum = "SELECT SUM(v) FROM t";
    tx.execute(bump, (1,)).unwrap();
    tx.execute("SAVEPOINT s", ()).unwrap();
    tx.execute(bump, (1,)).unwrap();
    tx.execute(bump, (2,)).unwrap();
    assert_eq!(tx.query_one::<i64, _>(sum, ()).unwrap(), 63);
    tx.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    assert_eq!(tx.query_one::<i64, _>(sum, ()).unwrap(), 61);
    tx.execute("SAVEPOINT s", ()).unwrap();
    tx.execute(bump, (3,)).unwrap();
    tx.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    assert_eq!(tx.query_one::<i64, _>(sum, ()).unwrap(), 61);
    tx.execute("SAVEPOINT s", ()).unwrap();
    tx.execute(bump, (2,)).unwrap();
    assert_eq!(tx.query_one::<i64, _>(sum, ()).unwrap(), 62);
    tx.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    assert_eq!(tx.query_one::<i64, _>(sum, ()).unwrap(), 61);
    tx.commit().unwrap();
    assert_eq!(db.query_one::<i64, _>(sum, ()).unwrap(), 61);
}

#[test]
fn repeated_sql_runs_on_after_a_failed_statement() {
    let db = open("tx_parse_cache_after_error");
    let mut tx = db.begin().unwrap();
    let insert = "INSERT INTO t VALUES ($1, $2)";
    assert!(tx.execute(insert, (1, 99)).is_err(), "duplicate key");
    tx.execute(insert, (4, 40)).unwrap();
    assert!(tx.execute(insert, (4, 41)).is_err(), "duplicate key again");
    tx.execute(insert, (5, 50)).unwrap();
    for _ in 0..3 {
        assert!(tx.query("SELECT missing FROM t", ()).is_err());
    }
    tx.commit().unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT SUM(v) FROM t", ()).unwrap(),
        150
    );
}

#[test]
fn empty_and_multi_statement_sql_behave_as_before() {
    let db = open("tx_parse_cache_programs");
    let mut tx = db.begin().unwrap();
    for _ in 0..2 {
        assert!(tx.execute("", ()).is_err());
        assert!(tx.execute("  ", ()).is_err());
    }
    let both = "UPDATE t SET v = v + 1 WHERE id = 1; UPDATE t SET v = v + 100 WHERE id = 2";
    tx.execute(both, ()).unwrap();
    tx.execute(both, ()).unwrap();
    assert_eq!(
        ids(&mut tx, "SELECT v FROM t ORDER BY id"),
        vec![12, 220, 30]
    );
    tx.commit().unwrap();
}

#[test]
fn repeated_sql_reads_the_schema_a_new_parse_would() {
    let db = open("tx_parse_cache_schema");
    let cached = "SELECT * FROM t WHERE id = 1";
    {
        let mut tx = db.begin().unwrap();
        for _ in 0..2 {
            let rows = tx.query(cached, ()).unwrap();
            assert_eq!(rows.columns().len(), 2);
        }
        db.execute("ALTER TABLE t ADD COLUMN w INTEGER DEFAULT 7", ())
            .unwrap();
        let rows = tx.query(cached, ()).unwrap();
        let again = rows.columns().to_vec();
        drop(rows);
        assert_eq!(again.len(), 3, "the cached statement missed the new column");
        let fresh = tx
            .query("SELECT  * FROM t WHERE id = 1", ())
            .unwrap()
            .columns()
            .to_vec();
        assert_eq!(again, fresh, "the cached statement resolved another schema");
        tx.commit().unwrap();
    }
}
