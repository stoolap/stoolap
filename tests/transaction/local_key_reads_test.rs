// Copyright 2026 Stoolap Contributors
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

//! An indexed read of a transaction's own rows finds them by their keys

use stoolap::api::Transaction;
use stoolap::Database;

/// Rows written in the transaction, past the count it keeps keys from
const ROWS: i64 = 200;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, email TEXT UNIQUE, k INTEGER, a INTEGER, b INTEGER)",
        "CREATE INDEX t_k ON t (k)",
        "CREATE INDEX t_ab ON t (a, b)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// A transaction holding rows 1..=ROWS, `k` = id, `a` = id % 10, `b` = id
fn written(db: &Database) -> Transaction {
    let mut tx = db.begin().unwrap();
    for id in 1..=ROWS {
        tx.execute(
            &format!(
                "INSERT INTO t VALUES ({id}, 'u{id}', {id}, {}, {id})",
                id % 10
            ),
            (),
        )
        .unwrap();
    }
    tx
}

fn ids(tx: &mut Transaction, sql: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = tx
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn reads_by_each_kind_of_key() {
    let db = open("local_keys_kinds");
    let mut tx = written(&db);
    for (sql, expected) in [
        ("SELECT id FROM t WHERE k = 150", vec![150]),
        ("SELECT id FROM t WHERE email = 'u150'", vec![150]),
        ("SELECT id FROM t WHERE k IN (3, 150, 999)", vec![3, 150]),
        ("SELECT id FROM t WHERE a = 7 AND b = 27", vec![27]),
        ("SELECT id FROM t WHERE k = 3 OR email = 'u4'", vec![3, 4]),
        ("SELECT id FROM t WHERE k = 3 AND email = 'u3'", vec![3]),
        ("SELECT id FROM t WHERE k = 3 AND email = 'u4'", vec![]),
        ("SELECT id FROM t WHERE k = 150.0", vec![150]),
        ("SELECT id FROM t WHERE k = '150'", vec![150]),
        ("SELECT id FROM t WHERE k = NULL", vec![]),
        ("SELECT id FROM t WHERE k = 999", vec![]),
    ] {
        assert_eq!(ids(&mut tx, sql), expected, "{sql}");
    }
    let found: Vec<i64> = tx
        .query("SELECT id FROM t WHERE k = $1", (42,))
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(found, vec![42], "a parameter");
    let found: Vec<i64> = tx
        .query("SELECT id FROM t WHERE k = $1", ("42",))
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(found, vec![42], "a TEXT parameter for an INTEGER column");
}

#[test]
fn reads_after_each_kind_of_write() {
    let db = open("local_keys_writes");
    let mut tx = written(&db);
    // The first read builds the keys; the writes after it keep them
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 7"), vec![7]);
    for sql in [
        "INSERT INTO t VALUES (500, 'u500', 7, 0, 500)",
        "UPDATE t SET k = 1000 WHERE id = 8",
        "DELETE FROM t WHERE id = 9",
        "UPDATE t SET id = id + 10000 WHERE id = 10",
        "UPDATE t SET email = 'moved' WHERE id = 11",
    ] {
        tx.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    for (sql, expected) in [
        ("SELECT id FROM t WHERE k = 7", vec![7, 500]),
        ("SELECT id FROM t WHERE k = 8", vec![]),
        ("SELECT id FROM t WHERE k = 1000", vec![8]),
        ("SELECT id FROM t WHERE k = 9", vec![]),
        ("SELECT id FROM t WHERE k = 10", vec![10010]),
        ("SELECT id FROM t WHERE email = 'u11'", vec![]),
        ("SELECT id FROM t WHERE email = 'moved'", vec![11]),
    ] {
        assert_eq!(ids(&mut tx, sql), expected, "{sql}");
    }
    tx.commit().unwrap();
    let mut after = db.begin().unwrap();
    assert_eq!(
        ids(&mut after, "SELECT id FROM t WHERE k = 7"),
        vec![7, 500]
    );
}

#[test]
fn writes_find_their_rows_by_key() {
    let db = open("local_keys_dml");
    let mut tx = written(&db);
    assert_eq!(
        tx.execute("UPDATE t SET b = -1 WHERE k = 150", ()).unwrap(),
        1
    );
    assert_eq!(
        tx.execute("DELETE FROM t WHERE email = 'u151'", ())
            .unwrap(),
        1
    );
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE b = -1"), vec![150]);
    assert!(ids(&mut tx, "SELECT id FROM t WHERE k = 151").is_empty());
}

#[test]
fn reads_after_a_rollback_to_a_savepoint() {
    let db = open("local_keys_savepoint");
    let mut tx = written(&db);
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 20"), vec![20]);
    tx.execute("SAVEPOINT s", ()).unwrap();
    tx.execute("UPDATE t SET k = 2000 WHERE id = 20", ())
        .unwrap();
    tx.execute("INSERT INTO t VALUES (600, 'u600', 20, 0, 600)", ())
        .unwrap();
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 2000"), vec![20]);
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 20"), vec![600]);
    tx.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 20"), vec![20]);
    assert!(ids(&mut tx, "SELECT id FROM t WHERE k = 2000").is_empty());
}

#[test]
fn reads_after_a_failed_statement() {
    let db = open("local_keys_failed");
    let mut tx = written(&db);
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 30"), vec![30]);
    // The second row's key is taken: the statement leaves nothing
    let failed = tx.execute(
        "INSERT INTO t VALUES (700, 'u700', 30, 0, 700), (1, 'u1', 1, 1, 1)",
        (),
    );
    assert!(failed.is_err());
    // Both rows take one unique email: the statement fails after a write
    tx.execute(
        "UPDATE t SET k = 3000, email = 'taken' WHERE id = 30 OR id = 31",
        (),
    )
    .unwrap_err();
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 30"), vec![30]);
    assert!(ids(&mut tx, "SELECT id FROM t WHERE k = 3000").is_empty());
}

#[test]
fn a_statement_reading_its_own_table_reads_it_as_it_began() {
    let db = open("local_keys_cut");
    let mut tx = written(&db);
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 40"), vec![40]);
    tx.execute(
        "INSERT INTO t SELECT id + 100000, email || 'x', k, a, b FROM t WHERE k = 40",
        (),
    )
    .unwrap();
    assert_eq!(
        ids(&mut tx, "SELECT id FROM t WHERE k = 40"),
        vec![40, 100040]
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_read_by_key_takes_only_the_rows_holding_it() {
    use stoolap::test_failpoints::local_rows_added;
    let db = open("local_keys_cost");
    let mut tx = written(&db);
    assert_eq!(ids(&mut tx, "SELECT id FROM t WHERE k = 150"), vec![150]);
    for (sql, expected) in [
        ("SELECT id FROM t WHERE k = 150", 1),
        ("SELECT id FROM t WHERE email = 'u150'", 1),
        ("SELECT id FROM t WHERE k IN (3, 150)", 2),
    ] {
        let before = local_rows_added();
        assert_eq!(ids(&mut tx, sql).len(), expected, "{sql}");
        assert_eq!(local_rows_added() - before, expected, "{sql}");
    }
}
