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

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::prepared_scalar_runs;
use stoolap::Database;

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "INSERT INTO t VALUES (1, 10, 'x'), (2, 10, 'y'), (3, 20, 'x'), (4, NULL, 'z'), (5, 20, 'x')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO u VALUES (1, 1, 3), (2, 2, 30), (3, 3, 10), (4, 5, NULL)",
        "CREATE VIEW tv AS SELECT id, a, b FROM t",
        "CREATE TABLE d (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO d VALUES (1, 1), (2, 1), (3, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Each row as text, NULL as written
fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".to_string())
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

#[test]
fn a_distinct_statement_or_a_set_operation_branch_runs_prepared() {
    let _guard = serial();
    let db = setup("distinct_set_operation_answers");
    for (sql, want) in [
        (
            "SELECT DISTINCT a FROM t WHERE id > (SELECT MAX(x) FROM u WHERE x < 5) ORDER BY a",
            &["20", "NULL"][..],
        ),
        (
            "SELECT DISTINCT a FROM t WHERE a > (SELECT x FROM u WHERE u.k = t.id) ORDER BY a",
            &["10", "20"],
        ),
        (
            "SELECT DISTINCT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id) ORDER BY a",
            &["10", "20"],
        ),
        (
            "SELECT DISTINCT b FROM t WHERE id < (SELECT MAX(k) FROM u) ORDER BY b LIMIT 1 OFFSET 1",
            &["y"],
        ),
        (
            "SELECT DISTINCT (SELECT x FROM u WHERE u.k = t.id) FROM t ORDER BY 1",
            &["3", "10", "30", "NULL"],
        ),
        (
            "SELECT DISTINCT a + (SELECT MIN(x) FROM u) AS s FROM t ORDER BY s",
            &["13", "23", "NULL"],
        ),
        (
            "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u WHERE x < 5) UNION SELECT 9 ORDER BY 1",
            &["4", "5", "9"],
        ),
        (
            "SELECT a FROM t WHERE id IN (SELECT k FROM u) EXCEPT SELECT a FROM t WHERE a = 10 ORDER BY 1",
            &["20"],
        ),
        (
            "SELECT id, (SELECT x FROM u WHERE u.k = t.id) FROM t UNION ALL SELECT 9, 9 ORDER BY 1",
            &["1|3", "2|30", "3|10", "4|NULL", "5|NULL", "9|9"],
        ),
        (
            "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u WHERE x < 5) UNION SELECT 9 ORDER BY 1 DESC LIMIT 2",
            &["9", "5"],
        ),
        (
            "SELECT DISTINCT a FROM t WHERE id > (SELECT MAX(x) FROM u WHERE x < 5) UNION SELECT 7 ORDER BY 1",
            &["7", "20", "NULL"],
        ),
        // A view, and a correlated block over a VALUES list, both failed here
        (
            "SELECT DISTINCT a FROM tv WHERE id > (SELECT MIN(x) FROM u) ORDER BY a",
            &["20", "NULL"],
        ),
        (
            "SELECT DISTINCT (SELECT v.x + t.a FROM (VALUES (1, 5), (3, 7)) AS v(k, x) \
             WHERE v.k = t.id) FROM t ORDER BY 1",
            &["15", "27", "NULL"],
        ),
        (
            "SELECT (SELECT v.x FROM (VALUES (1, 5), (3, 7)) AS v(k, x) WHERE v.k = t.id) \
             FROM t UNION SELECT 100 ORDER BY 1",
            &["5", "7", "100", "NULL"],
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        assert_eq!(rows(&db, sql), want, "{sql}");
    }
}

#[test]
fn distinct_reads_every_row_before_its_limit_counts_them() {
    let _guard = serial();
    let db = setup("distinct_set_operation_limits");
    // The first two rows hold one value: cutting before DISTINCT keeps one
    let sql = "SELECT DISTINCT g FROM d WHERE id > (SELECT MIN(x) FROM u) - 3 LIMIT 2";
    assert_eq!(prepared_line(&db, sql), "Prepared: yes");
    assert_eq!(rows(&db, sql), ["1", "2"]);
    let sql = "SELECT DISTINCT g FROM d WHERE id > (SELECT MIN(x) FROM u) - 3 LIMIT 1 OFFSET 1";
    assert_eq!(rows(&db, sql), ["2"]);
    // A UNION ALL takes LIMIT rows at most from each branch
    let sql = "SELECT id FROM d WHERE id > (SELECT MIN(x) FROM u) - 3 UNION ALL SELECT 9 LIMIT 2";
    assert_eq!(prepared_line(&db, sql), "Prepared: yes");
    assert_eq!(rows(&db, sql), ["1", "2"]);
}

#[test]
fn a_branch_runs_its_correlated_block_for_each_row_and_raises_its_error() {
    let _guard = serial();
    let db = setup("distinct_set_operation_runs");
    let sql = "SELECT DISTINCT (SELECT x FROM u WHERE u.k = t.id) FROM t \
               UNION SELECT 1 ORDER BY 1";
    let runs = prepared_scalar_runs();
    assert_eq!(rows(&db, sql), ["1", "3", "10", "30", "NULL"]);
    assert_eq!(prepared_scalar_runs() - runs, 5, "one run per row of t");
    for sql in [
        "SELECT DISTINCT a FROM t WHERE id > (SELECT x FROM u) ORDER BY a",
        "SELECT id FROM t WHERE id > (SELECT x FROM u) UNION SELECT 9",
    ] {
        let err = match db.query(sql, ()) {
            Err(e) => e.to_string(),
            Ok(found) => found
                .collect::<Result<Vec<_>, _>>()
                .expect_err("expected an error")
                .to_string(),
        };
        assert!(err.contains("more than one row"), "{sql}: {err}");
    }
}
