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

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::in_subquery_runs;
use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX t_a ON t (a)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1), (2, NULL)", ())
        .unwrap();
    db
}

/// The ids the statement gives, and the IN subqueries it ran for them
fn run(db: &Database, sql: &str) -> (Vec<i64>, usize) {
    let before = in_subquery_runs();
    let found = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    (found, in_subquery_runs() - before)
}

#[test]
fn test_not_in_over_a_set_holding_null_runs_the_subquery_once() {
    let db = setup("not_in_null_once");
    for (column, sql) in [
        (
            "primary key",
            "SELECT id FROM t WHERE id NOT IN (SELECT k FROM u)",
        ),
        ("index", "SELECT id FROM t WHERE a NOT IN (SELECT k FROM u)"),
        (
            "primary key beside a conjunct",
            "SELECT id FROM t WHERE id NOT IN (SELECT k FROM u) AND a > 0",
        ),
    ] {
        assert_eq!(
            run(&db, sql),
            (vec![], 1),
            "{column}: a NULL member keeps no row, and the subquery runs once"
        );
    }
}

#[test]
fn test_not_in_without_null_keeps_its_answer() {
    let db = setup("not_in_without_null");
    db.execute("DELETE FROM u WHERE k IS NULL", ()).unwrap();
    let (found, runs) = run(
        &db,
        "SELECT id FROM t WHERE id NOT IN (SELECT k FROM u) ORDER BY id",
    );
    assert_eq!(found, vec![2, 3]);
    assert!(runs <= 1, "ran the subquery {runs} times");
}
