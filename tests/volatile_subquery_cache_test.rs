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

//! A subquery that calls a non-deterministic function is evaluated again in
//! each statement

use std::collections::HashSet;

use stoolap::{Database, Value};

const RUNS: usize = 20;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (bid INTEGER PRIMARY KEY, bv INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO a SELECT g.value, g.value FROM generate_series(1, 100) g",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO b SELECT g.value, g.value * 10 FROM generate_series(1, 100) g",
        (),
    )
    .unwrap();
    db
}

/// The distinct first cells `sql` gives over `RUNS` statements
fn answers(db: &Database, sql: &str) -> HashSet<String> {
    (0..RUNS)
        .map(|_| {
            let mut rows = db.query(sql, ()).unwrap();
            let row = rows.next().expect("one row").unwrap();
            format!("{:?}", row.get_value(0).unwrap())
        })
        .collect()
}

#[test]
fn test_a_non_deterministic_subquery_runs_in_each_statement() {
    let db = setup("volatile_subquery_each_statement");
    let stuck: Vec<&str> = [
        "SELECT (SELECT RANDOM() FROM b LIMIT 1)",
        "SELECT (SELECT bid FROM b ORDER BY RANDOM() LIMIT 1)",
        "SELECT (SELECT MAX(bid) FROM b WHERE RANDOM() < 0.5)",
        "SELECT (SELECT COUNT(*) FROM (SELECT bid FROM b WHERE RANDOM() < 0.5) d)",
        "SELECT COUNT(*) FROM a WHERE id IN (SELECT bid FROM b WHERE RANDOM() < 0.5)",
        "SELECT COUNT(*) FROM a WHERE x IN (SELECT bid FROM b WHERE RANDOM() < 0.5)",
        "SELECT COUNT(*) FROM a WHERE id NOT IN (SELECT bid FROM b WHERE RANDOM() < 0.5)",
        "SELECT COUNT(*) FROM a WHERE EXISTS \
         (SELECT 1 FROM b WHERE b.bid = a.id AND RANDOM() < 0.5)",
        "SELECT COUNT(*) FROM a WHERE NOT EXISTS \
         (SELECT 1 FROM b WHERE b.bid = a.id AND RANDOM() < 0.5)",
    ]
    .into_iter()
    .filter(|sql| answers(&db, sql).len() == 1)
    .collect();
    assert!(
        stuck.is_empty(),
        "one answer over {RUNS} statements: {stuck:#?}"
    );
}

#[test]
fn test_a_non_deterministic_scalar_subquery_is_one_value_in_a_statement() {
    let db = setup("volatile_subquery_one_statement");
    let values: HashSet<String> = db
        .query("SELECT (SELECT RANDOM() FROM b LIMIT 1) FROM a", ())
        .unwrap()
        .map(|row| format!("{:?}", row.unwrap().get_value(0).unwrap()))
        .collect();
    assert_eq!(values.len(), 1, "{values:?}");
}

#[test]
fn test_a_deterministic_subquery_keeps_its_answer() {
    let db = setup("volatile_subquery_deterministic");
    for sql in [
        "SELECT (SELECT MAX(bid) FROM b)",
        "SELECT COUNT(*) FROM a WHERE id IN (SELECT bid FROM b WHERE bv > 500)",
    ] {
        let found = answers(&db, sql);
        assert_eq!(found.len(), 1, "{sql}: {found:?}");
    }
    assert_eq!(
        answers(&db, "SELECT (SELECT MAX(bid) FROM b)"),
        HashSet::from([format!("{:?}", Value::Integer(100))])
    );
}

#[test]
fn test_a_subquery_reading_a_cte_of_its_statement_runs_in_each_statement() {
    let db = setup("volatile_subquery_cte");
    for (n, expected) in [(1, Value::Integer(1)), (2, Value::Integer(2))] {
        let sql = format!("WITH c(r) AS (SELECT {n}) SELECT (SELECT r FROM c)");
        assert_eq!(
            answers(&db, &sql),
            HashSet::from([format!("{expected:?}")]),
            "{sql}"
        );
    }
    let sql = "WITH c(r) AS (SELECT RANDOM()) SELECT (SELECT r FROM c)";
    assert!(answers(&db, sql).len() > 1, "{sql} gave one answer");
}

#[test]
fn test_a_non_deterministic_as_of_runs_in_each_statement() {
    let db = setup("volatile_subquery_as_of");
    let sql = "SELECT (SELECT MAX(bid) FROM b AS OF TIMESTAMP CASE WHEN RANDOM() < 0.5 \
               THEN '1970-01-01 00:00:00' ELSE '2099-01-01 00:00:00' END)";
    assert!(answers(&db, sql).len() > 1, "{sql} gave one answer");
}
