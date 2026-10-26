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

//! A statement without FROM reads one row without columns, and its
//! subqueries answer as their values written as literals do

use stoolap::core::Value;
use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://fromless_statement_{name}")).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db
}

/// The column names and every value, or the error's text
fn answer(rows: stoolap::Result<stoolap::Rows>) -> Result<(Vec<String>, Vec<Vec<Value>>), String> {
    let rows = rows.map_err(|e| e.to_string())?;
    let names = rows.columns().to_vec();
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..names.len())
                .map(|i| row.get_value(i).unwrap().clone())
                .collect(),
        );
    }
    Ok((names, out))
}

/// The values a twin answers, whose names are its own literals
fn alike(db: &Database, sql: &str, twin: &str) {
    let values = |sql: &str| answer(db.query(sql, ())).map(|(_, values)| values);
    assert_eq!(values(sql), values(twin), "{sql}");
}

#[test]
fn a_statement_without_from_names_its_items_as_written() {
    let db = open("names");
    for (sql, names) in [
        (
            "SELECT (SELECT MAX(x) FROM u)",
            vec!["(SELECT MAX(x) FROM u)"],
        ),
        ("SELECT (SELECT MAX(x) FROM u) AS m, 1", vec!["m", "1"]),
        (
            "SELECT 1 + (SELECT MAX(x) FROM u), EXISTS (SELECT 1 FROM u)",
            vec!["1 + (SELECT MAX(x) FROM u)", "EXISTS (SELECT 1 FROM u)"],
        ),
    ] {
        let got = answer(db.query(sql, ())).unwrap().0;
        assert_eq!(got, names, "{sql}");
    }
}

#[test]
fn a_statement_without_from_answers_as_its_literal_twin() {
    let db = open("twins");
    for (sql, twin) in [
        ("SELECT (SELECT MAX(x) FROM u)", "SELECT 3"),
        ("SELECT (SELECT MAX(x) FROM u) AS m", "SELECT 3 AS m"),
        (
            "SELECT 1 + (SELECT MAX(x) FROM u), 'a'",
            "SELECT 1 + 3, 'a'",
        ),
        ("SELECT (SELECT x FROM u WHERE id = 9)", "SELECT NULL"),
        ("SELECT (SELECT x FROM u)", "SELECT (SELECT x FROM u)"),
        (
            "SELECT 1 WHERE EXISTS (SELECT 1 FROM u WHERE x = 2)",
            "SELECT 1 WHERE TRUE",
        ),
        (
            "SELECT 1 WHERE EXISTS (SELECT 1 FROM u WHERE x = 9)",
            "SELECT 1 WHERE FALSE",
        ),
        (
            "SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM u WHERE x = 9)",
            "SELECT 1 WHERE TRUE",
        ),
        (
            "SELECT 1 WHERE 2 IN (SELECT x FROM u)",
            "SELECT 1 WHERE TRUE",
        ),
        (
            "SELECT 1 WHERE 9 IN (SELECT x FROM u)",
            "SELECT 1 WHERE FALSE",
        ),
        (
            "SELECT 1 WHERE NULL IN (SELECT x FROM u)",
            "SELECT 1 WHERE NULL",
        ),
        (
            "SELECT 1 WHERE (SELECT MAX(x) FROM u) > 2",
            "SELECT 1 WHERE 3 > 2",
        ),
        (
            "SELECT 1 WHERE (SELECT x FROM u WHERE id = 9) > 2",
            "SELECT 1 WHERE NULL > 2",
        ),
        ("SELECT EXISTS (SELECT 1 FROM u)", "SELECT TRUE"),
        (
            "SELECT 2 IN (SELECT x FROM u), 9 IN (SELECT x FROM u)",
            "SELECT TRUE, FALSE",
        ),
        (
            "SELECT (SELECT MAX(x) FROM u) ORDER BY 1",
            "SELECT 3 ORDER BY 1",
        ),
        ("SELECT (SELECT MAX(x) FROM u) LIMIT 0", "SELECT 3 LIMIT 0"),
        (
            "SELECT (SELECT MAX(x) FROM u) LIMIT 1 OFFSET 1",
            "SELECT 3 LIMIT 1 OFFSET 1",
        ),
        (
            "SELECT DISTINCT (SELECT MAX(x) FROM u)",
            "SELECT DISTINCT 3",
        ),
        (
            "SELECT COUNT(*), (SELECT MAX(x) FROM u)",
            "SELECT COUNT(*), 3",
        ),
        ("SELECT SUM(2), (SELECT MAX(x) FROM u)", "SELECT SUM(2), 3"),
        (
            "SELECT COUNT(*), (SELECT MAX(x) FROM u) WHERE FALSE",
            "SELECT COUNT(*), 3 WHERE FALSE",
        ),
        (
            "SELECT MAX(2), COUNT(*) WHERE EXISTS (SELECT 1 FROM u WHERE x = 9)",
            "SELECT MAX(2), COUNT(*) WHERE FALSE",
        ),
        (
            "SELECT COUNT(*) HAVING (SELECT MAX(x) FROM u) > 2",
            "SELECT COUNT(*) HAVING 3 > 2",
        ),
        (
            "SELECT COUNT(*) HAVING (SELECT MAX(x) FROM u) > 5",
            "SELECT COUNT(*) HAVING 3 > 5",
        ),
        (
            "SELECT COUNT(*) + (SELECT MAX(x) FROM u)",
            "SELECT COUNT(*) + 3",
        ),
        (
            "SELECT ROW_NUMBER() OVER (), (SELECT MAX(x) FROM u)",
            "SELECT ROW_NUMBER() OVER (), 3",
        ),
        (
            "SELECT (SELECT MAX(x) FROM u), (SELECT MIN(x) FROM u)",
            "SELECT 3, 1",
        ),
        (
            "SELECT (SELECT (SELECT MAX(x) FROM u) + MIN(x) FROM u)",
            "SELECT 4",
        ),
    ] {
        alike(&db, sql, twin);
    }
}

#[test]
fn a_prepared_statement_without_from_reads_each_run_and_its_parameters() {
    let db = open("prepared");
    let stmt = db
        .prepare("SELECT (SELECT MAX(x) FROM u WHERE x < $1), $1")
        .unwrap();
    let run = |p: i64| answer(stmt.query(vec![Value::Integer(p)])).unwrap().1;
    assert_eq!(run(3), vec![vec![Value::Integer(2), Value::Integer(3)]]);
    assert_eq!(run(2), vec![vec![Value::Integer(1), Value::Integer(2)]]);
    db.execute("INSERT INTO u VALUES (4, 0)", ()).unwrap();
    assert_eq!(
        run(1),
        vec![vec![Value::Integer(0), Value::Integer(1)]],
        "a run reads the rows committed before it"
    );
    let filtered = db
        .prepare("SELECT 'yes' WHERE EXISTS (SELECT 1 FROM u WHERE x = $1)")
        .unwrap();
    let rows = |p: i64| {
        answer(filtered.query(vec![Value::Integer(p)]))
            .unwrap()
            .1
            .len()
    };
    assert_eq!((rows(2), rows(9)), (1, 0));
}

#[test]
fn a_statement_without_from_reads_its_transactions_rows() {
    let db = open("transaction");
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO u VALUES (5, 50)", ()).unwrap();
    let got: Vec<i64> = tx
        .query("SELECT (SELECT MAX(x) FROM u)", ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(got, vec![50]);
    tx.rollback().unwrap();
    let after: i64 = db.query_one("SELECT (SELECT MAX(x) FROM u)", ()).unwrap();
    assert_eq!(after, 3);
}

fn values(rows: stoolap::Result<stoolap::Rows>) -> Vec<Vec<Value>> {
    answer(rows).unwrap().1
}

fn aggregate_rule(db: &Database) {
    let none = |sql: &str| -> Vec<Vec<Value>> { values(db.query(sql, ())) };
    let zero = vec![vec![Value::Integer(0)]];
    for sql in [
        "SELECT COUNT(*) WHERE FALSE",
        "SELECT COUNT(*) WHERE NULL",
        "SELECT COUNT(*) WHERE 1 = 2",
        "SELECT COUNT(*) WHERE EXISTS (SELECT 1 FROM u WHERE x = 9)",
        "SELECT COUNT(*) WHERE (SELECT MAX(x) FROM u) > 5",
        "SELECT COUNT(*) WHERE FALSE HAVING COUNT(*) = 0",
        "SELECT COUNT(*) WHERE (SELECT MAX(x) FROM u) > 5 HAVING COUNT(*) = 0",
    ] {
        assert_eq!(none(sql), zero, "{sql}");
    }
    for sql in [
        "SELECT COUNT(*) WHERE FALSE HAVING COUNT(*) > 0",
        "SELECT 1 WHERE FALSE",
        "SELECT ROW_NUMBER() OVER () WHERE FALSE",
        "SELECT (SELECT MAX(x) FROM u) WHERE FALSE",
    ] {
        assert_eq!(none(sql), Vec::<Vec<Value>>::new(), "{sql}");
    }
    assert_eq!(
        none("SELECT SUM(2), CASE WHEN COUNT(*) = 0 THEN 'none' END WHERE FALSE"),
        vec![vec![
            Value::Null(stoolap::core::DataType::Null),
            Value::text("none")
        ]]
    );
    let stmt = db
        .prepare("SELECT COUNT(*), (SELECT MAX(x) FROM u) WHERE $1 > 1")
        .unwrap();
    let run = |p: i64| values(stmt.query(vec![Value::Integer(p)]));
    assert_eq!(run(2), vec![vec![Value::Integer(1), Value::Integer(3)]]);
    assert_eq!(run(0), vec![vec![Value::Integer(0), Value::Integer(3)]]);
    let literal = db.prepare("SELECT COUNT(*) WHERE $1 > 1").unwrap();
    assert_eq!(values(literal.query(vec![Value::Integer(0)])), zero);
}

#[test]
fn an_aggregate_without_from_answers_one_row_when_the_where_keeps_none() {
    aggregate_rule(&open("aggregate_hot"));
}

#[test]
fn an_aggregate_without_from_over_sealed_blocks_answers_one_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    aggregate_rule(&db);
}
