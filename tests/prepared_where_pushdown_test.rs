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

//! A prepared statement's WHERE, given each run's values for the scan,
//! answers as its literal twin does, hot and sealed

use stoolap::core::Value;
use stoolap::Database;

const OPS: [&str; 6] = ["=", "<>", "<", "<=", ">", ">="];
const COLUMNS: [&str; 6] = ["i", "f", "t", "ts", "b", "n"];

fn fill(db: &Database) {
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, i INTEGER NOT NULL, f FLOAT NOT NULL, \
         t TEXT NOT NULL, ts TIMESTAMP NOT NULL, b BOOLEAN NOT NULL, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_w_i ON w(i)", ()).unwrap();
    db.execute("CREATE INDEX idx_w_t ON w(t)", ()).unwrap();
    db.execute(
        "INSERT INTO w SELECT g.value, g.value % 100, (g.value % 100) * 1.0, \
         CAST(g.value % 100 AS TEXT), TIMESTAMP '2024-01-01 00:00:00' + g.value * INTERVAL '1 hour', \
         g.value % 2 = 0, CASE WHEN g.value % 7 = 0 THEN NULL ELSE g.value END \
         FROM generate_series(1, 300) g",
        (),
    )
    .unwrap();
}

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://prepared_where_pushdown_{name}")).unwrap();
    fill(&db);
    db
}

fn sealed(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    fill(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// Each value with its literal spelling
fn values() -> Vec<(Value, &'static str)> {
    vec![
        (Value::Integer(50), "50"),
        (Value::Float(50.0), "50.0"),
        (Value::Float(50.5), "50.5"),
        (Value::text("50"), "'50'"),
        (Value::text("abc"), "'abc'"),
        (Value::Null(stoolap::core::DataType::Null), "NULL"),
        (
            Value::text("2024-01-05 00:00:00"),
            "'2024-01-05 00:00:00'",
        ),
        (Value::Boolean(true), "TRUE"),
    ]
}

/// The sorted first column, or the error's text
fn answer(rows: stoolap::Result<stoolap::Rows>) -> Result<Vec<i64>, String> {
    let mut ids = Vec::new();
    for row in rows.map_err(|e| e.to_string())? {
        ids.push(row.map_err(|e| e.to_string())?.get::<i64>(0).unwrap());
    }
    ids.sort_unstable();
    Ok(ids)
}

fn prepared(db: &Database, sql: &str, params: Vec<Value>) -> Result<Vec<i64>, String> {
    let stmt = db.prepare(sql).map_err(|e| e.to_string())?;
    // The second run is the one that reuses what the first prepared
    let first = answer(stmt.query(params.clone()));
    let second = answer(stmt.query(params));
    assert_eq!(first, second, "{sql}: a second run answers as the first");
    second
}

fn literal(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    answer(db.query(sql, ()))
}

fn matrix(db: &Database) {
    for column in COLUMNS {
        for op in OPS {
            for (value, spelled) in values() {
                let p = prepared(
                    db,
                    &format!("SELECT id, (SELECT 1) FROM w WHERE {column} {op} $1"),
                    vec![value.clone()],
                );
                let l = literal(db, &format!("SELECT id FROM w WHERE {column} {op} {spelled}"));
                assert_eq!(p, l, "{column} {op} {spelled}");
                let p = prepared(
                    db,
                    &format!("SELECT id, (SELECT 1) FROM w WHERE $1 {op} {column}"),
                    vec![value],
                );
                let l = literal(db, &format!("SELECT id FROM w WHERE {spelled} {op} {column}"));
                assert_eq!(p, l, "{spelled} {op} {column}");
            }
        }
    }
}

fn shapes(db: &Database) {
    for (sql, params, twin) in [
        (
            "SELECT id, (SELECT 1) FROM w WHERE i >= $1 AND i <= $2 AND t = $3",
            vec![Value::Integer(10), Value::Integer(20), Value::text("15")],
            "SELECT id FROM w WHERE i >= 10 AND i <= 20 AND t = '15'",
        ),
        (
            "SELECT id, (SELECT 1) FROM w WHERE i > $1 AND f < $2 AND n <> $3 AND b = $4",
            vec![
                Value::Integer(10),
                Value::Float(60.5),
                Value::Integer(110),
                Value::Boolean(false),
            ],
            "SELECT id FROM w WHERE i > 10 AND f < 60.5 AND n <> 110 AND b = FALSE",
        ),
        (
            "SELECT id, (SELECT 1) FROM w WHERE id = $1 AND i = $2",
            vec![Value::Integer(142), Value::Integer(42)],
            "SELECT id FROM w WHERE id = 142 AND i = 42",
        ),
        (
            "SELECT id, (SELECT 1) FROM w WHERE i = $1 AND n IS NULL",
            vec![Value::Integer(14)],
            "SELECT id FROM w WHERE i = 14 AND n IS NULL",
        ),
        (
            "SELECT id, (SELECT 1) FROM w WHERE i IN ($1, $2) AND f > $3",
            vec![Value::Integer(5), Value::Integer(6), Value::Float(0.0)],
            "SELECT id FROM w WHERE i IN (5, 6) AND f > 0.0",
        ),
        (
            "SELECT a.id, (SELECT 1) FROM w a JOIN w c ON a.id = c.id WHERE a.i = $1 AND c.f < $2",
            vec![Value::Integer(7), Value::Float(50.0)],
            "SELECT a.id FROM w a JOIN w c ON a.id = c.id WHERE a.i = 7 AND c.f < 50.0",
        ),
    ] {
        assert_eq!(prepared(db, sql, params), literal(db, twin), "{sql}");
    }
}

fn updates(db: &Database) {
    let stmt = db
        .prepare("UPDATE w SET n = n WHERE i = $1 AND f >= $2 AND id > (SELECT 0)")
        .unwrap();
    for (i, f) in [(3, 0.0), (3, 50.0), (98, 98.0)] {
        let p = stmt
            .execute(vec![Value::Integer(i), Value::Float(f)])
            .unwrap();
        let l: i64 = db
            .query_one(
                &format!("SELECT COUNT(*) FROM w WHERE i = {i} AND f >= {f:.1}"),
                (),
            )
            .unwrap();
        assert_eq!(p, l, "i = {i} AND f >= {f}");
    }
}

#[test]
fn a_hot_prepared_where_answers_as_its_literal_twin() {
    let db = hot("hot");
    matrix(&db);
    shapes(&db);
    updates(&db);
}

#[test]
fn a_sealed_prepared_where_answers_as_its_literal_twin() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    matrix(&db);
    shapes(&db);
    updates(&db);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_where_of_comparisons_is_not_converted_per_run() {
    use stoolap::test_failpoints as fp;

    let db = hot("converted");
    let runs = |sql: &str, params: Vec<Value>| -> usize {
        let stmt = db.prepare(sql).unwrap();
        answer(stmt.query(params.clone())).unwrap();
        let before = fp::where_pushdowns_converted();
        answer(stmt.query(params)).unwrap();
        fp::where_pushdowns_converted() - before
    };
    for (sql, params) in [
        (
            "SELECT id, (SELECT 1) FROM w WHERE id = $1",
            vec![Value::Integer(42)],
        ),
        (
            "SELECT id, (SELECT 1) FROM w WHERE i >= $1 AND i <= $2 AND t = $3",
            vec![Value::Integer(10), Value::Integer(20), Value::text("15")],
        ),
        (
            "SELECT id, (SELECT 1) FROM w WHERE $1 < i",
            vec![Value::Integer(90)],
        ),
    ] {
        assert_eq!(runs(sql, params), 0, "{sql}");
    }
    // A conjunct that is not a comparison, or a value the column does not
    // read as its type, is converted as it was
    assert_eq!(
        runs(
            "SELECT id, (SELECT 1) FROM w WHERE i IN ($1, $2)",
            vec![Value::Integer(5), Value::Integer(6)],
        ),
        1
    );
    assert_eq!(
        runs(
            "SELECT id, (SELECT 1) FROM w WHERE i = $1",
            vec![Value::text("abc")],
        ),
        1
    );
}
