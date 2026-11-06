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

//! A grouped or aggregating statement that reads a name two columns of its
//! source carry is refused as ambiguous, wherever it reads it

use stoolap::Database;

const FROM: &str = "FROM (SELECT id, 10 AS x, 20 AS x FROM t) AS s";

fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://grouped_ambiguous_name_{name}")).unwrap();
    fill(&db);
    db
}

fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    fill(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn rows(db: &Database, sql: &str, prepared: bool) -> stoolap::Result<Vec<Vec<i64>>> {
    let result = match prepared {
        true => db.prepare(sql)?.query((0,))?,
        false => db.query(sql, (0,))?,
    };
    let width = result.columns().len();
    result
        .map(|row| {
            let row = row?;
            (0..width).map(|i| row.get::<i64>(i)).collect()
        })
        .collect()
}

#[test]
fn a_grouped_statement_reading_a_repeated_name_is_ambiguous() {
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in [("hot", hot("refused")), ("sealed", sealed(&dir))] {
        for sql in [
            format!("SELECT COUNT(*) {FROM} WHERE id > $1 GROUP BY s.x"),
            format!("SELECT COUNT(*) {FROM} WHERE id > $1 GROUP BY x"),
            format!("SELECT x, COUNT(*) {FROM} WHERE id > $1 GROUP BY 1"),
            format!("SELECT COUNT(*) {FROM} GROUP BY id HAVING MAX(x) > $1"),
            format!("SELECT MAX(x) {FROM} WHERE id > $1"),
            format!("SELECT MAX(s.x) + $1 {FROM}"),
            format!("SELECT CASE WHEN MAX(x) > $1 THEN 1 END {FROM}"),
            format!("SELECT id, COUNT(*) {FROM} WHERE id > $1 GROUP BY id ORDER BY MAX(s.x)"),
            format!(
                "SELECT COUNT(*) {FROM} WHERE id IN (SELECT id FROM t WHERE id > $1) GROUP BY s.x"
            ),
            // A key reading the name inside an expression, beside an alias of it
            format!("SELECT id AS x, COUNT(*) {FROM} WHERE id > $1 GROUP BY id, x + 1"),
            format!("SELECT COUNT(*) {FROM} WHERE id > $1 GROUP BY GROUPING SETS ((x))"),
            format!("SELECT MAX(x), ROW_NUMBER() OVER () {FROM} WHERE id > $1"),
            // Names that fold alike: two of one non-ASCII spelling, and the
            // Kelvin sign beside k
            "SELECT MAX(s.\"É\") FROM (SELECT id, 10 AS \"É\", 20 AS \"É\" FROM t) AS s \
             WHERE id > $1"
                .to_string(),
            "SELECT MAX(s.k) FROM (SELECT id, 10 AS \"\u{212A}\", 20 AS k FROM t) AS s \
             WHERE id > $1"
                .to_string(),
            // A final sigma folds as the name it ends, not as its letter alone
            "SELECT MAX(s.\"ος\") FROM (SELECT id, 10 AS \"ΟΣ\", 20 AS \"ος\" FROM t) AS s \
             WHERE id > $1"
                .to_string(),
        ] {
            for prepared in [false, true] {
                let err = rows(&db, &sql, prepared)
                    .expect_err(&format!("{context}, prepared {prepared}: {sql}"))
                    .to_string();
                assert!(
                    err.contains("is ambiguous"),
                    "{context}, prepared {prepared}: {sql}: {err}"
                );
            }
        }
    }
}

#[test]
fn a_grouped_statement_not_reading_the_repeated_name_answers() {
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in [("hot", hot("answers")), ("sealed", sealed(&dir))] {
        for (sql, expected) in [
            (
                format!("SELECT COUNT(*), MAX(id) {FROM} WHERE id > $1"),
                vec![vec![2, 2]],
            ),
            (
                format!("SELECT id + 1 AS x, COUNT(*) {FROM} WHERE id > $1 GROUP BY id ORDER BY x"),
                vec![vec![2, 1], vec![3, 1]],
            ),
        ] {
            for prepared in [false, true] {
                assert_eq!(
                    rows(&db, &sql, prepared).unwrap(),
                    expected,
                    "{context}, prepared {prepared}: {sql}"
                );
            }
        }
    }
}
