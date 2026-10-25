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

//! The DISTINCT of a set operation's first SELECT removes duplicates from
//! that SELECT's rows only, as the same SELECT written as a derived table

use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "INSERT INTO t SELECT value, value % 4, 'b' || (value % 3) FROM generate_series(1, 120)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO w SELECT value, value % 6 FROM generate_series(1, 30)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn sorted(db: &Database, sql: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            format!(
                "{:?}",
                (0..row.len()).map(|i| row.get_value(i)).collect::<Vec<_>>()
            )
        })
        .collect();
    rows.sort();
    rows
}

fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    // The first SELECT, written bare and in the general path's and the
    // prepared path's forms
    for first in [
        "SELECT DISTINCT a FROM t",
        "SELECT DISTINCT a FROM t WHERE id > (SELECT MIN(id) FROM t) - 1",
        "SELECT DISTINCT a, b FROM t",
    ] {
        let columns = if first.contains("a, b") { "a, b" } else { "a" };
        let other = if columns == "a" {
            "SELECT a FROM w"
        } else {
            "SELECT a, 'b1' FROM w"
        };
        for op in [
            "UNION ALL",
            "UNION",
            "INTERSECT",
            "INTERSECT ALL",
            "EXCEPT",
            "EXCEPT ALL",
        ] {
            let sql = format!("{first} {op} {other}");
            let want = format!("SELECT {columns} FROM ({first}) d {op} {other}");
            let (got, want_rows) = (sorted(db, &sql), sorted(db, &want));
            if got != want_rows {
                wrong.push(format!(
                    "{phase}: {sql}\n  got  {got:?}\n  want {want_rows:?}"
                ));
            }
        }
    }
}

#[test]
fn a_leading_distinct_removes_duplicates_from_its_own_rows() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://set_operation_leading_distinct").unwrap();
    setup(&db);
    check(&db, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    setup(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET a = 9 WHERE id % 40 = 0", ())
        .unwrap();
    check(&db, "sealed with hot updates", &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_union_all_after_a_distinct_keeps_its_duplicates() {
    let db = Database::open("memory://set_operation_leading_distinct_literals").unwrap();
    for (sql, want) in [
        ("SELECT DISTINCT 1 AS x UNION ALL SELECT 1", vec!["1", "1"]),
        (
            "SELECT DISTINCT 1 AS x UNION SELECT 2 UNION ALL SELECT 1",
            vec!["1", "1", "2"],
        ),
    ] {
        let mut got: Vec<String> = db
            .query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap().to_string())
            .collect();
        got.sort();
        assert_eq!(got, want, "{sql}");
    }
}
