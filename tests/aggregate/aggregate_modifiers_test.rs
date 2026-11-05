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

//! An aggregate's DISTINCT, FILTER, ORDER BY and extra arguments decide its
//! value on every path that answers it

use stoolap::core::Value;
use stoolap::Database;

fn open(db: Database, seal: bool) -> Database {
    for sql in [
        "CREATE TABLE d (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO d VALUES (1, 1), (2, 1), (3, 1), (4, 5)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "INSERT INTO s VALUES (1, 1, 'x'), (2, 1, 'z'), (3, 1, 'y')",
        "CREATE TABLE i (id INTEGER PRIMARY KEY, x INTEGER, c INTEGER)",
        "CREATE INDEX i_x ON i(x)",
        "CREATE INDEX i_c ON i(c)",
        "INSERT INTO i VALUES (1, 5, 1), (2, 20, 2), (3, 30, 2)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
        "INSERT INTO t VALUES (1, '2024-01-01 00:00:00'), (2, '2024-01-02 00:00:00'), \
         (3, '2024-01-02 00:00:00')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        Value::Text(s) => format!("t:{s}"),
        v if v.is_null() => "null".to_string(),
        other => format!("{other:?}"),
    }
}

/// The distinct rows, sorted, or the error
fn answer(db: &Database, sql: &str) -> Vec<String> {
    let rows = match db.query(sql, ()) {
        Ok(rows) => rows,
        Err(e) => return vec![format!("error: {e}")],
    };
    let mut out = Vec::new();
    for row in rows {
        match row {
            Ok(row) => out.push(
                (0..row.len())
                    .map(|i| render(row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
            Err(e) => return vec![format!("error: {e}")],
        }
    }
    out.sort();
    out.dedup();
    out
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let cases: &[(&str, &[&str])] = &[
        ("SELECT MEDIAN(DISTINCT x) FROM d", &["f:3.0"]),
        (
            "SELECT MEDIAN(DISTINCT x) FROM d WHERE id + 0 = id",
            &["f:3.0"],
        ),
        ("SELECT ARRAY_AGG(DISTINCT x) FROM d", &["t:[1,5]"]),
        ("SELECT STDDEV_POP(DISTINCT x) FROM d", &["f:2.0"]),
        (
            "SELECT VAR_POP(DISTINCT x), COUNT(*) FROM d",
            &["f:4.0|i:4"],
        ),
        (
            "SELECT a, STRING_AGG(b, ',' ORDER BY b DESC) FROM s GROUP BY ROLLUP(a)",
            &["i:1|t:z,y,x", "null|t:z,y,x"],
        ),
        (
            "SELECT a, STRING_AGG(b, ',' ORDER BY b DESC) FROM s GROUP BY a",
            &["i:1|t:z,y,x"],
        ),
        (
            "SELECT STRING_AGG(b, ' | ') OVER () FROM s",
            &["t:x | z | y"],
        ),
        (
            "SELECT STRING_AGG(b, ' | ' ORDER BY b) OVER () FROM s",
            &["t:x | y | z"],
        ),
        ("SELECT MIN(x) FILTER (WHERE x > 10) FROM i", &["i:20"]),
        ("SELECT MAX(x) FILTER (WHERE x < 25) FROM i", &["i:20"]),
        (
            "SELECT COUNT(DISTINCT c) FILTER (WHERE id < 0) FROM i",
            &["i:0"],
        ),
        (
            "SELECT COUNT(DISTINCT c) FILTER (WHERE id > 1) FROM i",
            &["i:1"],
        ),
        ("SELECT COUNT(*) FILTER (WHERE x > 10) FROM i", &["i:2"]),
        ("SELECT COUNT(DISTINCT ts) FROM t", &["i:2"]),
        ("SELECT COUNT(DISTINCT ts), COUNT(*) FROM t", &["i:2|i:3"]),
    ];
    for (sql, want) in cases {
        let got = answer(db, sql);
        if got != *want {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
        }
    }
}

#[test]
fn modifiers_decide_the_value_on_every_path() {
    let mut wrong = Vec::new();
    check(
        &open(
            Database::open("memory://aggregate_modifiers").unwrap(),
            false,
        ),
        "memory",
        &mut wrong,
    );
    for seal in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(
            Database::open(&format!(
                "file://{}?checkpoint_interval=0",
                dir.path().display()
            ))
            .unwrap(),
            seal,
        );
        check(&db, &format!("file sealed {seal}"), &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
