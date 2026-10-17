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

//! A BOOLEAN or TIMESTAMP and an INTEGER or FLOAT are different domains on
//! every path: never equal, their ordering UNKNOWN, neither read as the other

use stoolap::core::Value;
use stoolap::Database;

const ORDER: &[&str] = &["<", "<=", ">", ">="];
const TS: &str = "CAST('2025-01-01 00:00:00' AS TIMESTAMP)";

fn open(db: Database, seal: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, b BOOLEAN, ts TIMESTAMP, \
         z INTEGER, zf FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES \
         (1, 1, TRUE, '2025-01-01 00:00:00', 0, 0.0), \
         (2, 1, FALSE, '2025-06-01 00:00:00', 1, 1.5), \
         (3, 2, TRUE, '2025-03-01 00:00:00', 2, -1.0)",
        (),
    )
    .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    rows_with(db, sql, vec![])
}

fn rows_with(db: &Database, sql: &str, params: Vec<Value>) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for row in db.query(sql, params).map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push(format!("{}", row.get_value(0).unwrap()));
    }
    out.sort();
    Ok(out)
}

/// Each ordering of `b` or `ts` against a number, in both operand orders,
/// against a constant and against a column
fn predicates() -> Vec<String> {
    let mut out = Vec::new();
    for col in ["b", "ts"] {
        for num in ["0", "1.5", "z", "zf"] {
            for op in ORDER {
                out.push(format!("{col} {op} {num}"));
                out.push(format!("{num} {op} {col}"));
            }
            out.push(format!("{col} BETWEEN {num} AND 9"));
            out.push(format!("{col} NOT BETWEEN {num} AND 9"));
        }
    }
    for col in ["z", "zf"] {
        for op in ORDER {
            out.push(format!("{col} {op} {TS}"));
            out.push(format!("{TS} {op} {col}"));
        }
        out.push(format!("{col} BETWEEN {TS} AND {TS}"));
    }
    out
}

fn ts_param() -> Value {
    Value::Timestamp(chrono::DateTime::from_timestamp(1_735_689_600, 0).unwrap())
}

/// Each equality across the number domain and whether it holds: a number
/// against `b` or `ts`, a BOOLEAN against `z` or `zf`; never equal
fn equalities() -> Vec<(String, bool)> {
    let mut out = Vec::new();
    for (col, others) in [
        ("b", &["0", "1", "1.5", "z", "zf"][..]),
        ("ts", &["0", "1.5", "z", "zf"][..]),
        ("z", &["TRUE", "FALSE", "b", "ts", TS][..]),
        ("zf", &["TRUE", "FALSE", "b", "ts", TS][..]),
    ] {
        for other in others {
            out.push((format!("{col} = {other}"), false));
            out.push((format!("{other} = {col}"), false));
            out.push((format!("{col} <> {other}"), true));
            if !matches!(*other, "z" | "zf" | "b" | "ts") {
                out.push((format!("{col} IN ({other}, 7)"), false));
                out.push((format!("{col} NOT IN ({other}, 7)"), true));
            }
        }
    }
    out
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let ids = |holds: bool| -> Result<Vec<String>, String> {
        Ok(if holds {
            vec!["1".to_string(), "2".to_string(), "3".to_string()]
        } else {
            vec![]
        })
    };
    for (pred, holds) in equalities() {
        let truth = if holds { "true" } else { "false" };
        for (sql, want) in [
            (format!("SELECT id FROM t WHERE {pred}"), ids(holds)),
            (format!("SELECT id FROM t WHERE NOT ({pred})"), ids(!holds)),
            (
                format!("SELECT id FROM t WHERE ({pred}) IS TRUE"),
                ids(holds),
            ),
            (
                format!("SELECT id FROM t WHERE CASE WHEN {pred} THEN 1 ELSE 0 END = 1"),
                ids(holds),
            ),
            (
                format!("SELECT {pred} FROM t"),
                Ok(vec![truth.to_string(); 3]),
            ),
        ] {
            let got = rows(db, &sql);
            if got != want {
                wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
            }
        }
    }
    for (col, param) in [
        ("b", Value::Integer(0)),
        ("ts", Value::Integer(0)),
        ("z", Value::Boolean(false)),
        ("zf", Value::Boolean(true)),
        ("z", ts_param()),
        ("zf", ts_param()),
    ] {
        for (sql, want) in [
            (format!("SELECT id FROM t WHERE {col} = $1"), ids(false)),
            (format!("SELECT id FROM t WHERE {col} <> $1"), ids(true)),
        ] {
            let got = rows_with(db, &sql, vec![param.clone()]);
            let prepared = db
                .prepare(&sql)
                .and_then(|s| s.query(vec![param.clone()]))
                .map(|r| {
                    let mut v: Vec<String> = r
                        .map(|row| format!("{}", row.unwrap().get_value(0).unwrap()))
                        .collect();
                    v.sort();
                    v
                })
                .map_err(|e| e.to_string());
            if got != want || prepared != want {
                wrong.push(format!(
                    "{ctx}: {sql} with {param:?}: got {got:?}, prepared {prepared:?}, want {want:?}"
                ));
            }
        }
    }
    check_order(db, ctx, wrong);
}

fn check_order(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let all = Ok(vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    let none: Result<Vec<String>, String> = Ok(vec![]);
    for pred in predicates() {
        for (sql, want) in [
            (format!("SELECT id FROM t WHERE {pred}"), &none),
            (format!("SELECT id FROM t WHERE NOT ({pred})"), &none),
            (format!("SELECT id FROM t WHERE ({pred}) IS NULL"), &all),
            (
                format!("SELECT id FROM t WHERE ({pred}) OR id = 1"),
                &Ok(vec!["1".to_string()]),
            ),
            (
                format!("SELECT id FROM t WHERE ({pred}) IS NULL AND ({pred}) IS NOT TRUE"),
                &all,
            ),
        ] {
            let got = rows(db, &sql);
            if &got != want {
                wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
            }
        }
        let sql = format!("SELECT {pred} FROM t");
        let got = rows(db, &sql);
        let nulls = Ok(vec!["NULL".to_string(); 3]);
        if got != nulls {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, want three NULLs"));
        }
    }
    for (col, params) in [
        ("b", vec![Value::Integer(0), Value::Float(1.5)]),
        ("ts", vec![Value::Integer(0), Value::Float(1.5)]),
        ("z", vec![ts_param()]),
        ("zf", vec![ts_param()]),
    ] {
        for op in ORDER {
            for param in &params {
                for (sql, want) in [
                    (format!("SELECT id FROM t WHERE {col} {op} $1"), &none),
                    (format!("SELECT id FROM t WHERE NOT ({col} {op} $1)"), &none),
                    (
                        format!("SELECT id FROM t WHERE ({col} {op} $1) IS NULL"),
                        &all,
                    ),
                    (
                        format!(
                            "SELECT id FROM t WHERE id IN (SELECT id FROM t WHERE {col} {op} $1)"
                        ),
                        &none,
                    ),
                    (
                        format!(
                            "SELECT id FROM t WHERE EXISTS \
                             (SELECT 1 FROM t u WHERE u.{col} {op} $1 AND u.id = t.id)"
                        ),
                        &none,
                    ),
                ] {
                    let got = rows_with(db, &sql, vec![param.clone()]);
                    if &got != want {
                        wrong.push(format!(
                            "{ctx}: {sql} with {param:?}: got {got:?}, want {want:?}"
                        ));
                    }
                }
                let sql = format!(
                    "SELECT id, (SELECT COUNT(*) FROM t u WHERE u.{col} {op} $1) FROM t WHERE id = 1"
                );
                let got = db
                    .prepare(&sql)
                    .and_then(|stmt| stmt.query(vec![param.clone()]))
                    .map(|rows| {
                        rows.map(|r| r.unwrap().get::<i64>(1).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .map_err(|e| e.to_string());
                if got != Ok(vec![0]) {
                    wrong.push(format!("{ctx}: prepared {sql} with {param:?}: got {got:?}"));
                }
            }
        }
    }
    for (sql, want) in [
        ("SELECT TRUE > 0 FROM t WHERE id = 1", "NULL"),
        (
            "SELECT CAST('2025-01-01' AS TIMESTAMP) >= 0 FROM t WHERE id = 1",
            "NULL",
        ),
        ("SELECT TRUE = 1 FROM t WHERE id = 1", "false"),
        ("SELECT b = 1 FROM t WHERE id = 1", "false"),
        ("SELECT b <> 1 FROM t WHERE id = 1", "true"),
        ("SELECT ts = 0 FROM t WHERE id = 1", "false"),
        ("SELECT ts <> 0 FROM t WHERE id = 1", "true"),
    ] {
        let got = rows(db, sql);
        if got != Ok(vec![want.to_string()]) {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want}"));
        }
    }
}

#[test]
fn ordering_a_non_number_against_a_number_is_unknown_everywhere() {
    let mut wrong = Vec::new();
    check(
        &open(Database::open("memory://non_number_order").unwrap(), false),
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

#[test]
fn having_orders_a_non_number_aggregate_as_unknown_in_every_form() {
    let db = open(
        Database::open("memory://non_number_order_having").unwrap(),
        false,
    );
    let mut wrong = Vec::new();
    for col in ["b", "ts"] {
        for agg in ["MIN", "MAX"] {
            for op in ORDER {
                for c in ["0", "1.5"] {
                    let a = format!("{agg}({col}) {op} {c}");
                    for (sql, want) in [
                        (format!("SELECT g FROM t GROUP BY g HAVING {a}"), vec![]),
                        (
                            format!(
                                "SELECT g, {agg}({col}) AS m FROM t GROUP BY g HAVING m {op} {c}"
                            ),
                            vec![],
                        ),
                        (
                            format!("SELECT g FROM t GROUP BY g HAVING {a} OR COUNT(*) < 0"),
                            vec![],
                        ),
                        (
                            format!("SELECT g FROM t GROUP BY g HAVING NOT ({a})"),
                            vec![],
                        ),
                        (
                            format!("SELECT g FROM t GROUP BY g HAVING ({a}) IS NULL"),
                            vec!["1".to_string(), "2".to_string()],
                        ),
                        (
                            format!("SELECT g FROM t GROUP BY g HAVING {agg}({col}) = {c}"),
                            vec![],
                        ),
                        (
                            format!(
                                "SELECT g, {agg}({col}) AS m FROM t GROUP BY g HAVING m = {c}"
                            ),
                            vec![],
                        ),
                        (
                            format!(
                                "SELECT g FROM t GROUP BY g HAVING {agg}({col}) = {c} OR COUNT(*) < 0"
                            ),
                            vec![],
                        ),
                        (
                            format!("SELECT g FROM t GROUP BY g HAVING {agg}({col}) <> {c}"),
                            vec!["1".to_string(), "2".to_string()],
                        ),
                    ] {
                        let got = rows(&db, &sql);
                        if got != Ok(want.clone()) {
                            wrong.push(format!("{sql}: got {got:?}, want {want:?}"));
                        }
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
