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

//! An INTERVAL literal times a number, negated, or added to a timestamp
//! that way, answers the same in every context; the operators that have no
//! meaning for an interval are errors instead of NULL

use stoolap::core::Value;
use stoolap::Database;

const TS: &str = "TIMESTAMP '2024-01-31 00:00:00'";

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://interval_arithmetic_{name}")).unwrap();
    for sql in [
        "CREATE TABLE v (id INTEGER PRIMARY KEY, n INTEGER, f FLOAT, z INTEGER, ts TIMESTAMP)",
        "INSERT INTO v VALUES (1, 3, 1.5, NULL, TIMESTAMP '2024-01-31 00:00:00')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn value(db: &Database, sql: &str, params: Vec<Value>) -> String {
    match db.query(sql, params) {
        Ok(mut rows) => match rows.next() {
            Some(Ok(row)) => match row.get_value(0) {
                Some(Value::Null(_)) => "NULL".to_string(),
                Some(v) => format!("{v:?}"),
                None => "no column".to_string(),
            },
            Some(Err(e)) => format!("ERR {e}"),
            None => "no row".to_string(),
        },
        Err(e) => format!("ERR {e}"),
    }
}

fn ts(s: &str) -> String {
    format!("Timestamp({s}Z)")
}

fn text(s: &str) -> String {
    format!("Text({s:?})")
}

/// (shape over `v`, the same shape over literals, expected)
fn shapes() -> Vec<(&'static str, String, String)> {
    let lit = |s: &str| {
        s.replace("ts", TS)
            .replace("{n}", "3")
            .replace("{f}", "1.5")
            .replace("{z}", "NULL")
    };
    let col = |s: &str| {
        s.replace("{n}", "n")
            .replace("{f}", "f")
            .replace("{z}", "z")
    };
    [
        ("ts + {n} * INTERVAL '1 hour'", ts("2024-01-31T03:00:00")),
        ("ts + INTERVAL '1 hour' * {n}", ts("2024-01-31T03:00:00")),
        ("ts - {n} * INTERVAL '1 hour'", ts("2024-01-30T21:00:00")),
        ("{n} * INTERVAL '1 hour' + ts", ts("2024-01-31T03:00:00")),
        ("ts + ({n} * INTERVAL '1 hour')", ts("2024-01-31T03:00:00")),
        ("ts + {f} * INTERVAL '1 hour'", ts("2024-01-31T01:30:00")),
        ("ts + {n} * INTERVAL '1 month'", ts("2024-04-30T00:00:00")),
        ("ts + {n} * INTERVAL '1 year'", ts("2027-01-31T00:00:00")),
        ("ts + {n} * INTERVAL 1 DAY", ts("2024-02-03T00:00:00")),
        ("ts + {n} * INTERVAL '2 weeks'", ts("2024-03-13T00:00:00")),
        ("ts + -{n} * INTERVAL '1 hour'", ts("2024-01-30T21:00:00")),
        (
            "ts + {f} * INTERVAL '1 millisecond'",
            ts("2024-01-31T00:00:00.001500"),
        ),
        ("ts + {z} * INTERVAL '1 hour'", "NULL".to_string()),
        (
            "ts + 2 * ({n} * INTERVAL '1 hour')",
            ts("2024-01-31T06:00:00"),
        ),
        ("ts - -({n} * INTERVAL '1 hour')", ts("2024-01-31T03:00:00")),
        ("ts + -INTERVAL '1 hour'", ts("2024-01-30T23:00:00")),
        ("ts - -INTERVAL '1 hour'", ts("2024-01-31T01:00:00")),
        ("ts + INTERVAL '3 hours'", ts("2024-01-31T03:00:00")),
        ("{n} * INTERVAL '1 second'", text("3 second")),
        ("INTERVAL '1 second' * {n}", text("3 second")),
        ("{f} * INTERVAL '1 hour'", text("5400 second")),
        ("{f} * INTERVAL '1 second'", text("1500 millisecond")),
        ("{n} * INTERVAL '1 month'", text("3 month")),
        ("2.0 * INTERVAL '1 month'", text("2 month")),
        ("{z} * INTERVAL '1 hour'", "NULL".to_string()),
        ("2 * ({z} * INTERVAL '1 hour')", "NULL".to_string()),
        ("ts + 2 * ({z} * INTERVAL '1 hour')", "NULL".to_string()),
        ("{f} * ({n} * INTERVAL '1 hour')", text("16200 second")),
        (
            "2 * (9223372036854775807 * (0 * INTERVAL '1 microsecond'))",
            text("0 microsecond"),
        ),
        (
            "ts + 2 * (9223372036854775807 * (0 * INTERVAL '1 microsecond'))",
            ts("2024-01-31T00:00:00"),
        ),
        (
            "2 * (9223372036854775807 * ({z} * INTERVAL '1 hour'))",
            "NULL".to_string(),
        ),
        (
            "ts - 2 * (9223372036854775807 * ({z} * INTERVAL '1 hour'))",
            "NULL".to_string(),
        ),
        ("-INTERVAL '1 hour'", text("-1 hour")),
        ("2 * ({n} * INTERVAL '1 hour')", text("6 hour")),
    ]
    .into_iter()
    .map(|(s, want)| {
        let over_v: &'static str = Box::leak(col(s).into_boxed_str());
        (over_v, lit(s), want)
    })
    .collect()
}

#[test]
fn interval_products_answer_as_expected_over_literals_and_columns() {
    let db = open("shapes");
    let mut wrong = Vec::new();
    for (over_v, literal, want) in shapes() {
        for sql in [
            format!("SELECT {literal}"),
            format!("SELECT {over_v} FROM v"),
            format!("SELECT x FROM (SELECT {over_v} AS x FROM v) s"),
        ] {
            let got = value(&db, &sql, vec![]);
            if got != want {
                wrong.push(format!("{sql}: gave {got}, want {want}"));
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

#[test]
fn interval_products_take_parameters() {
    let db = open("params");
    let mut wrong = Vec::new();
    for (sql, params, want) in [
        (
            "SELECT ts + $1 * INTERVAL '1 hour' FROM v",
            vec![Value::Integer(3)],
            ts("2024-01-31T03:00:00"),
        ),
        (
            "SELECT ts - INTERVAL '1 hour' * $1 FROM v",
            vec![Value::Float(1.5)],
            ts("2024-01-30T22:30:00"),
        ),
        (
            "SELECT $1 * INTERVAL '1 day'",
            vec![Value::Integer(2)],
            text("2 day"),
        ),
    ] {
        let got = value(&db, sql, params);
        if got != want {
            wrong.push(format!("{sql}: gave {got}, want {want}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn interval_products_filter_update_and_insert() {
    let db = open("contexts");
    for sql in [
        "CREATE TABLE e (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
        "CREATE INDEX e_ts ON e(ts)",
        "INSERT INTO e SELECT value, DATE_ADD(TIMESTAMP '2024-01-31 00:00:00', value, 'hour') FROM generate_series(0, 99)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            format!("SELECT COUNT(*) FROM e WHERE ts >= {TS} + 3 * INTERVAL '1 hour'"),
            "Integer(97)".to_string(),
        ),
        (
            format!("SELECT COUNT(*) FROM e WHERE ts < {TS} + INTERVAL '1 hour' * 10"),
            "Integer(10)".to_string(),
        ),
        (
            format!("SELECT COUNT(*) FROM e WHERE ts BETWEEN {TS} + 2 * INTERVAL '1 hour' AND {TS} + 4 * INTERVAL '1 hour'"),
            "Integer(3)".to_string(),
        ),
        (
            format!("SELECT COUNT(*) FROM e WHERE ts = {TS} + id * INTERVAL '1 hour'"),
            "Integer(100)".to_string(),
        ),
    ] {
        let got = value(&db, &sql, vec![]);
        if got != want {
            wrong.push(format!("{sql}: gave {got}, want {want}"));
        }
    }
    db.execute("UPDATE v SET ts = ts + n * INTERVAL '1 day'", ())
        .unwrap();
    let got = value(&db, "SELECT ts FROM v", vec![]);
    if got != ts("2024-02-03T00:00:00") {
        wrong.push(format!("UPDATE: gave {got}"));
    }
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();
    db.execute(
        &format!("INSERT INTO o SELECT value, {TS} + value * INTERVAL '1 minute' FROM generate_series(1, 5)"),
        (),
    )
    .unwrap();
    let got = value(&db, "SELECT MAX(ts) FROM o", vec![]);
    if got != ts("2024-01-31T00:05:00") {
        wrong.push(format!("INSERT SELECT: gave {got}"));
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn interval_products_prune_sealed_rows_alike() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for sql in [
        "CREATE TABLE e (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
        "CREATE INDEX e_ts ON e(ts)",
        "INSERT INTO e SELECT value, DATE_ADD(TIMESTAMP '2024-01-31 00:00:00', value, 'hour') FROM generate_series(0, 99)",
        "PRAGMA CHECKPOINT",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let got = value(
        &db,
        &format!("SELECT COUNT(*) FROM e WHERE ts >= {TS} + 3 * INTERVAL '1 hour'"),
        vec![],
    );
    assert_eq!(got, "Integer(97)");
}

#[test]
fn interval_operators_without_a_meaning_are_errors() {
    let db = open("errors");
    let mut wrong = Vec::new();
    for sql in [
        "SELECT INTERVAL '1 hour' / 2".to_string(),
        "SELECT INTERVAL '1 hour' % 2".to_string(),
        "SELECT 2 / INTERVAL '1 hour'".to_string(),
        "SELECT INTERVAL '1 hour' + INTERVAL '2 hours'".to_string(),
        "SELECT INTERVAL '2 hours' - INTERVAL '1 hour'".to_string(),
        "SELECT INTERVAL '1 hour' * INTERVAL '1 hour'".to_string(),
        format!("SELECT INTERVAL '1 hour' - {TS}"),
        "SELECT id FROM v WHERE f * INTERVAL '1 month' IS NULL".to_string(),
        "SELECT id FROM v WHERE ts + f * INTERVAL '1 year' IS NULL".to_string(),
        "SELECT 'abc' * INTERVAL '1 hour'".to_string(),
        "SELECT id FROM v WHERE ts + 'abc' * INTERVAL '1 hour' IS NULL".to_string(),
        "SELECT 3 * INTERVAL '1 fortnight'".to_string(),
        "SELECT 9223372036854775807 * INTERVAL '1 hour'".to_string(),
        "SELECT id FROM v WHERE ts + 9223372036854775807 * INTERVAL '1 month' IS NULL".to_string(),
        "SELECT id FROM v WHERE ts + CAST('NaN' AS FLOAT) * INTERVAL '1 hour' IS NULL".to_string(),
        "SELECT id FROM v WHERE n + n * INTERVAL '1 hour' IS NULL".to_string(),
        "SELECT TRUE * INTERVAL '1 hour'".to_string(),
        "SELECT 2 * ('abc' * INTERVAL '1 hour')".to_string(),
        "SELECT 2 * (TRUE * INTERVAL '1 hour')".to_string(),
        "SELECT id FROM v WHERE ts + 2 * ('abc' * INTERVAL '1 hour') IS NULL".to_string(),
        "SELECT id FROM v WHERE ts - n * (TRUE * INTERVAL '1 hour') IS NULL".to_string(),
        "SELECT 2 * (9223372036854775807 * INTERVAL '1 microsecond')".to_string(),
    ] {
        let got = value(&db, &sql, vec![]);
        if !got.starts_with("ERR") {
            wrong.push(format!("{sql}: gave {got}, want an error"));
        }
    }
    for (sql, param) in [
        ("SELECT $1 * INTERVAL '1 hour'", Value::text("abc")),
        ("SELECT 2 * ($1 * INTERVAL '1 hour')", Value::text("abc")),
        ("SELECT 2 * ($1 * INTERVAL '1 hour')", Value::Boolean(true)),
        (
            "SELECT id FROM v WHERE ts + 2 * ($1 * INTERVAL '1 hour') IS NULL",
            Value::text("abc"),
        ),
    ] {
        let got = value(&db, sql, vec![param]);
        if !got.starts_with("ERR") {
            wrong.push(format!("{sql}: gave {got}, want an error"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
