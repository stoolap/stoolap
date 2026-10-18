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

//! A TIMESTAMP moved by an INTERVAL or a number of days reads the interval
//! one way, moves a month on the calendar with the day clamped, and is an
//! error for a result outside years 0000 to 9999, wherever it is computed.

use stoolap::{Database, Value};

/// `t` holds the last and the first representable day, and interval text
fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://timestamp_arithmetic_{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, '9999-12-31 12:00:00.5', '1 day'), \
         (2, '0000-01-01 00:00:00', '-1 day'), (3, '2024-01-31 10:00:00.75', 'banana'), \
         (4, '2024-01-31 10:00:00.75', '99999999999999999999 days')",
        (),
    )
    .unwrap();
    db
}

fn answer(db: &Database, sql: &str) -> Value {
    db.query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))[0]
        .get_value(0)
        .cloned()
        .unwrap()
}

fn fails(db: &Database, sql: &str) {
    let result = db
        .query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
    assert!(result.is_err(), "{sql} should fail");
}

fn ts(s: &str) -> Value {
    Value::Timestamp(stoolap::core::parse_timestamp(s).unwrap())
}

#[test]
fn test_a_result_outside_the_range_is_an_error() {
    let db = open("range");
    for expr in [
        "ts + INTERVAL '1 day'",
        "ts + INTERVAL '1 month'",
        "ts + INTERVAL 1 YEAR",
        "ts + 2 * INTERVAL '1 day'",
        "ts + 1",
        "1 + ts",
        "ts - -1",
        "ts + s",
        "DATE_ADD(ts, 1)",
        "DATE_ADD(ts, 1, 'month')",
        "DATE_SUB(ts, -1, 'year')",
        "ts + 9223372036854775807",
        "ts + INTERVAL '9223372036854775807 days'",
        "ts + INTERVAL '9223372036854775807 years'",
        "ts + '9223372036854775807 days 00:00:00'",
        "ts + '106751992 days 00:00:00'",
    ] {
        fails(&db, &format!("SELECT {expr} FROM t WHERE id = 1"));
        fails(
            &db,
            &format!("SELECT id FROM t WHERE id = 1 AND {expr} > ts"),
        );
    }
    for expr in [
        "ts - INTERVAL '1 day'",
        "ts - 1",
        "ts + s",
        "DATE_SUB(ts, 1)",
        "DATE_SUB(ts, 1, 'year')",
        "ts - 9223372036854775807",
        "DATE_SUB(ts, -9223372036854775808)",
    ] {
        fails(&db, &format!("SELECT {expr} FROM t WHERE id = 2"));
    }
    fails(&db, "SELECT ts + s FROM t WHERE id = 4");

    assert!(db
        .execute("UPDATE t SET ts = ts + INTERVAL '1 day' WHERE id = 1", ())
        .is_err());
    assert_eq!(
        answer(&db, "SELECT ts FROM t WHERE id = 1"),
        ts("9999-12-31 12:00:00.5")
    );
    assert!(db
        .execute(
            "INSERT INTO t (id, ts) SELECT 9, ts + 1 FROM t WHERE id = 1",
            ()
        )
        .is_err());
}

#[test]
fn test_a_parameter_moves_a_timestamp_as_a_literal_does() {
    let db = open("parameters");
    let sql = "SELECT ts + $1 FROM t WHERE id = 1";
    for param in [Value::text("1 day"), Value::Integer(1)] {
        let result = db
            .query(sql, (param.clone(),))
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
        assert!(result.is_err(), "{sql} with {param:?} should fail");
    }
    let rows = db
        .query(
            "SELECT ts - $1 FROM t WHERE id = 1",
            (Value::text("1 day"),),
        )
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        rows[0].get_value(0).cloned().unwrap(),
        ts("9999-12-30 12:00:00.5")
    );
}

#[test]
fn test_an_interval_reads_one_way() {
    let db = open("reading");
    let base = "TIMESTAMP '2024-01-01'";
    for (interval, expected) in [
        ("INTERVAL '5 ms'", "2024-01-01 00:00:00.005"),
        ("INTERVAL '5 ms' * 1", "2024-01-01 00:00:00.005"),
        ("INTERVAL '7 us'", "2024-01-01 00:00:00.000007"),
        ("INTERVAL '2 mins'", "2024-01-01 00:02:00"),
        ("INTERVAL '3 secs'", "2024-01-01 00:00:03"),
        ("INTERVAL 5 MILLISECONDS", "2024-01-01 00:00:00.005"),
        ("INTERVAL 1 WEEK", "2024-01-08 00:00:00"),
    ] {
        assert_eq!(
            answer(&db, &format!("SELECT {base} + {interval}")),
            ts(expected),
            "{interval}"
        );
    }
    for literal in ["INTERVAL '1 banana'", "INTERVAL 1 BANANA"] {
        fails(&db, &format!("SELECT {base} + {literal}"));
    }
    // Text that is not an interval reads as NULL
    assert!(answer(&db, "SELECT ts + s FROM t WHERE id = 3").is_null());
    assert!(answer(&db, "SELECT ts + '1.5 days' FROM t WHERE id = 3").is_null());
    for text in [
        "banana days 99999999999999999999:00:00",
        "99999999999999999999 days banana:00:00",
    ] {
        let rows = db
            .query("SELECT ts + $1 FROM t WHERE id = 3", (Value::text(text),))
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .unwrap_or_else(|e| panic!("{text}: {e}"));
        assert!(rows[0].get_value(0).unwrap().is_null(), "{text}");
    }
}

#[test]
fn test_a_month_moves_the_calendar_and_keeps_the_time() {
    let db = open("months");
    for (expr, expected) in [
        ("ts + INTERVAL '1 month'", "2024-02-29 10:00:00.75"),
        ("ts + INTERVAL '1 year'", "2025-01-31 10:00:00.75"),
        ("ts + INTERVAL '13 months'", "2025-02-28 10:00:00.75"),
        ("DATE_ADD(ts, 1, 'month')", "2024-02-29 10:00:00.75"),
        ("DATE_ADD(ts, 1, 'year')", "2025-01-31 10:00:00.75"),
        ("DATE_SUB(ts, 1, 'month')", "2023-12-31 10:00:00.75"),
        ("DATE_ADD(ts, 250, 'ms')", "2024-01-31 10:00:01"),
        ("DATE_ADD(ts, 1.9)", "2024-02-01 10:00:00.75"),
        ("DATE_SUB(ts, 1)", "2024-01-30 10:00:00.75"),
    ] {
        assert_eq!(
            answer(&db, &format!("SELECT {expr} FROM t WHERE id = 3")),
            ts(expected),
            "{expr}"
        );
    }
    assert_eq!(
        answer(&db, "SELECT TIMESTAMP '2024-03-31' - INTERVAL '1 month'"),
        ts("2024-02-29")
    );
}

#[test]
fn test_a_series_steps_by_calendar_months() {
    let db = open("series");
    let values = |sql: &str| -> Vec<Value> {
        db.query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get_value(0).cloned().unwrap())
            .collect()
    };
    assert_eq!(
        values(
            "SELECT value FROM generate_series(TIMESTAMP '2024-01-31', TIMESTAMP '2024-05-31', '1 month')"
        ),
        ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31"]
            .map(ts)
            .to_vec()
    );
    assert_eq!(
        values(
            "SELECT value FROM generate_series(TIMESTAMP '2024-02-29', TIMESTAMP '2028-02-29', '1 year')"
        ),
        ["2024-02-29", "2025-02-28", "2026-02-28", "2027-02-28", "2028-02-29"]
            .map(ts)
            .to_vec()
    );
    assert_eq!(
        values(
            "SELECT value FROM generate_series(TIMESTAMP '2024-05-31', TIMESTAMP '2024-03-01', '-1 month')"
        ),
        ["2024-05-31", "2024-04-30", "2024-03-31"].map(ts).to_vec()
    );
    assert_eq!(
        values(
            "SELECT value FROM generate_series(TIMESTAMP '9999-12-30', TIMESTAMP '9999-12-31', 1)"
        ),
        ["9999-12-30", "9999-12-31"].map(ts).to_vec()
    );
    assert_eq!(
        values(
            "SELECT value FROM generate_series(TIMESTAMP '2024-01-01', TIMESTAMP '2024-01-02', '106751991 days 00:00:00')"
        ),
        [ts("2024-01-01")].to_vec()
    );
    assert!(db
        .query(
            "SELECT value FROM generate_series(TIMESTAMP '2024-01-01', TIMESTAMP '2024-01-02', '106751992 days 00:00:00')",
            ()
        )
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .is_err());
}
