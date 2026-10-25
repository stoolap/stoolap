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

//! A TIMESTAMP is computed with from year 0000 to 9999, and stored or handed
//! out as nanoseconds since the epoch, from 1677-09-21 to 2262-04-11: storing
//! one outside that range, or reading a table AS OF one, is an error, never a
//! value moved to another time or an empty answer.

use stoolap::core::Value;
use stoolap::Database;

/// t holds one row in 2024 and one NULL, hot or sealed
fn open(name: &str, sealed: bool, dir: &tempfile::TempDir) -> (Database, String) {
    let dsn = match sealed {
        false => format!("memory://{name}"),
        true => format!("file://{}/{name}", dir.path().display()),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
        "INSERT INTO t VALUES (1, '2024-01-01 00:00:00'), (2, NULL)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    (db, dsn)
}

/// Each row's values with their types, one string per row
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    db.query(sql, ())
        .and_then(|rows| {
            rows.map(|row| {
                row.map(|row| {
                    (0..row.len())
                        .map(|i| match row.get_value(i) {
                            Some(Value::Timestamp(ts)) => ts.to_rfc3339(),
                            Some(value) => format!("{value:?}"),
                            None => "missing".into(),
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            })
            .collect()
        })
        .map_err(|e| e.to_string())
}

const STORED: &str = "SELECT id, ts FROM t ORDER BY id";

#[test]
fn a_timestamp_past_the_stored_range_is_computed_with() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let (db, _) = open(&format!("timestamp_range_computed_{sealed}"), sealed, &dir);
        for (sql, want) in [
            (
                "SELECT id FROM t WHERE ts < '9999-12-31'",
                vec!["Integer(1)"],
            ),
            (
                "SELECT id FROM t WHERE ts > '1600-01-01'",
                vec!["Integer(1)"],
            ),
            (
                "SELECT id FROM t WHERE ts BETWEEN '1000-01-01' AND '9999-12-31'",
                vec!["Integer(1)"],
            ),
            (
                "SELECT id FROM t WHERE ts < TIMESTAMP '3000-01-01'",
                vec!["Integer(1)"],
            ),
            ("SELECT id FROM t WHERE ts >= '3000-01-01'", vec![]),
            (
                "SELECT CAST('9999-12-31' AS TIMESTAMP)",
                vec!["9999-12-31T00:00:00+00:00"],
            ),
            (
                "SELECT TIMESTAMP '2262-01-01' + INTERVAL '200 years'",
                vec!["2462-01-01T00:00:00+00:00"],
            ),
        ] {
            let got = rows(&db, sql);
            if got.as_deref() != Ok(&want.iter().map(|s| s.to_string()).collect::<Vec<_>>()[..]) {
                wrong.push(format!(
                    "sealed {sealed}: {sql} gave {got:?}, want {want:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn storing_a_timestamp_past_the_range_is_an_error_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let (db, dsn) = open(&format!("timestamp_range_stored_{sealed}"), sealed, &dir);
        let before = rows(&db, STORED).unwrap();
        let far = chrono::DateTime::parse_from_rfc3339("9999-12-31T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        for (sql, params) in [
            ("INSERT INTO t VALUES (3, '9999-12-31')", vec![]),
            ("INSERT INTO t VALUES (3, '1600-01-01')", vec![]),
            ("INSERT INTO t VALUES (3, TIMESTAMP '2262-04-12')", vec![]),
            (
                "INSERT INTO t VALUES (3, $1)",
                vec![Value::text("9999-12-31")],
            ),
            ("INSERT INTO t VALUES (3, $1)", vec![Value::Timestamp(far)]),
            ("UPDATE t SET ts = '9999-12-31' WHERE id = 1", vec![]),
            (
                "UPDATE t SET ts = ts + INTERVAL '300 years' WHERE id = 1",
                vec![],
            ),
            (
                "UPDATE t SET ts = ts - INTERVAL '400 years' WHERE id = 1",
                vec![],
            ),
            (
                "INSERT INTO t SELECT 3, TIMESTAMP '1000-01-01' FROM t WHERE id = 1",
                vec![],
            ),
            (
                "INSERT INTO t VALUES (1, '2024-01-01') \
                 ON CONFLICT (id) DO UPDATE SET ts = '9999-12-31'",
                vec![],
            ),
        ] {
            match db.execute(sql, params.clone()) {
                Err(e) if e.to_string().contains("outside the range") => {}
                other => wrong.push(format!("sealed {sealed}: {sql} {params:?} gave {other:?}")),
            }
            let now = rows(&db, STORED).unwrap();
            if now != before {
                wrong.push(format!(
                    "sealed {sealed}: {sql} left {now:?}, was {before:?}"
                ));
            }
        }
        db.close().unwrap();
        if sealed {
            let db = Database::open(&dsn).unwrap();
            let reopened = rows(&db, STORED).unwrap();
            if reopened != before {
                wrong.push(format!("reopened {reopened:?}, was {before:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_time_past_the_range_is_an_error_where_it_becomes_nanoseconds() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _) = open("timestamp_range_nanoseconds", false, &dir);
    for sql in [
        "SELECT id FROM t AS OF TIMESTAMP '2263-01-01 00:00:00'",
        "SELECT id FROM t AS OF TIMESTAMP '1600-01-01 00:00:00'",
        "SELECT TIME_TRUNC('1h', TIMESTAMP '9999-12-31')",
    ] {
        match rows(&db, sql) {
            Err(e) if e.contains("outside the range") => {}
            other => panic!("{sql} gave {other:?}, want an error"),
        }
    }
    assert_eq!(
        rows(
            &db,
            "SELECT id FROM t AS OF TIMESTAMP '2262-01-01 00:00:00' ORDER BY id"
        ),
        Ok(vec!["Integer(1)".to_string(), "Integer(2)".to_string()]),
        "a time inside the range reads the rows"
    );
}

#[test]
fn a_schema_change_that_would_store_a_timestamp_past_the_range_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let (db, dsn) = open(&format!("timestamp_range_schema_{sealed}"), sealed, &dir);
        for sql in [
            "CREATE TABLE u (id INTEGER PRIMARY KEY, s TEXT)",
            "INSERT INTO u VALUES (1, '9999-12-31'), (2, '2024-01-01')",
            // d's default is what a volume sealed before d was added reads
            "ALTER TABLE u ADD COLUMN d TEXT DEFAULT '9999-12-31'",
            "UPDATE u SET d = '2024-01-01'",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let before = [
            rows(&db, STORED).unwrap(),
            rows(&db, "SELECT * FROM u ORDER BY id").unwrap(),
        ];
        for sql in [
            "ALTER TABLE t ADD COLUMN far TIMESTAMP DEFAULT '9999-12-31'",
            "ALTER TABLE t ADD COLUMN far TIMESTAMP DEFAULT TIMESTAMP '1600-01-01'",
            "ALTER TABLE u MODIFY COLUMN s TIMESTAMP",
            "ALTER TABLE u MODIFY COLUMN d TIMESTAMP",
        ] {
            match db.execute(sql, ()) {
                Err(e) if e.to_string().contains("outside the range") => {}
                other => wrong.push(format!("sealed {sealed}: {sql} gave {other:?}")),
            }
            let now = [
                rows(&db, STORED).unwrap(),
                rows(&db, "SELECT * FROM u ORDER BY id").unwrap(),
            ];
            if now != before {
                wrong.push(format!(
                    "sealed {sealed}: {sql} left {now:?}, was {before:?}"
                ));
            }
        }
        if let Err(e) = db.execute("PRAGMA CHECKPOINT", ()) {
            wrong.push(format!(
                "sealed {sealed}: the checkpoint after the refusals gave {e}"
            ));
        }
        db.close().unwrap();
        if sealed {
            let db = Database::open(&dsn).unwrap();
            let reopened = [
                rows(&db, STORED).unwrap(),
                rows(&db, "SELECT * FROM u ORDER BY id").unwrap(),
            ];
            if reopened != before {
                wrong.push(format!("reopened {reopened:?}, was {before:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
