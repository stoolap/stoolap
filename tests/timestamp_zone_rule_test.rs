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

//! A timestamp text may end with a zone after its time, `Z`, `+HH`,
//! `+HHMM` or `+HH:MM`, in the ISO form and the SQL form alike, and reads as
//! that instant in UTC wherever a text is read as a timestamp. A TIMESTAMP or
//! DATE literal whose text does not read as one is an error

use stoolap::{Database, Value};

/// Each text and the UTC instant it names, as `TIMESTAMP '...'` writes it
const FORMS: [(&str, &str); 11] = [
    ("2024-01-10 00:00:00+00:00", "2024-01-10 00:00:00"),
    ("2024-01-10 03:00:00+03:00", "2024-01-10 00:00:00"),
    ("2024-01-10 03:00:00+0300", "2024-01-10 00:00:00"),
    ("2024-01-09 19:00:00-05", "2024-01-10 00:00:00"),
    ("2024-01-10 00:00:00Z", "2024-01-10 00:00:00"),
    ("2024-01-10 00:00:00 +00:00", "2024-01-10 00:00:00"),
    ("2024-01-09 18:30:00.25-05:30", "2024-01-10 00:00:00.25"),
    ("2024-01-09T19:00:00-05", "2024-01-10 00:00:00"),
    ("2024-01-10T00:00:00.25Z", "2024-01-10 00:00:00.25"),
    ("2024-01-10T3:00:00+03:00", "2024-01-10 00:00:00"),
    ("2024-01-10 3:00:00+03", "2024-01-10 00:00:00"),
];

fn truth(db: &Database, sql: &str, params: Vec<Value>) -> Option<bool> {
    let mut rows = db
        .query(sql, params)
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    match rows.next().expect("one row").unwrap().get_value(0) {
        Some(Value::Boolean(b)) => Some(*b),
        Some(Value::Null(_)) => None,
        other => panic!("{sql}: not a truth value: {other:?}"),
    }
}

fn ids(db: &Database, sql: &str, params: Vec<Value>) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, params)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// One row a form, written as a literal; then the same row written as a
/// parameter under the id after it
fn load(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();
    for (i, (form, _)) in FORMS.iter().enumerate() {
        let id = 2 * i as i64;
        db.execute(&format!("INSERT INTO t VALUES ({id}, '{form}')"), ())
            .unwrap_or_else(|e| panic!("{form}: {e}"));
        db.execute(
            "INSERT INTO t VALUES ($1, $2)",
            vec![Value::Integer(id + 1), Value::text(*form)],
        )
        .unwrap_or_else(|e| panic!("{form}: {e}"));
    }
}

fn check(db: &Database) {
    let mut wrong = Vec::new();
    for (i, (form, utc)) in FORMS.iter().enumerate() {
        let same = FORMS
            .iter()
            .enumerate()
            .filter(|(_, (_, other))| other == utc)
            .flat_map(|(j, _)| [2 * j as i64, 2 * j as i64 + 1])
            .collect::<Vec<_>>();
        for sql in [
            format!("SELECT TIMESTAMP '{form}' = TIMESTAMP '{utc}'"),
            format!("SELECT CAST('{form}' AS TIMESTAMP) = TIMESTAMP '{utc}'"),
            format!("SELECT ts = TIMESTAMP '{utc}' FROM t WHERE id = {}", 2 * i),
            format!(
                "SELECT ts = TIMESTAMP '{utc}' FROM t WHERE id = {}",
                2 * i + 1
            ),
            format!("SELECT TIMESTAMP '{utc}' = '{form}'"),
        ] {
            if truth(db, &sql, vec![]) != Some(true) {
                wrong.push(sql);
            }
        }
        for (sql, params) in [
            (format!("SELECT id FROM t WHERE ts = '{form}'"), vec![]),
            (
                "SELECT id FROM t WHERE ts = $1".to_string(),
                vec![Value::text(*form)],
            ),
            (
                format!("SELECT id FROM t WHERE ts >= '{form}' AND ts <= TIMESTAMP '{utc}'"),
                vec![],
            ),
        ] {
            let got = ids(db, &sql, params.clone());
            if got != same {
                wrong.push(format!("{sql} with {params:?}: {got:?}, want {same:?}"));
            }
        }
        let prepared = db.prepare("SELECT id FROM t WHERE ts = $1").unwrap();
        for _ in 0..2 {
            let mut got: Vec<i64> = prepared
                .query(vec![Value::text(*form)])
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect();
            got.sort_unstable();
            if got != same {
                wrong.push(format!("prepared ts = {form:?}: {got:?}"));
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
fn a_zone_after_the_time_reads_as_its_instant_in_every_context() {
    let db = Database::open("memory://a_zone_after_the_time_reads_as_its_instant").unwrap();
    load(&db);
    check(&db);
}

#[test]
fn a_sealed_zone_after_the_time_reads_as_its_instant_in_every_context() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    load(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db);
}

#[test]
fn a_zone_out_of_range_or_not_written_whole_reads_as_no_timestamp() {
    let db = Database::open("memory://a_zone_out_of_range_reads_as_no_timestamp").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '2024-01-10 00:00:00')", ())
        .unwrap();
    for text in [
        "2024-01-10 00:00:00+25:00",
        "2024-01-10T00:00:00+25:00",
        "2024-01-10T03:00:00+03:",
        "2024-01-10 03:00:00+03:",
        "2024-01-10 03:00:00+03:0",
    ] {
        assert_eq!(
            truth(
                &db,
                &format!("SELECT CAST('{text}' AS TIMESTAMP) IS NULL"),
                vec![]
            ),
            Some(true),
            "{text}"
        );
        assert_eq!(
            truth(&db, &format!("SELECT ts = '{text}' FROM t"), vec![]),
            None,
            "{text}"
        );
        assert!(
            db.execute(&format!("INSERT INTO t VALUES (2, '{text}')"), ())
                .is_err(),
            "{text} written"
        );
    }
}

#[test]
fn a_typed_literal_that_reads_as_no_timestamp_is_an_error() {
    let db = Database::open("memory://a_typed_literal_that_reads_as_no_timestamp").unwrap();
    for sql in [
        "SELECT TIMESTAMP '2024-01-10 00:00:00 UTC'",
        "SELECT TIMESTAMP 'not a date'",
        "SELECT DATE '2024-13-01'",
        "SELECT TIMESTAMP '2024-01-10 00:00:00+25:00'",
        "SELECT TIMESTAMP '2024-01-10T03:00:00+03:'",
    ] {
        let failed = db
            .query(sql, ())
            .map_or(true, |mut rows| rows.next().is_some_and(|row| row.is_err()));
        assert!(failed, "{sql} answered");
    }
    for sql in [
        "SELECT TIMESTAMP '2024-01-10'",
        "SELECT TIMESTAMP '12:00:00'",
        "SELECT DATE '2024-01-10'",
    ] {
        assert!(db.query(sql, ()).is_ok(), "{sql} failed");
    }
}
