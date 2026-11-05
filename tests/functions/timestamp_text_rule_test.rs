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

//! A text reads as one TIMESTAMP whenever and wherever it is read: a time of
//! day alone is that time on 1970-01-01, a date alone is its midnight in any
//! of the documented forms, and a year is written with four digits.

use stoolap::core::Value;
use stoolap::Database;

const NOON: &str = "1970-01-01T12:00:00Z";

fn open(name: &str, sealed: bool, dir: &tempfile::TempDir) -> Database {
    let db = match sealed {
        false => Database::open(&format!("memory://{name}")).unwrap(),
        true => Database::open(&format!("file://{}/{name}", dir.path().display())).unwrap(),
    };
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP, tm TIME, txt TEXT)",
        "INSERT INTO t VALUES (1, '12:00:00', '12:00', '12:00:00'), \
         (2, '2024-01-15 12:00:00', '13:30:00', '2024/01/15')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, txt TEXT)",
        "INSERT INTO u VALUES (1, '12:00:00')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The first column of every row, as its text form, sorted
fn texts(db: &Database, sql: &str, params: Vec<Value>) -> Vec<String> {
    let mut out: Vec<String> = db
        .query(sql, params)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            match row.get_value(0).cloned() {
                Some(Value::Timestamp(ts)) => ts.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                Some(value) => value.to_string(),
                None => "NULL".into(),
            }
        })
        .collect();
    out.sort();
    out
}

#[test]
fn a_time_of_day_alone_is_that_time_on_1970_01_01_everywhere() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let db = open(
            &format!("timestamp_text_time_of_day_{sealed}"),
            sealed,
            &dir,
        );
        let noon = || vec![NOON.to_string()];
        let one = || vec!["1".to_string()];
        let text = |s: &str| vec![Value::text(s)];
        for (sql, params, want) in [
            ("SELECT ts FROM t WHERE id = 1", vec![], noon()),
            ("SELECT tm FROM t WHERE id = 1", vec![], noon()),
            ("SELECT CAST('12:00:00' AS TIMESTAMP)", vec![], noon()),
            ("SELECT TIMESTAMP '12:00:00'", vec![], noon()),
            ("SELECT CAST('12' || ':00' AS TIMESTAMP)", vec![], noon()),
            (
                "SELECT CAST(txt AS TIMESTAMP) FROM t WHERE id = 1",
                vec![],
                noon(),
            ),
            ("SELECT CAST($1 AS TIMESTAMP)", text("12:00"), noon()),
            ("SELECT id FROM t WHERE ts = '12:00:00'", vec![], one()),
            ("SELECT id FROM t WHERE ts = $1", text("12:00:00"), one()),
            ("SELECT id FROM t WHERE tm = '12:00:00'", vec![], one()),
            ("SELECT id FROM t WHERE tm < '13:00'", vec![], one()),
            (
                "SELECT id FROM t WHERE ts IN ('12:00:00', '09:00')",
                vec![],
                one(),
            ),
            (
                "SELECT id FROM t WHERE ts IN (SELECT txt FROM u)",
                vec![],
                one(),
            ),
            (
                "SELECT id FROM t WHERE ts = (SELECT txt FROM u)",
                vec![],
                one(),
            ),
            (
                "SELECT id FROM t WHERE ts IN (SELECT CAST(txt AS TIMESTAMP) FROM u)",
                vec![],
                one(),
            ),
            (
                "SELECT id FROM t WHERE CASE WHEN ts = '12:00:00' THEN 1 ELSE 0 END = 1",
                vec![],
                one(),
            ),
            (
                "SELECT id FROM t GROUP BY id, ts HAVING ts = '12:00:00'",
                vec![],
                one(),
            ),
        ] {
            // Twice, so a statement's kept plan or answer reads as the first run did
            for run in 0..2 {
                let got = texts(&db, sql, params.clone());
                if got != want {
                    wrong.push(format!(
                        "sealed {sealed}, run {run}: {sql} {params:?} gave {got:?}, want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_date_alone_reads_in_every_documented_form_with_a_four_digit_year() {
    let dir = tempfile::tempdir().unwrap();
    let db = open("timestamp_text_dates", false, &dir);
    let mut wrong = Vec::new();
    for (text, want) in [
        ("2024-03-04", "2024-03-04T00:00:00Z"),
        ("2024/03/04", "2024-03-04T00:00:00Z"),
        ("03/04/2024", "2024-03-04T00:00:00Z"),
        ("15/05/2023", "2023-05-15T00:00:00Z"),
        ("2024/03/04 05:06:07", "2024-03-04T05:06:07Z"),
        ("0024-03-04", "0024-03-04T00:00:00Z"),
        ("24-03-04", "NULL"),
        ("3/4/24", "NULL"),
    ] {
        let got = texts(&db, "SELECT CAST($1 AS TIMESTAMP)", vec![Value::text(text)]);
        if got != [want] {
            wrong.push(format!("{text:?} read as {got:?}, want {want}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
