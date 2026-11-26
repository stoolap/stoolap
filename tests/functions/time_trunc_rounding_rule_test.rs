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

//! TIME_TRUNC(interval, ts) is the start of the interval holding ts,
//! intervals counted from 1970-01-01 UTC: before 1970 as after, it rounds
//! down.

use stoolap::core::Value;
use stoolap::Database;

/// Before 1970 between buckets, inside one, on an edge, the first stored
/// nanosecond, and one row after 1970
const ROWS: &str = "INSERT INTO t VALUES \
    (1, '1969-12-31 22:30:00'), (2, '1969-12-31 23:59:59.5'), \
    (3, '1969-07-20 20:17:40'), (4, '1969-12-31 23:00:00'), \
    (5, '2024-06-15 10:37:45'), (6, '1677-09-21 00:12:43.145224192')";

/// t holds `ROWS`, hot or sealed
fn open(name: &str, sealed: bool, dir: &tempfile::TempDir) -> Database {
    let dsn = match sealed {
        false => format!("memory://{name}"),
        true => format!("file://{}/{name}", dir.path().display()),
    };
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();
    db.execute(ROWS, ()).unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each row's values, one string per row
fn rows(result: stoolap::Result<stoolap::Rows>) -> Vec<String> {
    result
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| match row.get_value(i) {
                    Some(Value::Timestamp(ts)) => ts.to_rfc3339(),
                    Some(value) => format!("{value}"),
                    None => "missing".into(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[test]
fn time_trunc_rounds_down_before_1970_in_every_context() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let db = open(&format!("time_trunc_rounding_rule_{sealed}"), sealed, &dir);
        let by_hour = vec![
            "1677-09-21T00:00:00+00:00 1",
            "1969-07-20T20:00:00+00:00 1",
            "1969-12-31T22:00:00+00:00 1",
            "1969-12-31T23:00:00+00:00 2",
            "2024-06-15T10:00:00+00:00 1",
        ];
        for (sql, want) in [
            (
                "SELECT TIME_TRUNC('1s', '1969-12-31 23:59:59.5'), \
                 TIME_TRUNC('15m', '1969-07-20 20:17:40'), \
                 TIME_TRUNC('1h', '1969-12-31 23:00:00')",
                vec![
                    "1969-12-31T23:59:59+00:00 1969-07-20T20:15:00+00:00 1969-12-31T23:00:00+00:00",
                ],
            ),
            (
                "SELECT id, TIME_TRUNC('1h', ts) FROM t ORDER BY id",
                vec![
                    "1 1969-12-31T22:00:00+00:00",
                    "2 1969-12-31T23:00:00+00:00",
                    "3 1969-07-20T20:00:00+00:00",
                    "4 1969-12-31T23:00:00+00:00",
                    "5 2024-06-15T10:00:00+00:00",
                    "6 1677-09-21T00:00:00+00:00",
                ],
            ),
            (
                "SELECT id FROM t WHERE TIME_TRUNC('1h', ts) = '1969-12-31 23:00:00' ORDER BY id",
                vec!["2", "4"],
            ),
            (
                "SELECT id, CASE WHEN TIME_TRUNC('1h', ts) < '1970-01-01' THEN 'before' \
                 ELSE 'after' END FROM t ORDER BY id",
                vec![
                    "1 before", "2 before", "3 before", "4 before", "5 after", "6 before",
                ],
            ),
            (
                "SELECT TIME_TRUNC('1h', ts), COUNT(*) FROM t \
                 GROUP BY TIME_TRUNC('1h', ts) ORDER BY 1",
                by_hour.clone(),
            ),
            (
                "SELECT TIME_TRUNC('1h', ts), COUNT(*) FROM t WHERE id + 0 = id \
                 GROUP BY TIME_TRUNC('1h', ts) ORDER BY 1",
                by_hour,
            ),
            (
                "SELECT TIME_TRUNC('1h', ts), COUNT(*) FROM t GROUP BY TIME_TRUNC('1h', ts) \
                 HAVING TIME_TRUNC('1h', ts) = '1969-12-31 22:00:00'",
                vec!["1969-12-31T22:00:00+00:00 1"],
            ),
        ] {
            let got = rows(db.query(sql, ()));
            if got != want {
                wrong.push(format!("sealed={sealed} {sql}: {got:?}"));
            }
        }
        let got = rows(db.query("SELECT TIME_TRUNC($1, $2)", ("1s", "1969-12-31 23:59:59.5")));
        if got != ["1969-12-31T23:59:59+00:00"] {
            wrong.push(format!("sealed={sealed} parameters: {got:?}"));
        }
        let prepared = db
            .prepare("SELECT id FROM t WHERE TIME_TRUNC('1h', ts) = $1 ORDER BY id")
            .unwrap();
        for _ in 0..2 {
            let got = rows(prepared.query(("1969-12-31 22:00:00",)));
            if got != ["1"] {
                wrong.push(format!("sealed={sealed} prepared: {got:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
