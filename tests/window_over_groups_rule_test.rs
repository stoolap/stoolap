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

//! A window function over groups reads a GROUP BY expression and the
//! group's aggregates in every part of its call, hot and sealed, prepared or
//! not; a column the GROUP BY does not name is an error in each of them

use stoolap::{Database, Value};

/// Each statement over the groups of `id % 2`, its parameters and its rows
const ANSWERED: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT id % 2, SUM((id % 2) * 10) OVER () FROM t GROUP BY id % 2 ORDER BY 1",
        &[],
        &["0|10", "1|10"],
    ),
    (
        "SELECT id % 2, SUM((id % 2) * $1) OVER () FROM t GROUP BY id % 2 ORDER BY 1",
        &[5],
        &["0|5", "1|5"],
    ),
    (
        "SELECT id % 2, MAX(SUM(a) + id % 2) OVER () FROM t GROUP BY id % 2 ORDER BY 1",
        &[],
        &["0|41", "1|41"],
    ),
    (
        "SELECT id % 2, LAG((id % 2) + 100, 1, 0) OVER (ORDER BY id % 2) FROM t \
         GROUP BY id % 2 ORDER BY 1",
        &[],
        &["0|0", "1|100"],
    ),
    (
        "SELECT id % 2, SUM(COUNT(*)) FILTER (WHERE id % 2 = 1) OVER () FROM t \
         GROUP BY id % 2 ORDER BY 1",
        &[],
        &["0|2", "1|2"],
    ),
    (
        "SELECT id % 2, SUM(SUM(a)) FILTER (WHERE id % 2 = $1) OVER () FROM t \
         GROUP BY id % 2 ORDER BY 1",
        &[0],
        &["0|20", "1|20"],
    ),
    (
        "SELECT id % 2, STRING_AGG(CAST(SUM(a) AS TEXT), ',' ORDER BY id % 2 DESC) OVER () \
         FROM t GROUP BY id % 2 ORDER BY 1",
        &[],
        &["0|40,20", "1|40,20"],
    ),
    (
        "SELECT id % 2, SUM(COUNT(*)) OVER (PARTITION BY (id % 2) * 2) FROM t \
         GROUP BY id % 2 ORDER BY 1",
        &[],
        &["0|1", "1|2"],
    ),
    (
        "SELECT id % 2, ROW_NUMBER() OVER (ORDER BY (id % 2) * -1) FROM t \
         GROUP BY id % 2 ORDER BY 1",
        &[],
        &["0|2", "1|1"],
    ),
];

/// Each statement reading `a` outside its groups in a part of the call
const REFUSED: &[&str] = &[
    "SELECT id % 2, SUM(a) OVER () FROM t GROUP BY id % 2",
    "SELECT id % 2, SUM(a * 2) OVER () FROM t GROUP BY id % 2",
    "SELECT id % 2, LAG(a) OVER (ORDER BY id % 2) FROM t GROUP BY id % 2",
    "SELECT id % 2, SUM(COUNT(*)) FILTER (WHERE a > 0) OVER () FROM t GROUP BY id % 2",
    "SELECT id % 2, STRING_AGG(CAST(SUM(a) AS TEXT), ',' ORDER BY a) OVER () FROM t \
     GROUP BY id % 2",
    "SELECT id % 2, SUM(COUNT(*)) OVER (PARTITION BY a) FROM t GROUP BY id % 2",
    "SELECT id % 2, ROW_NUMBER() OVER (ORDER BY a) FROM t GROUP BY id % 2",
];

fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn values(params: &[i64]) -> Vec<Value> {
    params.iter().map(|p| Value::Integer(*p)).collect()
}

fn answer(rows: stoolap::Result<stoolap::Rows>) -> Result<Vec<String>, String> {
    rows.map_err(|e| e.to_string())?
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|"))
        })
        .collect()
}

#[test]
fn a_window_over_groups_reads_its_groups_in_every_part_of_its_call() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        for (sql, params, want) in ANSWERED {
            let want: Result<Vec<String>, String> =
                Ok(want.iter().map(|s| s.to_string()).collect());
            let stmt = db.prepare(sql).unwrap();
            for run in 0..2 {
                let got = [
                    ("ad hoc", answer(db.query(sql, values(params)))),
                    ("prepared", answer(stmt.query(values(params)))),
                ];
                for (how, got) in got {
                    if got != want {
                        wrong.push(format!("sealed {sealed}, {how}, run {run}: {sql}: {got:?}"));
                    }
                }
            }
        }
        for sql in REFUSED {
            let got = answer(db.query(sql, ()));
            if !got.as_ref().is_err_and(|e| e.contains("'a' not found")) {
                wrong.push(format!("sealed {sealed}: {sql}: {got:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
