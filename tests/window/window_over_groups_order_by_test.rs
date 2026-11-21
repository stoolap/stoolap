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

//! A grouped statement with a window function sorts by any aggregate over
//! its groups, one the SELECT list does not compute included, as a grouped
//! statement without one does: hot and sealed, prepared or not

use stoolap::{Database, Value};

/// Each statement over the groups of `id % 2` (sums 20 and 40), its
/// parameters and its rows in order
const ANSWERED: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT id % 2, COUNT(*) OVER () FROM t GROUP BY id % 2 ORDER BY SUM(a)",
        &[],
        &["0|2", "1|2"],
    ),
    (
        "SELECT id % 2, COUNT(*) OVER () FROM t GROUP BY id % 2 ORDER BY SUM(a) DESC",
        &[],
        &["1|2", "0|2"],
    ),
    (
        "SELECT id % 2, RANK() OVER (ORDER BY id % 2) FROM t GROUP BY id % 2 \
         ORDER BY SUM(a) * $1",
        &[-1],
        &["1|2", "0|1"],
    ),
    (
        "SELECT id % 2, COUNT(*) OVER () FROM t GROUP BY id % 2 HAVING COUNT(*) > 0 \
         ORDER BY MAX(a) - MIN(a) DESC",
        &[],
        &["1|2", "0|2"],
    ),
    (
        "SELECT id % 2, RANK() OVER (ORDER BY SUM(a)) FROM t GROUP BY id % 2 \
         ORDER BY COUNT(*) DESC",
        &[],
        &["1|2", "0|1"],
    ),
    (
        "SELECT DISTINCT ON (SUM(a) > 15) id % 2, COUNT(*) OVER () FROM t \
         GROUP BY id % 2 ORDER BY SUM(a) > 15, SUM(a) DESC",
        &[],
        &["1|2"],
    ),
    (
        "SELECT id % 2 AS k, SUM(COUNT(*)) OVER () FROM t GROUP BY ROLLUP(id % 2) \
         ORDER BY SUM(a) DESC",
        &[],
        &["NULL|6", "1|6", "0|6"],
    ),
    (
        "SELECT id % 2, COUNT(*) OVER () FROM t GROUP BY id % 2 \
         ORDER BY STRING_AGG(CAST(a AS TEXT), ',' ORDER BY a)",
        &[],
        &["1|2", "0|2"],
    ),
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
fn a_grouped_window_statement_sorts_by_an_aggregate_its_select_list_lacks() {
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
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
