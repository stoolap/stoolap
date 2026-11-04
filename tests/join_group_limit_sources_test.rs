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

//! A GROUP BY + LIMIT join whose right side is a VALUES list or a table
//! function reads that side from its source with the keys the left side's
//! groups name: it answers as written, hot and sealed, prepared or not

use stoolap::Database;

/// Each statement, its parameters and its rows in order
const CASES: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT t.id, COUNT(*) FROM t JOIN (VALUES (1)) AS v(n) ON t.id = v.n GROUP BY t.id LIMIT 1",
        &[],
        &["1|1"],
    ),
    (
        "SELECT t.id, COUNT(*) FROM t JOIN (VALUES (2), (2), (3)) AS v(n) ON t.id = v.n \
         GROUP BY t.id ORDER BY t.id LIMIT 5",
        &[],
        &["2|2", "3|1"],
    ),
    (
        "SELECT t.id, SUM(v.w) FROM t JOIN (VALUES (1, 10), (3, 30), (3, 5)) AS v(n, w) \
         ON t.id = v.n WHERE v.w > 6 GROUP BY t.id ORDER BY t.id LIMIT 3",
        &[],
        &["1|10", "3|30"],
    ),
    (
        "SELECT t.id, COUNT(*) FROM t JOIN generate_series(5, 9) AS s(n) ON t.id = s.n \
         GROUP BY t.id LIMIT 1",
        &[],
        &["5|1"],
    ),
    (
        "SELECT t.id, SUM(s.n) FROM t JOIN generate_series(5, $1) AS s(n) ON t.id = s.n \
         GROUP BY t.id LIMIT $2",
        &[7, 1],
        &["5|5"],
    ),
    (
        "SELECT t.id, COUNT(*) FROM t JOIN generate_series(1, 3) AS s(n) ON t.id = s.n \
         GROUP BY t.id ORDER BY t.id LIMIT 2",
        &[],
        &["1|1", "2|1"],
    ),
    (
        "SELECT t.id, COUNT(*) FROM t JOIN generate_series(2, 9) AS s(n) ON t.id = s.n \
         GROUP BY t.id ORDER BY t.id LIMIT 2 OFFSET 1",
        &[],
        &["3|1", "4|1"],
    ),
    (
        "SELECT t.id, COUNT(*) FROM t JOIN generate_series(1, $1) AS s(n) ON t.id = s.n \
         GROUP BY t.id ORDER BY t.id LIMIT $2",
        &[3, 2],
        &["1|1", "2|1"],
    ),
];

fn open(dsn: &str, sealed: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 1), (3, 2), (4, 2), (5, 3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(result: stoolap::Result<stoolap::Rows>) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for row in result.map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        let values: Vec<String> = (0..row.len())
            .map(|i| row.get_value(i).unwrap().to_string())
            .collect();
        out.push(values.join("|"));
    }
    Ok(out)
}

#[test]
fn a_grouped_join_reduced_to_its_keys_reads_a_values_or_function_side() {
    let dir = tempfile::tempdir().unwrap();
    let sealed = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let mut wrong = Vec::new();
    for (context, db) in [
        ("hot", open("memory://join_group_limit_sources", false)),
        ("sealed", open(&sealed, true)),
    ] {
        for (sql, params, want) in CASES {
            let params: Vec<stoolap::Value> =
                params.iter().map(|&p| stoolap::Value::Integer(p)).collect();
            let want: Result<Vec<String>, String> =
                Ok(want.iter().map(|row| row.to_string()).collect());
            let statement = db.prepare(sql).unwrap();
            for run in 0..2 {
                let ad_hoc = rows(db.query(sql, params.clone()));
                let prepared = rows(statement.query(params.clone()));
                for (how, got) in [("ad hoc", ad_hoc), ("prepared", prepared)] {
                    if got != want {
                        wrong.push(format!("{context}, {how}, run {run}: {sql}: {got:?}"));
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
