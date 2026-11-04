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

//! A recursive CTE's members, and a statement over a CTE, run on the runner
//! under the statement's plan: hot and sealed, prepared or not, each answers
//! its rows and compiles nothing through the bridge when it runs again

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{bridge_compiles, general_path_total};
use stoolap::{Database, Value};

/// Each statement, its parameters and its rows
const CASES: &[(&str, &[i64], &[&str])] = &[
    (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 50) \
         SELECT COUNT(*), SUM(x) FROM cnt",
        &[],
        &["50|1275"],
    ),
    (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 30) \
         SELECT x FROM cnt WHERE x % 10 = 0 ORDER BY x",
        &[],
        &["10", "20", "30"],
    ),
    (
        "WITH RECURSIVE f(n, a, b) AS (SELECT 1, 0, 1 UNION ALL \
         SELECT n + 1, b, a + b FROM f WHERE n < 20) SELECT n, b FROM f WHERE n = 20",
        &[],
        &["20|6765"],
    ),
    (
        "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM s WHERE n < $1) \
         SELECT SUM(n) FROM s",
        &[10],
        &["55"],
    ),
    (
        "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL \
         SELECT n + 1 FROM s WHERE n < 3 OR n BETWEEN 50 AND 51) SELECT n FROM s ORDER BY n",
        &[],
        &["1", "2", "3"],
    ),
    (
        "WITH RECURSIVE l(n, str) AS (SELECT 1, 'a' UNION ALL \
         SELECT n + 1, str || 'a' FROM l WHERE n < 4) SELECT str FROM l ORDER BY n",
        &[],
        &["a", "aa", "aaa", "aaaa"],
    ),
    (
        "WITH RECURSIVE r(id, x) AS (SELECT 1, 0 UNION ALL \
         SELECT id + 10, x FROM r WHERE id < 10 UNION ALL \
         SELECT id + 1, x + 1 FROM r WHERE id < 3) SELECT id, x FROM r ORDER BY id, x",
        &[],
        &["1|0", "2|1", "3|2", "11|0", "12|1", "13|2"],
    ),
    (
        "WITH RECURSIVE c(id, v) AS (SELECT id, v FROM t WHERE id = 1 UNION ALL \
         SELECT c.id + 1, c.v * 2 FROM c WHERE c.id < 4) SELECT id, v FROM c ORDER BY id",
        &[],
        &["1|10", "2|20", "3|40", "4|80"],
    ),
    (
        "WITH c(id, v) AS (SELECT id, v FROM t) \
         SELECT id, v * 2 FROM c WHERE v > $1 ORDER BY id",
        &[15],
        &["2|40", "3|60"],
    ),
    (
        "WITH c(id, v) AS (SELECT id, v FROM t) SELECT SUM(v), MAX(id) FROM c",
        &[],
        &["60|3"],
    ),
];

fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
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

fn answer(rows: stoolap::Result<stoolap::Rows>) -> Vec<String> {
    rows.unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

#[test]
fn a_recursive_cte_runs_on_the_runner_hot_and_sealed() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        for (sql, params, want) in CASES {
            let stmt = db.prepare(sql).unwrap();
            for run in 0..2 {
                let (bridge, general) = (bridge_compiles(), general_path_total());
                let got = [
                    ("ad hoc", answer(db.query(sql, values(params)))),
                    ("prepared", answer(stmt.query(values(params)))),
                ];
                let compiled = bridge_compiles() - bridge;
                let reached = general_path_total() - general;
                for (how, got) in got {
                    if got != *want {
                        wrong.push(format!("sealed {sealed}, {how}: {sql}: {got:?}"));
                    }
                }
                if run > 0 && (compiled != 0 || reached != 0) {
                    wrong.push(format!(
                        "sealed {sealed}: {sql}: run again, {compiled} bridge compiles, \
                         {reached} general path runs"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
