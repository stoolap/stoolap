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

//! A statement whose windows run with its subqueries keeps the window plan
//! it was prepared with, and each run still reads its own parameters and
//! the values its subqueries give

use stoolap::{Database, Statement};

/// Each row's values joined by `|`, the rows joined by spaces
fn run(statement: &Statement, param: Option<i64>) -> String {
    let rows = match param {
        Some(param) => statement.query((param,)),
        None => statement.query(()),
    }
    .unwrap();
    rows.map(|row| {
        let row = row.unwrap();
        (0..row.len())
            .map(|i| row.get_value(i).unwrap().to_string())
            .collect::<Vec<_>>()
            .join("|")
    })
    .collect::<Vec<_>>()
    .join(" ")
}

/// A statement, the parameter of each of two runs, and each run's rows
type Case = (&'static str, [Option<i64>; 2], [&'static str; 2]);

#[test]
fn a_held_window_plan_reads_each_run_s_values() {
    let db = Database::open("memory://grouped_window_plan_test").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, 4)",
        "INSERT INTO u VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let cases: [Case; 5] = [
        (
            "SELECT id, SUM(x) OVER (ORDER BY id ROWS BETWEEN $1 PRECEDING AND CURRENT ROW), \
             (SELECT 5) FROM t",
            [Some(0), Some(2)],
            ["1|1|5 2|2|5 3|3|5 4|4|5", "1|1|5 2|3|5 3|6|5 4|9|5"],
        ),
        (
            "SELECT id, LAG(x, 1, $1) OVER (ORDER BY id), (SELECT 5) FROM t",
            [Some(0), Some(9)],
            ["1|0|5 2|1|5 3|2|5 4|3|5", "1|9|5 2|1|5 3|2|5 4|3|5"],
        ),
        (
            "SELECT id, NTILE($1) OVER (ORDER BY id), (SELECT 5) FROM t",
            [Some(1), Some(2)],
            ["1|1|5 2|1|5 3|1|5 4|1|5", "1|1|5 2|1|5 3|2|5 4|2|5"],
        ),
        (
            "SELECT id, SUM(x) OVER (PARTITION BY x > $1), (SELECT 5) FROM t",
            [Some(0), Some(2)],
            ["1|10|5 2|10|5 3|10|5 4|10|5", "1|3|5 2|3|5 3|7|5 4|7|5"],
        ),
        (
            "SELECT id, SUM(x) OVER w, (SELECT 5) FROM t WINDOW w AS (ORDER BY id \
             ROWS BETWEEN (SELECT COUNT(*) FROM u) PRECEDING AND CURRENT ROW)",
            [None, None],
            ["1|1|5 2|3|5 3|5|5 4|7|5", "1|1|5 2|3|5 3|6|5 4|9|5"],
        ),
    ];
    for (sql, params, want) in cases {
        let statement = db.prepare(sql).unwrap();
        assert_eq!(run(&statement, params[0]), want[0], "{sql}, first run");
        db.execute("INSERT INTO u VALUES (2)", ()).unwrap();
        assert_eq!(run(&statement, params[1]), want[1], "{sql}, second run");
        db.execute("DELETE FROM u WHERE id = 2", ()).unwrap();
    }
}
