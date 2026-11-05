// Copyright 2026 Stoolap Contributors
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

//! A grouped column the SELECT list, a window function or its OVER clause
//! names is that column, though an aggregate's alias has the same name

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://aggregate_alias_group_column_{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 10), (3, 4, 20)",
        (),
    )
    .unwrap();
    db
}

fn rows(db: &Database, sql: &str, prepared: bool) -> Vec<Vec<Option<i64>>> {
    let result = if prepared {
        db.prepare(sql).unwrap().query(()).unwrap()
    } else {
        db.query(sql, ()).unwrap()
    };
    let width = result.columns().len();
    result
        .map(|row| {
            let row = row.unwrap();
            (0..width)
                .map(|i| row.get::<Option<i64>>(i).unwrap())
                .collect()
        })
        .collect()
}

#[test]
fn a_grouped_column_named_like_an_aggregate_alias_reads_the_column() {
    let db = open("reads");
    let s = |v: [i64; 2]| vec![Some(v[0]), Some(v[1])];
    let s3 = |v: [Option<i64>; 3]| v.to_vec();
    for (sql, want) in [
        (
            "SELECT SUM(a) AS b, b FROM t GROUP BY b ORDER BY 2",
            vec![s([3, 10]), s([4, 20])],
        ),
        (
            "SELECT SUM(a) AS b, t.b FROM t GROUP BY b ORDER BY 2",
            vec![s([3, 10]), s([4, 20])],
        ),
        (
            "SELECT SUM(a) AS b, b FROM t GROUP BY t.b ORDER BY 2",
            vec![s([3, 10]), s([4, 20])],
        ),
        (
            "SELECT SUM(a) AS b, b, (SELECT 1) FROM t GROUP BY b ORDER BY 2",
            vec![
                s3([Some(3), Some(10), Some(1)]),
                s3([Some(4), Some(20), Some(1)]),
            ],
        ),
        (
            "SELECT SUM(a) AS b, SUM(b) OVER () FROM t GROUP BY b ORDER BY 1",
            vec![s([3, 30]), s([4, 30])],
        ),
        (
            "SELECT SUM(a) AS b, b, ROW_NUMBER() OVER (ORDER BY b DESC) FROM t GROUP BY b ORDER BY 2",
            vec![
                s3([Some(3), Some(10), Some(2)]),
                s3([Some(4), Some(20), Some(1)]),
            ],
        ),
        (
            "SELECT SUM(a) AS b, t.b, SUM(b) OVER (PARTITION BY b) FROM t GROUP BY b ORDER BY 2",
            vec![
                s3([Some(3), Some(10), Some(10)]),
                s3([Some(4), Some(20), Some(20)]),
            ],
        ),
        (
            "SELECT SUM(a) AS b, b AS gb, LAG(b) OVER (ORDER BY b) FROM t GROUP BY b ORDER BY 2",
            vec![
                s3([Some(3), Some(10), None]),
                s3([Some(4), Some(20), Some(10)]),
            ],
        ),
        (
            "SELECT SUM(a) AS b FROM t GROUP BY b HAVING b > 15",
            vec![vec![Some(4)]],
        ),
        (
            "SELECT SUM(a) AS b, b + 1 FROM t GROUP BY b ORDER BY 2",
            vec![s([3, 11]), s([4, 21])],
        ),
    ] {
        assert_eq!(rows(&db, sql, false), want, "{sql}");
        assert_eq!(rows(&db, sql, true), want, "prepared: {sql}");
    }
}
