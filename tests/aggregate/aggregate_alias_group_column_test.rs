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
//! names is that column, though a SELECT alias has the same name

use stoolap::Database;

/// t's two groups by b sum a alike, so an aggregate read in place of b
/// merges them where b would keep them apart
fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://aggregate_alias_group_column_{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 10), (3, 3, 20)",
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

fn row(values: &[i64]) -> Vec<Option<i64>> {
    values.iter().map(|&v| Some(v)).collect()
}

#[test]
fn a_grouped_column_named_like_a_select_alias_reads_the_column() {
    let db = open("reads");
    let by_b = vec![row(&[3, 10]), row(&[3, 20])];
    let plus_one = vec![row(&[3, 11]), row(&[3, 21])];
    for (sql, want) in [
        ("SELECT SUM(a) AS b, b FROM t GROUP BY b ORDER BY 2", by_b.clone()),
        ("SELECT SUM(a) AS b, t.b FROM t GROUP BY b ORDER BY 2", by_b.clone()),
        ("SELECT SUM(a) AS b, b FROM t GROUP BY t.b ORDER BY 2", by_b.clone()),
        ("SELECT SUM(a) AS b, b AS gb FROM t GROUP BY t.b ORDER BY 2", by_b.clone()),
        ("SELECT SUM(a) AS b, b + 1 FROM t GROUP BY t.b ORDER BY 2", plus_one.clone()),
        ("SELECT SUM(a) AS b, t.b + 1 FROM t GROUP BY b ORDER BY 2", plus_one.clone()),
        (
            "SELECT SUM(a) AS b, CASE WHEN b > 15 THEN 1 ELSE 0 END FROM t GROUP BY t.b ORDER BY 2",
            vec![row(&[3, 0]), row(&[3, 1])],
        ),
        (
            "SELECT a + 1 AS b, b, COUNT(*) FROM t GROUP BY a + 1, t.b ORDER BY 1",
            vec![row(&[2, 10, 1]), row(&[3, 10, 1]), row(&[4, 20, 1])],
        ),
        (
            "SELECT SUM(a) AS b, b, (SELECT 1) FROM t GROUP BY b ORDER BY 2",
            vec![row(&[3, 10, 1]), row(&[3, 20, 1])],
        ),
        (
            "SELECT SUM(a) AS b, SUM(b) OVER () FROM t GROUP BY b ORDER BY 2",
            vec![row(&[3, 30]), row(&[3, 30])],
        ),
        (
            "SELECT SUM(a) AS b, b, ROW_NUMBER() OVER (ORDER BY b DESC) FROM t GROUP BY b ORDER BY 2",
            vec![row(&[3, 10, 2]), row(&[3, 20, 1])],
        ),
        (
            "SELECT SUM(a) AS b, SUM(b) OVER (PARTITION BY b) FROM t GROUP BY b ORDER BY 2",
            by_b.clone(),
        ),
        (
            "SELECT SUM(a) AS b, ROW_NUMBER() OVER (PARTITION BY b ORDER BY b) FROM t GROUP BY b ORDER BY 2",
            vec![row(&[3, 1]), row(&[3, 1])],
        ),
        (
            "SELECT SUM(a) AS b, b AS gb, LAG(b) OVER (ORDER BY b) FROM t GROUP BY b ORDER BY 2",
            vec![vec![Some(3), Some(10), None], row(&[3, 20, 10])],
        ),
        (
            "SELECT SUM(a) AS b FROM t GROUP BY b HAVING b > 15",
            vec![row(&[3])],
        ),
    ] {
        assert_eq!(rows(&db, sql, false), want, "{sql}");
        assert_eq!(rows(&db, sql, true), want, "prepared: {sql}");
    }
    let sql = "SELECT SUM(a) AS b, b + 1, SUM(b) OVER (PARTITION BY b) FROM t GROUP BY b LIMIT 2";
    for prepared in [false, true] {
        let mut got = rows(&db, sql, prepared);
        got.sort();
        assert_eq!(got, vec![row(&[3, 11, 10]), row(&[3, 21, 20])], "{sql}");
    }
}

#[test]
fn a_grouped_column_named_like_a_window_result_reads_the_column() {
    let db = open("window_name");
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, __wf_0_0 INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO w VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let sql = "SELECT ROW_NUMBER() OVER (ORDER BY w.__wf_0_0) + 1 FROM w GROUP BY w.__wf_0_0 ORDER BY __wf_0_0";
    assert_eq!(rows(&db, sql, false), vec![row(&[2]), row(&[3])], "{sql}");
    assert_eq!(
        rows(&db, sql, true),
        vec![row(&[2]), row(&[3])],
        "prepared: {sql}"
    );
}
