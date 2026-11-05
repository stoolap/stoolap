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

//! A window function over a ROLLUP, CUBE or GROUPING SETS reads every group
//! of every set: the statement answers with the rows the grouping alone
//! gives, each with its window value, hot and sealed

use stoolap::Database;

/// (grouping with a window, the same grouping alone)
const CASES: &[(&str, &str)] = &[
    (
        "SELECT k, j, SUM(a), COUNT(*) OVER () FROM t GROUP BY ROLLUP (k, j)",
        "SELECT k, j, SUM(a) FROM t GROUP BY ROLLUP (k, j)",
    ),
    (
        "SELECT k, j, SUM(a), GROUPING(k), COUNT(*) OVER (PARTITION BY GROUPING(j)) \
         FROM t GROUP BY CUBE (k, j)",
        "SELECT k, j, SUM(a), GROUPING(k) FROM t GROUP BY CUBE (k, j)",
    ),
    (
        "SELECT k, j, SUM(a), RANK() OVER (PARTITION BY GROUPING(j) ORDER BY SUM(a) DESC) AS r \
         FROM t GROUP BY ROLLUP (k, j)",
        "SELECT k, j, SUM(a) FROM t GROUP BY ROLLUP (k, j)",
    ),
    (
        "SELECT k, j, COUNT(*), COUNT(*) OVER () FROM t WHERE a > 2 \
         GROUP BY GROUPING SETS ((k), (j), ())",
        "SELECT k, j, COUNT(*) FROM t WHERE a > 2 GROUP BY GROUPING SETS ((k), (j), ())",
    ),
    (
        "SELECT k, j, SUM(a), COUNT(*) OVER () FROM (SELECT k, j, a FROM t) d \
         GROUP BY ROLLUP (k, j)",
        "SELECT k, j, SUM(a) FROM (SELECT k, j, a FROM t) d GROUP BY ROLLUP (k, j)",
    ),
];

/// (statement, its rows sorted, each written with `|` between its values)
const ANSWERS: &[(&str, &[&str])] = &[
    (
        "SELECT k, j, COUNT(*) OVER (PARTITION BY GROUPING(j)) FROM t GROUP BY ROLLUP (k, j)",
        &[
            "0|0|6",
            "0|1|6",
            "0|NULL|4",
            "1|0|6",
            "1|1|6",
            "1|NULL|4",
            "2|0|6",
            "2|1|6",
            "2|NULL|4",
            "NULL|NULL|4",
        ],
    ),
    (
        "SELECT k, COUNT(*) OVER w FROM t GROUP BY ROLLUP (k) \
         WINDOW w AS (PARTITION BY GROUPING(k))",
        &["0|3", "1|3", "2|3", "NULL|1"],
    ),
    (
        "SELECT t.k, COUNT(*) OVER (PARTITION BY GROUPING(t.k)) FROM t GROUP BY ROLLUP (t.k)",
        &["0|3", "1|3", "2|3", "NULL|1"],
    ),
    (
        "SELECT t.k, GROUPING(t.k), COUNT(*) FROM t GROUP BY ROLLUP (t.k)",
        &["0|0|4", "1|0|4", "2|0|4", "NULL|1|12"],
    ),
    (
        "SELECT k, GROUPING(t.k), COUNT(*) FROM t GROUP BY ROLLUP (k)",
        &["0|0|4", "1|0|4", "2|0|4", "NULL|1|12"],
    ),
    (
        "SELECT t.k, GROUPING(k), COUNT(*) FROM t GROUP BY ROLLUP (t.k)",
        &["0|0|4", "1|0|4", "2|0|4", "NULL|1|12"],
    ),
    (
        "SELECT t.k AS g, COUNT(*) OVER (PARTITION BY GROUPING(k)) FROM t GROUP BY ROLLUP (g)",
        &["0|3", "1|3", "2|3", "NULL|1"],
    ),
    (
        "SELECT t.k, COUNT(*) OVER (PARTITION BY GROUPING(k)) FROM t GROUP BY ROLLUP (1)",
        &["0|3", "1|3", "2|3", "NULL|1"],
    ),
    (
        "SELECT k AS g, GROUPING(g), GROUPING(k), COUNT(*) FROM t GROUP BY ROLLUP (g)",
        &["0|0|0|4", "1|0|0|4", "2|0|0|4", "NULL|1|1|12"],
    ),
    (
        "SELECT GROUPING(k), COUNT(*) FROM t GROUP BY ROLLUP (k, k)",
        &["0|4", "0|4", "0|4", "0|4", "0|4", "0|4", "1|12"],
    ),
    (
        "SELECT COUNT(*) OVER (PARTITION BY GROUPING(k)) FROM t GROUP BY ROLLUP (k, k)",
        &["1", "6", "6", "6", "6", "6", "6"],
    ),
    (
        "SELECT GROUPING(k), GROUPING(t.k), COUNT(*) FROM t GROUP BY ROLLUP (k, t.k)",
        &[
            "0|0|4", "0|0|4", "0|0|4", "0|1|4", "0|1|4", "0|1|4", "1|1|12",
        ],
    ),
    (
        "SELECT COUNT(*) OVER (PARTITION BY GROUPING(k)) FROM t GROUP BY ROLLUP (k, t.k)",
        &["1", "6", "6", "6", "6", "6", "6"],
    ),
    (
        "SELECT COUNT(*) OVER (PARTITION BY GROUPING(t.k)) FROM t GROUP BY ROLLUP (k, t.k)",
        &["3", "3", "3", "4", "4", "4", "4"],
    ),
    (
        "SELECT GROUPING(k), COUNT(*) FROM t GROUP BY ROLLUP (t.k, t.k)",
        &["0|4", "0|4", "0|4", "0|4", "0|4", "0|4", "1|12"],
    ),
    (
        "SELECT GROUPING(\"οσ\"), COUNT(*) FROM u GROUP BY ROLLUP (\"ΟΣ\", \"οσ\")",
        &["0|1", "1|1", "1|1"],
    ),
    (
        "SELECT COUNT(*) OVER (PARTITION BY GROUPING(\"οσ\")) FROM u \
         GROUP BY ROLLUP (\"ΟΣ\", \"οσ\")",
        &["1", "2", "2"],
    ),
];

fn open(dir: &tempfile::TempDir, sealed: bool, name: &str) -> Database {
    let db = match sealed {
        true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
        false => Database::open(&format!("memory://{name}")).unwrap(),
    };
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, j INTEGER, a INTEGER)",
        "INSERT INTO t SELECT value, value % 3, value % 2, value FROM generate_series(1, 12)",
        // Two columns whose names differ only in case
        "CREATE TABLE u (id INTEGER PRIMARY KEY, \"ΟΣ\" INTEGER, \"οσ\" INTEGER)",
        "INSERT INTO u VALUES (1, 10, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each row of `sql` as text, sorted
fn rows(db: &Database, sql: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

fn check(db: &Database, context: &str) {
    for (windowed, alone) in CASES {
        let alone = rows(db, alone);
        let windowed = rows(db, windowed);
        let width = alone.first().map_or(0, Vec::len);
        let grouped: Vec<Vec<String>> = windowed.iter().map(|row| row[..width].to_vec()).collect();
        assert_eq!(grouped, alone, "{context}: {windowed:?}");
    }
    for (sql, want) in ANSWERS {
        let got: Vec<String> = rows(db, sql).iter().map(|row| row.join("|")).collect();
        assert_eq!(got, *want, "{context}: {sql}");
    }
    // The window counts every group of every set
    let counted = rows(db, "SELECT COUNT(*) OVER () FROM t GROUP BY ROLLUP (k, j)");
    assert_eq!(counted, vec![vec!["10".to_string()]; 10], "{context}");
}

#[test]
fn a_window_reads_every_group_of_every_set_hot_and_sealed() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed, "grouping_sets_window_runner");
        check(&db, &format!("sealed {sealed}"));
    }
}
