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

//! A HAVING makes a statement aggregate: without a GROUP BY, every row it
//! reads forms one group, over any source, with or without subqueries,
//! nested or not

use stoolap::Database;

/// t: (1, 1), (2, 1), (3, 2); v reads t as it is
fn open(dsn: &str, seal: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 1), (3, 2)",
        "CREATE VIEW v AS SELECT id, g FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each row's values joined by `|`, or the error
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    result
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|"))
        })
        .collect()
}

/// Each source of t's three rows under the name t, with the WITH it needs
const SOURCES: &[(&str, &str)] = &[
    ("", "t"),
    ("", "(SELECT id, g FROM t) t"),
    ("", "v t"),
    ("WITH c AS (SELECT id, g FROM t) ", "c t"),
    ("", "(VALUES (1, 1), (2, 1), (3, 2)) t(id, g)"),
    ("", "t JOIN t t2 ON t2.id = t.id"),
];

/// Each statement over `{s}` and its rows
const SHAPES: &[(&str, &[&str])] = &[
    ("SELECT 1 FROM {s} HAVING COUNT(*) > 0", &["1"]),
    ("SELECT 1 FROM {s} HAVING COUNT(*) > 5", &[]),
    ("SELECT 1 FROM {s} HAVING 1 = 1", &["1"]),
    ("SELECT (SELECT 1) FROM {s} HAVING COUNT(*) > 0", &["1"]),
    ("SELECT 1 FROM {s} HAVING COUNT(*) > (SELECT 0)", &["1"]),
    ("SELECT 1 FROM {s} HAVING COUNT(*) > (SELECT 5)", &[]),
    (
        "SELECT 1 FROM {s} WHERE t.g > (SELECT 1) HAVING COUNT(*) > 0",
        &["1"],
    ),
    (
        "SELECT CASE WHEN COUNT(*) > 2 THEN 'many' END FROM {s} HAVING COUNT(*) > 0",
        &["many"],
    ),
    (
        "SELECT 1 AS x FROM {s} HAVING COUNT(*) > 0 ORDER BY x",
        &["1"],
    ),
    (
        "SELECT 1, ROW_NUMBER() OVER () FROM {s} HAVING COUNT(*) > 0",
        &["1|1"],
    ),
];

#[test]
fn a_having_without_group_by_makes_one_group_of_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let sealed = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let mut wrong = Vec::new();
    for (name, db) in [
        ("hot", open("memory://having_aggregates_test", false)),
        ("sealed", open(&sealed, true)),
    ] {
        for (with, source) in SOURCES {
            for (shape, want) in SHAPES {
                let sql = format!("{with}{}", shape.replace("{s}", source));
                let got = rows(&db, &sql);
                let want: Vec<String> = want.iter().map(|s| s.to_string()).collect();
                if got.as_ref() != Ok(&want) {
                    wrong.push(format!("{name}: {sql}: got {got:?}, want {want:?}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_having_without_group_by_reads_its_columns_as_any_aggregate() {
    let db = open("memory://having_aggregates_columns", false);
    for (with, source) in SOURCES {
        for shape in [
            "SELECT t.g FROM {s} HAVING COUNT(*) > 0",
            "SELECT (SELECT t.g) FROM {s} HAVING COUNT(*) > 0",
        ] {
            let sql = format!("{with}{}", shape.replace("{s}", source));
            assert!(rows(&db, &sql).is_err(), "{sql}: {:?}", rows(&db, &sql));
        }
    }
    let sql = "SELECT SUM(g) OVER (ORDER BY id) AS z FROM t HAVING (SELECT z) > 0";
    assert!(rows(&db, sql).is_err(), "{sql}: {:?}", rows(&db, sql));
}

#[test]
fn a_prepared_having_without_group_by_makes_one_group() {
    let db = open("memory://having_aggregates_prepared", false);
    for sql in [
        "SELECT 1 FROM t HAVING COUNT(*) > $1",
        "SELECT (SELECT 1) FROM t HAVING COUNT(*) > $1",
        "SELECT 1 FROM t JOIN t t2 ON t2.id = t.id HAVING COUNT(*) > $1",
        "SELECT (SELECT 1) FROM (SELECT id FROM t) d HAVING COUNT(*) > $1",
    ] {
        let statement = db.prepare(sql).unwrap();
        for (bound, want) in [(0i64, 1), (5, 0)] {
            let got = statement.query((bound,)).unwrap().count();
            assert_eq!(got, want, "{sql} with {bound}");
        }
    }
}

#[test]
fn a_nested_having_without_group_by_makes_one_group() {
    let db = open("memory://having_aggregates_nested", false);
    let join = "t t1 JOIN t t2 ON t2.id = t1.id";
    for (sql, want) in [
        (
            format!("SELECT (SELECT 1 FROM {join} HAVING COUNT(*) > 0)"),
            vec!["1"],
        ),
        (
            format!(
                "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {join} WHERE t1.g = t.g \
                 HAVING COUNT(*) > 1) ORDER BY id"
            ),
            vec!["1", "2"],
        ),
        (
            format!("SELECT COUNT(*) FROM (SELECT 1 AS x FROM {join} HAVING COUNT(*) > 0) d"),
            vec!["1"],
        ),
        (
            format!(
                "WITH c AS (SELECT 1 AS x FROM {join} HAVING COUNT(*) > 0) SELECT COUNT(*) FROM c"
            ),
            vec!["1"],
        ),
        (
            format!("SELECT 1 FROM {join} HAVING COUNT(*) > 0 UNION ALL SELECT 2"),
            vec!["1", "2"],
        ),
    ] {
        assert_eq!(
            rows(&db, &sql),
            Ok(want.iter().map(|s| s.to_string()).collect()),
            "{sql}"
        );
    }
}

#[test]
fn a_having_without_group_by_counts_a_transaction_s_own_rows() {
    let db = open("memory://having_aggregates_transaction", false);
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 3)", ()).unwrap();
    for sql in [
        "SELECT (SELECT 1) FROM t HAVING COUNT(*) > 3",
        "SELECT 1 FROM t t1 JOIN t t2 ON t2.id = t1.id HAVING COUNT(*) > 3",
    ] {
        assert_eq!(rows(&db, sql), Ok(vec!["1".to_string()]), "{sql}");
    }
    db.execute("ROLLBACK", ()).unwrap();
}
