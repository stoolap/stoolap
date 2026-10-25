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

//! UNION, INTERSECT and EXCEPT answer one row of each value, by the key
//! equality GROUP BY groups by, in the order each value is first met

use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a FLOAT, b TEXT)",
        "INSERT INTO t SELECT value, CASE WHEN value % 7 = 0 THEN NULL ELSE value % 5 END, \
         'b' || (value % 3) FROM generate_series(1, 300)",
        "INSERT INTO w SELECT value, CASE WHEN value % 9 = 0 THEN NULL ELSE (value % 8) * 1.0 END, \
         'b' || (value % 4) FROM generate_series(1, 80)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            format!(
                "{:?}",
                (0..row.len()).map(|i| row.get_value(i)).collect::<Vec<_>>()
            )
        })
        .collect()
}

fn sorted(mut rows: Vec<String>) -> Vec<String> {
    rows.sort();
    rows
}

/// Each statement against a GROUP BY over a UNION ALL that asks the same:
/// `s` tells the side a row came from
fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    let sides = [
        (
            "SELECT a FROM t WHERE id > 20",
            "SELECT a FROM t WHERE id < 40",
        ),
        ("SELECT a, b FROM t", "SELECT a, b FROM t WHERE id < 50"),
        ("SELECT a FROM t", "SELECT a FROM w"),
        ("SELECT a, b FROM w", "SELECT a, b FROM t WHERE a > 1"),
        (
            "SELECT b FROM t WHERE a IS NULL",
            "SELECT b FROM w WHERE id > 70",
        ),
    ];
    for (left, right) in sides {
        let columns = if left.contains("a, b") {
            "a, b"
        } else if left.starts_with("SELECT b") {
            "b"
        } else {
            "a"
        };
        let tagged = format!(
            "SELECT {columns}, 1 AS s FROM ({left}) l UNION ALL SELECT {columns}, 2 FROM ({right}) r"
        );
        for (op, having) in [
            ("UNION", "COUNT(*) > 0"),
            ("INTERSECT", "MIN(s) = 1 AND MAX(s) = 2"),
            ("EXCEPT", "MAX(s) = 1"),
        ] {
            let got = sorted(rows(db, &format!("{left} {op} {right}")));
            let want = sorted(rows(
                db,
                &format!("SELECT {columns} FROM ({tagged}) x GROUP BY {columns} HAVING {having}"),
            ));
            if got != want {
                wrong.push(format!(
                    "{phase}: {left} {op} {right}\n  got  {got:?}\n  want {want:?}"
                ));
            }
        }
    }
}

#[test]
fn a_deduplicating_set_operation_answers_as_the_grouped_union_all() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://set_operation_dedup_hot").unwrap();
    setup(&db);
    check(&db, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    setup(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET a = 9 WHERE id % 50 = 0", ())
        .unwrap();
    check(&db, "sealed with hot updates", &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_chain_of_set_operations_keeps_duplicates_only_under_all() {
    let db = Database::open("memory://set_operation_dedup_chain").unwrap();
    setup(&db);
    let count = |sql: &str| rows(&db, sql).len();
    // t holds 0..4 and NULL, w 0.0..7.0 and NULL: nine values together
    assert_eq!(
        count("SELECT a FROM t UNION ALL SELECT a FROM t WHERE id < 5"),
        304
    );
    assert_eq!(
        count("SELECT a FROM t UNION ALL SELECT a FROM t UNION SELECT a FROM w"),
        9
    );
    assert_eq!(
        count("SELECT a FROM t UNION SELECT a FROM t WHERE id < 5 UNION ALL SELECT a FROM t WHERE id < 3"),
        8
    );
    // INTERSECT binds tighter: 300 rows and the six values both hold
    assert_eq!(
        count("SELECT a FROM t UNION ALL SELECT a FROM t INTERSECT SELECT a FROM w"),
        306
    );
    assert_eq!(
        count(
            "SELECT a FROM (SELECT a FROM t UNION SELECT a FROM w) u \
             EXCEPT ALL SELECT a FROM t WHERE id < 3"
        ),
        7
    );
    assert_eq!(
        count("SELECT a FROM t INTERSECT SELECT a FROM w INTERSECT ALL SELECT a FROM t"),
        6
    );
    // First met first: the left side's values in its order, then the right's new ones
    assert_eq!(
        rows(&db, "SELECT a FROM t WHERE id < 8 UNION SELECT a FROM w WHERE id > 75"),
        rows(&db, "SELECT a FROM t WHERE id IN (1, 2, 3, 4, 5, 7) UNION ALL SELECT a FROM w WHERE id IN (77, 78, 79)")
    );
}
