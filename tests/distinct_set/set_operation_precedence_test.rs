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

//! UNION and EXCEPT of one level apply left to right, and INTERSECT binds
//! tighter than either, as the same chain grouped by derived tables

use stoolap::Database;

const OPS: [&str; 6] = [
    "UNION",
    "UNION ALL",
    "EXCEPT",
    "EXCEPT ALL",
    "INTERSECT",
    "INTERSECT ALL",
];

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE a (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO a VALUES (1, 1), (2, 1), (3, 2), (4, 3), (5, 3), (6, 3), (7, 4)",
        "CREATE TABLE b (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO b VALUES (1, 1), (2, 3), (3, 3), (4, 5), (5, 5)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO c VALUES (1, 1), (2, 1), (3, 3), (4, 5), (5, 6)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn sorted(db: &Database, sql: &str) -> Vec<i64> {
    let mut rows: Vec<i64> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    rows.sort();
    rows
}

fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    for first in OPS {
        for second in OPS {
            let chain = format!("SELECT v FROM a {first} SELECT v FROM b {second} SELECT v FROM c");
            let tighter = second.starts_with("INTERSECT") && !first.starts_with("INTERSECT");
            let grouped = if tighter {
                format!(
                    "SELECT v FROM a {first} \
                     SELECT v FROM (SELECT v FROM b {second} SELECT v FROM c) y"
                )
            } else {
                format!(
                    "SELECT v FROM (SELECT v FROM a {first} SELECT v FROM b) x \
                     {second} SELECT v FROM c"
                )
            };
            let (got, want) = (sorted(db, &chain), sorted(db, &grouped));
            if got != want {
                wrong.push(format!("{phase}: {chain}\n  got  {got:?}\n  want {want:?}"));
            }
        }
    }
}

#[test]
fn a_set_operation_chain_groups_as_the_standard_says() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://set_operation_precedence").unwrap();
    setup(&db);
    check(&db, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    setup(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_longer_chain_applies_union_and_except_left_to_right() {
    let db = Database::open("memory://set_operation_precedence_literals").unwrap();
    for (sql, want) in [
        ("SELECT 2 AS x UNION SELECT 1 EXCEPT SELECT 2", vec![1]),
        ("SELECT 2 AS x UNION ALL SELECT 1 EXCEPT SELECT 2", vec![1]),
        (
            "SELECT 1 AS x UNION SELECT 2 EXCEPT SELECT 2 UNION SELECT 3",
            vec![1, 3],
        ),
        (
            "SELECT 1 AS x EXCEPT SELECT 1 UNION SELECT 1 INTERSECT SELECT 2",
            vec![],
        ),
    ] {
        assert_eq!(sorted(&db, sql), want, "{sql}");
    }
}
