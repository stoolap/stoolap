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

//! A CASE that compares an operand against its WHEN values answers with
//! its result and nothing else, wherever the expression around it reads it.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT)", ())
        .unwrap();
    for id in 1..=6i64 {
        db.execute(
            "INSERT INTO t VALUES ($1, $2)",
            (id, if id % 2 == 0 { "a" } else { "b" }),
        )
        .unwrap();
    }
    db
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

fn values(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect()
}

/// The comparison around the CASE reads the CASE's result, not its operand
#[test]
fn test_case_on_the_right_of_a_comparison() {
    let db = setup("case_operand_right");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE id > CASE 1 WHEN 1 THEN 2 ELSE 3 END"
        ),
        4,
        "the first WHEN matches"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE id > CASE 1 WHEN 9 THEN 2 ELSE 3 END"
        ),
        3,
        "the ELSE is taken"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE id > CASE 1 WHEN 1 THEN 2 END"
        ),
        4,
        "no ELSE to take"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE id = CASE 1 WHEN 1 THEN 2 ELSE 3 END"
        ),
        1,
        "equality"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE id > CASE 'x' WHEN 'x' THEN 2 ELSE 3 END"
        ),
        4,
        "a text operand"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE g <> 'z' AND id > CASE 1 WHEN 1 THEN 2 ELSE 3 END"
        ),
        4,
        "beside another predicate"
    );
}

/// A column operand is read per row
#[test]
fn test_case_on_a_column_operand() {
    let db = setup("case_operand_column");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE id > CASE g WHEN 'a' THEN 3 ELSE 2 END"
        ),
        4
    );
    assert_eq!(
        values(
            &db,
            "SELECT CASE g WHEN 'a' THEN 1 WHEN 'b' THEN 2 ELSE 3 END FROM t ORDER BY id"
        ),
        ["2", "1", "2", "1", "2", "1"]
    );
}

/// The same CASE reads the same either side of the comparison
#[test]
fn test_case_reads_the_same_on_either_side() {
    let db = setup("case_operand_sides");
    let right = count(
        &db,
        "SELECT COUNT(*) FROM t WHERE id > CASE 1 WHEN 1 THEN 2 ELSE 3 END",
    );
    let left = count(
        &db,
        "SELECT COUNT(*) FROM t WHERE CASE 1 WHEN 1 THEN 2 ELSE 3 END < id",
    );
    assert_eq!((right, left), (4, 4));
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE CASE 1 WHEN 1 THEN 2 ELSE 3 END < CASE 2 WHEN 2 THEN id ELSE 0 END"
        ),
        4,
        "a CASE on both sides"
    );
}

/// Nesting one CASE inside another leaves nothing of the outer operand
#[test]
fn test_nested_case_operands() {
    let db = setup("case_operand_nested");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t WHERE id > CASE 1 WHEN 1 THEN CASE 2 WHEN 2 THEN 2 ELSE 5 END ELSE 3 END"
        ),
        4
    );
    assert_eq!(
        values(
            &db,
            "SELECT CASE 1 WHEN 1 THEN CASE g WHEN 'a' THEN 'even' ELSE 'odd' END ELSE 'no' END FROM t ORDER BY id"
        ),
        ["odd", "even", "odd", "even", "odd", "even"]
    );
}
