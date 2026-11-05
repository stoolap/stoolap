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

//! AND stops at a left operand read as false and OR at one read as true,
//! whatever its type, so the right operand is not evaluated

use stoolap::Database;

/// `p` is a pattern REGEXP refuses, so evaluating `'x' REGEXP p` fails
fn setup(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER, p TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0, '[')", ()).unwrap();
}

fn answers(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    db.query(sql, ())
        .and_then(|rows| {
            rows.map(|row| Ok(format!("{:?}", row?.get_value(0))))
                .collect()
        })
        .map_err(|e| e.to_string())
}

fn cases() -> Vec<(&'static str, &'static str)> {
    vec![
        ("SELECT b AND 'x' REGEXP p FROM t", "Some(Boolean(false))"),
        ("SELECT b + 1 OR 'x' REGEXP p FROM t", "Some(Boolean(true))"),
        ("SELECT 'a' OR 'x' REGEXP p FROM t", "Some(Boolean(true))"),
        (
            "SELECT CASE WHEN b AND 'x' REGEXP p THEN 1 ELSE 2 END FROM t",
            "Some(Integer(2))",
        ),
        (
            "SELECT 7 FROM t WHERE NOT (b AND 'x' REGEXP p)",
            "Some(Integer(7))",
        ),
        (
            "SELECT 7 FROM t WHERE b + 1 OR 'x' REGEXP p",
            "Some(Integer(7))",
        ),
        (
            "SELECT 7 FROM t GROUP BY id, b, p HAVING b + 1 OR 'x' REGEXP p",
            "Some(Integer(7))",
        ),
    ]
}

fn check(db: &Database, rows: usize) -> Vec<String> {
    let mut wrong = Vec::new();
    for (sql, answer) in cases() {
        match answers(db, sql) {
            Ok(got) if got == vec![answer.to_string(); rows] => {}
            other => wrong.push(format!("{sql}: {other:?}")),
        }
    }
    for (sql, param, answer) in [
        (
            "SELECT $1 AND 'x' REGEXP p FROM t",
            0,
            "Some(Boolean(false))",
        ),
        ("SELECT $1 OR 'x' REGEXP p FROM t", 2, "Some(Boolean(true))"),
    ] {
        let statement = db.prepare(sql).unwrap();
        for _ in 0..2 {
            let got: Result<Vec<String>, String> = statement
                .query((param,))
                .and_then(|rows| {
                    rows.map(|row| Ok(format!("{:?}", row?.get_value(0))))
                        .collect()
                })
                .map_err(|e| e.to_string());
            match got {
                Ok(got) if got == vec![answer.to_string(); rows] => {}
                other => wrong.push(format!("prepared {sql}: {other:?}")),
            }
        }
    }
    wrong
}

#[test]
fn a_left_operand_of_any_type_decides_and_and_or() {
    let db = Database::open("memory://short_circuit_hot").unwrap();
    setup(&db);
    let wrong = check(&db, 1);
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn a_left_operand_of_any_type_decides_and_and_or_on_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    setup(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert!(!db.engine().volume_stats().is_empty(), "sealed");
    db.execute("INSERT INTO t VALUES (2, 0, '[')", ()).unwrap();
    let wrong = check(&db, 2);
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}
