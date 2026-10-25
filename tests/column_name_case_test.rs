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

//! A WHERE clause answers the same whatever case it writes a column's name in

use stoolap::{Database, Value};

fn ids(db: &Database, sql: &str, params: Vec<Value>) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, params)
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn create(db: &Database, index: bool) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(3), j JSON, k INTEGER, b BOOLEAN, ts TIMESTAMP)",
        "INSERT INTO t VALUES \
         (1, '[1,2,3]', '{\"a\":1}', 1, true, '2024-01-01 00:00:00'), \
         (2, '[4,5,6]', '{\"a\":2}', 2, false, '2024-01-02 00:00:00'), \
         (3, '[1,2,3]', '{\"a\":1}', 3, true, '2024-01-03 00:00:00')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if index {
        for sql in ["CREATE INDEX t_k ON t(k)", "CREATE INDEX t_ts ON t(ts)"] {
            db.execute(sql, ()).unwrap();
        }
    }
}

fn check(db: &Database, phase: &str) {
    let vector = || vec![Value::vector(vec![1.0, 2.0, 3.0])];
    let json = || vec![Value::json("{\"a\":1}")];
    for (sql, params) in [
        ("SELECT id FROM t WHERE v = $1", vector()),
        ("SELECT id FROM t WHERE v <> $1", vector()),
        ("SELECT id FROM t WHERE v IN ($1)", vector()),
        ("SELECT id FROM t WHERE v NOT IN ($1)", vector()),
        ("SELECT id FROM t WHERE v IN ($1) AND k > 0", vector()),
        ("SELECT id FROM t WHERE j = $1", json()),
        ("SELECT id FROM t WHERE j IN ($1)", json()),
        ("SELECT id FROM t WHERE k = '2'", vec![]),
        ("SELECT id FROM t WHERE k IN ('1', '3')", vec![]),
        ("SELECT id FROM t WHERE k > '1'", vec![]),
        ("SELECT id FROM t WHERE b = 'true'", vec![]),
        ("SELECT id FROM t WHERE ts >= '2024-01-02'", vec![]),
        (
            "SELECT id FROM t WHERE ts IN ('2024-01-01 00:00:00')",
            vec![],
        ),
    ] {
        let want = ids(db, sql, params.clone());
        assert!(!want.is_empty(), "{phase}: {sql} reads some rows");
        let column = sql
            .split_whitespace()
            .nth(5)
            .expect("the WHERE clause's column");
        for written in [
            column.to_uppercase(),
            format!("t.{}", column.to_uppercase()),
        ] {
            let other = sql.replacen(&format!("WHERE {column} "), &format!("WHERE {written} "), 1);
            assert_eq!(ids(db, &other, params.clone()), want, "{phase}: {other}");
        }
    }
}

#[test]
fn a_column_name_in_any_case_answers_the_same_in_memory() {
    for index in [false, true] {
        let db = Database::open(&format!(
            "memory://a_column_name_in_any_case_answers_the_same_{index}"
        ))
        .unwrap();
        create(&db, index);
        check(&db, &format!("memory, index {index}"));
    }
}

#[test]
fn a_column_name_in_any_case_answers_the_same_hot_and_sealed() {
    for index in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
        create(&db, index);
        check(&db, &format!("hot, index {index}"));
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        check(&db, &format!("sealed, index {index}"));
    }
}
