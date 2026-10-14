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

use stoolap::{named_params, DataType, Database, Value};

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO u VALUES (1, 5), (2, 25), (5, 60), (6, NULL)",
        (),
    )
    .unwrap();
    db
}

#[test]
fn test_pushdown_arithmetic_uses_current_positional_parameter() {
    let db = setup("test_pushdown_arithmetic_uses_current_positional_parameter");
    for predicate in [
        "v <= $1 * 10",
        "$1 * 10 >= v",
        "v <= CAST($1 AS INTEGER) * 10",
        "v <= -$1 * -10",
        "v <= $1 * 10 OR id = 99",
    ] {
        let sql = format!("SELECT id FROM u WHERE {predicate} ORDER BY id");
        for (parameter, expected) in [
            (Value::Integer(3), vec![1, 2]),
            (Value::Integer(1), vec![1]),
            (Value::Integer(7), vec![1, 2, 5]),
            (Value::Null(DataType::Integer), vec![]),
            (Value::Integer(3), vec![1, 2]),
        ] {
            let rows: Vec<i64> = db
                .query(&sql, (parameter,))
                .unwrap()
                .map(|row| row.unwrap().get(0).unwrap())
                .collect();
            assert_eq!(rows, expected, "{predicate}");
        }
    }
}

#[test]
fn test_pushdown_arithmetic_uses_current_named_parameter() {
    let db = setup("test_pushdown_arithmetic_uses_current_named_parameter");
    let sql = "SELECT id FROM u WHERE v <= :factor * 10 ORDER BY id";
    for (factor, expected) in [(3, vec![1, 2]), (1, vec![1]), (7, vec![1, 2, 5])] {
        let rows: Vec<i64> = db
            .query_named(sql, named_params! { factor: factor })
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(rows, expected, "factor = {factor}");
    }
}

#[test]
fn test_indexed_pushdown_evaluates_parameter_arithmetic() {
    let db = setup("test_indexed_pushdown_evaluates_parameter_arithmetic");
    db.execute("CREATE INDEX idx_u_v ON u(v)", ()).unwrap();
    for (sql, parameter, expected) in [
        ("SELECT id FROM u WHERE id = $1 + 1", 1, 2),
        ("SELECT id FROM u WHERE id = $1 + 1", 4, 5),
        ("SELECT id FROM u WHERE v = $1 * 10 + 5", 0, 1),
        ("SELECT id FROM u WHERE v = $1 * 10 + 5", 2, 2),
    ] {
        let rows: Vec<i64> = db
            .query(sql, (parameter,))
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(rows, vec![expected], "{sql} with {parameter}");
    }
}

#[test]
fn test_pushdown_between_and_in_use_parameter_expressions() {
    let db = setup("test_pushdown_between_and_in_use_parameter_expressions");
    for (predicate, expected) in [
        ("v BETWEEN $1 * 10 - 10 AND $1 * 10", vec![2]),
        ("v IN ($1 * 10 - 5, $1 * 10 + 30)", vec![2, 5]),
    ] {
        let sql = format!("SELECT id FROM u WHERE {predicate} ORDER BY id");
        let rows: Vec<i64> = db
            .query(&sql, (3,))
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(rows, expected, "{predicate}");
    }
}

#[test]
fn test_dml_pushdown_evaluates_parameter_arithmetic() {
    let db = setup("test_dml_pushdown_evaluates_parameter_arithmetic");
    assert_eq!(
        db.execute("UPDATE u SET v = v + 1 WHERE v <= $1 * 10", (3,))
            .unwrap(),
        2
    );
    assert_eq!(
        db.execute("DELETE FROM u WHERE v <= $1 * 10", (1,))
            .unwrap(),
        1
    );
    let rows: Vec<i64> = db
        .query("SELECT id FROM u ORDER BY id", ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(rows, vec![2, 5, 6]);
}
