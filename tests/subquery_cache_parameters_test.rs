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

//! A cached subquery result answers only the parameter values it was taken for

use stoolap::{named_params, DataType, Database, Value};

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (bid INTEGER PRIMARY KEY, bv INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO a VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db.execute("INSERT INTO b VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db
}

fn ids(db: &Database, sql: &str, value: i64) -> Vec<i64> {
    db.query(sql, (value,))
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

const IN_CASES: [(&str, [i64; 3]); 2] = [
    (
        "SELECT id FROM a WHERE id IN (SELECT bid FROM b WHERE bv = $1) ORDER BY id",
        [1, 2, 3],
    ),
    (
        "SELECT id FROM a WHERE x IN (SELECT bid FROM b WHERE bv = $1) ORDER BY id",
        [1, 2, 3],
    ),
];

const EXISTS_CASES: [(&str, [i64; 3]); 2] = [
    (
        "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.bid = a.id AND b.bv = $1) ORDER BY id",
        [1, 2, 3],
    ),
    (
        "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.bid = a.x AND b.bv = $1) ORDER BY id",
        [1, 2, 3],
    ),
];

#[test]
fn test_in_and_exists_subqueries_follow_each_parameter_value() {
    let db = setup("subquery_cache_params_in_exists");
    for (sql, expected) in IN_CASES.iter().chain(EXISTS_CASES.iter()) {
        for (value, id) in [
            (10, expected[0]),
            (20, expected[1]),
            (30, expected[2]),
            (10, expected[0]),
        ] {
            assert_eq!(ids(&db, sql, value), vec![id], "{sql} with {value}");
        }
    }
}

#[test]
fn test_negated_subqueries_follow_each_parameter_value() {
    let db = setup("subquery_cache_params_negated");
    for sql in [
        "SELECT id FROM a WHERE id NOT IN (SELECT bid FROM b WHERE bv = $1) ORDER BY id",
        "SELECT id FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.bid = a.id AND b.bv = $1) ORDER BY id",
    ] {
        for (value, expected) in [(10, vec![2, 3]), (20, vec![1, 3]), (30, vec![1, 2])] {
            assert_eq!(ids(&db, sql, value), expected, "{sql} with {value}");
        }
    }
}

#[test]
fn test_a_prepared_statement_follows_each_parameter_value() {
    let db = setup("subquery_cache_params_prepared");
    for sql in [
        "SELECT id FROM a WHERE id IN (SELECT bid FROM b WHERE bv = $1) ORDER BY id",
        "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.bid = a.id AND b.bv = $1) ORDER BY id",
    ] {
        let statement = db.prepare(sql).unwrap();
        for (value, id) in [(10, 1), (20, 2), (30, 3), (10, 1)] {
            let found: Vec<i64> = statement
                .query((value,))
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect();
            assert_eq!(found, vec![id], "{sql} with {value}");
        }
    }
}

#[test]
fn test_named_parameters_follow_each_value() {
    let db = setup("subquery_cache_params_named");
    for sql in [
        "SELECT id FROM a WHERE id IN (SELECT bid FROM b WHERE bv = :v) ORDER BY id",
        "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.bid = a.id AND b.bv = :v) ORDER BY id",
    ] {
        for (value, id) in [(10i64, 1i64), (20, 2), (30, 3)] {
            let found: Vec<i64> = db
                .query_named(sql, named_params! { v: value })
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect();
            assert_eq!(found, vec![id], "{sql} with {value}");
        }
    }
}

#[test]
fn test_a_parameter_of_another_type_is_another_value() {
    let db = setup("subquery_cache_params_types");
    let queries = [
        "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.bid = a.id AND TYPEOF($1) = 'INTEGER') ORDER BY id",
        "SELECT id FROM a WHERE id IN (SELECT bid FROM b WHERE TYPEOF($1) = 'INTEGER') ORDER BY id",
    ];
    let values = [
        (Value::Integer(1), vec![1, 2, 3]),
        (Value::Float(1.0), vec![]),
        (Value::Integer(1), vec![1, 2, 3]),
        (Value::Null(DataType::Integer), vec![]),
        (Value::Null(DataType::Float), vec![]),
    ];
    for sql in queries {
        for (value, expected) in values.iter() {
            let found: Vec<i64> = db
                .query(sql, (value.clone(),))
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect();
            assert_eq!(&found, expected, "{sql} with {value:?}");
        }
    }
    let named = "SELECT id FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.bid = a.id \
                 AND TYPEOF(:v) = 'INTEGER') ORDER BY id";
    for (value, expected) in [
        (Value::Integer(1), vec![1, 2, 3]),
        (Value::Float(1.0), vec![]),
    ] {
        let found: Vec<i64> = db
            .query_named(named, named_params! { v: value.clone() })
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(found, expected, "{named} with {value:?}");
    }
}
