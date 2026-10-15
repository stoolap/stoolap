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

//! An IN list with a NaN member gives sealed rows the answer hot rows get

use stoolap::core::{DataType, Value};
use stoolap::Database;

const NEG_NAN: u64 = 0xfff8_0000_0000_0000;

fn open(dir: &std::path::Path, indexed: bool) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, f FLOAT, i INTEGER)",
        (),
    )
    .unwrap();
    if indexed {
        db.execute("CREATE INDEX t_f ON t(f)", ()).unwrap();
        db.execute("CREATE INDEX t_i ON t(i)", ()).unwrap();
    }
    let rows = [
        (1, Value::Float(0.0), Value::Integer(0)),
        (2, Value::Float(-0.0), Value::Integer(5)),
        (3, Value::Float(f64::NAN), Value::Integer(1)),
        (4, Value::Float(1.5), Value::Integer(2)),
        (5, Value::Float(5.0), Value::Integer(3)),
        (6, Value::Float(f64::INFINITY), Value::Integer(4)),
        (
            7,
            Value::Null(DataType::Float),
            Value::Null(DataType::Integer),
        ),
    ];
    for (id, f, i) in rows {
        db.execute(
            "INSERT INTO t VALUES ($1, $2, $3)",
            vec![Value::Integer(id), f, i],
        )
        .unwrap();
    }
    db
}

fn ids(db: &Database, sql: &str, params: &[Value]) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, params.to_vec())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// Each case runs on hot rows, then on the same rows sealed, with and
/// without indexes on the column
fn check(cases: &[(&str, Vec<Value>, Vec<i64>)]) {
    for indexed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path(), indexed);
        for (sql, params, want) in cases {
            assert_eq!(
                &ids(&db, sql, params),
                want,
                "hot, indexed={indexed}: {sql} {params:?}"
            );
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        for (sql, params, want) in cases {
            assert_eq!(
                &ids(&db, sql, params),
                want,
                "sealed, indexed={indexed}: {sql} {params:?}"
            );
        }
    }
}

#[test]
fn a_nan_member_at_any_position_keeps_the_other_members() {
    let nan = Value::Float(f64::NAN);
    check(&[
        (
            "SELECT id FROM t WHERE f IN ($1, $2)",
            vec![nan.clone(), Value::Float(1.5)],
            vec![4],
        ),
        (
            "SELECT id FROM t WHERE f IN ($1, $2)",
            vec![Value::Float(5.0), nan.clone()],
            vec![5],
        ),
        (
            "SELECT id FROM t WHERE f IN ($1, $2, $3)",
            vec![Value::Float(1.5), nan.clone(), Value::Float(5.0)],
            vec![4, 5],
        ),
        (
            "SELECT id FROM t WHERE f IN ($1, $2)",
            vec![Value::Float(f64::from_bits(NEG_NAN)), Value::Float(0.0)],
            vec![1, 2],
        ),
        ("SELECT id FROM t WHERE f IN (1.5, $1)", vec![nan], vec![4]),
    ]);
}

#[test]
fn a_list_of_only_nan_matches_nothing() {
    let nan = Value::Float(f64::NAN);
    check(&[
        (
            "SELECT id FROM t WHERE f IN ($1)",
            vec![nan.clone()],
            vec![],
        ),
        (
            "SELECT id FROM t WHERE f IN ($1, $2)",
            vec![nan, Value::Float(f64::from_bits(NEG_NAN))],
            vec![],
        ),
    ]);
}

#[test]
fn a_nan_member_beside_a_null_member() {
    let nan = Value::Float(f64::NAN);
    let null = Value::Null(DataType::Float);
    check(&[
        (
            "SELECT id FROM t WHERE f IN ($1, $2, $3)",
            vec![nan.clone(), null.clone(), Value::Float(1.5)],
            vec![4],
        ),
        (
            "SELECT id FROM t WHERE f IN ($1, $2)",
            vec![null, nan],
            vec![],
        ),
    ]);
}

#[test]
fn not_in_with_a_nan_member() {
    let nan = Value::Float(f64::NAN);
    check(&[
        (
            "SELECT id FROM t WHERE f NOT IN ($1, $2)",
            vec![nan.clone(), Value::Float(1.5)],
            vec![1, 2, 3, 5, 6],
        ),
        (
            "SELECT id FROM t WHERE f NOT IN ($1, $2)",
            vec![nan, Value::Null(DataType::Float)],
            vec![],
        ),
    ]);
}

#[test]
fn an_integer_column_with_a_nan_member() {
    let nan = Value::Float(f64::NAN);
    check(&[
        (
            "SELECT id FROM t WHERE i IN ($1, $2)",
            vec![Value::Float(5.0), nan.clone()],
            vec![2],
        ),
        (
            "SELECT id FROM t WHERE i IN ($1, $2)",
            vec![nan, Value::Integer(2)],
            vec![4],
        ),
    ]);
}
