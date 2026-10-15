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

//! DISTINCT, GROUP BY and COUNT(DISTINCT) take 0.0 and -0.0 as one value
//! and every NaN as one value; predicates and parameters keep their own rules

use stoolap::core::{DataType, Value};
use stoolap::Database;

const NAN_PAYLOAD: u64 = 0x7ff8_0000_0000_0001;
const NEG_NAN: u64 = 0xfff8_0000_0000_0000;

/// g is 1 on every row; f holds both zeros, three NaNs, 1.5 and two NULLs
fn open(dir: &std::path::Path, indexed: bool, seal: bool) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, f FLOAT)",
        (),
    )
    .unwrap();
    if indexed {
        db.execute("CREATE INDEX t_f ON t(f)", ()).unwrap();
    }
    let values = [
        Value::Float(0.0),
        Value::Float(-0.0),
        Value::Float(f64::NAN),
        Value::Float(f64::from_bits(NAN_PAYLOAD)),
        Value::Float(f64::from_bits(NEG_NAN)),
        Value::Float(1.5),
        Value::Null(DataType::Float),
        Value::Null(DataType::Float),
    ];
    for (id, f) in values.into_iter().enumerate() {
        db.execute(
            "INSERT INTO t VALUES ($1, 1, $2)",
            vec![Value::Integer(id as i64), f],
        )
        .unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// A zero of either sign renders as 0, any NaN as NaN
fn render(v: &Value) -> String {
    match v {
        Value::Null(_) => "NULL".into(),
        Value::Float(x) if x.is_nan() => "NaN".into(),
        Value::Float(x) if *x == 0.0 => "0".into(),
        Value::Float(x) => format!("{x:?}"),
        Value::Integer(x) => x.to_string(),
        other => format!("{other:?}"),
    }
}

fn rows(db: &Database, sql: &str, params: Vec<Value>) -> Vec<String> {
    let mut out: Vec<String> = db
        .query(sql, params)
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            let mut cells = Vec::new();
            let mut i = 0;
            while let Some(v) = row.get_value(i) {
                cells.push(render(v));
                i += 1;
            }
            cells.join("|")
        })
        .collect();
    out.sort();
    out
}

fn check(cases: &[(&str, &[&str])]) {
    for indexed in [false, true] {
        for seal in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let db = open(dir.path(), indexed, seal);
            for (sql, want) in cases {
                assert_eq!(
                    rows(&db, sql, vec![]),
                    *want,
                    "indexed={indexed} sealed={seal}: {sql}"
                );
            }
        }
    }
}

#[test]
fn group_by_merges_zeros_and_nans() {
    check(&[
        (
            "SELECT f, COUNT(*) FROM t GROUP BY f",
            &["0|2", "1.5|1", "NULL|2", "NaN|3"],
        ),
        (
            "SELECT g, f, COUNT(*) FROM t GROUP BY g, f",
            &["1|0|2", "1|1.5|1", "1|NULL|2", "1|NaN|3"],
        ),
    ]);
}

#[test]
fn distinct_merges_zeros_and_nans() {
    check(&[("SELECT DISTINCT f FROM t", &["0", "1.5", "NULL", "NaN"])]);
}

#[test]
fn count_distinct_merges_zeros_and_nans_and_skips_null() {
    check(&[
        ("SELECT COUNT(DISTINCT f) FROM t", &["3"]),
        ("SELECT g, COUNT(DISTINCT f) FROM t GROUP BY g", &["1|3"]),
        ("SELECT COUNT(DISTINCT f) FROM t WHERE id >= 0", &["3"]),
    ]);
}

#[test]
fn predicates_keep_their_rules() {
    check(&[
        ("SELECT COUNT(*) FROM t WHERE f = 0.0", &["2"]),
        ("SELECT COUNT(*) FROM t WHERE f = -0.0", &["2"]),
        ("SELECT COUNT(*) FROM t WHERE f IN (0.0, 1.5)", &["3"]),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), false, true);
    let nan = Value::Float(f64::NAN);
    assert_eq!(
        rows(&db, "SELECT COUNT(*) FROM t WHERE f = $1", vec![nan]),
        ["0"]
    );
}

#[test]
fn parameters_keep_their_sign_and_bits() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), false, false);
    let negative = |p: f64| -> bool {
        db.query_one::<f64, _>("SELECT $1 * 2.0", vec![Value::Float(p)])
            .unwrap()
            .is_sign_negative()
    };
    assert!(!negative(0.0));
    assert!(negative(-0.0));
    assert!(!negative(0.0));
}
