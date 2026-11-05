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

//! Every keyed type under every index form, asked every predicate hot,
//! sealed and after a reopen: each answer matches the scan of a twin table
//! without indexes, and a rare key is read through the side file exactly
//! when the side file holds the index

use std::sync::Mutex;

use stoolap::core::{DataType, Value};
use stoolap::Database;

/// The read counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

const ROWS: i64 = 2000;

#[derive(Clone, Copy, PartialEq)]
enum Type {
    Integer,
    Float,
    Timestamp,
    Boolean,
    Text,
}

#[derive(Clone, Copy, PartialEq)]
enum Form {
    BTree,
    Hash,
    Bitmap,
    /// `(a, c)`
    Composite,
}

const FORMS: [Form; 4] = [Form::BTree, Form::Hash, Form::Bitmap, Form::Composite];

fn sql_type(ty: Type) -> &'static str {
    match ty {
        Type::Integer => "INTEGER",
        Type::Float => "FLOAT",
        Type::Timestamp => "TIMESTAMP",
        Type::Boolean => "BOOLEAN",
        Type::Text => "TEXT",
    }
}

fn ts(days: i64) -> Value {
    Value::timestamp(chrono::DateTime::from_timestamp(1_700_000_000 + days * 86_400, 0).unwrap())
}

/// About four rows per regular value, with each type's edge values
fn value(ty: Type, id: i64) -> Value {
    let step = id % 500;
    match (ty, id) {
        (Type::Integer, 0) => Value::Null(DataType::Integer),
        (Type::Integer, 1) => Value::Integer(i64::MIN),
        (Type::Integer, 2) => Value::Integer(i64::MAX),
        (Type::Integer, _) => Value::Integer(step - 250),
        (Type::Float, 0) => Value::Null(DataType::Float),
        (Type::Float, 1) => Value::Float(f64::NAN),
        (Type::Float, 2) => Value::Float(-0.0),
        (Type::Float, 3) => Value::Float(f64::INFINITY),
        (Type::Float, 4) => Value::Float(f64::NEG_INFINITY),
        (Type::Float, 5) => Value::Float(9_007_199_254_740_992.0),
        (Type::Float, _) => Value::Float(step as f64 * 0.5 - 125.0),
        (Type::Timestamp, 0) => Value::Null(DataType::Timestamp),
        (Type::Timestamp, _) => ts(step),
        (Type::Boolean, 0) => Value::Null(DataType::Boolean),
        (Type::Boolean, _) => Value::Boolean(id % 50 == 17),
        (Type::Text, 0) => Value::Null(DataType::Text),
        (Type::Text, 1) => Value::text(""),
        (Type::Text, 2) => Value::text("a\0b"),
        (Type::Text, _) => Value::text(format!("v{step:04}")),
    }
}

/// A value four rows hold, one no row holds, and the two ends of a range
fn probes(ty: Type) -> (Value, Value, Value, Value) {
    match ty {
        Type::Integer => (
            Value::Integer(-3),
            Value::Integer(10_000),
            Value::Integer(-10),
            Value::Integer(10),
        ),
        Type::Float => (
            Value::Float(1.5),
            Value::Float(1.25),
            Value::Float(-0.0),
            Value::Float(4.0),
        ),
        Type::Timestamp => (ts(17), ts(9_000), ts(10), ts(20)),
        Type::Boolean => (
            Value::Boolean(true),
            Value::Null(DataType::Boolean),
            Value::Boolean(false),
            Value::Boolean(true),
        ),
        Type::Text => (
            Value::text("v0017"),
            Value::text("v0017x"),
            Value::text("a"),
            Value::text("v0020"),
        ),
    }
}

/// Values of another type, or none of the type's own, the comparisons
/// must still answer as the scan does
fn strangers(ty: Type) -> Vec<Value> {
    match ty {
        Type::Integer => vec![
            Value::Float(-3.0),
            Value::Float(-3.5),
            Value::Float(f64::NAN),
        ],
        Type::Float => vec![
            Value::Integer(2),
            Value::Integer(0),
            Value::Float(0.0),
            Value::Float(f64::NAN),
        ],
        Type::Timestamp => vec![Value::text("2023-11-14 22:13:20")],
        Type::Boolean => vec![Value::Integer(1)],
        Type::Text => vec![Value::Integer(17), Value::text("a\0b"), Value::text("")],
    }
}

/// Whether the side file holds the index, so a rare key is read through it
fn held(ty: Type, form: Form) -> bool {
    match form {
        Form::Composite => false,
        Form::BTree => ty != Type::Boolean,
        Form::Hash => ty == Type::Text,
        Form::Bitmap => false,
    }
}

fn open(dir: &std::path::Path) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap()
}

/// `t` indexed by `form`, `u` its twin without indexes
fn create(db: &Database, ty: Type, form: Form) -> bool {
    for table in ["t", "u"] {
        db.execute(
            &format!(
                "CREATE TABLE {table} (id INTEGER PRIMARY KEY, a INTEGER, c {})",
                sql_type(ty)
            ),
            (),
        )
        .unwrap();
    }
    let index = match form {
        Form::BTree => "CREATE INDEX t_c ON t(c) USING BTREE",
        Form::Hash => "CREATE INDEX t_c ON t(c) USING HASH",
        Form::Bitmap => "CREATE INDEX t_c ON t(c) USING BITMAP",
        Form::Composite => "CREATE INDEX t_ac ON t(a, c)",
    };
    if db.execute(index, ()).is_err() {
        // A form the type does not take
        return false;
    }
    db.execute("BEGIN", ()).unwrap();
    for table in ["t", "u"] {
        let insert = format!("INSERT INTO {table} VALUES ($1, $2, $3)");
        for id in 0..ROWS {
            db.execute(
                &insert,
                vec![Value::Integer(id), Value::Integer(id % 100), value(ty, id)],
            )
            .unwrap();
        }
    }
    db.execute("COMMIT", ()).unwrap();
    true
}

fn ids(db: &Database, sql: &str, params: &[Value]) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, params.to_vec())
        .unwrap_or_else(|e| panic!("{sql} {params:?}: {e}"))
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn side_reads(db: &Database) -> i64 {
    let rows = db.query("PRAGMA INDEX_READ_STATS", ()).unwrap();
    let columns = rows.columns().to_vec();
    let at = |name: &str| columns.iter().position(|c| c == name).unwrap();
    let (probes, misses) = (at("probes"), at("misses"));
    let row = rows.into_iter().next().unwrap().unwrap();
    row.get::<i64>(probes).unwrap() + row.get::<i64>(misses).unwrap()
}

/// Every predicate on `c`, alone and after an equality on `a`
fn cases(ty: Type) -> Vec<(String, Vec<Value>)> {
    let (rare, absent, low, high) = probes(ty);
    let mut cases = Vec::new();
    for (prefix, lead) in [("", vec![]), ("a = $9 AND ", vec![Value::Integer(17)])] {
        let mut push = |predicate: &str, params: Vec<Value>| {
            // `$9` stands for the leading value, numbered after the rest
            let sql = format!("SELECT id FROM t WHERE {prefix}{predicate}");
            let mut all = params.clone();
            let sql = if lead.is_empty() {
                sql
            } else {
                all.push(lead[0].clone());
                sql.replace("$9", &format!("${}", params.len() + 1))
            };
            cases.push((sql, all));
        };
        for v in [&rare, &absent, &low, &high]
            .into_iter()
            .cloned()
            .chain(strangers(ty))
        {
            for op in ["=", "<", "<=", ">", ">=", "<>"] {
                push(&format!("c {op} $1"), vec![v.clone()]);
            }
        }
        push("c BETWEEN $1 AND $2", vec![low.clone(), high.clone()]);
        push("c >= $1 AND c < $2", vec![low.clone(), high.clone()]);
        push("c IN ($1, $2)", vec![rare.clone(), absent.clone()]);
        push("c NOT IN ($1, $2)", vec![rare.clone(), low.clone()]);
        push("c IS NULL", vec![]);
        push("c IS NOT NULL", vec![]);
    }
    cases
}

/// Each case answers as on the twin; with `served`, the equality on the
/// rare key is read through the side file exactly when the index is held
fn check(db: &Database, ty: Type, form: Form, served: bool) {
    for (sql, params) in cases(ty) {
        let want = ids(db, &sql.replace(" t ", " u "), &params);
        let got = ids(db, &sql, &params);
        assert_eq!(
            got,
            want,
            "{} {:?}: {sql} {params:?}",
            sql_type(ty),
            form_name(form)
        );
    }
    if !served {
        return;
    }
    let (rare, _, _, _) = probes(ty);
    let (sql, params) = match form {
        Form::Composite => {
            let holder = (0..ROWS).find(|&id| value(ty, id) == rare).unwrap();
            let lead = Value::Integer(holder % 100);
            ("SELECT id FROM t WHERE a = $1 AND c = $2", vec![lead, rare])
        }
        _ => ("SELECT id FROM t WHERE c = $1", vec![rare]),
    };
    // A BOOLEAN key is rare only in a composite
    let expect = held(ty, form);
    let before = side_reads(db);
    assert!(!ids(db, sql, &params).is_empty(), "{sql}");
    assert_eq!(
        side_reads(db) > before,
        expect,
        "side file read: {} {:?}: {sql}",
        sql_type(ty),
        form_name(form)
    );
}

fn form_name(form: Form) -> &'static str {
    match form {
        Form::BTree => "btree",
        Form::Hash => "hash",
        Form::Bitmap => "bitmap",
        Form::Composite => "composite",
    }
}

fn run(ty: Type) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for form in FORMS {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        if !create(&db, ty, form) {
            continue;
        }
        check(&db, ty, form, false);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        check(&db, ty, form, true);
        drop(db);
        let db = open(dir.path());
        check(&db, ty, form, true);
    }
}

#[test]
fn integer_keys_under_every_index_form() {
    run(Type::Integer);
}

#[test]
fn float_keys_under_every_index_form() {
    run(Type::Float);
}

#[test]
fn timestamp_keys_under_every_index_form() {
    run(Type::Timestamp);
}

#[test]
fn boolean_keys_under_every_index_form() {
    run(Type::Boolean);
}

#[test]
fn text_keys_under_every_index_form() {
    run(Type::Text);
}
