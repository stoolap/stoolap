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

//! A function registered in the global registry is the one every clause
//! of every statement calls, hot and sealed, prepared or not: each answers
//! as its twin without the call does. A name a built-in function holds
//! takes no registration

use stoolap::core::{Result, Value};
use stoolap::functions::{
    global_registry, AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature,
    FunctionType, ScalarFunction,
};
use stoolap::{Database, Rows};

/// An aggregate under the built-in SUM's name, answering 99
#[derive(Default)]
struct NinetyNine;

impl AggregateFunction for NinetyNine {
    fn name(&self) -> &str {
        "sum"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "sum",
            FunctionType::Aggregate,
            "test override",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn accumulate(&mut self, _value: &Value) {}
    fn result(&self) -> Result<Value> {
        Ok(Value::Integer(99))
    }
    fn reset(&mut self) {}
    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(NinetyNine)
    }
}

/// A scalar answering 99 under `$name`, a name the engine already reads
macro_rules! scalar_named {
    ($type:ident, $name:literal) => {
        #[derive(Default)]
        struct $type;

        impl ScalarFunction for $type {
            fn name(&self) -> &str {
                $name
            }
            fn info(&self) -> FunctionInfo {
                FunctionInfo {
                    name: $name.into(),
                    ..RegisteredId.info()
                }
            }
            fn evaluate(&self, _args: &[Value]) -> Result<Value> {
                Ok(Value::Integer(99))
            }
            fn clone_box(&self) -> Box<dyn ScalarFunction> {
                Box::new($type)
            }
        }
    };
}

scalar_named!(ScalarNtile, "NTILE");
scalar_named!(TransactionId, "current_transaction_id");
scalar_named!(Grouping, "GROUPING");

/// Answers its argument
#[derive(Default)]
struct RegisteredId;

impl ScalarFunction for RegisteredId {
    fn name(&self) -> &str {
        "REGISTERED_ID"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "REGISTERED_ID",
            FunctionType::Scalar,
            "test identity",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(RegisteredId)
    }
}

/// Each statement reading the tables, `F(` the registered call
const SELECTS: &[&str] = &[
    "SELECT F(a) FROM t ORDER BY id",
    "SELECT id FROM t WHERE F(a) = 20",
    "SELECT id FROM t WHERE id = 1 AND F(1) = 1",
    "SELECT id FROM t WHERE F(a) > $1 ORDER BY id",
    "SELECT id FROM t ORDER BY F(a) DESC LIMIT 2",
    "SELECT F(g), COUNT(*) FROM t GROUP BY F(g) ORDER BY 1",
    "SELECT g, SUM(a) FROM t GROUP BY g HAVING F(SUM(a)) > 30 ORDER BY g",
    "SELECT SUM(F(a)) FROM t",
    "SELECT g, MAX(F(a)) FROM t GROUP BY g ORDER BY g",
    "SELECT CASE WHEN F(a) > 15 THEN 'big' ELSE 'small' END FROM t ORDER BY id",
    "SELECT SUM(F(a)) OVER (PARTITION BY g) FROM t ORDER BY id",
    "SELECT ROW_NUMBER() OVER (ORDER BY F(a) DESC) FROM t ORDER BY id",
    "SELECT t.id FROM t JOIN u ON F(u.k) = t.g ORDER BY t.id, u.id",
    "SELECT id FROM t WHERE a IN (SELECT F(a) FROM t WHERE g = 1) ORDER BY id",
    "SELECT (SELECT F(MAX(a)) FROM t)",
    "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE F(u.k) = t.g) ORDER BY id",
    "SELECT F(x) FROM (VALUES (1), (2)) v(x) ORDER BY 1",
    "WITH c AS (SELECT F(a) AS x FROM t) SELECT x FROM c ORDER BY x",
    "SELECT x FROM (SELECT F(a) AS x FROM t) d ORDER BY x",
    "SELECT DISTINCT ON (F(g)) g, id FROM t ORDER BY F(g), id",
    "SELECT DISTINCT F(g) FROM t ORDER BY 1",
    "SELECT id FROM t ORDER BY id LIMIT F(2)",
];

/// Each write, `F(` the registered call, with the rows of `t` it leaves
const WRITES: &[&str] = &[
    "UPDATE t SET a = F(a) + 1 WHERE id = 1",
    "UPDATE t SET a = 0 WHERE F(a) = 20",
    "DELETE FROM t WHERE F(a) = 30",
    "INSERT INTO t VALUES (9, F(90), 3)",
    "INSERT INTO t SELECT id + 10, F(a), g FROM t WHERE id = 1",
];

fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, g INTEGER)",
        "INSERT INTO t VALUES (1, 10, 1), (2, 20, 1), (3, 30, 2)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn answer(rows: stoolap::Result<Rows>) -> std::result::Result<Vec<String>, String> {
    rows.map_err(|e| e.to_string())?
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|"))
        })
        .collect()
}

fn params(sql: &str) -> Vec<Value> {
    match sql.contains("$1") {
        true => vec![Value::Integer(15)],
        false => Vec::new(),
    }
}

fn table(db: &Database) -> Vec<String> {
    answer(db.query("SELECT id, a, g FROM t ORDER BY id", ())).unwrap()
}

#[test]
fn a_registered_function_is_called_in_every_clause_hot_and_sealed() {
    global_registry().register_scalar::<RegisteredId>().unwrap();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        for shape in SELECTS {
            let (sql, twin) = (
                shape.replace("F(", "REGISTERED_ID("),
                shape.replace("F(", "("),
            );
            let want = answer(db.query(&twin, params(shape)));
            let got = [
                answer(db.query(&sql, params(shape))),
                db.prepare(&sql)
                    .map_err(|e| e.to_string())
                    .and_then(|stmt| answer(stmt.query(params(shape)))),
            ];
            for (how, got) in ["ad hoc", "prepared"].into_iter().zip(got) {
                if want.is_err() || got != want {
                    wrong.push(format!(
                        "sealed {sealed}, {how}: {sql}: {got:?}, want {want:?}"
                    ));
                }
            }
        }
        for shape in WRITES {
            let (sql, twin) = (
                shape.replace("F(", "REGISTERED_ID("),
                shape.replace("F(", "("),
            );
            let (dir, twin_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            let (db, twin_db) = (open(&dir, sealed), open(&twin_dir, sealed));
            let got = db.execute(&sql, ()).map_err(|e| e.to_string());
            let want = twin_db.execute(&twin, ()).map_err(|e| e.to_string());
            if want.is_err() || got != want || table(&db) != table(&twin_db) {
                wrong.push(format!("sealed {sealed}: {sql}: {got:?}, want {want:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_registered_function_is_called_in_a_check_and_a_default() {
    global_registry().register_scalar::<RegisteredId>().unwrap();
    let db = Database::open("memory://registered_function_constraints").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, a INTEGER CHECK (REGISTERED_ID(a) > 0), \
         b INTEGER DEFAULT REGISTERED_ID(7))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c (id, a) VALUES (1, 5)", ())
        .unwrap();
    let refused = db.execute("INSERT INTO c (id, a) VALUES (2, -5)", ());
    assert!(refused.is_err(), "the CHECK let -5 in: {refused:?}");
    let refused = db.execute("UPDATE c SET a = -1 WHERE id = 1", ());
    assert!(refused.is_err(), "the CHECK let the update in: {refused:?}");
    assert_eq!(
        answer(db.query("SELECT id, a, b FROM c", ())),
        Ok(vec!["1|5|7".to_string()])
    );
}

#[test]
fn a_built_in_name_takes_no_registration_and_keeps_its_answer() {
    let registry = global_registry();
    let refused = [
        registry.register_aggregate::<NinetyNine>(),
        registry.register_scalar::<ScalarNtile>(),
        registry.register_scalar::<TransactionId>(),
        registry.register_scalar::<Grouping>(),
        registry.register_tvf(
            "generate_series",
            std::sync::Arc::new(|| Box::new(stoolap::functions::tvf::GenerateSeriesFunction)),
        ),
    ];
    for outcome in &refused {
        let message = outcome.as_ref().map_err(|e| e.to_string()).err();
        assert!(
            message.as_deref().is_some_and(|m| m.contains("built-in")),
            "a built-in name took a registration: {message:?}"
        );
    }
    let db = Database::open("memory://registered_function_built_in").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2)", ())
        .unwrap();
    for (sql, want) in [
        ("SELECT SUM(a) FROM t", vec!["3"]),
        ("SELECT SUM(a + 0) FROM t", vec!["3"]),
        (
            "SELECT SUM(a + 0) OVER () FROM t ORDER BY id",
            vec!["3", "3"],
        ),
        ("SELECT NTILE(2) OVER (ORDER BY id) FROM t", vec!["1", "2"]),
        (
            "SELECT a, GROUPING(a) FROM t GROUP BY ROLLUP(a) ORDER BY 2, 1",
            vec!["1|0", "2|0", "NULL|1"],
        ),
    ] {
        let want: Vec<String> = want.into_iter().map(String::from).collect();
        assert_eq!(answer(db.query(sql, ())), Ok(want), "{sql}");
    }
}
