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

//! Every path that answers a DISTINCT answers as the rows path does, and
//! every path is exercised: a path added without a case here fails the
//! coverage check

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::test_failpoints::{distinct_paths, distinct_paths_taken, reset_distinct_paths};
use stoolap::Database;

/// (context, path the query takes there, query, the same query on the rows
/// path)
const CASES: &[(&str, &str, &str, &str)] = &[
    (
        "memory",
        "values",
        "SELECT DISTINCT g FROM t",
        "SELECT DISTINCT g FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "values",
        "SELECT DISTINCT g AS x FROM t ORDER BY x DESC LIMIT 2",
        "SELECT DISTINCT g AS x FROM t WHERE id + 0 = id ORDER BY x DESC LIMIT 2",
    ),
    (
        "sealed",
        "values",
        "SELECT DISTINCT g FROM t",
        "SELECT DISTINCT g FROM t WHERE id + 0 = id",
    ),
    (
        "sealed",
        "values",
        "SELECT DISTINCT s FROM t ORDER BY s",
        "SELECT DISTINCT s FROM t WHERE id + 0 = id ORDER BY s",
    ),
    (
        "mixed",
        "values",
        "SELECT DISTINCT g FROM t",
        "SELECT DISTINCT g FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "rows",
        "SELECT DISTINCT a FROM t WHERE a > 0",
        "SELECT DISTINCT a FROM t WHERE a > 0 AND id + 0 = id",
    ),
    (
        "sealed",
        "rows",
        "SELECT DISTINCT g, s FROM t ORDER BY g, s",
        "SELECT DISTINCT g, s FROM t WHERE id + 0 = id ORDER BY g, s",
    ),
    (
        "memory",
        "scan",
        "SELECT DISTINCT a + 0 FROM t WHERE a > (SELECT MIN(g) FROM t)",
        "SELECT DISTINCT a + 0 FROM t WHERE a > 1 AND id + 0 = id",
    ),
];

fn open(db: Database, seal: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, s TEXT, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t(g)", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 'x', 1), (2, 1, 'x', 2), (3, 2, 'y', 10), \
         (4, 2, NULL, NULL), (5, NULL, 'y', 2)",
        (),
    )
    .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// A value with its type, so INTEGER 3 and FLOAT 3.0 differ
fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        other => format!("{other}"),
    }
}

/// The column names and the rows sorted, or the error
type Answer = Result<(Vec<String>, Vec<String>), String>;

/// The answer to `sql` and the paths that answered it
fn run(db: &Database, sql: &str) -> (Answer, Vec<&'static str>) {
    reset_distinct_paths();
    let answer = db
        .query(sql, ())
        .map_err(|e| e.to_string())
        .and_then(|rows| {
            let columns = rows.columns().to_vec();
            let mut out = Vec::new();
            for row in rows {
                let row = row.map_err(|e| e.to_string())?;
                out.push(
                    (0..row.len())
                        .map(|i| render(row.get_value(i).unwrap()))
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
            out.sort();
            Ok((columns, out))
        });
    (answer, distinct_paths_taken())
}

#[test]
fn every_distinct_path_answers_as_the_rows_path() {
    let memory = open(
        Database::open("memory://distinct_path_equivalence").unwrap(),
        false,
    );
    let sealed_dir = tempfile::tempdir().unwrap();
    let sealed = open(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            sealed_dir.path().display()
        ))
        .unwrap(),
        true,
    );
    let mixed_dir = tempfile::tempdir().unwrap();
    let mixed = open(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            mixed_dir.path().display()
        ))
        .unwrap(),
        true,
    );
    mixed
        .execute(
            "INSERT INTO t VALUES (6, 3, 'z', 3); DELETE FROM t WHERE id = 3",
            (),
        )
        .unwrap();
    let mut wrong = Vec::new();
    let mut covered: Vec<&str> = Vec::new();
    for &(ctx, path, sql, reference) in CASES {
        let db = match ctx {
            "sealed" => &sealed,
            "mixed" => &mixed,
            _ => &memory,
        };
        let (got, taken) = run(db, sql);
        let (want, reference_taken) = run(db, reference);
        if taken != [path] {
            wrong.push(format!("{ctx}: {sql}: took {taken:?}, expected [{path}]"));
        }
        if reference_taken != ["rows"] {
            wrong.push(format!("{ctx}: {reference}: took {reference_taken:?}"));
        }
        if got != want {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, rows path {want:?}"));
        }
        covered.extend(taken);
    }
    for path in distinct_paths() {
        if !covered.contains(&path) {
            wrong.push(format!("no case takes the {path} path"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_transaction_with_its_own_writes_reads_its_rows_not_the_values() {
    let db = open(
        Database::open("memory://distinct_path_transaction").unwrap(),
        false,
    );
    let mut tx = db.begin().unwrap();
    let values = |tx: &mut stoolap::api::Transaction, sql: &str| -> Vec<Option<i64>> {
        reset_distinct_paths();
        let mut values: Vec<Option<i64>> = tx
            .query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<Option<i64>>(0).unwrap())
            .collect();
        values.sort();
        values
    };
    let read = "SELECT DISTINCT g FROM t";
    assert_eq!(values(&mut tx, read), vec![None, Some(1), Some(2)]);
    assert_eq!(distinct_paths_taken(), ["values"], "before its writes");
    tx.execute("INSERT INTO t VALUES (6, 9, 'z', 3)", ())
        .unwrap();
    tx.execute("DELETE FROM t WHERE g = 2", ()).unwrap();
    assert_eq!(values(&mut tx, read), vec![None, Some(1), Some(9)]);
    assert_eq!(distinct_paths_taken(), ["rows"], "after its writes");
    tx.rollback().unwrap();
}
