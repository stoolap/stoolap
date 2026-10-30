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

//! Every path that builds an aggregate result answers as the general path
//! does, and every path is exercised: a path added without a case here
//! fails the coverage check

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::functions::{
    global_registry, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, NativeFn1,
    ScalarFunction,
};
use stoolap::test_failpoints::{aggregate_paths, aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

/// (context, path the query takes there, query, the same query forced
/// onto the general path)
const CASES: &[(&str, &str, &str, &str)] = &[
    (
        "memory",
        "stats",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT SUM(a) FROM t ORDER BY COUNT(*)",
        "SELECT SUM(a) FROM t WHERE id + 0 = id ORDER BY COUNT(*)",
    ),
    (
        "sealed",
        "storage_filtered",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a) FROM t WHERE a > 0",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a) FROM t WHERE a > 0 AND id + 0 = id",
    ),
    (
        "sealed",
        "storage_filtered",
        "SELECT MAX(a) FROM t WHERE a > 0 ORDER BY COUNT(*)",
        "SELECT MAX(a) FROM t WHERE a > 0 AND id + 0 = id ORDER BY COUNT(*)",
    ),
    (
        "sealed",
        "storage_filtered",
        "SELECT SUM(a) * 2 FROM t WHERE a > 0",
        "SELECT SUM(a) * 2 FROM t WHERE a > 0 AND id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT SUM(a) * 2 AS x, SUM(a), SUM(a) * 2 AS y FROM t",
        "SELECT SUM(a) * 2 AS x, SUM(a), SUM(a) * 2 AS y FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(a) * 2, COUNT(DISTINCT a), COUNT(a) * 2 AS z FROM t",
        "SELECT COUNT(a) * 2, COUNT(DISTINCT a), COUNT(a) * 2 AS z FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT SUM(a) AS x, g, SUM(a) AS y, COUNT(*) FROM t GROUP BY g",
        "SELECT SUM(a) AS x, g, SUM(a) AS y, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g",
    ),
    // An ORDER BY naming a SELECT alias reads the grouped slots all the same
    (
        "memory",
        "storage_grouped",
        "SELECT g, COUNT(*) AS n FROM t GROUP BY g ORDER BY n, g",
        "SELECT g, COUNT(*) AS n FROM t WHERE id + 0 = id GROUP BY g ORDER BY n, g",
    ),
    // Only COUNT counts the rows of a star; any other call of one is grouped
    // by the general path
    (
        "memory",
        "general",
        "SELECT g, SUM(*) FROM t GROUP BY g LIMIT 1",
        "SELECT g, SUM(*) FROM t WHERE id + 0 = id GROUP BY g LIMIT 1",
    ),
    (
        "memory",
        "general",
        "SELECT g, MAX(*) FROM t GROUP BY g LIMIT 1",
        "SELECT g, MAX(*) FROM t WHERE id + 0 = id GROUP BY g LIMIT 1",
    ),
    (
        "sealed",
        "storage_grouped",
        "SELECT g, MAX(a), MAX(a) AS m FROM t GROUP BY g ORDER BY COUNT(*)",
        "SELECT g, MAX(a), MAX(a) AS m FROM t WHERE id + 0 = id GROUP BY g ORDER BY COUNT(*)",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT COUNT(*) AS n, g, COUNT(*) FROM t GROUP BY g HAVING COUNT(*) > 0",
        "SELECT COUNT(*) AS n, g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g \
         HAVING COUNT(*) > 0",
    ),
    (
        "memory",
        "streaming_derived_table",
        "SELECT COUNT(*) AS n, s, COUNT(*) FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT COUNT(*) AS n, s, COUNT(*) FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "sealed",
        "streaming_derived_table",
        "SELECT s, MIN(a), MIN(a) AS lo FROM (SELECT s, a FROM t) d GROUP BY s",
        "SELECT s, MIN(a), MIN(a) AS lo FROM (SELECT s, a FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT g AS n, COUNT(*) AS n FROM t GROUP BY g",
        "SELECT g AS n, COUNT(*) AS n FROM t WHERE id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT g, COUNT(*) FROM t GROUP BY g HAVING MIN(s) > 0 LIMIT 10",
        "SELECT g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g HAVING MIN(s) > 0 LIMIT 10",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT g, COUNT(*) FROM t GROUP BY g HAVING g > 1 OR COUNT(*) + 0 >= 2 * MIN(s) LIMIT 10",
        "SELECT g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g \
         HAVING g > 1 OR COUNT(*) + 0 >= 2 * MIN(s) LIMIT 10",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT g, MAX(s), MIN(a) FROM t GROUP BY g HAVING COUNT(*) > 0 LIMIT 10",
        "SELECT g, MAX(s), MIN(a) FROM t WHERE id + 0 = id GROUP BY g HAVING COUNT(*) > 0 \
         LIMIT 10",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT g AS n, COUNT(*) AS g FROM t GROUP BY g",
        "SELECT g AS n, COUNT(*) AS g FROM t WHERE id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "streaming_group_by",
        "SELECT g AS n, COUNT(*) AS g FROM t GROUP BY g HAVING COUNT(*) > 0 LIMIT 10",
        "SELECT g AS n, COUNT(*) AS g FROM t WHERE id + 0 = id GROUP BY g \
         HAVING COUNT(*) > 0 LIMIT 10",
    ),
    (
        "memory",
        "streaming_derived_table",
        "SELECT s AS n, COUNT(*) AS s FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT s AS n, COUNT(*) AS s FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "memory",
        "streaming_derived_table",
        "SELECT COUNT(*) FROM (SELECT CAST(a AS TEXT) AS \"x.s\", s FROM t) d GROUP BY \"x.s\"",
        "SELECT COUNT(*) FROM (SELECT CAST(a AS TEXT) AS \"x.s\", s FROM t) d WHERE s = s \
         GROUP BY \"x.s\"",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT a AS g, COUNT(*) FROM t GROUP BY g",
        "SELECT a AS g, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "index_bound",
        "SELECT MIN(g) FROM t",
        "SELECT MIN(g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "index_bound",
        "SELECT MAX(t.g) FROM t",
        "SELECT MAX(t.g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "index_bound",
        "SELECT MAX(g) + 1 AS m FROM t",
        "SELECT MAX(g) + 1 AS m FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT COUNT(*) AS n FROM t",
        "SELECT COUNT(*) AS n FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT COUNT(1), COUNT('x') AS n FROM t",
        "SELECT COUNT(1), COUNT('x') AS n FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT COUNT(DISTINCT g) FROM t",
        "SELECT COUNT(DISTINCT g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "stats",
        "SELECT COUNT(DISTINCT g), SUM(a) FROM t",
        "SELECT COUNT(DISTINCT g), SUM(a) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(DISTINCT s), COUNT(a) * 2 FROM t",
        "SELECT COUNT(DISTINCT s), COUNT(a) * 2 FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "prepared_block",
        "SELECT id FROM t WHERE a <= (SELECT MAX(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
        "SELECT id FROM t WHERE a <= (SELECT MAX(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id ORDER BY 1)",
    ),
    (
        "sealed",
        "prepared_block",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id ORDER BY 1)",
    ),
    (
        "memory",
        "index_count",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(*) FROM t AS u WHERE u.g = t.g) LIMIT 10",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(*) FROM t AS u \
         WHERE u.g = t.g AND u.id + 0 = u.id ORDER BY 1) LIMIT 10",
    ),
    (
        "memory",
        "index_count",
        "SELECT id, (SELECT COUNT(*) FROM t AS u WHERE u.g = t.a) AS n FROM t LIMIT 10",
        "SELECT id, (SELECT COUNT(*) FROM t AS u \
         WHERE u.g = t.a AND u.id + 0 = u.id ORDER BY 1) AS n FROM t LIMIT 10",
    ),
    (
        "memory",
        "index_count",
        "SELECT id FROM t WHERE 4 = (SELECT COUNT(*) + COUNT(*) FROM t AS u WHERE u.g = t.g \
         HAVING COUNT(*) > 1) LIMIT 10",
        "SELECT id FROM t WHERE 4 = (SELECT COUNT(*) + COUNT(*) FROM t AS u \
         WHERE u.g = t.g AND u.id + 0 = u.id HAVING COUNT(*) > 1 ORDER BY 1) LIMIT 10",
    ),
    (
        "empty",
        "streaming_derived_table",
        "SELECT COUNT(*) AS n, s, COUNT(*) AS m FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT COUNT(*) AS n, s, COUNT(*) AS m FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
];

fn open(db: Database, seal: bool, rows: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, s TEXT, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t(g)", ()).unwrap();
    if rows {
        db.execute(
            "INSERT INTO t VALUES (1, 1, '1', 1), (2, 1, '1', 2), (3, 2, '2', 10), \
             (4, 2, '2', NULL)",
            (),
        )
        .unwrap();
    }
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
    reset_aggregate_paths();
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
    (answer, aggregate_paths_taken())
}

#[test]
fn every_aggregate_path_answers_as_the_general_path() {
    let memory = open(
        Database::open("memory://aggregate_path_equivalence").unwrap(),
        false,
        true,
    );
    let empty = open(
        Database::open("memory://aggregate_path_equivalence_empty").unwrap(),
        false,
        false,
    );
    let dir = tempfile::tempdir().unwrap();
    let sealed = open(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap(),
        true,
        true,
    );
    let mut wrong = Vec::new();
    let mut covered: Vec<&str> = Vec::new();
    for &(ctx, path, sql, reference) in CASES {
        let db = match ctx {
            "sealed" => &sealed,
            "empty" => &empty,
            _ => &memory,
        };
        let (got, taken) = run(db, sql);
        let (want, reference_taken) = run(db, reference);
        if taken != [path] {
            wrong.push(format!("{ctx}: {sql}: took {taken:?}, expected [{path}]"));
        }
        if reference_taken != ["general"] {
            wrong.push(format!("{ctx}: {reference}: took {reference_taken:?}"));
        }
        if got != want {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, general path {want:?}"));
        }
        covered.extend(taken);
        covered.extend(reference_taken);
    }
    for path in aggregate_paths() {
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
fn a_filtered_aggregate_keeps_every_conjunct_with_literals_and_parameters() {
    let int = Value::Integer;
    // Each WHERE with its parameters; the literal form writes them in
    let cases: &[(&str, Vec<Value>, &str)] = &[
        (
            "a > $1 AND a NOT BETWEEN $2 AND $3",
            vec![int(0), int(1), int(2)],
            "a > 0 AND a NOT BETWEEN 1 AND 2",
        ),
        (
            "id > $1 AND id NOT BETWEEN $2 AND $3",
            vec![int(0), int(1), int(2)],
            "id > 0 AND id NOT BETWEEN 1 AND 2",
        ),
        (
            "a > $1 AND a BETWEEN $2 AND $3",
            vec![int(0), int(2), int(10)],
            "a > 0 AND a BETWEEN 2 AND 10",
        ),
        (
            "a > $1 AND a NOT IN ($2, $3)",
            vec![int(0), int(1), int(10)],
            "a > 0 AND a NOT IN (1, 10)",
        ),
        (
            "a > $1 AND a IN ($2, $3)",
            vec![int(0), int(1), int(10)],
            "a > 0 AND a IN (1, 10)",
        ),
        ("a > $1 AND g IS NULL", vec![int(0)], "a > 0 AND g IS NULL"),
        (
            "a > $1 AND a <> $2",
            vec![int(0), int(2)],
            "a > 0 AND a <> 2",
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    let sealed = open(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap(),
        true,
        true,
    );
    let hot = open(
        Database::open("memory://aggregate_conjuncts_hot").unwrap(),
        false,
        true,
    );
    let mut wrong = Vec::new();
    for (ctx, db) in [("hot", &hot), ("sealed", &sealed)] {
        for (with_params, params, literal) in cases {
            let select = "SELECT COUNT(*), SUM(a), MIN(id) FROM t WHERE";
            let general = run(db, &format!("{select} {literal} AND id + 0 = id")).0;
            let written = run(db, &format!("{select} {literal}")).0;
            let prepared = db
                .query(&format!("{select} {with_params}"), params.clone())
                .map_err(|e| e.to_string())
                .map(|rows| {
                    let rows: Vec<String> = rows
                        .map(|row| {
                            let row = row.unwrap();
                            (0..row.len())
                                .map(|i| render(row.get_value(i).unwrap()))
                                .collect::<Vec<_>>()
                                .join("|")
                        })
                        .collect();
                    rows
                });
            let general_rows = general.clone().map(|(_, rows)| rows);
            if written != general {
                wrong.push(format!(
                    "{ctx}: {literal}: {written:?}, general {general:?}"
                ));
            }
            if prepared != general_rows {
                wrong.push(format!(
                    "{ctx}: {with_params} {params:?}: {prepared:?}, general {general_rows:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_having_that_calls_a_scalar_function_is_not_judged_inside_the_index_walk() {
    let db = open(
        Database::open("memory://aggregate_path_having_scalar_call").unwrap(),
        false,
        true,
    );
    let sql = "SELECT g, COUNT(*) FROM t GROUP BY g HAVING SLEEP(0) IS NOT NULL LIMIT 10";
    let (got, taken) = run(&db, sql);
    assert_eq!(
        taken,
        ["general"],
        "a memory table's B-tree walk holds its index lock while it judges HAVING"
    );
    let (plain, _) = run(&db, "SELECT g, COUNT(*) FROM t GROUP BY g LIMIT 10");
    assert_eq!(got, plain);
}

#[derive(Default)]
struct NativeHaving;

impl ScalarFunction for NativeHaving {
    fn name(&self) -> &str {
        "NATIVE_HAVING_PROBE"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "NATIVE_HAVING_PROBE",
            FunctionType::Scalar,
            "a one-argument function with a native pointer",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> stoolap::core::Result<Value> {
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(NativeHaving)
    }
    fn native_fn1(&self) -> Option<NativeFn1> {
        Some(|_| {})
    }
}

#[test]
fn a_having_that_calls_a_native_function_is_not_judged_inside_the_index_walk() {
    global_registry().register_scalar::<NativeHaving>();
    let db = open(
        Database::open("memory://aggregate_path_having_native_call").unwrap(),
        false,
        true,
    );
    let sql =
        "SELECT g, COUNT(*) FROM t GROUP BY g HAVING NATIVE_HAVING_PROBE(g) IS NOT NULL LIMIT 10";
    let (got, taken) = run(&db, sql);
    assert_eq!(
        taken,
        ["general"],
        "a native pointer runs user code under the walk's index lock as well"
    );
    let (plain, _) = run(&db, "SELECT g, COUNT(*) FROM t GROUP BY g LIMIT 10");
    assert_eq!(got, plain);
}
