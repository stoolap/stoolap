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
use stoolap::test_failpoints::{aggregate_paths, aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

/// (context, path the query takes there, query, the same query forced
/// onto the general path)
const CASES: &[(&str, &str, &str, &str)] = &[
    (
        "memory",
        "pushdown",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "pushdown",
        "SELECT SUM(a) FROM t ORDER BY COUNT(*)",
        "SELECT SUM(a) FROM t WHERE id + 0 = id ORDER BY COUNT(*)",
    ),
    (
        "sealed",
        "filtered_pushdown",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a) FROM t WHERE a > 0",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a) FROM t WHERE a > 0 AND id + 0 = id",
    ),
    (
        "sealed",
        "filtered_pushdown",
        "SELECT MAX(a) FROM t WHERE a > 0 ORDER BY COUNT(*)",
        "SELECT MAX(a) FROM t WHERE a > 0 AND id + 0 = id ORDER BY COUNT(*)",
    ),
    (
        "memory",
        "streaming_global",
        "SELECT SUM(a) * 2 AS x, SUM(a), SUM(a) * 2 AS y FROM t",
        "SELECT SUM(a) * 2 AS x, SUM(a), SUM(a) * 2 AS y FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "streaming_global_scan",
        "SELECT COUNT(a) * 2, COUNT(DISTINCT a), COUNT(a) * 2 AS z FROM t",
        "SELECT COUNT(a) * 2, COUNT(DISTINCT a), COUNT(a) * 2 AS z FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "storage_grouped",
        "SELECT SUM(a) AS x, g, SUM(a) AS y, COUNT(*) FROM t GROUP BY g",
        "SELECT SUM(a) AS x, g, SUM(a) AS y, COUNT(*) FROM t WHERE id + 0 = id GROUP BY g",
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
        "index_min_max",
        "SELECT MIN(g) FROM t",
        "SELECT MIN(g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "index_min_max",
        "SELECT MAX(t.g) FROM t",
        "SELECT MAX(t.g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "compiled_count_star",
        "SELECT COUNT(*) AS n FROM t",
        "SELECT COUNT(*) AS n FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "compiled_count_distinct",
        "SELECT COUNT(DISTINCT g) FROM t",
        "SELECT COUNT(DISTINCT g) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "pushdown",
        "SELECT COUNT(DISTINCT g), SUM(a) FROM t",
        "SELECT COUNT(DISTINCT g), SUM(a) FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "streaming_global_scan",
        "SELECT COUNT(DISTINCT s), COUNT(a) * 2 FROM t",
        "SELECT COUNT(DISTINCT s), COUNT(a) * 2 FROM t WHERE id + 0 = id",
    ),
    (
        "memory",
        "prepared_block",
        "SELECT id FROM t WHERE a <= (SELECT MAX(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
        "SELECT DISTINCT id FROM t WHERE a <= (SELECT MAX(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
    ),
    (
        "sealed",
        "prepared_block",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
        "SELECT DISTINCT id FROM t WHERE 1 < (SELECT COUNT(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
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
