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
use stoolap::test_failpoints::{
    aggregate_paths, aggregate_paths_taken, reset_aggregate_paths, PREPARED_FAMILIES_OFF,
};
use stoolap::Database;

// The families' switch is process-wide
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// What `run` gives with the prepared families refused, so a subquery block
/// aggregates on the general path
fn families_off<T>(run: impl FnOnce() -> T) -> T {
    use std::sync::atomic::Ordering::SeqCst;
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            PREPARED_FAMILIES_OFF.store(false, SeqCst);
        }
    }
    PREPARED_FAMILIES_OFF.store(true, SeqCst);
    let _restore = Restore;
    run()
}

/// (context, path the query takes there, query, the same query forced
/// onto the general path, run with the prepared families refused)
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
    // A WHERE hot rows hold: the runner feeds the rows its read passes
    (
        "memory",
        "global_scan",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t WHERE a > 0",
        "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y, MIN(a), MAX(a) FROM t WHERE a > 0 \
         AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT SUM(a) FROM t WHERE g = 2 ORDER BY COUNT(*)",
        "SELECT SUM(a) FROM t WHERE g = 2 AND id + 0 = id ORDER BY COUNT(*)",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(DISTINCT a), SUM(a + 1), COUNT(*) FILTER (WHERE a > 1) FROM t WHERE g > 0",
        "SELECT COUNT(DISTINCT a), SUM(a + 1), COUNT(*) FILTER (WHERE a > 1) FROM t \
         WHERE g > 0 AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT SUM(a), COUNT(a), MAX(s) FROM t WHERE a > 100",
        "SELECT SUM(a), COUNT(a), MAX(s) FROM t WHERE a > 100 AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT SUM(a), COUNT(*), MAX(s) FROM t WHERE id = 4",
        "SELECT SUM(a), COUNT(*), MAX(s) FROM t WHERE id = 4 AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT COUNT(*), SUM(a) FROM t WHERE 1 = 0",
        "SELECT COUNT(*), SUM(a) FROM t WHERE 1 = 0 AND id + 0 = id",
    ),
    (
        "memory",
        "global_scan",
        "SELECT 1 + COUNT(*), MIN(s) FROM t WHERE g = 1",
        "SELECT 1 + COUNT(*), MIN(s) FROM t WHERE g = 1 AND id + 0 = id",
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
    // from the rows the read passes
    (
        "memory",
        "hash_groups",
        "SELECT g, SUM(*) FROM t GROUP BY g LIMIT 1",
        "SELECT g, SUM(*) FROM t WHERE id + 0 = id GROUP BY g LIMIT 1",
    ),
    (
        "memory",
        "hash_groups",
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
        "hash_groups",
        "SELECT COUNT(*) AS n, s, COUNT(*) FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT COUNT(*) AS n, s, COUNT(*) FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "sealed",
        "hash_groups",
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
        "hash_groups",
        "SELECT s AS n, COUNT(*) AS s FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT s AS n, COUNT(*) AS s FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "memory",
        "hash_groups",
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
        "SELECT id FROM t WHERE a <= (SELECT MAX(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
    ),
    (
        "sealed",
        "prepared_block",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(a) FROM t AS u WHERE u.g = t.g AND u.id + 0 = u.id)",
    ),
    (
        "memory",
        "index_count",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(*) FROM t AS u WHERE u.g = t.g) LIMIT 10",
        "SELECT id FROM t WHERE 1 < (SELECT COUNT(*) FROM t AS u \
         WHERE u.g = t.g AND u.id + 0 = u.id) LIMIT 10",
    ),
    (
        "memory",
        "index_count",
        "SELECT id, (SELECT COUNT(*) FROM t AS u WHERE u.g = t.a) AS n FROM t LIMIT 10",
        "SELECT id, (SELECT COUNT(*) FROM t AS u \
         WHERE u.g = t.a AND u.id + 0 = u.id) AS n FROM t LIMIT 10",
    ),
    (
        "memory",
        "index_count",
        "SELECT id FROM t WHERE 4 = (SELECT COUNT(*) + COUNT(*) FROM t AS u WHERE u.g = t.g \
         HAVING COUNT(*) > 1) LIMIT 10",
        "SELECT id FROM t WHERE 4 = (SELECT COUNT(*) + COUNT(*) FROM t AS u \
         WHERE u.g = t.g AND u.id + 0 = u.id HAVING COUNT(*) > 1) LIMIT 10",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, s, COUNT(*), SUM(a) FROM t WHERE a > 0 GROUP BY g, s",
        "SELECT g, s, COUNT(*), SUM(a) FROM t WHERE a > 0 AND id + 0 = id GROUP BY g, s",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, s, COUNT(*), SUM(a) FROM t WHERE a > 0 GROUP BY g, s",
        "SELECT g, s, COUNT(*), SUM(a) FROM t WHERE a > 0 AND id + 0 = id GROUP BY g, s",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g + 1 AS k, AVG(a), MIN(s) FROM t GROUP BY g + 1 HAVING COUNT(a) > 0",
        "SELECT g + 1 AS k, AVG(a), MIN(s) FROM t WHERE id + 0 = id GROUP BY g + 1 \
         HAVING COUNT(a) > 0",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a) FROM t GROUP BY ROLLUP (g, s) ORDER BY 1, 2",
        "SELECT g, SUM(a) FROM t WHERE id + 0 = id GROUP BY ROLLUP (g, s) ORDER BY 1, 2",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, s, COUNT(*), GROUPING(g) FROM t GROUP BY CUBE (g, s)",
        "SELECT g, s, COUNT(*), GROUPING(g) FROM t WHERE id + 0 = id GROUP BY CUBE (g, s)",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, s, SUM(a) FROM t GROUP BY GROUPING SETS ((g), (s), ())",
        "SELECT g, s, SUM(a) FROM t WHERE id + 0 = id GROUP BY GROUPING SETS ((g), (s), ())",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT DISTINCT COUNT(*) FROM t WHERE a > 0 GROUP BY g",
        "SELECT DISTINCT COUNT(*) FROM t WHERE a > 0 AND id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g FROM t WHERE a IS NOT NULL GROUP BY g ORDER BY SUM(a) DESC LIMIT 1 OFFSET 1",
        "SELECT g FROM t WHERE a IS NOT NULL AND id + 0 = id GROUP BY g ORDER BY SUM(a) DESC \
         LIMIT 1 OFFSET 1",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT s, COUNT(DISTINCT a), STRING_AGG(s, ',' ORDER BY id) FROM t GROUP BY s",
        "SELECT s, COUNT(DISTINCT a), STRING_AGG(s, ',' ORDER BY id) FROM t \
         WHERE id + 0 = id GROUP BY s",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT COUNT(*), SUM(a) FROM t HAVING SUM(a) > 1",
        "SELECT COUNT(*), SUM(a) FROM t WHERE id + 0 = id HAVING SUM(a) > 1",
    ),
    (
        "memory",
        "stats",
        "SELECT DISTINCT SUM(a) > 0 FROM t",
        "SELECT DISTINCT SUM(a) > 0 FROM t WHERE id + 0 = id",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT COUNT(*), MAX(a) FROM t HAVING COUNT(*) = 0",
        "SELECT COUNT(*), MAX(a) FROM t WHERE id + 0 = id HAVING COUNT(*) = 0",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT g, COUNT(*) FROM t WHERE a > 0 GROUP BY g",
        "SELECT g, COUNT(*) FROM t WHERE a > 0 AND id + 0 = id GROUP BY g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT 42 AS answer, COUNT(*) FROM t WHERE id = 1 GROUP BY GROUPING SETS (())",
        "SELECT 42 AS answer, COUNT(*) FROM t WHERE id = 1 AND id + 0 = id \
         GROUP BY GROUPING SETS (())",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, COUNT(*) FROM t WHERE FALSE GROUP BY g",
        "SELECT g, COUNT(*) FROM t WHERE FALSE AND id + 0 = id GROUP BY g",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT COUNT(*) AS n, s, COUNT(*) AS m FROM (SELECT s FROM t) d GROUP BY s",
        "SELECT COUNT(*) AS n, s, COUNT(*) AS m FROM (SELECT s FROM t) d WHERE s = s GROUP BY s",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT COUNT(*), SUM(a), MAX(s) FROM (SELECT s, a FROM t WHERE a > 0) d",
        "SELECT COUNT(*), SUM(a), MAX(s) FROM (SELECT s, a FROM t WHERE a > 0) d WHERE s = s",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a) FROM (SELECT g, a, s FROM t) d WHERE a > 0 GROUP BY g \
         HAVING COUNT(*) > 0 ORDER BY g",
        "SELECT g, SUM(a) FROM (SELECT g, a, s FROM t) d WHERE a > 0 AND s = s GROUP BY g \
         HAVING COUNT(*) > 0 ORDER BY g",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, COUNT(*), MIN(s) FROM tv GROUP BY g",
        "SELECT g, COUNT(*), MIN(s) FROM tv WHERE s = s GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "WITH c AS (SELECT g, s, a FROM t) SELECT g, s, SUM(a) FROM c GROUP BY ROLLUP (g, s)",
        "WITH c AS (SELECT g, s, a FROM t) SELECT g, s, SUM(a) FROM c WHERE s = s \
         GROUP BY ROLLUP (g, s)",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT x, COUNT(*), SUM(y) FROM (VALUES (1, 2), (1, 3), (2, NULL)) AS v(x, y) GROUP BY x",
        "SELECT x, COUNT(*), SUM(y) FROM (VALUES (1, 2), (1, 3), (2, NULL)) AS v(x, y) \
         WHERE x = x GROUP BY x",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT value % 3 AS r, COUNT(*) FROM generate_series(1, 10) GROUP BY value % 3",
        "SELECT value % 3 AS r, COUNT(*) FROM generate_series(1, 10) WHERE value = value \
         GROUP BY value % 3",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT DISTINCT COUNT(*) FROM (SELECT g FROM t) d GROUP BY g",
        "SELECT DISTINCT COUNT(*) FROM (SELECT g FROM t) d WHERE g = g GROUP BY g",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT COUNT(*), SUM(a) FROM (SELECT a FROM t) d",
        "SELECT COUNT(*), SUM(a) FROM (SELECT a FROM t) d WHERE a = a",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, SUM(a), RANK() OVER (ORDER BY SUM(a) DESC) FROM t GROUP BY g",
        "SELECT g, SUM(a), RANK() OVER (ORDER BY SUM(a) DESC) FROM t WHERE id + 0 = id \
         GROUP BY g",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, s, SUM(a), ROW_NUMBER() OVER (ORDER BY g, s) FROM t GROUP BY ROLLUP (g, s)",
        "SELECT g, s, SUM(a), ROW_NUMBER() OVER (ORDER BY g, s) FROM t WHERE id + 0 = id \
         GROUP BY ROLLUP (g, s)",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT g, s, COUNT(*), COUNT(*) OVER (PARTITION BY GROUPING(s)) FROM t \
         GROUP BY CUBE (g, s)",
        "SELECT g, s, COUNT(*), COUNT(*) OVER (PARTITION BY GROUPING(s)) FROM t \
         WHERE id + 0 = id GROUP BY CUBE (g, s)",
    ),
    (
        "sealed",
        "hash_groups",
        "SELECT g, SUM(a), SUM(SUM(a)) OVER () FROM (SELECT g, a, s FROM t) d \
         WHERE a > 0 GROUP BY g HAVING COUNT(*) > 0 ORDER BY g LIMIT 1",
        "SELECT g, SUM(a), SUM(SUM(a)) OVER () FROM (SELECT g, a, s FROM t) d \
         WHERE a > 0 AND s = s GROUP BY g HAVING COUNT(*) > 0 ORDER BY g LIMIT 1",
    ),
    (
        "memory",
        "hash_groups",
        "SELECT COUNT(*), ROW_NUMBER() OVER () FROM t WHERE a > 0",
        "SELECT COUNT(*), ROW_NUMBER() OVER () FROM t WHERE a > 0 AND id + 0 = id",
    ),
    (
        "empty",
        "hash_groups",
        "SELECT g, COUNT(*), RANK() OVER (ORDER BY g) FROM t GROUP BY g",
        "SELECT g, COUNT(*), RANK() OVER (ORDER BY g) FROM t WHERE id + 0 = id GROUP BY g",
    ),
];

fn open(db: Database, seal: bool, rows: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, s TEXT, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t(g)", ()).unwrap();
    db.execute("CREATE VIEW tv AS SELECT g, s, a FROM t WHERE id > 0", ())
        .unwrap();
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
    let _guard = serial();
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
        let (want, reference_taken) = families_off(|| run(db, reference));
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
    let _guard = serial();
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
    let _guard = serial();
    let db = open(
        Database::open("memory://aggregate_path_having_scalar_call").unwrap(),
        false,
        true,
    );
    let sql = "SELECT g, COUNT(*) FROM t GROUP BY g HAVING SLEEP(0) IS NOT NULL LIMIT 10";
    let (got, taken) = run(&db, sql);
    assert_eq!(
        taken,
        ["hash_groups"],
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
    let _guard = serial();
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
        ["hash_groups"],
        "a native pointer runs user code under the walk's index lock as well"
    );
    let (plain, _) = run(&db, "SELECT g, COUNT(*) FROM t GROUP BY g LIMIT 10");
    assert_eq!(got, plain);
}
