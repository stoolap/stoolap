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

use std::sync::{Mutex, MutexGuard};

use stoolap::Database;

const REFUSED: &str = "is allowed only in an aggregate call";

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, a INTEGER, s TEXT)",
        "INSERT INTO t VALUES (1, 1, 30, 'x'), (2, 2, -10, 'y'), (3, 1, NULL, 'x')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 2, 7)",
        "CREATE INDEX t_a ON t (a)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, v VECTOR(2))",
        "INSERT INTO w VALUES (1, '[1,0]'), (2, '[0,1]')",
        "CREATE TABLE h (id INTEGER PRIMARY KEY, v VECTOR(2))",
        "INSERT INTO h VALUES (1, '[1,0]'), (2, '[0,1]')",
        "CREATE INDEX h_v ON h (v) USING HNSW",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The error `sql` gives, run as a query or as a statement
fn error(db: &Database, sql: &str) -> Option<String> {
    let reads = sql.starts_with("SELECT") || sql.starts_with("WITH");
    let outcome = match reads {
        true => db
            .query(sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .map(|_| ()),
        false => db.execute(sql, ()).map(|_| ()),
    };
    outcome.err().map(|e| e.to_string())
}

#[test]
fn a_modifier_on_a_call_to_no_aggregate_is_an_error_everywhere() {
    let _guard = serial();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("aggregate_modifier_everywhere", &dir, sealed);
        let mut wrong = Vec::new();
        for call in [
            "ABS(DISTINCT a)",
            "ABS(a ORDER BY t.id)",
            "ABS(a) FILTER (WHERE a > 0)",
        ] {
            for sql in [
                format!("SELECT {call} FROM t"),
                format!("SELECT id FROM t WHERE {call} > 5"),
                format!("SELECT id FROM t WHERE {call} = 30"),
                format!("SELECT id FROM t WHERE {call} > 5 AND id > (SELECT 0)"),
                format!("SELECT id FROM t ORDER BY {call}"),
                format!("SELECT CASE WHEN {call} > 0 THEN 1 ELSE 0 END FROM t"),
                format!("SELECT k FROM t GROUP BY k HAVING MAX({call}) > 0"),
                format!("SELECT {call} AS g, COUNT(*) FROM t GROUP BY g"),
                format!("SELECT t.id FROM t JOIN u ON u.k = t.k AND {call} > 0"),
                format!("SELECT id, (SELECT {call} FROM t t2 WHERE t2.id = t.id) FROM t"),
                format!(
                    "SELECT id FROM u WHERE EXISTS (SELECT 1 FROM t WHERE t.k = u.k AND {call} > 0)"
                ),
                format!("SELECT id FROM u WHERE v IN (SELECT {call} FROM t)"),
                format!("SELECT id FROM u WHERE EXISTS (SELECT {call} FROM t)"),
                format!("SELECT EXISTS (SELECT {call} AS x FROM t WHERE t.k = 1)"),
                format!("SELECT id FROM u WHERE EXISTS (SELECT {call} FROM t ORDER BY id LIMIT 1)"),
                format!("SELECT EXISTS (SELECT {call} FROM t JOIN u ON u.k = t.k)"),
                format!("SELECT id, SUM(k) OVER (PARTITION BY {call}) FROM t"),
                format!("WITH c AS (SELECT {call} AS x FROM t) SELECT x FROM c"),
                format!("SELECT x FROM (SELECT {call} AS x FROM t) d"),
                format!("UPDATE t SET k = k WHERE {call} > 100"),
                format!("UPDATE t SET k = {call} WHERE id > 100"),
                format!("DELETE FROM t WHERE {call} > 100"),
                format!("INSERT INTO u SELECT id + 100, k, {call} FROM t WHERE id > 100"),
            ] {
                match error(&db, &sql) {
                    Some(e) if e.contains(REFUSED) => {}
                    other => wrong.push(format!("sealed {sealed}: {sql}: {other:?}")),
                }
            }
        }
        for sql in [
            "SELECT ABS(DISTINCT 3)",
            "SELECT COALESCE(DISTINCT a, 0) FROM t",
            "SELECT UPPER(s ORDER BY id) FROM t",
            "SELECT ROW_NUMBER(DISTINCT a) OVER () FROM t",
            "SELECT id FROM u WHERE EXISTS (SELECT ROW_NUMBER(DISTINCT a) OVER () FROM t)",
            "SELECT id FROM w ORDER BY VEC_DISTANCE_L2(DISTINCT v, '[0,0]') LIMIT 1",
            "SELECT id FROM h ORDER BY VEC_DISTANCE_L2(DISTINCT v, '[0,0]') LIMIT 1",
            "SELECT id FROM h ORDER BY VEC_DISTANCE_L2(v, '[0,0]') FILTER (WHERE id > 0) LIMIT 1",
            "SELECT id, VEC_DISTANCE_L2(v, '[0,0]' ORDER BY id) AS d FROM h ORDER BY d LIMIT 1",
            "INSERT INTO u VALUES (50, 1, ABS(DISTINCT 3))",
            "CREATE TABLE c1 (id INTEGER PRIMARY KEY, a INTEGER CHECK (ABS(DISTINCT a) > 0))",
        ] {
            match error(&db, sql) {
                Some(e) if e.contains(REFUSED) => {}
                other => wrong.push(format!("sealed {sealed}: {sql}: {other:?}")),
            }
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    }
}

#[test]
fn a_parameter_a_prepared_statement_and_a_default_refuse_it_too() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("aggregate_modifier_parameter", &dir, false);
    let refused = |result: Result<usize, String>, what: &str| {
        let e = result.expect_err(what);
        assert!(e.contains(REFUSED), "{what}: {e}");
    };
    refused(
        db.query("SELECT ABS(DISTINCT $1)", (3,))
            .map(|rows| rows.count())
            .map_err(|e| e.to_string()),
        "parameter",
    );
    let prepared = db
        .prepare("SELECT id FROM t WHERE ABS(DISTINCT a) > $1")
        .map_err(|e| e.to_string());
    refused(
        prepared.and_then(|stmt| {
            stmt.query((5,))
                .map_err(|e| e.to_string())?
                .collect::<Result<Vec<_>, _>>()
                .map(|rows| rows.len())
                .map_err(|e| e.to_string())
        }),
        "prepared statement",
    );
    // A DEFAULT is computed when an INSERT reads it, and refused there
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, a INTEGER DEFAULT ABS(DISTINCT -4))",
        (),
    )
    .unwrap();
    refused(
        db.execute("INSERT INTO d (id) VALUES (1)", ())
            .map(|n| n as usize)
            .map_err(|e| e.to_string()),
        "default",
    );
}

#[test]
fn an_aggregate_call_keeps_its_modifiers() {
    let _guard = serial();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("aggregate_modifier_kept", &dir, sealed);
        let row = db
            .query(
                "SELECT COUNT(DISTINCT k), SUM(a) FILTER (WHERE a > 0), \
                 STRING_AGG(s, ',' ORDER BY id DESC) FROM t",
                (),
            )
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(row.get::<i64>(0).unwrap(), 2, "sealed {sealed}");
        assert_eq!(row.get::<i64>(1).unwrap(), 30, "sealed {sealed}");
        assert_eq!(row.get::<String>(2).unwrap(), "x,y,x", "sealed {sealed}");
        let counts: Vec<i64> = db
            .query(
                "SELECT COUNT(DISTINCT a) OVER (PARTITION BY k) FROM t ORDER BY id",
                (),
            )
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(counts, vec![1, 1, 1], "sealed {sealed}");
    }
}

/// Each statement and the error it gives, each answered by its own path
const ERRORS: &[(&str, &str)] = &[
    (
        "SELECT k, (SELECT MAX(v) FROM u), COUNT(*) FROM t GROUP BY 4",
        "GROUP BY position 4 is not in the select list",
    ),
    (
        "SELECT id FROM t WHERE a > (SELECT 1) LIMIT a",
        "Column 'a' not found",
    ),
    (
        "SELECT id FROM t WHERE a > (SELECT 1) LIMIT 1 OFFSET k",
        "Column 'k' not found",
    ),
    (
        "SELECT id, (SELECT MAX(x) FROM (SELECT v AS x FROM u ORDER BY v LIMIT t.k) d) FROM t",
        "Column 't.k' not found",
    ),
    (
        "SELECT id FROM t WHERE a IN (SELECT v FROM u LIMIT t.k)",
        "Column 't.k' not found",
    ),
    (
        "SELECT k FROM t WHERE a > (SELECT 1) ORDER BY MAX(a)",
        "aggregate function MAX is not allowed where the rows are not grouped",
    ),
    (
        "SELECT id FROM t WHERE a > (SELECT 1) ORDER BY COUNT(*)",
        "aggregate function COUNT is not allowed where the rows are not grouped",
    ),
    (
        "SELECT k FROM t ORDER BY MAX(a)",
        "aggregate function MAX is not allowed where the rows are not grouped",
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE COUNT(*) > 1)",
        "aggregate functions are not allowed in WHERE clause",
    ),
    (
        "SELECT id, (SELECT v FROM u WHERE MAX(v) > 1) FROM t",
        "aggregate functions are not allowed in WHERE clause",
    ),
];

#[test]
fn a_statement_the_families_used_to_refuse_gives_its_error() {
    let _guard = serial();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("aggregate_modifier_errors", &dir, sealed);
        for (sql, want) in ERRORS {
            let got = error(&db, sql).unwrap_or_default();
            assert!(got.contains(want), "sealed {sealed}: {sql}: {got}");
        }
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_statement_the_families_used_to_refuse_runs_no_general_path_mechanism() {
    use stoolap::test_failpoints::general_path_total;
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("aggregate_modifier_general_path", &dir, false);
    let refused = [
        "SELECT id FROM t WHERE COALESCE(DISTINCT a, 0) > (SELECT 1)",
        "SELECT id, (SELECT ABS(DISTINCT v) FROM u WHERE u.id = t.id) FROM t",
        "SELECT id FROM u WHERE EXISTS (SELECT ABS(DISTINCT a) FROM t WHERE t.k = u.k)",
    ];
    // An aggregate where no group is read is still the general path's to refuse
    let prepared = ERRORS
        .iter()
        .map(|(sql, _)| *sql)
        .filter(|sql| !sql.contains("ORDER BY MAX") && !sql.contains("ORDER BY COUNT"))
        .filter(|sql| !sql.contains("WHERE COUNT") && !sql.contains("WHERE MAX"));
    for sql in prepared.chain(refused) {
        let before = general_path_total();
        assert!(error(&db, sql).is_some(), "{sql}");
        assert_eq!(general_path_total(), before, "{sql}");
    }
}
