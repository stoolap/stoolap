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

//! Shape x path x parameter style x storage: a statement written with `$n`
//! and with `:name` answers what it answers with its values written as
//! literals, on every path, over hot and sealed rows; with a value missing
//! it is an error on every path and writes nothing

use stoolap::{Database, NamedParams, Value};

/// A statement with `{a}`, `{b}`, ... where its values go, and the values
const SHAPES: &[(&str, &[i64])] = &[
    ("SELECT id, v, s FROM t WHERE id = {a}", &[2]),
    ("SELECT id FROM t WHERE v > {a} AND v < {b} ORDER BY id", &[15, 45]),
    ("SELECT id FROM t WHERE v BETWEEN {a} AND {b} ORDER BY id", &[20, 40]),
    ("SELECT id FROM t WHERE id IN ({a}, {b}) ORDER BY id", &[1, 3]),
    ("SELECT {a} + v, {b} * id FROM t WHERE id = {a}", &[1, 7]),
    (
        "SELECT CASE WHEN v > {a} THEN {b} ELSE id END FROM t ORDER BY id",
        &[25, 99],
    ),
    (
        "SELECT id FROM t WHERE id IN (SELECT k FROM u WHERE w >= {a}) ORDER BY id",
        &[200],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id AND u.w > {a}) ORDER BY id",
        &[150],
    ),
    (
        "SELECT id, (SELECT MAX(w) FROM u WHERE u.k <= {a}) FROM t ORDER BY id",
        &[2],
    ),
    (
        "WITH c AS (SELECT id, v FROM t WHERE v >= {a}) SELECT id FROM c WHERE v <= {b} ORDER BY id",
        &[20, 40],
    ),
    (
        "SELECT t.id, u.w FROM t JOIN u ON u.k = t.id AND u.w > {a} ORDER BY t.id",
        &[150],
    ),
    (
        "SELECT g, SUM(v) FROM t WHERE v > {a} GROUP BY g HAVING SUM(v) > {b} ORDER BY g",
        &[5, 25],
    ),
    ("SELECT id FROM t ORDER BY id LIMIT {a} OFFSET {b}", &[2, 1]),
    (
        "SELECT id, SUM(v) OVER (ORDER BY id) + {a} FROM t ORDER BY id",
        &[1000],
    ),
    ("SELECT COUNT(*) FROM t WHERE g = {a}", &[1]),
    ("UPDATE t SET v = {a} WHERE id = {b}", &[77, 2]),
    ("UPDATE t SET v = v + {a} WHERE id = {b}", &[5, 3]),
    ("UPDATE t SET v = {a} WHERE g = {b}", &[66, 1]),
    ("UPDATE t SET v = {a} WHERE id = {b} RETURNING id, v", &[55, 1]),
    ("DELETE FROM t WHERE id = {a}", &[1]),
    ("DELETE FROM t WHERE v > {a}", &[25]),
    ("INSERT INTO t VALUES ({a}, {b}, 'n', {a})", &[9, 90]),
    ("INSERT INTO t VALUES ({a}, {b}, 'n', 1) RETURNING id, v", &[8, 80]),
    (
        "INSERT INTO t VALUES ({a}, 0, 'x', 0) ON CONFLICT (id) DO UPDATE SET v = {b}",
        &[2, 222],
    ),
    (
        "INSERT INTO t SELECT k + {a}, w, 'u', 0 FROM u WHERE w > {b}",
        &[100, 150],
    ),
    (
        "INSERT INTO t VALUES ({a}, (SELECT MAX(w) FROM u WHERE k <= {b}), 'q', 0)",
        &[7, 2],
    ),
];

const NAMES: [&str; 4] = ["a", "b", "c", "d"];

fn fill(template: &str, each: impl Fn(usize) -> String) -> String {
    let mut sql = template.to_string();
    for (i, name) in NAMES.iter().enumerate() {
        sql = sql.replace(&format!("{{{name}}}"), &each(i));
    }
    sql
}

fn literal(template: &str, values: &[i64]) -> String {
    fill(template, |i| {
        values.get(i).map_or(String::new(), i64::to_string)
    })
}

fn positional(template: &str) -> String {
    fill(template, |i| format!("${}", i + 1))
}

fn named(template: &str) -> String {
    fill(template, |i| format!(":{}", NAMES[i]))
}

/// The answer of a statement and the table it leaves
type Outcome = Result<(Vec<Vec<Value>>, Vec<Vec<Value>>), String>;

fn rows(result: stoolap::Result<stoolap::Rows>) -> stoolap::Result<Vec<Vec<Value>>> {
    let mut out = Vec::new();
    for row in result? {
        let row = row?;
        out.push(
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().clone())
                .collect(),
        );
    }
    Ok(out)
}

fn table(db: &Database) -> Vec<Vec<Value>> {
    rows(db.query("SELECT id, v, s, g FROM t ORDER BY id", ())).unwrap()
}

/// `t` and `u` in their first state, sealed when `sealed`
fn reset(db: &Database, sealed: bool) {
    db.execute("DELETE FROM t", ()).unwrap();
    db.execute("DELETE FROM u", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10, 'a', 1), (2, 20, 'b', 1), (3, 30, 'c', 2), (4, 40, 'd', 2)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, 100), (2, 200), (3, 300)", ())
        .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
}

fn open(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (k INTEGER PRIMARY KEY, w INTEGER)", ())
        .unwrap();
    db
}

#[derive(Clone, Copy, Debug)]
enum Path {
    AdHocParsed,
    AdHocCached,
    Prepared,
    PreparedInTx,
    AdHocInTx,
    Plan,
}

const PATHS: [Path; 6] = [
    Path::AdHocParsed,
    Path::AdHocCached,
    Path::Prepared,
    Path::PreparedInTx,
    Path::AdHocInTx,
    Path::Plan,
];

enum Bind {
    Positional(Vec<Value>),
    Named(Vec<(&'static str, Value)>),
}

impl Bind {
    fn named(&self) -> NamedParams {
        let mut params = NamedParams::new();
        if let Bind::Named(values) = self {
            for (name, value) in values {
                params.insert(*name, value.clone());
            }
        }
        params
    }

    fn positional(&self) -> Vec<Value> {
        match self {
            Bind::Positional(values) => values.clone(),
            Bind::Named(_) => Vec::new(),
        }
    }
}

/// Runs `sql` on `path` from the first state, in a session with an empty
/// plan cache
fn run(db: &Database, sealed: bool, path: Path, sql: &str, bind: &Bind) -> Outcome {
    reset(db, sealed);
    let session = db.clone();
    let named = matches!(bind, Bind::Named(_));
    let query = |session: &Database| {
        if named {
            rows(session.query_named(sql, bind.named()))
        } else {
            rows(session.query(sql, bind.positional()))
        }
    };
    let answer = match path {
        Path::AdHocParsed => query(&session),
        Path::AdHocCached => {
            // The first call caches the plan and is undone
            let _ = query(&session);
            reset(db, sealed);
            query(&session)
        }
        Path::Prepared => session.prepare(sql).and_then(|stmt| {
            if named {
                rows(stmt.query_named(bind.named()))
            } else {
                rows(stmt.query(bind.positional()))
            }
        }),
        Path::PreparedInTx | Path::AdHocInTx => {
            let mut tx = session.begin().unwrap();
            let answer = match path {
                Path::PreparedInTx => session.prepare(sql).and_then(|stmt| {
                    if named {
                        rows(stmt.query_named_in(&mut tx, bind.named()))
                    } else {
                        rows(stmt.query_in(&mut tx, bind.positional()))
                    }
                }),
                _ => {
                    if named {
                        rows(tx.query_named(sql, bind.named()))
                    } else {
                        rows(tx.query(sql, bind.positional()))
                    }
                }
            };
            match answer {
                Ok(_) => tx.commit().unwrap(),
                Err(_) => tx.rollback().unwrap(),
            }
            answer
        }
        Path::Plan => session.cached_plan(sql).and_then(|plan| {
            if named {
                rows(session.query_named_plan(&plan, bind.named()))
            } else {
                rows(session.query_plan(&plan, bind.positional()))
            }
        }),
    };
    answer
        .map(|answer| (answer, table(db)))
        .map_err(|e| e.to_string())
}

fn binds(values: &[i64], given: usize) -> [Bind; 2] {
    let values = &values[..given];
    [
        Bind::Positional(values.iter().map(|&v| Value::Integer(v)).collect()),
        Bind::Named(
            values
                .iter()
                .enumerate()
                .map(|(i, &v)| (NAMES[i], Value::Integer(v)))
                .collect(),
        ),
    ]
}

fn sql_of(template: &str, bind: &Bind) -> String {
    match bind {
        Bind::Positional(_) => positional(template),
        Bind::Named(_) => named(template),
    }
}

fn matrix(db: &Database, sealed: bool) {
    let mut failures = Vec::new();
    for &(template, values) in SHAPES {
        let oracle = run(
            db,
            sealed,
            Path::AdHocParsed,
            &literal(template, values),
            &Bind::Positional(Vec::new()),
        );
        assert!(oracle.is_ok(), "oracle {template}: {oracle:?}");
        for path in PATHS {
            for bind in binds(values, values.len()) {
                let sql = sql_of(template, &bind);
                let got = run(db, sealed, path, &sql, &bind);
                if got != oracle {
                    failures.push(format!(
                        "{path:?} {sql}\n   got  {got:?}\n   want {oracle:?}"
                    ));
                }
            }
            // One value fewer than the statement reads
            for bind in binds(values, values.len() - 1) {
                let sql = sql_of(template, &bind);
                reset(db, sealed);
                let before = table(db);
                match run(db, sealed, path, &sql, &bind) {
                    Err(_) if table(db) == before => {}
                    Err(e) => failures.push(format!("{path:?} {sql}: wrote before {e}")),
                    Ok(got) => failures.push(format!("{path:?} {sql}: missing value gave {got:?}")),
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cells differ:\n{}",
        failures.len(),
        SHAPES.len() * PATHS.len() * 4,
        failures.join("\n")
    );
}

#[test]
fn every_shape_answers_alike_on_every_path_over_hot_rows() {
    matrix(&open("memory://param_slots_matrix_hot"), false);
}

#[test]
fn every_shape_answers_alike_on_every_path_over_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().join("db").display());
    matrix(&open(&dsn), true);
}
