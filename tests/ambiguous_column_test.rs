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

//! In a statement the prepared families run, a bare name that more than
//! one source of a query has, and that no NATURAL or USING join merges, is
//! an error: `column reference 'x' is ambiguous`

use stoolap::Database;

const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 3, 30)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, a INTEGER, x INTEGER)",
    "INSERT INTO u VALUES (1, 2, 7), (2, 3, 1), (4, 1, 9)",
    "CREATE TABLE w (k INTEGER PRIMARY KEY, y INTEGER)",
    "INSERT INTO w VALUES (1, 1), (2, 2)",
    "CREATE VIEW uv AS SELECT id, a, x FROM u",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://ambiguous_column_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// The second source as each kind of FROM entry, named s, holding `id`
const SIDES: &[(&str, &str, &str)] = &[
    ("table", "", "u AS s"),
    ("view", "", "uv AS s"),
    ("cte", "WITH c AS (SELECT id, a, x FROM u) ", "c AS s"),
    ("derived", "", "(SELECT id, a, x FROM u) AS s"),
    ("table function", "", "generate_series(1, 3) AS s (id)"),
];

const JOINS: &[&str] = &["JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN"];

/// Each statement naming `id` bare where `{id}` stands, `{f}` its FROM and
/// `{on}` the rest of its ON
const SHAPES: &[&str] = &[
    "SELECT t.b FROM {f} WHERE {id} > (SELECT MIN(y) FROM w)",
    "SELECT t.b FROM {f} WHERE {id} > (SELECT MIN(y) FROM w) - $1",
    "SELECT t.b FROM {f} WHERE {id} IN (SELECT y FROM w)",
    "SELECT t.b FROM {f} WHERE CASE WHEN {id} > 1 THEN (SELECT 1) ELSE 0 END = 1",
    "SELECT t.b FROM {f} WHERE EXISTS (SELECT 1 FROM w WHERE w.k = {id})",
    "SELECT t.b, (SELECT COUNT(*) FROM w WHERE w.k = {id}) FROM {f}",
    "SELECT {id}, (SELECT MAX(y) FROM w) FROM {f}",
    "SELECT {id} + 1, (SELECT MAX(y) FROM w) FROM {f} ORDER BY t.b",
    "SELECT t.b, (SELECT MAX(y) FROM w) FROM {f} ORDER BY {id}",
    "SELECT t.b FROM {f} ORDER BY {id} + (SELECT 0)",
    "SELECT t.b FROM {f}{on} AND {id} > (SELECT 0)",
];

fn answer(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, (1,)).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            row.get::<Option<i64>>(0)
                .map_err(|e| e.to_string())?
                .unwrap_or(-1),
        );
    }
    out.sort_unstable();
    Ok(out)
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), (1,))
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

fn check(db: &Database, context: &str, wrong: &mut Vec<String>) {
    for (kind, with, side) in SIDES {
        for join in JOINS {
            for (from, on) in [
                (format!("t {join} {side} ON s.id = t.a"), String::new()),
                (
                    format!("t {join} {side} ON s.id = t.a JOIN w ON w.k = t.id"),
                    String::new(),
                ),
                (format!("t {join} {side}"), " ON s.id = t.a".to_string()),
            ] {
                for shape in SHAPES {
                    if shape.contains("{on}") != !on.is_empty() {
                        continue;
                    }
                    let shape = shape.replace("{f}", &from).replace("{on}", &on);
                    let sql = format!("{with}{}", shape.replace("{id}", "id"));
                    match answer(db, &sql) {
                        Err(e) if e.contains("column reference 'id' is ambiguous") => {}
                        got => wrong.push(format!("{context} {kind}: {sql}: {got:?}")),
                    }
                    let qualified = format!("{with}{}", shape.replace("{id}", "t.id"));
                    if let Err(e) = answer(db, &qualified) {
                        wrong.push(format!("{context} {kind}: {qualified}: {e}"));
                    }
                    let line = prepared_line(db, &qualified);
                    if line != "Prepared: yes" {
                        wrong.push(format!("{context} {kind}: {qualified}: {line}"));
                    }
                }
            }
        }
    }
    // A join that merges a name merges one column of each side, never a
    // source's repeated one or a third source's
    for (sql, name) in [
        (
            "SELECT t.b FROM t WHERE EXISTS \
             (SELECT 1 FROM u JOIN u AS v ON v.id = u.a WHERE id = t.a)",
            "id",
        ),
        (
            "SELECT t.b FROM t WHERE EXISTS (SELECT x FROM (SELECT 1 AS x, 2 AS x) AS q \
             JOIN (SELECT 1 AS x) AS r USING (x))",
            "x",
        ),
        (
            "SELECT t.b FROM t WHERE EXISTS (SELECT x FROM (SELECT 1 AS x) AS q \
             JOIN (SELECT 1 AS x, 2 AS x) AS r USING (x))",
            "x",
        ),
        (
            "SELECT t.b FROM t WHERE EXISTS (SELECT 1 FROM u JOIN uv USING (id) \
             JOIN u AS q ON q.id = u.a WHERE id = t.a)",
            "id",
        ),
        (
            "SELECT t.b FROM t JOIN (SELECT 1 AS id, 2 AS id) AS q USING (id) \
             WHERE id > (SELECT 0)",
            "q.id",
        ),
    ] {
        match answer(db, sql) {
            Err(e) if e.contains(&format!("column reference '{name}' is ambiguous")) => {}
            got => wrong.push(format!("{context}: {sql}: {got:?}")),
        }
    }
    // Each statement whose bare name one column answers
    for (sql, want) in [
        (
            "SELECT t.b FROM t JOIN u USING (id) WHERE id > (SELECT MIN(y) FROM w)",
            vec![20],
        ),
        (
            "SELECT t.b FROM t JOIN u USING (id, a) WHERE a < (SELECT MAX(y) + 9 FROM w)",
            vec![],
        ),
        (
            "SELECT t.b FROM t NATURAL JOIN u WHERE id > (SELECT MIN(y) FROM w)",
            vec![],
        ),
        (
            "SELECT t.b FROM t WHERE EXISTS (SELECT 1 FROM u JOIN uv USING (id) WHERE id = t.a)",
            vec![10, 20],
        ),
        (
            "SELECT t.b FROM t WHERE EXISTS (SELECT 1 FROM u JOIN uv USING (id) \
             JOIN u AS q USING (id) WHERE id = t.a)",
            vec![10, 20],
        ),
        (
            "SELECT t.b FROM t JOIN u ON u.id = t.a WHERE EXISTS (SELECT 1 FROM w WHERE w.k = x)",
            vec![20],
        ),
        (
            "SELECT t.b FROM t LEFT JOIN u ON u.id = t.a WHERE EXISTS \
             (SELECT 1 FROM w WHERE w.k = x)",
            vec![20],
        ),
        (
            "SELECT t.b FROM t JOIN u ON u.id = t.a WHERE EXISTS \
             (SELECT 1 FROM w WHERE w.k = b - 9)",
            vec![10],
        ),
        (
            "SELECT t.b FROM t JOIN u ON u.id = t.a WHERE EXISTS \
             (SELECT 1 FROM u AS q WHERE q.id = id)",
            vec![10, 20],
        ),
    ] {
        let got = answer(db, sql);
        if got != Ok(want.clone()) {
            wrong.push(format!("{context}: {sql}: {got:?}, want {want:?}"));
        }
    }
}

#[test]
fn a_bare_name_two_sources_have_is_ambiguous_in_every_clause() {
    let mut wrong = Vec::new();
    check(&hot("matrix"), "hot", &mut wrong);
    let dir = tempfile::tempdir().unwrap();
    check(&sealed(&dir), "sealed", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_conflict_update_reads_a_bare_name_from_the_row_met() {
    let db = hot("conflict");
    db.execute(
        "INSERT INTO t VALUES (1, 5, 0) ON CONFLICT (id) DO UPDATE SET b = b + excluded.a",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 5, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT b + 1)",
        (),
    )
    .unwrap();
    let b: i64 = db.query_one("SELECT b FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(b, 16, "b = 10 + 5, then + 1");
}
