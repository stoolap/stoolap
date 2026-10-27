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

//! In a join, `v.x` is the column `x` of the source named `v`: a name that
//! source does not have is an error, never another source's column of that
//! name, and every kind of source names its columns by its own name

use stoolap::Database;

const SETUP: &[&str] = &[
    "CREATE TABLE y (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO y VALUES (1, 1, 5), (2, 1, 50), (3, 1, 7), (4, 2, 9), (5, 2, 20)",
    "CREATE VIEW vy AS SELECT a, b FROM y",
    "CREATE TABLE ta (a INTEGER, b INTEGER)",
    "INSERT INTO ta VALUES (1, 5), (2, 9)",
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 4, 30), (4, NULL, 40), (5, 3, 50), (6, 1, 60)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 7), (2, 8), (3, 9), (5, NULL)",
    "CREATE TABLE w (k INTEGER PRIMARY KEY)",
    "INSERT INTO w VALUES (1), (2)",
];

/// An index on y.a, which a join on it reads y through
const INDEXED: &str = "CREATE INDEX y_a ON y (a)";

fn hot(name: &str, index: bool) -> Database {
    let db = Database::open(&format!("memory://joined_column_sources_{name}")).unwrap();
    for sql in SETUP.iter().chain(index.then_some(&INDEXED)) {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir, index: bool) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP.iter().chain(index.then_some(&INDEXED)) {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// A source named v holding a and b, never id, as each kind of FROM entry
const SIDES: &[(&str, &str, &str)] = &[
    ("table", "", "ta AS v"),
    ("view", "", "vy AS v"),
    ("cte", "WITH c AS (SELECT a, b FROM y) ", "c AS v"),
    ("derived", "", "(SELECT a, b FROM y) AS v"),
    ("values", "", "(VALUES (1, 5), (2, 9)) AS v (a, b)"),
    ("table function", "", "generate_series(1, 2) AS v (a)"),
];

/// Each statement naming `v.id`, which v does not have and y does
const MISSING: &[&str] = &[
    "SELECT v.id FROM {v} JOIN y ON y.a = v.a",
    "SELECT y.b FROM {v} JOIN y ON y.id = v.id",
    "SELECT y.b FROM {v} JOIN y ON y.a = v.a WHERE v.id = 1",
    "SELECT y.b FROM {v} JOIN y ON y.a = v.a ORDER BY v.id",
    "SELECT COUNT(*) FROM {v} JOIN y ON y.a = v.a GROUP BY v.id",
    "SELECT v.id, (SELECT MAX(k) FROM w) FROM {v} JOIN y ON y.a = v.a",
    "SELECT y.b FROM {v} JOIN y ON y.a = v.a WHERE v.id > (SELECT MIN(k) FROM w)",
    "SELECT y.b FROM {v} JOIN y ON y.a = v.a WHERE EXISTS (SELECT 1 FROM w WHERE w.k = v.id)",
    "SELECT y.id FROM {v} JOIN y ON y.a = v.a ORDER BY v.id",
    "SELECT y.b FROM {v} JOIN y ON y.a = v.a ORDER BY v.id LIMIT 2",
    "SELECT v.id FROM {v} JOIN y ON y.a = v.a LIMIT 2",
    "SELECT DISTINCT v.id FROM {v} JOIN y ON y.a = v.a",
    "SELECT DISTINCT ON (v.id) y.id FROM {v} JOIN y ON y.a = v.a",
    "SELECT SUM(v.id) FROM {v} JOIN y ON y.a = v.a",
    "SELECT y.a FROM {v} JOIN y ON y.a = v.a GROUP BY y.a HAVING MAX(v.id) > 0",
    "SELECT RANK() OVER (ORDER BY v.id) FROM {v} JOIN y ON y.a = v.a",
    "SELECT COUNT(*) OVER (PARTITION BY v.id) FROM {v} JOIN y ON y.a = v.a",
];

fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| match row.get::<Option<String>>(i) {
                    Ok(value) => value.unwrap_or_else(|| "NULL".into()),
                    Err(e) => format!("<{e}>"),
                })
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    out.sort();
    Ok(out)
}

fn check(db: &Database, context: &str, wrong: &mut Vec<String>) {
    for (kind, with, side) in SIDES {
        for shape in MISSING {
            let sql = format!("{with}{}", shape.replace("{v}", side));
            // A condition pushed to v's scan names the column without v
            let missing = |e: &str| {
                let e = e.to_lowercase();
                e.contains("column 'v.id' not found") || e.contains("column 'id' not found")
            };
            match answer(db, &sql) {
                Err(e) if missing(&e) => {}
                got => wrong.push(format!("{context} {kind}: {sql}: {got:?}")),
            }
        }
    }
    // A name a source has answers by its qualifier: a table's name or alias,
    // and a column a USING join merges through either table
    for (sql, same_as) in [
        ("SELECT y.a FROM y", "SELECT a FROM y"),
        (
            "SELECT COUNT(\"missing\") FROM (VALUES (1), (2)) AS v (x)",
            "SELECT 2",
        ),
        (
            "SELECT DISTINCT ON (v.a) v.a FROM ta AS v JOIN y ON y.a = v.a",
            "SELECT DISTINCT a FROM ta",
        ),
        (
            "SELECT q.a FROM y AS q ORDER BY q.b",
            "SELECT a FROM y ORDER BY b",
        ),
        (
            "SELECT t.id, u.x FROM t JOIN u USING (id)",
            "SELECT t.id, u.x FROM t JOIN u ON u.id = t.id",
        ),
        (
            "SELECT u.id, t.b FROM t JOIN u USING (id) ORDER BY u.id",
            "SELECT u.id, t.b FROM t JOIN u ON u.id = t.id ORDER BY u.id",
        ),
        (
            "SELECT t.id, q.x FROM t JOIN u USING (id) JOIN u AS q ON q.id = t.a",
            "SELECT t.id, q.x FROM t JOIN u ON u.id = t.id JOIN u AS q ON q.id = t.a",
        ),
        (
            "SELECT COUNT(*), t.id FROM t JOIN u USING (id) GROUP BY t.id",
            "SELECT COUNT(*), t.id FROM t JOIN u ON u.id = t.id GROUP BY t.id",
        ),
        (
            "SELECT t.id, u.id, id, v.a FROM t JOIN u USING (id) CROSS JOIN ta AS v",
            "SELECT t.id, u.id, t.id, v.a FROM t JOIN u ON u.id = t.id CROSS JOIN ta AS v",
        ),
        (
            "SELECT COUNT(*) FROM t JOIN u USING (id) CROSS JOIN ta AS v GROUP BY u.id",
            "SELECT COUNT(*) FROM t JOIN u ON u.id = t.id CROSS JOIN ta AS v GROUP BY u.id",
        ),
        (
            "SELECT RANK() OVER (ORDER BY u.id), COUNT(*) OVER (PARTITION BY t.id) FROM t JOIN u USING (id)",
            "SELECT RANK() OVER (ORDER BY u.id), COUNT(*) OVER (PARTITION BY t.id) FROM t JOIN u ON u.id = t.id",
        ),
    ] {
        let (got, want) = (answer(db, sql), answer(db, same_as));
        if got.is_err() || got != want {
            wrong.push(format!("{context}: {sql}: {got:?}, want {want:?}"));
        }
    }
    // A source of each kind joined in the middle answers as a table there
    let table = "SELECT t.id, s.id, u.x FROM t JOIN u AS s ON s.id = t.a JOIN u ON u.id = s.id";
    let want = answer(db, table);
    for (kind, with, side) in [
        ("values", "", "(VALUES (1), (2), (3), (5)) AS s (id)"),
        ("derived", "", "(SELECT id FROM u) AS s"),
        ("cte", "WITH c AS (SELECT id FROM u) ", "c AS s"),
        ("table function", "", "generate_series(1, 5) AS s (id)"),
    ] {
        let sql = format!(
            "{with}SELECT t.id, s.id, u.x FROM t JOIN {side} ON s.id = t.a JOIN u ON u.id = s.id"
        );
        let got = answer(db, &sql);
        let want = match kind {
            "table function" => answer(
                db,
                "SELECT t.id, t.a, u.x FROM t JOIN u ON u.id = t.a WHERE t.a BETWEEN 1 AND 5",
            ),
            _ => want.clone(),
        };
        if got != want {
            wrong.push(format!("{context} {kind}: {sql}: {got:?}, want {want:?}"));
        }
    }
}

#[test]
fn a_qualified_name_reads_its_own_source_and_no_other() {
    let mut wrong = Vec::new();
    check(&hot("matrix", false), "hot", &mut wrong);
    check(&hot("indexed", true), "hot indexed", &mut wrong);
    let dir = tempfile::tempdir().unwrap();
    check(&sealed(&dir, false), "sealed", &mut wrong);
    let dir = tempfile::tempdir().unwrap();
    check(&sealed(&dir, true), "sealed indexed", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
