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

//! Under SELECT DISTINCT every ORDER BY key is a SELECT item: by position, by
//! alias, as the same expression or as the column an item reads. Any other
//! key is refused, naming it, over every source and pipeline

use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "INSERT INTO t VALUES (1, 1, 'x'), (2, 1, 'y'), (3, 2, 'x'), (4, 3, 'z')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 2, 6)",
        "CREATE TABLE e (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "CREATE VIEW v AS SELECT id, a, b FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

const SOURCES: [(&str, &str, &str); 6] = [
    ("table", "", "t"),
    ("empty table", "", "e AS t"),
    ("view", "", "v AS t"),
    ("derived", "", "(SELECT id, a, b FROM t) AS t"),
    ("cte", "WITH c AS (SELECT id, a, b FROM t) ", "c AS t"),
    ("join", "", "t JOIN u ON u.k = t.a"),
];

fn refusal(result: Result<Vec<String>, String>, key: &str) -> Option<String> {
    match result {
        Err(e)
            if e.contains(&format!(
                "ORDER BY {key} is not in the SELECT DISTINCT list"
            )) =>
        {
            None
        }
        other => Some(format!("{other:?}")),
    }
}

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    db.query(sql, ())
        .map_err(|e| e.to_string())?
        .map(|row| {
            row.map(|row| {
                format!(
                    "{:?}",
                    (0..row.len()).map(|i| row.get_value(i)).collect::<Vec<_>>()
                )
            })
            .map_err(|e| e.to_string())
        })
        .collect()
}

fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    for (source, with, from) in SOURCES {
        for (items, order_by, key) in [
            ("t.a", "t.b", "t.b"),
            ("t.a", "t.id", "t.id"),
            ("t.a", "t.a + 1", "(t.a + 1)"),
            ("t.a", "t.a, t.b DESC", "t.b"),
            ("t.a AS k", "t.b", "t.b"),
            ("t.id", "t.b", "t.b"),
            (
                "t.a",
                "(SELECT MAX(x) FROM u WHERE u.k = t.a)",
                "(SELECT MAX(x) FROM u WHERE (u.k = t.a))",
            ),
            ("t.a", "t.b LIMIT 2", "t.b"),
        ] {
            let sql = format!("{with}SELECT DISTINCT {items} FROM {from} ORDER BY {order_by}");
            let key = if key.is_empty() { order_by } else { key };
            if let Some(got) = refusal(rows(db, &sql), key) {
                wrong.push(format!(
                    "{phase} {source}: {sql}\n  got {got}, want the refusal of {key}"
                ));
            }
        }
        for (items, order_by) in [
            ("t.a", "t.a"),
            ("t.a", "a"),
            ("a", "t.a DESC"),
            ("t.a AS k", "k"),
            ("t.a", "1"),
            ("t.a + 1", "t.a + 1"),
            ("t.a, t.b", "t.b, t.a"),
            ("*", "t.b"),
        ] {
            let sql = format!("{with}SELECT DISTINCT {items} FROM {from} ORDER BY {order_by}");
            if let Err(e) = rows(db, &sql) {
                wrong.push(format!("{phase} {source}: {sql}\n  refused: {e}"));
            }
        }
    }
    // The grouped, the prepared and the nested pipelines judge it the same way
    for (sql, key) in [
        ("SELECT DISTINCT a FROM t GROUP BY a, b ORDER BY b", "b"),
        (
            "SELECT DISTINCT a FROM t WHERE id > (SELECT MIN(id) FROM t) - 1 ORDER BY b",
            "b",
        ),
        (
            "SELECT id FROM u WHERE k IN (SELECT DISTINCT a FROM t ORDER BY b LIMIT 1)",
            "b",
        ),
    ] {
        if let Some(got) = refusal(rows(db, sql), key) {
            wrong.push(format!(
                "{phase}: {sql}\n  got {got}, want the refusal of {key}"
            ));
        }
    }
    for sql in [
        "SELECT DISTINCT a, COUNT(*) AS n FROM t GROUP BY a ORDER BY n, a",
        "SELECT DISTINCT a FROM t WHERE id > (SELECT MIN(id) FROM t) - 1 ORDER BY a",
        "SELECT DISTINCT a FROM t UNION SELECT 7 ORDER BY 1",
    ] {
        if let Err(e) = rows(db, sql) {
            wrong.push(format!("{phase}: {sql}\n  refused: {e}"));
        }
    }
    let prepared = db.prepare("SELECT DISTINCT a FROM t WHERE id > $1 ORDER BY b");
    let refused = match prepared {
        Err(e) => e.to_string(),
        Ok(statement) => match statement.query((0,)) {
            Err(e) => e.to_string(),
            Ok(rows) => format!("{} rows", rows.count()),
        },
    };
    if !refused.contains("ORDER BY b is not in the SELECT DISTINCT list") {
        wrong.push(format!(
            "{phase}: a prepared statement with a parameter: {refused}"
        ));
    }
}

#[test]
fn an_order_by_key_outside_the_select_distinct_list_is_refused() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://distinct_order_key_rule").unwrap();
    setup(&db);
    check(&db, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    setup(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET b = 'w' WHERE id = 4", ()).unwrap();
    check(&db, "sealed with hot updates", &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn the_refusal_names_the_key_and_the_ways_to_write_it() {
    let db = Database::open("memory://distinct_order_key_message").unwrap();
    setup(&db);
    let message = db
        .query("SELECT DISTINCT a FROM t ORDER BY b", ())
        .err()
        .expect("refused")
        .to_string();
    for part in [
        "ORDER BY b is not in the SELECT DISTINCT list",
        "Add it to the SELECT list",
        "MAX(b)",
    ] {
        assert!(message.contains(part), "{message}");
    }
}
