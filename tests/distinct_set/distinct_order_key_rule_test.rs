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

//! Under SELECT DISTINCT every ORDER BY key is one the SELECT items determine:
//! by position, by alias, as the same expression or as a value of the columns
//! they read. Any other key is refused, naming it, over every source and pipeline

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
            ("t.a", "t.a, t.b DESC", "t.b"),
            ("t.a AS k", "t.b", "t.b"),
            ("t.id", "t.b", "t.b"),
            (
                "t.a",
                "(SELECT MAX(x) FROM u WHERE u.k = t.a)",
                "(SELECT MAX(x) FROM u WHERE (u.k = t.a))",
            ),
            ("t.a", "t.b LIMIT 2", "t.b"),
            // A call that may answer anew for each merged row
            ("t.a", "RANDOM()", "RANDOM()"),
            ("t.a", "t.a + RANDOM()", "(t.a + RANDOM())"),
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
            ("a", "a DESC"),
            // The same column under a qualifier of this query's FROM, or none
            ("t.a", "a"),
            ("a", "t.a DESC"),
            // A value the selected columns compute
            ("t.a", "t.a + 1"),
            ("a", "-a, t.a * 2"),
            ("t.a", "CASE WHEN a > 1 THEN 0 ELSE 1 END"),
            ("t.b", "t.b LIKE 'x%'"),
            ("t.a", "t.a IN (1, 2)"),
            ("t.a", "CASE WHEN t.a NOT IN (1, 2) THEN 0 ELSE 1 END"),
            ("t.a", "ABS(t.a - 2)"),
            ("t.a AS k", "k"),
            ("t.a, t.b", "2"),
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
        // A path that never reads the rows, or reads none, judges it all the same
        (
            "SELECT DISTINCT a FROM t WHERE a > ANY (SELECT a FROM e) ORDER BY b",
            "b",
        ),
        (
            "SELECT t.id FROM t WHERE EXISTS \
             (SELECT DISTINCT u.k FROM u WHERE u.id = t.id ORDER BY u.x)",
            "u.x",
        ),
        // A column a sort appends is no SELECT item, a star beside it or not
        (
            "SELECT DISTINCT t.*, COUNT(*) OVER () AS n FROM t JOIN u ON u.k = t.a ORDER BY u.x",
            "u.x",
        ),
        (
            "SELECT DISTINCT t.* FROM t JOIN u ON u.k = t.a ORDER BY x",
            "x",
        ),
        // The inner query's column, though the outer one has a column of its name
        (
            "SELECT (SELECT DISTINCT p.a FROM e AS c ORDER BY a LIMIT 1) FROM t AS p",
            "a",
        ),
    ] {
        if let Some(got) = refusal(rows(db, sql), key) {
            wrong.push(format!(
                "{phase}: {sql}\n  got {got}, want the refusal of {key}"
            ));
        }
    }
    for sql in [
        // The item as written, before a SELECT subquery is folded
        "SELECT DISTINCT (SELECT 1) AS k, COUNT(*) ORDER BY (SELECT 1)",
        // An item and its key written alike
        "SELECT DISTINCT a + 1 FROM t ORDER BY a + 1",
        "SELECT DISTINCT a + 1 FROM t ORDER BY t.a + 1",
        "SELECT DISTINCT COUNT(t.a) AS n FROM t GROUP BY a ORDER BY COUNT(a)",
        "SELECT DISTINCT t.* FROM t ORDER BY b",
        // An enclosing query's column is constant in the subquery
        "SELECT (SELECT DISTINCT c.a FROM e AS c ORDER BY p.b LIMIT 1) FROM t AS p",
        "SELECT DISTINCT SUM(a) OVER () AS total FROM t ORDER BY SUM(a) OVER ()",
        "SELECT DISTINCT t.* FROM t JOIN u ON u.k = t.a ORDER BY t.b",
        "SELECT DISTINCT \"Ä\".* FROM t AS \"Ä\" ORDER BY \"Ä\".a",
        "SELECT DISTINCT a, COUNT(*) AS n FROM t GROUP BY a ORDER BY n, a",
        "SELECT DISTINCT a FROM t WHERE id > (SELECT MIN(id) FROM t) - 1 ORDER BY a",
        "SELECT DISTINCT a FROM t UNION SELECT 7 ORDER BY 1",
    ] {
        if let Err(e) = rows(db, sql) {
            wrong.push(format!("{phase}: {sql}\n  refused: {e}"));
        }
    }
    let sql = "SELECT DISTINCT SUM(a) OVER () AS total FROM t ORDER BY SUM(t.a) OVER ()";
    if !rows(db, sql).is_err_and(|e| e.contains("is not in the SELECT DISTINCT list")) {
        wrong.push(format!("{phase}: {sql}\n  want the refusal"));
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
        "Sort by a selected item",
        "MAX(b)",
    ] {
        assert!(message.contains(part), "{message}");
    }
    let message = db
        .query(
            "SELECT DISTINCT a FROM t GROUP BY a, b ORDER BY MAX(id)",
            (),
        )
        .err()
        .expect("refused")
        .to_string();
    assert!(
        message.contains("once every GROUP BY item is selected") && !message.contains("MAX(MAX"),
        "{message}"
    );
}

#[test]
fn every_group_selected_is_one_distinct_row() {
    let db = Database::open("memory://distinct_order_key_groups").unwrap();
    setup(&db);
    // One INTEGER 1 and one FLOAT 1.0: two TYPEOF groups, one distinct value
    let values = "(VALUES (1, 1, 'x'), (3, 1.0, 'y'), (2, 2, 'x')) AS t(id, a, b)";
    let mut wrong = Vec::new();
    for from in ["t", values] {
        for (items, group_by) in [
            ("a", "a"),
            ("t.a", "a"),
            ("a", "t.a"),
            ("a AS k", "a"),
            ("a AS k", "1"),
            ("a + 1", "a + 1"),
            ("a + 1 AS k", "1"),
            ("a, b", "a, b"),
            ("a, b", "2, 1"),
            ("b, COUNT(*) AS n", "b"),
            ("UPPER(b)", "UPPER(b)"),
        ] {
            for order_by in ["MAX(id)", "MAX(id) DESC", "COUNT(*), 1", "MIN(id) DESC, 1"] {
                // Merging no rows, DISTINCT answers as the query without it
                let sql = format!(
                    "SELECT DISTINCT {items} FROM {from} GROUP BY {group_by} ORDER BY {order_by}"
                );
                let plain = sql.replacen("DISTINCT ", "", 1);
                let (got, want) = (rows(&db, &sql), rows(&db, &plain));
                if got.is_err() || got != want {
                    wrong.push(format!("{sql}\n  got {got:?}, want {want:?}"));
                }
            }
        }
        for (items, group_by) in [
            // A name grouping binds to the column, not the aggregate's alias
            ("COUNT(*) AS a", "a"),
            ("a AS k", "k"),
            // A key DISTINCT does not see, or one finer than an item
            ("a", "a, b"),
            ("a + 1", "a"),
            ("a", "a, TYPEOF(a)"),
            ("a", "2"),
            ("*", "id, a, b"),
            // Grouping sets put one value in rows of more than one group
            ("a", "ROLLUP(a)"),
            ("a", "CUBE(a)"),
            ("a", "GROUPING SETS ((a), ())"),
        ] {
            let sql =
                format!("SELECT DISTINCT {items} FROM {from} GROUP BY {group_by} ORDER BY MAX(id)");
            let got = rows(&db, &sql);
            if !got
                .as_ref()
                .is_err_and(|e| e.contains("is not in the SELECT DISTINCT list"))
            {
                wrong.push(format!("{sql}\n  got {got:?}, want the refusal"));
            }
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
fn a_view_is_refused_when_it_is_created() {
    let db = Database::open("memory://distinct_order_key_view").unwrap();
    setup(&db);
    let created = db.execute("CREATE VIEW w AS SELECT DISTINCT a FROM t ORDER BY b", ());
    assert!(
        created.as_ref().is_err_and(|e| e
            .to_string()
            .contains("ORDER BY b is not in the SELECT DISTINCT list")),
        "{created:?}"
    );
}
