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

/// The same rule in statements holding no subquery
const PLAIN_SHAPES: &[&str] = &[
    "SELECT {id} FROM {f}",
    "SELECT {id} + 1 FROM {f}",
    "SELECT t.b FROM {f} WHERE {id} > 1",
    "SELECT t.b FROM {f} WHERE {id} > $1",
    "SELECT t.b FROM {f} WHERE CASE WHEN {id} > 1 THEN 1 ELSE 0 END = 1",
    "SELECT COUNT(*) FROM {f} GROUP BY {id}",
    "SELECT SUM({id}) FROM {f}",
    "SELECT COUNT(*) FROM {f} GROUP BY t.b HAVING MAX({id}) > 0",
    "SELECT ROW_NUMBER() OVER (ORDER BY {id}) FROM {f}",
    "SELECT t.b FROM {f} ORDER BY {id}",
    "SELECT DISTINCT ON ({id}) t.b FROM {f} ORDER BY {id}",
    "SELECT t.b FROM {f}{on} AND {id} > 0",
];

fn answer(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    // A statement with `$1` reads 1 there, the others no value
    let params = match sql.contains("$1") {
        true => vec![stoolap::Value::Integer(1)],
        false => Vec::new(),
    };
    let rows = db.query(sql, params).map_err(|e| e.to_string())?;
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
                let shapes = SHAPES.iter().map(|s| (s, true));
                for (shape, holds_subquery) in shapes.chain(PLAIN_SHAPES.iter().map(|s| (s, false)))
                {
                    if shape.contains("{on}") == on.is_empty() {
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
                    if holds_subquery && line != "Prepared: yes" {
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
        (
            "SELECT t.b FROM t JOIN u USING (id) JOIN u AS q ON q.id = t.a WHERE id > 1",
            "id",
        ),
        (
            "SELECT id, t.b FROM t LEFT JOIN u USING (id) JOIN u AS q ON q.id = t.a",
            "id",
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
        ("SELECT t.b FROM t JOIN u USING (id) WHERE id > 1", vec![20]),
        (
            "SELECT t.id AS id FROM t JOIN u ON u.id = t.a ORDER BY id DESC",
            vec![1, 2],
        ),
        (
            "SELECT t.b FROM t JOIN u USING (id) JOIN w ON w.k = t.id WHERE id > 1",
            vec![20],
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

#[test]
fn a_select_alias_spares_a_bare_name_only_where_the_alias_is_read() {
    let db = hot("alias_contexts");
    let mut wrong = Vec::new();
    // Each statement with `{k}` for a name both sources have, which a
    // SELECT alias `id` also gives, and the alias's item when `{k}` reads it
    let f = "FROM t JOIN u ON u.id = t.a";
    let first = "10 - t.id";
    let max = "MAX(t.id)";
    for (shape, reads) in [
        (
            "SELECT 10 - t.id AS id, t.id {f} ORDER BY {k}, t.id",
            Some(first),
        ),
        (
            "SELECT DISTINCT ON ({k}) 10 - t.id AS id, t.id {f} ORDER BY {k}, t.id",
            Some(first),
        ),
        (
            "SELECT 10 - t.id AS id, COUNT(*) {f} GROUP BY {k}",
            Some(first),
        ),
        ("SELECT MAX(t.id) AS id {f} HAVING {k} > 0", Some(max)),
        (
            "SELECT MAX(t.id) AS id {f} HAVING {k} > (SELECT 0)",
            Some(max),
        ),
        (
            "SELECT MAX(t.id) AS id, t.a {f} GROUP BY t.a HAVING {k} + 0 > 1",
            Some(max),
        ),
        (
            "SELECT t.a AS id, COUNT(*) {f} GROUP BY t.a HAVING {k} > 1",
            Some("t.a"),
        ),
        ("SELECT 10 - t.id AS id, t.id {f} ORDER BY {k} + 0", None),
        (
            "SELECT 10 - t.id AS id, t.id {f} ORDER BY ROW_NUMBER() OVER (ORDER BY {k})",
            None,
        ),
        (
            "SELECT 10 - t.id AS id, COUNT(*) {f} GROUP BY t.id ORDER BY MAX({k})",
            None,
        ),
        ("SELECT 0 AS id, COUNT(*) {f} GROUP BY {k} + 1", None),
        (
            "SELECT 10 - t.id AS id, COUNT(*) {f} GROUP BY t.id HAVING {k} > 1",
            None,
        ),
        (
            "SELECT MAX(t.id) AS id, t.a {f} GROUP BY t.a HAVING MIN({k}) > 1",
            None,
        ),
        ("SELECT 10 - t.id AS id {f} WHERE {k} > 1", None),
        (
            "SELECT 10 - t.id AS id, ROW_NUMBER() OVER (ORDER BY {k}, t.id) {f}",
            None,
        ),
        (
            "SELECT 10 - t.id AS id, COUNT(*) OVER (PARTITION BY {k}) {f}",
            None,
        ),
        (
            "SELECT 10 - t.id AS id, ROW_NUMBER() OVER w {f} WINDOW w AS (ORDER BY {k}, t.id)",
            None,
        ),
    ] {
        let shape = shape.replace("{f}", f);
        let bare = shape.replace("{k}", "id");
        let got = answer(&db, &bare);
        let right = match reads {
            Some(item) => {
                got.is_ok() && got == answer(&db, &shape.replace("{k}", &format!("({item})")))
            }
            None => got
                .as_ref()
                .is_err_and(|e| e.contains("column reference 'id' is ambiguous")),
        };
        if !right {
            wrong.push(format!("{bare}: {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_name_two_columns_of_one_source_carry_is_ambiguous_under_a_window() {
    let dir = tempfile::tempdir().unwrap();
    for db in [hot("window_one_source"), sealed(&dir)] {
        let mut wrong = Vec::new();
        for (with, source) in [
            ("", "(SELECT t.a, u.a FROM t JOIN u ON u.id = t.id) AS s"),
            (
                "WITH c AS (SELECT t.a, u.a FROM t JOIN u ON u.id = t.id) ",
                "c AS s",
            ),
        ] {
            for (shape, reference) in [
                ("SELECT {a}, ROW_NUMBER() OVER () FROM {f}", "a"),
                ("SELECT {a} + 1, ROW_NUMBER() OVER () FROM {f}", "a"),
                ("SELECT ROW_NUMBER() OVER (PARTITION BY {a}) FROM {f}", "a"),
                ("SELECT ROW_NUMBER() OVER (ORDER BY {a}) FROM {f}", "a"),
                ("SELECT MAX({a}) OVER () FROM {f}", "a"),
                (
                    "SELECT ROW_NUMBER() OVER w FROM {f} WINDOW w AS (ORDER BY {a})",
                    "a",
                ),
                ("SELECT ROW_NUMBER() OVER () FROM {f} ORDER BY {a}", "a"),
                ("SELECT {a}, ROW_NUMBER() OVER () FROM {f}", "s.a"),
            ] {
                let sql = format!(
                    "{with}{}",
                    shape.replace("{a}", reference).replace("{f}", source)
                );
                let got = answer(&db, &sql);
                let message = format!("column reference '{reference}' is ambiguous");
                if !got.as_ref().is_err_and(|e| e.contains(&message)) {
                    wrong.push(format!("{sql}: {got:?}"));
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}

#[test]
fn a_qualifier_keeps_its_source_beside_repeated_names_under_a_window() {
    let db = hot("window_qualifier_kept");
    let from = "FROM (SELECT 1 AS a, 2 AS a) AS s CROSS JOIN (SELECT 3 AS b) AS r";
    let err = answer(&db, &format!("SELECT s.b, ROW_NUMBER() OVER () {from}")).unwrap_err();
    assert!(err.contains("'s.b' not found"), "{err}");
    assert_eq!(
        answer(&db, &format!("SELECT r.b, ROW_NUMBER() OVER () {from}")),
        Ok(vec![3])
    );
}

#[test]
fn a_name_two_group_slots_carry_is_no_source_ambiguity_under_a_window() {
    let db = hot("window_group_slot_names");
    let rows = |sql: &str| -> Vec<Vec<i64>> {
        db.query(sql, ())
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .map(|r| {
                let r = r.unwrap();
                (0..r.len()).map(|i| r.get::<i64>(i).unwrap()).collect()
            })
            .collect()
    };
    assert_eq!(
        rows("SELECT a, SUM(b) AS a, SUM(SUM(a)) OVER (ORDER BY a) FROM t GROUP BY a ORDER BY 1"),
        vec![vec![1, 10, 1], vec![2, 20, 3], vec![3, 30, 6]]
    );
}
