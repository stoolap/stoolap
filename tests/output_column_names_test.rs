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

use stoolap::Database;

const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "CREATE INDEX t_b ON t (b)",
    "INSERT INTO t VALUES (1, 1, 5), (2, 2, 6), (3, 3, 7)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 2), (2, 4)",
    "CREATE VIEW v AS SELECT * FROM t",
];

/// SELECT items of every kind and the names they give
const ITEMS: &str = "a, t.b, A, ABS(a), CAST(a AS TEXT), a + 1, -a, 5, 'x', COALESCE(a, 0)";
const NAMES: &[&str] = &[
    "a",
    "b",
    "A",
    "ABS(a)",
    "CAST(a AS TEXT)",
    "a + 1",
    "-a",
    "5",
    "'x'",
    "COALESCE(a, 0)",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://output_column_names_{name}")).unwrap();
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

fn names(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .columns()
        .to_vec()
}

fn with(extra: &[&str]) -> Vec<String> {
    NAMES.iter().chain(extra).map(|n| n.to_string()).collect()
}

fn cases() -> Vec<(String, Vec<String>)> {
    let items = |sql: &str| (sql.replace("{I}", ITEMS), with(&[]));
    let mut cases = vec![
        items("SELECT {I} FROM t"),
        items("SELECT {I} FROM t WHERE a > 0"),
        items("SELECT {I} FROM t WHERE id = 1"),
        items("SELECT {I} FROM t WHERE b = 5"),
        items("SELECT {I} FROM t ORDER BY id LIMIT 2"),
        items("SELECT DISTINCT {I} FROM t"),
        items("SELECT {I} FROM t JOIN u ON u.id = t.id"),
        items("SELECT {I} FROM (SELECT * FROM t) AS t"),
        items("SELECT {I} FROM v AS t"),
        items("WITH c AS (SELECT * FROM t) SELECT {I} FROM c AS t"),
        items("SELECT * FROM (SELECT {I} FROM t) AS d"),
        items("WITH c AS (SELECT {I} FROM t) SELECT * FROM c"),
        items("SELECT {I} FROM t UNION ALL SELECT {I} FROM t"),
        (
            format!("SELECT {ITEMS}, ROW_NUMBER() OVER (ORDER BY id) FROM t"),
            with(&["ROW_NUMBER() OVER (ORDER BY id)"]),
        ),
    ];
    let fixed: &[(&str, &[&str])] = &[
        ("SELECT a, t.b, A FROM t ORDER BY id LIMIT 1", &["a", "b", "A"]),
        (
            "SELECT a, t.a, SUM(b), COUNT(*), COUNT(DISTINCT b), SUM(b) + 1, ABS(SUM(b)), a + 1 FROM t GROUP BY a",
            &["a", "a", "SUM(b)", "COUNT(*)", "COUNT(DISTINCT b)", "SUM(b) + 1", "ABS(SUM(b))", "a + 1"],
        ),
        ("SELECT t.a, COUNT(*) FROM t GROUP BY t.a", &["a", "COUNT(*)"]),
        (
            "SELECT COUNT(*), SUM(b), MIN(b), MAX(b), SUM(b) + 1 FROM t",
            &["COUNT(*)", "SUM(b)", "MIN(b)", "MAX(b)", "SUM(b) + 1"],
        ),
        ("SELECT COUNT(*) FROM t", &["COUNT(*)"]),
        ("SELECT COUNT(DISTINCT b) FROM t", &["COUNT(DISTINCT b)"]),
        ("SELECT MIN(b) FROM t", &["MIN(b)"]),
        ("SELECT MAX(b) FROM t", &["MAX(b)"]),
        (
            "SELECT a, SUM(b), RANK() OVER (ORDER BY SUM(b)) FROM t GROUP BY a",
            &["a", "SUM(b)", "RANK() OVER (ORDER BY SUM(b))"],
        ),
        (
            "SELECT ABS(-1), 1 + 2, 'x', CAST(1 AS TEXT)",
            &["ABS(-1)", "1 + 2", "'x'", "CAST(1 AS TEXT)"],
        ),
        ("SELECT ABS(-1), 1 + 2 WHERE 1 = 0", &["ABS(-1)", "1 + 2"]),
        ("SELECT (SELECT MAX(x) FROM u)", &["(SELECT MAX(x) FROM u)"]),
        (
            "SELECT (SELECT MAX(x) FROM u), COUNT(*)",
            &["(SELECT MAX(x) FROM u)", "COUNT(*)"],
        ),
        ("SELECT a, (SELECT MAX(x) FROM u) FROM t", &["a", "(SELECT MAX(x) FROM u)"]),
        (
            "SELECT a, (SELECT MAX(x) FROM u) FROM t GROUP BY a",
            &["a", "(SELECT MAX(x) FROM u)"],
        ),
        (
            "SELECT column1, column1 + 1 FROM (VALUES (1), (2)) AS w",
            &["column1", "column1 + 1"],
        ),
        (
            "SELECT value, value + 1 FROM generate_series(1, 2)",
            &["value", "value + 1"],
        ),
        ("SELECT t.*, u.x FROM t JOIN u ON u.id = t.id", &["id", "a", "b", "x"]),
        (
            "WITH s(id, a) AS (SELECT id, a FROM t) SELECT s.id, s.a FROM s",
            &["id", "a"],
        ),
        (
            "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM s WHERE n < 2) SELECT s.n FROM s",
            &["n"],
        ),
        ("SELECT g.x FROM generate_series(1, 2) AS g(x)", &["x"]),
    ];
    cases.extend(fixed.iter().map(|(sql, names)| {
        (
            sql.to_string(),
            names.iter().map(|n| n.to_string()).collect(),
        )
    }));
    cases
}

#[test]
fn a_result_column_is_named_by_its_alias_its_column_or_as_written() {
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in [("hot", hot("every_path")), ("sealed", sealed(&dir))] {
        for (sql, expected) in cases() {
            // The second run may answer from a cache
            for run in 0..2 {
                assert_eq!(names(&db, &sql), expected, "{context}, run {run}: {sql}");
            }
        }
    }
}

#[test]
fn an_alias_names_its_column_on_every_path() {
    let db = hot("aliases");
    for sql in [
        "SELECT a + 1 AS k FROM t",
        "SELECT SUM(b) AS k FROM t GROUP BY a",
        "SELECT COUNT(*) AS k FROM t",
        "SELECT MIN(b) AS k FROM t",
        "SELECT ROW_NUMBER() OVER (ORDER BY id) AS k FROM t",
        "SELECT 1 + 2 AS k",
        "SELECT * FROM (SELECT a + 1 AS k FROM t) AS d",
    ] {
        assert_eq!(names(&db, sql), ["k"], "{sql}");
    }
}

#[test]
fn a_prepared_statement_names_its_columns_as_the_statement_does() {
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in [("hot", hot("prepared")), ("sealed", sealed(&dir))] {
        let sql = format!("SELECT {ITEMS}, a + $1 FROM t WHERE id = $2");
        let stmt = db.prepare(&sql).unwrap();
        for run in 0..2 {
            let rows = stmt.query((1, 1)).unwrap();
            assert_eq!(rows.columns(), with(&["a + $1"]), "{context}, run {run}");
        }
    }
}

#[test]
fn returning_names_its_columns_as_a_select_list_does() {
    let db = hot("returning");
    let returning =
        "RETURNING a, t.b, A, ABS(a), CAST(a AS TEXT), a + 1, -a, 5, 'x', COALESCE(a, 0)";
    for sql in [
        format!("INSERT INTO t VALUES (9, 9, 9) {returning}"),
        format!("UPDATE t SET a = 8 WHERE id = 9 {returning}"),
        format!("DELETE FROM t WHERE id = 9 {returning}"),
    ] {
        assert_eq!(names(&db, &sql), with(&[]), "{sql}");
    }
}

#[test]
fn a_table_created_from_a_select_takes_its_names() {
    let db = hot("ctas");
    db.execute(
        "CREATE TABLE z AS SELECT a, t.b, ABS(a), a + 1, 1.5 FROM t",
        (),
    )
    .unwrap();
    assert_eq!(
        names(&db, "SELECT * FROM z"),
        ["a", "b", "ABS(a)", "a + 1", "1.5"]
    );
}

#[test]
fn a_grouped_column_keeps_the_spelling_of_its_item() {
    let db = hot("grouped_spelling");
    for (sql, expected) in [
        ("SELECT A, COUNT(*) FROM t GROUP BY a", ["A", "COUNT(*)"]),
        (
            "SELECT a AS K, COUNT(*) FROM t GROUP BY k",
            ["K", "COUNT(*)"],
        ),
        (
            "SELECT t.A, COUNT(*) FROM t GROUP BY t.a",
            ["A", "COUNT(*)"],
        ),
    ] {
        assert_eq!(names(&db, sql), expected, "{sql}");
    }
    assert_eq!(names(&db, "SELECT A FROM t GROUP BY a"), ["A"]);
}

#[test]
fn a_name_no_bare_word_spells_is_quoted_in_the_item_it_names() {
    let db = hot("quoted_names");
    db.execute(
        "CREATE TABLE q (id INTEGER PRIMARY KEY, \"A b\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO q VALUES (1, 2)", ()).unwrap();
    for (sql, expected) in [
        ("SELECT \"A b\", q.\"A b\" FROM q", vec!["A b", "A b"]),
        (
            "SELECT \"A b\" + 1, q.\"A b\" * 2, ABS(\"A b\") FROM q",
            vec!["\"A b\" + 1", "q.\"A b\" * 2", "ABS(\"A b\")"],
        ),
        (
            "SELECT \"A b\" + 1, COUNT(*) FROM q GROUP BY \"A b\"",
            vec!["\"A b\" + 1", "COUNT(*)"],
        ),
    ] {
        assert_eq!(names(&db, sql), expected, "{sql}");
    }
}
