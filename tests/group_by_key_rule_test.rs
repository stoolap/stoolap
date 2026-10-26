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

//! A GROUP BY key binds to a column of the rows the statement groups, to the
//! outer row of a subquery, or to a SELECT item by its alias or position; a
//! name that binds to none of them is an error, whatever path groups the rows

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE one (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO one VALUES (1, 10), (2, 10), (3, 20)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 5), (2, 6)",
        "CREATE VIEW v AS SELECT id, a FROM one",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn error(db: &Database, sql: &str) -> Option<String> {
    match db.query(sql, ()) {
        Err(e) => Some(e.to_string()),
        Ok(rows) => rows
            .map(|row| row.map(|_| ()))
            .collect::<Result<Vec<()>, _>>()
            .err()
            .map(|e| e.to_string()),
    }
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i)))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

#[test]
fn a_group_by_name_that_binds_to_nothing_is_an_error_on_every_path() {
    let db = open("group_by_name_binds_to_nothing");
    let mut wrong = Vec::new();
    for (sql, name) in [
        ("SELECT COUNT(*) FROM one GROUP BY nosuch", "nosuch"),
        ("SELECT a, COUNT(*) FROM one GROUP BY a, nosuch", "nosuch"),
        ("SELECT COUNT(*) FROM one GROUP BY one.nosuch", "one.nosuch"),
        (
            "SELECT COUNT(*) FROM one GROUP BY nosuch ORDER BY 1 LIMIT 1",
            "nosuch",
        ),
        (
            "SELECT COUNT(*) FROM one GROUP BY nosuch HAVING COUNT(*) > 0",
            "nosuch",
        ),
        (
            "SELECT COUNT(DISTINCT a) FROM one GROUP BY nosuch",
            "nosuch",
        ),
        ("SELECT COUNT(*) FROM one GROUP BY ROLLUP(nosuch)", "nosuch"),
        (
            "SELECT COUNT(*) FROM one GROUP BY GROUPING SETS ((a), (nosuch))",
            "nosuch",
        ),
        (
            "SELECT COUNT(*) FROM one JOIN u ON u.id = one.id GROUP BY nosuch",
            "nosuch",
        ),
        (
            "SELECT COUNT(*) FROM (SELECT * FROM one) d GROUP BY nosuch",
            "nosuch",
        ),
        (
            "WITH c AS (SELECT * FROM one) SELECT COUNT(*) FROM c GROUP BY nosuch",
            "nosuch",
        ),
        ("SELECT COUNT(*) FROM v GROUP BY nosuch", "nosuch"),
        (
            "SELECT COUNT(*) FROM (VALUES (1), (2)) w(z) GROUP BY nosuch",
            "nosuch",
        ),
        (
            "SELECT COUNT(*), (SELECT MAX(x) FROM u) FROM one GROUP BY nosuch",
            "nosuch",
        ),
        ("SELECT COUNT(*) GROUP BY nosuch", "nosuch"),
        (
            "SELECT id, (SELECT COUNT(*) FROM u GROUP BY nosuch) FROM one",
            "nosuch",
        ),
    ] {
        match error(&db, sql) {
            Some(e)
                if e.to_lowercase()
                    .contains(&format!("column '{name}' not found")) => {}
            other => wrong.push(format!("{sql}: {other:?}")),
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
fn a_group_by_name_that_binds_groups_as_before() {
    let db = open("group_by_name_binds");
    for (sql, want) in [
        (
            "SELECT a, COUNT(*) FROM one GROUP BY a ORDER BY a",
            vec![
                "Some(Integer(10))|Some(Integer(2))",
                "Some(Integer(20))|Some(Integer(1))",
            ],
        ),
        (
            "SELECT a + 1 AS k, COUNT(*) FROM one GROUP BY k ORDER BY k",
            vec![
                "Some(Integer(11))|Some(Integer(2))",
                "Some(Integer(21))|Some(Integer(1))",
            ],
        ),
        (
            "SELECT a, COUNT(*) FROM one GROUP BY 1 ORDER BY 1",
            vec![
                "Some(Integer(10))|Some(Integer(2))",
                "Some(Integer(20))|Some(Integer(1))",
            ],
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM u GROUP BY one.a) FROM one ORDER BY id",
            vec![
                "Some(Integer(1))|Some(Integer(2))",
                "Some(Integer(2))|Some(Integer(2))",
                "Some(Integer(3))|Some(Integer(2))",
            ],
        ),
    ] {
        assert_eq!(rows(&db, sql), want, "{sql}");
    }
}

#[test]
fn an_aggregate_group_by_key_is_one_error_however_it_is_written() {
    let db = open("aggregate_group_by_key");
    let mut wrong = Vec::new();
    for sql in [
        "SELECT COUNT(*) FROM one GROUP BY 1",
        "SELECT COUNT(*) AS c FROM one GROUP BY c",
        "SELECT COUNT(*) FROM one GROUP BY COUNT(*)",
        "SELECT a, SUM(id) + 1 FROM one GROUP BY 2",
        "SELECT a, SUM(id) + 1 AS s FROM one GROUP BY s",
        "SELECT a FROM one GROUP BY a + SUM(id)",
        "SELECT a, COUNT(*) FROM one GROUP BY a, 2",
        "SELECT a, COUNT(*) AS c FROM one GROUP BY a, c HAVING COUNT(*) > 0",
        "SELECT a, COUNT(*) FROM one GROUP BY ROLLUP(2)",
        "SELECT a, COUNT(*) FROM one GROUP BY GROUPING SETS ((a), (2))",
        "SELECT a, COUNT(*) FROM one JOIN u ON u.id = one.id GROUP BY 2",
        "SELECT COUNT(*) AS c FROM v GROUP BY c",
        "SELECT COUNT(*), (SELECT MAX(x) FROM u) FROM one GROUP BY 1",
        "SELECT COUNT(*) AS c, (SELECT MAX(x) FROM u) FROM one GROUP BY c",
        "SELECT COUNT(*) GROUP BY 1",
        "SELECT COUNT(*) GROUP BY 1 HAVING TRUE",
        "SELECT COUNT(*) AS c GROUP BY c",
        "SELECT COUNT(*) GROUP BY COUNT(*)",
        "SELECT id, (SELECT COUNT(*) AS c FROM u GROUP BY c) FROM one",
        "SELECT COUNT(*) FROM one GROUP BY SUM(id) IN ()",
        "SELECT COUNT(*) FROM one GROUP BY SUM(id) IN (1, 2)",
        "SELECT COUNT(*) FROM one GROUP BY a IN (SUM(id))",
        "SELECT COUNT(*) FROM one GROUP BY SUM(id) BETWEEN 1 AND 2",
        "SELECT COUNT(*) FROM one GROUP BY a BETWEEN 1 AND MAX(id)",
        "SELECT COUNT(*) FROM one GROUP BY CASE SUM(id) WHEN 1 THEN 1 END",
        "SELECT COUNT(*) FROM one GROUP BY CAST(SUM(id) AS TEXT) LIKE '1%'",
        "SELECT COUNT(*) FROM one GROUP BY 'x' LIKE CAST(MIN(id) AS TEXT)",
        "SELECT COUNT(*) FROM one GROUP BY (a, SUM(id)) = (1, 2)",
        "SELECT COUNT(*) AS c FROM one GROUP BY \"c\"",
        "SELECT COUNT(*) AS c, (SELECT MAX(x) FROM u) FROM one GROUP BY \"c\"",
        "SELECT id, (SELECT COUNT(*) AS c FROM u GROUP BY \"c\") FROM one",
    ] {
        match error(&db, sql) {
            Some(e) if e.contains("an aggregate function cannot be a GROUP BY key") => {}
            other => wrong.push(format!("{sql}: {other:?}")),
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
fn an_aggregate_alias_leaves_a_column_of_its_name_to_group_by() {
    let db = open("aggregate_alias_leaves_the_column");
    assert_eq!(
        rows(&db, "SELECT COUNT(*) AS a FROM one GROUP BY a ORDER BY 1"),
        vec!["Some(Integer(1))", "Some(Integer(2))"]
    );
    assert_eq!(
        rows(&db, "SELECT MAX(id) AS a FROM one GROUP BY a ORDER BY 1"),
        vec!["Some(Integer(2))", "Some(Integer(3))"]
    );
}

#[test]
fn a_quoted_group_by_name_binds_as_the_compiler_binds_it() {
    let db = open("quoted_group_by_name");
    for (sql, want) in [
        (
            "SELECT COUNT(*) FROM one GROUP BY \"constant\"",
            vec!["Some(Integer(3))"],
        ),
        (
            "SELECT \"constant\", COUNT(*) FROM one GROUP BY \"constant\"",
            vec!["Some(Text(\"constant\"))|Some(Integer(3))"],
        ),
        (
            "SELECT COUNT(*) FROM one GROUP BY \"constant\" || ''",
            vec!["Some(Integer(3))"],
        ),
        (
            "SELECT COUNT(*), (SELECT MAX(x) FROM u) FROM one GROUP BY \"constant\"",
            vec!["Some(Integer(3))|Some(Integer(6))"],
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM u GROUP BY \"constant\") FROM one WHERE id = 1",
            vec!["Some(Integer(1))|Some(Integer(2))"],
        ),
        (
            "SELECT \"a\", COUNT(*) FROM one GROUP BY \"a\" ORDER BY 1",
            vec![
                "Some(Integer(10))|Some(Integer(2))",
                "Some(Integer(20))|Some(Integer(1))",
            ],
        ),
        (
            "SELECT a, COUNT(*) FROM one GROUP BY \"a\" ORDER BY 1",
            vec![
                "Some(Integer(10))|Some(Integer(2))",
                "Some(Integer(20))|Some(Integer(1))",
            ],
        ),
        (
            "SELECT one.a, COUNT(*) FROM one GROUP BY \"a\" HAVING one.a > 0 ORDER BY one.a",
            vec![
                "Some(Integer(10))|Some(Integer(2))",
                "Some(Integer(20))|Some(Integer(1))",
            ],
        ),
        (
            "SELECT \"constant\", COUNT(*) FROM one GROUP BY 1",
            vec!["Some(Text(\"constant\"))|Some(Integer(3))"],
        ),
        (
            "SELECT \"constant\" AS k, COUNT(*) FROM one GROUP BY 1",
            vec!["Some(Text(\"constant\"))|Some(Integer(3))"],
        ),
        (
            "SELECT \"constant\", COUNT(*), (SELECT MAX(x) FROM u) FROM one GROUP BY 1",
            vec!["Some(Text(\"constant\"))|Some(Integer(3))|Some(Integer(6))"],
        ),
        (
            "SELECT \"c\" AS k, COUNT(*) AS c FROM one GROUP BY 1",
            vec!["Some(Text(\"c\"))|Some(Integer(3))"],
        ),
        (
            "SELECT \"c\" AS k, COUNT(*) AS c FROM one GROUP BY k",
            vec!["Some(Text(\"c\"))|Some(Integer(3))"],
        ),
        (
            "SELECT EXISTS (SELECT \"c\" AS k, COUNT(*) AS c FROM u GROUP BY k) FROM one \
             WHERE id = 1",
            vec!["Some(Boolean(true))"],
        ),
    ] {
        assert_eq!(rows(&db, sql), want, "{sql}");
    }
}
