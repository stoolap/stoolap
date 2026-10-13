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

//! A GROUP BY expression selected under an alias can be built on by another
//! column of the same SELECT, as it can when it is selected without one.

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Vec<(i64, Option<i64>)> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<i64>(0).unwrap(),
                row.get::<Option<i64>>(1).unwrap(),
            )
        })
        .collect()
}

#[test]
fn test_aliased_group_key_reused_by_another_column() {
    let db = Database::open("memory://group_by_aliased_key_reuse").unwrap();
    db.execute("CREATE TABLE t (ts INTEGER, price FLOAT)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (0, 1.0), (60, 2.0), (300, 3.0), (360, 5.0)",
        (),
    )
    .unwrap();

    let expected = vec![(0, Some(0)), (1, Some(300))];
    assert_eq!(
        rows(
            &db,
            "SELECT ts / 300 AS id, (ts / 300) * 300 AS bucket FROM t GROUP BY ts / 300 ORDER BY id"
        ),
        expected
    );
    assert_eq!(
        rows(
            &db,
            "SELECT ts / 300 AS id, (ts / 300) * 300 AS ts FROM t GROUP BY ts / 300 ORDER BY id"
        ),
        expected
    );
    assert_eq!(
        rows(
            &db,
            "SELECT ts / 300, (ts / 300) * 300 AS bucket FROM t GROUP BY ts / 300 ORDER BY 1"
        ),
        expected
    );
}

#[test]
fn test_aliased_function_group_key_reused_by_another_column() {
    let db = Database::open("memory://group_by_aliased_function_key_reuse").unwrap();
    db.execute("CREATE TABLE t (name TEXT)", ()).unwrap();
    db.execute("INSERT INTO t VALUES ('ab'), ('ab'), ('cde')", ())
        .unwrap();

    let lengths: Vec<Option<i64>> = db
        .query(
            "SELECT upper(name) AS u, length(upper(name)) AS l FROM t GROUP BY upper(name) ORDER BY u",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<Option<i64>>(1).unwrap())
        .collect();
    assert_eq!(lengths, vec![Some(2), Some(3)]);
}

fn strings(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{:?}", row.get::<stoolap::Value>(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

#[test]
fn test_aliased_group_key_with_string_literals_is_not_reused() {
    let db = Database::open("memory://group_by_literal_case").unwrap();
    db.execute("CREATE TABLE t (s TEXT)", ()).unwrap();
    db.execute("INSERT INTO t VALUES ('A')", ()).unwrap();
    assert_eq!(
        strings(
            &db,
            "SELECT replace(s, 'a', 'b') AS g, replace(s, 'A', 'B') AS h FROM t GROUP BY s, replace(s, 'a', 'b')"
        ),
        strings(&db, "SELECT 'A', 'B'")
    );

    db.execute("CREATE TABLE q (s TEXT)", ()).unwrap();
    db.execute("INSERT INTO q VALUES ('''A')", ()).unwrap();
    assert_eq!(
        strings(
            &db,
            "SELECT replace(s, '''a', 'b') AS g, replace(s, '''A', 'B') AS h FROM q GROUP BY s, replace(s, '''a', 'b')"
        ),
        strings(&db, "SELECT '''A', 'B'")
    );
}

#[test]
fn test_literal_case_in_other_aggregate_lookups() {
    let db = Database::open("memory://aggregate_literal_lookups").unwrap();
    db.execute("CREATE TABLE t (s TEXT)", ()).unwrap();
    db.execute("INSERT INTO t VALUES ('A'), ('b')", ()).unwrap();

    assert_eq!(
        strings(&db, "SELECT MAX('UPPER'), COUNT(*) + 1 FROM t"),
        strings(&db, "SELECT 'UPPER', 3")
    );
    assert_eq!(
        strings(
            &db,
            "SELECT s FROM t GROUP BY s HAVING SUM(CASE WHEN s = 'A' THEN 1 ELSE 0 END) > 0"
        ),
        strings(&db, "SELECT 'A'")
    );
    assert_eq!(
        strings(
            &db,
            "SELECT s, RANK() OVER (ORDER BY SUM(CASE WHEN s = 'A' THEN 1 ELSE 0 END) DESC) FROM t GROUP BY s ORDER BY s"
        ),
        strings(&db, "SELECT 'A', 1 UNION ALL SELECT 'b', 2")
    );

    db.execute("CREATE TABLE w (\"a'B\" INTEGER)", ()).unwrap();
    db.execute("INSERT INTO w VALUES (1), (2)", ()).unwrap();
    assert_eq!(
        strings(
            &db,
            "SELECT COUNT(*) OVER (PARTITION BY \"a'B\") FROM w ORDER BY 1"
        ),
        strings(&db, "SELECT 1 UNION ALL SELECT 1")
    );
}

#[test]
fn test_float_literal_is_not_matched_to_an_integer_group_key() {
    let db = Database::open("memory://group_by_float_literal").unwrap();
    db.execute("CREATE TABLE t (v INTEGER)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (5)", ()).unwrap();

    assert_eq!(
        strings(
            &db,
            "SELECT v / 2 AS g, v / 2.0 AS h FROM t GROUP BY v, v / 2"
        ),
        strings(&db, "SELECT 2, 2.5")
    );
}

#[test]
fn test_quoted_column_is_not_matched_to_an_expression_group_key() {
    let db = Database::open("memory://group_by_quoted_column").unwrap();
    db.execute("CREATE TABLE t (v INTEGER, \"v / 2\" INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (5, 99)", ()).unwrap();
    for (sql, expected) in [
        (
            "SELECT \"v / 2\" AS g, v / 2 AS h FROM t GROUP BY v, g",
            "SELECT 99, 2",
        ),
        (
            "SELECT \"v / 2\" + 1 AS g, (v / 2) + 1 AS h FROM t GROUP BY v, g",
            "SELECT 100, 3",
        ),
    ] {
        assert_eq!(strings(&db, sql), strings(&db, expected), "{}", sql);
    }
}

#[test]
fn test_quoted_column_is_not_matched_to_a_literal() {
    let db = Database::open("memory://group_by_quoted_null").unwrap();
    db.execute("CREATE TABLE t (\"null\" INTEGER)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (7)", ()).unwrap();
    assert_eq!(
        strings(&db, "SELECT \"null\" AS g, NULL AS h FROM t GROUP BY g"),
        strings(&db, "SELECT 7, NULL")
    );
}

#[test]
fn test_float_group_key_does_not_split_on_its_decimal_point() {
    let db = Database::open("memory://group_by_float_key_name").unwrap();
    db.execute("CREATE TABLE t (v INTEGER)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (5)", ()).unwrap();
    assert_eq!(
        strings(&db, "SELECT v / 2.0, 0 AS z FROM t GROUP BY v / 2.0"),
        strings(&db, "SELECT 2.5, 0")
    );
}
