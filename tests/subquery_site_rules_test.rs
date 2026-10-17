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

//! A subquery in any clause gives the answer its value gives written as a
//! literal, or an error until the clause runs on a prepared plan: never a
//! NULL it did not compute, a dropped predicate or a skipped check.

use stoolap::Database;

/// t: id 1..6, a = id % 3, b = id * 10. u: id 1..3, x = id
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    for id in 1..=6i64 {
        db.execute("INSERT INTO t VALUES ($1, $2, $3)", (id, id % 3, id * 10))
            .unwrap();
    }
    for id in 1..=3i64 {
        db.execute("INSERT INTO u VALUES ($1, $1)", (id,)).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<Vec<String>>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect(),
        );
    }
    Ok(out)
}

fn table(db: &Database) -> Vec<Vec<String>> {
    rows(db, "SELECT * FROM t ORDER BY id").unwrap()
}

/// The literal's answer, or an error until `owner` prepares the clause
fn never_wrong(name: &str, owner: &str, with_subquery: &str, with_literal: &str) {
    let db = setup(name);
    let expected = rows(&db, with_literal).unwrap();
    if let Ok(got) = rows(&db, with_subquery) {
        assert_eq!(got, expected, "{with_subquery} (prepared by {owner})");
    }
}

/// The literal's answer, now
fn same(name: &str, with_subquery: &str, with_literal: &str) {
    let db = setup(name);
    let expected = rows(&db, with_literal).unwrap();
    assert_eq!(rows(&db, with_subquery), Ok(expected), "{with_subquery}");
}

/// The literal's effect, or an error that leaves the table as it was
fn never_wrong_effect(name: &str, owner: &str, with_subquery: &str, with_literal: &str) {
    let left = setup(&format!("{name}_subquery"));
    let right = setup(&format!("{name}_literal"));
    let before = table(&left);
    right.execute(with_literal, ()).unwrap();
    let expected = match left.execute(with_subquery, ()) {
        Ok(_) => table(&right),
        Err(_) => before,
    };
    assert_eq!(
        table(&left),
        expected,
        "{with_subquery} (prepared by {owner})"
    );
}

#[test]
fn test_select_list_exists_and_quantified() {
    same(
        "site_select_exists",
        "SELECT id, EXISTS (SELECT 1 FROM u) FROM t ORDER BY id",
        "SELECT id, TRUE FROM t ORDER BY id",
    );
    same(
        "site_select_all",
        "SELECT id, a < ALL (SELECT x FROM u) FROM t ORDER BY id",
        "SELECT id, a < 1 FROM t ORDER BY id",
    );
    same(
        "site_select_correlated_exists",
        "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.id = t.a) FROM t ORDER BY id",
        "SELECT id, a BETWEEN 1 AND 3 FROM t ORDER BY id",
    );
}

#[test]
fn test_constant_arithmetic_on_a_subquery() {
    same(
        "site_constant_fold",
        "SELECT id FROM t WHERE b = (SELECT 1) + 9 ORDER BY id",
        "SELECT id FROM t WHERE b = 10 ORDER BY id",
    );
}

#[test]
fn test_limit_and_offset() {
    never_wrong(
        "site_offset",
        "step 1",
        "SELECT id FROM t ORDER BY id OFFSET (SELECT 2)",
        "SELECT id FROM t ORDER BY id OFFSET 2",
    );
    never_wrong(
        "site_limit",
        "step 1",
        "SELECT id FROM t ORDER BY id LIMIT (SELECT 2)",
        "SELECT id FROM t ORDER BY id LIMIT 2",
    );
}

#[test]
fn test_as_of_where() {
    never_wrong(
        "site_as_of",
        "step 1",
        "SELECT id FROM t AS OF TIMESTAMP '2999-01-01 00:00:00' WHERE a IN (SELECT x FROM u) ORDER BY id",
        "SELECT id FROM t AS OF TIMESTAMP '2999-01-01 00:00:00' WHERE a IN (1, 2, 3) ORDER BY id",
    );
}

#[test]
fn test_vector_search_select_list() {
    let db = Database::open("memory://site_vector").unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, e VECTOR(2))", ())
        .unwrap();
    for (id, e) in [(1, "[1,2]"), (2, "[2,2]"), (3, "[5,5]"), (4, "[9,9]")] {
        db.execute("INSERT INTO d VALUES ($1, $2)", (id, e))
            .unwrap();
    }
    let with_literal = "SELECT id, 4 FROM d ORDER BY VEC_DISTANCE_L2(e, '[1,2]') LIMIT 3";
    let with_subquery =
        "SELECT id, (SELECT COUNT(*) FROM d) FROM d ORDER BY VEC_DISTANCE_L2(e, '[1,2]') LIMIT 3";
    let expected = rows(&db, with_literal).unwrap();
    if let Ok(got) = rows(&db, with_subquery) {
        assert_eq!(got, expected, "{with_subquery} (prepared by step 1)");
    }
}

#[test]
fn test_correlated_conjunct_beside_an_indexed_in() {
    never_wrong(
        "site_index_correlated_conjunct",
        "step 2",
        "SELECT id FROM t WHERE id IN (SELECT x FROM u) AND EXISTS (SELECT 1 FROM u w WHERE w.id = t.a) ORDER BY id",
        "SELECT id FROM t WHERE id IN (1, 2, 3) AND a BETWEEN 1 AND 3 ORDER BY id",
    );
}

#[test]
fn test_correlated_join_on() {
    never_wrong(
        "site_correlated_on",
        "step 3",
        "SELECT t.id, u.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a) ORDER BY t.id",
        "SELECT t.id, u.id FROM t JOIN u ON u.id = t.a ORDER BY t.id",
    );
}

#[test]
fn test_group_by_expression() {
    never_wrong(
        "site_group_by",
        "step 4",
        "SELECT COUNT(*) FROM t GROUP BY id > (SELECT MAX(x) FROM u) ORDER BY 1",
        "SELECT COUNT(*) FROM t GROUP BY id > 3 ORDER BY 1",
    );
}

#[test]
fn test_having_beside_a_window() {
    never_wrong(
        "site_having_window",
        "step 4",
        "SELECT a, SUM(b), RANK() OVER (ORDER BY a) FROM t GROUP BY a HAVING SUM(b) > (SELECT 70) ORDER BY a",
        "SELECT a, SUM(b), RANK() OVER (ORDER BY a) FROM t GROUP BY a HAVING SUM(b) > 70 ORDER BY a",
    );
}

#[test]
fn test_grouped_items_mixing_correlated_and_uncorrelated() {
    never_wrong(
        "site_grouped_mixed",
        "step 4",
        "SELECT a, (SELECT MAX(x) FROM u), (SELECT COUNT(*) FROM u WHERE u.id = t.a) FROM t GROUP BY a ORDER BY a",
        "SELECT a, 3, CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END FROM t GROUP BY a ORDER BY a",
    );
}

#[test]
fn test_distinct_on_expression() {
    never_wrong(
        "site_distinct_on",
        "step 4",
        "SELECT DISTINCT ON (a + (SELECT COUNT(*) FROM u)) a FROM t ORDER BY a + (SELECT COUNT(*) FROM u), id",
        "SELECT DISTINCT ON (a + 3) a FROM t ORDER BY a + 3, id",
    );
}

#[test]
fn test_window_control_argument_reading_the_row() {
    let db = setup("site_window_control");
    let got = rows(
        &db,
        "SELECT id, NTILE((SELECT MAX(x) FROM u WHERE u.x = t.a)) OVER (ORDER BY id) FROM t ORDER BY id",
    );
    assert!(got.is_err(), "a control argument reading the row: {got:?}");
}

#[test]
fn test_insert_values() {
    never_wrong_effect(
        "site_insert_values",
        "step 5",
        "INSERT INTO t VALUES (7, (SELECT MAX(x) FROM u), 0)",
        "INSERT INTO t VALUES (7, 3, 0)",
    );
}

#[test]
fn test_on_conflict_set() {
    never_wrong_effect(
        "site_on_conflict",
        "step 5",
        "INSERT INTO t VALUES (1, 0, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT MAX(x) FROM u)",
        "INSERT INTO t VALUES (1, 0, 0) ON CONFLICT (id) DO UPDATE SET b = 3",
    );
}

#[test]
fn test_update_set_mixing_correlated_and_uncorrelated() {
    never_wrong_effect(
        "site_update_mixed",
        "step 5",
        "UPDATE t SET b = (SELECT MAX(x) FROM u), a = (SELECT COUNT(*) FROM u WHERE u.id = t.id)",
        "UPDATE t SET b = 3, a = CASE WHEN id <= 3 THEN 1 ELSE 0 END",
    );
}

#[test]
fn test_returning() {
    let db = setup("site_returning");
    let with_subquery = "DELETE FROM t WHERE id = 1 RETURNING id, (SELECT MAX(x) FROM u)";
    if let Ok(got) = rows(&db, with_subquery) {
        assert_eq!(
            got,
            vec![vec!["1".to_string(), "3".to_string()]],
            "{with_subquery} (prepared by step 5)"
        );
    }
}

#[test]
fn test_column_default() {
    let db = setup("site_default");
    if db
        .execute(
            "CREATE TABLE d (id INTEGER PRIMARY KEY, v INTEGER DEFAULT (SELECT MAX(x) FROM u))",
            (),
        )
        .is_err()
    {
        return;
    }
    db.execute("INSERT INTO d VALUES (1, 7)", ())
        .expect("a row that names the column takes no default");
    if db.execute("INSERT INTO d (id) VALUES (2)", ()).is_ok() {
        assert_eq!(
            rows(&db, "SELECT v FROM d WHERE id = 2"),
            Ok(vec![vec!["3".to_string()]]),
            "DEFAULT (SELECT MAX(x) FROM u) (prepared by step 5)"
        );
    }
}

#[test]
fn test_check() {
    let db = setup("site_check");
    if db
        .execute(
            "CREATE TABLE c (id INTEGER PRIMARY KEY, w INTEGER CHECK (w < (SELECT MAX(x) FROM u)))",
            (),
        )
        .is_err()
    {
        return;
    }
    let _ = db.execute("INSERT INTO c VALUES (1, 10)", ());
    assert_eq!(
        rows(&db, "SELECT COUNT(*) FROM c"),
        Ok(vec![vec!["0".to_string()]]),
        "a row breaking CHECK (w < (SELECT MAX(x) FROM u)) was stored (prepared by step 5)"
    );
}

#[test]
fn test_derived_table_where() {
    never_wrong(
        "site_derived_where",
        "step 6",
        "SELECT id FROM (SELECT id, a FROM t) s WHERE id > (SELECT MAX(x) FROM u) ORDER BY id",
        "SELECT id FROM (SELECT id, a FROM t) s WHERE id > 3 ORDER BY id",
    );
}

#[test]
fn test_values_source_where() {
    never_wrong(
        "site_values_where",
        "step 6",
        "SELECT n FROM (VALUES (1), (5)) AS w(n) WHERE n > (SELECT MAX(x) FROM u)",
        "SELECT n FROM (VALUES (1), (5)) AS w(n) WHERE n > 3",
    );
}

#[test]
fn test_view_where_and_select_list() {
    let db = setup("site_view");
    db.execute("CREATE VIEW v AS SELECT id, a FROM t", ())
        .unwrap();
    for (with_subquery, with_literal) in [
        (
            "SELECT id FROM v WHERE id > (SELECT MAX(x) FROM u) ORDER BY id",
            "SELECT id FROM v WHERE id > 3 ORDER BY id",
        ),
        (
            "SELECT id, (SELECT MAX(x) FROM u) FROM v ORDER BY id",
            "SELECT id, 3 FROM v ORDER BY id",
        ),
    ] {
        let expected = rows(&db, with_literal).unwrap();
        if let Ok(got) = rows(&db, with_subquery) {
            assert_eq!(got, expected, "{with_subquery} (prepared by step 6)");
        }
    }
}

#[test]
fn test_cte_block_reading_the_grandparent_row() {
    never_wrong(
        "site_cte_grandparent",
        "step 6",
        "WITH c AS (SELECT 1 AS x) SELECT id FROM t WHERE EXISTS (SELECT 1 FROM c WHERE c.x IN (SELECT u.x FROM u WHERE u.id = t.a)) ORDER BY id",
        "SELECT id FROM t WHERE a = 1 ORDER BY id",
    );
}

#[test]
fn test_table_function_argument() {
    never_wrong(
        "site_tvf_argument",
        "step 6",
        "SELECT * FROM generate_series(1, (SELECT 3))",
        "SELECT * FROM generate_series(1, 3)",
    );
}
