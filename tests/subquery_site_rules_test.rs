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

//! The ledger of subquery evaluation sites. Each site states what it does
//! today: `Right`, the answer its value gives written as a literal, or
//! `Fails`, an error that changes nothing until the named step prepares it.
//! A site that answers otherwise fails here, a wrong answer and a site a
//! step fixed without updating the ledger alike.

use stoolap::Database;

/// What a site does today
#[derive(Clone, Copy, Debug)]
enum State {
    /// The literal's answer
    Right,
    /// An error that changes nothing, until the plan step named prepares it
    Fails(&'static str),
}

use State::{Fails, Right};

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

/// That `got`, the outcome of `what`, is what `state` says, against `want`
fn holds<T: PartialEq + std::fmt::Debug>(
    state: State,
    got: Result<T, String>,
    want: &T,
    what: &str,
) {
    match (state, got) {
        (Right, Ok(got)) => assert_eq!(&got, want, "{what}"),
        (Right, Err(e)) => panic!("{what} should answer as its literal: {e}"),
        (Fails(_), Err(_)) => {}
        (Fails(owner), Ok(got)) if &got == want => {
            panic!("{what} now answers right: mark it Right, {owner} prepared it")
        }
        (Fails(owner), Ok(got)) => {
            panic!("{what} gave {got:?}, want {want:?}, or an error until {owner}")
        }
    }
}

/// A query site
fn site(name: &str, state: State, with_subquery: &str, with_literal: &str) {
    let db = setup(name);
    let want = rows(&db, with_literal).unwrap();
    holds(state, rows(&db, with_subquery), &want, with_subquery);
}

/// A write site: the literal's effect, or an error that leaves the table
fn write_site(name: &str, state: State, with_subquery: &str, with_literal: &str) {
    let left = setup(&format!("{name}_subquery"));
    let right = setup(&format!("{name}_literal"));
    let before = table(&left);
    right.execute(with_literal, ()).unwrap();
    let got = left.execute(with_subquery, ()).map_err(|e| e.to_string());
    if got.is_err() {
        assert_eq!(table(&left), before, "{with_subquery} failed but wrote");
    }
    holds(
        state,
        got.map(|_| table(&left)),
        &table(&right),
        with_subquery,
    );
}

#[test]
fn test_select_list_exists_and_quantified() {
    site(
        "site_select_exists",
        Right,
        "SELECT id, EXISTS (SELECT 1 FROM u) FROM t ORDER BY id",
        "SELECT id, TRUE FROM t ORDER BY id",
    );
    site(
        "site_select_all",
        Right,
        "SELECT id, a < ALL (SELECT x FROM u) FROM t ORDER BY id",
        "SELECT id, a < 1 FROM t ORDER BY id",
    );
    site(
        "site_select_correlated_exists",
        Right,
        "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.id = t.a) FROM t ORDER BY id",
        "SELECT id, a BETWEEN 1 AND 3 FROM t ORDER BY id",
    );
}

#[test]
fn test_constant_arithmetic_on_a_subquery() {
    site(
        "site_constant_fold",
        Right,
        "SELECT id FROM t WHERE b = (SELECT 1) + 9 ORDER BY id",
        "SELECT id FROM t WHERE b = 10 ORDER BY id",
    );
}

#[test]
fn test_limit_and_offset() {
    site(
        "site_offset",
        Right,
        "SELECT id FROM t ORDER BY id OFFSET (SELECT 2)",
        "SELECT id FROM t ORDER BY id OFFSET 2",
    );
    site(
        "site_limit",
        Right,
        "SELECT id FROM t ORDER BY id LIMIT (SELECT 2)",
        "SELECT id FROM t ORDER BY id LIMIT 2",
    );
}

#[test]
fn test_as_of_where() {
    site(
        "site_as_of",
        Fails("step 1"),
        "SELECT id FROM t AS OF TIMESTAMP '2262-01-01 00:00:00' WHERE a IN (SELECT x FROM u) ORDER BY id",
        "SELECT id FROM t AS OF TIMESTAMP '2262-01-01 00:00:00' WHERE a IN (1, 2, 3) ORDER BY id",
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
    let with_subquery =
        "SELECT id, (SELECT COUNT(*) FROM d) FROM d ORDER BY VEC_DISTANCE_L2(e, '[1,2]') LIMIT 3";
    let want = rows(
        &db,
        "SELECT id, 4 FROM d ORDER BY VEC_DISTANCE_L2(e, '[1,2]') LIMIT 3",
    )
    .unwrap();
    holds(Right, rows(&db, with_subquery), &want, with_subquery);
}

#[test]
fn test_correlated_conjunct_beside_an_indexed_in() {
    site(
        "site_index_correlated_conjunct",
        Right,
        "SELECT id FROM t WHERE id IN (SELECT x FROM u) AND EXISTS (SELECT 1 FROM u w WHERE w.id = t.a) ORDER BY id",
        "SELECT id FROM t WHERE id IN (1, 2, 3) AND a BETWEEN 1 AND 3 ORDER BY id",
    );
}

#[test]
fn test_correlated_join_on() {
    site(
        "site_correlated_on",
        Right,
        "SELECT t.id, u.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a) ORDER BY t.id",
        "SELECT t.id, u.id FROM t JOIN u ON u.id = t.a ORDER BY t.id",
    );
}

#[test]
fn test_group_by_expression() {
    site(
        "site_group_by",
        Right,
        "SELECT COUNT(*) FROM t GROUP BY id > (SELECT MAX(x) FROM u) ORDER BY 1",
        "SELECT COUNT(*) FROM t GROUP BY id > 3 ORDER BY 1",
    );
}

#[test]
fn test_having_beside_a_window() {
    site(
        "site_having_window",
        Right,
        "SELECT a, SUM(b), RANK() OVER (ORDER BY a) FROM t GROUP BY a HAVING SUM(b) > (SELECT 70) ORDER BY a",
        "SELECT a, SUM(b), RANK() OVER (ORDER BY a) FROM t GROUP BY a HAVING SUM(b) > 70 ORDER BY a",
    );
}

#[test]
fn test_grouped_items_mixing_correlated_and_uncorrelated() {
    site(
        "site_grouped_mixed",
        Right,
        "SELECT a, (SELECT MAX(x) FROM u), (SELECT COUNT(*) FROM u WHERE u.id = t.a) FROM t GROUP BY a ORDER BY a",
        "SELECT a, 3, CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END FROM t GROUP BY a ORDER BY a",
    );
}

#[test]
fn test_distinct_on_expression() {
    site(
        "site_distinct_on",
        Right,
        "SELECT DISTINCT ON (a + (SELECT COUNT(*) FROM u)) a FROM t ORDER BY a + (SELECT COUNT(*) FROM u), id",
        "SELECT DISTINCT ON (a + 3) a FROM t ORDER BY a + 3, id",
    );
}

#[test]
fn test_window_control_argument_reading_the_row() {
    // What a control argument that reads the row means is open (plan 5.8)
    let db = setup("site_window_control");
    let sql = "SELECT id, NTILE((SELECT MAX(x) FROM u WHERE u.x = t.a)) OVER (ORDER BY id) FROM t ORDER BY id";
    let got = rows(&db, sql);
    assert!(
        got.is_err(),
        "{sql} answered {got:?} before its rule is set"
    );
}

#[test]
fn test_insert_values() {
    write_site(
        "site_insert_values",
        Right,
        "INSERT INTO t VALUES (7, (SELECT MAX(x) FROM u), 0)",
        "INSERT INTO t VALUES (7, 3, 0)",
    );
}

#[test]
fn test_on_conflict_set() {
    write_site(
        "site_on_conflict",
        Right,
        "INSERT INTO t VALUES (1, 0, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT MAX(x) FROM u)",
        "INSERT INTO t VALUES (1, 0, 0) ON CONFLICT (id) DO UPDATE SET b = 3",
    );
}

#[test]
fn test_update_set_mixing_correlated_and_uncorrelated() {
    write_site(
        "site_update_mixed",
        Right,
        "UPDATE t SET b = (SELECT MAX(x) FROM u), a = (SELECT COUNT(*) FROM u WHERE u.id = t.id)",
        "UPDATE t SET b = 3, a = CASE WHEN id <= 3 THEN 1 ELSE 0 END",
    );
}

#[test]
fn test_returning() {
    let db = setup("site_returning");
    let before = table(&db);
    let sql = "DELETE FROM t WHERE id = 1 RETURNING id, (SELECT MAX(x) FROM u)";
    let got = rows(&db, sql);
    if got.is_err() {
        assert_eq!(table(&db), before, "{sql} failed but deleted");
    }
    let want = vec![vec!["1".to_string(), "3".to_string()]];
    holds(Right, got, &want, sql);
}

#[test]
fn test_column_default() {
    // A DEFAULT may not hold a subquery: CREATE refuses it (plan 5.3)
    let db = setup("site_default");
    let sql = "CREATE TABLE d (id INTEGER PRIMARY KEY, v INTEGER DEFAULT (SELECT MAX(x) FROM u))";
    assert!(db.execute(sql, ()).is_err(), "{sql} was accepted");
    assert!(
        db.execute("SELECT * FROM d", ()).is_err(),
        "{sql} failed but created d"
    );
}

#[test]
fn test_check() {
    // A CHECK may not hold a subquery: CREATE refuses it (plan 5.3)
    let db = setup("site_check");
    let sql =
        "CREATE TABLE c (id INTEGER PRIMARY KEY, w INTEGER CHECK (w < (SELECT MAX(x) FROM u)))";
    assert!(db.execute(sql, ()).is_err(), "{sql} was accepted");
    assert!(
        db.execute("SELECT * FROM c", ()).is_err(),
        "{sql} failed but created c"
    );
}

#[test]
fn test_derived_table_where() {
    site(
        "site_derived_where",
        Right,
        "SELECT id FROM (SELECT id, a FROM t) s WHERE id > (SELECT MAX(x) FROM u) ORDER BY id",
        "SELECT id FROM (SELECT id, a FROM t) s WHERE id > 3 ORDER BY id",
    );
}

#[test]
fn test_values_source_where() {
    site(
        "site_values_where",
        Right,
        "SELECT n FROM (VALUES (1), (5)) AS w(n) WHERE n > (SELECT MAX(x) FROM u)",
        "SELECT n FROM (VALUES (1), (5)) AS w(n) WHERE n > 3",
    );
}

#[test]
fn test_view_where_and_select_list() {
    let db = setup("site_view");
    db.execute("CREATE VIEW v AS SELECT id, a FROM t", ())
        .unwrap();
    for (state, with_subquery, with_literal) in [
        (
            Right,
            "SELECT id FROM v WHERE id > (SELECT MAX(x) FROM u) ORDER BY id",
            "SELECT id FROM v WHERE id > 3 ORDER BY id",
        ),
        (
            Right,
            "SELECT id, (SELECT MAX(x) FROM u) FROM v ORDER BY id",
            "SELECT id, 3 FROM v ORDER BY id",
        ),
    ] {
        let want = rows(&db, with_literal).unwrap();
        holds(state, rows(&db, with_subquery), &want, with_subquery);
    }
}

#[test]
fn test_cte_block_reading_the_grandparent_row() {
    site(
        "site_cte_grandparent",
        Right,
        "WITH c AS (SELECT 1 AS x) SELECT id FROM t WHERE EXISTS (SELECT 1 FROM c WHERE c.x IN (SELECT u.x FROM u WHERE u.id = t.a)) ORDER BY id",
        "SELECT id FROM t WHERE a = 1 ORDER BY id",
    );
}

#[test]
fn test_table_function_argument() {
    // The grammar does not take a subquery argument yet
    site(
        "site_tvf_argument",
        Fails("step 6, grammar first"),
        "SELECT * FROM generate_series(1, (SELECT 3))",
        "SELECT * FROM generate_series(1, 3)",
    );
}

#[test]
fn test_where_over_a_source_outside_the_family() {
    let db = setup("site_source_outside_family");
    db.execute("CREATE VIEW tv AS SELECT id, a, b FROM t", ())
        .unwrap();
    let sources = [
        ("", "(SELECT * FROM t) AS s"),
        ("", "tv AS s"),
        ("WITH c AS (SELECT * FROM t) ", "c AS s"),
    ];
    // (WHERE with a subquery, the same with its answer written out)
    let wheres = [
        ("s.a IN (SELECT x FROM u)", "s.a IN (1, 2, 3)"),
        ("EXISTS (SELECT 1 FROM u WHERE u.x = 2)", "TRUE"),
        ("s.b > (SELECT MAX(x) FROM u) * 10", "s.b > 30"),
        ("s.a = ANY (SELECT x FROM u)", "s.a IN (1, 2, 3)"),
    ];
    // Each keeps the statement out of the prepared family
    let outside = [
        ("", " LIMIT 5 + 5"),
        (" AND (s.id, s.a) IN (SELECT id, a FROM t)", ""),
    ];
    for (with, from) in sources {
        for (with_subquery, with_literal) in wheres {
            for (also, tail) in outside {
                let query = |filter: &str| {
                    format!(
                        "{with}SELECT s.id FROM {from} WHERE {filter}{also} ORDER BY s.id{tail}"
                    )
                };
                let sql = query(with_subquery);
                let want = rows(&db, &query(with_literal)).unwrap();
                holds(Right, rows(&db, &sql), &want, &sql);
            }
        }
    }
}
