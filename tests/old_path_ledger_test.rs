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

//! The ledger of which path answers each statement shape with a subquery:
//! `Old`, a general path mechanism, until the named step prepares it, or
//! `Prepared`, none. Every shape answers as its value written as a literal.
//! A shape whose path changes without its entry changing fails here.

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{general_path_hits, GeneralPath};
use stoolap::Database;

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Which path answers a shape today
#[derive(Clone, Copy, Debug)]
enum Path {
    /// A general path mechanism, until the step named prepares it
    #[allow(dead_code)]
    Old(&'static str),
    /// No general path mechanism
    Prepared,
}

use Path::{Old, Prepared};

const ALL: [GeneralPath; 15] = [
    GeneralPath::CorrelatedScan,
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::PushdownConstant,
    GeneralPath::ProjectionEvaluator,
    GeneralPath::OrderByEvaluator,
    GeneralPath::AntiJoin,
    GeneralPath::ExistsSemiJoin,
    GeneralPath::ExistsSubquery,
    GeneralPath::InSubquery,
    GeneralPath::ScalarSubquery,
    GeneralPath::SelectFold,
    GeneralPath::CorrelatedExpression,
    GeneralPath::LiftedSubqueries,
];

fn hits() -> Vec<usize> {
    ALL.iter().map(|path| general_path_hits(*path)).collect()
}

/// The mechanisms that ran between `before` and now
fn ran(before: &[usize]) -> Vec<GeneralPath> {
    ALL.iter()
        .zip(before.iter().zip(hits()))
        .filter(|(_, (before, now))| now > *before)
        .map(|(path, _)| *path)
        .collect()
}

/// t: id 1..6, a = id % 3, b = id * 10. u: id 1..3, x = id. vt is t, vj a
/// join of them with a subquery
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 0, 30), (4, 1, 40), (5, 2, 50), (6, 0, 60)",
        "INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)",
        "CREATE VIEW vt AS SELECT id, a, b FROM t",
        "CREATE VIEW vj AS SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The rows in the order they came when `ordered`, else sorted: a shape
/// without ORDER BY has no order to compare
fn rows(db: &Database, sql: &str, ordered: bool) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect()
        })
        .collect();
    if !ordered {
        out.sort();
    }
    out
}

fn check(name: &str, path: Path, ran: Vec<GeneralPath>, wrong: &mut Vec<String>) {
    match (path, ran.is_empty()) {
        (Old(_), false) | (Prepared, true) => {}
        (Old(owner), true) => wrong.push(format!(
            "{name} ran no general path mechanism: mark it Prepared, {owner} moved it"
        )),
        (Prepared, false) => wrong.push(format!("{name} is marked Prepared but ran {ran:?}")),
    }
}

/// Each query shape, the same statement with its subqueries' values written
/// as literals, and its path
const QUERIES: &[(&str, &str, &str, Path)] = &[
    (
        "where_scalar",
        "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u)",
        "SELECT id FROM t WHERE id > 3",
        Prepared,
    ),
    (
        "where_correlated_exists",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.x = t.a)",
        "SELECT id FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "grouped_select_list",
        "SELECT a, (SELECT MAX(x) FROM u) FROM t GROUP BY a",
        "SELECT a, 3 FROM t GROUP BY a",
        Prepared,
    ),
    (
        "join_where",
        "SELECT t.id FROM t JOIN u ON u.id = t.a WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN u ON u.id = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "select_list_scalar",
        "SELECT id, (SELECT MAX(x) FROM u) FROM t",
        "SELECT id, 3 FROM t",
        Prepared,
    ),
    (
        "select_list_correlated",
        "SELECT id, (SELECT COUNT(*) FROM u WHERE u.id = t.a) FROM t",
        "SELECT id, CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END FROM t",
        Prepared,
    ),
    (
        "select_list_correlated_exists",
        "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.id = t.a) FROM t",
        "SELECT id, a BETWEEN 1 AND 3 FROM t",
        Prepared,
    ),
    (
        "order_by_scalar",
        "SELECT id FROM t ORDER BY (SELECT MAX(x) FROM u), id",
        "SELECT id FROM t ORDER BY id",
        Prepared,
    ),
    (
        "order_by_correlated",
        "SELECT id FROM t ORDER BY (SELECT COUNT(*) FROM u WHERE u.id = t.a), id",
        "SELECT id FROM t ORDER BY CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END, id",
        Prepared,
    ),
    (
        "where_and_select_list",
        "SELECT id, (SELECT MAX(x) FROM u) FROM t WHERE a IN (SELECT x FROM u)",
        "SELECT id, 3 FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "fromless",
        "SELECT (SELECT MAX(x) FROM u)",
        "SELECT 3",
        Prepared,
    ),
    (
        "fromless_where_exists",
        "SELECT 1 WHERE EXISTS (SELECT 1 FROM u WHERE x = 2)",
        "SELECT 1",
        Prepared,
    ),
    (
        "fromless_distinct_ordered",
        "SELECT DISTINCT (SELECT MAX(x) FROM u) ORDER BY 1 LIMIT 1",
        "SELECT DISTINCT 3 ORDER BY 1 LIMIT 1",
        Prepared,
    ),
    (
        "fromless_aggregate",
        "SELECT COUNT(*), (SELECT MAX(x) FROM u)",
        "SELECT COUNT(*), 3",
        Prepared,
    ),
    (
        "fromless_having",
        "SELECT COUNT(*) HAVING (SELECT MAX(x) FROM u) > 2",
        "SELECT COUNT(*) HAVING 3 > 2",
        Prepared,
    ),
    (
        "fromless_window",
        "SELECT ROW_NUMBER() OVER (), (SELECT MAX(x) FROM u)",
        "SELECT ROW_NUMBER() OVER (), 3",
        Prepared,
    ),
    (
        "fromless_block_reading_the_row",
        "SELECT id FROM t WHERE (SELECT t.a) = 1",
        "SELECT id FROM t WHERE a = 1",
        Prepared,
    ),
    (
        "aggregate_where",
        "SELECT COUNT(*) FROM t WHERE a IN (SELECT x FROM u)",
        "SELECT COUNT(*) FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "group_by_where",
        "SELECT a, COUNT(*) FROM t WHERE id > (SELECT MAX(x) FROM u) GROUP BY a",
        "SELECT a, COUNT(*) FROM t WHERE id > 3 GROUP BY a",
        Prepared,
    ),
    (
        "aggregate_argument",
        "SELECT a, SUM((SELECT MAX(x) FROM u WHERE u.x = t.a)) FROM t GROUP BY a",
        "SELECT a, SUM(CASE WHEN a BETWEEN 1 AND 3 THEN a END) FROM t GROUP BY a",
        Prepared,
    ),
    (
        "having_correlated_over_a_view",
        "SELECT s.a, COUNT(*) FROM vt AS s GROUP BY s.a \
         HAVING COUNT(*) > (SELECT COUNT(*) FROM u WHERE u.x = s.a)",
        "SELECT a, COUNT(*) FROM t GROUP BY a \
         HAVING COUNT(*) > CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END",
        Prepared,
    ),
    (
        "grouped_over_a_derived_table",
        "SELECT d.a, SUM((SELECT MAX(x) FROM u WHERE u.x = d.a)) FROM (SELECT a FROM t) d \
         GROUP BY d.a",
        "SELECT a, SUM(CASE WHEN a BETWEEN 1 AND 3 THEN a END) FROM t GROUP BY a",
        Prepared,
    ),
    (
        "grouped_over_a_join",
        "SELECT t.a, (SELECT COUNT(*) FROM u WHERE u.x = t.a) FROM t JOIN u v ON v.id = t.id \
         GROUP BY t.a",
        "SELECT t.a, CASE WHEN t.a BETWEEN 1 AND 3 THEN 1 ELSE 0 END FROM t \
         JOIN u v ON v.id = t.id GROUP BY t.a",
        Prepared,
    ),
    (
        "grouped_rollup_and_distinct",
        "SELECT DISTINCT a, (SELECT MAX(x) FROM u) FROM t GROUP BY ROLLUP(a)",
        "SELECT DISTINCT a, 3 FROM t GROUP BY ROLLUP(a)",
        Prepared,
    ),
    (
        "grouped_set_operation_branch",
        "SELECT a, (SELECT MAX(x) FROM u) FROM t GROUP BY a UNION ALL SELECT -1, 0",
        "SELECT a, 3 FROM t GROUP BY a UNION ALL SELECT -1, 0",
        Prepared,
    ),
    (
        "block_order_by_limit",
        "SELECT id FROM t WHERE a = (SELECT x FROM u WHERE u.x >= t.a ORDER BY x LIMIT 1)",
        "SELECT id FROM t WHERE a IN (1, 2)",
        Prepared,
    ),
    (
        "tuple_in",
        "SELECT id FROM t WHERE (id, a) IN (SELECT id, x FROM u)",
        "SELECT id FROM t WHERE id IN (1, 2)",
        Prepared,
    ),
    (
        "distinct_where",
        "SELECT DISTINCT a FROM t WHERE id > (SELECT MAX(x) FROM u)",
        "SELECT DISTINCT a FROM t WHERE id > 3",
        Prepared,
    ),
    (
        "union_where",
        "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u) UNION SELECT 9",
        "SELECT id FROM t WHERE id > 3 UNION SELECT 9",
        Prepared,
    ),
    (
        "union_order_by",
        "SELECT id FROM t UNION SELECT 9 ORDER BY (SELECT MAX(x) FROM u) - id LIMIT 2",
        "SELECT id FROM t UNION SELECT 9 ORDER BY 3 - id LIMIT 2",
        Prepared,
    ),
    (
        "window_where",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM t WHERE id > (SELECT MAX(x) FROM u)",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM t WHERE id > 3",
        Prepared,
    ),
    (
        "window_select",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id), (SELECT COUNT(*) FROM u WHERE u.id = t.a) \
         FROM t",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id), CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END \
         FROM t",
        Prepared,
    ),
    (
        "named_window_select",
        "SELECT id, ROW_NUMBER() OVER w, (SELECT COUNT(*) FROM u WHERE u.id = t.a) \
         FROM t WINDOW w AS (ORDER BY id)",
        "SELECT id, ROW_NUMBER() OVER w, CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END \
         FROM t WINDOW w AS (ORDER BY id)",
        Prepared,
    ),
    (
        "named_window_key",
        "SELECT id, SUM(b) OVER w FROM t WINDOW w AS (PARTITION BY a > (SELECT MIN(x) FROM u))",
        "SELECT id, SUM(b) OVER w FROM t WINDOW w AS (PARTITION BY a > 1)",
        Prepared,
    ),
    (
        "named_window_having",
        "SELECT a, SUM(b), RANK() OVER w FROM t GROUP BY a \
         HAVING SUM(b) > (SELECT MAX(x) FROM u) * 20 WINDOW w AS (ORDER BY a)",
        "SELECT a, SUM(b), RANK() OVER w FROM t GROUP BY a HAVING SUM(b) > 60 \
         WINDOW w AS (ORDER BY a)",
        Prepared,
    ),
    (
        "grouped_order_by",
        "SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY (SELECT COUNT(*) FROM u WHERE u.id = t.a), a",
        "SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END, a",
        Prepared,
    ),
    (
        "grouped_distinct_on",
        "SELECT DISTINCT ON ((SELECT COUNT(*) FROM u WHERE u.id = t.a)) a FROM t GROUP BY a \
         ORDER BY a",
        "SELECT DISTINCT ON (CASE WHEN a BETWEEN 1 AND 3 THEN 1 ELSE 0 END) a FROM t GROUP BY a \
         ORDER BY a",
        Prepared,
    ),
    (
        "concatenation_beside_a_subquery",
        "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u) AND a || '' = '1'",
        "SELECT id FROM t WHERE id > 3 AND a || '' = '1'",
        Prepared,
    ),
    (
        "join_with_a_cte_side",
        "WITH c AS (SELECT id, x FROM u) SELECT t.id FROM t JOIN c ON c.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN u ON u.id = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_with_a_cte_side_its_limit_cuts",
        "WITH c (a, n) AS (SELECT a, COUNT(*) FROM t GROUP BY a) \
         SELECT c.a, u.x FROM c LEFT JOIN u ON u.id = c.a AND u.x < (SELECT MAX(x) FROM u) \
         ORDER BY c.a LIMIT 2",
        "WITH c (a, n) AS (SELECT a, COUNT(*) FROM t GROUP BY a) \
         SELECT c.a, u.x FROM c LEFT JOIN u ON u.id = c.a AND u.x < 3 ORDER BY c.a LIMIT 2",
        Prepared,
    ),
    (
        "join_inside_a_cte_its_limit_cuts",
        "WITH c (a, n) AS (SELECT t.a, COUNT(*) FROM t JOIN u ON u.id = t.id \
         WHERE t.b > (SELECT MIN(x) FROM u) GROUP BY t.a) \
         SELECT c.a, u.x FROM c LEFT JOIN u ON u.id = c.a AND u.x < (SELECT MAX(x) FROM u) \
         ORDER BY c.a LIMIT 2",
        "WITH c (a, n) AS (SELECT t.a, COUNT(*) FROM t JOIN u ON u.id = t.id \
         WHERE t.b > 1 GROUP BY t.a) \
         SELECT c.a, u.x FROM c LEFT JOIN u ON u.id = c.a AND u.x < 3 ORDER BY c.a LIMIT 2",
        Prepared,
    ),
    (
        "join_with_a_subquery_in_its_select_list",
        "SELECT t.id, (SELECT MAX(x) FROM u) FROM t JOIN u v ON v.id = t.a",
        "SELECT t.id, 3 FROM t JOIN u v ON v.id = t.a",
        Prepared,
    ),
    (
        "join_with_a_view_side",
        "SELECT t.id FROM t JOIN vt ON vt.id = t.a WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN vt ON vt.id = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_with_a_derived_side",
        "SELECT t.id FROM t JOIN (SELECT id, x FROM u) AS d ON d.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN (SELECT id, x FROM u) AS d ON d.id = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_with_a_values_side",
        "SELECT t.id FROM t JOIN (VALUES (1, 1), (2, 2)) AS v (id, x) ON v.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN (VALUES (1, 1), (2, 2)) AS v (id, x) ON v.id = t.a \
         WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_with_a_table_function_side",
        "SELECT t.id FROM t JOIN generate_series(1, 2) AS g (n) ON g.n = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN generate_series(1, 2) AS g (n) ON g.n = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_of_three_tables",
        "SELECT t.id FROM t JOIN u ON u.id = t.a JOIN u AS w ON w.id = u.id \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN u ON u.id = t.a JOIN u AS w ON w.id = u.id WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_with_a_joined_using_side",
        "SELECT t.id FROM t JOIN u USING (id) JOIN u AS w ON w.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN u USING (id) JOIN u AS w ON w.id = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_with_a_joined_values_side",
        "SELECT t.id FROM t JOIN (VALUES (1), (2)) AS v (n) ON v.n = t.a JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN (VALUES (1), (2)) AS v (n) ON v.n = t.a JOIN u ON u.id = t.a \
         WHERE t.b > 3",
        Prepared,
    ),
    (
        "block_over_a_derived_table",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = t.a)",
        "SELECT id FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "block_over_a_correlated_derived_table",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u WHERE u.x = t.a) AS q)",
        "SELECT id FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "block_over_a_derived_join",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT u.x FROM u JOIN u AS w ON w.id = u.id) \
         AS q WHERE q.x = t.a)",
        "SELECT id FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "block_over_a_join",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u JOIN u AS w ON w.id = u.id \
         WHERE u.x = t.a)",
        "SELECT id FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "join_inside_a_derived_table",
        "SELECT * FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) AS d",
        "SELECT * FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) AS d",
        Prepared,
    ),
    (
        "join_inside_a_derived_table_of_a_prepared_statement",
        "SELECT d.id FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) AS d WHERE d.id > (SELECT MIN(x) FROM u)",
        "SELECT d.id FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) AS d \
         WHERE d.id > 1",
        Prepared,
    ),
    (
        "join_inside_a_view",
        "SELECT * FROM vj",
        "SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_inside_a_view_of_a_prepared_statement",
        "SELECT vj.id FROM vj WHERE vj.id > (SELECT MIN(x) FROM u)",
        "SELECT t.id FROM t JOIN u ON u.id = t.a WHERE t.b > 3 AND t.id > 1",
        Prepared,
    ),
    (
        "block_over_a_view_of_a_join",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM vj WHERE vj.id = t.id)",
        "SELECT id FROM t WHERE a IN (1, 2)",
        Prepared,
    ),
    (
        "join_inside_a_cte",
        "WITH c AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) SELECT * FROM c",
        "WITH c AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) SELECT * FROM c",
        Prepared,
    ),
    (
        "join_inside_a_cte_of_a_prepared_statement",
        "WITH c AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) SELECT c.id FROM c WHERE c.id > (SELECT MIN(x) FROM u)",
        "WITH c AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) \
         SELECT c.id FROM c WHERE c.id > 1",
        Prepared,
    ),
    (
        "join_in_a_set_operation_branch",
        "SELECT x FROM u UNION SELECT t.id FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT x FROM u UNION SELECT t.id FROM t JOIN u ON u.id = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "join_with_a_derived_side_that_is_a_join",
        "SELECT t.id FROM t JOIN (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) AS d ON d.id = t.id WHERE t.b > (SELECT MIN(x) FROM u)",
        "SELECT t.id FROM t JOIN (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) AS d \
         ON d.id = t.id WHERE t.b > 1",
        Prepared,
    ),
    (
        "join_as_the_first_member_of_a_recursive_cte",
        "WITH RECURSIVE r(id, x) AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u) UNION ALL SELECT id + 10, x FROM r WHERE id < 10) \
         SELECT r.id FROM r WHERE r.id > (SELECT MIN(x) FROM u)",
        "WITH RECURSIVE r(id, x) AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3 \
         UNION ALL SELECT id + 10, x FROM r WHERE id < 10) SELECT r.id FROM r WHERE r.id > 1",
        Prepared,
    ),
    (
        "join_as_the_recursive_member_of_a_recursive_cte",
        "WITH RECURSIVE r(id, x) AS (SELECT id, x FROM u WHERE id = 1 UNION ALL \
         SELECT u.id, u.x FROM r JOIN u ON u.id = r.id + 1 WHERE u.x > (SELECT MIN(x) FROM u)) \
         SELECT r.id, r.x FROM r WHERE r.id > (SELECT MIN(x) FROM u) - 1",
        "WITH RECURSIVE r(id, x) AS (SELECT id, x FROM u WHERE id = 1 UNION ALL \
         SELECT u.id, u.x FROM r JOIN u ON u.id = r.id + 1 WHERE u.x > 1) \
         SELECT r.id, r.x FROM r WHERE r.id > 0",
        Prepared,
    ),
    (
        "join_inside_a_derived_table_of_an_aggregate",
        "SELECT COUNT(*) FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) AS d",
        "SELECT COUNT(*) FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) AS d",
        Prepared,
    ),
    (
        "join_inside_a_derived_table_of_a_distinct",
        "SELECT DISTINCT d.x FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) AS d",
        "SELECT DISTINCT d.x FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) AS d",
        Prepared,
    ),
    (
        "join_inside_a_derived_join_side",
        "SELECT t.id FROM t JOIN (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) AS d ON d.id = t.id",
        "SELECT t.id FROM t JOIN (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) AS d \
         ON d.id = t.id",
        Prepared,
    ),
    (
        "join_inside_a_derived_table_in_a_joined_side",
        "SELECT d.id FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) AS d JOIN t ON t.id = d.id JOIN u ON u.id = t.a",
        "SELECT d.id FROM (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) AS d \
         JOIN t ON t.id = d.id JOIN u ON u.id = t.a",
        Prepared,
    ),
    (
        "join_inside_a_cte_read_twice",
        "WITH c AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) SELECT c.id FROM c JOIN c AS e ON e.id = c.id",
        "WITH c AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > 3) \
         SELECT c.id FROM c JOIN c AS e ON e.id = c.id",
        Prepared,
    ),
    (
        "join_inside_a_derived_table_of_a_cte_cut_by_limit",
        "WITH c AS (SELECT d.id, COUNT(*) AS n FROM (SELECT t.id FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) AS d GROUP BY d.id) \
         SELECT c.id FROM c LEFT JOIN u ON u.id = c.id LIMIT 1",
        "WITH c AS (SELECT d.id, COUNT(*) AS n FROM (SELECT t.id FROM t JOIN u ON u.id = t.a \
         WHERE t.b > 3) AS d GROUP BY d.id) \
         SELECT c.id FROM c LEFT JOIN u ON u.id = c.id LIMIT 1",
        Prepared,
    ),
    (
        "join_inside_a_union_branch_of_a_cte_cut_by_limit",
        "WITH c AS (SELECT d.id, COUNT(*) AS n FROM (SELECT id FROM t WHERE id < 0 UNION ALL \
         SELECT t.id FROM t JOIN u ON u.id = t.a WHERE t.b > (SELECT MAX(x) FROM u)) AS d \
         GROUP BY d.id) SELECT c.id FROM c LEFT JOIN u ON u.id = c.id LIMIT 1",
        "WITH c AS (SELECT d.id, COUNT(*) AS n FROM (SELECT id FROM t WHERE id < 0 UNION ALL \
         SELECT t.id FROM t JOIN u ON u.id = t.a WHERE t.b > 3) AS d \
         GROUP BY d.id) SELECT c.id FROM c LEFT JOIN u ON u.id = c.id LIMIT 1",
        Prepared,
    ),
    (
        "in_block_with_distinct",
        "SELECT id FROM t WHERE a IN (SELECT DISTINCT x FROM u)",
        "SELECT id FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "in_block_with_union",
        "SELECT id FROM t WHERE a IN (SELECT x FROM u WHERE x < 2 UNION SELECT x FROM u WHERE x > 2)",
        "SELECT id FROM t WHERE a IN (1, 3)",
        Prepared,
    ),
    (
        "not_in_block_with_except",
        "SELECT id FROM t WHERE a NOT IN (SELECT x FROM u EXCEPT SELECT 1)",
        "SELECT id FROM t WHERE a NOT IN (2, 3)",
        Prepared,
    ),
    (
        "scalar_block_with_intersect",
        "SELECT id, (SELECT x FROM u INTERSECT SELECT 2) FROM t",
        "SELECT id, 2 FROM t",
        Prepared,
    ),
    (
        "scalar_block_with_grouping_sets",
        "SELECT id, (SELECT COUNT(*) FROM u GROUP BY GROUPING SETS ((x), ()) \
         ORDER BY 1 DESC LIMIT 1) FROM t",
        "SELECT id, 3 FROM t",
        Prepared,
    ),
    (
        "scalar_block_with_window_clause",
        "SELECT id, (SELECT SUM(x) OVER w FROM u WINDOW w AS (ORDER BY id) \
         ORDER BY 1 DESC LIMIT 1) FROM t",
        "SELECT id, 6 FROM t",
        Prepared,
    ),
    (
        "correlated_exists_block_with_union",
        "SELECT id FROM t WHERE EXISTS (SELECT x FROM u WHERE x = t.a \
         UNION SELECT x FROM u WHERE x = t.a)",
        "SELECT id FROM t WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "correlated_exists_block_with_union_over_a_computed_key",
        "SELECT id FROM t WHERE EXISTS (SELECT x FROM u WHERE x = t.a \
         UNION SELECT x FROM u WHERE x = t.a + 2)",
        "SELECT id FROM t WHERE a IN (1, 2, 3) OR a + 2 IN (1, 2, 3)",
        Old("7b Phase 3: a statement block's own statement runs on the general executor"),
    ),
    (
        "scalar_grouped_block_with_order_and_limit",
        "SELECT id, (SELECT COUNT(*) FROM u GROUP BY x ORDER BY COUNT(*) DESC, x LIMIT 1) FROM t",
        "SELECT id, 1 FROM t",
        Prepared,
    ),
    (
        "in_block_with_a_computed_limit",
        "SELECT id FROM t WHERE a IN (SELECT x FROM u ORDER BY x DESC LIMIT 1 + 0)",
        "SELECT id FROM t WHERE a IN (3)",
        Prepared,
    ),
    (
        "order_by_subquery_reading_an_alias",
        "SELECT b AS z FROM t ORDER BY (SELECT COUNT(*) FROM u WHERE u.x * 20 <= z) DESC, z",
        "SELECT b AS z FROM t ORDER BY CASE WHEN b >= 60 THEN 3 WHEN b >= 40 THEN 2 \
         WHEN b >= 20 THEN 1 ELSE 0 END DESC, z",
        Prepared,
    ),
    (
        "join_order_by_subquery_reading_an_alias",
        "SELECT t.b AS z, u.x FROM t JOIN u ON u.id = t.a ORDER BY (SELECT z) DESC",
        "SELECT t.b AS z, u.x FROM t JOIN u ON u.id = t.a ORDER BY z DESC",
        Prepared,
    ),
    (
        "grouped_order_by_subquery_reading_an_alias",
        "SELECT a, MAX(b) AS z FROM t GROUP BY a ORDER BY (SELECT z) DESC",
        "SELECT a, MAX(b) AS z FROM t GROUP BY a ORDER BY z DESC",
        Prepared,
    ),
];

/// The query shapes whose statement orders its rows, compared in that order
const ORDERED: &[&str] = &[
    "order_by_scalar",
    "order_by_correlated",
    "grouped_order_by",
    "grouped_distinct_on",
];

/// Each write shape, the same write with literals, and its path
const WRITES: &[(&str, &str, &str, Path)] = &[
    (
        "update_set",
        "UPDATE t SET b = (SELECT MAX(x) FROM u) WHERE id = 1",
        "UPDATE t SET b = 3 WHERE id = 1",
        Prepared,
    ),
    (
        "update_where_over_a_derived_table",
        "UPDATE t SET b = 0 WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = t.a)",
        "UPDATE t SET b = 0 WHERE a IN (1, 2, 3)",
        Prepared,
    ),
    (
        "delete_with_concatenation_beside_a_subquery",
        "DELETE FROM t WHERE id > (SELECT MAX(x) FROM u) AND a || '' = '1'",
        "DELETE FROM t WHERE id > 3 AND a || '' = '1'",
        Prepared,
    ),
    (
        "insert_select_of_a_join",
        "INSERT INTO t SELECT t.id + 10, u.x, t.b FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "INSERT INTO t SELECT t.id + 10, u.x, t.b FROM t JOIN u ON u.id = t.a WHERE t.b > 3",
        Prepared,
    ),
    (
        "update_with_a_cte_of_a_join",
        "WITH c AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) UPDATE t SET b = 0 WHERE id IN (SELECT id FROM c)",
        "UPDATE t SET b = 0 WHERE id IN (1, 2, 4, 5)",
        Prepared,
    ),
    (
        "delete_with_a_cte_of_a_join",
        "WITH c AS (SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)) DELETE FROM t WHERE id IN (SELECT id FROM c) AND a = 2",
        "DELETE FROM t WHERE id IN (2, 5)",
        Prepared,
    ),
];

#[test]
fn every_query_shape_takes_the_path_its_entry_names() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for (name, sql, literal, path) in QUERIES {
        let db = setup(&format!("old_path_ledger_{name}"));
        let ordered = ORDERED.contains(name);
        let want = rows(&db, literal, ordered);
        let before = hits();
        let got = rows(&db, sql, ordered);
        let ran = ran(&before);
        if got != want {
            wrong.push(format!("{name}: {got:?}, want {want:?}"));
        }
        check(name, *path, ran, &mut wrong);
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn every_write_shape_takes_the_path_its_entry_names() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for (name, sql, literal, path) in WRITES {
        let left = setup(&format!("old_path_ledger_{name}_subquery"));
        let right = setup(&format!("old_path_ledger_{name}_literal"));
        right.execute(literal, ()).unwrap();
        let before = hits();
        left.execute(sql, ())
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let ran = ran(&before);
        let (got, want) = (
            rows(&left, "SELECT * FROM t ORDER BY id", true),
            rows(&right, "SELECT * FROM t ORDER BY id", true),
        );
        if got != want {
            wrong.push(format!("{name}: {got:?}, want {want:?}"));
        }
        check(name, *path, ran, &mut wrong);
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
