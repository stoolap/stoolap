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

/// t: id 1..6, a = id % 3, b = id * 10. u: id 1..3, x = id. vt is t
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 0, 30), (4, 1, 40), (5, 2, 50), (6, 0, 60)",
        "INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)",
        "CREATE VIEW vt AS SELECT id, a, b FROM t",
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
        Old("a statement without FROM"),
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
        Old("block ORDER BY and LIMIT"),
    ),
    (
        "tuple_in",
        "SELECT id FROM t WHERE (id, a) IN (SELECT id, x FROM u)",
        "SELECT id FROM t WHERE id IN (1, 2)",
        Old("a tuple on the left of IN"),
    ),
    (
        "distinct_where",
        "SELECT DISTINCT a FROM t WHERE id > (SELECT MAX(x) FROM u)",
        "SELECT DISTINCT a FROM t WHERE id > 3",
        Old("DISTINCT, set operations and windows"),
    ),
    (
        "union_where",
        "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u) UNION SELECT 9",
        "SELECT id FROM t WHERE id > 3 UNION SELECT 9",
        Old("DISTINCT, set operations and windows"),
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
        "operator_outside_the_family",
        "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u) AND a || '' = '1'",
        "SELECT id FROM t WHERE id > 3 AND a || '' = '1'",
        Old("operators outside the family"),
    ),
    (
        "join_with_a_cte_side",
        "WITH c AS (SELECT id, x FROM u) SELECT t.id FROM t JOIN c ON c.id = t.a \
         WHERE t.b > (SELECT MAX(x) FROM u)",
        "SELECT t.id FROM t JOIN u ON u.id = t.a WHERE t.b > 3",
        Old("plan-less and source-side joins"),
    ),
    (
        "block_over_a_derived_table",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT x FROM u) AS q WHERE q.x = t.a)",
        "SELECT id FROM t WHERE a IN (1, 2, 3)",
        Old("plan-less and source-side joins"),
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
        Old("DML refusals"),
    ),
    (
        "delete_with_an_operator_outside_the_family",
        "DELETE FROM t WHERE id > (SELECT MAX(x) FROM u) AND a || '' = '1'",
        "DELETE FROM t WHERE id > 3 AND a || '' = '1'",
        Old("DML refusals"),
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
