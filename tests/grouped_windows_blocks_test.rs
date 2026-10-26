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

//! A statement with window functions, or a grouped one with an ORDER BY or
//! a DISTINCT ON holding subqueries, runs them through prepared blocks over
//! any source: it answers as its twin written without them

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{general_path_hits, GeneralPath};
use stoolap::Database;

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// t: id 1..200, a = id % 9, b = id % 13; u: id 1..40, k = id % 11,
/// x = id % 7; w: id 1..200; v and c read t as it is
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, z INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 9, g.value % 13 FROM generate_series(1, 200) g",
        "INSERT INTO u SELECT g.value, g.value % 11, g.value % 7 FROM generate_series(1, 40) g",
        "INSERT INTO w SELECT g.value, g.value % 4 FROM generate_series(1, 200) g",
        "CREATE VIEW v AS SELECT id, a, b FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// t's rows written as a VALUES list
fn values_source() -> String {
    let rows: Vec<String> = (1..=200)
        .map(|id| format!("({id}, {}, {})", id % 9, id % 13))
        .collect();
    format!("(VALUES {}) t(id, a, b)", rows.join(", "))
}

/// Each source reading t's rows under the name t, and the WITH it needs
fn sources() -> Vec<(&'static str, String, &'static str)> {
    vec![
        ("table", "t".into(), ""),
        ("derived", "(SELECT id, a, b FROM t) t".into(), ""),
        ("view", "v t".into(), ""),
        ("cte", "c t".into(), "WITH c AS (SELECT id, a, b FROM t) "),
        ("values", values_source(), ""),
        ("join", "t JOIN w ON w.id = t.id".into(), ""),
        (
            "left join",
            "t LEFT JOIN w ON w.id = t.id AND w.z < 0".into(),
            "",
        ),
    ]
}

/// How many rows of u have k = t.a, for a in 0..8
const COUNT_BY_A: &str = "CASE WHEN t.a = 0 OR t.a = 8 THEN 3 ELSE 4 END";

/// Each statement over `{t}`, and its twin over t written without subqueries
fn shapes() -> Vec<(String, String)> {
    let shape = |sql: &str, twin: &str| (sql.to_string(), twin.replace("{count}", COUNT_BY_A));
    vec![
        // Windows over the rows
        shape(
            "SELECT t.id, ROW_NUMBER() OVER (ORDER BY t.id), (SELECT MAX(u.x) FROM u) FROM {t} \
             ORDER BY t.id",
            "SELECT t.id, ROW_NUMBER() OVER (ORDER BY t.id), 6 FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, ROW_NUMBER() OVER (ORDER BY t.id), \
             (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} ORDER BY t.id",
            "SELECT t.id, ROW_NUMBER() OVER (ORDER BY t.id), {count} FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, SUM(t.b) OVER (PARTITION BY t.a > (SELECT MIN(u.x) FROM u) + 3) \
             FROM {t} ORDER BY t.id",
            "SELECT t.id, SUM(t.b) OVER (PARTITION BY t.a > 3) FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, SUM((SELECT COUNT(*) FROM u WHERE u.k = t.a)) OVER (ORDER BY t.id) \
             FROM {t} ORDER BY t.id",
            "SELECT t.id, SUM({count}) OVER (ORDER BY t.id) FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, ROW_NUMBER() OVER (ORDER BY (SELECT COUNT(*) FROM u WHERE u.k = t.a), \
             t.id) FROM {t} ORDER BY t.id",
            "SELECT t.id, ROW_NUMBER() OVER (ORDER BY {count}, t.id) FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, FIRST_VALUE((SELECT COUNT(*) FROM u WHERE u.k = t.a)) \
             OVER (PARTITION BY t.b ORDER BY t.id) FROM {t} ORDER BY t.id",
            "SELECT t.id, FIRST_VALUE(t.c) OVER (PARTITION BY t.b ORDER BY t.id) \
             FROM (SELECT id, b, {count} AS c FROM t) t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, LAG((SELECT COUNT(*) FROM u WHERE u.k = t.a), 2, -1) \
             OVER (ORDER BY t.id) FROM {t} ORDER BY t.id",
            "SELECT t.id, LAG(t.c, 2, -1) OVER (ORDER BY t.id) \
             FROM (SELECT id, {count} AS c FROM t) t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, ROW_NUMBER() OVER (ORDER BY t.id) FROM {t} \
             WHERE t.a IN (SELECT u.k FROM u WHERE u.x > 3) ORDER BY t.id",
            "SELECT t.id, ROW_NUMBER() OVER (ORDER BY t.id) FROM t \
             WHERE t.a IN (4, 5, 6, 8, 1, 2, 3, 7, 0) ORDER BY t.id",
        ),
        // Windows over groups
        shape(
            "SELECT t.a, SUM(t.b), RANK() OVER (ORDER BY t.a) FROM {t} GROUP BY t.a \
             HAVING SUM(t.b) > (SELECT MAX(u.x) FROM u) * 21 ORDER BY t.a",
            "SELECT t.a, SUM(t.b), RANK() OVER (ORDER BY t.a) FROM t GROUP BY t.a \
             HAVING SUM(t.b) > 126 ORDER BY t.a",
        ),
        shape(
            "SELECT t.a, RANK() OVER (ORDER BY t.a), (SELECT COUNT(*) FROM u WHERE u.k = t.a) \
             FROM {t} GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, RANK() OVER (ORDER BY t.a), {count} FROM t GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT t.a, SUM((SELECT COUNT(*) FROM u WHERE u.k = t.a)), RANK() OVER (ORDER BY t.a) \
             FROM {t} GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, SUM({count}), RANK() OVER (ORDER BY t.a) FROM t GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT t.a, SUM(SUM(t.b)) OVER (ORDER BY t.a) + (SELECT MAX(u.x) FROM u) FROM {t} \
             GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, SUM(SUM(t.b)) OVER (ORDER BY t.a) + 6 FROM t GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT t.a, RANK() OVER (PARTITION BY t.a > (SELECT MIN(u.x) FROM u) + 3 \
             ORDER BY t.a) FROM {t} GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, RANK() OVER (PARTITION BY t.a > 3 ORDER BY t.a) FROM t GROUP BY t.a \
             ORDER BY t.a",
        ),
        // A grouped ORDER BY
        shape(
            "SELECT t.a, COUNT(*) FROM {t} GROUP BY t.a ORDER BY t.a * (SELECT MIN(u.x) - 1 FROM u)",
            "SELECT t.a, COUNT(*) FROM t GROUP BY t.a ORDER BY t.a * -1",
        ),
        shape(
            "SELECT t.a, COUNT(*) FROM {t} GROUP BY t.a \
             ORDER BY (SELECT COUNT(*) FROM u WHERE u.k = t.a), t.a",
            "SELECT t.a, COUNT(*) FROM t GROUP BY t.a ORDER BY {count}, t.a",
        ),
        // A WHERE holding the only subqueries
        shape(
            "SELECT COUNT(*) FROM {t} WHERE t.a IN (SELECT u.x FROM u)",
            "SELECT COUNT(*) FROM t WHERE t.a IN (0, 1, 2, 3, 4, 5, 6)",
        ),
        shape(
            "SELECT t.a, SUM(t.b) FROM {t} WHERE t.id > (SELECT MAX(u.x) FROM u) * 20 \
             GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, SUM(t.b) FROM t WHERE t.id > 120 GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT t.a, COUNT(*) FROM {t} WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.a \
             AND u.x > 4) GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, COUNT(*) FROM t WHERE t.a IN (0, 1, 2, 4, 5, 6, 7, 8) GROUP BY t.a \
             ORDER BY t.a",
        ),
        // An argument read once for its call
        shape(
            "SELECT t.id, NTILE((SELECT MIN(u.x) FROM u) + 3) OVER (ORDER BY t.id) FROM {t} \
             ORDER BY t.id",
            "SELECT t.id, NTILE(3) OVER (ORDER BY t.id) FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, LAG(t.b, (SELECT MIN(u.x) + 2 FROM u), (SELECT MAX(u.x) FROM u)) \
             OVER (ORDER BY t.id) FROM {t} ORDER BY t.id",
            "SELECT t.id, LAG(t.b, 2, 6) OVER (ORDER BY t.id) FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.id, SUM(t.b) OVER (ORDER BY t.id ROWS BETWEEN (SELECT MIN(u.x) + 2 FROM u) \
             PRECEDING AND CURRENT ROW) FROM {t} ORDER BY t.id",
            "SELECT t.id, SUM(t.b) OVER (ORDER BY t.id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) \
             FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.a, STRING_AGG(CAST(t.b AS TEXT), CAST((SELECT MIN(u.x) FROM u) AS TEXT) \
             ORDER BY t.b, t.id) FROM {t} GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, STRING_AGG(CAST(t.b AS TEXT), '0' ORDER BY t.b, t.id) FROM t \
             GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT t.a, NTILE((SELECT MIN(u.x) FROM u) + 2) OVER (ORDER BY t.a) FROM {t} \
             GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, NTILE(2) OVER (ORDER BY t.a) FROM t GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT t.id, NTILE((SELECT MIN(u.x) FROM u) + 3) OVER (ORDER BY t.id), \
             (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} ORDER BY t.id",
            "SELECT t.id, NTILE(3) OVER (ORDER BY t.id), {count} FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT t.a, STRING_AGG(CAST(t.b AS TEXT), CAST((SELECT MIN(u.x) FROM u) AS TEXT) \
             ORDER BY t.b, t.id), (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} \
             GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, STRING_AGG(CAST(t.b AS TEXT), '0' ORDER BY t.b, t.id), {count} FROM t \
             GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT t.a, (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} GROUP BY t.a \
             ORDER BY STRING_AGG(CAST(t.b AS TEXT), CAST((SELECT MIN(u.x) FROM u) AS TEXT) \
             ORDER BY t.b, t.id), t.a",
            "SELECT t.a, {count} FROM t GROUP BY t.a \
             ORDER BY STRING_AGG(CAST(t.b AS TEXT), '0' ORDER BY t.b, t.id), t.a",
        ),
        // A DISTINCT ON
        shape(
            "SELECT DISTINCT ON (t.a % 2 + (SELECT MIN(u.x) FROM u)) t.a, COUNT(*) FROM {t} \
             GROUP BY t.a ORDER BY t.a",
            "SELECT DISTINCT ON (t.a % 2) t.a, COUNT(*) FROM t GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT DISTINCT ON ((SELECT COUNT(*) FROM u WHERE u.k = t.a)) t.a FROM {t} \
             GROUP BY t.a ORDER BY t.a",
            "SELECT DISTINCT ON ({count}) t.a FROM t GROUP BY t.a ORDER BY t.a",
        ),
        shape(
            "SELECT DISTINCT ON ((SELECT COUNT(*) FROM u WHERE u.k = t.a)) t.id, \
             ROW_NUMBER() OVER (ORDER BY t.id) FROM {t} ORDER BY t.id",
            "SELECT DISTINCT ON ({count}) t.id, ROW_NUMBER() OVER (ORDER BY t.id) FROM t \
             ORDER BY t.id",
        ),
        shape(
            "SELECT DISTINCT ON (ROW_NUMBER() OVER (ORDER BY t.id) % 3 + (SELECT MAX(u.x) FROM u)) \
             t.id, ROW_NUMBER() OVER (ORDER BY t.id) % 3 + (SELECT MAX(u.x) FROM u) AS k \
             FROM {t} ORDER BY t.id",
            "SELECT DISTINCT ON (ROW_NUMBER() OVER (ORDER BY t.id) % 3 + 6) t.id, \
             ROW_NUMBER() OVER (ORDER BY t.id) % 3 + 6 AS k FROM t ORDER BY t.id",
        ),
        shape(
            "SELECT DISTINCT ON (RANK() OVER (ORDER BY t.a) % 2 + (SELECT MIN(u.x) FROM u)) t.a \
             FROM {t} GROUP BY t.a ORDER BY t.a",
            "SELECT DISTINCT ON (RANK() OVER (ORDER BY t.a) % 2) t.a FROM t GROUP BY t.a \
             ORDER BY t.a",
        ),
    ]
}

/// The mechanisms a statement the blocks run never reaches
const GENERAL: &[GeneralPath] = &[
    GeneralPath::LiftedSubqueries,
    GeneralPath::SelectFold,
    GeneralPath::ScalarSubquery,
    GeneralPath::CorrelatedExpression,
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::ExistsSubquery,
    GeneralPath::InSubquery,
    GeneralPath::OrderByEvaluator,
];

fn general_hits() -> Vec<usize> {
    GENERAL
        .iter()
        .map(|path| general_path_hits(*path))
        .collect()
}

/// Each row, without the column names
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok(out)
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

#[test]
fn every_source_answers_a_windowed_or_grouped_statement_as_its_twin() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let hot = Database::open("memory://grouped_windows_blocks").unwrap();
    fill(&hot);
    let sealed = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    fill(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut wrong = Vec::new();
    for (sealed, db) in [(false, &hot), (true, &sealed)] {
        for (shape, twin) in shapes() {
            let want = rows(db, &twin).unwrap_or_else(|e| panic!("{twin}: {e}"));
            for (name, source, with) in sources() {
                let sql = format!("{with}{}", shape.replace("{t}", &source));
                let line = prepared_line(db, &sql);
                let before = general_hits();
                let got = rows(db, &sql);
                let general: Vec<String> = GENERAL
                    .iter()
                    .zip(general_hits().iter().zip(&before))
                    .filter(|(_, (after, before))| after != before)
                    .map(|(path, _)| format!("{path:?}"))
                    .collect();
                if got.as_deref() != Ok(&want[..]) || line != "Prepared: yes" || !general.is_empty()
                {
                    wrong.push(format!(
                        "sealed {sealed}, {name}: {sql}\n  [{line}] general {general:?}\n  \
                         got {got:?}\n  want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_statement_differing_only_inside_a_call_or_a_grouping_set_classifies_alone() {
    let _guard = serial();
    let db = Database::open("memory://grouped_windows_blocks_classified").unwrap();
    fill(&db);
    let mut wrong = Vec::new();
    for (twin, sql) in [
        (
            "SELECT t.a, SUM(t.b) FILTER (WHERE t.b > 0) FROM t GROUP BY t.a ORDER BY t.a",
            "SELECT t.a, SUM(t.b) FILTER (WHERE t.b > (SELECT MIN(u.x) FROM u)) FROM t \
             GROUP BY t.a ORDER BY t.a",
        ),
        (
            "SELECT t.a, STRING_AGG(CAST(t.b AS TEXT), ',' ORDER BY t.b * -1) FROM t GROUP BY t.a \
             ORDER BY t.a",
            "SELECT t.a, STRING_AGG(CAST(t.b AS TEXT), ',' ORDER BY t.b * (SELECT MIN(u.x) - 1 \
             FROM u)) FROM t GROUP BY t.a ORDER BY t.a",
        ),
        (
            "SELECT t.a, COUNT(*) FROM t GROUP BY GROUPING SETS ((t.a), (t.a + 0)) ORDER BY t.a",
            "SELECT t.a, COUNT(*) FROM t GROUP BY GROUPING SETS ((t.a), (t.a + (SELECT MIN(u.x) \
             FROM u))) ORDER BY t.a",
        ),
    ] {
        let want = rows(&db, twin).unwrap_or_else(|e| panic!("{twin}: {e}"));
        let line = prepared_line(&db, sql);
        let before = general_hits();
        let got = rows(&db, sql);
        if got.as_deref() != Ok(&want[..]) || line != "Prepared: yes" || general_hits() != before {
            wrong.push(format!("{sql}\n  [{line}] {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_where_set_raises_its_error_only_for_a_row_that_reaches_it() {
    let _guard = serial();
    let db = Database::open("memory://grouped_windows_blocks_where_error").unwrap();
    fill(&db);
    let set = "(SELECT 9223372036854775807 + u.x FROM u)";
    for (sql, fails) in [
        (
            format!("SELECT COUNT(*) FROM t WHERE FALSE AND t.a IN {set}"),
            false,
        ),
        (
            format!("SELECT COUNT(*) FROM t WHERE t.b = 99 AND t.a IN {set}"),
            false,
        ),
        (
            format!("SELECT t.a, COUNT(*) FROM t WHERE TRUE OR t.a IN {set} GROUP BY t.a"),
            false,
        ),
        (format!("SELECT COUNT(*) FROM t WHERE t.a IN {set}"), true),
    ] {
        assert_eq!(rows(&db, &sql).is_err(), fails, "{sql}");
    }
}

#[test]
fn an_argument_read_once_that_reads_the_row_is_an_error() {
    let _guard = serial();
    let db = Database::open("memory://grouped_windows_blocks_row_argument").unwrap();
    fill(&db);
    for sql in [
        "SELECT t.id, NTILE((SELECT COUNT(*) FROM u WHERE u.k = t.a)) OVER (ORDER BY t.id) FROM t",
        "SELECT t.a, STRING_AGG(CAST(t.b AS TEXT), CAST((SELECT COUNT(*) FROM u \
         WHERE u.k = t.a) AS TEXT)) FROM t GROUP BY t.a",
    ] {
        assert!(rows(&db, sql).is_err(), "{sql}");
    }
}

#[test]
fn a_value_function_reads_an_expression_argument_for_each_row() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let hot = Database::open("memory://grouped_windows_blocks_values").unwrap();
    fill(&hot);
    let sealed = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    fill(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let over = "OVER (PARTITION BY t.a ORDER BY t.id ROWS BETWEEN 1 PRECEDING AND 2 FOLLOWING)";
    for db in [&hot, &sealed] {
        for call in [
            "FIRST_VALUE({arg})",
            "LAST_VALUE({arg})",
            "NTH_VALUE({arg}, 2)",
            "LAG({arg})",
            "LEAD({arg}, 1, 0)",
        ] {
            let sql = format!(
                "SELECT t.id, {} {over} FROM t ORDER BY t.id",
                call.replace("{arg}", "t.b * 2 + 1")
            );
            let twin = format!(
                "SELECT t.id, {} {over} FROM (SELECT id, a, b * 2 + 1 AS c FROM t) t ORDER BY t.id",
                call.replace("{arg}", "t.c")
            );
            assert_eq!(rows(db, &sql), rows(db, &twin), "{sql}");
        }
    }
}

#[test]
fn an_order_by_subquery_reading_a_select_alias_sorts_by_it() {
    let _guard = serial();
    let db = Database::open("memory://grouped_windows_blocks_alias").unwrap();
    fill(&db);
    for (sql, twin) in [
        (
            "SELECT t.a AS grp, COUNT(*) FROM t GROUP BY t.a ORDER BY (SELECT t.grp) DESC",
            "SELECT t.a AS grp, COUNT(*) FROM t GROUP BY t.a ORDER BY t.a DESC",
        ),
        // A dotted name is one name, which no column has: its text
        (
            "SELECT t.a AS grp, COUNT(*) FROM t GROUP BY t.a \
             ORDER BY (SELECT \"t.grp\" + 0 * t.a), t.a DESC",
            "SELECT t.a AS grp, COUNT(*) FROM t GROUP BY t.a \
             ORDER BY (SELECT 't.grp' + 0 * t.a), t.a DESC",
        ),
    ] {
        assert_eq!(rows(&db, sql), rows(&db, twin), "{sql}");
    }
}

#[test]
fn an_item_holding_a_subquery_keeps_the_name_it_is_written_under() {
    let _guard = serial();
    let db = Database::open("memory://grouped_windows_blocks_names").unwrap();
    fill(&db);
    for (sql, names) in [
        (
            "SELECT t.id, (SELECT MAX(u.x) FROM u), \
             SUM(t.b) OVER (PARTITION BY t.a > (SELECT MIN(u.x) FROM u) + 3) FROM t",
            vec![
                "id",
                "(SELECT MAX(u.x) FROM u)",
                "SUM(t.b) OVER (PARTITION BY t.a > (SELECT MIN(u.x) FROM u) + 3)",
            ],
        ),
        (
            "SELECT t.a, SUM(SUM(t.b)) OVER (ORDER BY t.a) + (SELECT MAX(u.x) FROM u), \
             (SELECT MAX(u.x) FROM u) AS m FROM t GROUP BY t.a",
            vec![
                "a",
                "SUM(SUM(t.b)) OVER (ORDER BY t.a) + (SELECT MAX(u.x) FROM u)",
                "m",
            ],
        ),
    ] {
        assert_eq!(db.query(sql, ()).unwrap().columns(), names, "{sql}");
    }
}
