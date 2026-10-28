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

//! A grouped statement over one table whose SELECT list or HAVING holds a
//! subquery runs it through prepared blocks: it answers as recorded when the
//! general path gave the same answer, hot and sealed, and an uncorrelated
//! value keeps the aggregate path its literal takes

#![cfg(feature = "test-failpoints")]

mod oracle;

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    aggregate_paths_taken, before_grouped_table_opened, general_path_hits, reset_aggregate_paths,
    GeneralPath,
};
use stoolap::Database;

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// t: id 1..200, a = id % 9, b = id % 13; u: id 1..40, k = id % 11, x = id % 7
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 9, g.value % 13 FROM generate_series(1, 200) g",
        "INSERT INTO u SELECT g.value, g.value % 11, g.value % 7 FROM generate_series(1, 40) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// Each statement, and for one whose subqueries read no row, the same
/// statement with each subquery written as its value
const CASES: &[(&str, Option<&str>)] = &[
    (
        "SELECT a, SUM(b), (SELECT MAX(u.x) FROM u) FROM t GROUP BY a ORDER BY a",
        Some("SELECT a, SUM(b), 6 FROM t GROUP BY a ORDER BY a"),
    ),
    (
        "SELECT a, (SELECT MAX(u.x) FROM u) + SUM(b) AS s FROM t GROUP BY a ORDER BY a",
        Some("SELECT a, 6 + SUM(b) AS s FROM t GROUP BY a ORDER BY a"),
    ),
    (
        "SELECT SUM(b), (SELECT MAX(u.x) FROM u), COUNT(*) FROM t",
        Some("SELECT SUM(b), 6, COUNT(*) FROM t"),
    ),
    (
        "SELECT a, SUM(b) FROM t GROUP BY a HAVING SUM(b) > (SELECT MAX(u.x) FROM u) * 20 ORDER BY a",
        Some("SELECT a, SUM(b) FROM t GROUP BY a HAVING SUM(b) > 6 * 20 ORDER BY a"),
    ),
    (
        "SELECT a, COUNT(*), EXISTS (SELECT 1 FROM u WHERE u.x > 5) FROM t GROUP BY a ORDER BY a",
        Some("SELECT a, COUNT(*), TRUE FROM t GROUP BY a ORDER BY a"),
    ),
    (
        "SELECT a, SUM(b), (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM t GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, EXISTS (SELECT 1 FROM u WHERE u.k = t.a AND u.x > 3) FROM t GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, a IN (SELECT u.k FROM u WHERE u.x > 3) FROM t GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, a > ANY (SELECT u.k FROM u WHERE u.x = t.a) FROM t GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, (SELECT MAX(u.x) FROM u WHERE u.k = t.a) * COUNT(*) FROM t GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, SUM(b) > ANY (SELECT u.k * 12 FROM u WHERE u.x = t.a) FROM t GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, SUM(b) FROM t GROUP BY a HAVING COUNT(*) IN (SELECT u.k * 2 + 2 FROM u) ORDER BY a",
        None,
    ),
    (
        "SELECT a, b, (SELECT COUNT(*) FROM u WHERE u.k = t.a AND u.x = t.b) FROM t \
         GROUP BY a, b ORDER BY a, b",
        None,
    ),
    (
        "SELECT a, SUM(b) FROM t GROUP BY a HAVING EXISTS (SELECT 1 FROM u WHERE u.k = t.a) ORDER BY a",
        None,
    ),
    (
        "SELECT a, COUNT(*) FROM t GROUP BY a \
         HAVING COUNT(*) >= 6 * (SELECT COUNT(*) FROM u WHERE u.k = t.a) ORDER BY a",
        None,
    ),
    (
        "SELECT a, SUM(b) FROM t GROUP BY a HAVING a IN (SELECT u.k FROM u WHERE u.x > 3) ORDER BY a",
        None,
    ),
    (
        "SELECT a AS g, COUNT(*) FROM t GROUP BY a HAVING g IN (SELECT u.k FROM u WHERE u.x > 3) \
         ORDER BY g",
        None,
    ),
    (
        "SELECT a AS g, SUM(b) FROM t GROUP BY a HAVING g > ANY (SELECT u.k FROM u WHERE u.x = t.a) \
         ORDER BY g",
        None,
    ),
    (
        "SELECT a AS k, b AS k, COUNT(*) FROM t GROUP BY a, b \
         HAVING k = (SELECT MIN(u.k) + 1 FROM u) ORDER BY 1, 2",
        Some("SELECT a AS k, b AS k, COUNT(*) FROM t GROUP BY a, b HAVING k = 1 ORDER BY 1, 2"),
    ),
    (
        "SELECT a, (SELECT MAX(u.x) FROM u), (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM t \
         GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, SUM(b), (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM t WHERE b > 3 \
         GROUP BY a ORDER BY a LIMIT 4",
        None,
    ),
];

/// Each statement whose subqueries read the input rows, the same answer
/// with each subquery's value computed as a column of a derived table, and
/// for one whose subqueries read no row, the statement with their values
const INPUT_CASES: &[(&str, &str, Option<&str>)] = &[
    (
        "SELECT COUNT(*) FROM t GROUP BY id > (SELECT MAX(u.x) * 20 FROM u) ORDER BY 1",
        "SELECT COUNT(*) FROM (SELECT id > (SELECT MAX(u.x) * 20 FROM u) AS g FROM t) d \
         GROUP BY g ORDER BY 1",
        Some("SELECT COUNT(*) FROM t GROUP BY id > 120 ORDER BY 1"),
    ),
    (
        "SELECT COUNT(*), SUM(b) FROM t GROUP BY (SELECT COUNT(*) FROM u WHERE u.k = t.a) \
         ORDER BY 1, 2",
        "SELECT COUNT(*), SUM(b) FROM (SELECT b, (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS g \
         FROM t) d GROUP BY g ORDER BY 1, 2",
        None,
    ),
    (
        "SELECT id > (SELECT MAX(u.x) * 20 FROM u), COUNT(*) FROM t \
         GROUP BY id > (SELECT MAX(u.x) * 20 FROM u) ORDER BY 1",
        "SELECT g, COUNT(*) FROM (SELECT id > (SELECT MAX(u.x) * 20 FROM u) AS g FROM t) d \
         GROUP BY g ORDER BY 1",
        Some("SELECT id > 120, COUNT(*) FROM t GROUP BY id > 120 ORDER BY 1"),
    ),
    (
        "SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS c, COUNT(*) FROM t \
         GROUP BY (SELECT COUNT(*) FROM u WHERE u.k = t.a) ORDER BY 1",
        "SELECT g, COUNT(*) FROM (SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS g FROM t) d \
         GROUP BY g ORDER BY 1",
        None,
    ),
    (
        "SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS c, COUNT(*) FROM t GROUP BY c \
         ORDER BY 1",
        "SELECT g, COUNT(*) FROM (SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS g FROM t) d \
         GROUP BY g ORDER BY 1",
        None,
    ),
    (
        "SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a), COUNT(*) FROM t GROUP BY 1 ORDER BY 1",
        "SELECT g, COUNT(*) FROM (SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS g FROM t) d \
         GROUP BY g ORDER BY 1",
        None,
    ),
    (
        "SELECT a, SUM(b * (SELECT MAX(u.x) FROM u)) FROM t GROUP BY a ORDER BY a",
        "SELECT a, SUM(v) FROM (SELECT a, b * (SELECT MAX(u.x) FROM u) AS v FROM t) d \
         GROUP BY a ORDER BY a",
        Some("SELECT a, SUM(b * 6) FROM t GROUP BY a ORDER BY a"),
    ),
    (
        "SELECT a, SUM((SELECT COUNT(*) FROM u WHERE u.k = t.b)) FROM t GROUP BY a ORDER BY a",
        "SELECT a, SUM(v) FROM (SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.b) AS v FROM t) d \
         GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, COUNT(*) FILTER (WHERE b > (SELECT MIN(u.x) FROM u WHERE u.k = t.a)) FROM t \
         GROUP BY a ORDER BY a",
        "SELECT a, COUNT(*) FILTER (WHERE f) FROM (SELECT a, b > (SELECT MIN(u.x) FROM u \
         WHERE u.k = t.a) AS f FROM t) d GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, SUM(b) FILTER (WHERE b IN (SELECT u.x FROM u WHERE u.k < 3)) FROM t \
         GROUP BY a ORDER BY a",
        "SELECT a, SUM(b) FILTER (WHERE f) FROM (SELECT a, b, b IN (SELECT u.x FROM u \
         WHERE u.k < 3) AS f FROM t) d GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT a, STRING_AGG(CAST(b AS TEXT), ',' ORDER BY (SELECT COUNT(*) FROM u \
         WHERE u.x = t.b), b) FROM t GROUP BY a ORDER BY a",
        "SELECT a, STRING_AGG(CAST(b AS TEXT), ',' ORDER BY o, b) FROM (SELECT a, b, \
         (SELECT COUNT(*) FROM u WHERE u.x = t.b) AS o FROM t) d GROUP BY a ORDER BY a",
        None,
    ),
    (
        "SELECT COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.a AND u.x > 4)), \
         SUM(b) FROM t",
        "SELECT c, s FROM (SELECT COUNT(*) AS c FROM t WHERE EXISTS (SELECT 1 FROM u \
         WHERE u.k = t.a AND u.x > 4)) x, (SELECT SUM(b) AS s FROM t) y",
        None,
    ),
    (
        "SELECT a FROM t GROUP BY a HAVING SUM((SELECT COUNT(*) FROM u WHERE u.k = t.b)) > 60 \
         ORDER BY a",
        "SELECT a FROM (SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.b) AS v FROM t) d \
         GROUP BY a HAVING SUM(v) > 60 ORDER BY a",
        None,
    ),
    (
        "SELECT a, SUM(b * (SELECT MAX(u.x) FROM u WHERE u.k = t.a)) FROM t GROUP BY a \
         HAVING SUM(b * (SELECT MAX(u.x) FROM u WHERE u.k = t.a)) > 300 ORDER BY a",
        "SELECT a, SUM(v) FROM (SELECT a, b * (SELECT MAX(u.x) FROM u WHERE u.k = t.a) AS v \
         FROM t) d GROUP BY a HAVING SUM(v) > 300 ORDER BY a",
        None,
    ),
    (
        "SELECT a, SUM(b) FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.b) GROUP BY a \
         HAVING SUM(b) > (SELECT MAX(u.x) FROM u) * 10 ORDER BY a",
        "SELECT a, SUM(b) FROM (SELECT a, b FROM t WHERE EXISTS (SELECT 1 FROM u \
         WHERE u.k = t.b)) d GROUP BY a HAVING SUM(b) > 60 ORDER BY a",
        None,
    ),
    (
        "SELECT a, SUM((SELECT COUNT(*) FROM u WHERE u.k = t.b)) FROM t \
         WHERE b > (SELECT MIN(u.x) FROM u) + 2 GROUP BY a ORDER BY a",
        "SELECT a, SUM(v) FROM (SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.b) AS v FROM t \
         WHERE b > 2) d GROUP BY a ORDER BY a",
        None,
    ),
];

/// The paths a statement the blocks run never reaches
const GENERAL: &[GeneralPath] = &[
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::ExistsSubquery,
    GeneralPath::InSubquery,
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

fn columns(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ()).unwrap().columns().to_vec()
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

const RECORDED: oracle::Answers = answers!("grouped_output_blocks");
const RECORDED_INPUT: oracle::Answers = answers!("grouped_output_blocks_input");
const RECORDED_BEFORE: oracle::Answers = answers!("grouped_output_blocks_before");
const RECORDED_AROUND: oracle::Answers = answers!("grouped_output_blocks_around");
const RECORDED_ERRORS: oracle::Answers = answers!("grouped_output_blocks_errors");
const RECORDED_KEYED: oracle::Answers = answers!("grouped_output_blocks_keyed");
const RECORDED_SCHEMA: oracle::Answers = answers!("grouped_output_blocks_schema");

/// The rows of `sql`, or its error, as `answers` records them under `key`,
/// recorded only when the general path gives the same ones
fn confirmed(
    db: &Database,
    answers: &oracle::Answers,
    key: &str,
    sql: &str,
) -> Result<Vec<String>, String> {
    let got = rows(db, sql);
    let general = || rows(db, &oracle::general_twin(sql));
    if let Some(why) = oracle::check_confirmed(answers, key, &got, general) {
        panic!("{why}");
    }
    got
}

#[test]
fn a_grouped_statement_reads_its_output_s_subqueries_through_its_blocks() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://grouped_output_blocks").unwrap();
            fill(&db);
            db
        };
        for (sql, literal) in CASES {
            let hits = general_hits();
            let first = rows(&db, sql);
            reset_aggregate_paths();
            let again = rows(&db, sql);
            let taken = aggregate_paths_taken();
            let reached = general_hits();
            if reached != hits {
                wrong.push(format!(
                    "sealed {sealed}: {sql}: reached the general path {hits:?} -> {reached:?}"
                ));
            }
            // The general path cannot run a mixed SELECT list: it confirms its literal
            let general = match sql.contains("(SELECT MAX(u.x) FROM u), (SELECT COUNT") {
                true => sql.replace("(SELECT MAX(u.x) FROM u)", "6"),
                false => sql.to_string(),
            };
            wrong.extend(
                oracle::check_confirmed(&RECORDED, sql, &first, || {
                    rows(&db, &oracle::general_twin(&general))
                })
                .map(|why| format!("sealed {sealed}: {why}")),
            );
            let want = first.clone();
            if !want.as_ref().is_ok_and(|rows| !rows.is_empty()) {
                wrong.push(format!("sealed {sealed}: {sql}: answers {want:?}"));
                continue;
            }
            let line = prepared_line(&db, sql);
            if line != "Prepared: yes" {
                wrong.push(format!("sealed {sealed}: {sql}: {line}"));
            }
            if again != want {
                wrong.push(format!(
                    "sealed {sealed}: {sql}: gave {first:?} then {again:?}"
                ));
            }
            if let Some(literal) = literal {
                let _ = rows(&db, literal);
                reset_aggregate_paths();
                let literal_rows = rows(&db, literal);
                let literal_taken = aggregate_paths_taken();
                if taken != literal_taken {
                    wrong.push(format!(
                        "sealed {sealed}: {sql}: took {taken:?}, its literal {literal_taken:?}"
                    ));
                }
                if literal_rows != want {
                    wrong.push(format!("sealed {sealed}: {literal}: {literal_rows:?}"));
                }
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
fn an_aggregate_beside_an_uncorrelated_value_answers_as_its_literal_twin() {
    let db = Database::open("memory://grouped_output_value_twin").unwrap();
    fill(&db);
    for (sql, twin) in [
        (
            "SELECT COUNT(*) + (SELECT 1) FROM t",
            "SELECT COUNT(*) + 1 FROM t",
        ),
        (
            "SELECT SUM(b) + (SELECT 1) FROM t",
            "SELECT SUM(b) + 1 FROM t",
        ),
        (
            "SELECT COUNT(*) + (SELECT 1) FROM t WHERE a > (SELECT 4)",
            "SELECT COUNT(*) + 1 FROM t WHERE a > 4",
        ),
    ] {
        let want = rows(&db, twin);
        assert!(want.is_ok(), "{twin}: {want:?}");
        for _ in 0..2 {
            assert_eq!(rows(&db, sql), want, "{sql}");
        }
    }
}

#[test]
fn a_grouped_statement_reads_its_input_rows_subqueries_through_its_blocks() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://grouped_input_blocks").unwrap();
            fill(&db);
            db
        };
        for (sql, reference, literal) in INPUT_CASES {
            // The reference reads its subqueries as a derived table's columns,
            // recorded when the general path gives it the same answer
            let want = rows(&db, reference);
            wrong.extend(
                oracle::check_confirmed(&RECORDED_INPUT, reference, &want, || {
                    rows(&db, &oracle::general_twin(reference))
                })
                .map(|why| format!("sealed {sealed}: {why}")),
            );
            if !want.as_ref().is_ok_and(|rows| !rows.is_empty()) {
                wrong.push(format!("sealed {sealed}: {reference}: {want:?}"));
                continue;
            }
            let hits = general_hits();
            let first = rows(&db, sql);
            reset_aggregate_paths();
            let again = rows(&db, sql);
            let taken = aggregate_paths_taken();
            let reached = general_hits();
            if reached != hits {
                wrong.push(format!(
                    "sealed {sealed}: {sql}: reached the general path {hits:?} -> {reached:?}"
                ));
            }
            let line = prepared_line(&db, sql);
            if line != "Prepared: yes" {
                wrong.push(format!("sealed {sealed}: {sql}: {line}"));
            }
            if first != want || again != want {
                wrong.push(format!(
                    "sealed {sealed}: {sql}: gave {first:?} then {again:?}, the reference {want:?}"
                ));
            }
            if let Some(literal) = literal {
                let _ = rows(&db, literal);
                reset_aggregate_paths();
                let literal_rows = rows(&db, literal);
                let literal_taken = aggregate_paths_taken();
                if taken != literal_taken {
                    wrong.push(format!(
                        "sealed {sealed}: {sql}: took {taken:?}, its literal {literal_taken:?}"
                    ));
                }
                if literal_rows != want {
                    wrong.push(format!("sealed {sealed}: {literal}: {literal_rows:?}"));
                }
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
fn a_grouped_column_holding_a_subquery_is_named_by_its_expression() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_names").unwrap();
    fill(&db);
    for (sql, names) in [
        (
            "SELECT a, (SELECT MAX(u.x) FROM u) FROM t GROUP BY a",
            &["a", "(SELECT MAX(u.x) FROM u)"][..],
        ),
        (
            "SELECT a, (SELECT MAX(u.x) FROM u) + SUM(b) FROM t GROUP BY a",
            &["a", "(SELECT MAX(u.x) FROM u) + SUM(b)"],
        ),
        (
            "SELECT a, a IN (SELECT u.k FROM u) FROM t GROUP BY a",
            &["a", "a IN (SELECT u.k FROM u)"],
        ),
        (
            "SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM t GROUP BY a",
            &["a", "(SELECT COUNT(*) FROM u WHERE (u.k = t.a))"],
        ),
        (
            "SELECT SUM(b), (SELECT MAX(u.x) FROM u) AS m FROM t",
            &["SUM(b)", "m"],
        ),
    ] {
        assert_eq!(columns(&db, sql), names, "{sql}");
    }
}

#[test]
fn an_uncorrelated_grouped_subquery_is_read_before_any_group() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_before").unwrap();
    fill(&db);
    for sql in [
        "SELECT a, (SELECT u.x FROM u WHERE u.k = 1) FROM t WHERE a < 0 GROUP BY a",
        "SELECT a, COUNT(*) FROM t WHERE a < 0 GROUP BY a HAVING COUNT(*) > (SELECT u.x FROM u)",
        "SELECT a, a IN (SELECT u.x * 9223372036854775807 FROM u) FROM t WHERE a < 0 GROUP BY a",
    ] {
        let got = confirmed(&db, &RECORDED_BEFORE, sql, sql);
        assert!(got.is_err(), "{sql}: {got:?}");
    }
}

#[test]
fn a_value_around_an_uncorrelated_grouped_subquery_is_computed_only_for_a_group() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_around").unwrap();
    fill(&db);
    for sql in [
        "SELECT a, 9223372036854775807 + (SELECT 1) FROM t WHERE a < 0 GROUP BY a",
        "SELECT a FROM t WHERE a < 0 GROUP BY a HAVING 9223372036854775807 + (SELECT 1) > 0",
    ] {
        assert_eq!(
            confirmed(&db, &RECORDED_AROUND, sql, sql),
            Ok(Vec::new()),
            "{sql}"
        );
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    }
}

#[test]
fn a_correlated_grouped_subquery_raises_its_error_for_the_group_that_reaches_it() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_errors").unwrap();
    fill(&db);
    for sql in [
        "SELECT a, (SELECT u.x FROM u WHERE u.k = t.a) FROM t GROUP BY a",
        "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > (SELECT u.x FROM u WHERE u.k = t.a)",
        "SELECT a, (SELECT u.x FROM u WHERE u.k = t.a) FROM t WHERE a = 0 GROUP BY a",
    ] {
        let _ = confirmed(&db, &RECORDED_ERRORS, sql, sql);
    }
}

#[test]
fn an_input_row_subquery_raises_its_error_only_for_a_row_that_reaches_it() {
    let _guard = serial();
    let db = Database::open("memory://grouped_input_blocks_errors").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO t VALUES (1, 0, 1), (2, 1, 2), (3, 1, 3)",
        "INSERT INTO u VALUES (1, 0, 5), (2, 1, 6), (3, 1, 7)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let rows_of = |want: &[&str]| Ok(want.iter().map(|row| row.to_string()).collect::<Vec<_>>());
    for (sql, want) in [
        (
            "SELECT a, SUM((SELECT u.x FROM u WHERE u.k = t.a)) FILTER (WHERE a = 0) FROM t \
             GROUP BY a ORDER BY a",
            rows_of(&["Integer(0)|Integer(5)", "Integer(1)|Null(Null)"]),
        ),
        (
            "SELECT a, SUM(CASE WHEN a = 0 THEN (SELECT u.x FROM u WHERE u.k = t.a) ELSE 0 END) \
             FROM t GROUP BY a ORDER BY a",
            rows_of(&["Integer(0)|Integer(5)", "Integer(1)|Integer(0)"]),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE a = 0 GROUP BY (SELECT u.x FROM u WHERE u.k = t.a)",
            rows_of(&["Integer(1)"]),
        ),
    ] {
        assert_eq!(rows(&db, sql), want, "{sql}");
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    }
    for sql in [
        "SELECT COUNT(*) FROM t GROUP BY (SELECT u.x FROM u WHERE u.k = t.a)",
        "SELECT COUNT(*) FROM t WHERE a = 7 GROUP BY (SELECT u.x FROM u WHERE u.k = 1)",
        "SELECT a, SUM((SELECT u.x FROM u WHERE u.k = 1)) FROM t WHERE a = 7 GROUP BY a",
    ] {
        let got = rows(&db, sql);
        assert!(
            got.as_ref().is_err_and(|e| e.contains("more than one row")),
            "{sql}: {got:?}"
        );
    }
}

#[test]
fn two_anonymous_parameters_in_grouped_items_stay_two() {
    let _guard = serial();
    let db = Database::open("memory://grouped_input_blocks_anonymous").unwrap();
    fill(&db);
    for (sql, literal) in [
        (
            "SELECT a + (SELECT ?) AS x, a + (SELECT ?) AS y, COUNT(*) FROM t GROUP BY 1, 2 \
             ORDER BY 1, 2",
            "SELECT a + 1 AS x, a + 2 AS y, COUNT(*) FROM t GROUP BY 1, 2 ORDER BY 1, 2",
        ),
        (
            "SELECT a + (SELECT ?) AS x, a + (SELECT ?) AS y, COUNT(*) FROM t GROUP BY x, y \
             ORDER BY 1, 2",
            "SELECT a + 1 AS x, a + 2 AS y, COUNT(*) FROM t GROUP BY x, y ORDER BY 1, 2",
        ),
        (
            "SELECT COUNT(*), MIN(a) FROM t GROUP BY a + (SELECT ?), a - (SELECT ?) ORDER BY 1, 2",
            "SELECT COUNT(*), MIN(a) FROM t GROUP BY a + 1, a - 2 ORDER BY 1, 2",
        ),
    ] {
        let got: Vec<String> = db
            .query(sql, (1i64, 2i64))
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (0..row.len())
                    .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .collect();
        assert_eq!(Ok(got), rows(&db, literal), "{sql}");
    }
}

#[test]
fn a_grouped_item_named_by_alias_or_position_is_read_wherever_it_is_written() {
    let _guard = serial();
    let db = Database::open("memory://grouped_input_blocks_named_item").unwrap();
    fill(&db);
    for (sql, literal) in [
        (
            "SELECT a + (SELECT 1) AS g, COUNT(*) FROM t GROUP BY g \
             HAVING a + (SELECT 1) > 3 ORDER BY 1",
            "SELECT a + 1 AS g, COUNT(*) FROM t GROUP BY g HAVING a + 1 > 3 ORDER BY 1",
        ),
        (
            "SELECT a + (SELECT 1), COUNT(*), (a + (SELECT 1)) * 2 FROM t GROUP BY 1 \
             HAVING a + (SELECT 1) > 3 ORDER BY 1",
            "SELECT a + 1, COUNT(*), (a + 1) * 2 FROM t GROUP BY 1 HAVING a + 1 > 3 ORDER BY 1",
        ),
    ] {
        let got = rows(&db, sql);
        assert!(
            got.as_ref().is_ok_and(|rows| !rows.is_empty()),
            "{sql}: {got:?}"
        );
        assert_eq!(got, rows(&db, literal), "{sql}");
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    }
}

#[test]
fn a_where_naming_a_select_alias_reads_that_item() {
    let _guard = serial();
    let db = Database::open("memory://grouped_input_blocks_where_alias").unwrap();
    fill(&db);
    let sql = "SELECT a AS z, (SELECT MAX(u.x) FROM u) FROM t WHERE z IN (SELECT u.k FROM u \
               WHERE u.x > 3) GROUP BY a ORDER BY 1";
    let got = rows(&db, sql);
    assert!(
        got.as_ref().is_ok_and(|rows| !rows.is_empty()),
        "{sql}: {got:?}"
    );
    assert_eq!(
        got,
        rows(&db, &sql.replace("WHERE z IN", "WHERE a IN")),
        "{sql}"
    );
    assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
}

#[test]
fn a_grouped_statement_s_subqueries_read_parameters_and_the_transaction_s_writes() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_params").unwrap();
    fill(&db);
    let read = |sql: &str, params: (i64, i64)| -> Vec<String> {
        db.query(sql, params)
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (0..row.len())
                    .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .collect()
    };
    for sql in [
        "SELECT a, SUM(b) + (SELECT MAX(u.x) FROM u WHERE u.k < $1) + $2 FROM t \
         GROUP BY a ORDER BY a",
        "SELECT a, SUM(b) + $1, (SELECT COUNT(*) FROM u WHERE u.k = t.a + $2) FROM t \
         GROUP BY a ORDER BY a",
        "SELECT a, SUM(b * (SELECT MAX(u.x) FROM u WHERE u.k < $1)) + $2 FROM t \
         GROUP BY a ORDER BY a",
        "SELECT a, COUNT(*) FILTER (WHERE b > (SELECT MIN(u.x) FROM u WHERE u.k = t.a + $2)) + $1 \
         FROM t GROUP BY a ORDER BY a",
        "SELECT a, SUM(b) + $1 FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.b + $2) \
         GROUP BY a HAVING SUM(b) > (SELECT MIN(u.x) FROM u) ORDER BY a",
    ] {
        for (p1, p2) in [(3, 0), (11, 1), (0, -2)] {
            let literal = sql
                .replace("$1", &p1.to_string())
                .replace("$2", &p2.to_string());
            // Its literal twin, the parameters written in
            let want = rows(&db, &literal).unwrap();
            assert_eq!(read(sql, (p1, p2)), want, "{sql}: {p1} {p2}");
        }
    }
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO u VALUES (500, 4, 99)", ()).unwrap();
    let mut seen = tx
        .query(
            "SELECT a, (SELECT MAX(u.x) FROM u WHERE u.k = t.a) FROM t WHERE a = 4 GROUP BY a",
            (),
        )
        .unwrap();
    let row = seen.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 99, "the transaction's own row");
    drop(seen);
    let mut seen = tx
        .query(
            "SELECT a, MAX((SELECT MAX(u.x) FROM u WHERE u.k = t.a)) FROM t WHERE a = 4 GROUP BY a",
            (),
        )
        .unwrap();
    let row = seen.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 99, "an input row reads it too");
    drop(seen);
    tx.rollback().unwrap();
}

#[test]
fn a_grouped_statement_outside_the_blocks_says_why() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_outside").unwrap();
    fill(&db);
    for (sql, reason) in [
        (
            "SELECT a, RANK() OVER (ORDER BY a), (SELECT MAX(u.x) FROM u) FROM t GROUP BY a \
             ORDER BY (SELECT MIN(u.x) FROM u) + a",
            "an ORDER BY holding a subquery",
        ),
        (
            "SELECT a AS g, (SELECT MAX(u.x) FROM u) FROM t GROUP BY a \
             ORDER BY (SELECT MIN(u.x) FROM u) + g",
            "an ORDER BY subquery key naming a SELECT alias",
        ),
        (
            "SELECT a, (SELECT SUM((SELECT 1)) FROM u) FROM t GROUP BY a",
            "a subquery outside WHERE and the subquery blocks",
        ),
        (
            "SELECT a, SUM(b), RANK() OVER w FROM t GROUP BY a \
             HAVING SUM(b) > (SELECT MAX(u.x) FROM u) WINDOW w AS (ORDER BY a)",
            "a named window",
        ),
    ] {
        assert_eq!(
            prepared_line(&db, sql),
            format!("Prepared: no ({reason})"),
            "{sql}"
        );
    }
}

#[test]
fn a_grouped_statement_keyed_by_an_item_holding_a_subquery_groups_by_it() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_keyed").unwrap();
    fill(&db);
    for sql in [
        "SELECT a IN (SELECT u.k FROM u WHERE u.k < 4) AS g, COUNT(*) FROM t GROUP BY g",
        "SELECT a IN (SELECT u.k FROM u WHERE u.k < 4), COUNT(*) FROM t GROUP BY 1",
        "SELECT 0 AS g, a IN (SELECT u.k FROM u WHERE u.k < 4) AS g, COUNT(*) FROM t GROUP BY g",
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let sorted = |sql: &str| {
            rows(&db, sql).map(|mut rows| {
                rows.sort();
                rows
            })
        };
        let got = sorted(sql);
        assert!(
            got.as_ref().is_ok_and(|rows| rows.len() == 2),
            "{sql}: {got:?}"
        );
        let general = || sorted(&oracle::general_twin(sql));
        assert_eq!(
            oracle::check_confirmed(&RECORDED_KEYED, sql, &got, general),
            None
        );
    }
}

#[test]
fn a_table_whose_columns_changed_after_preparing_raises_schema_changed() {
    let _guard = serial();
    for (name, sql) in [
        (
            "grouped_output_blocks_retype",
            "SELECT k, k IN (SELECT u.a FROM u) FROM t GROUP BY k",
        ),
        (
            "grouped_output_blocks_retype_where",
            "SELECT k, k IN (SELECT u.a FROM u) FROM t \
             WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id) GROUP BY k",
        ),
    ] {
        let db = Database::open(&format!("memory://{name}")).unwrap();
        for setup in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
            "CREATE TABLE u (id INTEGER PRIMARY KEY, a TEXT)",
            "INSERT INTO t VALUES (1, 1)",
            "INSERT INTO u VALUES (1, '01')",
        ] {
            db.execute(setup, ()).unwrap();
        }
        // Twice, so the second run reads the plan the first one held
        for _ in 0..2 {
            let _ = confirmed(&db, &RECORDED_SCHEMA, &format!("{name} before: {sql}"), sql);
        }
        let writer = db.clone();
        let done = Rc::new(Cell::new(false));
        let flag = Rc::clone(&done);
        before_grouped_table_opened(move || {
            writer
                .execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
                .unwrap();
            flag.set(true);
        });
        let got = rows(&db, sql);
        assert!(
            done.get(),
            "{sql}: the columns changed inside the statement"
        );
        assert!(
            got.as_ref().is_err_and(|e| e.contains("schema changed")),
            "{sql}: {got:?}"
        );
        let _ = confirmed(&db, &RECORDED_SCHEMA, &format!("{name} after: {sql}"), sql);
    }
}

#[test]
fn a_having_name_reads_the_alias_the_grouped_path_reads() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_having_names").unwrap();
    fill(&db);
    for setup in [
        "CREATE TABLE v (id INTEGER PRIMARY KEY, g INTEGER, b BOOLEAN)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, s TEXT)",
        "INSERT INTO v VALUES (1, 1, TRUE), (2, 1, FALSE), (3, 2, TRUE)",
        "INSERT INTO m VALUES (1, '2')",
    ] {
        db.execute(setup, ()).unwrap();
    }
    let sql =
        "SELECT g, COUNT(*) AS b FROM v GROUP BY g HAVING b IN (SELECT m.s FROM m) ORDER BY g";
    let want = rows(&db, &sql.replace(" FROM v", " FROM (SELECT * FROM v) v"));
    assert_eq!(rows(&db, sql), want, "{sql}");
    assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    // A name inside a group item or an aggregate's argument is not an alias
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO w VALUES (1, 1, 2)", ()).unwrap();
    for (sql, want) in [
        (
            "SELECT SUM(a) AS b FROM w HAVING SUM(b) > (SELECT 0)",
            "Integer(1)",
        ),
        (
            "SELECT a + 1 AS a FROM w GROUP BY a + 1 HAVING a + 1 = (SELECT 2)",
            "Integer(2)",
        ),
    ] {
        let general = rows(&db, &sql.replace(" FROM w", " FROM (SELECT * FROM w) w"));
        assert_eq!(
            general,
            Ok(vec![want.to_string()]),
            "{sql}: the general path"
        );
        assert_eq!(rows(&db, sql), general, "{sql}");
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    }
    // A subquery that groups reads its own HAVING's aliases the same way
    let sql = "SELECT id FROM u WHERE EXISTS (SELECT t.a AS k, t.b AS k FROM t \
               WHERE t.a = u.x GROUP BY t.a, t.b HAVING k = 1) ORDER BY id";
    assert_eq!(
        rows(&db, sql),
        Ok(["1", "8", "15", "22", "29", "36"]
            .iter()
            .map(|id| format!("Integer({id})"))
            .collect()),
        "{sql}: k is t.a, the first item named k"
    );
    assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
}
