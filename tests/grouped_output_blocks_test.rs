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
//! subquery runs it through prepared blocks: it answers as the general path
//! does, hot and sealed, and an uncorrelated value keeps the aggregate path
//! its literal takes

#![cfg(feature = "test-failpoints")]

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

/// The statement over a derived table, out of the blocks
fn general(sql: &str) -> String {
    sql.replace(" FROM t", " FROM (SELECT * FROM t) t")
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
            // The general path cannot run a mixed SELECT list: compare its literal
            let mixed = sql.contains("(SELECT MAX(u.x) FROM u), (SELECT COUNT");
            let want = match mixed {
                true => rows(&db, &general(&sql.replace("(SELECT MAX(u.x) FROM u)", "6"))),
                false => rows(&db, &general(sql)),
            };
            if !want.as_ref().is_ok_and(|rows| !rows.is_empty()) {
                wrong.push(format!("sealed {sealed}: {sql}: the general path {want:?}"));
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
                    "sealed {sealed}: {sql}: gave {first:?} then {again:?}, the general path {want:?}"
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
        let got = rows(&db, sql);
        let want = rows(&db, &general(sql));
        assert!(
            got.is_err() && got == want,
            "{sql}: {got:?}, the general path {want:?}"
        );
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
        assert_eq!(rows(&db, sql), Ok(Vec::new()), "{sql}");
        assert_eq!(rows(&db, &general(sql)), Ok(Vec::new()), "{sql}");
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
        assert_eq!(rows(&db, sql), rows(&db, &general(sql)), "{sql}");
    }
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
    ] {
        for (p1, p2) in [(3, 0), (11, 1), (0, -2)] {
            let literal = sql
                .replace("$1", &p1.to_string())
                .replace("$2", &p2.to_string());
            let want = rows(&db, &general(&literal)).unwrap();
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
    tx.rollback().unwrap();
}

#[test]
fn a_grouped_statement_outside_the_blocks_says_why() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_outside").unwrap();
    fill(&db);
    for (sql, reason) in [
        (
            "SELECT a, (SELECT MAX(u.x) FROM u) FROM t WHERE b IN (SELECT u.x FROM u) GROUP BY a",
            "a WHERE, GROUP BY or ORDER BY holding a subquery",
        ),
        (
            "SELECT a, SUM((SELECT MAX(u.x) FROM u WHERE u.k = t.a)) FROM t GROUP BY a",
            "a subquery outside WHERE and the subquery blocks",
        ),
        (
            "SELECT a, SUM(b), RANK() OVER (ORDER BY a) FROM t GROUP BY a \
             HAVING SUM(b) > (SELECT MAX(u.x) FROM u)",
            "a window function",
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
fn a_grouped_statement_keyed_by_an_item_holding_a_subquery_stays_outside_the_blocks() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_keyed").unwrap();
    fill(&db);
    for sql in [
        "SELECT a IN (SELECT u.k FROM u WHERE u.k < 4) AS g, COUNT(*) FROM t GROUP BY g",
        "SELECT a IN (SELECT u.k FROM u WHERE u.k < 4), COUNT(*) FROM t GROUP BY 1",
    ] {
        assert_eq!(
            prepared_line(&db, sql),
            "Prepared: no (a WHERE, GROUP BY or ORDER BY holding a subquery)",
            "{sql}"
        );
        let sorted = |sql: &str| {
            rows(&db, sql).map(|mut rows| {
                rows.sort();
                rows
            })
        };
        let got = sorted(sql);
        assert!(
            got.as_ref().is_ok_and(|rows| rows.len() == 2) && got == sorted(&general(sql)),
            "{sql}: {got:?}"
        );
    }
}

#[test]
fn a_table_whose_columns_changed_after_preparing_raises_schema_changed() {
    let _guard = serial();
    let db = Database::open("memory://grouped_output_blocks_retype").unwrap();
    for setup in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, a TEXT)",
        "INSERT INTO t VALUES (1, 1)",
        "INSERT INTO u VALUES (1, '01')",
    ] {
        db.execute(setup, ()).unwrap();
    }
    let sql = "SELECT k, k IN (SELECT u.a FROM u) FROM t GROUP BY k";
    // Twice, so the second run reads the plan the first one held
    for _ in 0..2 {
        assert_eq!(rows(&db, sql), rows(&db, &general(sql)), "{sql}: prepared");
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
    assert_eq!(rows(&db, sql), rows(&db, &general(sql)), "{sql}: run again");
}
