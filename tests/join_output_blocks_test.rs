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

//! A join of two tables whose SELECT list or ORDER BY holds a subquery
//! runs it through the plan's blocks on every join path: it answers as the
//! general path does, hot and sealed, and run again compiles nothing

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    bridge_compiles, general_path_hits, join_paths_taken, program_compiles, reset_join_paths,
    GeneralPath,
};
use stoolap::Database;

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// l: id 1..60, k = id % 7, g = (id - 1) / 10; r: id 1..80, k = id % 7
/// (indexed), w = id % 5; s: id 1..90, a = id % 31, b = id % 51
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 FROM generate_series(1, 60) g",
        "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5 FROM generate_series(1, 80) g",
        "INSERT INTO s SELECT g.value, g.value % 31, g.value % 51 FROM generate_series(1, 90) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// Subqueries a SELECT list or an ORDER BY holds, correlated with one side,
/// with both, or with neither
const FORMS: &[&str] = &[
    "(SELECT MAX(s.a) FROM s WHERE s.b = l.k)",
    "(SELECT s.a FROM s WHERE s.id = r.id)",
    "(SELECT MAX(s.a) FROM s WHERE s.b < 5)",
    "EXISTS (SELECT 1 FROM s WHERE s.a = r.w AND s.b = l.k)",
    "r.w IN (SELECT s.a FROM s WHERE s.b < 10)",
    "l.g > ANY (SELECT s.a FROM s WHERE s.b = r.w)",
    "l.id + (SELECT MIN(s.b) FROM s WHERE s.a = 3)",
];

/// Each join path; `{o}` is the SELECT item, `{k}` the ORDER BY keys
const SHAPES: &[(&str, [&str; 2])] = &[
    (
        "SELECT l.id, r.id, {o} FROM l JOIN r ON l.g = r.w ORDER BY {k}",
        ["hash", "hash"],
    ),
    (
        "SELECT l.id, r.id, {o} FROM l JOIN r ON l.k = r.k ORDER BY {k}",
        [
            if cfg!(feature = "test-filedb") {
                "hash"
            } else {
                "index_nested_loop"
            },
            "hash",
        ],
    ),
    (
        "SELECT l.id, r.id, {o} FROM l JOIN r ON l.k = r.id ORDER BY {k}",
        ["batch_index_nested_loop", "batch_index_nested_loop"],
    ),
    (
        "SELECT l.id, r.id, {o} FROM l JOIN r ON l.g < r.w ORDER BY {k}",
        ["nested_loop", "nested_loop"],
    ),
    (
        "SELECT l.id, r.id, {o} FROM l LEFT JOIN r ON l.g = r.w AND r.k = 2 ORDER BY {k}",
        ["hash", "hash"],
    ),
    (
        "SELECT l.id, r.id, {o} FROM l JOIN r ON l.g = r.w \
         WHERE r.k IN (SELECT s.a FROM s WHERE s.b < 30) ORDER BY {k}",
        ["hash", "hash"],
    ),
    (
        "SELECT l.id, r.id, {o} FROM l JOIN r ON l.g = r.w ORDER BY {k} LIMIT 20",
        ["hash", "hash"],
    ),
];

/// The paths a statement the blocks run never reaches
const GENERAL: &[GeneralPath] = &[
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::OrderByEvaluator,
    GeneralPath::ExistsSubquery,
    GeneralPath::InSubquery,
];

fn general_hits() -> Vec<usize> {
    GENERAL
        .iter()
        .map(|path| general_path_hits(*path))
        .collect()
}

/// The column names, then each row
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = vec![result.columns().join("|")];
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

/// The statement with both sides read as derived tables, out of the blocks
fn general(sql: &str) -> String {
    sql.replace(" FROM l ", " FROM (SELECT * FROM l) l ")
        .replace(" JOIN r ", " JOIN (SELECT * FROM r) r ")
}

#[test]
fn a_join_reads_its_output_s_subqueries_through_its_blocks() {
    let _guard = serial();
    let mut wrong = Vec::new();
    let mut cases = 0;
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://join_output_blocks").unwrap();
            fill(&db);
            db
        };
        for (shape, paths) in SHAPES {
            for form in FORMS {
                for (item, keys) in [
                    (form.to_string(), "1, 2".to_string()),
                    ("0".to_string(), format!("{form}, 1, 2")),
                ] {
                    cases += 1;
                    let sql = shape.replace("{o}", &item).replace("{k}", &keys);
                    let want = rows(&db, &general(&sql));
                    if !want.as_ref().is_ok_and(|rows| rows.len() > 1) {
                        wrong.push(format!("sealed {sealed}: {sql}: the general path {want:?}"));
                        continue;
                    }
                    let hits = general_hits();
                    let first = rows(&db, &sql);
                    let compiles = (bridge_compiles(), program_compiles());
                    reset_join_paths();
                    let again = rows(&db, &sql);
                    let taken = join_paths_taken();
                    let compiled = (
                        bridge_compiles() - compiles.0,
                        program_compiles() - compiles.1,
                    );
                    let reached = general_hits();
                    if reached != hits {
                        wrong.push(format!(
                            "sealed {sealed}: {sql}: reached the general path {hits:?} -> {reached:?}"
                        ));
                    }
                    if taken != [paths[usize::from(sealed)]] {
                        wrong.push(format!("sealed {sealed}: {sql}: took {taken:?}"));
                    }
                    let line = prepared_line(&db, &sql);
                    if line != "Prepared: yes" {
                        wrong.push(format!("sealed {sealed}: {sql}: {line}"));
                    }
                    if compiled != (0, 0) {
                        wrong.push(format!(
                            "sealed {sealed}: {sql}: run again compiled {compiled:?}"
                        ));
                    }
                    if first != want || again != want {
                        wrong.push(format!(
                            "sealed {sealed}: {sql}: gave {first:?} then {again:?}, the general path {want:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {cases} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn an_uncorrelated_output_subquery_is_read_before_the_join_reads_a_row() {
    let _guard = serial();
    let db = Database::open("memory://join_output_blocks_before").unwrap();
    fill(&db);
    for sql in [
        "SELECT l.id, (SELECT s.a FROM s WHERE s.b < 3) FROM l JOIN r ON l.k = r.k WHERE l.id < 0",
        "SELECT l.id FROM l JOIN r ON l.k = r.k WHERE l.id < 0 ORDER BY (SELECT s.a FROM s WHERE s.b < 3)",
        "SELECT l.id, r.w IN (SELECT s.a * 9223372036854775807 FROM s WHERE s.b = 33) \
         FROM l JOIN r ON l.k = r.k WHERE l.id < 0",
    ] {
        let got = rows(&db, sql);
        let want = rows(&db, &general(sql));
        assert!(
            got.as_ref().is_err() && got == want,
            "{sql}: {got:?}, the general path {want:?}"
        );
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    }
}

#[test]
fn a_correlated_output_subquery_raises_its_error_for_the_row_that_reaches_it() {
    let _guard = serial();
    let db = Database::open("memory://join_output_blocks_errors").unwrap();
    fill(&db);
    for sql in [
        "SELECT l.id, (SELECT s.a FROM s WHERE s.b = l.k) FROM l JOIN r ON l.k = r.k",
        "SELECT l.id FROM l JOIN r ON l.k = r.k ORDER BY (SELECT s.a FROM s WHERE s.b = r.w)",
        "SELECT l.id, (SELECT s.a FROM s WHERE s.b = l.k) FROM l JOIN r ON l.k = r.k WHERE l.id = 50",
    ] {
        let got = rows(&db, sql);
        let want = rows(&db, &general(sql));
        assert_eq!(got, want, "{sql}");
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    }
}

#[test]
fn a_join_output_s_subqueries_answer_in_every_context() {
    let _guard = serial();
    let db = Database::open("memory://join_output_blocks_contexts").unwrap();
    fill(&db);
    for (sql, prepared) in [
        (
            "SELECT *, (SELECT MAX(s.a) FROM s WHERE s.b = l.k) AS m FROM l JOIN r ON l.k = r.k \
             ORDER BY m DESC, l.id, r.id LIMIT 5",
            true,
        ),
        (
            "SELECT l.*, r.w FROM l JOIN r ON l.k = r.k \
             ORDER BY (SELECT s.a FROM s WHERE s.id = r.id), l.id, r.id LIMIT 7",
            true,
        ),
        (
            "SELECT * FROM l JOIN r ON l.k = r.k \
             ORDER BY EXISTS (SELECT 1 FROM s WHERE s.a = r.w AND s.b = l.k), l.id, r.id LIMIT 7",
            true,
        ),
        (
            "SELECT DISTINCT l.k, (SELECT MAX(s.a) FROM s WHERE s.b = l.k) FROM l JOIN r ON l.k = r.k \
             ORDER BY 1",
            true,
        ),
        (
            "SELECT l.id, CASE WHEN r.w IN (SELECT s.a FROM s WHERE s.b < 10) THEN -1 ELSE \
             (SELECT MAX(s.a) FROM s WHERE s.b = r.w) END FROM l JOIN r ON l.k = r.k \
             ORDER BY 1, 2",
            true,
        ),
        (
            "SELECT l.id AS x, r.id FROM l JOIN r ON l.k = r.k \
             ORDER BY (SELECT s.a FROM s WHERE s.id = x), 1, 2",
            true,
        ),
        (
            "SELECT l.id, r.id, (SELECT s.a FROM s WHERE s.id = l.id) FROM l JOIN r USING (k) \
             ORDER BY 1, 2",
            false,
        ),
        (
            "SELECT l.k, COUNT(*), (SELECT MAX(s.a) FROM s WHERE s.b = l.k) FROM l JOIN r ON l.k = r.k \
             GROUP BY l.k ORDER BY 1",
            false,
        ),
    ] {
        let got = rows(&db, sql);
        // The general path cannot sort by a subquery beside a qualified star
        let want = rows(&db, &general(&sql.replace("l.*", "l.id, l.k, l.g")));
        match (&got, &want) {
            (Err(got), Err(want)) => assert!(
                got.to_lowercase().contains("column 'x' not found")
                    && want.to_lowercase().contains("column 'x' not found"),
                "{sql}: {got}, the general path {want}"
            ),
            _ => assert_eq!(got, want, "{sql}"),
        }
        let line = prepared_line(&db, sql);
        assert_eq!(line == "Prepared: yes", prepared, "{sql}: {line}");
    }
}

#[test]
fn a_join_output_s_subqueries_read_parameters_and_the_transaction_s_writes() {
    let _guard = serial();
    let db = Database::open("memory://join_output_blocks_params").unwrap();
    fill(&db);
    let sql = "SELECT l.id, r.id, (SELECT MAX(s.a) FROM s WHERE s.b = l.k + $1) FROM l JOIN r \
               ON l.k = r.k ORDER BY (SELECT s.a FROM s WHERE s.id = r.id + $2), 1, 2 LIMIT 9";
    let read = |db: &Database, params: (i64, i64)| -> Vec<String> {
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
    let literal = |a: i64, b: i64| {
        rows(
            &db,
            &general(
                &sql.replace("$1", &a.to_string())
                    .replace("$2", &b.to_string()),
            ),
        )
        .unwrap()[1..]
            .to_vec()
    };
    for params in [(0, 0), (2, 5), (-1, 40)] {
        assert_eq!(read(&db, params), literal(params.0, params.1), "{params:?}");
    }
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO s VALUES (500, 99, 1)", ()).unwrap();
    let mut seen = tx
        .query(
            "SELECT l.id, (SELECT MAX(s.a) FROM s WHERE s.b = l.k) FROM l JOIN r ON l.k = r.k \
             WHERE l.id = 1",
            (),
        )
        .unwrap();
    let row = seen.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 99, "the transaction's own row");
    drop(seen);
    tx.rollback().unwrap();
}
