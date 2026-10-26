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

//! A join of two tables whose ON holds subqueries runs them as the plan's
//! blocks, inside every join algorithm: over the pair an index join or a
//! nested loop asks, over the joined row a hash join asks. It answers as
//! the general path does, hot and sealed, and run again compiles nothing

#![cfg(feature = "test-failpoints")]

use std::collections::BTreeSet;

use stoolap::test_failpoints::{
    bridge_compiles, join_paths_taken, prepared_strategy_runs, program_compiles, reset_join_paths,
    PreparedStrategy,
};
use stoolap::Database;

/// The strategy and path counts are process-wide: every test here runs
/// prepared blocks, so they take turns
static COUNTED: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn turn() -> std::sync::MutexGuard<'static, ()> {
    COUNTED.lock().unwrap_or_else(|e| e.into_inner())
}

/// l: id 1..30, k = id % 7, g = (id - 1) / 10; r: id 1..50, k = id % 7
/// (indexed), w = id % 5; s: id 1..90, a = id % 31, b = id % 51;
/// bl and br: 12000 rows each, u = id, v = id % 100
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5 FROM generate_series(1, 50) g",
        "INSERT INTO s SELECT g.value, g.value % 31, g.value % 51 FROM generate_series(1, 90) g",
        "CREATE TABLE bl (id INTEGER PRIMARY KEY, u INTEGER, v INTEGER)",
        "CREATE TABLE br (id INTEGER PRIMARY KEY, u INTEGER, v INTEGER)",
        "INSERT INTO bl SELECT g.value, g.value, g.value % 100 FROM generate_series(1, 12000) g",
        "INSERT INTO br SELECT g.value, g.value, g.value % 100 FROM generate_series(1, 12000) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// The (a, b) pairs of s
fn s_pairs() -> BTreeSet<(i64, i64)> {
    (1..=90).map(|id| (id % 31, id % 51)).collect()
}

fn row(left: Option<i64>, right: Option<i64>) -> String {
    let show = |v: Option<i64>| v.map_or("Null(Null)".to_string(), |v| format!("Integer({v})"));
    format!("{}|{}", show(left), show(right))
}

/// `l LEFT JOIN r ON l.g = r.w AND r.id IN (SELECT s.b FROM s WHERE s.a = l.g)`
fn left_in() -> Vec<String> {
    let s = s_pairs();
    let mut out = Vec::new();
    for l in 1..=30i64 {
        let g = (l - 1) / 10;
        let matched: Vec<i64> = (1..=50i64)
            .filter(|r| g == r % 5 && s.contains(&(g, *r)))
            .collect();
        match matched.is_empty() {
            true => out.push(row(Some(l), None)),
            false => out.extend(matched.into_iter().map(|r| row(Some(l), Some(r)))),
        }
    }
    out.sort();
    out
}

/// `l RIGHT JOIN r ON l.g = r.w AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)`
fn right_exists() -> Vec<String> {
    let s = s_pairs();
    let mut out = Vec::new();
    for r in 1..=50i64 {
        let matched: Vec<i64> = (1..=30i64)
            .filter(|l| (l - 1) / 10 == r % 5 && s.contains(&(*l, r)))
            .collect();
        match matched.is_empty() {
            true => out.push(row(None, Some(r))),
            false => out.extend(matched.into_iter().map(|l| row(Some(l), Some(r)))),
        }
    }
    out.sort();
    out
}

/// `l LEFT JOIN r ON l.g < r.w AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)`
fn left_nested_exists() -> Vec<String> {
    let s = s_pairs();
    let mut out = Vec::new();
    for l in 1..=30i64 {
        let matched: Vec<i64> = (1..=50i64)
            .filter(|r| (l - 1) / 10 < r % 5 && s.contains(&(l, *r)))
            .collect();
        match matched.is_empty() {
            true => out.push(row(Some(l), None)),
            false => out.extend(matched.into_iter().map(|r| row(Some(l), Some(r)))),
        }
    }
    out.sort();
    out
}

/// `sql` as a derived table's statement, which holds no plan, so the
/// general path answers it
fn general(sql: &str) -> String {
    format!("SELECT * FROM ({sql}) AS general")
}

enum Reference {
    /// The same statement with the ON's subquery moved to its WHERE, which
    /// the general path answers per joined row
    Moved(&'static str),
    Computed(fn() -> Vec<String>),
}

struct Case {
    sql: &'static str,
    path: &'static str,
    reference: Reference,
    /// A LIMIT without ORDER BY may return any of the reference's rows
    limit: Option<usize>,
}

const CASES: &[Case] = &[
    Case {
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        path: "hash",
        reference: Reference::Moved(
            "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w WHERE EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        ),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND l.id < (SELECT MAX(s.b) FROM s WHERE s.a = r.w)",
        path: "hash",
        reference: Reference::Moved(
            "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w WHERE l.id < (SELECT MAX(s.b) FROM s WHERE s.a = r.w)",
        ),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND l.id > ANY (SELECT s.b FROM s WHERE s.a = r.w)",
        path: "hash",
        reference: Reference::Moved(
            "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w WHERE l.id > ANY (SELECT s.b FROM s WHERE s.a = r.w)",
        ),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND l.id + r.id > (SELECT MIN(s.b) FROM s WHERE s.a = 5)",
        path: "hash",
        reference: Reference::Moved(
            "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w WHERE l.id + r.id > (SELECT MIN(s.b) FROM s WHERE s.a = 5)",
        ),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l LEFT JOIN r ON l.g = r.w AND r.id IN (SELECT s.b FROM s WHERE s.a = l.g)",
        path: "hash",
        reference: Reference::Computed(left_in),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l RIGHT JOIN r ON l.g = r.w AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        path: "hash",
        reference: Reference::Computed(right_exists),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND l.id < (SELECT MAX(s.b) FROM s WHERE s.a = r.w) LIMIT 7",
        path: "limit_hash",
        reference: Reference::Moved(
            "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w WHERE l.id < (SELECT MAX(s.b) FROM s WHERE s.a = r.w)",
        ),
        limit: Some(7),
    },
    Case {
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.g < r.w AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        path: "nested_loop",
        reference: Reference::Moved(
            "SELECT l.id, r.id FROM l JOIN r ON l.g < r.w WHERE EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        ),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l LEFT JOIN r ON l.g < r.w AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        path: "nested_loop",
        reference: Reference::Computed(left_nested_exists),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.k = r.id AND r.w IN (SELECT s.b FROM s WHERE s.a = l.id)",
        path: "batch_index_nested_loop",
        reference: Reference::Moved(
            "SELECT l.id, r.id FROM l JOIN r ON l.k = r.id WHERE r.w IN (SELECT s.b FROM s WHERE s.a = l.id)",
        ),
        limit: None,
    },
    Case {
        sql: "SELECT r.id, l.id FROM r JOIN l ON r.id = l.k AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b > r.w) WHERE l.g = 1",
        path: "batch_index_nested_loop",
        reference: Reference::Moved(
            "SELECT r.id, l.id FROM r JOIN l ON r.id = l.k WHERE l.g = 1 AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b > r.w)",
        ),
        limit: None,
    },
    Case {
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.k = r.k AND r.id > (SELECT MIN(s.b) FROM s WHERE s.a = l.id) LIMIT 30",
        path: "index_nested_loop",
        reference: Reference::Moved(
            "SELECT l.id, r.id FROM l JOIN r ON l.k = r.k WHERE r.id > (SELECT MIN(s.b) FROM s WHERE s.a = l.id)",
        ),
        limit: Some(30),
    },
    Case {
        sql: "SELECT COUNT(*), SUM(bl.id) FROM bl JOIN br ON bl.u = br.u AND br.v IN (SELECT s.b FROM s WHERE s.a < 9)",
        path: "parallel_hash",
        reference: Reference::Moved(
            "SELECT COUNT(*), SUM(bl.id) FROM bl JOIN br ON bl.u = br.u WHERE br.v IN (SELECT s.b FROM s WHERE s.a < 9)",
        ),
        limit: None,
    },
];

/// The first two values of each row of `sql` as sorted text, or the error
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(format!(
            "{:?}|{:?}",
            row.get_value(0).unwrap(),
            row.get_value(1).unwrap()
        ));
    }
    out.sort();
    Ok(out)
}

/// Whether every row of `part` is in `whole`, as often as it is there
fn contained(part: &[String], whole: &[String]) -> bool {
    let mut left = whole.to_vec();
    part.iter()
        .all(|row| match left.iter().position(|w| w == row) {
            Some(at) => {
                left.swap_remove(at);
                true
            }
            None => false,
        })
}

#[test]
fn a_join_on_runs_its_subqueries_as_blocks() {
    let _turn = turn();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://join_on_blocks").unwrap();
            fill(&db);
            db
        };
        for case in CASES {
            let first = rows(&db, case.sql);
            let bridge = bridge_compiles();
            let programs = program_compiles();
            reset_join_paths();
            let again = rows(&db, case.sql);
            let taken = join_paths_taken();
            let compiled = (bridge_compiles() - bridge, program_compiles() - programs);
            let want = match &case.reference {
                Reference::Moved(sql) => rows(&db, &general(sql)),
                Reference::Computed(rows) => Ok(rows()),
            };
            if taken != [case.path] {
                wrong.push(format!(
                    "sealed {sealed}: {}: took {taken:?}, expected {}",
                    case.sql, case.path
                ));
            }
            if compiled != (0, 0) {
                wrong.push(format!(
                    "sealed {sealed}: {}: run again, compiled {compiled:?}",
                    case.sql
                ));
            }
            let answers = |got: &Result<Vec<String>, String>| match (got, &want, case.limit) {
                (Ok(got), Ok(want), Some(limit)) => {
                    got.len() == limit.min(want.len()) && contained(got, want)
                }
                (got, want, None) => got == want,
                _ => false,
            };
            if !answers(&first) || !answers(&again) {
                wrong.push(format!(
                    "sealed {sealed}: {}: gave {first:?} then {again:?}, the reference {want:?}",
                    case.sql
                ));
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
fn an_on_block_that_errors_errors_as_the_general_path_does() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_error").unwrap();
    fill(&db);
    let want = rows(
        &db,
        &general(
            "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w \
             WHERE l.id < (SELECT s.b FROM s WHERE s.a = r.w)",
        ),
    );
    assert!(
        want.as_ref()
            .is_err_and(|e| e.contains("more than one row")),
        "the general path errors: {want:?}"
    );
    let sql = "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND l.id < (SELECT s.b FROM s WHERE s.a = r.w)";
    for _ in 0..2 {
        assert_eq!(rows(&db, sql), want, "{sql}");
    }
}

#[test]
fn an_uncorrelated_on_block_no_pair_reaches_is_not_read() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_lazy").unwrap();
    fill(&db);
    let sql =
        "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND l.id < 0 AND l.id < (SELECT s.b FROM s)";
    for _ in 0..2 {
        assert_eq!(rows(&db, sql), Ok(Vec::new()), "no pair reads the block");
    }
}

#[test]
fn an_on_block_binds_a_dotted_column_by_position() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_dotted").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, \"x.y\" INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "CREATE TABLE s (v INTEGER PRIMARY KEY)",
        "INSERT INTO l VALUES (1, 10), (2, 20)",
        "INSERT INTO r VALUES (1), (2)",
        "INSERT INTO s VALUES (11)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT l.id, r.id FROM l JOIN r ON l.id = r.id \
               AND EXISTS (SELECT 1 FROM s WHERE s.v = l.\"x.y\" + r.id)";
    for _ in 0..2 {
        assert_eq!(
            rows(&db, sql),
            Ok(vec!["Integer(1)|Integer(1)".to_string()]),
            "{sql}"
        );
    }
}

#[test]
fn an_on_block_reads_the_transaction_s_own_writes() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_tx").unwrap();
    fill(&db);
    let sql = "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w \
               AND EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id + 1000)";
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO s VALUES (1000, 3, 1005)", ())
        .unwrap();
    let got: Vec<String> = tx
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            format!(
                "{:?}|{:?}",
                row.get_value(0).unwrap(),
                row.get_value(1).unwrap()
            )
        })
        .collect();
    tx.rollback().unwrap();
    assert_eq!(
        got,
        vec!["Integer(3)|Integer(5)".to_string()],
        "the block sees the row the transaction inserted"
    );
}

/// Tuples stay outside the subquery family: an ON with a subquery beside
/// one is not prepared
const TUPLE_FORMS: &[&str] = &[
    "(l.id, r.id) IN ((1, 1), (6, 1), (11, 11))",
    "(l.g, l.k) = (r.w, r.k)",
];

const PAIR_FORMS: &[&str] = &[
    "l.id + r.id BETWEEN 5 AND 40",
    "l.id NOT BETWEEN 3 AND 5",
    "CASE WHEN l.k > 3 THEN r.w ELSE l.g END = r.w",
    "CAST(l.id AS TEXT) LIKE '1%'",
    "COALESCE(l.k, 0) <= r.k",
    "l.k IS NOT NULL AND r.k IS NOT NULL",
    "NOT (l.id = r.id)",
    "l.k IN (1, 2, 3)",
    "l.id % 3 = r.id % 3",
    "ABS(l.k - r.k) < 2",
];

const SIDE_FORMS: &[&str] = &[
    "(l.k, l.g) IN ((1, 0), (2, 1), (3, 2))",
    "(l.k, l.g) = (1, 0)",
    "l.id NOT BETWEEN 3 AND 5",
    "CASE WHEN l.k > 3 THEN 1 ELSE 0 END = 1",
    "CAST(l.id AS TEXT) LIKE '1%'",
    "l.k IN (1, 2, 3)",
];

#[test]
fn a_join_rewrites_every_condition_form_beside_its_blocks() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_forms").unwrap();
    fill(&db);
    let exists = "EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)";
    let mut shapes: Vec<(String, String)> = Vec::new();
    for form in TUPLE_FORMS.iter().chain(PAIR_FORMS) {
        let moved = format!(
            "SELECT l.id, r.id FROM {{l}} JOIN {{r}} ON l.g = r.w AND {form} WHERE {exists}"
        );
        shapes.push((
            format!("SELECT l.id, r.id FROM {{l}} JOIN {{r}} ON {form} WHERE {exists}"),
            format!("SELECT l.id, r.id FROM {{l}} JOIN {{r}} ON {form} WHERE {exists}"),
        ));
        shapes.push((moved.clone(), moved.clone()));
        if TUPLE_FORMS.contains(form) {
            continue;
        }
        shapes.push((
            format!("SELECT l.id, r.id FROM {{l}} JOIN {{r}} ON l.g = r.w AND {form} AND {exists}"),
            moved,
        ));
    }
    for form in SIDE_FORMS {
        let sql = format!(
            "SELECT l.id, r.id FROM {{l}} JOIN {{r}} ON l.g = r.w WHERE {form} AND {exists}"
        );
        shapes.push((sql.clone(), sql));
    }
    for using in ["USING (k)", "USING (k, id)"] {
        let sql = format!("SELECT l.id, r.id FROM {{l}} JOIN {{r}} {using} WHERE {exists}");
        shapes.push((sql.clone(), sql));
    }
    let sql = format!("SELECT l.id, r.id FROM {{l}} NATURAL JOIN {{r}} WHERE {exists}");
    shapes.push((sql.clone(), sql));
    let mut wrong = Vec::new();
    for (shape, reference) in &shapes {
        let sql = shape.replace("{l}", "l").replace("{r}", "r");
        let want = rows(
            &db,
            &general(&reference.replace("{l}", "l").replace("{r}", "r")),
        );
        if want.is_err() {
            wrong.push(format!("the reference of {sql} fails: {want:?}"));
            continue;
        }
        let got = rows(&db, &sql);
        if got != want {
            wrong.push(format!("{sql}: gave {got:?}, the reference {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} wrong:\n{}",
        wrong.len(),
        shapes.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_using_join_keeps_its_keys_beside_its_blocks() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_using").unwrap();
    fill(&db);
    for sql in [
        "SELECT l.id, r.id FROM l JOIN r USING (k) WHERE EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)",
        "SELECT l.id, r.id FROM l NATURAL JOIN r WHERE EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.k)",
    ] {
        reset_join_paths();
        rows(&db, sql).unwrap();
        assert_eq!(join_paths_taken(), ["hash"], "{sql}");
    }
}

#[test]
fn a_where_block_after_the_join_chooses_its_strategy_by_the_joined_rows() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_strategy").unwrap();
    for sql in [
        "CREATE TABLE j1 (id INTEGER PRIMARY KEY, g INTEGER, k INTEGER)",
        "CREATE TABLE j2 (id INTEGER PRIMARY KEY, g INTEGER, k INTEGER)",
        "CREATE TABLE s2 (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE INDEX s2_k ON s2 (k)",
        "INSERT INTO j1 SELECT g.value, 0, 1 FROM generate_series(1, 40) g",
        "INSERT INTO j2 SELECT g.value, 0, 1 FROM generate_series(1, 25) g",
        "INSERT INTO s2 SELECT g.value, 1 FROM generate_series(1, 10000) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT j1.id, j2.id FROM j1 JOIN j2 ON j1.g = j2.g \
               WHERE EXISTS (SELECT 1 FROM s2 WHERE s2.k = j1.k)";
    let probes = prepared_strategy_runs(PreparedStrategy::Probe);
    let keyed = prepared_strategy_runs(PreparedStrategy::KeyedSet);
    assert_eq!(rows(&db, sql).map(|rows| rows.len()), Ok(1000));
    assert_eq!(
        (
            prepared_strategy_runs(PreparedStrategy::Probe) - probes,
            prepared_strategy_runs(PreparedStrategy::KeyedSet) - keyed
        ),
        (0, 1),
        "1000 joined rows over 10000 rows of s2 read the keyed set once, not a probe per row"
    );
}

#[test]
fn a_natural_join_binds_a_shared_dotted_column_beside_its_blocks() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_natural_dotted").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, \"x.y\" INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, \"x.y\" INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY)",
        "INSERT INTO l VALUES (1, 7), (2, 8)",
        "INSERT INTO r VALUES (1, 7), (2, 9)",
        "INSERT INTO s VALUES (2), (4)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT l.id, r.id FROM l NATURAL JOIN r \
               WHERE EXISTS (SELECT 1 FROM s WHERE s.id = l.id + r.id)";
    for _ in 0..2 {
        reset_join_paths();
        assert_eq!(
            rows(&db, sql),
            Ok(vec!["Integer(1)|Integer(1)".to_string()]),
            "{sql}"
        );
        assert_eq!(join_paths_taken(), ["hash"], "{sql}");
    }
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
fn explain_says_whether_a_join_runs_its_subqueries_as_blocks() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_explain").unwrap();
    fill(&db);
    let exists = "EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = r.id)";
    let cases = [
        (
            format!("SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND {exists}"),
            "Prepared: yes".to_string(),
        ),
        (
            format!("SELECT l.id, r.id FROM l LEFT JOIN r ON l.g = r.w AND {exists}"),
            "Prepared: yes".to_string(),
        ),
        (
            format!("SELECT l.id, r.id FROM l JOIN r ON l.g = r.w WHERE {exists}"),
            "Prepared: yes".to_string(),
        ),
        (
            format!("SELECT l.id, r.id FROM l JOIN r USING (k) WHERE {exists}"),
            "Prepared: yes".to_string(),
        ),
        (
            "SELECT l.id FROM l JOIN r ON r.id = l.id WHERE l.k > (SELECT s.b FROM s WHERE s.id = l.id)"
                .to_string(),
            "Prepared: yes".to_string(),
        ),
        (
            format!("SELECT l.id, r.id FROM l JOIN (SELECT * FROM r) r ON l.g = r.w WHERE {exists}"),
            "Prepared: yes".to_string(),
        ),
        (
            "WITH c AS (SELECT * FROM r) SELECT l.id, c.id FROM l JOIN c ON l.g = c.w \
             WHERE EXISTS (SELECT 1 FROM s WHERE s.a = l.id AND s.b = c.id)"
                .to_string(),
            "Prepared: yes".to_string(),
        ),
        (
            "WITH c AS (SELECT * FROM s) SELECT l.id, r.id FROM l JOIN r ON l.g = r.w \
             WHERE EXISTS (SELECT 1 FROM c WHERE c.a = l.id AND c.b = r.id) \
             AND EXISTS (SELECT 1 FROM c WHERE c.a = r.id)"
                .to_string(),
            "Prepared: yes".to_string(),
        ),
        (
            format!("SELECT l.id, r.id FROM l JOIN r ON l.g = r.w JOIN s ON s.id = l.id WHERE {exists}"),
            "Prepared: no (a side that is itself a join)".to_string(),
        ),
        (
            format!(
                "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w \
                 WHERE l.id > (SELECT MIN(s.b) FROM s) AND {exists}"
            ),
            "Prepared: yes".to_string(),
        ),
        (
            format!("SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND (l.g, l.k) = (r.w, r.k) AND {exists}"),
            "Prepared: no (an expression form outside the family)".to_string(),
        ),
    ];
    let mut wrong = Vec::new();
    for (sql, want) in &cases {
        let got = prepared_line(&db, sql);
        if got != *want {
            wrong.push(format!("{sql}: {got}, expected {want}"));
        }
        let ran = rows(&db, sql).and_then(|_| {
            let compiles = (bridge_compiles(), program_compiles());
            rows(&db, sql)?;
            Ok((bridge_compiles(), program_compiles()) == compiles)
        });
        if got == "Prepared: yes" && ran != Ok(true) {
            wrong.push(format!("{sql}: says yes, run again it compiled ({ran:?})"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn explain_says_no_for_a_join_outside_the_statement_s_own_select() {
    let _turn = turn();
    let db = Database::open("memory://join_on_blocks_explain_sites").unwrap();
    fill(&db);
    db.execute("CREATE TABLE tgt (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    let join = "SELECT l.id, r.k FROM l JOIN r ON l.id = r.id \
                WHERE EXISTS (SELECT 1 FROM s WHERE s.id = l.id)";
    let outside = "Prepared: no (a join outside the statement's own SELECT)";
    let lines = |sql: &str| -> Vec<String> {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap().trim().to_string())
            .filter(|line| line.starts_with("Prepared:"))
            .collect()
    };
    let selects = [
        (format!("SELECT 0, 0 UNION ALL {join}"), outside),
        (format!("{join} UNION ALL SELECT 0, 0"), "Prepared: yes"),
        (format!("SELECT d.id, d.k FROM ({join}) d"), outside),
    ];
    let mut wrong = Vec::new();
    for (sql, want) in &selects {
        let got = lines(sql);
        if got != [*want] {
            wrong.push(format!("{sql}: {got:?}, expected {want}"));
        }
        let ran = rows(&db, sql).and_then(|_| {
            let compiles = (bridge_compiles(), program_compiles());
            rows(&db, sql)?;
            Ok((bridge_compiles(), program_compiles()) == compiles)
        });
        if ran != Ok(*want == "Prepared: yes") {
            wrong.push(format!(
                "{sql}: says {want}, run again it compiled nothing: {ran:?}"
            ));
        }
    }
    let insert = format!("INSERT INTO tgt {join}");
    if lines(&insert) != [outside] {
        wrong.push(format!("{insert}: {:?}", lines(&insert)));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
