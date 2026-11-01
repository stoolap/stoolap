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

//! A NATURAL or USING join whose SELECT list or ORDER BY holds a subquery
//! runs it through the plan's blocks, its rows read with each merged pair
//! apart and shown merged: it answers as the general path does, hot and
//! sealed, and a subquery reads a merged column by its bare name

#![cfg(feature = "test-failpoints")]

mod oracle;

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{order_limit_paths_taken, reset_order_limit_paths};
use stoolap::Database;

// The prepared families' switch is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// l: id 1..30, k = id % 7; r: id 6..35, k = id % 9 (indexed); q: id
/// 2..40 by 2, k = id % 4; s: id 1..90, a = id % 31, b = id % 9
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "CREATE TABLE q (id INTEGER PRIMARY KEY, k INTEGER, z INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value + 5, (g.value + 5) % 9, g.value % 5 \
         FROM generate_series(1, 30) g",
        "INSERT INTO q SELECT g.value * 2, g.value % 4, g.value FROM generate_series(1, 20) g",
        "INSERT INTO s SELECT g.value, g.value % 31, g.value % 9 FROM generate_series(1, 90) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
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

fn databases(dir: &tempfile::TempDir) -> [(&'static str, Database); 2] {
    let hot = Database::open("memory://merged_join_output_blocks").unwrap();
    fill(&hot);
    let sealed = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    fill(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    [("hot", hot), ("sealed", sealed)]
}

const RECORDED: oracle::Answers = answers!("merged_join_output_blocks");

/// Each merging join and the join on the equalities it merges by; the last
/// two merge within a joined side
const JOINS: &[(&str, &str)] = &[
    ("l JOIN r USING (k)", "l JOIN r ON r.k = l.k"),
    ("l LEFT JOIN r USING (k)", "l LEFT JOIN r ON r.k = l.k"),
    ("l RIGHT JOIN r USING (k)", "l RIGHT JOIN r ON r.k = l.k"),
    ("l FULL JOIN r USING (k)", "l FULL JOIN r ON r.k = l.k"),
    ("l NATURAL JOIN r", "l JOIN r ON r.id = l.id AND r.k = l.k"),
    (
        "l JOIN r USING (id, k)",
        "l JOIN r ON r.id = l.id AND r.k = l.k",
    ),
    (
        "l JOIN r USING (k) JOIN q ON q.id = l.id",
        "l JOIN r ON r.k = l.k JOIN q ON q.id = l.id",
    ),
    (
        "l JOIN r USING (k) LEFT JOIN q USING (k)",
        "l JOIN r ON r.k = l.k LEFT JOIN q ON q.k = l.k",
    ),
];

/// SELECT lists naming every column through its table, then a star
const OUTPUTS: &[&str] = &[
    "l.id, r.id, (SELECT MAX(s.a) FROM s WHERE s.b = l.k) AS m",
    "l.id, r.id, EXISTS (SELECT 1 FROM s WHERE s.b = r.w AND s.a > l.g) AS e",
    "l.id, r.id, q.id, (SELECT MIN(s.a) FROM s WHERE s.b = r.k AND s.id > q.z) AS m",
    "*, (SELECT s.a FROM s WHERE s.id = r.id) AS m",
];

const TAILS: &[&str] = &[
    "ORDER BY l.id, r.id",
    "ORDER BY (SELECT MAX(s.a) FROM s WHERE s.b = l.k), l.id, r.id LIMIT 9",
];

#[test]
fn a_merging_join_answers_as_the_join_on_its_equalities() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (context, db) in databases(&dir) {
        for (join, on) in JOINS {
            let reads_q = join.contains(" q ");
            for output in OUTPUTS.iter().filter(|o| !o.starts_with('*')) {
                if output.contains("q.") && !reads_q {
                    continue;
                }
                for tail in TAILS {
                    let sql = format!("SELECT {output} FROM {join} {tail}");
                    let line = prepared_line(&db, &sql);
                    if line != "Prepared: yes" {
                        wrong.push(format!("{context}: {sql}: {line}"));
                    }
                    let (got, want) = (rows(&db, &sql), rows(&db, &sql.replace(join, on)));
                    if got != want || got.as_ref().is_ok_and(|r| r.len() < 2) {
                        wrong.push(format!("{context}: {sql}\n  got  {got:?}\n  want {want:?}"));
                    }
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
fn a_merging_join_runs_its_output_subqueries_as_the_general_path_does() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (context, db) in databases(&dir) {
        // The general path reads a joined side's merge otherwise
        for (join, _) in &JOINS[..6] {
            for output in OUTPUTS.iter().filter(|o| !o.contains("q.")) {
                for tail in TAILS {
                    let sql = format!("SELECT {output} FROM {join} {tail}");
                    let line = prepared_line(&db, &sql);
                    if line != "Prepared: yes" {
                        wrong.push(format!("{context}: {sql}: {line}"));
                    }
                    let answer = rows(&db, &sql);
                    let general = || oracle::general_answer(&sql, |twin| rows(&db, twin));
                    wrong.extend(
                        oracle::check_confirmed(&RECORDED, &sql, &answer, general)
                            .map(|why| format!("{context}: {why}")),
                    );
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
fn a_subquery_reads_a_merged_column_by_its_bare_name() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (context, db) in databases(&dir) {
        for join in [
            "l JOIN r USING (k)",
            "l LEFT JOIN r USING (k)",
            "l NATURAL JOIN r",
        ] {
            for output in [
                "l.id, r.id, (SELECT MAX(s.a) FROM s WHERE s.b = {k}) AS m",
                "*, (SELECT COUNT(*) FROM s WHERE s.b = {k} AND s.a > 9) AS c",
            ] {
                let sql = format!("SELECT {output} FROM {join} ORDER BY {{k}}, 1, 2, 3");
                let bare = sql.replace("{k}", "k");
                let twin = sql.replace("{k}", "COALESCE(l.k, r.k)");
                let line = prepared_line(&db, &bare);
                if line != "Prepared: yes" {
                    wrong.push(format!("{context}: {bare}: {line}"));
                }
                let (got, want) = (rows(&db, &bare), rows(&db, &twin));
                if got != want || got.as_ref().is_ok_and(|r| r.len() < 2) {
                    wrong.push(format!(
                        "{context}: {bare}\n  got  {got:?}\n  want {want:?}"
                    ));
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
fn a_bare_name_a_merged_pair_and_another_source_both_carry_is_ambiguous_in_the_output() {
    let _guard = serial();
    let db = Database::open("memory://merged_join_output_ambiguous").unwrap();
    fill(&db);
    for sql in [
        "SELECT l.id FROM l JOIN r USING (k) JOIN q ON q.id = l.id \
         WHERE k IN (SELECT s.b FROM s)",
        "SELECT k, (SELECT MAX(s.a) FROM s) FROM l JOIN r USING (k) JOIN q ON q.id = l.id",
        "SELECT l.id, (SELECT MAX(s.a) FROM s WHERE s.b = k) FROM l JOIN r USING (k) \
         JOIN q ON q.id = l.id",
    ] {
        let answer = rows(&db, sql);
        assert!(
            answer
                .as_ref()
                .is_err_and(|e| e.contains("column reference 'k' is ambiguous")),
            "{sql}: {answer:?}"
        );
    }
}

#[test]
fn an_ordered_subquery_in_a_merging_join_s_output_takes_a_prepared_path() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in databases(&dir) {
        for sql in [
            "SELECT l.id, (SELECT i.id FROM r i WHERE i.id > 5 AND i.id <> l.id \
             ORDER BY i.id LIMIT 1) FROM l JOIN r USING (k) ORDER BY 1, 2",
            "SELECT l.id, (SELECT i.id FROM r i WHERE i.k = 3 AND i.id <> l.id \
             ORDER BY i.w DESC, i.id LIMIT 1) FROM l NATURAL JOIN r ORDER BY 1, 2",
            "SELECT l.id, (SELECT i.id FROM r i ORDER BY i.k DESC, i.id LIMIT 1 OFFSET 1) \
             FROM l LEFT JOIN r USING (k) ORDER BY 1, 2",
        ] {
            reset_order_limit_paths();
            let answer = rows(&db, sql);
            let taken = order_limit_paths_taken();
            assert!(answer.is_ok_and(|r| r.len() > 1), "{context}: {sql}");
            assert!(
                taken.iter().all(|path| path.starts_with("prepared_")),
                "{context}: {sql}: took {taken:?}"
            );
        }
    }
}

#[test]
fn a_star_lines_up_with_the_items_and_qualified_stars_beside_it() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in databases(&dir) {
        // An alias after the star is read where the projection put it
        let sql = "SELECT *, l.id AS x, (SELECT MAX(s.a) FROM s WHERE s.id = r.id) AS c \
                   FROM l JOIN r USING (k) ORDER BY x + 0 DESC, r.id";
        let general = oracle::general_answer(sql, |twin| rows(&db, twin));
        assert_eq!(rows(&db, sql), general, "{context}: {sql}");
        // Each name stands over its own value
        let sql = "SELECT *, r.*, (SELECT 7) AS x FROM l JOIN r USING (k)";
        let result = db.query(sql, ()).unwrap();
        let columns = result.columns().to_vec();
        assert_eq!(
            columns,
            ["l.id", "k", "l.g", "r.id", "r.w", "id", "k", "w", "x"],
            "{context}: {sql}"
        );
        for row in result {
            let row = row.unwrap();
            assert_eq!(row.len(), columns.len(), "{context}: {sql}");
            assert_eq!(row.get_by_name::<i64>("x").unwrap(), 7, "{context}: {sql}");
        }
    }
}

#[test]
fn a_merging_join_that_groups_or_has_windows_stays_out_of_the_family() {
    let _guard = serial();
    let db = Database::open("memory://merged_join_output_grouped").unwrap();
    fill(&db);
    for sql in [
        "SELECT ROW_NUMBER() OVER () AS rn, (SELECT 1) AS x FROM l JOIN r USING (id)",
        "SELECT k, COUNT(*), (SELECT MAX(s.a) FROM s) AS m FROM l JOIN r USING (k) \
         GROUP BY k ORDER BY k",
    ] {
        assert_eq!(
            prepared_line(&db, sql),
            "Prepared: no (a NATURAL or USING join whose output holds a subquery)",
            "{sql}"
        );
        let answer = rows(&db, sql);
        assert!(
            answer.as_ref().is_ok_and(|r| r.len() > 1),
            "{sql}: {answer:?}"
        );
    }
}
