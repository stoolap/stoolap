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

//! A top-level `column [NOT] IN (block)` or `column op ANY|ALL (block)` over
//! an uncorrelated block is read before the scan and taken by the scan: the
//! statement answers as the program does when the conjunct is asked of each
//! row (`OR id < 0`, FALSE for every row), for every member set and column
//! type, hot and sealed

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{prepared_strategy_runs, PreparedStrategy};
use stoolap::Database;

/// t: id 1..40, i INTEGER (NULL every 9th), f FLOAT, s TEXT; m holds the
/// member sets, picked by `k`: 1 numbers, 2 numbers and a NULL, 3 NULL
/// only, 4 TEXT numbers, 5 no row at all
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, s TEXT)",
        "INSERT INTO t SELECT g.value, CASE WHEN g.value % 9 = 0 THEN NULL ELSE g.value % 11 END, \
         (g.value % 7) * 1.5, CAST(g.value % 5 AS TEXT) FROM generate_series(1, 40) g",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, w TEXT)",
        "INSERT INTO m VALUES (1, 1, 3, '3'), (2, 1, 7, '7'), (3, 2, 4, '4'), (4, 2, NULL, NULL), \
         (5, 3, NULL, NULL), (6, 4, 2, '2'), (7, 4, 9, '9')",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

const COLUMNS: &[&str] = &["i", "f", "s"];

const TESTS: &[&str] = &[
    "IN", "NOT IN", "= ANY", "<> ALL", "> ANY", "< ALL", ">= ALL", "<= ANY", "<> ANY", "= ALL",
];

const BLOCKS: &[&str] = &[
    "SELECT v FROM m WHERE k = 1",
    "SELECT v FROM m WHERE k = 2",
    "SELECT v FROM m WHERE k = 3",
    "SELECT w FROM m WHERE k = 4",
    "SELECT v FROM m WHERE k = 5",
];

fn ids(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(
            row.map_err(|e| e.to_string())?
                .get::<i64>(0)
                .map_err(|e| e.to_string())?,
        );
    }
    out.sort();
    Ok(out)
}

#[test]
fn a_scan_takes_a_set_conjunct_as_the_general_path_answers_it() {
    let mut wrong = Vec::new();
    let mut taken = 0;
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://prepared_set_scan").unwrap();
            fill(&db);
            db
        };
        for column in COLUMNS {
            for test in TESTS {
                for block in BLOCKS {
                    let condition = format!("{column} {test} ({block})");
                    for shape in [
                        "SELECT id FROM {t} WHERE {c}",
                        "SELECT id FROM {t} WHERE id > 5 AND {c}",
                        "SELECT id FROM {t} WHERE {c} ORDER BY id LIMIT 4 OFFSET 2",
                    ] {
                        let plain = shape.replace("{c}", &condition).replace("{t}", "t");
                        let per_row = shape
                            .replace("{c}", &format!("({condition} OR id < 0)"))
                            .replace("{t}", "t");
                        let before = prepared_strategy_runs(PreparedStrategy::ScanTakesMembers);
                        let got = ids(&db, &plain);
                        taken +=
                            prepared_strategy_runs(PreparedStrategy::ScanTakesMembers) - before;
                        let want = ids(&db, &per_row);
                        if got != want {
                            wrong.push(format!(
                                "sealed {sealed}: {plain}: {got:?}, per row {want:?}"
                            ));
                        }
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
    assert!(taken > 0, "no statement reached the set scan");
}

#[test]
fn a_block_error_belongs_to_the_row_that_needs_the_answer() {
    let db = Database::open("memory://prepared_set_scan_errors").unwrap();
    fill(&db);
    db.execute("CREATE TABLE e (id INTEGER PRIMARY KEY, i INTEGER)", ())
        .unwrap();
    for test in ["IN", "NOT IN", "> ANY", ">= ALL", "= ANY", "<> ALL"] {
        let block = "(SELECT v * 9223372036854775807 FROM m WHERE k = 1)";
        let empty = format!("SELECT id FROM e WHERE i {test} {block}");
        assert_eq!(ids(&db, &empty), Ok(vec![]), "{empty}: no row needs it");
        let rows = format!("SELECT id FROM t WHERE i {test} {block}");
        let got = ids(&db, &rows);
        let per_row = ids(&db, &rows.replace("WHERE i", "WHERE id < 0 OR i"));
        assert!(
            got.as_ref().is_err_and(|e| e.contains("overflow")),
            "{rows}: {got:?}"
        );
        assert_eq!(got, per_row, "{rows}: as it errs asked of each row");
    }
}

#[test]
fn a_set_conjunct_the_scan_takes_reaches_the_scan_strategy() {
    let db = Database::open("memory://prepared_set_scan_reached").unwrap();
    fill(&db);
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, Score INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 2), (2, 8)", ())
        .unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM u WHERE Score > ANY (SELECT v FROM m WHERE k = 1)"
        ),
        Ok(vec![2])
    );
    for sql in [
        "SELECT id FROM u WHERE Score > ANY (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i > ANY (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE f <= ALL (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i NOT IN (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i = ANY (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i <> ALL (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i > ANY (SELECT v FROM m WHERE k = 5)",
    ] {
        let before = prepared_strategy_runs(PreparedStrategy::ScanTakesMembers);
        ids(&db, sql).unwrap();
        assert!(
            prepared_strategy_runs(PreparedStrategy::ScanTakesMembers) > before,
            "{sql}: the scan took the set"
        );
    }
}
