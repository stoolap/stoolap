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

//! Every operator that produces a join's rows answers as another operator
//! does for the same join, and every one registered has a case here that it
//! answers alone, without falling back to another

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{join_paths, join_paths_taken, reset_join_paths};
use stoolap::Database;

/// l: id 1..30, k = id % 7, g = (id - 1) / 10; r: id 1..50, k = id % 7
/// (indexed), w = id % 5; bl and br: 12000 rows each, u = id unindexed,
/// v = id % 100
fn open() -> Database {
    let db = Database::open("memory://join_path_equivalence").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5 FROM generate_series(1, 50) g",
        "CREATE TABLE bl (id INTEGER PRIMARY KEY, u INTEGER, v INTEGER)",
        "CREATE TABLE br (id INTEGER PRIMARY KEY, u INTEGER, v INTEGER)",
        "INSERT INTO bl SELECT g.value, g.value, g.value % 100 FROM generate_series(1, 12000) g",
        "INSERT INTO br SELECT g.value, g.value, g.value % 100 FROM generate_series(1, 12000) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The rows the reference answers with
enum Reference {
    /// A statement, with the one path it runs
    Sql(&'static str, &'static str),
    /// Rows computed here from the tables' contents
    Computed(fn() -> Vec<String>),
}

/// A path, a statement it alone answers, the reference for it, and the
/// statement's LIMIT: a LIMIT without ORDER BY may return any of the
/// reference's rows, so they are checked as part of it with their counts
struct Case {
    path: &'static str,
    sql: &'static str,
    reference: Reference,
    limit: Option<usize>,
}

/// `l JOIN r ON l.g < r.w`, from the contents `open` gives the tables
fn l_g_below_r_w() -> Vec<String> {
    let mut out = Vec::new();
    for l in 1..=30i64 {
        for r in 1..=50i64 {
            if (l - 1) / 10 < r % 5 {
                out.push(format!("Integer({l})|Integer({r})"));
            }
        }
    }
    out.sort();
    out
}

const CASES: &[Case] = &[
    Case {
        path: "index_nested_loop",
        sql: "SELECT l.id, r.id, r.w FROM l JOIN r ON l.k = r.k LIMIT 40",
        reference: Reference::Sql(
            "SELECT l.id, r.id, r.w FROM l JOIN r ON l.k + 0 = r.k + 0",
            "nested_loop",
        ),
        limit: Some(40),
    },
    Case {
        path: "batch_index_nested_loop",
        sql: "SELECT l.id, r.id, r.w FROM l JOIN r ON l.k = r.id",
        reference: Reference::Sql(
            "SELECT l.id, r.id, r.w FROM l JOIN r ON l.k + 0 = r.id + 0",
            "nested_loop",
        ),
        limit: None,
    },
    Case {
        path: "grouped_index_join",
        sql: "SELECT l.id, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k = r.k GROUP BY l.id LIMIT 5",
        reference: Reference::Sql(
            "SELECT l.id, COUNT(*), SUM(r.w) FROM l JOIN r ON l.k + 0 = r.k + 0 GROUP BY l.id",
            "nested_loop",
        ),
        limit: Some(5),
    },
    Case {
        path: "limit_hash",
        sql: "SELECT l.id, r.id, r.w FROM l JOIN r ON l.g = r.w LIMIT 7",
        reference: Reference::Sql(
            "SELECT l.id, r.id, r.w FROM l JOIN r ON l.g + 0 = r.w + 0",
            "nested_loop",
        ),
        limit: Some(7),
    },
    Case {
        path: "parallel_hash",
        sql: "SELECT COUNT(*), SUM(bl.id), SUM(br.v) FROM bl JOIN br ON bl.u = br.u",
        // u equals id in both tables, so the primary key join holds the same pairs
        reference: Reference::Sql(
            "SELECT COUNT(*), SUM(bl.id), SUM(br.v) FROM bl JOIN br ON bl.id = br.id",
            "batch_index_nested_loop",
        ),
        limit: None,
    },
    Case {
        path: "hash",
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w",
        reference: Reference::Sql(
            "SELECT l.id, r.id FROM l JOIN r ON l.g + 0 = r.w + 0",
            "nested_loop",
        ),
        limit: None,
    },
    Case {
        path: "nested_loop",
        sql: "SELECT l.id, r.id FROM l JOIN r ON l.g < r.w",
        reference: Reference::Computed(l_g_below_r_w),
        limit: None,
    },
];

/// The column names of `sql` and its rows as sorted text, or the error
fn rows(db: &Database, sql: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = result.columns().to_vec();
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
    out.sort();
    Ok((columns, out))
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
fn every_join_path_answers_as_another_path_does() {
    let db = open();
    let mut wrong = Vec::new();
    let mut covered: Vec<&str> = Vec::new();
    for case in CASES {
        reset_join_paths();
        let got = rows(&db, case.sql);
        let taken = join_paths_taken();
        if taken != [case.path] {
            wrong.push(format!(
                "{}: took {taken:?}, expected only {}",
                case.sql, case.path
            ));
        }
        covered.extend(taken);
        let (columns, want) = match &case.reference {
            Reference::Sql(reference, reference_path) => {
                reset_join_paths();
                let want = rows(&db, reference);
                let reference_taken = join_paths_taken();
                if reference_taken != [*reference_path] {
                    wrong.push(format!(
                        "{reference}: took {reference_taken:?}, expected only {reference_path}"
                    ));
                }
                match want {
                    Ok((columns, want)) => (Some(columns), want),
                    Err(e) => {
                        wrong.push(format!("{reference}: {e}"));
                        continue;
                    }
                }
            }
            Reference::Computed(rows) => (None, rows()),
        };
        let (got_columns, got) = match got {
            Ok(got) => got,
            Err(e) => {
                wrong.push(format!("{}: {e}", case.sql));
                continue;
            }
        };
        if columns.is_some_and(|columns| columns != got_columns) {
            wrong.push(format!("{}: columns {got_columns:?}", case.sql));
        }
        let answers = match case.limit {
            Some(limit) => got.len() == limit.min(want.len()) && contained(&got, &want),
            None => got == want,
        };
        if !answers {
            wrong.push(format!(
                "{}: got {got:?}, the reference has {want:?}",
                case.sql
            ));
        }
    }
    for path in join_paths() {
        if !covered.contains(&path) {
            wrong.push(format!("no case takes the {path} path"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
