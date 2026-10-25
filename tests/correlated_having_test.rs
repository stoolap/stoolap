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

//! A subquery that reads the outer row only in its HAVING is correlated on
//! every path that asks whether a WHERE is

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| e.to_string())?.get::<i64>(0).unwrap());
    }
    out.sort_unstable();
    Ok(out)
}

fn create(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        "INSERT INTO t VALUES (1, 1, 'a'), (2, 2, 'b'), (3, 3, 'c')",
        "CREATE INDEX t_k ON t(k)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Each subquery beside one that differs only where it reads the outer
/// row, with the answer of each
const TWINS: [(&str, &[i64], &str, &[i64]); 2] = [
    (
        "(SELECT COUNT(*) FROM u HAVING COUNT(*) > 1) = 2",
        &[1, 2, 3],
        "(SELECT COUNT(*) FROM u HAVING COUNT(*) > t.k) = 2",
        &[1],
    ),
    (
        "(SELECT u.id FROM u ORDER BY (u.id - 1) * (u.id - 1) LIMIT 1) = 2",
        &[],
        "(SELECT u.id FROM u ORDER BY (u.id - t.k) * (u.id - t.k) LIMIT 1) = 2",
        &[2, 3],
    ),
];

fn check_twins(db: &Database, correlated_first: bool) {
    let mut failed = Vec::new();
    for (plain, plain_ids, correlated, correlated_ids) in TWINS {
        let mut order = [(plain, plain_ids), (correlated, correlated_ids)];
        if correlated_first {
            order.reverse();
        }
        for (subquery, want) in order {
            let sql = format!("SELECT id FROM t WHERE s || 'x' <> 'zx' AND {subquery}");
            let got = ids(db, &sql);
            if got.as_deref() != Ok(want) {
                failed.push(format!("{sql}: {got:?}"));
            }
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn a_correlated_subquery_after_its_uncorrelated_twin_reads_the_outer_row() {
    check_twins(&create("memory://correlated_after_its_twin"), false);
}

#[test]
fn an_uncorrelated_subquery_after_its_correlated_twin_answers_alike() {
    check_twins(&create("memory://uncorrelated_after_its_twin"), true);
}

#[test]
fn a_subquery_reading_the_outer_row_in_its_having_is_correlated() {
    let db = create("memory://a_subquery_reading_the_outer_row_in_its_having");
    let having = "(SELECT COUNT(*) FROM u HAVING COUNT(*) > t.k) = 2";
    let mut failed = Vec::new();
    for filter in [
        "",
        "s || 'x' <> 'zx' AND ",
        "id IN (1, 2, 3) AND s || 'x' <> 'zx' AND ",
        "k IN (1, 2, 3) AND s || 'x' <> 'zx' AND ",
        "k IN (SELECT g FROM u) AND s || 'x' <> 'zx' AND ",
        "id IN (SELECT id FROM u) AND s || 'x' <> 'zx' AND ",
        "id NOT IN (SELECT g + 5 FROM u) AND s || 'x' <> 'zx' AND ",
    ] {
        for tail in ["", " LIMIT 5"] {
            let sql = format!("SELECT id FROM t WHERE {filter}{having}{tail}");
            let got = ids(&db, &sql);
            if got != Ok(vec![1]) {
                failed.push(format!("{sql}: {got:?}"));
            }
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}
