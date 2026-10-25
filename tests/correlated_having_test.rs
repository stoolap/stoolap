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

#[test]
fn a_subquery_reading_the_outer_row_in_its_having_is_correlated() {
    let db = Database::open("memory://a_subquery_reading_the_outer_row_in_its_having").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        "INSERT INTO t VALUES (1, 1, 'a'), (2, 2, 'b'), (3, 3, 'c')",
        "CREATE INDEX t_k ON t(k)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
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
