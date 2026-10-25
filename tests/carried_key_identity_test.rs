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

//! A sort or DISTINCT ON key is read from the column that computes it, never
//! from an output column whose name is spelled like the key: renaming that
//! output changes no answer, over every source and pipeline

use stoolap::parser::ast::Statement;
use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE child (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "INSERT INTO child VALUES (1, 1, 100), (2, 2, 0), (3, 3, 50), (4, 1, 70)",
        "CREATE TABLE side (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO side VALUES (1, 1), (2, 2), (3, 3)",
        "CREATE VIEW cv AS SELECT id, a, b FROM child",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        let values: Vec<_> = (0..row.len()).map(|i| row.get_value(i)).collect();
        out.push(format!("{values:?}"));
    }
    Ok(out)
}

/// The name a key's column was given by its text
fn spelled(key: &str) -> String {
    let sql = format!("SELECT 1 FROM t AS c ORDER BY {key}");
    match &stoolap::parser::parse_sql(&sql).unwrap()[0] {
        Statement::Select(select) => select.order_by[0].expression.to_string(),
        _ => unreachable!(),
    }
}

const SOURCES: [(&str, &str); 5] = [
    ("", "child AS c"),
    ("", "(SELECT * FROM child) AS c"),
    ("", "cv AS c"),
    ("WITH w AS (SELECT * FROM child) ", "w AS c"),
    ("", "child AS c JOIN side AS s ON s.k = c.a"),
];

/// The general pipeline, and the blocks' one a WHERE subquery takes
const FILTERS: [&str; 2] = ["", " WHERE c.id > (SELECT MIN(id) FROM child) - 1"];

/// (items with `{alias}`, key, the DISTINCT ON head or none)
const SHAPES: [(&str, &str, &str); 15] = [
    ("c.b AS \"{alias}\"", "ABS(c.a - 2)", ""),
    ("c.b AS \"{alias}\"", "c.a + 0", ""),
    ("c.b AS \"{alias}\"", "-c.a", ""),
    ("c.b AS \"{alias}\"", "c.a % 2", ""),
    (
        "c.b AS \"{alias}\"",
        "CASE WHEN c.a > 1 THEN 0 ELSE 1 END",
        "",
    ),
    ("c.id, c.b AS \"{alias}\"", "c.a * 1", ""),
    // A key reading an alias
    ("c.a AS k, c.b AS \"{alias}\"", "k + 0", ""),
    ("c.a AS k, c.b AS \"{alias}\"", "-k", ""),
    ("c.a AS k, c.b AS \"{alias}\"", "ABS(k - 2)", ""),
    // One row per key
    ("c.b AS \"{alias}\"", "c.a % 2", "DISTINCT ON (c.a % 2) "),
    ("c.b AS \"{alias}\"", "c.a * 0", "DISTINCT ON (c.a * 0) "),
    ("c.b AS \"{alias}\"", "c.a + 0", "DISTINCT ON (c.a + 0) "),
    // A window key
    (
        "c.b AS \"{alias}\"",
        "ROW_NUMBER() OVER (ORDER BY c.a, c.id)",
        "",
    ),
    (
        "c.b AS \"{alias}\", ROW_NUMBER() OVER () AS n",
        "ROW_NUMBER() OVER (ORDER BY c.a, c.id)",
        "",
    ),
    (
        "c.b AS \"{alias}\"",
        "RANK() OVER (PARTITION BY c.a % 2 ORDER BY c.a, c.id)",
        "",
    ),
];

#[test]
fn an_output_spelled_like_a_key_does_not_answer_for_it() {
    let db = Database::open("memory://carried_key_identity").unwrap();
    setup(&db);
    let (mut cases, mut wrong) = (0, Vec::new());
    for (with, from) in SOURCES {
        for filter in FILTERS {
            for (items, key, head) in SHAPES {
                for limit in ["", " LIMIT 2"] {
                    let query = |alias: &str| {
                        let items = items.replace("{alias}", alias);
                        format!(
                            "{with}SELECT {head}{items} FROM {from}{filter} \
                             ORDER BY {key}, c.id{limit}"
                        )
                    };
                    let want = rows(&db, &query("zz"));
                    let sql = query(&spelled(key));
                    let got = rows(&db, &sql);
                    cases += 1;
                    // #664: a computed DISTINCT ON or window key over a derived
                    // source with a WHERE subquery fails whatever the item is called
                    let known = (head.starts_with("DISTINCT ON") || key.contains("OVER"))
                        && !filter.is_empty()
                        && from != "child AS c"
                        && !from.contains("JOIN");
                    if (want.is_err() && !known) || got != want {
                        wrong.push(format!("{sql}\n  got {got:?}\n  want {want:?}"));
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
