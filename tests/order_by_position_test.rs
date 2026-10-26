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

//! An ORDER BY position names one of the columns the SELECT list shows, or
//! the statement fails; a position that names one sorts as its column
//! written out, over every source, select list, pipeline and nesting

use std::collections::BTreeMap;
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

type Rows = Result<Vec<Vec<String>>, String>;

fn collect(rows: stoolap::Result<stoolap::Rows>) -> Rows {
    let mut out = Vec::new();
    for row in rows.map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i)))
                .collect(),
        );
    }
    Ok(out)
}

/// The answer run directly, and run as a prepared statement
fn runs(db: &Database, sql: &str) -> [(&'static str, Rows); 2] {
    let prepared = match db.prepare(sql) {
        Ok(stmt) => collect(stmt.query(())),
        Err(e) => Err(e.to_string()),
    };
    [
        ("direct", collect(db.query(sql, ()))),
        ("prepared", prepared),
    ]
}

/// (name, WITH clause, FROM)
const SOURCES: [(&str, &str, &str); 5] = [
    ("table", "", "child AS c"),
    ("derived", "", "(SELECT * FROM child) AS c"),
    ("view", "", "cv AS c"),
    ("cte", "WITH w AS (SELECT * FROM child) ", "w AS c"),
    ("join", "", "child AS c JOIN side AS s ON s.k = c.a"),
];

/// The general pipeline, and the blocks' one a WHERE subquery takes
const FILTERS: [(&str, &str); 2] = [
    ("general", ""),
    ("blocks", " WHERE c.id > (SELECT MIN(id) FROM child) - 1"),
];

/// (name, the items, the columns they show written out, what follows
/// WHERE before ORDER BY); `{star}` is the source's columns
const SHAPES: [(&str, &str, &str, &str); 10] = [
    ("columns", "c.a, c.b", "c.a, c.b", ""),
    ("alias", "c.a AS x, c.b", "c.a, c.b", ""),
    ("computed", "c.a % 2, c.b", "c.a % 2, c.b", ""),
    ("star", "*", "{star}", ""),
    ("qualified star", "c.*", "c.id, c.a, c.b", ""),
    ("star then item", "*, c.b", "{star}, c.b", ""),
    ("grouped", "c.a, COUNT(*)", "c.a, COUNT(*)", " GROUP BY c.a"),
    (
        "windowed",
        "c.a, ROW_NUMBER() OVER (ORDER BY c.id)",
        "c.a, ROW_NUMBER() OVER (ORDER BY c.id)",
        "",
    ),
    ("distinct", "DISTINCT c.a, c.b", "c.a, c.b", ""),
    (
        "union",
        "c.a, c.b",
        "",
        " UNION ALL SELECT c.b, c.a FROM child AS c",
    ),
];

/// Where the query stands: alone, as a derived table, or inside EXISTS
const NESTINGS: [&str; 3] = ["top", "derived", "exists"];

#[test]
fn an_order_by_position_names_a_shown_column() {
    let db = Database::open("memory://order_by_position").unwrap();
    setup(&db);
    let mut wrong: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut cases = 0;
    for (source, with, from) in SOURCES {
        let star = if from.contains("JOIN") {
            "c.id, c.a, c.b, s.id, s.k"
        } else {
            "c.id, c.a, c.b"
        };
        for (name, items, written, tail) in SHAPES {
            let written = written.replace("{star}", star);
            let width = match written.as_str() {
                "" => 2,
                written => written.split(", ").count(),
            };
            // A union's ORDER BY reads only its output, so its keys are
            // checked for refusal alone
            let union = written.is_empty();
            let written: Vec<&str> = written.split(", ").collect();
            for (pipeline, filter) in FILTERS {
                for limit in ["", " LIMIT 3"] {
                    let query = |keys: &str| {
                        format!("SELECT {items} FROM {from}{filter}{tail} ORDER BY {keys}{limit}")
                    };
                    // Every position, the last one first and descending
                    let positions: Vec<String> = (1..=width)
                        .rev()
                        .enumerate()
                        .map(|(i, p)| {
                            if i == 0 {
                                format!("{p} DESC")
                            } else {
                                p.to_string()
                            }
                        })
                        .collect();
                    let columns: Vec<String> = (1..=width)
                        .rev()
                        .enumerate()
                        .map(|(i, p)| {
                            let column = written.get(p - 1).copied().unwrap_or_default();
                            if i == 0 {
                                format!("{column} DESC")
                            } else {
                                column.to_string()
                            }
                        })
                        .collect();
                    let past = width + 1;
                    let mut keys: Vec<(String, Option<String>)> = vec![
                        ("0".into(), None),
                        (past.to_string(), None),
                        (format!("1, {past}"), None),
                        ("-1".into(), None),
                    ];
                    if !union {
                        keys.push((positions.join(", "), Some(columns.join(", "))));
                    }
                    for (keys, written_keys) in keys {
                        for nesting in NESTINGS {
                            let wrap = |q: String| match nesting {
                                "top" => format!("{with}{q}"),
                                "derived" => format!("{with}SELECT * FROM ({q}) AS d"),
                                _ => format!("{with}SELECT EXISTS ({q})"),
                            };
                            let sql = wrap(query(&keys));
                            let want = match &written_keys {
                                Some(written_keys) => {
                                    let base = wrap(query(written_keys));
                                    match collect(db.query(&base, ())) {
                                        Ok(rows) => Ok(rows),
                                        Err(e) => panic!("{base}: {e}"),
                                    }
                                }
                                None => Err(()),
                            };
                            for (run, got) in runs(&db, &sql) {
                                cases += 1;
                                let right = match (&want, &got) {
                                    (Ok(want), Ok(got)) if nesting == "derived" => {
                                        let (mut w, mut g) = (want.clone(), got.clone());
                                        w.sort();
                                        g.sort();
                                        w == g
                                    }
                                    (Ok(want), Ok(got)) => want == got,
                                    (Err(()), Err(e)) => e.contains("ORDER BY position"),
                                    _ => false,
                                };
                                if !right {
                                    let got = match &got {
                                        Ok(_) if want.is_err() => "answered".to_string(),
                                        Ok(rows) => format!("{rows:?}"),
                                        Err(e) => e.clone(),
                                    };
                                    wrong
                                        .entry(format!("{name:<15} {pipeline:<7} {nesting}"))
                                        .or_default()
                                        .push(format!("{source} {run} [{keys}]: {got}"));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let total: usize = wrong.values().map(Vec::len).sum();
    let table: Vec<String> = wrong
        .iter()
        .map(|(shape, runs)| {
            format!(
                "{:>3}  {shape}\n      {}",
                runs.len(),
                runs.iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n      ")
            )
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "{total} of {cases} wrong:\n{}",
        table.join("\n")
    );
}
