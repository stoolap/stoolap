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
        "CREATE TABLE nobody (id INTEGER PRIMARY KEY)",
        "CREATE TABLE parent (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO parent VALUES (1, 5), (2, 7), (9, 1)",
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
const SHAPES: [(&str, &str, &str, &str); 11] = [
    ("columns", "c.a, c.b", "c.a, c.b", ""),
    // No qualifier anywhere, as a statement the scope check may skip
    ("bare columns", "a, b", "a, b", ""),
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

/// Where the query stands: alone, as a derived table, inside EXISTS, or
/// inside an EXISTS or NOT EXISTS that reads the enclosing row, of a table
/// with rows or of an empty one
const NESTINGS: [&str; 7] = [
    "top",
    "derived",
    "exists",
    "correlated exists",
    "correlated not exists",
    "empty correlated exists",
    "empty correlated not exists",
];

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
                    let query = |keys: &str, correlated: bool| {
                        let filter = match (correlated, filter) {
                            (false, filter) => filter.to_string(),
                            (true, "") => " WHERE c.id = p.id".to_string(),
                            (true, filter) => format!("{filter} AND c.id = p.id"),
                        };
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
                            let correlated = nesting.contains("correlated");
                            // An empty enclosing table runs the subquery for no row
                            let outer = match nesting.starts_with("empty") {
                                true => "nobody",
                                false => "parent",
                            };
                            let wrap = |q: String| match nesting {
                                "top" => format!("{with}{q}"),
                                "derived" => format!("{with}SELECT * FROM ({q}) AS d"),
                                "exists" => format!("{with}SELECT EXISTS ({q})"),
                                nesting if nesting.ends_with("not exists") => format!(
                                    "{with}SELECT p.id FROM {outer} AS p WHERE NOT EXISTS ({q}) \
                                     ORDER BY p.id"
                                ),
                                _ => format!(
                                    "{with}SELECT p.id FROM {outer} AS p WHERE EXISTS ({q}) \
                                     ORDER BY p.id"
                                ),
                            };
                            let sql = wrap(query(&keys, correlated));
                            let want = match &written_keys {
                                Some(written_keys) => {
                                    let base = wrap(query(written_keys, correlated));
                                    match collect(db.query(&base, ())) {
                                        Ok(rows) => Ok(rows),
                                        Err(e) => panic!("{base}: {e}"),
                                    }
                                }
                                None => Err(()),
                            };
                            // #676: a join keeps the subquery out of the
                            // blocks, so with no enclosing row it never runs
                            let unchecked = outer == "nobody" && from.contains("JOIN");
                            for (run, got) in runs(&db, &sql) {
                                cases += 1;
                                let right = match (&want, &got) {
                                    (Err(()), Ok(rows)) if unchecked => rows.is_empty(),
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

#[test]
fn a_star_of_one_column_counts_one_position_in_every_subquery() {
    let db = Database::open("memory://order_by_position_one_column").unwrap();
    setup(&db);
    for sql in [
        "CREATE TABLE solo (id INTEGER PRIMARY KEY)",
        "INSERT INTO solo VALUES (1), (2), (9)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for inner in [
        "SELECT * FROM solo AS o ORDER BY {p}",
        "SELECT * FROM solo AS o WHERE o.id = p.id ORDER BY {p}",
        "SELECT o.* FROM solo AS o WHERE o.id >= p.id ORDER BY {p} LIMIT 1",
    ] {
        for outer in [
            "SELECT p.id FROM parent AS p WHERE p.id IN ({q}) ORDER BY p.id",
            "SELECT p.id FROM parent AS p WHERE p.id NOT IN ({q}) ORDER BY p.id",
            "SELECT p.id FROM parent AS p WHERE p.id = ANY ({q}) ORDER BY p.id",
            "SELECT p.id, ({q} LIMIT 1) FROM parent AS p ORDER BY p.id",
            "SELECT p.id FROM parent AS p WHERE EXISTS ({q}) ORDER BY p.id",
        ] {
            // A scalar subquery holds one LIMIT
            if outer.contains("LIMIT 1)") && inner.contains("LIMIT") {
                continue;
            }
            let sql = |p: &str| outer.replace("{q}", &inner.replace("{p}", p));
            for (run, got) in runs(&db, &sql("1")) {
                if let Err(e) = got {
                    wrong.push(format!("{} ({run}) refused: {e}", sql("1")));
                }
            }
            for p in ["0", "2"] {
                for (run, got) in runs(&db, &sql(p)) {
                    if !got.as_ref().is_err_and(|e| e.contains("ORDER BY position")) {
                        wrong.push(format!("{} ({run}): {got:?}", sql(p)));
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_valid_position_leaves_a_correlated_exists_plan_as_it_is() {
    let db = Database::open("memory://order_by_position_plan").unwrap();
    setup(&db);
    // The plan without the line that prints the subquery as written
    let plan = |sql: &str| -> Vec<String> {
        collect(db.query(&format!("EXPLAIN {sql}"), ()))
            .unwrap_or_else(|e| panic!("EXPLAIN {sql}: {e}"))
            .into_iter()
            .flatten()
            .filter(|line| !line.contains("EXISTS"))
            .collect()
    };
    let mut wrong = Vec::new();
    for (_, with, from) in SOURCES {
        for items in ["*", "c.*"] {
            let sql = |order: &str| {
                format!(
                    "{with}SELECT p.id FROM parent AS p WHERE EXISTS \
                     (SELECT {items} FROM {from} WHERE c.id = p.id{order})"
                )
            };
            let want = plan(&sql(""));
            for order in [" ORDER BY 1", " ORDER BY 2 DESC, 1"] {
                let got = plan(&sql(order));
                if got != want {
                    wrong.push(format!("{}\n  {got:?}\n  want {want:?}", sql(order)));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_position_in_a_derived_table_of_a_subquery_names_a_shown_column() {
    let db = Database::open("memory://order_by_position_derived_block").unwrap();
    setup(&db);
    // `{p}` is the position, `{c}` the correlation
    let derived = [
        "(SELECT * FROM child ORDER BY {p}) AS d",
        "(SELECT c.* FROM child AS c ORDER BY {p}) AS d",
        "(SELECT * FROM (SELECT * FROM child ORDER BY {p}) AS e) AS d",
    ];
    let mut wrong = Vec::new();
    for outer in ["parent", "nobody"] {
        for from in derived {
            for correlation in [" WHERE d.id = p.id", ""] {
                for form in [
                    "SELECT p.id FROM {o} AS p WHERE EXISTS (SELECT * FROM {d}{c})",
                    "SELECT p.id FROM {o} AS p WHERE NOT EXISTS (SELECT * FROM {d}{c})",
                    "SELECT p.id, (SELECT d.a FROM {d}{c} LIMIT 1) FROM {o} AS p",
                    "SELECT p.id FROM {o} AS p WHERE p.id IN (SELECT d.id FROM {d}{c})",
                ] {
                    let sql = |p: &str| {
                        form.replace("{d}", from)
                            .replace("{c}", correlation)
                            .replace("{o}", outer)
                            .replace("{p}", p)
                    };
                    for (run, got) in runs(&db, &sql("3 DESC")) {
                        if let Err(e) = got {
                            wrong.push(format!("{} ({run}) refused: {e}", sql("3 DESC")));
                        }
                    }
                    for p in ["4", "1, 4", "0"] {
                        for (run, got) in runs(&db, &sql(p)) {
                            if !got.as_ref().is_err_and(|e| e.contains("ORDER BY position")) {
                                wrong.push(format!("{} ({run}): {got:?}", sql(p)));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
