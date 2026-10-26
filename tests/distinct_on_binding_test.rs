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

//! A DISTINCT ON key is bound as an ORDER BY key is: it answers as the same
//! query whose key is written out, over every source, select list, pipeline,
//! nesting and prepared run

use std::collections::BTreeMap;
use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE parent (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO parent VALUES (1, 5), (2, 7)",
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

/// The answer run directly, and run twice as a prepared statement
fn runs(db: &Database, sql: &str) -> [(&'static str, Rows); 3] {
    let prepared = db.prepare(sql).map_err(|e| e.to_string());
    let run = || match &prepared {
        Ok(stmt) => collect(stmt.query(())),
        Err(e) => Err(e.clone()),
    };
    [
        ("direct", collect(db.query(sql, ()))),
        ("prepared", run()),
        ("prepared again", run()),
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

/// (name, items, DISTINCT ON key, the key written out over the input)
const SHAPES: [(&str, &str, &str, &str); 22] = [
    ("column", "c.b", "c.a", "c.a"),
    ("bare column", "c.b", "a", "c.a"),
    ("computed", "c.b", "c.a % 2", "c.a % 2"),
    ("computed twice read", "c.b", "c.a + c.id * 0", "c.a"),
    ("selected computed", "c.a % 2, c.b", "c.a % 2", "c.a % 2"),
    ("aliased computed", "c.a % 2 AS m", "c.a % 2", "c.a % 2"),
    ("alias", "c.a % 2 AS q, c.b", "q", "c.a % 2"),
    (
        "alias in expression",
        "c.a % 2 AS q, c.b",
        "q + 0",
        "c.a % 2",
    ),
    (
        "alias over a column name",
        "c.a % 2 AS a, c.b",
        "a",
        "c.a % 2",
    ),
    ("ordinal", "c.a % 2, c.b", "1", "c.a % 2"),
    ("star", "*", "c.a % 2", "c.a % 2"),
    ("item then star", "c.b, *", "c.a % 2", "c.a % 2"),
    ("qualified star", "c.*", "c.a % 2", "c.a % 2"),
    (
        "star then computed",
        "*, c.a % 2 AS m",
        "c.a % 2",
        "c.a % 2",
    ),
    ("enclosing", "c.b", "p.b + c.a % 2", "c.a % 2"),
    (
        "subquery",
        "c.b",
        "c.a % 2 + (SELECT MIN(id) FROM side) * 0",
        "c.a % 2",
    ),
    // A windowed statement computes its key as a hidden item
    (
        "windowed subquery",
        "ROW_NUMBER() OVER () * 0 + c.b",
        "c.a % 2 + (SELECT MIN(id) FROM side) * 0",
        "c.a % 2",
    ),
    (
        "windowed computed",
        "ROW_NUMBER() OVER () * 0 + c.b",
        "c.a % 2",
        "c.a % 2",
    ),
    (
        "windowed enclosing",
        "ROW_NUMBER() OVER () * 0 + c.b",
        "p.b + c.a % 2",
        "c.a % 2",
    ),
    // `{k}` is a column the star may not show: the join's other side
    ("qualified star, other column", "c.*", "{k} % 2", "{k} % 2"),
    (
        "windowed star",
        "*, ROW_NUMBER() OVER () * 0 AS w",
        "{k} % 2",
        "{k} % 2",
    ),
    (
        "windowed qualified star",
        "c.*, ROW_NUMBER() OVER () * 0 AS w",
        "{k} % 2",
        "{k} % 2",
    ),
];

/// Where the query stands: alone, as a derived table, or as scalar subqueries
const NESTINGS: [&str; 3] = ["top", "derived", "scalar"];

fn query(head: &str, items: &str, from: &str, filter: &str, order: &str, tail: &str) -> String {
    format!("SELECT DISTINCT ON ({head}) {items} FROM {from}{filter} ORDER BY {order}, c.id{tail}")
}

#[test]
fn a_distinct_on_key_answers_as_its_written_out_key() {
    let db = Database::open("memory://distinct_on_binding").unwrap();
    setup(&db);
    let mut wrong: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut cases = 0;
    for (source, with, from) in SOURCES {
        // The written out key over the table itself, or the join itself
        let base_from = if from.contains("JOIN") {
            from
        } else {
            "child AS c"
        };
        let other = if from.contains("JOIN") { "s.k" } else { "c.a" };
        for (name, items, key, written) in SHAPES {
            let (key, written) = (&*key.replace("{k}", other), &*written.replace("{k}", other));
            let enclosing = key.contains("p.");
            // A star written out as its columns
            let star = if from.contains("JOIN") {
                "c.id, c.a, c.b, s.id, s.k"
            } else {
                "c.id, c.a, c.b"
            };
            let base_items = items
                .split(", ")
                .map(|item| match item {
                    "*" => star,
                    "c.*" => "c.id, c.a, c.b",
                    other => other,
                })
                .collect::<Vec<_>>()
                .join(", ");
            let base =
                collect(db.query(&query(written, &base_items, base_from, "", written, ""), ()))
                    .unwrap_or_else(|e| panic!("{name} base: {e}"));
            for (pipeline, filter) in FILTERS {
                for nesting in NESTINGS {
                    let single = !items.contains(", ") && !matches!(items, "*" | "c.*");
                    let (sql, want) = match nesting {
                        "top" if !enclosing => (
                            format!("{with}{}", query(key, items, from, filter, written, "")),
                            sorted(base.clone(), false),
                        ),
                        "derived" if !enclosing => (
                            format!(
                                "{with}SELECT * FROM ({}) AS d",
                                query(key, items, from, filter, written, "")
                            ),
                            sorted(base.clone(), true),
                        ),
                        "scalar" if single => {
                            let one = |offset: usize| {
                                format!(
                                    "({})",
                                    query(
                                        key,
                                        items,
                                        from,
                                        filter,
                                        written,
                                        &format!(" LIMIT 1 OFFSET {offset}")
                                    )
                                )
                            };
                            let row: Vec<String> = (0..3)
                                .map(|i| {
                                    base.get(i)
                                        .map_or("Some(Null(Null))".into(), |r| r[0].clone())
                                })
                                .collect();
                            (
                                format!(
                                    "{with}SELECT {}, {}, {} FROM parent AS p ORDER BY p.id",
                                    one(0),
                                    one(1),
                                    one(2)
                                ),
                                vec![row.clone(), row],
                            )
                        }
                        _ => continue,
                    };
                    for (run, got) in runs(&db, &sql) {
                        cases += 1;
                        let got = got.map(|rows| sorted(rows, nesting == "derived"));
                        if got.as_ref() != Ok(&want) {
                            wrong
                                .entry(format!("{name:<24} {nesting:<7} {pipeline}"))
                                .or_default()
                                .push(format!("{source} {run}: {got:?}"));
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
                "{:>2}  {shape}\n      {}",
                runs.len(),
                runs.join("\n      ")
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
fn a_set_operation_reads_a_key_its_first_branch_selects() {
    let db = Database::open("memory://distinct_on_binding_set_operation").unwrap();
    setup(&db);
    // (first branch's item, key): each key is that item's value
    let shapes = [
        ("c.a % 2", "c.a % 2"),
        ("c.a % 2 AS m", "m"),
        ("c.a % 2 AS m", "c.a % 2"),
        ("c.a % 2", "1"),
        ("c.a", "c.a"),
        ("c.a", "a"),
    ];
    let mut wrong = Vec::new();
    for (_, with, from) in SOURCES {
        for (item, key) in shapes {
            let unaliased = item.split(" AS ").next().unwrap_or(item);
            let sql = format!(
                "{with}SELECT DISTINCT ON ({key}) {item} FROM {from} \
                 UNION ALL SELECT {unaliased} FROM {from}"
            );
            let all = format!("{with}SELECT {unaliased} FROM {from}");
            let mut want = collect(db.query(&all, ())).unwrap_or_else(|e| panic!("{all}: {e}"));
            want.sort();
            want.dedup();
            for (run, got) in runs(&db, &sql) {
                let got = got.map(|rows| sorted(rows, true));
                if got.as_ref() != Ok(&want) {
                    wrong.push(format!("{sql} ({run}): {got:?}, want {want:?}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_position_outside_the_list_names_no_column() {
    let db = Database::open("memory://distinct_on_binding_position").unwrap();
    setup(&db);
    for (_, with, from) in SOURCES {
        for (_, filter) in FILTERS {
            for position in ["0", "3"] {
                let sql = format!(
                    "{with}{}",
                    query(position, "c.a % 2, c.b", from, filter, "c.a % 2", "")
                );
                for (run, got) in runs(&db, &sql) {
                    assert!(
                        got.as_ref().is_err_and(|e| e.contains("names no column")),
                        "{sql} ({run}) answered {got:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_block_keeps_the_first_row_of_each_key_in_every_form() {
    let db = Database::open("memory://distinct_on_binding_forms").unwrap();
    setup(&db);
    let row = |cells: &[&str]| cells.iter().map(|c| c.to_string()).collect::<Vec<_>>();
    let cases = [
        (
            "SELECT p.id, (SELECT DISTINCT ON (c.a) c.b FROM child AS c WHERE c.a = p.id \
             ORDER BY c.a, c.b) FROM parent AS p ORDER BY p.id",
            vec![
                row(&["Some(Integer(1))", "Some(Integer(70))"]),
                row(&["Some(Integer(2))", "Some(Integer(0))"]),
            ],
        ),
        (
            "SELECT p.id FROM parent AS p WHERE p.b * 10 IN (SELECT DISTINCT ON (c.a) c.b \
             FROM child AS c WHERE c.id > p.id - 2 ORDER BY c.a, c.b DESC) ORDER BY p.id",
            vec![row(&["Some(Integer(1))"])],
        ),
        (
            "SELECT p.id FROM parent AS p WHERE EXISTS (SELECT DISTINCT ON (c.a % 2) c.id \
             FROM child AS c WHERE c.b >= p.b - 5 ORDER BY c.a % 2, c.id OFFSET 1) ORDER BY p.id",
            vec![row(&["Some(Integer(1))"])],
        ),
    ];
    for (sql, want) in cases {
        let explain = collect(db.query(&format!("EXPLAIN {sql}"), ())).unwrap();
        assert!(
            explain
                .iter()
                .flatten()
                .any(|line| line.contains("Prepared: yes")),
            "{sql}: {explain:?}"
        );
        for (run, got) in runs(&db, sql) {
            assert_eq!(got.as_ref(), Ok(&want), "{sql} ({run})");
        }
    }
}

#[test]
fn a_join_runs_a_distinct_on_subquery_as_its_block() {
    let db = Database::open("memory://distinct_on_binding_join").unwrap();
    setup(&db);
    let sql = "SELECT DISTINCT ON (c.a % 2 + (SELECT MIN(id) FROM side) * 0) c.b \
               FROM child AS c JOIN side AS s ON s.k = c.a ORDER BY c.a % 2, c.id";
    let explain = collect(db.query(&format!("EXPLAIN {sql}"), ())).unwrap();
    assert!(
        explain
            .iter()
            .flatten()
            .any(|line| line.contains("Prepared: yes")),
        "{explain:?}"
    );
    let want = vec![
        vec!["Some(Integer(0))".to_string()],
        vec!["Some(Integer(100))".to_string()],
    ];
    for (run, got) in runs(&db, sql) {
        assert_eq!(got.as_ref(), Ok(&want), "{run}");
    }
}

/// The rows, in order when the query fixes it
fn sorted(mut rows: Vec<Vec<String>>, unordered: bool) -> Vec<Vec<String>> {
    if unordered {
        rows.sort();
    }
    rows
}
