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

//! A column is found by the entry it comes from and its own name, never by
//! the text they spell together: spelling a FROM name or an output name
//! with a dot instead of without one changes no answer

use std::collections::BTreeMap;
use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE w (id INTEGER PRIMARY KEY, \"b.v\" INTEGER)",
        "INSERT INTO w VALUES (1, 7), (2, 5), (3, 7)",
        "CREATE TABLE uv (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO uv VALUES (1, 9), (2, 4), (3, 1)",
        "CREATE TABLE parent (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO parent VALUES (1, 5), (2, 7)",
        "CREATE TABLE child (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "INSERT INTO child VALUES (1, 1, 100), (2, 2, 0), (3, 3, 50)",
        "CREATE VIEW cv AS SELECT id, a, b FROM child",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

type Rows = Result<Vec<String>, String>;

fn collect(rows: stoolap::Result<stoolap::Rows>) -> Rows {
    let rows = rows.map_err(|e| e.to_string())?;
    let mut out = vec![format!("{:?}", rows.columns().len())];
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        let values: Vec<_> = (0..row.len()).map(|i| row.get_value(i)).collect();
        out.push(format!("{values:?}"));
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

/// Each case with a dotted name, against the same case spelling it `zz`
struct Matrix<'d> {
    db: &'d Database,
    cases: usize,
    wrong: BTreeMap<String, Vec<String>>,
}

impl Matrix<'_> {
    fn check(&mut self, cell: &str, dotted: &str, spelled: &str, plain: &str) {
        let want = collect(self.db.query(&dotted.replace(spelled, plain), ()));
        for (run, got) in runs(self.db, dotted) {
            self.cases += 1;
            if want.is_err() || got != want {
                self.wrong
                    .entry(cell.to_string())
                    .or_default()
                    .push(format!(
                        "{run}: {dotted}\n        got {got:?}\n        want {want:?}"
                    ));
            }
        }
    }
}

/// The general pipeline, and the blocks' one a WHERE subquery takes
const FILTERS: [(&str, &str); 2] = [
    ("general", ""),
    ("blocks", "id > (SELECT MIN(id) FROM uv) - 1"),
];

/// (name, the query over `{from}`, `{and}` before a filter or `{where}`
/// for one alone): two entries whose names spell one text, `a` with its
/// column "b.v" and "a.b" with its column v
const JOINED: [(&str, &str); 12] = [
    (
        "both columns",
        "SELECT \"a.b\".v, a.\"b.v\" FROM {from}{where} ORDER BY a.id",
    ),
    (
        "left column",
        "SELECT a.\"b.v\" FROM {from}{where} ORDER BY a.id",
    ),
    (
        "right column",
        "SELECT \"a.b\".v FROM {from}{where} ORDER BY a.id",
    ),
    (
        "filter",
        "SELECT a.id, a.\"b.v\" FROM {from} WHERE a.\"b.v\" = 7{and} ORDER BY a.id",
    ),
    (
        "right filter",
        "SELECT a.id FROM {from} WHERE \"a.b\".v = 9{and} ORDER BY a.id",
    ),
    (
        "order",
        "SELECT a.id FROM {from}{where} ORDER BY a.\"b.v\", a.id",
    ),
    ("left star", "SELECT a.* FROM {from}{where} ORDER BY a.id"),
    (
        "right star",
        "SELECT \"a.b\".* FROM {from}{where} ORDER BY 1",
    ),
    ("star", "SELECT * FROM {from}{where} ORDER BY 1"),
    (
        "grouped",
        "SELECT a.\"b.v\", COUNT(*) FROM {from}{where} GROUP BY a.\"b.v\" ORDER BY 1",
    ),
    (
        "window",
        "SELECT a.id, SUM(\"a.b\".v) OVER (PARTITION BY a.\"b.v\") FROM {from}{where} \
         ORDER BY a.id",
    ),
    (
        "distinct",
        "SELECT DISTINCT a.\"b.v\" FROM {from}{where} ORDER BY a.\"b.v\"",
    ),
];

#[test]
fn a_joined_column_is_found_by_its_entry_and_name() {
    let db = Database::open("memory://column_sources_join").unwrap();
    setup(&db);
    let mut matrix = Matrix {
        db: &db,
        cases: 0,
        wrong: BTreeMap::new(),
    };
    let from = "w AS a JOIN uv AS \"a.b\" ON \"a.b\".id = a.id";
    for (pipeline, filter) in FILTERS {
        let filter = filter.replace("id", "a.id");
        for (name, query) in JOINED {
            for limit in ["", " LIMIT 2"] {
                let (where_, and) = match filter.as_str() {
                    "" => (String::new(), String::new()),
                    f => (format!(" WHERE {f}"), format!(" AND {f}")),
                };
                let sql = format!(
                    "{}{limit}",
                    query
                        .replace("{from}", from)
                        .replace("{where}", &where_)
                        .replace("{and}", &and)
                );
                for (nesting, sql) in [
                    ("top", sql.clone()),
                    ("derived", format!("SELECT * FROM ({sql}) AS d")),
                ] {
                    matrix.check(
                        &format!("join {name:<12} {pipeline:<7} {nesting}"),
                        &sql,
                        "\"a.b\"",
                        "q",
                    );
                }
            }
        }
    }
    report(matrix);
}

/// (name, WITH clause, FROM)
const SOURCES: [(&str, &str, &str); 4] = [
    ("table", "", "child AS c"),
    ("derived", "", "(SELECT * FROM child) AS c"),
    ("view", "", "cv AS c"),
    ("cte", "WITH w2 AS (SELECT * FROM child) ", "w2 AS c"),
];

/// (name, the subquery with `{f}` its FROM and `{w}` its filter): its
/// output is called "p.b", which the enclosing query's column is spelled as
const CORRELATED: [(&str, &str); 10] = [
    (
        "order",
        "SELECT c.a AS \"p.b\" FROM {f}{w} ORDER BY p.b, c.a DESC LIMIT 1",
    ),
    (
        "order expression",
        "SELECT c.a AS \"p.b\" FROM {f}{w} ORDER BY p.b + 0, c.a DESC LIMIT 1",
    ),
    (
        "order negated",
        "SELECT c.a AS \"p.b\" FROM {f}{w} ORDER BY -p.b, c.a DESC LIMIT 1",
    ),
    (
        "distinct",
        "SELECT DISTINCT c.a AS \"p.b\" FROM {f}{w} ORDER BY p.b, c.a DESC LIMIT 1",
    ),
    (
        "distinct on",
        "SELECT DISTINCT ON (p.b) c.a AS \"p.b\" FROM {f}{w} ORDER BY p.b, c.a DESC LIMIT 1",
    ),
    (
        "grouped",
        "SELECT c.a AS \"p.b\" FROM {f}{w} GROUP BY c.a ORDER BY p.b, c.a DESC LIMIT 1",
    ),
    (
        "selected",
        "SELECT p.b + c.a AS \"p.b\" FROM {f}{w} ORDER BY c.a DESC LIMIT 1",
    ),
    (
        "filtered",
        "SELECT c.a AS \"p.b\" FROM {f} WHERE c.a < p.b{and} ORDER BY c.a DESC LIMIT 1",
    ),
    (
        "having",
        "SELECT c.a AS \"p.b\" FROM {f}{w} GROUP BY c.a HAVING c.a + p.b > 7 \
         ORDER BY c.a LIMIT 1",
    ),
    (
        "window",
        "SELECT ROW_NUMBER() OVER () * 0 + c.a AS \"p.b\" FROM {f}{w} ORDER BY p.b, c.a DESC \
         LIMIT 1",
    ),
];

#[test]
fn an_output_spelled_like_an_enclosing_column_is_not_that_column() {
    let db = Database::open("memory://column_sources_correlated").unwrap();
    setup(&db);
    let mut matrix = Matrix {
        db: &db,
        cases: 0,
        wrong: BTreeMap::new(),
    };
    for (source, with, from) in SOURCES {
        for (pipeline, filter) in FILTERS {
            let filter = filter.replace("id", "c.id").replace("uv", "child");
            let (where_, and) = match filter.as_str() {
                "" => (String::new(), String::new()),
                f => (format!(" WHERE {f}"), format!(" AND {f}")),
            };
            for (name, sub) in CORRELATED {
                let sub = sub
                    .replace("{f}", from)
                    .replace("{w}", &where_)
                    .replace("{and}", &and);
                for (nesting, sql) in [
                    (
                        "select",
                        format!("{with}SELECT ({sub}) FROM parent AS p ORDER BY p.id"),
                    ),
                    (
                        "where",
                        format!(
                            "{with}SELECT p.id FROM parent AS p WHERE ({sub}) IS NOT NULL \
                             ORDER BY p.id"
                        ),
                    ),
                ] {
                    matrix.check(
                        &format!("{name:<16} {pipeline:<7} {nesting:<6} {source}"),
                        &sql,
                        "\"p.b\"",
                        "zz",
                    );
                }
            }
        }
    }
    report(matrix);
}

fn report(matrix: Matrix<'_>) {
    let total: usize = matrix.wrong.values().map(Vec::len).sum();
    let table: Vec<String> = matrix
        .wrong
        .iter()
        .map(|(cell, cases)| {
            format!(
                "{:>3}  {cell}\n      {}",
                cases.len(),
                cases
                    .iter()
                    .take(2)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n      ")
            )
        })
        .collect();
    assert!(
        matrix.wrong.is_empty(),
        "{total} of {} wrong:\n{}",
        matrix.cases,
        table.join("\n")
    );
}

#[test]
fn a_quoted_name_cannot_hold_the_qualifier_separator() {
    let db = Database::open("memory://column_sources_separator").unwrap();
    setup(&db);
    for sql in [
        "CREATE TABLE \"x\u{1F}y\" (id INTEGER PRIMARY KEY)",
        "SELECT id AS \"a\u{1F}b\" FROM w",
        "SELECT \"w\u{1F}id\" FROM w",
    ] {
        let refused = db.query(sql, ()).err().map(|e| e.to_string());
        assert!(
            refused.as_deref().is_some_and(|e| e.contains("U+001F")),
            "{sql:?}: {refused:?}"
        );
    }
}

#[test]
fn a_quoted_name_no_column_has_is_its_text_over_every_source() {
    let db = Database::open("memory://column_sources_fallback").unwrap();
    for sql in [
        "CREATE TABLE q (id INTEGER PRIMARY KEY, \"a.b\" INTEGER)",
        "INSERT INTO q VALUES (1, 5)",
        "CREATE VIEW qv AS SELECT \"a.b\" FROM q",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for (with, from) in [
        ("", "q"),
        ("", "qv"),
        ("", "(SELECT 5 AS \"a.b\") AS d"),
        ("WITH c(\"a.b\") AS (SELECT 5) ", "c"),
        ("WITH c AS (SELECT 5 AS \"a.b\") ", "c"),
    ] {
        for (item, want) in [
            ("\"b\"", "[Some(Text(\"b\"))]"),
            ("\"b\" AS x", "[Some(Text(\"b\"))]"),
            ("\"a.b\"", "[Some(Integer(5))]"),
            ("\"a.b\" AS x", "[Some(Integer(5))]"),
        ] {
            let sql = format!("{with}SELECT {item} FROM {from}");
            for (run, got) in runs(&db, &sql) {
                if got.as_ref().map(|rows| rows.get(1).map(String::as_str)) != Ok(Some(want)) {
                    wrong.push(format!("{sql} ({run}): {got:?}, want {want}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_separator_inside_a_value_names_no_qualified_column() {
    let db = Database::open("memory://column_sources_value_separator").unwrap();
    setup(&db);
    let inner = "(SELECT 'a\u{1F}b', 'a.b') AS d";
    let names = |sql: &str| db.query(sql, ()).map(|rows| rows.columns().to_vec());
    let shown = names(&format!("SELECT * FROM {inner}")).unwrap();
    assert_ne!(shown[0], shown[1], "two items, two names: {shown:?}");
    let sql = format!("SELECT d.\"'a.b'\" FROM {inner}");
    let got = collect(db.query(&sql, ()));
    assert_eq!(
        got.as_ref().map(|rows| rows.get(1).cloned()),
        Ok(Some("[Some(Text(\"a.b\"))]".to_string())),
        "{sql}"
    );
    db.execute(&format!("CREATE TABLE copied AS SELECT * FROM {inner}"), ())
        .unwrap();
}

#[test]
fn a_caller_of_the_compiler_names_a_qualified_column_with_qualified() {
    use stoolap::executor::expression::compile_expression;
    use stoolap::executor::utils::qualified;
    use stoolap::parser::ast::Statement;
    let item = |sql: &str| match &stoolap::parser::parse_sql(sql).unwrap()[0] {
        Statement::Select(select) => select.columns[0].clone(),
        _ => unreachable!(),
    };
    let reference = item("SELECT t.id FROM t");
    assert!(compile_expression(&reference, &[qualified("t", "id")]).is_ok());
    // A dotted text is one name, which `t.id` is not
    assert!(compile_expression(&reference, &["t.id".to_string()]).is_err());
    assert!(compile_expression(&item("SELECT \"t.id\" FROM t"), &["t.id".to_string()]).is_ok());
}
