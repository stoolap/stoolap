// Copyright 2026 Stoolap Contributors
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

//! Subqueries over and inside FROM sources other than a table: views, CTEs,
//! derived tables, VALUES and table functions. Each cell answers as the same
//! statement over the tables the source copies, hot and sealed, unless
//! `state` says it fails until a plan step prepares it or names its
//! columns wrong until an issue is fixed

use stoolap::Database;

/// What a cell does today
#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    /// The answer of the same statement over the tables
    Right,
    /// An error that changes nothing, until the plan step named prepares it
    Fails(&'static str),
    /// The right rows under other column names, until the issue named is fixed
    WrongNames(&'static str),
}

use State::{Fails, Right, WrongNames};

/// A source over t or u: the WITH it needs, what FROM names, and whether
/// it holds every column of the table it copies
#[derive(Clone, Copy)]
struct Source {
    label: &'static str,
    with: &'static str,
    from: &'static str,
    whole: bool,
}

impl Source {
    /// Whether it reads the table it copies, rather than constants
    fn copies(&self) -> bool {
        !matches!(self.label, "values" | "series" | "recursive")
    }
}

/// t: id 1..6, a = id % 3, b = id * 10, and (7, NULL, NULL). u: id 1..3, x = id
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 0, 30), (4, 1, 40), (5, 2, 50), (6, 0, 60), (7, NULL, NULL)",
    "INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)",
    "CREATE TABLE t0 (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "CREATE TABLE u0 (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO t0 SELECT * FROM t",
    "INSERT INTO u0 SELECT * FROM u",
    "CREATE VIEW vt AS SELECT id, a, b FROM t",
    "CREATE VIEW vu AS SELECT id, x FROM u",
    "CREATE VIEW vb_where AS SELECT id, a FROM t WHERE id > (SELECT MAX(x) FROM u)",
    "CREATE VIEW vb_list AS SELECT id, (SELECT COUNT(*) FROM u WHERE u.id = t.a) AS c FROM t",
    "CREATE VIEW vb_exists AS SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.a)",
    "CREATE VIEW vb_having AS SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING COUNT(*) > (SELECT 1)",
    "CREATE VIEW vb_join AS SELECT t.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a)",
];

/// The sources over t, aliased s; the reference is t itself
const T_SOURCES: &[Source] = &[
    Source {
        label: "view",
        with: "",
        from: "vt AS s",
        whole: true,
    },
    Source {
        label: "derived",
        with: "",
        from: "(SELECT id, a, b FROM t) AS s",
        whole: true,
    },
    Source {
        label: "cte",
        with: "WITH s AS (SELECT id, a, b FROM t) ",
        from: "s",
        whole: true,
    },
    Source {
        label: "cte_columns",
        with: "WITH s(id, a, b) AS (SELECT id, a, b FROM t) ",
        from: "s",
        whole: true,
    },
    Source {
        label: "recursive",
        with: "WITH RECURSIVE s(id, a, b) AS (SELECT 1, 1, 10 UNION ALL SELECT id + 1, \
               CASE WHEN id + 1 = 7 THEN NULL ELSE (id + 1) % 3 END, \
               CASE WHEN id + 1 = 7 THEN NULL ELSE (id + 1) * 10 END FROM s WHERE id < 7) ",
        from: "s",
        whole: true,
    },
    Source {
        label: "values",
        with: "",
        from: "(VALUES (1, 1, 10), (2, 2, 20), (3, 0, 30), (4, 1, 40), (5, 2, 50), (6, 0, 60), \
               (7, NULL, NULL)) AS s(id, a, b)",
        whole: true,
    },
    Source {
        label: "series",
        with: "",
        from: "generate_series(1, 7) AS s(id)",
        whole: false,
    },
];

const T_REFERENCE: Source = Source {
    label: "table",
    with: "",
    from: "t AS s",
    whole: true,
};

/// The reference of a source of constants: a copy of t no write reaches
const T0_REFERENCE: Source = Source {
    label: "table",
    with: "",
    from: "t0 AS s",
    whole: true,
};

/// The sources over u, aliased q; the reference is u itself
const U_SOURCES: &[Source] = &[
    Source {
        label: "view",
        with: "",
        from: "vu AS q",
        whole: true,
    },
    Source {
        label: "derived",
        with: "",
        from: "(SELECT id, x FROM u) AS q",
        whole: true,
    },
    Source {
        label: "cte",
        with: "WITH q AS (SELECT id, x FROM u) ",
        from: "q",
        whole: true,
    },
    Source {
        label: "cte_columns",
        with: "WITH q(id, x) AS (SELECT id, x FROM u) ",
        from: "q",
        whole: true,
    },
    Source {
        label: "values",
        with: "",
        from: "(VALUES (1, 1), (2, 2), (3, 3)) AS q(id, x)",
        whole: true,
    },
    Source {
        label: "series",
        with: "",
        from: "generate_series(1, 3) AS q(x)",
        whole: false,
    },
];

const U_REFERENCE: Source = Source {
    label: "table",
    with: "",
    from: "u AS q",
    whole: true,
};

/// The reference of a source of constants: a copy of u no write reaches
const U0_REFERENCE: Source = Source {
    label: "table",
    with: "",
    from: "u0 AS q",
    whole: true,
};

/// A template: `{src}` is the source's FROM, its WITH goes first; `whole`
/// says it reads a column a table function's source lacks
struct Template {
    name: &'static str,
    sql: &'static str,
    whole: bool,
}

const fn tpl(name: &'static str, sql: &'static str, whole: bool) -> Template {
    Template { name, sql, whole }
}

/// The statement's subqueries over or beside a source over t
const OUTER: &[Template] = &[
    tpl(
        "where_scalar",
        "SELECT s.id FROM {src} WHERE s.id > (SELECT MAX(x) FROM u) ORDER BY s.id",
        false,
    ),
    tpl(
        "where_in",
        "SELECT s.id FROM {src} WHERE s.a IN (SELECT x FROM u) ORDER BY s.id",
        true,
    ),
    tpl(
        "where_not_in",
        "SELECT s.id FROM {src} WHERE s.a NOT IN (SELECT x FROM u) ORDER BY s.id",
        true,
    ),
    tpl(
        "where_all",
        "SELECT s.id FROM {src} WHERE s.a < ALL (SELECT x FROM u) ORDER BY s.id",
        true,
    ),
    tpl(
        "where_exists",
        "SELECT s.id FROM {src} WHERE EXISTS (SELECT 1 FROM u WHERE u.id = s.a) ORDER BY s.id",
        true,
    ),
    tpl(
        "where_correlated_scalar",
        "SELECT s.id FROM {src} WHERE s.a = (SELECT MAX(x) FROM u WHERE u.id = s.a) ORDER BY s.id",
        true,
    ),
    tpl(
        "where_both",
        "SELECT s.id FROM {src} WHERE s.id > (SELECT MAX(x) FROM u) \
         AND EXISTS (SELECT 1 FROM u WHERE u.id = s.a) ORDER BY s.id",
        true,
    ),
    tpl(
        "where_series_exists",
        "SELECT s.id FROM {src} WHERE EXISTS (SELECT 1 FROM u WHERE u.id = s.id) ORDER BY s.id",
        false,
    ),
    tpl(
        "select_scalar",
        "SELECT s.id, (SELECT MAX(x) FROM u) FROM {src} ORDER BY s.id",
        false,
    ),
    tpl(
        "select_correlated",
        "SELECT s.id, (SELECT COUNT(*) FROM u WHERE u.id = s.id) FROM {src} ORDER BY s.id",
        false,
    ),
    tpl(
        "order_by_correlated",
        "SELECT s.id FROM {src} ORDER BY (SELECT COUNT(*) FROM u WHERE u.id = s.id), s.id",
        false,
    ),
    tpl(
        "having",
        "SELECT s.a, COUNT(*) FROM {src} GROUP BY s.a HAVING COUNT(*) > (SELECT 1) ORDER BY s.a",
        true,
    ),
    tpl(
        "limit",
        "SELECT s.id FROM {src} ORDER BY s.id LIMIT (SELECT 2)",
        false,
    ),
];

/// A base table statement whose block reads a source over u
const IN_BLOCK: &[Template] = &[
    tpl(
        "block_in",
        "SELECT id FROM t WHERE a IN (SELECT x FROM {src}) ORDER BY id",
        false,
    ),
    tpl(
        "block_exists",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {src} WHERE q.x = t.a) ORDER BY id",
        false,
    ),
    tpl(
        "block_scalar",
        "SELECT id FROM t WHERE a = (SELECT MAX(x) FROM {src} WHERE q.x = t.a) ORDER BY id",
        false,
    ),
    tpl(
        "block_select_list",
        "SELECT id, (SELECT COUNT(*) FROM {src} WHERE q.x = t.a) FROM t ORDER BY id",
        false,
    ),
    tpl(
        "block_grandparent",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {src} \
         WHERE q.x IN (SELECT u.x FROM u WHERE u.id = t.a)) ORDER BY id",
        false,
    ),
    tpl(
        "block_unqualified_outer",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {src} WHERE x = a) ORDER BY id",
        false,
    ),
    tpl(
        "block_outer_arithmetic",
        "SELECT id, (SELECT MAX(q.x) FROM {src} WHERE q.x = t.id - 1) FROM t ORDER BY id",
        false,
    ),
];

/// A base table joined to a source over u
const JOINED: &[Template] = &[
    tpl(
        "join_where_scalar",
        "SELECT t.id FROM t JOIN {src} ON q.x = t.a WHERE t.id > (SELECT MIN(x) FROM u) ORDER BY t.id",
        false,
    ),
    tpl(
        "join_where_exists",
        "SELECT t.id FROM t JOIN {src} ON q.x = t.a WHERE EXISTS (SELECT 1 FROM u WHERE u.id = q.x) ORDER BY t.id",
        false,
    ),
    tpl("join_side_literal", "SELECT t.id FROM t JOIN {src} ON q.x = t.a WHERE q.x = 2 ORDER BY t.id", false),
    tpl(
        "join_on_correlated",
        "SELECT t.id FROM t JOIN {src} ON q.x = (SELECT MIN(w.x) FROM u w WHERE w.x = t.a) ORDER BY t.id",
        false,
    ),
];

/// Writes to t whose subqueries read a source over u
const WRITES: &[Template] = &[
    tpl("update_set", "UPDATE t SET b = (SELECT MAX(x) FROM {src})", false),
    tpl("update_set_correlated", "UPDATE t SET b = (SELECT COUNT(*) FROM {src} WHERE q.x = t.a)", false),
    tpl("update_where_in", "UPDATE t SET b = 0 WHERE a IN (SELECT x FROM {src})", false),
    tpl("delete_where_exists", "DELETE FROM t WHERE EXISTS (SELECT 1 FROM {src} WHERE q.x = t.a)", false),
    tpl("insert_values", "INSERT INTO t VALUES (8, (SELECT MAX(x) FROM {src}), 0)", false),
    tpl(
        "on_conflict_set",
        "INSERT INTO t VALUES (1, 0, 0) ON CONFLICT (id) DO UPDATE SET b = (SELECT MAX(x) FROM {src})",
        false,
    ),
    tpl("returning", "DELETE FROM t WHERE id = 1 RETURNING id, (SELECT MAX(x) FROM {src})", false),
];

/// Writes to t reading a source over t
const WRITES_OVER_T: &[Template] = &[
    tpl(
        "insert_select",
        "INSERT INTO t SELECT s.id + 10, s.a, s.b FROM {src} WHERE s.id > (SELECT MAX(x) FROM u)",
        true,
    ),
    tpl(
        "statement_point",
        "UPDATE t SET b = (SELECT s.b FROM {src} WHERE s.id = t.id - 1)",
        true,
    ),
];

/// What a source's body holding a subquery answers, against the statement
/// the body is: (source, cell, sql, reference)
const BODIES: &[(&str, &str, &str, &str)] = &[
    ("view", "body_where", "SELECT id FROM vb_where ORDER BY id", "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u) ORDER BY id"),
    ("view", "body_select_list", "SELECT id, c FROM vb_list ORDER BY id", "SELECT id, (SELECT COUNT(*) FROM u WHERE u.id = t.a) AS c FROM t ORDER BY id"),
    ("view", "body_exists", "SELECT id FROM vb_exists ORDER BY id", "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.a) ORDER BY id"),
    ("view", "body_having", "SELECT a, n FROM vb_having ORDER BY a", "SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING COUNT(*) > (SELECT 1) ORDER BY a"),
    ("view", "body_join_on", "SELECT id FROM vb_join ORDER BY id", "SELECT t.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a) ORDER BY t.id"),
    ("derived", "body_where", "SELECT id FROM (SELECT id, a FROM t WHERE id > (SELECT MAX(x) FROM u)) AS s ORDER BY id", "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u) ORDER BY id"),
    ("derived", "body_select_list", "SELECT id, c FROM (SELECT id, (SELECT COUNT(*) FROM u WHERE u.id = t.a) AS c FROM t) AS s ORDER BY id", "SELECT id, (SELECT COUNT(*) FROM u WHERE u.id = t.a) AS c FROM t ORDER BY id"),
    ("derived", "body_exists", "SELECT id FROM (SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.a)) AS s ORDER BY id", "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.a) ORDER BY id"),
    ("derived", "body_having", "SELECT a, n FROM (SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING COUNT(*) > (SELECT 1)) AS s ORDER BY a", "SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING COUNT(*) > (SELECT 1) ORDER BY a"),
    ("derived", "body_join_on", "SELECT id FROM (SELECT t.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a)) AS s ORDER BY id", "SELECT t.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a) ORDER BY t.id"),
    ("cte", "body_where", "WITH s AS (SELECT id, a FROM t WHERE id > (SELECT MAX(x) FROM u)) SELECT id FROM s ORDER BY id", "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u) ORDER BY id"),
    ("cte", "body_select_list", "WITH s AS (SELECT id, (SELECT COUNT(*) FROM u WHERE u.id = t.a) AS c FROM t) SELECT id, c FROM s ORDER BY id", "SELECT id, (SELECT COUNT(*) FROM u WHERE u.id = t.a) AS c FROM t ORDER BY id"),
    ("cte", "body_exists", "WITH s AS (SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.a)) SELECT id FROM s ORDER BY id", "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.a) ORDER BY id"),
    ("cte", "body_having", "WITH s AS (SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING COUNT(*) > (SELECT 1)) SELECT a, n FROM s ORDER BY a", "SELECT a, COUNT(*) AS n FROM t GROUP BY a HAVING COUNT(*) > (SELECT 1) ORDER BY a"),
    ("cte", "body_join_on", "WITH s AS (SELECT t.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a)) SELECT id FROM s ORDER BY id", "SELECT t.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a) ORDER BY t.id"),
    ("cte_columns", "body_where", "WITH s(id, a) AS (SELECT id, a FROM t WHERE id > (SELECT MAX(x) FROM u)) SELECT id FROM s ORDER BY id", "SELECT id FROM t WHERE id > (SELECT MAX(x) FROM u) ORDER BY id"),
    ("cte_columns", "body_join_on", "WITH s(id) AS (SELECT t.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a)) SELECT id FROM s ORDER BY id", "SELECT t.id FROM t JOIN u ON u.id = (SELECT MIN(w.id) FROM u w WHERE w.x = t.a) ORDER BY t.id"),
    ("recursive", "body_member", "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM s WHERE n < (SELECT MAX(x) FROM u)) SELECT n FROM s ORDER BY n", "SELECT x AS n FROM u ORDER BY x"),
    ("values", "body_row", "SELECT n FROM (VALUES ((SELECT MAX(x) FROM u)), (1)) AS s(n) ORDER BY n", "SELECT x AS n FROM u WHERE x IN (1, 3) ORDER BY x"),
    ("series", "body_argument", "SELECT * FROM generate_series(1, (SELECT MAX(x) FROM u))", "SELECT x AS value FROM u ORDER BY x"),
];

/// The states today; every cell not listed is Right
fn state(source: &str, cell: &str) -> State {
    const STEP_6: State = Fails("step 6");
    const GRAMMAR: State = Fails("step 6, grammar first");
    match (source, cell) {
        ("cte_columns" | "recursive" | "series", "limit") => WrongNames("#581"),
        ("series", "body_argument") => GRAMMAR,
        ("cte" | "cte_columns", "insert_values" | "on_conflict_set") => GRAMMAR,
        (_, "join_on_correlated") => STEP_6,
        (
            "derived",
            "block_grandparent"
            | "block_unqualified_outer"
            | "returning"
            | "insert_values"
            | "on_conflict_set",
        ) => STEP_6,
        ("view" | "derived" | "cte" | "cte_columns", "body_join_on") => STEP_6,
        ("values", "body_row") => STEP_6,
        _ => Right,
    }
}

fn open(name: &str, sealed: bool, dir: &tempfile::TempDir) -> Database {
    let db = if sealed {
        Database::open(&format!("file://{}/{name}", dir.path().display())).unwrap()
    } else {
        Database::open(&format!("memory://from_source_{name}")).unwrap()
    };
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The column names, then each row's values with their types; a NULL is
/// NULL whatever type it carries
fn rows(result: stoolap::Result<stoolap::Rows>) -> Result<Vec<Vec<String>>, String> {
    use stoolap::core::Value;
    let result = result.map_err(|e| e.to_string())?;
    let mut out = vec![result.columns().to_vec()];
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| match row.get_value(i) {
                    None | Some(Value::Null(_)) => "NULL".to_string(),
                    Some(Value::Integer(v)) => format!("integer {v}"),
                    Some(Value::Float(v)) => format!("float {v:?}"),
                    Some(Value::Text(v)) => format!("text {:?}", v.as_str()),
                    Some(Value::Boolean(v)) => format!("boolean {v}"),
                    Some(other) => format!("{other:?}"),
                })
                .collect(),
        );
    }
    Ok(out)
}

fn fill(template: &Template, source: Source) -> String {
    format!(
        "{}{}",
        source.with,
        template.sql.replace("{src}", source.from)
    )
}

/// How a read runs
#[derive(Clone, Copy, Debug)]
enum Context {
    Plain,
    /// Prepared, its second run
    Prepared,
    /// Inside a transaction that made the write, which changes the answer
    /// of a source copying the table, then rolls it back
    Transaction(&'static str),
}

fn read(db: &Database, sql: &str, context: Context) -> Result<Vec<Vec<String>>, String> {
    match context {
        Context::Plain => rows(db.query(sql, ())),
        Context::Prepared => {
            let statement = db.prepare(sql).map_err(|e| e.to_string())?;
            rows(statement.query(()))?;
            rows(statement.query(()))
        }
        Context::Transaction(write) => {
            let mut tx = db.begin().map_err(|e| e.to_string())?;
            tx.execute(write, ()).map_err(|e| e.to_string())?;
            let got = rows(tx.query(sql, ()));
            tx.rollback().map_err(|e| e.to_string())?;
            got
        }
    }
}

/// That a cell gave what its state says, else a line for the report
fn judge(
    what: String,
    state: State,
    got: Result<Vec<Vec<String>>, String>,
    want: Result<Vec<Vec<String>>, String>,
    wrong: &mut Vec<String>,
) {
    let want = match want {
        Ok(want) => want,
        Err(e) => return wrong.push(format!("{what}: the reference fails: {e}")),
    };
    match (state, got) {
        (Right, Ok(got)) if got == want => {}
        (Right, Ok(got)) => wrong.push(format!("{what}: gave {got:?}, want {want:?}")),
        (Right, Err(e)) => wrong.push(format!("{what}: should answer: {e}")),
        (Fails(_), Err(_)) => {}
        (Fails(owner), Ok(got)) if got == want => wrong.push(format!(
            "{what}: now answers right, mark it Right ({owner})"
        )),
        (Fails(owner), Ok(got)) => wrong.push(format!(
            "{what}: gave {got:?}, want {want:?} or an error until {owner}"
        )),
        (WrongNames(issue), Ok(got)) if got == want => wrong.push(format!(
            "{what}: now names its columns right, mark it Right ({issue})"
        )),
        (WrongNames(_), Ok(got)) if got[1..] == want[1..] => {}
        (WrongNames(issue), Ok(got)) => wrong.push(format!(
            "{what}: gave {got:?}, want {want:?} under other names ({issue})"
        )),
        (WrongNames(issue), Err(e)) => wrong.push(format!(
            "{what}: known wrong names ({issue}), now fails: {e}"
        )),
    }
}

/// Every read template over its sources, in `contexts`, hot and sealed
fn reads(
    group: &str,
    templates: &[Template],
    sources: &[Source],
    [copying, constant]: [Source; 2],
    contexts: &[Context],
) {
    let mut wrong = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    for sealed in [false, true] {
        let db = open(&format!("{group}_{sealed}"), sealed, &dir);
        for template in templates {
            for source in sources.iter().filter(|s| s.whole || !template.whole) {
                let sql = fill(template, *source);
                // A source of constants does not see a write its table takes
                let reference = if source.copies() { copying } else { constant };
                let want = fill(template, reference);
                for &context in contexts {
                    let what = format!(
                        "{}/{} {context:?} sealed {sealed}: {sql}",
                        source.label, template.name
                    );
                    judge(
                        what,
                        state(source.label, template.name),
                        read(&db, &sql, context),
                        read(&db, &want, context),
                        &mut wrong,
                    );
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} cells:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

const OVER_T_CONTEXTS: &[Context] = &[
    Context::Plain,
    Context::Prepared,
    Context::Transaction("INSERT INTO t VALUES (8, 0, 80)"),
];

const OVER_U_CONTEXTS: &[Context] = &[
    Context::Plain,
    Context::Prepared,
    Context::Transaction("INSERT INTO u VALUES (9, 0)"),
];

#[test]
fn subqueries_of_a_statement_over_a_source() {
    let references = [T_REFERENCE, T0_REFERENCE];
    reads("outer", OUTER, T_SOURCES, references, OVER_T_CONTEXTS);
}

#[test]
fn a_block_reading_a_source() {
    let references = [U_REFERENCE, U0_REFERENCE];
    reads("block", IN_BLOCK, U_SOURCES, references, OVER_U_CONTEXTS);
}

#[test]
fn a_table_joined_to_a_source() {
    let references = [U_REFERENCE, U0_REFERENCE];
    reads("joined", JOINED, U_SOURCES, references, OVER_U_CONTEXTS);
}

#[test]
fn a_source_whose_body_holds_a_subquery() {
    let mut wrong = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    for sealed in [false, true] {
        let db = open(&format!("bodies_{sealed}"), sealed, &dir);
        for (source, cell, sql, reference) in BODIES {
            for context in [Context::Plain, Context::Prepared] {
                judge(
                    format!("{source}/{cell} {context:?} sealed {sealed}: {sql}"),
                    state(source, cell),
                    read(&db, sql, context),
                    read(&db, reference, context),
                    &mut wrong,
                );
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} cells:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// What a write returned and left in t, or its error; a failed write must
/// leave t as it was
fn write(db: &Database, sql: &str) -> Result<Vec<Vec<String>>, String> {
    let table = || rows(db.query("SELECT * FROM t ORDER BY id", ()));
    written(table()?, rows(db.query(sql, ())), table()?)
}

/// [`write`] inside a transaction it then rolls back, so the next write
/// finds the rows as they were
fn write_rolled_back(db: &Database, sql: &str) -> Result<Vec<Vec<String>>, String> {
    let mut tx = db.begin().map_err(|e| e.to_string())?;
    let before = rows(tx.query("SELECT * FROM t ORDER BY id", ()))?;
    let got = rows(tx.query(sql, ()));
    let after = rows(tx.query("SELECT * FROM t ORDER BY id", ()))?;
    tx.rollback().map_err(|e| e.to_string())?;
    written(before, got, after)
}

fn written(
    before: Vec<Vec<String>>,
    got: Result<Vec<Vec<String>>, String>,
    after: Vec<Vec<String>>,
) -> Result<Vec<Vec<String>>, String> {
    match got {
        Ok(mut returned) => {
            returned.extend(after);
            Ok(returned)
        }
        Err(e) if after != before => Err(format!("failed but wrote: {e}")),
        Err(e) => Err(e),
    }
}

#[test]
fn a_write_reading_a_source() {
    let mut wrong = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    let groups: [(&[Template], &[Source], Source); 2] = [
        (WRITES, U_SOURCES, U_REFERENCE),
        (WRITES_OVER_T, T_SOURCES, T_REFERENCE),
    ];
    // Hot, each write on a database of its own in autocommit; sealed, every
    // write on one database inside a transaction it rolls back
    let sealed_db = open("writes_sealed", true, &dir);
    let mut n = 0;
    let mut run = |sealed: bool, sql: &str| match sealed {
        true => write_rolled_back(&sealed_db, sql),
        false => {
            n += 1;
            write(&open(&format!("writes_{n}"), false, &dir), sql)
        }
    };
    for sealed in [false, true] {
        for (templates, sources, reference) in groups {
            for template in templates {
                let want = run(sealed, &fill(template, reference));
                for source in sources.iter().filter(|s| s.whole || !template.whole) {
                    let sql = fill(template, *source);
                    let got = run(sealed, &sql);
                    if let Err(e) = &got {
                        if e.starts_with("failed but wrote") {
                            wrong.push(format!("{}/{}: {sql}: {e}", source.label, template.name));
                            continue;
                        }
                    }
                    judge(
                        format!("{}/{} sealed {sealed}: {sql}", source.label, template.name),
                        state(source.label, template.name),
                        got,
                        want.clone(),
                        &mut wrong,
                    );
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} cells:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
