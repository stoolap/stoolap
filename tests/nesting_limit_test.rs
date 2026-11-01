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

//! A statement nested past the parser's limits fails with an error where it
//! is parsed, instead of overflowing the stack of a later stage

use stoolap::parser::{MAX_EXPRESSION_DEPTH, MAX_STATEMENT_DEPTH};
use stoolap::Database;

fn selects_error() -> String {
    format!(
        "statement nested too deeply: more than {MAX_STATEMENT_DEPTH} SELECTs inside one another"
    )
}

fn expressions_error() -> String {
    format!(
        "statement nested too deeply: more than {MAX_EXPRESSION_DEPTH} expressions inside one another"
    )
}

/// `run` on a thread of 2 MB, the stack a spawned thread and many runtimes'
/// workers take by default
fn on_small_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
    db
}

fn answer(db: &Database, sql: &str) -> Result<usize, String> {
    db.query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map(|rows| rows.len())
        .map_err(|e| e.to_string())
}

/// Each shape, nesting `n` levels below its statement
fn nested(shape: &str, n: usize) -> String {
    let wrap = |open: &str, inner: &str, close: &str| {
        format!("{}{inner}{}", open.repeat(n), close.repeat(n))
    };
    match shape {
        "scalar" => format!("SELECT {}", wrap("(SELECT ", "1", ")")),
        "correlated" => format!("SELECT {} FROM t", wrap("(SELECT t.a + ", "1", ")")),
        "derived" => format!("SELECT * FROM {}", wrap("(SELECT * FROM ", "t", ") d")),
        "exists" => format!(
            "SELECT a FROM t WHERE {}",
            wrap("EXISTS (SELECT 1 WHERE ", "TRUE", ")")
        ),
        "cte" => format!(
            "WITH c AS ({}) SELECT * FROM c",
            wrap("SELECT * FROM (", "SELECT 1", ") d")
        ),
        "parentheses" => format!("SELECT {}", wrap("(", "1", ")")),
        "case" => format!("SELECT {}", wrap("CASE WHEN TRUE THEN ", "1", " END")),
        "call" => format!("SELECT {}", wrap("ABS(", "1", ")")),
        "not" => format!("SELECT {}", wrap("NOT ", "TRUE", "")),
        _ => unreachable!(),
    }
}

#[test]
fn a_statement_at_the_limits_answers_and_one_past_them_fails() {
    on_small_stack(|| {
        let db = setup("nesting_limit_edges");
        let (selects, expressions) = (MAX_STATEMENT_DEPTH, MAX_EXPRESSION_DEPTH);
        let mut wrong = Vec::new();
        // The outermost SELECT and its first expression count too; a CTE body
        // sits inside the statement that carries it
        for (shape, deepest, error) in [
            ("scalar", selects - 1, selects_error()),
            ("correlated", selects - 1, selects_error()),
            ("derived", selects - 1, selects_error()),
            ("exists", selects - 1, selects_error()),
            ("cte", selects - 2, selects_error()),
            ("parentheses", expressions - 1, expressions_error()),
            ("case", expressions - 1, expressions_error()),
            ("call", expressions - 1, expressions_error()),
            ("not", expressions - 1, expressions_error()),
        ] {
            let sql = nested(shape, deepest);
            if let Err(e) = answer(&db, &sql) {
                wrong.push(format!("{shape} at {deepest}: {e}"));
            }
            for past in [deepest + 1, 10_000] {
                match answer(&db, &nested(shape, past)) {
                    Err(e) if e.contains(&error) => {}
                    other => wrong.push(format!("{shape} at {past}: {other:?}")),
                }
            }
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    });
}

#[test]
fn expressions_count_across_the_selects_they_nest() {
    let db = setup("nesting_limit_across");
    // Each SELECT holds parentheses enough that two of them pass the limit
    let depth = MAX_EXPRESSION_DEPTH / 2;
    let level =
        |inner: String| format!("(SELECT {}{inner}{})", "(".repeat(depth), ")".repeat(depth));
    let sql = format!("SELECT {}", level(level("1".to_string())));
    let refused = answer(&db, &sql);
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains(&expressions_error())),
        "{refused:?}"
    );
}

#[test]
fn a_write_and_a_parse_alone_are_refused_too() {
    let db = setup("nesting_limit_writes");
    let deep = format!("{}1{}", "(".repeat(10_000), ")".repeat(10_000));
    for sql in [
        format!("UPDATE t SET a = {deep}"),
        format!("INSERT INTO t VALUES (2, {deep})"),
        format!("DELETE FROM t WHERE a = {deep}"),
    ] {
        let refused = db.execute(&sql, ()).err().map(|e| e.to_string());
        assert!(
            refused
                .as_deref()
                .is_some_and(|e| e.contains(&expressions_error())),
            "{}: {refused:?}",
            &sql[..20]
        );
    }
    let parsed = stoolap::parser::parse_sql(&nested("scalar", 10_000));
    assert!(
        parsed.is_err_and(|e| e.to_string().contains(&selects_error())),
        "the parser alone refuses it"
    );
}

#[test]
fn a_long_set_operation_is_not_nested() {
    let db = setup("nesting_limit_union");
    let sql = format!("SELECT 1{}", " UNION ALL SELECT 1".repeat(1000));
    assert_eq!(answer(&db, &sql), Ok(1001));
}

#[test]
fn text_the_engine_wrote_reads_back_past_the_limits() {
    let db = setup("nesting_limit_written");
    // The engine writes each operator of a chain in parentheses, nesting
    // them deeper than the expression limit
    let chain = |term: &str, op: &str| {
        let terms = vec![term; MAX_EXPRESSION_DEPTH * 2];
        terms.join(op)
    };
    let sum = chain("1", " + ");
    let checks = chain("v >= 0", " OR ");
    for sql in [
        format!(
            "CREATE TABLE w (id INTEGER PRIMARY KEY, v INTEGER DEFAULT ({sum}) CHECK ({checks}))"
        ),
        "INSERT INTO w (id) VALUES (1)".to_string(),
        format!("CREATE VIEW wv AS SELECT id FROM w WHERE {checks}"),
    ] {
        db.execute(&sql, ())
            .unwrap_or_else(|e| panic!("{}: {e}", &sql[..30]));
    }
    let v: i64 = db.query_one("SELECT v FROM w WHERE id = 1", ()).unwrap();
    assert_eq!(
        v,
        (MAX_EXPRESSION_DEPTH * 2) as i64,
        "the stored DEFAULT reads back"
    );
    assert_eq!(
        answer(&db, "SELECT * FROM wv"),
        Ok(1),
        "the stored view reads back"
    );
    // The view's text as the engine wrote it, read back, is still held to
    // the limits when a user sends it
    let written = stoolap::parser::parse_sql(&format!("SELECT id FROM w WHERE {checks}"))
        .unwrap()
        .remove(0)
        .to_string();
    let refused = answer(&db, &written);
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains(&expressions_error())),
        "{refused:?}"
    );
}

#[test]
fn a_with_that_starts_at_the_limit_is_counted() {
    let db = setup("nesting_limit_with_at_limit");
    let wrappers = MAX_STATEMENT_DEPTH - 1;
    let body = nested("derived", 10_000);
    let sql = format!(
        "SELECT * FROM {}(WITH c AS ({body}) SELECT * FROM c) w{}",
        "(SELECT * FROM ".repeat(wrappers),
        ") d".repeat(wrappers)
    );
    let refused = answer(&db, &sql);
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains(&selects_error())),
        "{refused:?}"
    );
}
