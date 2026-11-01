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

const SELECTS: &str = "statement nested too deeply: more than 32 SELECTs inside one another";
const EXPRESSIONS: &str =
    "statement nested too deeply: more than 256 expressions inside one another";

/// `run` on a thread with a main thread's stack: a debug build's stages at
/// the limits take more than a test thread's
fn on_main_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(8 << 20)
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
        "correlated" => format!("SELECT {} FROM t", wrap("(SELECT ", "t.a", ")")),
        "derived" => format!("SELECT * FROM {}", wrap("(SELECT * FROM ", "t", ") d")),
        "exists" => format!(
            "SELECT a FROM t WHERE {}",
            wrap("EXISTS (SELECT 1 WHERE ", "TRUE", ")")
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
    assert_eq!((MAX_STATEMENT_DEPTH, MAX_EXPRESSION_DEPTH), (32, 256));
    on_main_stack(|| {
        let db = setup("nesting_limit_edges");
        let mut wrong = Vec::new();
        // The outermost SELECT and its first expression count too
        for (shape, deepest, error) in [
            ("scalar", 31, SELECTS),
            ("correlated", 31, SELECTS),
            ("derived", 31, SELECTS),
            ("exists", 31, SELECTS),
            ("parentheses", 255, EXPRESSIONS),
            ("case", 255, EXPRESSIONS),
            ("call", 255, EXPRESSIONS),
            ("not", 255, EXPRESSIONS),
        ] {
            let sql = nested(shape, deepest);
            if let Err(e) = answer(&db, &sql) {
                wrong.push(format!("{shape} at {deepest}: {e}"));
            }
            for past in [deepest + 1, 10_000] {
                match answer(&db, &nested(shape, past)) {
                    Err(e) if e.contains(error) => {}
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
    // Twenty SELECTs of twenty parentheses each nest 420 expressions
    let level = |inner: String| format!("(SELECT {}{inner}{})", "(".repeat(20), ")".repeat(20));
    let sql = format!(
        "SELECT {}",
        (0..20).fold("1".to_string(), |inner, _| level(inner))
    );
    let refused = answer(&db, &sql);
    assert!(
        refused.as_ref().is_err_and(|e| e.contains(EXPRESSIONS)),
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
            refused.as_deref().is_some_and(|e| e.contains(EXPRESSIONS)),
            "{}: {refused:?}",
            &sql[..20]
        );
    }
    let parsed = stoolap::parser::parse_sql(&nested("scalar", 10_000));
    assert!(
        parsed.is_err_and(|e| e.to_string().contains(SELECTS)),
        "the parser alone refuses it"
    );
}

#[test]
fn a_long_set_operation_is_not_nested() {
    let db = setup("nesting_limit_union");
    let sql = format!("SELECT 1{}", " UNION ALL SELECT 1".repeat(1000));
    assert_eq!(answer(&db, &sql), Ok(1001));
}
