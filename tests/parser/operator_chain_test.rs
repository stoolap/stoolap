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

//! A long chain of one operator answers, or fails with an error where it is
//! parsed, instead of overflowing the stack of a later stage

use stoolap::parser::MAX_EXPRESSION_DEPTH;
use stoolap::Database;

const TERMS: usize = 20_000;

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

/// `term(1) op term(2) op ... term(n)`
fn chain(n: usize, operator: &str, term: impl Fn(usize) -> String) -> String {
    let terms: Vec<String> = (1..=n).map(term).collect();
    terms.join(&format!(" {operator} "))
}

fn ors() -> String {
    chain(TERMS, "OR", |i| format!("a = {i}"))
}

fn ands() -> String {
    chain(TERMS, "AND", |_| "a > 0".to_string())
}

fn setup(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 5), (3, 30000)", ())
        .unwrap();
}

fn first_column(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    db.query(sql, ())
        .and_then(|rows| rows.map(|row| row?.get::<i64>(0)).collect())
        .map_err(|e| e.to_string())
}

fn queries() -> Vec<(&'static str, String, Vec<i64>)> {
    let (ors, ands) = (ors(), ands());
    vec![
        (
            "where or",
            format!("SELECT a FROM t WHERE {ors} ORDER BY a"),
            vec![1, 5],
        ),
        (
            "where and",
            format!("SELECT a FROM t WHERE {ands} ORDER BY a"),
            vec![1, 5, 30000],
        ),
        (
            "having",
            format!("SELECT a FROM t GROUP BY a HAVING {ors} ORDER BY a"),
            vec![1, 5],
        ),
        (
            "join on",
            format!(
                "SELECT t.a FROM t JOIN t u ON t.id = u.id AND {} ORDER BY t.a",
                chain(TERMS, "AND", |_| "u.a > 1".to_string())
            ),
            vec![5, 30000],
        ),
        (
            "exists",
            format!(
                "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM t u WHERE u.id = t.id AND ({ors})) ORDER BY a"
            ),
            vec![1, 5],
        ),
        (
            "derived",
            format!("SELECT a FROM (SELECT a FROM t WHERE {ands}) d WHERE {ors} ORDER BY a"),
            vec![1, 5],
        ),
        (
            "select list",
            format!(
                "SELECT CASE WHEN {} THEN LENGTH({}) END",
                chain(TERMS, "AND", |_| "TRUE".to_string()),
                chain(TERMS, "||", |_| "'a'".to_string())
            ),
            vec![TERMS as i64],
        ),
    ]
}

#[test]
fn chains_of_and_or_and_concatenation_answer() {
    on_small_stack(|| {
        let db = Database::open("memory://operator_chain_answers").unwrap();
        setup(&db);
        let mut wrong = Vec::new();
        for (shape, sql, expected) in queries() {
            match first_column(&db, &sql) {
                Ok(rows) if rows == expected => {}
                other => wrong.push(format!("{shape}: {other:?}")),
            }
            let explain = db
                .query(&format!("EXPLAIN {sql}"), ())
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
            if let Err(e) = explain {
                wrong.push(format!("EXPLAIN {shape}: {e}"));
            }
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    });
}

#[test]
fn chains_answer_prepared_and_on_sealed_rows() {
    on_small_stack(|| {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        setup(&db);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert!(!db.engine().volume_stats().is_empty(), "sealed");
        db.execute("INSERT INTO t VALUES (4, 2)", ()).unwrap();
        let mut wrong = Vec::new();
        for (shape, sql, expected) in [
            (
                "or",
                format!("SELECT a FROM t WHERE {} ORDER BY a", ors()),
                vec![1, 2, 5],
            ),
            (
                "and",
                format!("SELECT a FROM t WHERE {} ORDER BY a", ands()),
                vec![1, 2, 5, 30000],
            ),
        ] {
            match first_column(&db, &sql) {
                Ok(rows) if rows == expected => {}
                other => wrong.push(format!("{shape}: {other:?}")),
            }
        }
        let params = chain(TERMS, "AND", |_| "a <> $1".to_string());
        let statement = db
            .prepare(&format!("SELECT a FROM t WHERE {params} ORDER BY a"))
            .unwrap();
        for _ in 0..2 {
            let rows: Vec<i64> = statement
                .query((5,))
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect();
            if rows != vec![1, 2, 30000] {
                wrong.push(format!("prepared: {rows:?}"));
            }
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    });
}

#[test]
fn chains_answer_in_writes_checks_and_views() {
    on_small_stack(|| {
        let db = Database::open("memory://operator_chain_writes").unwrap();
        setup(&db);
        let ors = ors();
        let updated = db
            .execute(&format!("UPDATE t SET a = a + 1 WHERE {ors}"), ())
            .unwrap();
        assert_eq!(updated, 2, "UPDATE");
        let deleted = db
            .execute(&format!("DELETE FROM t WHERE {}", ands()), ())
            .unwrap();
        assert_eq!(deleted, 3, "DELETE");

        // A stored CHECK's text is written with a 16-bit length
        let check = chain(2_000, "AND", |_| "x > 0".to_string());
        db.execute(&format!("CREATE TABLE c (x INTEGER CHECK ({check}))"), ())
            .unwrap();
        db.execute("INSERT INTO c VALUES (1)", ()).unwrap();
        assert!(db.execute("INSERT INTO c VALUES (0)", ()).is_err(), "CHECK");

        setup_rows(&db);
        db.execute(&format!("CREATE VIEW v AS SELECT a FROM t WHERE {ors}"), ())
            .unwrap();
        assert_eq!(
            first_column(&db, "SELECT a FROM v ORDER BY a"),
            Ok(vec![1, 5]),
            "view"
        );
    });
}

fn setup_rows(db: &Database) {
    db.execute("INSERT INTO t VALUES (1, 1), (2, 5), (3, 30000)", ())
        .unwrap();
}

#[test]
fn a_chain_cut_short_by_a_syntax_error_fails_with_it() {
    on_small_stack(|| {
        let db = Database::open("memory://operator_chain_syntax_error").unwrap();
        setup(&db);
        let terms = 10 * TERMS;
        let long = |operator: &str| chain(terms, operator, |i| format!("a = {i}"));
        let mut wrong = Vec::new();
        for (name, sql) in [
            (
                "and, then )",
                format!("SELECT a FROM t WHERE {} AND )", long("AND")),
            ),
            (
                "or, then (",
                format!("SELECT a FROM t WHERE {} OR (", long("OR")),
            ),
            (
                "or at the end",
                format!("SELECT a FROM t WHERE {} OR", long("OR")),
            ),
            (
                "concatenation, then )",
                format!("SELECT {} || )", chain(terms, "||", |_| "'a'".to_string())),
            ),
            (
                "concatenation, then ALL without (",
                format!(
                    "SELECT {} || ALL 1",
                    chain(terms, "||", |_| "'a'".to_string())
                ),
            ),
        ] {
            match first_column(&db, &sql) {
                Err(e) if e.contains("parse error") || e.contains("expected") => {}
                other => wrong.push(format!("{name}: {other:?}")),
            }
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    });
}

#[test]
fn other_chains_nest_one_level_per_operator() {
    on_small_stack(|| {
        let db = Database::open("memory://operator_chain_nesting").unwrap();
        // Operators, the first operand one level below the deepest
        let deepest = MAX_EXPRESSION_DEPTH - 1;
        let mut wrong = Vec::new();
        let operators = deepest as i64;
        for (operator, answer) in [("+", operators + 1), ("-", 1 - operators), ("*", 1)] {
            let at = format!(
                "SELECT {}",
                chain(deepest + 1, operator, |_| "1".to_string())
            );
            match first_column(&db, &at) {
                Ok(rows) if rows == vec![answer] => {}
                other => wrong.push(format!("{operator} at {deepest}: {other:?}")),
            }
            for past in [deepest + 1, TERMS] {
                let sql = format!("SELECT {}", chain(past + 1, operator, |_| "1".to_string()));
                match first_column(&db, &sql) {
                    Err(e) if e.contains(&expressions_error()) => {}
                    other => wrong.push(format!("{operator} at {past}: {other:?}")),
                }
            }
        }
        // A chain's operands nest under it: half the limit in CASE under a
        // chain of half the limit is past it
        let half = MAX_EXPRESSION_DEPTH / 2;
        let case = format!(
            "{}1{}",
            "CASE WHEN TRUE THEN ".repeat(half),
            " END".repeat(half)
        );
        let sql = format!(
            "SELECT {case} + {}",
            chain(half + 1, "+", |_| "1".to_string())
        );
        match first_column(&db, &sql) {
            Err(e) if e.contains(&expressions_error()) => {}
            other => wrong.push(format!("case under a chain: {other:?}")),
        }
        // An operand nests under the operators above it, not under the whole
        // chain: the `*` chain is the right operand of the last `+` only
        let ones = |n: usize, operator: &str| chain(n + 1, operator, |_| "1".to_string());
        let sum = |pluses: usize, times: usize| {
            format!("SELECT {} + {}", ones(pluses - 1, "+"), ones(times, "*"))
        };
        match first_column(&db, &sum(deepest, deepest - 1)) {
            Ok(rows) if rows == vec![deepest as i64 + 1] => {}
            other => wrong.push(format!("product under a sum: {other:?}")),
        }
        match first_column(&db, &sum(deepest, deepest)) {
            Err(e) if e.contains(&expressions_error()) => {}
            other => wrong.push(format!("product under a sum, past: {other:?}")),
        }
        // The fifth of five ANDs sits two levels below their root, one below
        // the CASE
        let fifth = |pluses: usize| {
            format!(
                "SELECT CASE WHEN TRUE AND TRUE AND TRUE AND TRUE AND {} = {} THEN 1 END",
                ones(pluses, "+"),
                pluses + 1
            )
        };
        match first_column(&db, &fifth(deepest - 4)) {
            Ok(rows) if rows == vec![1] => {}
            other => wrong.push(format!("sum in a balanced chain: {other:?}")),
        }
        match first_column(&db, &fifth(deepest - 3)) {
            Err(e) if e.contains(&expressions_error()) => {}
            other => wrong.push(format!("sum in a balanced chain, past: {other:?}")),
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    });
}

#[test]
fn a_balanced_chain_stops_where_the_written_one_does() {
    let db = Database::open("memory://operator_chain_short_circuit").unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, p TEXT)",
        (),
    )
    .unwrap();
    // REGEXP refuses the pattern '[', so the chain fails if it reads it
    db.execute("INSERT INTO r VALUES (1, 1, 0, '[')", ())
        .unwrap();
    let mut wrong = Vec::new();
    for (sql, expected) in [
        (
            "SELECT CASE WHEN a = 1 AND a = 1 AND b AND 'x' REGEXP p THEN 1 ELSE 2 END FROM r",
            vec![2],
        ),
        (
            "SELECT CASE WHEN a = 2 OR a = 2 OR b + 1 OR 'x' REGEXP p THEN 1 ELSE 2 END FROM r",
            vec![1],
        ),
        (
            "SELECT id FROM r WHERE NOT (a = 1 AND a = 1 AND b AND 'x' REGEXP p)",
            vec![1],
        ),
        (
            "SELECT id FROM r WHERE a = 2 OR a = 2 OR b + 1 OR 'x' REGEXP p",
            vec![1],
        ),
    ] {
        match first_column(&db, sql) {
            Ok(rows) if rows == expected => {}
            other => wrong.push(format!("{sql}: {other:?}")),
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn a_chain_groups_by_any_grouping_of_itself() {
    let db = Database::open("memory://operator_chain_grouping").unwrap();
    db.execute(
        "CREATE TABLE g (id INTEGER PRIMARY KEY, a TEXT, b TEXT, c TEXT, d TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO g VALUES (1, 'a', 'b', 'c', 'd')", ())
        .unwrap();
    let mut wrong = Vec::new();
    for (sql, expected) in [
        (
            "SELECT a || b || c || d, COUNT(*) FROM g GROUP BY (((a || b) || c) || d)",
            "abcd",
        ),
        (
            "SELECT ((a || b) || c) || d, COUNT(*) FROM g GROUP BY a || b || c || d",
            "abcd",
        ),
        (
            "SELECT a || (b || c), COUNT(*) FROM g GROUP BY (a || b) || c",
            "abc",
        ),
        (
            "SELECT CASE WHEN a = 'a' AND b = 'b' AND c = 'c' AND d = 'd' THEN 'abcd' END, COUNT(*) \
             FROM g GROUP BY (((a = 'a' AND b = 'b') AND c = 'c') AND d = 'd')",
            "abcd",
        ),
    ] {
        let answer = db.query(sql, ()).and_then(|rows| {
            rows.map(|row| {
                let row = row?;
                Ok((row.get::<String>(0)?, row.get::<i64>(1)?))
            })
            .collect::<Result<Vec<_>, _>>()
        });
        match answer {
            Ok(rows) if rows == vec![(expected.to_string(), 1)] => {}
            other => wrong.push(format!("{sql}: {other:?}")),
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}
