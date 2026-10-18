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

//! One comparison rule in every operator and context. A TEXT against a
//! TIMESTAMP is read as a timestamp and against a BOOLEAN by the CAST rule,
//! UNKNOWN when it does not read; JSON values are equal or not, with no
//! order; a BOOLEAN or TIMESTAMP against a number has no order and is not equal.

use std::cmp::Ordering;

use stoolap::{Database, Value};

/// A row's value, the value it is compared with, the same as a parameter
/// when one is written, and the answers of `=` and of the order
struct Case {
    column: &'static str,
    stored: &'static str,
    compared: &'static str,
    param: Option<Value>,
    eq: Option<bool>,
    ord: Option<Ordering>,
}

fn cases() -> Vec<Case> {
    use Ordering::*;
    let case = |column, stored, compared, param, eq, ord| Case {
        column,
        stored,
        compared,
        param,
        eq,
        ord,
    };
    vec![
        case(
            "INTEGER",
            "1",
            "'1'",
            Some(Value::text("1")),
            Some(true),
            Some(Equal),
        ),
        case(
            "INTEGER",
            "1",
            "'2'",
            Some(Value::text("2")),
            Some(false),
            Some(Less),
        ),
        case("INTEGER", "1", "'x'", Some(Value::text("x")), None, None),
        case(
            "FLOAT",
            "1.5",
            "'1.5'",
            Some(Value::text("1.5")),
            Some(true),
            Some(Equal),
        ),
        case(
            "TIMESTAMP",
            "'2024-01-01 00:00:00'",
            "'2024-01-01'",
            Some(Value::text("2024-01-01")),
            Some(true),
            Some(Equal),
        ),
        case(
            "TIMESTAMP",
            "'2024-01-01 00:00:00'",
            "'2024-01-02'",
            Some(Value::text("2024-01-02")),
            Some(false),
            Some(Less),
        ),
        case(
            "TIMESTAMP",
            "'2024-01-01 00:00:00'",
            "'junk'",
            Some(Value::text("junk")),
            None,
            None,
        ),
        case(
            "BOOLEAN",
            "TRUE",
            "'true'",
            Some(Value::text("true")),
            Some(true),
            Some(Equal),
        ),
        case(
            "BOOLEAN",
            "TRUE",
            "'yes'",
            Some(Value::text("yes")),
            Some(true),
            Some(Equal),
        ),
        case(
            "BOOLEAN",
            "TRUE",
            "'f'",
            Some(Value::text("f")),
            Some(false),
            Some(Greater),
        ),
        case(
            "BOOLEAN",
            "TRUE",
            "'junk'",
            Some(Value::text("junk")),
            None,
            None,
        ),
        case(
            "BOOLEAN",
            "FALSE",
            "TRUE",
            Some(Value::Boolean(true)),
            Some(false),
            Some(Less),
        ),
        case(
            "BOOLEAN",
            "TRUE",
            "1",
            Some(Value::Integer(1)),
            Some(false),
            None,
        ),
        case(
            "TIMESTAMP",
            "'2024-01-01 00:00:00'",
            "1",
            Some(Value::Integer(1)),
            Some(false),
            None,
        ),
        case(
            "TEXT",
            "'abc'",
            "'abd'",
            Some(Value::text("abd")),
            Some(false),
            Some(Less),
        ),
        case(
            "JSON",
            "'{\"a\":1}'",
            "CAST('{\"a\":1}' AS JSON)",
            None,
            Some(true),
            None,
        ),
        case(
            "JSON",
            "'{\"a\":1}'",
            "CAST('{\"a\":2}' AS JSON)",
            None,
            Some(false),
            None,
        ),
        case("JSON", "'{\"a\":1}'", "'{\"a\":1}'", None, None, None),
    ]
}

/// What `value op compared` answers by the case's `=` and order
fn expected(case: &Case, op: &str) -> Option<bool> {
    let ord = case.ord;
    match op {
        "=" => case.eq,
        "<>" => case.eq.map(|eq| !eq),
        "<" => ord.map(|o| o == Ordering::Less),
        "<=" => ord.map(|o| o != Ordering::Greater),
        ">" => ord.map(|o| o == Ordering::Greater),
        ">=" => ord.map(|o| o != Ordering::Less),
        "IS DISTINCT FROM" => Some(case.eq != Some(true)),
        "IS NOT DISTINCT FROM" => Some(case.eq == Some(true)),
        "BETWEEN" => ord.map(|o| o == Ordering::Equal),
        "IN" => case.eq,
        _ => unreachable!(),
    }
}

const OPS: [&str; 10] = [
    "=",
    "<>",
    "<",
    "<=",
    ">",
    ">=",
    "IS DISTINCT FROM",
    "IS NOT DISTINCT FROM",
    "BETWEEN",
    "IN",
];

/// `v op c`, with `c` as both bounds of BETWEEN and the one member of IN
fn predicate(v: &str, op: &str, c: &str) -> String {
    match op {
        "BETWEEN" => format!("{v} BETWEEN {c} AND {c}"),
        "IN" => format!("{v} IN ({c})"),
        _ => format!("{v} {op} {c}"),
    }
}

fn values(db: &Database, sql: &str, params: Vec<Value>) -> Result<Vec<Value>, String> {
    db.query(sql, params)
        .and_then(|rows| {
            rows.map(|row| row.map(|row| row.get_value(0).cloned().unwrap()))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|e| e.to_string())
}

fn truth(value: Option<bool>) -> Value {
    value.map_or(Value::Null(stoolap::DataType::Boolean), Value::Boolean)
}

/// Every wrong answer for `case` in every context, as `context: sql`
fn wrong_answers(db: &Database, case: &Case, wrong: &mut Vec<String>) {
    let mut check = |context: &str, sql: String, params: Vec<Value>, want: Vec<Value>| {
        let got = values(db, &sql, params);
        if got.as_ref() != Ok(&want) {
            wrong.push(format!("{context}: {sql} gave {got:?}, want {want:?}"));
        }
    };
    let (v, c) = ("v", case.compared);
    let one = || vec![Value::Integer(1)];
    let none = Vec::new;
    for op in OPS {
        let want = expected(case, op);
        let kept = |answer: Option<bool>| if answer == Some(true) { one() } else { none() };
        let p = predicate(v, op, c);
        check(
            "WHERE",
            format!("SELECT id FROM c WHERE {p}"),
            vec![],
            kept(want),
        );
        check(
            "WHERE NOT",
            format!("SELECT id FROM c WHERE NOT ({p})"),
            vec![],
            kept(want.map(|w| !w)),
        );
        check(
            "prepared WHERE",
            format!("SELECT id FROM c WHERE {p} AND (SELECT 1) = 1"),
            vec![],
            kept(want),
        );
        check(
            "SELECT list",
            format!("SELECT {p} FROM c"),
            vec![],
            vec![truth(want)],
        );
        let three = |w: Option<bool>| {
            Value::Integer(match w {
                Some(true) => 1,
                Some(false) => 0,
                None => 2,
            })
        };
        let searched = format!("CASE WHEN {p} THEN 1 WHEN NOT ({p}) THEN 0 ELSE 2 END");
        check(
            "CASE",
            format!("SELECT {searched} FROM c"),
            vec![],
            vec![three(want)],
        );
        check(
            "CASE fast path",
            format!("SELECT CASE WHEN {p} THEN 1 ELSE 0 END FROM c ORDER BY id LIMIT 1"),
            vec![],
            vec![Value::Integer(i64::from(want == Some(true)))],
        );
        check(
            "HAVING",
            format!("SELECT id FROM c GROUP BY id, {v} HAVING {p}"),
            vec![],
            kept(want),
        );
        if let Some(param) = &case.param {
            let p = predicate(v, op, "$1");
            check(
                "parameter",
                format!("SELECT id FROM c WHERE {p}"),
                vec![param.clone()],
                kept(want),
            );
            check(
                "parameter, prepared",
                format!("SELECT id FROM c WHERE {p} AND (SELECT 1) = 1"),
                vec![param.clone()],
                kept(want),
            );
        }
    }
    let matched = case.eq == Some(true);
    check(
        "simple CASE",
        format!("SELECT CASE {v} WHEN {c} THEN 1 ELSE 0 END FROM c"),
        vec![],
        vec![Value::Integer(matched as i64)],
    );
    check(
        "NULLIF",
        format!("SELECT NULLIF({v}, {c}) IS NULL FROM c"),
        vec![],
        vec![Value::Boolean(matched)],
    );
}

fn run(dsn: impl Fn(usize) -> String, seal: bool) -> Vec<String> {
    let mut wrong = Vec::new();
    for (n, case) in cases().iter().enumerate() {
        let db = Database::open(&dsn(n)).unwrap();
        db.execute(
            &format!("CREATE TABLE c (id INTEGER PRIMARY KEY, v {})", case.column),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO c VALUES (1, {})", case.stored), ())
            .unwrap();
        if seal {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        wrong_answers(&db, case, &mut wrong);
    }
    wrong
}

#[test]
fn test_one_comparison_rule_answers_every_operator_and_context() {
    let wrong = run(|n| format!("memory://comparison_rule_{n}"), false);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_one_comparison_rule_answers_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let wrong = run(
        |n| {
            format!(
                "file://{}/rule_{n}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
                dir.path().display()
            )
        },
        true,
    );
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
