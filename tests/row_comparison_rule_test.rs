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

//! Two rows of one width compare value by value: `=` is false when a pair
//! is unequal, else UNKNOWN when a pair holds a NULL, else true; `<>` is its
//! negation; an ordering is decided by the first pair that is not equal, and
//! is UNKNOWN when that pair holds a NULL; IS [NOT] DISTINCT FROM compares
//! every pair as it compares two values

use stoolap::{Database, Value};

/// Two integers or NULLs
type Pair = [Option<i64>; 2];

const OPERATORS: [&str; 8] = [
    "=",
    "<>",
    "<",
    "<=",
    ">",
    ">=",
    "IS DISTINCT FROM",
    "IS NOT DISTINCT FROM",
];

/// The rule, over pairs of integers or NULLs
fn expected(op: &str, a: Pair, b: Pair) -> Option<bool> {
    let pairs = || a.iter().zip(b.iter());
    let distinct = pairs().any(|(x, y)| x != y);
    let equal = if pairs().any(|(x, y)| matches!((x, y), (Some(x), Some(y)) if x != y)) {
        Some(false)
    } else if pairs().any(|(x, y)| x.is_none() || y.is_none()) {
        None
    } else {
        Some(true)
    };
    let ordering = || {
        for (x, y) in pairs() {
            match (x, y) {
                (Some(x), Some(y)) if x == y => {}
                (Some(x), Some(y)) => return Some(x.cmp(y)),
                _ => return None,
            }
        }
        Some(std::cmp::Ordering::Equal)
    };
    match op {
        "=" => equal,
        "<>" => equal.map(|eq| !eq),
        "<" => ordering().map(|o| o.is_lt()),
        "<=" => ordering().map(|o| o.is_le()),
        ">" => ordering().map(|o| o.is_gt()),
        ">=" => ordering().map(|o| o.is_ge()),
        "IS DISTINCT FROM" => Some(distinct),
        _ => Some(!distinct),
    }
}

fn value(v: Option<i64>) -> Value {
    v.map_or(Value::null_unknown(), Value::Integer)
}

fn truth(cell: Option<&Value>) -> Option<bool> {
    match cell {
        Some(Value::Boolean(b)) => Some(*b),
        Some(Value::Integer(i)) => Some(*i == 1),
        Some(Value::Null(_)) => None,
        other => panic!("not a truth value: {other:?}"),
    }
}

/// Every pair of rows over 1, 2 and NULL, one table row each
fn combinations() -> Vec<(i64, Pair, Pair)> {
    let values = [Some(1), Some(2), None];
    let mut out = Vec::new();
    for a1 in values {
        for a2 in values {
            for b1 in values {
                for b2 in values {
                    out.push((out.len() as i64 + 1, [a1, a2], [b1, b2]));
                }
            }
        }
    }
    out
}

fn load(db: &Database) {
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, a1 INTEGER, a2 INTEGER, b1 INTEGER, b2 INTEGER)",
        (),
    )
    .unwrap();
    for (id, a, b) in combinations() {
        db.execute(
            "INSERT INTO r VALUES ($1, $2, $3, $4, $5)",
            vec![
                Value::Integer(id),
                value(a[0]),
                value(a[1]),
                value(b[0]),
                value(b[1]),
            ],
        )
        .unwrap();
    }
}

/// Each row's truth value for `sql`, which selects the id and one cell
fn per_row(db: &Database, sql: &str) -> Vec<(i64, Option<bool>)> {
    let mut out: Vec<_> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), truth(row.get_value(1)))
        })
        .collect();
    out.sort_unstable_by_key(|(id, _)| *id);
    out
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// `a IN (list)` by the rule: OR over the rows' equalities
fn expected_in(a: Pair, list: &[Pair]) -> Option<bool> {
    list.iter()
        .map(|b| expected("=", a, *b))
        .try_fold(Some(false), |found, eq| match (found, eq) {
            (_, Some(true)) => Err(()),
            (Some(false), Some(false)) => Ok(Some(false)),
            _ => Ok(None),
        })
        .unwrap_or(Some(true))
}

/// A test's truth for a table row's two pairs
type Rule = Box<dyn Fn(Pair, Pair) -> Option<bool>>;

/// The tests a row's cells answer, the same with parameters for the cells,
/// and their truth by the rule
fn cases() -> Vec<(String, String, Rule)> {
    let mut cases: Vec<(String, String, Rule)> = OPERATORS
        .iter()
        .map(|op| {
            (
                format!("(a1, a2) {op} (b1, b2)"),
                format!("($1, $2) {op} ($3, $4)"),
                Box::new(move |a, b| expected(op, a, b)) as Rule,
            )
        })
        .collect();
    cases.push((
        "(a1, a2) IN ((b1, b2), (b2, b1))".to_string(),
        "($1, $2) IN (($3, $4), ($4, $3))".to_string(),
        Box::new(|a, b| expected_in(a, &[b, [b[1], b[0]]])),
    ));
    cases.push((
        "(a1, a2) NOT IN ((b1, b2), (b2, b1))".to_string(),
        "($1, $2) NOT IN (($3, $4), ($4, $3))".to_string(),
        Box::new(|a, b| expected_in(a, &[b, [b[1], b[0]]]).map(|found| !found)),
    ));
    cases.push((
        "(a1, a2) IN ((1, 2), (2, NULL))".to_string(),
        "($1, $2) IN ((1, 2), (2, NULL))".to_string(),
        Box::new(|a, _| expected_in(a, &[[Some(1), Some(2)], [Some(2), None]])),
    ));
    cases
}

fn check_every_context(db: &Database) {
    let rows = combinations();
    let mut wrong = Vec::new();
    for (test, with_parameters, rule) in cases() {
        let want: Vec<(i64, Option<bool>)> =
            rows.iter().map(|(id, a, b)| (*id, rule(*a, *b))).collect();
        let holding = |truth: Option<bool>| -> Vec<i64> {
            want.iter()
                .filter(|(_, t)| *t == truth)
                .map(|(id, _)| *id)
                .collect()
        };
        for sql in [
            format!("SELECT id, {test} FROM r"),
            format!("SELECT id, {test} FROM r LIMIT 100"),
            format!("SELECT id, CASE WHEN {test} THEN 1 WHEN NOT ({test}) THEN 0 END FROM r"),
        ] {
            let got = per_row(db, &sql);
            if got != want {
                wrong.push(format!("{sql}: {got:?}"));
            }
        }
        for (sql, truth) in [
            (format!("SELECT id FROM r WHERE {test}"), Some(true)),
            (format!("SELECT id FROM r WHERE NOT ({test})"), Some(false)),
            (
                format!("SELECT id FROM r GROUP BY id, a1, a2, b1, b2 HAVING {test}"),
                Some(true),
            ),
        ] {
            let got = ids(db, &sql);
            if got != holding(truth) {
                wrong.push(format!("{sql}: {got:?}"));
            }
        }
        let prepared = db.prepare(&format!("SELECT {with_parameters}")).unwrap();
        for (_, a, b) in &rows {
            let params = vec![value(a[0]), value(a[1]), value(b[0]), value(b[1])];
            for _ in 0..2 {
                let mut result = prepared.query(params.clone()).unwrap();
                let row = result.next().unwrap().unwrap();
                let got = truth(row.get_value(0));
                if got != rule(*a, *b) {
                    wrong.push(format!("{with_parameters} with {a:?}, {b:?}: {got:?}"));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_rows_compare_by_the_rule_in_every_context() {
    let db = Database::open("memory://rows_compare_by_the_rule_in_every_context").unwrap();
    load(&db);
    check_every_context(&db);
}

#[test]
fn test_sealed_rows_compare_by_the_rule_in_every_context() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    load(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check_every_context(&db);
}

#[test]
fn test_a_row_compares_its_values_as_two_values_compare() {
    let db = Database::open("memory://a_row_compares_its_values_as_two_values_compare").unwrap();
    for (sql, want) in [
        ("SELECT (1, '1') = (1, 1)", Some(true)),
        ("SELECT (1, 'x') = (1, 1)", None),
        ("SELECT (1, 2.0) = (1, 2)", Some(true)),
        ("SELECT (1, 2, 3) < (1, 2, 4)", Some(true)),
        ("SELECT (1, 2, NULL) < (1, 3, 0)", Some(true)),
        ("SELECT (1, 2, NULL) <= (1, 2, 0)", None),
        ("SELECT (NULL, 2) = (1, 3)", Some(false)),
        ("SELECT ('b', 'a') > ('a', 'z')", Some(true)),
    ] {
        let mut rows = db.query(sql, ()).unwrap();
        let got = truth(rows.next().unwrap().unwrap().get_value(0));
        assert_eq!(got, want, "{sql}");
    }
}

#[test]
fn test_a_row_is_compared_with_a_row_of_its_width_only() {
    let db = Database::open("memory://a_row_is_compared_with_a_row_of_its_width").unwrap();
    for sql in [
        "SELECT (1, 2) = (1, 2, 3)",
        "SELECT (1, 2) = 1",
        "SELECT 1 < (1, 2)",
        "SELECT (1, 2)",
        "SELECT (1, 2) + 1",
        "SELECT (1, 2) IS NULL",
        "SELECT (1, 2) IN ((1, 2, 3))",
        "SELECT (1, 2) IN ((1, 2), 3)",
        "SELECT (1, 2) IN (1, 2)",
    ] {
        let failed = db
            .query(sql, ())
            .map_or(true, |mut rows| rows.next().is_some_and(|row| row.is_err()));
        assert!(failed, "{sql} answered");
    }
}

/// The rows a set test's block gives, over 1, 2 and NULL
const MEMBERS: [Pair; 3] = [[Some(1), Some(2)], [Some(2), None], [None, Some(1)]];

/// `a op ALL (members)` when `all`, else `a op ANY (members)`, by the rule:
/// a member that settles it wins, else UNKNOWN when one's comparison is
fn expected_quantified(op: &str, all: bool, a: Pair, members: &[Pair]) -> Option<bool> {
    let mut unknown = false;
    for member in members {
        match expected(op, a, *member) {
            Some(answer) if answer != all => return Some(answer),
            Some(_) => {}
            None => unknown = true,
        }
    }
    (!unknown).then_some(all)
}

/// The members whose value at `at` equals `value`, as `=` keeps them
fn members_where(at: usize, value: Option<i64>, equal: bool) -> Vec<Pair> {
    MEMBERS
        .iter()
        .filter(|m| match (m[at], value) {
            (Some(x), Some(y)) => (x == y) == equal,
            _ => false,
        })
        .copied()
        .collect()
}

fn check_set_tests(db: &Database) {
    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY, x1 INTEGER, x2 INTEGER)",
        (),
    )
    .unwrap();
    for (id, member) in MEMBERS.iter().enumerate() {
        db.execute(
            "INSERT INTO m VALUES ($1, $2, $3)",
            vec![
                Value::Integer(id as i64),
                value(member[0]),
                value(member[1]),
            ],
        )
        .unwrap();
    }
    let mut cases: Vec<(String, Rule)> = Vec::new();
    for op in ["=", "<>", "<", "<=", ">", ">="] {
        for all in [false, true] {
            let quantifier = if all { "ALL" } else { "ANY" };
            cases.push((
                format!("(a1, a2) {op} {quantifier} (SELECT x1, x2 FROM m)"),
                Box::new(move |a, _| expected_quantified(op, all, a, &MEMBERS)),
            ));
        }
    }
    cases.push((
        "(a1, a2) IN (SELECT x1, x2 FROM m)".to_string(),
        Box::new(|a, _| expected_quantified("=", false, a, &MEMBERS)),
    ));
    cases.push((
        "(a1, a2) NOT IN (SELECT x1, x2 FROM m)".to_string(),
        Box::new(|a, _| expected_quantified("=", false, a, &MEMBERS).map(|found| !found)),
    ));
    cases.push((
        "(a1, a2) IN (SELECT x1, x2 FROM m WHERE m.x1 = r.b1)".to_string(),
        Box::new(|a, b| expected_quantified("=", false, a, &members_where(0, b[0], true))),
    ));
    cases.push((
        "(a1, a2) < ALL (SELECT x1, x2 FROM m WHERE m.x2 <> r.b2)".to_string(),
        Box::new(|a, b| expected_quantified("<", true, a, &members_where(1, b[1], false))),
    ));
    cases.push((
        "(a1, a2) IN (SELECT x1, MAX(x2) FROM m GROUP BY x1)".to_string(),
        Box::new(|a, _| expected_quantified("=", false, a, &MEMBERS)),
    ));
    cases.push((
        "(a1, a2) >= ANY (SELECT x1, x2 FROM m ORDER BY x2 LIMIT 1)".to_string(),
        Box::new(|a, _| expected_quantified(">=", false, a, &[[None, Some(1)]])),
    ));
    cases.push((
        "(a1, a2) IN (SELECT x1, x2 FROM m WHERE id < 0)".to_string(),
        Box::new(|_, _| Some(false)),
    ));
    cases.push((
        "(a1, a2) > ALL (SELECT x1, x2 FROM m WHERE id < 0)".to_string(),
        Box::new(|_, _| Some(true)),
    ));
    let rows = combinations();
    let mut wrong = Vec::new();
    for (test, rule) in cases {
        let want: Vec<(i64, Option<bool>)> =
            rows.iter().map(|(id, a, b)| (*id, rule(*a, *b))).collect();
        let holding = |truth: Option<bool>| -> Vec<i64> {
            want.iter()
                .filter(|(_, t)| *t == truth)
                .map(|(id, _)| *id)
                .collect()
        };
        let select = format!("SELECT id, {test} FROM r");
        let plan: Vec<String> = db
            .query(&format!("EXPLAIN {select}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect();
        if !plan.iter().any(|line| line.trim() == "Prepared: yes") {
            wrong.push(format!("{select} is not prepared: {plan:?}"));
        }
        let got = per_row(db, &select);
        if got != want {
            wrong.push(format!("{select}: {got:?}"));
        }
        for (sql, truth) in [
            (format!("SELECT id FROM r WHERE {test}"), Some(true)),
            (format!("SELECT id FROM r WHERE NOT ({test})"), Some(false)),
        ] {
            let got = ids(db, &sql);
            if got != holding(truth) {
                wrong.push(format!("{sql}: {got:?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_a_row_is_tested_against_a_subquery_by_the_rule() {
    let db = Database::open("memory://a_row_is_tested_against_a_subquery_by_the_rule").unwrap();
    load(&db);
    check_set_tests(&db);
}

#[test]
fn test_a_sealed_row_is_tested_against_a_subquery_by_the_rule() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    load(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check_set_tests(&db);
}
