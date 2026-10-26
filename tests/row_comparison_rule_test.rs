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

fn check_every_context(db: &Database) {
    let rows = combinations();
    let mut wrong = Vec::new();
    for op in OPERATORS {
        let test = format!("(a1, a2) {op} (b1, b2)");
        let want: Vec<(i64, Option<bool>)> = rows
            .iter()
            .map(|(id, a, b)| (*id, expected(op, *a, *b)))
            .collect();
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
        let prepared = db
            .prepare(&format!("SELECT ($1, $2) {op} ($3, $4)"))
            .unwrap();
        for (_, a, b) in &rows {
            let params = vec![value(a[0]), value(a[1]), value(b[0]), value(b[1])];
            for _ in 0..2 {
                let mut result = prepared.query(params.clone()).unwrap();
                let row = result.next().unwrap().unwrap();
                let got = truth(row.get_value(0));
                if got != expected(op, *a, *b) {
                    wrong.push(format!("({a:?}) {op} ({b:?}) as parameters: {got:?}"));
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
    ] {
        let failed = db
            .query(sql, ())
            .map_or(true, |mut rows| rows.next().is_some_and(|row| row.is_err()));
        assert!(failed, "{sql} answered");
    }
}
