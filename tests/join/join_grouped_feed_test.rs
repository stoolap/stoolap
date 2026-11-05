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

//! A grouped join whose clauses hold no subquery feeds its calls through
//! programs its plan holds: it answers as its rows, summed here, say

use std::collections::BTreeMap;

use stoolap::{Database, Value};

/// The column names and each row, or the error
type Answer = Result<(Vec<String>, Vec<Vec<Option<i64>>>), String>;

fn answer(db: &Database, sql: &str, params: &[Value]) -> Answer {
    let result = db.query(sql, params.to_vec()).map_err(|e| e.to_string())?;
    let columns = result.columns().to_vec();
    let mut rows = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        // Every value is an INTEGER or NULL: another type is a wrong answer
        let values = (0..row.len()).map(|i| match row.get_value(i).unwrap() {
            Value::Integer(n) => Ok(Some(*n)),
            value if value.is_null() => Ok(None),
            value => Err(format!("{value:?} where an INTEGER or NULL is written")),
        });
        rows.push(values.collect::<Result<_, _>>()?);
    }
    Ok((columns, rows))
}

/// l: id 1..40, k = id % 7, g = id % 4
fn l_rows() -> impl Iterator<Item = (i64, i64, i64)> {
    (1..=40).map(|id| (id, id % 7, id % 4))
}

/// r: id 1..30, k = id % 5, w = id * 3
fn r_rows() -> impl Iterator<Item = (i64, i64, i64)> {
    (1..=30).map(|id| (id, id % 5, id * 3))
}

/// The rows of r joined to an l row with key `k`
fn joined(k: i64) -> Vec<(i64, i64, i64)> {
    r_rows().filter(|r| r.1 == k).collect()
}

fn open(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (id, k, g) in l_rows() {
        db.execute("INSERT INTO l VALUES ($1, $2, $3)", (id, k, g))
            .unwrap();
    }
    for (id, k, w) in r_rows() {
        db.execute("INSERT INTO r VALUES ($1, $2, $3)", (id, k, w))
            .unwrap();
    }
    db
}

/// For each l.g, `value` folded over the inner join's pairs with l.id
/// past `after`, in g order
fn per_g(
    after: i64,
    value: impl Fn((i64, i64, i64), (i64, i64, i64)) -> i64,
    fold: impl Fn(i64, i64) -> i64,
) -> Vec<(i64, i64)> {
    let mut groups: BTreeMap<i64, i64> = BTreeMap::new();
    for l in l_rows().filter(|l| l.0 > after) {
        for r in joined(l.1) {
            let v = value(l, r);
            groups
                .entry(l.2)
                .and_modify(|acc| *acc = fold(*acc, v))
                .or_insert(v);
        }
    }
    groups.into_iter().collect()
}

fn sum(a: i64, b: i64) -> i64 {
    a + b
}

/// Each statement, its columns, the parameter values of two runs and the
/// rows it answers for a run's first value
type Shape = (
    &'static str,
    &'static [&'static str],
    &'static [i64],
    &'static [i64],
    fn(i64) -> Vec<Vec<Option<i64>>>,
);

const SHAPES: &[Shape] = &[
    (
        "SELECT l.id, SUM(r.w + 1) FROM l LEFT JOIN r ON l.k = r.k GROUP BY l.id ORDER BY l.id",
        &["id", "SUM(r.w + 1)"],
        &[],
        &[],
        |_| {
            l_rows()
                .map(|l| {
                    let m = joined(l.1);
                    let total = (!m.is_empty()).then(|| m.iter().map(|r| r.2 + 1).sum());
                    vec![Some(l.0), total]
                })
                .collect()
        },
    ),
    (
        "SELECT 10 - l.id AS id, COUNT(*) FROM l JOIN r ON r.k = l.k GROUP BY 10 - l.id \
         ORDER BY 1",
        &["id", "COUNT(*)"],
        &[],
        &[],
        |_| {
            let mut rows: Vec<Vec<Option<i64>>> = l_rows()
                .filter_map(|l| {
                    let n = joined(l.1).len() as i64;
                    (n > 0).then(|| vec![Some(10 - l.0), Some(n)])
                })
                .collect();
            rows.sort();
            rows
        },
    ),
    (
        "SELECT COUNT(*), SUM(CASE WHEN r.id IS NULL THEN 1 ELSE 0 END) FROM l LEFT JOIN r \
         ON l.k = r.k",
        &[
            "COUNT(*)",
            "SUM(CASE WHEN (r.id IS NULL) THEN 1 ELSE 0 END)",
        ],
        &[],
        &[],
        |_| {
            let counts = l_rows().map(|l| joined(l.1).len() as i64);
            let (rows, unmatched) = counts.fold((0, 0), |(rows, unmatched), n| match n {
                0 => (rows + 1, unmatched + 1),
                n => (rows + n, unmatched),
            });
            vec![vec![Some(rows), Some(unmatched)]]
        },
    ),
    (
        "SELECT l.g, SUM(r.w * l.k) FROM l JOIN r ON r.k = l.k GROUP BY l.g \
         HAVING SUM(r.w) > 10 ORDER BY l.g",
        &["g", "SUM(r.w * l.k)"],
        &[],
        &[],
        |_| {
            let weights: BTreeMap<i64, i64> = per_g(0, |_, r| r.2, sum).into_iter().collect();
            per_g(0, |l, r| r.2 * l.1, sum)
                .into_iter()
                .filter(|(g, _)| weights[g] > 10)
                .map(|(g, v)| vec![Some(g), Some(v)])
                .collect()
        },
    ),
    (
        "SELECT l.g, SUM(r.w + $1) FROM l JOIN r ON r.k = l.k GROUP BY l.g ORDER BY l.g",
        &["g", "SUM(r.w + $1)"],
        &[1],
        &[100],
        |p| {
            per_g(0, |_, r| r.2 + p, sum)
                .into_iter()
                .map(|(g, v)| vec![Some(g), Some(v)])
                .collect()
        },
    ),
    (
        "SELECT l.g, MAX(r.w - l.id) FROM l JOIN r ON r.k = l.k WHERE l.id > $1 GROUP BY l.g \
         ORDER BY l.g",
        &["g", "MAX(r.w - l.id)"],
        &[5],
        &[30],
        |p| {
            per_g(p, |l, r| r.2 - l.0, i64::max)
                .into_iter()
                .map(|(g, v)| vec![Some(g), Some(v)])
                .collect()
        },
    ),
    (
        "SELECT l.g, SUM(r.w + 1) + ROW_NUMBER() OVER (ORDER BY l.g) FROM l JOIN r \
         ON r.k = l.k GROUP BY l.g ORDER BY l.g",
        &["g", "SUM(r.w + 1) + ROW_NUMBER() OVER (ORDER BY l.g)"],
        &[],
        &[],
        |_| {
            (1..)
                .zip(per_g(0, |_, r| r.2 + 1, sum))
                .map(|(n, (g, v))| vec![Some(g), Some(v + n)])
                .collect()
        },
    ),
    (
        "SELECT d.g, SUM(d.w + 1) FROM (SELECT l.g, r.w FROM l JOIN r ON r.k = l.k) AS d \
         GROUP BY d.g ORDER BY d.g",
        &["g", "SUM(d.w + 1)"],
        &[],
        &[],
        |_| {
            per_g(0, |_, r| r.2 + 1, sum)
                .into_iter()
                .map(|(g, v)| vec![Some(g), Some(v)])
                .collect()
        },
    ),
];

fn values(params: &[i64]) -> Vec<Value> {
    params.iter().map(|&p| Value::Integer(p)).collect()
}

#[test]
fn a_grouped_join_answers_as_its_rows_say_run_after_run() {
    let db = open("memory://join_grouped_feed_answers");
    let mut wrong = Vec::new();
    for (sql, columns, first, second, rows) in SHAPES {
        for params in [first, second, first] {
            let got = answer(&db, sql, &values(params));
            let want: Answer = Ok((
                columns.iter().map(|c| c.to_string()).collect(),
                rows(params.first().copied().unwrap_or(0)),
            ));
            if got != want {
                wrong.push(format!("{sql} {params:?}\n  got  {got:?}\n  want {want:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
