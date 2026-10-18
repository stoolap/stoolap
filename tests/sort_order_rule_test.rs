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

//! Values sort in one total order: numbers by exact value with NaN after
//! them, then BOOLEAN < numbers < TEXT < TIMESTAMP across types; NULL is
//! placed by NULLS FIRST/LAST and skipped by MIN and MAX

use stoolap::core::Value;
use stoolap::Database;

/// The values in their sorted order, as SQL literals
const SORTED: &[&str] = &[
    "TRUE",
    "2",
    "2.5",
    "10",
    "CAST('NaN' AS FLOAT)",
    "'10'",
    "'a'",
    "CAST('2024-01-01 00:00:00' AS TIMESTAMP)",
];

/// The same values rendered, in sorted order
const RENDERED: &[&str] = &[
    "b:true",
    "i:2",
    "f:2.5",
    "i:10",
    "f:NaN",
    "t:10",
    "t:a",
    "ts:2024-01-01 00:00:00",
];

/// Orders in which the values are assigned to the rows
const PERMUTATIONS: &[&[usize]] = &[
    &[0, 1, 2, 3, 4, 5, 6, 7],
    &[7, 6, 5, 4, 3, 2, 1, 0],
    &[5, 2, 7, 0, 3, 6, 1, 4],
];

fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) if f.is_nan() => "f:NaN".to_string(),
        Value::Float(f) => format!("f:{f}"),
        Value::Text(s) => format!("t:{s}"),
        Value::Boolean(b) => format!("b:{b}"),
        Value::Timestamp(t) => format!("ts:{}", t.format("%Y-%m-%d %H:%M:%S")),
        v if v.is_null() => "null".to_string(),
        other => format!("{other:?}"),
    }
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    rows_with(db, sql, ())
}

fn rows_with<P: stoolap::Params>(db: &Database, sql: &str, params: P) -> Vec<String> {
    let result = match db.query(sql, params) {
        Ok(result) => result,
        Err(e) => return vec![format!("error: {e}")],
    };
    let mut out = Vec::new();
    for row in result {
        match row {
            Ok(row) => out.push(
                (0..row.len())
                    .map(|i| render(row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
            Err(e) => out.push(format!("error: {e}")),
        }
    }
    out
}

fn open(db: Database, seal: bool) -> Database {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    for id in 1..=9 {
        db.execute(&format!("INSERT INTO t VALUES ({id}, {})", id % 2), ())
            .unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// A derived table `s(id, g, v)` giving row `id` (1..=8) the value at
/// `perm[id - 1]` and row 9 NULL
fn source(perm: &[usize]) -> String {
    let arms: String = perm
        .iter()
        .enumerate()
        .map(|(i, &v)| format!(" WHEN id = {} THEN {}", i + 1, SORTED[v]))
        .collect();
    format!("(SELECT id, g, CASE{arms} ELSE NULL END AS v FROM t) s")
}

fn sorted(with_null_last: bool) -> Vec<String> {
    let mut want: Vec<String> = RENDERED.iter().map(|s| s.to_string()).collect();
    if with_null_last {
        want.push("null".to_string());
    }
    want
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let asc = sorted(true);
    let mut desc = sorted(false);
    desc.reverse();
    desc.insert(0, "null".to_string());
    let min_max = format!("{}|{}", RENDERED[0], RENDERED[RENDERED.len() - 1]);
    for perm in PERMUTATIONS {
        let s = source(perm);
        let cases: Vec<(String, Vec<String>)> = vec![
            (format!("SELECT v FROM {s} ORDER BY v"), asc.clone()),
            (format!("SELECT v FROM {s} ORDER BY v DESC"), desc.clone()),
            (
                format!("SELECT v FROM {s} ORDER BY v LIMIT 3"),
                asc[..3].to_vec(),
            ),
            (
                format!("SELECT v FROM {s} ORDER BY v DESC LIMIT 3"),
                desc[..3].to_vec(),
            ),
            (
                format!("SELECT DISTINCT v FROM {s} ORDER BY v"),
                asc.clone(),
            ),
            (
                format!("SELECT MIN(v), MAX(v) FROM {s}"),
                vec![min_max.clone()],
            ),
            (
                format!("SELECT MIN(v), MAX(v) FROM {s} WHERE id > 0"),
                vec![min_max.clone()],
            ),
            (
                format!("SELECT MIN(v), MAX(v) FROM {s} GROUP BY 1 = 1"),
                vec![min_max.clone()],
            ),
            (
                format!("SELECT MIN(v), MAX(v) FROM {s} GROUP BY 1 = 1 HAVING COUNT(*) > 0"),
                vec![min_max.clone()],
            ),
            (
                format!("SELECT DISTINCT MIN(v) OVER (), MAX(v) OVER () FROM {s}"),
                vec![min_max.clone()],
            ),
            (
                format!("SELECT v FROM {s} ORDER BY ROW_NUMBER() OVER (ORDER BY v)"),
                asc.clone(),
            ),
        ];
        let ids_in_order: Vec<String> = (0..SORTED.len())
            .map(|v| (perm.iter().position(|&p| p == v).unwrap() + 1).to_string())
            .chain(std::iter::once("9".to_string()))
            .collect();
        let args = perm
            .iter()
            .map(|&v| SORTED[v])
            .collect::<Vec<_>>()
            .join(", ");
        let mut checks: Vec<(String, Vec<String>, Vec<String>)> = cases
            .into_iter()
            .map(|(sql, want)| {
                let got = rows(db, &sql);
                (sql, got, want)
            })
            .collect();
        let sql = format!("SELECT MIN(v), MAX(v) FROM {s} WHERE id > $1");
        checks.push((
            sql.clone(),
            rows_with(db, &sql, (0i64,)),
            vec![min_max.clone()],
        ));
        let sql = format!("SELECT STRING_AGG(CAST(id AS TEXT), ',' ORDER BY v) FROM {s}");
        checks.push((
            sql.clone(),
            rows(db, &sql),
            vec![format!("t:{}", ids_in_order.join(","))],
        ));
        let sql = format!("SELECT GREATEST({args}), LEAST({args})");
        checks.push((
            sql.clone(),
            rows(db, &sql),
            vec![format!("{}|{}", RENDERED[RENDERED.len() - 1], RENDERED[0])],
        ));
        for (sql, got, want) in checks {
            if got != want {
                wrong.push(format!(
                    "{ctx} {perm:?}: {sql}\n    got  {got:?}\n    want {want:?}"
                ));
            }
        }
    }
}

#[test]
fn values_sort_in_one_total_order_everywhere() {
    let mut wrong = Vec::new();
    check(
        &open(Database::open("memory://sort_order_rule").unwrap(), false),
        "memory",
        &mut wrong,
    );
    let dir = tempfile::tempdir().unwrap();
    let db = open(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap(),
        true,
    );
    check(&db, "sealed", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
