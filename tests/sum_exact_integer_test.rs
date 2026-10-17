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

//! SUM over INTEGER values is an exact INTEGER on every path, a final sum
//! outside the INTEGER range is an overflow error, and AVG divides the exact sum

use stoolap::core::Value;
use stoolap::Database;

const MAX: i64 = i64::MAX;
const BIG: i64 = 9_007_199_254_740_993; // 2^53 + 1

fn open(db: Database, seal: bool) -> Database {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, i INTEGER, f FLOAT, b BOOLEAN)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, i INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, i INTEGER)",
        "CREATE TABLE o (id INTEGER PRIMARY KEY, g INTEGER, i INTEGER)",
        "CREATE TABLE p (id INTEGER PRIMARY KEY, i INTEGER)",
        "CREATE TABLE a (id INTEGER PRIMARY KEY, g INTEGER, i INTEGER)",
        "INSERT INTO a VALUES (1, 1, 9007199254740992), (2, 1, 9007199254740995)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, g INTEGER, ts TIMESTAMP)",
        "INSERT INTO s VALUES (1, 1, '2024-01-01 00:00:00'), (2, 1, '2024-01-02 00:00:00')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db.execute(
        &format!(
            "INSERT INTO t VALUES (1, 1, {BIG}, 2.5, TRUE), (2, 2, 1, 2.5, FALSE), \
             (3, 2, 2, NULL, TRUE), (4, 3, {MAX}, 1.0, TRUE), (5, 3, 10, 1.0, TRUE), \
             (6, 3, -20, 1.0, FALSE), (7, 4, {BIG}, 0.5, TRUE), \
             (8, 4, -9007199254740992, 0.5, TRUE), (9, 4, 1, 1.0, TRUE)"
        ),
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO u VALUES (1, {MAX}), (2, 10), (3, -20), (4, {BIG}), \
             (5, -9007199254740992), (6, 1)"
        ),
        (),
    )
    .unwrap();
    db.execute(
        &format!("INSERT INTO w VALUES (1, {BIG}), (2, -9007199254740992), (3, 1)"),
        (),
    )
    .unwrap();
    db.execute(
        &format!("INSERT INTO o VALUES (1, 1, {MAX}), (2, 1, 1)"),
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO p SELECT value, CASE WHEN value = 1 THEN {MAX} WHEN value = 3 \
             THEN -{MAX} ELSE 1 END FROM generate_series(1, 120000)"
        ),
        (),
    )
    .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// A value with its type, so INTEGER 3 and FLOAT 3.0 differ
fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        v if v.is_null() => "null".to_string(),
        other => format!("{other:?}"),
    }
}

/// The distinct rows, sorted, or "error" when the query fails with an overflow
fn answer(db: &Database, sql: &str) -> Vec<String> {
    let rows = match db.query(sql, ()) {
        Ok(rows) => rows,
        Err(e) if e.to_string().to_lowercase().contains("overflow") => {
            return vec!["error".to_string()]
        }
        Err(e) => return vec![format!("unexpected error: {e}")],
    };
    let mut out = Vec::new();
    for row in rows {
        match row {
            Ok(row) => out.push(
                (0..row.len())
                    .map(|i| render(row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|"),
            ),
            Err(e) if e.to_string().to_lowercase().contains("overflow") => {
                return vec!["error".to_string()]
            }
            Err(e) => return vec![format!("unexpected error: {e}")],
        }
    }
    out.sort();
    out.dedup();
    out
}

const GROUPED: &[&str] = &[
    "SELECT g, {agg} FROM {t} GROUP BY g",
    "SELECT g, {agg} FROM {t} WHERE id + 0 = id GROUP BY g",
    "SELECT g, {agg} FROM {t} GROUP BY g HAVING COUNT(*) > 0 LIMIT 10",
    "SELECT g, {agg} FROM (SELECT * FROM {t}) d GROUP BY g",
    "SELECT g, {agg} OVER (PARTITION BY g) FROM {t}",
];

const GLOBAL: &[&str] = &[
    "SELECT {agg} FROM {t}",
    "SELECT {agg} FROM {t} WHERE id > 0",
    "SELECT {agg} FROM {t} WHERE id + 0 = id",
    "SELECT {agg} + 0 FROM {t}",
    "SELECT {agg} FROM (SELECT * FROM {t}) d",
];

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let f = |x: f64| format!("f:{x:?}");
    let grouped: Vec<(&str, &str, Vec<String>)> = vec![
        (
            "t",
            "SUM(i)",
            vec![
                format!("i:1|i:{BIG}"),
                "i:2|i:3".into(),
                format!("i:3|i:{}", MAX - 10),
                "i:4|i:2".into(),
            ],
        ),
        (
            "t",
            "SUM(f)",
            vec![
                format!("i:1|{}", f(2.5)),
                format!("i:2|{}", f(2.5)),
                format!("i:3|{}", f(3.0)),
                format!("i:4|{}", f(2.0)),
            ],
        ),
        (
            "t",
            "SUM(b)",
            vec![
                "i:1|i:1".into(),
                "i:2|i:1".into(),
                "i:3|i:2".into(),
                "i:4|i:3".into(),
            ],
        ),
        (
            "t",
            "AVG(i)",
            vec![
                format!("i:1|{}", f(BIG as f64)),
                format!("i:2|{}", f(1.5)),
                format!("i:3|{}", f(((MAX - 10) / 3) as f64)),
                format!("i:4|{}", f(2.0 / 3.0)),
            ],
        ),
        ("o", "SUM(i)", vec!["error".into()]),
        ("s", "SUM(ts)", vec!["i:1|null".into()]),
        ("s", "AVG(ts)", vec!["i:1|null".into()]),
        (
            "a",
            "AVG(i)",
            vec![format!("i:1|{}", f(9_007_199_254_740_994.0))],
        ),
    ];
    let global: Vec<(&str, &str, Vec<String>)> = vec![
        ("a", "AVG(i)", vec![f(9_007_199_254_740_994.0)]),
        ("s", "SUM(ts)", vec!["null".into()]),
        ("s", "AVG(ts)", vec!["null".into()]),
        ("u", "SUM(i)", vec![format!("i:{}", MAX - 8)]),
        ("t", "SUM(f)", vec![f(10.0)]),
        ("w", "SUM(i)", vec!["i:2".into()]),
        ("w", "AVG(i)", vec![f(2.0 / 3.0)]),
        ("p", "SUM(i)", vec!["i:119998".into()]),
        ("p", "AVG(i)", vec![f(119_998.0 / 120_000.0)]),
        ("o", "SUM(i)", vec!["error".into()]),
    ];
    let cases = grouped
        .iter()
        .flat_map(|c| GROUPED.iter().map(move |form| (form, c)))
        .chain(
            global
                .iter()
                .flat_map(|c| GLOBAL.iter().map(move |form| (form, c))),
        );
    for (form, (table, agg, want)) in cases {
        let sql = form.replace("{agg}", agg).replace("{t}", table);
        let got = answer(db, &sql);
        if &got != want {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
        }
    }
    let having = [
        (
            format!("SELECT g FROM t GROUP BY g HAVING SUM(i) = {BIG}"),
            vec!["i:1".to_string()],
        ),
        (
            format!("SELECT g FROM t WHERE id + 0 = id GROUP BY g HAVING SUM(i) = {BIG}"),
            vec!["i:1".to_string()],
        ),
        (
            "SELECT g FROM t GROUP BY g HAVING SUM(i) = 9007199254740992".to_string(),
            vec![],
        ),
        (
            "SELECT g FROM t GROUP BY g HAVING SUM(i) = 9007199254740992 LIMIT 10".to_string(),
            vec![],
        ),
        (
            "SELECT AVG(x) FROM (SELECT 9007199254740990 AS x UNION ALL \
             SELECT -9007199254740989.0 UNION ALL SELECT 0) d"
                .to_string(),
            vec![f(1.0 / 3.0)],
        ),
        (
            "SELECT SUM(i), AVG(i), COUNT(*) FROM p WHERE id + 0 = id".to_string(),
            vec![format!("i:119998|{}|i:120000", f(119_998.0 / 120_000.0))],
        ),
    ];
    for (sql, want) in having {
        let got = answer(db, &sql);
        if got != want {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
        }
    }
}

#[test]
fn sum_is_exact_on_every_path() {
    let mut wrong = Vec::new();
    check(
        &open(Database::open("memory://sum_exact_integer").unwrap(), false),
        "memory",
        &mut wrong,
    );
    for seal in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(
            Database::open(&format!(
                "file://{}?checkpoint_interval=0",
                dir.path().display()
            ))
            .unwrap(),
            seal,
        );
        check(&db, &format!("file sealed {seal}"), &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
