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

//! Values that `=` calls equal are one key wherever rows are grouped or
//! deduplicated: INTEGER 1 and FLOAT 1.0 are one group, 0, 0.0 and -0.0 are
//! one, every NULL is one, and 9007199254740993 and 9007199254740992.0 stay
//! two. A group shows one of its values

use stoolap::core::Value;
use stoolap::Database;

/// An INTEGER for even ids, a FLOAT for odd ones
const M: &str = "CASE WHEN id % 2 = 0 THEN a ELSE f END";

/// A number by its value, whatever its type; other values as written
fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("n:{n}"),
        Value::Float(f) if f.fract() == 0.0 && f.abs() < 1e18 => format!("n:{}", *f as i64),
        Value::Float(f) => format!("n:{f}"),
        v if v.is_null() => "null".to_string(),
        other => format!("{other}"),
    }
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    let result = match db.query(sql, ()) {
        Ok(result) => result,
        Err(e) => return vec![format!("error: {e}")],
    };
    let mut out: Vec<String> = result
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| render(row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    out.sort();
    out
}

fn open(db: Database, seal: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, f FLOAT)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, NULL, 1.0), (2, 1, NULL), (3, NULL, 2.5), (4, NULL, NULL), \
         (5, NULL, 0.0), (6, 0, NULL), (7, NULL, -0.0), (8, 9007199254740993, NULL), \
         (9, NULL, 9007199254740992.0), (10, 2, NULL)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, 0), (2, 1), (3, 2)", ())
        .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn sorted(values: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    out.sort();
    out
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let counts = sorted(&[
        "n:1|n:2",
        "n:2.5|n:1",
        "null|n:1",
        "n:0|n:3",
        "n:9007199254740993|n:1",
        "n:9007199254740992|n:1",
        "n:2|n:1",
    ]);
    let keys = sorted(&[
        "n:1",
        "n:2.5",
        "null",
        "n:0",
        "n:9007199254740993",
        "n:9007199254740992",
        "n:2",
    ]);
    let mut rollup = counts.clone();
    rollup.push("null|n:10".to_string());
    rollup.sort();
    let partitions = sorted(&[
        "n:1|n:2",
        "n:1|n:2",
        "n:2.5|n:1",
        "null|n:1",
        "n:0|n:3",
        "n:0|n:3",
        "n:0|n:3",
        "n:9007199254740993|n:1",
        "n:9007199254740992|n:1",
        "n:2|n:1",
    ]);
    let cases: Vec<(String, Vec<String>)> = vec![
        (
            format!("SELECT {M}, COUNT(*) FROM t GROUP BY {M}"),
            counts.clone(),
        ),
        (
            format!("SELECT {M}, COUNT(*) FROM t GROUP BY {M} LIMIT 100"),
            counts.clone(),
        ),
        (
            format!("SELECT {M}, COUNT(*) FROM t GROUP BY {M} HAVING COUNT(*) > 0"),
            counts.clone(),
        ),
        (
            format!("SELECT k, COUNT(*) FROM (SELECT {M} AS k FROM t) d GROUP BY k"),
            counts.clone(),
        ),
        (
            format!("SELECT {M}, COUNT(*) FROM t GROUP BY ROLLUP({M})"),
            rollup,
        ),
        (format!("SELECT DISTINCT {M} FROM t"), keys.clone()),
        (
            format!("SELECT DISTINCT ON ({M}) {M} FROM t ORDER BY {M}, id"),
            keys.clone(),
        ),
        (
            format!("SELECT {M} FROM t UNION SELECT {M} FROM t"),
            keys.clone(),
        ),
        (
            format!("SELECT {M} FROM t INTERSECT SELECT {M} FROM t"),
            keys.clone(),
        ),
        (
            format!("SELECT {M} FROM t EXCEPT SELECT a FROM t WHERE a IS NULL"),
            keys.iter().filter(|k| *k != "null").cloned().collect(),
        ),
        (
            format!("SELECT COUNT(DISTINCT {M}) FROM t"),
            vec!["n:6".to_string()],
        ),
        (
            format!("SELECT {M}, COUNT(*) OVER (PARTITION BY {M}) FROM t"),
            partitions,
        ),
        (
            format!(
                "SELECT k FROM u WHERE k IN (SELECT {M} FROM t GROUP BY {M} HAVING COUNT(*) >= 2)"
            ),
            sorted(&["n:0", "n:1"]),
        ),
    ];
    for (sql, want) in cases {
        let got = rows(db, &sql);
        if got != want {
            wrong.push(format!("{ctx}: {sql}\n    got  {got:?}\n    want {want:?}"));
        }
    }
}

#[test]
fn values_equal_by_eq_are_one_key_everywhere() {
    let mut wrong = Vec::new();
    check(
        &open(
            Database::open("memory://grouping_identity_rule").unwrap(),
            false,
        ),
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
