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

//! An INTEGER compared with a FLOAT is decided exactly, never through a
//! cast that rounds above 2^53 or saturates at 2^63, on hot rows, sealed
//! volumes and in HAVING

use stoolap::core::Value;
use stoolap::Database;

const P53: i64 = 9_007_199_254_740_992;

/// (id, i, f): i64::MAX, 2^53 + 1 and 1 against 2^53, 2^53 + 2, 2^53 + 4
fn rows() -> [(i64, i64, f64); 3] {
    [
        (1, i64::MAX, P53 as f64),
        (2, P53 + 1, (P53 + 2) as f64),
        (3, 1, (P53 + 4) as f64),
    ]
}

fn open(dir: &std::path::Path, seal: bool) -> Database {
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", dir.display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, g INTEGER)",
        (),
    )
    .unwrap();
    for (id, i, f) in rows() {
        db.execute(
            "INSERT INTO t VALUES ($1, $2, $3, 1)",
            vec![Value::Integer(id), Value::Integer(i), Value::Float(f)],
        )
        .unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn one(db: &Database, sql: &str) -> Result<i64, String> {
    let mut rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    match rows.next() {
        Some(row) => row
            .map_err(|e| e.to_string())?
            .get::<i64>(0)
            .map_err(|e| e.to_string()),
        None => Ok(0),
    }
}

/// The ids meeting a predicate by exact arithmetic on the rows
fn want(pred: &str) -> i64 {
    let p53 = P53 as i128;
    let max = i64::MAX as i128;
    rows()
        .into_iter()
        .filter(|&(_, i, f)| {
            let (i, f) = (i as i128, f as i128);
            match pred {
                "i = 9223372036854775808.0" => i == max + 1,
                "i < 9223372036854775808.0" => i < max + 1,
                "i >= 9223372036854775808.0" => i > max,
                "i <= 9007199254740992.5" => i <= p53,
                "f >= 9007199254740993" => f > p53,
                "f > 9007199254740993" => f > p53 + 1,
                "f < 9007199254740993" => f < p53 + 1,
                "f = 9007199254740993" => f == p53 + 1,
                "f BETWEEN 9007199254740993 AND 9007199254740997" => {
                    (p53 + 1..=p53 + 5).contains(&f)
                }
                "f NOT BETWEEN 9007199254740993 AND 9007199254740997" => {
                    !(p53 + 1..=p53 + 5).contains(&f)
                }
                other => panic!("no expectation for {other}"),
            }
        })
        .count() as i64
}

const PREDICATES: &[&str] = &[
    "i = 9223372036854775808.0",
    "i < 9223372036854775808.0",
    "i >= 9223372036854775808.0",
    "i <= 9007199254740992.5",
    "f >= 9007199254740993",
    "f > 9007199254740993",
    "f < 9007199254740993",
    "f = 9007199254740993",
    "f BETWEEN 9007199254740993 AND 9007199254740997",
    "f NOT BETWEEN 9007199254740993 AND 9007199254740997",
];

#[test]
fn integer_and_float_compare_exactly_hot_and_sealed() {
    let mut wrong = Vec::new();
    for seal in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path(), seal);
        for pred in PREDICATES {
            let want = want(pred);
            for sql in [
                format!("SELECT COUNT(*) FROM t WHERE {pred}"),
                format!("SELECT COUNT(*) FROM (SELECT id FROM t WHERE {pred}) s"),
                format!("SELECT COALESCE(SUM(g), 0) FROM t WHERE {pred}"),
            ] {
                let got = one(&db, &sql);
                if got != Ok(want) {
                    wrong.push(format!("sealed {seal}: {sql}: got {got:?}, want {want}"));
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
fn having_compares_a_large_aggregate_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), false);
    let mut wrong = Vec::new();
    for (sql, want) in [
        ("SELECT COUNT(*) FROM (SELECT g FROM t WHERE id = 2 GROUP BY g HAVING MAX(i) = 9007199254740992) s", 0),
        ("SELECT COUNT(*) FROM (SELECT g FROM t WHERE id = 2 GROUP BY g HAVING MAX(i) = 9007199254740993) s", 1),
        ("SELECT COUNT(*) FROM (SELECT g FROM t WHERE id = 2 GROUP BY g HAVING MIN(i) > 9007199254740992) s", 1),
        ("SELECT COUNT(*) FROM (SELECT g FROM t WHERE id = 1 GROUP BY g HAVING MAX(i) < 9223372036854775807) s", 0),
        ("SELECT COUNT(*) FROM (SELECT g FROM t WHERE id = 1 GROUP BY g HAVING MAX(i) = 9223372036854775807) s", 1),
    ] {
        let got = one(&db, sql);
        if got != Ok(want) {
            wrong.push(format!("{sql}: got {got:?}, want {want}"));
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
fn a_column_default_compares_exactly_on_older_volumes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), true);
    db.execute(
        "ALTER TABLE t ADD COLUMN d FLOAT DEFAULT 9007199254740992.0",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for (pred, want) in [
        ("d >= 9007199254740993", 0),
        ("d < 9007199254740993", 3),
        ("d = 9007199254740992", 3),
    ] {
        for sql in [
            format!("SELECT COUNT(*) FROM t WHERE {pred}"),
            format!("SELECT COUNT(*) FROM (SELECT id FROM t WHERE {pred}) s"),
        ] {
            let got = one(&db, &sql);
            if got != Ok(want) {
                wrong.push(format!("{sql}: got {got:?}, want {want}"));
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

fn groups(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    let mut out = Vec::new();
    for row in db.query(sql, ()).map_err(|e| e.to_string())? {
        out.push(
            row.map_err(|e| e.to_string())?
                .get::<i64>(0)
                .map_err(|e| e.to_string())?,
        );
    }
    out.sort_unstable();
    Ok(out)
}

#[test]
fn having_on_an_aggregate_answers_as_on_its_alias_for_every_type() {
    let db = Database::open("memory://having_alias_types").unwrap();
    db.execute(
        "CREATE TABLE h (id INTEGER PRIMARY KEY, g INTEGER, b BOOLEAN, ts TIMESTAMP, \
         s TEXT, i INTEGER, f FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO h VALUES \
         (1, 1, FALSE, '2025-01-01 00:00:00', '0', 9007199254740993, 0.5), \
         (2, 2, TRUE, '2025-06-01 00:00:00', 'abc', 1, 1.5), \
         (3, 2, FALSE, '2025-03-01 00:00:00', '2', -1, -2.0)",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for col in ["b", "ts", "s", "i", "f"] {
        for agg in ["MIN", "MAX"] {
            for op in ["=", "<>", "<", "<=", ">", ">="] {
                for c in ["0", "1", "1.5", "9007199254740992"] {
                    let inline = format!("SELECT g FROM h GROUP BY g HAVING {agg}({col}) {op} {c}");
                    let alias =
                        format!("SELECT g, {agg}({col}) AS m FROM h GROUP BY g HAVING m {op} {c}");
                    let (a, b) = (groups(&db, &inline), groups(&db, &alias));
                    if a != b {
                        wrong.push(format!("{inline}: {a:?}, as alias: {b:?}"));
                    }
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
    assert_eq!(
        groups(&db, "SELECT g FROM h GROUP BY g HAVING MAX(b) > 0"),
        Ok(vec![]),
        "a BOOLEAN is not above a number"
    );
}
