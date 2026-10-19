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

//! LIKE, ILIKE, GLOB and REGEXP read both sides as the text CAST AS TEXT
//! gives them, a NULL on either side gives NULL, and every path that
//! answers them, hot or sealed, indexed or not, answers alike.

use stoolap::{Database, Value};

const COLUMNS: [&str; 5] = ["s", "n", "f", "ts", "b"];

/// (operator, patterns), each pattern a SQL literal
const PREDICATES: [(&str, &[&str]); 9] = [
    (
        "LIKE",
        &[
            "'1%'",
            "'%5'",
            "'_'",
            "'a\\%b'",
            "'J%n'",
            "'User_1%'",
            "'%'",
            "'2024%'",
            "'t%'",
            "'%mile'",
            "'a_b'",
        ],
    ),
    ("NOT LIKE", &["'1%'", "'a%'", "'%e'"]),
    (
        "ILIKE",
        &["'JOHN'", "'é%'", "'%MILE'", "'TR%'", "'USER_1%'"],
    ),
    ("NOT ILIKE", &["'john'", "'É%'"]),
    ("GLOB", &["'1*'", "'a*'", "'?'", "'*mile'"]),
    ("NOT GLOB", &["'a*'", "'1*'"]),
    ("REGEXP", &["'^1'", "'5$'", "'^[Jj]'"]),
    ("LIKE", &["DATE '2024-01-01'", "15", "TRUE"]),
    ("NOT REGEXP", &["'^1'"]),
];

/// The sorted ids a query returns, or its error
type Ids = Result<Vec<i64>, String>;

fn ids(db: &Database, sql: &str, params: Vec<Value>) -> Ids {
    let rows = db.query(sql, params).map_err(|e| e.to_string())?;
    let mut ids: Vec<i64> = rows
        .map(|row| {
            row.map_err(|e| e.to_string())
                .and_then(|row| row.get::<i64>(0).map_err(|e| e.to_string()))
        })
        .collect::<Result<_, _>>()?;
    ids.sort_unstable();
    Ok(ids)
}

fn fill(db: &Database, index: bool) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, n INTEGER, f FLOAT, ts TIMESTAMP, b BOOLEAN)",
        (),
    )
    .unwrap();
    if index {
        for column in COLUMNS {
            db.execute(&format!("CREATE INDEX t_{column} ON t ({column})"), ())
                .unwrap();
        }
    }
    db.execute(
        "INSERT INTO t VALUES \
         (1, 'John', 15, 1.5, '2024-01-01 12:00:00', TRUE), \
         (2, 'a%b', 123, 12.25, '2023-06-01 00:00:00', FALSE), \
         (3, 'a\\x', 7, 0.5, '2024-03-01 00:00:00', TRUE), \
         (4, NULL, NULL, NULL, NULL, NULL), \
         (5, 'axb', 1, 100.0, '2024-01-02 00:00:00', FALSE), \
         (6, 'Émile', 25, 2.5, '2024-01-01 00:00:00', TRUE), \
         (7, 'émile', -5, -1.5, '2022-12-31 23:59:59', FALSE), \
         (8, 'User_12', 1000, 1e20, '2024-11-11 11:11:11', TRUE), \
         (9, '2024-01-01T00:00:00+00:00', 15, 15.0, '2024-01-01 00:00:00', TRUE), \
         (10, 'true', 0, 0.0, '2025-01-01 00:00:00', TRUE)",
        (),
    )
    .unwrap();
}

/// Every wrong answer of `db`, against what the select list answers when
/// both sides are already TEXT
fn wrong_answers(db: &Database, context: &str, oracle: &Database, wrong: &mut Vec<String>) {
    for column in COLUMNS {
        for (op, patterns) in PREDICATES {
            for pattern in patterns.iter() {
                let want = ids(
                    oracle,
                    &format!(
                        "SELECT id FROM (SELECT id, CAST({column} AS TEXT) {op} CAST({pattern} AS TEXT) AS m FROM t) x WHERE m = TRUE"
                    ),
                    Vec::new(),
                );
                let unknown = ids(
                    oracle,
                    &format!(
                        "SELECT id FROM (SELECT id, CAST({column} AS TEXT) {op} CAST({pattern} AS TEXT) AS m FROM t) x WHERE m IS NULL"
                    ),
                    Vec::new(),
                );
                let pattern_text = ids(
                    oracle,
                    &format!("SELECT 1 WHERE CAST({pattern} AS TEXT) IS NOT NULL"),
                    Vec::new(),
                );
                // NOT (x op p) holds where x op p is FALSE: neither TRUE nor NULL
                let not_want: Result<Vec<i64>, String> = match (&want, &unknown) {
                    (Ok(want), Ok(unknown)) => Ok((1..=10)
                        .filter(|id| !want.contains(id) && !unknown.contains(id))
                        .collect()),
                    (Err(e), _) | (_, Err(e)) => Err(e.clone()),
                };
                let test = format!("{column} {op} {pattern}");
                let mut shapes: Vec<(String, Vec<Value>, Ids)> = vec![
                    (
                        format!("SELECT id FROM t WHERE {test}"),
                        Vec::new(),
                        want.clone(),
                    ),
                    (
                        format!("SELECT id FROM t WHERE NOT ({test})"),
                        Vec::new(),
                        not_want,
                    ),
                    (
                        format!("SELECT id FROM t WHERE ({test}) IS NULL"),
                        Vec::new(),
                        unknown,
                    ),
                    (
                        format!("SELECT id FROM (SELECT id, {test} AS m FROM t) x WHERE m = TRUE"),
                        Vec::new(),
                        want.clone(),
                    ),
                    (
                        format!("SELECT id FROM t WHERE CASE WHEN {test} THEN TRUE ELSE FALSE END"),
                        Vec::new(),
                        want.clone(),
                    ),
                    (
                        format!("SELECT id FROM t WHERE {test} LIMIT 100"),
                        Vec::new(),
                        want.clone(),
                    ),
                ];
                if pattern_text.as_deref() == Ok(&[1][..]) && pattern.starts_with('\'') {
                    let text = pattern.trim_matches('\'').replace("''", "'");
                    shapes.push((
                        format!("SELECT id FROM t WHERE {column} {op} $1"),
                        vec![Value::text(text)],
                        want.clone(),
                    ));
                }
                for (sql, params, want) in shapes {
                    let got = ids(db, &sql, params.clone());
                    if got != want {
                        wrong.push(format!(
                            "{context}: {sql} {params:?} gave {got:?}, want {want:?}"
                        ));
                    }
                }
            }
        }
    }
}

#[test]
fn test_a_pattern_reads_both_sides_as_text_on_every_path() {
    let oracle = Database::open("memory://pattern_rule_oracle").unwrap();
    fill(&oracle, false);
    let mut wrong = Vec::new();
    for index in [false, true] {
        let hot = Database::open(&format!("memory://pattern_rule_hot_{index}")).unwrap();
        fill(&hot, index);
        let context = if index { "indexed hot" } else { "hot" };
        wrong_answers(&hot, context, &oracle, &mut wrong);

        let dir = tempfile::tempdir().unwrap();
        let sealed = Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap();
        fill(&sealed, index);
        sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let context = if index { "indexed sealed" } else { "sealed" };
        wrong_answers(&sealed, context, &oracle, &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_an_index_range_reads_every_row_a_prefix_pattern_matches() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            Database::open(&format!(
                "file://{}?checkpoint_interval=0",
                dir.path().display()
            ))
            .unwrap()
        } else {
            Database::open("memory://pattern_rule_index_range").unwrap()
        };
        db.execute(
            "CREATE TABLE r (id INTEGER PRIMARY KEY, s TEXT, n INTEGER)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX r_s ON r (s)", ()).unwrap();
        db.execute("CREATE INDEX r_n ON r (n)", ()).unwrap();
        db.execute("BEGIN", ()).unwrap();
        for id in 1..=3000i64 {
            db.execute(
                "INSERT INTO r VALUES ($1, $2, $3)",
                vec![
                    Value::Integer(id),
                    Value::text(format!("zz{id}")),
                    Value::Integer(id + 200_000),
                ],
            )
            .unwrap();
        }
        for (id, s, n) in [
            (5001, "ab%x", 15),
            (5002, "ab\\x", 150),
            (5003, "ab\u{1F600}", 1),
            (5004, "abc", 7),
            (5005, "aé", 25),
        ] {
            db.execute(
                "INSERT INTO r VALUES ($1, $2, $3)",
                vec![Value::Integer(id), Value::text(s), Value::Integer(n)],
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (sql, want) in [
            ("SELECT id FROM r WHERE s LIKE 'ab\\%x'", vec![5001]),
            (
                "SELECT id FROM r WHERE s LIKE 'ab%'",
                vec![5001, 5002, 5003, 5004],
            ),
            ("SELECT id FROM r WHERE s LIKE 'ab\u{1F600}%'", vec![5003]),
            ("SELECT id FROM r WHERE n LIKE '1%'", vec![5001, 5002, 5003]),
            (
                "SELECT id FROM r WHERE s ILIKE 'AB_%'",
                vec![5001, 5002, 5003, 5004],
            ),
            ("SELECT id FROM r WHERE s ILIKE 'aÉ%'", vec![5005]),
        ] {
            let got = ids(&db, sql, Vec::new());
            if got != Ok(want.clone()) {
                wrong.push(format!(
                    "sealed {sealed}: {sql} gave {got:?}, want {want:?}"
                ));
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
fn test_a_wildcard_and_a_trailing_escape_read_alike_on_every_path() {
    let long = "x".repeat(10_000);
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            Database::open(&format!(
                "file://{}?checkpoint_interval=0",
                dir.path().display()
            ))
            .unwrap()
        } else {
            Database::open("memory://pattern_rule_wildcards").unwrap()
        };
        db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, s TEXT)", ())
            .unwrap();
        for (id, s) in [
            (1, "a\nb"),
            (2, "abc\\"),
            (3, "abc"),
            (4, "x"),
            (5, long.as_str()),
            (6, "\n"),
        ] {
            db.execute(
                "INSERT INTO w VALUES ($1, $2)",
                vec![Value::Integer(id), Value::text(s)],
            )
            .unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let backslash = Value::text("abc\\");
        for (sql, params, want) in [
            (
                "SELECT id FROM w WHERE s LIKE '%%'",
                vec![],
                vec![1, 2, 3, 4, 5, 6],
            ),
            (
                "SELECT id FROM w WHERE s LIKE 'a%b%'",
                vec![],
                vec![1, 2, 3],
            ),
            ("SELECT id FROM w WHERE s LIKE 'a_b'", vec![], vec![1]),
            ("SELECT id FROM w WHERE s LIKE '_'", vec![], vec![4, 6]),
            (
                "SELECT id FROM w WHERE s ILIKE 'A%B%'",
                vec![],
                vec![1, 2, 3],
            ),
            (
                "SELECT id FROM w WHERE s GLOB '**'",
                vec![],
                vec![1, 2, 3, 4, 5, 6],
            ),
            ("SELECT id FROM w WHERE s GLOB 'a?b'", vec![], vec![1]),
            (
                "SELECT id FROM w WHERE s GLOB $1",
                vec![backslash.clone()],
                vec![2],
            ),
            (
                "SELECT id FROM (SELECT id, s GLOB $1 AS m FROM w) x WHERE m",
                vec![backslash.clone()],
                vec![2],
            ),
        ] {
            let got = ids(&db, sql, params.clone());
            if got != Ok(want.clone()) {
                wrong.push(format!(
                    "sealed {sealed}: {sql} {params:?} gave {got:?}, want {want:?}"
                ));
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
