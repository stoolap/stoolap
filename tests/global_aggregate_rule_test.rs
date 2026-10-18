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

//! An aggregate over a whole table, or over the rows a WHERE keeps, answers
//! as it answers over the same rows read through a derived table, whether
//! the rows are hot, sealed, or both

use stoolap::core::Value;
use stoolap::Database;

fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        other => format!("{other}"),
    }
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    let result = db.query(sql, ()).and_then(|rows| {
        let mut out = Vec::new();
        for row in rows {
            let row = row?;
            out.push(
                (0..row.len())
                    .map(|i| render(row.get_value(i).unwrap()))
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
        Ok(out)
    });
    result.unwrap_or_else(|e| vec![format!("error: {e}")])
}

fn create(db: &Database) {
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, b BOOLEAN, \
         t TIMESTAMP, s TEXT)",
        (),
    )
    .unwrap();
}

fn insert(db: &Database, ids: std::ops::RangeInclusive<i64>) {
    for id in ids {
        let i = if id % 6 == 0 {
            "NULL".to_string()
        } else {
            (id * 7 % 23 - 11).to_string()
        };
        let f = match id % 9 {
            0 => "NULL".to_string(),
            4 => "CAST('NaN' AS FLOAT)".to_string(),
            5 => "-0.0".to_string(),
            k => format!("{}", k as f64 * 1.25 - 3.0),
        };
        let b = match id % 4 {
            0 => "NULL",
            1 => "TRUE",
            _ => "FALSE",
        };
        let t = if id % 5 == 0 {
            "NULL".to_string()
        } else {
            format!("'2024-01-{:02} 10:{:02}:00'", id % 28 + 1, id % 60)
        };
        db.execute(
            &format!(
                "INSERT INTO d VALUES ({id}, {i}, {f}, {b}, {t}, 's{}')",
                id % 3
            ),
            (),
        )
        .unwrap();
    }
}

const CALLS: &[&str] = &[
    "COUNT(*)", "COUNT(i)", "SUM(i)", "AVG(i)", "MIN(i)", "MAX(i)", "COUNT(f)", "SUM(f)", "AVG(f)",
    "MIN(f)", "MAX(f)", "SUM(b)", "AVG(b)", "COUNT(b)", "MIN(t)", "MAX(t)", "COUNT(t)", "COUNT(s)",
];

const FILTERS: &[&str] = &[
    "",
    "WHERE id > 7",
    "WHERE i >= 0",
    "WHERE f < 2.0",
    "WHERE s = 's1'",
];

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    for filter in FILTERS {
        let all = CALLS.join(", ");
        let direct = rows(db, &format!("SELECT {all} FROM d {filter}"));
        let derived = rows(
            db,
            &format!("SELECT {all} FROM (SELECT * FROM d {filter}) x"),
        );
        if direct != derived {
            wrong.push(format!(
                "{ctx}: {filter}: all calls: got {direct:?}, derived {derived:?}"
            ));
        }
        for call in CALLS {
            for select in [call.to_string(), format!("COALESCE({call}, NULL)")] {
                let direct = rows(db, &format!("SELECT {select} FROM d {filter}"));
                let derived = rows(
                    db,
                    &format!("SELECT {select} FROM (SELECT * FROM d {filter}) x"),
                );
                if direct != derived {
                    wrong.push(format!(
                        "{ctx}: SELECT {select} {filter}: got {direct:?}, derived {derived:?}"
                    ));
                }
            }
        }
    }
}

#[test]
fn global_aggregates_answer_as_over_the_same_rows() {
    let mut wrong = Vec::new();
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::reset_aggregate_paths();

    let memory = Database::open("memory://global_aggregate_rule").unwrap();
    create(&memory);
    insert(&memory, 1..=60);
    check(&memory, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    create(&db);
    insert(&db, 1..=60);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed", &mut wrong);

    insert(&db, 61..=80);
    db.execute("DELETE FROM d WHERE id % 10 = 3", ()).unwrap();
    db.execute("UPDATE d SET i = i + 100, f = f * 2 WHERE id % 10 = 7", ())
        .unwrap();
    check(&db, "sealed and hot", &mut wrong);

    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    #[cfg(feature = "test-failpoints")]
    {
        let taken = stoolap::test_failpoints::aggregate_paths_taken();
        for path in [
            "pushdown",
            "filtered_pushdown",
            "streaming_global",
            "general",
        ] {
            assert!(taken.contains(&path), "{path} not taken: {taken:?}");
        }
    }
}
