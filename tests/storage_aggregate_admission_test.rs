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

//! The storage aggregates answer an operation over a column type as the
//! general path does, or leave it to the general path

use stoolap::core::Value;
use stoolap::Database;

const OPS: &[&str] = &["COUNT", "SUM", "AVG", "MIN", "MAX"];
const COLUMNS: &[&str] = &["i", "f", "s", "b", "ts"];

fn open(db: Database, seal: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, i INTEGER, f FLOAT, s TEXT, \
         b BOOLEAN, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES \
         (1, 1, 3, 1.5, 'b', TRUE, '2024-01-02 00:00:00'), \
         (2, 1, 1, 0.5, 'a', FALSE, '2024-01-01 00:00:00'), \
         (3, 2, 7, 2.5, 'c', TRUE, '2024-01-03 00:00:00'), \
         (4, 2, NULL, NULL, NULL, NULL, NULL)",
        (),
    )
    .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute(
            "INSERT INTO t VALUES (5, 2, 5, 3.5, 'd', FALSE, '2024-01-04 00:00:00')",
            (),
        )
        .unwrap();
    }
    db
}

fn render(value: &Value) -> String {
    match value {
        Value::Integer(n) => format!("i:{n}"),
        Value::Float(f) => format!("f:{f:?}"),
        Value::Text(s) => format!("t:{s}"),
        v if v.is_null() => "null".to_string(),
        other => format!("{other:?}"),
    }
}

/// The rows sorted, or the error
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| render(row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    out.sort();
    Ok(out)
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    for op in OPS {
        for col in COLUMNS {
            for (sql, general) in [
                (
                    format!("SELECT {op}({col}) FROM t"),
                    format!("SELECT {op}({col}) FROM t WHERE id + 0 = id"),
                ),
                (
                    format!("SELECT {op}({col}) FROM t WHERE id > 0"),
                    format!("SELECT {op}({col}) FROM t WHERE id > 0 AND id + 0 = id"),
                ),
                (
                    format!("SELECT g, {op}({col}) FROM t GROUP BY g"),
                    format!("SELECT g, {op}({col}) FROM t WHERE id + 0 = id GROUP BY g"),
                ),
            ] {
                let (got, want) = (answer(db, &sql), answer(db, &general));
                if got != want {
                    wrong.push(format!("{ctx}: {sql}: got {got:?}, general {want:?}"));
                }
            }
        }
    }
}

#[test]
fn storage_aggregates_answer_every_type_as_the_general_path() {
    let mut wrong = Vec::new();
    check(
        &open(
            Database::open("memory://storage_aggregate_admission").unwrap(),
            false,
        ),
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
