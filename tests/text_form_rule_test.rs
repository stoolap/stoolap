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

//! A value read as text is the text CAST AS TEXT gives it, whichever
//! function reads it.

use stoolap::{Database, Value};

/// Each reading of `x` as text, written with `{}` for `x`
const READINGS: [&str; 23] = [
    "{} || ''",
    "CONCAT({})",
    "CONCAT_WS('', {})",
    "UPPER({})",
    "LOWER({})",
    "LENGTH({})",
    "CHAR_LENGTH({})",
    "REVERSE({})",
    "LEFT({}, 3)",
    "RIGHT({}, 3)",
    "SUBSTRING({}, 2)",
    "REPLACE({}, '0', '-')",
    "LPAD({}, 40, '*')",
    "RPAD({}, 40, '*')",
    "TRIM({})",
    "LTRIM({})",
    "RTRIM({})",
    "INSTR({}, 'e')",
    "STRPOS({}, 'e')",
    "SPLIT_PART({}, 'e', 1)",
    "STARTS_WITH({}, '1')",
    "ENDS_WITH({}, '0')",
    "REPEAT({}, 2)",
];

const COLUMNS: [&str; 5] = ["f", "ts", "b", "n", "j"];

fn first_row(db: &Database, sql: &str, params: Vec<Value>) -> Result<Vec<Value>, String> {
    let mut rows = db.query(sql, params).map_err(|e| e.to_string())?;
    let row = rows.next().ok_or("no row")?.map_err(|e| e.to_string())?;
    Ok((0..row.len())
        .map(|i| row.get_value(i).cloned().unwrap())
        .collect())
}

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://text_form_rule_{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, f FLOAT, ts TIMESTAMP, b BOOLEAN, n INTEGER, j JSON)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES \
         (1, 1e20, '2024-01-01 12:00:00.123456789', TRUE, -5, '{\"a\":1}'), \
         (2, 1.5, '2024-06-01 00:00:00', FALSE, 0, '[1,2]'), \
         (3, 1e-7, '1999-12-31 23:59:59.5', TRUE, 9223372036854775807, '\"s\"'), \
         (4, 123456789.125, '2024-01-01', FALSE, 10, 'null')",
        (),
    )
    .unwrap();
    db
}

#[test]
fn test_every_reading_of_a_value_as_text_is_its_cast() {
    let db = open("columns");
    let mut wrong = Vec::new();
    for id in 1..=4 {
        for column in COLUMNS {
            for reading in READINGS {
                let sql = format!(
                    "SELECT {}, {} FROM t WHERE id = {id}",
                    reading.replace("{}", column),
                    reading.replace("{}", &format!("CAST({column} AS TEXT)"))
                );
                match first_row(&db, &sql, Vec::new()) {
                    Ok(pair) if pair[0] == pair[1] => {}
                    got => wrong.push(format!("{sql} gave {got:?}")),
                }
            }
            let sql = format!(
                "SELECT STRING_AGG({column}, ''), MAX(CAST({column} AS TEXT)) FROM t WHERE id = {id}"
            );
            match first_row(&db, &sql, Vec::new()) {
                Ok(pair) if pair[0] == pair[1] => {}
                got => wrong.push(format!("{sql} gave {got:?}")),
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
fn test_a_parameter_read_as_text_is_its_cast() {
    let db = open("parameters");
    let mut wrong = Vec::new();
    for value in [
        Value::Float(1e20),
        Value::Float(1e-7),
        Value::Float(123456789.125),
        Value::Boolean(true),
        Value::Integer(-5),
    ] {
        for reading in READINGS {
            let sql = format!(
                "SELECT {}, {}",
                reading.replace("{}", "$1"),
                reading.replace("{}", "CAST($1 AS TEXT)")
            );
            match first_row(&db, &sql, vec![value.clone()]) {
                Ok(pair) if pair[0] == pair[1] => {}
                got => wrong.push(format!("{sql} with {value:?} gave {got:?}")),
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
