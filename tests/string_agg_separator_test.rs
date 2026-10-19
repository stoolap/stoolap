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

//! A STRING_AGG or GROUP_CONCAT separator is one value for the statement,
//! read as CONCAT_WS reads its separator: a NULL one gives NULL, and one
//! that reads the row is an error.

use stoolap::{Database, Value};

const SEPARATORS: [&str; 9] = [
    "' | '",
    "''",
    "1",
    "1.5",
    "TRUE",
    "NULL",
    "TIMESTAMP '2024-01-01 12:00:00'",
    "'a' || 'b'",
    "$1",
];

/// Every statement shape a separator is read in, with `sep` in it
const SHAPES: [&str; 8] = [
    "SELECT STRING_AGG(b, {sep}) FROM t",
    "SELECT STRING_AGG(b, {sep} ORDER BY id) FROM t",
    "SELECT GROUP_CONCAT(b, {sep} ORDER BY id) FROM t",
    "SELECT g, STRING_AGG(b, {sep} ORDER BY id) FROM t GROUP BY g",
    "SELECT g, STRING_AGG(b, {sep} ORDER BY id) FROM t GROUP BY g HAVING COUNT(*) > 1",
    "SELECT a FROM (SELECT STRING_AGG(b, {sep} ORDER BY id) AS a FROM t) d",
    "SELECT STRING_AGG(b, {sep} ORDER BY id) OVER (PARTITION BY g) FROM t ORDER BY id LIMIT 1",
    "SELECT STRING_AGG(b, {sep}) FROM t WHERE b <> 'z'",
];

fn last_of_first_row(rows: stoolap::Result<stoolap::Rows>) -> Result<Value, String> {
    let mut rows = rows.map_err(|e| e.to_string())?;
    let row = rows.next().ok_or("no row")?.map_err(|e| e.to_string())?;
    Ok(row.get_value(row.len() - 1).cloned().unwrap())
}

fn create(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, b TEXT, sep TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 'x', '-'), (2, 1, 'y', '-')",
        (),
    )
    .unwrap();
}

fn check(db: &Database, storage: &str, wrong: &mut Vec<String>) {
    for sep in SEPARATORS {
        let want =
            last_of_first_row(db.query(&format!("SELECT CONCAT_WS({sep}, 'x', 'y')"), ("+",)));
        for shape in SHAPES {
            let sql = shape.replace("{sep}", sep);
            let got = last_of_first_row(db.query(&sql, ("+",)));
            if got != want {
                wrong.push(format!("{storage}: {sql} gave {got:?}, want {want:?}"));
            }
        }
    }
    for shape in SHAPES {
        let sql = shape.replace("{sep}", "sep");
        if let Ok(got) = last_of_first_row(db.query(&sql, ())) {
            wrong.push(format!("{storage}: {sql} gave {got:?}, want an error"));
        }
    }
    for shape in SHAPES {
        let sql = shape.replace("{sep}", "$1");
        let prepared = db.prepare(&sql).unwrap();
        for sep in ["+", "-"] {
            let got = last_of_first_row(prepared.query((sep,)));
            let want = Ok(Value::text(format!("x{sep}y")));
            if got != want {
                wrong.push(format!(
                    "{storage}: prepared {sql} with {sep} gave {got:?}, want {want:?}"
                ));
            }
        }
    }
}

#[test]
fn test_a_separator_is_read_as_concat_ws_reads_it() {
    let mut wrong = Vec::new();
    let memory = Database::open("memory://string_agg_separator").unwrap();
    create(&memory);
    check(&memory, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    create(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed", &mut wrong);

    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_a_separator_names_its_column_as_written() {
    let db = Database::open("memory://string_agg_separator_names").unwrap();
    create(&db);
    let mut wrong = Vec::new();
    for sep in [
        "' | '",
        "'it''s'",
        "TIMESTAMP '2024-01-01 12:00:00'",
        "'a' || 'b'",
        "$1",
    ] {
        let sql = format!("SELECT STRING_AGG(b, {sep}) FROM t");
        let rows = db.query(&sql, ("+",)).unwrap();
        let want = format!("STRING_AGG(b, {sep})");
        if rows.columns() != [want.as_str()] {
            wrong.push(format!("{sql} named {:?}, want {want}", rows.columns()));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
