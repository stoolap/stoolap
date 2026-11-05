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

//! A statement the runner prepares fails with the error preparing it gives

use stoolap::Database;

/// Each statement over t and a text its error holds
const ERRORS: &[(&str, &str)] = &[
    ("SELECT id FROM t WHERE val REGEXP '['", "Invalid regex"),
    (
        "SELECT id FROM t WHERE val REGEXP '[' || ''",
        "Invalid regex",
    ),
    (
        "SELECT id FROM t WHERE val NOT REGEXP LOWER('[')",
        "Invalid regex",
    ),
    (
        "SELECT id FROM t WHERE SUM(id) > 1",
        "aggregate functions are not allowed in WHERE clause",
    ),
    (
        "SELECT id FROM t ORDER BY MAX(id)",
        "aggregate function MAX is not allowed where the rows are not grouped",
    ),
];

#[test]
fn a_statement_fails_as_preparing_it_fails() {
    let db = Database::open("memory://prepare_error").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a')", ()).unwrap();
    let mut wrong = Vec::new();
    for (sql, want) in ERRORS {
        for run in 0..2 {
            let got = match db.query(sql, ()) {
                Ok(rows) => rows
                    .map(|row| row.map(|_| ()))
                    .collect::<Result<Vec<_>, _>>()
                    .err()
                    .map(|e| e.to_string()),
                Err(e) => Some(e.to_string()),
            };
            if !got.as_deref().is_some_and(|got| got.contains(want)) {
                wrong.push(format!("{sql}, run {run}: {got:?}, want {want:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
