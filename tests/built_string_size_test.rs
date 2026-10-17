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

use stoolap::Database;

/// The length of the one text `sql` returns, or its error
fn built(db: &Database, sql: &str) -> Result<usize, String> {
    let mut rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let row = rows.next().expect("one row").map_err(|e| e.to_string())?;
    Ok(row.get::<String>(0).map_err(|e| e.to_string())?.len())
}

#[test]
fn test_a_string_built_past_the_cap_is_an_error() {
    let db = Database::open("memory://built_string_cap").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 9223372036854775807)", ())
        .unwrap();
    for sql in [
        "SELECT LPAD('a', 9223372036854775807)",
        "SELECT RPAD('a', 9223372036854775807)",
        "SELECT LPAD('a', n) FROM t",
        "SELECT RPAD('a', n, 'xy') FROM t",
        "SELECT LPAD('a', 10485761)",
        "SELECT RPAD('a', 6000000, 'é')",
        "SELECT REPEAT('ab', 5242881)",
    ] {
        let got = built(&db, sql);
        assert!(
            got.as_ref().is_err_and(|e| e.contains("exceeding maximum")),
            "{sql}: {got:?}"
        );
    }
}

#[test]
fn test_a_string_built_within_the_cap_keeps_its_answer() {
    let db = Database::open("memory://built_string_within").unwrap();
    for (sql, len) in [
        ("SELECT LPAD('a', 10485760)", 10_485_760),
        ("SELECT RPAD('a', 5, 'xy')", 5),
        ("SELECT LPAD('a', 3, 'é')", 5),
        ("SELECT RPAD('a', 5000000, 'é')", 9_999_999),
        ("SELECT LPAD('abc', 2)", 2),
        ("SELECT RPAD('a', 1000, '')", 1),
        ("SELECT REPEAT('ab', 5242880)", 10_485_760),
        ("SELECT REPLACE(REPEAT('a', 1000), 'a', 'bb')", 2000),
        (
            "SELECT REPLACE(REPEAT('ab', 1000000), 'ab', 'c')",
            1_000_000,
        ),
    ] {
        assert_eq!(built(&db, sql), Ok(len), "{sql}");
    }
}

#[test]
fn test_a_replace_that_grows_past_the_cap_is_an_error() {
    let db = Database::open("memory://built_string_replace").unwrap();
    let sql = "SELECT REPLACE(REPEAT('a', 1000000), 'a', 'bbbbbbbbbbbb')";
    let got = built(&db, sql);
    assert!(
        got.as_ref().is_err_and(|e| e.contains("exceeding maximum")),
        "{sql}: {got:?}"
    );
}
