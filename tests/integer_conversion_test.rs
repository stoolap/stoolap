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

//! A conversion into INTEGER truncates a FLOAT toward zero and is an error
//! for a number outside the INTEGER range, wherever the conversion happens.

use std::io::Write;

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://integer_conversion_{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, s TEXT)",
        (),
    )
    .unwrap();
    db
}

fn int(db: &Database, sql: &str) -> Option<i64> {
    db.query_one::<Option<i64>, _>(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn fails(db: &Database, sql: &str) {
    let result = if sql.starts_with("SELECT") {
        db.query(sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .map(|_| ())
    } else {
        db.execute(sql, ()).map(|_| ())
    };
    assert!(result.is_err(), "{sql} should fail");
}

#[test]
fn test_cast_outside_the_range_is_an_error() {
    let db = open("cast");
    assert_eq!(int(&db, "SELECT CAST(3.7 AS INTEGER)"), Some(3));
    assert_eq!(int(&db, "SELECT CAST(-3.7 AS INTEGER)"), Some(-3));
    assert_eq!(
        int(&db, "SELECT CAST(-9223372036854775808.0 AS INTEGER)"),
        Some(i64::MIN)
    );
    assert_eq!(int(&db, "SELECT CAST(' 42 ' AS INTEGER)"), Some(42));
    assert_eq!(int(&db, "SELECT CAST('3.7' AS INTEGER)"), None);
    for sql in [
        "SELECT CAST(9223372036854775807.0 AS INTEGER)",
        "SELECT CAST(1e30 AS INTEGER)",
        "SELECT CAST(-1e30 AS INTEGER)",
        "SELECT CAST('9223372036854775808' AS INTEGER)",
        "SELECT CAST(CAST('inf' AS FLOAT) AS INTEGER)",
    ] {
        fails(&db, sql);
    }
}

#[test]
fn test_cast_of_a_stored_value_outside_the_range_is_an_error() {
    let db = open("stored");
    db.execute(
        "INSERT INTO t VALUES (1, 1, 1e30, '99999999999999999999')",
        (),
    )
    .unwrap();
    fails(&db, "SELECT CAST(f AS INTEGER) FROM t");
    fails(&db, "SELECT CAST(s AS INTEGER) FROM t");
    fails(&db, "SELECT id FROM t WHERE CAST(f AS INTEGER) > 0");
}

#[test]
fn test_a_write_outside_the_range_is_an_error() {
    let db = open("write");
    db.execute("INSERT INTO t VALUES (1, 1, 1e30, NULL)", ())
        .unwrap();
    for sql in [
        "INSERT INTO t (id, i) VALUES (2, 1e30)",
        "INSERT INTO t (id, i) VALUES (3, -1e30)",
        "INSERT INTO t (id, i) VALUES (4, '99999999999999999999')",
        "INSERT INTO t (id, i) SELECT 5, f FROM t WHERE id = 1",
        "UPDATE t SET i = 1e30 WHERE id = 1",
        "UPDATE t SET i = f WHERE id = 1",
        "UPDATE t SET i = f",
        "INSERT INTO t (id, i) VALUES (1, 5) ON DUPLICATE KEY UPDATE i = 1e30",
        "INSERT INTO t (id, i) VALUES (1, 5) ON DUPLICATE KEY UPDATE i = f",
    ] {
        fails(&db, sql);
    }
    assert!(db
        .execute("INSERT INTO t (id, i) VALUES (6, $1)", (1e30f64,))
        .is_err());
    assert!(db
        .execute("UPDATE t SET i = $1 WHERE id = $2", (1e30f64, 1i64))
        .is_err());
    assert_eq!(int(&db, "SELECT COUNT(*) FROM t"), Some(1));
    assert_eq!(int(&db, "SELECT i FROM t WHERE id = 1"), Some(1));

    db.execute("INSERT INTO t (id, i) VALUES (2, -3.7)", ())
        .unwrap();
    assert_eq!(int(&db, "SELECT i FROM t WHERE id = 2"), Some(-3));
}

#[test]
fn test_a_default_outside_the_range_is_an_error() {
    let db = open("default");
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, a INTEGER DEFAULT 1e30, b INTEGER DEFAULT (1e30 + 0))",
        (),
    )
    .unwrap();
    fails(&db, "INSERT INTO d (id, b) VALUES (1, 2)");
    fails(&db, "INSERT INTO d (id, a) VALUES (2, 2)");
    db.execute("INSERT INTO t (id) VALUES (1)", ()).unwrap();
    fails(&db, "ALTER TABLE t ADD COLUMN c INTEGER DEFAULT 1e30");
}

#[test]
fn test_copy_outside_the_range_is_an_error() {
    let db = open("copy");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "1,99999999999999999999").unwrap();
    let sql = format!(
        "COPY t (id, i) FROM '{}' WITH (FORMAT CSV, HEADER false)",
        tmp.path().display()
    );
    assert!(db.execute(&sql, ()).is_err());
    assert_eq!(int(&db, "SELECT COUNT(*) FROM t"), Some(0));

    let padded = tempfile::NamedTempFile::new().unwrap();
    writeln!(padded.as_file(), "1, 42").unwrap();
    let sql = format!(
        "COPY t (id, i) FROM '{}' WITH (FORMAT CSV, HEADER false)",
        padded.path().display()
    );
    db.execute(&sql, ()).unwrap();
    assert_eq!(int(&db, "SELECT i FROM t WHERE id = 1"), Some(42));
}

#[test]
fn test_an_integer_argument_outside_the_range_is_an_error() {
    let db = open("argument");
    for sql in [
        "SELECT LEFT('abc', 1e30)",
        "SELECT SUBSTRING('abc', 1, 1e30)",
        "SELECT CHAR(1e30)",
        "SELECT DATE_ADD('2024-01-01', 1e30)",
    ] {
        fails(&db, sql);
    }
    assert_eq!(
        db.query_one::<String, _>("SELECT LEFT('abc', 2.9)", ())
            .unwrap(),
        "ab"
    );
}

#[test]
fn test_a_column_changed_to_integer_holds_null_for_a_value_outside_the_range() {
    let db = open("modify");
    db.execute("INSERT INTO t (id, f) VALUES (1, 1e30), (2, -3.7)", ())
        .unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN f INTEGER", ())
        .unwrap();
    assert_eq!(int(&db, "SELECT f FROM t WHERE id = 1"), None);
    assert_eq!(int(&db, "SELECT f FROM t WHERE id = 2"), Some(-3));
}

#[test]
fn test_reading_a_float_as_i64_outside_the_range_is_an_error() {
    let db = open("read");
    assert!(db.query_one::<i64, _>("SELECT 1e30", ()).is_err());
    assert_eq!(db.query_one::<i64, _>("SELECT -3.7", ()).unwrap(), -3);
}
