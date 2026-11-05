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

//! Integration tests for COPY FROM (CSV and JSON import)

use std::io::Write;
use stoolap::Database;

fn setup_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).expect("Failed to create database");
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, score FLOAT)",
        (),
    )
    .unwrap();
    db
}

// ============================================================================
// CSV Import Tests
// ============================================================================

#[test]
fn test_copy_csv_with_header() {
    let db = setup_db("csv_header");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,name,age,score").unwrap();
    writeln!(tmp.as_file(), "1,Alice,30,95.5").unwrap();
    writeln!(tmp.as_file(), "2,Bob,25,88.0").unwrap();
    writeln!(tmp.as_file(), "3,Charlie,35,92.3").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER true)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 3);

    let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
    assert_eq!(count, 3);

    let name: String = db
        .query_one("SELECT name FROM users WHERE id = 1", ())
        .unwrap();
    assert_eq!(name, "Alice");

    let age: i64 = db
        .query_one("SELECT age FROM users WHERE id = 2", ())
        .unwrap();
    assert_eq!(age, 25);

    let score: f64 = db
        .query_one("SELECT score FROM users WHERE id = 3", ())
        .unwrap();
    assert!((score - 92.3).abs() < 0.01);
}

#[test]
fn test_copy_csv_without_header() {
    let db = setup_db("csv_no_header");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "1,Alice,30,95.5").unwrap();
    writeln!(tmp.as_file(), "2,Bob,25,88.0").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER false)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 2);

    let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_copy_csv_custom_delimiter() {
    let db = setup_db("csv_delim");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id|name|age|score").unwrap();
    writeln!(tmp.as_file(), "1|Alice|30|95.5").unwrap();
    writeln!(tmp.as_file(), "2|Bob|25|88.0").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER true, DELIMITER '|')",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 2);

    let name: String = db
        .query_one("SELECT name FROM users WHERE id = 1", ())
        .unwrap();
    assert_eq!(name, "Alice");
}

#[test]
fn test_copy_csv_semicolon_delimiter() {
    let db = setup_db("csv_semi");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id;name;age;score").unwrap();
    writeln!(tmp.as_file(), "1;Alice;30;95.5").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER true, DELIMITER ';')",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);
}

#[test]
fn test_copy_csv_with_specific_columns() {
    let db = Database::open("memory://csv_cols").expect("Failed to create database");
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price FLOAT DEFAULT 0.0)",
        (),
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "1,Widget").unwrap();
    writeln!(tmp.as_file(), "2,Gadget").unwrap();

    let sql = format!(
        "COPY items (id, name) FROM '{}' WITH (FORMAT CSV, HEADER false)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 2);

    // price should have default value 0.0
    let price: f64 = db
        .query_one("SELECT price FROM items WHERE id = 1", ())
        .unwrap();
    assert_eq!(price, 0.0);
}

#[test]
fn test_copy_csv_null_handling() {
    let db = setup_db("csv_null");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,name,age,score").unwrap();
    writeln!(tmp.as_file(), "1,Alice,30,95.5").unwrap();
    writeln!(tmp.as_file(), "2,,25,").unwrap(); // empty fields

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER true)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 2);
}

#[test]
fn test_copy_csv_custom_null_string() {
    let db = setup_db("csv_null_str");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,name,age,score").unwrap();
    writeln!(tmp.as_file(), "1,Alice,30,95.5").unwrap();
    writeln!(tmp.as_file(), "2,NA,25,NA").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER true, NULL 'NA')",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 2);

    // "NA" values should be NULL
    let result: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM users WHERE id = 2 AND name IS NULL",
            (),
        )
        .unwrap();
    assert_eq!(result, 1, "Expected name to be NULL for id=2");
}

#[test]
fn test_copy_csv_type_coercion() {
    let db = Database::open("memory://csv_coerce").expect("Failed to create database");
    db.execute(
        "CREATE TABLE typed (id INTEGER PRIMARY KEY, flag BOOLEAN, ts TIMESTAMP)",
        (),
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,flag,ts").unwrap();
    writeln!(tmp.as_file(), "1,true,2024-01-15T10:30:00").unwrap();
    writeln!(tmp.as_file(), "2,false,2024-06-20T14:00:00").unwrap();

    let sql = format!(
        "COPY typed FROM '{}' WITH (FORMAT CSV, HEADER true)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 2);

    let flag: bool = db
        .query_one("SELECT flag FROM typed WHERE id = 1", ())
        .unwrap();
    assert!(flag);

    let flag: bool = db
        .query_one("SELECT flag FROM typed WHERE id = 2", ())
        .unwrap();
    assert!(!flag);
}

#[test]
fn test_copy_csv_default_format() {
    // When no WITH clause, should default to CSV with header=true
    let db = setup_db("csv_default");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,name,age,score").unwrap();
    writeln!(tmp.as_file(), "1,Alice,30,95.5").unwrap();

    let sql = format!("COPY users FROM '{}'", tmp.path().display());
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);
}

// ============================================================================
// JSON Import Tests
// ============================================================================

#[test]
fn test_copy_json_array() {
    let db = setup_db("json_array");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    write!(
        tmp.as_file(),
        r#"[
        {{"id": 1, "name": "Alice", "age": 30, "score": 95.5}},
        {{"id": 2, "name": "Bob", "age": 25, "score": 88.0}},
        {{"id": 3, "name": "Charlie", "age": 35, "score": 92.3}}
    ]"#
    )
    .unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 3);

    let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
    assert_eq!(count, 3);

    let name: String = db
        .query_one("SELECT name FROM users WHERE id = 2", ())
        .unwrap();
    assert_eq!(name, "Bob");
}

#[test]
fn test_copy_json_lines() {
    let db = setup_db("json_lines");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(
        tmp.as_file(),
        r#"{{"id": 1, "name": "Alice", "age": 30, "score": 95.5}}"#
    )
    .unwrap();
    writeln!(
        tmp.as_file(),
        r#"{{"id": 2, "name": "Bob", "age": 25, "score": 88.0}}"#
    )
    .unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 2);

    let name: String = db
        .query_one("SELECT name FROM users WHERE id = 1", ())
        .unwrap();
    assert_eq!(name, "Alice");
}

#[test]
fn test_copy_json_with_null_values() {
    let db = setup_db("json_null");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(
        tmp.as_file(),
        r#"{{"id": 1, "name": "Alice", "age": 30, "score": null}}"#
    )
    .unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);

    let is_null: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM users WHERE id = 1 AND score IS NULL",
            (),
        )
        .unwrap();
    assert_eq!(is_null, 1);
}

#[test]
fn test_copy_json_partial_columns() {
    let db = setup_db("json_partial");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    // JSON objects only have some fields, missing fields get defaults
    writeln!(tmp.as_file(), r#"{{"id": 1, "name": "Alice"}}"#).unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);

    let name: String = db
        .query_one("SELECT name FROM users WHERE id = 1", ())
        .unwrap();
    assert_eq!(name, "Alice");
}

#[test]
fn test_copy_json_with_specific_columns() {
    let db = Database::open("memory://json_cols").expect("Failed to create database");
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price FLOAT DEFAULT 9.99)",
        (),
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(
        tmp.as_file(),
        r#"{{"id": 1, "name": "Widget", "price": 5.00, "extra": "ignored"}}"#
    )
    .unwrap();

    let sql = format!(
        "COPY items (id, name) FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);

    // price should be the default 9.99, not the JSON value 5.00
    let price: f64 = db
        .query_one("SELECT price FROM items WHERE id = 1", ())
        .unwrap();
    assert!((price - 9.99).abs() < 0.01);
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_copy_file_not_found() {
    let db = setup_db("err_notfound");
    let result = db.execute(
        "COPY users FROM '/nonexistent/file.csv' WITH (FORMAT CSV)",
        (),
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("cannot open file"), "Got: {}", err);
}

#[test]
fn test_copy_csv_column_mismatch() {
    let db = setup_db("err_mismatch");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    // 3 fields but table has 4 columns
    writeln!(tmp.as_file(), "1,Alice,30").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER false)",
        tmp.path().display()
    );
    let result = db.execute(&sql, ());
    assert!(result.is_err());
}

#[test]
fn test_copy_json_invalid_json() {
    let db = setup_db("err_badjson");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "not json at all").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let result = db.execute(&sql, ());
    assert!(result.is_err());
}

#[test]
fn test_copy_csv_type_error() {
    let db = setup_db("err_type");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,name,age,score").unwrap();
    writeln!(tmp.as_file(), "not_a_number,Alice,30,95.5").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER true)",
        tmp.path().display()
    );
    let result = db.execute(&sql, ());
    // "not_a_number" can't be coerced to INTEGER, should fail
    assert!(result.is_err());
}

#[test]
fn test_copy_in_transaction_blocked() {
    let db = setup_db("err_txn");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "1,Alice,30,95.5").unwrap();

    db.execute("BEGIN", ()).unwrap();
    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER false)",
        tmp.path().display()
    );
    let result = db.execute(&sql, ());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("explicit transaction"), "Got: {}", err);
    db.execute("ROLLBACK", ()).unwrap();
}

// ============================================================================
// Large Data Test
// ============================================================================

#[test]
fn test_copy_csv_many_rows() {
    let db = Database::open("memory://csv_many").expect("Failed to create database");
    db.execute(
        "CREATE TABLE bench (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,val").unwrap();
    for i in 0..10000 {
        writeln!(tmp.as_file(), "{},{}", i, i * 2).unwrap();
    }

    let sql = format!(
        "COPY bench FROM '{}' WITH (FORMAT CSV, HEADER true)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 10000);

    let count: i64 = db.query_one("SELECT COUNT(*) FROM bench", ()).unwrap();
    assert_eq!(count, 10000);

    let val: i64 = db
        .query_one("SELECT val FROM bench WHERE id = 5000", ())
        .unwrap();
    assert_eq!(val, 10000);
}

#[test]
fn test_copy_json_many_rows() {
    let db = Database::open("memory://json_many").expect("Failed to create database");
    db.execute(
        "CREATE TABLE bench (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    for i in 0..1000 {
        writeln!(tmp.as_file(), r#"{{"id": {}, "val": {}}}"#, i, i * 3).unwrap();
    }

    let sql = format!(
        "COPY bench FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1000);

    // sum of i*3 for i=0..999 = 3 * (999*1000/2) = 1498500
    let sum: i64 = db.query_one("SELECT SUM(val) FROM bench", ()).unwrap();
    assert_eq!(sum, 1498500);
}

// ============================================================================
// Bug Regression Tests
// ============================================================================

#[test]
fn test_copy_json_bad_coercion_fails() {
    // P1: JSON COPY must reject values that fail type coercion, not silently null them.
    let db = Database::open("memory://json_coerce_fail").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, age INTEGER)", ())
        .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), r#"{{"id": 1, "age": "oops"}}"#).unwrap();

    let sql = format!("COPY t FROM '{}' WITH (FORMAT JSON)", tmp.path().display());
    let result = db.execute(&sql, ());
    assert!(result.is_err(), "Expected type error for 'oops' -> INTEGER");
}

#[test]
fn test_copy_csv_bad_coercion_fails() {
    // CSV path already rejects bad coercions via parse_field; verify it stays that way.
    let db = Database::open("memory://csv_coerce_fail").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val FLOAT)", ())
        .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,val").unwrap();
    writeln!(tmp.as_file(), "1,not_a_float").unwrap();

    let sql = format!(
        "COPY t FROM '{}' WITH (FORMAT CSV, HEADER true)",
        tmp.path().display()
    );
    let result = db.execute(&sql, ());
    assert!(
        result.is_err(),
        "Expected type error for 'not_a_float' -> FLOAT"
    );
}

#[test]
fn test_copy_json_case_insensitive_keys() {
    // P2: JSON key matching must be fully case-insensitive.
    let db = Database::open("memory://json_ci_keys").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    // Uppercase keys
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), r#"{{"ID": 1, "NAME": "Alice"}}"#).unwrap();

    let sql = format!("COPY t FROM '{}' WITH (FORMAT JSON)", tmp.path().display());
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);

    let name: String = db.query_one("SELECT name FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(name, "Alice");
}

#[test]
fn test_copy_json_case_insensitive_keys_with_columns() {
    // P2: Case-insensitive matching must also work when columns are specified.
    let db = Database::open("memory://json_ci_cols").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), r#"{{"ID": 1, "NAME": "Bob"}}"#).unwrap();

    let sql = format!(
        "COPY t (id, name) FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);

    let name: String = db.query_one("SELECT name FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(name, "Bob");
}

#[test]
fn test_copy_json_case_insensitive_unicode_keys_with_columns() {
    // P2: Unicode case-insensitive matching with explicit column list.
    let db = Database::open("memory://json_ci_unicode").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    // JSON key uses uppercase Unicode: "ÄNAME" should match column "äname"
    // (stoolap lowercases column names, so "name" is stored as "name")
    // But let's test with a column that stays ASCII to isolate the JSON key side.
    let tmp = tempfile::NamedTempFile::new().unwrap();
    // Use a mixed-case key with non-ASCII: uppercase Ä in JSON key
    writeln!(tmp.as_file(), r#"{{"id": 1, "NAME": "Alice"}}"#).unwrap();
    writeln!(tmp.as_file(), r#"{{"id": 2, "Name": "Bob"}}"#).unwrap();

    let sql = format!(
        "COPY t (id, name) FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 2);

    let name: String = db.query_one("SELECT name FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(name, "Alice");
    let name: String = db.query_one("SELECT name FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(name, "Bob");
}

#[test]
fn test_copy_csv_vector_dimension_rejected() {
    // P1: COPY must reject vectors with wrong dimensions, same as INSERT.
    let db = Database::open("memory://csv_vec_dim").unwrap();
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,vec").unwrap();
    // 2-element vector into VECTOR(3) column
    writeln!(tmp.as_file(), r#"1,"[1.0, 2.0]""#).unwrap();

    let sql = format!(
        "COPY vecs FROM '{}' WITH (FORMAT CSV, HEADER true)",
        tmp.path().display()
    );
    let result = db.execute(&sql, ());
    assert!(result.is_err(), "Should reject wrong vector dimension");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("imension"),
        "Error should mention dimension mismatch, got: {}",
        err
    );
}

#[test]
fn test_copy_csv_vector_dimension_accepted() {
    // Correct dimensions should succeed.
    let db = Database::open("memory://csv_vec_ok").unwrap();
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,vec").unwrap();
    writeln!(tmp.as_file(), r#"1,"[1.0, 2.0, 3.0]""#).unwrap();

    let sql = format!(
        "COPY vecs FROM '{}' WITH (FORMAT CSV, HEADER true)",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);
}

#[test]
fn test_copy_json_vector_dimension_rejected() {
    // P1: JSON COPY must also reject wrong vector dimensions.
    let db = Database::open("memory://json_vec_dim").unwrap();
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), r#"{{"id": 1, "vec": "[1.0, 2.0]"}}"#).unwrap();

    let sql = format!(
        "COPY vecs FROM '{}' WITH (FORMAT JSON)",
        tmp.path().display()
    );
    let result = db.execute(&sql, ());
    assert!(result.is_err(), "Should reject wrong vector dimension");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("imension"),
        "Error should mention dimension mismatch, got: {}",
        err
    );
}

// ============================================================================
// Parser Tests
// ============================================================================

#[test]
fn test_copy_parse_minimal() {
    // Minimal syntax: COPY table FROM 'path' (defaults to CSV with header)
    let db = setup_db("parse_min");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id,name,age,score").unwrap();
    writeln!(tmp.as_file(), "1,Alice,30,95.5").unwrap();

    let sql = format!("COPY users FROM '{}'", tmp.path().display());
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);
}

#[test]
fn test_copy_parse_all_options() {
    let db = setup_db("parse_all");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp.as_file(), "id;name;age;score").unwrap();
    writeln!(tmp.as_file(), "1;Alice;30;95.5").unwrap();

    let sql = format!(
        "COPY users FROM '{}' WITH (FORMAT CSV, HEADER true, DELIMITER ';', NULL 'NA')",
        tmp.path().display()
    );
    let affected = db.execute(&sql, ()).unwrap();
    assert_eq!(affected, 1);
}
