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

//! Integration tests for new SQL functions and syntax
//! Tests CONCAT_WS, STRPOS, POSITION(x IN y), and CURRENT_TRANSACTION_ID

use stoolap::Database;

// ============================================================================
// CONCAT_WS Function Tests
// ============================================================================

#[test]
fn test_concat_ws_basic() {
    let db = Database::open("memory://test_concat_ws").expect("Failed to create database");

    // Basic usage with hyphen separator
    let result: String = db
        .query_one("SELECT CONCAT_WS('-', 'hello', 'world', 'test')", ())
        .expect("Failed to query");
    assert_eq!(result, "hello-world-test");
}

#[test]
fn test_concat_ws_with_nulls() {
    let db = Database::open("memory://test_concat_ws_null").expect("Failed to create database");

    // NULLs are skipped in CONCAT_WS
    let result: String = db
        .query_one(
            "SELECT CONCAT_WS(', ', 'apple', NULL, 'banana', NULL, 'cherry')",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "apple, banana, cherry");
}

#[test]
fn test_concat_ws_null_separator() {
    let db = Database::open("memory://test_concat_ws_null_sep").expect("Failed to create database");

    // If separator is NULL, result is NULL
    let result = db.query("SELECT CONCAT_WS(NULL, 'a', 'b')", ()).unwrap();
    let row = result.into_iter().next().unwrap().unwrap();
    assert!(row.get::<Option<String>>(0).unwrap().is_none());
}

#[test]
fn test_concat_ws_two_args() {
    let db = Database::open("memory://test_concat_ws_two").expect("Failed to create database");

    // Minimum args: separator and one value
    let result: String = db
        .query_one("SELECT CONCAT_WS('-', 'only')", ())
        .expect("Failed to query");
    assert_eq!(result, "only");
}

#[test]
fn test_concat_ws_empty_separator() {
    let db = Database::open("memory://test_concat_ws_empty").expect("Failed to create database");

    // Empty string separator
    let result: String = db
        .query_one("SELECT CONCAT_WS('', 'a', 'b', 'c')", ())
        .expect("Failed to query");
    assert_eq!(result, "abc");
}

// ============================================================================
// STRPOS Function Tests
// ============================================================================

#[test]
fn test_strpos_basic() {
    let db = Database::open("memory://test_strpos").expect("Failed to create database");

    // Basic substring search
    let result: i64 = db
        .query_one("SELECT STRPOS('hello world', 'world')", ())
        .expect("Failed to query");
    assert_eq!(result, 7); // 1-indexed position
}

#[test]
fn test_strpos_not_found() {
    let db = Database::open("memory://test_strpos_notfound").expect("Failed to create database");

    // Substring not found returns 0
    let result: i64 = db
        .query_one("SELECT STRPOS('hello world', 'xyz')", ())
        .expect("Failed to query");
    assert_eq!(result, 0);
}

#[test]
fn test_strpos_at_start() {
    let db = Database::open("memory://test_strpos_start").expect("Failed to create database");

    // Substring at start
    let result: i64 = db
        .query_one("SELECT STRPOS('hello', 'he')", ())
        .expect("Failed to query");
    assert_eq!(result, 1);
}

#[test]
fn test_strpos_single_char() {
    let db = Database::open("memory://test_strpos_char").expect("Failed to create database");

    // Single character search
    let result: i64 = db
        .query_one("SELECT STRPOS('hello', 'e')", ())
        .expect("Failed to query");
    assert_eq!(result, 2);
}

#[test]
fn test_strpos_null_handling() {
    let db = Database::open("memory://test_strpos_null").expect("Failed to create database");

    // NULL returns NULL
    let result = db.query("SELECT STRPOS(NULL, 'test')", ()).unwrap();
    let row = result.into_iter().next().unwrap().unwrap();
    assert!(row.get::<Option<i64>>(0).unwrap().is_none());
}

// ============================================================================
// POSITION(x IN y) Syntax Tests
// ============================================================================

#[test]
fn test_position_in_syntax() {
    let db = Database::open("memory://test_position_in").expect("Failed to create database");

    // SQL standard POSITION(substring IN string) syntax
    let result: i64 = db
        .query_one("SELECT POSITION('world' IN 'hello world')", ())
        .expect("Failed to query");
    assert_eq!(result, 7);
}

#[test]
fn test_position_in_not_found() {
    let db = Database::open("memory://test_position_in_nf").expect("Failed to create database");

    // Not found returns 0
    let result: i64 = db
        .query_one("SELECT POSITION('xyz' IN 'hello world')", ())
        .expect("Failed to query");
    assert_eq!(result, 0);
}

#[test]
fn test_position_comma_syntax() {
    let db = Database::open("memory://test_position_comma").expect("Failed to create database");

    // Both syntaxes should work: POSITION(x IN y) and POSITION(x, y)
    let result_in: i64 = db
        .query_one("SELECT POSITION('e' IN 'hello')", ())
        .expect("Failed to query");
    let result_comma: i64 = db
        .query_one("SELECT POSITION('e', 'hello')", ())
        .expect("Failed to query");
    assert_eq!(result_in, result_comma);
    assert_eq!(result_in, 2);
}

#[test]
fn test_position_in_with_expression() {
    let db = Database::open("memory://test_position_expr").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_strings (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO test_strings VALUES (1, 'hello world'), (2, 'world hello')",
        (),
    )
    .unwrap();

    // POSITION in column expressions
    let result = db
        .query(
            "SELECT id, POSITION('world' IN name) as pos FROM test_strings ORDER BY id",
            (),
        )
        .unwrap();
    let rows: Vec<_> = result.into_iter().collect();
    assert_eq!(rows.len(), 2);

    let row1 = rows[0].as_ref().unwrap();
    assert_eq!(row1.get::<i64>(1).unwrap(), 7); // 'world' at position 7 in 'hello world'

    let row2 = rows[1].as_ref().unwrap();
    assert_eq!(row2.get::<i64>(1).unwrap(), 1); // 'world' at position 1 in 'world hello'
}

// ============================================================================
// CURRENT_TRANSACTION_ID Function Tests
// ============================================================================

#[test]
fn test_current_transaction_id_no_transaction() {
    let db = Database::open("memory://test_cur_txn_none").expect("Failed to create database");

    // Outside explicit transaction, returns NULL
    let result = db.query("SELECT CURRENT_TRANSACTION_ID()", ()).unwrap();
    let row = result.into_iter().next().unwrap().unwrap();
    assert!(row.get::<Option<i64>>(0).unwrap().is_none());
}

#[test]
fn test_current_transaction_id_with_transaction() {
    let db = Database::open("memory://test_cur_txn").expect("Failed to create database");

    // Start a transaction
    db.execute("BEGIN", ()).unwrap();

    // Inside transaction, returns the transaction ID (positive integer)
    let txn_id: i64 = db
        .query_one("SELECT CURRENT_TRANSACTION_ID()", ())
        .expect("Failed to query");
    assert!(txn_id > 0);

    db.execute("COMMIT", ()).unwrap();
}

#[test]
fn test_current_transaction_id_consistent_in_transaction() {
    let db = Database::open("memory://test_cur_txn_consistent").expect("Failed to create database");

    db.execute("BEGIN", ()).unwrap();

    // Multiple calls within same transaction should return same ID
    let txn_id1: i64 = db
        .query_one("SELECT CURRENT_TRANSACTION_ID()", ())
        .expect("Failed to query");
    let txn_id2: i64 = db
        .query_one("SELECT CURRENT_TRANSACTION_ID()", ())
        .expect("Failed to query");

    assert_eq!(txn_id1, txn_id2);

    db.execute("COMMIT", ()).unwrap();
}

#[test]
fn test_current_transaction_id_different_transactions() {
    let db = Database::open("memory://test_cur_txn_diff").expect("Failed to create database");

    // First transaction
    db.execute("BEGIN", ()).unwrap();
    let txn_id1: i64 = db
        .query_one("SELECT CURRENT_TRANSACTION_ID()", ())
        .expect("Failed to query");
    db.execute("COMMIT", ()).unwrap();

    // Second transaction
    db.execute("BEGIN", ()).unwrap();
    let txn_id2: i64 = db
        .query_one("SELECT CURRENT_TRANSACTION_ID()", ())
        .expect("Failed to query");
    db.execute("COMMIT", ()).unwrap();

    // Different transactions should have different IDs
    assert_ne!(txn_id1, txn_id2);
}

#[test]
fn test_current_transaction_id_after_rollback() {
    let db = Database::open("memory://test_cur_txn_rollback").expect("Failed to create database");

    db.execute("BEGIN", ()).unwrap();
    let _txn_id: i64 = db
        .query_one("SELECT CURRENT_TRANSACTION_ID()", ())
        .expect("Failed to query");
    db.execute("ROLLBACK", ()).unwrap();

    // After rollback, should be NULL again
    let result = db.query("SELECT CURRENT_TRANSACTION_ID()", ()).unwrap();
    let row = result.into_iter().next().unwrap().unwrap();
    assert!(row.get::<Option<i64>>(0).unwrap().is_none());
}
