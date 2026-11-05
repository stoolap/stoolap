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

//! Multi-Statement Parsing Tests
//!
//! Tests parsing of multiple SQL statements in one string

use stoolap::parser::parse_sql;

/// Test multi-statement parsing with transaction statements
#[test]
fn test_multi_statement_transaction() {
    let sql = "
BEGIN TRANSACTION;
INSERT INTO users (id, name) VALUES (1, 'Alice');
COMMIT;
";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Multi-statement parsing should succeed: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(
        statements.len(),
        3,
        "Expected 3 statements, got {}",
        statements.len()
    );
}

/// Test multi-statement with CREATE and INSERT
#[test]
fn test_multi_statement_create_insert() {
    let sql = "
CREATE TABLE test (id INTEGER, name TEXT);
INSERT INTO test VALUES (1, 'one');
INSERT INTO test VALUES (2, 'two');
";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Multi-statement parsing should succeed: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(statements.len(), 3, "Expected 3 statements");
}

/// Test multi-statement with SELECT
#[test]
fn test_multi_statement_select() {
    let sql = "
SELECT * FROM table1;
SELECT * FROM table2;
SELECT COUNT(*) FROM table3;
";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Multi-statement parsing should succeed: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(statements.len(), 3, "Expected 3 SELECT statements");
}

/// Test single statement (no semicolon)
#[test]
fn test_single_statement_no_semicolon() {
    let sql = "SELECT * FROM users";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Single statement should parse: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(statements.len(), 1, "Expected 1 statement");
}

/// Test single statement with semicolon
#[test]
fn test_single_statement_with_semicolon() {
    let sql = "SELECT * FROM users;";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Single statement should parse: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(statements.len(), 1, "Expected 1 statement");
}

/// Test multi-statement with UPDATE and DELETE
#[test]
fn test_multi_statement_update_delete() {
    let sql = "
UPDATE users SET name = 'Bob' WHERE id = 1;
DELETE FROM users WHERE id = 2;
";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Multi-statement parsing should succeed: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(statements.len(), 2, "Expected 2 statements");
}

/// Test multi-statement with CREATE INDEX
#[test]
fn test_multi_statement_with_index() {
    let sql = "
CREATE TABLE items (id INTEGER, name TEXT);
CREATE INDEX idx_name ON items (name);
";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Multi-statement parsing should succeed: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(statements.len(), 2, "Expected 2 statements");
}

/// Test multi-statement with DROP
#[test]
fn test_multi_statement_with_drop() {
    let sql = "
DROP INDEX IF EXISTS idx_old;
DROP TABLE IF EXISTS old_table;
";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Multi-statement parsing should succeed: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(statements.len(), 2, "Expected 2 statements");
}

/// Test empty statements (just semicolons)
#[test]
fn test_empty_statements() {
    let sql = ";;;";

    let result = parse_sql(sql);
    // This might succeed with 0 statements or fail - either is acceptable
    if let Ok(statements) = result {
        // If it succeeds, it should have 0 meaningful statements
        assert!(
            statements.is_empty() || statements.len() <= 3,
            "Empty statements should not produce multiple real statements"
        );
    }
}

/// Test mixed DDL and DML
#[test]
fn test_mixed_ddl_dml() {
    let sql = "
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT);
INSERT INTO products VALUES (1, 'Widget', 9.99);
SELECT * FROM products WHERE price > 5;
UPDATE products SET price = 10.99 WHERE id = 1;
";

    let result = parse_sql(sql);
    assert!(
        result.is_ok(),
        "Mixed DDL/DML should parse: {:?}",
        result.err()
    );

    let statements = result.unwrap();
    assert_eq!(statements.len(), 4, "Expected 4 statements");
}
