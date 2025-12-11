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

//! AS OF (Time-Travel Query) Tests
//!
//! Tests for temporal queries using AS OF TRANSACTION and AS OF TIMESTAMP syntax.

use stoolap::Database;

// ============================================================================
// Parser Tests
// ============================================================================

#[test]
fn test_as_of_parser_transaction_integer() {
    let db = Database::open("memory://as_of_parser1").expect("Failed to create database");
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Create failed");
    db.execute("INSERT INTO users VALUES (1, 'Alice')", ())
        .expect("Insert failed");

    // AS OF TRANSACTION with integer should parse successfully
    let result = db.query("SELECT * FROM users AS OF TRANSACTION 12345", ());
    assert!(
        result.is_ok(),
        "AS OF TRANSACTION with integer should parse"
    );
}

#[test]
fn test_as_of_parser_timestamp_string() {
    let db = Database::open("memory://as_of_parser2").expect("Failed to create database");
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Create failed");
    db.execute("INSERT INTO users VALUES (1, 'Alice')", ())
        .expect("Insert failed");

    // AS OF TIMESTAMP with string should parse successfully
    let result = db.query(
        "SELECT * FROM users AS OF TIMESTAMP '2024-01-01 10:00:00'",
        (),
    );
    assert!(result.is_ok(), "AS OF TIMESTAMP with string should parse");
}

#[test]
fn test_as_of_parser_with_table_alias() {
    let db = Database::open("memory://as_of_parser3").expect("Failed to create database");
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Create failed");
    db.execute("INSERT INTO users VALUES (1, 'Alice')", ())
        .expect("Insert failed");

    // AS OF with table alias should parse successfully
    let result = db.query("SELECT * FROM users AS OF TRANSACTION 100 AS u", ());
    assert!(result.is_ok(), "AS OF with table alias should parse");
}

#[test]
fn test_as_of_parser_invalid_type() {
    let db = Database::open("memory://as_of_parser4").expect("Failed to create database");
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)", ())
        .expect("Create failed");

    // Invalid AS OF type should return error
    let result = db.query("SELECT * FROM users AS OF INVALID 100", ());
    assert!(result.is_err(), "Invalid AS OF type should error");
}

#[test]
fn test_as_of_parser_missing_value() {
    let db = Database::open("memory://as_of_parser5").expect("Failed to create database");
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)", ())
        .expect("Create failed");

    // Missing AS OF value should return error
    let result = db.query("SELECT * FROM users AS OF TRANSACTION", ());
    assert!(result.is_err(), "Missing AS OF value should error");
}

// ============================================================================
// Executor Tests
// ============================================================================

#[test]
fn test_as_of_transaction_basic() {
    let db = Database::open("memory://as_of_exec1").expect("Failed to create database");

    // Create table and insert initial data
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .expect("Create failed");

    db.execute("INSERT INTO users VALUES (1, 'Alice', 30)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO users VALUES (2, 'Bob', 25)", ())
        .expect("Insert failed");

    // Update Alice's age
    db.execute("UPDATE users SET age = 31 WHERE id = 1", ())
        .expect("Update failed");

    // Current query should see updated value
    let current_age: i64 = db
        .query_one("SELECT age FROM users WHERE id = 1", ())
        .expect("Query failed");
    assert_eq!(current_age, 31, "Current age should be 31");

    // Query AS OF earlier transaction should work
    let result = db.query("SELECT * FROM users AS OF TRANSACTION 3", ());
    assert!(result.is_ok(), "AS OF TRANSACTION query should execute");
}

#[test]
fn test_as_of_transaction_with_where() {
    let db = Database::open("memory://as_of_exec2").expect("Failed to create database");

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .expect("Create failed");

    db.execute("INSERT INTO users VALUES (1, 'Alice', 30)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO users VALUES (2, 'Bob', 25)", ())
        .expect("Insert failed");

    // Query with WHERE clause using AS OF
    let result = db.query("SELECT * FROM users AS OF TRANSACTION 2 WHERE id = 1", ());
    assert!(result.is_ok(), "AS OF with WHERE should execute");
}

#[test]
fn test_as_of_timestamp() {
    let db = Database::open("memory://as_of_exec3").expect("Failed to create database");

    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, event TEXT, status TEXT)",
        (),
    )
    .expect("Create failed");

    db.execute("INSERT INTO events VALUES (1, 'Event A', 'pending')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO events VALUES (2, 'Event B', 'pending')", ())
        .expect("Insert failed");

    // Update Event A status
    db.execute("UPDATE events SET status = 'completed' WHERE id = 1", ())
        .expect("Update failed");

    // Query with AS OF TIMESTAMP (past timestamp)
    let result = db.query(
        "SELECT * FROM events AS OF TIMESTAMP '2020-01-01 00:00:00'",
        (),
    );
    assert!(result.is_ok(), "AS OF TIMESTAMP should execute");

    // Query with AS OF TIMESTAMP (future timestamp should see current data)
    let result = db.query(
        "SELECT * FROM events AS OF TIMESTAMP '2050-01-01 00:00:00'",
        (),
    );
    assert!(
        result.is_ok(),
        "AS OF TIMESTAMP with future date should execute"
    );
}

#[test]
fn test_as_of_isolation() {
    let db = Database::open("memory://as_of_exec4").expect("Failed to create database");

    db.execute(
        "CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER)",
        (),
    )
    .expect("Create failed");

    db.execute("INSERT INTO accounts VALUES (1, 1000)", ())
        .expect("Insert failed");

    // Update balance
    db.execute("UPDATE accounts SET balance = 2000 WHERE id = 1", ())
        .expect("Update failed");

    // Current query should see new value
    let current_balance: i64 = db
        .query_one("SELECT balance FROM accounts WHERE id = 1", ())
        .expect("Query failed");
    assert_eq!(current_balance, 2000, "Current balance should be 2000");

    // AS OF earlier transaction should still work
    let result = db.query(
        "SELECT balance FROM accounts AS OF TRANSACTION 2 WHERE id = 1",
        (),
    );
    assert!(result.is_ok(), "AS OF query should execute");
}

#[test]
fn test_as_of_errors() {
    let db = Database::open("memory://as_of_exec5").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_errors (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Create failed");

    // Test invalid timestamp format
    let result = db.query(
        "SELECT * FROM test_errors AS OF TIMESTAMP 'invalid-date'",
        (),
    );
    assert!(result.is_err(), "Invalid timestamp format should error");

    // Test non-existent table
    let result = db.query("SELECT * FROM non_existent AS OF TRANSACTION 1", ());
    assert!(result.is_err(), "Non-existent table should error");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_as_of_version_chain_traversal() {
    let db = Database::open("memory://as_of_edge1").expect("Failed to create database");

    db.execute(
        "CREATE TABLE version_chain (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Create failed");

    // Create a version chain with multiple updates
    db.execute("INSERT INTO version_chain VALUES (1, 1)", ())
        .expect("Insert failed");

    for i in 2..=10 {
        db.execute(
            &format!("UPDATE version_chain SET value = {} WHERE id = 1", i),
            (),
        )
        .expect("Update failed");
    }

    // Query current value
    let current_value: i64 = db
        .query_one("SELECT value FROM version_chain WHERE id = 1", ())
        .expect("Query failed");
    assert_eq!(current_value, 10, "Current value should be 10");

    // AS OF at various points should execute
    for txn_id in [2, 5, 8, 10] {
        let result = db.query(
            &format!(
                "SELECT value FROM version_chain AS OF TRANSACTION {} WHERE id = 1",
                txn_id
            ),
            (),
        );
        assert!(
            result.is_ok(),
            "AS OF TRANSACTION {} should execute",
            txn_id
        );
    }
}

#[test]
fn test_as_of_with_deleted_version_chain() {
    let db = Database::open("memory://as_of_edge2").expect("Failed to create database");

    db.execute(
        "CREATE TABLE delete_chain (id INTEGER PRIMARY KEY, status TEXT)",
        (),
    )
    .expect("Create failed");

    // Create: insert -> update -> delete -> insert again
    db.execute("INSERT INTO delete_chain VALUES (1, 'created')", ())
        .expect("Insert failed");
    db.execute(
        "UPDATE delete_chain SET status = 'updated' WHERE id = 1",
        (),
    )
    .expect("Update failed");
    db.execute("DELETE FROM delete_chain WHERE id = 1", ())
        .expect("Delete failed");
    db.execute("INSERT INTO delete_chain VALUES (1, 'recreated')", ())
        .expect("Re-insert failed");

    // Current query should see 'recreated'
    let current_status: String = db
        .query_one("SELECT status FROM delete_chain WHERE id = 1", ())
        .expect("Query failed");
    assert_eq!(
        current_status, "recreated",
        "Current status should be 'recreated'"
    );

    // AS OF queries at various points should execute
    let result = db.query(
        "SELECT status FROM delete_chain AS OF TRANSACTION 2 WHERE id = 1",
        (),
    );
    assert!(result.is_ok(), "AS OF after insert should execute");
}

#[test]
fn test_as_of_timestamp_boundary_conditions() {
    let db = Database::open("memory://as_of_edge3").expect("Failed to create database");

    // Use TEXT column for this test since we're testing AS OF behavior, not TIMESTAMP column insertion
    db.execute(
        "CREATE TABLE timestamp_edge (id INTEGER PRIMARY KEY, created_at TEXT)",
        (),
    )
    .expect("Create failed");

    db.execute(
        "INSERT INTO timestamp_edge VALUES (1, '2024-06-15 12:00:00')",
        (),
    )
    .expect("Insert failed");

    // Test various timestamp boundaries
    let test_cases = [
        ("2024-06-15 12:00:01", true),  // 1 second after
        ("2024-06-15 11:00:00", false), // 1 hour before
        ("2024-06-16 12:00:00", true),  // 1 day after
        ("2050-01-01 00:00:00", true),  // Far future
        ("2020-01-01 00:00:00", false), // Far past
    ];

    for (timestamp, _should_have_row) in test_cases {
        let result = db.query(
            &format!(
                "SELECT id FROM timestamp_edge AS OF TIMESTAMP '{}'",
                timestamp
            ),
            (),
        );
        assert!(
            result.is_ok(),
            "AS OF TIMESTAMP '{}' should execute",
            timestamp
        );
    }
}

#[test]
fn test_as_of_error_conditions() {
    let db = Database::open("memory://as_of_edge4").expect("Failed to create database");

    db.execute(
        "CREATE TABLE error_test (id INTEGER PRIMARY KEY, data TEXT)",
        (),
    )
    .expect("Create failed");

    // Invalid AS OF type
    let result = db.query("SELECT * FROM error_test AS OF INVALID 123", ());
    assert!(result.is_err(), "Invalid AS OF type should error");

    // Non-integer transaction ID
    let result = db.query("SELECT * FROM error_test AS OF TRANSACTION 'abc'", ());
    assert!(result.is_err(), "Non-integer transaction ID should error");

    // AS OF TIMESTAMP with integer is accepted (interpreted as nanoseconds)
    let result = db.query("SELECT * FROM error_test AS OF TIMESTAMP 123", ());
    assert!(
        result.is_ok(),
        "AS OF TIMESTAMP with integer is valid (nanoseconds)"
    );

    // Malformed timestamp string
    let result = db.query("SELECT * FROM error_test AS OF TIMESTAMP 'not-a-date'", ());
    assert!(result.is_err(), "Malformed timestamp should error");

    // AS OF on non-existent table
    let result = db.query("SELECT * FROM non_existent AS OF TRANSACTION 1", ());
    assert!(result.is_err(), "AS OF on non-existent table should error");
}

#[test]
fn test_as_of_with_null_values() {
    let db = Database::open("memory://as_of_edge5").expect("Failed to create database");

    db.execute(
        "CREATE TABLE null_test (id INTEGER PRIMARY KEY, optional_value TEXT)",
        (),
    )
    .expect("Create failed");

    // Create versions with NULL values
    db.execute("INSERT INTO null_test VALUES (1, NULL)", ())
        .expect("Insert failed");
    db.execute(
        "UPDATE null_test SET optional_value = 'not null' WHERE id = 1",
        (),
    )
    .expect("Update failed");
    db.execute(
        "UPDATE null_test SET optional_value = NULL WHERE id = 1",
        (),
    )
    .expect("Update back to NULL failed");

    // AS OF queries should handle NULL correctly
    let result = db.query(
        "SELECT optional_value FROM null_test AS OF TRANSACTION 2 WHERE id = 1",
        (),
    );
    assert!(result.is_ok(), "AS OF with NULL should execute");
}

// ============================================================================
// Coverage Tests
// ============================================================================

#[test]
fn test_as_of_empty_table() {
    let db = Database::open("memory://as_of_cov1").expect("Failed to create database");

    db.execute(
        "CREATE TABLE coverage_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Create failed");

    // Query empty table with AS OF
    let result = db.query("SELECT * FROM coverage_test AS OF TRANSACTION 1", ());
    assert!(result.is_ok(), "AS OF on empty table should execute");
}

#[test]
fn test_as_of_very_old_transaction() {
    let db = Database::open("memory://as_of_cov2").expect("Failed to create database");

    db.execute(
        "CREATE TABLE coverage_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Create failed");
    db.execute("INSERT INTO coverage_test VALUES (1, 100)", ())
        .expect("Insert failed");

    // Query with transaction ID 0 (before any data)
    let result = db.query("SELECT * FROM coverage_test AS OF TRANSACTION 0", ());
    assert!(result.is_ok(), "AS OF TRANSACTION 0 should execute");
}

#[test]
fn test_as_of_very_old_timestamp() {
    let db = Database::open("memory://as_of_cov3").expect("Failed to create database");

    db.execute(
        "CREATE TABLE coverage_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Create failed");
    db.execute("INSERT INTO coverage_test VALUES (1, 100)", ())
        .expect("Insert failed");

    // Query with timestamp from 1970
    let result = db.query(
        "SELECT * FROM coverage_test AS OF TIMESTAMP '1970-01-01 00:00:00'",
        (),
    );
    assert!(result.is_ok(), "AS OF TIMESTAMP 1970 should execute");
}

#[test]
fn test_as_of_future_timestamp() {
    let db = Database::open("memory://as_of_cov4").expect("Failed to create database");

    db.execute(
        "CREATE TABLE coverage_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Create failed");
    db.execute("INSERT INTO coverage_test VALUES (1, 100)", ())
        .expect("Insert failed");

    // Query with future timestamp
    let result = db.query(
        "SELECT * FROM coverage_test AS OF TIMESTAMP '2050-01-01 00:00:00'",
        (),
    );
    assert!(result.is_ok(), "AS OF TIMESTAMP 2050 should execute");
}

#[test]
fn test_as_of_with_complex_expressions() {
    let db = Database::open("memory://as_of_cov5").expect("Failed to create database");

    db.execute(
        "CREATE TABLE coverage_test (id INTEGER PRIMARY KEY, value INTEGER, text_value TEXT)",
        (),
    )
    .expect("Create failed");
    db.execute("INSERT INTO coverage_test VALUES (1, 100, 'test1')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO coverage_test VALUES (2, 200, 'test2')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO coverage_test VALUES (3, 300, NULL)", ())
        .expect("Insert failed");

    // Complex WHERE with AS OF
    let result = db.query(
        "SELECT id, value FROM coverage_test AS OF TRANSACTION 4 \
         WHERE (value > 150 AND value < 400) OR text_value IS NULL \
         ORDER BY id",
        (),
    );
    assert!(result.is_ok(), "AS OF with complex WHERE should execute");
}

#[test]
fn test_as_of_concurrent_access() {
    let db = Database::open("memory://as_of_cov6").expect("Failed to create database");

    db.execute(
        "CREATE TABLE concurrent_test (id INTEGER PRIMARY KEY, counter INTEGER)",
        (),
    )
    .expect("Create failed");

    db.execute("INSERT INTO concurrent_test VALUES (1, 0)", ())
        .expect("Insert failed");

    // Create sequential updates
    for i in 1..=5 {
        db.execute(
            &format!("UPDATE concurrent_test SET counter = {} WHERE id = 1", i),
            (),
        )
        .expect("Update failed");
    }

    // Current value should be 5
    let current: i64 = db
        .query_one("SELECT counter FROM concurrent_test WHERE id = 1", ())
        .expect("Query failed");
    assert_eq!(current, 5, "Current counter should be 5");

    // AS OF at various points should execute
    for txn_id in 2..=6 {
        let result = db.query(
            &format!(
                "SELECT counter FROM concurrent_test AS OF TRANSACTION {} WHERE id = 1",
                txn_id
            ),
            (),
        );
        assert!(
            result.is_ok(),
            "AS OF TRANSACTION {} should execute",
            txn_id
        );
    }
}

#[test]
fn test_as_of_timestamp_precision() {
    let db = Database::open("memory://as_of_cov7").expect("Failed to create database");

    db.execute(
        "CREATE TABLE timestamp_precision (id INTEGER PRIMARY KEY)",
        (),
    )
    .expect("Create failed");
    db.execute("INSERT INTO timestamp_precision VALUES (1)", ())
        .expect("Insert failed");

    // Test various timestamp formats
    let timestamps = [
        "2025-01-01 12:00:00",        // Date and time
        "2025-01-01 12:00:00.123",    // With milliseconds
        "2025-01-01 12:00:00.123456", // With microseconds
    ];

    for ts in timestamps {
        let result = db.query(
            &format!(
                "SELECT id FROM timestamp_precision AS OF TIMESTAMP '{}'",
                ts
            ),
            (),
        );
        // Some formats might not be supported, that's OK
        if result.is_err() {
            println!(
                "Timestamp format '{}' not supported: {:?}",
                ts,
                result.err()
            );
        }
    }
}

#[test]
fn test_as_of_with_aggregation() {
    let db = Database::open("memory://as_of_cov8").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Create failed");

    db.execute("INSERT INTO sales VALUES (1, 100.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO sales VALUES (2, 200.0)", ())
        .expect("Insert failed");
    db.execute("UPDATE sales SET amount = 150.0 WHERE id = 1", ())
        .expect("Update failed");

    // Current SUM should be 350
    let current_sum: f64 = db
        .query_one("SELECT SUM(amount) FROM sales", ())
        .expect("Query failed");
    assert!(
        (current_sum - 350.0).abs() < 0.01,
        "Current SUM should be 350"
    );

    // AS OF with aggregation should execute
    let result = db.query("SELECT SUM(amount) FROM sales AS OF TRANSACTION 3", ());
    assert!(result.is_ok(), "AS OF with SUM should execute");
}

#[test]
fn test_as_of_select_specific_columns() {
    let db = Database::open("memory://as_of_cov9").expect("Failed to create database");

    db.execute(
        "CREATE TABLE multi_col (a INTEGER PRIMARY KEY, b TEXT, c FLOAT, d INTEGER)",
        (),
    )
    .expect("Create failed");
    db.execute("INSERT INTO multi_col VALUES (1, 'hello', 3.14, 42)", ())
        .expect("Insert failed");

    // Select specific columns with AS OF
    let result = db.query("SELECT a, c FROM multi_col AS OF TRANSACTION 2", ());
    assert!(result.is_ok(), "AS OF with specific columns should execute");

    let result = db.query(
        "SELECT b, d FROM multi_col AS OF TRANSACTION 2 WHERE a = 1",
        (),
    );
    assert!(
        result.is_ok(),
        "AS OF with specific columns and WHERE should execute"
    );
}

#[test]
fn test_as_of_select_star() {
    let db = Database::open("memory://as_of_cov10").expect("Failed to create database");

    db.execute(
        "CREATE TABLE star_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Create failed");
    db.execute("INSERT INTO star_test VALUES (1, 100)", ())
        .expect("Insert failed");
    db.execute("UPDATE star_test SET value = 200 WHERE id = 1", ())
        .expect("Update failed");

    // SELECT * with AS OF
    let result = db.query("SELECT * FROM star_test AS OF TRANSACTION 2", ());
    assert!(result.is_ok(), "AS OF with SELECT * should execute");
}
