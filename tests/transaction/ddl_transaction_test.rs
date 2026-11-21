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

//! DDL Transaction Rollback Tests
//!
//! Tests for DDL statements (CREATE TABLE, DROP TABLE) being properly
//! rolled back within transactions (Bug #86)

use stoolap::Database;

#[test]
fn test_create_table_rollback() {
    let db = Database::open("memory://ddl_create_rollback").expect("Failed to create database");

    // Begin transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Create table within transaction
    db.execute("CREATE TABLE rollback_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert data
    db.execute("INSERT INTO rollback_test VALUES (1, 'test')", ())
        .expect("Failed to insert");

    // Verify table exists within transaction
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM rollback_test", ())
        .expect("Table should exist within transaction");
    assert_eq!(count, 1, "Should have 1 row within transaction");

    // Rollback transaction
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // Verify table no longer exists after rollback
    let result = db.query("SELECT * FROM rollback_test", ());
    assert!(result.is_err(), "Table should not exist after rollback");
}

#[test]
fn test_create_table_commit() {
    let db = Database::open("memory://ddl_create_commit").expect("Failed to create database");

    // Begin transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Create table within transaction
    db.execute("CREATE TABLE commit_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert data
    db.execute("INSERT INTO commit_test VALUES (1, 'test')", ())
        .expect("Failed to insert");

    // Commit transaction
    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Verify table exists after commit
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM commit_test", ())
        .expect("Table should exist after commit");
    assert_eq!(count, 1, "Should have 1 row after commit");
}

#[test]
fn test_drop_table_rollback() {
    let db = Database::open("memory://ddl_drop_rollback").expect("Failed to create database");

    // First create a table outside transaction
    db.execute("CREATE TABLE persist_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Insert some data
    db.execute("INSERT INTO persist_test VALUES (1, 'test')", ())
        .expect("Failed to insert");

    // Verify table exists
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM persist_test", ())
        .expect("Table should exist");
    assert_eq!(count, 1, "Should have 1 row");

    // Begin transaction and drop table
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute("DROP TABLE persist_test", ())
        .expect("Failed to drop table");

    // Verify table is gone within transaction
    let result = db.query("SELECT * FROM persist_test", ());
    assert!(
        result.is_err(),
        "Table should not exist after DROP within transaction"
    );

    // Rollback transaction
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // Verify table is restored after rollback (schema only - data may be lost)
    let result = db.query("SELECT * FROM persist_test", ());
    assert!(
        result.is_ok(),
        "Table schema should be restored after rollback"
    );
}

#[test]
fn test_drop_table_commit() {
    let db = Database::open("memory://ddl_drop_commit").expect("Failed to create database");

    // First create a table
    db.execute("CREATE TABLE drop_commit_test (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    // Begin transaction and drop table
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");
    db.execute("DROP TABLE drop_commit_test", ())
        .expect("Failed to drop table");

    // Commit transaction
    db.execute("COMMIT", ())
        .expect("Failed to commit transaction");

    // Verify table is gone after commit
    let result = db.query("SELECT * FROM drop_commit_test", ());
    assert!(
        result.is_err(),
        "Table should not exist after committed DROP"
    );
}

#[test]
fn test_multiple_ddl_in_transaction() {
    let db = Database::open("memory://ddl_multiple").expect("Failed to create database");

    // Begin transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Create multiple tables
    db.execute("CREATE TABLE table1 (id INTEGER)", ())
        .expect("Failed to create table1");
    db.execute("CREATE TABLE table2 (id INTEGER)", ())
        .expect("Failed to create table2");
    db.execute("CREATE TABLE table3 (id INTEGER)", ())
        .expect("Failed to create table3");

    // Verify all tables exist
    db.execute("INSERT INTO table1 VALUES (1)", ())
        .expect("table1 should exist");
    db.execute("INSERT INTO table2 VALUES (2)", ())
        .expect("table2 should exist");
    db.execute("INSERT INTO table3 VALUES (3)", ())
        .expect("table3 should exist");

    // Rollback
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // Verify all tables are gone
    assert!(
        db.query("SELECT * FROM table1", ()).is_err(),
        "table1 should not exist after rollback"
    );
    assert!(
        db.query("SELECT * FROM table2", ()).is_err(),
        "table2 should not exist after rollback"
    );
    assert!(
        db.query("SELECT * FROM table3", ()).is_err(),
        "table3 should not exist after rollback"
    );
}

#[test]
fn test_ddl_outside_transaction_auto_commits() {
    let db = Database::open("memory://ddl_auto_commit").expect("Failed to create database");

    // Create table outside explicit transaction (should auto-commit)
    db.execute("CREATE TABLE auto_commit_test (id INTEGER)", ())
        .expect("Failed to create table");

    // Table should exist
    let result = db.query("SELECT * FROM auto_commit_test", ());
    assert!(result.is_ok(), "Table should exist after auto-commit");
}

#[test]
fn test_mixed_ddl_and_dml_rollback() {
    let db = Database::open("memory://ddl_dml_mixed").expect("Failed to create database");

    // Create a table first
    db.execute("CREATE TABLE existing_table (id INTEGER, value TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO existing_table VALUES (1, 'original')", ())
        .expect("Failed to insert");

    // Begin transaction
    db.execute("BEGIN", ())
        .expect("Failed to begin transaction");

    // Create new table
    db.execute("CREATE TABLE new_table (id INTEGER)", ())
        .expect("Failed to create new table");

    // Modify existing table
    db.execute(
        "UPDATE existing_table SET value = 'modified' WHERE id = 1",
        (),
    )
    .expect("Failed to update");

    // Rollback
    db.execute("ROLLBACK", ())
        .expect("Failed to rollback transaction");

    // New table should not exist
    assert!(
        db.query("SELECT * FROM new_table", ()).is_err(),
        "New table should not exist after rollback"
    );

    // Existing table should have original value (DML rollback)
    let value: String = db
        .query_one("SELECT value FROM existing_table WHERE id = 1", ())
        .expect("Should be able to query existing table");
    assert_eq!(value, "original", "Value should be rolled back to original");
}

/// Test: MODIFY COLUMN to NOT NULL fails when existing rows have NULLs
#[test]
fn test_modify_column_not_null_rejects_existing_nulls() {
    let db = Database::open("memory://modify_not_null_reject").expect("Failed to create database");

    db.execute(
        "CREATE TABLE t_nullable (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t_nullable VALUES (1, NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO t_nullable VALUES (2, 'hello')", ())
        .expect("Failed to insert");

    // Should fail because row 1 has NULL in val
    let result = db.execute("ALTER TABLE t_nullable MODIFY COLUMN val TEXT NOT NULL", ());
    assert!(result.is_err(), "Should reject NOT NULL when NULLs exist");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not null constraint"),
        "Error should mention not null constraint, got: {}",
        err
    );
}

/// Test: MODIFY COLUMN to NOT NULL succeeds when no NULLs exist
#[test]
fn test_modify_column_not_null_succeeds_without_nulls() {
    let db = Database::open("memory://modify_not_null_success").expect("Failed to create database");

    db.execute(
        "CREATE TABLE t_non_null (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t_non_null VALUES (1, 'a')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO t_non_null VALUES (2, 'b')", ())
        .expect("Failed to insert");

    // Should succeed because no NULLs
    db.execute("ALTER TABLE t_non_null MODIFY COLUMN val TEXT NOT NULL", ())
        .expect("MODIFY COLUMN should succeed when no NULLs exist");

    // Verify the constraint is now enforced on new inserts
    let result = db.execute("INSERT INTO t_non_null VALUES (3, NULL)", ());
    assert!(
        result.is_err(),
        "INSERT with NULL should fail after NOT NULL constraint"
    );
}

/// Test: MODIFY COLUMN to NOT NULL on empty table succeeds
#[test]
fn test_modify_column_not_null_empty_table() {
    let db = Database::open("memory://modify_not_null_empty").expect("Failed to create database");

    db.execute(
        "CREATE TABLE t_empty (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Empty table, no NULLs to violate
    db.execute("ALTER TABLE t_empty MODIFY COLUMN val TEXT NOT NULL", ())
        .expect("Should succeed on empty table");
}

fn ids(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    rows.map(|row| row.and_then(|r| r.get::<i64>(0)).map_err(|e| e.to_string()))
        .collect()
}

#[test]
fn create_table_as_select_in_a_transaction_is_undone_by_rollback() {
    let db = Database::open("memory://ddl_ctas_rollback").unwrap();
    db.execute("CREATE TABLE src (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO src VALUES (1), (2)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO src VALUES (3)", ()).unwrap();
    db.execute("CREATE TABLE ctas AS SELECT id FROM src", ())
        .unwrap();
    assert_eq!(
        ids(&db, "SELECT id FROM ctas ORDER BY id"),
        Ok(vec![1, 2, 3]),
        "the transaction reads the table it made, with its own rows"
    );
    db.execute("ROLLBACK", ()).unwrap();
    match ids(&db, "SELECT id FROM ctas") {
        Err(e) if e.contains("not found") => {}
        got => panic!("ctas after ROLLBACK: {got:?}, want table not found"),
    }
    assert_eq!(ids(&db, "SELECT id FROM src ORDER BY id"), Ok(vec![1, 2]));
}

#[test]
fn create_table_as_select_in_a_transaction_keeps_its_rows_on_commit() {
    let db = Database::open("memory://ddl_ctas_commit").unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("CREATE TABLE ctas AS SELECT 1 AS id UNION ALL SELECT 2", ())
        .unwrap();
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(ids(&db, "SELECT id FROM ctas ORDER BY id"), Ok(vec![1, 2]));
}

#[test]
fn create_table_as_select_after_a_savepoint_is_undone_by_rollback_to_it() {
    let db = Database::open("memory://ddl_ctas_savepoint").unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("CREATE TABLE kept AS SELECT 1 AS id", ())
        .unwrap();
    db.execute("SAVEPOINT s", ()).unwrap();
    db.execute("CREATE TABLE gone AS SELECT 2 AS id", ())
        .unwrap();
    db.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(ids(&db, "SELECT id FROM kept"), Ok(vec![1]));
    match ids(&db, "SELECT id FROM gone") {
        Err(e) if e.contains("not found") => {}
        got => panic!("gone after ROLLBACK TO SAVEPOINT: {got:?}, want table not found"),
    }
}

#[test]
fn create_table_as_select_rolled_back_stays_gone_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("BEGIN", ()).unwrap();
        db.execute("CREATE TABLE kept AS SELECT 1 AS id", ())
            .unwrap();
        db.execute("COMMIT", ()).unwrap();
        db.execute("BEGIN", ()).unwrap();
        db.execute("CREATE TABLE gone AS SELECT 2 AS id", ())
            .unwrap();
        db.execute("ROLLBACK", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(ids(&db, "SELECT id FROM kept"), Ok(vec![1]));
    match ids(&db, "SELECT id FROM gone") {
        Err(e) if e.contains("not found") => {}
        got => panic!("gone after reopen: {got:?}, want table not found"),
    }
}

#[test]
fn a_table_a_transaction_dropped_or_rolled_back_can_be_made_again_under_its_name() {
    let db = Database::open("memory://ddl_recreate_same_name").unwrap();
    db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("CREATE TABLE p (id INTEGER)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("DROP TABLE t", ()).unwrap();
    db.execute("CREATE TABLE t AS SELECT 2 AS id", ()).unwrap();
    db.execute("DROP TABLE t", ()).unwrap();
    db.execute("CREATE TABLE t AS SELECT 3 AS id", ()).unwrap();
    db.execute("SAVEPOINT s", ()).unwrap();
    db.execute("CREATE TABLE u AS SELECT 3 AS id", ()).unwrap();
    db.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    db.execute("CREATE TABLE u AS SELECT 4 AS id", ()).unwrap();
    db.execute("INSERT INTO p VALUES (6)", ()).unwrap();
    db.execute("DROP TABLE p", ()).unwrap();
    db.execute("CREATE TABLE p (id INTEGER)", ()).unwrap();
    db.execute("INSERT INTO p VALUES (5)", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    for (table, want) in [("t", 3), ("u", 4), ("p", 5)] {
        assert_eq!(
            ids(&db, &format!("SELECT id FROM {table}")),
            Ok(vec![want]),
            "{table} after COMMIT"
        );
    }
}

#[test]
fn a_row_written_through_a_held_handle_commits_to_its_table_renamed_meanwhile() {
    use stoolap::core::{Row, Value};
    use stoolap::storage::traits::Engine;
    let db = Database::open("memory://ddl_held_handle_rename").unwrap();
    db.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut held = tx.get_table("t").unwrap();
    let other = db.clone();
    other.execute("ALTER TABLE t RENAME TO u", ()).unwrap();
    other.execute("CREATE TABLE t (id INTEGER)", ()).unwrap();
    let _ = tx.get_table("t");
    held.insert(Row::from_values(vec![Value::Integer(1)]))
        .unwrap();
    drop(held);
    let committed = tx.commit();
    assert_eq!(
        (committed.is_ok(), ids(&db, "SELECT id FROM u")),
        (true, Ok(vec![1])),
        "(commit succeeded, rows of the renamed table)"
    );
}
