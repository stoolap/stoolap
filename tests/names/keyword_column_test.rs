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

//! Keyword Column Tests
//!
//! Tests using SQL keywords as column names with proper quoting

use stoolap::Database;

/// Test creating table with double-quoted keyword columns
#[test]
fn test_double_quoted_keywords() {
    let db = Database::open("memory://keyword_double").expect("Failed to create database");

    db.execute(r#"CREATE TABLE test_keywords ("select" INTEGER, "from" TEXT, "where" BOOLEAN, "order" INTEGER)"#, ())
        .expect("Failed to create table with double-quoted keyword columns");

    // Verify table was created by inserting data
    db.execute(r#"INSERT INTO test_keywords ("select", "from", "where", "order") VALUES (1, 'test', true, 2)"#, ())
        .expect("Failed to insert into table with keyword columns");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_keywords", ())
        .expect("Failed to count");
    assert_eq!(count, 1);
}

/// Test creating table with backtick-quoted keyword columns
#[test]
fn test_backtick_keywords() {
    let db = Database::open("memory://keyword_backtick").expect("Failed to create database");

    db.execute("CREATE TABLE test_keywords2 (`insert` INTEGER, `update` TEXT, `delete` BOOLEAN, `table` INTEGER)", ())
        .expect("Failed to create table with backtick-quoted keyword columns");

    // Verify table was created by inserting data
    db.execute("INSERT INTO test_keywords2 (`insert`, `update`, `delete`, `table`) VALUES (10, 'data', false, 20)", ())
        .expect("Failed to insert into table with backtick columns");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM test_keywords2", ())
        .expect("Failed to count");
    assert_eq!(count, 1);
}

/// Test mixed quoted and unquoted columns
#[test]
fn test_mixed_quoted_unquoted() {
    let db = Database::open("memory://keyword_mixed").expect("Failed to create database");

    db.execute(
        r#"CREATE TABLE test_mixed (id INTEGER, "key" TEXT, name TEXT, "default" INTEGER)"#,
        (),
    )
    .expect("Failed to create table with mixed columns");

    db.execute(
        r#"INSERT INTO test_mixed (id, "key", name, "default") VALUES (1, 'mykey', 'test', 100)"#,
        (),
    )
    .expect("Failed to insert");

    let result = db
        .query(r#"SELECT id, "key", name, "default" FROM test_mixed"#, ())
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let key: String = row.get(1).unwrap();
        let name: String = row.get(2).unwrap();
        let default_val: i64 = row.get(3).unwrap();

        assert_eq!(id, 1);
        assert_eq!(key, "mykey");
        assert_eq!(name, "test");
        assert_eq!(default_val, 100);
        count += 1;
    }
    assert_eq!(count, 1);
}

/// Test INSERT and SELECT with quoted column names
#[test]
fn test_insert_select_quoted_columns() {
    let db = Database::open("memory://keyword_insert").expect("Failed to create database");

    db.execute(r#"CREATE TABLE keyword_ops ("select" INTEGER, "from" TEXT, "where" BOOLEAN, "order" INTEGER)"#, ())
        .expect("Failed to create table");

    db.execute(r#"INSERT INTO keyword_ops ("select", "from", "where", "order") VALUES (1, 'test', true, 2)"#, ())
        .expect("Failed to insert");

    let result = db
        .query(
            r#"SELECT "select", "from", "where", "order" FROM keyword_ops"#,
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let select_val: i64 = row.get(0).unwrap();
        let from_val: String = row.get(1).unwrap();
        let where_val: bool = row.get(2).unwrap();
        let order_val: i64 = row.get(3).unwrap();

        assert_eq!(select_val, 1);
        assert_eq!(from_val, "test");
        assert!(where_val);
        assert_eq!(order_val, 2);
    }
}

/// Test UPDATE with quoted column names
#[test]
fn test_update_quoted_columns() {
    let db = Database::open("memory://keyword_update").expect("Failed to create database");

    db.execute(
        r#"CREATE TABLE keyword_update ("select" INTEGER, "from" TEXT)"#,
        (),
    )
    .expect("Failed to create table");

    db.execute(
        r#"INSERT INTO keyword_update ("select", "from") VALUES (1, 'original')"#,
        (),
    )
    .expect("Failed to insert");

    db.execute(
        r#"UPDATE keyword_update SET "from" = 'updated' WHERE "select" = 1"#,
        (),
    )
    .expect("Failed to update");

    let result: String = db
        .query_one(
            r#"SELECT "from" FROM keyword_update WHERE "select" = 1"#,
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "updated");
}

/// Test CREATE INDEX on keyword column
#[test]
fn test_create_index_on_keyword_column() {
    let db = Database::open("memory://keyword_index").expect("Failed to create database");

    db.execute(
        r#"CREATE TABLE indexed_keywords ("from" TEXT, id INTEGER)"#,
        (),
    )
    .expect("Failed to create table");

    db.execute(r#"CREATE INDEX idx_from ON indexed_keywords ("from")"#, ())
        .expect("Failed to create index on keyword column");

    // Insert some data to test the index works
    db.execute(
        r#"INSERT INTO indexed_keywords ("from", id) VALUES ('value1', 1)"#,
        (),
    )
    .expect("Failed to insert");
    db.execute(
        r#"INSERT INTO indexed_keywords ("from", id) VALUES ('value2', 2)"#,
        (),
    )
    .expect("Failed to insert");

    let count: i64 = db
        .query_one(
            r#"SELECT COUNT(*) FROM indexed_keywords WHERE "from" = 'value1'"#,
            (),
        )
        .expect("Failed to query");
    assert_eq!(count, 1);
}

/// Test case sensitivity with quoted identifiers (backtick quoting)
#[test]
fn test_case_sensitivity() {
    let db = Database::open("memory://keyword_case").expect("Failed to create database");

    db.execute(
        "CREATE TABLE case_test (`MyColumn` INTEGER, id INTEGER)",
        (),
    )
    .expect("Failed to create table with mixed case column");

    db.execute("INSERT INTO case_test (`MyColumn`, id) VALUES (1, 1)", ())
        .expect("Failed to insert with mixed case column");

    // Query returns lowercase column names (Stoolap normalizes internally)
    let result = db
        .query("SELECT `MyColumn` FROM case_test", ())
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let val: i64 = row.get(0).unwrap();
        assert_eq!(val, 1);
        count += 1;
    }
    assert_eq!(count, 1);
}

/// Test with multiple SQL keywords as columns
#[test]
fn test_many_keyword_columns() {
    let db = Database::open("memory://keyword_many").expect("Failed to create database");

    db.execute(
        r#"CREATE TABLE keyword_test (
            "select" INTEGER,
            "from" INTEGER,
            "where" INTEGER,
            "insert" INTEGER,
            "update" INTEGER,
            "delete" INTEGER,
            "create" INTEGER,
            "table" INTEGER,
            "drop" INTEGER,
            "alter" INTEGER
        )"#,
        (),
    )
    .expect("Failed to create table with many keyword columns");

    db.execute(
        r#"INSERT INTO keyword_test ("select", "from", "where", "insert", "update", "delete", "create", "table", "drop", "alter")
           VALUES (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)"#, ())
    .expect("Failed to insert");

    let result: i64 = db
        .query_one(r#"SELECT "select" FROM keyword_test"#, ())
        .expect("Failed to query");
    assert_eq!(result, 1);

    let result: i64 = db
        .query_one(r#"SELECT "alter" FROM keyword_test"#, ())
        .expect("Failed to query");
    assert_eq!(result, 10);
}

/// Test DELETE with quoted column in WHERE
#[test]
fn test_delete_quoted_columns() {
    let db = Database::open("memory://keyword_delete").expect("Failed to create database");

    db.execute(
        r#"CREATE TABLE keyword_del ("where" INTEGER, value TEXT)"#,
        (),
    )
    .expect("Failed to create table");

    db.execute(
        r#"INSERT INTO keyword_del ("where", value) VALUES (1, 'keep')"#,
        (),
    )
    .unwrap();
    db.execute(
        r#"INSERT INTO keyword_del ("where", value) VALUES (2, 'delete')"#,
        (),
    )
    .unwrap();

    db.execute(r#"DELETE FROM keyword_del WHERE "where" = 2"#, ())
        .expect("Failed to delete");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM keyword_del", ())
        .expect("Failed to count");
    assert_eq!(count, 1);
}
