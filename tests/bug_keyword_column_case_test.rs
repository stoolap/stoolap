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

//! Test: reserved keywords used as aliases should preserve lowercase in results.

use stoolap::Database;

#[test]
fn test_keyword_alias_preserves_lowercase() {
    let db = Database::open("memory://keyword_alias_case").unwrap();

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, ts TEXT, val TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, '2024-01-01', 'hello')", ())
        .unwrap();

    // Alias with keyword: `time` is a keyword and should not become TIME
    let rows = db
        .query("SELECT ts AS time, val AS value FROM t", ())
        .unwrap();
    for row in rows {
        let row = row.unwrap();
        let columns = row.columns();
        assert_eq!(
            columns,
            &["time", "value"],
            "Keyword aliases should be lowercase, got: {:?}",
            columns
        );
    }
}

#[test]
fn test_keyword_alias_with_backtick_column() {
    let db = Database::open("memory://keyword_alias_backtick").unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, `time` TEXT)", ())
        .unwrap();

    db.execute("INSERT INTO t VALUES (1, '10:30:00')", ())
        .unwrap();

    // Backtick column + keyword alias
    let rows = db.query("SELECT `time` AS time FROM t", ()).unwrap();
    for row in rows {
        let row = row.unwrap();
        let columns = row.columns();
        assert_eq!(
            columns,
            &["time"],
            "Keyword alias should be lowercase, got: {:?}",
            columns
        );
    }
}

#[test]
fn test_keyword_alias_various_keywords() {
    let db = Database::open("memory://keyword_alias_various").unwrap();

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 'x', 42)", ()).unwrap();

    // Test multiple keyword aliases
    let rows = db
        .query("SELECT a AS type, b AS order, id AS key FROM t", ())
        .unwrap();
    for row in rows {
        let row = row.unwrap();
        let columns = row.columns();
        assert_eq!(
            columns,
            &["type", "order", "key"],
            "All keyword aliases should be lowercase, got: {:?}",
            columns
        );
    }
}

#[test]
fn test_expression_keyword_alias() {
    let db = Database::open("memory://expr_keyword_alias").unwrap();

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, price FLOAT)", ())
        .unwrap();

    db.execute("INSERT INTO t VALUES (1, 99.5)", ()).unwrap();

    // Expression with keyword alias
    let rows = db.query("SELECT price * 1.1 AS value FROM t", ()).unwrap();
    for row in rows {
        let row = row.unwrap();
        let columns = row.columns();
        assert_eq!(
            columns,
            &["value"],
            "Expression keyword alias should be lowercase, got: {:?}",
            columns
        );
    }
}

#[test]
fn test_quoted_column_names_preserve_case() {
    let db = Database::open("memory://quoted_col_case").unwrap();

    db.execute(
        "CREATE TABLE t (
            id INTEGER PRIMARY KEY,
            \"key\" TEXT,
            \"value\" TEXT
        )",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 'mykey', 'myvalue')", ())
        .unwrap();

    // Double-quoted keyword columns should preserve original case
    let rows = db.query("SELECT \"key\", \"value\" FROM t", ()).unwrap();
    for row in rows {
        let row = row.unwrap();
        let columns = row.columns();
        assert_eq!(
            columns,
            &["key", "value"],
            "Quoted column names should preserve case, got: {:?}",
            columns
        );
    }
}

#[test]
fn test_unquoted_keyword_column_in_create_table() {
    let db = Database::open("memory://keyword_col_create").unwrap();

    // 'level' is a keyword — when used unquoted as a column name,
    // it should appear lowercase in results, not as LEVEL
    db.execute(
        "CREATE TABLE logs (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            level TEXT NOT NULL,
            message TEXT NOT NULL
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO logs (level, message) VALUES ('info', 'hello world')",
        (),
    )
    .unwrap();

    let rows = db.query("SELECT level, message FROM logs", ()).unwrap();
    for row in rows {
        let row = row.unwrap();
        let columns = row.columns();
        assert_eq!(
            columns,
            &["level", "message"],
            "Unquoted keyword column should be lowercase, got: {:?}",
            columns
        );
    }

    // Also check SELECT *
    let rows = db.query("SELECT * FROM logs", ()).unwrap();
    for row in rows {
        let row = row.unwrap();
        let columns = row.columns();
        assert!(
            columns.contains(&"level".to_string()),
            "Expected 'level' (lowercase) in SELECT * columns, got: {:?}",
            columns
        );
    }
}
