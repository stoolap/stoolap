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

//! Integration tests for UNION, INTERSECT, and EXCEPT set operations

use stoolap::Database;

fn create_test_db(name: &str) -> Database {
    Database::open(&format!("memory://{}", name)).expect("Failed to create in-memory database")
}

// ============================================================================
// UNION Tests
// ============================================================================

#[test]
fn test_union_basic() {
    let db = create_test_db("union_basic");

    let result = db
        .query("SELECT 1 AS x UNION SELECT 2 UNION SELECT 3", ())
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    values.sort();
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_union_removes_duplicates() {
    let db = create_test_db("union_dedup");

    let result = db
        .query(
            "SELECT 1 AS x UNION SELECT 1 UNION SELECT 2 UNION SELECT 2 UNION SELECT 3",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    values.sort();
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_union_all_keeps_duplicates() {
    let db = create_test_db("union_all");

    let result = db
        .query("SELECT 1 AS x UNION ALL SELECT 1 UNION ALL SELECT 2", ())
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    values.sort();
    assert_eq!(values, vec![1, 1, 2]);
}

#[test]
fn test_union_with_tables() {
    let db = create_test_db("union_tables");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1, 'Alice'), (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (2, 'Bob'), (3, 'Charlie')", ())
        .unwrap();

    // UNION should remove duplicate Bob
    let result = db
        .query("SELECT name FROM t1 UNION SELECT name FROM t2", ())
        .unwrap();

    let mut names: Vec<String> = Vec::new();
    for row in result {
        let row = row.unwrap();
        names.push(row.get(0).unwrap());
    }

    names.sort();
    assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
}

#[test]
fn test_union_all_with_tables() {
    let db = create_test_db("union_all_tables");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1, 'Alice'), (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (2, 'Bob'), (3, 'Charlie')", ())
        .unwrap();

    // UNION ALL should keep duplicate Bob
    let result = db
        .query("SELECT name FROM t1 UNION ALL SELECT name FROM t2", ())
        .unwrap();

    let mut names: Vec<String> = Vec::new();
    for row in result {
        let row = row.unwrap();
        names.push(row.get(0).unwrap());
    }

    names.sort();
    assert_eq!(names, vec!["Alice", "Bob", "Bob", "Charlie"]);
}

#[test]
fn test_union_with_order_by() {
    let db = create_test_db("union_order");

    let result = db
        .query("SELECT 3 AS x UNION SELECT 1 UNION SELECT 2 ORDER BY x", ())
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    // ORDER BY applies to entire result
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_union_with_limit() {
    let db = create_test_db("union_limit");

    let result = db
        .query(
            "SELECT 1 AS x UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 LIMIT 2",
            (),
        )
        .unwrap();

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 2);
}

// ============================================================================
// INTERSECT Tests
// ============================================================================

#[test]
fn test_intersect_basic() {
    let db = create_test_db("intersect_basic");

    db.execute("CREATE TABLE t1 (x INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (x INTEGER PRIMARY KEY)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1), (2), (3)", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (2), (3), (4)", ())
        .unwrap();

    let result = db
        .query("SELECT x FROM t1 INTERSECT SELECT x FROM t2", ())
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    values.sort();
    assert_eq!(values, vec![2, 3]);
}

#[test]
fn test_intersect_no_common_elements() {
    let db = create_test_db("intersect_none");

    db.execute("CREATE TABLE t1 (x INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (x INTEGER PRIMARY KEY)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1), (2)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (3), (4)", ()).unwrap();

    let result = db
        .query("SELECT x FROM t1 INTERSECT SELECT x FROM t2", ())
        .unwrap();

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 0);
}

// ============================================================================
// EXCEPT Tests
// ============================================================================

#[test]
fn test_except_basic() {
    let db = create_test_db("except_basic");

    db.execute("CREATE TABLE t1 (x INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (x INTEGER PRIMARY KEY)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1), (2), (3)", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (2), (3), (4)", ())
        .unwrap();

    let result = db
        .query("SELECT x FROM t1 EXCEPT SELECT x FROM t2", ())
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![1]);
}

#[test]
fn test_except_removes_all_matching() {
    let db = create_test_db("except_all_match");

    db.execute("CREATE TABLE t1 (x INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (x INTEGER PRIMARY KEY)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1), (2), (3)", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (1), (2), (3)", ())
        .unwrap();

    let result = db
        .query("SELECT x FROM t1 EXCEPT SELECT x FROM t2", ())
        .unwrap();

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 0);
}

#[test]
fn test_except_order_matters() {
    let db = create_test_db("except_order");

    db.execute("CREATE TABLE t1 (x INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (x INTEGER PRIMARY KEY)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1), (2)", ()).unwrap();
    db.execute("INSERT INTO t2 VALUES (2), (3)", ()).unwrap();

    // t1 EXCEPT t2 = {1}
    let result1 = db
        .query("SELECT x FROM t1 EXCEPT SELECT x FROM t2", ())
        .unwrap();
    let mut values1: Vec<i64> = Vec::new();
    for row in result1 {
        let row = row.unwrap();
        values1.push(row.get(0).unwrap());
    }
    assert_eq!(values1, vec![1]);

    // t2 EXCEPT t1 = {3}
    let result2 = db
        .query("SELECT x FROM t2 EXCEPT SELECT x FROM t1", ())
        .unwrap();
    let mut values2: Vec<i64> = Vec::new();
    for row in result2 {
        let row = row.unwrap();
        values2.push(row.get(0).unwrap());
    }
    assert_eq!(values2, vec![3]);
}

// ============================================================================
// Combined Set Operations Tests
// ============================================================================

#[test]
fn test_multiple_unions() {
    let db = create_test_db("multi_union");

    let result = db
        .query(
            "SELECT 1 AS x UNION SELECT 2 UNION SELECT 3 UNION SELECT 4",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    values.sort();
    assert_eq!(values, vec![1, 2, 3, 4]);
}

#[test]
fn test_union_with_where_clause() {
    let db = create_test_db("union_where");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (4, 40), (5, 50), (6, 60)", ())
        .unwrap();

    let result = db
        .query(
            "SELECT val FROM t1 WHERE val > 15 UNION SELECT val FROM t2 WHERE val < 55",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    values.sort();
    assert_eq!(values, vec![20, 30, 40, 50]);
}

#[test]
fn test_union_multiple_columns() {
    let db = create_test_db("union_multi_col");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (2, 'Bob')", ()).unwrap();

    let result = db
        .query("SELECT id, name FROM t1 UNION SELECT id, name FROM t2", ())
        .unwrap();

    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        rows.push((id, name));
    }

    rows.sort_by_key(|(id, _)| *id);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], (1, "Alice".to_string()));
    assert_eq!(rows[1], (2, "Bob".to_string()));
}
