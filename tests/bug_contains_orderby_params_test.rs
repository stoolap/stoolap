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

//! Regression test for Bug: CONTAINS() + ORDER BY with parameterized argument returns 0 rows.
//!
//! Root cause: parallel_filter() and sequential_filter() in parallel.rs created
//! RowFilter/ExpressionEval WITHOUT passing ExecutionContext, so query parameters
//! ($1, $2, ...) evaluated to NULL in the batch (ORDER BY) path.

use stoolap::Database;

fn setup_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).unwrap();
    db.execute(
        "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, body TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO articles VALUES (1, 'Rust Guide', 'Learn Rust programming language')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO articles VALUES (2, 'SQL Tips', 'Advanced SQL query optimization')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO articles VALUES (3, 'Web Dev', 'Building web applications with Rust')",
        (),
    )
    .unwrap();
    db
}

#[test]
fn test_contains_with_param_no_order_by() {
    let db = setup_db("contains_no_order");
    let mut rows = db
        .query(
            "SELECT id, title FROM articles WHERE CONTAINS(body, $1)",
            ("Rust",),
        )
        .unwrap();
    let mut count = 0;
    while let Some(Ok(_)) = rows.next() {
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
fn test_contains_with_param_order_by() {
    let db = setup_db("contains_order_by");
    let mut rows = db
        .query(
            "SELECT id, title FROM articles WHERE CONTAINS(body, $1) ORDER BY id",
            ("Rust",),
        )
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![1, 3]);
}

#[test]
fn test_contains_with_param_order_by_desc() {
    let db = setup_db("contains_order_desc");
    let mut rows = db
        .query(
            "SELECT id, title FROM articles WHERE CONTAINS(body, $1) ORDER BY id DESC",
            ("Rust",),
        )
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![3, 1]);
}

#[test]
fn test_contains_lower_with_param_order_by() {
    let db = setup_db("contains_lower_order");
    let mut rows = db
        .query(
            "SELECT id FROM articles WHERE CONTAINS(LOWER(body), LOWER($1)) ORDER BY id",
            ("rust",),
        )
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![1, 3]);
}

#[test]
fn test_function_with_param_order_by_no_match() {
    let db = setup_db("contains_no_match");
    let mut rows = db
        .query(
            "SELECT id FROM articles WHERE CONTAINS(body, $1) ORDER BY id",
            ("nonexistent",),
        )
        .unwrap();
    let mut count = 0;
    while let Some(Ok(_)) = rows.next() {
        count += 1;
    }
    assert_eq!(count, 0);
}

#[test]
fn test_equality_with_param_order_by() {
    let db = setup_db("eq_param_order");
    let mut rows = db
        .query(
            "SELECT id, title FROM articles WHERE title = $1 ORDER BY id",
            ("Rust Guide",),
        )
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![1]);
}

#[test]
fn test_multiple_params_with_order_by() {
    let db = setup_db("multi_params_order");
    let mut rows = db
        .query(
            "SELECT id FROM articles WHERE CONTAINS(body, $1) AND id > $2 ORDER BY id",
            ("Rust", 1i64),
        )
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![3]);
}
