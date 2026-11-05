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

//! Regression tests for: Aggregate + IN subquery fails when predicate column is indexed/PK.
//!
//! The bug: When the WHERE column used with IN (subquery) has an index or is a PRIMARY KEY,
//! the query optimizer selected an index fast-path that cannot handle aggregate functions,
//! resulting in "Function not found: COUNT" errors.

use std::sync::atomic::{AtomicUsize, Ordering};
use stoolap::Database;

static TEST_ID: AtomicUsize = AtomicUsize::new(0);

fn setup_db() -> Database {
    let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
    let db = Database::open(&format!("memory://agg_subquery_{}", id))
        .expect("Failed to create database");
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, r TEXT)", ())
        .expect("Failed to create table p");
    db.execute(
        "CREATE TABLE plain5 (account_id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .expect("Failed to create table plain5");
    db.execute("INSERT INTO p VALUES (1, 'x'), (2, 'y'), (3, 'x')", ())
        .expect("Failed to insert into p");
    db.execute(
        "INSERT INTO plain5 VALUES (1, 10), (2, 20), (3, 30), (4, 40)",
        (),
    )
    .expect("Failed to insert into plain5");
    db
}

#[test]
fn test_count_with_in_subquery_on_pk() {
    let db = setup_db();
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM plain5 WHERE account_id IN (SELECT id FROM p WHERE r = 'x')",
            (),
        )
        .expect("COUNT with IN subquery on PK should work");
    // p where r='x' returns ids 1, 3; plain5 has account_id 1 and 3
    assert_eq!(count, 2);
}

#[test]
fn test_sum_with_in_subquery_on_pk() {
    let db = setup_db();
    let sum: i64 = db
        .query_one(
            "SELECT SUM(v) FROM plain5 WHERE account_id IN (SELECT id FROM p WHERE r = 'x')",
            (),
        )
        .expect("SUM with IN subquery on PK should work");
    // account_id 1 → v=10, account_id 3 → v=30
    assert_eq!(sum, 40);
}

#[test]
fn test_avg_with_in_subquery_on_pk() {
    let db = setup_db();
    let avg: f64 = db
        .query_one(
            "SELECT AVG(v) FROM plain5 WHERE account_id IN (SELECT id FROM p WHERE r = 'x')",
            (),
        )
        .expect("AVG with IN subquery on PK should work");
    // (10 + 30) / 2 = 20.0
    assert!((avg - 20.0).abs() < 0.001);
}

#[test]
fn test_min_max_with_in_subquery_on_pk() {
    let db = setup_db();
    let min: i64 = db
        .query_one(
            "SELECT MIN(v) FROM plain5 WHERE account_id IN (SELECT id FROM p WHERE r = 'x')",
            (),
        )
        .expect("MIN with IN subquery on PK should work");
    assert_eq!(min, 10);

    let max: i64 = db
        .query_one(
            "SELECT MAX(v) FROM plain5 WHERE account_id IN (SELECT id FROM p WHERE r = 'x')",
            (),
        )
        .expect("MAX with IN subquery on PK should work");
    assert_eq!(max, 30);
}

#[test]
fn test_count_with_in_subquery_on_indexed_column() {
    let db = Database::open("memory://agg_in_idx").expect("Failed to create database");
    db.execute(
        "CREATE TABLE lookup (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .expect("create");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, lookup_id INTEGER, amount FLOAT)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_lookup ON data(lookup_id)", ())
        .expect("create index");
    db.execute("INSERT INTO lookup VALUES (1, 'A'), (2, 'B'), (3, 'A')", ())
        .expect("insert");
    db.execute(
        "INSERT INTO data VALUES (1, 1, 100.0), (2, 2, 200.0), (3, 3, 300.0), (4, 1, 150.0)",
        (),
    )
    .expect("insert");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM data WHERE lookup_id IN (SELECT id FROM lookup WHERE category = 'A')",
            (),
        )
        .expect("COUNT with IN subquery on indexed column should work");
    // lookup_id IN (1, 3) → data rows with lookup_id 1 (2 rows) and 3 (1 row)
    assert_eq!(count, 3);
}

#[test]
fn test_aggregate_with_in_literal_list_on_pk() {
    let db = setup_db();
    // Also test IN list literals (not subquery) on indexed column
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM plain5 WHERE account_id IN (1, 3)", ())
        .expect("COUNT with IN literal list on PK should work");
    assert_eq!(count, 2);

    let sum: i64 = db
        .query_one("SELECT SUM(v) FROM plain5 WHERE account_id IN (1, 3)", ())
        .expect("SUM with IN literal list on PK should work");
    assert_eq!(sum, 40);
}

#[test]
fn test_group_by_with_in_subquery_on_pk() {
    let db = Database::open("memory://grp_in_pk").expect("Failed to create database");
    db.execute(
        "CREATE TABLE categories (id INTEGER PRIMARY KEY, active BOOLEAN)",
        (),
    )
    .expect("create");
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, cat_id INTEGER, price INTEGER)",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_cat ON items(cat_id)", ())
        .expect("create index");
    db.execute(
        "INSERT INTO categories VALUES (1, true), (2, false), (3, true)",
        (),
    )
    .expect("insert");
    db.execute(
        "INSERT INTO items VALUES (1, 1, 10), (2, 1, 20), (3, 2, 30), (4, 3, 40), (5, 3, 50)",
        (),
    )
    .expect("insert");

    // GROUP BY with IN subquery on indexed column
    let result = db
        .query(
            "SELECT cat_id, SUM(price) FROM items WHERE cat_id IN (SELECT id FROM categories WHERE active = true) GROUP BY cat_id ORDER BY cat_id",
            (),
        )
        .expect("GROUP BY with IN subquery on indexed column should work");
    let mut rows = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cat_id: i64 = row.get(0).unwrap();
        let sum: i64 = row.get(1).unwrap();
        rows.push((cat_id, sum));
    }
    assert_eq!(rows, vec![(1, 30), (3, 90)]);
}
