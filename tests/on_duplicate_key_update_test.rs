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

//! ON DUPLICATE KEY UPDATE Tests (UPSERT)
//!
//! Tests INSERT ... ON DUPLICATE KEY UPDATE functionality

use stoolap::Database;

fn setup_users_table(db: &Database) {
    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT,
            age INTEGER
        )",
        (),
    )
    .expect("Failed to create users table");

    // Create a unique index on username
    db.execute("CREATE UNIQUE INDEX idx_username ON users (username)", ())
        .expect("Failed to create unique index");
}

/// Test basic insert
#[test]
fn test_basic_insert() {
    let db = Database::open("memory://upsert_basic").expect("Failed to create database");
    setup_users_table(&db);

    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES (1, 'user1', 'user1@example.com', 25)",
        (),
    )
    .expect("Failed to insert first row");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM users", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row");
}

/// Test ON DUPLICATE KEY UPDATE with primary key violation
#[test]
fn test_on_duplicate_key_primary_key() {
    let db = Database::open("memory://upsert_pk").expect("Failed to create database");
    setup_users_table(&db);

    // Insert initial row
    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES (1, 'user1', 'user1@example.com', 25)",
        (),
    )
    .expect("Failed to insert first row");

    // Try to insert with same primary key - should update instead
    db.execute(
        "INSERT INTO users (id, username, email, age)
         VALUES (1, 'different_user', 'new_email@example.com', 40)
         ON DUPLICATE KEY UPDATE
         username = 'primary_key_updated',
         email = 'pk_updated@example.com',
         age = 45",
        (),
    )
    .expect("Failed to update with ON DUPLICATE KEY");

    // Verify the update worked
    let result = db
        .query(
            "SELECT id, username, email, age FROM users WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let username: String = row.get(1).unwrap();
        let email: String = row.get(2).unwrap();
        let age: i64 = row.get(3).unwrap();

        assert_eq!(id, 1);
        assert_eq!(username, "primary_key_updated");
        assert_eq!(email, "pk_updated@example.com");
        assert_eq!(age, 45);
        found = true;
    }

    assert!(found, "Expected to find updated row");

    // Verify only 1 row exists
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM users", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 row after upsert");
}

/// Test ON DUPLICATE KEY UPDATE with unique constraint violation
#[test]
fn test_on_duplicate_key_unique_constraint() {
    let db = Database::open("memory://upsert_unique").expect("Failed to create database");
    setup_users_table(&db);

    // Insert two rows
    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES (1, 'user1', 'user1@example.com', 25)",
        (),
    )
    .expect("Failed to insert user1");
    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES (2, 'user2', 'user2@example.com', 30)",
        (),
    )
    .expect("Failed to insert user2");

    // Try to insert with different id but same username (unique constraint violation)
    db.execute(
        "INSERT INTO users (id, username, email, age)
         VALUES (999, 'user2', 'different_email@example.com', 50)
         ON DUPLICATE KEY UPDATE
         id = 2,
         email = 'updated_by_unique_constraint@example.com',
         age = 55",
        (),
    )
    .expect("Failed to update with ON DUPLICATE KEY for unique constraint");

    // Verify user2 was updated
    let result = db
        .query(
            "SELECT id, username, email, age FROM users WHERE username = 'user2'",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let email: String = row.get(2).unwrap();
        let age: i64 = row.get(3).unwrap();

        assert_eq!(id, 2);
        assert_eq!(email, "updated_by_unique_constraint@example.com");
        assert_eq!(age, 55);
    }

    // Verify still only 2 rows exist
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM users", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows after upsert");
}

/// Test ON DUPLICATE KEY UPDATE with no conflict (normal insert)
#[test]
fn test_on_duplicate_key_no_conflict() {
    let db = Database::open("memory://upsert_no_conflict").expect("Failed to create database");
    setup_users_table(&db);

    // Insert initial row
    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES (1, 'user1', 'user1@example.com', 25)",
        (),
    )
    .expect("Failed to insert first row");

    // Insert with ON DUPLICATE KEY but no conflict - should just insert
    db.execute(
        "INSERT INTO users (id, username, email, age)
         VALUES (2, 'user2', 'user2@example.com', 30)
         ON DUPLICATE KEY UPDATE
         email = 'not_used@example.com'",
        (),
    )
    .expect("Failed to insert new row with ON DUPLICATE KEY");

    // Verify we have 2 rows
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM users", ())
        .expect("Failed to count");
    assert_eq!(count, 2, "Expected 2 rows");

    // Verify user2 has original email (not the UPDATE clause)
    let email: String = db
        .query_one("SELECT email FROM users WHERE id = 2", ())
        .expect("Failed to query email");
    assert_eq!(email, "user2@example.com", "Email should be original value");
}

/// Test ON DUPLICATE KEY UPDATE with products table (unique constraint on code)
#[test]
fn test_on_duplicate_key_with_unique_index() {
    let db = Database::open("memory://upsert_products").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            name TEXT,
            price FLOAT
        )",
        (),
    )
    .expect("Failed to create products table");

    db.execute("CREATE UNIQUE INDEX idx_code ON products (code)", ())
        .expect("Failed to create unique index");

    // Insert initial product
    db.execute("INSERT INTO products (id, code, name, price) VALUES (1, 'PROD-001', 'Initial Product', 19.99)", ())
        .expect("Failed to insert product");

    // Try to insert with same code (unique constraint) - should update
    db.execute(
        "INSERT INTO products (id, code, name, price)
         VALUES (999, 'PROD-001', 'Duplicate Code Product', 29.99)
         ON DUPLICATE KEY UPDATE
         name = 'Updated Product',
         price = 39.99",
        (),
    )
    .expect("Failed to update product with ON DUPLICATE KEY");

    // Verify the product was updated
    let result = db
        .query(
            "SELECT id, code, name, price FROM products WHERE code = 'PROD-001'",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(2).unwrap();
        let price: f64 = row.get(3).unwrap();

        assert_eq!(id, 1, "Original ID should be preserved");
        assert_eq!(name, "Updated Product");
        assert_eq!(price, 39.99);
    }

    // Verify only 1 product exists
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM products", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Expected 1 product after upsert");
}

/// Test multiple upserts in sequence
#[test]
fn test_multiple_upserts() {
    let db = Database::open("memory://upsert_multiple").expect("Failed to create database");

    db.execute(
        "CREATE TABLE counters (
            id INTEGER PRIMARY KEY,
            name TEXT,
            count INTEGER
        )",
        (),
    )
    .expect("Failed to create counters table");

    // First insert
    db.execute(
        "INSERT INTO counters (id, name, count) VALUES (1, 'visits', 1)
         ON DUPLICATE KEY UPDATE count = count + 1",
        (),
    )
    .expect("Failed first upsert");

    // Second upsert (should update)
    db.execute(
        "INSERT INTO counters (id, name, count) VALUES (1, 'visits', 1)
         ON DUPLICATE KEY UPDATE count = count + 1",
        (),
    )
    .expect("Failed second upsert");

    // Third upsert (should update again)
    db.execute(
        "INSERT INTO counters (id, name, count) VALUES (1, 'visits', 1)
         ON DUPLICATE KEY UPDATE count = count + 1",
        (),
    )
    .expect("Failed third upsert");

    // Verify count is 3
    let count: i64 = db
        .query_one("SELECT count FROM counters WHERE id = 1", ())
        .expect("Failed to query count");
    assert_eq!(count, 3, "Expected count to be 3 after 3 upserts");
}

/// Test that we get correct row count
#[test]
fn test_row_count_after_upserts() {
    let db = Database::open("memory://upsert_rowcount").expect("Failed to create database");
    setup_users_table(&db);

    // Insert 3 users
    for i in 1..=3 {
        db.execute(
            &format!(
                "INSERT INTO users (id, username, email, age)
             VALUES ({}, 'user{}', 'user{}@example.com', {})
             ON DUPLICATE KEY UPDATE age = age + 1",
                i,
                i,
                i,
                20 + i
            ),
            (),
        )
        .expect("Failed to insert user");
    }

    // Verify 3 distinct users
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM users", ())
        .expect("Failed to count");
    assert_eq!(count, 3, "Expected 3 users");

    // Get all usernames
    let result = db
        .query("SELECT username FROM users ORDER BY id", ())
        .expect("Failed to query");

    let mut usernames: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let username: String = row.get(0).unwrap();
        usernames.push(username);
    }

    assert_eq!(usernames.len(), 3, "Expected 3 distinct usernames");
}

#[test]
fn test_on_duplicate_key_update_positional_params() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_users_table(&db);

    // Insert initial row
    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES (1, 'alice', 'alice@example.com', 20)",
        (),
    )
    .unwrap();

    // Upsert with positional params — conflict on id=1 triggers UPDATE branch
    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES ($1, $2, $3, $4)
         ON DUPLICATE KEY UPDATE username = $5, age = $6",
        (
            1_i64,
            "ignored",
            "ignored@example.com",
            0_i64,
            "bob",
            30_i64,
        ),
    )
    .unwrap();

    let name: String = db
        .query_one("SELECT username FROM users WHERE id = 1", ())
        .expect("Failed to query username");
    let age: i64 = db
        .query_one("SELECT age FROM users WHERE id = 1", ())
        .expect("Failed to query age");

    assert_eq!(
        name, "bob",
        "Username should be updated via positional param"
    );
    assert_eq!(age, 30, "Age should be updated via positional param");
}

#[test]
fn test_on_duplicate_key_update_named_params() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_users_table(&db);

    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES (1, 'alice', 'alice@example.com', 20)",
        (),
    )
    .unwrap();

    // Upsert with named params
    db.execute_named(
        "INSERT INTO users (id, username, email, age) VALUES (:id, :ins_name, :ins_email, :ins_age)
         ON DUPLICATE KEY UPDATE username = :new_name, age = :new_age",
        stoolap::named_params! {
            id: 1_i64,
            ins_name: "ignored",
            ins_email: "ignored@example.com",
            ins_age: 0_i64,
            new_name: "charlie",
            new_age: 40_i64
        },
    )
    .unwrap();

    let name: String = db
        .query_one("SELECT username FROM users WHERE id = 1", ())
        .expect("Failed to query username");
    let age: i64 = db
        .query_one("SELECT age FROM users WHERE id = 1", ())
        .expect("Failed to query age");

    assert_eq!(
        name, "charlie",
        "Username should be updated via named param"
    );
    assert_eq!(age, 40, "Age should be updated via named param");
}

#[test]
fn test_on_duplicate_key_update_mixed_literal_and_params() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_users_table(&db);

    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES (1, 'alice', 'alice@example.com', 20)",
        (),
    )
    .unwrap();

    // Mix of literal and param in UPDATE clause
    db.execute(
        "INSERT INTO users (id, username, email, age) VALUES ($1, $2, $3, $4)
         ON DUPLICATE KEY UPDATE username = 'literal_name', age = $5",
        (1_i64, "ignored", "ignored@example.com", 0_i64, 99_i64),
    )
    .unwrap();

    let name: String = db
        .query_one("SELECT username FROM users WHERE id = 1", ())
        .expect("Failed to query username");
    let age: i64 = db
        .query_one("SELECT age FROM users WHERE id = 1", ())
        .expect("Failed to query age");

    assert_eq!(
        name, "literal_name",
        "Username should be updated via literal"
    );
    assert_eq!(age, 99, "Age should be updated via positional param");
}
