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

//! FromRow Tests - Row to Struct mapping

use stoolap::{named_params, Database, FromRow, Result, ResultRow};

// Simple user struct
struct User {
    id: i64,
    name: String,
}

impl FromRow for User {
    fn from_row(row: &ResultRow) -> Result<Self> {
        Ok(User {
            id: row.get(0)?,
            name: row.get(1)?,
        })
    }
}

// Struct with optional field
struct Product {
    id: i64,
    name: String,
    description: Option<String>,
    price: f64,
}

impl FromRow for Product {
    fn from_row(row: &ResultRow) -> Result<Self> {
        Ok(Product {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            price: row.get(3)?,
        })
    }
}

// Struct using column names
struct Employee {
    id: i64,
    name: String,
    department: String,
    active: bool,
}

impl FromRow for Employee {
    fn from_row(row: &ResultRow) -> Result<Self> {
        Ok(Employee {
            id: row.get_by_name("id")?,
            name: row.get_by_name("name")?,
            department: row.get_by_name("department")?,
            active: row.get_by_name("active")?,
        })
    }
}

#[test]
fn test_query_as_basic() {
    let db = Database::open("memory://from_row_basic").expect("Failed to create database");

    db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO users VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (3, 'Charlie')", ())
        .unwrap();

    let users: Vec<User> = db
        .query_as("SELECT id, name FROM users ORDER BY id", ())
        .expect("Failed to query_as");

    assert_eq!(users.len(), 3);
    assert_eq!(users[0].id, 1);
    assert_eq!(users[0].name, "Alice");
    assert_eq!(users[1].id, 2);
    assert_eq!(users[1].name, "Bob");
    assert_eq!(users[2].id, 3);
    assert_eq!(users[2].name, "Charlie");
}

#[test]
fn test_query_as_with_params() {
    let db = Database::open("memory://from_row_params").expect("Failed to create database");

    db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO users VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (3, 'Charlie')", ())
        .unwrap();

    let users: Vec<User> = db
        .query_as("SELECT id, name FROM users WHERE id > ?", (1,))
        .expect("Failed to query_as");

    assert_eq!(users.len(), 2);
    assert_eq!(users[0].name, "Bob");
    assert_eq!(users[1].name, "Charlie");
}

#[test]
fn test_query_as_with_optional_field() {
    let db = Database::open("memory://from_row_optional").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER, name TEXT, description TEXT, price FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products VALUES (1, 'Apple', 'A red fruit', 1.50)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO products VALUES (2, 'Banana', NULL, 0.75)", ())
        .unwrap();

    let products: Vec<Product> = db
        .query_as(
            "SELECT id, name, description, price FROM products ORDER BY id",
            (),
        )
        .expect("Failed to query_as");

    assert_eq!(products.len(), 2);

    assert_eq!(products[0].id, 1);
    assert_eq!(products[0].name, "Apple");
    assert_eq!(products[0].description, Some("A red fruit".to_string()));
    assert!((products[0].price - 1.50).abs() < 0.01);

    assert_eq!(products[1].id, 2);
    assert_eq!(products[1].name, "Banana");
    assert_eq!(products[1].description, None);
    assert!((products[1].price - 0.75).abs() < 0.01);
}

#[test]
fn test_query_as_by_column_name() {
    let db = Database::open("memory://from_row_by_name").expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO employees VALUES (1, 'Alice', 'Engineering', true)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO employees VALUES (2, 'Bob', 'Sales', false)",
        (),
    )
    .unwrap();

    // Select columns in different order - FromRow uses column names
    let employees: Vec<Employee> = db
        .query_as(
            "SELECT department, name, active, id FROM employees ORDER BY id",
            (),
        )
        .expect("Failed to query_as");

    assert_eq!(employees.len(), 2);
    assert_eq!(employees[0].id, 1);
    assert_eq!(employees[0].name, "Alice");
    assert_eq!(employees[0].department, "Engineering");
    assert!(employees[0].active);

    assert_eq!(employees[1].id, 2);
    assert_eq!(employees[1].name, "Bob");
    assert_eq!(employees[1].department, "Sales");
    assert!(!employees[1].active);
}

#[test]
fn test_query_as_named() {
    let db = Database::open("memory://from_row_named").expect("Failed to create database");

    db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO users VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (3, 'Charlie')", ())
        .unwrap();

    let users: Vec<User> = db
        .query_as_named(
            "SELECT id, name FROM users WHERE id >= :min_id ORDER BY id",
            named_params! { min_id: 2 },
        )
        .expect("Failed to query_as_named");

    assert_eq!(users.len(), 2);
    assert_eq!(users[0].name, "Bob");
    assert_eq!(users[1].name, "Charlie");
}

#[test]
fn test_query_as_empty_result() {
    let db = Database::open("memory://from_row_empty").expect("Failed to create database");

    db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    let users: Vec<User> = db
        .query_as("SELECT id, name FROM users", ())
        .expect("Failed to query_as");

    assert!(users.is_empty());
}

#[test]
fn test_query_as_single_row() {
    let db = Database::open("memory://from_row_single").expect("Failed to create database");

    db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO users VALUES (42, 'TheOne')", ())
        .unwrap();

    let users: Vec<User> = db
        .query_as("SELECT id, name FROM users WHERE id = ?", (42,))
        .expect("Failed to query_as");

    assert_eq!(users.len(), 1);
    assert_eq!(users[0].id, 42);
    assert_eq!(users[0].name, "TheOne");
}

// Test with all supported types
struct AllTypes {
    int_val: i64,
    float_val: f64,
    text_val: String,
    bool_val: bool,
}

impl FromRow for AllTypes {
    fn from_row(row: &ResultRow) -> Result<Self> {
        Ok(AllTypes {
            int_val: row.get(0)?,
            float_val: row.get(1)?,
            text_val: row.get(2)?,
            bool_val: row.get(3)?,
        })
    }
}

#[test]
fn test_query_as_all_types() {
    let db = Database::open("memory://from_row_types").expect("Failed to create database");

    db.execute(
        "CREATE TABLE all_types (int_val INTEGER, float_val FLOAT, text_val TEXT, bool_val BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO all_types VALUES (42, 3.5, 'hello', true)", ())
        .unwrap();

    let rows: Vec<AllTypes> = db
        .query_as(
            "SELECT int_val, float_val, text_val, bool_val FROM all_types",
            (),
        )
        .expect("Failed to query_as");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].int_val, 42);
    assert!((rows[0].float_val - 3.5).abs() < 0.01);
    assert_eq!(rows[0].text_val, "hello");
    assert!(rows[0].bool_val);
}
