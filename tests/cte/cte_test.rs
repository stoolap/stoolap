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

//! CTE (WITH clause) Tests
//!
//! Tests Common Table Expressions (WITH clause)

use stoolap::Database;

fn setup_cte_tables(db: &Database) {
    // Create employees table
    db.execute(
        "CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary FLOAT
        )",
        (),
    )
    .expect("Failed to create employees table");

    // Insert test data
    db.execute(
        "INSERT INTO employees (id, name, department, salary) VALUES
        (1, 'Alice', 'Engineering', 100000.0),
        (2, 'Bob', 'Engineering', 90000.0),
        (3, 'Charlie', 'Sales', 80000.0),
        (4, 'David', 'Sales', 75000.0),
        (5, 'Eve', 'HR', 70000.0)",
        (),
    )
    .expect("Failed to insert employees");
}

/// Test simple CTE
#[test]
fn test_simple_cte() {
    let db = Database::open("memory://cte_simple").expect("Failed to create database");
    setup_cte_tables(&db);

    let result = db
        .query(
            "WITH high_earners AS (
                SELECT * FROM employees WHERE salary > 85000
            )
            SELECT * FROM high_earners",
            (),
        )
        .expect("Failed to execute CTE query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let salary: f64 = row.get(3).unwrap();
        assert!(salary > 85000.0, "Expected salary > 85000, got {}", salary);
        count += 1;
    }

    assert_eq!(count, 2, "Expected 2 high earners");
}

/// Test CTE with WHERE clause
#[test]
fn test_cte_with_where() {
    let db = Database::open("memory://cte_where").expect("Failed to create database");
    setup_cte_tables(&db);

    let result = db
        .query(
            "WITH engineering_team AS (
                SELECT * FROM employees WHERE department = 'Engineering'
            )
            SELECT name, salary FROM engineering_team WHERE salary > 95000",
            (),
        )
        .expect("Failed to execute CTE query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let salary: f64 = row.get(1).unwrap();
        assert_eq!(name, "Alice", "Expected name to be Alice, got {}", name);
        assert_eq!(
            salary, 100000.0,
            "Expected salary to be 100000, got {}",
            salary
        );
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 result");
}

/// Test CTE with column selection and aggregation
#[test]
fn test_cte_with_aggregation() {
    let db = Database::open("memory://cte_agg").expect("Failed to create database");
    setup_cte_tables(&db);

    let result = db
        .query(
            "WITH dept_summary AS (
                SELECT department, COUNT(*) as emp_count, AVG(salary) as avg_salary
                FROM employees
                GROUP BY department
            )
            SELECT * FROM dept_summary WHERE emp_count > 1",
            (),
        )
        .expect("Failed to execute CTE query");

    let mut results: std::collections::HashMap<String, (i64, f64)> =
        std::collections::HashMap::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let department: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        let avg_salary: f64 = row.get(2).unwrap();
        results.insert(department, (count, avg_salary));
    }

    assert_eq!(results.len(), 2, "Expected 2 departments with > 1 employee");

    // Check Engineering department
    if let Some((count, avg)) = results.get("Engineering") {
        assert_eq!(*count, 2, "Expected Engineering count to be 2");
        assert_eq!(*avg, 95000.0, "Expected Engineering avg salary to be 95000");
    } else {
        panic!("Expected Engineering department");
    }

    // Check Sales department
    if let Some((count, avg)) = results.get("Sales") {
        assert_eq!(*count, 2, "Expected Sales count to be 2");
        assert_eq!(*avg, 77500.0, "Expected Sales avg salary to be 77500");
    } else {
        panic!("Expected Sales department");
    }
}

/// Test multiple CTEs
#[test]
fn test_multiple_ctes() {
    let db = Database::open("memory://cte_multiple").expect("Failed to create database");

    // Create employees table
    db.execute(
        "CREATE TABLE employees2 (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER,
            salary FLOAT
        )",
        (),
    )
    .expect("Failed to create employees table");

    // Create departments table
    db.execute(
        "CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT,
            budget FLOAT
        )",
        (),
    )
    .expect("Failed to create departments table");

    // Insert departments
    db.execute(
        "INSERT INTO departments (id, name, budget) VALUES
        (1, 'Engineering', 500000.0),
        (2, 'Sales', 300000.0),
        (3, 'HR', 200000.0)",
        (),
    )
    .expect("Failed to insert departments");

    // Insert employees
    db.execute(
        "INSERT INTO employees2 (id, name, department_id, salary) VALUES
        (1, 'Alice', 1, 100000.0),
        (2, 'Bob', 1, 90000.0),
        (3, 'Charlie', 2, 80000.0),
        (4, 'David', 2, 75000.0),
        (5, 'Eve', 3, 70000.0)",
        (),
    )
    .expect("Failed to insert employees");

    // Test multiple CTEs with JOIN
    let result = db
        .query(
            "WITH high_earners AS (
                SELECT * FROM employees2 WHERE salary > 80000
            ),
            big_departments AS (
                SELECT * FROM departments WHERE budget > 250000
            )
            SELECT e.name, e.salary, d.name as department
            FROM high_earners e
            JOIN big_departments d ON e.department_id = d.id",
            (),
        )
        .expect("Failed to execute multiple CTEs query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let salary: f64 = row.get(1).unwrap();
        let department: String = row.get(2).unwrap();
        assert!(salary > 80000.0, "Expected salary > 80000, got {}", salary);
        assert!(
            department == "Engineering" || department == "Sales",
            "Expected Engineering or Sales, got {}",
            department
        );
        count += 1;
    }

    assert_eq!(count, 2, "Expected 2 results");
}

/// Test CTE referencing another CTE
#[test]
fn test_cte_referencing_cte() {
    let db = Database::open("memory://cte_ref_cte").expect("Failed to create database");

    // Create employees table
    db.execute(
        "CREATE TABLE employees3 (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department_id INTEGER,
            salary FLOAT
        )",
        (),
    )
    .expect("Failed to create employees table");

    // Create departments table
    db.execute(
        "CREATE TABLE departments2 (
            id INTEGER PRIMARY KEY,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create departments table");

    // Insert departments
    db.execute(
        "INSERT INTO departments2 (id, name) VALUES
        (1, 'Engineering'),
        (2, 'Sales'),
        (3, 'HR')",
        (),
    )
    .expect("Failed to insert departments");

    // Insert employees
    db.execute(
        "INSERT INTO employees3 (id, name, department_id, salary) VALUES
        (1, 'Alice', 1, 100000.0),
        (2, 'Bob', 1, 90000.0),
        (3, 'Charlie', 2, 80000.0),
        (4, 'David', 2, 75000.0),
        (5, 'Eve', 3, 70000.0)",
        (),
    )
    .expect("Failed to insert employees");

    // Test CTE referencing another CTE
    let result = db
        .query(
            "WITH dept_stats AS (
                SELECT department_id, COUNT(*) as emp_count, AVG(salary) as avg_salary
                FROM employees3
                GROUP BY department_id
            ),
            expensive_depts AS (
                SELECT * FROM dept_stats WHERE avg_salary > 75000
            )
            SELECT d.name, e.emp_count, e.avg_salary
            FROM expensive_depts e
            JOIN departments2 d ON e.department_id = d.id",
            (),
        )
        .expect("Failed to execute CTE referencing CTE query");

    let mut results: std::collections::HashMap<String, (i64, f64)> =
        std::collections::HashMap::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let count: i64 = row.get(1).unwrap();
        let avg_salary: f64 = row.get(2).unwrap();
        results.insert(name, (count, avg_salary));
    }

    assert_eq!(results.len(), 2, "Expected 2 expensive departments");

    // Check Engineering department
    if let Some((count, avg)) = results.get("Engineering") {
        assert_eq!(*count, 2, "Expected Engineering count to be 2");
        assert_eq!(*avg, 95000.0, "Expected Engineering avg salary to be 95000");
    } else {
        panic!("Expected Engineering department");
    }

    // Check Sales department
    if let Some((count, avg)) = results.get("Sales") {
        assert_eq!(*count, 2, "Expected Sales count to be 2");
        assert_eq!(*avg, 77500.0, "Expected Sales avg salary to be 77500");
    } else {
        panic!("Expected Sales department");
    }
}

/// Test CTE with sales data
#[test]
fn test_cte_with_sales() {
    let db = Database::open("memory://cte_sales").expect("Failed to create database");

    // Create sales table
    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product TEXT,
            region TEXT,
            amount FLOAT,
            year INTEGER
        )",
        (),
    )
    .expect("Failed to create sales table");

    // Insert test data
    db.execute(
        "INSERT INTO sales (id, product, region, amount, year) VALUES
        (1, 'Widget', 'North', 1000.0, 2023),
        (2, 'Widget', 'South', 1500.0, 2023),
        (3, 'Gadget', 'North', 2000.0, 2023),
        (4, 'Gadget', 'South', 2500.0, 2023),
        (5, 'Widget', 'North', 1200.0, 2024),
        (6, 'Widget', 'South', 1800.0, 2024),
        (7, 'Gadget', 'North', 2200.0, 2024),
        (8, 'Gadget', 'South', 2800.0, 2024)",
        (),
    )
    .expect("Failed to insert sales");

    // Test CTE with aggregation
    let result = db
        .query(
            "WITH recent_sales AS (
                SELECT * FROM sales WHERE year = 2024
            )
            SELECT product, SUM(amount) as total
            FROM recent_sales
            GROUP BY product",
            (),
        )
        .expect("Failed to execute CTE query");

    let mut results: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let product: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        results.insert(product, total);
    }

    // Widget: 1200 + 1800 = 3000
    // Gadget: 2200 + 2800 = 5000
    if let Some(total) = results.get("Widget") {
        assert_eq!(*total, 3000.0, "Expected Widget total to be 3000");
    } else {
        panic!("Expected Widget in results");
    }

    if let Some(total) = results.get("Gadget") {
        assert_eq!(*total, 5000.0, "Expected Gadget total to be 5000");
    } else {
        panic!("Expected Gadget in results");
    }
}

/// Test CTE referenced in a subquery (Bug #20 regression test)
/// Previously failed with: "table not found: high_depts"
#[test]
fn test_cte_in_subquery() {
    let db = Database::open("memory://cte_in_subquery").expect("Failed to create database");

    setup_cte_tables(&db);

    // Test CTE referenced from IN subquery
    let result = db
        .query(
            "WITH high_depts AS (
                SELECT department FROM employees WHERE salary > 90000
            )
            SELECT name FROM employees
            WHERE department IN (SELECT department FROM high_depts)
            ORDER BY name",
            (),
        )
        .expect("Failed to execute CTE in subquery");

    let names: Vec<String> = result
        .into_iter()
        .map(|row| row.expect("Failed to get row").get::<String>(0).unwrap())
        .collect();

    // Engineering department has salary > 90000 (Alice=100000)
    // So all Engineering employees should be returned: Alice, Bob
    assert_eq!(names, vec!["Alice", "Bob"]);
}

/// Test CTE in FROM subquery
#[test]
fn test_cte_in_from_subquery() {
    let db = Database::open("memory://cte_from_subquery").expect("Failed to create database");

    setup_cte_tables(&db);

    // Test CTE referenced from FROM subquery
    let result = db
        .query(
            "WITH depts AS (
                SELECT DISTINCT department FROM employees WHERE salary > 80000
            )
            SELECT * FROM (SELECT * FROM depts) AS sub
            ORDER BY department",
            (),
        )
        .expect("Failed to execute CTE in FROM subquery");

    let departments: Vec<String> = result
        .into_iter()
        .map(|row| row.expect("Failed to get row").get::<String>(0).unwrap())
        .collect();

    // Employees with salary > 80000: Alice (Eng, 100K), Bob (Eng, 90K), Charlie (Sales, 80K)
    // Wait, Charlie is exactly 80K, not > 80K
    // So departments with salary > 80000: Engineering only (Alice=100K, Bob=90K)
    assert_eq!(departments, vec!["Engineering"]);
}
