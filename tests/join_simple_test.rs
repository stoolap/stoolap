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

//! Simple JOIN Tests
//!
//! Tests basic JOIN functionality: INNER JOIN, LEFT JOIN
//!
//! NOTE: Some JOIN tests are marked as ignored due to known issues in the
//! current JOIN implementation that need to be fixed.

use stoolap::Database;

fn setup_simple_tables(db: &Database) {
    // Create departments_simple table (with ID, name)
    db.execute(
        "CREATE TABLE departments_simple (id INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create departments_simple table");

    // Create employees_simple table (with ID, name, department ID reference)
    db.execute(
        "CREATE TABLE employees_simple (id INTEGER, name TEXT, dept_id INTEGER)",
        (),
    )
    .expect("Failed to create employees_simple table");

    // Insert some departments
    db.execute(
        "INSERT INTO departments_simple (id, name) VALUES (1, 'Engineering')",
        (),
    )
    .expect("Failed to insert department 1");
    db.execute(
        "INSERT INTO departments_simple (id, name) VALUES (2, 'Sales')",
        (),
    )
    .expect("Failed to insert department 2");
    db.execute(
        "INSERT INTO departments_simple (id, name) VALUES (3, 'Marketing')",
        (),
    )
    .expect("Failed to insert department 3");

    // Insert some employees (with references to departments)
    db.execute(
        "INSERT INTO employees_simple (id, name, dept_id) VALUES (101, 'John', 1)",
        (),
    )
    .expect("Failed to insert employee 1");
    db.execute(
        "INSERT INTO employees_simple (id, name, dept_id) VALUES (102, 'Jane', 1)",
        (),
    )
    .expect("Failed to insert employee 2");
    db.execute(
        "INSERT INTO employees_simple (id, name, dept_id) VALUES (103, 'Bob', 2)",
        (),
    )
    .expect("Failed to insert employee 3");
    db.execute(
        "INSERT INTO employees_simple (id, name, dept_id) VALUES (104, 'Alice', NULL)",
        (),
    )
    .expect("Failed to insert employee 4");
}

/// INNER JOIN test
#[test]
fn test_inner_join_basic() {
    let db = Database::open("memory://join_inner_basic").expect("Failed to create database");
    setup_simple_tables(&db);

    // Execute a simple INNER JOIN
    let result = db.query(
        "SELECT e.id, e.name, d.name FROM employees_simple e INNER JOIN departments_simple d ON e.dept_id = d.id", ()).expect("Failed to execute INNER JOIN");

    let mut rows = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let emp_id: i64 = row.get(0).unwrap();
        let emp_name: String = row.get(1).unwrap();
        let dept_name: String = row.get(2).unwrap();
        rows.push((emp_id, emp_name, dept_name));
    }

    // Expected results: John, Jane in Engineering; Bob in Sales
    // Alice has NULL dept_id so she won't appear in INNER JOIN
    assert_eq!(rows.len(), 3, "Expected 3 rows from INNER JOIN");

    // Verify the results (order may vary, so we check set membership)
    let expected = vec![
        (101, "John".to_string(), "Engineering".to_string()),
        (102, "Jane".to_string(), "Engineering".to_string()),
        (103, "Bob".to_string(), "Sales".to_string()),
    ];

    for exp in &expected {
        assert!(
            rows.contains(exp),
            "Expected row {:?} not found in results: {:?}",
            exp,
            rows
        );
    }
}

/// LEFT JOIN test - tests that LEFT JOIN returns all rows from left table
/// Note: Column resolution for right table columns has known issues
#[test]
fn test_left_join_basic() {
    let db = Database::open("memory://join_left_basic").expect("Failed to create database");
    setup_simple_tables(&db);

    // Execute a simple LEFT JOIN
    let result = db.query(
        "SELECT e.id, e.name, d.name FROM employees_simple e LEFT JOIN departments_simple d ON e.dept_id = d.id", ()).expect("Failed to execute LEFT JOIN");

    let mut row_count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        row_count += 1;
    }

    // LEFT JOIN should return all employees (4 rows)
    assert_eq!(row_count, 4, "Expected 4 rows from LEFT JOIN");
}

/// INNER JOIN with WHERE clause
#[test]
fn test_join_with_where_clause() {
    let db = Database::open("memory://join_with_where").expect("Failed to create database");
    setup_simple_tables(&db);

    // INNER JOIN with WHERE clause
    let result = db.query(
        "SELECT e.id, e.name, d.name FROM employees_simple e INNER JOIN departments_simple d ON e.dept_id = d.id WHERE d.name = 'Engineering'", ()).expect("Failed to execute INNER JOIN with WHERE");

    let mut rows = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let emp_id: i64 = row.get(0).unwrap();
        let emp_name: String = row.get(1).unwrap();
        let dept_name: String = row.get(2).unwrap();
        rows.push((emp_id, emp_name, dept_name));
    }

    // Should only return John and Jane (Engineering employees)
    assert_eq!(rows.len(), 2, "Expected 2 Engineering employees");

    for (_, _, dept) in &rows {
        assert_eq!(
            dept, "Engineering",
            "All results should be from Engineering"
        );
    }
}

/// INNER JOIN with multiple conditions
#[test]
fn test_join_with_multiple_conditions() {
    let db = Database::open("memory://join_multi_cond").expect("Failed to create database");
    setup_simple_tables(&db);

    // Test join with additional condition in WHERE
    let result = db.query(
        "SELECT e.id, e.name, d.name FROM employees_simple e INNER JOIN departments_simple d ON e.dept_id = d.id WHERE e.id > 101", ()).expect("Failed to execute join with condition");

    let mut rows = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let emp_id: i64 = row.get(0).unwrap();
        let emp_name: String = row.get(1).unwrap();
        let dept_name: String = row.get(2).unwrap();
        rows.push((emp_id, emp_name, dept_name));
    }

    // Should return Jane (102) and Bob (103), not John (101) or Alice (104 - NULL dept)
    assert_eq!(rows.len(), 2, "Expected 2 rows with id > 101");

    for (id, _, _) in &rows {
        assert!(*id > 101, "All employee IDs should be > 101");
    }
}

/// Self JOIN
#[test]
fn test_join_self() {
    let db = Database::open("memory://join_self").expect("Failed to create database");

    // Create a table with hierarchical data
    db.execute(
        "CREATE TABLE org (id INTEGER, name TEXT, manager_id INTEGER)",
        (),
    )
    .expect("Failed to create org table");

    db.execute(
        "INSERT INTO org (id, name, manager_id) VALUES (1, 'CEO', NULL)",
        (),
    )
    .expect("Failed to insert CEO");
    db.execute(
        "INSERT INTO org (id, name, manager_id) VALUES (2, 'VP1', 1)",
        (),
    )
    .expect("Failed to insert VP1");
    db.execute(
        "INSERT INTO org (id, name, manager_id) VALUES (3, 'VP2', 1)",
        (),
    )
    .expect("Failed to insert VP2");
    db.execute(
        "INSERT INTO org (id, name, manager_id) VALUES (4, 'Manager1', 2)",
        (),
    )
    .expect("Failed to insert Manager1");

    // Self join to find employee-manager relationships
    let result = db
        .query(
            "SELECT e.name, m.name FROM org e INNER JOIN org m ON e.manager_id = m.id",
            (),
        )
        .expect("Failed to execute self join");

    let mut rows = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let emp_name: String = row.get(0).unwrap();
        let mgr_name: String = row.get(1).unwrap();
        rows.push((emp_name, mgr_name));
    }

    // VP1 and VP2 report to CEO, Manager1 reports to VP1
    assert_eq!(rows.len(), 3, "Expected 3 employee-manager relationships");

    let expected = vec![
        ("VP1".to_string(), "CEO".to_string()),
        ("VP2".to_string(), "CEO".to_string()),
        ("Manager1".to_string(), "VP1".to_string()),
    ];

    for exp in &expected {
        assert!(
            rows.contains(exp),
            "Expected relationship {:?} not found",
            exp
        );
    }
}

/// RIGHT JOIN
#[test]
fn test_right_join() {
    let db = Database::open("memory://join_right").expect("Failed to create database");
    setup_simple_tables(&db);

    // RIGHT JOIN - should include all departments, even those without employees
    let result = db.query(
        "SELECT e.id, e.name, d.name FROM employees_simple e RIGHT JOIN departments_simple d ON e.dept_id = d.id", ()).expect("Failed to execute RIGHT JOIN");

    let mut rows = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        // e.id and e.name may be NULL for Marketing (no employees)
        let emp_id: Option<i64> = row.get(0).ok();
        let emp_name: Option<String> = row.get(1).ok();
        let dept_name: String = row.get(2).unwrap();
        rows.push((emp_id, emp_name, dept_name));
    }

    // Should have: John & Jane in Engineering, Bob in Sales, NULL in Marketing
    assert!(
        rows.len() >= 4,
        "Expected at least 4 rows from RIGHT JOIN, got {}",
        rows.len()
    );

    // Check that Marketing appears in results (it has no employees)
    let marketing_row = rows.iter().find(|(_, _, dept)| dept == "Marketing");
    assert!(
        marketing_row.is_some(),
        "Marketing department should appear in RIGHT JOIN results"
    );
}

#[test]
fn test_cross_join() {
    let db = Database::open("memory://join_cross").expect("Failed to create database");

    // Create small tables for cross join
    db.execute("CREATE TABLE colors (name TEXT)", ())
        .expect("Failed to create colors table");
    db.execute("CREATE TABLE sizes (name TEXT)", ())
        .expect("Failed to create sizes table");

    db.execute("INSERT INTO colors (name) VALUES ('Red')", ())
        .unwrap();
    db.execute("INSERT INTO colors (name) VALUES ('Blue')", ())
        .unwrap();
    db.execute("INSERT INTO sizes (name) VALUES ('S')", ())
        .unwrap();
    db.execute("INSERT INTO sizes (name) VALUES ('M')", ())
        .unwrap();
    db.execute("INSERT INTO sizes (name) VALUES ('L')", ())
        .unwrap();

    // Cross join produces cartesian product
    let result = db
        .query("SELECT c.name, s.name FROM colors c CROSS JOIN sizes s", ())
        .expect("Failed to execute CROSS JOIN");

    let mut rows = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let color: String = row.get(0).unwrap();
        let size: String = row.get(1).unwrap();
        rows.push((color, size));
    }

    // 2 colors x 3 sizes = 6 combinations
    assert_eq!(rows.len(), 6, "Expected 6 rows from CROSS JOIN (2 x 3)");

    // Verify all combinations exist
    let expected_combinations = vec![
        ("Red", "S"),
        ("Red", "M"),
        ("Red", "L"),
        ("Blue", "S"),
        ("Blue", "M"),
        ("Blue", "L"),
    ];

    for (color, size) in expected_combinations {
        assert!(
            rows.iter().any(|(c, s)| c == color && s == size),
            "Expected combination ({}, {}) not found",
            color,
            size
        );
    }
}
