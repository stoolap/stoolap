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

//! VIEW Tests
//!
//! Tests for CREATE VIEW, DROP VIEW, and querying views

use stoolap::Database;

fn setup_base_table(db: &Database) {
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

/// Test CREATE VIEW basic functionality
#[test]
fn test_create_view_basic() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a simple view
    db.execute(
        "CREATE VIEW engineering_employees AS SELECT * FROM employees WHERE department = 'Engineering'",
        (),
    )
    .expect("Failed to create view");

    // Query the view
    let rows = db
        .query("SELECT * FROM engineering_employees ORDER BY id", ())
        .expect("Failed to query view");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
    assert_eq!(results[1].get_by_name::<String>("name").unwrap(), "Bob");
}

/// Test CREATE VIEW IF NOT EXISTS
#[test]
fn test_create_view_if_not_exists() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create view
    db.execute("CREATE VIEW my_view AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // CREATE VIEW IF NOT EXISTS should not fail
    db.execute(
        "CREATE VIEW IF NOT EXISTS my_view AS SELECT id FROM employees",
        (),
    )
    .expect("CREATE VIEW IF NOT EXISTS should not fail");

    // Original view should still work
    let rows = db
        .query("SELECT * FROM my_view LIMIT 1", ())
        .expect("Failed to query view");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);
}

/// Test DROP VIEW basic functionality
#[test]
fn test_drop_view_basic() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW temp_view AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Verify view works
    let rows = db
        .query("SELECT * FROM temp_view", ())
        .expect("Failed to query view");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 5);

    // Drop the view
    db.execute("DROP VIEW temp_view", ())
        .expect("Failed to drop view");

    // View should no longer exist
    let result = db.query("SELECT * FROM temp_view", ());
    assert!(result.is_err());
}

/// Test DROP VIEW IF EXISTS
#[test]
fn test_drop_view_if_exists() {
    let db = Database::open_in_memory().expect("Failed to open database");

    // DROP VIEW IF EXISTS on non-existent view should not fail
    db.execute("DROP VIEW IF EXISTS non_existent_view", ())
        .expect("DROP VIEW IF EXISTS should not fail");
}

/// Test view with column selection
#[test]
fn test_view_column_selection() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create view with specific columns
    db.execute(
        "CREATE VIEW employee_names AS SELECT id, name FROM employees",
        (),
    )
    .expect("Failed to create view");

    // Query specific column from view
    let rows = db
        .query("SELECT name FROM employee_names WHERE id = 1", ())
        .expect("Failed to query view");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
}

/// Test view with aggregation
#[test]
fn test_view_with_aggregation() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create view with aggregation
    db.execute(
        "CREATE VIEW dept_stats AS SELECT department, COUNT(*) as emp_count, AVG(salary) as avg_salary FROM employees GROUP BY department",
        (),
    )
    .expect("Failed to create view");

    // Query the view
    let rows = db
        .query("SELECT * FROM dept_stats ORDER BY department", ())
        .expect("Failed to query view");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 3);

    // Check Engineering department
    let eng = results
        .iter()
        .find(|r| r.get_by_name::<String>("department").unwrap() == "Engineering")
        .unwrap();
    assert_eq!(eng.get_by_name::<i64>("emp_count").unwrap(), 2);
}

/// Test view with WHERE clause on outer query
#[test]
fn test_view_with_outer_where() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view of all employees
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query with WHERE on the outer query
    let rows = db
        .query("SELECT * FROM all_employees WHERE salary > 80000", ())
        .expect("Failed to query view with WHERE");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2); // Alice (100000) and Bob (90000)
}

/// Test that view and table names cannot conflict
#[test]
fn test_view_table_name_conflict() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Try to create a view with the same name as existing table
    let result = db.execute("CREATE VIEW employees AS SELECT 1", ());
    assert!(result.is_err());
}

/// Test case-insensitive view names
#[test]
fn test_view_case_insensitive() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create view with mixed case
    db.execute("CREATE VIEW MyView AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query with different case
    let rows = db
        .query("SELECT * FROM myview", ())
        .expect("Failed to query view with different case");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 5);

    // Drop with different case
    db.execute("DROP VIEW MYVIEW", ())
        .expect("Failed to drop view with different case");
}

/// Test view with ORDER BY on outer query
#[test]
fn test_view_with_order_by() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query with ORDER BY ASC
    let rows = db
        .query("SELECT * FROM all_employees ORDER BY salary", ())
        .expect("Failed to query view with ORDER BY");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 5);
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Eve"); // 70000
    assert_eq!(results[4].get_by_name::<String>("name").unwrap(), "Alice"); // 100000

    // Query with ORDER BY DESC
    let rows = db
        .query("SELECT * FROM all_employees ORDER BY salary DESC", ())
        .expect("Failed to query view with ORDER BY DESC");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice"); // 100000
    assert_eq!(results[4].get_by_name::<String>("name").unwrap(), "Eve"); // 70000
}

/// Test view with LIMIT
#[test]
fn test_view_with_limit() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query with LIMIT
    let rows = db
        .query("SELECT * FROM all_employees ORDER BY id LIMIT 3", ())
        .expect("Failed to query view with LIMIT");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get_by_name::<i64>("id").unwrap(), 1);
    assert_eq!(results[2].get_by_name::<i64>("id").unwrap(), 3);
}

/// Test view with OFFSET
#[test]
fn test_view_with_offset() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query with LIMIT and OFFSET
    let rows = db
        .query(
            "SELECT * FROM all_employees ORDER BY id LIMIT 2 OFFSET 2",
            (),
        )
        .expect("Failed to query view with LIMIT OFFSET");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].get_by_name::<i64>("id").unwrap(), 3); // Charlie
    assert_eq!(results[1].get_by_name::<i64>("id").unwrap(), 4); // David
}

/// Test view with DISTINCT
#[test]
fn test_view_with_distinct() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute(
        "CREATE VIEW departments AS SELECT department FROM employees",
        (),
    )
    .expect("Failed to create view");

    // Query with DISTINCT
    let rows = db
        .query("SELECT DISTINCT * FROM departments ORDER BY department", ())
        .expect("Failed to query view with DISTINCT");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 3); // Engineering, HR, Sales
}

/// Test view in JOIN with a table
#[test]
fn test_view_in_join_with_table() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a projects table
    db.execute(
        "CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT
        )",
        (),
    )
    .expect("Failed to create projects table");

    db.execute(
        "INSERT INTO projects (id, name, department) VALUES
        (1, 'Project Alpha', 'Engineering'),
        (2, 'Project Beta', 'Sales')",
        (),
    )
    .expect("Failed to insert projects");

    // Create a view
    db.execute(
        "CREATE VIEW eng_employees AS SELECT id, name, department FROM employees WHERE department = 'Engineering'",
        (),
    )
    .expect("Failed to create view");

    // JOIN view with table
    let rows = db
        .query(
            "SELECT e.name, p.name FROM eng_employees e JOIN projects p ON e.department = p.department ORDER BY e.name",
            (),
        )
        .expect("Failed to query view in JOIN");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2); // Alice and Bob both match Project Alpha
}

/// Test view in JOIN with another view
#[test]
fn test_view_join_with_view() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create two views
    db.execute(
        "CREATE VIEW high_salary AS SELECT * FROM employees WHERE salary >= 90000",
        (),
    )
    .expect("Failed to create high_salary view");

    db.execute(
        "CREATE VIEW eng_dept AS SELECT * FROM employees WHERE department = 'Engineering'",
        (),
    )
    .expect("Failed to create eng_dept view");

    // JOIN two views (should get employees who are both high salary AND engineering)
    let rows = db
        .query(
            "SELECT h.name FROM high_salary h JOIN eng_dept e ON h.id = e.id ORDER BY h.name",
            (),
        )
        .expect("Failed to join two views");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2); // Alice (100000, Engineering) and Bob (90000, Engineering)
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
    assert_eq!(results[1].get_by_name::<String>("name").unwrap(), "Bob");
}

/// Test nested views (view referencing another view)
#[test]
fn test_nested_views() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create first view
    db.execute(
        "CREATE VIEW high_salary AS SELECT * FROM employees WHERE salary >= 80000",
        (),
    )
    .expect("Failed to create first view");

    // Create second view that references the first view
    db.execute(
        "CREATE VIEW high_salary_eng AS SELECT * FROM high_salary WHERE department = 'Engineering'",
        (),
    )
    .expect("Failed to create nested view");

    // Query the nested view
    let rows = db
        .query("SELECT * FROM high_salary_eng ORDER BY salary DESC", ())
        .expect("Failed to query nested view");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2); // Alice (100000) and Bob (90000)
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
    assert_eq!(results[1].get_by_name::<String>("name").unwrap(), "Bob");
}

/// Test view with JOIN in its definition
#[test]
fn test_view_with_join_definition() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a departments table
    db.execute(
        "CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT,
            budget FLOAT
        )",
        (),
    )
    .expect("Failed to create departments table");

    db.execute(
        "INSERT INTO departments (id, name, budget) VALUES
        (1, 'Engineering', 500000.0),
        (2, 'Sales', 300000.0),
        (3, 'HR', 200000.0)",
        (),
    )
    .expect("Failed to insert departments");

    // Create a view with JOIN
    db.execute(
        "CREATE VIEW employee_with_budget AS
         SELECT e.name, e.salary, d.budget
         FROM employees e
         JOIN departments d ON e.department = d.name",
        (),
    )
    .expect("Failed to create view with JOIN");

    // Query the view
    let rows = db
        .query(
            "SELECT * FROM employee_with_budget ORDER BY salary DESC LIMIT 3",
            (),
        )
        .expect("Failed to query view with JOIN definition");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
    assert_eq!(results[0].get_by_name::<f64>("budget").unwrap(), 500000.0);
}

/// Test that CREATE TABLE with same name as view fails
#[test]
fn test_create_table_conflicts_with_view() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW my_view AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Try to create a table with the same name - should fail
    let result = db.execute("CREATE TABLE my_view (id INTEGER PRIMARY KEY)", ());
    assert!(result.is_err());
}

/// Test view with multiple ORDER BY columns
#[test]
fn test_view_with_multiple_order_by() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query with multiple ORDER BY columns
    let rows = db
        .query(
            "SELECT * FROM all_employees ORDER BY department ASC, salary DESC",
            (),
        )
        .expect("Failed to query view with multiple ORDER BY");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 5);

    // Engineering first (Alice 100000, Bob 90000), then HR (Eve 70000), then Sales (Charlie 80000, David 75000)
    assert_eq!(
        results[0].get_by_name::<String>("department").unwrap(),
        "Engineering"
    );
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
    assert_eq!(
        results[1].get_by_name::<String>("department").unwrap(),
        "Engineering"
    );
    assert_eq!(results[1].get_by_name::<String>("name").unwrap(), "Bob");
}

/// Test view with WHERE and ORDER BY and LIMIT combined
#[test]
fn test_view_with_combined_clauses() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query with WHERE + ORDER BY + LIMIT
    let rows = db
        .query(
            "SELECT * FROM all_employees WHERE salary > 70000 ORDER BY salary DESC LIMIT 2",
            (),
        )
        .expect("Failed to query view with combined clauses");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice"); // 100000
    assert_eq!(results[1].get_by_name::<String>("name").unwrap(), "Bob"); // 90000
}

/// Test selecting specific columns from view with ORDER BY
#[test]
fn test_view_select_columns_with_order_by() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query specific columns with ORDER BY
    let rows = db
        .query(
            "SELECT name, salary FROM all_employees ORDER BY salary DESC LIMIT 3",
            (),
        )
        .expect("Failed to query view with column selection and ORDER BY");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
    assert_eq!(results[0].get_by_name::<f64>("salary").unwrap(), 100000.0);
}

/// Test view data reflects underlying table changes
#[test]
fn test_view_reflects_table_changes() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Check initial count
    let rows = db
        .query("SELECT * FROM all_employees", ())
        .expect("Failed to query view");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 5);

    // Insert a new employee
    db.execute(
        "INSERT INTO employees (id, name, department, salary) VALUES (6, 'Frank', 'Engineering', 85000.0)",
        (),
    )
    .expect("Failed to insert employee");

    // View should now have 6 rows
    let rows = db
        .query("SELECT * FROM all_employees", ())
        .expect("Failed to query view after insert");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 6);

    // Delete an employee
    db.execute("DELETE FROM employees WHERE id = 1", ())
        .expect("Failed to delete employee");

    // View should now have 5 rows
    let rows = db
        .query("SELECT * FROM all_employees", ())
        .expect("Failed to query view after delete");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 5);
}

/// Test querying non-existent view
#[test]
fn test_query_nonexistent_view() {
    let db = Database::open_in_memory().expect("Failed to open database");

    let result = db.query("SELECT * FROM nonexistent_view", ());
    assert!(result.is_err());
}

/// Test dropping view doesn't affect underlying table
#[test]
fn test_drop_view_preserves_table() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create and drop a view
    db.execute("CREATE VIEW temp AS SELECT * FROM employees", ())
        .expect("Failed to create view");
    db.execute("DROP VIEW temp", ())
        .expect("Failed to drop view");

    // Underlying table should still work
    let rows = db
        .query("SELECT * FROM employees", ())
        .expect("Failed to query table after dropping view");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 5);
}

/// Test that deeply nested views are detected and return an error
#[test]
fn test_view_depth_limit() {
    let db = Database::open_in_memory().expect("Failed to open database");

    // Create a base table
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, value INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t VALUES (1, 100)", ())
        .expect("Failed to insert");

    // Create a chain of 35 nested views (exceeds limit of 32)
    db.execute("CREATE VIEW v0 AS SELECT * FROM t", ())
        .expect("Failed to create v0");

    for i in 1..35 {
        let sql = format!("CREATE VIEW v{} AS SELECT * FROM v{}", i, i - 1);
        db.execute(&sql, ())
            .unwrap_or_else(|_| panic!("Failed to create v{}", i));
    }

    // Querying v34 should fail due to depth limit
    let result = db.query("SELECT * FROM v34", ());
    match result {
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("depth") || err_msg.contains("nesting"),
                "Error message should mention depth/nesting: {}",
                err_msg
            );
        }
        Ok(_) => panic!("Expected error due to view depth limit exceeded"),
    }

    // Querying a view within the limit should still work (v30 = 31 levels deep)
    let rows = db
        .query("SELECT * FROM v30", ())
        .expect("Failed to query v30");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get::<i64>(0).unwrap(), 1);
}

/// Test SHOW VIEWS command
#[test]
fn test_show_views() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Initially no views
    let rows = db
        .query("SHOW VIEWS", ())
        .expect("Failed to execute SHOW VIEWS");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 0);

    // Create some views
    db.execute("CREATE VIEW view_a AS SELECT * FROM employees", ())
        .expect("Failed to create view_a");
    db.execute(
        "CREATE VIEW view_b AS SELECT name, salary FROM employees WHERE salary > 80000",
        (),
    )
    .expect("Failed to create view_b");

    // SHOW VIEWS should list both
    let rows = db
        .query("SHOW VIEWS", ())
        .expect("Failed to execute SHOW VIEWS");
    // Check column name is view_name
    let column_names = rows.columns().to_vec();
    assert!(!column_names.is_empty() && column_names[0] == "view_name");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2);

    // View names should be present (order may vary)
    let view_names: Vec<String> = results
        .iter()
        .map(|r| r.get_by_name::<String>("view_name").unwrap())
        .collect();
    assert!(view_names.contains(&"view_a".to_string()));
    assert!(view_names.contains(&"view_b".to_string()));

    // Drop a view and check again
    db.execute("DROP VIEW view_a", ())
        .expect("Failed to drop view_a");
    let rows = db
        .query("SHOW VIEWS", ())
        .expect("Failed to execute SHOW VIEWS");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get_by_name::<String>("view_name").unwrap(),
        "view_b"
    );
}

/// Test view with function call expression
#[test]
fn test_view_with_function_expression() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query view with UPPER function
    let rows = db
        .query(
            "SELECT UPPER(name) AS upper_name FROM all_employees ORDER BY name LIMIT 3",
            (),
        )
        .expect("Failed to query view with function");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 3);
    assert_eq!(
        results[0].get_by_name::<String>("upper_name").unwrap(),
        "ALICE"
    );
    assert_eq!(
        results[1].get_by_name::<String>("upper_name").unwrap(),
        "BOB"
    );
    assert_eq!(
        results[2].get_by_name::<String>("upper_name").unwrap(),
        "CHARLIE"
    );
}

/// Test view with arithmetic expression
#[test]
fn test_view_with_arithmetic_expression() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Query view with arithmetic: salary * 1.1 (10% raise)
    let rows = db
        .query(
            "SELECT name, salary * 1.1 AS raised_salary FROM all_employees WHERE name = 'Alice'",
            (),
        )
        .expect("Failed to query view with arithmetic");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
    let raised: f64 = results[0].get_by_name::<f64>("raised_salary").unwrap();
    assert!((raised - 110000.0).abs() < 0.01);
}

/// Test view with CONCAT function
#[test]
fn test_view_with_concat_expression() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Use CONCAT to combine name and department
    let rows = db
        .query(
            "SELECT CONCAT(name, ' - ', department) AS full_info FROM all_employees WHERE id = 1",
            (),
        )
        .expect("Failed to query view with CONCAT");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get_by_name::<String>("full_info").unwrap(),
        "Alice - Engineering"
    );
}

/// Test view with COALESCE expression
#[test]
fn test_view_with_coalesce_expression() {
    let db = Database::open_in_memory().expect("Failed to open database");

    // Create table with nullable column
    db.execute(
        "CREATE TABLE test_null (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO test_null (id, value) VALUES (1, 'hello'), (2, NULL)",
        (),
    )
    .expect("Failed to insert");

    db.execute("CREATE VIEW test_view AS SELECT * FROM test_null", ())
        .expect("Failed to create view");

    // Use COALESCE to handle NULL
    let rows = db
        .query(
            "SELECT id, COALESCE(value, 'default') AS safe_value FROM test_view ORDER BY id",
            (),
        )
        .expect("Failed to query view with COALESCE");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].get_by_name::<String>("safe_value").unwrap(),
        "hello"
    );
    assert_eq!(
        results[1].get_by_name::<String>("safe_value").unwrap(),
        "default"
    );
}

/// Test view with multiple complex expressions
#[test]
fn test_view_with_multiple_complex_expressions() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Multiple expressions: function, arithmetic, and alias
    let rows = db
        .query(
            "SELECT UPPER(name) AS upper_name, salary * 0.3 AS bonus, department FROM all_employees WHERE salary >= 90000 ORDER BY salary DESC",
            (),
        )
        .expect("Failed to query view with multiple expressions");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 2); // Alice (100000) and Bob (90000)

    assert_eq!(
        results[0].get_by_name::<String>("upper_name").unwrap(),
        "ALICE"
    );
    let bonus1: f64 = results[0].get_by_name::<f64>("bonus").unwrap();
    assert!((bonus1 - 30000.0).abs() < 0.01);

    assert_eq!(
        results[1].get_by_name::<String>("upper_name").unwrap(),
        "BOB"
    );
    let bonus2: f64 = results[1].get_by_name::<f64>("bonus").unwrap();
    assert!((bonus2 - 27000.0).abs() < 0.01);
}

/// Test view with LENGTH function
#[test]
fn test_view_with_length_function() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    db.execute("CREATE VIEW all_employees AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // Get length of names - verify LENGTH function works
    let rows = db
        .query(
            "SELECT name, LENGTH(name) AS name_len FROM all_employees WHERE name = 'Charlie'",
            (),
        )
        .expect("Failed to query view with LENGTH");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);
    // Charlie has 7 chars
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Charlie");
    assert_eq!(results[0].get_by_name::<i64>("name_len").unwrap(), 7);

    // Test another name
    let rows = db
        .query(
            "SELECT name, LENGTH(name) AS name_len FROM all_employees WHERE name = 'Bob'",
            (),
        )
        .expect("Failed to query view with LENGTH");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);
    // Bob has 3 chars
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Bob");
    assert_eq!(results[0].get_by_name::<i64>("name_len").unwrap(), 3);
}

/// Test SHOW CREATE VIEW command
#[test]
fn test_show_create_view() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view
    db.execute(
        "CREATE VIEW high_earners AS SELECT name, salary FROM employees WHERE salary > 80000",
        (),
    )
    .expect("Failed to create view");

    // SHOW CREATE VIEW should return the view definition
    let rows = db
        .query("SHOW CREATE VIEW high_earners", ())
        .expect("Failed to execute SHOW CREATE VIEW");

    // Check column names
    let column_names = rows.columns().to_vec();
    assert_eq!(column_names.len(), 2);
    assert_eq!(column_names[0], "View");
    assert_eq!(column_names[1], "Create View");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);

    // Check view name
    assert_eq!(
        results[0].get_by_name::<String>("View").unwrap(),
        "high_earners"
    );

    // Check the CREATE VIEW statement contains the expected parts
    let create_stmt = results[0].get_by_name::<String>("Create View").unwrap();
    assert!(create_stmt.contains("CREATE VIEW"));
    assert!(create_stmt.contains("high_earners"));
    assert!(create_stmt.contains("SELECT"));
    assert!(create_stmt.contains("salary"));
    assert!(create_stmt.contains("80000"));
}

/// Test SHOW CREATE VIEW with case insensitive name
#[test]
fn test_show_create_view_case_insensitive() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view with mixed case
    db.execute("CREATE VIEW MyTestView AS SELECT * FROM employees", ())
        .expect("Failed to create view");

    // SHOW CREATE VIEW with different case
    let rows = db
        .query("SHOW CREATE VIEW mytestview", ())
        .expect("Failed to execute SHOW CREATE VIEW");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);

    // Should return original case
    assert_eq!(
        results[0].get_by_name::<String>("View").unwrap(),
        "MyTestView"
    );
}

/// Test SHOW CREATE VIEW for non-existent view
#[test]
fn test_show_create_view_not_exists() {
    let db = Database::open_in_memory().expect("Failed to open database");

    // SHOW CREATE VIEW for non-existent view should fail
    let result = db.query("SHOW CREATE VIEW nonexistent_view", ());
    assert!(result.is_err());
}

/// Test SHOW CREATE VIEW with complex query
#[test]
fn test_show_create_view_complex() {
    let db = Database::open_in_memory().expect("Failed to open database");
    setup_base_table(&db);

    // Create a view with a more complex query
    db.execute(
        "CREATE VIEW dept_summary AS SELECT department, COUNT(*) AS cnt, AVG(salary) AS avg_sal FROM employees GROUP BY department",
        (),
    )
    .expect("Failed to create view");

    let rows = db
        .query("SHOW CREATE VIEW dept_summary", ())
        .expect("Failed to execute SHOW CREATE VIEW");
    let results: Vec<_> = rows.collect_vec().expect("Failed to collect rows");
    assert_eq!(results.len(), 1);

    let create_stmt = results[0].get_by_name::<String>("Create View").unwrap();
    assert!(create_stmt.contains("GROUP BY"));
    assert!(create_stmt.contains("COUNT"));
    assert!(create_stmt.contains("AVG"));
}
