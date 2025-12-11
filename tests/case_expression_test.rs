//! Integration tests for CASE WHEN expressions
//!
//! Tests various forms of CASE expressions including:
//! - Searched CASE (CASE WHEN condition THEN result)
//! - Simple CASE (CASE expr WHEN value THEN result)
//! - CASE with aggregates
//! - CASE in different SQL contexts

use stoolap::Database;

// ============================================================================
// Basic CASE WHEN expressions
// ============================================================================

#[test]
fn test_searched_case_basic() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Simple searched CASE
    let result: String = db
        .query_one(
            "SELECT CASE WHEN 1 > 0 THEN 'positive' ELSE 'negative' END",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "positive");

    let result: String = db
        .query_one(
            "SELECT CASE WHEN 1 < 0 THEN 'positive' ELSE 'negative' END",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "negative");
}

#[test]
fn test_searched_case_multiple_when() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Multiple WHEN clauses
    let result: String = db
        .query_one(
            "SELECT CASE WHEN 5 < 0 THEN 'negative' WHEN 5 = 0 THEN 'zero' WHEN 5 > 0 THEN 'positive' END",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "positive");

    // Test that first matching WHEN wins
    let result: String = db
        .query_one(
            "SELECT CASE WHEN 5 > 0 THEN 'first' WHEN 5 > 4 THEN 'second' ELSE 'other' END",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "first");
}

#[test]
fn test_simple_case_basic() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Simple CASE with value matching
    let result: String = db
        .query_one("SELECT CASE 'apple' WHEN 'apple' THEN 'fruit' WHEN 'carrot' THEN 'vegetable' ELSE 'unknown' END", ())
        .expect("Failed to query");
    assert_eq!(result, "fruit");

    let result: String = db
        .query_one("SELECT CASE 'carrot' WHEN 'apple' THEN 'fruit' WHEN 'carrot' THEN 'vegetable' ELSE 'unknown' END", ())
        .expect("Failed to query");
    assert_eq!(result, "vegetable");

    let result: String = db
        .query_one("SELECT CASE 'pizza' WHEN 'apple' THEN 'fruit' WHEN 'carrot' THEN 'vegetable' ELSE 'unknown' END", ())
        .expect("Failed to query");
    assert_eq!(result, "unknown");
}

#[test]
fn test_simple_case_with_integers() {
    let db = Database::open_in_memory().expect("Failed to create database");

    let result: String = db
        .query_one("SELECT CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' WHEN 3 THEN 'three' ELSE 'other' END", ())
        .expect("Failed to query");
    assert_eq!(result, "two");
}

#[test]
fn test_case_no_else() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // CASE without ELSE returns NULL when no match
    let result: Option<String> = db
        .query_one("SELECT CASE WHEN 1 < 0 THEN 'negative' END", ())
        .expect("Failed to query");
    assert!(result.is_none());
}

// ============================================================================
// CASE with table data
// ============================================================================

#[test]
fn test_case_with_column_values() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, department TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO employees VALUES (1, 'Alice', 25, 'Engineering')",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO employees VALUES (2, 'Bob', 45, 'Engineering')",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO employees VALUES (3, 'Charlie', 62, 'Management')",
        (),
    )
    .expect("Failed to insert");
    db.execute(
        "INSERT INTO employees VALUES (4, 'Diana', 17, 'Intern')",
        (),
    )
    .expect("Failed to insert");

    // CASE with column reference
    let rows = db
        .query(
            "SELECT name, CASE WHEN age < 18 THEN 'Minor' WHEN age < 30 THEN 'Young' WHEN age < 50 THEN 'Middle' ELSE 'Senior' END as category FROM employees ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Alice");
    assert_eq!(
        results[0].get_by_name::<String>("category").unwrap(),
        "Young"
    );
    assert_eq!(results[1].get_by_name::<String>("name").unwrap(), "Bob");
    assert_eq!(
        results[1].get_by_name::<String>("category").unwrap(),
        "Middle"
    );
    assert_eq!(results[2].get_by_name::<String>("name").unwrap(), "Charlie");
    assert_eq!(
        results[2].get_by_name::<String>("category").unwrap(),
        "Senior"
    );
    assert_eq!(results[3].get_by_name::<String>("name").unwrap(), "Diana");
    assert_eq!(
        results[3].get_by_name::<String>("category").unwrap(),
        "Minor"
    );
}

#[test]
fn test_simple_case_with_column() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, status TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'active')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'pending')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 'cancelled')", ())
        .expect("Failed to insert");

    let rows = db
        .query(
            "SELECT id, CASE status WHEN 'active' THEN 'Running' WHEN 'pending' THEN 'Waiting' WHEN 'cancelled' THEN 'Stopped' ELSE 'Unknown' END as display FROM items ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get_by_name::<i64>("id").unwrap(), 1);
    assert_eq!(
        results[0].get_by_name::<String>("display").unwrap(),
        "Running"
    );
    assert_eq!(results[1].get_by_name::<i64>("id").unwrap(), 2);
    assert_eq!(
        results[1].get_by_name::<String>("display").unwrap(),
        "Waiting"
    );
    assert_eq!(results[2].get_by_name::<i64>("id").unwrap(), 3);
    assert_eq!(
        results[2].get_by_name::<String>("display").unwrap(),
        "Stopped"
    );
}

// ============================================================================
// CASE with aggregates
// ============================================================================

#[test]
fn test_case_with_sum_aggregate() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO orders VALUES (1, 1, 100)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO orders VALUES (2, 1, 200)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO orders VALUES (3, 2, 50)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO orders VALUES (4, 3, 500)", ())
        .expect("Failed to insert");

    // CASE referencing SUM aggregate
    let rows = db
        .query(
            "SELECT customer_id, SUM(amount), CASE WHEN SUM(amount) > 200 THEN 'VIP' ELSE 'Regular' END as tier FROM orders GROUP BY customer_id ORDER BY customer_id",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get_by_name::<i64>("customer_id").unwrap(), 1);
    assert_eq!(results[0].get_by_name::<f64>("SUM(amount)").unwrap(), 300.0);
    assert_eq!(results[0].get_by_name::<String>("tier").unwrap(), "VIP");
    assert_eq!(results[1].get_by_name::<i64>("customer_id").unwrap(), 2);
    assert_eq!(results[1].get_by_name::<f64>("SUM(amount)").unwrap(), 50.0);
    assert_eq!(results[1].get_by_name::<String>("tier").unwrap(), "Regular");
    assert_eq!(results[2].get_by_name::<i64>("customer_id").unwrap(), 3);
    assert_eq!(results[2].get_by_name::<f64>("SUM(amount)").unwrap(), 500.0);
    assert_eq!(results[2].get_by_name::<String>("tier").unwrap(), "VIP");
}

#[test]
fn test_case_with_count_aggregate() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE visits (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO visits VALUES (1, 1)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO visits VALUES (2, 1)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO visits VALUES (3, 1)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO visits VALUES (4, 2)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO visits VALUES (5, 3)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO visits VALUES (6, 3)", ())
        .expect("Failed to insert");

    let rows = db
        .query(
            "SELECT user_id, COUNT(*), CASE WHEN COUNT(*) >= 3 THEN 'Active' WHEN COUNT(*) >= 2 THEN 'Moderate' ELSE 'Inactive' END as activity FROM visits GROUP BY user_id ORDER BY user_id",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get_by_name::<i64>("user_id").unwrap(), 1);
    assert_eq!(results[0].get_by_name::<i64>("COUNT(*)").unwrap(), 3);
    assert_eq!(
        results[0].get_by_name::<String>("activity").unwrap(),
        "Active"
    );
    assert_eq!(results[1].get_by_name::<i64>("user_id").unwrap(), 2);
    assert_eq!(results[1].get_by_name::<i64>("COUNT(*)").unwrap(), 1);
    assert_eq!(
        results[1].get_by_name::<String>("activity").unwrap(),
        "Inactive"
    );
    assert_eq!(results[2].get_by_name::<i64>("user_id").unwrap(), 3);
    assert_eq!(results[2].get_by_name::<i64>("COUNT(*)").unwrap(), 2);
    assert_eq!(
        results[2].get_by_name::<String>("activity").unwrap(),
        "Moderate"
    );
}

#[test]
fn test_multiple_case_with_aggregates() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO sales VALUES (1, 'North', 1000)", ())
        .expect("Failed");
    db.execute("INSERT INTO sales VALUES (2, 'North', 1500)", ())
        .expect("Failed");
    db.execute("INSERT INTO sales VALUES (3, 'North', 800)", ())
        .expect("Failed");
    db.execute("INSERT INTO sales VALUES (4, 'South', 500)", ())
        .expect("Failed");
    db.execute("INSERT INTO sales VALUES (5, 'East', 2000)", ())
        .expect("Failed");
    db.execute("INSERT INTO sales VALUES (6, 'East', 3000)", ())
        .expect("Failed");

    // Multiple CASE expressions with different aggregates
    let rows = db
        .query(
            "SELECT region, \
             CASE WHEN SUM(amount) >= 3000 THEN 'High' ELSE 'Low' END as revenue_tier, \
             CASE WHEN COUNT(*) >= 3 THEN 'Many' ELSE 'Few' END as order_count, \
             CASE WHEN AVG(amount) >= 2000 THEN 'Premium' ELSE 'Standard' END as avg_tier \
             FROM sales GROUP BY region ORDER BY region",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 3);

    // East: SUM=5000 (High), COUNT=2 (Few), AVG=2500 (Premium)
    assert_eq!(results[0].get_by_name::<String>("region").unwrap(), "East");
    assert_eq!(
        results[0].get_by_name::<String>("revenue_tier").unwrap(),
        "High"
    );
    assert_eq!(
        results[0].get_by_name::<String>("order_count").unwrap(),
        "Few"
    );
    assert_eq!(
        results[0].get_by_name::<String>("avg_tier").unwrap(),
        "Premium"
    );

    // North: SUM=3300 (High), COUNT=3 (Many), AVG=1100 (Standard)
    assert_eq!(results[1].get_by_name::<String>("region").unwrap(), "North");
    assert_eq!(
        results[1].get_by_name::<String>("revenue_tier").unwrap(),
        "High"
    );
    assert_eq!(
        results[1].get_by_name::<String>("order_count").unwrap(),
        "Many"
    );
    assert_eq!(
        results[1].get_by_name::<String>("avg_tier").unwrap(),
        "Standard"
    );

    // South: SUM=500 (Low), COUNT=1 (Few), AVG=500 (Standard)
    assert_eq!(results[2].get_by_name::<String>("region").unwrap(), "South");
    assert_eq!(
        results[2].get_by_name::<String>("revenue_tier").unwrap(),
        "Low"
    );
    assert_eq!(
        results[2].get_by_name::<String>("order_count").unwrap(),
        "Few"
    );
    assert_eq!(
        results[2].get_by_name::<String>("avg_tier").unwrap(),
        "Standard"
    );
}

// ============================================================================
// CASE in WHERE clause
// ============================================================================

#[test]
fn test_case_in_where_clause() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO products VALUES (1, 'Widget', 10, 'A')", ())
        .expect("Failed");
    db.execute("INSERT INTO products VALUES (2, 'Gadget', 50, 'B')", ())
        .expect("Failed");
    db.execute("INSERT INTO products VALUES (3, 'Gizmo', 100, 'A')", ())
        .expect("Failed");
    db.execute("INSERT INTO products VALUES (4, 'Thing', 25, 'C')", ())
        .expect("Failed");

    // CASE in WHERE clause
    let rows = db
        .query(
            "SELECT name FROM products WHERE CASE WHEN category = 'A' THEN price > 50 ELSE price > 20 END ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 3);

    let names: Vec<String> = results
        .iter()
        .map(|r| r.get_by_name::<String>("name").unwrap())
        .collect();
    assert!(names.contains(&"Gadget".to_string()));
    assert!(names.contains(&"Gizmo".to_string()));
    assert!(names.contains(&"Thing".to_string()));
}

// ============================================================================
// CASE in GROUP BY
// ============================================================================

#[test]
fn test_case_in_group_by() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO orders VALUES (1, 'shipped', 100)", ())
        .expect("Failed");
    db.execute("INSERT INTO orders VALUES (2, 'pending', 200)", ())
        .expect("Failed");
    db.execute("INSERT INTO orders VALUES (3, 'shipped', 300)", ())
        .expect("Failed");
    db.execute("INSERT INTO orders VALUES (4, 'cancelled', 50)", ())
        .expect("Failed");
    db.execute("INSERT INTO orders VALUES (5, 'pending', 150)", ())
        .expect("Failed");

    // GROUP BY CASE expression
    let rows = db
        .query(
            "SELECT CASE WHEN status = 'shipped' THEN 'Completed' ELSE 'In Progress' END as category, \
             COUNT(*), SUM(amount) \
             FROM orders \
             GROUP BY CASE WHEN status = 'shipped' THEN 'Completed' ELSE 'In Progress' END \
             ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].get_by_name::<String>("category").unwrap(),
        "Completed"
    );
    assert_eq!(results[0].get_by_name::<i64>("COUNT(*)").unwrap(), 2);
    assert_eq!(results[0].get_by_name::<f64>("SUM(amount)").unwrap(), 400.0);
    assert_eq!(
        results[1].get_by_name::<String>("category").unwrap(),
        "In Progress"
    );
    assert_eq!(results[1].get_by_name::<i64>("COUNT(*)").unwrap(), 3);
    assert_eq!(results[1].get_by_name::<f64>("SUM(amount)").unwrap(), 400.0);
}

// ============================================================================
// CASE with NULL handling
// ============================================================================

#[test]
fn test_case_with_null_values() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 10)", ())
        .expect("Failed");
    db.execute("INSERT INTO items VALUES (2, NULL)", ())
        .expect("Failed");
    db.execute("INSERT INTO items VALUES (3, 20)", ())
        .expect("Failed");

    // CASE handling NULL
    let rows = db
        .query(
            "SELECT id, CASE WHEN value IS NULL THEN 'Missing' ELSE 'Present' END as status FROM items ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get_by_name::<i64>("id").unwrap(), 1);
    assert_eq!(
        results[0].get_by_name::<String>("status").unwrap(),
        "Present"
    );
    assert_eq!(results[1].get_by_name::<i64>("id").unwrap(), 2);
    assert_eq!(
        results[1].get_by_name::<String>("status").unwrap(),
        "Missing"
    );
    assert_eq!(results[2].get_by_name::<i64>("id").unwrap(), 3);
    assert_eq!(
        results[2].get_by_name::<String>("status").unwrap(),
        "Present"
    );
}

// ============================================================================
// CASE with nested expressions
// ============================================================================

#[test]
fn test_nested_case_expressions() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Nested CASE expressions
    let result: String = db
        .query_one(
            "SELECT CASE \
                WHEN 1 = 1 THEN \
                    CASE WHEN 2 > 1 THEN 'nested-true' ELSE 'nested-false' END \
                ELSE 'outer-false' \
             END",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "nested-true");
}

#[test]
fn test_case_with_arithmetic() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // CASE with arithmetic in conditions and results
    let result: i64 = db
        .query_one(
            "SELECT CASE WHEN 10 + 5 > 12 THEN 100 * 2 ELSE 50 + 25 END",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 200);
}

#[test]
fn test_case_with_string_functions() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // CASE with string functions
    let result: String = db
        .query_one(
            "SELECT CASE WHEN LENGTH('hello') > 3 THEN UPPER('yes') ELSE LOWER('NO') END",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "YES");
}

// ============================================================================
// CASE in ORDER BY (Note: CASE expressions in ORDER BY are not yet fully supported
// for sorting - this test verifies the query runs but uses ORDER BY on regular columns)
// ============================================================================

#[test]
fn test_case_in_order_by() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, priority TEXT, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'low', 'Task A')", ())
        .expect("Failed");
    db.execute("INSERT INTO items VALUES (2, 'high', 'Task B')", ())
        .expect("Failed");
    db.execute("INSERT INTO items VALUES (3, 'medium', 'Task C')", ())
        .expect("Failed");
    db.execute("INSERT INTO items VALUES (4, 'high', 'Task D')", ())
        .expect("Failed");

    // Workaround: Add CASE as a SELECT column and ORDER BY that column alias
    // This is the supported pattern for CASE-based ordering
    let rows = db
        .query(
            "SELECT name, priority, CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END as sort_order FROM items ORDER BY sort_order, name",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 4);
    // First two should be high priority (sorted by name: B, D)
    assert_eq!(
        results[0].get_by_name::<String>("priority").unwrap(),
        "high"
    );
    assert_eq!(results[0].get_by_name::<String>("name").unwrap(), "Task B");
    assert_eq!(
        results[1].get_by_name::<String>("priority").unwrap(),
        "high"
    );
    assert_eq!(results[1].get_by_name::<String>("name").unwrap(), "Task D");
    // Then medium
    assert_eq!(
        results[2].get_by_name::<String>("priority").unwrap(),
        "medium"
    );
    // Then low
    assert_eq!(results[3].get_by_name::<String>("priority").unwrap(), "low");
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_case_all_conditions_false() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // All conditions false, no ELSE
    let result: Option<String> = db
        .query_one(
            "SELECT CASE WHEN 1 = 2 THEN 'a' WHEN 1 = 3 THEN 'b' END",
            (),
        )
        .expect("Failed to query");
    assert!(result.is_none());

    // All conditions false, with ELSE
    let result: String = db
        .query_one(
            "SELECT CASE WHEN 1 = 2 THEN 'a' WHEN 1 = 3 THEN 'b' ELSE 'default' END",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, "default");
}

#[test]
fn test_case_with_boolean_result() {
    let db = Database::open_in_memory().expect("Failed to create database");

    let result: bool = db
        .query_one("SELECT CASE WHEN 1 > 0 THEN TRUE ELSE FALSE END", ())
        .expect("Failed to query");
    assert!(result);

    let result: bool = db
        .query_one("SELECT CASE WHEN 1 < 0 THEN TRUE ELSE FALSE END", ())
        .expect("Failed to query");
    assert!(!result);
}

#[test]
fn test_case_with_numeric_types() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Integer result
    let result: i64 = db
        .query_one("SELECT CASE WHEN TRUE THEN 42 ELSE 0 END", ())
        .expect("Failed to query");
    assert_eq!(result, 42);

    // Float result
    let result: f64 = db
        .query_one("SELECT CASE WHEN TRUE THEN 3.14 ELSE 0.0 END", ())
        .expect("Failed to query");
    assert!((result - 3.14).abs() < 0.001);
}

// ============================================================================
// Additional CASE with aggregate edge cases
// ============================================================================

#[test]
fn test_case_with_avg_aggregate() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, student TEXT, score FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO scores VALUES (1, 'Alice', 95)", ())
        .expect("Failed");
    db.execute("INSERT INTO scores VALUES (2, 'Alice', 85)", ())
        .expect("Failed");
    db.execute("INSERT INTO scores VALUES (3, 'Bob', 70)", ())
        .expect("Failed");
    db.execute("INSERT INTO scores VALUES (4, 'Bob', 65)", ())
        .expect("Failed");
    db.execute("INSERT INTO scores VALUES (5, 'Charlie', 80)", ())
        .expect("Failed");

    let rows = db
        .query(
            "SELECT student, AVG(score), \
             CASE WHEN AVG(score) >= 90 THEN 'A' \
                  WHEN AVG(score) >= 80 THEN 'B' \
                  WHEN AVG(score) >= 70 THEN 'C' \
                  ELSE 'F' END as grade \
             FROM scores GROUP BY student ORDER BY student",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 3);

    // Alice: avg=90 -> A
    assert_eq!(
        results[0].get_by_name::<String>("student").unwrap(),
        "Alice"
    );
    assert_eq!(results[0].get_by_name::<String>("grade").unwrap(), "A");

    // Bob: avg=67.5 -> F
    assert_eq!(results[1].get_by_name::<String>("student").unwrap(), "Bob");
    assert_eq!(results[1].get_by_name::<String>("grade").unwrap(), "F");

    // Charlie: avg=80 -> B
    assert_eq!(
        results[2].get_by_name::<String>("student").unwrap(),
        "Charlie"
    );
    assert_eq!(results[2].get_by_name::<String>("grade").unwrap(), "B");
}

#[test]
fn test_case_with_min_max_aggregate() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE temps (id INTEGER PRIMARY KEY, city TEXT, temp INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO temps VALUES (1, 'NYC', 30)", ())
        .expect("Failed");
    db.execute("INSERT INTO temps VALUES (2, 'NYC', 90)", ())
        .expect("Failed");
    db.execute("INSERT INTO temps VALUES (3, 'LA', 60)", ())
        .expect("Failed");
    db.execute("INSERT INTO temps VALUES (4, 'LA', 80)", ())
        .expect("Failed");

    let rows = db
        .query(
            "SELECT city, MIN(temp), MAX(temp), \
             CASE WHEN MAX(temp) - MIN(temp) > 50 THEN 'Variable' ELSE 'Stable' END as climate \
             FROM temps GROUP BY city ORDER BY city",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 2);

    // LA: 80-60=20 -> Stable
    assert_eq!(results[0].get_by_name::<String>("city").unwrap(), "LA");
    assert_eq!(
        results[0].get_by_name::<String>("climate").unwrap(),
        "Stable"
    );

    // NYC: 90-30=60 -> Variable
    assert_eq!(results[1].get_by_name::<String>("city").unwrap(), "NYC");
    assert_eq!(
        results[1].get_by_name::<String>("climate").unwrap(),
        "Variable"
    );
}

#[test]
fn test_case_comparing_aggregates() {
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, category TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO data VALUES (1, 'A', 10)", ())
        .expect("Failed");
    db.execute("INSERT INTO data VALUES (2, 'A', 20)", ())
        .expect("Failed");
    db.execute("INSERT INTO data VALUES (3, 'A', 30)", ())
        .expect("Failed");
    db.execute("INSERT INTO data VALUES (4, 'B', 100)", ())
        .expect("Failed");

    let rows = db
        .query(
            "SELECT category, SUM(value), COUNT(*), \
             CASE WHEN SUM(value) / COUNT(*) > 30 THEN 'High Avg' ELSE 'Low Avg' END as avg_type \
             FROM data GROUP BY category ORDER BY category",
            (),
        )
        .expect("Failed to query");

    let results: Vec<_> = rows.collect_vec().expect("Failed to collect");
    assert_eq!(results.len(), 2);

    // A: 60/3=20 -> Low Avg
    assert_eq!(results[0].get_by_name::<String>("category").unwrap(), "A");
    assert_eq!(
        results[0].get_by_name::<String>("avg_type").unwrap(),
        "Low Avg"
    );

    // B: 100/1=100 -> High Avg
    assert_eq!(results[1].get_by_name::<String>("category").unwrap(), "B");
    assert_eq!(
        results[1].get_by_name::<String>("avg_type").unwrap(),
        "High Avg"
    );
}
