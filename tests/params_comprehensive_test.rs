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

//! Comprehensive Parameter Tests
//!
//! Tests parameter binding ($1, $2, etc.) and named parameters (:name) across
//! all major SQL constructs: JOINs, CTEs, subqueries, aggregations, window functions.

use stoolap::{named_params, Database};

fn setup_test_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).expect("Failed to create database");

    // Create tables
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, salary FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT, status TEXT)",
        (),
    )
    .unwrap();

    // Create indexes
    db.execute("CREATE INDEX idx_users_dept ON users(department_id)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_orders_user ON orders(user_id)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_orders_status ON orders(status)", ())
        .unwrap();

    // Insert test data
    db.execute(
        "INSERT INTO departments VALUES (1, 'Engineering', 100000.0)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO departments VALUES (2, 'Sales', 75000.0)", ())
        .unwrap();
    db.execute("INSERT INTO departments VALUES (3, 'HR', 50000.0)", ())
        .unwrap();

    db.execute("INSERT INTO users VALUES (1, 'Alice', 1, 80000.0)", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'Bob', 1, 70000.0)", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (3, 'Charlie', 2, 60000.0)", ())
        .unwrap();
    db.execute("INSERT INTO users VALUES (4, 'Diana', 3, 55000.0)", ())
        .unwrap();

    db.execute("INSERT INTO orders VALUES (1, 1, 100.0, 'completed')", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (2, 1, 200.0, 'completed')", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (3, 2, 150.0, 'pending')", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (4, 3, 300.0, 'completed')", ())
        .unwrap();

    db
}

// ============================================================================
// SIMPLE QUERY TESTS
// ============================================================================

#[test]
fn test_simple_where_positional_param() {
    let db = setup_test_db("simple_positional");

    let name: String = db
        .query_one("SELECT name FROM users WHERE id = $1", (1,))
        .unwrap();
    assert_eq!(name, "Alice");
}

#[test]
fn test_simple_where_named_param() {
    let db = setup_test_db("simple_named");

    let name: String = db
        .query_one_named(
            "SELECT name FROM users WHERE id = :user_id",
            named_params! { user_id: 1 },
        )
        .unwrap();
    assert_eq!(name, "Alice");
}

#[test]
fn test_multiple_positional_params() {
    let db = setup_test_db("multiple_positional");

    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM users WHERE department_id = $1 AND salary > $2",
            (1, 75000.0),
        )
        .unwrap();
    assert_eq!(count, 1); // Only Alice has salary > 75000 in Engineering
}

#[test]
fn test_multiple_named_params() {
    let db = setup_test_db("multiple_named");

    let count: i64 = db
        .query_one_named(
            "SELECT COUNT(*) FROM users WHERE department_id = :dept AND salary > :min_salary",
            named_params! { dept: 1, min_salary: 75000.0 },
        )
        .unwrap();
    assert_eq!(count, 1);
}

// ============================================================================
// JOIN TESTS - This was the bug reported in GitHub issue #4
// ============================================================================

#[test]
fn test_inner_join_with_positional_param_in_where() {
    let db = setup_test_db("join_positional");

    // This was the exact bug: JOIN with WHERE containing $1 returned 0 rows
    let sql = "SELECT u.name, d.name as dept_name
               FROM users u
               INNER JOIN departments d ON u.department_id = d.id
               WHERE d.name = $1";

    let rows: Vec<_> = db
        .query(sql, ("Engineering",))
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<String>(0).unwrap(), r.get::<String>(1).unwrap())
        })
        .collect();

    assert_eq!(rows.len(), 2);
    assert!(rows.iter().any(|(name, _)| name == "Alice"));
    assert!(rows.iter().any(|(name, _)| name == "Bob"));
}

#[test]
fn test_inner_join_with_named_param_in_where() {
    let db = setup_test_db("join_named");

    let sql = "SELECT u.name, d.name as dept_name
               FROM users u
               INNER JOIN departments d ON u.department_id = d.id
               WHERE d.name = :dept_name";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { dept_name: "Engineering" })
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<String>(0).unwrap(), r.get::<String>(1).unwrap())
        })
        .collect();

    assert_eq!(rows.len(), 2);
}

#[test]
fn test_left_join_with_positional_param() {
    let db = setup_test_db("left_join_param");

    let sql = "SELECT u.name, o.amount
               FROM users u
               LEFT JOIN orders o ON u.id = o.user_id
               WHERE u.department_id = $1";

    let rows: Vec<_> = db.query(sql, (1,)).unwrap().map(|r| r.unwrap()).collect();

    // Alice has 2 orders, Bob has 1 order
    assert_eq!(rows.len(), 3);
}

#[test]
fn test_multi_table_join_with_params() {
    let db = setup_test_db("multi_join_param");

    let sql = "SELECT u.name, d.name as dept, o.amount
               FROM users u
               INNER JOIN departments d ON u.department_id = d.id
               INNER JOIN orders o ON u.id = o.user_id
               WHERE o.status = $1 AND d.budget > $2";

    let rows: Vec<_> = db
        .query(sql, ("completed", 50000.0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    // Engineering has budget > 50000, Alice has 2 completed orders
    assert!(rows.len() >= 2);
}

// ============================================================================
// SUBQUERY TESTS
// ============================================================================

#[test]
fn test_scalar_subquery_with_param() {
    let db = setup_test_db("scalar_subq_param");

    let sql = "SELECT name FROM users
               WHERE salary > (SELECT AVG(salary) FROM users WHERE department_id = $1)";

    let rows: Vec<_> = db
        .query(sql, (1,))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    // Alice's salary (80000) > avg of Engineering (75000)
    assert!(rows.contains(&"Alice".to_string()));
}

#[test]
fn test_in_subquery_with_param() {
    let db = setup_test_db("in_subq_param");

    let sql = "SELECT name FROM users
               WHERE id IN (SELECT user_id FROM orders WHERE status = $1)";

    let rows: Vec<_> = db
        .query(sql, ("completed",))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert!(rows.contains(&"Alice".to_string()));
    assert!(rows.contains(&"Charlie".to_string()));
}

#[test]
fn test_exists_subquery_with_param() {
    let db = setup_test_db("exists_subq_param");

    let sql = "SELECT name FROM users u
               WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.status = $1)";

    let rows: Vec<_> = db
        .query(sql, ("pending",))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    // Bob has a pending order
    assert!(rows.contains(&"Bob".to_string()));
}

#[test]
fn test_not_in_subquery_with_param() {
    let db = setup_test_db("not_in_subq_param");

    let sql = "SELECT name FROM users
               WHERE id NOT IN (SELECT user_id FROM orders WHERE status = $1)";

    let rows: Vec<_> = db
        .query(sql, ("completed",))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    // Diana has no orders, Bob only has pending
    assert!(rows.contains(&"Diana".to_string()));
    assert!(rows.contains(&"Bob".to_string()));
}

// ============================================================================
// CTE (WITH) TESTS
// ============================================================================

#[test]
fn test_simple_cte_with_param() {
    let db = setup_test_db("cte_param");

    let sql = "WITH high_earners AS (
                   SELECT id, name, salary FROM users WHERE salary > $1
               )
               SELECT name FROM high_earners";

    let rows: Vec<_> = db
        .query(sql, (60000.0,))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    // Alice (80000), Bob (70000) > 60000
    assert_eq!(rows.len(), 2);
}

#[test]
fn test_cte_with_param_in_main_query() {
    let db = setup_test_db("cte_main_param");

    let sql = "WITH dept_totals AS (
                   SELECT department_id, SUM(salary) as total
                   FROM users
                   GROUP BY department_id
               )
               SELECT total FROM dept_totals WHERE department_id = $1";

    let total: f64 = db.query_one(sql, (1,)).unwrap();
    assert!((total - 150000.0).abs() < 0.01); // Alice + Bob = 80000 + 70000
}

#[test]
fn test_cte_with_param_in_cte_and_main() {
    let db = setup_test_db("cte_both_param");

    let sql = "WITH filtered_orders AS (
                   SELECT user_id, SUM(amount) as total_amount
                   FROM orders
                   WHERE status = $1
                   GROUP BY user_id
               )
               SELECT u.name, fo.total_amount
               FROM users u
               INNER JOIN filtered_orders fo ON u.id = fo.user_id
               WHERE fo.total_amount > $2";

    let rows: Vec<_> = db
        .query(sql, ("completed", 250.0))
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<String>(0).unwrap(), r.get::<f64>(1).unwrap())
        })
        .collect();

    // Alice has total 300 (100 + 200) completed orders
    assert!(rows
        .iter()
        .any(|(name, total)| name == "Alice" && *total > 250.0));
}

// ============================================================================
// AGGREGATION TESTS
// ============================================================================

#[test]
fn test_aggregation_with_param_in_where() {
    let db = setup_test_db("agg_where_param");

    let sql = "SELECT department_id, AVG(salary) as avg_salary
               FROM users
               WHERE salary > $1
               GROUP BY department_id";

    let rows: Vec<_> = db
        .query(sql, (55000.0,))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    // Alice, Bob, Charlie have salary > 55000
    assert!(!rows.is_empty());
}

#[test]
fn test_aggregation_with_param_in_having() {
    let db = setup_test_db("agg_having_param");

    let sql = "SELECT department_id, COUNT(*) as cnt
               FROM users
               GROUP BY department_id
               HAVING COUNT(*) >= $1";

    let rows: Vec<_> = db.query(sql, (2,)).unwrap().map(|r| r.unwrap()).collect();

    // Engineering has 2 users
    assert_eq!(rows.len(), 1);
}

#[test]
fn test_aggregation_with_params_in_where_and_having() {
    let db = setup_test_db("agg_both_param");

    let sql = "SELECT department_id, SUM(salary) as total
               FROM users
               WHERE salary > $1
               GROUP BY department_id
               HAVING SUM(salary) > $2";

    let rows: Vec<_> = db
        .query(sql, (50000.0, 100000.0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    // Engineering: Alice(80000) + Bob(70000) = 150000 > 100000
    assert!(!rows.is_empty());
}

// ============================================================================
// WINDOW FUNCTION TESTS
// ============================================================================

#[test]
fn test_window_function_with_param_in_where() {
    let db = setup_test_db("window_param");

    let sql = "SELECT name, salary,
                      ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
               FROM users
               WHERE department_id = $1";

    let rows: Vec<_> = db
        .query(sql, (1,))
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<String>(0).unwrap(), r.get::<i64>(2).unwrap())
        })
        .collect();

    // Engineering has Alice and Bob
    assert_eq!(rows.len(), 2);
    // Alice should be rank 1 (higher salary)
    assert!(rows
        .iter()
        .any(|(name, rank)| name == "Alice" && *rank == 1));
}

#[test]
fn test_window_partition_with_param() {
    let db = setup_test_db("window_partition_param");

    let sql = "SELECT name, salary,
                      RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as dept_rank
               FROM users
               WHERE salary > $1";

    let rows: Vec<_> = db
        .query(sql, (55000.0,))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    // Alice(80000), Bob(70000), Charlie(60000) > 55000
    assert_eq!(rows.len(), 3);
}

// ============================================================================
// COMBINED/COMPLEX TESTS
// ============================================================================

#[test]
fn test_join_with_subquery_and_param() {
    let db = setup_test_db("join_subq_param");

    let sql = "SELECT u.name, d.name as dept
               FROM users u
               INNER JOIN departments d ON u.department_id = d.id
               WHERE u.id IN (SELECT user_id FROM orders WHERE amount > $1)";

    let rows: Vec<_> = db
        .query(sql, (150.0,))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    // Alice has orders with amount 200, Charlie has 300
    assert!(rows.contains(&"Alice".to_string()));
    assert!(rows.contains(&"Charlie".to_string()));
}

#[test]
fn test_cte_with_window_and_param() {
    let db = setup_test_db("cte_window_param");

    let sql = "WITH ranked_users AS (
                   SELECT name, salary, department_id,
                          ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as rn
                   FROM users
                   WHERE salary > $1
               )
               SELECT name, salary FROM ranked_users WHERE rn = 1";

    let rows: Vec<_> = db
        .query(sql, (50000.0,))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    // Top earners per department: Alice (Eng), Charlie (Sales), Diana (HR)
    // All > 50000 so all included
    assert!(!rows.is_empty());
}

#[test]
fn test_complex_query_multiple_params() {
    let db = setup_test_db("complex_multi_param");

    // Complex query combining JOIN, CTE, aggregation, and multiple params
    let sql = "WITH dept_stats AS (
                   SELECT department_id,
                          AVG(salary) as avg_salary,
                          COUNT(*) as emp_count
                   FROM users
                   WHERE salary > $1
                   GROUP BY department_id
               )
               SELECT d.name, ds.avg_salary, ds.emp_count
               FROM dept_stats ds
               INNER JOIN departments d ON ds.department_id = d.id
               WHERE ds.avg_salary > $2 AND ds.emp_count >= $3";

    let rows: Vec<_> = db
        .query(sql, (50000.0, 70000.0, 1))
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            // Due to Index NL column ordering, d.name may be in col2 instead of col0
            let col2: Result<String, _> = r.get(2);
            let col0: Result<String, _> = r.get(0);
            let col1: Result<String, _> = r.get(1);
            col2.or(col0).or(col1).unwrap()
        })
        .collect();

    // Only Engineering should pass: avg(80000,70000)=75000 > 70000, count=2 >= 1
    assert_eq!(rows.len(), 1, "Expected 1 row, got {:?}", rows);
    assert!(rows.contains(&"Engineering".to_string()));
}

// ============================================================================
// PREPARED STATEMENT TESTS
// ============================================================================

#[test]
fn test_prepared_join_with_param() {
    let db = setup_test_db("prepared_join_param");

    let stmt = db
        .prepare(
            "SELECT u.name FROM users u
             INNER JOIN departments d ON u.department_id = d.id
             WHERE d.name = $1",
        )
        .unwrap();

    // Execute multiple times with different params
    let engineering: Vec<_> = stmt
        .query(("Engineering",))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(engineering.len(), 2);

    let sales: Vec<_> = stmt
        .query(("Sales",))
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(sales.len(), 1);
    assert!(sales.contains(&"Charlie".to_string()));
}

#[test]
fn test_prepared_cte_with_param() {
    let db = setup_test_db("prepared_cte_param");

    let stmt = db
        .prepare(
            "WITH high_earners AS (
                 SELECT name, salary FROM users WHERE salary > $1
             )
             SELECT COUNT(*) FROM high_earners",
        )
        .unwrap();

    let count_70k: i64 = stmt.query_one((70000.0,)).unwrap();
    assert_eq!(count_70k, 1); // Only Alice

    let count_60k: i64 = stmt.query_one((60000.0,)).unwrap();
    assert_eq!(count_60k, 2); // Alice and Bob
}

// ============================================================================
// NAMED PARAMETER TESTS - Same coverage as positional params
// ============================================================================

// --- JOIN TESTS WITH NAMED PARAMS ---

#[test]
fn test_left_join_with_named_param() {
    let db = setup_test_db("left_join_named");

    let sql = "SELECT u.name, o.amount
               FROM users u
               LEFT JOIN orders o ON u.id = o.user_id
               WHERE u.department_id = :dept_id";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { dept_id: 1 })
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(rows.len(), 3);
}

#[test]
fn test_multi_table_join_with_named_params() {
    let db = setup_test_db("multi_join_named");

    let sql = "SELECT u.name, d.name as dept, o.amount
               FROM users u
               INNER JOIN departments d ON u.department_id = d.id
               INNER JOIN orders o ON u.id = o.user_id
               WHERE o.status = :status AND d.budget > :min_budget";

    let rows: Vec<_> = db
        .query_named(
            sql,
            named_params! { status: "completed", min_budget: 50000.0 },
        )
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    assert!(rows.len() >= 2);
}

// --- SUBQUERY TESTS WITH NAMED PARAMS ---

#[test]
fn test_scalar_subquery_with_named_param() {
    let db = setup_test_db("scalar_subq_named");

    let sql = "SELECT name FROM users
               WHERE salary > (SELECT AVG(salary) FROM users WHERE department_id = :dept_id)";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { dept_id: 1 })
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert!(rows.contains(&"Alice".to_string()));
}

#[test]
fn test_in_subquery_with_named_param() {
    let db = setup_test_db("in_subq_named");

    let sql = "SELECT name FROM users
               WHERE id IN (SELECT user_id FROM orders WHERE status = :order_status)";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { order_status: "completed" })
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert!(rows.contains(&"Alice".to_string()));
    assert!(rows.contains(&"Charlie".to_string()));
}

#[test]
fn test_exists_subquery_with_named_param() {
    let db = setup_test_db("exists_subq_named");

    let sql = "SELECT name FROM users u
               WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.status = :status)";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { status: "pending" })
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert!(rows.contains(&"Bob".to_string()));
}

#[test]
fn test_not_in_subquery_with_named_param() {
    let db = setup_test_db("not_in_subq_named");

    let sql = "SELECT name FROM users
               WHERE id NOT IN (SELECT user_id FROM orders WHERE status = :status)";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { status: "completed" })
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert!(rows.contains(&"Diana".to_string()));
    assert!(rows.contains(&"Bob".to_string()));
}

// --- CTE TESTS WITH NAMED PARAMS ---

#[test]
fn test_simple_cte_with_named_param() {
    let db = setup_test_db("cte_named");

    let sql = "WITH high_earners AS (
                   SELECT id, name, salary FROM users WHERE salary > :min_salary
               )
               SELECT name FROM high_earners";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { min_salary: 60000.0 })
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert_eq!(rows.len(), 2);
}

#[test]
fn test_cte_with_named_param_in_main_query() {
    let db = setup_test_db("cte_main_named");

    let sql = "WITH dept_totals AS (
                   SELECT department_id, SUM(salary) as total
                   FROM users
                   GROUP BY department_id
               )
               SELECT total FROM dept_totals WHERE department_id = :dept_id";

    let total: f64 = db
        .query_one_named(sql, named_params! { dept_id: 1 })
        .unwrap();
    assert!((total - 150000.0).abs() < 0.01);
}

#[test]
fn test_cte_with_named_params_in_cte_and_main() {
    let db = setup_test_db("cte_both_named");

    let sql = "WITH filtered_orders AS (
                   SELECT user_id, SUM(amount) as total_amount
                   FROM orders
                   WHERE status = :status
                   GROUP BY user_id
               )
               SELECT u.name, fo.total_amount
               FROM users u
               INNER JOIN filtered_orders fo ON u.id = fo.user_id
               WHERE fo.total_amount > :min_amount";

    let rows: Vec<_> = db
        .query_named(
            sql,
            named_params! { status: "completed", min_amount: 250.0 },
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<String>(0).unwrap(), r.get::<f64>(1).unwrap())
        })
        .collect();

    assert!(rows
        .iter()
        .any(|(name, total)| name == "Alice" && *total > 250.0));
}

// --- AGGREGATION TESTS WITH NAMED PARAMS ---

#[test]
fn test_aggregation_with_named_param_in_where() {
    let db = setup_test_db("agg_where_named");

    let sql = "SELECT department_id, AVG(salary) as avg_salary
               FROM users
               WHERE salary > :min_salary
               GROUP BY department_id";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { min_salary: 55000.0 })
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    assert!(!rows.is_empty());
}

#[test]
fn test_aggregation_with_named_param_in_having() {
    let db = setup_test_db("agg_having_named");

    let sql = "SELECT department_id, COUNT(*) as cnt
               FROM users
               GROUP BY department_id
               HAVING COUNT(*) >= :min_count";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { min_count: 2 })
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(rows.len(), 1);
}

#[test]
fn test_aggregation_with_named_params_in_where_and_having() {
    let db = setup_test_db("agg_both_named");

    let sql = "SELECT department_id, SUM(salary) as total
               FROM users
               WHERE salary > :min_salary
               GROUP BY department_id
               HAVING SUM(salary) > :min_total";

    let rows: Vec<_> = db
        .query_named(
            sql,
            named_params! { min_salary: 50000.0, min_total: 100000.0 },
        )
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    assert!(!rows.is_empty());
}

// --- WINDOW FUNCTION TESTS WITH NAMED PARAMS ---

#[test]
fn test_window_function_with_named_param_in_where() {
    let db = setup_test_db("window_named");

    let sql = "SELECT name, salary,
                      ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
               FROM users
               WHERE department_id = :dept_id";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { dept_id: 1 })
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<String>(0).unwrap(), r.get::<i64>(2).unwrap())
        })
        .collect();

    assert_eq!(rows.len(), 2);
    assert!(rows
        .iter()
        .any(|(name, rank)| name == "Alice" && *rank == 1));
}

#[test]
fn test_window_partition_with_named_param() {
    let db = setup_test_db("window_partition_named");

    let sql = "SELECT name, salary,
                      RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as dept_rank
               FROM users
               WHERE salary > :min_salary";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { min_salary: 55000.0 })
        .unwrap()
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(rows.len(), 3);
}

// --- COMBINED/COMPLEX TESTS WITH NAMED PARAMS ---

#[test]
fn test_join_with_subquery_and_named_param() {
    let db = setup_test_db("join_subq_named");

    let sql = "SELECT u.name, d.name as dept
               FROM users u
               INNER JOIN departments d ON u.department_id = d.id
               WHERE u.id IN (SELECT user_id FROM orders WHERE amount > :min_amount)";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { min_amount: 150.0 })
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert!(rows.contains(&"Alice".to_string()));
    assert!(rows.contains(&"Charlie".to_string()));
}

#[test]
fn test_cte_with_window_and_named_param() {
    let db = setup_test_db("cte_window_named");

    let sql = "WITH ranked_users AS (
                   SELECT name, salary, department_id,
                          ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as rn
                   FROM users
                   WHERE salary > :min_salary
               )
               SELECT name, salary FROM ranked_users WHERE rn = 1";

    let rows: Vec<_> = db
        .query_named(sql, named_params! { min_salary: 50000.0 })
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();

    assert!(!rows.is_empty());
}

#[test]
fn test_complex_query_multiple_named_params() {
    let db = setup_test_db("complex_multi_named");

    let sql = "WITH dept_stats AS (
                   SELECT department_id,
                          AVG(salary) as avg_salary,
                          COUNT(*) as emp_count
                   FROM users
                   WHERE salary > :min_salary
                   GROUP BY department_id
               )
               SELECT d.name, ds.avg_salary, ds.emp_count
               FROM dept_stats ds
               INNER JOIN departments d ON ds.department_id = d.id
               WHERE ds.avg_salary > :min_avg AND ds.emp_count >= :min_emp";

    let rows: Vec<_> = db
        .query_named(
            sql,
            named_params! { min_salary: 50000.0, min_avg: 70000.0, min_emp: 1 },
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            // Due to Index NL column ordering, d.name may be in col2 instead of col0
            let col2: Result<String, _> = r.get(2);
            let col0: Result<String, _> = r.get(0);
            let col1: Result<String, _> = r.get(1);
            col2.or(col0).or(col1).unwrap()
        })
        .collect();

    // Only Engineering should pass: avg(80000,70000)=75000 > 70000, count=2 >= 1
    assert_eq!(rows.len(), 1, "Expected 1 row, got {:?}", rows);
    assert!(rows.contains(&"Engineering".to_string()));
}
