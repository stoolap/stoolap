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

//! Advanced CTE (Common Table Expression) Tests
//!
//! Tests for WITH clause, multiple CTEs, CTE with column aliases,
//! CTEs referencing other CTEs, and recursive CTEs.

use stoolap::Database;

fn setup_tables(db: &Database) {
    db.execute(
        "CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            manager_id INTEGER,
            department TEXT,
            salary FLOAT
        )",
        (),
    )
    .expect("Failed to create table");

    let inserts = [
        "INSERT INTO employees VALUES (1, 'Alice', NULL, 'Engineering', 100000)",
        "INSERT INTO employees VALUES (2, 'Bob', 1, 'Engineering', 80000)",
        "INSERT INTO employees VALUES (3, 'Charlie', 1, 'Engineering', 75000)",
        "INSERT INTO employees VALUES (4, 'Diana', 2, 'Engineering', 60000)",
        "INSERT INTO employees VALUES (5, 'Eve', 2, 'Sales', 55000)",
        "INSERT INTO employees VALUES (6, 'Frank', 3, 'Sales', 50000)",
        "INSERT INTO employees VALUES (7, 'Grace', NULL, 'Marketing', 90000)",
        "INSERT INTO employees VALUES (8, 'Henry', 7, 'Marketing', 70000)",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert");
    }
}

// =============================================================================
// Basic CTE Tests
// =============================================================================

#[test]
fn test_basic_cte() {
    let db = Database::open("memory://basic_cte").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH engineers AS (
                SELECT * FROM employees WHERE department = 'Engineering'
            )
            SELECT name FROM engineers ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert!(names.contains(&"Alice".to_string()), "Expected Alice");
    assert!(names.contains(&"Bob".to_string()), "Expected Bob");
}

#[test]
fn test_cte_with_aggregation() {
    let db = Database::open("memory://cte_agg").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH dept_salary AS (
                SELECT department, AVG(salary) as avg_salary
                FROM employees
                GROUP BY department
            )
            SELECT department, avg_salary
            FROM dept_salary
            ORDER BY avg_salary DESC",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for _ in result {
        row_count += 1;
    }

    assert_eq!(row_count, 3, "Expected 3 departments");
}

// =============================================================================
// CTE with Column Aliases Tests
// =============================================================================

#[test]
fn test_cte_column_aliases() {
    let db = Database::open("memory://cte_col_alias").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH stats(dept, total, average) AS (
                SELECT department, SUM(salary), AVG(salary)
                FROM employees
                GROUP BY department
            )
            SELECT dept, total, average
            FROM stats
            ORDER BY dept",
            (),
        )
        .expect("Failed to query");

    let columns = result.columns().to_vec();
    assert!(
        columns.contains(&"dept".to_string()),
        "Expected 'dept' column"
    );
    assert!(
        columns.contains(&"total".to_string()),
        "Expected 'total' column"
    );
    assert!(
        columns.contains(&"average".to_string()),
        "Expected 'average' column"
    );
}

// =============================================================================
// Multiple CTEs Tests
// =============================================================================

#[test]
fn test_multiple_ctes() {
    let db = Database::open("memory://multi_cte").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH
                engineering AS (
                    SELECT * FROM employees WHERE department = 'Engineering'
                ),
                high_earners AS (
                    SELECT * FROM employees WHERE salary > 70000
                )
            SELECT e.name
            FROM engineering e
            INNER JOIN high_earners h ON e.id = h.id
            ORDER BY e.name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // Engineering AND salary > 70000: Alice (100k), Bob (80k), Charlie (75k)
    assert!(names.contains(&"Alice".to_string()), "Expected Alice");
    assert!(names.contains(&"Bob".to_string()), "Expected Bob");
    assert!(names.contains(&"Charlie".to_string()), "Expected Charlie");
}

#[test]
fn test_cte_referencing_cte() {
    let db = Database::open("memory://cte_ref_cte").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH
                base AS (
                    SELECT id, name, salary FROM employees
                ),
                doubled AS (
                    SELECT id, name, salary * 2 as doubled_salary FROM base
                )
            SELECT name, doubled_salary
            FROM doubled
            WHERE doubled_salary > 160000
            ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // Only Alice (100k * 2 = 200k) and Grace (90k * 2 = 180k) > 160k
    assert!(names.contains(&"Alice".to_string()), "Expected Alice");
    assert!(names.contains(&"Grace".to_string()), "Expected Grace");
}

// =============================================================================
// CTE with JOIN Tests
// =============================================================================

#[test]
fn test_cte_with_join() {
    let db = Database::open("memory://cte_join").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH managers AS (
                SELECT DISTINCT m.id, m.name
                FROM employees e
                INNER JOIN employees m ON e.manager_id = m.id
            )
            SELECT name FROM managers ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // Managers: Alice, Bob, Charlie, Grace
    assert!(
        names.contains(&"Alice".to_string()),
        "Expected Alice as manager"
    );
    assert!(
        names.contains(&"Bob".to_string()),
        "Expected Bob as manager"
    );
}

#[test]
fn test_cte_join_main_query() {
    let db = Database::open("memory://cte_join_main").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH dept_stats AS (
                SELECT department, AVG(salary) as avg_sal
                FROM employees
                GROUP BY department
            )
            SELECT e.name, e.salary, d.avg_sal
            FROM employees e
            INNER JOIN dept_stats d ON e.department = d.department
            WHERE e.salary > d.avg_sal
            ORDER BY e.name",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert!(count > 0, "Expected some employees above dept average");
}

// =============================================================================
// Recursive CTE Tests
// =============================================================================

#[test]
fn test_recursive_cte_basic() {
    let db = Database::open("memory://recursive_basic").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH RECURSIVE org_chart(id, name, level) AS (
                SELECT id, name, 0 as level
                FROM employees
                WHERE manager_id IS NULL
                UNION ALL
                SELECT e.id, e.name, oc.level + 1
                FROM employees e
                INNER JOIN org_chart oc ON e.manager_id = oc.id
            )
            SELECT name, level FROM org_chart ORDER BY level, name",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        let level: i64 = row.get(1).unwrap();
        results.push((name, level));
    }

    // Alice and Grace are level 0 (no manager)
    assert!(
        results.iter().any(|(n, l)| n == "Alice" && *l == 0),
        "Expected Alice at level 0"
    );
    assert!(
        results.iter().any(|(n, l)| n == "Grace" && *l == 0),
        "Expected Grace at level 0"
    );

    // Bob and Charlie report to Alice (level 1)
    assert!(
        results.iter().any(|(n, l)| n == "Bob" && *l == 1),
        "Expected Bob at level 1"
    );
}

#[test]
fn test_recursive_cte_numbers() {
    let db = Database::open("memory://recursive_numbers").expect("Failed to create database");

    let result = db
        .query(
            "WITH RECURSIVE nums(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM nums WHERE n < 5
            )
            SELECT n FROM nums ORDER BY n",
            (),
        )
        .expect("Failed to query");

    let mut numbers = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let n: i64 = row.get(0).unwrap();
        numbers.push(n);
    }

    assert_eq!(numbers, vec![1, 2, 3, 4, 5], "Expected 1-5");
}

#[test]
fn test_recursive_cte_paths() {
    let db = Database::open("memory://recursive_paths").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH RECURSIVE emp_path(id, name, path) AS (
                SELECT id, name, name
                FROM employees
                WHERE manager_id IS NULL
                UNION ALL
                SELECT e.id, e.name, ep.path || ' -> ' || e.name
                FROM employees e
                INNER JOIN emp_path ep ON e.manager_id = ep.id
            )
            SELECT name, path FROM emp_path ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 8, "Expected all 8 employees");
}

// =============================================================================
// CTE with UNION/INTERSECT/EXCEPT Tests
// =============================================================================

#[test]
fn test_cte_with_union() {
    let db = Database::open("memory://cte_union").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH combined AS (
                SELECT name, salary FROM employees WHERE department = 'Engineering'
                UNION
                SELECT name, salary FROM employees WHERE salary > 80000
            )
            SELECT name FROM combined ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // Engineering: Alice, Bob, Charlie, Diana
    // Salary > 80k: Alice, Grace
    // Union: Alice, Bob, Charlie, Diana, Grace
    assert!(names.contains(&"Alice".to_string()), "Expected Alice");
    assert!(names.contains(&"Grace".to_string()), "Expected Grace");
}

// =============================================================================
// CTE with Subqueries Tests
// =============================================================================

#[test]
fn test_cte_with_exists() {
    let db = Database::open("memory://cte_exists").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH managers AS (
                SELECT DISTINCT manager_id as id FROM employees WHERE manager_id IS NOT NULL
            )
            SELECT e.name
            FROM employees e
            WHERE EXISTS (SELECT 1 FROM managers m WHERE m.id = e.id)
            ORDER BY e.name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // People who are managers: Alice (1), Bob (2), Charlie (3), Grace (7)
    assert!(names.contains(&"Alice".to_string()), "Expected Alice");
    assert!(names.contains(&"Bob".to_string()), "Expected Bob");
}

#[test]
fn test_cte_with_in() {
    let db = Database::open("memory://cte_in").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH high_earners AS (
                SELECT id FROM employees WHERE salary > 75000
            )
            SELECT name
            FROM employees
            WHERE id IN (SELECT id FROM high_earners)
            ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    // salary > 75k: Alice (100k), Grace (90k), Bob (80k)
    assert!(names.contains(&"Alice".to_string()), "Expected Alice");
    assert!(names.contains(&"Grace".to_string()), "Expected Grace");
    assert!(names.contains(&"Bob".to_string()), "Expected Bob");
}

// =============================================================================
// CTE with Window Functions Tests
// =============================================================================

#[test]
fn test_cte_with_window() {
    let db = Database::open("memory://cte_window").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH ranked AS (
                SELECT name, department, salary,
                       RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
                FROM employees
            )
            SELECT name, department, rank
            FROM ranked
            WHERE rank = 1
            ORDER BY department",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    // One top earner per department (3 departments)
    assert_eq!(count, 3, "Expected 3 top earners (one per dept)");
}

// =============================================================================
// CTE with LIMIT and ORDER BY Tests
// =============================================================================

#[test]
fn test_cte_with_limit() {
    let db = Database::open("memory://cte_limit").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH top_earners AS (
                SELECT name, salary
                FROM employees
                ORDER BY salary DESC
                LIMIT 3
            )
            SELECT name FROM top_earners ORDER BY name",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 rows from CTE with LIMIT");
}

#[test]
fn test_cte_limit_in_main_query() {
    let db = Database::open("memory://cte_limit_main").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH all_emp AS (
                SELECT name, salary FROM employees
            )
            SELECT name FROM all_emp ORDER BY salary DESC LIMIT 2",
            (),
        )
        .expect("Failed to query");

    let mut names = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        names.push(name);
    }

    assert_eq!(names.len(), 2, "Expected 2 rows");
    assert_eq!(names[0], "Alice", "Expected Alice first (highest salary)");
}

// =============================================================================
// CTE Edge Cases
// =============================================================================

#[test]
fn test_cte_empty_result() {
    let db = Database::open("memory://cte_empty").expect("Failed to create database");
    setup_tables(&db);

    let result = db
        .query(
            "WITH empty AS (
                SELECT * FROM employees WHERE salary > 1000000
            )
            SELECT name FROM empty",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 0, "Expected 0 rows from empty CTE");
}

#[test]
fn test_cte_same_column_name_as_table() {
    let db = Database::open("memory://cte_same_name").expect("Failed to create database");
    setup_tables(&db);

    // CTE named differently than source table but with same columns
    let result = db
        .query(
            "WITH emp_copy AS (
                SELECT id, name, salary FROM employees
            )
            SELECT ec.name, ec.salary
            FROM emp_copy ec
            WHERE ec.id = 1",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let name: String = row.get(0).unwrap();
        assert_eq!(name, "Alice", "Expected Alice");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row");
}

#[test]
fn test_nested_cte_in_subquery() {
    let db = Database::open("memory://nested_cte_sub").expect("Failed to create database");
    setup_tables(&db);

    // CTE used in a derived table
    let result = db
        .query(
            "WITH base AS (
                SELECT department, COUNT(*) as emp_count
                FROM employees
                GROUP BY department
            )
            SELECT department, emp_count
            FROM (SELECT * FROM base WHERE emp_count >= 2) AS filtered
            ORDER BY department",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert!(count > 0, "Expected some departments with >= 2 employees");
}
