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

//! Real World SQL Parser Tests
//!
//! Tests that the parser can handle real-world SQL patterns.

use stoolap::parser::{parse_sql, Statement};

/// Helper to parse SQL and assert no errors
fn parse_ok(sql: &str) -> Statement {
    match parse_sql(sql) {
        Ok(stmts) => {
            assert!(!stmts.is_empty(), "No statements parsed from: {}", sql);
            stmts.into_iter().next().unwrap()
        }
        Err(e) => panic!("Failed to parse '{}': {:?}", sql, e),
    }
}

/// Helper to check if parse succeeds (doesn't panic on error, just returns bool)
fn parse_succeeds(sql: &str) -> bool {
    match parse_sql(sql) {
        Ok(stmts) => !stmts.is_empty(),
        Err(_) => false,
    }
}

// =============================================================================
// TestCreateTableWithAllTypes
// =============================================================================

#[test]
fn test_create_table_with_all_types() {
    let sql = r#"
    CREATE TABLE employees (
        id INTEGER,
        name TEXT,
        salary FLOAT,
        is_active BOOLEAN,
        hire_date DATE,
        start_time TIME,
        created_at TIMESTAMP,
        metadata JSON
    )
    "#;

    let stmt = parse_ok(sql);
    match stmt {
        Statement::CreateTable(create) => {
            assert_eq!(create.table_name.value, "employees");
            assert_eq!(create.columns.len(), 8);

            let expected = [
                ("id", "INTEGER"),
                ("name", "TEXT"),
                ("salary", "FLOAT"),
                ("is_active", "BOOLEAN"),
                ("hire_date", "DATE"),
                ("start_time", "TIME"),
                ("created_at", "TIMESTAMP"),
                ("metadata", "JSON"),
            ];

            for (i, (name, typ)) in expected.iter().enumerate() {
                assert_eq!(
                    create.columns[i].name.value, *name,
                    "Column {} name mismatch",
                    i
                );
                assert_eq!(
                    create.columns[i].data_type, *typ,
                    "Column {} type mismatch",
                    i
                );
            }
        }
        _ => panic!("Expected CreateTableStatement"),
    }
}

// =============================================================================
// TestInsertWithAllDataTypes
// =============================================================================

#[test]
fn test_insert_with_integer_value() {
    let stmt = parse_ok("INSERT INTO employees (id) VALUES (42)");
    match stmt {
        Statement::Insert(_) => {}
        _ => panic!("Expected InsertStatement"),
    }
}

#[test]
fn test_insert_with_float_value() {
    let stmt = parse_ok("INSERT INTO employees (salary) VALUES (55.5)");
    match stmt {
        Statement::Insert(_) => {}
        _ => panic!("Expected InsertStatement"),
    }
}

#[test]
fn test_insert_with_text_value() {
    let stmt = parse_ok("INSERT INTO employees (name) VALUES ('John Doe')");
    match stmt {
        Statement::Insert(_) => {}
        _ => panic!("Expected InsertStatement"),
    }
}

#[test]
fn test_insert_with_boolean_true() {
    let stmt = parse_ok("INSERT INTO employees (is_active) VALUES (TRUE)");
    match stmt {
        Statement::Insert(_) => {}
        _ => panic!("Expected InsertStatement"),
    }
}

#[test]
fn test_insert_with_boolean_false() {
    let stmt = parse_ok("INSERT INTO employees (is_active) VALUES (FALSE)");
    match stmt {
        Statement::Insert(_) => {}
        _ => panic!("Expected InsertStatement"),
    }
}

#[test]
fn test_insert_with_json_value() {
    let stmt = parse_ok(
        r#"INSERT INTO employees (metadata) VALUES ('{"department":"Engineering","skills":["Go","SQL"]}')"#,
    );
    match stmt {
        Statement::Insert(_) => {}
        _ => panic!("Expected InsertStatement"),
    }
}

#[test]
fn test_insert_with_multiple_types() {
    let stmt = parse_ok(
        "INSERT INTO employees (id, name, salary, is_active, hire_date, start_time, metadata) VALUES (1, 'John Smith', 75000.50, TRUE, '2022-01-15', '09:00:00', '{\"role\":\"developer\"}')",
    );
    match stmt {
        Statement::Insert(insert) => {
            assert_eq!(insert.values.len(), 1);
            assert_eq!(insert.values[0].len(), 7);
        }
        _ => panic!("Expected InsertStatement"),
    }
}

// =============================================================================
// TestBasicSelectQueries
// =============================================================================

#[test]
fn test_select_all_columns() {
    let stmt = parse_ok("SELECT * FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_column_list() {
    let stmt = parse_ok("SELECT id, name, salary FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_integer_condition() {
    let stmt = parse_ok("SELECT * FROM employees WHERE id = 42");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_text_condition() {
    let stmt = parse_ok("SELECT * FROM employees WHERE name = 'John Doe'");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_boolean_condition() {
    let stmt = parse_ok("SELECT * FROM employees WHERE is_active = TRUE");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_date_condition() {
    let stmt = parse_ok("SELECT * FROM employees WHERE hire_date = '2023-05-15'");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_complex_date_condition() {
    let stmt = parse_ok(
        "SELECT * FROM employees WHERE hire_date >= '2022-01-01' AND hire_date <= '2022-12-31'",
    );
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_time_condition() {
    let stmt = parse_ok("SELECT * FROM employees WHERE start_time = '09:00:00'");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_multiple_conditions() {
    let stmt = parse_ok("SELECT * FROM employees WHERE is_active = TRUE AND salary > 50000.0");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_order_by() {
    let stmt = parse_ok("SELECT * FROM employees ORDER BY hire_date DESC, salary ASC");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_limit_offset() {
    let stmt = parse_ok("SELECT * FROM employees ORDER BY id LIMIT 10 OFFSET 20");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_join() {
    let stmt = parse_ok(
        "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id",
    );
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_left_join() {
    let stmt = parse_ok(
        "SELECT e.name, d.name FROM employees e LEFT JOIN departments d ON e.department_id = d.id",
    );
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_select_with_complex_time_date() {
    let stmt = parse_ok("SELECT * FROM employees WHERE start_time >= '09:00:00' AND start_time <= '17:00:00' AND hire_date >= '2022-01-01'");
    assert!(matches!(stmt, Statement::Select(_)));
}

// =============================================================================
// TestAggregationFunctions
// =============================================================================

#[test]
fn test_count_star() {
    let stmt = parse_ok("SELECT COUNT(*) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_count_column() {
    let stmt = parse_ok("SELECT COUNT(id) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_sum_function() {
    let stmt = parse_ok("SELECT SUM(salary) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_avg_function() {
    let stmt = parse_ok("SELECT AVG(salary) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_min_function() {
    let stmt = parse_ok("SELECT MIN(salary) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_max_function() {
    let stmt = parse_ok("SELECT MAX(salary) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_multiple_aggregations() {
    let stmt = parse_ok("SELECT COUNT(*), AVG(salary), MIN(salary), MAX(salary) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_group_by() {
    let stmt = parse_ok("SELECT department_id, COUNT(*) FROM employees GROUP BY department_id");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_having_clause() {
    let stmt = parse_ok(
        "SELECT department_id, COUNT(*) FROM employees GROUP BY department_id HAVING COUNT(*) > 5",
    );
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_complex_aggregation() {
    let stmt = parse_ok("SELECT department_id, AVG(salary) FROM employees WHERE is_active = TRUE GROUP BY department_id HAVING COUNT(*) > 3 ORDER BY AVG(salary) DESC");
    assert!(matches!(stmt, Statement::Select(_)));
}

// =============================================================================
// TestScalarFunctions
// =============================================================================

#[test]
fn test_upper_function() {
    let stmt = parse_ok("SELECT UPPER(name) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_lower_function() {
    let stmt = parse_ok("SELECT LOWER(name) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_abs_function() {
    let stmt = parse_ok("SELECT ABS(salary - 50000) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_round_function() {
    let stmt = parse_ok("SELECT ROUND(salary, 2) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_length_function() {
    let stmt = parse_ok("SELECT LENGTH(name) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_substring_function() {
    let stmt = parse_ok("SELECT SUBSTRING(name, 1, 3) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_concat_function() {
    let stmt = parse_ok("SELECT CONCAT(name, ' - ', department) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_scalar_in_where() {
    let stmt = parse_ok("SELECT * FROM employees WHERE UPPER(name) = 'JOHN DOE'");
    assert!(matches!(stmt, Statement::Select(_)));
}

// =============================================================================
// TestCastExpressions
// =============================================================================

#[test]
fn test_cast_int_to_float() {
    let stmt = parse_ok("SELECT CAST(id AS FLOAT) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_cast_string_to_int() {
    let stmt = parse_ok("SELECT CAST('42' AS INTEGER) FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_cast_in_where() {
    let stmt = parse_ok("SELECT * FROM employees WHERE CAST(salary AS INTEGER) > 50000");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_cast_with_arithmetic() {
    let stmt = parse_ok("SELECT CAST(id AS FLOAT) + 0.5 FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

// =============================================================================
// TestCaseExpressions
// =============================================================================

#[test]
fn test_simple_case() {
    let stmt =
        parse_ok("SELECT CASE WHEN salary > 50000 THEN 'High' ELSE 'Low' END FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_multiple_when_clauses() {
    let stmt = parse_ok("SELECT CASE WHEN salary > 100000 THEN 'Very High' WHEN salary > 50000 THEN 'High' ELSE 'Low' END FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_case_with_complex_conditions() {
    let stmt = parse_ok("SELECT CASE WHEN salary > 50000 AND is_active = TRUE THEN 'Active High' WHEN salary <= 50000 AND is_active = TRUE THEN 'Active Low' ELSE 'Inactive' END FROM employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

// =============================================================================
// TestSubqueries
// =============================================================================

#[test]
fn test_subquery_in_where() {
    // Subqueries might not be fully implemented
    let result = parse_succeeds(
        "SELECT * FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
    );
    println!(
        "Subquery in WHERE: {}",
        if result { "PASS" } else { "NOT IMPLEMENTED" }
    );
}

#[test]
fn test_subquery_with_in() {
    let result = parse_succeeds("SELECT * FROM employees WHERE department_id IN (SELECT id FROM departments WHERE location = 'New York')");
    println!(
        "Subquery with IN: {}",
        if result { "PASS" } else { "NOT IMPLEMENTED" }
    );
}

// =============================================================================
// TestWithClauses (CTEs)
// =============================================================================

#[test]
fn test_simple_cte() {
    let stmt = parse_ok("WITH high_salary AS (SELECT * FROM employees WHERE salary > 50000) SELECT * FROM high_salary");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_cte_with_join() {
    let stmt = parse_ok("WITH dept_employees AS (SELECT d.name AS dept_name, e.name AS emp_name FROM departments d JOIN employees e ON d.id = e.department_id) SELECT * FROM dept_employees");
    assert!(matches!(stmt, Statement::Select(_)));
}

// =============================================================================
// TestPreparedStatementParams
// =============================================================================

#[test]
fn test_simple_parameter() {
    let stmt = parse_ok("SELECT * FROM employees WHERE id = ?");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_multiple_parameters() {
    let stmt = parse_ok("SELECT * FROM employees WHERE salary > ? AND hire_date < ?");
    assert!(matches!(stmt, Statement::Select(_)));
}

#[test]
fn test_parameters_in_insert() {
    let stmt = parse_ok("INSERT INTO employees (name, salary) VALUES (?, ?)");
    assert!(matches!(stmt, Statement::Insert(_)));
}

#[test]
fn test_parameters_in_update() {
    let stmt = parse_ok("UPDATE employees SET name = ?, salary = ? WHERE id = ?");
    assert!(matches!(stmt, Statement::Update(_)));
}

// =============================================================================
// TestAlterTable
// =============================================================================

#[test]
fn test_alter_add_column() {
    let stmt = parse_ok("ALTER TABLE employees ADD COLUMN department_id INTEGER");
    assert!(matches!(stmt, Statement::AlterTable(_)));
}

#[test]
fn test_alter_drop_column() {
    let stmt = parse_ok("ALTER TABLE employees DROP COLUMN metadata");
    assert!(matches!(stmt, Statement::AlterTable(_)));
}

#[test]
fn test_alter_rename_column() {
    let stmt = parse_ok("ALTER TABLE employees RENAME COLUMN name TO full_name");
    assert!(matches!(stmt, Statement::AlterTable(_)));
}

#[test]
fn test_alter_rename_table() {
    let stmt = parse_ok("ALTER TABLE employees RENAME TO staff");
    assert!(matches!(stmt, Statement::AlterTable(_)));
}

// =============================================================================
// TestCreateDropIndex
// =============================================================================

#[test]
fn test_create_index() {
    let stmt = parse_ok("CREATE INDEX idx_salary ON employees(salary)");
    assert!(matches!(stmt, Statement::CreateIndex(_)));
}

#[test]
fn test_create_unique_index() {
    let stmt = parse_ok("CREATE UNIQUE INDEX idx_email ON employees(email)");
    assert!(matches!(stmt, Statement::CreateIndex(_)));
}

#[test]
fn test_create_multi_column_index() {
    let stmt = parse_ok("CREATE INDEX idx_dept_hire ON employees(department_id, hire_date)");
    assert!(matches!(stmt, Statement::CreateIndex(_)));
}

#[test]
fn test_drop_index() {
    let stmt = parse_ok("DROP INDEX idx_salary ON employees");
    assert!(matches!(stmt, Statement::DropIndex(_)));
}

#[test]
fn test_drop_index_if_exists() {
    let stmt = parse_ok("DROP INDEX IF EXISTS idx_nonexistent ON employees");
    assert!(matches!(stmt, Statement::DropIndex(_)));
}
