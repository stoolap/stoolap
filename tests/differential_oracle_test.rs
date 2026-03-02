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

#![cfg(feature = "sqlite")]

//! Differential Oracle Tests
//!
//! Compares Stoolap query results against SQLite for correctness validation.
//! Both databases run in-memory and execute identical SQL statements, then
//! results are normalized and compared for equivalence.

use rusqlite::Connection;
use stoolap::Database;
use stoolap::Value;

/// Create both an in-memory Stoolap database and an in-memory SQLite database.
fn setup_both() -> (Database, Connection) {
    let stoolap = Database::open_in_memory().expect("Failed to create Stoolap database");
    let sqlite = Connection::open_in_memory().expect("Failed to create SQLite database");
    (stoolap, sqlite)
}

/// Execute a DDL or DML statement on both databases.
fn exec_both(stoolap: &Database, sqlite: &Connection, sql: &str) {
    stoolap
        .execute(sql, ())
        .unwrap_or_else(|e| panic!("Stoolap failed on '{}': {}", sql, e));
    sqlite
        .execute_batch(sql)
        .unwrap_or_else(|e| panic!("SQLite failed on '{}': {}", sql, e));
}

/// Query Stoolap and return results as a Vec of Vec<String>.
///
/// Value conversion:
/// - Integer: to_string()
/// - Float: format with 3 decimal places
/// - Boolean: "1" for true, "0" for false (to match SQLite)
/// - Text: as-is
/// - Null: "NULL"
/// - Timestamp: to_string()
/// - Extension: to_string()
fn query_stoolap(db: &Database, sql: &str) -> Vec<Vec<String>> {
    let mut rows = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("Stoolap query failed on '{}': {}", sql, e));
    let num_cols = rows.columns().len();
    let mut result: Vec<Vec<String>> = Vec::new();

    for row in &mut rows {
        let row = row.unwrap_or_else(|e| panic!("Stoolap row error on '{}': {}", sql, e));
        let mut string_row = Vec::with_capacity(num_cols);
        for i in 0..num_cols {
            let value = row.get_value(i);
            let s = match value {
                Some(Value::Integer(v)) => v.to_string(),
                Some(Value::Float(v)) => format!("{:.3}", v),
                Some(Value::Boolean(b)) => {
                    if *b {
                        "1".to_string()
                    } else {
                        "0".to_string()
                    }
                }
                Some(Value::Text(t)) => t.to_string(),
                Some(Value::Null(_)) | None => "NULL".to_string(),
                Some(Value::Timestamp(ts)) => ts.to_string(),
                Some(Value::Extension(_)) => value.unwrap().to_string(),
            };
            string_row.push(s);
        }
        result.push(string_row);
    }
    result
}

/// Query SQLite and return results as a Vec of Vec<String>.
///
/// For each column, try reading as i64, then f64, then String.
/// Floats are formatted with 3 decimal places. NULL becomes "NULL".
fn query_sqlite(conn: &Connection, sql: &str) -> Vec<Vec<String>> {
    let mut stmt = conn
        .prepare(sql)
        .unwrap_or_else(|e| panic!("SQLite prepare failed on '{}': {}", sql, e));
    let col_count = stmt.column_count();

    let rows = stmt
        .query_map([], |row| {
            let mut string_row = Vec::with_capacity(col_count);
            for i in 0..col_count {
                let val: rusqlite::types::Value = row.get(i).unwrap();
                let s = match val {
                    rusqlite::types::Value::Null => "NULL".to_string(),
                    rusqlite::types::Value::Integer(v) => v.to_string(),
                    rusqlite::types::Value::Real(v) => format!("{:.3}", v),
                    rusqlite::types::Value::Text(ref t) => t.clone(),
                    rusqlite::types::Value::Blob(_) => "<blob>".to_string(),
                };
                string_row.push(s);
            }
            Ok(string_row)
        })
        .unwrap_or_else(|e| panic!("SQLite query_map failed on '{}': {}", sql, e));

    rows.map(|r| r.expect("SQLite row error")).collect()
}

/// Normalize a single cell value for comparison.
/// - Normalize numeric representations so "30" and "30.000" are comparable
///
/// Boolean normalization is NOT done here because Stoolap already converts
/// `Value::Boolean` to "1"/"0" at extraction time. Normalizing text "true"/"false"
/// globally would mask real type/value differences if a TEXT column contained those literals.
fn normalize_cell(s: &str) -> String {
    if s == "NULL" {
        return "NULL".to_string();
    }
    // Try to parse as f64 for numeric normalization
    if let Ok(f) = s.parse::<f64>() {
        // Check if this is an integer-like value (no fractional part)
        if f == f.trunc() && !s.contains('.') {
            return format!("{}", f as i64);
        }
        return format!("{:.3}", f);
    }
    s.to_string()
}

/// Compare two cell values with numeric tolerance.
/// Returns true if the cells are equivalent.
fn cells_equal(a: &str, b: &str) -> bool {
    let na = normalize_cell(a);
    let nb = normalize_cell(b);
    if na == nb {
        return true;
    }
    // Try numeric comparison with epsilon
    if let (Ok(fa), Ok(fb)) = (na.parse::<f64>(), nb.parse::<f64>()) {
        let epsilon = 1e-3;
        return (fa - fb).abs() < epsilon;
    }
    false
}

/// Normalize rows (sort lexicographically) and compare the two result sets.
/// If they differ, panic with a descriptive error showing the SQL and both result sets.
fn normalize_and_compare(sql: &str, stoolap_rows: Vec<Vec<String>>, sqlite_rows: Vec<Vec<String>>) {
    // Sort both row sets for order-independent comparison
    let mut stoolap_sorted: Vec<Vec<String>> = stoolap_rows
        .iter()
        .map(|row| row.iter().map(|c| normalize_cell(c)).collect())
        .collect();
    stoolap_sorted.sort();

    let mut sqlite_sorted: Vec<Vec<String>> = sqlite_rows
        .iter()
        .map(|row| row.iter().map(|c| normalize_cell(c)).collect())
        .collect();
    sqlite_sorted.sort();

    // Check row count
    if stoolap_sorted.len() != sqlite_sorted.len() {
        panic!(
            "Row count mismatch for SQL: {}\n  Stoolap rows: {}\n  SQLite rows: {}\n  Stoolap: {:?}\n  SQLite:  {:?}",
            sql,
            stoolap_sorted.len(),
            sqlite_sorted.len(),
            stoolap_sorted,
            sqlite_sorted
        );
    }

    // Compare each row cell by cell
    for (row_idx, (s_row, q_row)) in stoolap_sorted.iter().zip(sqlite_sorted.iter()).enumerate() {
        if s_row.len() != q_row.len() {
            panic!(
                "Column count mismatch at row {} for SQL: {}\n  Stoolap: {:?}\n  SQLite:  {:?}",
                row_idx, sql, s_row, q_row
            );
        }
        for (col_idx, (s_cell, q_cell)) in s_row.iter().zip(q_row.iter()).enumerate() {
            if !cells_equal(s_cell, q_cell) {
                panic!(
                    "Value mismatch at row {}, col {} for SQL: {}\n  Stoolap cell: {:?}\n  SQLite cell:  {:?}\n  Stoolap full row: {:?}\n  SQLite full row:  {:?}\n  All Stoolap rows: {:?}\n  All SQLite rows:  {:?}",
                    row_idx, col_idx, sql, s_cell, q_cell, s_row, q_row, stoolap_sorted, sqlite_sorted
                );
            }
        }
    }
}

/// Compare two ordered result sets row-by-row without re-sorting.
/// Use this for ORDER BY queries where row order matters.
fn assert_ordered_equal(sql: &str, stoolap_rows: &[Vec<String>], sqlite_rows: &[Vec<String>]) {
    assert_eq!(
        stoolap_rows.len(),
        sqlite_rows.len(),
        "Row count mismatch for: {}\n  Stoolap: {:?}\n  SQLite:  {:?}",
        sql,
        stoolap_rows,
        sqlite_rows
    );
    for (i, (s, q)) in stoolap_rows.iter().zip(sqlite_rows.iter()).enumerate() {
        assert_eq!(
            s.len(),
            q.len(),
            "Column count mismatch at row {} for: {}",
            i,
            sql
        );
        for (j, (sc, qc)) in s.iter().zip(q.iter()).enumerate() {
            assert!(
                cells_equal(sc, qc),
                "Mismatch at row {}, col {} for SQL: {}\n  Stoolap: {:?}\n  SQLite:  {:?}",
                i,
                j,
                sql,
                stoolap_rows,
                sqlite_rows
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test functions
// ---------------------------------------------------------------------------

#[test]
fn test_oracle_dml() {
    let (stoolap, sqlite) = setup_both();

    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
    );
    exec_both(
        &stoolap,
        &sqlite,
        "INSERT INTO t1 VALUES (1, 'Alice', 100), (2, 'Bob', 200), (3, 'Carol', 300)",
    );

    // SELECT * after initial insert
    let sql = "SELECT * FROM t1 ORDER BY id";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // UPDATE
    exec_both(
        &stoolap,
        &sqlite,
        "UPDATE t1 SET value = value + 10 WHERE id = 2",
    );

    let sql = "SELECT * FROM t1 ORDER BY id";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // DELETE
    exec_both(&stoolap, &sqlite, "DELETE FROM t1 WHERE id = 3");

    let sql = "SELECT COUNT(*) FROM t1";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);
}

#[test]
fn test_oracle_select_where() {
    let (stoolap, sqlite) = setup_both();

    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, category TEXT, value INTEGER, flag INTEGER)",
    );

    // Insert 10 rows with various data
    let inserts = [
        "INSERT INTO t VALUES (1, 'Alice', 'A', 10, 1)",
        "INSERT INTO t VALUES (2, 'Bob', 'B', 20, 0)",
        "INSERT INTO t VALUES (3, 'Carol', 'A', 30, 1)",
        "INSERT INTO t VALUES (4, 'Dave', 'C', 40, 0)",
        "INSERT INTO t VALUES (5, 'Eve', 'B', 50, 1)",
        "INSERT INTO t VALUES (6, 'Frank', 'A', 60, 0)",
        "INSERT INTO t VALUES (7, 'Grace', 'C', 70, 1)",
        "INSERT INTO t VALUES (8, 'Hank', 'B', 80, 0)",
        "INSERT INTO t VALUES (9, 'Ivy', 'A', 90, 1)",
        "INSERT INTO t VALUES (10, 'Jack', 'C', NULL, 0)",
    ];
    for insert in &inserts {
        exec_both(&stoolap, &sqlite, insert);
    }

    let queries = [
        "SELECT * FROM t WHERE value > 50",
        "SELECT * FROM t WHERE category = 'A'",
        "SELECT * FROM t WHERE value BETWEEN 20 AND 80",
        "SELECT * FROM t WHERE category IN ('A', 'C')",
        "SELECT * FROM t WHERE name LIKE 'A%'",
        "SELECT * FROM t WHERE value IS NULL",
        "SELECT * FROM t WHERE value IS NOT NULL",
        "SELECT * FROM t WHERE value > 30 AND category = 'B'",
        "SELECT * FROM t WHERE value > 90 OR category = 'A'",
        "SELECT id, name FROM t WHERE value IS NOT NULL ORDER BY value DESC LIMIT 3",
    ];

    for sql in &queries {
        let sr = query_stoolap(&stoolap, sql);
        let qr = query_sqlite(&sqlite, sql);
        normalize_and_compare(sql, sr, qr);
    }

    // LIMIT with OFFSET: Stoolap and SQLite both support this syntax
    let sql = "SELECT id, name FROM t WHERE value IS NOT NULL ORDER BY value ASC LIMIT 3 OFFSET 2";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);
}

#[test]
fn test_oracle_aggregates() {
    let (stoolap, sqlite) = setup_both();

    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, category TEXT, value INTEGER)",
    );

    let inserts = [
        "INSERT INTO t VALUES (1, 'A', 10)",
        "INSERT INTO t VALUES (2, 'A', 20)",
        "INSERT INTO t VALUES (3, 'B', 30)",
        "INSERT INTO t VALUES (4, 'B', 40)",
        "INSERT INTO t VALUES (5, 'B', 50)",
        "INSERT INTO t VALUES (6, 'C', 60)",
    ];
    for insert in &inserts {
        exec_both(&stoolap, &sqlite, insert);
    }

    // Basic aggregates without GROUP BY
    let sql = "SELECT COUNT(*), SUM(value), MIN(value), MAX(value) FROM t";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // GROUP BY count
    let sql = "SELECT category, COUNT(*) FROM t GROUP BY category";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // GROUP BY avg
    let sql = "SELECT category, AVG(value) FROM t GROUP BY category";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // HAVING
    let sql = "SELECT category, COUNT(*) FROM t GROUP BY category HAVING COUNT(*) > 1";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // COUNT DISTINCT
    let sql = "SELECT COUNT(DISTINCT category) FROM t";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // Empty result set
    let sql = "SELECT COUNT(*) FROM t WHERE id = 999";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // NULL result from aggregate on empty set
    let sql = "SELECT SUM(value) FROM t WHERE id = 999";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);
}

#[test]
fn test_oracle_joins() {
    let (stoolap, sqlite) = setup_both();

    // Create employees table
    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE emp (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER)",
    );
    // Create departments table
    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE dept (id INTEGER PRIMARY KEY, dept_name TEXT)",
    );
    // Create projects table for 3-way join
    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE proj (id INTEGER PRIMARY KEY, proj_name TEXT, dept_id INTEGER)",
    );

    // Insert departments
    let dept_inserts = [
        "INSERT INTO dept VALUES (1, 'Engineering')",
        "INSERT INTO dept VALUES (2, 'Marketing')",
        "INSERT INTO dept VALUES (3, 'Sales')",
    ];
    for ins in &dept_inserts {
        exec_both(&stoolap, &sqlite, ins);
    }

    // Insert employees (some with dept_id that exists, one with NULL dept)
    let emp_inserts = [
        "INSERT INTO emp VALUES (1, 'Alice', 1)",
        "INSERT INTO emp VALUES (2, 'Bob', 2)",
        "INSERT INTO emp VALUES (3, 'Carol', 1)",
        "INSERT INTO emp VALUES (4, 'Dave', NULL)",
        "INSERT INTO emp VALUES (5, 'Eve', 3)",
    ];
    for ins in &emp_inserts {
        exec_both(&stoolap, &sqlite, ins);
    }

    // Insert projects
    let proj_inserts = [
        "INSERT INTO proj VALUES (1, 'ProjectX', 1)",
        "INSERT INTO proj VALUES (2, 'ProjectY', 2)",
    ];
    for ins in &proj_inserts {
        exec_both(&stoolap, &sqlite, ins);
    }

    // INNER JOIN
    let sql = "SELECT e.name, d.dept_name FROM emp e INNER JOIN dept d ON e.dept_id = d.id";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // LEFT JOIN (Dave should have NULL dept_name)
    let sql = "SELECT e.name, d.dept_name FROM emp e LEFT JOIN dept d ON e.dept_id = d.id";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // CROSS JOIN with ORDER BY for deterministic output
    let sql =
        "SELECT e.name, d.dept_name FROM emp e CROSS JOIN dept d ORDER BY e.name, d.dept_name";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // Multi-table join (3 tables)
    let sql = "SELECT e.name, d.dept_name, p.proj_name FROM emp e INNER JOIN dept d ON e.dept_id = d.id INNER JOIN proj p ON d.id = p.dept_id";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);
}

#[test]
fn test_oracle_subqueries() {
    let (stoolap, sqlite) = setup_both();

    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, value INTEGER)",
    );
    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, ref_id INTEGER)",
    );

    let t_inserts = [
        "INSERT INTO t VALUES (1, 10)",
        "INSERT INTO t VALUES (2, 50)",
        "INSERT INTO t VALUES (3, 80)",
        "INSERT INTO t VALUES (4, 30)",
    ];
    for ins in &t_inserts {
        exec_both(&stoolap, &sqlite, ins);
    }

    let t2_inserts = [
        "INSERT INTO t2 VALUES (1, 1)",
        "INSERT INTO t2 VALUES (2, 1)",
        "INSERT INTO t2 VALUES (3, 3)",
    ];
    for ins in &t2_inserts {
        exec_both(&stoolap, &sqlite, ins);
    }

    // IN subquery
    let sql = "SELECT * FROM t WHERE id IN (SELECT id FROM t WHERE value > 50)";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // EXISTS subquery
    let sql = "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.ref_id = t.id)";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // Scalar subquery
    let sql = "SELECT id, (SELECT COUNT(*) FROM t2 WHERE t2.ref_id = t.id) AS cnt FROM t";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);
}

#[test]
fn test_oracle_expressions() {
    let (stoolap, sqlite) = setup_both();

    // Arithmetic expressions
    let sql = "SELECT 2 + 3, 10 - 4, 3 * 7, 15 / 4, 17 % 5";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // CASE expression
    let sql = "SELECT CASE WHEN 5 > 3 THEN 'yes' ELSE 'no' END";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // COALESCE
    let sql = "SELECT COALESCE(NULL, NULL, 'default')";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // NULLIF
    let sql = "SELECT NULLIF(10, 10), NULLIF(10, 20)";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // CAST
    let sql = "SELECT CAST(42 AS TEXT), CAST('123' AS INTEGER)";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);
}

#[test]
fn test_oracle_string_functions() {
    let (stoolap, sqlite) = setup_both();

    // UPPER and LOWER
    let sql = "SELECT UPPER('hello'), LOWER('HELLO')";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // LENGTH
    let sql = "SELECT LENGTH('hello'), LENGTH('')";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // SUBSTRING vs SUBSTR: run different SQL on each engine, compare results
    let stoolap_sql = "SELECT SUBSTRING('hello world', 1, 5)";
    let sqlite_sql = "SELECT SUBSTR('hello world', 1, 5)";
    let sr = query_stoolap(&stoolap, stoolap_sql);
    let qr = query_sqlite(&sqlite, sqlite_sql);
    normalize_and_compare("SUBSTRING/SUBSTR('hello world', 1, 5)", sr, qr);

    // REPLACE
    let sql = "SELECT REPLACE('hello world', 'world', 'there')";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // TRIM
    let sql = "SELECT TRIM('  hello  ')";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);
}

#[test]
fn test_oracle_set_operations() {
    let (stoolap, sqlite) = setup_both();

    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE t1 (id INTEGER PRIMARY KEY)",
    );
    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY)",
    );

    // t1: {1, 2, 3, 4}
    let t1_inserts = [
        "INSERT INTO t1 VALUES (1)",
        "INSERT INTO t1 VALUES (2)",
        "INSERT INTO t1 VALUES (3)",
        "INSERT INTO t1 VALUES (4)",
    ];
    for ins in &t1_inserts {
        exec_both(&stoolap, &sqlite, ins);
    }

    // t2: {3, 4, 5, 6}
    let t2_inserts = [
        "INSERT INTO t2 VALUES (3)",
        "INSERT INTO t2 VALUES (4)",
        "INSERT INTO t2 VALUES (5)",
        "INSERT INTO t2 VALUES (6)",
    ];
    for ins in &t2_inserts {
        exec_both(&stoolap, &sqlite, ins);
    }

    // UNION (deduplicates)
    let sql = "SELECT id FROM t1 UNION SELECT id FROM t2";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // UNION ALL (preserves duplicates)
    let sql = "SELECT id FROM t1 UNION ALL SELECT id FROM t2";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // INTERSECT
    let sql = "SELECT id FROM t1 INTERSECT SELECT id FROM t2";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);

    // EXCEPT
    let sql = "SELECT id FROM t1 EXCEPT SELECT id FROM t2";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    normalize_and_compare(sql, sr, qr);
}

#[test]
fn test_oracle_order_by() {
    let (stoolap, sqlite) = setup_both();

    exec_both(
        &stoolap,
        &sqlite,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score INTEGER, grade TEXT)",
    );

    let inserts = [
        "INSERT INTO t VALUES (1, 'Alice', 85, 'B')",
        "INSERT INTO t VALUES (2, 'Bob', 92, 'A')",
        "INSERT INTO t VALUES (3, 'Carol', 78, 'C')",
        "INSERT INTO t VALUES (4, 'Dave', 92, 'A')",
        "INSERT INTO t VALUES (5, 'Eve', NULL, 'B')",
        "INSERT INTO t VALUES (6, 'Frank', 65, 'D')",
    ];
    for ins in &inserts {
        exec_both(&stoolap, &sqlite, ins);
    }

    // ORDER BY single column ASC - add , id tiebreaker for deterministic order on score ties (92, 92)
    let sql = "SELECT id, name, score FROM t WHERE score IS NOT NULL ORDER BY score ASC, id ASC";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    assert_ordered_equal(sql, &sr, &qr);

    // ORDER BY single column DESC - add , id tiebreaker
    let sql = "SELECT id, name, score FROM t WHERE score IS NOT NULL ORDER BY score DESC, id ASC";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    assert_ordered_equal(sql, &sr, &qr);

    // ORDER BY multiple columns - add , id tiebreaker within same (grade, score)
    let sql = "SELECT id, name, score, grade FROM t WHERE score IS NOT NULL ORDER BY grade ASC, score DESC, id ASC";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    assert_ordered_equal(sql, &sr, &qr);

    // ORDER BY with LIMIT - add , id tiebreaker
    let sql = "SELECT id, name FROM t WHERE score IS NOT NULL ORDER BY score DESC, id ASC LIMIT 3";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    assert_ordered_equal(sql, &sr, &qr);

    // ORDER BY with NULLs - add , id tiebreaker, compare row-by-row (not re-sorted)
    let sql = "SELECT id, name, score FROM t ORDER BY score ASC, id ASC";
    let sr = query_stoolap(&stoolap, sql);
    let qr = query_sqlite(&sqlite, sql);
    assert_ordered_equal(sql, &sr, &qr);
}
