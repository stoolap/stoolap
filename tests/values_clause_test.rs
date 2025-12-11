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

//! Integration tests for VALUES clause as table expression

use stoolap::Database;

fn create_test_db(name: &str) -> Database {
    Database::open(&format!("memory://{}", name)).expect("Failed to create in-memory database")
}

// ============================================================================
// Basic VALUES Tests
// ============================================================================

#[test]
fn test_values_single_column() {
    let db = create_test_db("values_single");

    let result = db
        .query("SELECT * FROM (VALUES (1), (2), (3)) AS t(x)", ())
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_values_multiple_columns() {
    let db = create_test_db("values_multi");

    let result = db
        .query(
            "SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')) AS t(id, name)",
            (),
        )
        .unwrap();

    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        rows.push((id, name));
    }

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (1, "Alice".to_string()));
    assert_eq!(rows[1], (2, "Bob".to_string()));
    assert_eq!(rows[2], (3, "Charlie".to_string()));
}

#[test]
fn test_values_without_alias() {
    let db = create_test_db("values_no_alias");

    // Should use default column names: column1, column2, ...
    let result = db
        .query("SELECT * FROM (VALUES (10, 20), (30, 40)) AS t", ())
        .unwrap();

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let col1: i64 = row.get(0).unwrap();
        let col2: i64 = row.get(1).unwrap();
        rows.push((col1, col2));
    }

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], (10, 20));
    assert_eq!(rows[1], (30, 40));
}

// ============================================================================
// Column Alias Tests
// ============================================================================

#[test]
fn test_values_column_aliases() {
    let db = create_test_db("values_col_alias");

    let result = db
        .query(
            "SELECT id, name FROM (VALUES (1, 'Test')) AS t(id, name)",
            (),
        )
        .unwrap();

    for row in result {
        let row = row.unwrap();
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        assert_eq!(id, 1);
        assert_eq!(name, "Test");
    }
}

#[test]
fn test_values_qualified_column_reference() {
    let db = create_test_db("values_qualified");

    let result = db
        .query(
            "SELECT t.id, t.name FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)",
            (),
        )
        .unwrap();

    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let id: i64 = row.get(0).unwrap();
        let name: String = row.get(1).unwrap();
        rows.push((id, name));
    }

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], (1, "Alice".to_string()));
    assert_eq!(rows[1], (2, "Bob".to_string()));
}

// ============================================================================
// WHERE Clause Filtering Tests
// ============================================================================

#[test]
fn test_values_with_where_filter() {
    let db = create_test_db("values_where");

    let result = db
        .query(
            "SELECT * FROM (VALUES (1), (2), (3), (4), (5)) AS t(x) WHERE x > 2",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![3, 4, 5]);
}

#[test]
fn test_values_with_where_on_multiple_columns() {
    let db = create_test_db("values_where_multi");

    let result = db
        .query(
            "SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30), (4, 40)) AS t(id, val) WHERE val >= 20 AND id <= 3", ())
        .unwrap();

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let id: i64 = row.get(0).unwrap();
        let val: i64 = row.get(1).unwrap();
        rows.push((id, val));
    }

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], (2, 20));
    assert_eq!(rows[1], (3, 30));
}

#[test]
fn test_values_with_where_string_filter() {
    let db = create_test_db("values_where_string");

    let result = db
        .query(
            "SELECT * FROM (VALUES ('apple'), ('banana'), ('cherry'), ('apricot')) AS t(fruit) WHERE fruit LIKE 'a%'", ())
        .unwrap();

    let mut values: Vec<String> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values.len(), 2);
    assert!(values.contains(&"apple".to_string()));
    assert!(values.contains(&"apricot".to_string()));
}

// ============================================================================
// Expression Evaluation Tests
// ============================================================================

#[test]
fn test_values_with_expression_in_select() {
    let db = create_test_db("values_expr_select");

    let result = db
        .query(
            "SELECT id * 2 AS doubled FROM (VALUES (1), (2), (3)) AS t(id)",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![2, 4, 6]);
}

#[test]
fn test_values_with_arithmetic_expressions() {
    let db = create_test_db("values_arith");

    let result = db
        .query(
            "SELECT x + y AS sum, x * y AS product FROM (VALUES (2, 3), (4, 5), (6, 7)) AS t(x, y)",
            (),
        )
        .unwrap();

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let sum: i64 = row.get(0).unwrap();
        let product: i64 = row.get(1).unwrap();
        rows.push((sum, product));
    }

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (5, 6)); // 2+3, 2*3
    assert_eq!(rows[1], (9, 20)); // 4+5, 4*5
    assert_eq!(rows[2], (13, 42)); // 6+7, 6*7
}

#[test]
fn test_values_with_string_concatenation() {
    let db = create_test_db("values_concat");

    let result = db
        .query(
            "SELECT fname || ' ' || lname AS fullname FROM (VALUES ('John', 'Doe'), ('Jane', 'Smith')) AS t(fname, lname)", ())
        .unwrap();

    let mut values: Vec<String> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec!["John Doe", "Jane Smith"]);
}

#[test]
fn test_values_with_case_expression() {
    let db = create_test_db("values_case");

    let result = db
        .query(
            "SELECT val, CASE WHEN val < 50 THEN 'low' WHEN val < 100 THEN 'medium' ELSE 'high' END AS category
             FROM (VALUES (25), (75), (150)) AS t(val)", ())
        .unwrap();

    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let val: i64 = row.get(0).unwrap();
        let category: String = row.get(1).unwrap();
        rows.push((val, category));
    }

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (25, "low".to_string()));
    assert_eq!(rows[1], (75, "medium".to_string()));
    assert_eq!(rows[2], (150, "high".to_string()));
}

// ============================================================================
// ORDER BY and LIMIT Tests
// ============================================================================

#[test]
fn test_values_with_order_by() {
    let db = create_test_db("values_order");

    let result = db
        .query(
            "SELECT * FROM (VALUES (3), (1), (4), (1), (5), (9), (2)) AS t(x) ORDER BY x",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![1, 1, 2, 3, 4, 5, 9]);
}

#[test]
fn test_values_with_order_by_desc() {
    let db = create_test_db("values_order_desc");

    let result = db
        .query(
            "SELECT * FROM (VALUES (3), (1), (5), (2), (4)) AS t(x) ORDER BY x DESC",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_values_with_limit() {
    let db = create_test_db("values_limit");

    let result = db
        .query(
            "SELECT * FROM (VALUES (1), (2), (3), (4), (5)) AS t(x) LIMIT 3",
            (),
        )
        .unwrap();

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 3);
}

#[test]
fn test_values_with_offset() {
    let db = create_test_db("values_offset");

    let result = db
        .query(
            "SELECT * FROM (VALUES (1), (2), (3), (4), (5)) AS t(x) LIMIT 2 OFFSET 2",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![3, 4]);
}

// ============================================================================
// Aggregation Tests
// ============================================================================

#[test]
fn test_values_with_count() {
    let db = create_test_db("values_count");

    let result = db
        .query(
            "SELECT COUNT(*) FROM (VALUES (1), (2), (3), (4), (5)) AS t(x)",
            (),
        )
        .unwrap();

    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }

    assert_eq!(count, Some(5));
}

#[test]
fn test_values_with_sum() {
    let db = create_test_db("values_sum");

    let result = db
        .query(
            "SELECT SUM(x) FROM (VALUES (1), (2), (3), (4), (5)) AS t(x)",
            (),
        )
        .unwrap();

    let mut sum: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        sum = Some(row.get(0).unwrap());
    }

    assert_eq!(sum, Some(15));
}

#[test]
fn test_values_with_avg() {
    let db = create_test_db("values_avg");

    let result = db
        .query("SELECT AVG(x) FROM (VALUES (10), (20), (30)) AS t(x)", ())
        .unwrap();

    let mut avg: Option<f64> = None;
    for row in result {
        let row = row.unwrap();
        avg = Some(row.get(0).unwrap());
    }

    assert_eq!(avg, Some(20.0));
}

#[test]
fn test_values_with_min_max() {
    let db = create_test_db("values_minmax");

    let result = db
        .query(
            "SELECT MIN(x), MAX(x) FROM (VALUES (5), (2), (8), (1), (9)) AS t(x)",
            (),
        )
        .unwrap();

    for row in result {
        let row = row.unwrap();
        let min: i64 = row.get(0).unwrap();
        let max: i64 = row.get(1).unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 9);
    }
}

// ============================================================================
// Mixed Data Types Tests
// ============================================================================

#[test]
fn test_values_with_null() {
    let db = create_test_db("values_null");

    let result = db
        .query("SELECT * FROM (VALUES (1), (NULL), (3)) AS t(x)", ())
        .unwrap();

    let mut count = 0;
    for row in result {
        let row = row.unwrap();
        let val: Option<i64> = row.get(0).ok();
        match count {
            0 => assert_eq!(val, Some(1)),
            1 => assert!(row.get::<i64>(0).is_err() || val.is_none()),
            2 => assert_eq!(val, Some(3)),
            _ => panic!("Too many rows"),
        }
        count += 1;
    }

    assert_eq!(count, 3);
}

#[test]
fn test_values_with_boolean() {
    let db = create_test_db("values_bool");

    let result = db
        .query(
            "SELECT * FROM (VALUES (TRUE), (FALSE), (TRUE)) AS t(flag)",
            (),
        )
        .unwrap();

    let mut values: Vec<bool> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![true, false, true]);
}

#[test]
fn test_values_with_float() {
    let db = create_test_db("values_float");

    let result = db
        .query("SELECT * FROM (VALUES (1.5), (2.5), (3.5)) AS t(x)", ())
        .unwrap();

    let mut values: Vec<f64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![1.5, 2.5, 3.5]);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_values_single_row() {
    let db = create_test_db("values_single_row");

    let result = db
        .query("SELECT * FROM (VALUES (42)) AS t(answer)", ())
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![42]);
}

#[test]
fn test_values_large_dataset() {
    let db = create_test_db("values_large");

    // Generate a VALUES clause with 100 rows
    let values_rows: Vec<String> = (1..=100).map(|i| format!("({})", i)).collect();
    let query = format!("SELECT * FROM (VALUES {}) AS t(x)", values_rows.join(", "));

    let result = db.query(&query, ()).unwrap();

    let mut count = 0;
    let mut sum = 0i64;
    for row in result {
        let row = row.unwrap();
        let val: i64 = row.get(0).unwrap();
        sum += val;
        count += 1;
    }

    assert_eq!(count, 100);
    assert_eq!(sum, 5050); // Sum of 1 to 100
}
