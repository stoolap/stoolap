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

//! New SQL Features Integration Tests
//!
//! Tests for recently added SQL features:
//! - EXISTS as scalar expression
//! - RETURNING clause for INSERT/UPDATE/DELETE
//! - ALL/ANY subquery operators
//! - FILTER clause for aggregates
//! - JSON_ARRAY_LENGTH function

use stoolap::Database;

// ============================================================================
// EXISTS as Scalar Expression
// ============================================================================

#[test]
fn test_exists_scalar_true() {
    let db = Database::open("memory://exists_scalar_true").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO test_data VALUES (1, 100), (2, 200)", ())
        .expect("Failed to insert");

    let result: bool = db
        .query_one(
            "SELECT EXISTS(SELECT 1 FROM test_data WHERE value > 50)",
            (),
        )
        .expect("Failed to query");
    assert!(
        result,
        "EXISTS should return true when subquery returns rows"
    );
}

#[test]
fn test_exists_scalar_false() {
    let db = Database::open("memory://exists_scalar_false").expect("Failed to create database");

    db.execute(
        "CREATE TABLE test_data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO test_data VALUES (1, 100), (2, 200)", ())
        .expect("Failed to insert");

    let result: bool = db
        .query_one(
            "SELECT EXISTS(SELECT 1 FROM test_data WHERE value > 500)",
            (),
        )
        .expect("Failed to query");
    assert!(
        !result,
        "EXISTS should return false when subquery returns no rows"
    );
}

#[test]
fn test_exists_scalar_empty_table() {
    let db = Database::open("memory://exists_scalar_empty").expect("Failed to create database");

    db.execute("CREATE TABLE empty_table (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create table");

    let result: bool = db
        .query_one("SELECT EXISTS(SELECT 1 FROM empty_table)", ())
        .expect("Failed to query");
    assert!(!result, "EXISTS should return false for empty table");
}

#[test]
fn test_exists_scalar_with_literal() {
    let db = Database::open("memory://exists_scalar_literal").expect("Failed to create database");

    // SELECT EXISTS(SELECT 1) should always return true
    let result: bool = db
        .query_one("SELECT EXISTS(SELECT 1)", ())
        .expect("Failed to query");
    assert!(result, "EXISTS(SELECT 1) should return true");
}

#[test]
fn test_exists_scalar_multiple_columns() {
    let db = Database::open("memory://exists_scalar_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price DOUBLE)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO products VALUES (1, 'Widget', 10.0), (2, 'Gadget', 100.0)",
        (),
    )
    .expect("Failed to insert");

    // Test multiple EXISTS in same SELECT
    let result = db
        .query(
            "SELECT EXISTS(SELECT 1 FROM products WHERE price < 50) AS has_cheap,
                    EXISTS(SELECT 1 FROM products WHERE price > 1000) AS has_expensive",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let has_cheap: bool = row.get(0).expect("Failed to get has_cheap");
        let has_expensive: bool = row.get(1).expect("Failed to get has_expensive");
        assert!(has_cheap, "Should have cheap products");
        assert!(!has_expensive, "Should not have expensive products");
    }
}

// ============================================================================
// RETURNING Clause for INSERT
// ============================================================================

#[test]
fn test_insert_returning_single_row() {
    let db = Database::open("memory://insert_ret_single").expect("Failed to create database");

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    let result = db
        .query(
            "INSERT INTO users VALUES (1, 'Alice') RETURNING id, name",
            (),
        )
        .expect("Failed to insert");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let name: String = row.get(1).expect("Failed to get name");
        assert_eq!(id, 1);
        assert_eq!(name, "Alice");
        count += 1;
    }
    assert_eq!(count, 1, "Should return exactly 1 row");
}

#[test]
fn test_insert_returning_multiple_rows() {
    let db = Database::open("memory://insert_ret_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    let result = db
        .query(
            "INSERT INTO items VALUES (1, 100), (2, 200), (3, 300) RETURNING id, value",
            (),
        )
        .expect("Failed to insert");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let value: i64 = row.get(1).expect("Failed to get value");
        rows.push((id, value));
    }

    assert_eq!(rows.len(), 3, "Should return 3 rows");
    assert_eq!(rows[0], (1, 100));
    assert_eq!(rows[1], (2, 200));
    assert_eq!(rows[2], (3, 300));
}

#[test]
fn test_insert_returning_subset_columns() {
    let db = Database::open("memory://insert_ret_subset").expect("Failed to create database");

    db.execute(
        "CREATE TABLE records (id INTEGER PRIMARY KEY, a TEXT, b TEXT, c TEXT)",
        (),
    )
    .expect("Failed to create table");

    let result = db
        .query(
            "INSERT INTO records VALUES (1, 'x', 'y', 'z') RETURNING id, b",
            (),
        )
        .expect("Failed to insert");

    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let b: String = row.get(1).expect("Failed to get b");
        assert_eq!(id, 1);
        assert_eq!(b, "y");
    }
}

// ============================================================================
// RETURNING Clause for UPDATE
// ============================================================================

#[test]
fn test_update_returning_single_row() {
    let db = Database::open("memory://update_ret_single").expect("Failed to create database");

    db.execute(
        "CREATE TABLE counters (id INTEGER PRIMARY KEY, count INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO counters VALUES (1, 10)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "UPDATE counters SET count = count + 5 WHERE id = 1 RETURNING id, count",
            (),
        )
        .expect("Failed to update");

    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let count: i64 = row.get(1).expect("Failed to get count");
        assert_eq!(id, 1);
        assert_eq!(count, 15, "Count should be updated to 15");
    }
}

#[test]
fn test_update_returning_multiple_rows() {
    let db = Database::open("memory://update_ret_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO scores VALUES (1, 100), (2, 200), (3, 300)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "UPDATE scores SET score = score * 2 WHERE score >= 200 RETURNING id, score",
            (),
        )
        .expect("Failed to update");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let score: i64 = row.get(1).expect("Failed to get score");
        rows.push((id, score));
    }

    assert_eq!(rows.len(), 2, "Should return 2 updated rows");
    // Check the updated values
    assert!(rows.iter().any(|&(id, score)| id == 2 && score == 400));
    assert!(rows.iter().any(|&(id, score)| id == 3 && score == 600));
}

#[test]
fn test_update_returning_no_match() {
    let db = Database::open("memory://update_ret_nomatch").expect("Failed to create database");

    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO data VALUES (1, 100)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "UPDATE data SET value = 999 WHERE id = 999 RETURNING id, value",
            (),
        )
        .expect("Failed to update");

    let mut count = 0;
    for _row in result {
        count += 1;
    }
    assert_eq!(count, 0, "Should return 0 rows when no match");
}

// ============================================================================
// RETURNING Clause for DELETE
// ============================================================================

#[test]
fn test_delete_returning_single_row() {
    let db = Database::open("memory://delete_ret_single").expect("Failed to create database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO items VALUES (1, 'Apple'), (2, 'Banana')", ())
        .expect("Failed to insert");

    let result = db
        .query("DELETE FROM items WHERE id = 1 RETURNING id, name", ())
        .expect("Failed to delete");

    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let name: String = row.get(1).expect("Failed to get name");
        assert_eq!(id, 1);
        assert_eq!(name, "Apple");
    }

    // Verify row is actually deleted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM items", ())
        .expect("Failed to count");
    assert_eq!(count, 1, "Should have 1 row remaining");
}

#[test]
fn test_delete_returning_multiple_rows() {
    let db = Database::open("memory://delete_ret_multi").expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, price DOUBLE)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO products VALUES (1, 'A', 10.0), (2, 'B', 20.0), (3, 'A', 30.0)",
        (),
    )
    .expect("Failed to insert");

    let result = db
        .query(
            "DELETE FROM products WHERE category = 'A' RETURNING id, price",
            (),
        )
        .expect("Failed to delete");

    let mut deleted: Vec<(i64, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let price: f64 = row.get(1).expect("Failed to get price");
        deleted.push((id, price));
    }

    assert_eq!(deleted.len(), 2, "Should delete 2 rows");
    assert!(deleted.iter().any(|&(id, _)| id == 1));
    assert!(deleted.iter().any(|&(id, _)| id == 3));
}

// ============================================================================
// ALL/ANY Subquery Operators
// ============================================================================

#[test]
fn test_all_greater_than() {
    let db = Database::open("memory://all_gt").expect("Failed to create database");

    db.execute("CREATE TABLE numbers (n INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO numbers VALUES (1), (2), (3)", ())
        .expect("Failed to insert");

    // 5 > ALL(1,2,3) should be true
    let result: bool = db
        .query_one("SELECT 5 > ALL(SELECT n FROM numbers)", ())
        .expect("Failed to query");
    assert!(result, "5 > ALL(1,2,3) should be true");

    // 2 > ALL(1,2,3) should be false
    let result: bool = db
        .query_one("SELECT 2 > ALL(SELECT n FROM numbers)", ())
        .expect("Failed to query");
    assert!(!result, "2 > ALL(1,2,3) should be false");
}

#[test]
fn test_all_equals() {
    let db = Database::open("memory://all_eq").expect("Failed to create database");

    db.execute("CREATE TABLE vals (v INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO vals VALUES (5), (5), (5)", ())
        .expect("Failed to insert");

    // 5 = ALL(5,5,5) should be true
    let result: bool = db
        .query_one("SELECT 5 = ALL(SELECT v FROM vals)", ())
        .expect("Failed to query");
    assert!(result, "5 = ALL(5,5,5) should be true");

    // Add a different value
    db.execute("INSERT INTO vals VALUES (6)", ())
        .expect("Failed to insert");

    // 5 = ALL(5,5,5,6) should be false
    let result: bool = db
        .query_one("SELECT 5 = ALL(SELECT v FROM vals)", ())
        .expect("Failed to query");
    assert!(!result, "5 = ALL(5,5,5,6) should be false");
}

#[test]
fn test_any_equals() {
    let db = Database::open("memory://any_eq").expect("Failed to create database");

    db.execute("CREATE TABLE items (x INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO items VALUES (1), (2), (3)", ())
        .expect("Failed to insert");

    // 2 = ANY(1,2,3) should be true
    let result: bool = db
        .query_one("SELECT 2 = ANY(SELECT x FROM items)", ())
        .expect("Failed to query");
    assert!(result, "2 = ANY(1,2,3) should be true");

    // 5 = ANY(1,2,3) should be false
    let result: bool = db
        .query_one("SELECT 5 = ANY(SELECT x FROM items)", ())
        .expect("Failed to query");
    assert!(!result, "5 = ANY(1,2,3) should be false");
}

#[test]
fn test_any_less_than() {
    let db = Database::open("memory://any_lt").expect("Failed to create database");

    db.execute("CREATE TABLE nums (n INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO nums VALUES (10), (20), (30)", ())
        .expect("Failed to insert");

    // 15 < ANY(10,20,30) should be true (15 < 20, 15 < 30)
    let result: bool = db
        .query_one("SELECT 15 < ANY(SELECT n FROM nums)", ())
        .expect("Failed to query");
    assert!(result, "15 < ANY(10,20,30) should be true");

    // 50 < ANY(10,20,30) should be false
    let result: bool = db
        .query_one("SELECT 50 < ANY(SELECT n FROM nums)", ())
        .expect("Failed to query");
    assert!(!result, "50 < ANY(10,20,30) should be false");
}

#[test]
fn test_all_any_empty_subquery() {
    let db = Database::open("memory://all_any_empty").expect("Failed to create database");

    db.execute("CREATE TABLE empty_tbl (x INTEGER)", ())
        .expect("Failed to create table");

    // ALL against empty set is true (vacuous truth)
    let result: bool = db
        .query_one("SELECT 5 > ALL(SELECT x FROM empty_tbl)", ())
        .expect("Failed to query");
    assert!(result, "x > ALL(empty) should be true");

    // ANY against empty set is false
    let result: bool = db
        .query_one("SELECT 5 = ANY(SELECT x FROM empty_tbl)", ())
        .expect("Failed to query");
    assert!(!result, "x = ANY(empty) should be false");
}

#[test]
fn test_all_any_in_select() {
    let db = Database::open("memory://all_any_select").expect("Failed to create database");

    db.execute("CREATE TABLE values_tbl (v INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO values_tbl VALUES (1), (2), (3)", ())
        .expect("Failed to insert");

    // Test ALL/ANY as column expressions
    let result = db
        .query(
            "SELECT 5 > ALL(SELECT v FROM values_tbl) AS all_result,
                    2 = ANY(SELECT v FROM values_tbl) AS any_result",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let all_result: bool = row.get(0).expect("Failed to get all_result");
        let any_result: bool = row.get(1).expect("Failed to get any_result");
        assert!(all_result, "5 > ALL(1,2,3) should be true");
        assert!(any_result, "2 = ANY(1,2,3) should be true");
    }
}

// ============================================================================
// FILTER Clause for Aggregates
// ============================================================================

#[test]
fn test_filter_count() {
    let db = Database::open("memory://filter_count").expect("Failed to create database");

    db.execute(
        "CREATE TABLE sales (id INTEGER, region TEXT, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO sales VALUES
         (1, 'east', 100), (2, 'west', 200),
         (3, 'east', 150), (4, 'west', 300)",
        (),
    )
    .expect("Failed to insert");

    let result = db
        .query(
            "SELECT COUNT(*) FILTER (WHERE region = 'east') AS east_count,
                    COUNT(*) FILTER (WHERE region = 'west') AS west_count
             FROM sales",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let east_count: i64 = row.get(0).expect("Failed to get east_count");
        let west_count: i64 = row.get(1).expect("Failed to get west_count");
        assert_eq!(east_count, 2, "East count should be 2");
        assert_eq!(west_count, 2, "West count should be 2");
    }
}

#[test]
fn test_filter_sum() {
    let db = Database::open("memory://filter_sum").expect("Failed to create database");

    db.execute(
        "CREATE TABLE orders (id INTEGER, status TEXT, total INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO orders VALUES
         (1, 'completed', 100), (2, 'pending', 50),
         (3, 'completed', 200), (4, 'cancelled', 75)",
        (),
    )
    .expect("Failed to insert");

    let result: i64 = db
        .query_one(
            "SELECT SUM(total) FILTER (WHERE status = 'completed') FROM orders",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 300, "Sum of completed orders should be 300");
}

#[test]
fn test_filter_avg() {
    let db = Database::open("memory://filter_avg").expect("Failed to create database");

    db.execute(
        "CREATE TABLE scores (id INTEGER, subject TEXT, score INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO scores VALUES
         (1, 'math', 80), (2, 'english', 90),
         (3, 'math', 100), (4, 'english', 70)",
        (),
    )
    .expect("Failed to insert");

    let result: f64 = db
        .query_one(
            "SELECT AVG(score) FILTER (WHERE subject = 'math') FROM scores",
            (),
        )
        .expect("Failed to query");
    assert!((result - 90.0).abs() < 0.001, "Avg math score should be 90");
}

#[test]
fn test_filter_min_max() {
    let db = Database::open("memory://filter_minmax").expect("Failed to create database");

    db.execute(
        "CREATE TABLE temps (day INTEGER, type TEXT, temp INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO temps VALUES
         (1, 'high', 30), (1, 'low', 15),
         (2, 'high', 35), (2, 'low', 20)",
        (),
    )
    .expect("Failed to insert");

    let result = db
        .query(
            "SELECT MAX(temp) FILTER (WHERE type = 'high') AS max_high,
                    MIN(temp) FILTER (WHERE type = 'low') AS min_low
             FROM temps",
            (),
        )
        .expect("Failed to query");

    for row in result {
        let row = row.expect("Failed to get row");
        let max_high: i64 = row.get(0).expect("Failed to get max_high");
        let min_low: i64 = row.get(1).expect("Failed to get min_low");
        assert_eq!(max_high, 35, "Max high should be 35");
        assert_eq!(min_low, 15, "Min low should be 15");
    }
}

#[test]
fn test_filter_with_group_by() {
    let db = Database::open("memory://filter_groupby").expect("Failed to create database");

    db.execute(
        "CREATE TABLE transactions (id INTEGER, year INTEGER, type TEXT, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO transactions VALUES
         (1, 2023, 'income', 1000), (2, 2023, 'expense', 500),
         (3, 2024, 'income', 1500), (4, 2024, 'expense', 700)",
        (),
    )
    .expect("Failed to insert");

    let result = db
        .query(
            "SELECT year,
                    SUM(amount) FILTER (WHERE type = 'income') AS income,
                    SUM(amount) FILTER (WHERE type = 'expense') AS expense
             FROM transactions
             GROUP BY year
             ORDER BY year",
            (),
        )
        .expect("Failed to query");

    let mut rows: Vec<(i64, i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let year: i64 = row.get(0).expect("Failed to get year");
        let income: i64 = row.get(1).expect("Failed to get income");
        let expense: i64 = row.get(2).expect("Failed to get expense");
        rows.push((year, income, expense));
    }

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], (2023, 1000, 500));
    assert_eq!(rows[1], (2024, 1500, 700));
}

#[test]
fn test_filter_no_match() {
    let db = Database::open("memory://filter_nomatch").expect("Failed to create database");

    db.execute(
        "CREATE TABLE data (id INTEGER, category TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO data VALUES (1, 'A', 100), (2, 'A', 200)", ())
        .expect("Failed to insert");

    // Filter that matches nothing should return NULL for most aggregates
    let result: Option<i64> = db
        .query_one(
            "SELECT SUM(value) FILTER (WHERE category = 'Z') FROM data",
            (),
        )
        .expect("Failed to query");
    assert!(result.is_none(), "SUM with no matches should be NULL");

    // COUNT(*) with no matches returns 0
    let result: i64 = db
        .query_one(
            "SELECT COUNT(*) FILTER (WHERE category = 'Z') FROM data",
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 0, "COUNT(*) with no matches should be 0");
}

// ============================================================================
// JSON_ARRAY_LENGTH Function
// ============================================================================

#[test]
fn test_json_array_length_basic() {
    let db = Database::open("memory://json_arr_len_basic").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT JSON_ARRAY_LENGTH('[1, 2, 3, 4, 5]')", ())
        .expect("Failed to query");
    assert_eq!(result, 5, "Array length should be 5");
}

#[test]
fn test_json_array_length_empty() {
    let db = Database::open("memory://json_arr_len_empty").expect("Failed to create database");

    let result: i64 = db
        .query_one("SELECT JSON_ARRAY_LENGTH('[]')", ())
        .expect("Failed to query");
    assert_eq!(result, 0, "Empty array length should be 0");
}

#[test]
fn test_json_array_length_with_path() {
    let db = Database::open("memory://json_arr_len_path").expect("Failed to create database");

    let result: i64 = db
        .query_one(
            r#"SELECT JSON_ARRAY_LENGTH('{"items": [1, 2, 3]}', '$.items')"#,
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 3, "Nested array length should be 3");
}

#[test]
fn test_json_array_length_not_array() {
    let db = Database::open("memory://json_arr_len_notarr").expect("Failed to create database");

    let result: Option<i64> = db
        .query_one(r#"SELECT JSON_ARRAY_LENGTH('{"a": 1}')"#, ())
        .expect("Failed to query");
    assert!(result.is_none(), "Non-array should return NULL");
}

#[test]
fn test_json_array_length_null_input() {
    let db = Database::open("memory://json_arr_len_null").expect("Failed to create database");

    let result: Option<i64> = db
        .query_one("SELECT JSON_ARRAY_LENGTH(NULL)", ())
        .expect("Failed to query");
    assert!(result.is_none(), "NULL input should return NULL");
}

#[test]
fn test_json_array_length_nested_path() {
    let db = Database::open("memory://json_arr_len_nested").expect("Failed to create database");

    let json = r#"{"data": {"users": ["alice", "bob", "charlie"]}}"#;
    let result: i64 = db
        .query_one(
            &format!("SELECT JSON_ARRAY_LENGTH('{}', '$.data.users')", json),
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 3, "Deeply nested array length should be 3");
}

#[test]
fn test_json_array_length_with_table() {
    let db = Database::open("memory://json_arr_len_table").expect("Failed to create database");

    db.execute(
        "CREATE TABLE json_data (id INTEGER PRIMARY KEY, data TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        r#"INSERT INTO json_data VALUES
           (1, '[1, 2]'),
           (2, '[1, 2, 3, 4]'),
           (3, '[]')"#,
        (),
    )
    .expect("Failed to insert");

    let result = db
        .query(
            "SELECT id, JSON_ARRAY_LENGTH(data) AS len FROM json_data ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).expect("Failed to get id");
        let len: i64 = row.get(1).expect("Failed to get len");
        rows.push((id, len));
    }

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (1, 2));
    assert_eq!(rows[1], (2, 4));
    assert_eq!(rows[2], (3, 0));
}

#[test]
fn test_json_array_length_invalid_path() {
    let db = Database::open("memory://json_arr_len_invalid").expect("Failed to create database");

    let result: Option<i64> = db
        .query_one(
            r#"SELECT JSON_ARRAY_LENGTH('{"a": [1,2]}', '$.nonexistent')"#,
            (),
        )
        .expect("Failed to query");
    assert!(result.is_none(), "Invalid path should return NULL");
}

#[test]
fn test_json_array_length_mixed_types() {
    let db = Database::open("memory://json_arr_len_mixed").expect("Failed to create database");

    // Array with mixed types
    let result: i64 = db
        .query_one(
            r#"SELECT JSON_ARRAY_LENGTH('[1, "two", true, null, {"a": 1}]')"#,
            (),
        )
        .expect("Failed to query");
    assert_eq!(result, 5, "Mixed type array length should be 5");
}
