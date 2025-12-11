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

//! Integration tests for recursive Common Table Expressions (CTEs)

use stoolap::Database;

fn create_test_db(name: &str) -> Database {
    Database::open(&format!("memory://{}", name)).expect("Failed to create in-memory database")
}

// ============================================================================
// Basic Recursive CTE Tests
// ============================================================================

#[test]
fn test_recursive_cte_count_to_10() {
    let db = create_test_db("rcte_count10");

    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 10
            )
            SELECT x FROM cnt",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
fn test_recursive_cte_with_alias_in_anchor() {
    let db = create_test_db("rcte_alias_anchor");

    let result = db
        .query(
            "WITH RECURSIVE cnt AS (
                SELECT 1 AS x
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 5
            )
            SELECT x FROM cnt",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_recursive_cte_fibonacci() {
    let db = create_test_db("rcte_fib");

    let result = db
        .query(
            "WITH RECURSIVE fib(n, fib_n, fib_n_plus_1) AS (
                SELECT 1, 0, 1
                UNION ALL
                SELECT n + 1, fib_n_plus_1, fib_n + fib_n_plus_1 FROM fib WHERE n < 10
            )
            SELECT n, fib_n FROM fib",
            (),
        )
        .unwrap();

    let mut fib_values: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let n: i64 = row.get(0).unwrap();
        let fib_n: i64 = row.get(1).unwrap();
        fib_values.push((n, fib_n));
    }

    // Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    assert_eq!(fib_values.len(), 10);
    assert_eq!(fib_values[0], (1, 0));
    assert_eq!(fib_values[1], (2, 1));
    assert_eq!(fib_values[2], (3, 1));
    assert_eq!(fib_values[3], (4, 2));
    assert_eq!(fib_values[4], (5, 3));
    assert_eq!(fib_values[5], (6, 5));
    assert_eq!(fib_values[6], (7, 8));
    assert_eq!(fib_values[7], (8, 13));
    assert_eq!(fib_values[8], (9, 21));
    assert_eq!(fib_values[9], (10, 34));
}

#[test]
fn test_recursive_cte_cumulative_sum() {
    let db = create_test_db("rcte_cumsum");

    let result = db
        .query(
            "WITH RECURSIVE nums(n, total) AS (
                SELECT 1, 1
                UNION ALL
                SELECT n + 1, total + n + 1 FROM nums WHERE n < 5
            )
            SELECT n, total FROM nums",
            (),
        )
        .unwrap();

    let mut values: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let n: i64 = row.get(0).unwrap();
        let total: i64 = row.get(1).unwrap();
        values.push((n, total));
    }

    // Sum 1 to n: 1=1, 2=3, 3=6, 4=10, 5=15
    assert_eq!(values, vec![(1, 1), (2, 3), (3, 6), (4, 10), (5, 15)]);
}

#[test]
fn test_recursive_cte_powers_of_two() {
    let db = create_test_db("rcte_pow2");

    let result = db
        .query(
            "WITH RECURSIVE pow2(n, val) AS (
                SELECT 0, 1
                UNION ALL
                SELECT n + 1, val * 2 FROM pow2 WHERE n < 10
            )
            SELECT n, val FROM pow2",
            (),
        )
        .unwrap();

    let mut values: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let n: i64 = row.get(0).unwrap();
        let val: i64 = row.get(1).unwrap();
        values.push((n, val));
    }

    assert_eq!(values.len(), 11);
    assert_eq!(values[0], (0, 1));
    assert_eq!(values[1], (1, 2));
    assert_eq!(values[2], (2, 4));
    assert_eq!(values[3], (3, 8));
    assert_eq!(values[4], (4, 16));
    assert_eq!(values[10], (10, 1024));
}

// ============================================================================
// Recursive CTE with Aggregation Tests
// ============================================================================

#[test]
fn test_recursive_cte_with_count() {
    let db = create_test_db("rcte_count");

    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 100
            )
            SELECT COUNT(*) FROM cnt",
            (),
        )
        .unwrap();

    let mut count_result: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count_result = Some(row.get(0).unwrap());
    }

    assert_eq!(count_result, Some(100));
}

#[test]
fn test_recursive_cte_with_sum() {
    let db = create_test_db("rcte_sum");

    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 10
            )
            SELECT SUM(x) FROM cnt",
            (),
        )
        .unwrap();

    let mut sum_result: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        sum_result = Some(row.get(0).unwrap());
    }

    // Sum of 1 to 10 = 55
    assert_eq!(sum_result, Some(55));
}

#[test]
fn test_recursive_cte_with_max() {
    let db = create_test_db("rcte_max");

    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 50
            )
            SELECT MAX(x) FROM cnt",
            (),
        )
        .unwrap();

    let mut max_result: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        max_result = Some(row.get(0).unwrap());
    }

    assert_eq!(max_result, Some(50));
}

// ============================================================================
// Recursive CTE with Filtering Tests
// ============================================================================

#[test]
fn test_recursive_cte_with_where_on_result() {
    let db = create_test_db("rcte_where");

    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 20
            )
            SELECT x FROM cnt WHERE x > 15",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![16, 17, 18, 19, 20]);
}

#[test]
fn test_recursive_cte_with_limit() {
    let db = create_test_db("rcte_limit");

    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 100
            )
            SELECT x FROM cnt LIMIT 5",
            (),
        )
        .unwrap();

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 5);
}

#[test]
fn test_recursive_cte_with_offset() {
    let db = create_test_db("rcte_offset");

    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 10
            )
            SELECT x FROM cnt LIMIT 3 OFFSET 5",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    assert_eq!(values, vec![6, 7, 8]);
}

// ============================================================================
// INSERT with Recursive CTE Tests
// ============================================================================

#[test]
fn test_insert_with_recursive_cte() {
    let db = create_test_db("rcte_insert");

    db.execute("CREATE TABLE numbers (n INTEGER PRIMARY KEY)", ())
        .unwrap();

    db.execute(
        "INSERT INTO numbers
        WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM cnt WHERE x < 10
        )
        SELECT x FROM cnt",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM numbers", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(10));

    let result = db.query("SELECT SUM(n) FROM numbers", ()).unwrap();
    let mut sum: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        sum = Some(row.get(0).unwrap());
    }
    assert_eq!(sum, Some(55));
}

#[test]
fn test_insert_with_recursive_cte_multiple_columns() {
    let db = create_test_db("rcte_insert_multi");

    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO data
        WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM cnt WHERE x < 100
        )
        SELECT x, x * 10 FROM cnt",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM data", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(100));

    let result = db.query("SELECT * FROM data WHERE id = 50", ()).unwrap();
    for row in result {
        let row = row.unwrap();
        let id: i64 = row.get(0).unwrap();
        let value: i64 = row.get(1).unwrap();
        assert_eq!(id, 50);
        assert_eq!(value, 500);
    }
}

#[test]
fn test_insert_with_recursive_cte_large_dataset() {
    let db = create_test_db("rcte_insert_large");

    db.execute("CREATE TABLE big_table (n INTEGER PRIMARY KEY)", ())
        .unwrap();

    db.execute(
        "INSERT INTO big_table
        WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM cnt WHERE x < 1000
        )
        SELECT x FROM cnt",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM big_table", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(1000));

    let result = db
        .query("SELECT MIN(n), MAX(n) FROM big_table", ())
        .unwrap();
    for row in result {
        let row = row.unwrap();
        let min: i64 = row.get(0).unwrap();
        let max: i64 = row.get(1).unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 1000);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_recursive_cte_single_row() {
    let db = create_test_db("rcte_single");

    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (
                SELECT 1
                UNION ALL
                SELECT x + 1 FROM cnt WHERE x < 1
            )
            SELECT x FROM cnt",
            (),
        )
        .unwrap();

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }

    // Only anchor row since condition x < 1 is immediately false
    assert_eq!(values, vec![1]);
}

#[test]
fn test_recursive_cte_with_string_concatenation() {
    let db = create_test_db("rcte_string");

    let result = db
        .query(
            "WITH RECURSIVE letters(n, str) AS (
                SELECT 1, 'a'
                UNION ALL
                SELECT n + 1, str || 'a' FROM letters WHERE n < 5
            )
            SELECT n, str FROM letters",
            (),
        )
        .unwrap();

    let mut values: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let n: i64 = row.get(0).unwrap();
        let str: String = row.get(1).unwrap();
        values.push((n, str));
    }

    assert_eq!(values.len(), 5);
    assert_eq!(values[0], (1, "a".to_string()));
    assert_eq!(values[1], (2, "aa".to_string()));
    assert_eq!(values[2], (3, "aaa".to_string()));
    assert_eq!(values[3], (4, "aaaa".to_string()));
    assert_eq!(values[4], (5, "aaaaa".to_string()));
}

#[test]
fn test_recursive_cte_factorial() {
    let db = create_test_db("rcte_factorial");

    let result = db
        .query(
            "WITH RECURSIVE fact(n, factorial) AS (
                SELECT 1, 1
                UNION ALL
                SELECT n + 1, factorial * (n + 1) FROM fact WHERE n < 10
            )
            SELECT n, factorial FROM fact",
            (),
        )
        .unwrap();

    let mut values: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        let n: i64 = row.get(0).unwrap();
        let factorial: i64 = row.get(1).unwrap();
        values.push((n, factorial));
    }

    // Factorials: 1!, 2!, 3!, ..., 10!
    assert_eq!(values.len(), 10);
    assert_eq!(values[0], (1, 1)); // 1!
    assert_eq!(values[1], (2, 2)); // 2!
    assert_eq!(values[2], (3, 6)); // 3!
    assert_eq!(values[3], (4, 24)); // 4!
    assert_eq!(values[4], (5, 120)); // 5!
    assert_eq!(values[9], (10, 3628800)); // 10!
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_recursive_cte_requires_union_all() {
    let db = create_test_db("rcte_err_union");

    // UNION (without ALL) should fail for recursive CTE
    let result = db.query(
        "WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION
            SELECT x + 1 FROM cnt WHERE x < 10
        )
        SELECT x FROM cnt",
        (),
    );

    assert!(result.is_err());
}

#[test]
fn test_recursive_cte_without_union() {
    let db = create_test_db("rcte_err_no_union");

    // Recursive CTE without UNION ALL should fail
    let result = db.query(
        "WITH RECURSIVE cnt(x) AS (
            SELECT 1
        )
        SELECT x FROM cnt",
        (),
    );

    assert!(result.is_err());
}
