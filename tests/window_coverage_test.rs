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

//! Integration Tests for Window Function Coverage
//!
//! These tests exercise the code paths in src/executor/window.rs

use stoolap::Database;

// =============================================================================
// ROW_NUMBER Tests
// =============================================================================

#[test]
fn test_row_number_basic() {
    let db = Database::open("memory://rn_basic").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t1 VALUES ({}, 'item{}')", i, i), ())
            .expect("Failed");
    }

    let result = db
        .query("SELECT id, ROW_NUMBER() OVER () as rn FROM t1", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_row_number_with_order_by() {
    let db = Database::open("memory://rn_order").expect("Failed");
    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO scores VALUES ({}, {})", i, (11 - i) * 10),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, score, ROW_NUMBER() OVER (ORDER BY score DESC) as rn FROM scores",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<(i64, i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed");
        let id: i64 = row.get(0).unwrap();
        let score: i64 = row.get(1).unwrap();
        let rn: i64 = row.get(2).unwrap();
        rows.push((id, score, rn));
    }
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[0].2, 1); // First row has rn=1
}

#[test]
fn test_row_number_with_partition() {
    let db = Database::open("memory://rn_partition").expect("Failed");
    db.execute(
        "CREATE TABLE emp (id INTEGER PRIMARY KEY, dept TEXT, salary INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=12 {
        let dept = if i <= 4 {
            "A"
        } else if i <= 8 {
            "B"
        } else {
            "C"
        };
        db.execute(
            &format!("INSERT INTO emp VALUES ({}, '{}', {})", i, dept, i * 100),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, dept, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary) as rn FROM emp",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 12);
}

// =============================================================================
// RANK and DENSE_RANK Tests
// =============================================================================

#[test]
fn test_rank_basic() {
    let db = Database::open("memory://rank_basic").expect("Failed");
    db.execute(
        "CREATE TABLE students (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed");

    db.execute("INSERT INTO students VALUES (1, 100)", ())
        .expect("Failed");
    db.execute("INSERT INTO students VALUES (2, 100)", ())
        .expect("Failed");
    db.execute("INSERT INTO students VALUES (3, 90)", ())
        .expect("Failed");
    db.execute("INSERT INTO students VALUES (4, 80)", ())
        .expect("Failed");

    let result = db
        .query(
            "SELECT id, score, RANK() OVER (ORDER BY score DESC) as r FROM students",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed");
        let score: i64 = row.get(1).unwrap();
        let r: i64 = row.get(2).unwrap();
        rows.push((score, r));
    }
    // Two 100s should both have rank 1, then 90 has rank 3, 80 has rank 4
    assert_eq!(rows.len(), 4);
}

#[test]
fn test_dense_rank_basic() {
    let db = Database::open("memory://dense_rank_basic").expect("Failed");
    db.execute(
        "CREATE TABLE students (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed");

    db.execute("INSERT INTO students VALUES (1, 100)", ())
        .expect("Failed");
    db.execute("INSERT INTO students VALUES (2, 100)", ())
        .expect("Failed");
    db.execute("INSERT INTO students VALUES (3, 90)", ())
        .expect("Failed");
    db.execute("INSERT INTO students VALUES (4, 80)", ())
        .expect("Failed");

    let result = db
        .query(
            "SELECT id, score, DENSE_RANK() OVER (ORDER BY score DESC) as dr FROM students",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed");
        let score: i64 = row.get(1).unwrap();
        let dr: i64 = row.get(2).unwrap();
        rows.push((score, dr));
    }
    // Two 100s have rank 1, 90 has rank 2, 80 has rank 3
    assert_eq!(rows.len(), 4);
}

#[test]
fn test_rank_with_partition() {
    let db = Database::open("memory://rank_partition").expect("Failed");
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        let region = if i % 2 == 0 { "North" } else { "South" };
        db.execute(
            &format!(
                "INSERT INTO sales VALUES ({}, '{}', {})",
                i,
                region,
                i as f64 * 10.0
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, region, amount, RANK() OVER (PARTITION BY region ORDER BY amount DESC) as r FROM sales",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 20);
}

// =============================================================================
// NTILE Tests
// =============================================================================

#[test]
fn test_ntile_basic() {
    let db = Database::open("memory://ntile_basic").expect("Failed");
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO items VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, NTILE(4) OVER (ORDER BY id) as quartile FROM items",
            (),
        )
        .expect("Query failed");

    let mut quartiles: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed");
        let q: i64 = row.get(1).unwrap();
        quartiles.push(q);
    }
    assert_eq!(quartiles.len(), 100);
    // Should have 25 rows in each quartile
    assert_eq!(quartiles.iter().filter(|&&q| q == 1).count(), 25);
    assert_eq!(quartiles.iter().filter(|&&q| q == 4).count(), 25);
}

#[test]
fn test_ntile_with_partition() {
    let db = Database::open("memory://ntile_partition").expect("Failed");
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, price FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=30 {
        let cat = if i <= 10 {
            "A"
        } else if i <= 20 {
            "B"
        } else {
            "C"
        };
        db.execute(
            &format!(
                "INSERT INTO products VALUES ({}, '{}', {})",
                i,
                cat,
                i as f64 * 5.0
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, category, NTILE(2) OVER (PARTITION BY category ORDER BY price) as half FROM products",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 30);
}

// =============================================================================
// LAG and LEAD Tests
// =============================================================================

#[test]
fn test_lag_basic() {
    let db = Database::open("memory://lag_basic").expect("Failed");
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO events VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, value, LAG(value, 1) OVER (ORDER BY id) as prev_value FROM events",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_lead_basic() {
    let db = Database::open("memory://lead_basic").expect("Failed");
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO events VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, value, LEAD(value, 1) OVER (ORDER BY id) as next_value FROM events",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_lag_with_default() {
    let db = Database::open("memory://lag_default").expect("Failed");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=5 {
        db.execute(&format!("INSERT INTO data VALUES ({}, {})", i, i * 10), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, val, LAG(val, 1, 0) OVER (ORDER BY id) as prev FROM data",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 5);
}

#[test]
fn test_lag_offset_2() {
    let db = Database::open("memory://lag_offset2").expect("Failed");
    db.execute(
        "CREATE TABLE series (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO series VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, val, LAG(val, 2) OVER (ORDER BY id) as prev2 FROM series",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

// =============================================================================
// FIRST_VALUE, LAST_VALUE, NTH_VALUE Tests
// =============================================================================

#[test]
fn test_first_value() {
    let db = Database::open("memory://first_val").expect("Failed");
    db.execute(
        "CREATE TABLE prices (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO prices VALUES ({}, {})", i, i as f64 * 10.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, price, FIRST_VALUE(price) OVER (ORDER BY id) as first FROM prices",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_last_value() {
    let db = Database::open("memory://last_val").expect("Failed");
    db.execute(
        "CREATE TABLE prices (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO prices VALUES ({}, {})", i, i as f64 * 10.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, price, LAST_VALUE(price) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last FROM prices",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_nth_value() {
    let db = Database::open("memory://nth_val").expect("Failed");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO data VALUES ({}, {})", i, i * 5), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, val, NTH_VALUE(val, 3) OVER (ORDER BY id) as third FROM data",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

// =============================================================================
// PERCENT_RANK and CUME_DIST Tests
// =============================================================================

#[test]
fn test_percent_rank() {
    let db = Database::open("memory://pct_rank").expect("Failed");
    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO scores VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, score, PERCENT_RANK() OVER (ORDER BY score) as pct FROM scores",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_cume_dist() {
    let db = Database::open("memory://cume_dist").expect("Failed");
    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO scores VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, score, CUME_DIST() OVER (ORDER BY score) as cd FROM scores",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

// =============================================================================
// Aggregate Window Functions Tests
// =============================================================================

#[test]
fn test_sum_over() {
    let db = Database::open("memory://sum_over").expect("Failed");
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO sales VALUES ({}, {})", i, i as f64 * 10.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, amount, SUM(amount) OVER () as total FROM sales",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_sum_over_partition() {
    let db = Database::open("memory://sum_partition").expect("Failed");
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        let region = if i % 2 == 0 { "North" } else { "South" };
        db.execute(
            &format!(
                "INSERT INTO sales VALUES ({}, '{}', {})",
                i,
                region,
                i as f64 * 10.0
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query("SELECT id, region, amount, SUM(amount) OVER (PARTITION BY region) as region_total FROM sales", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 20);
}

#[test]
fn test_avg_over() {
    let db = Database::open("memory://avg_over").expect("Failed");
    db.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, val FLOAT)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO data VALUES ({}, {})", i, i as f64),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query("SELECT id, val, AVG(val) OVER () as avg_val FROM data", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_count_over() {
    let db = Database::open("memory://count_over").expect("Failed");
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .expect("Failed");

    for i in 1..=15 {
        let cat = if i % 3 == 0 {
            "A"
        } else if i % 3 == 1 {
            "B"
        } else {
            "C"
        };
        db.execute(&format!("INSERT INTO items VALUES ({}, '{}')", i, cat), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, category, COUNT(*) OVER (PARTITION BY category) as cat_count FROM items",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 15);
}

#[test]
fn test_min_max_over() {
    let db = Database::open("memory://minmax_over").expect("Failed");
    db.execute(
        "CREATE TABLE prices (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO prices VALUES ({}, {})", i, i as f64 * 15.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query("SELECT id, price, MIN(price) OVER () as min_p, MAX(price) OVER () as max_p FROM prices", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

// =============================================================================
// Running Aggregates (Cumulative) Tests
// =============================================================================

#[test]
fn test_running_sum() {
    let db = Database::open("memory://running_sum").expect("Failed");
    db.execute(
        "CREATE TABLE transactions (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO transactions VALUES ({}, {})", i, 100.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, amount, SUM(amount) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total FROM transactions",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_running_avg() {
    let db = Database::open("memory://running_avg").expect("Failed");
    db.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, val FLOAT)", ())
        .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO data VALUES ({}, {})", i, i as f64 * 10.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, val, AVG(val) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_avg FROM data",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

// =============================================================================
// Multiple Window Functions Tests
// =============================================================================

#[test]
fn test_multiple_window_functions() {
    let db = Database::open("memory://multi_window").expect("Failed");
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO sales VALUES ({}, {})", i, i as f64 * 100.0),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, amount,
                    ROW_NUMBER() OVER (ORDER BY amount) as rn,
                    RANK() OVER (ORDER BY amount) as r,
                    SUM(amount) OVER () as total
             FROM sales",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_different_partitions() {
    let db = Database::open("memory://diff_partitions").expect("Failed");
    db.execute(
        "CREATE TABLE emp (id INTEGER PRIMARY KEY, dept TEXT, team TEXT, salary INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=12 {
        let dept = if i <= 6 { "A" } else { "B" };
        let team = if i % 2 == 0 { "X" } else { "Y" };
        db.execute(
            &format!(
                "INSERT INTO emp VALUES ({}, '{}', '{}', {})",
                i,
                dept,
                team,
                i * 1000
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, dept, team, salary,
                    ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary) as dept_rank,
                    ROW_NUMBER() OVER (PARTITION BY team ORDER BY salary) as team_rank
             FROM emp",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 12);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_window_single_row() {
    let db = Database::open("memory://win_single").expect("Failed");
    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed");

    db.execute("INSERT INTO t1 VALUES (1, 100)", ())
        .expect("Failed");

    let result = db
        .query(
            "SELECT id, val, ROW_NUMBER() OVER () as rn, SUM(val) OVER () as total FROM t1",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed");
        let rn: i64 = row.get(2).unwrap();
        assert_eq!(rn, 1);
        count += 1;
    }
    assert_eq!(count, 1);
}

#[test]
fn test_window_with_nulls() {
    let db = Database::open("memory://win_nulls").expect("Failed");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    db.execute("INSERT INTO data VALUES (1, 10)", ())
        .expect("Failed");
    db.execute("INSERT INTO data VALUES (2, NULL)", ())
        .expect("Failed");
    db.execute("INSERT INTO data VALUES (3, 30)", ())
        .expect("Failed");

    let result = db
        .query(
            "SELECT id, val, ROW_NUMBER() OVER (ORDER BY val) as rn FROM data",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 3);
}

#[test]
fn test_window_empty_partition() {
    let db = Database::open("memory://win_empty_part").expect("Failed");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .expect("Failed");

    db.execute("INSERT INTO data VALUES (1, 'A', 10)", ())
        .expect("Failed");
    db.execute("INSERT INTO data VALUES (2, 'A', 20)", ())
        .expect("Failed");
    // No category B

    let result = db
        .query(
            "SELECT id, category, val, SUM(val) OVER (PARTITION BY category) as cat_sum FROM data",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
fn test_window_with_where() {
    let db = Database::open("memory://win_where").expect("Failed");
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, amount FLOAT)",
        (),
    )
    .expect("Failed");

    for i in 1..=20 {
        db.execute(
            &format!("INSERT INTO sales VALUES ({}, {})", i, i as f64 * 10.0),
            (),
        )
        .expect("Failed");
    }

    // WHERE filter applied before window function
    let result = db
        .query("SELECT id, amount, ROW_NUMBER() OVER (ORDER BY amount) as rn FROM sales WHERE amount > 100", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert!(count < 20); // Should have fewer rows due to WHERE
}

#[test]
fn test_window_with_limit() {
    let db = Database::open("memory://win_limit").expect("Failed");
    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=50 {
        db.execute(&format!("INSERT INTO data VALUES ({}, {})", i, i), ())
            .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, val, ROW_NUMBER() OVER (ORDER BY val) as rn FROM data LIMIT 10",
            (),
        )
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn test_window_desc_order() {
    let db = Database::open("memory://win_desc").expect("Failed");
    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO scores VALUES ({}, {})", i, i * 10),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query(
            "SELECT id, score, ROW_NUMBER() OVER (ORDER BY score DESC) as rn FROM scores",
            (),
        )
        .expect("Query failed");

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed");
        let score: i64 = row.get(1).unwrap();
        let rn: i64 = row.get(2).unwrap();
        rows.push((score, rn));
    }
    assert_eq!(rows.len(), 10);
    // Highest score (100) should have rn=1
    let max_score_rn = rows.iter().find(|(s, _)| *s == 100).unwrap().1;
    assert_eq!(max_score_rn, 1);
}

#[test]
fn test_window_multiple_order_columns() {
    let db = Database::open("memory://win_multi_order").expect("Failed");
    db.execute(
        "CREATE TABLE emp (id INTEGER PRIMARY KEY, dept TEXT, salary INTEGER)",
        (),
    )
    .expect("Failed");

    for i in 1..=12 {
        let dept = if i % 3 == 0 {
            "A"
        } else if i % 3 == 1 {
            "B"
        } else {
            "C"
        };
        db.execute(
            &format!(
                "INSERT INTO emp VALUES ({}, '{}', {})",
                i,
                dept,
                (i % 5) * 1000
            ),
            (),
        )
        .expect("Failed");
    }

    let result = db
        .query("SELECT id, dept, salary, ROW_NUMBER() OVER (ORDER BY dept, salary DESC) as rn FROM emp", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let _ = row.expect("Failed");
        count += 1;
    }
    assert_eq!(count, 12);
}
