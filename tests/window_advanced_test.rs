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

//! Advanced Window Function Tests
//!
//! Tests for ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG, LEAD,
//! FIRST_VALUE, LAST_VALUE, NTH_VALUE, PERCENT_RANK, CUME_DIST
//! with PARTITION BY, ORDER BY, and frame specifications.

use stoolap::Database;

fn setup_sales_table(db: &Database) {
    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            salesperson TEXT,
            region TEXT,
            amount FLOAT,
            sale_date TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    let inserts = [
        "INSERT INTO sales VALUES (1, 'Alice', 'North', 1000.0, '2024-01-15')",
        "INSERT INTO sales VALUES (2, 'Alice', 'North', 1500.0, '2024-01-20')",
        "INSERT INTO sales VALUES (3, 'Alice', 'North', 800.0, '2024-01-25')",
        "INSERT INTO sales VALUES (4, 'Bob', 'North', 1200.0, '2024-01-10')",
        "INSERT INTO sales VALUES (5, 'Bob', 'North', 900.0, '2024-01-18')",
        "INSERT INTO sales VALUES (6, 'Charlie', 'South', 2000.0, '2024-01-05')",
        "INSERT INTO sales VALUES (7, 'Charlie', 'South', 1800.0, '2024-01-12')",
        "INSERT INTO sales VALUES (8, 'Diana', 'South', 1100.0, '2024-01-08')",
        "INSERT INTO sales VALUES (9, 'Diana', 'South', 1300.0, '2024-01-22')",
        "INSERT INTO sales VALUES (10, 'Diana', 'South', 1100.0, '2024-01-28')",
    ];

    for insert in &inserts {
        db.execute(insert, ()).expect("Failed to insert");
    }
}

// =============================================================================
// ROW_NUMBER Tests
// =============================================================================

#[test]
fn test_row_number_basic() {
    let db = Database::open("memory://row_number_basic").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, salesperson, ROW_NUMBER() OVER (ORDER BY id) as rn
             FROM sales
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_numbers = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let rn: i64 = row.get(2).unwrap();
        row_numbers.push(rn);
    }

    assert_eq!(
        row_numbers,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Expected sequential row numbers"
    );
}

#[test]
fn test_row_number_with_partition() {
    let db = Database::open("memory://row_number_partition").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, salesperson, ROW_NUMBER() OVER (PARTITION BY salesperson ORDER BY id) as rn
             FROM sales
             ORDER BY salesperson, id",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let salesperson: String = row.get(1).unwrap();
        let rn: i64 = row.get(2).unwrap();
        results.push((salesperson, rn));
    }

    // Alice should have rn 1, 2, 3
    let alice_rns: Vec<i64> = results
        .iter()
        .filter(|(s, _)| s == "Alice")
        .map(|(_, rn)| *rn)
        .collect();
    assert_eq!(
        alice_rns,
        vec![1, 2, 3],
        "Expected Alice row numbers 1, 2, 3"
    );
}

#[test]
fn test_row_number_desc_order() {
    let db = Database::open("memory://row_number_desc").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, amount, ROW_NUMBER() OVER (ORDER BY amount DESC) as rn
             FROM sales
             ORDER BY rn",
            (),
        )
        .expect("Failed to query");

    let mut first_row = None;
    for row in result {
        let row = row.expect("Failed to get row");
        let amount: f64 = row.get(1).unwrap();
        let rn: i64 = row.get(2).unwrap();
        if rn == 1 {
            first_row = Some(amount);
        }
    }

    // Highest amount is 2000 (Charlie)
    assert!(
        (first_row.unwrap() - 2000.0).abs() < 0.01,
        "Expected highest amount first"
    );
}

// =============================================================================
// RANK Tests
// =============================================================================

#[test]
fn test_rank_basic() {
    let db = Database::open("memory://rank_basic").expect("Failed to create database");
    setup_sales_table(&db);

    // Diana has two sales of 1100, so they should have the same rank
    let result = db
        .query(
            "SELECT salesperson, amount, RANK() OVER (ORDER BY amount) as rnk
             FROM sales
             ORDER BY amount, salesperson",
            (),
        )
        .expect("Failed to query");

    let mut ranks = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let amount: f64 = row.get(1).unwrap();
        let rnk: i64 = row.get(2).unwrap();
        ranks.push((amount, rnk));
    }

    // Two 1100 values should have same rank
    let rank_1100: Vec<i64> = ranks
        .iter()
        .filter(|(a, _)| (*a - 1100.0).abs() < 0.01)
        .map(|(_, r)| *r)
        .collect();
    assert!(
        rank_1100.len() == 2 && rank_1100[0] == rank_1100[1],
        "Expected same rank for tied values"
    );
}

#[test]
fn test_rank_with_gaps() {
    let db = Database::open("memory://rank_gaps").expect("Failed to create database");

    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert scores with ties
    db.execute("INSERT INTO scores VALUES (1, 100)", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (2, 100)", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (3, 90)", ()).unwrap();
    db.execute("INSERT INTO scores VALUES (4, 90)", ()).unwrap();
    db.execute("INSERT INTO scores VALUES (5, 80)", ()).unwrap();

    let result = db
        .query(
            "SELECT score, RANK() OVER (ORDER BY score DESC) as rnk FROM scores ORDER BY score DESC, id",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let score: i64 = row.get(0).unwrap();
        let rnk: i64 = row.get(1).unwrap();
        results.push((score, rnk));
    }

    // Should be: 100->1, 100->1, 90->3, 90->3, 80->5 (gaps after ties)
    assert_eq!(results[0].1, 1, "First 100 should be rank 1");
    assert_eq!(results[1].1, 1, "Second 100 should be rank 1");
    assert_eq!(results[2].1, 3, "First 90 should be rank 3 (gap)");
    assert_eq!(results[4].1, 5, "80 should be rank 5");
}

// =============================================================================
// DENSE_RANK Tests
// =============================================================================

#[test]
fn test_dense_rank_no_gaps() {
    let db = Database::open("memory://dense_rank").expect("Failed to create database");

    db.execute(
        "CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO scores VALUES (1, 100)", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (2, 100)", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (3, 90)", ()).unwrap();
    db.execute("INSERT INTO scores VALUES (4, 90)", ()).unwrap();
    db.execute("INSERT INTO scores VALUES (5, 80)", ()).unwrap();

    let result = db
        .query(
            "SELECT score, DENSE_RANK() OVER (ORDER BY score DESC) as drnk
             FROM scores ORDER BY score DESC, id",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let score: i64 = row.get(0).unwrap();
        let drnk: i64 = row.get(1).unwrap();
        results.push((score, drnk));
    }

    // Should be: 100->1, 100->1, 90->2, 90->2, 80->3 (no gaps)
    assert_eq!(results[0].1, 1, "First 100 should be dense rank 1");
    assert_eq!(results[1].1, 1, "Second 100 should be dense rank 1");
    assert_eq!(results[2].1, 2, "First 90 should be dense rank 2 (no gap)");
    assert_eq!(results[4].1, 3, "80 should be dense rank 3");
}

// =============================================================================
// NTILE Tests
// =============================================================================

#[test]
fn test_ntile_quartiles() {
    let db = Database::open("memory://ntile_quartiles").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, amount, NTILE(4) OVER (ORDER BY amount) as quartile
             FROM sales
             ORDER BY amount",
            (),
        )
        .expect("Failed to query");

    let mut quartiles: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let quartile: i64 = row.get(2).unwrap();
        quartiles.push(quartile);
    }

    // 10 rows into 4 groups: 3, 3, 2, 2
    assert!(quartiles.iter().filter(|&&q| q == 1).count() >= 2);
    assert!(quartiles.iter().filter(|&&q| q == 4).count() >= 2);
}

#[test]
fn test_ntile_with_partition() {
    let db = Database::open("memory://ntile_partition").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson, amount, NTILE(2) OVER (PARTITION BY salesperson ORDER BY amount) as half
             FROM sales
             ORDER BY salesperson, amount",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let salesperson: String = row.get(0).unwrap();
        let half: i64 = row.get(2).unwrap();
        results.push((salesperson, half));
    }

    // Each salesperson's sales split into 2 groups
    let alice_halves: Vec<i64> = results
        .iter()
        .filter(|(s, _)| s == "Alice")
        .map(|(_, h)| *h)
        .collect();
    // Alice has 3 sales, so should be 2 in first half, 1 in second half (or similar split)
    assert!(
        alice_halves.contains(&1) && alice_halves.contains(&2),
        "Expected both halves for Alice"
    );
}

// =============================================================================
// LAG/LEAD Tests
// =============================================================================

#[test]
fn test_lag_basic() {
    let db = Database::open("memory://lag_basic").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, amount, LAG(amount) OVER (ORDER BY id) as prev_amount
             FROM sales
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _id: i64 = row.get(0).unwrap();
        // LAG returns NULL for first row, amount is column 1, prev_amount is column 2
        let _amount: f64 = row.get(1).unwrap();
        count += 1;
    }

    // All rows should be returned
    assert_eq!(count, 10, "Expected 10 rows");
}

#[test]
fn test_lag_with_offset() {
    let db = Database::open("memory://lag_offset").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, amount, LAG(amount, 2) OVER (ORDER BY id) as prev2_amount
             FROM sales
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 10, "Expected all rows returned");
}

#[test]
fn test_lag_with_default() {
    let db = Database::open("memory://lag_default").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, LAG(amount, 1, 0) OVER (ORDER BY id) as prev_amount
             FROM sales
             ORDER BY id
             LIMIT 1",
            (),
        )
        .expect("Failed to query");

    let mut first_prev = None;
    for row in result {
        let row = row.expect("Failed to get row");
        let prev: f64 = row.get(1).unwrap();
        first_prev = Some(prev);
    }

    // First row should have default value 0
    assert!(
        (first_prev.unwrap()).abs() < 0.01,
        "Expected default value 0 for first row"
    );
}

#[test]
fn test_lead_basic() {
    let db = Database::open("memory://lead_basic").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, amount, LEAD(amount) OVER (ORDER BY id) as next_amount
             FROM sales
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 10, "Expected all rows returned");
}

#[test]
fn test_lead_with_partition() {
    let db = Database::open("memory://lead_partition").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson, sale_date, LEAD(sale_date) OVER (PARTITION BY salesperson ORDER BY sale_date) as next_sale
             FROM sales
             ORDER BY salesperson, sale_date",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 10, "Expected all rows");
}

// =============================================================================
// FIRST_VALUE/LAST_VALUE Tests
// =============================================================================

#[test]
fn test_first_value_basic() {
    let db = Database::open("memory://first_value").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson, amount,
                    FIRST_VALUE(amount) OVER (PARTITION BY salesperson ORDER BY amount) as first_amt
             FROM sales
             ORDER BY salesperson, amount",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let salesperson: String = row.get(0).unwrap();
        let first_amt: f64 = row.get(2).unwrap();
        results.push((salesperson, first_amt));
    }

    // Alice's first (lowest) amount is 800
    let alice_first: f64 = results
        .iter()
        .find(|(s, _)| s == "Alice")
        .map(|(_, f)| *f)
        .unwrap();
    assert!(
        (alice_first - 800.0).abs() < 0.01,
        "Expected Alice's first value 800"
    );
}

#[test]
fn test_last_value_with_frame() {
    let db = Database::open("memory://last_value").expect("Failed to create database");
    setup_sales_table(&db);

    // LAST_VALUE requires proper frame specification
    let result = db
        .query(
            "SELECT salesperson, amount,
                    LAST_VALUE(amount) OVER (
                        PARTITION BY salesperson
                        ORDER BY amount
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) as last_amt
             FROM sales
             ORDER BY salesperson, amount",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let salesperson: String = row.get(0).unwrap();
        let last_amt: f64 = row.get(2).unwrap();
        results.push((salesperson, last_amt));
    }

    // Alice's last (highest) amount is 1500
    let alice_last: f64 = results
        .iter()
        .find(|(s, _)| s == "Alice")
        .map(|(_, l)| *l)
        .unwrap();
    assert!(
        (alice_last - 1500.0).abs() < 0.01,
        "Expected Alice's last value 1500"
    );
}

// =============================================================================
// NTH_VALUE Tests
// =============================================================================

#[test]
fn test_nth_value() {
    let db = Database::open("memory://nth_value").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson, amount,
                    NTH_VALUE(amount, 2) OVER (PARTITION BY salesperson ORDER BY amount) as second_amt
             FROM sales
             ORDER BY salesperson, amount",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 10, "Expected all rows");
}

// =============================================================================
// PERCENT_RANK / CUME_DIST Tests
// =============================================================================

#[test]
fn test_percent_rank() {
    let db = Database::open("memory://percent_rank").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT amount, PERCENT_RANK() OVER (ORDER BY amount) as pct_rank
             FROM sales
             ORDER BY amount",
            (),
        )
        .expect("Failed to query");

    let mut pct_ranks = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let pct: f64 = row.get(1).unwrap();
        pct_ranks.push(pct);
    }

    // First should be 0, last should be 1 (or close to it)
    assert!(pct_ranks[0].abs() < 0.01, "First percent rank should be 0");
}

#[test]
fn test_cume_dist() {
    let db = Database::open("memory://cume_dist").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT amount, CUME_DIST() OVER (ORDER BY amount) as cum_dist
             FROM sales
             ORDER BY amount",
            (),
        )
        .expect("Failed to query");

    let mut cum_dists = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cum: f64 = row.get(1).unwrap();
        cum_dists.push(cum);
    }

    // Last should be 1.0
    assert!(
        (cum_dists.last().unwrap() - 1.0).abs() < 0.01,
        "Last cume_dist should be 1.0"
    );
}

// =============================================================================
// Window Frame Tests
// =============================================================================

#[test]
fn test_window_frame_rows_preceding() {
    let db = Database::open("memory://frame_preceding").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, amount,
                    SUM(amount) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) as rolling_sum
             FROM sales
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut results = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let amount: f64 = row.get(1).unwrap();
        let rolling: f64 = row.get(2).unwrap();
        results.push((id, amount, rolling));
    }

    // First row: just its own amount
    assert!(
        (results[0].2 - results[0].1).abs() < 0.01,
        "First row rolling sum should equal its amount"
    );

    // Second row: sum of first + second amounts
    assert!(
        (results[1].2 - (results[0].1 + results[1].1)).abs() < 0.01,
        "Second row should be sum of first two"
    );
}

#[test]
fn test_window_frame_rows_between() {
    let db = Database::open("memory://frame_between").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, amount,
                    AVG(amount) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as moving_avg
             FROM sales
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 10, "Expected all rows");
}

#[test]
fn test_window_frame_unbounded() {
    let db = Database::open("memory://frame_unbounded").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT id, amount,
                    SUM(amount) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total
             FROM sales
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut running_totals = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let running: f64 = row.get(2).unwrap();
        running_totals.push(running);
    }

    // Running total should be monotonically increasing
    for i in 1..running_totals.len() {
        assert!(
            running_totals[i] >= running_totals[i - 1],
            "Running total should be increasing"
        );
    }
}

// =============================================================================
// Named Window Tests
// =============================================================================

#[test]
fn test_named_window() {
    let db = Database::open("memory://named_window").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson, amount,
                    ROW_NUMBER() OVER w as rn,
                    SUM(amount) OVER w as total
             FROM sales
             WINDOW w AS (PARTITION BY salesperson ORDER BY amount)
             ORDER BY salesperson, amount",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 10, "Expected all rows");
}

// =============================================================================
// Multiple Window Functions Tests
// =============================================================================

#[test]
fn test_multiple_window_functions() {
    let db = Database::open("memory://multiple_windows").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson, amount,
                    ROW_NUMBER() OVER (PARTITION BY salesperson ORDER BY amount) as rn,
                    RANK() OVER (PARTITION BY salesperson ORDER BY amount) as rnk,
                    SUM(amount) OVER (PARTITION BY salesperson) as total,
                    AVG(amount) OVER (PARTITION BY salesperson) as avg_amt
             FROM sales
             ORDER BY salesperson, amount",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 10, "Expected all rows");
}

// =============================================================================
// Window Functions with Aggregates
// =============================================================================

#[test]
fn test_window_sum() {
    let db = Database::open("memory://window_sum").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson, amount,
                    SUM(amount) OVER (PARTITION BY salesperson) as person_total,
                    SUM(amount) OVER () as grand_total
             FROM sales
             ORDER BY salesperson, id",
            (),
        )
        .expect("Failed to query");

    let mut grand_totals = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let grand: f64 = row.get(3).unwrap();
        grand_totals.push(grand);
    }

    // All grand totals should be the same
    let first = grand_totals[0];
    assert!(
        grand_totals.iter().all(|&g| (g - first).abs() < 0.01),
        "All grand totals should be equal"
    );
}

#[test]
fn test_window_count() {
    let db = Database::open("memory://window_count").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson,
                    COUNT(*) OVER (PARTITION BY salesperson) as person_count,
                    COUNT(*) OVER () as total_count
             FROM sales
             ORDER BY salesperson",
            (),
        )
        .expect("Failed to query");

    let mut total_counts = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let total: i64 = row.get(2).unwrap();
        total_counts.push(total);
    }

    assert!(
        total_counts.iter().all(|&c| c == 10),
        "Total count should be 10 for all rows"
    );
}

#[test]
fn test_window_avg() {
    let db = Database::open("memory://window_avg").expect("Failed to create database");
    setup_sales_table(&db);

    let result = db
        .query(
            "SELECT salesperson, amount,
                    AVG(amount) OVER (PARTITION BY salesperson) as person_avg
             FROM sales
             ORDER BY salesperson, id",
            (),
        )
        .expect("Failed to query");

    let mut results: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let salesperson: String = row.get(0).unwrap();
        let avg: f64 = row.get(2).unwrap();
        results.entry(salesperson).or_default().push(avg);
    }

    // Alice: (1000 + 1500 + 800) / 3 = 1100
    let alice_avgs = results.get("Alice").unwrap();
    assert!(
        alice_avgs.iter().all(|&a| (a - 1100.0).abs() < 0.01),
        "Alice's average should be 1100"
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_window_with_null_values() {
    let db = Database::open("memory://window_null").expect("Failed to create database");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, NULL)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, NULL)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (5, 50)", ()).unwrap();

    let result = db
        .query(
            "SELECT id, val, ROW_NUMBER() OVER (ORDER BY val NULLS LAST) as rn FROM t",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for _ in result {
        count += 1;
    }

    assert_eq!(count, 5, "Expected all 5 rows");
}

#[test]
fn test_window_empty_partition() {
    let db = Database::open("memory://window_empty_part").expect("Failed to create database");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Only one category
    db.execute("INSERT INTO t VALUES (1, 'A', 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 'A', 20)", ()).unwrap();

    let result = db
        .query(
            "SELECT cat, val, SUM(val) OVER (PARTITION BY cat) as cat_sum FROM t",
            (),
        )
        .expect("Failed to query");

    let mut sums = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let sum: i64 = row.get(2).unwrap();
        sums.push(sum);
    }

    assert!(
        sums.iter().all(|&s| s == 30),
        "Sum should be 30 for category A"
    );
}
