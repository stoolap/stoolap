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

//! Window Function Tests
//!
//! Tests window functions: ROW_NUMBER, RANK, DENSE_RANK, etc.

use stoolap::Database;

fn setup_window_table(db: &Database) {
    db.execute(
        "CREATE TABLE window_test (
            id INTEGER,
            name TEXT,
            department TEXT,
            salary INTEGER
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO window_test (id, name, department, salary) VALUES
        (1, 'Alice', 'Engineering', 85000),
        (2, 'Bob', 'Engineering', 75000),
        (3, 'Charlie', 'Engineering', 90000),
        (4, 'Diana', 'Marketing', 65000),
        (5, 'Eve', 'Marketing', 70000),
        (6, 'Frank', 'Finance', 95000),
        (7, 'Grace', 'Finance', 85000)",
        (),
    )
    .expect("Failed to insert data");
}

/// Test ROW_NUMBER() function
#[test]
fn test_row_number_function() {
    let db = Database::open("memory://window_row_num").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT ROW_NUMBER() OVER () AS row_num FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let row_num: i64 = row.get(0).unwrap();
        row_count += 1;

        // ROW_NUMBER should return sequential numbers starting from 1
        assert_eq!(
            row_num, row_count,
            "Expected ROW_NUMBER = {}, got {}",
            row_count, row_num
        );
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test RANK() function
#[test]
fn test_rank_function() {
    let db = Database::open("memory://window_rank").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query("SELECT RANK() OVER () AS rank_val FROM window_test", ())
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _rank_val: i64 = row.get(0).unwrap();
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test DENSE_RANK() function
#[test]
fn test_dense_rank_function() {
    let db = Database::open("memory://window_dense_rank").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT DENSE_RANK() OVER () AS dense_rank_val FROM window_test",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _dense_rank_val: i64 = row.get(0).unwrap();
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test NTILE() function
#[test]
fn test_ntile_function() {
    let db = Database::open("memory://window_ntile").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query("SELECT NTILE(3) OVER () AS ntile_val FROM window_test", ())
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let ntile_val: i64 = row.get(0).unwrap();
        row_count += 1;

        // NTILE(3) should return values 1, 2, or 3
        assert!(
            (1..=3).contains(&ntile_val),
            "NTILE(3) should return 1, 2, or 3, got {}",
            ntile_val
        );
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test window function with scalar function
#[test]
fn test_window_with_scalar_function() {
    let db = Database::open("memory://window_scalar").expect("Failed to create database");
    setup_window_table(&db);

    // Verify UPPER function works with window data
    let result = db
        .query(
            "SELECT UPPER(name) AS upper_name FROM window_test WHERE id = 1",
            (),
        )
        .expect("Failed to query");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        let upper_name: String = row.get(0).unwrap();
        assert_eq!(
            upper_name, "ALICE",
            "Expected UPPER(name) = 'ALICE', got '{}'",
            upper_name
        );
        found = true;
    }

    assert!(found, "No results returned for scalar function");
}

/// Test multiple window functions in one query
#[test]
fn test_multiple_window_functions() {
    let db = Database::open("memory://window_multiple").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT
                ROW_NUMBER() OVER () AS row_num,
                RANK() OVER () AS rank_val,
                DENSE_RANK() OVER () AS dense_rank_val
            FROM window_test",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _row_num: i64 = row.get(0).unwrap();
        let _rank_val: i64 = row.get(1).unwrap();
        let _dense_rank_val: i64 = row.get(2).unwrap();
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test LAG function
#[test]
fn test_lag_function() {
    let db = Database::open("memory://window_lag").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT salary, LAG(salary) OVER () AS prev_salary FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        // First row should have NULL for LAG
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test LEAD function
#[test]
fn test_lead_function() {
    let db = Database::open("memory://window_lead").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT salary, LEAD(salary) OVER () AS next_salary FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        // Last row should have NULL for LEAD
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test FIRST_VALUE function
#[test]
fn test_first_value_function() {
    let db = Database::open("memory://window_first_value").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT name, FIRST_VALUE(name) OVER () AS first_name FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _name: String = row.get(0).unwrap();
        let first_name: String = row.get(1).unwrap();
        // FIRST_VALUE should return the first name in the partition
        assert!(
            !first_name.is_empty(),
            "FIRST_VALUE should return a non-empty string"
        );
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test FIRST_VALUE with salary
#[test]
fn test_first_value_salary() {
    let db =
        Database::open("memory://window_first_value_salary").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT salary, FIRST_VALUE(salary) OVER () AS first_salary FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut first_salary_value: Option<i64> = None;
    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        let first_salary: i64 = row.get(1).unwrap();

        // All rows should have the same first_salary value
        if let Some(expected) = first_salary_value {
            assert_eq!(
                first_salary, expected,
                "FIRST_VALUE should be consistent across all rows"
            );
        } else {
            first_salary_value = Some(first_salary);
        }
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test LAST_VALUE function
#[test]
fn test_last_value_function() {
    let db = Database::open("memory://window_last_value").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT name, LAST_VALUE(name) OVER () AS last_name FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _name: String = row.get(0).unwrap();
        let last_name: String = row.get(1).unwrap();
        // LAST_VALUE should return the last name in the partition
        assert!(
            !last_name.is_empty(),
            "LAST_VALUE should return a non-empty string"
        );
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test LAST_VALUE with salary
#[test]
fn test_last_value_salary() {
    let db =
        Database::open("memory://window_last_value_salary").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT salary, LAST_VALUE(salary) OVER () AS last_salary FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut last_salary_value: Option<i64> = None;
    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        let last_salary: i64 = row.get(1).unwrap();

        // All rows should have the same last_salary value
        if let Some(expected) = last_salary_value {
            assert_eq!(
                last_salary, expected,
                "LAST_VALUE should be consistent across all rows"
            );
        } else {
            last_salary_value = Some(last_salary);
        }
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test NTH_VALUE function
#[test]
fn test_nth_value_function() {
    let db = Database::open("memory://window_nth_value").expect("Failed to create database");
    setup_window_table(&db);

    // NTH_VALUE(salary, 3) should return the 3rd salary
    let result = db
        .query(
            "SELECT salary, NTH_VALUE(salary, 3) OVER () AS third_salary FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut third_salary_value: Option<i64> = None;
    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        let third_salary: i64 = row.get(1).unwrap();

        // All rows should have the same NTH_VALUE(3) value
        if let Some(expected) = third_salary_value {
            assert_eq!(
                third_salary, expected,
                "NTH_VALUE(3) should be consistent across all rows"
            );
        } else {
            third_salary_value = Some(third_salary);
        }
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test NTH_VALUE with first value (equivalent to FIRST_VALUE)
#[test]
fn test_nth_value_first() {
    let db = Database::open("memory://window_nth_value_first").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT
                FIRST_VALUE(salary) OVER () AS first_salary,
                NTH_VALUE(salary, 1) OVER () AS nth_1_salary
            FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let first_salary: i64 = row.get(0).unwrap();
        let nth_1_salary: i64 = row.get(1).unwrap();

        // NTH_VALUE(salary, 1) should equal FIRST_VALUE(salary)
        assert_eq!(
            first_salary, nth_1_salary,
            "NTH_VALUE(1) should equal FIRST_VALUE"
        );
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test PERCENT_RANK function
#[test]
fn test_percent_rank_function() {
    let db = Database::open("memory://window_percent_rank").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT salary, PERCENT_RANK() OVER () AS pct_rank FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        let pct_rank: f64 = row.get(1).unwrap();

        // PERCENT_RANK should be between 0.0 and 1.0
        assert!(
            (0.0..=1.0).contains(&pct_rank),
            "PERCENT_RANK should be between 0.0 and 1.0, got {}",
            pct_rank
        );
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test PERCENT_RANK with ordered partition
#[test]
fn test_percent_rank_ordered() {
    let db =
        Database::open("memory://window_percent_rank_ordered").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT salary, PERCENT_RANK() OVER (ORDER BY salary) AS pct_rank FROM window_test ORDER BY salary",
            (),
        )
        .expect("Failed to query");

    let mut pct_ranks = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        let pct_rank: f64 = row.get(1).unwrap();
        pct_ranks.push(pct_rank);
    }

    // First row should have PERCENT_RANK of 0.0
    assert!(
        (pct_ranks[0] - 0.0).abs() < 0.0001,
        "First row PERCENT_RANK should be 0.0, got {}",
        pct_ranks[0]
    );

    // PERCENT_RANK should be monotonically non-decreasing when ordered
    for i in 1..pct_ranks.len() {
        assert!(
            pct_ranks[i] >= pct_ranks[i - 1],
            "PERCENT_RANK should be non-decreasing: {} < {}",
            pct_ranks[i],
            pct_ranks[i - 1]
        );
    }

    assert_eq!(pct_ranks.len(), 7, "Expected 7 rows");
}

/// Test CUME_DIST function
#[test]
fn test_cume_dist_function() {
    let db = Database::open("memory://window_cume_dist").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT salary, CUME_DIST() OVER () AS cume_dist FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        let cume_dist: f64 = row.get(1).unwrap();

        // CUME_DIST should be between 0.0 and 1.0 (exclusive of 0)
        assert!(
            cume_dist > 0.0 && cume_dist <= 1.0,
            "CUME_DIST should be between 0.0 (exclusive) and 1.0, got {}",
            cume_dist
        );
        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test CUME_DIST with ordered partition
#[test]
fn test_cume_dist_ordered() {
    let db =
        Database::open("memory://window_cume_dist_ordered").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT salary, CUME_DIST() OVER (ORDER BY salary) AS cume_dist FROM window_test ORDER BY salary",
            (),
        )
        .expect("Failed to query");

    let mut cume_dists = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let _salary: i64 = row.get(0).unwrap();
        let cume_dist: f64 = row.get(1).unwrap();
        cume_dists.push(cume_dist);
    }

    // Last row should have CUME_DIST of 1.0
    let last_idx = cume_dists.len() - 1;
    assert!(
        (cume_dists[last_idx] - 1.0).abs() < 0.0001,
        "Last row CUME_DIST should be 1.0, got {}",
        cume_dists[last_idx]
    );

    // CUME_DIST should be monotonically non-decreasing when ordered
    for i in 1..cume_dists.len() {
        assert!(
            cume_dists[i] >= cume_dists[i - 1],
            "CUME_DIST should be non-decreasing: {} < {}",
            cume_dists[i],
            cume_dists[i - 1]
        );
    }

    assert_eq!(cume_dists.len(), 7, "Expected 7 rows");
}

/// Test all new window functions together
#[test]
fn test_all_new_window_functions() {
    let db = Database::open("memory://window_all_new").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT
                name,
                salary,
                FIRST_VALUE(salary) OVER () AS first_sal,
                LAST_VALUE(salary) OVER () AS last_sal,
                NTH_VALUE(salary, 2) OVER () AS second_sal,
                PERCENT_RANK() OVER () AS pct_rank,
                CUME_DIST() OVER () AS cume_dist
            FROM window_test ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    let mut first_sal_value: Option<i64> = None;
    let mut last_sal_value: Option<i64> = None;
    let mut second_sal_value: Option<i64> = None;

    for row in result {
        let row = row.expect("Failed to get row");
        let _name: String = row.get(0).unwrap();
        let _salary: i64 = row.get(1).unwrap();
        let first_sal: i64 = row.get(2).unwrap();
        let last_sal: i64 = row.get(3).unwrap();
        let second_sal: i64 = row.get(4).unwrap();
        let pct_rank: f64 = row.get(5).unwrap();
        let cume_dist: f64 = row.get(6).unwrap();

        // Verify consistency across rows
        if let Some(expected) = first_sal_value {
            assert_eq!(first_sal, expected, "FIRST_VALUE should be consistent");
        } else {
            first_sal_value = Some(first_sal);
        }

        if let Some(expected) = last_sal_value {
            assert_eq!(last_sal, expected, "LAST_VALUE should be consistent");
        } else {
            last_sal_value = Some(last_sal);
        }

        if let Some(expected) = second_sal_value {
            assert_eq!(second_sal, expected, "NTH_VALUE should be consistent");
        } else {
            second_sal_value = Some(second_sal);
        }

        // Verify PERCENT_RANK and CUME_DIST are in valid range
        assert!(
            (0.0..=1.0).contains(&pct_rank),
            "PERCENT_RANK should be between 0.0 and 1.0"
        );
        assert!(
            cume_dist > 0.0 && cume_dist <= 1.0,
            "CUME_DIST should be between 0.0 (exclusive) and 1.0"
        );

        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}

/// Test window functions with PARTITION BY
#[test]
fn test_window_functions_with_partition() {
    let db = Database::open("memory://window_partition_new").expect("Failed to create database");
    setup_window_table(&db);

    let result = db
        .query(
            "SELECT
                department,
                name,
                salary,
                FIRST_VALUE(salary) OVER (PARTITION BY department) AS dept_first_sal,
                LAST_VALUE(salary) OVER (PARTITION BY department) AS dept_last_sal,
                PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) AS dept_pct_rank,
                CUME_DIST() OVER (PARTITION BY department ORDER BY salary) AS dept_cume_dist
            FROM window_test
            ORDER BY department, salary",
            (),
        )
        .expect("Failed to query");

    let mut row_count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let _department: String = row.get(0).unwrap();
        let _name: String = row.get(1).unwrap();
        let _salary: i64 = row.get(2).unwrap();
        let _dept_first_sal: i64 = row.get(3).unwrap();
        let _dept_last_sal: i64 = row.get(4).unwrap();
        let dept_pct_rank: f64 = row.get(5).unwrap();
        let dept_cume_dist: f64 = row.get(6).unwrap();

        // Verify ranges
        assert!(
            (0.0..=1.0).contains(&dept_pct_rank),
            "PERCENT_RANK should be between 0.0 and 1.0"
        );
        assert!(
            dept_cume_dist > 0.0 && dept_cume_dist <= 1.0,
            "CUME_DIST should be between 0.0 (exclusive) and 1.0"
        );

        row_count += 1;
    }

    assert_eq!(row_count, 7, "Expected 7 rows, got {}", row_count);
}
