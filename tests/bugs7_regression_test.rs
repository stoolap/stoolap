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

// Regression tests for Bug Batch 7
// Tests for bugs #44, #45, #47 discovered during exploratory testing

use stoolap::Database;

fn setup_db() -> Database {
    Database::open_in_memory().expect("Failed to create in-memory database")
}

// =============================================================================
// BUG #44: IN subquery on CTE returns empty results
// Problem: WITH cte AS (...) SELECT * FROM cte WHERE val IN (SELECT val FROM cte...)
//          returned empty results instead of matching rows
// =============================================================================

#[test]
fn test_bugs7_in_subquery_on_same_cte() {
    let db = setup_db();

    db.execute("CREATE TABLE t44 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t44 VALUES (1, 'a'), (2, 'b'), (3, 'b')", ())
        .expect("Failed to insert data");

    // IN subquery on the same CTE should work
    let mut rows = db
        .query(
            "WITH cte AS (SELECT * FROM t44)
             SELECT * FROM cte WHERE val IN (SELECT val FROM cte WHERE id = 2)
             ORDER BY id",
            (),
        )
        .expect("Query failed");

    // Should return rows with val='b' (ids 2 and 3)
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<String>(1).unwrap(), "b");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<String>(1).unwrap(), "b");

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs7_in_subquery_on_different_cte() {
    let db = setup_db();

    db.execute("CREATE TABLE t44b (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t44b VALUES (1, 'x'), (2, 'y'), (3, 'z')", ())
        .expect("Failed to insert data");

    // IN subquery referencing a different CTE
    let mut rows = db
        .query(
            "WITH
                cte1 AS (SELECT * FROM t44b),
                cte2 AS (SELECT val FROM t44b WHERE id > 1)
             SELECT * FROM cte1 WHERE val IN (SELECT val FROM cte2)
             ORDER BY id",
            (),
        )
        .expect("Query failed");

    // Should return rows 2 and 3 (val = 'y' and 'z')
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<String>(1).unwrap(), "y");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<String>(1).unwrap(), "z");

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs7_not_in_subquery_on_cte() {
    let db = setup_db();

    db.execute("CREATE TABLE t44c (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t44c VALUES (1, 'a'), (2, 'b'), (3, 'c')", ())
        .expect("Failed to insert data");

    // NOT IN subquery on CTE
    let mut rows = db
        .query(
            "WITH cte AS (SELECT * FROM t44c)
             SELECT * FROM cte WHERE val NOT IN (SELECT val FROM cte WHERE id > 1)
             ORDER BY id",
            (),
        )
        .expect("Query failed");

    // Should return only row 1 (val='a' is not in ('b', 'c'))
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<String>(1).unwrap(), "a");

    assert!(rows.next().is_none());
}

// =============================================================================
// BUG #45: GROUPING() function returns NULL instead of 0/1
// Problem: GROUPING(col) in ROLLUP/CUBE queries returned NULL for all rows
//          instead of 0 (actively grouped) or 1 (rolled up)
// =============================================================================

#[test]
fn test_bugs7_grouping_function_rollup() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t45 (region TEXT, product TEXT, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t45 VALUES ('East', 'Widget', 100), ('East', 'Gadget', 200)",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO t45 VALUES ('West', 'Widget', 150), ('West', 'Gadget', 250)",
        (),
    )
    .expect("Failed to insert data");

    // Test GROUPING() function with ROLLUP
    let mut rows = db
        .query(
            "SELECT region, product, SUM(amount), GROUPING(region) as gr, GROUPING(product) as gp
             FROM t45
             GROUP BY ROLLUP(region, product)
             ORDER BY gr, gp, region NULLS LAST, product NULLS LAST",
            (),
        )
        .expect("Query failed");

    // Regular rows: GROUPING should be 0 for both
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(3).unwrap(), 0); // GROUPING(region) = 0
    assert_eq!(row.get::<i64>(4).unwrap(), 0); // GROUPING(product) = 0

    // Skip to subtotals (region, NULL): GROUPING(product) = 1
    // We need to find a row where gp = 1
    let mut found_subtotal = false;
    loop {
        let row_opt = rows.next();
        if row_opt.is_none() {
            break;
        }
        let row = row_opt.unwrap().unwrap();
        let gp = row.get::<i64>(4).unwrap();
        if gp == 1 {
            let gr = row.get::<i64>(3).unwrap();
            if gr == 0 {
                // Subtotal row (region, NULL)
                found_subtotal = true;
                assert_eq!(gp, 1, "GROUPING(product) should be 1 for subtotal");
                break;
            }
        }
    }
    assert!(
        found_subtotal,
        "Should have at least one subtotal row with GROUPING(product) = 1"
    );
}

#[test]
fn test_bugs7_grouping_function_cube() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t45b (dept TEXT, year INTEGER, sales INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t45b VALUES ('IT', 2023, 100), ('IT', 2024, 200), ('HR', 2023, 150), ('HR', 2024, 250)",
        (),
    )
    .expect("Failed to insert data");

    // Test GROUPING() function with CUBE - should return all combinations
    let mut rows = db
        .query(
            "SELECT dept, year, SUM(sales), GROUPING(dept) as gd, GROUPING(year) as gy
             FROM t45b
             GROUP BY CUBE(dept, year)
             ORDER BY gd, gy",
            (),
        )
        .expect("Query failed");

    // Count rows with different grouping patterns
    let mut pattern_00 = 0; // Regular rows
    let mut pattern_01 = 0; // Subtotal by dept
    let mut pattern_10 = 0; // Subtotal by year
    let mut pattern_11 = 0; // Grand total

    loop {
        let row_opt = rows.next();
        if row_opt.is_none() {
            break;
        }
        let row = row_opt.unwrap().unwrap();
        let gd = row.get::<i64>(3).unwrap();
        let gy = row.get::<i64>(4).unwrap();
        match (gd, gy) {
            (0, 0) => pattern_00 += 1,
            (0, 1) => pattern_01 += 1,
            (1, 0) => pattern_10 += 1,
            (1, 1) => pattern_11 += 1,
            _ => panic!("Unexpected grouping pattern: ({}, {})", gd, gy),
        }
    }

    assert!(pattern_00 > 0, "Should have regular grouped rows");
    assert!(pattern_01 > 0, "Should have subtotals by dept");
    assert!(pattern_10 > 0, "Should have subtotals by year");
    assert_eq!(pattern_11, 1, "Should have exactly one grand total");
}

#[test]
fn test_bugs7_grouping_function_grand_total() {
    let db = setup_db();

    db.execute("CREATE TABLE t45c (cat TEXT, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t45c VALUES ('A', 10), ('B', 20)", ())
        .expect("Failed to insert data");

    // Grand total row should have GROUPING() = 1
    let mut rows = db
        .query(
            "SELECT cat, SUM(val), GROUPING(cat) as g
             FROM t45c
             GROUP BY ROLLUP(cat)
             ORDER BY g DESC",
            (),
        )
        .expect("Query failed");

    // First row should be grand total with GROUPING = 1
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(2).unwrap(), 1);
    assert_eq!(row.get::<i64>(1).unwrap(), 30); // SUM of all
}

// =============================================================================
// BUG #47: NTH_VALUE ignores frame bounds
// Problem: NTH_VALUE(val, N) returned value from entire partition
//          instead of respecting window frame bounds
// =============================================================================

#[test]
fn test_bugs7_nth_value_default_frame() {
    let db = setup_db();

    db.execute("CREATE TABLE t47 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t47 VALUES (1, 100), (2, 200), (3, 300)", ())
        .expect("Failed to insert data");

    // Default frame is RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    // Row 1: frame=[100], NTH_VALUE(2) should be NULL
    // Row 2: frame=[100, 200], NTH_VALUE(2) should be 200
    // Row 3: frame=[100, 200, 300], NTH_VALUE(2) should be 200
    let mut rows = db
        .query(
            "SELECT id, val, NTH_VALUE(val, 2) OVER (ORDER BY id) as nth2
             FROM t47
             ORDER BY id",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert!(
        row.get::<i64>(2).is_err() || row.get::<i64>(2).unwrap_or(-1) == -1,
        "Row 1 NTH_VALUE(2) should be NULL (only 1 row in frame)"
    );

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(2).unwrap(), 200);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<i64>(2).unwrap(), 200);
}

#[test]
fn test_bugs7_nth_value_explicit_frame() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t47b (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t47b VALUES (1, 'A', 100), (2, 'A', 200), (3, 'A', 300)",
        (),
    )
    .expect("Failed to insert data");
    db.execute("INSERT INTO t47b VALUES (4, 'B', 400), (5, 'B', 500)", ())
        .expect("Failed to insert data");

    // Explicit frame: ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING
    let mut rows = db
        .query(
            "SELECT id, grp, val,
                    NTH_VALUE(val, 2) OVER (
                        PARTITION BY grp ORDER BY id
                        ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING
                    ) as nth2
             FROM t47b
             ORDER BY id",
            (),
        )
        .expect("Query failed");

    // Row 1: frame=[100, 200], NTH_VALUE(2) = 200
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(3).unwrap(), 200);

    // Row 2: frame=[200, 300], NTH_VALUE(2) = 300
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(3).unwrap(), 300);

    // Row 3: frame=[300] (last in partition), NTH_VALUE(2) = NULL
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert!(
        row.get::<i64>(3).is_err(),
        "Row 3 NTH_VALUE(2) should be NULL (only 1 row in frame)"
    );

    // Row 4: frame=[400, 500], NTH_VALUE(2) = 500
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 4);
    assert_eq!(row.get::<i64>(3).unwrap(), 500);

    // Row 5: frame=[500] (last in partition), NTH_VALUE(2) = NULL
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);
    assert!(
        row.get::<i64>(3).is_err(),
        "Row 5 NTH_VALUE(2) should be NULL (only 1 row in frame)"
    );
}

#[test]
fn test_bugs7_nth_value_current_row_frame() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t47c (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t47c VALUES (1, 100), (2, 200), (3, 300)", ())
        .expect("Failed to insert data");

    // Frame of exactly 1 row: ROWS BETWEEN CURRENT ROW AND CURRENT ROW
    // NTH_VALUE(val, 2) should always be NULL since frame only has 1 row
    let mut rows = db
        .query(
            "SELECT id, val,
                    NTH_VALUE(val, 2) OVER (
                        ORDER BY id
                        ROWS BETWEEN CURRENT ROW AND CURRENT ROW
                    ) as nth2
             FROM t47c
             ORDER BY id",
            (),
        )
        .expect("Query failed");

    for _ in 0..3 {
        let row = rows.next().unwrap().unwrap();
        assert!(
            row.get::<i64>(2).is_err(),
            "NTH_VALUE(2) should be NULL when frame has only 1 row"
        );
    }
}

#[test]
fn test_bugs7_first_value_default_frame() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t47d (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t47d VALUES (1, 100), (2, 200), (3, 300)", ())
        .expect("Failed to insert data");

    // FIRST_VALUE should use default frame (UNBOUNDED PRECEDING to CURRENT ROW)
    let mut rows = db
        .query(
            "SELECT id, val, FIRST_VALUE(val) OVER (ORDER BY id) as first_val
             FROM t47d
             ORDER BY id",
            (),
        )
        .expect("Query failed");

    // All rows should see 100 as first value
    for _ in 0..3 {
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get::<i64>(2).unwrap(), 100);
    }
}
