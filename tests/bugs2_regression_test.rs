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

// Integration tests for all bugs tracked in BUGS2.md
// These tests ensure that fixed bugs don't regress

use stoolap::Database;

fn setup_db(name: &str) -> Database {
    Database::open(&format!("memory://{}", name)).expect("Failed to create database")
}

// =============================================================================
// Bug #1: ORDER BY fails with ROLLUP/CUBE
// Description: Parser fails when ORDER BY clause is used with GROUP BY ROLLUP or CUBE.
// =============================================================================
#[test]
fn test_bugs2_01_order_by_with_rollup() {
    let db = setup_db("bugs2_01");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO sales VALUES (1, 'North', 100)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO sales VALUES (2, 'North', 200)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO sales VALUES (3, 'South', 150)", ())
        .expect("Insert failed");

    // ORDER BY with ROLLUP should parse and execute
    let result = db
        .query(
            "SELECT region, SUM(amount) as total FROM sales GROUP BY ROLLUP(region) ORDER BY total",
            (),
        )
        .expect("Query with ROLLUP and ORDER BY should succeed");

    let rows: Vec<_> = result.collect();
    assert!(rows.len() >= 3, "Should have at least 3 rows with rollup");
}

// =============================================================================
// Bug #2: Correlated subqueries in UPDATE return NULL
// Description: UPDATE statements with correlated subqueries set all values to NULL
// =============================================================================
#[test]
fn test_bugs2_02_correlated_update() {
    let db = setup_db("bugs2_02");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "CREATE TABLE discounts (id INTEGER PRIMARY KEY, product_id INTEGER, amount INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO products VALUES (1, 'A', 100)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO products VALUES (2, 'B', 200)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO discounts VALUES (1, 1, 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO discounts VALUES (2, 2, 20)", ())
        .expect("Insert failed");

    db.execute(
        "UPDATE products SET price = price - (SELECT amount FROM discounts WHERE product_id = products.id)",
        (),
    )
    .expect("Correlated UPDATE should succeed");

    let result = db
        .query("SELECT id, price FROM products ORDER BY id", ())
        .expect("Query failed");
    let rows: Vec<_> = result.collect();

    let row0 = rows[0].as_ref().unwrap();
    let price0: i64 = row0.get(1).unwrap();
    assert_eq!(price0, 90, "Product 1 price should be 100 - 10 = 90");

    let row1 = rows[1].as_ref().unwrap();
    let price1: i64 = row1.get(1).unwrap();
    assert_eq!(price1, 180, "Product 2 price should be 200 - 20 = 180");
}

// =============================================================================
// Bug #3: CASE WHEN with window function comparison returns NULL
// Description: When a window function is used in a comparison inside a CASE WHEN clause
// =============================================================================
#[test]
fn test_bugs2_03_case_with_window_comparison() {
    let db = setup_db("bugs2_03");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT id, val, CASE WHEN ROW_NUMBER() OVER (ORDER BY val) = 1 THEN 'first' ELSE 'other' END as label FROM t",
            (),
        )
        .expect("CASE with window comparison should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // Check that labels are not NULL
    for row in &rows {
        let label: String = row.as_ref().unwrap().get(2).unwrap();
        assert!(
            label == "first" || label == "other",
            "Label should be 'first' or 'other', not NULL"
        );
    }
}

// =============================================================================
// Bug #4: ARRAY_AGG returns NULL
// Description: ARRAY_AGG aggregate function was not implemented
// =============================================================================
#[test]
fn test_bugs2_04_array_agg() {
    let db = setup_db("bugs2_04");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT ARRAY_AGG(val) FROM t", ())
        .expect("ARRAY_AGG should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let arr: String = row.get(0).unwrap();
    // Should be a JSON array like [10,20,30]
    assert!(
        arr.starts_with('[') && arr.ends_with(']'),
        "Should be JSON array format"
    );
}

// =============================================================================
// Bug #5: RANGE frame specification doesn't work correctly
// Description: Window functions with RANGE frame specification ignored the bounds
// =============================================================================
#[test]
fn test_bugs2_05_range_frame() {
    let db = setup_db("bugs2_05");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT id, val, SUM(val) OVER (ORDER BY val RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running FROM t ORDER BY val",
            (),
        )
        .expect("RANGE frame should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // Check running sums
    let row0 = rows[0].as_ref().unwrap();
    let running0: i64 = row0.get(2).unwrap();
    assert_eq!(running0, 10, "Running sum at 10 should be 10");

    let row1 = rows[1].as_ref().unwrap();
    let running1: i64 = row1.get(2).unwrap();
    assert_eq!(running1, 30, "Running sum at 20 should be 30");

    let row2 = rows[2].as_ref().unwrap();
    let running2: i64 = row2.get(2).unwrap();
    assert_eq!(running2, 60, "Running sum at 30 should be 60");
}

// =============================================================================
// Bug #6: CTE with column list syntax not supported
// Description: CTEs with column list syntax should work
// =============================================================================
#[test]
fn test_bugs2_06_cte_column_list() {
    let db = setup_db("bugs2_06");

    let result = db
        .query(
            "WITH cte(col1, col2) AS (SELECT 1, 2) SELECT * FROM cte",
            (),
        )
        .expect("CTE with column list should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let col1: i64 = row.get(0).unwrap();
    let col2: i64 = row.get(1).unwrap();
    assert_eq!(col1, 1);
    assert_eq!(col2, 2);
}

// =============================================================================
// Bug #7: Window function with expression argument returns wrong value
// Description: When a window aggregate has an expression argument (like val * 2)
// =============================================================================
#[test]
fn test_bugs2_07_window_expression_argument() {
    let db = setup_db("bugs2_07");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT id, val, SUM(val * 2) OVER (ORDER BY id) as running FROM t ORDER BY id",
            (),
        )
        .expect("Window function with expression argument should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // Should be: 20, 60, 120 (cumulative sum of val * 2)
    let row0 = rows[0].as_ref().unwrap();
    let running0: i64 = row0.get(2).unwrap();
    assert_eq!(running0, 20, "Running sum at id=1 should be 10*2=20");

    let row1 = rows[1].as_ref().unwrap();
    let running1: i64 = row1.get(2).unwrap();
    assert_eq!(running1, 60, "Running sum at id=2 should be 20+40=60");

    let row2 = rows[2].as_ref().unwrap();
    let running2: i64 = row2.get(2).unwrap();
    assert_eq!(running2, 120, "Running sum at id=3 should be 60+60=120");
}

// =============================================================================
// Bug #8: Window PARTITION BY ignores partition after JOIN
// Description: PARTITION BY was ignored in JOIN queries
// =============================================================================
#[test]
fn test_bugs2_08_window_partition_after_join() {
    let db = setup_db("bugs2_08");

    db.execute(
        "CREATE TABLE left_t (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "CREATE TABLE right_t (id INTEGER PRIMARY KEY, left_id INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO left_t VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 30)",
        (),
    )
    .expect("Insert failed");
    db.execute("INSERT INTO right_t VALUES (1, 1), (2, 2), (3, 3)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT l.id, l.grp, l.val, SUM(l.val) OVER (PARTITION BY l.grp) as sum_grp FROM left_t l JOIN right_t r ON l.id = r.left_id ORDER BY l.id",
            (),
        )
        .expect("PARTITION BY should work with JOIN");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // Group A sum should be 30 (10+20), Group B sum should be 30
    let row0 = rows[0].as_ref().unwrap();
    let sum0: i64 = row0.get(3).unwrap();
    assert_eq!(sum0, 30, "Group A sum should be 30");

    let row2 = rows[2].as_ref().unwrap();
    let sum2: i64 = row2.get(3).unwrap();
    assert_eq!(sum2, 30, "Group B sum should be 30");
}

// =============================================================================
// Bug #9: Nested aggregate in window function returns wrong value
// Description: SUM(SUM(val)) OVER () returned wrong value
// =============================================================================
#[test]
fn test_bugs2_09_nested_aggregate_window() {
    let db = setup_db("bugs2_09");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t VALUES (1, 'X', 100), (2, 'X', 200), (3, 'Y', 300), (4, 'Y', 400)",
        (),
    )
    .expect("Insert failed");

    let result = db
        .query(
            "SELECT grp, SUM(val) as grp_sum, SUM(SUM(val)) OVER () as total FROM t GROUP BY grp ORDER BY grp",
            (),
        )
        .expect("Nested aggregate in window should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);

    // X: sum=300, Y: sum=700, total=1000
    let row0 = rows[0].as_ref().unwrap();
    let grp_sum0: i64 = row0.get(1).unwrap();
    let total0: i64 = row0.get(2).unwrap();
    assert_eq!(grp_sum0, 300, "Group X sum should be 300");
    assert_eq!(total0, 1000, "Total should be 1000");

    let row1 = rows[1].as_ref().unwrap();
    let grp_sum1: i64 = row1.get(1).unwrap();
    let total1: i64 = row1.get(2).unwrap();
    assert_eq!(grp_sum1, 700, "Group Y sum should be 700");
    assert_eq!(total1, 1000, "Total should be 1000");
}

// =============================================================================
// Bug #10: Table-level constraint syntax not supported
// Description: Table-level UNIQUE constraint should be parsed
// =============================================================================
#[test]
fn test_bugs2_10_table_level_unique() {
    let db = setup_db("bugs2_10");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER, UNIQUE(val))",
        (),
    )
    .expect("Table-level UNIQUE should be parsed");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert failed");

    // Duplicate should fail
    let err = db.execute("INSERT INTO t VALUES (2, 10)", ());
    assert!(
        err.is_err(),
        "Duplicate value should violate unique constraint"
    );
}

// =============================================================================
// Bug #11: CAST of aggregate function fails
// Description: CAST(SUM(val) AS TEXT) failed with "Unknown function"
// =============================================================================
#[test]
fn test_bugs2_11_cast_aggregate() {
    let db = setup_db("bugs2_11");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT CAST(SUM(val) AS TEXT) as total_str FROM t", ())
        .expect("CAST of aggregate should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let total_str: String = row.get(0).unwrap();
    assert_eq!(total_str, "30", "CAST(SUM(val) AS TEXT) should be '30'");
}

// =============================================================================
// Bug #12: ORDER BY expression doesn't work when aliased in SELECT
// Description: ORDER BY val * 2 with SELECT val * 2 as doubled didn't work
// =============================================================================
#[test]
fn test_bugs2_12_order_by_aliased_expression() {
    let db = setup_db("bugs2_12");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 30), (2, 10), (3, 20)", ())
        .expect("Insert failed");

    // ORDER BY alias
    let result = db
        .query("SELECT id, val * 2 as doubled FROM t ORDER BY doubled", ())
        .expect("ORDER BY alias should work");

    let rows: Vec<_> = result.collect();
    let row0 = rows[0].as_ref().unwrap();
    let doubled0: i64 = row0.get(1).unwrap();
    assert_eq!(
        doubled0, 20,
        "First row should have smallest doubled value (10*2=20)"
    );

    // ORDER BY expression
    let result2 = db
        .query("SELECT id, val * 2 as doubled FROM t ORDER BY val * 2", ())
        .expect("ORDER BY expression should work");

    let rows2: Vec<_> = result2.collect();
    let row2_0 = rows2[0].as_ref().unwrap();
    let doubled2_0: i64 = row2_0.get(1).unwrap();
    assert_eq!(
        doubled2_0, 20,
        "First row should have smallest doubled value"
    );
}

// =============================================================================
// Bug #13: Window function fails with CTE + JOIN combination
// Description: CTE + JOIN + window function failed
// =============================================================================
#[test]
fn test_bugs2_13_cte_join_window() {
    let db = setup_db("bugs2_13");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, price FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 30)",
        (),
    )
    .expect("Insert failed");

    let result = db
        .query(
            "WITH cat_avg AS (SELECT category, AVG(price) as avg FROM products GROUP BY category)
             SELECT p.category, p.price, RANK() OVER (PARTITION BY p.category ORDER BY p.price) as rnk
             FROM products p JOIN cat_avg c ON p.category = c.category ORDER BY p.category, p.price",
            (),
        )
        .expect("CTE + JOIN + window should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);
}

// =============================================================================
// Bug #15: Modulo with NULL throws error instead of returning NULL
// Description: 10 % NULL should return NULL, not error
// =============================================================================
#[test]
fn test_bugs2_15_modulo_null() {
    let db = setup_db("bugs2_15");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t VALUES (1, 10, 3), (2, NULL, 5), (3, 20, NULL)",
        (),
    )
    .expect("Insert failed");

    let result = db
        .query("SELECT id, a, b, a % b AS modulo FROM t ORDER BY id", ())
        .expect("Modulo with NULL should not error");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // First row: 10 % 3 = 1
    let row0 = rows[0].as_ref().unwrap();
    let mod0: i64 = row0.get(3).unwrap();
    assert_eq!(mod0, 1);

    // Second row: NULL % 5 = NULL
    let row1 = rows[1].as_ref().unwrap();
    let mod1: Option<i64> = row1.get(3).ok();
    assert!(mod1.is_none(), "NULL % 5 should be NULL");

    // Third row: 20 % NULL = NULL
    let row2 = rows[2].as_ref().unwrap();
    let mod2: Option<i64> = row2.get(3).ok();
    assert!(mod2.is_none(), "20 % NULL should be NULL");
}

// =============================================================================
// Bug #16: NOT col = val includes NULL rows incorrectly
// Description: NOT val = 10 should not include NULL rows
// =============================================================================
#[test]
fn test_bugs2_16_not_equals_null() {
    let db = setup_db("bugs2_16");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10), (2, NULL), (3, 20)", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT id FROM t WHERE NOT val = 10 ORDER BY id", ())
        .expect("NOT val = 10 should work");

    let rows: Vec<_> = result.collect();
    // Should only return id=3 (val=20), NOT id=2 (val=NULL)
    assert_eq!(
        rows.len(),
        1,
        "Only row with val=20 should match NOT val = 10"
    );

    let row = rows[0].as_ref().unwrap();
    let id: i64 = row.get(0).unwrap();
    assert_eq!(id, 3, "Only id=3 should be returned");
}

// =============================================================================
// Bug #17: ORDER BY column name broken when alias present
// Description: ORDER BY val with SELECT val AS amount didn't work
// =============================================================================
#[test]
fn test_bugs2_17_order_by_column_with_alias() {
    let db = setup_db("bugs2_17");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (3, 100), (1, 200), (2, 150)", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT id, val AS amount FROM t ORDER BY val", ())
        .expect("ORDER BY original column name should work");

    let rows: Vec<_> = result.collect();

    // Should be ordered by val: 100, 150, 200
    let row0 = rows[0].as_ref().unwrap();
    let amount0: i64 = row0.get(1).unwrap();
    assert_eq!(amount0, 100, "First should be val=100");

    let row1 = rows[1].as_ref().unwrap();
    let amount1: i64 = row1.get(1).unwrap();
    assert_eq!(amount1, 150, "Second should be val=150");

    let row2 = rows[2].as_ref().unwrap();
    let amount2: i64 = row2.get(1).unwrap();
    assert_eq!(amount2, 200, "Third should be val=200");
}

// =============================================================================
// Bug #18: String concatenation (||) converts NULL to literal "NULL"
// Description: 'hello' || NULL should return NULL, not 'helloNULL'
// =============================================================================
#[test]
fn test_bugs2_18_concat_null() {
    let db = setup_db("bugs2_18");

    let result = db
        .query("SELECT 'hello' || NULL as result", ())
        .expect("Concatenation with NULL should work");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    // Use get_value to check for actual NULL instead of type conversion
    let value = row.get_value(0);
    assert!(
        value.map(|v| v.is_null()).unwrap_or(false),
        "'hello' || NULL should be NULL"
    );

    let result2 = db
        .query("SELECT NULL || 'world' as result", ())
        .expect("NULL concatenation should work");

    let rows2: Vec<_> = result2.collect();
    let row2 = rows2[0].as_ref().unwrap();
    let value2 = row2.get_value(0);
    assert!(
        value2.map(|v| v.is_null()).unwrap_or(false),
        "NULL || 'world' should be NULL"
    );
}

// =============================================================================
// Bug #19: ARRAY_AGG ORDER BY clause is ignored
// Description: ARRAY_AGG(val ORDER BY val) should sort the result
// =============================================================================
#[test]
fn test_bugs2_19_array_agg_order_by() {
    let db = setup_db("bugs2_19");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO t VALUES (1, 50), (2, 30), (3, 80), (4, 10)",
        (),
    )
    .expect("Insert failed");

    let result = db
        .query("SELECT ARRAY_AGG(val ORDER BY val) FROM t", ())
        .expect("ARRAY_AGG with ORDER BY should work");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let arr: String = row.get(0).unwrap();
    // Should be sorted: [10,30,50,80]
    assert_eq!(arr, "[10,30,50,80]", "ARRAY_AGG should be sorted ASC");

    let result_desc = db
        .query("SELECT ARRAY_AGG(val ORDER BY val DESC) FROM t", ())
        .expect("ARRAY_AGG with ORDER BY DESC should work");

    let rows_desc: Vec<_> = result_desc.collect();
    let row_desc = rows_desc[0].as_ref().unwrap();
    let arr_desc: String = row_desc.get(0).unwrap();
    assert_eq!(arr_desc, "[80,50,30,10]", "ARRAY_AGG should be sorted DESC");
}

// =============================================================================
// Bug #20: DECIMAL(precision, scale) syntax fails
// Description: CREATE TABLE with DECIMAL(10,2) failed
// =============================================================================
#[test]
fn test_bugs2_20_decimal_syntax() {
    let db = setup_db("bugs2_20");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, amount DECIMAL(10, 2))",
        (),
    )
    .expect("DECIMAL(precision, scale) should parse");

    db.execute("INSERT INTO t VALUES (1, 123.45)", ())
        .expect("Insert should work");

    let result = db.query("SELECT * FROM t", ()).expect("Query should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
}

// =============================================================================
// Bug #21: ANY/ALL subqueries return empty results
// Description: price > ALL (SELECT ...) returned empty results
// =============================================================================
#[test]
fn test_bugs2_21_any_all_subqueries() {
    let db = setup_db("bugs2_21");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT, category_id INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO products VALUES (1, 'Phone', 500, 1)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO products VALUES (2, 'Laptop', 1000, 1)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO products VALUES (3, 'Apple', 2, 3)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO products VALUES (4, 'Bread', 3, 3)", ())
        .expect("Insert failed");

    // > ALL: price greater than all prices in category 3 (max is 3)
    let result = db
        .query(
            "SELECT name, price FROM products WHERE price > ALL (SELECT price FROM products WHERE category_id = 3) ORDER BY price",
            (),
        )
        .expect("ANY/ALL subquery should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Phone and Laptop have price > 3");

    // = ANY: price equals any price in category 3
    let result2 = db
        .query(
            "SELECT name FROM products WHERE price = ANY (SELECT price FROM products WHERE category_id = 3) ORDER BY name",
            (),
        )
        .expect("= ANY should work");

    let rows2: Vec<_> = result2.collect();
    assert_eq!(rows2.len(), 2, "Apple and Bread match");
}

// =============================================================================
// Bug #22: NULLS FIRST/LAST causes parse error
// Description: ORDER BY val NULLS FIRST should work
// =============================================================================
#[test]
fn test_bugs2_22_nulls_first_last() {
    let db = setup_db("bugs2_22");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 30), (2, NULL), (3, 10)", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT id, val FROM t ORDER BY val ASC NULLS FIRST", ())
        .expect("NULLS FIRST should work");

    let rows: Vec<_> = result.collect();
    // NULL should be first
    let row0 = rows[0].as_ref().unwrap();
    let val0: Option<i64> = row0.get(1).ok();
    assert!(val0.is_none(), "NULL should be first");

    let result2 = db
        .query("SELECT id, val FROM t ORDER BY val ASC NULLS LAST", ())
        .expect("NULLS LAST should work");

    let rows2: Vec<_> = result2.collect();
    // NULL should be last
    let row2_last = rows2[2].as_ref().unwrap();
    let val_last: Option<i64> = row2_last.get(1).ok();
    assert!(val_last.is_none(), "NULL should be last");
}

// =============================================================================
// Bug #25: EXPLAIN produces no output
// Description: EXPLAIN statement was not implemented
// =============================================================================
#[test]
fn test_bugs2_25_explain() {
    let db = setup_db("bugs2_25");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    let result = db
        .query("EXPLAIN SELECT * FROM t WHERE val > 10", ())
        .expect("EXPLAIN should work");

    let rows: Vec<_> = result.collect();
    assert!(!rows.is_empty(), "EXPLAIN should produce output");
}

// =============================================================================
// Bug #26: INSERT with CTE fails
// Description: WITH ... INSERT INTO ... SELECT failed
// =============================================================================
#[test]
fn test_bugs2_26_insert_with_cte() {
    let db = setup_db("bugs2_26");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "WITH data AS (SELECT 1 as id, 100 as val) INSERT INTO t SELECT * FROM data",
        (),
    )
    .expect("INSERT with CTE should work");

    let result = db.query("SELECT * FROM t", ()).expect("Query should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let id: i64 = row.get(0).unwrap();
    let val: i64 = row.get(1).unwrap();
    assert_eq!(id, 1);
    assert_eq!(val, 100);
}

// =============================================================================
// Bug #27: View with t.* in JOIN stores columns with alias prefix
// Description: View column names had alias prefix like "e.name" instead of "name"
// =============================================================================
#[test]
fn test_bugs2_27_view_qualified_star() {
    let db = setup_db("bugs2_27");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, dname TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO employees VALUES (1, 'Alice')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO departments VALUES (1, 'Engineering')", ())
        .expect("Insert failed");

    db.execute(
        "CREATE VIEW emp_dept AS SELECT e.id, e.name, d.dname FROM employees e JOIN departments d ON e.id = d.id",
        (),
    )
    .expect("Create view should work");

    // Selecting specific column should work
    let result = db
        .query("SELECT name FROM emp_dept", ())
        .expect("SELECT name should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let name: String = row.get(0).unwrap();
    assert_eq!(name, "Alice");
}

// =============================================================================
// Bug #28: RETURNING * fails for INSERT and DELETE
// Description: RETURNING * caused parse error
// =============================================================================
#[test]
fn test_bugs2_28_returning_star() {
    let db = setup_db("bugs2_28");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    let result = db
        .query("INSERT INTO t VALUES (1, 100) RETURNING *", ())
        .expect("INSERT RETURNING * should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let id: i64 = row.get(0).unwrap();
    let val: i64 = row.get(1).unwrap();
    assert_eq!(id, 1);
    assert_eq!(val, 100);

    let result2 = db
        .query("DELETE FROM t WHERE id = 1 RETURNING *", ())
        .expect("DELETE RETURNING * should work");

    let rows2: Vec<_> = result2.collect();
    assert_eq!(rows2.len(), 1);
}

// =============================================================================
// Bug #29: SHOW CREATE TABLE doesn't show CHECK and UNIQUE constraints
// Description: CHECK and UNIQUE constraints were not displayed
// =============================================================================
#[test]
fn test_bugs2_29_show_create_constraints() {
    let db = setup_db("bugs2_29");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, age INTEGER CHECK(age >= 0), email TEXT UNIQUE)",
        (),
    )
    .expect("Failed to create table");

    let result = db
        .query("SHOW CREATE TABLE t", ())
        .expect("SHOW CREATE TABLE should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let create_sql: String = row.get(1).unwrap();

    // Should contain CHECK and UNIQUE
    assert!(create_sql.contains("CHECK"), "Should show CHECK constraint");
    assert!(
        create_sql.contains("UNIQUE"),
        "Should show UNIQUE constraint"
    );
}

// =============================================================================
// Bug #30: Unqualified column names disappear with JOIN + window function
// Description: SELECT name, ... with JOIN and window function lost the column
// =============================================================================
#[test]
fn test_bugs2_30_unqualified_columns_with_window() {
    let db = setup_db("bugs2_30");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "CREATE TABLE salaries (id INTEGER PRIMARY KEY, emp_id INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO employees VALUES (1, 'Alice', 100), (2, 'Bob', 200)",
        (),
    )
    .expect("Insert failed");
    db.execute("INSERT INTO salaries VALUES (1, 1), (2, 2)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) as rn FROM employees e JOIN salaries s ON e.id = s.emp_id ORDER BY salary DESC",
            (),
        )
        .expect("Unqualified columns with JOIN + window should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);

    // Check that name column is not null
    let row0 = rows[0].as_ref().unwrap();
    let name0: String = row0.get(0).unwrap();
    assert!(!name0.is_empty(), "name column should not be empty");
}

// =============================================================================
// Bug #31: DELETE with NOT EXISTS correlated subquery doesn't work
// Description: DELETE with NOT EXISTS didn't delete matching rows
// =============================================================================
#[test]
fn test_bugs2_31_delete_not_exists() {
    let db = setup_db("bugs2_31");

    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, active INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO customers VALUES (1, 'Alice', 1), (2, 'Bob', 0), (3, 'Charlie', 1)",
        (),
    )
    .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (1, 1), (2, 3)", ())
        .expect("Insert failed");

    // Delete inactive customers with no orders
    db.execute(
        "DELETE FROM customers WHERE active = 0 AND NOT EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id)",
        (),
    )
    .expect("DELETE with NOT EXISTS should work");

    let result = db
        .query("SELECT name FROM customers ORDER BY name", ())
        .expect("Query failed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Bob should be deleted");

    let names: Vec<String> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert!(names.contains(&"Alice".to_string()));
    assert!(names.contains(&"Charlie".to_string()));
    assert!(!names.contains(&"Bob".to_string()));
}

// =============================================================================
// Bug #32: Window function on aggregate with JOIN returns NULL
// Description: JOIN + GROUP BY + window function on aggregate returned NULL
// =============================================================================
#[test]
fn test_bugs2_32_window_on_aggregate_with_join() {
    let db = setup_db("bugs2_32");

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())
        .expect("Insert failed");
    db.execute(
        "INSERT INTO orders VALUES (1, 1, 100), (2, 1, 200), (3, 2, 50)",
        (),
    )
    .expect("Insert failed");

    let result = db
        .query(
            "SELECT u.name, SUM(o.amount) as total, SUM(SUM(o.amount)) OVER () as grand_total FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.name ORDER BY u.name",
            (),
        )
        .expect("Window function on aggregate with JOIN should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);

    // Alice: total=300, Bob: total=50, grand_total=350
    let row0 = rows[0].as_ref().unwrap();
    let total0: f64 = row0.get(1).unwrap();
    let grand0: f64 = row0.get(2).unwrap();
    assert!((total0 - 300.0).abs() < 0.01, "Alice total should be 300");
    assert!((grand0 - 350.0).abs() < 0.01, "Grand total should be 350");

    let row1 = rows[1].as_ref().unwrap();
    let total1: f64 = row1.get(1).unwrap();
    let grand1: f64 = row1.get(2).unwrap();
    assert!((total1 - 50.0).abs() < 0.01, "Bob total should be 50");
    assert!((grand1 - 350.0).abs() < 0.01, "Grand total should be 350");
}

// =============================================================================
// Bug #33: RANK/DENSE_RANK return wrong values with GROUP BY
// Description: RANK() OVER (ORDER BY SUM(val) DESC) returned 1 for all rows
// =============================================================================
#[test]
fn test_bugs2_33_rank_with_group_by() {
    let db = setup_db("bugs2_33");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 5), (4, 'C', 50)",
        (),
    )
    .expect("Insert failed");

    let result = db
        .query(
            "SELECT grp, SUM(val) as total, RANK() OVER (ORDER BY SUM(val) DESC) as rnk FROM t GROUP BY grp ORDER BY total DESC",
            (),
        )
        .expect("RANK with GROUP BY should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // C: 50 -> rank 1, A: 30 -> rank 2, B: 5 -> rank 3
    let row0 = rows[0].as_ref().unwrap();
    let grp0: String = row0.get(0).unwrap();
    let rnk0: i64 = row0.get(2).unwrap();
    assert_eq!(grp0, "C");
    assert_eq!(rnk0, 1, "C should have rank 1");

    let row1 = rows[1].as_ref().unwrap();
    let grp1: String = row1.get(0).unwrap();
    let rnk1: i64 = row1.get(2).unwrap();
    assert_eq!(grp1, "A");
    assert_eq!(rnk1, 2, "A should have rank 2");

    let row2 = rows[2].as_ref().unwrap();
    let grp2: String = row2.get(0).unwrap();
    let rnk2: i64 = row2.get(2).unwrap();
    assert_eq!(grp2, "B");
    assert_eq!(rnk2, 3, "B should have rank 3");
}

// =============================================================================
// Bug #34: COALESCE ignored when wrapping aggregate functions
// Description: COALESCE(SUM(val), 0) returned NULL instead of 0
// =============================================================================
#[test]
fn test_bugs2_34_coalesce_aggregate() {
    let db = setup_db("bugs2_34");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert failed");

    // No matching rows - should return 0, not NULL
    let result = db
        .query(
            "SELECT COALESCE(SUM(val), 0) as total FROM t WHERE id = 99",
            (),
        )
        .expect("COALESCE with aggregate should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let total: i64 = row.get(0).unwrap();
    assert_eq!(
        total, 0,
        "COALESCE(SUM(val), 0) should return 0 when no rows match"
    );

    // Test with AVG
    let result2 = db
        .query(
            "SELECT COALESCE(AVG(val), -1) as avg FROM t WHERE id = 99",
            (),
        )
        .expect("COALESCE with AVG should work");

    let rows2: Vec<_> = result2.collect();
    let row2 = rows2[0].as_ref().unwrap();
    let avg: f64 = row2.get(0).unwrap();
    assert!(
        (avg - (-1.0)).abs() < 0.01,
        "COALESCE(AVG(val), -1) should return -1"
    );

    // Test with matching rows - should return actual sum
    let result3 = db
        .query(
            "SELECT COALESCE(SUM(val), 0) as total FROM t WHERE id = 1",
            (),
        )
        .expect("Query should work");

    let rows3: Vec<_> = result3.collect();
    let row3 = rows3[0].as_ref().unwrap();
    let total3: i64 = row3.get(0).unwrap();
    assert_eq!(
        total3, 10,
        "COALESCE should return actual sum when rows exist"
    );
}
