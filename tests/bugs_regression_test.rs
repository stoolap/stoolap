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

// Integration tests for all bugs tracked in BUGS.md
// These tests ensure that fixed bugs don't regress

use stoolap::Database;

fn setup_db(name: &str) -> Database {
    Database::open(&format!("memory://{}", name)).expect("Failed to create database")
}

// =============================================================================
// Bug #1: HAVING clause with aggregates fails
// Description: HAVING SUM(price) > 100 gives "Unknown function: SUM"
// =============================================================================
#[test]
fn test_bug_01_having_with_aggregates() {
    let db = setup_db("bug01");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, category TEXT, price FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO orders VALUES (1, 'A', 50.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (2, 'A', 75.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (3, 'B', 30.0)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT category, SUM(price) as total FROM orders GROUP BY category HAVING SUM(price) > 100",
            (),
        )
        .expect("Query should succeed - HAVING with aggregate");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "Only category A has sum > 100");

    let row = rows[0].as_ref().expect("Row should exist");
    let category: String = row.get(0).unwrap();
    assert_eq!(category, "A");
}

// =============================================================================
// Bug #2: JOIN ON with AND ignores non-equality conditions
// Description: When using AND in JOIN ON clause, non-equality conditions are ignored
// =============================================================================
#[test]
fn test_bug_02_join_on_with_and_conditions() {
    let db = setup_db("bug02");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 'Apple', 'Fruit')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, 'Banana', 'Fruit')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, 'Carrot', 'Veg')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (4, 'Broccoli', 'Veg')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (5, 'Orange', 'Fruit')", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT a.id, a.name, b.id, b.name FROM t a INNER JOIN t b ON a.id < b.id AND a.category = b.category",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();

    // Fruit pairs where a.id < b.id: (1,2), (1,5), (2,5) = 3 pairs
    // Veg pairs where a.id < b.id: (3,4) = 1 pair
    // Total = 4 pairs
    assert_eq!(
        rows.len(),
        4,
        "Should have 4 pairs matching both conditions"
    );

    // Verify a.id < b.id for all rows
    for row in &rows {
        let row = row.as_ref().expect("Row should exist");
        let a_id: i64 = row.get(0).unwrap();
        let b_id: i64 = row.get(2).unwrap();
        assert!(a_id < b_id, "a.id ({}) should be < b.id ({})", a_id, b_id);
    }
}

// =============================================================================
// Bug #3: UPDATE with scalar subquery doesn't apply value
// Description: UPDATE SET column = (SELECT ...) doesn't update the value
// =============================================================================
#[test]
fn test_bug_03_update_with_scalar_subquery() {
    let db = setup_db("bug03");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, 30)", ())
        .expect("Insert failed");

    db.execute(
        "UPDATE t SET val = (SELECT MAX(val) FROM t) WHERE id = 1",
        (),
    )
    .expect("Update should succeed");

    let result = db
        .query("SELECT val FROM t WHERE id = 1", ())
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().expect("Row should exist");
    let val: i64 = row.get(0).unwrap();
    assert_eq!(val, 30, "id=1 should have val=30 (MAX of all vals)");
}

// =============================================================================
// Bug #4: Multiple column aliases ignored
// Description: When aliasing multiple columns with AS, aliases are ignored
// =============================================================================
#[test]
fn test_bug_04_multiple_column_aliases() {
    let db = setup_db("bug04");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT id AS a, val AS b FROM t", ())
        .expect("Query should succeed");

    // Check that column names are the aliases
    let column_names = result.columns();
    assert_eq!(
        column_names[0], "a",
        "First column should be aliased as 'a'"
    );
    assert_eq!(
        column_names[1], "b",
        "Second column should be aliased as 'b'"
    );
}

// =============================================================================
// Bug #5: Aggregate window functions not supported
// Description: SUM, COUNT, AVG as window functions give "Unknown window function" error
// =============================================================================
#[test]
fn test_bug_05_aggregate_window_functions() {
    let db = setup_db("bug05");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 'A', 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, 'A', 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, 'B', 30)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT id, val, SUM(val) OVER (PARTITION BY category ORDER BY id) as running_sum FROM t",
            (),
        )
        .expect("Query should succeed - SUM as window function");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "Should have 3 rows");

    // Check running sums
    let mut found_running_sums = Vec::new();
    for row in &rows {
        let row = row.as_ref().expect("Row should exist");
        let running_sum: i64 = row.get(2).unwrap();
        found_running_sums.push(running_sum);
    }

    // Category A: 10, then 10+20=30
    // Category B: 30
    assert!(
        found_running_sums.contains(&10),
        "Should have running sum 10"
    );
    assert!(
        found_running_sums.contains(&30),
        "Should have running sum 30 (either A total or B)"
    );
}

// =============================================================================
// Bug #6: ORDER BY with arithmetic expression doesn't evaluate
// Description: ORDER BY with expressions like `amount * -1` doesn't sort correctly
// =============================================================================
#[test]
fn test_bug_06_order_by_arithmetic_expression() {
    let db = setup_db("bug06");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, amount FLOAT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 1.50)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, 0.75)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, 0.50)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (4, 1.25)", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT id, amount FROM t ORDER BY amount * -1", ())
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();

    // ORDER BY amount * -1 should give descending order: 1.50, 1.25, 0.75, 0.50
    let expected_order = [1, 4, 2, 3];
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_ref().expect("Row should exist");
        let id: i64 = row.get(0).unwrap();
        assert_eq!(
            id, expected_order[i],
            "Row {} should have id {}",
            i, expected_order[i]
        );
    }
}

// =============================================================================
// Bug #7: ORDER BY aggregate expression doesn't sort correctly
// Description: ORDER BY SUM(x) DESC doesn't sort properly
// =============================================================================
#[test]
fn test_bug_07_order_by_aggregate_expression() {
    let db = setup_db("bug07");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO orders VALUES (1, 'Alice', 100.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (2, 'Bob', 200.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (3, 'Alice', 150.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (4, 'Bob', 100.0)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT customer, SUM(amount) as total FROM orders GROUP BY customer ORDER BY SUM(amount) DESC",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);

    // Bob has 300, Alice has 250, so Bob should be first
    let first_row = rows[0].as_ref().expect("Row should exist");
    let first_customer: String = first_row.get(0).unwrap();
    assert_eq!(
        first_customer, "Bob",
        "Bob (300) should be first when ORDER BY SUM DESC"
    );
}

// =============================================================================
// Bug #8: NULLS FIRST returns wrong result
// Description: ORDER BY col NULLS FIRST returns unexpected output
// =============================================================================
#[test]
fn test_bug_08_nulls_first() {
    let db = setup_db("bug08");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, NULL)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, 5)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, NULL)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (4, 10)", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT id, val FROM t ORDER BY val NULLS FIRST", ())
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 4, "Should have 4 rows");

    // First two rows should have NULL values (ids 1 and 3)
    let first_row = rows[0].as_ref().expect("Row should exist");
    let first_val: Option<i64> = first_row.get(1).ok();
    assert!(
        first_val.is_none(),
        "First row should have NULL with NULLS FIRST"
    );

    let second_row = rows[1].as_ref().expect("Row should exist");
    let second_val: Option<i64> = second_row.get(1).ok();
    assert!(
        second_val.is_none(),
        "Second row should have NULL with NULLS FIRST"
    );
}

// =============================================================================
// Bug #9: CTE JOIN with expression in ON clause returns empty
// Description: Joining CTEs with expressions fails (works with regular tables)
// =============================================================================
#[test]
fn test_bug_09_cte_join_with_expression() {
    let db = setup_db("bug09");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t1 VALUES (1, 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t1 VALUES (2, 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t2 VALUES (2, 100)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t2 VALUES (3, 200)", ())
        .expect("Insert failed");

    // First verify with regular tables
    let result = db
        .query("SELECT * FROM t1 INNER JOIN t2 ON t1.id + 1 = t2.id", ())
        .expect("Regular table join should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Regular table join should return 2 rows");

    // Now test with CTEs
    let result = db
        .query(
            "WITH cte1 AS (SELECT * FROM t1), cte2 AS (SELECT * FROM t2)
             SELECT * FROM cte1 INNER JOIN cte2 ON cte1.id + 1 = cte2.id",
            (),
        )
        .expect("CTE join should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        2,
        "CTE join with expression should return same as regular join"
    );
}

// =============================================================================
// Bug #10: GROUP BY with column alias returns NULL
// Description: Using alias in GROUP BY causes NULL values in output
// =============================================================================
#[test]
fn test_bug_10_group_by_with_column_alias() {
    let db = setup_db("bug10");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, product TEXT, qty INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO orders VALUES (1, 'Apple', 5)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (2, 'Banana', 3)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (3, 'Apple', 2)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT product, qty * 10 as qty_10 FROM orders GROUP BY product, qty * 10",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();

    // Check that no qty_10 values are NULL
    for row in &rows {
        let row = row.as_ref().expect("Row should exist");
        let qty_10: i64 = row.get(1).expect("qty_10 should not be NULL");
        assert!(qty_10 > 0, "qty_10 should be a positive value");
    }
}

// =============================================================================
// Bug #11: CURRENT_DATE/TIME/TIMESTAMP return NULL
// Description: SQL standard date/time keywords return NULL
// =============================================================================
#[test]
fn test_bug_11_current_date_time_timestamp() {
    let db = setup_db("bug11");

    // Test CURRENT_DATE
    let result = db
        .query("SELECT CURRENT_DATE", ())
        .expect("CURRENT_DATE query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].as_ref().expect("Row should exist");
    let date: String = row.get(0).expect("CURRENT_DATE should not be NULL");
    assert!(!date.is_empty(), "CURRENT_DATE should return a value");

    // Test CURRENT_TIMESTAMP
    let result = db
        .query("SELECT CURRENT_TIMESTAMP", ())
        .expect("CURRENT_TIMESTAMP query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].as_ref().expect("Row should exist");
    let ts: String = row.get(0).expect("CURRENT_TIMESTAMP should not be NULL");
    assert!(!ts.is_empty(), "CURRENT_TIMESTAMP should return a value");
}

// =============================================================================
// Bug #12: Three-table JOIN incorrect results
// Description: When joining 3 tables, the third join condition is not properly evaluated
// =============================================================================
#[test]
fn test_bug_12_three_table_join() {
    let db = setup_db("bug12");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, t1_id INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "CREATE TABLE t3 (id INTEGER PRIMARY KEY, t2_id INTEGER, descr TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t1 VALUES (1, 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t1 VALUES (2, 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t1 VALUES (3, 30)", ())
        .expect("Insert failed");

    db.execute("INSERT INTO t2 VALUES (1, 1, 'A')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t2 VALUES (2, 1, 'B')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t2 VALUES (3, 2, 'C')", ())
        .expect("Insert failed");

    db.execute("INSERT INTO t3 VALUES (1, 1, 'Desc1')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t3 VALUES (2, 2, 'Desc2')", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT t1.id, t2.id as t2_id, t2.name, t3.t2_id, t3.descr
             FROM t1
             INNER JOIN t2 ON t1.id = t2.t1_id
             INNER JOIN t3 ON t2.id = t3.t2_id",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();

    // Expected matches:
    // t1.id=1 -> t2.id=1,name=A -> t3.t2_id=1,descr=Desc1
    // t1.id=1 -> t2.id=2,name=B -> t3.t2_id=2,descr=Desc2
    // t1.id=2 -> t2.id=3,name=C -> no match in t3
    assert_eq!(rows.len(), 2, "Should have 2 matching rows");

    for row in &rows {
        let row = row.as_ref().expect("Row should exist");
        let t2_id: i64 = row.get(1).unwrap();
        let t3_t2_id: i64 = row.get(3).unwrap();
        assert_eq!(
            t2_id, t3_t2_id,
            "t2.id ({}) should equal t3.t2_id ({})",
            t2_id, t3_t2_id
        );
    }
}

// =============================================================================
// Bug #13: Derived table (subquery in FROM) ignores outer WHERE
// Description: WHERE clause on derived table is ignored
// =============================================================================
#[test]
fn test_bug_13_derived_table_where() {
    let db = setup_db("bug13");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, 30)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT * FROM (SELECT id, val * 2 as doubled FROM t) as sub WHERE doubled > 30",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();

    // doubled values: 20, 40, 60 - only 40 and 60 are > 30
    assert_eq!(
        rows.len(),
        2,
        "Should have 2 rows where doubled > 30 (40 and 60)"
    );

    for row in &rows {
        let row = row.as_ref().expect("Row should exist");
        let doubled: i64 = row.get(1).unwrap();
        assert!(doubled > 30, "All doubled values should be > 30");
    }
}

// =============================================================================
// Bug #14: HAVING with aggregate not in SELECT fails
// Description: HAVING clause fails when aggregate is not also in SELECT list
// =============================================================================
#[test]
fn test_bug_14_having_aggregate_not_in_select() {
    let db = setup_db("bug14");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER, grp TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10, 'A')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, 20, 'A')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, 30, 'B')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (4, 40, 'B')", ())
        .expect("Insert failed");

    // This should work - aggregate only in HAVING, not in SELECT
    let result = db
        .query("SELECT grp FROM t GROUP BY grp HAVING SUM(val) > 50", ())
        .expect("Query should succeed - HAVING aggregate not in SELECT");

    let rows: Vec<_> = result.collect();

    // Group A has sum 30, Group B has sum 70 - only B qualifies
    assert_eq!(rows.len(), 1, "Only group B has SUM(val) > 50");

    let row = rows[0].as_ref().expect("Row should exist");
    let grp: String = row.get(0).unwrap();
    assert_eq!(grp, "B");
}

// =============================================================================
// Bug #15: DENSE_RANK returns wrong values
// Description: DENSE_RANK behaves like RANK instead of dense ranking
// =============================================================================
#[test]
fn test_bug_15_dense_rank() {
    let db = setup_db("bug15");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (4, 30)", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT id, val, DENSE_RANK() OVER (ORDER BY val) as drank FROM t ORDER BY id",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 4);

    // Expected dense ranks: 1, 2, 2, 3 (not 1, 2, 2, 4 like RANK)
    let expected_ranks = [1i64, 2, 2, 3];
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_ref().expect("Row should exist");
        let drank: i64 = row.get(2).unwrap();
        assert_eq!(
            drank, expected_ranks[i],
            "Row {} should have DENSE_RANK {}",
            i, expected_ranks[i]
        );
    }
}

// =============================================================================
// Bug #16: LEFT/RIGHT string functions conflict with JOIN keywords
// Description: LEFT() and RIGHT() functions fail to parse due to keyword conflict
// =============================================================================
#[test]
fn test_bug_16_left_right_string_functions() {
    let db = setup_db("bug16");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 'hello')", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT LEFT(name, 3), RIGHT(name, 3) FROM t", ())
        .expect("LEFT/RIGHT functions should parse correctly");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().expect("Row should exist");
    let left_val: String = row.get(0).unwrap();
    let right_val: String = row.get(1).unwrap();

    assert_eq!(left_val, "hel", "LEFT('hello', 3) should be 'hel'");
    assert_eq!(right_val, "llo", "RIGHT('hello', 3) should be 'llo'");
}

// =============================================================================
// Bug #17: CHAR function conflicts with CHAR data type
// Description: CHAR() function fails to parse due to data type keyword conflict
// =============================================================================
#[test]
fn test_bug_17_char_function() {
    let db = setup_db("bug17");

    let result = db
        .query("SELECT CHAR(65)", ())
        .expect("CHAR function should parse correctly");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().expect("Row should exist");
    let char_val: String = row.get(0).unwrap();

    assert_eq!(char_val, "A", "CHAR(65) should be 'A'");
}

// =============================================================================
// Bug #18: Qualified star (table.*) not supported
// Description: SELECT t.* syntax is not recognized
// =============================================================================
#[test]
fn test_bug_18_qualified_star() {
    let db = setup_db("bug18");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert failed");

    // Simple qualified star
    let result = db
        .query("SELECT t.* FROM t", ())
        .expect("Qualified star should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().expect("Row should exist");
    let id: i64 = row.get(0).unwrap();
    let val: i64 = row.get(1).unwrap();
    assert_eq!(id, 1);
    assert_eq!(val, 10);

    // Qualified star with join
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t2 VALUES (1, 'Alice')", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT t.*, t2.name FROM t INNER JOIN t2 ON t.id = t2.id",
            (),
        )
        .expect("Qualified star in join should work");

    let column_names = result.columns();
    assert_eq!(
        column_names.len(),
        3,
        "Should have 3 columns: t.id, t.val, t2.name"
    );

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
}

// =============================================================================
// Bug #19: String escape sequences not processed
// Description: SQL standard escaped quotes are shown literally
// =============================================================================
#[test]
fn test_bug_19_string_escape_sequences() {
    let db = setup_db("bug19");

    let result = db
        .query("SELECT 'Hello ''World''' as escaped", ())
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().expect("Row should exist");
    let escaped: String = row.get(0).unwrap();

    // Two single quotes ('') should become one single quote (')
    assert_eq!(
        escaped, "Hello 'World'",
        "Escaped quotes should be processed: '' -> '"
    );
}

// =============================================================================
// Additional edge case tests
// =============================================================================

#[test]
fn test_qualified_star_with_alias() {
    let db = setup_db("qualified_star_alias");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO employees VALUES (1, 'Alice', 'Eng')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO employees VALUES (2, 'Bob', 'Sales')", ())
        .expect("Insert failed");

    // Using table alias with qualified star
    let result = db
        .query("SELECT e.* FROM employees e WHERE e.dept = 'Eng'", ())
        .expect("Qualified star with alias should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().expect("Row should exist");
    let name: String = row.get(1).unwrap();
    assert_eq!(name, "Alice");
}

#[test]
fn test_multiple_qualified_stars_in_join() {
    let db = setup_db("multi_qualified_star");

    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, a_val TEXT)", ())
        .expect("Failed to create table");
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, b_val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO a VALUES (1, 'A1')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO b VALUES (1, 'B1')", ())
        .expect("Insert failed");

    let result = db
        .query("SELECT a.*, b.* FROM a INNER JOIN b ON a.id = b.id", ())
        .expect("Multiple qualified stars should work");

    let column_names = result.columns();
    assert_eq!(column_names.len(), 4, "Should have 4 columns total");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
}

#[test]
fn test_complex_string_escapes() {
    let db = setup_db("complex_escapes");

    // Multiple escapes in one string
    let result = db
        .query("SELECT 'It''s a ''test'' string' as complex", ())
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().expect("Row should exist");
    let complex: String = row.get(0).unwrap();

    assert_eq!(complex, "It's a 'test' string");

    // Empty string between quotes
    let result = db
        .query("SELECT '''' as just_quote", ())
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().expect("Row should exist");
    let just_quote: String = row.get(0).unwrap();

    assert_eq!(just_quote, "'", "Four quotes should become one quote");
}

// =============================================================================
// Bug #20: CTE referenced in subquery fails with "table not found"
// Description: When a CTE is referenced inside a subquery, it fails
// =============================================================================
#[test]
fn test_bug_20_cte_in_subquery() {
    let db = setup_db("bug20");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 95000.0)",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 85000.0)",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO employees VALUES (3, 'Charlie', 'Sales', 75000.0)",
        (),
    )
    .expect("Insert failed");

    // CTE referenced in IN subquery - was broken
    let result = db
        .query(
            "WITH high_depts AS (SELECT department FROM employees WHERE salary > 90000)
             SELECT name FROM employees WHERE department IN (SELECT department FROM high_depts)
             ORDER BY name",
            (),
        )
        .expect("Query should succeed - CTE in subquery");

    let names: Vec<String> = result
        .into_iter()
        .filter_map(|r| r.ok())
        .map(|row| row.get::<String>(0).unwrap())
        .collect();

    // Engineering has salary > 90000 (Alice=95000), so all Engineering employees
    assert_eq!(names, vec!["Alice", "Bob"]);
}

// =============================================================================
// Bug #21: GROUP BY with extra columns drops aggregates from result
// Description: When GROUP BY includes columns not in SELECT, aggregate columns disappear
// =============================================================================
#[test]
fn test_bug_21_group_by_extra_columns() {
    let db = setup_db("bug21");

    db.execute("CREATE TABLE test (id INTEGER, name TEXT, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO test VALUES (1, 'A', 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO test VALUES (1, 'A', 20)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO test VALUES (2, 'B', 30)", ())
        .expect("Insert failed");

    // GROUP BY has extra column (id) not in SELECT - was dropping SUM column
    let result = db
        .query(
            "SELECT name, SUM(val) as total FROM test GROUP BY id, name ORDER BY name",
            (),
        )
        .expect("Query should succeed - GROUP BY with extra columns");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have 2 groups");

    let row0 = rows[0].as_ref().expect("Row should exist");
    let name: String = row0.get(0).unwrap();
    let total: f64 = row0.get(1).unwrap();
    assert_eq!(name, "A");
    assert_eq!(total, 30.0); // 10 + 20

    let row1 = rows[1].as_ref().expect("Row should exist");
    let name: String = row1.get(0).unwrap();
    let total: f64 = row1.get(1).unwrap();
    assert_eq!(name, "B");
    assert_eq!(total, 30.0);
}

// =============================================================================
// Bug #22: Arithmetic with window functions not supported
// Description: val - LAG(val, 1) OVER (...) gives "Window expression in evaluator not yet implemented"
// =============================================================================
#[test]
fn test_bug_22_window_function_arithmetic() {
    let db = setup_db("bug22");

    db.execute(
        "CREATE TABLE test_wf (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_wf VALUES (1, 10), (2, 20), (3, 35), (4, 50)",
        (),
    )
    .expect("Insert failed");

    // Simple arithmetic with window function
    let result = db
        .query(
            "SELECT id, val, val - LAG(val, 1) OVER (ORDER BY id) AS diff FROM test_wf",
            (),
        )
        .expect("Query should succeed - arithmetic with window function");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 4);

    // Row 1: diff = 10 - NULL = NULL
    let row0 = rows[0].as_ref().expect("Row should exist");
    let diff0: Option<i64> = row0.get(2).ok();
    assert!(diff0.is_none(), "First row diff should be NULL");

    // Row 2: diff = 20 - 10 = 10
    let row1 = rows[1].as_ref().expect("Row should exist");
    let diff1: i64 = row1.get(2).unwrap();
    assert_eq!(diff1, 10);

    // Row 3: diff = 35 - 20 = 15
    let row2 = rows[2].as_ref().expect("Row should exist");
    let diff2: i64 = row2.get(2).unwrap();
    assert_eq!(diff2, 15);

    // Row 4: diff = 50 - 35 = 15
    let row3 = rows[3].as_ref().expect("Row should exist");
    let diff3: i64 = row3.get(2).unwrap();
    assert_eq!(diff3, 15);
}

// =============================================================================
// Bug #22b: Multiple expressions with window functions
// Description: When multiple expressions contain window functions, all should work
// =============================================================================
#[test]
fn test_bug_22_multiple_window_expressions() {
    let db = setup_db("bug22b");

    db.execute(
        "CREATE TABLE test_wf2 (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO test_wf2 VALUES (1, 10), (2, 20), (3, 35), (4, 50)",
        (),
    )
    .expect("Insert failed");

    // Multiple expressions with different window functions
    let result = db
        .query(
            "SELECT id,
                    val + LAG(val, 1) OVER (ORDER BY id) AS sum_with_prev,
                    val * 2 - LEAD(val, 1) OVER (ORDER BY id) AS double_minus_next
             FROM test_wf2",
            (),
        )
        .expect("Query should succeed - multiple window function expressions");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 4);

    // Row 2: sum_with_prev = 20 + 10 = 30, double_minus_next = 40 - 35 = 5
    let row1 = rows[1].as_ref().expect("Row should exist");
    let sum_with_prev: i64 = row1.get(1).unwrap();
    let double_minus_next: i64 = row1.get(2).unwrap();
    assert_eq!(sum_with_prev, 30);
    assert_eq!(double_minus_next, 5);

    // Row 3: sum_with_prev = 35 + 20 = 55, double_minus_next = 70 - 50 = 20
    let row2 = rows[2].as_ref().expect("Row should exist");
    let sum_with_prev: i64 = row2.get(1).unwrap();
    let double_minus_next: i64 = row2.get(2).unwrap();
    assert_eq!(sum_with_prev, 55);
    assert_eq!(double_minus_next, 20);
}

// =============================================================================
// Bug #23: EXTRACT function not supported
// Description: SQL standard EXTRACT(field FROM date) syntax was not supported
// =============================================================================
#[test]
fn test_bug_23_extract_function() {
    let db = setup_db("bug23");

    // Test basic EXTRACT with date literal
    let result = db
        .query("SELECT EXTRACT(YEAR FROM DATE '2024-01-15')", ())
        .expect("EXTRACT YEAR should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let year: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(year, 2024);

    let result = db
        .query("SELECT EXTRACT(MONTH FROM DATE '2024-01-15')", ())
        .expect("EXTRACT MONTH should work");
    let rows: Vec<_> = result.collect();
    let month: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(month, 1);

    let result = db
        .query("SELECT EXTRACT(DAY FROM DATE '2024-01-15')", ())
        .expect("EXTRACT DAY should work");
    let rows: Vec<_> = result.collect();
    let day: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(day, 15);

    // Test EXTRACT with timestamp
    let result = db
        .query(
            "SELECT EXTRACT(HOUR FROM TIMESTAMP '2024-01-15 14:30:45')",
            (),
        )
        .expect("EXTRACT HOUR should work");
    let rows: Vec<_> = result.collect();
    let hour: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(hour, 14);

    // Test EXTRACT with column
    db.execute(
        "CREATE TABLE dates (id INTEGER PRIMARY KEY, dt TIMESTAMP)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO dates VALUES (1, '2024-06-20 09:15:30')", ())
        .expect("Insert failed");

    let result = db
        .query(
            "SELECT EXTRACT(YEAR FROM dt), EXTRACT(MONTH FROM dt) FROM dates WHERE id = 1",
            (),
        )
        .expect("EXTRACT from column should work");
    let rows: Vec<_> = result.collect();
    let year: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    let month: i64 = rows[0].as_ref().unwrap().get(1).unwrap();
    assert_eq!(year, 2024);
    assert_eq!(month, 6);
}

// =============================================================================
// Bug #24: JSON operators not supported
// Description: SQL JSON operators -> and ->> were not supported
// =============================================================================
#[test]
fn test_bug_24_json_operators() {
    let db = setup_db("bug24");

    // Test basic -> operator (returns JSON)
    let result = db
        .query(r#"SELECT '{"name": "test", "value": 42}' -> 'name'"#, ())
        .expect("JSON -> operator should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let json_result: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(json_result, "\"test\""); // JSON string includes quotes

    // Test basic ->> operator (returns TEXT)
    let result = db
        .query(r#"SELECT '{"name": "test", "value": 42}' ->> 'name'"#, ())
        .expect("JSON ->> operator should work");
    let rows: Vec<_> = result.collect();
    let text_result: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(text_result, "test"); // TEXT extracts without quotes

    // Test numeric value extraction
    let result = db
        .query(r#"SELECT '{"name": "test", "value": 42}' ->> 'value'"#, ())
        .expect("JSON ->> with number should work");
    let rows: Vec<_> = result.collect();
    let value: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(value, "42");

    // Test array index access with ->
    let result = db
        .query(r#"SELECT '[1, 2, 3]' -> 0"#, ())
        .expect("JSON array access should work");
    let rows: Vec<_> = result.collect();
    let first: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(first, "1");

    // Test array index access with ->>
    let result = db
        .query(r#"SELECT '["a", "b", "c"]' ->> 1"#, ())
        .expect("JSON array ->> should work");
    let rows: Vec<_> = result.collect();
    let second: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(second, "b");

    // Test chained JSON access
    let result = db
        .query(
            r#"SELECT '{"user": {"name": "Alice", "age": 30}}' -> 'user' ->> 'name'"#,
            (),
        )
        .expect("Chained JSON access should work");
    let rows: Vec<_> = result.collect();
    let name: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(name, "Alice");

    // Test with table and JSON column
    db.execute(
        "CREATE TABLE json_data (id INTEGER PRIMARY KEY, data JSON)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        r#"INSERT INTO json_data VALUES (1, '{"product": "Widget", "price": 19.99}')"#,
        (),
    )
    .expect("Insert failed");
    db.execute(
        r#"INSERT INTO json_data VALUES (2, '{"product": "Gadget", "price": 29.99}')"#,
        (),
    )
    .expect("Insert failed");

    let result = db
        .query(
            "SELECT id, data ->> 'product' AS product FROM json_data ORDER BY id",
            (),
        )
        .expect("JSON access from column should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);
    let product1: String = rows[0].as_ref().unwrap().get(1).unwrap();
    let product2: String = rows[1].as_ref().unwrap().get(1).unwrap();
    assert_eq!(product1, "Widget");
    assert_eq!(product2, "Gadget");

    // Test NULL handling - missing key returns NULL
    let result = db
        .query(r#"SELECT '{"a": 1}' -> 'missing'"#, ())
        .expect("Missing key should return NULL");
    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let null_val: Option<String> = row.get(0).unwrap();
    assert!(null_val.is_none(), "Missing key should return NULL");
}

// =============================================================================
// Bug #25: Multi-column IN not supported
// Description: IN clause with multiple columns (tuple comparison) was not supported
// =============================================================================
#[test]
fn test_bug_25_multi_column_in() {
    let db = setup_db("bug25");

    db.execute(
        "CREATE TABLE multi_in_test (a INTEGER, b INTEGER, c TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO multi_in_test VALUES (1, 10, 'x'), (2, 20, 'y'), (3, 30, 'x'), (1, 10, 'z')",
        (),
    )
    .expect("Insert failed");

    // Test basic multi-column IN
    let result = db
        .query(
            "SELECT a, b, c FROM multi_in_test WHERE (a, b) IN ((1, 10), (3, 30)) ORDER BY a, c",
            (),
        )
        .expect("Multi-column IN should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        3,
        "Should match 3 rows (two with (1,10), one with (3,30))"
    );

    // Verify the first result (1, 10, 'x')
    let row0 = rows[0].as_ref().unwrap();
    assert_eq!(row0.get::<i64>(0).unwrap(), 1);
    assert_eq!(row0.get::<i64>(1).unwrap(), 10);

    // Test NOT IN with multiple columns
    let result = db
        .query(
            "SELECT a, b FROM multi_in_test WHERE (a, b) NOT IN ((1, 10), (3, 30))",
            (),
        )
        .expect("Multi-column NOT IN should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "Only (2, 20) should not match");
    let row = rows[0].as_ref().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(1).unwrap(), 20);

    // Test with three columns
    let result = db
        .query(
            "SELECT a, b, c FROM multi_in_test WHERE (a, b, c) IN ((1, 10, 'x'), (2, 20, 'y'))",
            (),
        )
        .expect("Three-column IN should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should match exactly 2 rows");

    // Verify results
    let texts: Vec<String> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get::<String>(2).unwrap())
        .collect();
    assert!(texts.contains(&"x".to_string()));
    assert!(texts.contains(&"y".to_string()));
}

// =============================================================================
// Bug #26: UNION with AS keyword fails to parse
// Description: When using column aliases with AS in UNION queries, parser failed
// =============================================================================
#[test]
fn test_bug_26_union_with_as_keyword() {
    let db = setup_db("bug26");

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 95000)",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 85000)",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO employees VALUES (3, 'Charlie', 'Sales', 75000)",
        (),
    )
    .expect("Insert failed");

    // UNION with AS keyword in column aliases
    let result = db
        .query(
            "SELECT name, 'High' as level FROM employees WHERE salary > 90000
             UNION
             SELECT name, 'Low' as level FROM employees WHERE salary < 80000",
            (),
        )
        .expect("UNION with AS keyword should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        2,
        "Should have 2 rows (Alice=High, Charlie=Low)"
    );

    // Verify data
    let mut found_alice = false;
    let mut found_charlie = false;
    for row in &rows {
        let row = row.as_ref().unwrap();
        let name: String = row.get(0).unwrap();
        let level: String = row.get(1).unwrap();
        if name == "Alice" {
            assert_eq!(level, "High");
            found_alice = true;
        }
        if name == "Charlie" {
            assert_eq!(level, "Low");
            found_charlie = true;
        }
    }
    assert!(found_alice && found_charlie);
}

// =============================================================================
// Bug #27: Aggregate function inside expression fails
// Description: Aggregates inside functions or expressions failed with "Unknown function"
// =============================================================================
#[test]
fn test_bug_27_aggregate_in_expression() {
    let db = setup_db("bug27");

    db.execute("CREATE TABLE t27 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t27 VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    // COALESCE with aggregate
    let result = db
        .query("SELECT COALESCE(SUM(val), 0) FROM t27", ())
        .expect("COALESCE(SUM(val), 0) should work");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let sum: i64 = row.get(0).unwrap();
    assert_eq!(sum, 60, "COALESCE(SUM(val), 0) should be 60");

    // Aggregate + literal
    let result = db
        .query("SELECT SUM(val) + 10 as sum_plus FROM t27", ())
        .expect("SUM(val) + 10 should work");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let sum_plus: i64 = row.get(0).unwrap();
    assert_eq!(sum_plus, 70, "SUM(val) + 10 should be 70");

    // Function wrapping aggregate
    let result = db
        .query("SELECT ABS(SUM(val)) FROM t27", ())
        .expect("ABS(SUM(val)) should work");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let abs_sum: i64 = row.get(0).unwrap();
    assert_eq!(abs_sum, 60, "ABS(SUM(val)) should be 60");
}

// =============================================================================
// Bug #28: Scalar subquery breaks outer column resolution
// Description: When a scalar subquery is present in SELECT, other columns failed
// =============================================================================
#[test]
fn test_bug_28_scalar_subquery_column_resolution() {
    let db = setup_db("bug28");

    db.execute(
        "CREATE TABLE mc (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO mc VALUES (1, 10, 100), (2, 20, 200)", ())
        .expect("Insert failed");

    // Scalar subquery with other columns
    let result = db
        .query(
            "SELECT id, (SELECT MAX(b) FROM mc) as max_b FROM mc ORDER BY id",
            (),
        )
        .expect("Scalar subquery with other columns should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);

    // Both rows should have max_b = 200
    for row in &rows {
        let row = row.as_ref().unwrap();
        let max_b: i64 = row.get(1).unwrap();
        assert_eq!(max_b, 200, "Scalar subquery should return MAX(b) = 200");
    }

    // Verify id column still works
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 2]);
}

// =============================================================================
// Bug #29: VIEW with aggregate returns wrong results
// Description: Querying a VIEW with aggregate functions returned incorrect results
// =============================================================================
#[test]
fn test_bug_29_view_with_aggregate() {
    let db = setup_db("bug29");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products VALUES (1, 'Apple', 1.50, 'Fruit')",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO products VALUES (2, 'Banana', 0.75, 'Fruit')",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO products VALUES (3, 'Orange', 2.00, 'Fruit')",
        (),
    )
    .expect("Insert failed");

    // Create view with aggregate
    db.execute(
        "CREATE VIEW fruit_stats AS SELECT COUNT(*) as cnt, AVG(price) as avg_price FROM products WHERE category = 'Fruit'",
        (),
    )
    .expect("Failed to create view");

    // Query the view
    let result = db
        .query("SELECT * FROM fruit_stats", ())
        .expect("VIEW with aggregate should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let row = rows[0].as_ref().unwrap();
    let cnt: i64 = row.get(0).unwrap();
    let avg_price: f64 = row.get(1).unwrap();

    assert_eq!(cnt, 3, "COUNT(*) should be 3");
    // AVG = (1.50 + 0.75 + 2.00) / 3 = 1.4166...
    assert!(
        (avg_price - 1.4167).abs() < 0.01,
        "AVG(price) should be ~1.4167"
    );
}

// =============================================================================
// Bug #30: Aggregation on VIEWs returns NULL
// Description: When querying a VIEW with aggregate functions, results were NULL
// =============================================================================
#[test]
fn test_bug_30_aggregation_on_views() {
    let db = setup_db("bug30");

    db.execute(
        "CREATE TABLE t30 (id INTEGER PRIMARY KEY, cat TEXT, val FLOAT, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t30 VALUES (1, 'A', 100.0, TRUE), (2, 'A', 150.0, TRUE), (3, 'B', 200.0, TRUE)",
        (),
    )
    .expect("Insert failed");

    db.execute(
        "CREATE VIEW v30 AS SELECT id, cat, val FROM t30 WHERE active = TRUE",
        (),
    )
    .expect("Failed to create view");

    // Test aggregation on view
    let result = db
        .query(
            "SELECT cat, SUM(val) as total FROM v30 GROUP BY cat ORDER BY cat",
            (),
        )
        .expect("Aggregation on view should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have 2 category groups");

    let row0 = rows[0].as_ref().unwrap();
    let cat0: String = row0.get(0).unwrap();
    let total0: f64 = row0.get(1).unwrap();
    assert_eq!(cat0, "A");
    assert_eq!(total0, 250.0);

    let row1 = rows[1].as_ref().unwrap();
    let cat1: String = row1.get(0).unwrap();
    let total1: f64 = row1.get(1).unwrap();
    assert_eq!(cat1, "B");
    assert_eq!(total1, 200.0);
}

// =============================================================================
// Bug #31: Window frame ROWS BETWEEN not working correctly
// Description: Window frames were ignoring explicit ROWS BETWEEN specifications
// =============================================================================
#[test]
fn test_bug_31_window_frame_rows_between() {
    let db = setup_db("bug31");

    db.execute("CREATE TABLE t31 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO t31 VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
        (),
    )
    .expect("Insert failed");

    let result = db
        .query(
            "SELECT id, val, SUM(val) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as win_sum FROM t31",
            (),
        )
        .expect("Window frame ROWS BETWEEN should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 5);

    // Expected results:
    // id=1: 10+20 = 30
    // id=2: 10+20+30 = 60
    // id=3: 20+30+40 = 90
    // id=4: 30+40+50 = 120
    // id=5: 40+50 = 90
    let expected_sums = [30, 60, 90, 120, 90];
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_ref().unwrap();
        let win_sum: i64 = row.get(2).unwrap();
        assert_eq!(
            win_sum,
            expected_sums[i],
            "Window sum at row {} should be {}",
            i + 1,
            expected_sums[i]
        );
    }
}

// =============================================================================
// Bug #32: GROUP BY ordinal (position number) returns NULL
// Description: Using column position in GROUP BY caused NULL values
// =============================================================================
#[test]
fn test_bug_32_group_by_ordinal() {
    let db = setup_db("bug32");

    db.execute(
        "CREATE TABLE t32 (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t32 VALUES (1, 10, 100), (2, 10, 200), (3, 20, 150), (4, 20, 250)",
        (),
    )
    .expect("Insert failed");

    // GROUP BY 1 should group by the first column (a)
    let result = db
        .query(
            "SELECT a, SUM(b) as total FROM t32 GROUP BY 1 ORDER BY 1",
            (),
        )
        .expect("GROUP BY ordinal should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have 2 groups (a=10 and a=20)");

    let row0 = rows[0].as_ref().unwrap();
    let a0: i64 = row0.get(0).unwrap();
    let sum0: i64 = row0.get(1).unwrap();
    assert_eq!(a0, 10);
    assert_eq!(sum0, 300); // 100 + 200

    let row1 = rows[1].as_ref().unwrap();
    let a1: i64 = row1.get(0).unwrap();
    let sum1: i64 = row1.get(1).unwrap();
    assert_eq!(a1, 20);
    assert_eq!(sum1, 400); // 150 + 250
}

// =============================================================================
// Bug #33: NATURAL JOIN not working correctly
// Description: NATURAL JOIN was behaving like CROSS JOIN
// =============================================================================
#[test]
fn test_bug_33_natural_join() {
    let db = setup_db("bug33");

    db.execute(
        "CREATE TABLE t33_left (id INTEGER PRIMARY KEY, name TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "CREATE TABLE t33_right (id INTEGER PRIMARY KEY, name TEXT, extra TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t33_left VALUES (1, 'Alice', 100), (2, 'Bob', 200)",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO t33_right VALUES (1, 'Alice', 'X'), (3, 'Charlie', 'Y')",
        (),
    )
    .expect("Insert failed");

    // NATURAL JOIN should match on both id AND name
    let result = db
        .query("SELECT * FROM t33_left NATURAL JOIN t33_right", ())
        .expect("NATURAL JOIN should work");

    let rows: Vec<_> = result.collect();
    // Only (1, 'Alice') should match
    assert_eq!(
        rows.len(),
        1,
        "Only one row should match (id=1, name='Alice')"
    );
}

// =============================================================================
// Bug #34: Implicit CROSS JOIN (comma syntax) not supported
// Description: FROM t1, t2 syntax was not recognized
// =============================================================================
#[test]
fn test_bug_34_implicit_cross_join_comma_syntax() {
    let db = setup_db("bug34");

    db.execute("CREATE TABLE colors (id INTEGER, color TEXT)", ())
        .expect("Failed to create table");
    db.execute("CREATE TABLE sizes (id INTEGER, size TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO colors VALUES (1, 'Red'), (2, 'Blue')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO sizes VALUES (1, 'Small'), (2, 'Large')", ())
        .expect("Insert failed");

    // Comma syntax should work as implicit CROSS JOIN
    let result = db
        .query(
            "SELECT colors.color, sizes.size FROM colors, sizes ORDER BY colors.color, sizes.size",
            (),
        )
        .expect("Implicit CROSS JOIN should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 4, "Should have 4 rows (2x2 cross product)");

    // Test with WHERE clause (effectively an inner join)
    let result = db
        .query("SELECT colors.color, sizes.size FROM colors, sizes WHERE colors.id = sizes.id ORDER BY colors.id", ())
        .expect("Implicit CROSS JOIN with WHERE should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have 2 rows where id matches");
}

// =============================================================================
// Bug #35: DELETE/UPDATE with expression in WHERE fails
// Description: DELETE and UPDATE statements failed with arithmetic expressions in WHERE
// =============================================================================
#[test]
fn test_bug_35_delete_update_with_expression_in_where() {
    let db = setup_db("bug35");

    db.execute(
        "CREATE TABLE t35 (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t35 VALUES (1, 10, 20), (2, 30, 40), (3, 50, 60)",
        (),
    )
    .expect("Insert failed");

    // DELETE with expression in WHERE
    db.execute("DELETE FROM t35 WHERE a + b > 100", ())
        .expect("DELETE with expression should work");

    let result = db.query("SELECT id FROM t35 ORDER BY id", ()).unwrap();
    let rows: Vec<_> = result.collect();
    // Rows 2 (70>100? no), 3 (110>100? yes) - row 3 deleted
    // Actually: row 1 (30>100? no), row 2 (70>100? no), row 3 (110>100? yes)
    assert_eq!(rows.len(), 2, "Row with a+b > 100 should be deleted");

    // UPDATE with expression in WHERE
    db.execute("UPDATE t35 SET a = 999 WHERE a + b < 50", ())
        .expect("UPDATE with expression should work");

    let result = db.query("SELECT a FROM t35 WHERE id = 1", ()).unwrap();
    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let a: i64 = row.get(0).unwrap();
    assert_eq!(a, 999, "Row with a+b < 50 (id=1, 30<50) should be updated");
}

// =============================================================================
// Bug #36: Aggregate functions with expressions inside return wrong result
// Description: SUM(val * 2) was returning COUNT instead of the actual sum
// =============================================================================
#[test]
fn test_bug_36_aggregate_with_expression_inside() {
    let db = setup_db("bug36");

    db.execute("CREATE TABLE t36 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t36 VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    // SUM with expression
    let result = db
        .query("SELECT SUM(val * 2) as doubled_sum FROM t36", ())
        .expect("SUM with expression should work");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let sum: i64 = row.get(0).unwrap();
    assert_eq!(sum, 120, "SUM(val * 2) should be 120 (10*2 + 20*2 + 30*2)");

    // AVG with expression
    let result = db
        .query("SELECT AVG(val + 10) as avg_plus_10 FROM t36", ())
        .expect("AVG with expression should work");

    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let avg: f64 = row.get(0).unwrap();
    assert!(
        (avg - 30.0).abs() < 0.01,
        "AVG(val + 10) should be 30 ((20+30+40)/3)"
    );
}

// =============================================================================
// Bug #37: Recursive CTE with table JOIN fails
// Description: Recursive CTE worked for simple cases but failed when joining tables
// =============================================================================
#[test]
fn test_bug_37_recursive_cte_with_table_join() {
    let db = setup_db("bug37");

    db.execute(
        "CREATE TABLE tree (id INTEGER PRIMARY KEY, parent_id INTEGER, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO tree VALUES (1, NULL, 'Root'), (2, 1, 'Child1'), (3, 1, 'Child2'), (4, 2, 'GrandChild')",
        (),
    )
    .expect("Insert failed");

    // Recursive CTE that joins with the tree table
    let result = db
        .query(
            "WITH RECURSIVE h AS (
                SELECT id, name FROM tree WHERE parent_id IS NULL
                UNION ALL
                SELECT t.id, t.name FROM tree t JOIN h ON t.parent_id = h.id
            )
            SELECT * FROM h ORDER BY id",
            (),
        )
        .expect("Recursive CTE with table JOIN should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 4, "Should traverse entire tree (4 nodes)");

    // Verify we got all nodes
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 2, 3, 4]);
}

// =============================================================================
// Bug #38: ILIKE operator parsed incorrectly
// Description: ILIKE was not properly recognized as a case-insensitive pattern match
// =============================================================================
#[test]
fn test_bug_38_ilike_operator() {
    let db = setup_db("bug38");

    db.execute("CREATE TABLE t38 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO t38 VALUES (1, 'test'), (2, 'TEST'), (3, 'Test'), (4, 'Other')",
        (),
    )
    .expect("Insert failed");

    // ILIKE should be case-insensitive
    let result = db
        .query("SELECT id FROM t38 WHERE val ILIKE 'test' ORDER BY id", ())
        .expect("ILIKE should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        3,
        "ILIKE 'test' should match 3 rows (case-insensitive)"
    );

    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 2, 3]);

    // ILIKE with wildcard
    let result = db
        .query("SELECT id FROM t38 WHERE val ILIKE '%EST' ORDER BY id", ())
        .expect("ILIKE with wildcard should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "ILIKE '%EST' should match 3 rows");

    // NOT ILIKE
    let result = db
        .query(
            "SELECT id FROM t38 WHERE val NOT ILIKE 'test' ORDER BY id",
            (),
        )
        .expect("NOT ILIKE should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "NOT ILIKE 'test' should match 1 row");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 4);
}

// =============================================================================
// Bug #40: HAVING with expression GROUP BY shows NULL for grouped column
// Description: When GROUP BY uses an expression and HAVING is present, grouped column was NULL
// =============================================================================
#[test]
fn test_bug_40_having_with_expression_group_by() {
    let db = setup_db("bug40");

    db.execute(
        "CREATE TABLE t40 (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t40 VALUES (1, 'Alice', 10), (2, 'Alice', 20), (3, 'Bob', 15), (4, 'Bob', 25), (5, 'Charlie', 30)",
        (),
    )
    .expect("Insert failed");

    // GROUP BY expression with HAVING - grouped column should not be NULL
    let result = db
        .query(
            "SELECT UPPER(name) as upper_name, SUM(value) as total FROM t40 GROUP BY UPPER(name) HAVING SUM(value) > 30 ORDER BY upper_name",
            (),
        )
        .expect("HAVING with expression GROUP BY should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "Only BOB has SUM > 30 (40)");

    let row = rows[0].as_ref().unwrap();
    let upper_name: String = row.get(0).unwrap();
    let total: i64 = row.get(1).unwrap();
    assert_eq!(upper_name, "BOB", "Grouped expression should not be NULL");
    assert_eq!(total, 40);
}

// =============================================================================
// Bug #41: Scalar subquery in CASE expression not supported
// Description: Using a scalar subquery inside a CASE WHEN condition failed
// =============================================================================
#[test]
fn test_bug_41_scalar_subquery_in_case() {
    let db = setup_db("bug41");

    db.execute(
        "CREATE TABLE emp41 (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO emp41 VALUES (1, 'Alice', 'Engineering', 100000.0)",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO emp41 VALUES (2, 'Bob', 'Engineering', 80000.0)",
        (),
    )
    .expect("Insert failed");
    db.execute(
        "INSERT INTO emp41 VALUES (3, 'Charlie', 'Sales', 70000.0)",
        (),
    )
    .expect("Insert failed");

    // Scalar subquery in CASE WHEN condition
    let result = db
        .query(
            "SELECT name, salary,
                CASE
                    WHEN salary > (SELECT AVG(salary) FROM emp41) THEN 'Above Average'
                    ELSE 'Below Average'
                END as salary_level
             FROM emp41 ORDER BY name",
            (),
        )
        .expect("Scalar subquery in CASE should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // AVG salary = (100000 + 80000 + 70000) / 3 = 83333.33
    // Alice (100000) > avg -> Above Average
    // Bob (80000) < avg -> Below Average
    // Charlie (70000) < avg -> Below Average
    let row0 = rows[0].as_ref().unwrap();
    let name0: String = row0.get(0).unwrap();
    let level0: String = row0.get(2).unwrap();
    assert_eq!(name0, "Alice");
    assert_eq!(level0, "Above Average");

    let row1 = rows[1].as_ref().unwrap();
    let name1: String = row1.get(0).unwrap();
    let level1: String = row1.get(2).unwrap();
    assert_eq!(name1, "Bob");
    assert_eq!(level1, "Below Average");
}

// =============================================================================
// Bug #51: IS TRUE / IS FALSE not supported
// Description: SQL standard IS TRUE and IS FALSE predicates were not recognized
// =============================================================================
#[test]
fn test_bug_51_is_true_false() {
    let db = setup_db("bug51");

    db.execute(
        "CREATE TABLE t51 (id INTEGER PRIMARY KEY, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t51 VALUES (1, true), (2, false), (3, NULL)",
        (),
    )
    .expect("Insert failed");

    // IS TRUE
    let result = db
        .query("SELECT id FROM t51 WHERE active IS TRUE ORDER BY id", ())
        .expect("IS TRUE should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "Only id=1 has active IS TRUE");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 1);

    // IS FALSE
    let result = db
        .query("SELECT id FROM t51 WHERE active IS FALSE ORDER BY id", ())
        .expect("IS FALSE should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "Only id=2 has active IS FALSE");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 2);

    // IS NOT TRUE (false or NULL)
    let result = db
        .query(
            "SELECT id FROM t51 WHERE active IS NOT TRUE ORDER BY id",
            (),
        )
        .expect("IS NOT TRUE should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "id=2 and id=3 have active IS NOT TRUE");
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![2, 3]);

    // IS NOT FALSE (true or NULL)
    let result = db
        .query(
            "SELECT id FROM t51 WHERE active IS NOT FALSE ORDER BY id",
            (),
        )
        .expect("IS NOT FALSE should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "id=1 and id=3 have active IS NOT FALSE");
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 3]);
}

// =============================================================================
// Bug #51b: IS TRUE/FALSE in storage expressions (DELETE/UPDATE)
// Description: IS TRUE/FALSE should work with storage layer expressions
// =============================================================================
#[test]
fn test_bug_51_is_true_false_storage_expressions() {
    let db = setup_db("bug51b");

    db.execute(
        "CREATE TABLE t51b (id INTEGER PRIMARY KEY, active BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t51b VALUES (1, true), (2, false), (3, NULL)",
        (),
    )
    .expect("Insert failed");

    // DELETE with IS TRUE (should use storage expression)
    db.execute("DELETE FROM t51b WHERE active IS TRUE", ())
        .expect("DELETE with IS TRUE should work");

    let result = db.query("SELECT id FROM t51b ORDER BY id", ()).unwrap();
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Row with active IS TRUE should be deleted");
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![2, 3]);

    // Reset table
    db.execute("DROP TABLE t51b", ()).unwrap();
    db.execute(
        "CREATE TABLE t51b (id INTEGER PRIMARY KEY, active BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t51b VALUES (1, true), (2, false), (3, NULL)",
        (),
    )
    .unwrap();

    // UPDATE with IS FALSE
    db.execute("UPDATE t51b SET active = true WHERE active IS FALSE", ())
        .expect("UPDATE with IS FALSE should work");

    let result = db
        .query("SELECT id, active FROM t51b WHERE id = 2", ())
        .unwrap();
    let rows: Vec<_> = result.collect();
    let row = rows[0].as_ref().unwrap();
    let active: bool = row.get(1).unwrap();
    assert!(active, "Row with IS FALSE should be updated to true");
}

// =============================================================================
// Bug #52: XOR operator returns NULL
// Description: XOR operator existed but returned NULL instead of logical XOR
// =============================================================================
#[test]
fn test_bug_52_xor_operator() {
    let db = setup_db("bug52");

    // Test basic XOR in SELECT
    let result = db
        .query("SELECT true XOR false", ())
        .expect("XOR should work");
    let rows: Vec<_> = result.collect();
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(val, "true XOR false should be true");

    let result = db
        .query("SELECT true XOR true", ())
        .expect("XOR should work");
    let rows: Vec<_> = result.collect();
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(!val, "true XOR true should be false");

    let result = db
        .query("SELECT false XOR false", ())
        .expect("XOR should work");
    let rows: Vec<_> = result.collect();
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(!val, "false XOR false should be false");

    let result = db
        .query("SELECT false XOR true", ())
        .expect("XOR should work");
    let rows: Vec<_> = result.collect();
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(val, "false XOR true should be true");

    // Test XOR in WHERE clause
    db.execute(
        "CREATE TABLE t52 (id INTEGER PRIMARY KEY, a BOOLEAN, b BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t52 VALUES (1, true, true), (2, true, false), (3, false, true), (4, false, false)",
        (),
    )
    .expect("Insert failed");

    let result = db
        .query("SELECT id FROM t52 WHERE a XOR b ORDER BY id", ())
        .expect("XOR in WHERE should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "XOR should be true for rows 2 and 3");
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![2, 3]);
}

// =============================================================================
// Bug #52b: XOR in storage expressions (DELETE/UPDATE)
// Description: XOR should work with storage layer expressions
// =============================================================================
#[test]
fn test_bug_52_xor_storage_expressions() {
    let db = setup_db("bug52b");

    db.execute(
        "CREATE TABLE t52b (id INTEGER PRIMARY KEY, a BOOLEAN, b BOOLEAN)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t52b VALUES (1, true, true), (2, true, false), (3, false, true), (4, false, false)",
        (),
    )
    .expect("Insert failed");

    // DELETE with XOR (should delete rows where exactly one is true)
    db.execute("DELETE FROM t52b WHERE a XOR b", ())
        .expect("DELETE with XOR should work");

    let result = db.query("SELECT id FROM t52b ORDER BY id", ()).unwrap();
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Rows 2 and 3 should be deleted (XOR = true)");
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 4], "Only rows where both same should remain");
}

// =============================================================================
// Bug #56: SUBSTR function not implemented
// Description: SUBSTR should be an alias for SUBSTRING
// =============================================================================
#[test]
fn test_bug_56_substr_function() {
    let db = setup_db("bug56");

    // SUBSTR with 2 arguments (from start position to end)
    let result = db
        .query("SELECT SUBSTR('hello world', 7)", ())
        .expect("SUBSTR with 2 args should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let value: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(value, "world");

    // SUBSTR with 3 arguments (from position, length)
    let result = db
        .query("SELECT SUBSTR('hello world', 1, 5)", ())
        .expect("SUBSTR with 3 args should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let value: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(value, "hello");
}

// =============================================================================
// Bug #57: JOIN USING clause not supported
// Description: JOIN ... USING (column) should work like JOIN ... ON t1.column = t2.column
// =============================================================================
#[test]
fn test_bug_57_join_using() {
    let db = setup_db("bug57");

    db.execute("CREATE TABLE t57a (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Create table failed");

    db.execute(
        "CREATE TABLE t57b (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Create table failed");

    db.execute(
        "INSERT INTO t57a VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .expect("Insert failed");

    db.execute("INSERT INTO t57b VALUES (1, 100), (2, 200), (4, 400)", ())
        .expect("Insert failed");

    // INNER JOIN USING
    let result = db
        .query(
            "SELECT t57a.name, t57b.value FROM t57a JOIN t57b USING (id) ORDER BY t57a.name",
            (),
        )
        .expect("JOIN USING should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should return 2 matching rows");

    let names: Vec<String> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(names, vec!["Alice", "Bob"]);

    // LEFT JOIN USING
    let result = db
        .query(
            "SELECT t57a.name, t57b.value FROM t57a LEFT JOIN t57b USING (id) ORDER BY t57a.name",
            (),
        )
        .expect("LEFT JOIN USING should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        3,
        "Should return 3 rows (with NULL for Charlie)"
    );
}

// =============================================================================
// Bug #58: CREATE TABLE AS SELECT not supported
// Description: CREATE TABLE ... AS SELECT should create and populate a table
// =============================================================================
#[test]
fn test_bug_58_create_table_as_select() {
    let db = setup_db("bug58");

    db.execute(
        "CREATE TABLE t58 (id INTEGER PRIMARY KEY, name TEXT, value FLOAT)",
        (),
    )
    .expect("Create table failed");

    db.execute(
        "INSERT INTO t58 VALUES (1, 'Alice', 100.0), (2, 'Bob', 200.0), (3, 'Charlie', 150.0)",
        (),
    )
    .expect("Insert failed");

    // Basic CREATE TABLE AS SELECT
    db.execute("CREATE TABLE t58_copy AS SELECT * FROM t58", ())
        .expect("CREATE TABLE AS SELECT should work");

    let result = db.query("SELECT * FROM t58_copy ORDER BY id", ()).unwrap();
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "Should have copied all 3 rows");

    // CREATE TABLE AS SELECT with WHERE clause
    db.execute(
        "CREATE TABLE t58_filtered AS SELECT id, name FROM t58 WHERE value > 120.0",
        (),
    )
    .expect("CREATE TABLE AS SELECT with WHERE should work");

    let result = db
        .query("SELECT * FROM t58_filtered ORDER BY id", ())
        .unwrap();
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have 2 filtered rows");

    let names: Vec<String> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(1).unwrap())
        .collect();
    assert_eq!(names, vec!["Bob", "Charlie"]);

    // CREATE TABLE AS SELECT with aggregation
    db.execute(
        "CREATE TABLE t58_agg AS SELECT COUNT(*) as cnt, SUM(value) as total FROM t58",
        (),
    )
    .expect("CREATE TABLE AS SELECT with aggregation should work");

    let result = db.query("SELECT * FROM t58_agg", ()).unwrap();
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let cnt: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(cnt, 3);
}

// =============================================================================
// Bug #44: CHECK constraint not enforced (partial fix for parsing)
// Description: CHECK constraints should be validated during INSERT
// =============================================================================
#[test]
fn test_check_constraint() {
    let db = setup_db("check_constraint");

    db.execute(
        "CREATE TABLE t_check (id INTEGER PRIMARY KEY, value INTEGER CHECK (value > 0))",
        (),
    )
    .expect("Create table with CHECK constraint should work");

    // Valid value should succeed
    db.execute("INSERT INTO t_check VALUES (1, 100)", ())
        .expect("Insert with valid value should work");

    // Invalid value should fail
    let result = db.execute("INSERT INTO t_check VALUES (2, -50)", ());
    assert!(
        result.is_err(),
        "Insert with value <= 0 should fail CHECK constraint"
    );

    // Verify only the valid row is inserted
    let result = db.query("SELECT * FROM t_check", ()).unwrap();
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "Only one row should be inserted");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 1);
}

// =============================================================================
// Bug #45: DEFAULT values not applied (now fixed)
// Description: DEFAULT values should be applied when columns are not specified
// =============================================================================
#[test]
fn test_default_values() {
    let db = setup_db("default_values");

    db.execute(
        "CREATE TABLE t_default (id INTEGER PRIMARY KEY, name TEXT DEFAULT 'unnamed', value INTEGER DEFAULT 42)",
        (),
    )
    .expect("Create table with DEFAULT should work");

    // Insert with only id
    db.execute("INSERT INTO t_default (id) VALUES (1)", ())
        .expect("Insert with partial columns should work");

    // Insert with id and name
    db.execute("INSERT INTO t_default (id, name) VALUES (2, 'Bob')", ())
        .expect("Insert with partial columns should work");

    // Insert with all columns
    db.execute("INSERT INTO t_default VALUES (3, 'Alice', 100)", ())
        .expect("Insert with all columns should work");

    // Verify defaults were applied
    let result = db.query("SELECT * FROM t_default ORDER BY id", ()).unwrap();
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // Row 1: defaults for name and value
    let name1: String = rows[0].as_ref().unwrap().get(1).unwrap();
    let value1: i64 = rows[0].as_ref().unwrap().get(2).unwrap();
    assert_eq!(name1, "unnamed");
    assert_eq!(value1, 42);

    // Row 2: default for value only
    let name2: String = rows[1].as_ref().unwrap().get(1).unwrap();
    let value2: i64 = rows[1].as_ref().unwrap().get(2).unwrap();
    assert_eq!(name2, "Bob");
    assert_eq!(value2, 42);

    // Row 3: no defaults
    let name3: String = rows[2].as_ref().unwrap().get(1).unwrap();
    let value3: i64 = rows[2].as_ref().unwrap().get(2).unwrap();
    assert_eq!(name3, "Alice");
    assert_eq!(value3, 100);
}

// =============================================================================
// Bug #42: UNIQUE constraint enforcement (already fixed, adding test)
// Description: UNIQUE constraints should reject duplicate values
// =============================================================================
#[test]
fn test_unique_constraint() {
    let db = setup_db("unique_constraint");

    db.execute(
        "CREATE TABLE t_unique (id INTEGER PRIMARY KEY, email TEXT UNIQUE)",
        (),
    )
    .expect("Create table with UNIQUE constraint should work");

    // First insert should succeed
    db.execute("INSERT INTO t_unique VALUES (1, 'test@example.com')", ())
        .expect("First insert should work");

    // Second insert with same email should fail
    let result = db.execute("INSERT INTO t_unique VALUES (2, 'test@example.com')", ());
    assert!(
        result.is_err(),
        "Insert with duplicate UNIQUE value should fail"
    );

    // Different email should succeed
    db.execute("INSERT INTO t_unique VALUES (3, 'other@example.com')", ())
        .expect("Insert with different value should work");

    // Verify correct rows
    let result = db.query("SELECT * FROM t_unique ORDER BY id", ()).unwrap();
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);
}

// =============================================================================
// Bug #62: GROUP BY with qualified column name returns NULL
// Description: SELECT t.name FROM t GROUP BY t.name returns NULL instead of actual values
// =============================================================================
#[test]
fn test_bug_62_group_by_qualified_column() {
    let db = setup_db("bug62");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, price FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO products VALUES (1, 'Electronics', 100.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO products VALUES (2, 'Electronics', 200.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO products VALUES (3, 'Books', 50.0)", ())
        .expect("Insert failed");

    // Test qualified column in SELECT and GROUP BY
    let result = db
        .query(
            "SELECT products.category, SUM(products.price) as total FROM products GROUP BY products.category ORDER BY products.category",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have 2 categories");

    let row1 = rows[0].as_ref().expect("Row should exist");
    let category1: String = row1.get(0).unwrap();
    let total1: f64 = row1.get(1).unwrap();
    assert_eq!(category1, "Books");
    assert_eq!(total1, 50.0);

    let row2 = rows[1].as_ref().expect("Row should exist");
    let category2: String = row2.get(0).unwrap();
    let total2: f64 = row2.get(1).unwrap();
    assert_eq!(category2, "Electronics");
    assert_eq!(total2, 300.0);
}

// =============================================================================
// Bug #63: Scalar subquery in SELECT with GROUP BY returns NULL
// Description: SELECT (SELECT MAX(val) FROM t2) FROM t1 GROUP BY col returns NULL
// =============================================================================
#[test]
fn test_bug_63_scalar_subquery_with_group_by() {
    let db = setup_db("bug63");

    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO customers VALUES (1, 'Alice', 'USA')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO customers VALUES (2, 'Bob', 'UK')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO customers VALUES (3, 'Charlie', 'USA')", ())
        .expect("Insert failed");

    db.execute("INSERT INTO orders VALUES (1, 1, 100.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (2, 1, 200.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (3, 3, 150.0)", ())
        .expect("Insert failed");

    // Scalar subquery with GROUP BY
    let result = db
        .query(
            "SELECT country, COUNT(*) as cnt, (SELECT MAX(amount) FROM orders) as max_order FROM customers GROUP BY country ORDER BY country",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have 2 countries");

    // UK row
    let row1 = rows[0].as_ref().expect("Row should exist");
    let country1: String = row1.get(0).unwrap();
    let cnt1: i64 = row1.get(1).unwrap();
    let max_order1: f64 = row1.get(2).unwrap();
    assert_eq!(country1, "UK");
    assert_eq!(cnt1, 1);
    assert_eq!(
        max_order1, 200.0,
        "Scalar subquery should return 200.0, not NULL"
    );

    // USA row
    let row2 = rows[1].as_ref().expect("Row should exist");
    let country2: String = row2.get(0).unwrap();
    let cnt2: i64 = row2.get(1).unwrap();
    let max_order2: f64 = row2.get(2).unwrap();
    assert_eq!(country2, "USA");
    assert_eq!(cnt2, 2);
    assert_eq!(
        max_order2, 200.0,
        "Scalar subquery should return 200.0, not NULL"
    );
}

// =============================================================================
// Bug #64: Subquery in HAVING returns no results
// Description: HAVING (SELECT MAX(x) FROM t) > 0 gives "Scalar subquery not yet implemented"
// =============================================================================
#[test]
fn test_bug_64_subquery_in_having() {
    let db = setup_db("bug64");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "CREATE TABLE thresholds (id INTEGER PRIMARY KEY, min_amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO sales VALUES (1, 'North', 100.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO sales VALUES (2, 'North', 200.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO sales VALUES (3, 'South', 50.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO sales VALUES (4, 'South', 75.0)", ())
        .expect("Insert failed");

    db.execute("INSERT INTO thresholds VALUES (1, 150.0)", ())
        .expect("Insert failed");

    // Subquery in HAVING clause
    let result = db
        .query(
            "SELECT region, SUM(amount) as total FROM sales GROUP BY region HAVING SUM(amount) > (SELECT min_amount FROM thresholds WHERE id = 1)",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "Only North region has sum > 150");

    let row = rows[0].as_ref().expect("Row should exist");
    let region: String = row.get(0).unwrap();
    let total: f64 = row.get(1).unwrap();
    assert_eq!(region, "North");
    assert_eq!(total, 300.0);
}

// =============================================================================
// Bug #67: Window function with JOIN drops non-window columns
// Description: SELECT c.name, ROW_NUMBER() OVER ... FROM a JOIN b loses c.name
// =============================================================================
#[test]
fn test_bug_67_window_function_with_join() {
    let db = setup_db("bug67");

    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO customers VALUES (1, 'Alice')", ())
        .expect("Insert failed");
    db.execute("INSERT INTO customers VALUES (2, 'Bob')", ())
        .expect("Insert failed");

    db.execute("INSERT INTO orders VALUES (1, 1, 100.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (2, 1, 200.0)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO orders VALUES (3, 2, 150.0)", ())
        .expect("Insert failed");

    // Window function with JOIN - qualified column names should be preserved
    // The bug was that c.name would return NULL instead of actual values
    let result = db
        .query(
            "SELECT c.name, o.amount, ROW_NUMBER() OVER (ORDER BY o.amount DESC) as rn FROM customers c INNER JOIN orders o ON c.id = o.customer_id",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "Should have 3 rows");

    // Verify that qualified column names (c.name, o.amount) are not NULL
    // This was the actual bug - qualified names from JOINs were being dropped
    // We verify all names are valid strings (not NULL) and amounts are valid floats
    let mut names: Vec<String> = Vec::new();
    let mut amounts: Vec<f64> = Vec::new();
    for row in &rows {
        let row = row.as_ref().expect("Row should exist");
        let name: String = row.get(0).unwrap();
        let amount: f64 = row.get(1).unwrap();
        names.push(name);
        amounts.push(amount);
    }

    // All names should be valid (not NULL or empty) - this was the bug
    assert!(
        names.iter().all(|n| !n.is_empty()),
        "All names should be non-empty - bug #67 fix"
    );

    // Check we have correct names (Alice appears twice, Bob once)
    let alice_count = names.iter().filter(|n| *n == "Alice").count();
    let bob_count = names.iter().filter(|n| *n == "Bob").count();
    assert_eq!(alice_count, 2, "Alice should appear twice");
    assert_eq!(bob_count, 1, "Bob should appear once");

    // Check amounts are valid
    amounts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(amounts, vec![100.0, 150.0, 200.0]);
}

// =============================================================================
// Bug #68: Three-valued logic incorrect (NULL in boolean expressions)
// Description: NULL AND TRUE returns FALSE instead of NULL
// =============================================================================
#[test]
fn test_bug_68_three_valued_logic() {
    let db = setup_db("bug68");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (2, NULL)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t VALUES (3, 30)", ())
        .expect("Insert failed");

    // NULL AND TRUE should be NULL (unknown), not FALSE
    // So WHERE (val IS NULL) AND TRUE should match row 2
    let result = db
        .query("SELECT id FROM t WHERE (val IS NULL) AND TRUE", ())
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 2);

    // Test NULL OR FALSE = NULL (should not match in WHERE)
    // WHERE NULL OR FALSE filters out because result is NULL
    let result = db
        .query("SELECT id FROM t WHERE NULL OR FALSE", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 0, "NULL OR FALSE = NULL, should match no rows");

    // Test NULL OR TRUE = TRUE (should match all rows)
    let result = db
        .query("SELECT id FROM t WHERE NULL OR TRUE ORDER BY id", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "NULL OR TRUE = TRUE, should match all rows");

    // Test NULL AND FALSE = FALSE (should match no rows)
    let result = db
        .query("SELECT id FROM t WHERE NULL AND FALSE", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        0,
        "NULL AND FALSE = FALSE, should match no rows"
    );

    // Test NOT NULL = NULL
    let result = db
        .query("SELECT id FROM t WHERE NOT NULL", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 0, "NOT NULL = NULL, should match no rows");
}

// =============================================================================
// Bug #69: LIMIT+OFFSET without ORDER BY returns empty results
// Description: SELECT * FROM t LIMIT 3 OFFSET 3 returns empty when it shouldn't
// =============================================================================
#[test]
fn test_bug_69_limit_offset_without_order_by() {
    let db = setup_db("bug69");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO t VALUES ({})", i), ())
            .expect("Insert failed");
    }

    // LIMIT only should work
    let result = db
        .query("SELECT * FROM t LIMIT 3", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "LIMIT 3 should return 3 rows");

    // OFFSET only should work
    let result = db
        .query("SELECT * FROM t OFFSET 7", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "OFFSET 7 should return 3 rows (8, 9, 10)");

    // LIMIT + OFFSET without ORDER BY should work (this was the bug)
    let result = db
        .query("SELECT * FROM t LIMIT 3 OFFSET 3", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "LIMIT 3 OFFSET 3 should return 3 rows");

    // LIMIT + OFFSET with ORDER BY for comparison
    let result = db
        .query("SELECT * FROM t ORDER BY id LIMIT 3 OFFSET 3", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        3,
        "ORDER BY + LIMIT 3 OFFSET 3 should return 3 rows"
    );
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![4, 5, 6]);
}

// =============================================================================
// Bug #70: Negative prefix on aggregate from derived table ignored
// Description: -SUM(val) returns positive value instead of negated
// =============================================================================
#[test]
fn test_bug_70_negative_prefix_aggregate() {
    let db = setup_db("bug70");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .expect("Insert failed");

    // Direct negative aggregate
    let result: i64 = db
        .query_one("SELECT -SUM(val) FROM t", ())
        .expect("Query should succeed");
    assert_eq!(result, -30, "-SUM(val) should be -30");

    // Arithmetic with aggregate
    let result: i64 = db
        .query_one("SELECT SUM(val) * -1 FROM t", ())
        .expect("Query should succeed");
    assert_eq!(result, -30, "SUM(val) * -1 should be -30");

    // Addition with aggregate
    let result: i64 = db
        .query_one("SELECT SUM(val) + 5 FROM t", ())
        .expect("Query should succeed");
    assert_eq!(result, 35, "SUM(val) + 5 should be 35");

    // Derived table with negation
    let result: i64 = db
        .query_one(
            "SELECT -total FROM (SELECT SUM(val) AS total FROM t) sub",
            (),
        )
        .expect("Query should succeed");
    assert_eq!(result, -30, "-total from derived table should be -30");

    // Original bug reproduction case
    let result: i64 = db
        .query_one("SELECT -SUM(val) FROM (SELECT 10 as val) sub", ())
        .expect("Query should succeed");
    assert_eq!(result, -10, "-SUM(val) from derived should be -10");
}

// =============================================================================
// Bug #71: TRUNCATE TABLE does nothing
// Description: TRUNCATE command is parsed but doesn't delete any rows
// =============================================================================
#[test]
fn test_bug_71_truncate_table() {
    let db = setup_db("bug71");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    // Verify rows exist
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("Query should succeed");
    assert_eq!(count, 3, "Should have 3 rows before TRUNCATE");

    // TRUNCATE TABLE t
    db.execute("TRUNCATE TABLE t", ())
        .expect("TRUNCATE should succeed");

    // Verify rows are deleted
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("Query should succeed");
    assert_eq!(count, 0, "Should have 0 rows after TRUNCATE");

    // Table should still exist (TRUNCATE != DROP)
    db.execute("INSERT INTO t VALUES (1, 100)", ())
        .expect("Should be able to insert after TRUNCATE");

    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t", ())
        .expect("Query should succeed");
    assert_eq!(count, 1, "Should have 1 row after new INSERT");
}

// =============================================================================
// Bug #72: Aggregate + window + GROUP BY returns NULL
// Description: When combining aggregate functions with window functions and GROUP BY,
//              the aggregate results are NULL
// =============================================================================
#[test]
fn test_bug_72_aggregate_window_group_by() {
    let db = setup_db("bug72");

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, category TEXT, amount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO orders VALUES (1, 'A', 100.0), (2, 'A', 150.0), (3, 'B', 200.0), (4, 'B', 50.0)",
        (),
    )
    .expect("Insert failed");

    // Aggregate with window function and GROUP BY
    let result = db
        .query(
            "SELECT category, SUM(amount) AS total, RANK() OVER (ORDER BY SUM(amount) DESC) AS rnk FROM orders GROUP BY category",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have 2 rows (A and B)");

    // Extract values
    let mut data: Vec<(String, f64, i64)> = rows
        .iter()
        .map(|r| {
            let r = r.as_ref().unwrap();
            (
                r.get::<String>(0).unwrap(),
                r.get::<f64>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();
    data.sort_by(|a, b| a.0.cmp(&b.0)); // Sort by category name for deterministic order

    // Both categories have total 250.0 and rank 1 (tied)
    // Category A: 100 + 150 = 250
    assert_eq!(data[0].0, "A");
    assert_eq!(data[0].1, 250.0);
    assert_eq!(data[0].2, 1);

    // Category B: 200 + 50 = 250
    assert_eq!(data[1].0, "B");
    assert_eq!(data[1].1, 250.0);
    assert_eq!(data[1].2, 1);
}

// =============================================================================
// Bug #73: DISTINCT with ORDER BY non-SELECT column
// Description: SELECT DISTINCT a FROM t ORDER BY b returns duplicates
// =============================================================================
#[test]
fn test_bug_73_distinct_order_by_non_select_column() {
    let db = setup_db("bug73");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO t VALUES (1, 'X', 1), (2, 'Y', 2), (3, 'X', 3)",
        (),
    )
    .expect("Insert failed");

    // DISTINCT with ORDER BY on column not in SELECT
    let result = db
        .query("SELECT DISTINCT a FROM t ORDER BY b", ())
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        2,
        "DISTINCT should return only 2 unique values (X, Y)"
    );

    let values: Vec<String> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();

    // Should be X, Y (ordered by their first appearance in b order)
    assert_eq!(values, vec!["X", "Y"]);

    // Without ORDER BY for comparison
    let result = db
        .query("SELECT DISTINCT a FROM t", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        2,
        "DISTINCT without ORDER BY should also return 2 rows"
    );

    // ORDER BY on SELECT column should also work
    let result = db
        .query("SELECT DISTINCT a FROM t ORDER BY a DESC", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);
    let values: Vec<String> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(values, vec!["Y", "X"]);
}

// =============================================================================
// Bug #74: JSON_EXTRACT function not supported
// Description: The JSON_EXTRACT function was not implemented
// =============================================================================
#[test]
fn test_bug_74_json_extract_function() {
    let db = setup_db("bug74");

    db.execute("CREATE TABLE j (id INTEGER PRIMARY KEY, data JSON)", ())
        .expect("Failed to create table");

    db.execute(
        "INSERT INTO j VALUES (1, '{\"name\": \"Alice\", \"age\": 30}')",
        (),
    )
    .expect("Failed to insert data");

    // Test JSON_EXTRACT
    let result = db
        .query("SELECT JSON_EXTRACT(data, '$.name') FROM j", ())
        .expect("JSON_EXTRACT should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let name: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(name, "Alice");

    // Test extracting a number
    let result = db
        .query("SELECT JSON_EXTRACT(data, '$.age') FROM j", ())
        .expect("JSON_EXTRACT should work for numbers");
    let rows: Vec<_> = result.collect();
    let age: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(age, 30);
}

// =============================================================================
// Bug #75: TYPEOF function not supported
// Description: The TYPEOF function to get the type of a value was not implemented
// =============================================================================
#[test]
fn test_bug_75_typeof_function() {
    let db = setup_db("bug75");

    // Test TYPEOF with various types
    let result = db
        .query("SELECT TYPEOF(123)", ())
        .expect("TYPEOF should work with integers");
    let rows: Vec<_> = result.collect();
    let type_name: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(type_name, "INTEGER");

    let result = db
        .query("SELECT TYPEOF('hello')", ())
        .expect("TYPEOF should work with text");
    let rows: Vec<_> = result.collect();
    let type_name: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(type_name, "TEXT");

    let result = db
        .query("SELECT TYPEOF(3.14)", ())
        .expect("TYPEOF should work with floats");
    let rows: Vec<_> = result.collect();
    let type_name: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(type_name, "FLOAT");

    let result = db
        .query("SELECT TYPEOF(TRUE)", ())
        .expect("TYPEOF should work with booleans");
    let rows: Vec<_> = result.collect();
    let type_name: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(type_name, "BOOLEAN");

    let result = db
        .query("SELECT TYPEOF(NULL)", ())
        .expect("TYPEOF should work with NULL");
    let rows: Vec<_> = result.collect();
    let type_name: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(type_name, "NULL");
}

// =============================================================================
// Bug #76: CAST to DATE not supported
// Description: CAST to DATE type was not supported
// =============================================================================
#[test]
fn test_bug_76_cast_to_date() {
    let db = setup_db("bug76");

    // Test CAST to DATE
    let result = db
        .query("SELECT CAST('2024-01-15' AS DATE)", ())
        .expect("CAST to DATE should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    // The result should be a valid timestamp
    let date_str: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(
        date_str.contains("2024-01-15"),
        "Date should contain 2024-01-15"
    );

    // Test CAST to TIMESTAMP
    let result = db
        .query("SELECT CAST('2024-06-15 12:30:45' AS TIMESTAMP)", ())
        .expect("CAST to TIMESTAMP should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
}

// =============================================================================
// Bug #77: DATE_DIFF vs DATEDIFF inconsistent naming
// Description: DATE_DIFF (with underscore) didn't work but DATEDIFF did
// =============================================================================
#[test]
fn test_bug_77_date_diff_alias() {
    let db = setup_db("bug77");

    // Test DATEDIFF (original)
    let result = db
        .query("SELECT DATEDIFF('2024-03-20', '2024-03-15')", ())
        .expect("DATEDIFF should work");
    let rows: Vec<_> = result.collect();
    let diff: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(diff, 5);

    // Test DATE_DIFF (alias)
    let result = db
        .query("SELECT DATE_DIFF('2024-03-20', '2024-03-15')", ())
        .expect("DATE_DIFF should work as alias");
    let rows: Vec<_> = result.collect();
    let diff: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(diff, 5);
}

// =============================================================================
// Bug #78: Integer overflow wraps silently
// Description: Integer overflow wrapped around without error
// =============================================================================
#[test]
fn test_bug_78_integer_overflow_handling() {
    let db = setup_db("bug78");

    // Test overflow on addition
    let result = db.query("SELECT 9223372036854775807 + 1", ());
    assert!(
        result.is_err(),
        "Integer overflow should return an error, not wrap"
    );

    // Test overflow on multiplication
    let result = db.query("SELECT 9223372036854775807 * 2", ());
    assert!(
        result.is_err(),
        "Integer overflow on multiplication should return an error"
    );

    // Test overflow on subtraction (negative direction)
    let result = db.query("SELECT -9223372036854775808 - 1", ());
    assert!(result.is_err(), "Integer underflow should return an error");

    // Normal arithmetic should still work
    let result = db
        .query("SELECT 100 + 200", ())
        .expect("Normal arithmetic should work");
    let rows: Vec<_> = result.collect();
    let sum: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(sum, 300);
}

// =============================================================================
// Bug #92: Boolean short-circuit evaluation not implemented
// Description: Boolean AND/OR does not short-circuit, causing division by zero
//              errors even when they shouldn't be evaluated
// =============================================================================
#[test]
fn test_bug_92_boolean_short_circuit() {
    let db = setup_db("bug92");

    // FALSE AND <anything> should short-circuit and return FALSE
    // without evaluating the right side (which would cause division by zero)
    let result = db
        .query("SELECT false AND (1/0 > 0)", ())
        .expect("FALSE AND (error) should short-circuit to FALSE");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(!val, "FALSE AND anything = FALSE");

    // TRUE OR <anything> should short-circuit and return TRUE
    // without evaluating the right side (which would cause division by zero)
    let result = db
        .query("SELECT true OR (1/0 > 0)", ())
        .expect("TRUE OR (error) should short-circuit to TRUE");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(val, "TRUE OR anything = TRUE");

    // Normal AND/OR should still work correctly
    let result = db
        .query("SELECT true AND false", ())
        .expect("Normal AND should work");
    let rows: Vec<_> = result.collect();
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(!val, "TRUE AND FALSE = FALSE");

    let result = db
        .query("SELECT false OR true", ())
        .expect("Normal OR should work");
    let rows: Vec<_> = result.collect();
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(val, "FALSE OR TRUE = TRUE");

    // Multiple short-circuit conditions in sequence
    let result = db
        .query("SELECT false AND false AND (1/0 > 0)", ())
        .expect("Chained FALSE AND should short-circuit");
    let rows: Vec<_> = result.collect();
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(!val, "FALSE AND FALSE AND anything = FALSE");

    let result = db
        .query("SELECT true OR true OR (1/0 > 0)", ())
        .expect("Chained TRUE OR should short-circuit");
    let rows: Vec<_> = result.collect();
    let val: bool = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(val, "TRUE OR TRUE OR anything = TRUE");
}

// =============================================================================
// Bug #93: DESCRIBE table returns empty result
// Description: The DESCRIBE command parses but returns no data
// =============================================================================
#[test]
fn test_bug_93_describe_table() {
    let db = setup_db("bug93");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price FLOAT)",
        (),
    )
    .expect("Failed to create table");

    // DESCRIBE should return table structure
    let result = db
        .query("DESCRIBE products", ())
        .expect("DESCRIBE should work");

    let columns = result.columns();
    assert_eq!(columns.len(), 6, "DESCRIBE should return 6 columns");
    assert_eq!(columns[0], "Field");
    assert_eq!(columns[1], "Type");
    assert_eq!(columns[2], "Null");
    assert_eq!(columns[3], "Key");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "Should return 3 rows for 3 columns");

    // Check first column is id with primary key
    let row = rows[0].as_ref().unwrap();
    let field: String = row.get(0).unwrap();
    let key: String = row.get(3).unwrap();
    assert_eq!(field, "id");
    assert_eq!(key, "PRI");

    // DESC shorthand should also work
    let result = db
        .query("DESC products", ())
        .expect("DESC shorthand should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);
}

// =============================================================================
// Bug #96: NATURAL JOIN duplicates common columns
// Description: SELECT * FROM t1 NATURAL JOIN t2 shows id twice
// =============================================================================
#[test]
fn test_bug_96_natural_join_column_deduplication() {
    let db = setup_db("bug96");

    db.execute("CREATE TABLE t1 (id INTEGER, val TEXT)", ())
        .expect("Create t1");
    db.execute("CREATE TABLE t2 (id INTEGER, data TEXT)", ())
        .expect("Create t2");
    db.execute("INSERT INTO t1 VALUES (1, 'A'), (2, 'B')", ())
        .expect("Insert t1");
    db.execute("INSERT INTO t2 VALUES (1, 'X'), (2, 'Y')", ())
        .expect("Insert t2");

    // NATURAL JOIN should show 'id' only once, not twice
    let result = db
        .query("SELECT * FROM t1 NATURAL JOIN t2", ())
        .expect("NATURAL JOIN should work");

    let columns = result.columns();
    // Should have 3 columns: id, val, data (not 4: t1.id, t1.val, t2.id, t2.data)
    assert_eq!(
        columns.len(),
        3,
        "NATURAL JOIN should deduplicate common columns, got: {:?}",
        columns
    );

    // The join column should appear first with an unqualified name
    assert_eq!(columns[0], "id", "Join column should be unqualified 'id'");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should return 2 matched rows");
}

// =============================================================================
// Bug #97: JOIN USING duplicates common columns
// Description: SELECT * FROM t1 JOIN t2 USING (id) shows id twice
// =============================================================================
#[test]
fn test_bug_97_join_using_column_deduplication() {
    let db = setup_db("bug97");

    db.execute("CREATE TABLE t1 (id INTEGER, val TEXT)", ())
        .expect("Create t1");
    db.execute("CREATE TABLE t2 (id INTEGER, data TEXT)", ())
        .expect("Create t2");
    db.execute("INSERT INTO t1 VALUES (1, 'A'), (2, 'B')", ())
        .expect("Insert t1");
    db.execute("INSERT INTO t2 VALUES (1, 'X'), (2, 'Y')", ())
        .expect("Insert t2");

    // JOIN USING should show 'id' only once
    let result = db
        .query("SELECT * FROM t1 JOIN t2 USING (id)", ())
        .expect("JOIN USING should work");

    let columns = result.columns();
    // Should have 3 columns: id, val, data (not 4)
    assert_eq!(
        columns.len(),
        3,
        "JOIN USING should deduplicate the USING column, got: {:?}",
        columns
    );

    // The USING column should appear first with an unqualified name
    assert_eq!(columns[0], "id", "USING column should be unqualified 'id'");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should return 2 matched rows");

    // Verify the data is correct
    let row = rows[0].as_ref().unwrap();
    let id: i64 = row.get(0).unwrap();
    let val: String = row.get(1).unwrap();
    let data: String = row.get(2).unwrap();
    assert_eq!(id, 1);
    assert_eq!(val, "A");
    assert_eq!(data, "X");
}

// =============================================================================
// Bug #98: Anti-join (LEFT JOIN ... WHERE IS NULL) returns extra NULL columns
// Description: SELECT t1.* shows NULLs from the right table
// =============================================================================
#[test]
fn test_bug_98_qualified_star_in_join() {
    let db = setup_db("bug98");

    db.execute("CREATE TABLE t1 (id INTEGER, grp TEXT, val INTEGER)", ())
        .expect("Create t1");
    db.execute("CREATE TABLE t2 (id INTEGER, grp TEXT, data TEXT)", ())
        .expect("Create t2");
    db.execute("INSERT INTO t1 VALUES (1, 'A', 10), (2, 'B', 20)", ())
        .expect("Insert t1");
    db.execute("INSERT INTO t2 VALUES (1, 'A', 'x')", ())
        .expect("Insert t2");

    // Anti-join pattern: SELECT t1.* should only show t1's columns
    let result = db
        .query(
            "SELECT t1.* FROM t1 LEFT JOIN t2 ON t1.id = t2.id AND t1.grp = t2.grp WHERE t2.id IS NULL",
            (),
        )
        .expect("Anti-join should work");

    let columns = result.columns();
    // Should have exactly 3 columns from t1 (id, grp, val), not 6
    assert_eq!(
        columns.len(),
        3,
        "SELECT t1.* should only return t1's columns, got: {:?}",
        columns
    );

    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        1,
        "Only row (2, 'B', 20) should match anti-join"
    );

    // Verify the matched row has correct values (no NULLs appended)
    let row = rows[0].as_ref().unwrap();
    let id: i64 = row.get(0).unwrap();
    let grp: String = row.get(1).unwrap();
    let val: i64 = row.get(2).unwrap();
    assert_eq!(id, 2);
    assert_eq!(grp, "B");
    assert_eq!(val, 20);
}

// =============================================================================
// Bug #47: GLOB and REGEXP operators
// Description: GLOB and REGEXP operators now work for pattern matching
// =============================================================================
#[test]
fn test_bug_47_glob_operator() {
    let db = setup_db("bug47_glob");

    db.execute("CREATE TABLE t47 (id INTEGER PRIMARY KEY, data TEXT)", ())
        .expect("Create table");

    db.execute(
        "INSERT INTO t47 VALUES (1, 'hello world'), (2, 'hello there'), (3, 'goodbye'), (4, 'test123'), (5, NULL)",
        (),
    )
    .expect("Insert");

    // Test GLOB with * wildcard (matches any sequence)
    let result = db
        .query(
            "SELECT id FROM t47 WHERE data GLOB 'hello*' ORDER BY id",
            (),
        )
        .expect("GLOB should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "GLOB 'hello*' should match 2 rows");
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 2]);

    // Test GLOB with ? wildcard (matches single character)
    let result = db
        .query(
            "SELECT id FROM t47 WHERE data GLOB 'goodby?' ORDER BY id",
            (),
        )
        .expect("GLOB should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "GLOB 'goodby?' should match 1 row");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 3);

    // Test GLOB with suffix pattern
    let result = db
        .query("SELECT id FROM t47 WHERE data GLOB '*123' ORDER BY id", ())
        .expect("GLOB should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "GLOB '*123' should match 1 row");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 4);

    // Test NOT GLOB
    let result = db
        .query(
            "SELECT id FROM t47 WHERE data NOT GLOB 'hello*' ORDER BY id",
            (),
        )
        .expect("NOT GLOB should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        2,
        "NOT GLOB 'hello*' should match goodbye and test123"
    );
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![3, 4]);
}

#[test]
fn test_bug_47_regexp_operator() {
    let db = setup_db("bug47_regexp");

    db.execute("CREATE TABLE t47r (id INTEGER PRIMARY KEY, data TEXT)", ())
        .expect("Create table");

    db.execute(
        "INSERT INTO t47r VALUES (1, 'hello world'), (2, 'hello there'), (3, 'goodbye'), (4, 'test123'), (5, NULL)",
        (),
    )
    .expect("Insert");

    // Test REGEXP with ^ anchor (start of string)
    let result = db
        .query(
            "SELECT id FROM t47r WHERE data REGEXP '^hello' ORDER BY id",
            (),
        )
        .expect("REGEXP should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "REGEXP '^hello' should match 2 rows");
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 2]);

    // Test REGEXP with $ anchor (end of string)
    let result = db
        .query(
            "SELECT id FROM t47r WHERE data REGEXP 'world$' ORDER BY id",
            (),
        )
        .expect("REGEXP should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "REGEXP 'world$' should match 1 row");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 1);

    // Test REGEXP with character class
    let result = db
        .query(
            "SELECT id FROM t47r WHERE data REGEXP '[0-9]+' ORDER BY id",
            (),
        )
        .expect("REGEXP should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "REGEXP '[0-9]+' should match test123");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 4);

    // Test NOT REGEXP
    let result = db
        .query(
            "SELECT id FROM t47r WHERE data NOT REGEXP '^hello' ORDER BY id",
            (),
        )
        .expect("NOT REGEXP should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        2,
        "NOT REGEXP '^hello' should match goodbye and test123"
    );
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![3, 4]);

    // Test RLIKE (alias for REGEXP)
    let result = db
        .query(
            "SELECT id FROM t47r WHERE data RLIKE 'good' ORDER BY id",
            (),
        )
        .expect("RLIKE should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1, "RLIKE 'good' should match goodbye");
    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 3);
}

// =============================================================================
// Bug #66: Aggregate in WHERE should error
// Description: Using aggregate function in WHERE clause should give an error
// =============================================================================
#[test]
fn test_bug_66_aggregate_in_where_error() {
    let db = setup_db("bug66");

    db.execute("CREATE TABLE t66 (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .expect("Create table");

    db.execute("INSERT INTO t66 VALUES (1, 10), (2, 20)", ())
        .expect("Insert");

    // Aggregate in WHERE should error
    let result = db.query("SELECT * FROM t66 WHERE SUM(a) > 10", ());
    match result {
        Ok(_) => panic!("Aggregate in WHERE should error"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("aggregate") && err.contains("WHERE"),
                "Error should mention aggregate and WHERE: {}",
                err
            );
        }
    }

    // COUNT in WHERE should also error
    let result = db.query("SELECT * FROM t66 WHERE COUNT(*) > 1", ());
    assert!(result.is_err(), "COUNT in WHERE should error");

    // AVG in WHERE should also error
    let result = db.query("SELECT * FROM t66 WHERE AVG(a) > 10", ());
    assert!(result.is_err(), "AVG in WHERE should error");

    // Aggregate in HAVING is fine (for comparison)
    let result = db.query("SELECT a FROM t66 GROUP BY a HAVING SUM(a) > 5", ());
    assert!(result.is_ok(), "Aggregate in HAVING should work");
}

// =============================================================================
// Bug #104: UNION column count mismatch not validated
// Description: UNION with different column counts should error
// =============================================================================
#[test]
fn test_bug_104_union_column_count_validation() {
    let db = setup_db("bug104");

    db.execute("CREATE TABLE t104a (a INTEGER, b INTEGER, c INTEGER)", ())
        .expect("Create table");
    db.execute("CREATE TABLE t104b (x INTEGER)", ())
        .expect("Create table");

    db.execute("INSERT INTO t104a VALUES (1, 2, 3)", ())
        .expect("Insert");
    db.execute("INSERT INTO t104b VALUES (100)", ())
        .expect("Insert");

    // UNION with mismatched column counts should error
    let result = db.query(
        "SELECT a, b, c FROM t104a UNION ALL SELECT x FROM t104b",
        (),
    );
    match result {
        Ok(_) => panic!("UNION with mismatched columns should error"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("UNION") && err.contains("column"),
                "Error should mention UNION and columns: {}",
                err
            );
        }
    }

    // INTERSECT with mismatched columns should also error
    let result = db.query("SELECT a, b FROM t104a INTERSECT SELECT x FROM t104b", ());
    assert!(
        result.is_err(),
        "INTERSECT with mismatched columns should error"
    );

    // EXCEPT with mismatched columns should also error
    let result = db.query("SELECT a FROM t104a EXCEPT SELECT x, x FROM t104b", ());
    assert!(
        result.is_err(),
        "EXCEPT with mismatched columns should error"
    );

    // Matching column counts should work
    let result = db.query("SELECT a FROM t104a UNION ALL SELECT x FROM t104b", ());
    assert!(
        result.is_ok(),
        "UNION with matching column counts should work"
    );
    let rows: Vec<_> = result.unwrap().collect();
    assert_eq!(rows.len(), 2, "Should have 2 rows from UNION");
}

// =============================================================================
// Bug #105: Window frame ROWS UNBOUNDED PRECEDING shorthand doesn't work
// Description: Shorthand syntax should be equivalent to BETWEEN ... AND CURRENT ROW
// =============================================================================
#[test]
fn test_bug_105_window_frame_shorthand() {
    let db = setup_db("bug105");

    db.execute(
        "CREATE TABLE t105 (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Create table");

    db.execute(
        "INSERT INTO t105 VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
        (),
    )
    .expect("Insert");

    // Test shorthand syntax: ROWS UNBOUNDED PRECEDING
    let result = db
        .query(
            "SELECT id, val, SUM(val) OVER (ORDER BY id ROWS UNBOUNDED PRECEDING) as running FROM t105",
            (),
        )
        .expect("Window frame shorthand should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 5);

    let expected_running = [10i64, 30, 60, 100, 150];
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_ref().unwrap();
        let running: i64 = row.get(2).unwrap();
        assert_eq!(
            running,
            expected_running[i],
            "Row {} running sum should be {}",
            i + 1,
            expected_running[i]
        );
    }

    // Verify it matches the explicit BETWEEN syntax
    let result = db
        .query(
            "SELECT id, SUM(val) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running FROM t105",
            (),
        )
        .expect("Full BETWEEN syntax should work");

    let rows: Vec<_> = result.collect();
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_ref().unwrap();
        let running: i64 = row.get(1).unwrap();
        assert_eq!(
            running,
            expected_running[i],
            "BETWEEN syntax row {} should match shorthand",
            i + 1
        );
    }
}

// =============================================================================
// Bug #106: Arithmetic between aggregates with expression arguments returns NULL
// Description: SUM(a) / SUM(a + b) should work correctly
// =============================================================================
#[test]
fn test_bug_106_aggregate_arithmetic_with_expressions() {
    let db = setup_db("bug106");

    db.execute("CREATE TABLE t106 (a INTEGER, b INTEGER)", ())
        .expect("Create table");

    db.execute("INSERT INTO t106 VALUES (10, 5), (20, 10), (30, 15)", ())
        .expect("Insert");

    // Test arithmetic between aggregates with expression arguments
    let result = db
        .query("SELECT SUM(a) / SUM(a + b) FROM t106", ())
        .expect("Aggregate arithmetic should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);
    let val: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    // SUM(a) = 60, SUM(a + b) = 90, 60 / 90 = 0 (integer division)
    assert_eq!(val, 0, "60 / 90 should be 0 (integer division)");

    // Test with float division (multiply by 1.0 to convert to float)
    let result = db
        .query("SELECT SUM(a) * 1.0 / SUM(a + b) FROM t106", ())
        .expect("Float division should work");

    let rows: Vec<_> = result.collect();
    let val: f64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert!(
        (val - 0.666666).abs() < 0.001,
        "60.0 / 90 should be ~0.666, got {}",
        val
    );

    // Test subtraction
    let result = db
        .query("SELECT SUM(a + b) - SUM(a) FROM t106", ())
        .expect("Aggregate subtraction should work");

    let rows: Vec<_> = result.collect();
    let val: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    // SUM(a + b) = 90, SUM(a) = 60, 90 - 60 = 30
    assert_eq!(val, 30, "90 - 60 should be 30");

    // Test multiplication
    let result = db
        .query("SELECT SUM(a) * SUM(b) / 100 FROM t106", ())
        .expect("Aggregate multiplication should work");

    let rows: Vec<_> = result.collect();
    let val: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    // SUM(a) = 60, SUM(b) = 30, 60 * 30 / 100 = 18
    assert_eq!(val, 18, "60 * 30 / 100 should be 18");
}

// =============================================================================
// Bug #107: Simple CASE with NULL incorrectly matches NULL values
// Description: CASE val WHEN NULL should never match (SQL standard)
// =============================================================================
#[test]
fn test_bug_107_case_null_matching() {
    let db = setup_db("bug107");

    db.execute(
        "CREATE TABLE t107 (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Create table");

    db.execute("INSERT INTO t107 VALUES (1, 10), (2, NULL), (3, 20)", ())
        .expect("Insert");

    // Simple CASE with NULL should never match
    let result = db
        .query(
            "SELECT id, CASE val WHEN NULL THEN 'matched' ELSE 'no' END as result FROM t107 ORDER BY id",
            (),
        )
        .expect("CASE should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // All rows should have 'no' because NULL = NULL is UNKNOWN, not TRUE
    for row in &rows {
        let row = row.as_ref().unwrap();
        let result: String = row.get(1).unwrap();
        assert_eq!(result, "no", "CASE val WHEN NULL should never match");
    }

    // Searched CASE with IS NULL should work correctly
    let result = db
        .query(
            "SELECT id, CASE WHEN val IS NULL THEN 'null' ELSE 'not null' END as result FROM t107 ORDER BY id",
            (),
        )
        .expect("Searched CASE should work");

    let rows: Vec<_> = result.collect();
    let row1: String = rows[0].as_ref().unwrap().get(1).unwrap();
    let row2: String = rows[1].as_ref().unwrap().get(1).unwrap();
    let row3: String = rows[2].as_ref().unwrap().get(1).unwrap();

    assert_eq!(row1, "not null", "id=1 should be not null");
    assert_eq!(row2, "null", "id=2 should be null");
    assert_eq!(row3, "not null", "id=3 should be not null");

    // Simple CASE with non-NULL values should still work
    let result = db
        .query(
            "SELECT id, CASE val WHEN 10 THEN 'ten' WHEN 20 THEN 'twenty' ELSE 'other' END as result FROM t107 ORDER BY id",
            (),
        )
        .expect("CASE with values should work");

    let rows: Vec<_> = result.collect();
    let row1: String = rows[0].as_ref().unwrap().get(1).unwrap();
    let row2: String = rows[1].as_ref().unwrap().get(1).unwrap();
    let row3: String = rows[2].as_ref().unwrap().get(1).unwrap();

    assert_eq!(row1, "ten", "id=1 (val=10) should be 'ten'");
    assert_eq!(row2, "other", "id=2 (val=NULL) should be 'other'");
    assert_eq!(row3, "twenty", "id=3 (val=20) should be 'twenty'");
}

// =============================================================================
// Bug #108: CAST(timestamp AS DATE) doesn't truncate time component
// Description: When casting TIMESTAMP to DATE, time should be set to midnight
// =============================================================================
#[test]
fn test_bug_108_cast_timestamp_to_date() {
    let db = setup_db("bug108");

    db.execute(
        "CREATE TABLE t108 (id INTEGER PRIMARY KEY, ts TIMESTAMP, val INTEGER)",
        (),
    )
    .expect("Create table");

    // Insert timestamps with different times on the same day
    db.execute("INSERT INTO t108 VALUES (1, '2024-01-15 10:00:00', 10)", ())
        .expect("Insert 1");
    db.execute("INSERT INTO t108 VALUES (2, '2024-01-15 14:30:00', 20)", ())
        .expect("Insert 2");
    db.execute("INSERT INTO t108 VALUES (3, '2024-01-16 08:00:00', 30)", ())
        .expect("Insert 3");

    // GROUP BY CAST(ts AS DATE) should group same-day timestamps together
    let result = db
        .query(
            "SELECT CAST(ts AS DATE) as day, SUM(val) as total FROM t108 GROUP BY CAST(ts AS DATE) ORDER BY day",
            (),
        )
        .expect("Query should succeed");

    let rows: Vec<_> = result.collect();

    // Should have 2 groups: 2024-01-15 and 2024-01-16
    assert_eq!(rows.len(), 2, "Should group same-day timestamps together");

    // First group (2024-01-15): 10 + 20 = 30
    let total1: i64 = rows[0].as_ref().unwrap().get(1).unwrap();
    assert_eq!(total1, 30, "Jan 15 total should be 30");

    // Second group (2024-01-16): 30
    let total2: i64 = rows[1].as_ref().unwrap().get(1).unwrap();
    assert_eq!(total2, 30, "Jan 16 total should be 30");
}

// =============================================================================
// Bug #109: COUNT(DISTINCT col) OVER() window function doesn't count distinct
// Description: COUNT(DISTINCT col) over window should count distinct values
// =============================================================================
#[test]
fn test_bug_109_count_distinct_over() {
    let db = setup_db("bug109");

    db.execute("CREATE TABLE t109 (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .expect("Create table");

    db.execute("INSERT INTO t109 VALUES (1, 10)", ())
        .expect("Insert 1");
    db.execute("INSERT INTO t109 VALUES (2, 10)", ())
        .expect("Insert 2");
    db.execute("INSERT INTO t109 VALUES (3, 20)", ())
        .expect("Insert 3");
    db.execute("INSERT INTO t109 VALUES (4, 20)", ())
        .expect("Insert 4");

    let result = db
        .query(
            "SELECT id, a, COUNT(DISTINCT a) OVER () as dist_count FROM t109 ORDER BY id",
            (),
        )
        .expect("COUNT DISTINCT OVER should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 4);

    // All rows should show 2 (distinct values: 10, 20)
    for row in &rows {
        let count: i64 = row.as_ref().unwrap().get(2).unwrap();
        assert_eq!(count, 2, "COUNT(DISTINCT a) should be 2");
    }
}

// =============================================================================
// Bug #110: Negative LIMIT returns all rows instead of error
// Description: LIMIT -1 should return an error
// =============================================================================
#[test]
fn test_bug_110_negative_limit() {
    let db = setup_db("bug110");

    db.execute("CREATE TABLE t110 (id INTEGER PRIMARY KEY)", ())
        .expect("Create table");

    db.execute("INSERT INTO t110 VALUES (1), (2), (3)", ())
        .expect("Insert");

    // Negative LIMIT should return an error
    let result = db.query("SELECT * FROM t110 LIMIT -1", ());
    match result {
        Ok(_) => panic!("Negative LIMIT should return an error"),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("non-negative") || err_msg.contains("LIMIT"),
                "Error should mention LIMIT must be non-negative: {}",
                err_msg
            );
        }
    }
}

// =============================================================================
// Bug #111: Negative OFFSET returns empty result instead of error
// Description: OFFSET -1 should return an error
// =============================================================================
#[test]
fn test_bug_111_negative_offset() {
    let db = setup_db("bug111");

    db.execute("CREATE TABLE t111 (id INTEGER PRIMARY KEY)", ())
        .expect("Create table");

    db.execute("INSERT INTO t111 VALUES (1), (2), (3)", ())
        .expect("Insert");

    // Negative OFFSET should return an error
    let result = db.query("SELECT * FROM t111 LIMIT 2 OFFSET -1", ());
    match result {
        Ok(_) => panic!("Negative OFFSET should return an error"),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("non-negative") || err_msg.contains("OFFSET"),
                "Error should mention OFFSET must be non-negative: {}",
                err_msg
            );
        }
    }
}

// =============================================================================
// Bug #112: TEXT PRIMARY KEY silently fails at CREATE, errors at INSERT
// Description: CREATE TABLE should error for non-INTEGER PRIMARY KEY
// =============================================================================
#[test]
fn test_bug_112_text_primary_key() {
    let db = setup_db("bug112");

    // TEXT PRIMARY KEY should fail at CREATE TABLE time
    let result = db.execute("CREATE TABLE t112 (id TEXT PRIMARY KEY, val INTEGER)", ());
    assert!(
        result.is_err(),
        "TEXT PRIMARY KEY should fail at CREATE TABLE time"
    );

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("INTEGER") || err_msg.contains("PRIMARY KEY"),
        "Error should mention PRIMARY KEY must be INTEGER: {}",
        err_msg
    );
}

// =============================================================================
// Bug #113: NOT BETWEEN with NULL incorrectly includes NULL row
// Description: NULL values should not satisfy NOT BETWEEN condition
// =============================================================================
#[test]
fn test_bug_113_not_between_with_null() {
    let db = setup_db("bug113");

    db.execute(
        "CREATE TABLE t113 (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Create table");

    db.execute("INSERT INTO t113 VALUES (1, 10)", ())
        .expect("Insert 1");
    db.execute("INSERT INTO t113 VALUES (2, NULL)", ())
        .expect("Insert 2");
    db.execute("INSERT INTO t113 VALUES (3, 30)", ())
        .expect("Insert 3");

    // NOT BETWEEN should not include NULL rows
    let result = db
        .query(
            "SELECT id, val FROM t113 WHERE val NOT BETWEEN 10 AND 20 ORDER BY id",
            (),
        )
        .expect("NOT BETWEEN should work");

    let rows: Vec<_> = result.collect();

    // Only id=3 (val=30) should match; id=2 (val=NULL) should NOT be included
    assert_eq!(
        rows.len(),
        1,
        "Only non-NULL values outside range should match"
    );

    let id: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(id, 3, "Only id=3 (val=30) should be returned");

    // Also test that BETWEEN correctly excludes NULL
    let result = db
        .query(
            "SELECT id FROM t113 WHERE val BETWEEN 5 AND 35 ORDER BY id",
            (),
        )
        .expect("BETWEEN should work");

    let rows: Vec<_> = result.collect();
    // Should return id=1 (val=10) and id=3 (val=30), but not id=2 (val=NULL)
    assert_eq!(rows.len(), 2, "BETWEEN should exclude NULL values");
}

// =============================================================================
// BUGS2.md - Additional Bug Regression Tests
// =============================================================================

// =============================================================================
// BUGS2 Bug #1: ORDER BY fails with ROLLUP/CUBE
// Description: Parser fails when ORDER BY clause is used with GROUP BY ROLLUP or CUBE
// =============================================================================
#[test]
fn test_bugs2_01_order_by_with_rollup() {
    let db = setup_db("bugs2_01");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, product TEXT, amount FLOAT)",
        (),
    )
    .expect("Create table");

    db.execute("INSERT INTO sales VALUES (1, 'North', 'A', 100)", ())
        .expect("Insert 1");
    db.execute("INSERT INTO sales VALUES (2, 'South', 'B', 200)", ())
        .expect("Insert 2");
    db.execute("INSERT INTO sales VALUES (3, 'North', 'B', 150)", ())
        .expect("Insert 3");

    // Test ROLLUP with ORDER BY
    let result = db
        .query(
            "SELECT region, SUM(amount) as total FROM sales GROUP BY ROLLUP(region) ORDER BY region",
            (),
        )
        .expect("ROLLUP with ORDER BY should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(
        rows.len(),
        3,
        "Should have 3 rows (2 regions + grand total)"
    );

    // Test CUBE with ORDER BY
    let result = db
        .query(
            "SELECT region, product, SUM(amount) as total FROM sales GROUP BY CUBE(region, product) ORDER BY region, product",
            (),
        )
        .expect("CUBE with ORDER BY should work");

    let rows: Vec<_> = result.collect();
    assert!(!rows.is_empty(), "CUBE with ORDER BY should return results");
}

// =============================================================================
// BUGS2 Bug #2: Correlated subqueries in UPDATE return NULL
// Description: UPDATE with correlated subqueries sets all values to NULL
// =============================================================================
#[test]
fn test_bugs2_02_correlated_subquery_in_update() {
    let db = setup_db("bugs2_02");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert 1");
    db.execute("INSERT INTO t VALUES (2, 20)", ())
        .expect("Insert 2");
    db.execute("INSERT INTO t VALUES (3, 30)", ())
        .expect("Insert 3");

    // Correlated subquery UPDATE - each row gets max of other rows' values
    db.execute(
        "UPDATE t SET val = (SELECT MAX(val) FROM t t2 WHERE t2.id != t.id)",
        (),
    )
    .expect("Correlated UPDATE should work");

    let result = db
        .query("SELECT id, val FROM t ORDER BY id", ())
        .expect("Select after update");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // id=1: max of (20, 30) = 30
    let val1: i64 = rows[0].as_ref().unwrap().get(1).unwrap();
    assert_eq!(val1, 30, "id=1 should have val=30 (max of 20, 30)");

    // id=2: max of (10, 30) = 30
    let val2: i64 = rows[1].as_ref().unwrap().get(1).unwrap();
    assert_eq!(val2, 30, "id=2 should have val=30 (max of 10, 30)");

    // id=3: max of (10, 20) = 20
    let val3: i64 = rows[2].as_ref().unwrap().get(1).unwrap();
    assert_eq!(val3, 20, "id=3 should have val=20 (max of 10, 20)");
}

// =============================================================================
// BUGS2 Bug #3: CASE WHEN with window function comparison returns NULL
// Description: Window function inside CASE WHEN condition returns NULL
// =============================================================================
#[test]
fn test_bugs2_03_case_when_with_window_function() {
    let db = setup_db("bugs2_03");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert 1");
    db.execute("INSERT INTO t VALUES (2, 20)", ())
        .expect("Insert 2");
    db.execute("INSERT INTO t VALUES (3, 30)", ())
        .expect("Insert 3");

    // CASE WHEN with window function comparison
    let result = db
        .query(
            "SELECT id, val, CASE WHEN val > AVG(val) OVER () THEN 'above' ELSE 'below' END as position FROM t ORDER BY id",
            (),
        )
        .expect("CASE with window function should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // AVG is 20, so val=10 and val=20 are 'below', val=30 is 'above'
    let pos1: String = rows[0].as_ref().unwrap().get(2).unwrap();
    assert_eq!(pos1, "below", "id=1 (val=10) should be 'below' (10 < 20)");

    let pos2: String = rows[1].as_ref().unwrap().get(2).unwrap();
    assert_eq!(
        pos2, "below",
        "id=2 (val=20) should be 'below' (20 = 20, not >)"
    );

    let pos3: String = rows[2].as_ref().unwrap().get(2).unwrap();
    assert_eq!(pos3, "above", "id=3 (val=30) should be 'above' (30 > 20)");
}

// =============================================================================
// BUGS2 Bug #4: ARRAY_AGG returns NULL
// Description: ARRAY_AGG aggregate function was not implemented
// =============================================================================
#[test]
fn test_bugs2_04_array_agg() {
    let db = setup_db("bugs2_04");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .expect("Create table");

    db.execute("INSERT INTO t VALUES (1, 'A', 10)", ())
        .expect("Insert 1");
    db.execute("INSERT INTO t VALUES (2, 'A', 20)", ())
        .expect("Insert 2");
    db.execute("INSERT INTO t VALUES (3, 'B', 30)", ())
        .expect("Insert 3");

    let result = db
        .query(
            "SELECT grp, ARRAY_AGG(val) as vals FROM t GROUP BY grp ORDER BY grp",
            (),
        )
        .expect("ARRAY_AGG should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);

    // Group A should have [10,20]
    let vals_a: String = rows[0].as_ref().unwrap().get(1).unwrap();
    assert!(
        vals_a.contains("10") && vals_a.contains("20"),
        "Group A should contain 10 and 20, got: {}",
        vals_a
    );

    // Group B should have [30]
    let vals_b: String = rows[1].as_ref().unwrap().get(1).unwrap();
    assert!(
        vals_b.contains("30"),
        "Group B should contain 30, got: {}",
        vals_b
    );
}

// =============================================================================
// BUGS2 Bug #5: RANGE frame specification doesn't work correctly
// Description: RANGE frame ignored bounds and returned entire partition sum
// =============================================================================
#[test]
fn test_bugs2_05_range_frame_specification() {
    let db = setup_db("bugs2_05");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Create table");

    db.execute("INSERT INTO t VALUES (1, 10)", ())
        .expect("Insert 1");
    db.execute("INSERT INTO t VALUES (2, 20)", ())
        .expect("Insert 2");
    db.execute("INSERT INTO t VALUES (3, 30)", ())
        .expect("Insert 3");
    db.execute("INSERT INTO t VALUES (4, 40)", ())
        .expect("Insert 4");
    db.execute("INSERT INTO t VALUES (5, 50)", ())
        .expect("Insert 5");

    let result = db
        .query(
            "SELECT id, val, SUM(val) OVER (ORDER BY val RANGE BETWEEN 15 PRECEDING AND 15 FOLLOWING) as range_sum FROM t ORDER BY id",
            (),
        )
        .expect("RANGE frame should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 5);

    // id=1 (val=10): range [-5, 25] -> sum(10, 20) = 30
    let sum1: i64 = rows[0].as_ref().unwrap().get(2).unwrap();
    assert_eq!(sum1, 30, "id=1: range [-5, 25] should sum to 30 (10+20)");

    // id=2 (val=20): range [5, 35] -> sum(10, 20, 30) = 60
    let sum2: i64 = rows[1].as_ref().unwrap().get(2).unwrap();
    assert_eq!(sum2, 60, "id=2: range [5, 35] should sum to 60 (10+20+30)");

    // id=3 (val=30): range [15, 45] -> sum(20, 30, 40) = 90
    let sum3: i64 = rows[2].as_ref().unwrap().get(2).unwrap();
    assert_eq!(sum3, 90, "id=3: range [15, 45] should sum to 90 (20+30+40)");

    // id=4 (val=40): range [25, 55] -> sum(30, 40, 50) = 120
    let sum4: i64 = rows[3].as_ref().unwrap().get(2).unwrap();
    assert_eq!(
        sum4, 120,
        "id=4: range [25, 55] should sum to 120 (30+40+50)"
    );

    // id=5 (val=50): range [35, 65] -> sum(40, 50) = 90
    let sum5: i64 = rows[4].as_ref().unwrap().get(2).unwrap();
    assert_eq!(sum5, 90, "id=5: range [35, 65] should sum to 90 (40+50)");
}

// =============================================================================
// BUGS2 Bug #6: CTE with column list syntax not supported
// Description: CTEs with column list syntax should work
// =============================================================================
#[test]
fn test_bugs2_06_cte_with_column_list() {
    let db = setup_db("bugs2_06");

    // Basic CTE with column list
    let result = db
        .query(
            "WITH cte(col1, col2) AS (SELECT 1, 2) SELECT * FROM cte",
            (),
        )
        .expect("CTE with column list should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 1);

    let col1: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    let col2: i64 = rows[0].as_ref().unwrap().get(1).unwrap();
    assert_eq!(col1, 1);
    assert_eq!(col2, 2);

    // Recursive CTE with column list
    let result = db
        .query(
            "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 5) SELECT * FROM cnt",
            (),
        )
        .expect("Recursive CTE with column list should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 5, "Recursive CTE should generate 5 rows");

    // Verify values 1 through 5
    for (i, row) in rows.iter().enumerate() {
        let x: i64 = row.as_ref().unwrap().get(0).unwrap();
        assert_eq!(x, (i + 1) as i64, "Row {} should have x={}", i, i + 1);
    }
}

// =============================================================================
// BUGS2 Bug #7: Window function with expression argument returns wrong value
// Description: SUM(val * 2) OVER () returns count instead of sum
// =============================================================================
#[test]
fn test_bugs2_07_window_expression_argument() {
    let db = setup_db("bugs2_07");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Create failed");
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    // Test SUM(val * 2) with ORDER BY (running sum)
    let result = db
        .query(
            "SELECT id, SUM(val * 2) OVER (ORDER BY id) as running FROM t ORDER BY id",
            (),
        )
        .expect("Window with expression should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // Expected running sums: 20, 60, 120
    let running1: i64 = rows[0].as_ref().unwrap().get(1).unwrap();
    let running2: i64 = rows[1].as_ref().unwrap().get(1).unwrap();
    let running3: i64 = rows[2].as_ref().unwrap().get(1).unwrap();
    assert_eq!(running1, 20, "First running sum of val*2 should be 20");
    assert_eq!(running2, 60, "Second running sum of val*2 should be 60");
    assert_eq!(running3, 120, "Third running sum of val*2 should be 120");

    // Test SUM(val + 100) without ORDER BY (total)
    let result = db
        .query("SELECT SUM(val + 100) OVER () as total FROM t LIMIT 1", ())
        .expect("Window with addition expression should work");

    let rows: Vec<_> = result.collect();
    let total: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(total, 360, "SUM(val + 100) should be 360 (110+120+130)");
}

// =============================================================================
// BUGS2 Bug #8: Window PARTITION BY ignores partition after JOIN
// Description: PARTITION BY l.grp treats all rows as one partition after JOIN
// =============================================================================
#[test]
fn test_bugs2_08_window_partition_after_join() {
    let db = setup_db("bugs2_08");

    db.execute(
        "CREATE TABLE left_t (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .expect("Create left failed");
    db.execute(
        "CREATE TABLE right_t (id INTEGER PRIMARY KEY, left_id INTEGER)",
        (),
    )
    .expect("Create right failed");

    db.execute(
        "INSERT INTO left_t VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 30)",
        (),
    )
    .expect("Insert left failed");
    db.execute("INSERT INTO right_t VALUES (1, 1), (2, 2), (3, 3)", ())
        .expect("Insert right failed");

    let result = db
        .query(
            "SELECT l.id, l.grp, l.val, SUM(l.val) OVER (PARTITION BY l.grp) as sum_grp \
             FROM left_t l JOIN right_t r ON l.id = r.left_id ORDER BY l.id",
            (),
        )
        .expect("Window PARTITION BY after JOIN should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // Group A (ids 1, 2): sum should be 30 (10+20)
    // Group B (id 3): sum should be 30
    let sum1: i64 = rows[0].as_ref().unwrap().get(3).unwrap();
    let sum2: i64 = rows[1].as_ref().unwrap().get(3).unwrap();
    let sum3: i64 = rows[2].as_ref().unwrap().get(3).unwrap();

    assert_eq!(sum1, 30, "Group A sum should be 30");
    assert_eq!(sum2, 30, "Group A sum should be 30");
    assert_eq!(sum3, 30, "Group B sum should be 30");
}

// =============================================================================
// BUGS2 Bug #9: Nested aggregate in window function returns wrong value
// Description: SUM(SUM(val)) OVER () returns group count instead of sum
// =============================================================================
#[test]
fn test_bugs2_09_nested_aggregate_window() {
    let db = setup_db("bugs2_09");

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .expect("Create failed");
    db.execute(
        "INSERT INTO t VALUES (1, 'X', 100), (2, 'X', 200), (3, 'Y', 300), (4, 'Y', 400)",
        (),
    )
    .expect("Insert failed");

    // Test SUM(SUM(val)) OVER ()
    let result = db
        .query(
            "SELECT grp, SUM(val) as grp_sum, SUM(SUM(val)) OVER () as total \
             FROM t GROUP BY grp ORDER BY grp",
            (),
        )
        .expect("Nested aggregate in window should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2);

    // X: grp_sum=300, total=1000
    // Y: grp_sum=700, total=1000
    let grp_sum_x: i64 = rows[0].as_ref().unwrap().get(1).unwrap();
    let total_x: i64 = rows[0].as_ref().unwrap().get(2).unwrap();
    let grp_sum_y: i64 = rows[1].as_ref().unwrap().get(1).unwrap();
    let total_y: i64 = rows[1].as_ref().unwrap().get(2).unwrap();

    assert_eq!(grp_sum_x, 300, "X group sum should be 300");
    assert_eq!(grp_sum_y, 700, "Y group sum should be 700");
    assert_eq!(total_x, 1000, "Total of group sums should be 1000");
    assert_eq!(total_y, 1000, "Total of group sums should be 1000");

    // Test MAX(SUM(val)) OVER () - need to include the aliased aggregate for mapping
    let result = db
        .query(
            "SELECT grp, SUM(val) as grp_sum, MAX(SUM(val)) OVER () as max_of_sums \
             FROM t GROUP BY grp ORDER BY grp LIMIT 1",
            (),
        )
        .expect("MAX of SUM in window should work");

    let rows: Vec<_> = result.collect();
    let max: i64 = rows[0].as_ref().unwrap().get(2).unwrap();
    assert_eq!(max, 700, "MAX of sums should be 700");
}

// =============================================================================
// BUGS2.md Bug #10: Table-level constraint syntax not supported
// Description: Table-level UNIQUE(col) and CHECK(expr) constraints failed to parse
// =============================================================================
#[test]
fn test_bugs2_10_table_level_unique_constraint() {
    let db = setup_db("bugs2_10");

    // Test table-level UNIQUE constraint
    db.execute(
        "CREATE TABLE t1 (id INTEGER PRIMARY KEY, code TEXT, name TEXT, UNIQUE(code))",
        (),
    )
    .expect("Table-level UNIQUE should be parsed");

    db.execute("INSERT INTO t1 VALUES (1, 'A', 'Alice')", ())
        .expect("First insert should succeed");
    db.execute("INSERT INTO t1 VALUES (2, 'B', 'Bob')", ())
        .expect("Second insert should succeed");

    // This should fail due to unique constraint
    let result = db.execute("INSERT INTO t1 VALUES (3, 'A', 'Charlie')", ());
    assert!(
        result.is_err(),
        "Duplicate code should violate unique constraint"
    );

    // Verify data
    let result = db
        .query("SELECT * FROM t1 ORDER BY id", ())
        .expect("Query should succeed");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 2, "Should have exactly 2 rows");
}

// =============================================================================
// BUGS2.md Bug #11: CAST of aggregate function fails
// Description: CAST(SUM(val) AS TEXT) returned "Unknown function: SUM"
// =============================================================================
#[test]
fn test_bugs2_11_cast_of_aggregate() {
    let db = setup_db("bugs2_11");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Create failed");
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Insert failed");

    // Test CAST(SUM(val) AS TEXT)
    let result = db
        .query("SELECT CAST(SUM(val) AS TEXT) as sum_text FROM t", ())
        .expect("CAST of SUM should work");
    let rows: Vec<_> = result.collect();
    let sum_text: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(sum_text, "60", "SUM should be cast to text '60'");

    // Test CAST(AVG(val) AS INTEGER)
    let result = db
        .query("SELECT CAST(AVG(val) AS INTEGER) as avg_int FROM t", ())
        .expect("CAST of AVG should work");
    let rows: Vec<_> = result.collect();
    let avg_int: i64 = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(avg_int, 20, "AVG should be cast to integer 20");

    // Test CAST(COUNT(*) AS TEXT)
    let result = db
        .query("SELECT CAST(COUNT(*) AS TEXT) as cnt_text FROM t", ())
        .expect("CAST of COUNT should work");
    let rows: Vec<_> = result.collect();
    let cnt_text: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(cnt_text, "3", "COUNT should be cast to text '3'");
}

// =============================================================================
// BUGS2.md Bug #12: ORDER BY expression doesn't work when aliased in SELECT
// Description: ORDER BY val * 2 didn't sort correctly when val * 2 as doubled was in SELECT
// =============================================================================
#[test]
fn test_bugs2_12_order_by_expression_with_alias() {
    let db = setup_db("bugs2_12");

    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .expect("Create failed");
    db.execute("INSERT INTO t VALUES (1, 30), (2, 10), (3, 20)", ())
        .expect("Insert failed");

    // Test ORDER BY alias
    let result = db
        .query("SELECT id, val * 2 as doubled FROM t ORDER BY doubled", ())
        .expect("ORDER BY alias should work");
    let rows: Vec<_> = result.collect();
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![2, 3, 1], "ORDER BY doubled: should be 2,3,1");

    // Test ORDER BY same expression (should match alias and use same column)
    let result = db
        .query("SELECT id, val * 2 as doubled FROM t ORDER BY val * 2", ())
        .expect("ORDER BY expression should work");
    let rows: Vec<_> = result.collect();
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![2, 3, 1], "ORDER BY val * 2: should also be 2,3,1");

    // Test ORDER BY expression DESC
    let result = db
        .query(
            "SELECT id, val * 2 as doubled FROM t ORDER BY val * 2 DESC",
            (),
        )
        .expect("ORDER BY expression DESC should work");
    let rows: Vec<_> = result.collect();
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| r.as_ref().unwrap().get(0).unwrap())
        .collect();
    assert_eq!(ids, vec![1, 3, 2], "ORDER BY val * 2 DESC: should be 1,3,2");
}

// =============================================================================
// BUGS2.md Bug #13: Window function fails with CTE + JOIN combination
// Description: Using window function with CTE and JOIN gave "Window expression in evaluator not yet implemented"
// =============================================================================
#[test]
fn test_bugs2_13_window_with_cte_and_join() {
    let db = setup_db("bugs2_13");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, price FLOAT)",
        (),
    )
    .expect("Create failed");
    db.execute(
        "INSERT INTO products VALUES (1, 'A', 10.0), (2, 'A', 20.0), (3, 'B', 30.0)",
        (),
    )
    .expect("Insert failed");

    // CTE + JOIN without window function should work
    let result = db
        .query(
            "WITH cat_avg AS (SELECT category, AVG(price) as avg FROM products GROUP BY category) \
             SELECT p.category, p.price, c.avg \
             FROM products p JOIN cat_avg c ON p.category = c.category \
             ORDER BY p.id",
            (),
        )
        .expect("CTE + JOIN should work");
    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3);

    // CTE + JOIN + window function should now work
    let result = db
        .query(
            "WITH cat_avg AS (SELECT category, AVG(price) as avg FROM products GROUP BY category) \
             SELECT p.category, p.price, \
                    RANK() OVER (PARTITION BY p.category ORDER BY p.price) as rnk \
             FROM products p JOIN cat_avg c ON p.category = c.category \
             ORDER BY p.category, rnk",
            (),
        )
        .expect("CTE + JOIN + window function should work");

    let rows: Vec<_> = result.collect();
    assert_eq!(rows.len(), 3, "Should have 3 rows");

    // Check ranks: A has prices 10 (rank 1), 20 (rank 2); B has price 30 (rank 1)
    let rank1: i64 = rows[0].as_ref().unwrap().get(2).unwrap();
    let rank2: i64 = rows[1].as_ref().unwrap().get(2).unwrap();
    let rank3: i64 = rows[2].as_ref().unwrap().get(2).unwrap();

    assert_eq!(rank1, 1, "First A product should have rank 1");
    assert_eq!(rank2, 2, "Second A product should have rank 2");
    assert_eq!(rank3, 1, "B product should have rank 1");
}

// =============================================================================
// Test for reserved keyword error messages
// Description: Using reserved keywords as identifiers should give clear error messages
// =============================================================================
#[test]
fn test_reserved_keyword_error_message() {
    let db = setup_db("reserved_keyword");

    // Using reserved keyword 'key' as column name should fail with clear message
    let result = db.execute("CREATE TABLE t (key TEXT PRIMARY KEY, val INTEGER)", ());
    assert!(result.is_err(), "Using reserved keyword should fail");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("reserved keyword") || err_msg.contains("KEY"),
        "Error should mention reserved keyword: {}",
        err_msg
    );

    // Using double quotes to escape should work
    db.execute("CREATE TABLE t (\"key\" TEXT, id INTEGER PRIMARY KEY)", ())
        .expect("Quoted reserved keyword should work");

    db.execute("INSERT INTO t VALUES ('test', 1)", ())
        .expect("Insert should work");

    let result = db
        .query("SELECT \"key\" FROM t", ())
        .expect("Query with quoted identifier should work");
    let rows: Vec<_> = result.collect();
    let key_val: String = rows[0].as_ref().unwrap().get(0).unwrap();
    assert_eq!(key_val, "test");
}
