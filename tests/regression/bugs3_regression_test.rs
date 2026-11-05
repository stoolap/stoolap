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

// Regression tests for Bug Batch 3
// Tests for bugs discovered during exploratory testing

use stoolap::Database;

fn setup_db() -> Database {
    Database::open_in_memory().expect("Failed to create in-memory database")
}

// =============================================================================
// BUG 1: Window ORDER BY with COALESCE on aggregate
// Problem: RANK() OVER (ORDER BY COALESCE(SUM(val), 0)) returned all rank 1
// =============================================================================

#[test]
fn test_bugs3_window_coalesce_aggregate() {
    let db = setup_db();

    // Setup
    db.execute(
        "CREATE TABLE window_coalesce_test (grp TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO window_coalesce_test VALUES ('A', 10), ('B', 30), ('C', 20)",
        (),
    )
    .expect("Failed to insert data");

    // Test RANK with COALESCE on aggregate in ORDER BY
    let mut rows = db
        .query(
            "SELECT grp, COALESCE(SUM(val), 0) as total, \
             RANK() OVER (ORDER BY COALESCE(SUM(val), 0) DESC) as rnk \
             FROM window_coalesce_test GROUP BY grp ORDER BY rnk",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(1).unwrap(), 30);
    assert_eq!(row.get::<i64>(2).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "C");
    assert_eq!(row.get::<i64>(1).unwrap(), 20);
    assert_eq!(row.get::<i64>(2).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 10);
    assert_eq!(row.get::<i64>(2).unwrap(), 3);
}

#[test]
fn test_bugs3_window_dense_rank_coalesce() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE dense_rank_test (grp TEXT, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute(
        "INSERT INTO dense_rank_test VALUES ('A', 10), ('B', 20), ('C', 20)",
        (),
    )
    .expect("Failed to insert data");

    // Test DENSE_RANK with COALESCE
    let mut rows = db
        .query(
            "SELECT grp, COALESCE(SUM(val), 0) as total, \
             DENSE_RANK() OVER (ORDER BY COALESCE(SUM(val), 0) DESC) as rnk \
             FROM dense_rank_test GROUP BY grp ORDER BY grp",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(2).unwrap(), 2); // A=10 gets rank 2

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(2).unwrap(), 1); // B=20 gets rank 1

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "C");
    assert_eq!(row.get::<i64>(2).unwrap(), 1); // C=20 gets rank 1
}

#[test]
fn test_bugs3_window_row_number_coalesce() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE row_num_test (grp TEXT, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute(
        "INSERT INTO row_num_test VALUES ('A', 10), ('B', 30), ('C', 20)",
        (),
    )
    .expect("Failed to insert data");

    // Test ROW_NUMBER with COALESCE
    let mut rows = db
        .query(
            "SELECT grp, COALESCE(SUM(val), 0) as total, \
             ROW_NUMBER() OVER (ORDER BY COALESCE(SUM(val), 0) DESC) as rn \
             FROM row_num_test GROUP BY grp ORDER BY rn",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(2).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "C");
    assert_eq!(row.get::<i64>(2).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(2).unwrap(), 3);
}

// =============================================================================
// BUG 2: JOIN between subqueries ignores ON condition
// Problem: SELECT * FROM (SELECT..) a JOIN (SELECT..) b ON a.id = b.id
//          produced a CROSS JOIN instead of applying ON condition
// =============================================================================

#[test]
fn test_bugs3_subquery_join_on_condition() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE subq_a (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("CREATE TABLE subq_b (id INTEGER, data TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO subq_a VALUES (1, 'a'), (2, 'b')", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO subq_b VALUES (1, 'x'), (3, 'y')", ())
        .expect("Failed to insert data");

    // Test JOIN between subqueries with ON condition
    let mut rows = db
        .query(
            "SELECT * FROM (SELECT * FROM subq_a) AS a \
             JOIN (SELECT * FROM subq_b) AS b ON a.id = b.id",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1); // a.id
    assert_eq!(row.get::<String>(1).unwrap(), "a"); // a.val
    assert_eq!(row.get::<i64>(2).unwrap(), 1); // b.id
    assert_eq!(row.get::<String>(3).unwrap(), "x"); // b.data

    // Should only have 1 row where id=1 matches
    assert!(
        rows.next().is_none(),
        "Should have only 1 row (only id=1 matches)"
    );
}

#[test]
fn test_bugs3_subquery_left_join() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE left_a (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("CREATE TABLE left_b (id INTEGER, data TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO left_a VALUES (1, 'a'), (2, 'b')", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO left_b VALUES (1, 'x'), (3, 'y')", ())
        .expect("Failed to insert data");

    // Test LEFT JOIN between subqueries
    let mut rows = db
        .query(
            "SELECT * FROM (SELECT * FROM left_a) AS a \
             LEFT JOIN (SELECT * FROM left_b) AS b ON a.id = b.id \
             ORDER BY a.id",
            (),
        )
        .expect("Query failed");

    // First row: id=1 matched
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<String>(1).unwrap(), "a");
    assert_eq!(row.get::<i64>(2).unwrap(), 1);
    assert_eq!(row.get::<String>(3).unwrap(), "x");

    // Second row: id=2 unmatched (right side should be NULL)
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<String>(1).unwrap(), "b");
    // Check that b.id is NULL
    let value = row.get_value(2);
    assert!(
        value.map(|v| v.is_null()).unwrap_or(true),
        "b.id should be NULL for unmatched row"
    );

    assert!(rows.next().is_none(), "Should have only 2 rows");
}

#[test]
fn test_bugs3_regular_table_join_still_works() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE reg_a (id INTEGER, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("CREATE TABLE reg_b (id INTEGER, data TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO reg_a VALUES (1, 'a'), (2, 'b')", ())
        .expect("Failed to insert data");
    db.execute("INSERT INTO reg_b VALUES (1, 'x'), (3, 'y')", ())
        .expect("Failed to insert data");

    // Regression test: regular table JOIN should still work
    let mut rows = db
        .query("SELECT * FROM reg_a a JOIN reg_b b ON a.id = b.id", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);

    assert!(rows.next().is_none(), "Should have only 1 row");
}

// =============================================================================
// BUG 3: INSERT RETURNING doesn't show AUTO_INCREMENT values
// Problem: INSERT INTO t (name) VALUES ('x') RETURNING * showed NULL for id
// =============================================================================

#[test]
fn test_bugs3_insert_returning_auto_increment() {
    let db = setup_db();

    // Setup
    db.execute(
        "CREATE TABLE auto_ret_test (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Test INSERT RETURNING with AUTO_INCREMENT
    let mut rows = db
        .query(
            "INSERT INTO auto_ret_test (name) VALUES ('test1') RETURNING *",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    // The AUTO_INCREMENT id should be 1, not NULL
    let id = row.get::<i64>(0).unwrap();
    assert_eq!(id, 1, "AUTO_INCREMENT id should be 1, got {}", id);
    assert_eq!(row.get::<String>(1).unwrap(), "test1");

    assert!(rows.next().is_none(), "Should return only 1 row");
}

#[test]
fn test_bugs3_insert_returning_auto_increment_multiple() {
    let db = setup_db();

    // Setup
    db.execute(
        "CREATE TABLE auto_multi_test (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert multiple rows and check RETURNING
    let mut rows1 = db
        .query(
            "INSERT INTO auto_multi_test (name) VALUES ('first') RETURNING id",
            (),
        )
        .expect("Query failed");
    let row = rows1.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);

    let mut rows2 = db
        .query(
            "INSERT INTO auto_multi_test (name) VALUES ('second') RETURNING id",
            (),
        )
        .expect("Query failed");
    let row = rows2.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);

    let mut rows3 = db
        .query(
            "INSERT INTO auto_multi_test (name) VALUES ('third') RETURNING id",
            (),
        )
        .expect("Query failed");
    let row = rows3.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
}

#[test]
fn test_bugs3_insert_returning_specific_columns() {
    let db = setup_db();

    // Setup
    db.execute(
        "CREATE TABLE auto_cols_test (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT, age INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Test RETURNING specific columns including AUTO_INCREMENT
    let mut rows = db
        .query(
            "INSERT INTO auto_cols_test (name, age) VALUES ('Alice', 30) RETURNING id, name",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1); // id should be 1
    assert_eq!(row.get::<String>(1).unwrap(), "Alice");
}

#[test]
fn test_bugs3_insert_returning_with_explicit_id() {
    let db = setup_db();

    // Setup
    db.execute(
        "CREATE TABLE auto_explicit_test (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    // Insert with explicit ID
    let mut rows = db
        .query(
            "INSERT INTO auto_explicit_test (id, name) VALUES (100, 'explicit') RETURNING *",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 100); // explicit ID should be preserved

    // Next auto-increment should be 101
    let mut rows2 = db
        .query(
            "INSERT INTO auto_explicit_test (name) VALUES ('next') RETURNING id",
            (),
        )
        .expect("Query failed");
    let row = rows2.next().unwrap().unwrap();
    assert_eq!(
        row.get::<i64>(0).unwrap(),
        101,
        "Next AUTO_INCREMENT should be 101 after explicit 100"
    );
}

// =============================================================================
// BUG 4: WHERE clause with expression compared to column fails
// Problem: WHERE a * 2 = b returned no rows when expression equals column value
// Root cause: evaluate_literal_with_ctx silently returned NULL for expressions
//             containing column references instead of returning an error
// =============================================================================

#[test]
fn test_bugs3_where_expression_equals_column() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE expr_cmp (a INTEGER, b INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO expr_cmp VALUES (2, 4), (3, 6), (5, 10)", ())
        .expect("Failed to insert data");

    // Test: expression = column
    let mut rows = db
        .query("SELECT * FROM expr_cmp WHERE a * 2 = b ORDER BY a", ())
        .expect("Query failed");

    // All rows should match since 2*2=4, 3*2=6, 5*2=10
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(1).unwrap(), 4);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<i64>(1).unwrap(), 6);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);
    assert_eq!(row.get::<i64>(1).unwrap(), 10);

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs3_where_column_equals_expression() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE expr_cmp2 (a INTEGER, b INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO expr_cmp2 VALUES (2, 4), (3, 6), (5, 10)", ())
        .expect("Failed to insert data");

    // Test: column = expression
    let mut rows = db
        .query("SELECT * FROM expr_cmp2 WHERE a = b / 2 ORDER BY a", ())
        .expect("Query failed");

    // All rows should match since 4/2=2, 6/2=3, 10/2=5
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs3_where_expression_greater_than_column() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE expr_cmp3 (a INTEGER, b INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO expr_cmp3 VALUES (2, 4), (3, 6), (5, 10)", ())
        .expect("Failed to insert data");

    // Test: expression > column
    let mut rows = db
        .query("SELECT * FROM expr_cmp3 WHERE a * 3 > b ORDER BY a", ())
        .expect("Query failed");

    // All rows should match: 2*3=6>4, 3*3=9>6, 5*3=15>10
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs3_where_both_sides_expressions() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE expr_cmp4 (a INTEGER, b INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO expr_cmp4 VALUES (2, 4), (3, 7), (5, 10)", ())
        .expect("Failed to insert data");

    // Test: expression = expression (this already worked, regression test)
    let mut rows = db
        .query("SELECT * FROM expr_cmp4 WHERE a * 2 = b * 1 ORDER BY a", ())
        .expect("Query failed");

    // Only (2,4) and (5,10) match: 2*2=4*1, 5*2=10*1
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs3_where_expression_minus_column_equals_zero() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE expr_cmp5 (a INTEGER, b INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO expr_cmp5 VALUES (2, 4), (3, 6), (5, 10)", ())
        .expect("Failed to insert data");

    // Test: expression - column = literal (this already worked, regression test)
    let mut rows = db
        .query("SELECT * FROM expr_cmp5 WHERE a * 2 - b = 0 ORDER BY a", ())
        .expect("Query failed");

    // All rows should match
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs3_delete_with_expression_comparison() {
    let db = setup_db();

    // Setup WITH primary key (required for DELETE with complex WHERE)
    db.execute(
        "CREATE TABLE delete_expr (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO delete_expr VALUES (1, 1, 2), (2, 2, 4), (3, 3, 5)",
        (),
    )
    .expect("Failed to insert data");

    // Delete rows where a * 2 = b (should delete id=1 and id=2)
    db.execute("DELETE FROM delete_expr WHERE a * 2 = b", ())
        .expect("Delete failed");

    // Only id=3 should remain
    let mut rows = db
        .query("SELECT * FROM delete_expr ORDER BY id", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<i64>(1).unwrap(), 3);
    assert_eq!(row.get::<i64>(2).unwrap(), 5);

    assert!(rows.next().is_none(), "Should have only 1 row after delete");
}

#[test]
fn test_bugs3_update_with_expression_comparison() {
    let db = setup_db();

    // Setup
    db.execute("CREATE TABLE update_expr (a INTEGER, b INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO update_expr VALUES (2, 4), (3, 6), (5, 10)", ())
        .expect("Failed to insert data");

    // Update rows where a * 2 = b (all rows should match)
    db.execute("UPDATE update_expr SET b = b + 1 WHERE a * 2 = b", ())
        .expect("Update failed");

    // All b values should be incremented by 1
    let mut rows = db
        .query("SELECT * FROM update_expr ORDER BY a", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(1).unwrap(), 5); // was 4, now 5

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<i64>(1).unwrap(), 7); // was 6, now 7

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);
    assert_eq!(row.get::<i64>(1).unwrap(), 11); // was 10, now 11

    assert!(rows.next().is_none());
}
