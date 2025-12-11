// Regression tests for Bug Batch 4
// Tests for bugs discovered during exploratory testing session 2

use stoolap::Database;

fn setup_db() -> Database {
    Database::open_in_memory().expect("Failed to create in-memory database")
}

// =============================================================================
// BUG 1: ORDER BY on CTE (WITH clause) was completely ignored
// Problem: SELECT * FROM cte ORDER BY val returned rows in original order
// =============================================================================

#[test]
fn test_bugs4_cte_order_by_simple() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE test_order (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO test_order VALUES (1, 30), (2, 10), (3, 20)",
        (),
    )
    .expect("Failed to insert data");

    // CTE with ORDER BY should now work
    let mut rows = db
        .query(
            "WITH data AS (SELECT id, val FROM test_order) \
             SELECT * FROM data ORDER BY val ASC",
            (),
        )
        .expect("Query failed");

    // Should be sorted by val: 10, 20, 30
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2); // id=2 has val=10
    assert_eq!(row.get::<i64>(1).unwrap(), 10);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3); // id=3 has val=20
    assert_eq!(row.get::<i64>(1).unwrap(), 20);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1); // id=1 has val=30
    assert_eq!(row.get::<i64>(1).unwrap(), 30);
}

#[test]
fn test_bugs4_cte_order_by_desc() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE test_desc (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO test_desc VALUES (1, 30), (2, 10), (3, 20)", ())
        .expect("Failed to insert data");

    // CTE with ORDER BY DESC
    let mut rows = db
        .query(
            "WITH data AS (SELECT id, val FROM test_desc) \
             SELECT * FROM data ORDER BY val DESC",
            (),
        )
        .expect("Query failed");

    // Should be sorted by val DESC: 30, 20, 10
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 30);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 20);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 10);
}

#[test]
fn test_bugs4_cte_with_window_function_order_by() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, product TEXT, qty INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO orders VALUES (1, 'A', 10), (2, 'B', 20), (3, 'A', 15), (4, 'C', 5)",
        (),
    )
    .expect("Failed to insert data");

    // Multiple CTEs with window function and ORDER BY
    let mut rows = db
        .query(
            "WITH total_by_product AS ( \
                 SELECT product, SUM(qty) as total FROM orders GROUP BY product \
             ), \
             ranked AS ( \
                 SELECT product, total, RANK() OVER (ORDER BY total DESC) as rnk \
                 FROM total_by_product \
             ) \
             SELECT * FROM ranked ORDER BY rnk",
            (),
        )
        .expect("Query failed");

    // A=25, B=20, C=5 - should be ordered by rank
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 25);
    assert_eq!(row.get::<i64>(2).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(1).unwrap(), 20);
    assert_eq!(row.get::<i64>(2).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "C");
    assert_eq!(row.get::<i64>(1).unwrap(), 5);
    assert_eq!(row.get::<i64>(2).unwrap(), 3);
}

// =============================================================================
// BUG 2: GROUP_CONCAT ORDER BY didn't respect ordering
// Problem: GROUP_CONCAT(val ORDER BY val) returned values in insertion order
// =============================================================================

#[test]
fn test_bugs4_group_concat_order_by_asc() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE gc_test (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO gc_test VALUES (1, 'c'), (2, 'a'), (3, 'b')",
        (),
    )
    .expect("Failed to insert data");

    let mut rows = db
        .query(
            "SELECT GROUP_CONCAT(val ORDER BY val) as sorted FROM gc_test",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "a,b,c");
}

#[test]
fn test_bugs4_group_concat_order_by_desc() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE gc_desc (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO gc_desc VALUES (1, 'c'), (2, 'a'), (3, 'b')",
        (),
    )
    .expect("Failed to insert data");

    let mut rows = db
        .query(
            "SELECT GROUP_CONCAT(val ORDER BY val DESC) as reverse_sorted FROM gc_desc",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "c,b,a");
}

// =============================================================================
// BUG 3: STRING_AGG ORDER BY didn't respect ordering
// Problem: STRING_AGG(val, '-' ORDER BY val) returned values in insertion order
// =============================================================================

#[test]
fn test_bugs4_string_agg_order_by_asc() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE sa_test (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO sa_test VALUES (1, 'c'), (2, 'a'), (3, 'b')",
        (),
    )
    .expect("Failed to insert data");

    let mut rows = db
        .query(
            "SELECT STRING_AGG(val, '-' ORDER BY val) as sorted FROM sa_test",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "a-b-c");
}

#[test]
fn test_bugs4_string_agg_order_by_desc() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE sa_desc (id INTEGER PRIMARY KEY, val TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO sa_desc VALUES (1, 'c'), (2, 'a'), (3, 'b')",
        (),
    )
    .expect("Failed to insert data");

    let mut rows = db
        .query(
            "SELECT STRING_AGG(val, '-' ORDER BY val DESC) as reverse_sorted FROM sa_desc",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "c-b-a");
}

// =============================================================================
// BUG 4: Window navigation functions (NTH_VALUE, FIRST_VALUE, LAST_VALUE)
//        ignored bounded frame clauses
// Problem: FIRST_VALUE with ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
//          returned the first value of the entire partition
// =============================================================================

#[test]
fn test_bugs4_nth_value_with_frame() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE nth_test (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO nth_test VALUES (1, 100), (2, 200), (3, 300), (4, 400), (5, 500)",
        (),
    )
    .expect("Failed to insert data");

    // Get 2nd value within a 3-row sliding window
    let mut rows = db
        .query(
            "SELECT id, val, \
             NTH_VALUE(val, 2) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as nth2 \
             FROM nth_test ORDER BY id",
            (),
        )
        .expect("Query failed");

    // Row 1: frame [100, 200], 2nd = 200
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(2).unwrap(), 200);

    // Row 2: frame [100, 200, 300], 2nd = 200
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(2).unwrap(), 200);

    // Row 3: frame [200, 300, 400], 2nd = 300
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<i64>(2).unwrap(), 300);

    // Row 4: frame [300, 400, 500], 2nd = 400
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 4);
    assert_eq!(row.get::<i64>(2).unwrap(), 400);

    // Row 5: frame [400, 500], 2nd = 500
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);
    assert_eq!(row.get::<i64>(2).unwrap(), 500);
}

#[test]
fn test_bugs4_first_value_with_frame() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE fv_test (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO fv_test VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
        (),
    )
    .expect("Failed to insert data");

    // Sliding window of 3 rows
    let mut rows = db
        .query(
            "SELECT id, val, \
             FIRST_VALUE(val) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as first_in_window \
             FROM fv_test ORDER BY id",
            (),
        )
        .expect("Query failed");

    // Row 1: frame [10, 20], first = 10
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(2).unwrap(), 10);

    // Row 2: frame [10, 20, 30], first = 10
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(2).unwrap(), 10);

    // Row 3: frame [20, 30, 40], first = 20
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<i64>(2).unwrap(), 20);

    // Row 4: frame [30, 40, 50], first = 30
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 4);
    assert_eq!(row.get::<i64>(2).unwrap(), 30);

    // Row 5: frame [40, 50], first = 40
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);
    assert_eq!(row.get::<i64>(2).unwrap(), 40);
}

#[test]
fn test_bugs4_last_value_with_frame() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE lv_test (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO lv_test VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
        (),
    )
    .expect("Failed to insert data");

    // Sliding window of 3 rows
    let mut rows = db
        .query(
            "SELECT id, val, \
             LAST_VALUE(val) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as last_in_window \
             FROM lv_test ORDER BY id",
            (),
        )
        .expect("Query failed");

    // Row 1: frame [10, 20], last = 20
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(2).unwrap(), 20);

    // Row 2: frame [10, 20, 30], last = 30
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(2).unwrap(), 30);

    // Row 3: frame [20, 30, 40], last = 40
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<i64>(2).unwrap(), 40);

    // Row 4: frame [30, 40, 50], last = 50
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 4);
    assert_eq!(row.get::<i64>(2).unwrap(), 50);

    // Row 5: frame [40, 50], last = 50
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 5);
    assert_eq!(row.get::<i64>(2).unwrap(), 50);
}

#[test]
fn test_bugs4_first_value_unbounded_following() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE fv_unbounded (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO fv_unbounded VALUES (1, 10), (2, 20), (3, 30)",
        (),
    )
    .expect("Failed to insert data");

    // ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    let mut rows = db
        .query(
            "SELECT id, val, \
             FIRST_VALUE(val) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as first_all, \
             LAST_VALUE(val) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_all \
             FROM fv_unbounded ORDER BY id",
            (),
        )
        .expect("Query failed");

    // All rows should see entire partition, so first=10, last=30 for all
    for _ in 0..3 {
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get::<i64>(2).unwrap(), 10); // first_all
        assert_eq!(row.get::<i64>(3).unwrap(), 30); // last_all
    }
}
