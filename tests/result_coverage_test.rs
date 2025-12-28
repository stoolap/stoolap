// Result types integration tests
// Tests for result.rs - ExecResult, ExecutorMemoryResult, FilteredResult,
// ExprMappedResult, LimitedResult, OrderedResult, TopNResult, DistinctResult,
// AliasedResult, ProjectedResult, ScannerResult, StreamingProjectionResult

use stoolap::Database;

// ============================================================================
// LIMIT and OFFSET Tests
// ============================================================================

#[test]
fn test_limit_only() {
    let db = Database::open("memory://test_limit_only").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO items VALUES ({}, 'item{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    let result = db
        .query("SELECT id FROM items ORDER BY id LIMIT 5", ())
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_offset_only() {
    let db = Database::open("memory://test_offset_only").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO items VALUES ({}, 'item{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    let result = db
        .query("SELECT id FROM items ORDER BY id OFFSET 7", ())
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![8, 9, 10]);
}

#[test]
fn test_limit_and_offset() {
    let db = Database::open("memory://test_limit_and_offset").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO items VALUES ({}, 'item{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    let result = db
        .query("SELECT id FROM items ORDER BY id LIMIT 3 OFFSET 2", ())
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![3, 4, 5]);
}

#[test]
fn test_limit_exceeds_rows() {
    let db = Database::open("memory://test_limit_exceeds_rows").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    for i in 1..=3 {
        db.execute(
            &format!("INSERT INTO items VALUES ({}, 'item{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    let result = db
        .query("SELECT id FROM items ORDER BY id LIMIT 10", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 3);
}

#[test]
fn test_offset_exceeds_rows() {
    let db = Database::open("memory://test_offset_exceeds_rows").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    for i in 1..=3 {
        db.execute(
            &format!("INSERT INTO items VALUES ({}, 'item{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    let result = db
        .query("SELECT id FROM items OFFSET 10", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 0);
}

#[test]
fn test_zero_limit() {
    let db = Database::open("memory://test_zero_limit").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    for i in 1..=5 {
        db.execute(
            &format!("INSERT INTO items VALUES ({}, 'item{}')", i, i),
            (),
        )
        .expect("Failed to insert");
    }

    let result = db
        .query("SELECT id FROM items LIMIT 0", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 0);
}

// ============================================================================
// ORDER BY Tests (including radix sort for integers)
// ============================================================================

#[test]
fn test_order_by_ascending() {
    let db = Database::open("memory://test_order_by_asc").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 30)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 10)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 20)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT value FROM items ORDER BY value ASC", ())
        .expect("Query failed");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn test_order_by_descending() {
    let db = Database::open("memory://test_order_by_desc").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 30)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 10)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 20)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT value FROM items ORDER BY value DESC", ())
        .expect("Query failed");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(values, vec![30, 20, 10]);
}

#[test]
fn test_order_by_multiple_columns() {
    let db = Database::open("memory://test_order_by_multi").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category INTEGER, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 1, 30)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 2, 10)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 1, 20)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (4, 2, 40)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT category, value FROM items ORDER BY category ASC, value DESC",
            (),
        )
        .expect("Query failed");

    let mut results: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        results.push((row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()));
    }

    assert_eq!(results, vec![(1, 30), (1, 20), (2, 40), (2, 10)]);
}

#[test]
fn test_order_by_with_nulls() {
    let db = Database::open("memory://test_order_by_nulls").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 30)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 10)", ())
        .expect("Failed to insert");

    // Default behavior: NULLS FIRST for ASC in Stoolap
    let result = db
        .query("SELECT value FROM items ORDER BY value ASC", ())
        .expect("Query failed");

    let mut values: Vec<Option<i64>> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).ok());
    }

    // NULLs are at the beginning for ASC (default behavior)
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], None);
    assert_eq!(values[1], Some(10));
    assert_eq!(values[2], Some(30));
}

#[test]
fn test_order_by_with_nulls_desc() {
    let db = Database::open("memory://test_order_by_nulls_desc").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 30)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 10)", ())
        .expect("Failed to insert");

    // Default behavior: NULLS LAST for DESC in Stoolap
    let result = db
        .query("SELECT value FROM items ORDER BY value DESC", ())
        .expect("Query failed");

    let mut values: Vec<Option<i64>> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).ok());
    }

    // NULLs are at the end for DESC (default behavior)
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Some(30));
    assert_eq!(values[1], Some(10));
    assert_eq!(values[2], None);
}

#[test]
fn test_order_by_text() {
    let db = Database::open("memory://test_order_by_text").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'Charlie')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'Alice')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 'Bob')", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT name FROM items ORDER BY name ASC", ())
        .expect("Query failed");

    let mut names: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        names.push(row.get::<String>(0).unwrap());
    }

    assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
}

// ============================================================================
// Top-N Result Tests (ORDER BY + LIMIT optimization)
// ============================================================================

#[test]
fn test_top_n_small_limit() {
    let db = Database::open("memory://test_top_n_small").expect("Failed to open database");

    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    // Insert 100 rows
    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO numbers VALUES ({}, {})", i, 101 - i),
            (),
        )
        .expect("Failed to insert");
    }

    // Get top 3 by value DESC
    let result = db
        .query("SELECT value FROM numbers ORDER BY value DESC LIMIT 3", ())
        .expect("Query failed");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(values, vec![100, 99, 98]);
}

#[test]
fn test_top_n_with_offset() {
    let db = Database::open("memory://test_top_n_offset").expect("Failed to open database");

    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=100 {
        db.execute(
            &format!("INSERT INTO numbers VALUES ({}, {})", i, 101 - i),
            (),
        )
        .expect("Failed to insert");
    }

    // Skip first 2, then get 3 rows
    let result = db
        .query(
            "SELECT value FROM numbers ORDER BY value DESC LIMIT 3 OFFSET 2",
            (),
        )
        .expect("Query failed");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(values, vec![98, 97, 96]);
}

// ============================================================================
// DISTINCT Tests
// ============================================================================

#[test]
fn test_distinct_simple() {
    let db = Database::open("memory://test_distinct_simple").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'A')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'B')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 'A')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (4, 'C')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (5, 'B')", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT DISTINCT category FROM items ORDER BY category", ())
        .expect("Query failed");

    let mut categories: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        categories.push(row.get::<String>(0).unwrap());
    }

    assert_eq!(categories, vec!["A", "B", "C"]);
}

#[test]
fn test_distinct_multiple_columns() {
    let db = Database::open("memory://test_distinct_multi").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, cat1 TEXT, cat2 TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'A', 'X')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'A', 'Y')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 'A', 'X')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (4, 'B', 'X')", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT DISTINCT cat1, cat2 FROM items ORDER BY cat1, cat2",
            (),
        )
        .expect("Query failed");

    let mut pairs: Vec<(String, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        pairs.push((row.get::<String>(0).unwrap(), row.get::<String>(1).unwrap()));
    }

    assert_eq!(
        pairs,
        vec![
            ("A".to_string(), "X".to_string()),
            ("A".to_string(), "Y".to_string()),
            ("B".to_string(), "X".to_string()),
        ]
    );
}

#[test]
fn test_distinct_with_nulls() {
    let db = Database::open("memory://test_distinct_nulls").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'A')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 'A')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (4, NULL)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT DISTINCT value FROM items", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        row.expect("Failed to get row");
        count += 1;
    }

    // Should have 2 distinct values: 'A' and NULL
    assert_eq!(count, 2);
}

#[test]
fn test_distinct_with_limit() {
    let db = Database::open("memory://test_distinct_limit").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'A')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'B')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 'C')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (4, 'D')", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT DISTINCT category FROM items ORDER BY category LIMIT 2",
            (),
        )
        .expect("Query failed");

    let mut categories: Vec<String> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        categories.push(row.get::<String>(0).unwrap());
    }

    assert_eq!(categories, vec!["A", "B"]);
}

// ============================================================================
// Expression Projection Tests (ExprMappedResult)
// ============================================================================

#[test]
fn test_projection_arithmetic() {
    let db = Database::open("memory://test_proj_arith").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, price INTEGER, quantity INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 10, 5)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 20, 3)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT id, price * quantity AS total FROM items ORDER BY id",
            (),
        )
        .expect("Query failed");

    let mut totals: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        totals.push((row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()));
    }

    assert_eq!(totals, vec![(1, 50), (2, 60)]);
}

#[test]
fn test_projection_case_expression() {
    let db = Database::open("memory://test_proj_case").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 10)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 50)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 100)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT id, CASE WHEN value < 30 THEN 'low' WHEN value < 80 THEN 'medium' ELSE 'high' END AS level FROM items ORDER BY id",
            (),
        )
        .expect("Query failed");

    let mut levels: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        levels.push((row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap()));
    }

    assert_eq!(
        levels,
        vec![
            (1, "low".to_string()),
            (2, "medium".to_string()),
            (3, "high".to_string())
        ]
    );
}

#[test]
fn test_projection_coalesce() {
    let db = Database::open("memory://test_proj_coalesce").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, alias TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, NULL, 'Bob')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'Alice', NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, NULL, NULL)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT id, COALESCE(name, alias, 'Unknown') AS display FROM items ORDER BY id",
            (),
        )
        .expect("Query failed");

    let mut displays: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        displays.push((row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap()));
    }

    assert_eq!(
        displays,
        vec![
            (1, "Bob".to_string()),
            (2, "Alice".to_string()),
            (3, "Unknown".to_string())
        ]
    );
}

#[test]
fn test_projection_concat() {
    let db = Database::open("memory://test_proj_concat").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'John', 'Doe')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'Jane', 'Smith')", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT id, first_name || ' ' || last_name AS full_name FROM items ORDER BY id",
            (),
        )
        .expect("Query failed");

    let mut names: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        names.push((row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap()));
    }

    assert_eq!(
        names,
        vec![(1, "John Doe".to_string()), (2, "Jane Smith".to_string())]
    );
}

#[test]
fn test_projection_functions() {
    let db = Database::open("memory://test_proj_functions").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'hello', 3.7)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'WORLD', 2.2)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT id, UPPER(name), ROUND(value) FROM items ORDER BY id",
            (),
        )
        .expect("Query failed");

    let mut results: Vec<(i64, String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        results.push((
            row.get::<i64>(0).unwrap(),
            row.get::<String>(1).unwrap(),
            row.get::<f64>(2).unwrap(),
        ));
    }

    assert_eq!(
        results,
        vec![(1, "HELLO".to_string(), 4.0), (2, "WORLD".to_string(), 2.0)]
    );
}

// ============================================================================
// Column Alias Tests (AliasedResult)
// ============================================================================

#[test]
fn test_column_alias_simple() {
    let db = Database::open("memory://test_alias_simple").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'Alice')", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT id AS item_id, name AS item_name FROM items", ())
        .expect("Query failed");

    // Check column names
    let columns = result.columns();
    assert!(columns.contains(&"item_id".to_string()) || columns.contains(&"ITEM_ID".to_string()));
    assert!(
        columns.contains(&"item_name".to_string()) || columns.contains(&"ITEM_NAME".to_string())
    );
}

#[test]
fn test_column_alias_expression() {
    let db = Database::open("memory://test_alias_expr").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, price INTEGER, qty INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 10, 5)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT price * qty AS total_value FROM items", ())
        .expect("Query failed");

    let columns = result.columns();
    assert!(
        columns.contains(&"total_value".to_string())
            || columns.contains(&"TOTAL_VALUE".to_string())
    );

    for row in result {
        let row = row.expect("Failed to get row");
        assert_eq!(row.get::<i64>(0).unwrap(), 50);
    }
}

// ============================================================================
// Star Expansion Tests (QualifiedStar)
// ============================================================================

#[test]
fn test_star_expansion() {
    let db = Database::open("memory://test_star_expansion").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'test', 100)", ())
        .expect("Failed to insert");

    let result = db.query("SELECT * FROM items", ()).expect("Query failed");

    let columns = result.columns();
    assert_eq!(columns.len(), 3);

    for row in result {
        let row = row.expect("Failed to get row");
        assert_eq!(row.get::<i64>(0).unwrap(), 1);
        assert_eq!(row.get::<String>(1).unwrap(), "test");
        assert_eq!(row.get::<i64>(2).unwrap(), 100);
    }
}

#[test]
fn test_qualified_star() {
    let db = Database::open("memory://test_qualified_star").expect("Failed to open database");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t1 VALUES (1, 'Alice')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO t2 VALUES (1, 100)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT t1.* FROM t1 JOIN t2 ON t1.id = t2.id", ())
        .expect("Query failed");

    let columns = result.columns();
    // Should only have columns from t1
    assert!(columns.len() == 2);
}

// ============================================================================
// Scan and Memory Result Tests
// ============================================================================

#[test]
fn test_exec_result_insert() {
    let db = Database::open("memory://test_exec_insert").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    let rows_affected = db
        .execute("INSERT INTO items VALUES (1, 'test')", ())
        .expect("Failed to insert");

    assert_eq!(rows_affected, 1);
}

#[test]
fn test_exec_result_update() {
    let db = Database::open("memory://test_exec_update").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=5 {
        db.execute(&format!("INSERT INTO items VALUES ({}, 10)", i), ())
            .expect("Failed to insert");
    }

    let rows_affected = db
        .execute("UPDATE items SET value = 20 WHERE id > 2", ())
        .expect("Failed to update");

    assert_eq!(rows_affected, 3);
}

#[test]
fn test_exec_result_delete() {
    let db = Database::open("memory://test_exec_delete").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=5 {
        db.execute(&format!("INSERT INTO items VALUES ({}, {})", i, i), ())
            .expect("Failed to insert");
    }

    let rows_affected = db
        .execute("DELETE FROM items WHERE value < 3", ())
        .expect("Failed to delete");

    assert_eq!(rows_affected, 2);
}

// ============================================================================
// Filtered Result Tests
// ============================================================================

#[test]
fn test_filtered_result_equal() {
    let db = Database::open("memory://test_filtered_equal").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'A')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'B')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 'A')", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT id FROM items WHERE category = 'A' ORDER BY id", ())
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![1, 3]);
}

#[test]
fn test_filtered_result_range() {
    let db = Database::open("memory://test_filtered_range").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO items VALUES ({}, {})", i, i * 10), ())
            .expect("Failed to insert");
    }

    let result = db
        .query(
            "SELECT id FROM items WHERE value >= 30 AND value <= 70 ORDER BY id",
            (),
        )
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![3, 4, 5, 6, 7]);
}

#[test]
fn test_filtered_result_like() {
    let db = Database::open("memory://test_filtered_like").expect("Failed to open database");

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'apple')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'banana')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 'apricot')", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT id FROM items WHERE name LIKE 'ap%' ORDER BY id", ())
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![1, 3]);
}

#[test]
fn test_filtered_result_in() {
    let db = Database::open("memory://test_filtered_in").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO items VALUES ({}, {})", i, i), ())
            .expect("Failed to insert");
    }

    let result = db
        .query(
            "SELECT id FROM items WHERE value IN (2, 4, 6, 8) ORDER BY id",
            (),
        )
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![2, 4, 6, 8]);
}

#[test]
fn test_filtered_result_is_null() {
    let db = Database::open("memory://test_filtered_is_null").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 10)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 30)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (4, NULL)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT id FROM items WHERE value IS NULL ORDER BY id", ())
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![2, 4]);
}

// ============================================================================
// Streaming Results Tests
// ============================================================================

#[test]
fn test_streaming_with_where_and_limit() {
    let db =
        Database::open("memory://test_streaming_where_limit").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=100 {
        db.execute(&format!("INSERT INTO items VALUES ({}, {})", i, i), ())
            .expect("Failed to insert");
    }

    // This should filter and limit streaming (early termination)
    let result = db
        .query(
            "SELECT id FROM items WHERE value > 50 ORDER BY id LIMIT 5",
            (),
        )
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(ids, vec![51, 52, 53, 54, 55]);
}

#[test]
fn test_streaming_projection() {
    let db = Database::open("memory://test_streaming_proj").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, value INTEGER, extra TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 'A', 10, 'x')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 'B', 20, 'y')", ())
        .expect("Failed to insert");

    // Project only specific columns
    let result = db
        .query("SELECT id, value FROM items ORDER BY id", ())
        .expect("Query failed");

    let columns = result.columns();
    assert_eq!(columns.len(), 2);

    for row in result {
        let row = row.expect("Failed to get row");
        let id = row.get::<i64>(0).unwrap();
        let value = row.get::<i64>(1).unwrap();
        assert_eq!(value, id * 10);
    }
}

// ============================================================================
// Complex Query Tests (combining multiple result types)
// ============================================================================

#[test]
fn test_complex_filter_order_limit() {
    let db = Database::open("memory://test_complex_fol").expect("Failed to open database");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, amount INTEGER, region TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO sales VALUES (1, 'Widget', 100, 'North')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO sales VALUES (2, 'Widget', 200, 'South')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO sales VALUES (3, 'Gadget', 150, 'North')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO sales VALUES (4, 'Widget', 50, 'East')", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO sales VALUES (5, 'Gadget', 300, 'South')", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT product, amount FROM sales WHERE amount >= 100 ORDER BY amount DESC LIMIT 3",
            (),
        )
        .expect("Query failed");

    let mut results: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        results.push((row.get::<String>(0).unwrap(), row.get::<i64>(1).unwrap()));
    }

    assert_eq!(results.len(), 3);
    assert_eq!(results[0], ("Gadget".to_string(), 300));
    assert_eq!(results[1], ("Widget".to_string(), 200));
    assert_eq!(results[2], ("Gadget".to_string(), 150));
}

#[test]
fn test_complex_distinct_filter_order() {
    let db = Database::open("memory://test_complex_dfo").expect("Failed to open database");

    db.execute(
        "CREATE TABLE logs (id INTEGER PRIMARY KEY, event_type TEXT, severity INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO logs VALUES (1, 'error', 3)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO logs VALUES (2, 'warning', 2)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO logs VALUES (3, 'error', 3)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO logs VALUES (4, 'info', 1)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO logs VALUES (5, 'warning', 2)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO logs VALUES (6, 'error', 3)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT DISTINCT event_type, severity FROM logs WHERE severity >= 2 ORDER BY severity DESC",
            (),
        )
        .expect("Query failed");

    let mut results: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        results.push((row.get::<String>(0).unwrap(), row.get::<i64>(1).unwrap()));
    }

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], ("error".to_string(), 3));
    assert_eq!(results[1], ("warning".to_string(), 2));
}

#[test]
fn test_complex_expression_filter_order() {
    let db = Database::open("memory://test_complex_efo").expect("Failed to open database");

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price FLOAT, discount FLOAT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO products VALUES (1, 'A', 100.0, 0.1)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO products VALUES (2, 'B', 200.0, 0.2)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO products VALUES (3, 'C', 50.0, 0.05)", ())
        .expect("Failed to insert");

    let result = db
        .query(
            "SELECT name, price * (1 - discount) AS final_price FROM products ORDER BY price * (1 - discount) DESC",
            (),
        )
        .expect("Query failed");

    let mut results: Vec<(String, f64)> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        results.push((row.get::<String>(0).unwrap(), row.get::<f64>(1).unwrap()));
    }

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, "B");
    assert!((results[0].1 - 160.0).abs() < 0.001);
    assert_eq!(results[1].0, "A");
    assert!((results[1].1 - 90.0).abs() < 0.001);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_result_set() {
    let db = Database::open("memory://test_empty_result").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    let result = db
        .query("SELECT * FROM items WHERE value > 100", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 0);
}

#[test]
fn test_single_row_result() {
    let db = Database::open("memory://test_single_row").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 100)", ())
        .expect("Failed to insert");

    let result = db.query("SELECT * FROM items", ()).expect("Query failed");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        assert_eq!(row.get::<i64>(0).unwrap(), 1);
        assert_eq!(row.get::<i64>(1).unwrap(), 100);
        count += 1;
    }

    assert_eq!(count, 1);
}

#[test]
fn test_all_columns_null() {
    let db = Database::open("memory://test_all_null").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, NULL, NULL)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT a, b FROM items", ())
        .expect("Query failed");

    for row in result {
        let row = row.expect("Failed to get row");
        // Use get_value to check for actual NULL
        let val_a = row.get_value(0);
        let val_b = row.get_value(1);
        assert!(
            val_a.map(|v| v.is_null()).unwrap_or(false),
            "a should be NULL"
        );
        assert!(
            val_b.map(|v| v.is_null()).unwrap_or(false),
            "b should be NULL"
        );
    }
}

#[test]
fn test_large_offset() {
    let db = Database::open("memory://test_large_offset").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO items VALUES ({}, {})", i, i), ())
            .expect("Failed to insert");
    }

    let result = db
        .query("SELECT * FROM items OFFSET 1000", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 0);
}

#[test]
fn test_order_by_expression() {
    let db = Database::open("memory://test_order_by_expr").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 2, 3)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, 5, 1)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 1, 8)", ())
        .expect("Failed to insert");

    // Order by expression (a + b)
    let result = db
        .query("SELECT id FROM items ORDER BY a + b ASC", ())
        .expect("Query failed");

    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        ids.push(row.get::<i64>(0).unwrap());
    }

    // (1,2,3)=5, (2,5,1)=6, (3,1,8)=9
    assert_eq!(ids, vec![1, 2, 3]);
}

#[test]
fn test_nulls_first() {
    let db = Database::open("memory://test_nulls_first").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 10)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 5)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT value FROM items ORDER BY value ASC NULLS FIRST", ())
        .expect("Query failed");

    let mut values: Vec<Option<i64>> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).ok());
    }

    assert_eq!(values.len(), 3);
    assert_eq!(values[0], None); // NULL first
    assert_eq!(values[1], Some(5));
    assert_eq!(values[2], Some(10));
}

#[test]
fn test_nulls_last() {
    let db = Database::open("memory://test_nulls_last").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO items VALUES (1, 10)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (2, NULL)", ())
        .expect("Failed to insert");
    db.execute("INSERT INTO items VALUES (3, 5)", ())
        .expect("Failed to insert");

    let result = db
        .query("SELECT value FROM items ORDER BY value DESC NULLS LAST", ())
        .expect("Query failed");

    let mut values: Vec<Option<i64>> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).ok());
    }

    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Some(10));
    assert_eq!(values[1], Some(5));
    assert_eq!(values[2], None); // NULL last
}

#[test]
fn test_distinct_all_duplicates() {
    let db = Database::open("memory://test_distinct_all_dups").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=5 {
        db.execute(&format!("INSERT INTO items VALUES ({}, 'same')", i), ())
            .expect("Failed to insert");
    }

    let result = db
        .query("SELECT DISTINCT value FROM items", ())
        .expect("Query failed");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        assert_eq!(row.get::<String>(0).unwrap(), "same");
        count += 1;
    }

    assert_eq!(count, 1);
}

#[test]
fn test_limit_one() {
    let db = Database::open("memory://test_limit_one").expect("Failed to open database");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    for i in 1..=10 {
        db.execute(&format!("INSERT INTO items VALUES ({}, {})", i, i * 10), ())
            .expect("Failed to insert");
    }

    let result = db
        .query("SELECT value FROM items ORDER BY value DESC LIMIT 1", ())
        .expect("Query failed");

    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        values.push(row.get::<i64>(0).unwrap());
    }

    assert_eq!(values, vec![100]);
}
