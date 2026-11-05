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

//! Integration tests for INSERT INTO ... SELECT support

use stoolap::Database;

fn create_test_db(name: &str) -> Database {
    Database::open(&format!("memory://{}", name)).expect("Failed to create in-memory database")
}

// ============================================================================
// Basic INSERT ... SELECT Tests
// ============================================================================

#[test]
fn test_insert_select_from_same_table() {
    let db = create_test_db("insert_select_same");

    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO numbers VALUES (1, 100), (2, 200), (3, 300)",
        (),
    )
    .unwrap();

    // Insert copy of existing rows with new IDs
    db.execute(
        "INSERT INTO numbers (id, value) SELECT id + 10, value * 2 FROM numbers",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM numbers", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(6));

    // Check the new rows
    let result = db
        .query(
            "SELECT id, value FROM numbers WHERE id > 10 ORDER BY id",
            (),
        )
        .unwrap();
    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (11, 200));
    assert_eq!(rows[1], (12, 400));
    assert_eq!(rows[2], (13, 600));
}

#[test]
fn test_insert_select_from_another_table() {
    let db = create_test_db("insert_select_other");

    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE target (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO source VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO target (id, name) SELECT id, name FROM source",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM target", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(3));

    let result = db.query("SELECT name FROM target ORDER BY id", ()).unwrap();
    let mut names: Vec<String> = Vec::new();
    for row in result {
        let row = row.unwrap();
        names.push(row.get(0).unwrap());
    }
    assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
}

#[test]
fn test_insert_select_with_where_clause() {
    let db = create_test_db("insert_select_where");

    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE target (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO source VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
        (),
    )
    .unwrap();

    // Only insert rows where value > 25
    db.execute(
        "INSERT INTO target (id, value) SELECT id, value FROM source WHERE value > 25",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM target", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(3));

    let result = db.query("SELECT id FROM target ORDER BY id", ()).unwrap();
    let mut ids: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        ids.push(row.get(0).unwrap());
    }
    assert_eq!(ids, vec![3, 4, 5]);
}

// ============================================================================
// INSERT ... SELECT with VALUES Tests
// ============================================================================

#[test]
fn test_insert_select_from_values() {
    let db = create_test_db("insert_select_values");

    db.execute(
        "CREATE TABLE numbers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO numbers (id, name) SELECT * FROM (VALUES (1, 'one'), (2, 'two'), (3, 'three')) AS t(id, name)", ())
        .unwrap();

    let result = db.query("SELECT COUNT(*) FROM numbers", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(3));

    let result = db
        .query("SELECT id, name FROM numbers ORDER BY id", ())
        .unwrap();
    let mut rows: Vec<(i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(rows[0], (1, "one".to_string()));
    assert_eq!(rows[1], (2, "two".to_string()));
    assert_eq!(rows[2], (3, "three".to_string()));
}

#[test]
fn test_insert_select_from_values_with_expressions() {
    let db = create_test_db("insert_select_values_expr");

    db.execute(
        "CREATE TABLE data (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO data (id, value) SELECT x, x * x FROM (VALUES (1), (2), (3), (4), (5)) AS t(x)", ())
        .unwrap();

    let result = db
        .query("SELECT id, value FROM data ORDER BY id", ())
        .unwrap();
    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0], (1, 1)); // 1^2
    assert_eq!(rows[1], (2, 4)); // 2^2
    assert_eq!(rows[2], (3, 9)); // 3^2
    assert_eq!(rows[3], (4, 16)); // 4^2
    assert_eq!(rows[4], (5, 25)); // 5^2
}

#[test]
fn test_insert_select_from_values_filtered() {
    let db = create_test_db("insert_select_values_filter");

    db.execute(
        "CREATE TABLE positive (id INTEGER PRIMARY KEY, num INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO positive (id, num)
         SELECT ROW_NUMBER() OVER (), x
         FROM (VALUES (-2), (-1), (0), (1), (2), (3)) AS t(x)
         WHERE x > 0",
        (),
    )
    .unwrap();

    let result = db
        .query("SELECT num FROM positive ORDER BY num", ())
        .unwrap();
    let mut nums: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        nums.push(row.get(0).unwrap());
    }
    assert_eq!(nums, vec![1, 2, 3]);
}

// ============================================================================
// INSERT ... SELECT with Aggregation Tests
// ============================================================================

#[test]
fn test_insert_select_with_aggregation() {
    let db = create_test_db("insert_select_agg");

    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER)",
        (),
    )
    .unwrap();
    // Simple table without primary key constraint on category
    db.execute("CREATE TABLE summary (category TEXT, total INTEGER)", ())
        .unwrap();

    db.execute(
        "INSERT INTO sales VALUES
         (1, 'Electronics', 100),
         (2, 'Electronics', 200),
         (3, 'Books', 50),
         (4, 'Books', 75),
         (5, 'Clothing', 150)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO summary (category, total)
         SELECT category, SUM(amount) FROM sales GROUP BY category",
        (),
    )
    .unwrap();

    let result = db
        .query("SELECT category, total FROM summary ORDER BY category", ())
        .unwrap();
    let mut rows: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], ("Books".to_string(), 125));
    assert_eq!(rows[1], ("Clothing".to_string(), 150));
    assert_eq!(rows[2], ("Electronics".to_string(), 300));
}

#[test]
fn test_insert_select_with_count() {
    let db = create_test_db("insert_select_count");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT)",
        (),
    )
    .unwrap();
    // Simple table without primary key constraint
    db.execute(
        "CREATE TABLE category_counts (category TEXT, item_count INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO items VALUES (1, 'A'), (2, 'A'), (3, 'B'), (4, 'A'), (5, 'C')",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO category_counts (category, item_count)
         SELECT category, COUNT(*) FROM items GROUP BY category",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT category, item_count FROM category_counts ORDER BY category",
            (),
        )
        .unwrap();
    let mut rows: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], ("A".to_string(), 3));
    assert_eq!(rows[1], ("B".to_string(), 1));
    assert_eq!(rows[2], ("C".to_string(), 1));
}

// ============================================================================
// INSERT ... SELECT with JOINs Tests
// ============================================================================

#[test]
fn test_insert_select_with_join() {
    let db = create_test_db("insert_select_join");

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)",
        (),
    )
    .unwrap();
    // Simple table without primary key constraint
    db.execute(
        "CREATE TABLE user_totals (user_name TEXT, total_amount INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (1, 1, 100), (2, 1, 50), (3, 2, 200), (4, 3, 75)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO user_totals (user_name, total_amount)
         SELECT u.name, SUM(o.amount)
         FROM users u
         JOIN orders o ON u.id = o.user_id
         GROUP BY u.name",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT user_name, total_amount FROM user_totals ORDER BY user_name",
            (),
        )
        .unwrap();
    let mut rows: Vec<(String, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], ("Alice".to_string(), 150));
    assert_eq!(rows[1], ("Bob".to_string(), 200));
    assert_eq!(rows[2], ("Charlie".to_string(), 75));
}

// ============================================================================
// INSERT ... SELECT with UNION Tests
// ============================================================================

#[test]
fn test_insert_select_with_union() {
    let db = create_test_db("insert_select_union");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE combined (val INTEGER)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (1, 20), (2, 30)", ())
        .unwrap();

    // UNION removes duplicate 20
    db.execute(
        "INSERT INTO combined (val) SELECT val FROM t1 UNION SELECT val FROM t2",
        (),
    )
    .unwrap();

    let result = db
        .query("SELECT val FROM combined ORDER BY val", ())
        .unwrap();
    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn test_insert_select_with_union_all() {
    let db = create_test_db("insert_select_union_all");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE combined (val INTEGER)", ())
        .unwrap();

    db.execute("INSERT INTO t1 VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("INSERT INTO t2 VALUES (1, 20), (2, 30)", ())
        .unwrap();

    // UNION ALL keeps duplicate 20
    db.execute(
        "INSERT INTO combined (val) SELECT val FROM t1 UNION ALL SELECT val FROM t2",
        (),
    )
    .unwrap();

    let result = db
        .query("SELECT val FROM combined ORDER BY val", ())
        .unwrap();
    let mut values: Vec<i64> = Vec::new();
    for row in result {
        let row = row.unwrap();
        values.push(row.get(0).unwrap());
    }
    assert_eq!(values, vec![10, 20, 20, 30]);
}

// ============================================================================
// INSERT ... SELECT with CTE Tests
// ============================================================================

#[test]
fn test_insert_select_with_cte() {
    let db = create_test_db("insert_select_cte");

    db.execute(
        "CREATE TABLE results (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO results (id, value)
         WITH doubled AS (
             SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(id, val)
         )
         SELECT id, val * 2 FROM doubled",
        (),
    )
    .unwrap();

    let result = db
        .query("SELECT id, value FROM results ORDER BY id", ())
        .unwrap();
    let mut rows: Vec<(i64, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], (1, 20));
    assert_eq!(rows[1], (2, 40));
    assert_eq!(rows[2], (3, 60));
}

// ============================================================================
// INSERT ... SELECT with Different Column Orders
// ============================================================================

#[test]
fn test_insert_select_column_reorder() {
    let db = create_test_db("insert_select_reorder");

    db.execute(
        "CREATE TABLE source (a INTEGER PRIMARY KEY, b TEXT, c INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE target (x INTEGER, y INTEGER PRIMARY KEY, z TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO source VALUES (1, 'hello', 100)", ())
        .unwrap();

    // Insert with reordered columns
    db.execute(
        "INSERT INTO target (z, y, x) SELECT b, a, c FROM source",
        (),
    )
    .unwrap();

    let result = db.query("SELECT x, y, z FROM target", ()).unwrap();
    for row in result {
        let row = row.unwrap();
        let x: i64 = row.get(0).unwrap();
        let y: i64 = row.get(1).unwrap();
        let z: String = row.get(2).unwrap();
        assert_eq!(x, 100);
        assert_eq!(y, 1);
        assert_eq!(z, "hello");
    }
}

#[test]
fn test_insert_select_partial_columns() {
    let db = create_test_db("insert_select_partial");

    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE target (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO source VALUES (1, 'Alice', 100), (2, 'Bob', 200)",
        (),
    )
    .unwrap();

    // Only insert id and name, not value
    db.execute(
        "INSERT INTO target (id, name) SELECT id, name FROM source",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM target", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(2));
}

// ============================================================================
// INSERT ... SELECT with ORDER BY and LIMIT
// ============================================================================

#[test]
fn test_insert_select_with_limit_only() {
    let db = create_test_db("insert_select_limit_only");

    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE limited (value INTEGER)", ())
        .unwrap();

    db.execute(
        "INSERT INTO source VALUES (1, 50), (2, 30), (3, 80), (4, 10), (5, 90)",
        (),
    )
    .unwrap();

    // Insert first 3 rows (without ORDER BY)
    db.execute(
        "INSERT INTO limited (value) SELECT value FROM source LIMIT 3",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM limited", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(3));
}

// ============================================================================
// Large Dataset Tests
// ============================================================================

#[test]
fn test_insert_select_large_dataset() {
    let db = create_test_db("insert_select_large");

    db.execute("CREATE TABLE source (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE target (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    // Insert 1000 rows using recursive CTE
    db.execute(
        "INSERT INTO source
         WITH RECURSIVE cnt(x) AS (
             SELECT 1
             UNION ALL
             SELECT x + 1 FROM cnt WHERE x < 1000
         )
         SELECT x FROM cnt",
        (),
    )
    .unwrap();

    // Copy all rows to target
    db.execute("INSERT INTO target (id) SELECT id FROM source", ())
        .unwrap();

    let result = db.query("SELECT COUNT(*) FROM target", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(1000));

    // Verify min and max
    let result = db.query("SELECT MIN(id), MAX(id) FROM target", ()).unwrap();
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
fn test_insert_select_empty_result() {
    let db = create_test_db("insert_select_empty");

    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE target (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO source VALUES (1, 100)", ())
        .unwrap();

    // Insert with WHERE that matches nothing
    db.execute(
        "INSERT INTO target (id, value) SELECT id, value FROM source WHERE value > 1000",
        (),
    )
    .unwrap();

    let result = db.query("SELECT COUNT(*) FROM target", ()).unwrap();
    let mut count: Option<i64> = None;
    for row in result {
        let row = row.unwrap();
        count = Some(row.get(0).unwrap());
    }
    assert_eq!(count, Some(0));
}

#[test]
fn test_insert_select_single_row() {
    let db = create_test_db("insert_select_single");

    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE target (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO source VALUES (1, 'only')", ())
        .unwrap();

    db.execute(
        "INSERT INTO target (id, name) SELECT id, name FROM source",
        (),
    )
    .unwrap();

    let result = db.query("SELECT name FROM target", ()).unwrap();
    let mut names: Vec<String> = Vec::new();
    for row in result {
        let row = row.unwrap();
        names.push(row.get(0).unwrap());
    }
    assert_eq!(names, vec!["only"]);
}

// ============================================================================
// INSERT ... SELECT with DEFAULT Values (Regression Tests)
// ============================================================================

/// Regression test for DEFAULT values in INSERT...SELECT
/// Previously, INSERT...SELECT would not apply DEFAULT values for omitted columns,
/// instead inserting NULL. This test ensures DEFAULT values are properly applied.
#[test]
fn test_insert_select_with_default_values() {
    let db = create_test_db("insert_select_default");

    // Source table without DEFAULT
    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .unwrap();

    // Target table with DEFAULT values on multiple columns
    db.execute(
        "CREATE TABLE target (
            id INTEGER PRIMARY KEY,
            value INTEGER,
            status TEXT DEFAULT 'active',
            priority INTEGER DEFAULT 5,
            category TEXT DEFAULT 'general'
        )",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO source VALUES (1, 100), (2, 200), (3, 300)", ())
        .unwrap();

    // INSERT...SELECT omitting columns that have DEFAULT values
    db.execute(
        "INSERT INTO target (id, value) SELECT id, value FROM source",
        (),
    )
    .unwrap();

    // Verify DEFAULT values were applied
    let result = db
        .query(
            "SELECT id, value, status, priority, category FROM target ORDER BY id",
            (),
        )
        .unwrap();
    let mut rows: Vec<(i64, i64, String, i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
            row.get(3).unwrap(),
            row.get(4).unwrap(),
        ));
    }

    assert_eq!(rows.len(), 3);
    // All rows should have the DEFAULT values for status, priority, and category
    assert_eq!(
        rows[0],
        (1, 100, "active".to_string(), 5, "general".to_string())
    );
    assert_eq!(
        rows[1],
        (2, 200, "active".to_string(), 5, "general".to_string())
    );
    assert_eq!(
        rows[2],
        (3, 300, "active".to_string(), 5, "general".to_string())
    );
}

/// Test that DEFAULT values work correctly when INSERT...SELECT uses expressions
#[test]
fn test_insert_select_default_with_expressions() {
    let db = create_test_db("insert_select_default_expr");

    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE target (
            id INTEGER PRIMARY KEY,
            name TEXT,
            created_at TEXT DEFAULT 'now',
            is_active INTEGER DEFAULT 1
        )",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO source VALUES (1, 'Alice'), (2, 'Bob')", ())
        .unwrap();

    // INSERT...SELECT with expression in SELECT list
    db.execute(
        "INSERT INTO target (id, name) SELECT id * 10, UPPER(name) FROM source",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "SELECT id, name, created_at, is_active FROM target ORDER BY id",
            (),
        )
        .unwrap();
    let mut rows: Vec<(i64, String, String, i64)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
            row.get(3).unwrap(),
        ));
    }

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], (10, "ALICE".to_string(), "now".to_string(), 1));
    assert_eq!(rows[1], (20, "BOB".to_string(), "now".to_string(), 1));
}

/// Test INSERT...SELECT with DEFAULT value and explicit NULL override
#[test]
fn test_insert_select_default_vs_explicit_null() {
    let db = create_test_db("insert_select_default_null");

    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE target (
            id INTEGER PRIMARY KEY,
            val INTEGER,
            status TEXT DEFAULT 'pending'
        )",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO source VALUES (1, NULL), (2, 100)", ())
        .unwrap();

    // INSERT...SELECT - status should get DEFAULT value 'pending'
    db.execute(
        "INSERT INTO target (id, val) SELECT id, val FROM source",
        (),
    )
    .unwrap();

    let result = db
        .query("SELECT id, val, status FROM target ORDER BY id", ())
        .unwrap();

    let mut row1_status: Option<String> = None;
    let mut row2_val: Option<i64> = None;
    let mut row2_status: Option<String> = None;
    let mut count = 0;

    for row in result {
        let row = row.unwrap();
        let id: i64 = row.get(0).unwrap();
        if id == 1 {
            // val is NULL from source, status should be DEFAULT
            row1_status = Some(row.get::<String>(2).unwrap());
        } else if id == 2 {
            row2_val = Some(row.get(1).unwrap());
            row2_status = Some(row.get(2).unwrap());
        }
        count += 1;
    }

    assert_eq!(count, 2);
    // status should be 'pending' (DEFAULT value), not NULL
    assert_eq!(row1_status, Some("pending".to_string()));
    assert_eq!(row2_val, Some(100));
    assert_eq!(row2_status, Some("pending".to_string()));
}

/// Test that regular INSERT and INSERT...SELECT both apply DEFAULT values consistently
#[test]
fn test_insert_select_default_consistency() {
    let db = create_test_db("insert_select_default_consistent");

    db.execute(
        "CREATE TABLE mixed (
            id INTEGER PRIMARY KEY,
            value INTEGER,
            tag TEXT DEFAULT 'default_tag'
        )",
        (),
    )
    .unwrap();

    // Regular INSERT omitting 'tag' column
    db.execute("INSERT INTO mixed (id, value) VALUES (1, 100)", ())
        .unwrap();

    // INSERT...SELECT omitting 'tag' column
    db.execute("INSERT INTO mixed (id, value) SELECT 2, 200", ())
        .unwrap();

    let result = db
        .query("SELECT id, value, tag FROM mixed ORDER BY id", ())
        .unwrap();
    let mut rows: Vec<(i64, i64, String)> = Vec::new();
    for row in result {
        let row = row.unwrap();
        rows.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
        ));
    }

    assert_eq!(rows.len(), 2);
    // Both regular INSERT and INSERT...SELECT should apply the same DEFAULT value
    assert_eq!(rows[0], (1, 100, "default_tag".to_string()));
    assert_eq!(rows[1], (2, 200, "default_tag".to_string()));
}
