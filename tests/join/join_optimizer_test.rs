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

//! Integration tests for join optimizer

use stoolap::Database;

#[test]
fn test_explain_shows_hash_join() {
    let db = Database::open("memory://join_opt_hash").unwrap();

    // Create tables with non-indexed columns for join condition
    // Using non-PK columns forces hash join instead of index nested loop
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, dept_id INTEGER, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, dept_code INTEGER, name TEXT)",
        (),
    )
    .unwrap();

    // Insert enough data to make hash join beneficial (planner uses nested loop for < 200 rows)
    for i in 0..300 {
        db.execute(
            "INSERT INTO users (id, dept_id, name) VALUES (?, ?, ?)",
            (i, i % 10, format!("User {}", i)),
        )
        .unwrap();
    }
    for i in 0..50 {
        db.execute(
            "INSERT INTO departments (id, dept_code, name) VALUES (?, ?, ?)",
            (i, i, format!("Dept {}", i)),
        )
        .unwrap();
    }

    // Collect statistics so the planner can make informed decisions
    db.execute("ANALYZE users", ()).unwrap();
    db.execute("ANALYZE departments", ()).unwrap();

    // EXPLAIN should show Hash Join for equality condition on non-indexed columns
    let result = db
        .query(
            "EXPLAIN SELECT * FROM users u INNER JOIN departments d ON u.dept_id = d.dept_code",
            (),
        )
        .unwrap();

    let mut found_hash_join = false;
    let mut all_lines = Vec::new();
    for row in result {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        all_lines.push(plan_line.clone());
        if plan_line.contains("Hash Join") {
            found_hash_join = true;
        }
    }

    assert!(
        found_hash_join,
        "EXPLAIN should show Hash Join for equality condition on non-indexed columns. Got:\n{}",
        all_lines.join("\n")
    );
}

#[test]
fn test_explain_shows_nested_loop_for_cross_join() {
    let db = Database::open("memory://join_opt_nested").unwrap();

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY)", ())
        .unwrap();

    // CROSS JOIN should show Nested Loop
    let result = db
        .query("EXPLAIN SELECT * FROM t1 CROSS JOIN t2", ())
        .unwrap();

    let mut found_nested_loop = false;
    for row in result {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        if plan_line.contains("Nested Loop") {
            found_nested_loop = true;
            break;
        }
    }

    assert!(
        found_nested_loop,
        "EXPLAIN should show Nested Loop for CROSS JOIN"
    );
}

#[test]
fn test_explain_shows_rows_and_no_cost() {
    let db = Database::open("memory://join_opt_cost").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();

    let result = db
        .query(
            "EXPLAIN SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            (),
        )
        .unwrap();

    let mut found_cost = false;
    let mut found_rows = false;
    for row in result {
        let row = row.unwrap();
        let plan_line: String = row.get(0).unwrap();
        if plan_line.contains("cost=") {
            found_cost = true;
        }
        if plan_line.contains("rows=") {
            found_rows = true;
        }
    }

    assert!(!found_cost, "EXPLAIN prints no cost it does not plan by");
    assert!(found_rows, "EXPLAIN should show row estimate");
}

/// a of 20,000 rows, x as `x` gives it, indexed, and b of 20,000 rows,
/// both sealed
fn sealed_pair(dir: &tempfile::TempDir, x: &str) -> Database {
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for sql in [
        "CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)".to_string(),
        "CREATE TABLE b (id INTEGER PRIMARY KEY)".to_string(),
        format!("INSERT INTO a SELECT g.value, {x} FROM generate_series(1, 20000) g"),
        "INSERT INTO b SELECT g.value FROM generate_series(1, 20000) g".to_string(),
        "CREATE INDEX a_x ON a (x)".to_string(),
        "PRAGMA CHECKPOINT".to_string(),
    ] {
        db.execute(&sql, ()).unwrap();
    }
    db
}

/// The join line EXPLAIN prints for a join of a and b under `filter` on a
fn join_line(db: &Database, filter: &str) -> String {
    db.query(
        &format!("EXPLAIN SELECT * FROM a JOIN b ON b.id = a.id WHERE {filter}"),
        (),
    )
    .unwrap()
    .map(|row| row.unwrap().get::<String>(0).unwrap())
    .find(|line| line.contains("Join)"))
    .unwrap()
}

#[test]
fn test_explain_counts_an_equality_over_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_pair(&dir, "CASE WHEN g.value <= 10 THEN 1 ELSE 0 END");
    let line = join_line(&db, "a.x = 1");
    assert!(
        line.ends_with("(rows=10)"),
        "the side file counts the 10 sealed rows of x = 1: {line}"
    );
}

#[test]
fn test_explain_does_not_count_sealed_rows_most_of_which_are_gone() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_pair(&dir, "CASE WHEN g.value <= 10000 THEN 1 ELSE 0 END");
    db.execute("UPDATE a SET x = 0 WHERE id BETWEEN 2 AND 10000", ())
        .unwrap();
    let line = join_line(&db, "a.x = 1");
    assert!(
        line.ends_with("(rows=2000)"),
        "one row keeps x = 1, the side file still holds 10,000: the default \
         tenth, not the stale count: {line}"
    );
}

#[test]
fn test_explain_counts_no_more_sealed_keys_than_its_budget() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_pair(&dir, "g.value % 1000");
    let members: Vec<String> = (0..300).map(|x| x.to_string()).collect();
    let line = join_line(&db, &format!("a.x IN ({})", members.join(", ")));
    assert!(
        line.ends_with("(rows=20000)"),
        "300 keys pass the 256 probes the estimate takes: the default share \
         of an IN list, not a count: {line}"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_explain_takes_one_cold_view_for_an_in_list() {
    use stoolap::test_failpoints::estimate_cold_views;
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_pair(&dir, "g.value % 1000");
    let views = |filter: &str| {
        let before = estimate_cold_views();
        join_line(&db, filter);
        estimate_cold_views() - before
    };
    let one = views("a.x = 1");
    assert_eq!(
        views("a.x IN (1, 2, 3, 4, 5)"),
        one,
        "the five keys share the view the one key takes"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_explain_does_not_count_across_a_seal_after_its_cold_view() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_pair(&dir, "0");
    db.execute(
        "INSERT INTO a SELECT g.value, 1 FROM generate_series(20001, 40000) g",
        (),
    )
    .unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_estimate_cold_view(move || {
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    let line = join_line(&db, "a.x = 1");
    assert!(
        line.ends_with("(rows=2000)"),
        "the 20,000 rows of x = 1 left the hot index for a volume the view \
         did not capture: the default tenth, not a count of 0: {line}"
    );
}

#[test]
fn test_explain_estimates_one_row_for_a_primary_key_equality() {
    let db = Database::open("memory://join_opt_pk_one_row").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO a SELECT g.value, g.value FROM generate_series(1, 100) g",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO b SELECT g.value, g.value FROM generate_series(1, 100) g",
        (),
    )
    .unwrap();
    let line = db
        .query(
            "EXPLAIN SELECT * FROM a JOIN b ON a.k = b.k WHERE a.id = 1",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.contains("Join)"))
        .unwrap();
    assert!(
        line.ends_with("(rows=1)"),
        "a.id = 1 keeps the one row of a it names: {line}"
    );
}

#[test]
fn test_explain_analyze_shows_actual_rows() {
    let db = Database::open("memory://join_opt_analyze").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO orders VALUES (1, 1), (2, 1), (3, 2)", ())
        .unwrap();

    let result = db
        .query(
            "EXPLAIN ANALYZE SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id",
            (),
        )
        .unwrap();

    let top: String = result.into_iter().next().unwrap().unwrap().get(0).unwrap();
    assert!(
        top.ends_with(", rows=3)"),
        "EXPLAIN ANALYZE should show the 3 joined rows: {top}"
    );
}

#[test]
fn test_statistics_improve_estimates() {
    let db = Database::open("memory://join_opt_stats").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users SELECT g.value, 'u' FROM generate_series(1, 1000) g",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders SELECT g.value, g.value % 2 FROM generate_series(1, 1000) g",
        (),
    )
    .unwrap();
    let join_line = || {
        db.query(
            "EXPLAIN SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id \
             WHERE o.user_id = 1",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.contains("Join)"))
        .unwrap()
    };

    assert!(
        join_line().ends_with("(rows=100)"),
        "an equality keeps a tenth of orders without statistics: {}",
        join_line()
    );
    db.execute("ANALYZE users", ()).unwrap();
    db.execute("ANALYZE orders", ()).unwrap();
    assert!(
        join_line().ends_with("(rows=500)"),
        "ANALYZE finds user_id takes two values: {}",
        join_line()
    );
    db.execute("CREATE INDEX orders_user ON orders (user_id)", ())
        .unwrap();
    db.execute("DELETE FROM orders WHERE id > 10", ()).unwrap();
    assert!(
        join_line().ends_with("(rows=5)"),
        "the index counts the rows orders holds now: {}",
        join_line()
    );
}

#[test]
fn test_hash_join_executes_correctly() {
    let db = Database::open("memory://join_opt_exec").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (1, 1, 100.0), (2, 1, 200.0), (3, 2, 150.0)",
        (),
    )
    .unwrap();

    // Execute the join
    let result = db
        .query("SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id ORDER BY o.total", ())
        .unwrap();

    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        let total: f64 = row.get(1).unwrap();
        rows.push((name, total));
    }

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], ("Alice".to_string(), 100.0));
    assert_eq!(rows[1], ("Bob".to_string(), 150.0));
    assert_eq!(rows[2], ("Alice".to_string(), 200.0));
}

#[test]
fn test_left_join_with_hash_join() {
    let db = Database::open("memory://join_opt_left").unwrap();

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO orders VALUES (1, 1), (2, 1)", ())
        .unwrap();

    // LEFT JOIN should include users without orders
    let result = db
        .query(
            "SELECT u.name, o.id FROM users u LEFT JOIN orders o ON u.id = o.user_id ORDER BY u.id",
            (),
        )
        .unwrap();

    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        let order_id: Option<i64> = row.get(1).ok();
        rows.push((name, order_id));
    }

    // Alice has 2 orders, Bob has 0, Charlie has 0
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().any(|(n, id)| n == "Alice" && id.is_some()));
    assert!(rows.iter().any(|(n, id)| n == "Bob" && id.is_none()));
    assert!(rows.iter().any(|(n, id)| n == "Charlie" && id.is_none()));
}

#[test]
fn test_join_with_complex_condition() {
    let db = Database::open("memory://join_opt_complex").unwrap();

    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, price FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE order_items (id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO products VALUES (1, 10.0), (2, 20.0), (3, 30.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO order_items VALUES (1, 1, 2), (2, 2, 1), (3, 1, 3)",
        (),
    )
    .unwrap();

    // Join with equality and additional filter
    let result = db
        .query(
            "SELECT p.price, oi.quantity
             FROM products p
             INNER JOIN order_items oi ON p.id = oi.product_id
             WHERE oi.quantity > 1
             ORDER BY p.price",
            (),
        )
        .unwrap();

    let mut rows = Vec::new();
    for row in result {
        let row = row.unwrap();
        let price: f64 = row.get(0).unwrap();
        let quantity: i64 = row.get(1).unwrap();
        rows.push((price, quantity));
    }

    // Only items with quantity > 1
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|(_, q)| *q > 1));
}
