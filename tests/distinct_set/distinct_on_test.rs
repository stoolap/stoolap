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

//! Tests for PostgreSQL-compatible DISTINCT ON (expr, ...) support.

use stoolap::Database;

fn setup_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).unwrap();

    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer TEXT,
            amount FLOAT,
            order_date TEXT
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO orders VALUES (1, 'Alice', 100.0, '2024-01-01')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (2, 'Alice', 200.0, '2024-01-15')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (3, 'Alice', 150.0, '2024-02-01')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (4, 'Bob', 300.0, '2024-01-10')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (5, 'Bob', 250.0, '2024-02-05')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (6, 'Charlie', 50.0, '2024-01-20')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (7, 'Charlie', 175.0, '2024-02-10')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO orders VALUES (8, 'Charlie', 225.0, '2024-03-01')",
        (),
    )
    .unwrap();

    db
}

fn collect_rows(db: &Database, sql: &str) -> Vec<Vec<String>> {
    let mut rows_out = Vec::new();
    let mut rows = db.query(sql, ()).unwrap();
    while let Some(Ok(row)) = rows.next() {
        let mut vals = Vec::new();
        for i in 0..row.len() {
            vals.push(format!("{}", row.get_value(i).unwrap()));
        }
        rows_out.push(vals);
    }
    rows_out
}

#[test]
fn test_distinct_on_basic() {
    let db = setup_db("distinct_on_basic");

    // Get the first order per customer (ordered by order_date ascending)
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount, order_date FROM orders ORDER BY customer, order_date",
    );

    assert_eq!(rows.len(), 3, "Should return exactly one row per customer");
    // Sorted by customer alphabetically, first order_date per customer
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][2], "2024-01-01");
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][2], "2024-01-10");
    assert_eq!(rows[2][0], "Charlie");
    assert_eq!(rows[2][2], "2024-01-20");
}

#[test]
fn test_distinct_on_last_per_group() {
    let db = setup_db("distinct_on_last");

    // Get the most recent order per customer (ordered by order_date descending)
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount, order_date FROM orders ORDER BY customer, order_date DESC",
    );

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][2], "2024-02-01"); // Alice's latest
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][2], "2024-02-05"); // Bob's latest
    assert_eq!(rows[2][0], "Charlie");
    assert_eq!(rows[2][2], "2024-03-01"); // Charlie's latest
}

#[test]
fn test_distinct_on_with_limit() {
    let db = setup_db("distinct_on_limit");

    // DISTINCT ON with LIMIT — should get first 2 customers only
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount FROM orders ORDER BY customer, amount DESC LIMIT 2",
    );

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][1], "200"); // Alice's highest amount
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][1], "300"); // Bob's highest amount
}

#[test]
fn test_distinct_on_with_limit_and_offset() {
    let db = setup_db("distinct_on_limit_offset");

    // DISTINCT ON with LIMIT + OFFSET — skip first customer
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount FROM orders ORDER BY customer, amount LIMIT 2 OFFSET 1",
    );

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], "Bob");
    assert_eq!(rows[0][1], "250"); // Bob's lowest amount
    assert_eq!(rows[1][0], "Charlie");
    assert_eq!(rows[1][1], "50"); // Charlie's lowest amount
}

#[test]
fn test_distinct_on_without_order_by() {
    let db = setup_db("distinct_on_no_order");

    // DISTINCT ON without explicit ORDER BY — should still return one row per customer
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount FROM orders",
    );

    assert_eq!(rows.len(), 3, "Should return exactly one row per customer");
    // Sort to check all three customers are present (order is non-deterministic without ORDER BY)
    let mut customers: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    customers.sort();
    assert_eq!(customers, vec!["Alice", "Bob", "Charlie"]);
}

#[test]
fn test_distinct_on_single_group() {
    let db = setup_db("distinct_on_single");

    // All rows have same customer — should return exactly one row
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount FROM orders WHERE customer = 'Alice' ORDER BY customer, amount DESC",
    );

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][1], "200"); // Highest amount for Alice
}

#[test]
fn test_distinct_on_multiple_keys() {
    let db = Database::open("memory://distinct_on_multi_keys").unwrap();

    db.execute(
        "CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            region TEXT,
            category TEXT,
            amount FLOAT,
            sale_date TEXT
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO sales VALUES (1, 'East', 'Electronics', 500.0, '2024-01-01')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO sales VALUES (2, 'East', 'Electronics', 750.0, '2024-01-15')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO sales VALUES (3, 'East', 'Clothing', 200.0, '2024-01-10')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO sales VALUES (4, 'West', 'Electronics', 600.0, '2024-01-05')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO sales VALUES (5, 'West', 'Clothing', 300.0, '2024-01-20')",
        (),
    )
    .unwrap();

    // DISTINCT ON two columns: get first sale per (region, category)
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (region, category) region, category, amount FROM sales ORDER BY region, category, sale_date",
    );

    assert_eq!(
        rows.len(),
        4,
        "Should return one row per (region, category) pair"
    );
    assert_eq!(&rows[0], &["East", "Clothing", "200"]);
    assert_eq!(&rows[1], &["East", "Electronics", "500"]);
    assert_eq!(&rows[2], &["West", "Clothing", "300"]);
    assert_eq!(&rows[3], &["West", "Electronics", "600"]);
}

#[test]
fn test_distinct_on_with_where_clause() {
    let db = setup_db("distinct_on_where");

    // DISTINCT ON with WHERE filter
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount FROM orders WHERE amount > 100 ORDER BY customer, amount DESC",
    );

    assert_eq!(rows.len(), 3);
    assert_eq!(&rows[0], &["Alice", "200"]);
    assert_eq!(&rows[1], &["Bob", "300"]);
    assert_eq!(&rows[2], &["Charlie", "225"]);
}

#[test]
fn test_distinct_on_empty_result() {
    let db = setup_db("distinct_on_empty");

    // DISTINCT ON with no matching rows
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount FROM orders WHERE amount > 9999 ORDER BY customer",
    );

    assert_eq!(rows.len(), 0);
}

#[test]
fn test_distinct_on_with_expressions_in_select() {
    let db = setup_db("distinct_on_expr");

    // DISTINCT ON with expressions in the SELECT list
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount * 2 AS double_amount FROM orders ORDER BY customer, amount DESC",
    );

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][1], "400"); // 200 * 2
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][1], "600"); // 300 * 2
}

#[test]
fn test_distinct_on_integer_key() {
    let db = Database::open("memory://distinct_on_int_key").unwrap();

    db.execute(
        "CREATE TABLE scores (
            id INTEGER PRIMARY KEY,
            player_id INTEGER,
            score INTEGER,
            game_date TEXT
        )",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO scores VALUES (1, 1, 100, '2024-01-01')", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (2, 1, 250, '2024-01-15')", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (3, 2, 300, '2024-01-10')", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (4, 2, 150, '2024-01-20')", ())
        .unwrap();
    db.execute("INSERT INTO scores VALUES (5, 3, 200, '2024-01-05')", ())
        .unwrap();

    // DISTINCT ON integer column — get highest score per player
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (player_id) player_id, score FROM scores ORDER BY player_id, score DESC",
    );

    assert_eq!(rows.len(), 3);
    assert_eq!(&rows[0], &["1", "250"]);
    assert_eq!(&rows[1], &["2", "300"]);
    assert_eq!(&rows[2], &["3", "200"]);
}

#[test]
fn test_distinct_on_star_select() {
    let db = setup_db("distinct_on_star");

    // DISTINCT ON with SELECT * — all columns should be returned
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) * FROM orders ORDER BY customer, order_date",
    );

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][1], "Alice"); // customer
    assert_eq!(rows[1][1], "Bob");
    assert_eq!(rows[2][1], "Charlie");
}

#[test]
fn test_distinct_on_complex_order_by() {
    let db = setup_db("distinct_on_complex_ob");

    // Complex ORDER BY expression (amount + 0) that triggers expression-evaluation sort path
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount FROM orders ORDER BY customer, amount + 0 DESC",
    );

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][1], "200"); // highest
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][1], "300");
    assert_eq!(rows[2][0], "Charlie");
    assert_eq!(rows[2][1], "225");
}

#[test]
fn test_distinct_on_non_leading_order_by() {
    let db = Database::open("memory://distinct_on_nonlead").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .unwrap();
    // When sorted by val: A(10), B(20), A(30), B(40) — groups are interleaved
    db.execute("INSERT INTO t VALUES (1, 'A', 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 'B', 20)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 'A', 30)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (4, 'B', 40)", ()).unwrap();

    // ORDER BY val does NOT lead with DISTINCT ON column grp
    let rows = collect_rows(&db, "SELECT DISTINCT ON (grp) grp, val FROM t ORDER BY val");

    assert_eq!(rows.len(), 2, "Should return exactly 1 row per group");
    // First row seen per group in val-ascending order: A(10), B(20)
    assert_eq!(&rows[0], &["A", "10"]);
    assert_eq!(&rows[1], &["B", "20"]);
}

#[test]
fn test_distinct_on_key_not_in_select() {
    let db = setup_db("distinct_on_key_not_in_select");

    // DISTINCT ON (customer) but customer is NOT in SELECT list
    // Should still deduplicate by customer; only return amount column
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) amount FROM orders ORDER BY customer, amount DESC",
    );

    assert_eq!(rows.len(), 3, "Should return one row per customer");
    // Alice's highest=200, Bob's highest=300, Charlie's highest=225
    assert_eq!(rows[0], vec!["200"]);
    assert_eq!(rows[1], vec!["300"]);
    assert_eq!(rows[2], vec!["225"]);
}

#[test]
fn test_distinct_on_key_not_in_select_complex_order_by() {
    let db = setup_db("distinct_on_key_not_in_select_complex");

    // DISTINCT ON key not in SELECT + complex ORDER BY expression
    // Exercises both the extra-columns logic and the complex ORDER BY path
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) amount FROM orders ORDER BY customer, amount + 0 DESC",
    );

    assert_eq!(rows.len(), 3, "Should return one row per customer");
    assert_eq!(rows[0], vec!["200"]);
    assert_eq!(rows[1], vec!["300"]);
    assert_eq!(rows[2], vec!["225"]);
}

#[test]
fn test_distinct_on_key_not_in_select_no_order_by() {
    let db = setup_db("distinct_on_key_nosel_no_ob");

    // DISTINCT ON key not in SELECT, no ORDER BY
    let rows = collect_rows(&db, "SELECT DISTINCT ON (customer) amount FROM orders");

    assert_eq!(rows.len(), 3, "Should return one row per customer");
    // Each row should have exactly one column (amount)
    for row in &rows {
        assert_eq!(row.len(), 1, "Should only return the amount column");
    }
}

#[test]
fn test_distinct_on_aliased_key() {
    let db = setup_db("distinct_on_aliased");

    // DISTINCT ON (customer) where customer is aliased as 'c' in SELECT
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer AS c, amount FROM orders ORDER BY customer, amount DESC",
    );

    assert_eq!(rows.len(), 3, "Should return one row per customer");
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][1], "200"); // Alice's highest
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][1], "300"); // Bob's highest
    assert_eq!(rows[2][0], "Charlie");
    assert_eq!(rows[2][1], "225"); // Charlie's highest
}

#[test]
fn test_distinct_on_computed_expression() {
    let db = setup_db("distinct_on_computed");

    // DISTINCT ON with a computed expression
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (amount + 0) customer, amount FROM orders ORDER BY amount + 0 DESC",
    );

    // Each unique amount gets one row; 8 rows have 7 unique amounts
    // (100, 200, 150, 300, 250, 50, 175, 225 — all unique)
    assert_eq!(
        rows.len(),
        8,
        "All amounts are unique, so all 8 rows should be returned"
    );
}

#[test]
fn test_distinct_on_computed_expression_dedup() {
    let db = Database::open("memory://distinct_on_computed_dedup").unwrap();

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT, price FLOAT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO items VALUES (1, 'A', 10.0)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (2, 'B', 10.0)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (3, 'A', 20.0)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (4, 'B', 20.0)", ())
        .unwrap();

    // DISTINCT ON (price * 1) — computed expression, deduplicates by price
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (price * 1) category, price FROM items ORDER BY price * 1",
    );

    assert_eq!(rows.len(), 2, "Should return one row per unique price");
    // price 10: first row is category A (id=1), price 20: first is A (id=3)
    assert_eq!(rows[0][1], "10");
    assert_eq!(rows[1][1], "20");
}

#[test]
fn test_distinct_on_null_keys() {
    let db = Database::open("memory://distinct_on_nulls").unwrap();
    db.execute(
        "CREATE TABLE t_null (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t_null VALUES (1, NULL, 10)", ())
        .unwrap();
    db.execute("INSERT INTO t_null VALUES (2, NULL, 20)", ())
        .unwrap();
    db.execute("INSERT INTO t_null VALUES (3, 'A', 30)", ())
        .unwrap();
    db.execute("INSERT INTO t_null VALUES (4, 'A', 40)", ())
        .unwrap();
    db.execute("INSERT INTO t_null VALUES (5, 'B', 50)", ())
        .unwrap();

    // Two NULLs should be treated as the same group
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (grp) grp, val FROM t_null ORDER BY grp, val",
    );

    assert_eq!(rows.len(), 3, "NULL, A, B — three groups");
    // Check all three groups are represented
    let mut groups: Vec<&str> = rows.iter().map(|r| r[0].as_str()).collect();
    groups.sort();
    assert_eq!(groups, vec!["A", "B", "NULL"]);
}

#[test]
fn test_distinct_on_with_join() {
    let db = Database::open("memory://distinct_on_join").unwrap();

    db.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE purchases (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT, purchase_date TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO customers VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO customers VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute(
        "INSERT INTO purchases VALUES (1, 1, 100.0, '2024-01-01')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO purchases VALUES (2, 1, 200.0, '2024-02-01')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO purchases VALUES (3, 2, 150.0, '2024-01-15')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO purchases VALUES (4, 2, 300.0, '2024-03-01')",
        (),
    )
    .unwrap();

    // DISTINCT ON with JOIN — latest purchase per customer
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (c.name) c.name, p.amount, p.purchase_date \
         FROM customers c JOIN purchases p ON c.id = p.customer_id \
         ORDER BY c.name, p.purchase_date DESC",
    );

    assert_eq!(rows.len(), 2, "One row per customer");
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][1], "200"); // Alice's latest
    assert_eq!(rows[0][2], "2024-02-01");
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][1], "300"); // Bob's latest
    assert_eq!(rows[1][2], "2024-03-01");
}

#[test]
fn test_distinct_on_with_cte() {
    let db = setup_db("distinct_on_cte");

    // DISTINCT ON inside a CTE
    let rows = collect_rows(
        &db,
        "WITH latest_orders AS ( \
             SELECT DISTINCT ON (customer) customer, amount, order_date \
             FROM orders ORDER BY customer, order_date DESC \
         ) SELECT customer, amount FROM latest_orders ORDER BY customer",
    );

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][1], "150"); // Alice's latest (2024-02-01)
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][1], "250"); // Bob's latest (2024-02-05)
    assert_eq!(rows[2][0], "Charlie");
    assert_eq!(rows[2][1], "225"); // Charlie's latest (2024-03-01)
}

#[test]
fn test_distinct_on_with_subquery() {
    let db = setup_db("distinct_on_subquery");

    // DISTINCT ON in a subquery
    let rows = collect_rows(
        &db,
        "SELECT customer, amount FROM ( \
             SELECT DISTINCT ON (customer) customer, amount \
             FROM orders ORDER BY customer, amount DESC \
         ) sub ORDER BY customer",
    );

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], vec!["Alice", "200"]);
    assert_eq!(rows[1], vec!["Bob", "300"]);
    assert_eq!(rows[2], vec!["Charlie", "225"]);
}

#[test]
fn test_distinct_on_pushdown_does_not_fire() {
    let db = Database::open("memory://distinct_on_no_pushdown").unwrap();

    db.execute(
        "CREATE TABLE indexed_t (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_cat ON indexed_t(category)", ())
        .unwrap();

    db.execute("INSERT INTO indexed_t VALUES (1, 'A', 10)", ())
        .unwrap();
    db.execute("INSERT INTO indexed_t VALUES (2, 'A', 20)", ())
        .unwrap();
    db.execute("INSERT INTO indexed_t VALUES (3, 'B', 30)", ())
        .unwrap();
    db.execute("INSERT INTO indexed_t VALUES (4, 'B', 40)", ())
        .unwrap();

    // Without DISTINCT ON, plain DISTINCT on indexed column could use pushdown.
    // With DISTINCT ON (category), we must NOT use distinct pushdown — we want
    // one row per category, not distinct values of val.
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (category) val FROM indexed_t ORDER BY category, val DESC",
    );

    assert_eq!(rows.len(), 2, "One row per category");
    assert_eq!(rows[0], vec!["20"]); // A's highest val
    assert_eq!(rows[1], vec!["40"]); // B's highest val
}

#[test]
fn test_distinct_on_alias_shadow_ambiguity() {
    let db = Database::open("memory://distinct_on_alias_shadow").unwrap();

    db.execute(
        "CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT, val INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t1 VALUES (1, 'Alice', 10)", ())
        .unwrap();
    db.execute("INSERT INTO t1 VALUES (2, 'Alice', 20)", ())
        .unwrap();
    db.execute("INSERT INTO t1 VALUES (3, 'Bob', 30)", ())
        .unwrap();

    // DISTINCT ON (name) where name is aliased as 'n' — should still dedup by name
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (name) name AS n, val FROM t1 ORDER BY name, val DESC",
    );

    assert_eq!(rows.len(), 2, "Should return one row per name");
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][1], "20"); // Alice's highest
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][1], "30");

    // DISTINCT ON (name) where a different column is also aliased as 'name'
    // This tests the ambiguity: val is aliased to 'name', but DISTINCT ON (name)
    // should resolve to the original 'name' column, not the alias
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (name) name, val AS name_alias FROM t1 ORDER BY name, val DESC",
    );

    assert_eq!(rows.len(), 2, "Should still dedup by original name column");
    assert_eq!(rows[0][0], "Alice");
    assert_eq!(rows[0][1], "20");
    assert_eq!(rows[1][0], "Bob");
    assert_eq!(rows[1][1], "30");
}

#[test]
fn test_distinct_on_qualified_key_not_in_select_join() {
    let db = Database::open("memory://distinct_on_qual_key_join").unwrap();

    db.execute(
        "CREATE TABLE customers2 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE purchases2 (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO customers2 VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO customers2 VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO purchases2 VALUES (1, 1, 100.0)", ())
        .unwrap();
    db.execute("INSERT INTO purchases2 VALUES (2, 1, 200.0)", ())
        .unwrap();
    db.execute("INSERT INTO purchases2 VALUES (3, 2, 150.0)", ())
        .unwrap();
    db.execute("INSERT INTO purchases2 VALUES (4, 2, 300.0)", ())
        .unwrap();

    // DISTINCT ON (c.name) where c.name is NOT in SELECT — only p.amount is selected.
    // The qualified key must be materialized as an extra column for dedup.
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (c.name) p.amount \
         FROM customers2 c JOIN purchases2 p ON c.id = p.customer_id \
         ORDER BY c.name, p.amount DESC",
    );

    assert_eq!(rows.len(), 2, "One row per customer");
    // Alice's highest = 200, Bob's highest = 300
    assert_eq!(rows[0], vec!["200"]);
    assert_eq!(rows[1], vec!["300"]);
}

#[test]
fn test_distinct_on_qualified_key_same_column_name_join() {
    let db = Database::open("memory://distinct_on_qual_same_name").unwrap();

    db.execute(
        "CREATE TABLE customers3 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE purchases3 (id INTEGER PRIMARY KEY, customer_id INTEGER, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO customers3 VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO customers3 VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO purchases3 VALUES (1, 1, 'X')", ())
        .unwrap();
    db.execute("INSERT INTO purchases3 VALUES (2, 1, 'Y')", ())
        .unwrap();
    db.execute("INSERT INTO purchases3 VALUES (3, 2, 'Z')", ())
        .unwrap();

    // Both tables have a "name" column. DISTINCT ON (c.name) must bind to customers3.name,
    // NOT purchases3.name. Should return one row per customer (Alice, Bob).
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (c.name) p.name \
         FROM customers3 c JOIN purchases3 p ON c.id = p.customer_id \
         ORDER BY c.name, p.name DESC",
    );

    assert_eq!(rows.len(), 2, "One row per customer");
    // Alice (c.name='Alice'): purchases Y, X -> DESC -> first is Y
    // Bob (c.name='Bob'): purchase Z -> Z
    assert_eq!(rows[0], vec!["Y"]);
    assert_eq!(rows[1], vec!["Z"]);
}

#[test]
fn test_distinct_on_qualified_key_both_selected_join() {
    // Regression: DISTINCT ON (c.name) when both c.name and p.name are in SELECT.
    // Both project as bare "name" in result_columns, so resolve_distinct_on_indices
    // must fall back to select_exprs to find the correct column position.
    let db = Database::open("memory://distinct_on_qual_both_sel").unwrap();

    db.execute(
        "CREATE TABLE customers5 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE purchases5 (id INTEGER PRIMARY KEY, customer_id INTEGER, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO customers5 VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO customers5 VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO purchases5 VALUES (1, 1, 'X')", ())
        .unwrap();
    db.execute("INSERT INTO purchases5 VALUES (2, 1, 'Y')", ())
        .unwrap();
    db.execute("INSERT INTO purchases5 VALUES (3, 2, 'Z')", ())
        .unwrap();

    // Both c.name and p.name selected — dedup must be on c.name (column 0), not p.name
    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (c.name) c.name, p.name \
         FROM customers5 c JOIN purchases5 p ON c.id = p.customer_id \
         ORDER BY c.name, p.name DESC",
    );

    assert_eq!(rows.len(), 2, "One row per customer");
    assert_eq!(rows[0], vec!["Alice", "Y"]); // Alice's highest p.name DESC
    assert_eq!(rows[1], vec!["Bob", "Z"]);
}

#[test]
fn test_distinct_on_qualified_key_aliased_join() {
    // Regression: DISTINCT ON (c.name) when c.name is aliased as "customer".
    // result_columns has "customer", not "c.name", so resolve_distinct_on_indices
    // must check select_exprs to find the aliased source.
    let db = Database::open("memory://distinct_on_qual_alias").unwrap();

    db.execute(
        "CREATE TABLE customers6 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE purchases6 (id INTEGER PRIMARY KEY, customer_id INTEGER, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO customers6 VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO customers6 VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO purchases6 VALUES (1, 1, 'X')", ())
        .unwrap();
    db.execute("INSERT INTO purchases6 VALUES (2, 1, 'Y')", ())
        .unwrap();
    db.execute("INSERT INTO purchases6 VALUES (3, 2, 'Z')", ())
        .unwrap();

    let rows = collect_rows(
        &db,
        "SELECT DISTINCT ON (c.name) c.name AS customer, p.name \
         FROM customers6 c JOIN purchases6 p ON c.id = p.customer_id \
         ORDER BY c.name, p.name DESC",
    );

    assert_eq!(rows.len(), 2, "One row per customer");
    assert_eq!(rows[0], vec!["Alice", "Y"]);
    assert_eq!(rows[1], vec!["Bob", "Z"]);
}

#[test]
fn test_qualified_order_by_extra_columns_join() {
    // Regression: ORDER BY c.name (QualifiedIdentifier not in SELECT) must be
    // materialized as an extra column so the sort sees c.name, not just p.amount.
    let db = Database::open("memory://qual_ob_extra_join").unwrap();

    db.execute(
        "CREATE TABLE customers4 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE purchases4 (id INTEGER PRIMARY KEY, customer_id INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO customers4 VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO customers4 VALUES (2, 'Bob')", ())
        .unwrap();
    db.execute("INSERT INTO purchases4 VALUES (1, 1, 100.0)", ())
        .unwrap();
    db.execute("INSERT INTO purchases4 VALUES (2, 1, 200.0)", ())
        .unwrap();
    db.execute("INSERT INTO purchases4 VALUES (3, 2, 150.0)", ())
        .unwrap();
    db.execute("INSERT INTO purchases4 VALUES (4, 2, 300.0)", ())
        .unwrap();

    // ORDER BY c.name ASC, p.amount DESC — c.name is NOT in SELECT
    let rows = collect_rows(
        &db,
        "SELECT p.amount \
         FROM customers4 c JOIN purchases4 p ON c.id = p.customer_id \
         ORDER BY c.name, p.amount DESC",
    );

    // Alice(200, 100), Bob(300, 150)
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0], vec!["200"]);
    assert_eq!(rows[1], vec!["100"]);
    assert_eq!(rows[2], vec!["300"]);
    assert_eq!(rows[3], vec!["150"]);
}

#[test]
fn test_distinct_on_classification_cache_separation() {
    let db = setup_db("distinct_on_cache_sep");

    // Run plain DISTINCT first — this populates the classification cache
    let rows1 = collect_rows(
        &db,
        "SELECT DISTINCT customer, amount FROM orders ORDER BY customer LIMIT 2",
    );
    // Should return 2 distinct (customer, amount) pairs
    assert_eq!(rows1.len(), 2);

    // Now run DISTINCT ON with the same columns — must NOT reuse the cached
    // classification from the plain DISTINCT query
    let rows2 = collect_rows(
        &db,
        "SELECT DISTINCT ON (customer) customer, amount FROM orders ORDER BY customer, amount DESC LIMIT 2",
    );

    assert_eq!(rows2.len(), 2, "Should return first 2 customers");
    assert_eq!(rows2[0][0], "Alice");
    assert_eq!(rows2[0][1], "200"); // Alice's highest
    assert_eq!(rows2[1][0], "Bob");
    assert_eq!(rows2[1][1], "300"); // Bob's highest
}
