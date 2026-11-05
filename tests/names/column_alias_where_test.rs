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

//! Column Alias in WHERE Clause Tests
//!
//! Tests column alias usage in WHERE clauses

use stoolap::Database;

fn setup_products_db(test_name: &str) -> Database {
    let db = Database::open(&format!("memory://col_alias_where_{}", test_name))
        .expect("Failed to create database");

    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            price INTEGER,
            name TEXT
        )",
        (),
    )
    .expect("Failed to create table");

    db.execute(
        "INSERT INTO products (id, price, name) VALUES (1, 50, 'Widget')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO products (id, price, name) VALUES (2, 150, 'Gadget')",
        (),
    )
    .expect("Failed to insert data");
    db.execute(
        "INSERT INTO products (id, price, name) VALUES (3, 80, 'Thing')",
        (),
    )
    .expect("Failed to insert data");

    db
}

/// Test using column alias in WHERE clause
#[test]
fn test_alias_in_where_clause() {
    let db = setup_products_db("alias_where");

    let result = db
        .query("SELECT price AS cost FROM products WHERE cost > 100", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();

        assert_eq!(cost, 150, "Expected cost = 150");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row matching WHERE cost > 100");
}

/// Test multiple aliases in SELECT and WHERE
#[test]
fn test_multiple_aliases_in_where() {
    let db = setup_products_db("multi_alias");

    let result = db
        .query(
            "SELECT id, price AS cost, name AS product_name
            FROM products
            WHERE cost > 60 AND product_name = 'Thing'",
            (),
        )
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let cost: i64 = row.get(1).unwrap();
        let product_name: String = row.get(2).unwrap();

        assert_eq!(id, 3, "Expected id = 3");
        assert_eq!(cost, 80, "Expected cost = 80");
        assert_eq!(product_name, "Thing", "Expected product_name = 'Thing'");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row matching complex condition");
}

/// Test alias without WHERE (should work)
#[test]
fn test_alias_without_where() {
    let db = setup_products_db("without_where");

    let result = db
        .query("SELECT price AS cost FROM products ORDER BY cost", ())
        .expect("Failed to execute query");

    let mut costs: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();
        costs.push(cost);
    }

    assert_eq!(costs, vec![50, 80, 150], "Expected sorted costs");
}

/// Test alias with ORDER BY (should work)
#[test]
fn test_alias_with_order_by() {
    let db = setup_products_db("order_by");

    let result = db
        .query("SELECT price AS cost FROM products ORDER BY cost DESC", ())
        .expect("Failed to execute query");

    let mut costs: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();
        costs.push(cost);
    }

    assert_eq!(costs, vec![150, 80, 50], "Expected descending sorted costs");
}

/// Test original column in WHERE, alias in SELECT
#[test]
fn test_original_column_in_where() {
    let db = setup_products_db("orig_where");

    let result = db
        .query("SELECT price AS cost FROM products WHERE price > 100", ())
        .expect("Failed to execute query");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let cost: i64 = row.get(0).unwrap();

        assert_eq!(cost, 150, "Expected cost = 150");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row matching WHERE price > 100");
}

/// Test expression alias in SELECT
#[test]
fn test_expression_alias_in_select() {
    let db = setup_products_db("expr_alias");

    let result = db
        .query(
            "SELECT price * 2 AS double_price FROM products ORDER BY double_price",
            (),
        )
        .expect("Failed to execute query");

    let mut prices: Vec<i64> = Vec::new();
    for row in result {
        let row = row.expect("Failed to get row");
        let double_price: i64 = row.get(0).unwrap();
        prices.push(double_price);
    }

    assert_eq!(
        prices,
        vec![100, 160, 300],
        "Expected doubled prices sorted"
    );
}

/// Test alias with aggregate function
#[test]
fn test_alias_with_aggregate() {
    let db = setup_products_db("aggregate");

    let total: i64 = db
        .query_one("SELECT SUM(price) AS total_price FROM products", ())
        .expect("Failed to execute query");

    // 50 + 150 + 80 = 280
    assert_eq!(total, 280, "Expected total price = 280");
}

/// Test alias with COUNT
#[test]
fn test_alias_with_count() {
    let db = setup_products_db("count");

    let count: i64 = db
        .query_one("SELECT COUNT(*) AS product_count FROM products", ())
        .expect("Failed to execute query");

    assert_eq!(count, 3, "Expected product count = 3");
}

#[test]
fn test_alias_on_the_left_of_a_subquery_comparison() {
    let db = setup_products_db("alias_subquery_left");
    db.execute(
        "CREATE TABLE limits (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO limits VALUES (1, 60), (2, 100)", ())
        .unwrap();
    let ids = |sql: &str, floor: i64| -> Vec<i64> {
        db.query(sql, (floor,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    };
    for (test, want) in [
        ("cost > ANY (SELECT v FROM limits WHERE v > $1)", vec![2, 3]),
        ("cost > ALL (SELECT v FROM limits WHERE v > $1)", vec![2]),
        ("cost IN (SELECT v + 50 FROM limits WHERE v > $1)", vec![2]),
    ] {
        for from in ["products", "(SELECT * FROM products) p"] {
            let sql = format!("SELECT id, price AS cost FROM {from} WHERE {test} ORDER BY id");
            assert_eq!(ids(&sql, 0), want, "{sql}");
            assert_eq!(
                ids(&sql, 0),
                ids(&sql.replace("WHERE cost", "WHERE price"), 0),
                "{sql}: the alias reads its column"
            );
        }
    }
}

#[test]
fn where_reads_a_select_alias_over_every_source_a_from_column_winning() {
    let db = Database::open("memory://col_alias_where_every_source").unwrap();
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 2)",
        "CREATE VIEW v AS SELECT x FROM u",
        "CREATE TABLE t (id INTEGER PRIMARY KEY)",
        "INSERT INTO t VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let values = |sql: &str| -> Vec<i64> {
        db.query(sql, ())
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    };
    let prepared = |sql: &str| -> Vec<i64> {
        db.prepare(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .query(())
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    };
    for (sql, want) in [
        ("SELECT x + 1 AS k FROM u WHERE k < 5", vec![3]),
        ("SELECT v.x AS k FROM v WHERE k < 5", vec![2]),
        (
            "SELECT q.x AS k FROM (SELECT x FROM u) AS q WHERE k < 5",
            vec![2],
        ),
        (
            "SELECT column1 AS k FROM (VALUES (1)) AS w WHERE k < 5",
            vec![1],
        ),
        (
            "SELECT value AS k FROM generate_series(1, 2) WHERE k < 5",
            vec![1, 2],
        ),
        (
            "WITH c AS (SELECT x FROM u) SELECT x AS k FROM c WHERE k < 5",
            vec![2],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT v.x AS k FROM v WHERE k < 5 AND v.x = t.id + 1)",
            vec![1],
        ),
        ("SELECT x AS id FROM u WHERE id = 2", vec![]),
        (
            "SELECT q.x AS id FROM (SELECT id, x FROM u) AS q WHERE id = 2",
            vec![],
        ),
    ] {
        assert_eq!(values(sql), want, "{sql}");
        assert_eq!(prepared(sql), want, "prepared: {sql}");
    }
}
