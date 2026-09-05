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

//! EXISTS Subquery Tests
//!
//! Tests EXISTS and NOT EXISTS subqueries

use stoolap::Database;

fn setup_exists_tables(db: &Database) {
    // Create customers table
    db.execute(
        "CREATE TABLE customers (
            id INTEGER,
            name TEXT,
            country TEXT
        )",
        (),
    )
    .expect("Failed to create customers table");

    // Create orders table
    db.execute(
        "CREATE TABLE orders (
            id INTEGER,
            customer_id INTEGER,
            amount FLOAT
        )",
        (),
    )
    .expect("Failed to create orders table");

    // Insert customers
    db.execute(
        "INSERT INTO customers (id, name, country) VALUES
        (1, 'Alice', 'USA'),
        (2, 'Bob', 'UK'),
        (3, 'Charlie', 'USA'),
        (4, 'David', 'Canada')",
        (),
    )
    .expect("Failed to insert customers");

    // Insert orders
    db.execute(
        "INSERT INTO orders (id, customer_id, amount) VALUES
        (1, 1, 100.0),
        (2, 1, 200.0),
        (3, 3, 150.0),
        (4, 4, 300.0)",
        (),
    )
    .expect("Failed to insert orders");
}

/// Test EXISTS with subquery that returns rows
#[test]
fn test_exists_with_results() {
    let db = Database::open("memory://exists_results").expect("Failed to create database");
    setup_exists_tables(&db);

    // EXISTS should return all customers when orders exist
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE EXISTS (SELECT * FROM orders)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 4, "Expected 4 customers when orders exist");
}

/// Test EXISTS with empty subquery
#[test]
fn test_exists_with_no_results() {
    let db = Database::open("memory://exists_empty").expect("Failed to create database");
    setup_exists_tables(&db);

    // Delete all orders
    db.execute("DELETE FROM orders", ())
        .expect("Failed to delete orders");

    // EXISTS should return no customers when no orders exist
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE EXISTS (SELECT 1 FROM orders)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 0, "Expected 0 customers when no orders exist");
}

/// Test NOT EXISTS with empty subquery
#[test]
fn test_not_exists_with_no_results() {
    let db = Database::open("memory://not_exists_empty").expect("Failed to create database");
    setup_exists_tables(&db);

    // Delete all orders
    db.execute("DELETE FROM orders", ())
        .expect("Failed to delete orders");

    // NOT EXISTS should return all customers when no orders exist
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE NOT EXISTS (SELECT * FROM orders)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    assert_eq!(count, 4, "Expected 4 customers when no orders exist");
}

/// Test EXISTS with condition in subquery
#[test]
fn test_exists_with_condition() {
    let db = Database::open("memory://exists_condition").expect("Failed to create database");
    setup_exists_tables(&db);

    // EXISTS with WHERE condition in subquery
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE EXISTS (SELECT * FROM orders WHERE amount > 150)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    // Since there is at least one order > 150, all customers should be returned
    assert_eq!(count, 4, "Expected 4 customers (order > 150 exists)");
}

/// Test NOT EXISTS with condition in subquery
#[test]
fn test_not_exists_with_condition() {
    let db = Database::open("memory://not_exists_condition").expect("Failed to create database");
    setup_exists_tables(&db);

    // NOT EXISTS with condition that never matches
    let result = db
        .query(
            "SELECT id, name FROM customers
             WHERE NOT EXISTS (SELECT * FROM orders WHERE amount > 500)
             ORDER BY id",
            (),
        )
        .expect("Failed to query");

    let mut count = 0;
    for row in result {
        let _row = row.expect("Failed to get row");
        count += 1;
    }

    // Since no order > 500, all customers should be returned
    assert_eq!(count, 4, "Expected 4 customers (no order > 500)");
}

/// Test DELETE with EXISTS
#[test]
fn test_delete_with_exists() {
    let db = Database::open("memory://delete_exists").expect("Failed to create database");

    // Create products and inventory tables
    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            name TEXT,
            in_stock BOOLEAN
        )",
        (),
    )
    .expect("Failed to create products table");

    db.execute(
        "CREATE TABLE inventory (
            id INTEGER,
            product_id INTEGER,
            quantity INTEGER
        )",
        (),
    )
    .expect("Failed to create inventory table");

    // Insert products
    db.execute(
        "INSERT INTO products (id, name, in_stock) VALUES
        (1, 'Laptop', true),
        (2, 'Mouse', true),
        (3, 'Book', true),
        (4, 'Phone', true)",
        (),
    )
    .expect("Failed to insert products");

    // Empty inventory - no rows
    // DELETE with EXISTS on empty table should not delete anything
    db.execute(
        "DELETE FROM products
         WHERE EXISTS (SELECT * FROM inventory)",
        (),
    )
    .expect("Failed to execute DELETE with EXISTS");

    // All 4 products should remain
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM products", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected 4 products (no deletions)");
}

/// Test UPDATE with EXISTS
#[test]
fn test_update_with_exists() {
    let db = Database::open("memory://update_exists").expect("Failed to create database");

    // Create products and inventory tables
    db.execute(
        "CREATE TABLE products (
            id INTEGER,
            name TEXT,
            in_stock BOOLEAN
        )",
        (),
    )
    .expect("Failed to create products table");

    db.execute(
        "CREATE TABLE inventory (
            id INTEGER,
            product_id INTEGER,
            quantity INTEGER
        )",
        (),
    )
    .expect("Failed to create inventory table");

    // Insert products
    db.execute(
        "INSERT INTO products (id, name, in_stock) VALUES
        (1, 'Laptop', true),
        (2, 'Mouse', true),
        (3, 'Book', true),
        (4, 'Phone', true)",
        (),
    )
    .expect("Failed to insert products");

    // Add one inventory record
    db.execute(
        "INSERT INTO inventory (id, product_id, quantity) VALUES (1, 1, 10)",
        (),
    )
    .expect("Failed to insert inventory");

    // UPDATE with EXISTS - should update all products since inventory exists
    db.execute(
        "UPDATE products
         SET in_stock = false
         WHERE EXISTS (SELECT 1 FROM inventory)",
        (),
    )
    .expect("Failed to execute UPDATE with EXISTS");

    // All 4 products should be marked as out of stock
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM products WHERE in_stock = false", ())
        .expect("Failed to count");
    assert_eq!(count, 4, "Expected all 4 products to be out of stock");
}

/// A subquery that groups its rows and then keeps only some of the groups
/// is asked whether a group survived, not whether a row matched
#[test]
fn test_exists_over_a_subquery_that_groups_and_filters() {
    let db = Database::open("memory://exists_group_having").unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, g TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE e (id INTEGER PRIMARY KEY, d_id INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO d VALUES (1,'a'),(2,'a'),(3,'b')", ())
        .unwrap();
    db.execute("INSERT INTO e VALUES (1,1),(2,1),(3,2)", ())
        .unwrap();

    let count = |sql: &str| -> i64 {
        db.query(sql, ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap()
    };

    assert_eq!(
        count(
            "SELECT COUNT(*) FROM d WHERE EXISTS (SELECT 1 FROM e WHERE e.d_id = d.id GROUP BY e.d_id HAVING COUNT(*) > 1)"
        ),
        1,
        "only the row with two children"
    );
    assert_eq!(
        count(
            "SELECT COUNT(*) FROM d WHERE EXISTS (SELECT 1 FROM e WHERE e.d_id = d.id GROUP BY e.d_id HAVING COUNT(*) > 99)"
        ),
        0
    );
    assert_eq!(
        count(
            "SELECT COUNT(*) FROM d WHERE NOT EXISTS (SELECT 1 FROM e WHERE e.d_id = d.id GROUP BY e.d_id HAVING COUNT(*) > 1)"
        ),
        2
    );
    assert_eq!(
        count("SELECT COUNT(*) FROM d WHERE EXISTS (SELECT 1 FROM e WHERE e.d_id = d.id GROUP BY e.d_id)"),
        2,
        "grouping alone does not change whether a row is there"
    );
}

/// A bare aggregate returns its one row whether or not anything matched
#[test]
fn test_exists_over_a_bare_aggregate() {
    let db = Database::open("memory://exists_bare_aggregate").unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE e (id INTEGER PRIMARY KEY, d_id INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO d VALUES (1),(2),(3)", ()).unwrap();
    db.execute("INSERT INTO e VALUES (1,1),(2,1)", ()).unwrap();

    let count = |sql: &str| -> i64 {
        db.query(sql, ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap()
    };

    assert_eq!(
        count("SELECT COUNT(*) FROM d WHERE EXISTS (SELECT COUNT(*) FROM e WHERE e.d_id = d.id)"),
        3
    );
    assert_eq!(
        count(
            "SELECT COUNT(*) FROM d WHERE NOT EXISTS (SELECT COUNT(*) FROM e WHERE e.d_id = d.id)"
        ),
        0
    );
    assert_eq!(
        count("SELECT COUNT(*) FROM d WHERE EXISTS (SELECT 1 FROM e WHERE e.d_id = d.id LIMIT 0)"),
        0
    );
}

/// An index on the correlation column offers a shorter road to the same
/// question, and it is only the same question under the same conditions
#[test]
fn test_exists_reads_the_same_with_an_index_on_the_correlation() {
    for indexed in [false, true] {
        let name = if indexed {
            "exists_guard_indexed"
        } else {
            "exists_guard_plain"
        };
        let db = Database::open(&format!("memory://{name}")).unwrap();
        db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("CREATE TABLE e (id INTEGER PRIMARY KEY, d_id INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO d VALUES (1),(2),(3)", ()).unwrap();
        db.execute("INSERT INTO e VALUES (1,1),(2,1),(3,2)", ())
            .unwrap();
        if indexed {
            db.execute("CREATE INDEX ed ON e (d_id)", ()).unwrap();
        }

        let count = |sql: &str| -> i64 {
            db.query(sql, ())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .get(0)
                .unwrap()
        };

        assert_eq!(
            count(
                "SELECT COUNT(*) FROM d WHERE EXISTS (SELECT 1 FROM e WHERE e.d_id = d.id LIMIT 0)"
            ),
            0,
            "indexed: {indexed}"
        );
        assert_eq!(
            count("SELECT COUNT(*) FROM d WHERE EXISTS (SELECT 1 FROM e WHERE e.d_id = d.id GROUP BY e.d_id HAVING COUNT(*) > 1)"),
            1,
            "indexed: {indexed}"
        );
        assert_eq!(
            count(
                "SELECT COUNT(*) FROM d WHERE EXISTS (SELECT COUNT(*) FROM e WHERE e.d_id = d.id)"
            ),
            3,
            "indexed: {indexed}"
        );
        assert_eq!(
            count("SELECT COUNT(*) FROM d WHERE EXISTS (SELECT 1 FROM e WHERE e.d_id = d.id)"),
            2,
            "indexed: {indexed}"
        );
    }
}

/// No inner row equals a NULL, so EXISTS over one is false and NOT EXISTS
/// is true; the set the rewrite reads must not turn that into unknown
#[test]
fn test_not_exists_over_a_null_outer_value() {
    for keyed in [true, false] {
        let db = Database::open(&format!(
            "memory://not_exists_null_outer_{}",
            if keyed { "pk" } else { "nopk" }
        ))
        .unwrap();
        let pk = if keyed { "PRIMARY KEY" } else { "" };
        db.execute(&format!("CREATE TABLE t (id INTEGER {pk}, k INTEGER)"), ())
            .unwrap();
        db.execute("CREATE TABLE r (k INTEGER)", ()).unwrap();
        db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, NULL), (4, 9)", ())
            .unwrap();
        db.execute("INSERT INTO r VALUES (1), (2), (NULL)", ())
            .unwrap();

        let count = |sql: &str| -> i64 {
            db.query(sql, ())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .get(0)
                .unwrap()
        };
        assert_eq!(
            count("SELECT COUNT(*) FROM t WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.k = t.k)"),
            2,
            "the NULL row and the 9 row, keyed: {keyed}"
        );
        assert_eq!(
            count("SELECT COUNT(*) FROM t WHERE EXISTS (SELECT 1 FROM r WHERE r.k = t.k)"),
            2,
            "keyed: {keyed}"
        );
        let removed = db
            .execute(
                "DELETE FROM t WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.k = t.k)",
                (),
            )
            .unwrap();
        assert_eq!(removed, 2, "keyed: {keyed}");
        assert_eq!(count("SELECT COUNT(*) FROM t"), 2, "keyed: {keyed}");
    }
}

#[test]
fn test_having_binds_the_group_to_a_correlated_exists() {
    let db = Database::open("memory://having_correlated_exists").unwrap();
    db.execute("CREATE TABLE hx (a INTEGER, b INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE hy (a INTEGER)", ()).unwrap();
    db.execute(
        "INSERT INTO hx VALUES (1, 10), (1, 11), (2, 20), (3, 30), (NULL, 40)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO hy VALUES (1), (3), (NULL)", ())
        .unwrap();

    let groups = |having: &str| -> Vec<i64> {
        db.query(
            &format!("SELECT COALESCE(a, -1) FROM hx GROUP BY a HAVING {having} ORDER BY a"),
            (),
        )
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
    };
    assert_eq!(
        groups("EXISTS (SELECT 1 FROM hy WHERE hy.a = hx.a)"),
        vec![1, 3]
    );
    assert_eq!(
        groups("NOT EXISTS (SELECT 1 FROM hy WHERE hy.a = hx.a)"),
        vec![2, -1]
    );
    assert_eq!(
        groups("COUNT(*) = 1 AND EXISTS (SELECT 1 FROM hy WHERE hy.a = hx.a)"),
        vec![3]
    );
}

#[test]
fn test_update_binds_each_row_to_a_where_the_semi_join_rewrite_refused() {
    for keyed in [true, false] {
        let db =
            Database::open(&format!("memory://update_two_conjunct_not_exists_{keyed}")).unwrap();
        let pk = if keyed { "PRIMARY KEY" } else { "" };
        db.execute(
            &format!("CREATE TABLE uy (id INTEGER {pk}, a INTEGER, s TEXT, b INTEGER)"),
            (),
        )
        .unwrap();
        db.execute("CREATE TABLE ux (a INTEGER, s TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO ux VALUES (1, 'p'), (2, 'q')", ())
            .unwrap();
        db.execute(
            "INSERT INTO uy VALUES (1, 1, 'p', 1), (2, 1, 'q', 1), (3, 2, 'q', 1), (4, 3, 'p', 1), (5, NULL, 'p', 1)",
            (),
        )
        .unwrap();
        let count = |sql: &str| -> i64 {
            db.query(sql, ())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .get(0)
                .unwrap()
        };

        let changed = db
            .execute(
                "UPDATE uy SET b = 0 WHERE NOT EXISTS (SELECT 1 FROM ux WHERE ux.a = uy.a AND ux.s = uy.s)",
                (),
            )
            .unwrap();
        assert_eq!(changed, 3, "keyed: {keyed}");
        assert_eq!(
            count("SELECT COUNT(*) FROM uy WHERE b = 0"),
            3,
            "keyed: {keyed}"
        );

        let changed = db
            .execute(
                "UPDATE uy SET b = 5 WHERE EXISTS (SELECT 1 FROM ux WHERE ux.a = uy.a AND ux.s = uy.s)",
                (),
            )
            .unwrap();
        assert_eq!(changed, 2, "keyed: {keyed}");

        // a correlated SET alongside the correlated WHERE
        let changed = db
            .execute(
                "UPDATE uy SET b = (SELECT COUNT(*) FROM ux WHERE ux.a = uy.a) WHERE NOT EXISTS (SELECT 1 FROM ux WHERE ux.a = uy.a AND ux.s = uy.s)",
                (),
            )
            .unwrap();
        assert_eq!(changed, 3, "keyed: {keyed}");
        assert_eq!(count("SELECT SUM(b) FROM uy"), 11, "keyed: {keyed}");
    }
}
