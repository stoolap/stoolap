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

//! Rust-only benchmark matching Go's benchmark format
//!
//! Run with: cargo run --release --example rust_benchmark

use rand::Rng;
use std::time::Instant;
use stoolap::Database;

const ROW_COUNT: usize = 10_000;
const ITERATIONS: usize = 100;

fn main() {
    println!("Starting Stoolap-Rust benchmark...");
    println!(
        "Configuration: {} rows, {} iterations per test\n",
        ROW_COUNT, ITERATIONS
    );

    let mut rng = rand::rng();
    let db = Database::open("memory://").unwrap();

    // Create schema
    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER NOT NULL,
            balance REAL NOT NULL,
            active BOOLEAN NOT NULL,
            created_at TEXT NOT NULL
        )",
        (),
    )
    .unwrap();

    db.execute("CREATE INDEX idx_users_age ON users(age)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_users_active ON users(active)", ())
        .unwrap();

    // Populate using prepared statement
    let insert_stmt = db
        .prepare("INSERT INTO users (id, name, email, age, balance, active, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)")
        .unwrap();

    for i in 1..=ROW_COUNT {
        let age = rng.random_range(18..80);
        let balance = rng.random_range(0.0..100000.0);
        let active = rng.random_bool(0.7);
        let name = format!("User_{}", i);
        let email = format!("user{}@example.com", i);
        insert_stmt
            .execute((
                i as i64,
                &name,
                &email,
                age,
                balance,
                active,
                "2024-01-01 00:00:00",
            ))
            .unwrap();
    }

    println!("Benchmarking Stoolap-Rust...\n");
    println!("============================================================");
    println!("STOOLAP-RUST BENCHMARK (10,000 rows, 100 iterations, in-memory)");
    println!("============================================================\n");
    println!(
        "{:<25} | {:>12} | {:>12}",
        "Operation", "Avg (μs)", "ops/sec"
    );
    println!("------------------------------------------------------------");

    // SELECT by ID (prepared statement)
    let select_by_id = db.prepare("SELECT * FROM users WHERE id = $1").unwrap();
    let mut total = std::time::Duration::ZERO;
    for i in 0..ITERATIONS {
        let id = ((i % ROW_COUNT) + 1) as i64;
        let start = Instant::now();
        let rows = select_by_id.query((id,)).unwrap();
        let _ = rows.into_iter().next();
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "SELECT by ID",
        avg_us,
        1_000_000.0 / avg_us
    );

    // SELECT complex (prepared statement)
    let select_complex = db
        .prepare("SELECT id, name, balance FROM users WHERE age >= 25 AND age <= 45 AND active = true ORDER BY balance DESC LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let rows = select_complex.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "SELECT complex",
        avg_us,
        1_000_000.0 / avg_us
    );

    // SELECT * (full scan) (prepared statement)
    let select_all = db.prepare("SELECT * FROM users").unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let rows = select_all.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "SELECT * (full scan)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // UPDATE by ID (prepared statement)
    let update_by_id = db
        .prepare("UPDATE users SET balance = $1 WHERE id = $2")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for i in 0..ITERATIONS {
        let id = ((i % ROW_COUNT) + 1) as i64;
        let new_balance: f64 = rng.random_range(0.0..100000.0);
        let start = Instant::now();
        update_by_id.execute((new_balance, id)).unwrap();
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "UPDATE by ID",
        avg_us,
        1_000_000.0 / avg_us
    );

    // UPDATE complex (prepared statement) - use small range like DELETE for fair comparison
    let update_complex = db
        .prepare("UPDATE users SET balance = $1 WHERE age >= $2 AND age <= $3 AND active = true")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let new_balance: f64 = rng.random_range(0.0..100000.0);
        let start = Instant::now();
        update_complex
            .execute((new_balance, 27_i64, 28_i64))
            .unwrap(); // Small range like DELETE
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "UPDATE complex",
        avg_us,
        1_000_000.0 / avg_us
    );

    // INSERT single (prepared statement)
    let insert_single = db
        .prepare("INSERT INTO users (id, name, email, age, balance, active, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for i in 0..ITERATIONS {
        let id = (ROW_COUNT + 1000 + i) as i64;
        let age = rng.random_range(18..80);
        let name = format!("New_{}", id);
        let email = format!("new{}@example.com", id);
        let start = Instant::now();
        insert_single
            .execute((
                id,
                &name,
                &email,
                age,
                100.0_f64,
                true,
                "2024-01-01 00:00:00",
            ))
            .unwrap();
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "INSERT single",
        avg_us,
        1_000_000.0 / avg_us
    );

    // DELETE by ID (prepared statement)
    let delete_by_id = db.prepare("DELETE FROM users WHERE id = $1").unwrap();
    let mut total = std::time::Duration::ZERO;
    for i in 0..ITERATIONS {
        let id = (ROW_COUNT + 1000 + i) as i64;
        let start = Instant::now();
        delete_by_id.execute((id,)).unwrap();
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "DELETE by ID",
        avg_us,
        1_000_000.0 / avg_us
    );

    // DELETE complex (prepared statement) - similar to UPDATE complex
    let delete_complex = db
        .prepare("DELETE FROM users WHERE age >= $1 AND age <= $2 AND active = true")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        delete_complex.execute((25_i64, 26_i64)).unwrap(); // Small range to not delete too many
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "DELETE complex",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Aggregation (GROUP BY) (prepared statement)
    let agg_stmt = db
        .prepare("SELECT age, COUNT(*), AVG(balance) FROM users GROUP BY age")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let rows = agg_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "Aggregation (GROUP BY)",
        avg_us,
        1_000_000.0 / avg_us
    );

    println!("============================================================");
    println!(
        "\n{:<25} | {:>12} | {:>12}",
        "Advanced Operations", "Avg (μs)", "ops/sec"
    );
    println!("------------------------------------------------------------");

    // Create orders table for JOIN benchmarks
    db.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL
        )",
        (),
    )
    .unwrap();

    db.execute("CREATE INDEX idx_orders_user_id ON orders(user_id)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_orders_status ON orders(status)", ())
        .unwrap();

    // Populate orders (3 orders per user on average)
    let insert_order = db
        .prepare("INSERT INTO orders (id, user_id, amount, status, order_date) VALUES ($1, $2, $3, $4, $5)")
        .unwrap();

    let statuses = ["pending", "completed", "shipped", "cancelled"];
    for i in 1..=(ROW_COUNT * 3) {
        let user_id = rng.random_range(1..=ROW_COUNT) as i64;
        let amount = rng.random_range(10.0..1000.0);
        let status = statuses[rng.random_range(0..4)];
        insert_order
            .execute((i as i64, user_id, amount, status, "2024-01-15"))
            .unwrap();
    }

    // INNER JOIN (20 iterations - moderately slow)
    let join_stmt = db
        .prepare("SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..20 {
        let start = Instant::now();
        let rows = join_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / 20.0;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "INNER JOIN",
        avg_us,
        1_000_000.0 / avg_us
    );

    // LEFT JOIN with aggregation (20 iterations - slow)
    let left_join_stmt = db
        .prepare("SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..20 {
        let start = Instant::now();
        let rows = left_join_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / 20.0;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "LEFT JOIN + GROUP BY",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Scalar subquery
    let subquery_stmt = db
        .prepare("SELECT name, balance, (SELECT AVG(balance) FROM users) as avg_balance FROM users WHERE balance > (SELECT AVG(balance) FROM users) LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let rows = subquery_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "Scalar subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    // IN subquery (10 iterations - slow query)
    let in_subquery_stmt = db
        .prepare("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed') LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..10 {
        let start = Instant::now();
        let rows = in_subquery_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / 10.0;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "IN subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    // EXISTS subquery (10 iterations - correlated, slow)
    let exists_stmt = db
        .prepare("SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 500) LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..10 {
        let start = Instant::now();
        let rows = exists_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / 10.0;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "EXISTS subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    // CTE (Common Table Expression) - 20 iterations
    let cte_stmt = db
        .prepare("WITH high_value AS (SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id HAVING SUM(amount) > 1000) SELECT u.name, h.total FROM users u INNER JOIN high_value h ON u.id = h.user_id LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..20 {
        let start = Instant::now();
        let rows = cte_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / 20.0;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "CTE + JOIN",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Window function - ROW_NUMBER
    let window_stmt = db
        .prepare("SELECT name, balance, ROW_NUMBER() OVER (ORDER BY balance DESC) as rank FROM users LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let rows = window_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "Window ROW_NUMBER",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Window function - PARTITION BY
    let window_partition_stmt = db
        .prepare("SELECT name, age, balance, RANK() OVER (PARTITION BY age ORDER BY balance DESC) as age_rank FROM users LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let rows = window_partition_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "Window PARTITION BY",
        avg_us,
        1_000_000.0 / avg_us
    );

    // UNION
    let union_stmt = db
        .prepare("SELECT name, 'high' as category FROM users WHERE balance > 50000 UNION ALL SELECT name, 'low' as category FROM users WHERE balance <= 50000 LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let rows = union_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "UNION ALL",
        avg_us,
        1_000_000.0 / avg_us
    );

    // CASE expression
    let case_stmt = db
        .prepare("SELECT name, CASE WHEN balance > 75000 THEN 'platinum' WHEN balance > 50000 THEN 'gold' WHEN balance > 25000 THEN 'silver' ELSE 'bronze' END as tier FROM users LIMIT 100")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let rows = case_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "CASE expression",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Multi-table JOIN (20 iterations - slow)
    let multi_join_stmt = db
        .prepare("SELECT u.name, COUNT(DISTINCT o.id) as orders, SUM(o.amount) as total FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE u.active = true AND o.status IN ('completed', 'shipped') GROUP BY u.id, u.name HAVING COUNT(o.id) > 1 LIMIT 50")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..20 {
        let start = Instant::now();
        let rows = multi_join_stmt.query(()).unwrap();
        for _ in rows {}
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / 20.0;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "Complex JOIN+GROUP+HAVING",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Batch INSERT (100 rows)
    let mut total = std::time::Duration::ZERO;
    for iter in 0..ITERATIONS {
        let base_id = (ROW_COUNT * 10 + iter * 100) as i64;
        let start = Instant::now();
        for i in 0..100 {
            insert_order
                .execute((base_id + i as i64, 1_i64, 100.0, "pending", "2024-02-01"))
                .unwrap();
        }
        total += start.elapsed();
    }
    let avg_us = total.as_micros() as f64 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>12.1} | {:>12.0}",
        "Batch INSERT (100 rows)",
        avg_us,
        1_000_000.0 / avg_us
    );

    println!("============================================================");
}
