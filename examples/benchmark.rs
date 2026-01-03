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
const ITERATIONS: usize = 500; // Point queries
const ITERATIONS_MEDIUM: usize = 250; // Index scans, aggregations
const ITERATIONS_HEAVY: usize = 50; // Full scans, JOINs
const WARMUP: usize = 10;

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
    println!(
        "STOOLAP-RUST BENCHMARK ({} rows, {} iterations, in-memory)",
        ROW_COUNT, ITERATIONS
    );
    println!("============================================================\n");
    println!(
        "{:<25} | {:>15} | {:>12}",
        "Operation", "Avg (μs)", "ops/sec"
    );
    println!("---------------------------------------------------------------");

    // SELECT by ID (prepared statement)
    let select_by_id = db.prepare("SELECT * FROM users WHERE id = $1").unwrap();
    let ids: Vec<i64> = (0..ITERATIONS)
        .map(|i| ((i % ROW_COUNT) + 1) as i64)
        .collect();
    // Warmup
    for &id in ids.iter().take(WARMUP) {
        let rows = select_by_id.query((id,)).unwrap();
        let _ = rows.into_iter().next();
    }
    let start = Instant::now();
    for &id in &ids {
        let rows = select_by_id.query((id,)).unwrap();
        let _ = rows.into_iter().next();
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "SELECT by ID",
        avg_us,
        1_000_000.0 / avg_us
    );

    // SELECT by index (exact match on age)
    let select_by_index = db.prepare("SELECT * FROM users WHERE age = $1").unwrap();
    let ages: Vec<i64> = (0..ITERATIONS).map(|i| ((i % 62) + 18) as i64).collect();
    // Warmup
    for &age in ages.iter().take(WARMUP) {
        let rows = select_by_index.query((age,)).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for &age in &ages {
        let rows = select_by_index.query((age,)).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "SELECT by index (exact)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // SELECT by index (range query on age)
    let select_by_index_range = db
        .prepare("SELECT * FROM users WHERE age >= $1 AND age <= $2")
        .unwrap();
    // Warmup
    for _ in 0..WARMUP {
        let rows = select_by_index_range.query((30_i64, 40_i64)).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = select_by_index_range.query((30_i64, 40_i64)).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "SELECT by index (range)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // SELECT complex (prepared statement)
    let select_complex = db
        .prepare("SELECT id, name, balance FROM users WHERE age >= 25 AND age <= 45 AND active = true ORDER BY balance DESC LIMIT 100")
        .unwrap();
    // Warmup
    for _ in 0..WARMUP {
        let rows = select_complex.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = select_complex.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "SELECT complex",
        avg_us,
        1_000_000.0 / avg_us
    );

    // SELECT * (full scan) (prepared statement)
    let select_all = db.prepare("SELECT * FROM users").unwrap();
    // Warmup
    for _ in 0..10 {
        let rows = select_all.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS_HEAVY {
        let rows = select_all.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS_HEAVY as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "SELECT * (full scan)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // UPDATE by ID (prepared statement)
    let update_by_id = db
        .prepare("UPDATE users SET balance = $1 WHERE id = $2")
        .unwrap();
    let update_params: Vec<(f64, i64)> = (0..ITERATIONS)
        .map(|i| {
            (
                rng.random_range(0.0..100000.0),
                ((i % ROW_COUNT) + 1) as i64,
            )
        })
        .collect();
    // Warmup
    for &(balance, id) in update_params.iter().take(WARMUP) {
        update_by_id.execute((balance, id)).unwrap();
    }
    let start = Instant::now();
    for &(balance, id) in &update_params {
        update_by_id.execute((balance, id)).unwrap();
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "UPDATE by ID",
        avg_us,
        1_000_000.0 / avg_us
    );

    // UPDATE complex (prepared statement) - use small range like DELETE for fair comparison
    let update_complex = db
        .prepare("UPDATE users SET balance = $1 WHERE age >= $2 AND age <= $3 AND active = true")
        .unwrap();
    let update_complex_balances: Vec<f64> = (0..ITERATIONS)
        .map(|_| rng.random_range(0.0..100000.0))
        .collect();
    // Warmup
    for &balance in update_complex_balances.iter().take(WARMUP) {
        update_complex.execute((balance, 27_i64, 28_i64)).unwrap();
    }
    let start = Instant::now();
    for &balance in &update_complex_balances {
        update_complex.execute((balance, 27_i64, 28_i64)).unwrap(); // Small range like DELETE
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "UPDATE complex",
        avg_us,
        1_000_000.0 / avg_us
    );

    // INSERT single (prepared statement)
    let insert_single = db
        .prepare("INSERT INTO users (id, name, email, age, balance, active, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)")
        .unwrap();
    let insert_params: Vec<(i64, String, String, i64)> = (0..ITERATIONS)
        .map(|i| {
            let id = (ROW_COUNT + 1000 + i) as i64;
            (
                id,
                format!("New_{}", id),
                format!("new{}@example.com", id),
                rng.random_range(18..80),
            )
        })
        .collect();
    let start = Instant::now();
    for (id, name, email, age) in &insert_params {
        insert_single
            .execute((
                *id,
                name,
                email,
                *age,
                100.0_f64,
                true,
                "2024-01-01 00:00:00",
            ))
            .unwrap();
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "INSERT single",
        avg_us,
        1_000_000.0 / avg_us
    );

    // DELETE by ID (prepared statement)
    let delete_by_id = db.prepare("DELETE FROM users WHERE id = $1").unwrap();
    let delete_ids: Vec<i64> = (0..ITERATIONS)
        .map(|i| (ROW_COUNT + 1000 + i) as i64)
        .collect();
    let start = Instant::now();
    for &id in &delete_ids {
        delete_by_id.execute((id,)).unwrap();
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "DELETE by ID",
        avg_us,
        1_000_000.0 / avg_us
    );

    // DELETE complex (prepared statement) - similar to UPDATE complex
    let delete_complex = db
        .prepare("DELETE FROM users WHERE age >= $1 AND age <= $2 AND active = true")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        delete_complex.execute((25_i64, 26_i64)).unwrap(); // Small range to not delete too many
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "DELETE complex",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Aggregation (GROUP BY) (prepared statement)
    let agg_stmt = db
        .prepare("SELECT age, COUNT(*), AVG(balance) FROM users GROUP BY age")
        .unwrap();
    // Warmup
    for _ in 0..10 {
        let rows = agg_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS_MEDIUM {
        let rows = agg_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS_MEDIUM as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Aggregation (GROUP BY)",
        avg_us,
        1_000_000.0 / avg_us
    );

    println!("============================================================");
    println!(
        "\n{:<25} | {:>15} | {:>12}",
        "Advanced Operations", "Avg (μs)", "ops/sec"
    );
    println!("---------------------------------------------------------------");

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

    // INNER JOIN (100 iterations with warmup)
    let join_stmt = db
        .prepare("SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' LIMIT 100")
        .unwrap();
    // Warmup
    for _ in 0..WARMUP {
        let rows = join_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..100 {
        let rows = join_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 100.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "INNER JOIN",
        avg_us,
        1_000_000.0 / avg_us
    );

    // LEFT JOIN with aggregation (100 iterations with warmup)
    let left_join_stmt = db
        .prepare("SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 100")
        .unwrap();
    // Warmup
    for _ in 0..WARMUP {
        let rows = left_join_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..100 {
        let rows = left_join_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 100.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "LEFT JOIN + GROUP BY",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Scalar subquery
    let subquery_stmt = db
        .prepare("SELECT name, balance, (SELECT AVG(balance) FROM users) as avg_balance FROM users WHERE balance > (SELECT AVG(balance) FROM users) LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = subquery_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Scalar subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    // IN subquery (10 iterations - slow query)
    let in_subquery_stmt = db
        .prepare("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed') LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..10 {
        let rows = in_subquery_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 10.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "IN subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    // EXISTS subquery (100 iterations with warmup for semi-join cache)
    let exists_stmt = db
        .prepare("SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 500) LIMIT 100")
        .unwrap();
    // Warmup to populate semi-join cache
    for _ in 0..WARMUP {
        let rows = exists_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..100 {
        let rows = exists_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 100.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "EXISTS subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    // CTE (Common Table Expression) - 20 iterations
    let cte_stmt = db
        .prepare("WITH high_value AS (SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id HAVING SUM(amount) > 1000) SELECT u.name, h.total FROM users u INNER JOIN high_value h ON u.id = h.user_id LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..20 {
        let rows = cte_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 20.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "CTE + JOIN",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Window function - ROW_NUMBER (non-indexed column)
    let window_stmt = db
        .prepare("SELECT name, balance, ROW_NUMBER() OVER (ORDER BY balance DESC) as rank FROM users LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = window_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Window ROW_NUMBER",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Window function - ROW_NUMBER with PK (index optimization)
    let window_pk_stmt = db
        .prepare("SELECT name, ROW_NUMBER() OVER (ORDER BY id) as rank FROM users LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = window_pk_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Window ROW_NUMBER (PK)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Window function - PARTITION BY
    let window_partition_stmt = db
        .prepare("SELECT name, age, balance, RANK() OVER (PARTITION BY age ORDER BY balance DESC) as age_rank FROM users LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = window_partition_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Window PARTITION BY",
        avg_us,
        1_000_000.0 / avg_us
    );

    // UNION
    let union_stmt = db
        .prepare("SELECT name, 'high' as category FROM users WHERE balance > 50000 UNION ALL SELECT name, 'low' as category FROM users WHERE balance <= 50000 LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = union_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "UNION ALL",
        avg_us,
        1_000_000.0 / avg_us
    );

    // CASE expression
    let case_stmt = db
        .prepare("SELECT name, CASE WHEN balance > 75000 THEN 'platinum' WHEN balance > 50000 THEN 'gold' WHEN balance > 25000 THEN 'silver' ELSE 'bronze' END as tier FROM users LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = case_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "CASE expression",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Multi-table JOIN (20 iterations - slow)
    let multi_join_stmt = db
        .prepare("SELECT u.name, COUNT(DISTINCT o.id) as orders, SUM(o.amount) as total FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE u.active = true AND o.status IN ('completed', 'shipped') GROUP BY u.id, u.name HAVING COUNT(o.id) > 1 LIMIT 50")
        .unwrap();
    let start = Instant::now();
    for _ in 0..20 {
        let rows = multi_join_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 20.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Complex JOIN+GROUP+HAVING",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Batch INSERT in transaction (100 individual INSERTs wrapped in BEGIN/COMMIT)
    let batch_base_ids: Vec<i64> = (0..ITERATIONS)
        .map(|iter| (ROW_COUNT * 10 + iter * 100) as i64)
        .collect();
    let start = Instant::now();
    for &base_id in &batch_base_ids {
        db.execute("BEGIN", ()).unwrap();
        for i in 0..100 {
            insert_order
                .execute((base_id + i as i64, 1_i64, 100.0, "pending", "2024-02-01"))
                .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Batch INSERT (100 rows)",
        avg_us,
        1_000_000.0 / avg_us
    );

    println!("============================================================");
    println!(
        "\n{:<25} | {:>15} | {:>12}",
        "Bottleneck Hunters", "Avg (μs)", "ops/sec"
    );
    println!("---------------------------------------------------------------");

    // DISTINCT without ORDER BY
    let distinct_stmt = db.prepare("SELECT DISTINCT age FROM users").unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = distinct_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "DISTINCT (no ORDER)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // DISTINCT with ORDER BY
    let distinct_order_stmt = db
        .prepare("SELECT DISTINCT age FROM users ORDER BY age")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = distinct_order_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "DISTINCT + ORDER BY",
        avg_us,
        1_000_000.0 / avg_us
    );

    // COUNT DISTINCT
    let count_distinct_stmt = db.prepare("SELECT COUNT(DISTINCT age) FROM users").unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = count_distinct_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "COUNT DISTINCT",
        avg_us,
        1_000_000.0 / avg_us
    );

    // String LIKE pattern (prefix)
    let like_prefix_stmt = db
        .prepare("SELECT * FROM users WHERE name LIKE 'User_1%' LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = like_prefix_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "LIKE prefix (User_1%)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // String LIKE pattern (contains) - harder to optimize
    let like_contains_stmt = db
        .prepare("SELECT * FROM users WHERE email LIKE '%50%' LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = like_contains_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "LIKE contains (%50%)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // OR condition (often hard to optimize)
    let or_stmt = db
        .prepare("SELECT * FROM users WHERE age = 25 OR age = 50 OR age = 75 LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = or_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "OR conditions (3 vals)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // IN list
    let in_list_stmt = db
        .prepare("SELECT * FROM users WHERE age IN (20, 25, 30, 35, 40, 45, 50) LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = in_list_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "IN list (7 values)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // NOT IN subquery (often slow - anti-join)
    let not_in_stmt = db
        .prepare("SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM orders WHERE status = 'cancelled') LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..10 {
        let rows = not_in_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 10.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "NOT IN subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    // NOT EXISTS (alternative anti-join) - 100 iterations with warmup
    let not_exists_stmt = db
        .prepare("SELECT * FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.status = 'cancelled') LIMIT 100")
        .unwrap();
    // Warmup to populate semi-join cache
    for _ in 0..WARMUP {
        let rows = not_exists_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..100 {
        let rows = not_exists_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 100.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "NOT EXISTS subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    // OFFSET pagination (often slow for large offsets)
    let offset_stmt = db
        .prepare("SELECT * FROM users ORDER BY id LIMIT 100 OFFSET 5000")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = offset_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "OFFSET pagination (5000)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Multi-column ORDER BY
    let multi_order_stmt = db
        .prepare("SELECT * FROM users ORDER BY age DESC, balance ASC, name LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = multi_order_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Multi-col ORDER BY (3)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Self-join (users with same age) - 100 iterations with warmup
    let self_join_stmt = db
        .prepare("SELECT u1.name, u2.name, u1.age FROM users u1 INNER JOIN users u2 ON u1.age = u2.age AND u1.id < u2.id LIMIT 100")
        .unwrap();
    // Warmup
    for _ in 0..WARMUP {
        let rows = self_join_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..100 {
        let rows = self_join_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 100.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Self JOIN (same age)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Multiple window functions
    let multi_window_stmt = db
        .prepare("SELECT name, balance, ROW_NUMBER() OVER (ORDER BY balance DESC) as rn, RANK() OVER (ORDER BY balance DESC) as rnk, LAG(balance) OVER (ORDER BY balance DESC) as prev_bal FROM users LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = multi_window_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Multi window funcs (3)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Nested subquery (3 levels)
    let nested_stmt = db
        .prepare("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)) LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..20 {
        let rows = nested_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 20.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Nested subquery (3 lvl)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Multiple aggregates in one query
    let multi_agg_stmt = db
        .prepare("SELECT COUNT(*), SUM(balance), AVG(balance), MIN(balance), MAX(balance), COUNT(DISTINCT age) FROM users")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = multi_agg_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Multi aggregates (6)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // COALESCE / NULL handling
    let coalesce_stmt = db
        .prepare("SELECT name, COALESCE(balance, 0) as bal FROM users WHERE balance IS NOT NULL LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = coalesce_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "COALESCE + IS NOT NULL",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Expression in WHERE (function call)
    let expr_where_stmt = db
        .prepare(
            "SELECT * FROM users WHERE LENGTH(name) > 7 AND UPPER(name) LIKE 'USER_%' LIMIT 100",
        )
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = expr_where_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Expr in WHERE (funcs)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Math expressions
    let math_stmt = db
        .prepare("SELECT name, balance * 1.1 as new_bal, ROUND(balance / 1000, 2) as k_bal, ABS(balance - 50000) as diff FROM users LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = math_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Math expressions",
        avg_us,
        1_000_000.0 / avg_us
    );

    // String concatenation
    let concat_stmt = db
        .prepare("SELECT name || ' (' || email || ')' as full_info FROM users LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = concat_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "String concat (||)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Large result set (no LIMIT)
    let large_result_stmt = db
        .prepare("SELECT id, name, balance FROM users WHERE active = true")
        .unwrap();
    let start = Instant::now();
    for _ in 0..20 {
        let rows = large_result_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 20.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Large result (no LIMIT)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Multiple CTEs (100 iterations with warmup)
    let multi_cte_stmt = db
        .prepare("WITH young AS (SELECT * FROM users WHERE age < 30), rich AS (SELECT * FROM users WHERE balance > 70000) SELECT y.name, r.name FROM young y INNER JOIN rich r ON y.id = r.id LIMIT 50")
        .unwrap();
    // Warmup
    for _ in 0..WARMUP {
        let rows = multi_cte_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..100 {
        let rows = multi_cte_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 100.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Multiple CTEs (2)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Correlated subquery in SELECT
    let corr_select_stmt = db
        .prepare("SELECT u.name, (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count FROM users u LIMIT 100")
        .unwrap();
    // Warm up
    for _ in 0..5 {
        let rows = corr_select_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let start = Instant::now();
    for _ in 0..100 {
        let rows = corr_select_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / 100.0;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Correlated in SELECT",
        avg_us,
        1_000_000.0 / avg_us
    );

    // BETWEEN range
    let between_stmt = db
        .prepare("SELECT * FROM users WHERE balance BETWEEN 25000 AND 75000 LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = between_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "BETWEEN (non-indexed)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // GROUP BY with multiple columns
    let multi_group_stmt = db
        .prepare("SELECT age, active, COUNT(*), AVG(balance) FROM users GROUP BY age, active")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = multi_group_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "GROUP BY (2 columns)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Cross join (cartesian product) - limited
    let cross_join_stmt = db
        .prepare("SELECT u.name, o.status FROM users u CROSS JOIN (SELECT DISTINCT status FROM orders) o LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = cross_join_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "CROSS JOIN (limited)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Derived table (subquery in FROM)
    let derived_stmt = db
        .prepare("SELECT t.age_group, COUNT(*) FROM (SELECT CASE WHEN age < 30 THEN 'young' WHEN age < 50 THEN 'middle' ELSE 'senior' END as age_group FROM users) t GROUP BY t.age_group")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = derived_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Derived table (FROM sub)",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Window with frame clause
    let window_frame_stmt = db
        .prepare("SELECT name, balance, SUM(balance) OVER (ORDER BY balance ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) as rolling_sum FROM users LIMIT 100")
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = window_frame_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Window ROWS frame",
        avg_us,
        1_000_000.0 / avg_us
    );

    // HAVING without GROUP BY columns in SELECT
    let having_stmt = db
        .prepare(
            "SELECT age FROM users GROUP BY age HAVING COUNT(*) > 100 AND AVG(balance) > 40000",
        )
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = having_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "HAVING complex",
        avg_us,
        1_000_000.0 / avg_us
    );

    // Comparison with subquery result
    let compare_sub_stmt = db
        .prepare(
            "SELECT * FROM users WHERE balance > (SELECT AVG(amount) * 100 FROM orders) LIMIT 100",
        )
        .unwrap();
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let rows = compare_sub_stmt.query(()).unwrap();
        for _ in rows {}
    }
    let total = start.elapsed();
    let avg_us = total.as_nanos() as f64 / 1000.0 / ITERATIONS as f64;
    println!(
        "{:<25} | {:>15.3} | {:>12.0}",
        "Compare with subquery",
        avg_us,
        1_000_000.0 / avg_us
    );

    println!("============================================================");
}
