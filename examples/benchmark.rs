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

    // UPDATE complex (prepared statement)
    let update_complex = db
        .prepare("UPDATE users SET balance = $1 WHERE age >= 25 AND age <= 45 AND active = true")
        .unwrap();
    let mut total = std::time::Duration::ZERO;
    for _ in 0..ITERATIONS {
        let new_balance: f64 = rng.random_range(0.0..100000.0);
        let start = Instant::now();
        update_complex.execute((new_balance,)).unwrap();
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
}
