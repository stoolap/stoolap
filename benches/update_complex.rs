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

//! Fair benchmark comparison: Stoolap vs SQLite for complex UPDATE queries
//!
//! Run with: cargo bench --bench update_complex
//!
//! This benchmark compares complex UPDATE patterns:
//! - UPDATE with WHERE + multiple conditions (range + boolean)
//! - UPDATE with index range scan
//! - UPDATE with subquery in WHERE (EXISTS)
//! - UPDATE with IN subquery

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusqlite::Connection;
use stoolap::Database;

const ROW_COUNT: usize = 10_000;

/// Setup Stoolap database with test data
fn setup_stoolap() -> Database {
    let db = Database::open("memory://").unwrap();

    db.execute(
        "CREATE TABLE IF NOT EXISTS users (
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

    db.execute("CREATE INDEX IF NOT EXISTS idx_users_age ON users(age)", ())
        .unwrap();
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_active ON users(active)",
        (),
    )
    .unwrap();

    // Check if data already exists
    let count_result = db.query("SELECT COUNT(*) FROM users", ()).unwrap();
    let count_row = count_result.into_iter().next().unwrap().unwrap();
    let count: i64 = count_row.get(0).unwrap();
    if count > 0 {
        return db;
    }

    let insert_stmt = db
        .prepare("INSERT INTO users (id, name, email, age, balance, active, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)")
        .unwrap();

    for i in 1..=ROW_COUNT {
        let name = format!("User_{}", i);
        let email = format!("user{}@example.com", i);
        insert_stmt
            .execute((
                i as i64,
                &name,
                &email,
                ((i % 62) + 18) as i64,
                (i as f64) * 10.0,
                i % 2 == 0,
                "2024-01-01 00:00:00",
            ))
            .unwrap();
    }

    // Create orders table for subquery benchmarks
    db.execute(
        "CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL
        )",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
        (),
    )
    .unwrap();

    // Populate orders (3 orders per user on average)
    let insert_order = db
        .prepare("INSERT INTO orders (id, user_id, amount, status, order_date) VALUES ($1, $2, $3, $4, $5)")
        .unwrap();

    let statuses = ["pending", "completed", "shipped", "cancelled"];
    for i in 1..=(ROW_COUNT * 3) {
        let user_id = ((i % ROW_COUNT) + 1) as i64;
        let amount = ((i % 990) + 10) as f64;
        let status = statuses[i % 4];
        insert_order
            .execute((i as i64, user_id, amount, status, "2024-01-15"))
            .unwrap();
    }

    db
}

/// Setup SQLite database with test data
fn setup_sqlite() -> Connection {
    let conn = Connection::open_in_memory().unwrap();

    conn.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER NOT NULL,
            balance REAL NOT NULL,
            active INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )",
        [],
    )
    .unwrap();

    conn.execute("CREATE INDEX idx_users_age ON users(age)", [])
        .unwrap();
    conn.execute("CREATE INDEX idx_users_active ON users(active)", [])
        .unwrap();

    for i in 1..=ROW_COUNT {
        let name = format!("User_{}", i);
        let email = format!("user{}@example.com", i);
        conn.execute(
            "INSERT INTO users (id, name, email, age, balance, active, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                i as i64,
                &name,
                &email,
                ((i % 62) + 18) as i64,
                (i as f64) * 10.0,
                if i % 2 == 0 { 1 } else { 0 },
                "2024-01-01 00:00:00",
            ],
        )
        .unwrap();
    }

    // Create orders table
    conn.execute(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL
        )",
        [],
    )
    .unwrap();

    conn.execute("CREATE INDEX idx_orders_user_id ON orders(user_id)", [])
        .unwrap();
    conn.execute("CREATE INDEX idx_orders_status ON orders(status)", [])
        .unwrap();

    let statuses = ["pending", "completed", "shipped", "cancelled"];
    for i in 1..=(ROW_COUNT * 3) {
        let user_id = ((i % ROW_COUNT) + 1) as i64;
        let amount = ((i % 990) + 10) as f64;
        let status = statuses[i % 4];
        conn.execute(
            "INSERT INTO orders (id, user_id, amount, status, order_date) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![i as i64, user_id, amount, status, "2024-01-15"],
        )
        .unwrap();
    }

    conn
}

fn bench_update_range_with_boolean(c: &mut Criterion) {
    let mut group = c.benchmark_group("UPDATE range + boolean");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    // UPDATE with range condition on age AND boolean condition
    // Uses small range (2 years) to limit rows affected per iteration
    let stoolap_stmt = stoolap_db
        .prepare("UPDATE users SET balance = $1 WHERE age >= $2 AND age <= $3 AND active = true")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("UPDATE users SET balance = ?1 WHERE age >= ?2 AND age <= ?3 AND active = 1")
        .unwrap();

    group.bench_function("stoolap", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let new_balance = (iter_count as f64) * 10.0;
            iter_count += 1;
            stoolap_stmt
                .execute(black_box((new_balance, 27_i64, 28_i64)))
                .unwrap();
            black_box(())
        });
    });

    group.bench_function("sqlite", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let new_balance = (iter_count as f64) * 10.0;
            iter_count += 1;
            sqlite_stmt
                .execute(rusqlite::params![
                    black_box(new_balance),
                    black_box(27_i64),
                    black_box(28_i64)
                ])
                .unwrap();
            black_box(())
        });
    });

    group.finish();
}

fn bench_update_index_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("UPDATE index range scan");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    // UPDATE with index range scan only (no boolean filter)
    let stoolap_stmt = stoolap_db
        .prepare("UPDATE users SET balance = $1 WHERE age >= $2 AND age <= $3")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("UPDATE users SET balance = ?1 WHERE age >= ?2 AND age <= ?3")
        .unwrap();

    group.bench_function("stoolap", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let new_balance = (iter_count as f64) * 10.0;
            iter_count += 1;
            // Small range to limit affected rows
            stoolap_stmt
                .execute(black_box((new_balance, 30_i64, 31_i64)))
                .unwrap();
            black_box(())
        });
    });

    group.bench_function("sqlite", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let new_balance = (iter_count as f64) * 10.0;
            iter_count += 1;
            sqlite_stmt
                .execute(rusqlite::params![
                    black_box(new_balance),
                    black_box(30_i64),
                    black_box(31_i64)
                ])
                .unwrap();
            black_box(())
        });
    });

    group.finish();
}

fn bench_update_with_exists(c: &mut Criterion) {
    let mut group = c.benchmark_group("UPDATE with EXISTS subquery");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    // UPDATE users who have high-value orders
    let stoolap_stmt = stoolap_db
        .prepare("UPDATE users SET balance = balance + $1 WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id AND orders.amount > 800) AND id <= 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("UPDATE users SET balance = balance + ?1 WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id AND orders.amount > 800) AND id <= 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let bonus = ((iter_count % 100) as f64) * 0.1;
            iter_count += 1;
            stoolap_stmt.execute(black_box((bonus,))).unwrap();
            black_box(())
        });
    });

    group.bench_function("sqlite", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let bonus = ((iter_count % 100) as f64) * 0.1;
            iter_count += 1;
            sqlite_stmt
                .execute(rusqlite::params![black_box(bonus)])
                .unwrap();
            black_box(())
        });
    });

    group.finish();
}

fn bench_update_with_in_subquery(c: &mut Criterion) {
    let mut group = c.benchmark_group("UPDATE with IN subquery");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    // UPDATE users who have completed orders (limited to first 100 for consistent performance)
    let stoolap_stmt = stoolap_db
        .prepare("UPDATE users SET balance = balance + $1 WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed' LIMIT 50)")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("UPDATE users SET balance = balance + ?1 WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed' LIMIT 50)")
        .unwrap();

    group.bench_function("stoolap", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let bonus = ((iter_count % 100) as f64) * 0.1;
            iter_count += 1;
            stoolap_stmt.execute(black_box((bonus,))).unwrap();
            black_box(())
        });
    });

    group.bench_function("sqlite", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let bonus = ((iter_count % 100) as f64) * 0.1;
            iter_count += 1;
            sqlite_stmt
                .execute(rusqlite::params![black_box(bonus)])
                .unwrap();
            black_box(())
        });
    });

    group.finish();
}

fn bench_update_full_scan_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("UPDATE full scan (small range)");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    // Full table scan UPDATE with boolean condition only
    let stoolap_stmt = stoolap_db
        .prepare("UPDATE users SET balance = $1 WHERE active = false AND id <= 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("UPDATE users SET balance = ?1 WHERE active = 0 AND id <= 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let new_balance = (iter_count as f64) * 10.0;
            iter_count += 1;
            stoolap_stmt.execute(black_box((new_balance,))).unwrap();
            black_box(())
        });
    });

    group.bench_function("sqlite", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            let new_balance = (iter_count as f64) * 10.0;
            iter_count += 1;
            sqlite_stmt
                .execute(rusqlite::params![black_box(new_balance)])
                .unwrap();
            black_box(())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_update_range_with_boolean,
    bench_update_index_range,
    bench_update_with_exists,
    bench_update_with_in_subquery,
    bench_update_full_scan_small
);
criterion_main!(benches);
