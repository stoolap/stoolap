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

//! Fair benchmark comparison: Stoolap vs SQLite for complex DELETE queries
//!
//! Run with: cargo bench --bench delete_complex
//!
//! This benchmark compares complex DELETE patterns:
//! - DELETE with WHERE + multiple conditions (range + boolean)
//! - DELETE with index range scan
//! - DELETE with subquery in WHERE (EXISTS)
//! - DELETE with IN subquery

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

fn bench_delete_range_with_boolean(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("DELETE range + boolean");
    group.sample_size(50);

    // Age 27: ~161 rows out of 10K, half active = ~80 rows deleted
    let target_age = 27i64;

    // Benchmark Stoolap - fresh database per batch
    group.bench_function("stoolap", |b| {
        b.iter_batched(
            || {
                let db = setup_stoolap();
                let stmt = db
                    .prepare("DELETE FROM users WHERE age = $1 AND active = true")
                    .unwrap();
                (db, stmt)
            },
            |(_db, stmt)| {
                stmt.execute(black_box((target_age,))).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    // Benchmark SQLite - fresh database per batch
    group.bench_function("sqlite", |b| {
        b.iter_batched(
            setup_sqlite,
            |conn| {
                conn.execute(
                    "DELETE FROM users WHERE age = ?1 AND active = 1",
                    rusqlite::params![black_box(target_age)],
                )
                .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_delete_index_range(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("DELETE index range scan");
    group.sample_size(50);

    // Age 30: ~161 rows deleted per iteration
    let target_age = 30i64;

    // Benchmark Stoolap - fresh database per batch
    group.bench_function("stoolap", |b| {
        b.iter_batched(
            || {
                let db = setup_stoolap();
                let stmt = db.prepare("DELETE FROM users WHERE age = $1").unwrap();
                (db, stmt)
            },
            |(_db, stmt)| {
                stmt.execute(black_box((target_age,))).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    // Benchmark SQLite - fresh database per batch
    group.bench_function("sqlite", |b| {
        b.iter_batched(
            setup_sqlite,
            |conn| {
                conn.execute(
                    "DELETE FROM users WHERE age = ?1",
                    rusqlite::params![black_box(target_age)],
                )
                .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_delete_with_exists(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("DELETE with EXISTS subquery");
    group.sample_size(50);

    // Benchmark Stoolap - fresh database per batch
    group.bench_function("stoolap", |b| {
        b.iter_batched(
            || {
                let db = setup_stoolap();
                let stmt = db
                    .prepare("DELETE FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id AND orders.amount > 900) AND id <= 50")
                    .unwrap();
                (db, stmt)
            },
            |(_db, stmt)| {
                stmt.execute(()).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    // Benchmark SQLite - fresh database per batch
    group.bench_function("sqlite", |b| {
        b.iter_batched(
            setup_sqlite,
            |conn| {
                conn.execute(
                    "DELETE FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id AND orders.amount > 900) AND id <= 50",
                    [],
                )
                .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_delete_with_in_subquery(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("DELETE with IN subquery");
    group.sample_size(50);

    // Benchmark Stoolap - fresh database per batch
    group.bench_function("stoolap", |b| {
        b.iter_batched(
            || {
                let db = setup_stoolap();
                let stmt = db
                    .prepare("DELETE FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed' LIMIT 30)")
                    .unwrap();
                (db, stmt)
            },
            |(_db, stmt)| {
                stmt.execute(()).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    // Benchmark SQLite - fresh database per batch
    group.bench_function("sqlite", |b| {
        b.iter_batched(
            setup_sqlite,
            |conn| {
                conn.execute(
                    "DELETE FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed' LIMIT 30)",
                    [],
                )
                .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_delete_full_scan_small(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("DELETE full scan (small range)");
    group.sample_size(50);

    // Benchmark Stoolap - fresh database per batch
    group.bench_function("stoolap", |b| {
        b.iter_batched(
            || {
                // Setup: create fresh database with data
                let db = Database::open_in_memory().unwrap();
                db.execute(
                    "CREATE TABLE users (id INTEGER PRIMARY KEY, active BOOLEAN NOT NULL)",
                    (),
                )
                .unwrap();
                for i in 1..=100i64 {
                    db.execute("INSERT INTO users VALUES ($1, $2)", (i, i % 2 == 0))
                        .unwrap();
                }
                db
            },
            |db| {
                // Benchmark: DELETE with full scan (~50 rows where active = false)
                db.execute("DELETE FROM users WHERE active = false", ())
                    .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    // Benchmark SQLite - fresh database per batch
    group.bench_function("sqlite", |b| {
        b.iter_batched(
            || {
                // Setup: create fresh database with data
                let conn = Connection::open_in_memory().unwrap();
                conn.execute(
                    "CREATE TABLE users (id INTEGER PRIMARY KEY, active INTEGER NOT NULL)",
                    [],
                )
                .unwrap();
                for i in 1..=100i64 {
                    conn.execute(
                        "INSERT INTO users VALUES (?1, ?2)",
                        rusqlite::params![i, if i % 2 == 0 { 1 } else { 0 }],
                    )
                    .unwrap();
                }
                conn
            },
            |conn| {
                // Benchmark: DELETE with full scan (~50 rows where active = 0)
                conn.execute("DELETE FROM users WHERE active = 0", [])
                    .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_delete_range_with_boolean,
    bench_delete_index_range,
    bench_delete_with_exists,
    bench_delete_with_in_subquery,
    bench_delete_full_scan_small
);
criterion_main!(benches);
