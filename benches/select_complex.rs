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

//! Fair benchmark comparison: Stoolap vs SQLite for complex SELECT queries
//!
//! Run with: cargo bench --bench select_complex
//!
//! This benchmark compares complex query patterns:
//! - SELECT with WHERE + ORDER BY + LIMIT
//! - Aggregation (GROUP BY)
//! - JOINs
//! - Subqueries
//! - Window functions
//! - CTEs

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

    // Create orders table for JOIN benchmarks
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

fn bench_select_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("SELECT complex (WHERE+ORDER+LIMIT)");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT id, name, balance FROM users WHERE age >= 25 AND age <= 45 AND active = true ORDER BY balance DESC LIMIT 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT id, name, balance FROM users WHERE age >= 25 AND age <= 45 AND active = 1 ORDER BY balance DESC LIMIT 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, i64>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Aggregation (GROUP BY)");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT age, COUNT(*), AVG(balance) FROM users GROUP BY age")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT age, COUNT(*), AVG(balance) FROM users GROUP BY age")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, i64>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_inner_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("INNER JOIN");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' LIMIT 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' LIMIT 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, String>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_left_join_with_agg(c: &mut Criterion) {
    let mut group = c.benchmark_group("LEFT JOIN + GROUP BY");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT u.name, COUNT(o.id) as order_count, SUM(o.amount) as total FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, String>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_scalar_subquery(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalar subquery");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT name, balance FROM users WHERE balance > (SELECT AVG(balance) FROM users) LIMIT 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT name, balance FROM users WHERE balance > (SELECT AVG(balance) FROM users) LIMIT 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, String>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_in_subquery(c: &mut Criterion) {
    let mut group = c.benchmark_group("IN subquery");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed') LIMIT 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'completed') LIMIT 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, i64>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_exists_subquery(c: &mut Criterion) {
    let mut group = c.benchmark_group("EXISTS subquery");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 500) LIMIT 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 500) LIMIT 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, i64>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_cte(c: &mut Criterion) {
    let mut group = c.benchmark_group("CTE + JOIN");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("WITH high_value AS (SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id HAVING SUM(amount) > 1000) SELECT u.name, h.total FROM users u INNER JOIN high_value h ON u.id = h.user_id LIMIT 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("WITH high_value AS (SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id HAVING SUM(amount) > 1000) SELECT u.name, h.total FROM users u INNER JOIN high_value h ON u.id = h.user_id LIMIT 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, String>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_window_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("Window ROW_NUMBER");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT name, balance, ROW_NUMBER() OVER (ORDER BY balance DESC) as rank FROM users LIMIT 100")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT name, balance, ROW_NUMBER() OVER (ORDER BY balance DESC) as rank FROM users LIMIT 100")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, String>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_distinct(c: &mut Criterion) {
    let mut group = c.benchmark_group("DISTINCT");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT DISTINCT age FROM users")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT DISTINCT age FROM users")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, i64>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_count_distinct(c: &mut Criterion) {
    let mut group = c.benchmark_group("COUNT DISTINCT");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db
        .prepare("SELECT COUNT(DISTINCT age) FROM users")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT COUNT(DISTINCT age) FROM users")
        .unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, i64>(0).unwrap());
            }
        });
    });

    group.finish();
}

fn bench_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("SELECT * (full scan)");

    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    let stoolap_stmt = stoolap_db.prepare("SELECT * FROM users").unwrap();

    let mut sqlite_stmt = sqlite_conn.prepare("SELECT * FROM users").unwrap();

    group.bench_function("stoolap", |b| {
        b.iter(|| {
            let rows = stoolap_stmt.query(()).unwrap();
            for row in rows {
                black_box(row.unwrap());
            }
        });
    });

    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut rows = sqlite_stmt.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                black_box(row.get::<_, i64>(0).unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_select_complex,
    bench_aggregation,
    bench_inner_join,
    bench_left_join_with_agg,
    bench_scalar_subquery,
    bench_in_subquery,
    bench_exists_subquery,
    bench_cte,
    bench_window_function,
    bench_distinct,
    bench_count_distinct,
    bench_full_scan
);
criterion_main!(benches);
