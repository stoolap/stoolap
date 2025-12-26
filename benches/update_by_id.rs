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

//! Fair benchmark comparison: Stoolap vs SQLite for UPDATE by ID
//!
//! Run with: cargo bench --bench update_by_id
//!
//! This benchmark ensures fair comparison by:
//! 1. Using prepared statements for both databases
//! 2. Same data setup (10K rows)
//! 3. Same query pattern (UPDATE ... WHERE id = ?)
//! 4. Proper warmup via criterion
//! 5. Statistical analysis of results

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

    // Check if data already exists (for shared memory database case)
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
                (i % 62 + 18) as i64,
                (i as f64) * 10.0,
                i % 2 == 0,
                "2024-01-01 00:00:00",
            ))
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

    for i in 1..=ROW_COUNT {
        let name = format!("User_{}", i);
        let email = format!("user{}@example.com", i);
        conn.execute(
            "INSERT INTO users (id, name, email, age, balance, active, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                i as i64,
                &name,
                &email,
                (i % 62 + 18) as i64,
                (i as f64) * 10.0,
                if i % 2 == 0 { 1 } else { 0 },
                "2024-01-01 00:00:00",
            ],
        )
        .unwrap();
    }

    conn
}

fn bench_update_by_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("UPDATE by ID");

    // Setup databases
    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    // Prepare statements - update balance by ID
    let stoolap_stmt = stoolap_db
        .prepare("UPDATE users SET balance = $1 WHERE id = $2")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("UPDATE users SET balance = ?1 WHERE id = ?2")
        .unwrap();

    // Generate IDs to update (spread across all rows)
    let ids: Vec<i64> = (1..=1000).map(|i| ((i % ROW_COUNT) + 1) as i64).collect();

    // Benchmark Stoolap
    group.bench_function("stoolap", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            let new_balance = (idx as f64) * 10.0;
            idx += 1;
            stoolap_stmt.execute(black_box((new_balance, id))).unwrap();
            black_box(())
        });
    });

    // Benchmark SQLite
    group.bench_function("sqlite", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            let new_balance = (idx as f64) * 10.0;
            idx += 1;
            sqlite_stmt
                .execute(rusqlite::params![black_box(new_balance), black_box(id)])
                .unwrap();
            black_box(())
        });
    });

    group.finish();
}

fn bench_update_by_id_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("UPDATE by ID (batch of 100)");

    // Setup databases
    let stoolap_db = setup_stoolap();
    let sqlite_conn = setup_sqlite();

    // Prepare statements
    let stoolap_stmt = stoolap_db
        .prepare("UPDATE users SET balance = $1 WHERE id = $2")
        .unwrap();

    let mut sqlite_stmt = sqlite_conn
        .prepare("UPDATE users SET balance = ?1 WHERE id = ?2")
        .unwrap();

    // Generate IDs
    let ids: Vec<i64> = (1..=100).map(|i| ((i % ROW_COUNT) + 1) as i64).collect();

    // Benchmark Stoolap - 100 updates per iteration
    group.bench_function("stoolap", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            for (i, &id) in ids.iter().enumerate() {
                let new_balance = ((iter_count * 100 + i) as f64) * 10.0;
                stoolap_stmt.execute(black_box((new_balance, id))).unwrap();
            }
            iter_count += 1;
            black_box(())
        });
    });

    // Benchmark SQLite - 100 updates per iteration
    group.bench_function("sqlite", |b| {
        let mut iter_count = 0;
        b.iter(|| {
            for (i, &id) in ids.iter().enumerate() {
                let new_balance = ((iter_count * 100 + i) as f64) * 10.0;
                sqlite_stmt
                    .execute(rusqlite::params![black_box(new_balance), black_box(id)])
                    .unwrap();
            }
            iter_count += 1;
            black_box(())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_update_by_id, bench_update_by_id_batch);
criterion_main!(benches);
