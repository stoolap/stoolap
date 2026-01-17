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

//! Fair benchmark comparison: Stoolap vs SQLite for SELECT by ID
//!
//! Run with: cargo bench --bench select_by_id
//! Run with SQLite comparison: cargo bench --bench select_by_id --features sqlite
//!
//! This benchmark ensures fair comparison by:
//! 1. Using prepared statements for both databases
//! 2. Same data setup (10K rows)
//! 3. Same query pattern (SELECT * WHERE id = ?)
//! 4. Proper warmup via criterion
//! 5. Statistical analysis of results

use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "sqlite")]
use rusqlite::Connection;
use std::hint::black_box;
use stoolap::Database;

const ROW_COUNT: usize = 10_000;

/// Setup Stoolap database with test data
fn setup_stoolap() -> Database {
    let db = Database::open_in_memory().unwrap();

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
#[cfg(feature = "sqlite")]
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

fn bench_select_by_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("SELECT by ID");

    // Setup databases
    let stoolap_db = setup_stoolap();
    #[cfg(feature = "sqlite")]
    let sqlite_conn = setup_sqlite();

    // Prepare statements
    let stoolap_stmt = stoolap_db
        .prepare("SELECT * FROM users WHERE id = $1")
        .unwrap();

    #[cfg(feature = "sqlite")]
    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT * FROM users WHERE id = ?1")
        .unwrap();

    // Generate IDs to query (spread across all rows for cache fairness)
    let ids: Vec<i64> = (1..=1000).map(|i| ((i % ROW_COUNT) + 1) as i64).collect();

    // Benchmark Stoolap
    group.bench_function("stoolap", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            idx += 1;
            let rows = stoolap_stmt.query(black_box((id,))).unwrap();
            let row = rows.into_iter().next().unwrap().unwrap();
            black_box(row.get::<i64>(0).unwrap())
        });
    });

    // Benchmark SQLite
    #[cfg(feature = "sqlite")]
    group.bench_function("sqlite", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            idx += 1;
            let result: i64 = sqlite_stmt
                .query_row([black_box(id)], |row| row.get(0))
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_select_by_id_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("SELECT by ID (batch of 100)");

    // Setup databases
    let stoolap_db = setup_stoolap();
    #[cfg(feature = "sqlite")]
    let sqlite_conn = setup_sqlite();

    // Prepare statements
    let stoolap_stmt = stoolap_db
        .prepare("SELECT * FROM users WHERE id = $1")
        .unwrap();

    #[cfg(feature = "sqlite")]
    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT * FROM users WHERE id = ?1")
        .unwrap();

    // Generate IDs
    let ids: Vec<i64> = (1..=100).map(|i| ((i % ROW_COUNT) + 1) as i64).collect();

    // Benchmark Stoolap - 100 queries per iteration
    group.bench_function("stoolap", |b| {
        b.iter(|| {
            for &id in &ids {
                let rows = stoolap_stmt.query(black_box((id,))).unwrap();
                let row = rows.into_iter().next().unwrap().unwrap();
                black_box(row.get::<i64>(0).unwrap());
            }
        });
    });

    // Benchmark SQLite - 100 queries per iteration
    #[cfg(feature = "sqlite")]
    group.bench_function("sqlite", |b| {
        b.iter(|| {
            for &id in &ids {
                let result: i64 = sqlite_stmt
                    .query_row([black_box(id)], |row| row.get(0))
                    .unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

fn bench_select_by_id_with_result_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("SELECT by ID (full row processing)");

    // Setup databases
    let stoolap_db = setup_stoolap();
    #[cfg(feature = "sqlite")]
    let sqlite_conn = setup_sqlite();

    // Prepare statements
    let stoolap_stmt = stoolap_db
        .prepare("SELECT * FROM users WHERE id = $1")
        .unwrap();

    #[cfg(feature = "sqlite")]
    let mut sqlite_stmt = sqlite_conn
        .prepare("SELECT * FROM users WHERE id = ?1")
        .unwrap();

    let ids: Vec<i64> = (1..=1000).map(|i| ((i % ROW_COUNT) + 1) as i64).collect();

    // Benchmark Stoolap - read all columns
    group.bench_function("stoolap", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            idx += 1;
            let rows = stoolap_stmt.query(black_box((id,))).unwrap();
            let row = rows.into_iter().next().unwrap().unwrap();
            // Read all columns
            let _id: i64 = row.get(0).unwrap();
            let _name: String = row.get(1).unwrap();
            let _email: String = row.get(2).unwrap();
            let _age: i64 = row.get(3).unwrap();
            let _balance: f64 = row.get(4).unwrap();
            let _active: bool = row.get(5).unwrap();
            let _created_at: String = row.get(6).unwrap();
            black_box((_id, _name, _email, _age, _balance, _active, _created_at))
        });
    });

    // Benchmark SQLite - read all columns
    #[cfg(feature = "sqlite")]
    group.bench_function("sqlite", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            idx += 1;
            let result: (i64, String, String, i64, f64, i64, String) = sqlite_stmt
                .query_row([black_box(id)], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                })
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_select_by_id,
    bench_select_by_id_batch,
    bench_select_by_id_with_result_processing
);
criterion_main!(benches);
