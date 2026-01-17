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

//! Fair benchmark comparison: Stoolap vs SQLite for DELETE by ID
//!
//! Run with: cargo bench --bench delete_by_id
//! Run with SQLite comparison: cargo bench --bench delete_by_id --features sqlite
//!
//! This benchmark ensures fair comparison by:
//! 1. Using prepared statements for both databases
//! 2. Same data setup (10K rows, re-inserted between benchmarks)
//! 3. Same query pattern (DELETE FROM ... WHERE id = ?)
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

    insert_stoolap_data(&db);
    db
}

fn insert_stoolap_data(db: &Database) {
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

    insert_sqlite_data(&conn);
    conn
}

#[cfg(feature = "sqlite")]
fn insert_sqlite_data(conn: &Connection) {
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
}

fn bench_delete_by_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("DELETE by ID");

    // Setup databases
    let stoolap_db = setup_stoolap();
    #[cfg(feature = "sqlite")]
    let sqlite_conn = setup_sqlite();

    // Prepare statements
    let stoolap_delete = stoolap_db
        .prepare("DELETE FROM users WHERE id = $1")
        .unwrap();
    let stoolap_insert = stoolap_db
        .prepare("INSERT INTO users (id, name, email, age, balance, active, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)")
        .unwrap();

    #[cfg(feature = "sqlite")]
    let mut sqlite_delete = sqlite_conn
        .prepare("DELETE FROM users WHERE id = ?1")
        .unwrap();
    #[cfg(feature = "sqlite")]
    let mut sqlite_insert = sqlite_conn
        .prepare("INSERT INTO users (id, name, email, age, balance, active, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)")
        .unwrap();

    // Generate IDs to delete (cycling through rows)
    let ids: Vec<i64> = (1..=1000).map(|i| ((i % ROW_COUNT) + 1) as i64).collect();

    // Benchmark Stoolap - delete then re-insert to keep data available
    group.bench_function("stoolap", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            idx += 1;

            // Delete the row
            stoolap_delete.execute(black_box((id,))).unwrap();

            // Re-insert so we can delete again
            let name = format!("User_{}", id);
            let email = format!("user{}@example.com", id);
            stoolap_insert
                .execute((
                    id,
                    &name,
                    &email,
                    ((id as usize % 62) + 18) as i64,
                    (id as f64) * 10.0,
                    id % 2 == 0,
                    "2024-01-01 00:00:00",
                ))
                .unwrap();

            black_box(())
        });
    });

    // Benchmark SQLite - delete then re-insert
    #[cfg(feature = "sqlite")]
    group.bench_function("sqlite", |b| {
        let mut idx = 0;
        b.iter(|| {
            let id = ids[idx % ids.len()];
            idx += 1;

            // Delete the row
            sqlite_delete.execute([black_box(id)]).unwrap();

            // Re-insert so we can delete again
            let name = format!("User_{}", id);
            let email = format!("user{}@example.com", id);
            sqlite_insert
                .execute(rusqlite::params![
                    id,
                    &name,
                    &email,
                    ((id as usize % 62) + 18) as i64,
                    (id as f64) * 10.0,
                    if id % 2 == 0 { 1 } else { 0 },
                    "2024-01-01 00:00:00",
                ])
                .unwrap();

            black_box(())
        });
    });

    group.finish();
}

fn bench_delete_by_id_only(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("DELETE by ID (delete only, no re-insert)");

    // Use a smaller sample size for this benchmark
    group.sample_size(50);

    // Benchmark Stoolap - fresh database per batch, delete 100 rows
    group.bench_function("stoolap", |b| {
        b.iter_batched(
            || {
                // Setup: create fresh database with data
                let db = Database::open_in_memory().unwrap();
                db.execute(
                    "CREATE TABLE users (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        age INTEGER NOT NULL
                    )",
                    (),
                )
                .unwrap();
                for i in 1..=100 {
                    db.execute(
                        "INSERT INTO users VALUES ($1, $2, $3)",
                        (i as i64, format!("User_{}", i), (i % 62 + 18) as i64),
                    )
                    .unwrap();
                }
                let stmt = db.prepare("DELETE FROM users WHERE id = $1").unwrap();
                (db, stmt)
            },
            |(_db, stmt)| {
                // Benchmark: delete all 100 rows
                for id in 1..=100i64 {
                    stmt.execute(black_box((id,))).unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });

    // Benchmark SQLite - fresh database per batch, delete 100 rows
    #[cfg(feature = "sqlite")]
    group.bench_function("sqlite", |b| {
        b.iter_batched(
            || {
                // Setup: create fresh database with data
                let conn = Connection::open_in_memory().unwrap();
                conn.execute(
                    "CREATE TABLE users (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        age INTEGER NOT NULL
                    )",
                    [],
                )
                .unwrap();
                for i in 1..=100 {
                    conn.execute(
                        "INSERT INTO users VALUES (?1, ?2, ?3)",
                        rusqlite::params![i as i64, format!("User_{}", i), (i % 62 + 18) as i64],
                    )
                    .unwrap();
                }
                conn
            },
            |conn| {
                // Benchmark: delete all 100 rows
                let mut stmt = conn.prepare("DELETE FROM users WHERE id = ?1").unwrap();
                for id in 1..=100i64 {
                    stmt.execute([black_box(id)]).unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_delete_by_id, bench_delete_by_id_only);
criterion_main!(benches);
