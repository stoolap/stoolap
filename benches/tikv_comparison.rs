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

//! MVCC vs TiKV comparison benchmarks.
//!
//! Requires a running TiKV cluster and TIKV_PD_ENDPOINTS env var.
//!
//! Run:
//!   make tikv-up
//!   make bench-tikv
//!   # or directly:
//!   TIKV_PD_ENDPOINTS=127.0.0.1:2379 cargo bench --features tikv --bench tikv_comparison

#![cfg(feature = "tikv")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use stoolap::Database;

const SMALL_DATASET: usize = 100;
const MEDIUM_DATASET: usize = 1_000;

fn tikv_dsn() -> Option<String> {
    std::env::var("TIKV_PD_ENDPOINTS")
        .ok()
        .map(|ep| format!("tikv://{}", ep))
}

/// Setup a database (either memory or TiKV) with test data
fn setup_db(dsn: &str, table: &str, rows: usize) -> Database {
    let db = Database::open(dsn).unwrap();
    let _ = db.execute(&format!("DROP TABLE IF EXISTS {}", table), ());
    db.execute(
        &format!(
            "CREATE TABLE {} (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER NOT NULL,
                score FLOAT NOT NULL
            )",
            table
        ),
        (),
    )
    .unwrap();

    for i in 1..=rows {
        db.execute(
            &format!(
                "INSERT INTO {} VALUES ({}, 'item_{}', {}, {:.2})",
                table,
                i,
                i,
                i * 10,
                i as f64 * 1.5
            ),
            (),
        )
        .unwrap();
    }
    db
}

fn cleanup(db: &Database, table: &str) {
    let _ = db.execute(&format!("DROP TABLE IF EXISTS {}", table), ());
}

// ─── Point SELECT by PK ───────────────────────────────────────────

fn bench_select_by_id(c: &mut Criterion) {
    let tikv_dsn = match tikv_dsn() {
        Some(d) => d,
        None => {
            eprintln!("Skipping TiKV benchmarks: TIKV_PD_ENDPOINTS not set");
            return;
        }
    };

    let mut group = c.benchmark_group("select_by_pk");

    let mvcc_db = setup_db("memory://", "bench_sel_mvcc", SMALL_DATASET);
    let tikv_db = setup_db(&tikv_dsn, "bench_sel_tikv", SMALL_DATASET);

    group.bench_function("mvcc", |b| {
        let mut i = 1;
        b.iter(|| {
            let _: i64 = mvcc_db
                .query_one(
                    &format!("SELECT value FROM bench_sel_mvcc WHERE id = {}", i),
                    (),
                )
                .unwrap();
            i = (i % SMALL_DATASET) + 1;
            black_box(());
        });
    });

    group.bench_function("tikv", |b| {
        let mut i = 1;
        b.iter(|| {
            let _: i64 = tikv_db
                .query_one(
                    &format!("SELECT value FROM bench_sel_tikv WHERE id = {}", i),
                    (),
                )
                .unwrap();
            i = (i % SMALL_DATASET) + 1;
            black_box(());
        });
    });

    group.finish();
    cleanup(&tikv_db, "bench_sel_tikv");
}

// ─── Single-row INSERT ────────────────────────────────────────────

fn bench_insert(c: &mut Criterion) {
    let tikv_dsn = match tikv_dsn() {
        Some(d) => d,
        None => return,
    };

    let mut group = c.benchmark_group("insert_single_row");

    let mvcc_db = Database::open("memory://").unwrap();
    mvcc_db
        .execute(
            "CREATE TABLE bench_ins_mvcc (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
            (),
        )
        .unwrap();

    let tikv_db = Database::open(&tikv_dsn).unwrap();
    let _ = tikv_db.execute("DROP TABLE IF EXISTS bench_ins_tikv", ());
    tikv_db
        .execute(
            "CREATE TABLE bench_ins_tikv (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
            (),
        )
        .unwrap();

    let mut mvcc_id = 1i64;
    let mut tikv_id = 1i64;

    group.bench_function("mvcc", |b| {
        b.iter(|| {
            mvcc_db
                .execute(
                    &format!(
                        "INSERT INTO bench_ins_mvcc VALUES ({}, 'item', {})",
                        mvcc_id, mvcc_id
                    ),
                    (),
                )
                .unwrap();
            mvcc_id += 1;
            black_box(());
        });
    });

    group.bench_function("tikv", |b| {
        b.iter(|| {
            tikv_db
                .execute(
                    &format!(
                        "INSERT INTO bench_ins_tikv VALUES ({}, 'item', {})",
                        tikv_id, tikv_id
                    ),
                    (),
                )
                .unwrap();
            tikv_id += 1;
            black_box(());
        });
    });

    group.finish();
    cleanup(&tikv_db, "bench_ins_tikv");
}

// ─── UPDATE by PK ─────────────────────────────────────────────────

fn bench_update_by_id(c: &mut Criterion) {
    let tikv_dsn = match tikv_dsn() {
        Some(d) => d,
        None => return,
    };

    let mut group = c.benchmark_group("update_by_pk");

    let mvcc_db = setup_db("memory://", "bench_upd_mvcc", SMALL_DATASET);
    let tikv_db = setup_db(&tikv_dsn, "bench_upd_tikv", SMALL_DATASET);

    group.bench_function("mvcc", |b| {
        let mut i = 1;
        b.iter(|| {
            mvcc_db
                .execute(
                    &format!(
                        "UPDATE bench_upd_mvcc SET value = {} WHERE id = {}",
                        i * 100,
                        i
                    ),
                    (),
                )
                .unwrap();
            i = (i % SMALL_DATASET) + 1;
            black_box(());
        });
    });

    group.bench_function("tikv", |b| {
        let mut i = 1;
        b.iter(|| {
            tikv_db
                .execute(
                    &format!(
                        "UPDATE bench_upd_tikv SET value = {} WHERE id = {}",
                        i * 100,
                        i
                    ),
                    (),
                )
                .unwrap();
            i = (i % SMALL_DATASET) + 1;
            black_box(());
        });
    });

    group.finish();
    cleanup(&tikv_db, "bench_upd_tikv");
}

// ─── COUNT(*) ─────────────────────────────────────────────────────

fn bench_count(c: &mut Criterion) {
    let tikv_dsn = match tikv_dsn() {
        Some(d) => d,
        None => return,
    };

    let mut group = c.benchmark_group("count_star");

    for &size in &[SMALL_DATASET, MEDIUM_DATASET] {
        let mvcc_table = format!("bench_cnt_mvcc_{}", size);
        let tikv_table = format!("bench_cnt_tikv_{}", size);

        let mvcc_db = setup_db("memory://", &mvcc_table, size);
        let tikv_db = setup_db(&tikv_dsn, &tikv_table, size);

        group.bench_with_input(BenchmarkId::new("mvcc", size), &size, |b, _| {
            b.iter(|| {
                let _: i64 = mvcc_db
                    .query_one(&format!("SELECT COUNT(*) FROM {}", mvcc_table), ())
                    .unwrap();
                black_box(());
            });
        });

        group.bench_with_input(BenchmarkId::new("tikv", size), &size, |b, _| {
            b.iter(|| {
                let _: i64 = tikv_db
                    .query_one(&format!("SELECT COUNT(*) FROM {}", tikv_table), ())
                    .unwrap();
                black_box(());
            });
        });

        cleanup(&tikv_db, &tikv_table);
    }

    group.finish();
}

// ─── Full Table Scan ──────────────────────────────────────────────

fn bench_full_scan(c: &mut Criterion) {
    let tikv_dsn = match tikv_dsn() {
        Some(d) => d,
        None => return,
    };

    let mut group = c.benchmark_group("full_table_scan");

    for &size in &[SMALL_DATASET, MEDIUM_DATASET] {
        let mvcc_table = format!("bench_scan_mvcc_{}", size);
        let tikv_table = format!("bench_scan_tikv_{}", size);

        let mvcc_db = setup_db("memory://", &mvcc_table, size);
        let tikv_db = setup_db(&tikv_dsn, &tikv_table, size);

        group.bench_with_input(BenchmarkId::new("mvcc", size), &size, |b, _| {
            b.iter(|| {
                let mut count = 0;
                for row in mvcc_db
                    .query(&format!("SELECT * FROM {}", mvcc_table), ())
                    .unwrap()
                {
                    let _row = row.unwrap();
                    count += 1;
                }
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("tikv", size), &size, |b, _| {
            b.iter(|| {
                let mut count = 0;
                for row in tikv_db
                    .query(&format!("SELECT * FROM {}", tikv_table), ())
                    .unwrap()
                {
                    let _row = row.unwrap();
                    count += 1;
                }
                black_box(count);
            });
        });

        cleanup(&tikv_db, &tikv_table);
    }

    group.finish();
}

// ─── GROUP BY Aggregation ─────────────────────────────────────────

fn bench_group_by(c: &mut Criterion) {
    let tikv_dsn = match tikv_dsn() {
        Some(d) => d,
        None => return,
    };

    let mut group = c.benchmark_group("group_by_aggregation");

    let mvcc_db = setup_db("memory://", "bench_gb_mvcc", MEDIUM_DATASET);
    let tikv_db = setup_db(&tikv_dsn, "bench_gb_tikv", MEDIUM_DATASET);

    group.bench_function("mvcc", |b| {
        b.iter(|| {
            let mut count = 0;
            for row in mvcc_db
                .query(
                    "SELECT value % 10 as grp, SUM(score), COUNT(*) FROM bench_gb_mvcc GROUP BY value % 10",
                    (),
                )
                .unwrap()
            {
                let _row = row.unwrap();
                count += 1;
            }
            black_box(count);
        });
    });

    group.bench_function("tikv", |b| {
        b.iter(|| {
            let mut count = 0;
            for row in tikv_db
                .query(
                    "SELECT value % 10 as grp, SUM(score), COUNT(*) FROM bench_gb_tikv GROUP BY value % 10",
                    (),
                )
                .unwrap()
            {
                let _row = row.unwrap();
                count += 1;
            }
            black_box(count);
        });
    });

    group.finish();
    cleanup(&tikv_db, "bench_gb_tikv");
}

// ─── Transaction Commit ───────────────────────────────────────────

fn bench_transaction_commit(c: &mut Criterion) {
    let tikv_dsn = match tikv_dsn() {
        Some(d) => d,
        None => return,
    };

    let mut group = c.benchmark_group("transaction_commit");

    let mvcc_db = Database::open("memory://").unwrap();
    mvcc_db
        .execute(
            "CREATE TABLE bench_txn_mvcc (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .unwrap();

    let tikv_db = Database::open(&tikv_dsn).unwrap();
    let _ = tikv_db.execute("DROP TABLE IF EXISTS bench_txn_tikv", ());
    tikv_db
        .execute(
            "CREATE TABLE bench_txn_tikv (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .unwrap();

    let mut mvcc_id = 1i64;
    let mut tikv_id = 1i64;

    group.bench_function("mvcc", |b| {
        b.iter(|| {
            let mut txn = mvcc_db.begin().unwrap();
            txn.execute(
                &format!(
                    "INSERT INTO bench_txn_mvcc VALUES ({}, {})",
                    mvcc_id, mvcc_id
                ),
                (),
            )
            .unwrap();
            txn.commit().unwrap();
            mvcc_id += 1;
            black_box(());
        });
    });

    group.bench_function("tikv", |b| {
        b.iter(|| {
            let mut txn = tikv_db.begin().unwrap();
            txn.execute(
                &format!(
                    "INSERT INTO bench_txn_tikv VALUES ({}, {})",
                    tikv_id, tikv_id
                ),
                (),
            )
            .unwrap();
            txn.commit().unwrap();
            tikv_id += 1;
            black_box(());
        });
    });

    group.finish();
    cleanup(&tikv_db, "bench_txn_tikv");
}

criterion_group!(
    benches,
    bench_select_by_id,
    bench_insert,
    bench_update_by_id,
    bench_count,
    bench_full_scan,
    bench_group_by,
    bench_transaction_commit,
);
criterion_main!(benches);
