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

//! GROUP BY throughput across grouping-key distributions
//!
//! Run with: cargo bench --bench group_by_keys
//!
//! The grouping fast path keys its hash map by the raw integer for an integer
//! column, and by an encoded form of the bit pattern for a float column. The
//! encoding exists because some perfectly ordinary column values are
//! adversarial for a hash that can only see part of the key: a DOUBLE column
//! holding round values has all-zero low mantissa bits, and it used to be
//! keyed by `f64::to_bits()` directly, which put every group in one bucket.
//! An integer column holding a large stride has the same shape and is still
//! keyed raw. This benchmark keeps those distributions next to the benign ones
//! so a change that reintroduces the clustering shows up as a large, obvious
//! regression rather than as a slow query somebody reports later.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use stoolap::Database;

/// One row per group, so the grouping map holds exactly this many keys.
///
/// The value is chosen so the map sits at its 3/4 grow threshold rather than
/// wherever a round number happens to land: capacity is
/// `next_power_of_two(n * 4 / 3)`, so 12288 keys fill 16384 slots exactly.
/// Probe length is worst there, which is where a weak hash shows itself.
const ROW_COUNT: i64 = 12_288;
fn setup(name: &str, key: impl Fn(i64) -> f64) -> Database {
    let db = Database::open(&format!("memory://bench_group_by_{}", name)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k FLOAT, v INTEGER)",
        (),
    )
    .unwrap();
    let insert = db
        .prepare("INSERT INTO t (id, k, v) VALUES ($1, $2, $3)")
        .unwrap();
    for i in 1..=ROW_COUNT {
        insert.execute((i, key(i), i)).unwrap();
    }
    db
}

fn setup_int(name: &str, key: impl Fn(i64) -> i64) -> Database {
    let db = Database::open(&format!("memory://bench_group_by_{}", name)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    let insert = db
        .prepare("INSERT INTO t (id, k, v) VALUES ($1, $2, $3)")
        .unwrap();
    for i in 1..=ROW_COUNT {
        insert.execute((i, key(i), i)).unwrap();
    }
    db
}

fn bench_group_by_keys(c: &mut Criterion) {
    let mut group = c.benchmark_group("GROUP BY key distribution");

    let databases = vec![
        // Benign shapes: dense row ids and values with a full mantissa.
        ("integer_sequential", setup_int("int_seq", |i| i)),
        (
            "float_irregular",
            setup("float_irregular", |i| i as f64 * 1.000_000_123_4),
        ),
        // Adversarial shapes: entropy sits above the low bits of the key.
        ("float_whole", setup("float_whole", |i| i as f64)),
        ("float_money", setup("float_money", |i| i as f64 * 0.01)),
        ("integer_stride_2p32", setup_int("int_stride", |i| i << 32)),
    ];

    for (name, db) in &databases {
        let stmt = db.prepare("SELECT k, SUM(v) FROM t GROUP BY k").unwrap();
        group.bench_function(*name, |b| {
            b.iter(|| {
                let rows = stmt.query(()).unwrap();
                for row in rows {
                    black_box(row.unwrap());
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_group_by_keys);
criterion_main!(benches);
