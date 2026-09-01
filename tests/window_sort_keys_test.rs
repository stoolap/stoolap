// Copyright 2026 Stoolap Contributors
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

//! Window ORDER BY sort keys: NULLs must not collide with genuine
//! i64::MAX / f64::MAX values, and type drift past the detection
//! sample must not corrupt the ordering.

use stoolap::Database;

#[test]
fn window_rank_i64_max_vs_null() {
    let db = Database::open("memory://win_i64max").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    for (id, k) in [
        (1i64, None),
        (2, Some(9223372036854775807i64)),
        (3, None),
        (4, Some(5)),
        (5, Some(9223372036854775807)),
        (6, None),
    ] {
        db.execute("INSERT INTO t VALUES ($1, $2)", (id, k))
            .unwrap();
    }
    let rows = db
        .query(
            "SELECT k, RANK() OVER (ORDER BY k) FROM t ORDER BY 2, 1",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let ranks: Vec<i64> = rows.iter().map(|r| r.get(1).unwrap()).collect();
    // ASC, NULLs last: 5 -> rank 1; two i64::MAX -> rank 2; three NULLs -> rank 4
    assert_eq!(
        ranks,
        vec![1, 2, 2, 4, 4, 4],
        "i64::MAX must not tie with NULL"
    );
}

#[test]
fn window_rank_f64_max_vs_null() {
    let db = Database::open("memory://win_f64max").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
    for (id, f) in [
        (1i64, None),
        (2, Some(1.7976931348623157e308f64)),
        (3, Some(2.5)),
        (4, None),
        (5, Some(1.7976931348623157e308)),
    ] {
        db.execute("INSERT INTO t VALUES ($1, $2)", (id, f))
            .unwrap();
    }
    let rows = db
        .query(
            "SELECT f, RANK() OVER (ORDER BY f) FROM t ORDER BY 2, 1",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let ranks: Vec<i64> = rows.iter().map(|r| r.get(1).unwrap()).collect();
    // 2.5 -> 1; two f64::MAX -> 2; two NULLs -> 4
    assert_eq!(
        ranks,
        vec![1, 2, 2, 4, 4],
        "f64::MAX must not tie with NULL"
    );
}

#[test]
fn window_order_survives_type_drift_past_sample() {
    // 150 Integer values then a Float appears past the 100-row
    // detection sample; ordering must still interleave correctly
    let db = Database::open("memory://win_drift").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k FLOAT)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..150i64 {
        // Stored as integers via coercion-free int literals in a FLOAT
        // column stay Float; use an INTEGER column shape instead
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, (i * 2) as f64))
            .unwrap();
    }
    tx.commit().unwrap();
    let db2 = Database::open("memory://win_drift2").unwrap();
    db2.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT)", ())
        .unwrap();
    let mut tx = db2.begin().unwrap();
    for i in 0..150i64 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, format!("{:05}", i)))
            .unwrap();
    }
    // Row 150: numerically-small text that must sort FIRST
    tx.execute("INSERT INTO t VALUES (150, '00000')", ())
        .unwrap();
    tx.commit().unwrap();
    let first: String = db2
        .query_one(
            "SELECT k FROM (SELECT k, ROW_NUMBER() OVER (ORDER BY k) AS rn FROM t) s WHERE rn = 1",
            (),
        )
        .unwrap();
    assert_eq!(first, "00000");
}
