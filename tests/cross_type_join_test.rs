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

//! Cross-type numeric equi-joins: INTEGER keys must meet equal FLOAT keys
//! in every join algorithm (they hashed into different buckets before,
//! returning zero rows).

use stoolap::Database;

fn setup(dsn: &str, n: i64, shuffled: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE ta (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE tb (id INTEGER PRIMARY KEY, k FLOAT, g FLOAT)",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..n {
        let (ka, kb) = if shuffled {
            ((i * 37) % n, ((i * 53) % n) as f64)
        } else {
            (i, i as f64)
        };
        tx.execute("INSERT INTO ta VALUES ($1, $2, $3)", (i, ka, i % 5))
            .unwrap();
        tx.execute(
            "INSERT INTO tb VALUES ($1, $2, $3)",
            (i, kb, (i % 5) as f64),
        )
        .unwrap();
    }
    tx.commit().unwrap();
    db
}

#[test]
fn int_float_equi_join_hash_path() {
    // Shuffled keys defeat the sorted-input merge join
    let db = setup("memory://xtype_hash", 100, true);
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(c, 100);
}

#[test]
fn int_float_equi_join_sorted_path() {
    // 400+400 sorted rows: past MERGE_JOIN_MIN_ROWS (500 total), below
    // the 200-row nested-loop ceiling per side, so the planner picks
    // merge join for sorted inputs
    let db = setup("memory://xtype_sorted", 400, false);
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(c, 400);
}

#[test]
fn int_float_multi_key_join() {
    let db = setup("memory://xtype_multi", 100, false);
    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k AND ta.g = tb.g",
            (),
        )
        .unwrap();
    assert_eq!(c, 100);
}

#[test]
fn int_float_left_join_and_fractional() {
    let db = setup("memory://xtype_left", 100, false);
    // A fractional float can never match an INTEGER key
    db.execute("INSERT INTO tb VALUES (1000, 5.5, 0.0)", ())
        .unwrap();
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(c, 100);
    // LEFT JOIN keeps every left row exactly once when all match
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta LEFT JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(c, 100);
}

#[test]
fn int_float_join_null_keys_never_match() {
    let db = setup("memory://xtype_null", 10, false);
    db.execute("INSERT INTO ta VALUES (100, NULL, 0)", ())
        .unwrap();
    db.execute("INSERT INTO tb VALUES (100, NULL, 0.0)", ())
        .unwrap();
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(c, 10);
}

#[test]
fn int_float_join_parallel_path() {
    // Above the parallel join threshold (10K build rows)
    let db = setup("memory://xtype_par", 12_000, true);
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(c, 12_000);
}

#[test]
fn large_int_float_boundary_exact() {
    // 2^63 as float must not join i64::MAX (exact comparison above 2^53)
    let db = Database::open("memory://xtype_bound").unwrap();
    db.execute("CREATE TABLE ta (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE tb (id INTEGER PRIMARY KEY, k FLOAT)", ())
        .unwrap();
    db.execute("INSERT INTO ta VALUES (1, 9223372036854775807)", ())
        .unwrap();
    db.execute("INSERT INTO tb VALUES (1, 9223372036854775807.0)", ())
        .unwrap();
    db.execute("INSERT INTO ta VALUES (2, 42)", ()).unwrap();
    db.execute("INSERT INTO tb VALUES (2, 42.0)", ()).unwrap();
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    // 9223372036854775807.0 rounds to 2^63 which is not representable as
    // i64, so only the 42 pair matches
    assert_eq!(c, 1);
}

#[test]
fn float_zero_sign_join_matches_where_semantics() {
    let db = Database::open("memory://xtype_zero").unwrap();
    db.execute("CREATE TABLE ta (id INTEGER PRIMARY KEY, k FLOAT)", ())
        .unwrap();
    db.execute("CREATE TABLE tb (id INTEGER PRIMARY KEY, k FLOAT)", ())
        .unwrap();
    db.execute("INSERT INTO ta VALUES (1, 0.0)", ()).unwrap();
    db.execute("INSERT INTO tb VALUES (1, 0.0 / -1.0)", ())
        .unwrap();
    db.execute("INSERT INTO ta VALUES (2, 7.5)", ()).unwrap();
    db.execute("INSERT INTO tb VALUES (2, 7.5)", ()).unwrap();

    // The WHERE evaluator says 0.0 = -0.0; the join must agree
    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM ta WHERE k = (SELECT k FROM tb WHERE id = 1)",
            (),
        )
        .unwrap();
    assert_eq!(c, 1, "expression equality baseline");
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(c, 2, "join must match IEEE equality (0.0 = -0.0)");
}

#[test]
fn float_infinity_self_join() {
    let db = Database::open("memory://xtype_inf").unwrap();
    db.execute("CREATE TABLE ta (id INTEGER PRIMARY KEY, k FLOAT)", ())
        .unwrap();
    db.execute("CREATE TABLE tb (id INTEGER PRIMARY KEY, k FLOAT)", ())
        .unwrap();
    db.execute("INSERT INTO ta VALUES (1, 1e308 * 10.0)", ())
        .unwrap();
    db.execute("INSERT INTO tb VALUES (1, 1e308 * 10.0)", ())
        .unwrap();
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(c, 1, "Infinity = Infinity must join");
}

#[test]
fn float_zero_and_infinity_parallel_path() {
    // 11K build rows exceed the 10K parallel-join threshold, so this
    // exercises verify_composite_key_equality's float arm (an epsilon
    // there dropped Infinity self-joins and previously diverged from
    // the streaming verifier on 0.0 vs -0.0)
    let db = Database::open("memory://xtype_par_float").unwrap();
    db.execute("CREATE TABLE ta (id INTEGER PRIMARY KEY, k FLOAT)", ())
        .unwrap();
    db.execute("CREATE TABLE tb (id INTEGER PRIMARY KEY, k FLOAT)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..11_000i64 {
        let k = i as f64 + 0.5;
        tx.execute("INSERT INTO ta VALUES ($1, $2)", (i, k))
            .unwrap();
        tx.execute("INSERT INTO tb VALUES ($1, $2)", (i, k))
            .unwrap();
    }
    tx.execute("INSERT INTO ta VALUES (20000, 0.0)", ())
        .unwrap();
    tx.execute("INSERT INTO tb VALUES (20000, 0.0 / -1.0)", ())
        .unwrap();
    tx.execute("INSERT INTO ta VALUES (20001, 1e308 * 10.0)", ())
        .unwrap();
    tx.execute("INSERT INTO tb VALUES (20001, 1e308 * 10.0)", ())
        .unwrap();
    tx.commit().unwrap();

    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM ta JOIN tb ON ta.k = tb.k", ())
        .unwrap();
    assert_eq!(
        c, 11_002,
        "parallel join must match 0.0 = -0.0 and Infinity"
    );
}
