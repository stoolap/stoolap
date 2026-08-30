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

//! Float constants compared against INTEGER columns must compare
//! numerically, never by truncating the float to i64.

use stoolap::api::Database;

fn row_count(db: &Database, sql: &str) -> usize {
    db.query(sql, ()).unwrap().collect_vec().unwrap().len()
}

#[test]
fn pk_lookup_float_literal_matches_standard_path() {
    let db = Database::open(&format!("memory://float_int_{}", line!())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (5, 'x')", ()).unwrap();

    // id = 5.5 matches nothing; a fast path that truncates 5.5 to 5 is wrong
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id = 5.5"), 0);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id = 5.5"), 0);
    // id = 5.0 equals integer 5
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id = 5.0"), 1);
}

#[test]
fn pk_update_delete_float_key_does_not_touch_truncated_row() {
    let db = Database::open(&format!("memory://float_int_{}", line!())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (5, 'x')", ()).unwrap();

    // UPDATE via fractional literal key must not touch row 5
    for _ in 0..2 {
        let n = db
            .execute("UPDATE t SET v = 'y' WHERE id = 5.5", ())
            .unwrap();
        assert_eq!(n, 0);
    }
    let v: String = db.query_one("SELECT v FROM t WHERE id = 5", ()).unwrap();
    assert_eq!(v, "x");

    // Same through a prepared statement with a float parameter
    let stmt = db.prepare("UPDATE t SET v = 'z' WHERE id = $1").unwrap();
    for _ in 0..2 {
        assert_eq!(stmt.execute((5.5f64,)).unwrap(), 0);
    }
    let v: String = db.query_one("SELECT v FROM t WHERE id = 5", ()).unwrap();
    assert_eq!(v, "x");

    // DELETE via fractional key must not delete row 5
    for _ in 0..2 {
        assert_eq!(db.execute("DELETE FROM t WHERE id = 5.5", ()).unwrap(), 0);
    }
    let n: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(n, 1);
}

#[test]
fn pk_lookup_float_at_i64_boundary_does_not_hit_max_row() {
    let db = Database::open(&format!("memory://float_int_{}", line!())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (9223372036854775807, 'max')", ())
        .unwrap();

    // 2^63 is not representable as i64; a saturating cast targets i64::MAX
    for _ in 0..2 {
        assert_eq!(
            row_count(&db, "SELECT * FROM t WHERE id = 9223372036854775808.0"),
            0
        );
    }
    let stmt = db.prepare("SELECT * FROM t WHERE id = $1").unwrap();
    for _ in 0..2 {
        assert_eq!(
            stmt.query((9.223372036854776e18f64,))
                .unwrap()
                .collect_vec()
                .unwrap()
                .len(),
            0
        );
    }
    // UPDATE/DELETE must not touch the i64::MAX row either
    assert_eq!(
        db.execute("UPDATE t SET v = 'y' WHERE id = 9223372036854775808.0", ())
            .unwrap(),
        0
    );
    assert_eq!(
        db.execute("DELETE FROM t WHERE id = 9223372036854775808.0", ())
            .unwrap(),
        0
    );
    let v: String = db
        .query_one("SELECT v FROM t WHERE id = 9223372036854775807", ())
        .unwrap();
    assert_eq!(v, "max");
}

#[test]
fn integer_column_float_comparisons_are_numeric() {
    let db = Database::open("memory://float_int_cmp").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (5, 5)", ()).unwrap();

    // Equality and inequality: 5 vs 5.4 / 5.5
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n = 5.4"), 0);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n != 5.5"), 1);
    // Ordering: 5 < 5.5, 5 > 4.5, 5 <= 4.5 false, 5 >= 5.5 false
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n < 5.5"), 1);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n > 4.5"), 1);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n <= 4.5"), 0);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n >= 5.5"), 0);
    // Exact float boundaries still match
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n = 5.0"), 1);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n <= 5.0"), 1);
    // The same comparisons on the PK column (index-probe path)
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id < 5.5"), 1);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id >= 5.5"), 0);
    // Expression on the left defeats pushdown; VM path must agree
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n + 0 = 5.5"), 0);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n + 0 < 5.5"), 1);
}

#[test]
fn integer_column_float_in_list_and_between() {
    let db = Database::open("memory://float_int_inlist").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (5, 5)", ()).unwrap();

    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n IN (5.5, 6.5)"), 0);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE n IN (5.0, 6.5)"), 1);
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE n BETWEEN 5.1 AND 5.9"),
        0
    );
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE n BETWEEN 4.5 AND 5.5"),
        1
    );
}

#[test]
fn integer_column_float_aggregate_pushdown() {
    let db = Database::open("memory://float_int_agg").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..1000i64 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();

    // 0..=999: values > 499.5 are 500..=999 -> 500 rows
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE n > 499.5", ())
        .unwrap();
    assert_eq!(c, 500);
    let c: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE n = 499.5", ())
        .unwrap();
    assert_eq!(c, 0);
    let s: i64 = db
        .query_one("SELECT SUM(n) FROM t WHERE n < 2.5", ())
        .unwrap();
    assert_eq!(s, 3); // 0 + 1 + 2
}

#[test]
fn in_list_out_of_range_whole_float_does_not_match_extremes() {
    let db = Database::open("memory://float_int_inrange").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 9223372036854775807)", ())
        .unwrap();

    // 2^63 is whole but saturates to i64::MAX under a cast
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE n IN (1, 9223372036854775808.0)"),
        0
    );
    assert_eq!(
        row_count(
            &db,
            "SELECT * FROM t WHERE n NOT IN (1, 9223372036854775808.0)"
        ),
        1
    );
    // 2^53 + 1 rounds to 2^53 as f64; the linear fallback must not match
    db.execute("INSERT INTO t VALUES (2, 9007199254740993)", ())
        .unwrap();
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE n IN (1.5, 9007199254740992.0)"),
        0
    );
}

#[test]
fn join_on_integer_float_keys_is_exact_at_extremes() {
    let db = Database::open("memory://float_int_join").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, i INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
    db.execute("INSERT INTO a VALUES (1, 9223372036854775807)", ())
        .unwrap();
    db.execute("INSERT INTO b VALUES (1, 9223372036854775808.0)", ())
        .unwrap();

    // i64::MAX must not join Float(2^63)
    assert_eq!(row_count(&db, "SELECT * FROM a JOIN b ON a.i = b.f"), 0);
    // NOTE: representable cross-type pairs (42 vs 42.0) do not hash-join
    // on main either; the join hash table buckets Integer and Float keys
    // separately. Tracked as a separate pre-existing bug.
}

#[test]
fn float_column_integer_threshold_is_exact_at_extremes() {
    let db = Database::open("memory://float_int_thresh").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
    // Largest f64 below i64::MAX
    db.execute("INSERT INTO t VALUES (1, 9223372036854774784.0)", ())
        .unwrap();

    // cell < threshold, so >= must reject; a rounded cast accepts it
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE f >= 9223372036854774785"),
        0
    );
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE f < 9223372036854774785"),
        1
    );
}

#[test]
fn semantic_cache_distinguishes_large_integer_predicates() {
    let db = Database::open("memory://float_int_semcache").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 9223372036854775807)", ())
        .unwrap();

    // Warm the cache with the broader predicate, then run one that differs
    // only below f64 precision: they must not share a cache identity
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE n > 9223372036854775806"),
        1
    );
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE n > 9223372036854775807"),
        0
    );
}

#[test]
fn between_mixed_bounds_are_exact_at_extremes() {
    let db = Database::open("memory://float_int_btwmix").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 9223372036854775807)", ())
        .unwrap();

    // Integer lower bound must not round through f64: i64::MAX stays
    // within [i64::MAX, 2^63]
    assert_eq!(
        row_count(
            &db,
            "SELECT * FROM t WHERE n BETWEEN 9223372036854775807 AND 9223372036854775808.0"
        ),
        1
    );
    // i64::MAX - 1 is below the integer lower bound
    db.execute("UPDATE t SET n = 9223372036854775806 WHERE id = 1", ())
        .unwrap();
    assert_eq!(
        row_count(
            &db,
            "SELECT * FROM t WHERE n BETWEEN 9223372036854775807 AND 9223372036854775808.0"
        ),
        0
    );

    // Float cell against an integer upper bound: 2^63 is above i64::MAX
    db.execute("CREATE TABLE tf (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
    db.execute("INSERT INTO tf VALUES (1, 9223372036854775808.0)", ())
        .unwrap();
    assert_eq!(
        row_count(
            &db,
            "SELECT * FROM tf WHERE f BETWEEN 0 AND 9223372036854775807"
        ),
        0
    );
    assert_eq!(
        row_count(
            &db,
            "SELECT * FROM tf WHERE f BETWEEN 9223372036854775807 AND 1e19"
        ),
        1
    );
}

#[test]
fn i64_min_against_float_column_does_not_panic() {
    let db = Database::open("memory://float_int_min").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
    // -2^63 is exactly representable as f64
    db.execute("INSERT INTO t VALUES (1, -9223372036854775808.0)", ())
        .unwrap();

    let stmt = db.prepare("SELECT * FROM t WHERE f = $1").unwrap();
    for _ in 0..2 {
        assert_eq!(
            stmt.query((i64::MIN,))
                .unwrap()
                .collect_vec()
                .unwrap()
                .len(),
            1
        );
    }
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE f <= -9223372036854775807"),
        1
    );
}
