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
