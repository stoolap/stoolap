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

//! A boolean handed to SUM or AVG counts as one or nought, so a condition
//! can be summed to count the rows it holds for, and averaged for the
//! share of them. A NULL is left out as it is for any other value.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, b BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 50, TRUE), (2, 150, FALSE), (3, NULL, NULL), (4, 250, TRUE)",
        (),
    )
    .unwrap();
    db
}

fn one(db: &Database, sql: &str) -> String {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get::<Option<String>>(0)
        .unwrap()
        .unwrap_or_else(|| "NULL".into())
}

#[test]
fn test_summing_a_condition_counts_the_rows_it_holds_for() {
    let db = setup("boolean_agg_sum");
    assert_eq!(one(&db, "SELECT SUM(v > 100) FROM t"), "2");
    assert_eq!(one(&db, "SELECT SUM(b) FROM t"), "2");
    assert_eq!(one(&db, "SELECT SUM(TRUE) FROM t"), "4");
    assert_eq!(
        one(&db, "SELECT SUM(v > 100) FROM t"),
        one(
            &db,
            "SELECT SUM(CASE WHEN v > 100 THEN 1 ELSE 0 END) FROM t"
        ),
        "the same count the spelled-out form gives"
    );
}

#[test]
fn test_averaging_a_condition_gives_the_share_of_rows() {
    let db = setup("boolean_agg_avg");
    // Three rows carry a boolean; two of them are true
    let share: f64 = db
        .query("SELECT AVG(b) FROM t", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert!((share - 2.0 / 3.0).abs() < 1e-9, "{share}");
}

#[test]
fn test_a_null_boolean_is_left_out() {
    let db = setup("boolean_agg_null");
    assert_eq!(one(&db, "SELECT SUM(b) FROM t WHERE id = 3"), "NULL");
    assert_eq!(one(&db, "SELECT COUNT(b) FROM t"), "3");
}

fn int(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

#[test]
fn test_summing_distinct_booleans() {
    let db = setup("boolean_sum_distinct");
    assert_eq!(int(&db, "SELECT SUM(DISTINCT b) FROM t"), 1);
    assert_eq!(int(&db, "SELECT SUM(DISTINCT b) FROM t WHERE NOT b"), 0);
}

#[test]
fn test_boolean_sums_over_many_rows_and_groups() {
    let db = Database::open("memory://boolean_sum_many").unwrap();
    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, b BOOLEAN)",
        (),
    )
    .unwrap();
    // per k: (count of TRUE, count of non-NULL)
    let mut expect = [(0i64, 0i64); 4];
    let mut values = String::new();
    for id in 1..=20_000i64 {
        let k = (id % 4) as usize;
        let b = if id % 7 == 0 {
            "NULL"
        } else if id % 3 == 0 {
            expect[k].0 += 1;
            "TRUE"
        } else {
            "FALSE"
        };
        if b != "NULL" {
            expect[k].1 += 1;
        }
        if !values.is_empty() {
            values.push(',');
        }
        values.push_str(&format!("({id}, {k}, {b})"));
        if id % 500 == 0 {
            db.execute(&format!("INSERT INTO m VALUES {values}"), ())
                .unwrap();
            values.clear();
        }
    }
    let total: i64 = expect.iter().map(|e| e.0).sum();
    assert_eq!(int(&db, "SELECT SUM(b) FROM m"), total);
    assert_eq!(int(&db, "SELECT SUM(b) FROM m WHERE k = 1"), expect[1].0);

    let grouped = |sql: &str| -> Vec<(i64, i64, f64)> {
        db.query(sql, ())
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (
                    r.get::<i64>(0).unwrap(),
                    r.get::<i64>(1).unwrap(),
                    r.get::<f64>(2).unwrap(),
                )
            })
            .collect()
    };
    let rows = grouped("SELECT k, SUM(b), AVG(b) FROM m GROUP BY k ORDER BY k");
    assert_eq!(rows.len(), 4);
    for (k, sum, avg) in rows {
        let (ones, seen) = expect[k as usize];
        assert_eq!(sum, ones, "k {k}");
        assert!(
            (avg - ones as f64 / seen as f64).abs() < 1e-9,
            "k {k}: {avg}"
        );
    }

    // tombstones send the sum down the scanning path
    db.execute("DELETE FROM m WHERE id % 5 = 0", ()).unwrap();
    let mut kept = [(0i64, 0i64); 4];
    for id in (1..=20_000i64).filter(|id| id % 5 != 0 && id % 7 != 0) {
        let k = (id % 4) as usize;
        kept[k].1 += 1;
        if id % 3 == 0 {
            kept[k].0 += 1;
        }
    }
    let total: i64 = kept.iter().map(|e| e.0).sum();
    assert_eq!(int(&db, "SELECT SUM(b) FROM m"), total);
    for (k, sum, avg) in grouped("SELECT k, SUM(b), AVG(b) FROM m GROUP BY k ORDER BY k") {
        let (ones, seen) = kept[k as usize];
        assert_eq!(sum, ones, "after delete, k {k}");
        assert!(
            (avg - ones as f64 / seen as f64).abs() < 1e-9,
            "after delete, k {k}: {avg}"
        );
    }
}
