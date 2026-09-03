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

//! Two window functions with the same name keep their own values, and a
//! window's ORDER BY honours NULLS FIRST and NULLS LAST.

use stoolap::Database;

/// 100 rows: k is NULL for ids 1..=20 and 200 - id otherwise, amount is the id.
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();
    for i in 1..=100i64 {
        let k: Option<i64> = if i <= 20 { None } else { Some(200 - i) };
        db.execute("INSERT INTO t VALUES ($1, $2, $3)", (i, k, i as f64))
            .unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<Option<i64>>> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get::<Option<i64>>(i).unwrap())
                .collect()
        })
        .collect()
}

#[test]
fn test_two_windows_with_the_same_function_keep_their_own_columns() {
    let db = setup("window_same_name_columns");
    let got = rows(
        &db,
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id), ROW_NUMBER() OVER (ORDER BY amount DESC) FROM t ORDER BY id",
    );
    assert_eq!(got.len(), 100);
    assert_eq!(got[0], vec![Some(1), Some(1), Some(100)]);
    assert_eq!(got[45], vec![Some(46), Some(46), Some(55)]);
    assert_eq!(got[99], vec![Some(100), Some(100), Some(1)]);
    let got = rows(
        &db,
        "SELECT id, COUNT(*) OVER (ORDER BY id), COUNT(*) OVER (ORDER BY amount DESC), COUNT(*) OVER () FROM t ORDER BY id",
    );
    assert_eq!(got[0], vec![Some(1), Some(1), Some(100), Some(100)]);
    assert_eq!(got[99], vec![Some(100), Some(100), Some(1), Some(100)]);
}

#[test]
fn test_window_order_by_honours_nulls_first_and_last() {
    let db = setup("window_nulls_placement");
    // default: NULLs last ascending, first descending; explicit forms flip that
    for (order, null_row_rank, first_non_null_rank) in [
        ("ASC", 81, 1),
        ("DESC", 1, 21),
        ("ASC NULLS LAST", 81, 1),
        ("ASC NULLS FIRST", 1, 21),
        ("DESC NULLS FIRST", 1, 21),
        ("DESC NULLS LAST", 81, 1),
    ] {
        let got = rows(
            &db,
            &format!("SELECT id, ROW_NUMBER() OVER (ORDER BY k {order}) FROM t ORDER BY id"),
        );
        let rank_of = |id: i64| got[(id - 1) as usize][1].unwrap();
        assert!(
            (1..=20).all(|id| rank_of(id) >= null_row_rank && rank_of(id) < null_row_rank + 20),
            "ORDER BY k {order}: NULL rows ranked {:?}",
            (1..=20).map(rank_of).collect::<Vec<_>>()
        );
        // id 100 has the smallest non-null k, id 21 the largest
        let smallest = if order.starts_with("ASC") { 100 } else { 21 };
        assert_eq!(rank_of(smallest), first_non_null_rank, "ORDER BY k {order}");
    }
}
