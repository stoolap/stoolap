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

//! GROUP BY across hash-map resize boundaries: manual raw-entry hashes
//! must match the map's own hasher exactly, or the resize rehash
//! scatters entries and duplicate groups appear.

use stoolap::Database;

#[test]
fn group_by_distinct_survives_map_resize() {
    let db = Database::open("memory://gb_resize_slow").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    let mut id = 0i64;
    // 200 distinct keys, each with two distinct v values
    for round in 0..2i64 {
        for k in 0..200i64 {
            tx.execute("INSERT INTO t VALUES ($1, $2, $3)", (id, k, round))
                .unwrap();
            id += 1;
        }
    }
    tx.commit().unwrap();

    // COUNT(DISTINCT) forces the general grouping path
    let rows = db
        .query("SELECT k, COUNT(DISTINCT v) FROM t GROUP BY k", ())
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 200, "groups split across a map resize");
    for r in &rows {
        let c: i64 = r.get(1).unwrap();
        assert_eq!(c, 2, "a group lost rows to a duplicate entry");
    }
}

#[test]
fn multi_column_group_by_survives_map_resize() {
    let db = Database::open("memory://gb_resize_fast").unwrap();
    db.execute(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    let mut id = 0i64;
    // 500 distinct (a, b) pairs inserted twice
    for round in 0..2i64 {
        let _ = round;
        for p in 0..500i64 {
            tx.execute("INSERT INTO t2 VALUES ($1, $2, $3)", (id, p / 25, p % 25))
                .unwrap();
            id += 1;
        }
    }
    tx.commit().unwrap();

    // Simple aggregate keeps this on the multi-column fast path
    let rows = db
        .query("SELECT a, b, COUNT(*) FROM t2 GROUP BY a, b", ())
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 500, "groups split across a map resize");
    for r in &rows {
        let c: i64 = r.get(2).unwrap();
        assert_eq!(c, 2, "a group lost rows to a duplicate entry");
    }
}
