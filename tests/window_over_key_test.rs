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

//! Window functions sharing partition work must only share it when the
//! OVER clauses are genuinely identical.

use stoolap::Database;

#[test]
fn positional_parameters_do_not_share_partitions() {
    let db = Database::open("memory://win_key_param").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for i in 1..=6i64 {
        db.execute("INSERT INTO t VALUES ($1, $2)", (i, i)).unwrap();
    }
    // Two OVER clauses differing only in a bound parameter value
    let rows = db
        .query(
            "SELECT id, \
             ROW_NUMBER() OVER (PARTITION BY v % ? ORDER BY v) AS a, \
             ROW_NUMBER() OVER (PARTITION BY v % ? ORDER BY v) AS b \
             FROM t ORDER BY id",
            (2i64, 3i64),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let a: Vec<i64> = rows.iter().map(|r| r.get(1).unwrap()).collect();
    let b: Vec<i64> = rows.iter().map(|r| r.get(2).unwrap()).collect();
    assert_eq!(a, vec![1, 1, 2, 2, 3, 3], "v % 2 partitions");
    assert_eq!(b, vec![1, 1, 1, 2, 2, 2], "v % 3 partitions");
}

#[test]
fn quoted_comma_identifier_does_not_collide_with_two_columns() {
    let db = Database::open("memory://win_key_comma").unwrap();
    db.execute(
        "CREATE TABLE qc (id INTEGER PRIMARY KEY, \"a,b\" INTEGER, a INTEGER, b INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    for (id, ab, a, b, v) in [
        (1i64, 1i64, 1i64, 1i64, 10i64),
        (2, 1, 1, 2, 20),
        (3, 2, 2, 1, 30),
        (4, 2, 2, 2, 40),
    ] {
        db.execute(
            "INSERT INTO qc VALUES ($1, $2, $3, $4, $5)",
            (id, ab, a, b, v),
        )
        .unwrap();
    }
    let rows = db
        .query(
            "SELECT v, \
             ROW_NUMBER() OVER (PARTITION BY \"a,b\" ORDER BY v) AS k1, \
             ROW_NUMBER() OVER (PARTITION BY a, b ORDER BY v) AS k2 \
             FROM qc ORDER BY v",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let k1: Vec<i64> = rows.iter().map(|r| r.get(1).unwrap()).collect();
    let k2: Vec<i64> = rows.iter().map(|r| r.get(2).unwrap()).collect();
    assert_eq!(k1, vec![1, 2, 1, 2], "partition by the quoted column");
    assert_eq!(
        k2,
        vec![1, 1, 1, 1],
        "each (a, b) pair is its own partition"
    );
}
