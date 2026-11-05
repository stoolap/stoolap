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

//! A join with no condition under a small LIMIT pairs every row: it
//! streams, and its build side reads no more rows than the limit

use std::collections::BTreeSet;

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE x (id INTEGER PRIMARY KEY, v INTEGER)",
        "CREATE TABLE y (id INTEGER PRIMARY KEY, w INTEGER)",
        "INSERT INTO x VALUES (1, 1), (2, 2), (3, 3)",
        "INSERT INTO y VALUES (1, 10), (2, 20)",
        "CREATE TABLE big (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE bigger (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO big SELECT g.value, g.value FROM generate_series(1, 1000) g",
        "INSERT INTO bigger SELECT g.value, g.value FROM generate_series(1, 3000) g",
        "ANALYZE big",
        "ANALYZE bigger",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The (left id, right id) pairs `sql` answers, each once
fn pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect()
}

fn distinct(found: &[(i64, i64)]) -> BTreeSet<(i64, i64)> {
    let set: BTreeSet<_> = found.iter().copied().collect();
    assert_eq!(set.len(), found.len(), "a pair answered twice: {found:?}");
    set
}

fn every_pair(left: &[i64], right: &[i64]) -> BTreeSet<(i64, i64)> {
    left.iter()
        .flat_map(|l| right.iter().map(move |r| (*l, *r)))
        .collect()
}

#[test]
fn a_join_with_no_condition_under_a_limit_answers_pairs_of_its_sides() {
    let db = open("cross_join_limit_answers");
    let all = every_pair(&[1, 2, 3], &[1, 2]);
    for from in ["x CROSS JOIN y", "x, y"] {
        let sql = format!("SELECT x.id, y.id FROM {from} LIMIT 100");
        assert_eq!(distinct(&pairs(&db, &sql)), all, "{sql}");
        let sql = format!("SELECT x.id, y.id FROM {from} LIMIT 4");
        let found = distinct(&pairs(&db, &sql));
        assert!(
            found.len() == 4 && found.is_subset(&all),
            "{sql}: {found:?}"
        );
        let sql = format!("SELECT x.id, y.id FROM {from} LIMIT 3 OFFSET 2");
        let found = distinct(&pairs(&db, &sql));
        assert!(
            found.len() == 3 && found.is_subset(&all),
            "{sql}: {found:?}"
        );
        let sql = format!("SELECT x.id, y.id FROM {from} LIMIT 10 OFFSET 4");
        let found = distinct(&pairs(&db, &sql));
        assert!(
            found.len() == 2 && found.is_subset(&all),
            "{sql}: {found:?}"
        );
        let sql = format!("SELECT x.id, y.id FROM {from} WHERE x.v > 1 AND y.w < 20 LIMIT 100");
        assert_eq!(
            distinct(&pairs(&db, &sql)),
            every_pair(&[2, 3], &[1]),
            "{sql}"
        );
    }
    let sql = "SELECT y.id, x.id FROM y CROSS JOIN x LIMIT 100";
    assert_eq!(
        distinct(&pairs(&db, sql)),
        every_pair(&[1, 2], &[1, 2, 3]),
        "{sql}"
    );
    let found = pairs(
        &db,
        "SELECT x.id, y.id FROM x CROSS JOIN y WHERE y.w > 50 LIMIT 5",
    );
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn a_join_with_no_condition_over_large_sides_answers_distinct_pairs_up_to_the_limit() {
    let db = open("cross_join_limit_large");
    let sides = [("big", "big.id", 1000), ("bigger", "bigger.id", 3000)];
    for [(l, lid, ln), (r, rid, rn)] in [[sides[0], sides[1]], [sides[1], sides[0]]] {
        for (limit, offset, filter) in [
            (100, 0, ""),
            (7, 30, ""),
            (50, 0, " WHERE big.a % 2 = 0 AND bigger.b > 2900"),
        ] {
            let sql = format!(
                "SELECT {lid}, {rid} FROM {l} CROSS JOIN {r}{filter} LIMIT {limit} OFFSET {offset}"
            );
            let found = distinct(&pairs(&db, &sql));
            assert_eq!(found.len(), limit, "{sql}");
            for (a, b) in &found {
                assert!(
                    (1..=ln).contains(a) && (1..=rn).contains(b),
                    "{sql}: {a}, {b}"
                );
                if !filter.is_empty() {
                    let (big, bigger) = if l == "big" { (a, b) } else { (b, a) };
                    assert!(big % 2 == 0 && *bigger > 2900, "{sql}: {a}, {b}");
                }
            }
        }
    }
    let stmt = db
        .prepare("SELECT big.id, bigger.id FROM big CROSS JOIN bigger LIMIT $1 OFFSET $2")
        .unwrap();
    for (limit, offset) in [(1, 0), (100, 0), (20, 999)] {
        let found: Vec<(i64, i64)> = stmt
            .query((limit, offset))
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
            })
            .collect();
        assert_eq!(distinct(&found).len(), limit as usize, "{limit} {offset}");
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_join_with_no_condition_under_a_limit_streams_and_builds_on_no_more_than_the_limit() {
    use stoolap::test_failpoints::{limit_hash_build_rows, streaming_joins_streamed};
    let db = open("cross_join_limit_streams");
    for (sql, cap) in [
        (
            "SELECT big.id, bigger.id FROM big CROSS JOIN bigger LIMIT 100",
            100,
        ),
        (
            "SELECT bigger.id, big.id FROM bigger CROSS JOIN big LIMIT 100",
            100,
        ),
        (
            "SELECT big.id, bigger.id FROM big, bigger LIMIT 10 OFFSET 5",
            15,
        ),
        (
            "SELECT big.id, bigger.id FROM big CROSS JOIN bigger WHERE big.a > 500 LIMIT 20",
            20,
        ),
    ] {
        let (streamed, built) = (streaming_joins_streamed(), limit_hash_build_rows());
        db.query(sql, ()).unwrap().for_each(|row| {
            row.unwrap();
        });
        assert_eq!(streaming_joins_streamed() - streamed, 1, "{sql} streams");
        let built = limit_hash_build_rows() - built;
        assert!(built <= cap, "{sql} built on {built} rows");
    }
}
