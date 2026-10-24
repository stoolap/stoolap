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

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

const COUNTS: &str =
    "SELECT u.id, (SELECT COUNT(*) FROM o WHERE o.k = u.r) FROM u ORDER BY u.id LIMIT 100";

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, r INTEGER, t TEXT)",
        "INSERT INTO u VALUES (1, 1, '1'), (2, 2, '2'), (3, 3, 'x'), (4, NULL, NULL)",
        "CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "CREATE INDEX o_k ON o(k)",
        "INSERT INTO o VALUES (1, 2, 10), (2, 2, 20), (3, NULL, 30), (4, NULL, 40)",
        "INSERT INTO o SELECT 100 + g.value, 1, g.value FROM generate_series(1, 50) g",
        // Keys no outer row asks for, so a probe per outer row reads less
        "INSERT INTO o SELECT 1000 + g.value, 10 + g.value % 50, 0 FROM generate_series(1, 2000) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Each row's (first, second) column, and the aggregate paths that answered
fn pairs(db: &Database, sql: &str) -> (Vec<(i64, Option<i64>)>, Vec<&'static str>) {
    reset_aggregate_paths();
    let found = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<i64>(0).unwrap(),
                row.get::<Option<i64>>(1).unwrap(),
            )
        })
        .collect();
    (found, aggregate_paths_taken())
}

#[test]
fn a_correlated_count_of_an_indexed_key_is_counted_from_the_index() {
    let db = setup("index_count_counts");
    let (found, taken) = pairs(&db, COUNTS);
    // A NULL key equals no row, the rows holding NULL included
    assert_eq!(
        found,
        [(1, Some(50)), (2, Some(2)), (3, Some(0)), (4, Some(0))]
    );
    assert_eq!(taken, ["index_count"]);
    let having = "SELECT u.id, (SELECT COUNT(*) * 2 FROM o WHERE o.k = u.r HAVING COUNT(*) > 1) \
                  FROM u ORDER BY u.id LIMIT 100";
    let (found, _) = pairs(&db, having);
    assert_eq!(found, [(1, Some(100)), (2, Some(4)), (3, None), (4, None)]);
}

#[test]
fn an_index_count_sees_what_the_statement_sees() {
    let db = setup("index_count_view");
    db.execute("DELETE FROM o WHERE id = 1", ()).unwrap();
    db.execute("UPDATE o SET k = 3 WHERE id = 101", ()).unwrap();
    let (found, _) = pairs(&db, COUNTS);
    assert_eq!(
        found,
        [(1, Some(49)), (2, Some(1)), (3, Some(1)), (4, Some(0))]
    );
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO o VALUES (500, 2, 0), (501, 2, 0)", ())
        .unwrap();
    db.execute("DELETE FROM o WHERE id = 102", ()).unwrap();
    let (found, _) = pairs(&db, COUNTS);
    assert_eq!(
        found,
        [(1, Some(48)), (2, Some(3)), (3, Some(1)), (4, Some(0))],
        "the transaction's own writes"
    );
    db.execute("ROLLBACK", ()).unwrap();
    let (found, _) = pairs(&db, COUNTS);
    assert_eq!(
        found,
        [(1, Some(49)), (2, Some(1)), (3, Some(1)), (4, Some(0))]
    );
}

#[test]
fn a_count_the_index_cannot_answer_counts_its_rows() {
    let db = setup("index_count_declined");
    // A TEXT key against an INTEGER column, as the general path reads it
    let crossing =
        "SELECT u.id, (SELECT COUNT(*) FROM o WHERE o.k = u.t) FROM u ORDER BY u.id LIMIT 100";
    let general =
        "SELECT DISTINCT u.id, (SELECT COUNT(*) FROM o WHERE o.k = u.t AND o.id + 0 = o.id) \
                   FROM u ORDER BY u.id LIMIT 100";
    assert_eq!(pairs(&db, crossing).0, pairs(&db, general).0);
    for (block, want) in [
        ("SELECT COUNT(v) FROM o WHERE o.k = u.r", [50, 2, 0, 0]),
        (
            "SELECT COUNT(DISTINCT v) FROM o WHERE o.k = u.r",
            [50, 2, 0, 0],
        ),
        (
            "SELECT COUNT(*) FILTER (WHERE v > 10) FROM o WHERE o.k = u.r",
            [40, 1, 0, 0],
        ),
        (
            "SELECT COUNT(*) FROM o WHERE o.k = u.r AND v > 10",
            [40, 1, 0, 0],
        ),
        (
            "SELECT COUNT(*) FROM o WHERE o.k >= u.r",
            [2052, 2002, 2000, 0],
        ),
    ] {
        let sql = format!("SELECT u.id, ({block}) FROM u ORDER BY u.id LIMIT 100");
        let (found, taken) = pairs(&db, &sql);
        let want: Vec<_> = (1..=4).zip(want.map(Some)).collect();
        assert_eq!(found, want, "{sql}");
        assert!(!taken.contains(&"index_count"), "{sql} took {taken:?}");
    }
}
