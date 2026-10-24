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

use stoolap::test_failpoints::subquery_cache_hits;
use stoolap::Database;

#[test]
fn the_same_block_written_twice_is_kept_once_for_both() {
    let db = Database::open("memory://block_cache_identity").unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v FLOAT)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 10.0), (2, 20.0), (3, 30.0)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE q (id INTEGER PRIMARY KEY, b INTEGER, \"a.b\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO q VALUES (1, 10, 20)", ()).unwrap();
    db.execute(
        "CREATE TABLE \"q AS a\" (id INTEGER PRIMARY KEY, b INTEGER, \"a.b\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO \"q AS a\" VALUES (1, 30, 40)", ())
        .unwrap();
    // Blocks a key that ignores quoting or qualification would take as one
    for (sql, want) in [
        (
            "SELECT (SELECT \"a.b\" FROM q AS a), (SELECT a.b FROM q AS a) FROM q",
            ["20", "10"],
        ),
        (
            "SELECT (SELECT a.b FROM q AS a), (SELECT \"a.b\" FROM q AS a) FROM q",
            ["10", "20"],
        ),
        (
            "SELECT (SELECT \"UPPER\" FROM q), (SELECT \"upper\" FROM q) FROM q",
            ["UPPER", "upper"],
        ),
        (
            "SELECT (SELECT b FROM \"q AS a\"), (SELECT b FROM q AS a) FROM q",
            ["30", "10"],
        ),
        (
            "SELECT (SELECT b FROM q AS a), (SELECT b FROM \"q AS a\") FROM q",
            ["10", "30"],
        ),
    ] {
        let statement = db.prepare(sql).unwrap();
        for _ in 0..2 {
            let row = statement.query(()).unwrap().next().unwrap().unwrap();
            let found = [0, 1].map(|i| row.get::<String>(i).unwrap());
            assert_eq!(found, want, "{sql}");
        }
    }
    for sql in [
        "SELECT id, (SELECT AVG(v) FROM u) FROM u WHERE v > (SELECT AVG(v) FROM u)",
        "SELECT id FROM u WHERE v > (SELECT AVG(v) FROM u) AND v < (SELECT AVG(v) FROM u) + 20",
        "SELECT id FROM u WHERE id IN (SELECT id FROM u WHERE v > 15) \
         AND id + 0 IN (SELECT id FROM u WHERE v > 15)",
        "SELECT id FROM u WHERE EXISTS (SELECT 1 FROM u WHERE v > 25) \
         AND NOT NOT EXISTS (SELECT 1 FROM u WHERE v > 25)",
    ] {
        let statement = db.prepare(sql).unwrap();
        let run = || {
            statement
                .query(())
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect::<Vec<_>>()
        };
        let first = run();
        let hits = subquery_cache_hits();
        assert_eq!(run(), first, "{sql}");
        assert_eq!(
            subquery_cache_hits() - hits,
            2,
            "{sql}: each of the two blocks reads the answer kept for both"
        );
    }
}
