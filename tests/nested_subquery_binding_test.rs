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

//! A subquery two levels down that reads only the outermost row is bound
//! to that row and folded once, not run again for every row of the query
//! that holds it; and the plain conjuncts beside a correlated subquery
//! still narrow the outer scan.

use std::time::{Duration, Instant};
use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .next()
        .unwrap()
}

#[test]
fn test_grandparent_only_subquery_folds_once() {
    let db = Database::open("memory://nested_subquery_binding").unwrap();
    for t in ["x", "y"] {
        db.execute(
            &format!("CREATE TABLE {t} (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, s TEXT)"),
            (),
        )
        .unwrap();
        let rows: Vec<String> = (1..=2500)
            .map(|i| format!("({i}, {}, {}, 's{}')", i % 40, (i * 3) % 25, i % 30))
            .collect();
        for chunk in rows.chunks(500) {
            db.execute(&format!("INSERT INTO {t} VALUES {}", chunk.join(", ")), ())
                .unwrap();
        }
    }

    let started = Instant::now();
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM x WHERE id <= 20 AND b > \
             (SELECT AVG(b) FROM y WHERE y.a = (SELECT MIN(a) FROM y WHERE y.s = x.s))"
        ),
        10
    );
    // Twenty outer rows, one inner fold each: well under a second; the
    // per-row shape ran the inner subquery fifty thousand times
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "took {:?}",
        started.elapsed()
    );

    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM x WHERE id <= 20 AND EXISTS (SELECT 1 FROM y WHERE y.a = x.a \
             AND EXISTS (SELECT 1 FROM (SELECT b FROM y z WHERE z.a = x.a) w WHERE w.b > y.b))"
        ),
        20
    );
}
