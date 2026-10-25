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

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .next()
        .unwrap()
}

/// Row `i` of x and of y: (a, b, s)
fn row(i: i64) -> (i64, i64, String) {
    (i % 40, (i * 3) % 25, format!("s{}", i % 30))
}

fn tables(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for t in ["x", "y"] {
        db.execute(
            &format!("CREATE TABLE {t} (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, s TEXT)"),
            (),
        )
        .unwrap();
        let rows: Vec<String> = (1..=2500)
            .map(|i| {
                let (a, b, s) = row(i);
                format!("({i}, {a}, {b}, '{s}')")
            })
            .collect();
        for chunk in rows.chunks(500) {
            db.execute(&format!("INSERT INTO {t} VALUES {}", chunk.join(", ")), ())
                .unwrap();
        }
    }
    db
}

/// The outer rows `id <= 20` whose count of y rows passing `passes` is above `above`
fn expected(above: usize, passes: impl Fn(i64, i64) -> bool) -> i64 {
    (1..=20)
        .filter(|&i| (1..=2500).filter(|&j| passes(i, j)).count() > above)
        .count() as i64
}

#[test]
fn test_grandparent_only_subquery_folds_once() {
    let db = tables("memory://nested_subquery_binding");

    #[cfg(feature = "test-failpoints")]
    let runs = stoolap::test_failpoints::prepared_scalar_runs();
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM x WHERE id <= 20 AND b > \
             (SELECT AVG(b) FROM y WHERE y.a = (SELECT MIN(a) FROM y WHERE y.s = x.s))"
        ),
        10
    );
    #[cfg(feature = "test-failpoints")]
    assert_eq!(
        stoolap::test_failpoints::prepared_scalar_runs() - runs,
        40,
        "one AVG and one MIN run per outer row, not one MIN per row of y"
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

#[test]
fn test_grandparent_only_exists_probes_once_per_outer_row() {
    let db = tables("memory://nested_subquery_binding_exists");
    // A z row of 's7' whose a is above x.a + 20
    let found: Vec<bool> = (0..=20)
        .map(|i| (1..=2500).any(|k| row(k).2 == "s7" && row(k).0 > row(i).0 + 20))
        .collect();
    #[cfg(feature = "test-failpoints")]
    let probes = stoolap::test_failpoints::prepared_exists_counts().0;
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM x WHERE id <= 20 AND (SELECT COUNT(*) FROM y WHERE y.b > x.b \
             AND EXISTS (SELECT 1 FROM y z WHERE z.s = 's7' AND z.a > x.a + 20)) > 1000"
        ),
        expected(1000, |i, j| found[i as usize] && row(j).1 > row(i).1)
    );
    #[cfg(feature = "test-failpoints")]
    assert!(
        stoolap::test_failpoints::prepared_exists_counts().0 - probes <= 20,
        "at most one EXISTS probe per outer row, not one per row of y"
    );
}

#[test]
fn test_grandparent_only_in_reads_members_once_per_outer_row() {
    let db = tables("memory://nested_subquery_binding_in");
    let members: Vec<Vec<i64>> = (0..=20)
        .map(|i| {
            (1..=2500)
                .filter(|&k| row(k).2 == "s7" && row(k).0 > row(i).0 + 20)
                .map(|k| row(k).1)
                .collect()
        })
        .collect();
    #[cfg(feature = "test-failpoints")]
    let runs = stoolap::test_failpoints::in_subquery_runs();
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM x WHERE id <= 20 AND (SELECT COUNT(*) FROM y WHERE y.b IN \
             (SELECT z.b FROM y z WHERE z.s = 's7' AND z.a > x.a + 20)) > 100"
        ),
        expected(100, |i, j| members[i as usize].contains(&row(j).1))
    );
    #[cfg(feature = "test-failpoints")]
    assert!(
        stoolap::test_failpoints::in_subquery_runs() - runs <= 20,
        "at most one member read per outer row, not one per row of y"
    );
}
