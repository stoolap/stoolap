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

//! Every registered prepared strategy answers a corpus of EXISTS, IN, ANY,
//! ALL and aggregating scalar statements as its written answer says, and the
//! corpus reaches every one of them: a strategy added without a case fails here.

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{force_exists_strategy, prepared_strategy_runs, PreparedStrategy};
use stoolap::Database;

/// The forced strategy is process-wide: tests that force one take turns
static FORCED: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn setup() -> Database {
    let db = Database::open("memory://prepared_strategy_corpus").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER, s TEXT)",
        "INSERT INTO t VALUES (1, 1, 1, '1'), (2, 2, 2, '2'), (3, NULL, NULL, NULL), (4, 4, 4, '4'), (5, 5, 5, '5')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, d INTEGER)",
        "INSERT INTO u VALUES (1, 1, 1, 2), (2, 1, 10, 1), (3, 2, NULL, 1), (4, NULL, 4, 1), (5, 4, 4, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            row.unwrap_or_else(|e| panic!("{sql}: {e}"))
                .get::<i64>(0)
                .unwrap()
        })
        .collect()
}

/// Each statement with the ids its written answer gives
const CORPUS: &[(&str, &[i64])] = &[
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) ORDER BY id", &[1, 2, 4]),
    ("SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k) ORDER BY id", &[3, 5]),
    ("SELECT id FROM t WHERE x > 100 OR EXISTS (SELECT 1 FROM u WHERE u.k = t.k) ORDER BY id", &[1, 2, 4]),
    ("SELECT id FROM t WHERE x > 1 AND EXISTS (SELECT 1 FROM u WHERE u.v > 5) ORDER BY id", &[2, 4, 5]),
    ("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id AND u.d = 1) ORDER BY id", &[2, 3, 4, 5]),
    ("SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.id = t.id AND u.d = 1) ORDER BY id", &[1]),
    ("SELECT id FROM t WHERE x IN (SELECT v FROM u) ORDER BY id", &[1, 4]),
    ("SELECT id FROM t WHERE id IN (SELECT v FROM u WHERE v IS NOT NULL) ORDER BY id", &[1, 4]),
    ("SELECT id FROM t WHERE id NOT IN (SELECT v FROM u WHERE v IS NOT NULL) ORDER BY id", &[2, 3, 5]),
    ("SELECT id FROM t WHERE x IN (SELECT v FROM u WHERE u.k = t.k) ORDER BY id", &[1, 4]),
    ("SELECT id FROM t WHERE x NOT IN (SELECT v FROM u WHERE u.k = t.k) ORDER BY id", &[3, 5]),
    ("SELECT id FROM t WHERE x < ANY (SELECT v FROM u WHERE u.k = t.k) ORDER BY id", &[1]),
    ("SELECT id FROM t WHERE x >= ALL (SELECT v FROM u WHERE v IS NOT NULL) ORDER BY id", &[]),
    ("SELECT id FROM t WHERE x < (SELECT MAX(v) FROM u WHERE u.k = t.k) ORDER BY id", &[1]),
    ("SELECT id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.k = t.k) = 0 ORDER BY id", &[3, 5]),
    ("SELECT id FROM t WHERE EXISTS (SELECT u.k FROM u WHERE u.k = t.k GROUP BY u.k HAVING COUNT(*) > 1) ORDER BY id", &[1]),
    ("SELECT id FROM t WHERE x IN (SELECT MAX(v) FROM u WHERE u.k = t.k) ORDER BY id", &[4]),
    ("SELECT id FROM t WHERE x IN (SELECT MIN(v) FROM u GROUP BY k) ORDER BY id", &[1, 4]),
    ("SELECT id FROM t WHERE x <= ALL (SELECT COUNT(*) FROM u WHERE u.k = t.k GROUP BY u.d) ORDER BY id", &[1, 3, 5]),
    ("SELECT id FROM t WHERE x = (SELECT MIN(v) FROM u WHERE u.k = t.k HAVING MIN(v) < t.x + 1) ORDER BY id", &[1, 4]),
    ("SELECT id FROM t WHERE x > (SELECT AVG(v) FROM u) ORDER BY id", &[5]),
    ("SELECT id FROM (SELECT id, x FROM t) AS t WHERE x IN (SELECT v FROM u) ORDER BY id", &[1, 4]),
];

#[test]
fn test_every_registered_strategy_answers_the_corpus_and_is_reached() {
    let _turn = FORCED.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();
    let before: Vec<usize> = PreparedStrategy::ALL
        .iter()
        .map(|s| prepared_strategy_runs(*s))
        .collect();
    for strategy in [None, Some(false), Some(true)] {
        force_exists_strategy(strategy);
        for (sql, expected) in CORPUS {
            assert_eq!(ids(&db, sql), *expected, "{sql} under {strategy:?}");
        }
    }
    force_exists_strategy(None);
    for (strategy, before) in PreparedStrategy::ALL.iter().zip(before) {
        assert!(
            prepared_strategy_runs(*strategy) > before,
            "no case of the corpus reaches {strategy:?}"
        );
    }
}

#[test]
fn test_a_group_error_belongs_to_its_key_under_every_strategy() {
    let _turn = FORCED.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://prepared_strategy_group_error").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO t VALUES (1, 1)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO u VALUES (1, 1, 1), (2, 2, 9223372036854775807), (3, 2, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT id FROM t WHERE 2 = (SELECT SUM(v) + 1 FROM u WHERE u.k = t.k)";
    for strategy in [Some(false), Some(true)] {
        force_exists_strategy(strategy);
        let got = db
            .query(sql, ())
            .map_err(|e| e.to_string())
            .and_then(|rows| {
                rows.map(|row| {
                    row.map_err(|e| e.to_string())?
                        .get::<i64>(0)
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<Vec<_>, _>>()
            });
        assert_eq!(got, Ok(vec![1]), "{sql} under {strategy:?}");
    }
    force_exists_strategy(None);
}
