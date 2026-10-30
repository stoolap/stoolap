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

//! IN, NOT IN, ANY and ALL over a subquery in the prepared family. The
//! members are the SELECT expression of the block's candidates; a member
//! that settles the test wins, else the first error a candidate raised,
//! else the three-valued answer. Every strategy gives that answer.

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{force_exists_strategy, general_path_hits, GeneralPath};
use stoolap::Database;

static SERIAL: Mutex<()> = Mutex::new(());

/// Holds the lock and restores the rule's choice when dropped
struct Serial(#[allow(dead_code)] MutexGuard<'static, ()>);

impl Drop for Serial {
    fn drop(&mut self) {
        force_exists_strategy(None);
    }
}

fn serial() -> Serial {
    Serial(SERIAL.lock().unwrap_or_else(|e| e.into_inner()))
}

/// `t` and its copy `w` are the tested rows; `u` the members. Key 1 holds
/// 1 and 10, the first of them on a row whose `d * MAX` overflows; key 2
/// holds NULL; a NULL key holds 4; key 4 holds 4; key 5 holds nothing
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://prepared_in_{name}")).unwrap();
    for table in ["t", "w"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER, s TEXT)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} VALUES (1, 1, 1, '1'), (2, 2, 2, '2'), \
                 (3, NULL, NULL, NULL), (4, 4, 4, '4'), (5, 5, 5, '5')"
            ),
            (),
        )
        .unwrap();
    }
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, d INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO u VALUES (1, 1, 1, 2), (2, 1, 10, 1), (3, 2, NULL, 1), \
         (4, NULL, 4, 1), (5, 4, 4, 1)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE e (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    db.query(sql, ())
        .and_then(|rows| {
            rows.map(|row| row.map(|row| row.get::<i64>(0).unwrap()))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|e| e.to_string())
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

/// `SELECT id FROM t WHERE {test}` at the statement, and the same test
/// inside a scalar block over `w`
fn at_each_depth(test: &str) -> [String; 2] {
    [
        format!("SELECT id FROM t WHERE {test} ORDER BY id"),
        format!(
            "SELECT id FROM t WHERE 1 = (SELECT 1 FROM w WHERE w.id = t.id AND {}) ORDER BY id",
            test.replace("t.", "w.")
        ),
    ]
}

const STRATEGIES: [Option<bool>; 3] = [None, Some(false), Some(true)];

const CASES: &[(&str, &[i64])] = &[
    // Uncorrelated: members 1, 10, NULL, 4, 4
    ("t.x IN (SELECT v FROM u)", &[1, 4]),
    ("t.x NOT IN (SELECT v FROM u)", &[]),
    ("t.x IN (SELECT v FROM u WHERE v IS NOT NULL)", &[1, 4]),
    ("t.x NOT IN (SELECT v FROM u WHERE v IS NOT NULL)", &[2, 5]),
    ("t.x IN (SELECT v FROM e)", &[]),
    ("t.x NOT IN (SELECT v FROM e)", &[1, 2, 3, 4, 5]),
    ("t.x = ANY (SELECT v FROM u)", &[1, 4]),
    ("t.x <> ALL (SELECT v FROM u WHERE v IS NOT NULL)", &[2, 5]),
    (
        "t.x < ANY (SELECT v FROM u WHERE v IS NOT NULL)",
        &[1, 2, 4, 5],
    ),
    ("t.x > ALL (SELECT v FROM u WHERE v IS NOT NULL)", &[]),
    ("t.x >= ALL (SELECT v FROM e)", &[1, 2, 3, 4, 5]),
    ("t.x < ANY (SELECT v FROM e)", &[]),
    // Correlated: key 1 holds 1 and 10, key 2 NULL, key 4 holds 4
    ("t.x IN (SELECT v FROM u WHERE u.k = t.k)", &[1, 4]),
    ("t.x NOT IN (SELECT v FROM u WHERE u.k = t.k)", &[3, 5]),
    ("t.x < ANY (SELECT v FROM u WHERE u.k = t.k)", &[1]),
    ("t.x >= ALL (SELECT v FROM u WHERE u.k = t.k)", &[3, 4, 5]),
    ("t.x IN (SELECT v + 0 FROM u WHERE u.k = t.k)", &[1, 4]),
    ("t.x IN (SELECT v FROM u WHERE t.k = u.k AND u.v > 1)", &[4]),
    // A TEXT column against number members, and a number against TEXT ones
    ("t.s IN (SELECT v FROM u)", &[1, 4]),
    ("t.s IN (SELECT v FROM u WHERE v IS NOT NULL)", &[1, 4]),
    ("t.s NOT IN (SELECT v FROM u WHERE v IS NOT NULL)", &[2, 5]),
    ("t.x IN (SELECT s FROM w)", &[1, 2, 4, 5]),
    (
        "t.x NOT IN (SELECT s FROM w WHERE s IS NOT NULL AND s <> '2')",
        &[2],
    ),
    ("t.s = ANY (SELECT v FROM u WHERE u.k = t.k)", &[1, 4]),
];

#[test]
fn test_every_strategy_answers_the_contract() {
    let _guard = serial();
    let db = setup("contract");
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        for (test, expected) in CASES {
            for sql in at_each_depth(test) {
                assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
                assert_eq!(
                    ids(&db, &sql),
                    Ok(expected.to_vec()),
                    "{sql} under {strategy:?}"
                );
            }
        }
    }
}

#[test]
fn test_a_member_that_settles_the_test_wins_over_an_error() {
    let _guard = serial();
    let db = setup("errors");
    let raises = "u.d * 9223372036854775807 > 0";
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        for (sql, expected) in [
            // Key 1: 1 raises, 10 is a member that settles nothing for x = 1
            (format!("SELECT id FROM t WHERE t.id = 1 AND x IN (SELECT v FROM u WHERE u.k = t.k AND {raises})"), None),
            // 10 settles `x >= ALL` as FALSE
            (format!("SELECT id FROM t WHERE t.id = 1 AND x >= ALL (SELECT v FROM u WHERE u.k = t.k AND {raises})"), Some(vec![])),
            (format!("SELECT id FROM t WHERE t.id = 1 AND x <= ALL (SELECT v FROM u WHERE u.k = t.k AND {raises})"), None),
            // 10 settles `x < ANY` as TRUE
            (format!("SELECT id FROM t WHERE t.id = 1 AND x < ANY (SELECT v FROM u WHERE u.k = t.k AND {raises})"), Some(vec![1])),
            // Keys 2 and 4 hold no failing row
            (format!("SELECT id FROM t WHERE t.id <> 1 AND x IN (SELECT v FROM u WHERE u.k = t.k AND {raises}) ORDER BY id"), Some(vec![4])),
            // Uncorrelated: 10 is a member for no x, so the error stands
            (format!("SELECT id FROM t WHERE x IN (SELECT v FROM u WHERE {raises})"), None),
            // 10 settles it for every x but NULL, for which the error stands
            (format!("SELECT id FROM t WHERE x IS NOT NULL AND x < ANY (SELECT v FROM u WHERE {raises}) ORDER BY id"), Some(vec![1, 2, 4, 5])),
            (format!("SELECT id FROM t WHERE x < ANY (SELECT v FROM u WHERE {raises})"), None),
        ] {
            let found = ids(&db, &sql);
            match expected {
                Some(expected) => assert_eq!(found, Ok(expected), "{sql} under {strategy:?}"),
                None => assert!(found.is_err(), "{sql} under {strategy:?}: {found:?}"),
            }
        }
    }
}

#[test]
fn test_an_in_family_statement_never_reaches_the_general_path() {
    let _guard = serial();
    let db = setup("general");
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        for (test, _) in CASES {
            for sql in at_each_depth(test) {
                let before = general_path_hits(GeneralPath::InSubquery);
                ids(&db, &sql).unwrap();
                assert_eq!(general_path_hits(GeneralPath::InSubquery), before, "{sql}");
            }
        }
    }
}

#[test]
fn test_a_set_test_belongs_to_the_family_wherever_where_holds_it() {
    let _guard = serial();
    let db = setup("family");
    for sql in [
        "SELECT id FROM t WHERE x IN (SELECT v FROM u)",
        "SELECT id FROM t WHERE x > 1 OR x NOT IN (SELECT v FROM u WHERE u.k = t.k)",
        "SELECT id FROM t WHERE CASE WHEN x > 1 THEN x = ANY (SELECT v FROM u) ELSE FALSE END",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.v IN (SELECT x FROM w WHERE w.k = u.k))",
        "SELECT id FROM t WHERE x >= ALL (SELECT v + 1 FROM u WHERE u.k = t.k) LIMIT 2",
        "SELECT id FROM t WHERE x IN (SELECT MAX(v) FROM u)",
        "SELECT id, x IN (SELECT v FROM u) FROM t WHERE x IN (SELECT v FROM u)",
        "SELECT id FROM t WHERE (x, k) IN (SELECT v, k FROM u)",
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    }
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE x IN (SELECT v, k FROM u)",
            "a subquery selects 2 values where its place takes 1",
        ),
        (
            "SELECT id FROM t WHERE (x, k) IN (SELECT v FROM u)",
            "a subquery selects 1 values where its place takes 2",
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let err = db
            .query(sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .unwrap_err()
            .to_string();
        assert!(err.contains(want), "{sql}: {err}");
    }
}

#[test]
fn test_a_set_reads_the_statement_view() {
    use std::cell::Cell;
    use std::rc::Rc;
    let _guard = serial();
    for level in ["READ COMMITTED", "SNAPSHOT"] {
        for strategy in STRATEGIES {
            force_exists_strategy(strategy);
            for (query, before, after) in [
                (
                    "SELECT id FROM t WHERE x IN (SELECT v FROM u WHERE v IS NOT NULL) ORDER BY id",
                    vec![1, 4],
                    vec![1, 2, 4],
                ),
                (
                    "SELECT id FROM t WHERE x IN (SELECT v FROM u WHERE u.k = t.k) ORDER BY id",
                    vec![1, 4],
                    vec![1, 2, 4],
                ),
            ] {
                let db = setup(&format!(
                    "view_{}_{strategy:?}_{}",
                    level.len(),
                    query.len()
                ));
                db.execute(&format!("SET isolation_level = '{level}'"), ())
                    .unwrap();
                let writer = db.clone();
                let done = Rc::new(Cell::new(false));
                let flag = Rc::clone(&done);
                stoolap::test_failpoints::before_prepared_exists(move || {
                    writer
                        .execute("INSERT INTO u VALUES (9, 2, 2, 1)", ())
                        .unwrap();
                    flag.set(true);
                });
                assert_eq!(ids(&db, query), Ok(before), "{level} {strategy:?}: {query}");
                assert!(
                    done.get(),
                    "{level} {strategy:?}: the commit ran inside {query}"
                );
                assert_eq!(
                    ids(&db, query),
                    Ok(after),
                    "{level} {strategy:?}: the next {query}"
                );
            }
        }
    }
}

#[test]
fn test_members_are_kept_while_their_stamp_holds_and_by_parameters() {
    use stoolap::test_failpoints::subquery_cache_hits;
    let _guard = serial();
    let db = setup("kept");
    let sql = "SELECT id FROM t WHERE x IN (SELECT v FROM u WHERE v IS NOT NULL) ORDER BY id";
    let hits = subquery_cache_hits();
    assert_eq!(ids(&db, sql), Ok(vec![1, 4]));
    assert_eq!(subquery_cache_hits(), hits, "{sql} kept nothing yet");
    assert_eq!(ids(&db, sql), Ok(vec![1, 4]));
    assert_eq!(subquery_cache_hits(), hits + 1, "{sql} kept its members");
    db.execute("INSERT INTO u VALUES (9, 2, 2, 1)", ()).unwrap();
    assert_eq!(ids(&db, sql), Ok(vec![1, 2, 4]), "{sql} after a write");
    let param = "SELECT id FROM t WHERE x IN (SELECT v FROM u WHERE v > $1) ORDER BY id";
    for (value, expected) in [(3, vec![4]), (0, vec![1, 2, 4]), (3, vec![4])] {
        let found: Vec<i64> = db
            .query(param, (value,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(found, expected, "{param} with {value}");
    }
}

#[test]
fn test_a_top_level_set_test_reads_only_the_rows_its_members_name() {
    use stoolap::test_failpoints::prepared_rows_narrowed;
    let _guard = serial();
    let db = setup("narrowing");
    db.execute(
        "CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    let values: Vec<String> = (1..=300)
        .map(|i| format!("({i}, {}, '{}')", i % 30, i % 30))
        .collect();
    db.execute(&format!("INSERT INTO big VALUES {}", values.join(", ")), ())
        .unwrap();
    db.execute("CREATE INDEX big_k ON big (k)", ()).unwrap();
    let of = |keys: &[i64]| {
        (1..=300)
            .filter(|i| keys.contains(&(i % 30)))
            .collect::<Vec<i64>>()
    };
    for (sql, expected, narrowed) in [
        (
            "SELECT id FROM big WHERE k IN (SELECT v FROM u) ORDER BY id",
            of(&[1, 4, 10]),
            true,
        ),
        (
            "SELECT id FROM big WHERE id IN (SELECT v FROM u) ORDER BY id",
            vec![1, 4, 10],
            true,
        ),
        (
            "SELECT id FROM big WHERE id > 5 AND id IN (SELECT v FROM u) ORDER BY id",
            vec![10],
            true,
        ),
        (
            "SELECT id FROM big WHERE id NOT IN (SELECT v FROM u WHERE v IS NOT NULL) ORDER BY id",
            (1..=300).filter(|i| ![1, 4, 10].contains(i)).collect(),
            true,
        ),
        // A NULL member leaves NOT IN TRUE for no row
        (
            "SELECT id FROM big WHERE id NOT IN (SELECT v FROM u) ORDER BY id",
            vec![],
            true,
        ),
        // A TEXT column is not looked up for number members; the scan answers
        (
            "SELECT id FROM big WHERE s IN (SELECT v FROM u) ORDER BY id",
            of(&[1, 4, 10]),
            false,
        ),
        // NOT IN on a column that is not the primary key scans
        (
            "SELECT id FROM big WHERE k NOT IN (SELECT v FROM u WHERE v IS NOT NULL) ORDER BY id",
            (1..=300)
                .filter(|i| ![1, 4, 10].contains(&(i % 30)))
                .collect(),
            false,
        ),
    ] {
        let before = prepared_rows_narrowed();
        assert_eq!(ids(&db, sql), Ok(expected), "{sql}");
        assert_eq!(
            prepared_rows_narrowed() > before,
            narrowed,
            "{sql} read big by its members"
        );
    }
    let limited = "SELECT id FROM big WHERE k IN (SELECT v FROM u) LIMIT 3";
    let found = ids(&db, limited).unwrap();
    assert_eq!(found.len(), 3, "{limited}");
    assert!(
        found.iter().all(|id| [1, 4, 10].contains(&(id % 30))),
        "{limited}: {found:?}"
    );
    // A member's error leaves the rows to the scan, which raises it
    let failing =
        "SELECT id FROM big WHERE k IN (SELECT v FROM u WHERE u.d * 9223372036854775807 > 0)";
    assert!(ids(&db, failing).is_err(), "{failing}");
}

#[test]
fn test_a_probe_reads_only_members_its_equality_answers() {
    let _guard = serial();
    let db = setup("domains");
    db.execute(
        "CREATE TABLE b (id INTEGER PRIMARY KEY, flag BOOLEAN, n FLOAT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO b VALUES (1, TRUE, 1.5), (2, FALSE, 4.0)", ())
        .unwrap();
    db.execute("CREATE TABLE j (id INTEGER PRIMARY KEY, doc JSON)", ())
        .unwrap();
    db.execute(
        "INSERT INTO j VALUES (1, '{\"a\":1}'), (2, '{\"a\":2}')",
        (),
    )
    .unwrap();
    for (sql, expected) in [
        // A BOOLEAN is no number: TRUE equals no k
        (
            "SELECT id FROM t WHERE k IN (SELECT flag FROM b) ORDER BY id",
            vec![],
        ),
        (
            "SELECT id FROM t WHERE k NOT IN (SELECT flag FROM b) ORDER BY id",
            vec![1, 2, 4, 5],
        ),
        (
            "SELECT id FROM t WHERE id IN (SELECT flag FROM b) ORDER BY id",
            vec![],
        ),
        // 1.5 equals no integer, 4.0 equals 4
        (
            "SELECT id FROM t WHERE k IN (SELECT n FROM b) ORDER BY id",
            vec![4],
        ),
        (
            "SELECT id FROM t WHERE id IN (SELECT n FROM b) ORDER BY id",
            vec![4],
        ),
    ] {
        assert_eq!(ids(&db, sql), Ok(expected), "{sql}");
    }
    let json = "SELECT id FROM j WHERE doc IN (SELECT doc FROM j WHERE id = 2) ORDER BY id";
    let general =
        "SELECT DISTINCT id FROM j WHERE doc IN (SELECT doc FROM j WHERE id = 2) ORDER BY id";
    assert_eq!(ids(&db, json), ids(&db, general), "{json}");
}

#[test]
fn test_a_long_probe_sees_its_statement_cancelled() {
    use stoolap::test_failpoints::{before_prepared_exists, before_prepared_scalar};
    let _guard = serial();
    let db = setup("cancel");
    db.execute(
        "CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO big SELECT value, 1, value FROM generate_series(1, 5000)",
        (),
    )
    .unwrap();
    let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
    force_exists_strategy(Some(false));
    for (sql, scalar) in [
        ("SELECT id FROM t WHERE t.id = 1 AND 10000 > ALL (SELECT v FROM big WHERE big.k = t.k)", false),
        // `v * 1 < 0` is no comparison the scan takes, so every row reaches the probe
        ("SELECT id FROM t WHERE t.id = 1 AND EXISTS (SELECT 1 FROM big WHERE big.k = t.k AND big.v * 1 < 0)", false),
        ("SELECT id FROM t WHERE t.id = 1 AND 0 = (SELECT v FROM big WHERE big.k = t.k AND big.v * 1 < 0)", true),
    ] {
        let ctx = stoolap::ExecutionContext::new();
        let handle = ctx.clone();
        if scalar {
            before_prepared_scalar(move || handle.cancel());
        } else {
            before_prepared_exists(move || handle.cancel());
        }
        let result = executor.execute_with_context(sql, &ctx);
        assert!(
            matches!(result, Err(stoolap::Error::QueryCancelled)),
            "{sql}: {:?}",
            result.map(|_| ())
        );
    }
}

#[test]
fn test_a_cancel_inside_a_block_is_raised_and_never_kept() {
    use stoolap::test_failpoints::before_prepared_scalar;
    let _guard = serial();
    let db = Database::open("memory://prepared_in_cancel_kept").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY)",
        "INSERT INTO t VALUES (1)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO u VALUES (1, 1)",
        "CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO big SELECT value, 1, value FROM generate_series(1, 5000)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
    let sql = "SELECT id FROM t WHERE id IN (SELECT (SELECT v FROM big WHERE big.k = u.k AND big.v * 1 < 0) FROM u)";
    let ctx = stoolap::ExecutionContext::new();
    let handle = ctx.clone();
    before_prepared_scalar(move || handle.cancel());
    let result = executor.execute_with_context(sql, &ctx);
    assert!(
        matches!(result, Err(stoolap::Error::QueryCancelled)),
        "{sql}: {:?}",
        result.map(|_| ())
    );
    // The scalar block gives NULL, so the IN is UNKNOWN for every row
    let again = executor
        .execute_with_context(sql, &stoolap::ExecutionContext::new())
        .map(|mut rows| {
            let mut count = 0;
            while rows.next() {
                count += 1;
            }
            count
        });
    assert_eq!(again.ok(), Some(0), "{sql} kept the cancel");
}

#[test]
fn test_a_conjunct_beside_a_narrowing_in_raises_its_error() {
    let _guard = serial();
    let db = Database::open("memory://prepared_in_narrowed_error").unwrap();
    for sql in [
        "CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO a VALUES (1, -9223372036854775808)",
        "CREATE TABLE b (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO b VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for sql in [
        "SELECT id FROM a WHERE id IN (SELECT k FROM b) AND ABS(x) >= 0",
        "SELECT id FROM a WHERE id IN (SELECT MAX(k) FROM b) AND ABS(x) >= 0",
    ] {
        let got = ids(&db, sql);
        assert!(
            got.is_err(),
            "{sql} answered {got:?} for a row whose ABS overflows"
        );
    }
}

#[test]
fn test_a_cancel_before_the_scan_stops_at_its_limit_is_raised() {
    use stoolap::test_failpoints::before_prepared_scalar;
    let _guard = serial();
    let db = Database::open("memory://prepared_in_cancel_limit").unwrap();
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO u VALUES (1, 1)",
        "CREATE TABLE big (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO big SELECT value, value FROM generate_series(1, 5000)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
    let sql = "SELECT id FROM big WHERE id >= (SELECT k FROM u WHERE u.id = 1) LIMIT 1000000";
    let ctx = stoolap::ExecutionContext::new();
    let handle = ctx.clone();
    before_prepared_scalar(move || handle.cancel());
    let result = executor.execute_with_context(sql, &ctx);
    assert!(
        matches!(result, Err(stoolap::Error::QueryCancelled)),
        "{sql}: {:?}",
        result.map(|_| ())
    );
}
