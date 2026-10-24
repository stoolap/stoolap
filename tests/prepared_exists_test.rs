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

//! EXISTS and NOT EXISTS in the prepared family. A block's correlation
//! equalities pick its candidate rows; EXISTS is TRUE when a candidate
//! passes the rest of its WHERE, else the first error a candidate raised,
//! else FALSE. The probe, the keyed set and the once-per-run answer agree.

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    before_prepared_exists, force_exists_strategy, general_path_hits, prepared_exists_counts,
    prepared_rows_narrowed, subquery_cache_hits, GeneralPath,
};
use stoolap::Database;

// The counters and the forced strategy are process-wide
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

/// `t` and its copy `w` are the outer rows; `u` the inner. `d * MAX`
/// overflows for `d = 2`: key 1 has a candidate it raises for, then one it
/// passes for, key 2 only one it raises for, a NULL key one it raises for,
/// key 4 one it passes for; key 5 has none
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://prepared_exists_{name}")).unwrap();
    for table in ["t", "w"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER, s TEXT, f FLOAT)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} VALUES (1, 1, 'a', 1.0), (2, 2, 'b', 2.0), \
                 (3, NULL, NULL, NULL), (4, 4, 'd', 4.0), (5, 5, 'e', 5.0)"
            ),
            (),
        )
        .unwrap();
    }
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, s TEXT, f FLOAT, v INTEGER, d INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO u VALUES (1, 1, 'a', 1.0, 10, 2), (2, 1, 'a', 1.0, 20, 1), \
         (3, 2, 'b', 2.0, 5, 2), (4, NULL, NULL, NULL, 99, 2), (5, 4, 'd', 4.0, 1, 1)",
        (),
    )
    .unwrap();
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

fn fails(db: &Database, sql: &str) -> bool {
    db.query(sql, ())
        .and_then(|rows| {
            rows.map(|row| row.map(|_| ()))
                .collect::<Result<Vec<()>, _>>()
        })
        .is_err()
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

const ALL_GENERAL: [GeneralPath; 10] = [
    GeneralPath::CorrelatedScan,
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::PushdownConstant,
    GeneralPath::ProjectionEvaluator,
    GeneralPath::OrderByEvaluator,
    GeneralPath::AntiJoin,
    GeneralPath::ExistsSemiJoin,
    GeneralPath::ExistsSubquery,
];

fn general_hits() -> Vec<usize> {
    ALL_GENERAL.iter().map(|p| general_path_hits(*p)).collect()
}

/// The block `EXISTS (SELECT 1 FROM u WHERE {block})` at the statement,
/// and inside a scalar block over `w`, which reads it per `w` row
fn at_each_depth(block: &str, negated: bool) -> [String; 2] {
    let not = if negated { "NOT " } else { "" };
    [
        format!("SELECT id FROM t WHERE {not}EXISTS (SELECT 1 FROM u WHERE {block}) ORDER BY id"),
        format!(
            "SELECT id FROM t WHERE 1 = (SELECT 1 FROM w WHERE w.id = t.id AND \
             {not}EXISTS (SELECT 1 FROM u WHERE {})) ORDER BY id",
            block.replace("t.", "w.")
        ),
    ]
}

/// Each strategy a block may take: the rule's choice, the probe, the keyed set
const STRATEGIES: [Option<bool>; 3] = [None, Some(false), Some(true)];

#[test]
fn test_exists_belongs_to_the_family_wherever_where_holds_it() {
    let _guard = serial();
    let db = setup("family");
    for sql in [
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
        "SELECT id FROM t WHERE NOT EXISTS (SELECT * FROM u WHERE u.k = t.k)",
        "SELECT id FROM t WHERE id > 1 OR EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
        "SELECT id FROM t WHERE CASE WHEN id > 2 THEN EXISTS (SELECT 1 FROM u) ELSE FALSE END",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE EXISTS (SELECT 1 FROM w WHERE w.k = u.k))",
        "SELECT id FROM t WHERE 1 = (SELECT 1 FROM w WHERE w.id = t.id AND EXISTS (SELECT 1 FROM u WHERE u.k = w.k))",
        "SELECT id FROM t WHERE EXISTS (SELECT u.v + 1 AS x, u.s FROM u WHERE u.k = t.k) LIMIT 2",
        "SELECT id FROM t WHERE EXISTS (SELECT COUNT(*) FROM u WHERE u.k = t.k)",
        "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.k = t.k) FROM t WHERE EXISTS (SELECT 1 FROM u)",
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
    }
    for (sql, reason) in [
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k ORDER BY 1)",
            "an ORDER BY position in a subquery block",
        ),
        // The WHERE reads the alias, so the SELECT list's forms count
        (
            "SELECT id FROM t WHERE EXISTS (SELECT (u.k, u.id) IN (SELECT k, id FROM w) AS x FROM u WHERE x)",
            "a tuple on the left of IN (subquery)",
        ),
    ] {
        assert_eq!(
            prepared_line(&db, sql),
            format!("Prepared: no ({reason})"),
            "{sql}"
        );
    }
}

#[test]
fn test_every_strategy_answers_the_contract() {
    let _guard = serial();
    let db = setup("contract");
    let cases: [(&str, bool, Vec<i64>); 8] = [
        ("u.k = t.k", false, vec![1, 2, 4]),
        ("u.k = t.k", true, vec![3, 5]),
        ("u.k = t.k AND u.v > 8", false, vec![1]),
        ("u.k = t.k AND u.v > 8", true, vec![2, 3, 4, 5]),
        ("t.k = u.k AND u.v > 100", false, vec![]),
        ("t.k = u.k AND u.v > 100", true, vec![1, 2, 3, 4, 5]),
        // TEXT keys take the keyed set as INTEGER keys do
        ("u.s = t.s AND u.v > 8", false, vec![1]),
        ("u.s = t.s", true, vec![3, 5]),
    ];
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        for (block, negated, expected) in &cases {
            for sql in at_each_depth(block, *negated) {
                assert_eq!(ids(&db, &sql), *expected, "{sql} under {strategy:?}");
            }
        }
    }
}

#[test]
fn test_a_block_without_rows_answers_false_and_never_unknown() {
    let _guard = serial();
    let db = setup("empty");
    db.execute("CREATE TABLE e (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        for (sql, expected) in [
            (
                "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM e WHERE e.k = t.k) ORDER BY id",
                vec![],
            ),
            (
                "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM e WHERE e.k = t.k) ORDER BY id",
                vec![1, 2, 3, 4, 5],
            ),
            (
                "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM e) ORDER BY id",
                vec![1, 2, 3, 4, 5],
            ),
            (
                "SELECT id FROM t WHERE (EXISTS (SELECT 1 FROM e WHERE e.k = t.k)) IS FALSE ORDER BY id",
                vec![1, 2, 3, 4, 5],
            ),
        ] {
            assert_eq!(ids(&db, sql), expected, "{sql} under {strategy:?}");
        }
    }
}

#[test]
fn test_a_passing_candidate_wins_over_an_error() {
    let _guard = serial();
    let db = setup("errors");
    let block = "u.k = t.k AND u.d * 9223372036854775807 > 0";
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        // Key 1 passes beside an error, 4 passes, 5 has no candidate, and a
        // NULL key has none, so the failing row with a NULL key never counts
        for sql in at_each_depth(block, false) {
            let sql = sql.replace("ORDER BY id", "AND t.id <> 2 ORDER BY id");
            assert_eq!(ids(&db, &sql), vec![1, 4], "{sql} under {strategy:?}");
        }
        // Key 2's only candidate raises
        for sql in at_each_depth(block, false) {
            assert!(fails(&db, &sql), "{sql} under {strategy:?}");
        }
        for sql in at_each_depth(block, true) {
            assert!(fails(&db, &sql), "{sql} under {strategy:?}");
        }
    }
    // An uncorrelated block's error is the statement's
    assert!(fails(
        &db,
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.d * 9223372036854775807 < 0)"
    ));
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.d * 9223372036854775807 > 0) ORDER BY id"
        ),
        vec![1, 2, 3, 4, 5]
    );
}

#[test]
fn test_a_keyed_set_takes_only_keys_a_hash_tells_apart() {
    let _guard = serial();
    let db = setup("key_types");
    force_exists_strategy(Some(true));
    for (block, expected, keyed) in [
        ("u.k = t.k", vec![1, 2, 4], true),
        ("u.s = t.s", vec![1, 2, 4], true),
        ("u.f = t.f", vec![1, 2, 4], false),
        // INTEGER against FLOAT compares by value, which a hash does not
        ("u.f = t.k", vec![1, 2, 4], false),
        ("u.k = t.k AND u.v > t.id", vec![1, 2], false),
        ("u.k = t.k + 0", vec![1, 2, 4], false),
    ] {
        let sql =
            format!("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE {block}) ORDER BY id");
        let (_, built, _) = prepared_exists_counts();
        assert_eq!(ids(&db, &sql), expected, "{sql}");
        let (_, after, _) = prepared_exists_counts();
        assert_eq!(after > built, keyed, "{sql} takes a keyed set");
    }
}

#[test]
fn test_the_rule_probes_for_few_outer_rows_or_a_small_limit_on_an_index() {
    let _guard = serial();
    let db = setup("choice");
    db.execute(
        "CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    let values: Vec<String> = (1..=200)
        .map(|i| format!("({i}, {}, {i})", i % 7))
        .collect();
    db.execute(&format!("INSERT INTO big VALUES {}", values.join(", ")), ())
        .unwrap();
    let pure = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM big WHERE big.k = t.k) ORDER BY id";
    let limited = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) LIMIT 2";
    // Whether the run took the keyed set: it looked keys up or read the
    // statement's rows by them, and probed none
    let keyed = |sql: &str, expected: Vec<i64>| {
        let (probes, _, lookups) = prepared_exists_counts();
        let narrowed = prepared_rows_narrowed();
        assert_eq!(ids(&db, sql), expected, "{sql}");
        let (after_probes, _, after_lookups) = prepared_exists_counts();
        let used = after_lookups > lookups || prepared_rows_narrowed() > narrowed;
        match (after_probes > probes, used) {
            (false, true) => true,
            (true, false) => false,
            other => panic!("{sql} probed and used the set: {other:?}"),
        }
    };
    assert!(keyed(pure, vec![1, 2, 4, 5]), "{pure} without an index");
    assert!(keyed(limited, vec![1, 2]), "{limited} without an index");
    db.execute("CREATE INDEX big_k ON big (k)", ()).unwrap();
    db.execute("CREATE INDEX u_k ON u (k)", ()).unwrap();
    // Five outer rows against 200 inner ones, and a LIMIT of two
    assert!(!keyed(pure, vec![1, 2, 4, 5]), "{pure} probes the index");
    assert!(!keyed(limited, vec![1, 2]), "{limited} probes the index");
    // Five outer rows against five inner ones, no LIMIT: the set
    let few = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) ORDER BY id";
    assert!(keyed(few, vec![1, 2, 4]), "{few}");
    // More than the keys in the WHERE: the set, as the general path chooses
    let more =
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM big WHERE big.k = t.k AND big.v > 0) ORDER BY id";
    assert!(keyed(more, vec![1, 2, 4, 5]), "{more}");
}

#[test]
fn test_a_keyed_outcome_uses_at_most_one_word() {
    let bytes = std::mem::size_of::<stoolap::executor::context::ExistsOutcome>();
    assert!(
        bytes <= std::mem::size_of::<usize>(),
        "a successful EXISTS key reserves {bytes} bytes for its outcome"
    );
}

#[test]
fn test_dropping_a_session_releases_prepared_block_answers() {
    let _guard = serial();
    let db = setup("drop_cached_answers");
    force_exists_strategy(Some(true));
    let sql =
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.v > 8) ORDER BY id";
    let (_, before, _) = prepared_exists_counts();
    assert_eq!(ids(&db, sql), vec![1]);
    let (_, built, _) = prepared_exists_counts();
    assert_eq!(built, before + 1);
    let hits = subquery_cache_hits();
    assert_eq!(ids(&db, sql), vec![1]);
    assert_eq!(prepared_exists_counts().1, built);
    assert_eq!(subquery_cache_hits(), hits + 1);

    drop(db.clone());

    let hits = subquery_cache_hits();
    assert_eq!(ids(&db, sql), vec![1]);
    assert_eq!(
        prepared_exists_counts().1,
        built + 1,
        "dropping a database session must release its thread's block answers"
    );
    assert_eq!(subquery_cache_hits(), hits);
}

#[test]
fn test_an_answer_is_computed_once_per_run_and_kept_while_its_stamp_holds() {
    let _guard = serial();
    let db = setup("kept");
    let keyed =
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.v > 8) ORDER BY id";
    let once = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.v > 50) ORDER BY id";
    let counts = |sql: &str, expected: Vec<i64>| {
        let (probes, built, _) = prepared_exists_counts();
        let hits = subquery_cache_hits();
        assert_eq!(ids(&db, sql), expected, "{sql}");
        let (after_probes, after_built, _) = prepared_exists_counts();
        (
            after_probes - probes,
            after_built - built,
            subquery_cache_hits() - hits,
        )
    };
    assert_eq!(counts(keyed, vec![1]), (0, 1, 0), "{keyed} builds its set");
    assert_eq!(counts(keyed, vec![1]), (0, 0, 1), "{keyed} keeps its set");
    assert_eq!(
        counts(once, vec![1, 2, 3, 4, 5]),
        (1, 0, 0),
        "{once} probes once"
    );
    assert_eq!(
        counts(once, vec![1, 2, 3, 4, 5]),
        (0, 0, 1),
        "{once} is kept"
    );
    db.execute("INSERT INTO u VALUES (6, 5, 'e', 5.0, 9, 1)", ())
        .unwrap();
    assert_eq!(
        counts(keyed, vec![1, 5]),
        (0, 1, 0),
        "{keyed} after a write"
    );
    db.execute("DELETE FROM u WHERE v > 50", ()).unwrap();
    assert_eq!(counts(once, vec![]), (1, 0, 0), "{once} after a write");

    // The parameters are part of what is kept
    let param =
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.v > $1) ORDER BY id";
    for (value, expected) in [(8, vec![1, 5]), (0, vec![1, 2, 4, 5]), (8, vec![1, 5])] {
        let found: Vec<i64> = db
            .query(param, (value,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(found, expected, "{param} with {value}");
    }
}

#[test]
fn test_a_transaction_reads_its_own_writes_through_the_block() {
    let _guard = serial();
    let db = setup("transaction");
    let sql = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) ORDER BY id";
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        assert_eq!(ids(&db, sql), vec![1, 2, 4], "{sql} under {strategy:?}");
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO u VALUES (9, 5, 'e', 5.0, 1, 1)", ())
            .unwrap();
        let found: Vec<i64> = tx
            .query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(
            found,
            vec![1, 2, 4, 5],
            "{sql} in the transaction under {strategy:?}"
        );
        tx.rollback().unwrap();
        assert_eq!(
            ids(&db, sql),
            vec![1, 2, 4],
            "{sql} after the rollback under {strategy:?}"
        );
    }
}

#[test]
fn test_every_block_reads_the_statement_view() {
    let _guard = serial();
    let sql = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) ORDER BY id";
    let once = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = 5) ORDER BY id";
    for level in ["READ COMMITTED", "SNAPSHOT"] {
        for strategy in STRATEGIES {
            force_exists_strategy(strategy);
            for (query, before, after) in [
                (sql, vec![1, 2, 4], vec![1, 2, 4, 5]),
                (once, vec![], vec![1, 2, 3, 4, 5]),
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
                before_prepared_exists(move || {
                    writer
                        .execute("INSERT INTO u VALUES (7, 5, 'e', 5.0, 1, 1)", ())
                        .unwrap();
                    flag.set(true);
                });
                assert_eq!(ids(&db, query), before, "{level} {strategy:?}: {query}");
                assert!(
                    done.get(),
                    "{level} {strategy:?}: the commit ran inside {query}"
                );
                assert_eq!(
                    ids(&db, query),
                    after,
                    "{level} {strategy:?}: the next {query}"
                );
            }
        }
    }
}

#[test]
fn test_a_branch_not_taken_reads_no_block() {
    let _guard = serial();
    let db = setup("branch");
    for sql in [
        "SELECT id FROM t WHERE id > 100 AND (EXISTS (SELECT 1 FROM u WHERE u.k = t.k) OR id > 200)",
        "SELECT id FROM t WHERE CASE WHEN id > 100 THEN EXISTS (SELECT 1 FROM u WHERE u.k = t.k) ELSE FALSE END",
        "SELECT id FROM t WHERE id < 100 OR EXISTS (SELECT 1 FROM u WHERE u.v > 0) ORDER BY id",
    ] {
        let before = prepared_exists_counts();
        let found = ids(&db, sql);
        assert_eq!(prepared_exists_counts(), before, "{sql} read a block");
        assert_eq!(found.is_empty(), !sql.contains("id < 100"), "{sql}");
    }
    // A top-level conjunct's keyed set is built before the scan, which its
    // keys narrow, as a block compared with a column there runs before it
    let sql = "SELECT id FROM t WHERE id > 100 AND EXISTS (SELECT 1 FROM u WHERE u.k = t.k)";
    let (_, built, _) = prepared_exists_counts();
    assert!(ids(&db, sql).is_empty());
    assert_eq!(prepared_exists_counts().1, built + 1, "{sql}");
}

#[test]
fn test_a_top_level_exists_reads_only_the_rows_its_keys_name() {
    let _guard = serial();
    let db = setup("narrowing");
    db.execute(
        "CREATE TABLE big (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    let values: Vec<String> = (1..=300)
        .map(|i| format!("({i}, {}, {i})", i % 30))
        .collect();
    db.execute(&format!("INSERT INTO big VALUES {}", values.join(", ")), ())
        .unwrap();
    db.execute("CREATE INDEX big_k ON big (k)", ()).unwrap();
    force_exists_strategy(Some(true));
    // By the index on k, and by the primary key
    for (sql, expected) in [
        (
            "SELECT id FROM big WHERE EXISTS (SELECT 1 FROM u WHERE u.k = big.k) ORDER BY id",
            (1..=300).filter(|i| [1, 2, 4].contains(&(i % 30))).collect::<Vec<i64>>(),
        ),
        (
            "SELECT id FROM big WHERE EXISTS (SELECT 1 FROM u WHERE u.id = big.id AND u.v > 8) ORDER BY id",
            vec![1, 2, 4],
        ),
        (
            "SELECT id FROM big WHERE v > 100 AND EXISTS (SELECT 1 FROM u WHERE u.k = big.k) ORDER BY id",
            (101..=300).filter(|i| [1, 2, 4].contains(&(i % 30))).collect::<Vec<i64>>(),
        ),
    ] {
        let before = prepared_rows_narrowed();
        assert_eq!(ids(&db, sql), expected, "{sql}");
        assert_eq!(prepared_rows_narrowed(), before + 1, "{sql} read big by its keys");
    }
    // Key 2's only candidate raises, and big holds rows with key 2
    let failing =
        "SELECT id FROM big WHERE EXISTS (SELECT 1 FROM u WHERE u.k = big.k AND u.d * 9223372036854775807 > 0)";
    assert!(fails(&db, failing), "{failing}");
    let passing = "SELECT id FROM big WHERE EXISTS (SELECT 1 FROM u WHERE u.k = big.k AND u.d * 9223372036854775807 > 0) AND k <> 2 ORDER BY id";
    assert_eq!(
        ids(&db, passing),
        (1..=300)
            .filter(|i| [1, 4].contains(&(i % 30)))
            .collect::<Vec<i64>>(),
        "{passing}"
    );
    let limited = "SELECT id FROM big WHERE EXISTS (SELECT 1 FROM u WHERE u.k = big.k) LIMIT 3";
    let before = prepared_rows_narrowed();
    let found = ids(&db, limited);
    assert_eq!(found.len(), 3, "{limited}");
    assert!(
        found.iter().all(|id| [1, 2, 4].contains(&(id % 30))),
        "{limited}: {found:?}"
    );
    // A table that holds volumes reads the keys as candidates, which do not
    // answer under a LIMIT, so it scans
    let narrowed = if cfg!(feature = "test-filedb") { 0 } else { 1 };
    assert_eq!(prepared_rows_narrowed(), before + narrowed, "{limited}");
    // NOT EXISTS reads by the primary key only, the keys that passed left out
    let not = "SELECT id FROM big WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = big.k)";
    let before = prepared_rows_narrowed();
    assert_eq!(ids(&db, not).len(), 270, "{not}");
    assert_eq!(prepared_rows_narrowed(), before, "{not}");
    for (sql, expected) in [
        (
            "SELECT id FROM big WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.id = big.id) ORDER BY id",
            (6..=300).collect::<Vec<i64>>(),
        ),
        (
            "SELECT id FROM big WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.id = big.id) LIMIT 3",
            vec![6, 7, 8],
        ),
        // Ids 1, 3 and 4 raise; 2 and 5 pass
        (
            "SELECT id FROM big WHERE id > 4 AND NOT EXISTS (SELECT 1 FROM u WHERE u.id = big.id AND u.d * 9223372036854775807 > 0) ORDER BY id",
            (6..=300).collect::<Vec<i64>>(),
        ),
    ] {
        let before = prepared_rows_narrowed();
        assert_eq!(ids(&db, sql), expected, "{sql}");
        assert_eq!(prepared_rows_narrowed(), before + 1, "{sql} read big by its keys");
    }
    let failing = "SELECT id FROM big WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.id = big.id AND u.d * 9223372036854775807 > 0) LIMIT 3";
    assert!(fails(&db, failing), "{failing}");
}

#[test]
fn test_an_in_family_statement_never_reaches_the_general_path() {
    let _guard = serial();
    let db = setup("general");
    let mut corpus: Vec<String> = Vec::new();
    for block in [
        "u.k = t.k",
        "u.k = t.k AND u.v > 8",
        "u.s = t.s",
        "u.f = t.f",
    ] {
        for negated in [false, true] {
            corpus.extend(at_each_depth(block, negated));
        }
    }
    corpus.push("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.v > 50)".to_string());
    corpus.push(
        "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k) LIMIT 1".to_string(),
    );
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        for sql in &corpus {
            assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
            let before = general_hits();
            ids(&db, sql);
            assert_eq!(general_hits(), before, "{sql} under {strategy:?}");
        }
    }
    // The same NOT EXISTS with an inner ORDER BY position takes the anti-join
    let before = general_path_hits(GeneralPath::AntiJoin);
    let old = "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k ORDER BY 1)";
    assert_eq!(ids(&db, old).len(), 2);
    assert!(general_path_hits(GeneralPath::AntiJoin) > before, "{old}");
}

#[test]
fn test_a_kept_answer_is_of_the_scopes_its_names_bound_in() {
    let _guard = serial();
    let db = Database::open("memory://prepared_exists_binding").unwrap();
    for sql in [
        "CREATE TABLE t1 (id INTEGER PRIMARY KEY, k INTEGER, p TEXT)",
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "INSERT INTO t1 VALUES (1, 1, 'p')",
        "INSERT INTO t2 VALUES (1, 1)",
        "INSERT INTO u VALUES (1, 1, 'p')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    force_exists_strategy(Some(true));
    // "p" is t1's column, and a TEXT literal over t2
    for table in ["t1", "t2", "t1"] {
        let sql = format!(
            "SELECT id FROM {table} t WHERE id = 0 OR EXISTS (SELECT 1 FROM u WHERE u.a = t.k AND u.b = \"p\")"
        );
        assert_eq!(ids(&db, &sql), vec![1], "{sql}");
    }
}

#[test]
fn test_a_volatile_block_takes_no_keyed_set() {
    let _guard = serial();
    let db = setup("volatile");
    force_exists_strategy(Some(true));
    let sql = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND RANDOM() < 2) ORDER BY id";
    let (probes, built, _) = prepared_exists_counts();
    assert_eq!(ids(&db, sql), vec![1, 2, 4]);
    let (after_probes, after_built, _) = prepared_exists_counts();
    assert_eq!(after_built, built, "{sql} built a keyed set");
    assert!(after_probes > probes, "{sql} probed");
}

#[test]
fn test_a_value_the_where_computes_raises_only_on_a_candidate() {
    let _guard = serial();
    let db = setup("hoisted");
    db.execute(
        "CREATE TABLE e (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    let rows = |sql: &str, param: i64| {
        db.query(sql, (param,))
            .and_then(|rows| {
                rows.map(|row| row.map(|row| row.get::<i64>(0).unwrap()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(|e| e.to_string())
    };
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        for sql in [
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM e WHERE e.v > $1 * 9223372036854775807)",
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM e WHERE e.k = t.k AND e.v > $1 * 9223372036854775807)",
            "SELECT id FROM t WHERE 1 = (SELECT 1 FROM w WHERE w.id = t.id AND EXISTS (SELECT 1 FROM e WHERE e.k = w.k AND e.v > $1 * 9223372036854775807))",
        ] {
            assert_eq!(rows(sql, 2), Ok(vec![]), "{sql} under {strategy:?}");
        }
        // u holds candidates for keys 1, 2 and 4: each raises
        let sql = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.v > $1 * 9223372036854775807)";
        assert!(rows(sql, 2).is_err(), "{sql} under {strategy:?}");
        assert_eq!(rows(sql, 1), Ok(vec![]), "{sql} under {strategy:?}");
        // The value reads the outer row: key 5 has no candidate, key 2 has one
        let correlated = "SELECT id FROM t WHERE t.id = $1 AND EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.v > t.k * 9223372036854775807)";
        assert_eq!(
            rows(correlated, 5),
            Ok(vec![]),
            "{correlated} under {strategy:?}"
        );
        assert!(
            rows(correlated, 2).is_err(),
            "{correlated} under {strategy:?}"
        );
    }
}

#[test]
fn test_a_block_drawing_per_call_at_any_depth_takes_no_keyed_set() {
    let _guard = serial();
    let db = setup("volatile_nested");
    force_exists_strategy(Some(true));
    let sql = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND EXISTS (SELECT 1 FROM w WHERE w.k = u.k AND RANDOM() < 2)) ORDER BY id";
    let (_, built, _) = prepared_exists_counts();
    assert_eq!(ids(&db, sql), vec![1, 2, 4]);
    assert_eq!(prepared_exists_counts().1, built, "{sql} built a keyed set");
}

#[test]
fn test_a_failed_value_raises_only_where_it_is_read() {
    let _guard = serial();
    let db = setup("failed_reads");
    let rows = |sql: &str| {
        db.query(sql, (2,))
            .and_then(|rows| {
                rows.map(|row| row.map(|row| row.get::<i64>(0).unwrap()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(|e| e.to_string())
    };
    for strategy in STRATEGIES {
        force_exists_strategy(strategy);
        // The scalar block reads no failed slot, and its FALSE ends the conjunction
        let scalar = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id AND 0 = (SELECT 1 FROM w WHERE w.id = u.id) AND u.v > $1 * 9223372036854775807)";
        assert_eq!(rows(scalar), Ok(vec![]), "{scalar} under {strategy:?}");
        // A FLOAT operand of AND is TRUE to the conjunction, which goes on to the failed slot
        let float = "SELECT id FROM t WHERE t.id = 1 AND EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.f AND u.v > $1 * 9223372036854775807)";
        assert!(rows(float).is_err(), "{float} under {strategy:?}");
        // A FALSE first ends it before the failed slot
        let first = "SELECT id FROM t WHERE t.id = 1 AND EXISTS (SELECT 1 FROM u WHERE u.k = t.k AND u.v < 0 AND u.v > $1 * 9223372036854775807)";
        assert_eq!(rows(first), Ok(vec![]), "{first} under {strategy:?}");
    }
}
