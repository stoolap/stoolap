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

//! A join plans how it runs once; what moves without changing the plan,
//! its statistics, is read per run, and a run
//! that cannot take the planned index reads each side once all the same

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::limit_hash_left_builds;
use stoolap::Database;

// The failpoint hooks are per thread, the counters process-wide in effect
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// The file engine plans no index join on a secondary index
#[cfg(not(feature = "test-filedb"))]
mod index_replaced {
    use super::*;
    use stoolap::test_failpoints::{after_join_index_chosen, join_side_runs, reset_all};

    fn rows(db: &Database, sql: &str) -> Vec<Vec<i64>> {
        let mut found: Vec<Vec<i64>> = db
            .query(sql, ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                vec![row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()]
            })
            .collect();
        found.sort();
        found
    }

    fn fill(db: &Database) {
        for sql in [
            "CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER)",
            "CREATE TABLE b (id INTEGER PRIMARY KEY, w INTEGER)",
            "CREATE INDEX b_w ON b (w)",
            "INSERT INTO a SELECT g.value, g.value % 7 FROM generate_series(1, 40) g",
            "INSERT INTO b SELECT g.value, g.value % 11 FROM generate_series(1, 60) g",
        ] {
            db.execute(sql, ()).unwrap();
        }
    }

    #[test]
    fn an_index_replaced_after_the_join_chose_it_leaves_each_side_read_once() {
        let _guard = serial();
        let db = Database::open("memory://join_plan_replaced_index").unwrap();
        fill(&db);
        let sql = "SELECT a.id, b.id FROM a JOIN b ON b.w = a.k WHERE a.id < 30 LIMIT 500";
        let want = rows(&db, sql);
        let other = db.clone();
        after_join_index_chosen(move || {
            other.execute("DROP INDEX b_w ON b", ()).unwrap();
            other.execute("CREATE INDEX b_w ON b (w)", ()).unwrap();
        });
        let before = join_side_runs();
        let got = rows(&db, sql);
        let sides = join_side_runs() - before;
        reset_all();
        assert_eq!(got, want);
        assert_eq!(
            sides, 2,
            "the outer side read for the index join is read again"
        );
    }

    #[test]
    fn a_grouped_join_whose_index_was_replaced_reads_each_side_once() {
        let _guard = serial();
        let db = Database::open("memory://join_plan_replaced_grouped_index").unwrap();
        fill(&db);
        let every = "SELECT a.id, COUNT(b.id) FROM a LEFT JOIN b ON b.w = a.k GROUP BY a.id";
        let sql = &format!("{every} LIMIT 5");
        let want = rows(&db, every);
        rows(&db, sql);
        let other = db.clone();
        let chosen = std::rc::Rc::new(std::cell::Cell::new(false));
        let seen = std::rc::Rc::clone(&chosen);
        after_join_index_chosen(move || {
            other.execute("DROP INDEX b_w ON b", ()).unwrap();
            other.execute("CREATE INDEX b_w ON b (w)", ()).unwrap();
            seen.set(true);
        });
        let before = join_side_runs();
        let got = rows(&db, sql);
        let sides = join_side_runs() - before;
        reset_all();
        assert!(chosen.get(), "the grouped join chose the index");
        assert_eq!(got.len(), 5);
        assert!(got.iter().all(|row| want.contains(row)), "{got:?}");
        assert_eq!(
            sides, 2,
            "the left side read for the grouped join is read again"
        );
    }
}

#[test]
fn a_limit_hash_join_builds_on_the_side_estimated_smaller_under_its_conditions() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_limit_hash_analyze").unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO s SELECT g.value, g.value % 2 FROM generate_series(1, 1000) g",
        "INSERT INTO t SELECT g.value, g.value % 2 FROM generate_series(1, 300) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let stmt = db
        .prepare("SELECT s.id, t.id FROM s JOIN t ON t.k = s.k WHERE s.k = 1 LIMIT $1")
        .unwrap();
    let builds = || {
        let before = limit_hash_left_builds();
        assert_eq!(stmt.query((1,)).unwrap().count(), 1);
        limit_hash_left_builds() - before
    };
    assert_eq!(
        builds(),
        1,
        "without statistics an equality keeps a tenth of s, fewer rows than t holds"
    );
    db.execute("ANALYZE s", ()).unwrap();
    db.execute("ANALYZE t", ()).unwrap();
    assert_eq!(
        builds(),
        0,
        "after ANALYZE the equality keeps half of s, more rows than t holds"
    );
}

#[test]
fn a_limit_hash_join_builds_on_the_side_its_index_counts_smaller() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_limit_hash_index_count").unwrap();
    for sql in [
        "CREATE TABLE ic_s (id INTEGER PRIMARY KEY, k INTEGER, f INTEGER)",
        "CREATE TABLE ic_t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ic_s SELECT g.value, g.value % 200, g.value % 200 \
         FROM generate_series(1, 20000) g",
        "INSERT INTO ic_t SELECT g.value, g.value % 200 FROM generate_series(1, 1500) g",
        "CREATE INDEX ic_s_f ON ic_s (f)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let literal = db
        .prepare(
            "SELECT ic_s.id, ic_t.id FROM ic_s JOIN ic_t ON ic_t.k = ic_s.k \
             WHERE ic_s.f = 1 LIMIT $1",
        )
        .unwrap();
    let before = limit_hash_left_builds();
    assert_eq!(literal.query((1,)).unwrap().count(), 1);
    assert_eq!(
        limit_hash_left_builds() - before,
        1,
        "the index counts 100 rows of ic_s for f = 1, fewer than ic_t holds"
    );
    let bound = db
        .prepare(
            "SELECT ic_s.id, ic_t.id FROM ic_s JOIN ic_t ON ic_t.k = ic_s.k \
             WHERE ic_s.f = $2 LIMIT $1",
        )
        .unwrap();
    let before = limit_hash_left_builds();
    assert_eq!(bound.query((1, 1)).unwrap().count(), 1);
    assert_eq!(
        limit_hash_left_builds() - before,
        1,
        "the index counts the rows of the run's parameter"
    );
    let before = limit_hash_left_builds();
    assert_eq!(bound.query((1, 1000)).unwrap().count(), 0);
    assert_eq!(
        limit_hash_left_builds() - before,
        1,
        "the index counts no row of ic_s for f = 1000"
    );
}

/// How many times a LIMIT hash join of `ts_s` and `ts_t` under `cond`, run
/// with `params` after the limit and giving `rows` rows, built from the left
fn left_builds(db: &Database, cond: &str, params: Vec<stoolap::Value>, rows: usize) -> usize {
    let sql = format!(
        "SELECT ts_s.id, ts_t.id FROM ts_s JOIN ts_t ON ts_t.k = ts_s.k WHERE {cond} LIMIT $1"
    );
    let stmt = db.prepare(&sql).unwrap();
    let mut bound = vec![stoolap::Value::Integer(1)];
    bound.extend(params);
    let before = limit_hash_left_builds();
    assert_eq!(stmt.query(bound).unwrap().count(), rows, "{sql}");
    limit_hash_left_builds() - before
}

#[test]
fn a_range_counted_by_its_index_reads_its_bound_as_the_column_compares_it() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_count_bound_type").unwrap();
    for sql in [
        "CREATE TABLE ts_s (id INTEGER PRIMARY KEY, k INTEGER, ts TIMESTAMP)",
        "CREATE TABLE ts_t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ts_s SELECT g.value, g.value % 200, '2024-01-02 00:00:00' \
         FROM generate_series(1, 3000) g",
        "INSERT INTO ts_t SELECT g.value, g.value % 200 FROM generate_series(1, 1500) g",
        "CREATE INDEX ts_s_ts ON ts_s (ts)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(
        left_builds(&db, "ts_s.ts < '3000-01-01'", vec![], 1),
        0,
        "every one of the 3,000 rows of ts_s is before 3000-01-01, more than ts_t holds"
    );
    assert_eq!(
        left_builds(&db, "ts_s.ts < $2", vec!["3000-01-01".into()], 1),
        0,
        "a TEXT parameter is read as a TIMESTAMP too"
    );
}

#[test]
fn a_range_counted_by_its_index_counts_no_null_key() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_count_no_null").unwrap();
    for sql in [
        "CREATE TABLE ts_s (id INTEGER PRIMARY KEY, k INTEGER, f INTEGER)",
        "CREATE TABLE ts_t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ts_s SELECT g.value, g.value % 200, \
         CASE WHEN g.value <= 2000 THEN NULL WHEN g.value <= 2500 THEN 1 ELSE 9 END \
         FROM generate_series(1, 3000) g",
        "INSERT INTO ts_t SELECT g.value, g.value % 200 FROM generate_series(1, 1000) g",
        "CREATE INDEX ts_s_f ON ts_s (f)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(
        left_builds(&db, "ts_s.f < 5", vec![], 1),
        1,
        "f < 5 keeps the 500 rows of 1, fewer than ts_t holds, and none of the 2,000 NULLs"
    );
}

#[test]
fn an_in_list_counted_by_its_index_counts_each_key_once() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_count_in_once").unwrap();
    for sql in [
        "CREATE TABLE ts_s (id INTEGER PRIMARY KEY, k INTEGER, f INTEGER)",
        "CREATE TABLE ts_t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ts_s SELECT g.value, g.value % 200, g.value % 20 \
         FROM generate_series(1, 20000) g",
        "INSERT INTO ts_t SELECT g.value, g.value % 200 FROM generate_series(1, 1500) g",
        "CREATE INDEX ts_s_f ON ts_s (f)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(
        left_builds(&db, "ts_s.f IN ($2, $3)", vec![1.into(), 1.into()], 1),
        1,
        "two parameters bound to 1 keep the 1,000 rows of 1 once, fewer than ts_t holds"
    );
    assert_eq!(
        left_builds(&db, "ts_s.f IN (1, 1.0)", vec![], 1),
        1,
        "1.0 is the key 1 of an INTEGER column"
    );
}

#[test]
fn an_in_list_on_an_index_that_counts_nothing_reads_no_key() {
    use stoolap::test_failpoints::estimate_in_keys_read;
    let _guard = serial();
    let db = Database::open("memory://join_plan_in_pk_no_keys").unwrap();
    for sql in [
        "CREATE TABLE ts_s (id INTEGER PRIMARY KEY, k INTEGER, f INTEGER)",
        "CREATE TABLE ts_t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ts_s SELECT g.value, g.value % 200, g.value % 20 \
         FROM generate_series(1, 2000) g",
        "INSERT INTO ts_t SELECT g.value, g.value % 200 FROM generate_series(1, 1500) g",
        "CREATE INDEX ts_s_f ON ts_s (f)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let list: Vec<String> = (2..42).map(|at| format!("${at}")).collect();
    let keys = || (1..=40).map(stoolap::Value::Integer).collect::<Vec<_>>();
    for (column, counts) in [("id", false), ("f", true)] {
        let before = estimate_in_keys_read();
        left_builds(
            &db,
            &format!("ts_s.{column} IN ({})", list.join(", ")),
            keys(),
            1,
        );
        assert_eq!(
            estimate_in_keys_read() > before,
            counts,
            "{column}: the primary key's index counts no key, the index on f counts them"
        );
    }
}

#[test]
fn an_index_created_after_a_run_counts_on_the_next() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_index_after_run").unwrap();
    for sql in [
        "CREATE TABLE ia_s (id INTEGER PRIMARY KEY, k INTEGER, f INTEGER)",
        "CREATE TABLE ia_t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ia_s SELECT g.value, g.value % 200, g.value % 200 \
         FROM generate_series(1, 20000) g",
        "INSERT INTO ia_t SELECT g.value, g.value % 200 FROM generate_series(1, 1500) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let stmt = db
        .prepare(
            "SELECT ia_s.id, ia_t.id FROM ia_s JOIN ia_t ON ia_t.k = ia_s.k \
             WHERE ia_s.f = $2 LIMIT $1",
        )
        .unwrap();
    let builds = || {
        let before = limit_hash_left_builds();
        assert_eq!(stmt.query((1, 1)).unwrap().count(), 1);
        limit_hash_left_builds() - before
    };
    assert_eq!(builds(), 0, "a tenth of ia_s is more rows than ia_t holds");
    db.execute("CREATE INDEX ia_s_f ON ia_s (f)", ()).unwrap();
    assert_eq!(
        builds(),
        1,
        "the index created after the run counts 100 rows of ia_s"
    );
}

#[test]
fn explain_analyze_leaves_no_estimate_behind() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_explain_analyze_stateless").unwrap();
    for sql in [
        "CREATE TABLE ea_s (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE ea_t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ea_s SELECT g.value, g.value % 200 FROM generate_series(1, 20000) g",
        "INSERT INTO ea_t SELECT g.value, g.value % 200 FROM generate_series(1, 1500) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let stmt = db
        .prepare(
            "SELECT ea_s.id, ea_t.id FROM ea_s JOIN ea_t ON ea_t.k = ea_s.k \
             WHERE ea_s.k = 1 LIMIT $1",
        )
        .unwrap();
    let builds = || {
        let before = limit_hash_left_builds();
        assert_eq!(stmt.query((1,)).unwrap().count(), 1);
        limit_hash_left_builds() - before
    };
    assert_eq!(builds(), 0, "a tenth of ea_s is more rows than ea_t holds");
    for _ in 0..2 {
        let explained = db.query(
            "EXPLAIN ANALYZE SELECT ea_s.id FROM ea_s WHERE ea_s.k = 1",
            (),
        );
        explained.unwrap().for_each(|row| {
            row.unwrap();
        });
    }
    assert_eq!(
        builds(),
        0,
        "the estimate after EXPLAIN ANALYZE is the estimate before it"
    );
}
