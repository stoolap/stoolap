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
//! statistics and the workload's memory limit, is read per run, and a run
//! that cannot take the planned index reads each side once all the same

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::optimizer::{global_workload_learner, WorkloadConfig};
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
fn a_limit_hash_join_builds_on_the_side_statistics_found_smaller_after_analyze() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_limit_hash_analyze").unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO s VALUES (1, 3)",
        "INSERT INTO t SELECT g.value, g.value % 10 FROM generate_series(1, 3000) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let stmt = db
        .prepare("SELECT s.id, t.id FROM s JOIN t ON t.k = s.k LIMIT $1")
        .unwrap();
    let builds = || {
        let before = limit_hash_left_builds();
        assert_eq!(stmt.query((1,)).unwrap().count(), 1);
        limit_hash_left_builds() - before
    };
    assert_eq!(
        builds(),
        0,
        "without statistics neither side is known smaller"
    );
    db.execute("ANALYZE s", ()).unwrap();
    db.execute("ANALYZE t", ()).unwrap();
    assert_eq!(builds(), 1, "after ANALYZE the one-row side builds");
}

#[test]
fn a_workload_memory_limit_keeps_a_join_off_a_hash_table() {
    let _guard = serial();
    let db = Database::open("memory://join_plan_memory_limit").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE q (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO p SELECT g.value, g.value % 100 FROM generate_series(1, 12000) g",
        "INSERT INTO q SELECT g.value, g.value % 100 FROM generate_series(1, 12000) g",
        "ANALYZE p",
        "ANALYZE q",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let plan = |db: &Database| -> String {
        db.query("EXPLAIN SELECT p.id, q.id FROM p JOIN q ON q.k = p.k", ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let learner = global_workload_learner();
    let before = learner.config();
    let free = plan(&db);
    learner.set_config(WorkloadConfig {
        memory_limit_mb: 1,
        ..before.clone()
    });
    let limited = plan(&db);
    learner.set_config(before);
    assert!(free.contains("Hash Join"), "{free}");
    assert!(limited.contains("Nested Loop"), "{limited}");
}
