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

//! A join plans how it runs once; a run that cannot take the planned index
//! reads each side once all the same

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{after_join_index_chosen, join_side_runs, reset_all};
use stoolap::Database;

// The failpoint hooks are per thread, the counters process-wide in effect
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
