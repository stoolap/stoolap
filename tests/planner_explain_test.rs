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

//! EXPLAIN shows the access a statement runs by: the plan it prints is the
//! one the run takes

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::test_failpoints as fp;
use stoolap::Database;

fn table(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, balance FLOAT, active BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_name ON t(name)", ()).unwrap();
    db.execute("CREATE INDEX t_active ON t(active)", ())
        .unwrap();
    for id in 1..=50i64 {
        db.execute(
            "INSERT INTO t VALUES ($1, $2, $3, $4)",
            (id, format!("n{}", id % 7), id as f64 * 10.0, id % 2 == 0),
        )
        .unwrap();
    }
    db
}

/// The access EXPLAIN shows: the first word of its access step
fn explained(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .find(|word| matches!(word.as_str(), "PkGet" | "IndexScan" | "SeqScan"))
        .unwrap_or_default()
}

/// What running `sql` did: read its row by the primary key plan, and how
/// many ids the hot indexes gave
fn ran(db: &Database, sql: &str) -> (bool, usize) {
    let keyed = Rc::new(Cell::new(false));
    let hook = Rc::clone(&keyed);
    fp::after_pk_lookup_resolved(move || hook.set(true));
    let before = fp::hot_index_ids_read();
    let rows = db.query(sql, ()).unwrap().count();
    assert!(rows > 0, "{sql} keeps no row");
    // A hook a run did not fire must not fire for the next statement
    fp::after_pk_lookup_resolved(|| {});
    (keyed.get(), fp::hot_index_ids_read() - before)
}

#[test]
fn explain_shows_the_access_the_run_takes() {
    let db = table("planner_explain_access");
    for sql in [
        "SELECT * FROM t WHERE id = 7",
        "SELECT name FROM t WHERE id = 7 AND balance > 0",
        "SELECT name FROM t WHERE name = 'n3'",
        "SELECT * FROM t WHERE name = 'n3' AND balance > 0 LIMIT 2",
        "SELECT * FROM t WHERE balance > 10",
        "SELECT * FROM t WHERE balance > 10 LIMIT 3",
        "SELECT * FROM t WHERE active = true",
    ] {
        let access = explained(&db, sql);
        // Twice: the first run plans, the second runs the plan it kept
        ran(&db, sql);
        let (keyed, index_ids) = ran(&db, sql);
        match access.as_str() {
            "PkGet" => assert!(keyed && index_ids == 0, "{sql}: {access}"),
            "IndexScan" => assert!(!keyed && index_ids > 0, "{sql}: {access}"),
            "SeqScan" => assert!(!keyed && index_ids == 0, "{sql}: {access}"),
            other => panic!("{sql}: no planned access in EXPLAIN, {other}"),
        }
    }
}

#[test]
fn explain_shows_the_source_a_global_aggregate_reads() {
    let db = table("planner_explain_global");
    for (sql, plan, path) in [
        (
            "SELECT COUNT(*) FROM t",
            vec!["Aggregate COUNT(*)", "  Stats t", "  else SeqScan t"],
            "stats",
        ),
        (
            "SELECT COUNT(DISTINCT name) AS n, SUM(balance) FROM t",
            vec!["Aggregate n, SUM(balance)", "  Stats t", "  else SeqScan t"],
            "stats",
        ),
        (
            "SELECT MAX(id) FROM t",
            vec![
                "Aggregate MAX(id)",
                "  IndexBound t (id)",
                "  else Stats t",
                "  else SeqScan t",
            ],
            "index_bound",
        ),
        // A hash index keeps no order to give a bound
        (
            "SELECT MIN(name) FROM t",
            vec!["Aggregate MIN(name)", "  Stats t", "  else SeqScan t"],
            "stats",
        ),
    ] {
        let lines: Vec<String> = db
            .query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect();
        assert_eq!(lines, plan, "{sql}");
        // The run that plans, then a run of the plan
        for run in ["planning", "planned"] {
            fp::reset_aggregate_paths();
            db.query(sql, ()).unwrap().for_each(drop);
            assert_eq!(fp::aggregate_paths_taken(), [path], "{sql}, the {run} run");
        }
    }
}

#[test]
fn an_index_created_after_a_plan_is_read_by_its_next_run() {
    let db = table("planner_explain_new_index");
    let sql = "SELECT * FROM t WHERE balance > 400";
    assert_eq!(explained(&db, sql), "SeqScan");
    ran(&db, sql);
    assert_eq!(ran(&db, sql).1, 0, "a scan read an index");

    db.execute("CREATE INDEX t_balance ON t(balance)", ())
        .unwrap();
    assert_eq!(explained(&db, sql), "IndexScan");
    assert!(ran(&db, sql).1 > 0, "the plan kept from before the index");

    db.execute("DROP INDEX t_balance ON t", ()).unwrap();
    assert_eq!(explained(&db, sql), "SeqScan");
    assert_eq!(ran(&db, sql).1, 0, "the plan kept the dropped index");
}
