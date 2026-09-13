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

//! The rest of an index join's ON clause is compiled once per plan, not
//! once per execution, and stays correct across parameters and DDL.
//!
//! Under `test-filedb` a memory DSN opens a file database, whose join
//! reads through the volume path and allocates differently.
#![cfg(not(feature = "test-filedb"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use stoolap::Database;

static ALLOCS: AtomicU64 = AtomicU64::new(0);

/// The counter is process-wide, so the tests in this file never overlap
static SERIAL: Mutex<()> = Mutex::new(());

thread_local! {
    /// Only the measuring thread counts, so a background thread that wakes
    /// while a loaded machine stretches the loop does not add to it
    static COUNTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const SELF_JOIN: &str = "SELECT u1.id, u2.id, u1.age FROM users u1 INNER JOIN users u2 ON u1.age = u2.age AND u1.id < u2.id LIMIT 100";

fn users(name: &str, rows: i64) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_users_age ON users(age)", ())
        .unwrap();
    let insert = db.prepare("INSERT INTO users VALUES (?, ?, ?)").unwrap();
    for id in 1..=rows {
        insert.execute((id, "u", 18 + id % 60)).unwrap();
    }
    db
}

fn pairs(db: &Database, sql: &str, params: (i64,)) -> Vec<(i64, i64, i64)> {
    let stmt = db.prepare(sql).unwrap();
    let rows = if sql.contains('?') {
        stmt.query(params).unwrap()
    } else {
        stmt.query(()).unwrap()
    };
    rows.map(|r| {
        let r = r.unwrap();
        (
            r.get::<i64>(0).unwrap(),
            r.get::<i64>(1).unwrap(),
            r.get::<i64>(2).unwrap(),
        )
    })
    .collect()
}

#[test]
fn the_rest_of_the_on_clause_is_compiled_once_per_plan() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = users("join_residual_allocs", 2000);
    let stmt = db.prepare(SELF_JOIN).unwrap();
    let run = || {
        let mut n = 0;
        for row in stmt.query(()).unwrap() {
            row.unwrap();
            n += 1;
        }
        assert_eq!(n, 100);
    };
    for _ in 0..20 {
        run();
    }
    let reps = 200u64;
    let before = ALLOCS.load(Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    for _ in 0..reps {
        run();
    }
    COUNTING.with(|c| c.set(false));
    let per_query = (ALLOCS.load(Ordering::Relaxed) - before) as f64 / reps as f64;
    eprintln!("allocations per self join execution: {per_query:.2}");
    // The rest of the ON clause was compiled again on every execution,
    // lowercasing every column name of both tables into fresh maps
    assert!(
        per_query <= 250.0,
        "a prepared self join allocates {per_query:.1} times per execution"
    );
}

#[test]
fn the_cached_rest_of_the_on_clause_still_filters_every_pair() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = users("join_residual_filters", 500);
    for _ in 0..3 {
        let rows = pairs(&db, SELF_JOIN, (0,));
        assert_eq!(rows.len(), 100);
        for (a, b, age) in rows {
            assert!(a < b, "pair ({a}, {b}) breaks u1.id < u2.id");
            let age_b: i64 = db
                .query_one("SELECT age FROM users WHERE id = ?", (b,))
                .unwrap();
            assert_eq!(age, age_b, "pair ({a}, {b}) has different ages");
        }
    }
}

#[test]
fn a_parameter_in_the_rest_of_the_on_clause_is_read_on_every_execution() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = users("join_residual_params", 500);
    let sql = "SELECT u1.id, u2.id, u1.age FROM users u1 INNER JOIN users u2 ON u1.age = u2.age AND u1.id < ? ORDER BY u1.id, u2.id";
    let stmt = db.prepare(sql).unwrap();
    let count = |bound: i64| stmt.query((bound,)).unwrap().count();
    let five = count(5);
    let three = count(3);
    let five_again = count(5);
    assert!(three < five, "bound 3 gave {three}, bound 5 gave {five}");
    assert_eq!(five, five_again);
}

#[test]
fn the_cached_rest_of_the_on_clause_follows_a_schema_change() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = users("join_residual_ddl", 300);
    let stmt = db.prepare(SELF_JOIN).unwrap();
    for _ in 0..2 {
        assert_eq!(stmt.query(()).unwrap().count(), 100);
    }
    // The same table with its columns in another order: a program compiled
    // against the old positions would read the wrong columns
    db.execute("DROP TABLE users", ()).unwrap();
    db.execute(
        "CREATE TABLE users (age INTEGER NOT NULL, name TEXT NOT NULL, id INTEGER PRIMARY KEY)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_users_age ON users(age)", ())
        .unwrap();
    let insert = db.prepare("INSERT INTO users VALUES (?, ?, ?)").unwrap();
    for id in 1..=300i64 {
        insert.execute((18 + id % 60, "u", id)).unwrap();
    }
    let rows: Vec<(i64, i64, i64)> = stmt
        .query(())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<i64>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();
    assert_eq!(rows.len(), 100);
    for (a, b, age) in rows {
        assert!(
            a < b,
            "pair ({a}, {b}) breaks u1.id < u2.id after the schema change"
        );
        assert!((18..78).contains(&age), "age column misread as {age}");
    }
}

const NESTED_JOIN: &str = "SELECT a.id, c.v FROM a JOIN (SELECT * FROM b) b ON a.id = b.a_id JOIN c ON a.id = c.id AND a.v = c.v";

/// An index on the nested join's key flips that join's output between
/// `[b.*, a.*]` and `[a.*, b.*]` without a schema change; the enclosing
/// join's residual reads `a.v` by position, so it must follow
fn nested_join_follows_an_index_change(name: &str, indexed_first: bool) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, a_id INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO a VALUES (1, 10)", ()).unwrap();
    db.execute("INSERT INTO b VALUES (2, 1)", ()).unwrap();
    db.execute("INSERT INTO c VALUES (1, 10)", ()).unwrap();
    if indexed_first {
        db.execute("CREATE INDEX idx_b_a ON b(a_id)", ()).unwrap();
    }
    let stmt = db.prepare(NESTED_JOIN).unwrap();
    let read = || {
        stmt.query(())
            .unwrap()
            .map(|r| {
                let r = r.unwrap();
                (r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap())
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(read(), vec![(1, 10)]);
    assert_eq!(read(), vec![(1, 10)]);
    if indexed_first {
        db.execute("DROP INDEX idx_b_a ON b", ()).unwrap();
    } else {
        db.execute("CREATE INDEX idx_b_a ON b(a_id)", ()).unwrap();
    }
    assert_eq!(
        read(),
        vec![(1, 10)],
        "the residual kept reading the outer columns at their old positions"
    );
}

#[test]
fn the_cached_rest_of_the_on_clause_follows_an_index_created_under_a_nested_join() {
    nested_join_follows_an_index_change("join_residual_nested_create_index", false);
}

#[test]
fn the_cached_rest_of_the_on_clause_follows_an_index_dropped_under_a_nested_join() {
    nested_join_follows_an_index_change("join_residual_nested_drop_index", true);
}

#[test]
fn a_plan_run_with_an_outer_row_keeps_no_program() {
    use stoolap::common::CompactArc;
    use stoolap::core::Value;
    use stoolap::executor::{context::ExecutionContext, Executor};

    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_residual_outer_row").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO a VALUES (1, 1)", ()).unwrap();
    db.execute("INSERT INTO b VALUES (1, 10)", ()).unwrap();
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    // The outer row's value is written into the WHERE before the join
    // splits it, so a program kept from one execution would carry it
    let sql = "SELECT a.id FROM a JOIN b ON a.k = b.id WHERE b.v = parent.v";
    let read = |value: i64| {
        let ctx = ExecutionContext::new().with_outer_row(
            [("parent.v".into(), Value::Integer(value))]
                .into_iter()
                .collect(),
            CompactArc::new(vec!["parent.v".to_string()]),
        );
        let mut rows = executor.execute_with_context(sql, &ctx).unwrap();
        let mut ids = Vec::new();
        while rows.next() {
            ids.push(rows.row()[0].clone());
        }
        assert!(rows.last_error().is_none());
        ids
    };
    assert_eq!(read(10), vec![Value::Integer(1)]);
    assert_eq!(read(20), Vec::<Value>::new());
    assert_eq!(read(10), vec![Value::Integer(1)]);
}
