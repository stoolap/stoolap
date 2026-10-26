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

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(not(feature = "jemalloc"), not(feature = "test-filedb")))]

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
    use stoolap::executor::utils::qualified;
    use stoolap::executor::{context::ExecutionContext, Executor};
    use stoolap::parser::ast::{Expression, ParamSlots, Statement};

    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_residual_outer_row").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE parent (v INTEGER)", ()).unwrap();
    db.execute("INSERT INTO a VALUES (1, 1)", ()).unwrap();
    db.execute("INSERT INTO b VALUES (1, 10)", ()).unwrap();
    // The correlated block of a statement that parses, kept as a plan of its own
    let outer = stoolap::parser::parse_sql(
        "SELECT parent.v FROM parent WHERE EXISTS \
         (SELECT a.id FROM a JOIN b ON a.k = b.id WHERE a.id + b.v = parent.v)",
    )
    .unwrap();
    let Statement::Select(outer) = &outer[0] else {
        panic!("not a SELECT");
    };
    let Some(Expression::Exists(exists)) = outer.where_clause.as_deref() else {
        panic!("not an EXISTS");
    };
    let block = Statement::Select((*exists.subquery).clone());
    let executor = Executor::new(std::sync::Arc::clone(db.engine()));
    let plan = executor.query_cache().put(
        "join residual outer row block",
        std::sync::Arc::new(block),
        ParamSlots::default(),
    );
    // The outer row's value is written into the WHERE before the join
    // splits it, so a program kept from one execution would carry it
    let read = |value: i64| {
        let ctx = ExecutionContext::new().with_outer_row(
            [(qualified("parent", "v").into(), Value::Integer(value))]
                .into_iter()
                .collect(),
            CompactArc::new(vec![qualified("parent", "v")]),
        );
        let mut rows = executor.execute_with_cached_plan(&plan, &ctx).unwrap();
        let mut ids = Vec::new();
        while rows.next() {
            ids.push(rows.row()[0].clone());
        }
        assert!(rows.last_error().is_none());
        ids
    };
    assert_eq!(read(11), vec![Value::Integer(1)]);
    assert_eq!(read(12), Vec::<Value>::new());
    assert_eq!(read(11), vec![Value::Integer(1)]);
}

/// The allocations of one more run of `sql` on `db`, prepared and warmed up
fn allocations_per_run(db: &Database, sql: &str) -> f64 {
    let stmt = db.prepare(sql).unwrap();
    let run = || {
        for row in stmt.query(()).unwrap() {
            row.unwrap();
        }
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
    (ALLOCS.load(Ordering::Relaxed) - before) as f64 / reps as f64
}

#[test]
fn a_hash_join_reads_any_number_of_held_residuals_without_allocating() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_residual_three_conditions").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, w INTEGER)",
        "INSERT INTO l SELECT g.value, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value, g.value % 5 FROM generate_series(1, 50) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // Every condition holds for every pair, so the joins return the same
    // rows, and each condition costs what the one before it did
    let with = |conditions: &[&str]| {
        allocations_per_run(
            &db,
            &format!(
                "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND {}",
                conditions.join(" AND ")
            ),
        )
    };
    let two = with(&["l.id < 100", "r.id < 100"]);
    let three = with(&["l.id < 100", "r.id < 100", "l.g < 100"]);
    let four = with(&["l.id < 100", "r.id < 100", "l.g < 100", "r.w < 100"]);
    assert_eq!(
        three - two,
        four - three,
        "a third held residual costs {} allocations per run, a fourth {}",
        three - two,
        four - three
    );
}

#[test]
fn a_join_without_a_plan_builds_no_shared_list() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_residual_without_a_plan").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, w INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY)",
        "INSERT INTO l SELECT g.value, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value, g.value % 5 FROM generate_series(1, 50) g",
        "INSERT INTO s SELECT g.value FROM generate_series(1, 50) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // The inner join of a nested join runs without the statement's plan
    let with = |conditions: &[&str]| {
        let on: String = conditions.iter().map(|c| format!(" AND {c}")).collect();
        allocations_per_run(
            &db,
            &format!(
                "SELECT l.id, r.id, s.id FROM l JOIN r ON l.g = r.w{on} JOIN s ON s.id = r.id"
            ),
        )
    };
    let zero = with(&[]);
    let one = with(&["l.id < 100"]);
    let two = with(&["l.id < 100", "r.id < 100"]);
    let three = with(&["l.id < 100", "r.id < 100", "l.g < 100"]);
    let four = with(&["l.id < 100", "r.id < 100", "l.g < 100", "r.w < 100"]);
    assert_eq!(
        (one - zero) - (two - one),
        2.0,
        "a first residual costs {} allocations per run, a second {}: beyond its \
         program, only the analysis's condition list and the filter list should",
        one - zero,
        two - one
    );
    assert_eq!(
        (two - one, three - two),
        (four - three, four - three),
        "a second, third and fourth residual cost {}, {} and {} allocations per run",
        two - one,
        three - two,
        four - three
    );
}

#[test]
fn a_held_join_analysis_copies_no_condition_per_run() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://join_residual_held_analysis").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, w INTEGER)",
        "INSERT INTO l SELECT g.value, (g.value - 1) / 10 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value, g.value % 5 FROM generate_series(1, 50) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let with = |conditions: &[&str]| {
        allocations_per_run(
            &db,
            &format!(
                "SELECT l.id, r.id FROM l JOIN r ON l.g = r.w AND {}",
                conditions.join(" AND ")
            ),
        )
    };
    let one = with(&["l.id < 100"]);
    let four = with(&["l.id < 100", "r.id < 100", "l.g < 100", "r.w < 100"]);
    assert_eq!(
        one, four,
        "a join with one residual condition costs {one} allocations per run, with four {four}"
    );
}

#[test]
fn a_joined_where_it_cannot_bind_costs_no_allocation_per_joined_row() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let open = |rows: i64| {
        let db = Database::open(&format!("memory://join_residual_unbound_where_{rows}")).unwrap();
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)".to_string(),
            "CREATE TABLE u (id INTEGER PRIMARY KEY, b INTEGER)".to_string(),
            "CREATE TABLE v (id INTEGER PRIMARY KEY, z INTEGER)".to_string(),
            "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)".to_string(),
            format!("INSERT INTO t SELECT g.value, g.value % 7 FROM generate_series(1, {rows}) g"),
            format!("INSERT INTO u SELECT g.value, g.value % 5 FROM generate_series(1, {rows}) g"),
            "INSERT INTO v VALUES (1, 0), (2, 3)".to_string(),
            "INSERT INTO w VALUES (1, 1, 1), (2, 2, 2), (3, 3, 3)".to_string(),
        ] {
            db.execute(&sql, ()).unwrap();
        }
        db
    };
    let (small, large) = (open(50), open(200));
    let cost = |value: &str| {
        let sql = format!(
            "SELECT t.id FROM t JOIN u ON u.id = t.id \
             WHERE t.a + u.b > {value} AND (t.a, u.b) IN (SELECT x, y FROM w)"
        );
        allocations_per_run(&large, &sql) - allocations_per_run(&small, &sql)
    };
    let literal = cost("0");
    let scalar = cost("(SELECT MIN(z) FROM v)");
    assert_eq!(
        literal, scalar,
        "150 more joined rows cost {literal} more allocations per run with a literal, \
         {scalar} with a scalar subquery beside a row IN"
    );
}
