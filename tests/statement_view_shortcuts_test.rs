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

//! A read committed statement's shortcuts, over counts, bounds, indexes and
//! joins, answer for its view when a commit lands after the view was taken

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

use stoolap::Database;

const QUERIES: &[&str] = &[
    "SELECT COUNT(*) FROM t",
    "SELECT SUM(v) FROM t",
    "SELECT MIN(v) FROM t",
    "SELECT MAX(v) FROM t",
    "SELECT MIN(k) FROM t",
    "SELECT MAX(k) FROM t",
    "SELECT COUNT(DISTINCT k), COUNT(*) FROM t",
    "SELECT DISTINCT k FROM t ORDER BY k",
    "SELECT k, COUNT(*), MAX(v) FROM t GROUP BY k ORDER BY k",
    "SELECT COUNT(*), SUM(v) FROM t WHERE v >= 20",
    "SELECT id FROM t WHERE k = 2 ORDER BY id",
    "SELECT id FROM t WHERE k = 9 ORDER BY id",
    "SELECT id FROM t WHERE k IN (2, 3) ORDER BY id",
    "SELECT id FROM t WHERE k IN (2, 3) AND v + 0 > 0 ORDER BY id",
    "SELECT id FROM t WHERE k IN (2, 3) AND ABS(v) > 0",
    "SELECT id, v FROM t WHERE k IN (2, 3) AND (v > 0 OR id < 0)",
    "SELECT id FROM t WHERE k IN (SELECT x FROM u) ORDER BY id",
    "SELECT id, k FROM t ORDER BY k, id LIMIT 3",
    "SELECT id, k FROM t ORDER BY k DESC, id LIMIT 3",
    "SELECT id FROM t ORDER BY id DESC LIMIT 2",
    "SELECT id FROM t ORDER BY k LIMIT 3",
    "SELECT id FROM t ORDER BY k DESC LIMIT 2",
    "SELECT id, v FROM t WHERE k = 2 ORDER BY v DESC LIMIT 2",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY k ORDER BY id) FROM t ORDER BY id",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY k ORDER BY id) FROM t LIMIT 10",
    "SELECT t.id, u.x FROM t JOIN u ON u.x = t.k ORDER BY t.id",
    "SELECT u.x, t.id FROM u JOIN t ON t.k = u.x ORDER BY u.x, t.id",
    "SELECT u.x, t.id FROM u LEFT JOIN t ON t.k = u.x ORDER BY u.x, t.id LIMIT 10",
    "SELECT u.x, t.id FROM u JOIN t ON t.k = u.x LIMIT 10",
];

const COMMIT: &[&str] = &[
    "UPDATE t SET k = 9, v = 99 WHERE id = 2",
    "DELETE FROM t WHERE id = 4",
    "INSERT INTO t VALUES (6, 0, 5)",
];

fn setup(db: &Database, sealed: bool) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t (k)", ()).unwrap();
    db.execute("CREATE INDEX t_kv ON t (k, v)", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 2, 30), (4, 3, 40), (5, 5, 50)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 2), (2, 3), (3, 9)", ())
        .unwrap();
    if sealed {
        db.engine().seal_for_test().unwrap();
    }
}

/// The rows `sql` gives, sorted: a query without ORDER BY may give them in
/// another order on another path
fn answer(db: &Database, sql: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i)))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    rows.sort_unstable();
    rows
}

/// The answer to `sql` while another handle commits once the statement
/// has taken its view
fn answer_with_commit_inside(db: &Database, sql: &str) -> Vec<String> {
    let committed = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&committed));
    stoolap::test_failpoints::after_statement_view_taken(move || {
        let mut tx = other.begin().unwrap();
        for change in COMMIT {
            tx.execute(change, ()).unwrap();
        }
        tx.commit().unwrap();
        flag.set(true);
    });
    let seen = answer(db, sql);
    assert!(committed.get(), "{sql}: the commit did not land inside");
    seen
}

fn check(open: impl Fn(&str) -> Database, sealed: bool) {
    let mut wrong = Vec::new();
    for (n, sql) in QUERIES.iter().enumerate() {
        let before = open(&format!("before_{n}"));
        setup(&before, sealed);
        let expected = answer(&before, sql);
        let during = open(&format!("during_{n}"));
        setup(&during, sealed);
        let seen = answer_with_commit_inside(&during, sql);
        if seen != expected {
            wrong.push(format!("{sql}: {seen:?}, expected {expected:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn open_file(dir: &Path) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap()
}

#[test]
fn hot_shortcuts_answer_for_the_view() {
    check(
        |name| Database::open(&format!("memory://statement_view_shortcuts_{name}")).unwrap(),
        false,
    );
}

#[test]
fn sealed_shortcuts_answer_for_the_view() {
    let dir = tempfile::tempdir().unwrap();
    check(|name| open_file(&dir.path().join(name)), true);
}

#[test]
fn a_commit_between_the_hot_and_the_sealed_read_leaves_the_view_answer() {
    let mut wrong = Vec::new();
    for sql in [
        "SELECT COUNT(*) FROM t",
        "SELECT SUM(v) FROM t",
        "SELECT MIN(v) FROM t",
        "SELECT MAX(v) FROM t",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = open_file(dir.path());
        setup(&db, true);
        let expected = answer(&db, sql);
        let committed = Rc::new(Cell::new(false));
        let (other, flag) = (db.clone(), Rc::clone(&committed));
        stoolap::test_failpoints::between_hot_and_cold_reads(move || {
            other
                .execute("DELETE FROM t WHERE id IN (1, 5)", ())
                .unwrap();
            flag.set(true);
        });
        let seen = answer(&db, sql);
        assert!(committed.get(), "{sql}: the commit did not land inside");
        // A count read without a view drops a crossed answer and takes its
        // view then, after the commit
        let after = answer(&db, sql);
        if seen != expected && !(sql.contains("COUNT") && seen == after) {
            wrong.push(format!("{sql}: {seen:?}, expected {expected:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn a_partition_read_over_sealed_rows_keeps_the_transaction_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    setup(&db, true);
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (6, 2, 60)", ()).unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY k ORDER BY id) FROM t ORDER BY id"
        ),
        [1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY k ORDER BY id) FROM t LIMIT 10"
        )
        .len(),
        6
    );
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn an_indexed_in_subquery_sees_the_transaction_writes() {
    let db = Database::open("memory://statement_view_shortcuts_in_subquery_tx").unwrap();
    setup(&db, false);
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (6, 9, 60)", ()).unwrap();
    db.execute("UPDATE t SET k = 8 WHERE id = 2", ()).unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE k IN (SELECT x FROM u) ORDER BY id"
        ),
        [3, 4, 6]
    );
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn a_cached_distinct_count_its_index_cannot_answer_takes_the_regular_path() {
    let sql = "SELECT COUNT(DISTINCT k) FROM t";
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open_file(dir.path());
        setup(&db, sealed);
        // The first run compiles the count the later runs take
        let before = answer(&db, sql);
        let mut seen = None;
        // A publishing commit keeps the count from the quiet read: it is
        // read through a view
        while_publishing(&db, "UPDATE t SET k = 9, v = 99 WHERE id = 2", || {
            seen = Some(pk_answer(&db, sql));
        });
        assert_eq!(
            seen,
            Some((before, true)),
            "sealed={sealed}: the count beside a publishing commit"
        );
    }
}

#[test]
fn a_result_read_through_a_view_that_misses_a_commit_is_not_cached_for_later_statements() {
    let mut wrong = Vec::new();
    for sql in [
        "SELECT * FROM t WHERE v > 15",
        "SELECT id FROM u WHERE x IN (SELECT k FROM t WHERE v > 15) ORDER BY id",
        "SELECT id FROM u WHERE EXISTS (SELECT 1 FROM t WHERE t.k = u.x) ORDER BY id",
        "SELECT id FROM u WHERE x < (SELECT MAX(k) FROM t WHERE v > 15) ORDER BY id",
    ] {
        let db = Database::open("memory://statement_view_shortcuts_cache_window").unwrap();
        db.execute("DROP TABLE IF EXISTS t", ()).unwrap();
        db.execute("DROP TABLE IF EXISTS u", ()).unwrap();
        setup(&db, false);
        answer_with_commit_inside(&db, sql);
        let after = answer(&db, sql);
        let fresh = Database::open("memory://statement_view_shortcuts_cache_fresh").unwrap();
        fresh.execute("DROP TABLE IF EXISTS t", ()).unwrap();
        fresh.execute("DROP TABLE IF EXISTS u", ()).unwrap();
        setup(&fresh, false);
        let mut tx = fresh.begin().unwrap();
        for change in COMMIT {
            tx.execute(change, ()).unwrap();
        }
        tx.commit().unwrap();
        let expected = answer(&fresh, sql);
        if after != expected {
            wrong.push(format!("{sql}: {after:?}, expected {expected:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `during` runs on this thread while `sql` commits on another handle,
/// stopped after it published and before it is visible
fn while_publishing(db: &Database, sql: &'static str, during: impl FnOnce()) {
    let watchdog = std::time::Duration::from_secs(15);
    let (published_tx, published_rx) = std::sync::mpsc::channel();
    let (finish_tx, finish_rx) = std::sync::mpsc::channel::<()>();
    let writer = db.clone();
    let commit = std::thread::spawn(move || {
        let mut tx = writer.begin().unwrap();
        tx.execute(sql, ()).unwrap();
        stoolap::test_failpoints::before_commit_visible(move || {
            published_tx.send(()).unwrap();
            finish_rx.recv_timeout(watchdog).unwrap();
        });
        tx.commit().unwrap();
    });
    published_rx.recv_timeout(watchdog).unwrap();
    during();
    finish_tx.send(()).unwrap();
    commit.join().unwrap();
}

/// The answer to `sql`, and whether the statement took a view for it
fn pk_answer(db: &Database, sql: &str) -> (Vec<String>, bool) {
    let viewed = Rc::new(Cell::new(false));
    let flag = Rc::clone(&viewed);
    stoolap::test_failpoints::after_statement_view_taken(move || flag.set(true));
    let seen = answer(db, sql);
    // A hook the statement did not run must not reach the next one
    stoolap::test_failpoints::after_statement_view_taken(|| {});
    (seen, viewed.get())
}

#[test]
fn a_pk_read_beside_another_rows_publishing_commit_answers_without_a_view() {
    let db = Database::open("memory://statement_view_shortcuts_pk_other_row").unwrap();
    setup(&db, false);
    while_publishing(&db, "UPDATE t SET v = 99 WHERE id = 1", || {
        assert_eq!(
            pk_answer(&db, "SELECT v FROM t WHERE id = 3"),
            (vec!["Some(Integer(30))".to_string()], false),
            "row 3 while a commit to row 1 publishes"
        );
    });
}

#[test]
fn a_pk_read_beside_its_own_rows_publishing_commit_sees_the_state_before_it() {
    let mut wrong = Vec::new();
    for (n, (sql, read, expected)) in [
        (
            "UPDATE t SET v = 99 WHERE id = 2",
            2,
            vec!["Some(Integer(20))"],
        ),
        ("DELETE FROM t WHERE id = 2", 2, vec!["Some(Integer(20))"]),
        ("INSERT INTO t VALUES (9, 9, 90)", 9, vec![]),
    ]
    .into_iter()
    .enumerate()
    {
        let db = Database::open(&format!("memory://statement_view_shortcuts_pk_own_{n}")).unwrap();
        setup(&db, false);
        while_publishing(&db, sql, || {
            let (seen, _) = pk_answer(&db, &format!("SELECT v FROM t WHERE id = {read}"));
            if seen != expected {
                wrong.push(format!(
                    "{sql}: row {read} read {seen:?}, expected {expected:?}"
                ));
            }
        });
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_pk_read_of_a_deleted_row_beside_a_publishing_commit_finds_nothing() {
    let db = Database::open("memory://statement_view_shortcuts_pk_deleted").unwrap();
    setup(&db, false);
    db.execute("DELETE FROM t WHERE id = 4", ()).unwrap();
    while_publishing(&db, "UPDATE t SET v = 99 WHERE id = 1", || {
        assert_eq!(
            pk_answer(&db, "SELECT v FROM t WHERE id = 4").0,
            Vec::<String>::new(),
            "a deleted row while a commit to row 1 publishes"
        );
    });
}

/// Row 2's value read by a parsed statement on its own executor, while
/// another handle commits once the statement has taken its view
fn pk_of_a_parsed_statement_with_commit_inside(
    db: &Database,
    executor: &stoolap::Executor,
) -> Vec<String> {
    let committed = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&committed));
    stoolap::test_failpoints::after_statement_view_taken(move || {
        other
            .execute("UPDATE t SET v = 99 WHERE id = 2", ())
            .unwrap();
        flag.set(true);
    });
    let statement = stoolap::parser::parse_sql("SELECT v FROM t WHERE id = 2")
        .unwrap()
        .remove(0);
    let mut result = executor
        .execute_statement(&statement, &stoolap::ExecutionContext::new())
        .unwrap();
    let mut seen = Vec::new();
    while result.next() {
        seen.push(format!("{:?}", result.row().get(0)));
    }
    assert!(committed.get(), "the commit did not land inside");
    seen
}

#[test]
fn a_pk_lookup_reads_through_the_view_its_statement_took() {
    for in_txn in [false, true] {
        let db = Database::open(&format!(
            "memory://statement_view_shortcuts_pk_taken_{in_txn}"
        ))
        .unwrap();
        setup(&db, false);
        let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
        if in_txn {
            executor.execute("BEGIN").unwrap();
        }
        assert_eq!(
            pk_of_a_parsed_statement_with_commit_inside(&db, &executor),
            ["Some(Integer(20))"],
            "in a transaction: {in_txn}"
        );
    }
}

#[test]
fn a_count_without_a_view_drops_an_answer_a_commit_moved_rows_under() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 10)", ()).unwrap();
    db.engine().seal_for_test().unwrap();
    db.execute("INSERT INTO c VALUES (2, 20)", ()).unwrap();
    let sql = "SELECT COUNT(*) FROM c";
    // The first run compiles the count the later runs take
    answer(&db, sql);
    assert_eq!(
        pk_answer(&db, sql),
        (vec!["Some(Integer(2))".to_string()], false),
        "a quiet count takes no view"
    );
    let committed = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&committed));
    stoolap::test_failpoints::between_hot_and_cold_reads(move || {
        other
            .execute("UPDATE c SET v = 11 WHERE id = 1", ())
            .unwrap();
        flag.set(true);
    });
    let seen = pk_answer(&db, sql);
    assert!(committed.get(), "the commit did not land between the reads");
    assert_eq!(
        seen,
        (vec!["Some(Integer(2))".to_string()], true),
        "a count a commit moved a sealed row under is read again through a view"
    );
}

#[test]
fn a_distinct_count_without_a_view_drops_an_answer_a_commit_moved_rows_under() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX d_k ON d (k)", ()).unwrap();
    db.execute("INSERT INTO d VALUES (1, 1)", ()).unwrap();
    db.engine().seal_for_test().unwrap();
    db.execute("INSERT INTO d VALUES (2, 2)", ()).unwrap();
    let sql = "SELECT COUNT(DISTINCT k) FROM d";
    // The first run compiles the count the later runs take
    answer(&db, sql);
    assert_eq!(
        pk_answer(&db, sql),
        (vec!["Some(Integer(2))".to_string()], false),
        "a quiet distinct count takes no view"
    );
    let committed = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&committed));
    stoolap::test_failpoints::between_hot_and_cold_reads(move || {
        other
            .execute("UPDATE d SET k = 3 WHERE id = 1", ())
            .unwrap();
        flag.set(true);
    });
    let seen = pk_answer(&db, sql);
    assert!(committed.get(), "the commit did not land between the reads");
    assert_eq!(
        seen,
        (vec!["Some(Integer(2))".to_string()], true),
        "a distinct count a commit moved a sealed row under is read again through a view"
    );
}

#[test]
fn a_count_reads_through_a_view_its_context_already_holds() {
    let mut wrong = Vec::new();
    for (n, sql) in ["SELECT COUNT(*) FROM g", "SELECT COUNT(DISTINCT k) FROM g"]
        .into_iter()
        .enumerate()
    {
        let db =
            Database::open(&format!("memory://statement_view_shortcuts_count_held_{n}")).unwrap();
        db.execute("CREATE TABLE g (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute("CREATE INDEX g_k ON g (k)", ()).unwrap();
        db.execute("INSERT INTO g VALUES (1, 1)", ()).unwrap();
        let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
        let ctx = stoolap::ExecutionContext::new()
            .with_statement_view(db.engine().registry().capture_view());
        db.execute("INSERT INTO g VALUES (2, 2)", ()).unwrap();
        for run in ["first", "cached"] {
            let mut result = executor.execute_with_context(sql, &ctx).unwrap();
            assert!(result.next(), "{sql}, {run} run: a row");
            let seen = format!("{:?}", result.row().get(0));
            if seen != "Some(Integer(1))" {
                wrong.push(format!("{sql}, {run} run: {seen}, the view sees 1"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn each_statement_of_a_program_takes_its_own_view() {
    let db = Database::open("memory://statement_view_shortcuts_program_views").unwrap();
    setup(&db, false);
    let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
    let committed = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&committed));
    stoolap::test_failpoints::after_statement_view_taken(move || {
        other
            .execute("UPDATE t SET v = 99 WHERE id = 2", ())
            .unwrap();
        flag.set(true);
    });
    let mut result = executor
        .execute("SELECT v FROM t WHERE id = 1; SELECT v FROM t WHERE v > 0 AND id = 2")
        .unwrap();
    assert!(
        committed.get(),
        "the commit did not land after the first view"
    );
    assert!(result.next(), "a row");
    assert_eq!(
        format!("{:?}", result.row().get(0)),
        "Some(Integer(99))",
        "the second statement read through the first statement's view"
    );
}

/// The rows `sql` gives and the views its run took
fn viewed_answer(db: &Database, sql: &str) -> (Vec<String>, usize) {
    let taken = stoolap::test_failpoints::statement_views_taken();
    let seen = answer(db, sql);
    (
        seen,
        stoolap::test_failpoints::statement_views_taken() - taken,
    )
}

#[test]
fn a_whole_semantic_cache_answer_takes_no_view() {
    let db = Database::open("memory://statement_view_shortcuts_semantic_hits").unwrap();
    setup(&db, false);
    let cached = "SELECT * FROM t WHERE v > 15";
    let (first, views) = viewed_answer(&db, cached);
    assert_eq!(views, 1, "a miss reads through a view");
    let before = db.semantic_cache_stats().unwrap();
    assert_eq!(viewed_answer(&db, cached), (first, 0), "an exact hit");
    let narrower = answer(&db, "SELECT id, k, v FROM t WHERE v > 35");
    assert_eq!(
        viewed_answer(&db, "SELECT * FROM t WHERE v > 35"),
        (narrower, 0),
        "a subsumption hit"
    );
    let after = db.semantic_cache_stats().unwrap();
    assert_eq!(
        (
            after.exact_hits - before.exact_hits,
            after.subsumption_hits - before.subsumption_hits
        ),
        (1, 1)
    );
}

#[test]
fn a_semantic_cache_miss_is_looked_up_once_and_cached_through_its_view() {
    let db = Database::open("memory://statement_view_shortcuts_semantic_miss").unwrap();
    setup(&db, false);
    let sql = "SELECT * FROM t WHERE v > 25";
    let before = db.semantic_cache_stats().unwrap();
    let (seen, views) = viewed_answer(&db, sql);
    assert_eq!(views, 1, "a miss reads through a view");
    let after = db.semantic_cache_stats().unwrap();
    assert_eq!(after.misses - before.misses, 1, "lookups for one miss");
    assert_eq!(
        viewed_answer(&db, sql),
        (seen, 0),
        "the miss cached its rows"
    );
}

#[test]
fn a_semantic_cache_answer_is_not_given_for_a_view_the_caller_holds() {
    let db = Database::open("memory://statement_view_shortcuts_semantic_given").unwrap();
    setup(&db, false);
    let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
    let ctx =
        stoolap::ExecutionContext::new().with_statement_view(db.engine().registry().capture_view());
    db.execute("INSERT INTO t VALUES (6, 0, 60)", ()).unwrap();
    let sql = "SELECT * FROM t WHERE v > 45";
    for _ in 0..2 {
        let mut result = executor.execute(sql).unwrap();
        while result.next() {}
    }
    let mut result = executor.execute_with_context(sql, &ctx).unwrap();
    let mut rows = 0;
    while result.next() {
        rows += 1;
    }
    assert_eq!(rows, 1, "the rows the given view sees");
}

#[test]
fn a_cached_scan_keeps_what_its_statement_adds_after_it() {
    let db = Database::open("memory://statement_view_shortcuts_semantic_shapes").unwrap();
    setup(&db, false);
    let rows = |sql: &str| db.query(sql, ()).unwrap().count();
    for _ in 0..2 {
        rows("SELECT * FROM t WHERE v > 15");
        rows("SELECT * FROM u WHERE x > 0");
    }
    let mut wrong = Vec::new();
    for (sql, expected) in [
        ("SELECT * FROM t WHERE v > 15 OFFSET 1", 3),
        (
            "SELECT * FROM t WHERE v > 15 UNION ALL SELECT * FROM t WHERE v > 45",
            5,
        ),
        (
            "WITH u AS (SELECT id, v AS x FROM t WHERE v > 45) SELECT * FROM u WHERE x > 0",
            1,
        ),
    ] {
        let seen = rows(sql);
        if seen != expected {
            wrong.push(format!("{sql}: {seen} rows, expected {expected}"));
        }
    }
    let taken = stoolap::test_failpoints::statement_views_taken();
    db.query_with_timeout("SELECT * FROM t WHERE v > 15", (), 60_000)
        .unwrap()
        .count();
    if stoolap::test_failpoints::statement_views_taken() - taken != 1 {
        wrong.push("a timed call left the regular path".to_string());
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_semantic_cache_answer_is_not_given_inside_a_transaction() {
    use stoolap::core::{Row, Value};
    let db = Database::open("memory://statement_view_shortcuts_semantic_txn").unwrap();
    setup(&db, false);
    let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
    let sql = "SELECT * FROM t WHERE v > 45";
    for _ in 0..2 {
        let mut result = executor.execute(sql).unwrap();
        while result.next() {}
    }
    let tx = executor.begin_transaction().unwrap();
    tx.get_table("t")
        .unwrap()
        .insert(Row::from_values(vec![
            Value::Integer(6),
            Value::Integer(0),
            Value::Integer(60),
        ]))
        .unwrap();
    executor.install_transaction(tx);
    let mut result = executor.execute(sql).unwrap();
    let mut rows = 0;
    while result.next() {
        rows += 1;
    }
    executor.take_transaction().unwrap().rollback().unwrap();
    assert_eq!(rows, 2, "the transaction's own row");
}

#[test]
fn a_table_shortcut_without_a_view_drops_an_answer_a_commit_moved_rows_under() {
    let mut wrong = Vec::new();
    for shortcut in ["count", "sum", "row_count"] {
        let dir = tempfile::tempdir().unwrap();
        let db = open_file(dir.path());
        db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO c VALUES (1, 10)", ()).unwrap();
        db.engine().seal_for_test().unwrap();
        db.execute("INSERT INTO c VALUES (2, 20)", ()).unwrap();
        let committed = Rc::new(Cell::new(false));
        let (other, flag) = (db.clone(), Rc::clone(&committed));
        stoolap::test_failpoints::between_hot_and_cold_reads(move || {
            other
                .execute("UPDATE c SET v = 11 WHERE id = 1", ())
                .unwrap();
            flag.set(true);
        });
        // A handle of the storage API reads the newest state, with no view
        let tx = stoolap::storage::traits::Engine::begin_transaction(&**db.engine()).unwrap();
        let table = tx.get_table("c").unwrap();
        let seen = match shortcut {
            "count" => format!("{:?}", table.fast_row_count().unwrap()),
            "sum" => format!("{:?}", table.sum_column(1).unwrap()),
            _ => format!("{:?}", table.row_count().unwrap()),
        };
        if !committed.get() {
            wrong.push(format!(
                "{shortcut}: the commit did not land between the reads"
            ));
        }
        // The right answers are the totals before and after the commit
        let right: &[&str] = match shortcut {
            "count" => &["None", "Some(2)"],
            "sum" => &["None", "Some((30.0, 2))", "Some((31.0, 2))"],
            _ => &["2"],
        };
        if !right.contains(&seen.as_str()) {
            wrong.push(format!("{shortcut}: {seen}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
