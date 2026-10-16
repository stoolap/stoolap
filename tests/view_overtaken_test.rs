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

//! A truncate or drop that overtakes a statement's view: a SELECT that took
//! the view starts again with a new one, anything else fails and keeps
//! nothing it wrote

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use stoolap::test_failpoints as fp;
use stoolap::{Database, Error, ExecutionContext, Executor};

fn memory(name: &str) -> Database {
    let db = Database::open(&format!("memory://view_overtaken_{name}")).unwrap();
    fill(&db);
    db
}

fn file(dir: &Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    fill(&db);
    db.engine().seal_for_test().unwrap();
    db
}

fn fill(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("CREATE TABLE other (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO other VALUES (1)", ()).unwrap();
}

/// Once this thread's next statement has its view, another handle runs
/// `sqls`; the flag says they ran
fn after_the_view(db: &Database, sqls: &'static [&'static str]) -> Rc<Cell<bool>> {
    let done = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&done));
    fp::after_statement_view_taken(move || {
        for sql in sqls {
            other.execute(sql, ()).unwrap();
        }
        flag.set(true);
    });
    done
}

const TRUNCATE_AND_ADD: &[&str] = &["TRUNCATE TABLE t", "INSERT INTO t VALUES (3, 30)"];
const TRUNCATE_OTHER: &[&str] = &["TRUNCATE TABLE other"];

fn values(rows: stoolap::Result<stoolap::Rows>) -> stoolap::Result<Vec<i64>> {
    let mut out = Vec::new();
    for row in rows? {
        out.push(row?.get::<i64>(0)?);
    }
    Ok(out)
}

#[test]
fn a_select_overtaken_after_its_view_runs_again_with_a_new_one() {
    let dir = tempfile::tempdir().unwrap();
    for (name, db) in [("hot", memory("again_hot")), ("sealed", file(dir.path()))] {
        let ran = after_the_view(&db, TRUNCATE_AND_ADD);
        let taken = fp::statement_views_taken();
        let seen = values(db.query("SELECT v FROM t ORDER BY v", ()));
        assert!(
            ran.get(),
            "{name}: the truncate did not land after the view"
        );
        assert_eq!(
            seen.unwrap(),
            [30],
            "{name}: the table once the truncate is done"
        );
        assert_eq!(
            fp::statement_views_taken() - taken,
            2,
            "{name}: the second run took a view of its own"
        );
    }
}

#[test]
fn a_view_taken_while_a_truncate_runs_waits_it_out_and_runs_again() {
    let db = memory("during");
    let destruction = db.engine().registry().begin_destruction();
    let (waiting_tx, waiting) = std::sync::mpsc::channel();
    let reader = db.clone();
    let read = std::thread::spawn(move || {
        fp::on_destruction_wait(move || waiting_tx.send(()).unwrap());
        let taken = fp::statement_views_taken();
        let seen = values(reader.query("SELECT v FROM t ORDER BY v", ()));
        (seen, fp::statement_views_taken() - taken)
    });
    waiting
        .recv_timeout(Duration::from_secs(15))
        .expect("the read waited for the truncate");
    drop(destruction);
    let (seen, taken) = read.join().unwrap();
    assert_eq!(seen.unwrap(), [10, 20]);
    assert_eq!(taken, 2, "the view taken inside the truncate was not kept");
}

#[test]
fn cached_and_prepared_entries_run_again_alike() {
    let db = memory("entries");
    let sql = "SELECT v FROM t WHERE v > 0 ORDER BY v";
    // The first run caches the plan the second takes
    values(db.query(sql, ())).unwrap();
    after_the_view(&db, TRUNCATE_AND_ADD);
    assert_eq!(values(db.query(sql, ())).unwrap(), [30], "a cached plan");
    let prepared = db.prepare(sql).unwrap();
    after_the_view(&db, &["TRUNCATE TABLE t", "INSERT INTO t VALUES (4, 40)"]);
    assert_eq!(values(prepared.query(())).unwrap(), [40], "a prepared plan");
}

#[test]
fn a_compiled_pk_lookup_in_a_transaction_runs_again() {
    let db = memory("compiled_pk");
    let executor = Executor::new(Arc::clone(db.engine()));
    executor.execute("BEGIN").unwrap();
    let sql = "SELECT v FROM t WHERE id = 1";
    for _ in 0..2 {
        // The first runs compile the lookup the last takes
        let mut result = executor.execute(sql).unwrap();
        while result.next() {}
    }
    let ran = after_the_view(&db, TRUNCATE_AND_ADD);
    let taken = fp::statement_views_taken();
    let mut result = executor.execute(sql).unwrap();
    assert!(ran.get(), "the truncate did not land after the view");
    assert!(!result.next(), "row 1 is gone once the truncate is done");
    assert_eq!(fp::statement_views_taken() - taken, 2);
    executor.execute("COMMIT").unwrap();
}

#[test]
fn a_lazy_result_a_truncate_overtakes_ends_with_the_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = file(dir.path());
    let executor = Executor::new(Arc::clone(db.engine()));
    let mut result = executor.execute("SELECT id FROM t").unwrap();
    assert!(result.next(), "a first row");
    db.execute("TRUNCATE TABLE other", ()).unwrap();
    while result.next() {}
    assert!(
        matches!(result.last_error(), Some(Error::ViewOvertaken)),
        "the end of the rows is not their end"
    );
}

#[test]
fn a_statement_a_truncate_overtakes_keeps_nothing_it_wrote() {
    let mut wrong = Vec::new();
    for (n, sql) in [
        "INSERT INTO kept SELECT id + 100, v FROM t",
        "INSERT INTO kept SELECT id + 100, v FROM t RETURNING id",
        "UPDATE kept SET v = v + (SELECT MAX(v) FROM t)",
    ]
    .into_iter()
    .enumerate()
    {
        let db = memory(&format!("dml_{n}"));
        db.execute("CREATE TABLE kept (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO kept VALUES (1, 1)", ()).unwrap();
        let executor = Executor::new(Arc::clone(db.engine()));
        executor.execute("BEGIN").unwrap();
        executor.execute("INSERT INTO kept VALUES (2, 2)").unwrap();
        let ran = after_the_view(&db, TRUNCATE_OTHER);
        let outcome = executor.execute(sql);
        if !ran.get() {
            wrong.push(format!("{sql}: took no view, so the truncate did not land"));
        }
        match outcome {
            Err(Error::ViewOvertaken) => {}
            other => wrong.push(format!("{sql}: {:?}", other.map(|r| r.rows_affected()))),
        }
        executor.execute("COMMIT").unwrap();
        let rows = values(db.query("SELECT v FROM kept ORDER BY id", ())).unwrap();
        if rows != [1, 2] {
            wrong.push(format!(
                "{sql}: committed {rows:?}, expected the earlier write only"
            ));
        }
    }
    let db = memory("dml_autocommit");
    db.execute("CREATE TABLE kept (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    after_the_view(&db, TRUNCATE_OTHER);
    if !matches!(
        db.execute("INSERT INTO kept SELECT id, v FROM t", ()),
        Err(Error::ViewOvertaken)
    ) {
        wrong.push("an autocommit INSERT ... SELECT went on".to_string());
    }
    let rows = values(db.query("SELECT v FROM kept", ())).unwrap();
    if !rows.is_empty() {
        wrong.push(format!("an autocommit INSERT ... SELECT kept {rows:?}"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn the_runs_of_an_overtaken_select_share_its_first_deadline() {
    let db = memory("deadline");
    // The first view is overtaken 250 ms in; the second run's view takes as
    // long again, past the 400 ms the statement has
    let other = db.clone();
    fp::after_statement_view_taken(move || {
        std::thread::sleep(Duration::from_millis(250));
        other.execute("TRUNCATE TABLE other", ()).unwrap();
        fp::after_statement_view_taken(|| std::thread::sleep(Duration::from_millis(250)));
    });
    let seen = values(db.query_with_timeout("SELECT v FROM t ORDER BY v", (), 400));
    assert!(
        matches!(seen, Err(Error::QueryCancelled)),
        "a later run started its own deadline: {seen:?}"
    );
}

#[test]
fn a_select_calling_a_registered_function_is_not_run_again() {
    let db = memory("registered");
    let registry = stoolap::FunctionRegistry::new();
    registry.register_tvf(
        "MY_SERIES",
        Arc::new(|| Box::new(stoolap::functions::tvf::GenerateSeriesFunction)),
    );
    let executor = Executor::with_function_registry(Arc::clone(db.engine()), Arc::new(registry));
    after_the_view(&db, TRUNCATE_OTHER);
    assert!(matches!(
        executor.execute("SELECT v FROM t ORDER BY v"),
        Err(Error::ViewOvertaken)
    ));
}

#[test]
fn a_view_the_caller_gave_is_not_taken_again() {
    let db = memory("given");
    let executor = Executor::new(Arc::clone(db.engine()));
    let ctx = ExecutionContext::new().with_statement_view(db.engine().registry().capture_view());
    db.execute("TRUNCATE TABLE other", ()).unwrap();
    assert!(matches!(
        executor.execute_with_context("SELECT v FROM t ORDER BY v", &ctx),
        Err(Error::ViewOvertaken)
    ));
}
