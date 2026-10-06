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

//! Column positions are used only against the table instance and layout
//! they were resolved for: a statement compiled before a column change, or
//! a transaction holding rows written before one, runs on the new table or
//! is refused with `SchemaChanged`, and never reads or writes a column
//! through a position that now names another.

#[cfg(feature = "test-failpoints")]
use std::cell::Cell;
#[cfg(feature = "test-failpoints")]
use std::rc::Rc;

use stoolap::core::Error;
use stoolap::Database;

fn row3(db: &Database, sql: &str) -> Vec<(i64, i64, i64)> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap(), r.get(2).unwrap())
        })
        .collect()
}

fn refused(result: &Result<i64, Error>) -> bool {
    matches!(result, Err(Error::SchemaChanged { .. }))
}

fn file_dsn(dir: &tempfile::TempDir) -> String {
    format!("file://{}", dir.path().display())
}

// --- Compiled statements resumed after a column change -------------------

#[cfg(feature = "test-failpoints")]
fn prepared_update_after(dsn: &str, change: &'static [&'static str]) {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100, 1000)", ())
        .unwrap();
    let update = db.prepare("UPDATE t SET a = $1 WHERE id = $2").unwrap();
    update.execute((101, 1)).unwrap();
    update.execute((100, 1)).unwrap();
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_compiled_epoch_checked(move || {
        for sql in change {
            other.execute(sql, ()).unwrap();
        }
        hook_fired.set(true);
    });
    let result = update.execute((555, 1));
    assert!(fired.get(), "the change ran inside the compiled update");
    let after = row3(&db, "SELECT id, a, b FROM t");
    let expected = if refused(&result) { 100 } else { 555 };
    assert_eq!(after, [(1, expected, 1000)], "{result:?}");
}

#[cfg(feature = "test-failpoints")]
const DROP_X: &[&str] = &["ALTER TABLE t DROP COLUMN x"];

#[cfg(feature = "test-failpoints")]
const RECREATE_SWAPPED: &[&str] = &[
    "DROP TABLE t",
    "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, b INTEGER, a INTEGER)",
    "INSERT INTO t VALUES (1, 10, 1000, 100)",
];

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_prepared_update_resumed_after_a_drop_column_writes_its_column() {
    prepared_update_after("memory://compiled_positions_update_drop", DROP_X);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_update_resumed_after_a_drop_column_writes_its_column_on_file() {
    let dir = tempfile::tempdir().unwrap();
    prepared_update_after(&file_dsn(&dir), DROP_X);
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_prepared_update_resumed_on_a_recreated_table_writes_its_column() {
    prepared_update_after(
        "memory://compiled_positions_update_recreate",
        RECREATE_SWAPPED,
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_update_resumed_on_a_recreated_table_writes_its_column_on_file() {
    let dir = tempfile::tempdir().unwrap();
    prepared_update_after(&file_dsn(&dir), RECREATE_SWAPPED);
}

#[cfg(feature = "test-failpoints")]
fn prepared_select_after_drop(dsn: &str) {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100, 1000)", ())
        .unwrap();
    let select = db.prepare("SELECT a FROM t WHERE id = $1").unwrap();
    for _ in 0..2 {
        assert_eq!(select.query((1,)).unwrap().count(), 1);
    }
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_compiled_epoch_checked(move || {
        other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
        hook_fired.set(true);
    });
    let result = select.query((1,));
    assert!(fired.get(), "the drop ran inside the compiled select");
    match result {
        Err(Error::SchemaChanged { .. }) => {}
        Err(e) => panic!("{e:?}"),
        Ok(rows) => {
            let a: Vec<i64> = rows.map(|r| r.unwrap().get(0).unwrap()).collect();
            assert_eq!(a, [100], "the prepared SELECT reads a");
        }
    }
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_prepared_select_resumed_after_a_drop_column_reads_its_column() {
    prepared_select_after_drop("memory://compiled_positions_select_drop");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_select_resumed_after_a_drop_column_reads_its_column_on_file() {
    let dir = tempfile::tempdir().unwrap();
    prepared_select_after_drop(&file_dsn(&dir));
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_select_falling_back_to_cold_reads_its_own_table() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&file_dsn(&dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100, 1000)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let select = db.prepare("SELECT a FROM t WHERE id = $1").unwrap();
    for _ in 0..2 {
        assert_eq!(select.query((1,)).unwrap().count(), 1);
    }
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_pk_hot_rows_fetched(move || {
        for sql in [
            "DROP TABLE t",
            "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, b INTEGER, a INTEGER)",
            "INSERT INTO t VALUES (1, 10, 1000, 100)",
            "PRAGMA CHECKPOINT",
        ] {
            other.execute(sql, ()).unwrap();
        }
        hook_fired.set(true);
    });
    let result = select.query((1,));
    assert!(
        fired.get(),
        "the table was replaced between the hot miss and cold"
    );
    match result {
        Err(Error::SchemaChanged { .. }) => {}
        Err(e) => panic!("{e:?}"),
        Ok(rows) => {
            let a: Vec<i64> = rows.map(|r| r.unwrap().get(0).unwrap()).collect();
            assert_eq!(a, [100], "the prepared SELECT reads a of the new table");
        }
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_select_falling_back_to_cold_reads_its_own_table_across_a_rename() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&file_dsn(&dir)).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        "INSERT INTO t VALUES (1, 10, 100, 1000)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, b INTEGER, a INTEGER)",
        "INSERT INTO u VALUES (1, 10, 2000, 200)",
        "PRAGMA CHECKPOINT",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let select = db.prepare("SELECT a FROM t WHERE id = $1").unwrap();
    for _ in 0..2 {
        assert_eq!(select.query((1,)).unwrap().count(), 1);
    }
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_pk_hot_rows_fetched(move || {
        other.execute("ALTER TABLE t RENAME TO t_old", ()).unwrap();
        other.execute("ALTER TABLE u RENAME TO t", ()).unwrap();
        hook_fired.set(true);
    });
    let result = select.query((1,));
    assert!(fired.get(), "the name moved between the hot miss and cold");
    match result {
        Err(Error::SchemaChanged { .. }) => {}
        Err(e) => panic!("{e:?}"),
        Ok(rows) => {
            let a: Vec<i64> = rows.map(|r| r.unwrap().get(0).unwrap()).collect();
            assert_eq!(
                a,
                [200],
                "the prepared SELECT reads a of the table now named t"
            );
        }
    }
}

#[cfg(feature = "test-failpoints")]
fn prepared_insert_compiled_across_a_drop(dsn: &str) {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    let insert = db.prepare("INSERT INTO t (id, a) VALUES ($1, $2)").unwrap();
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_dml_table_opened(move || {
        other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
        hook_fired.set(true);
    });
    let first = insert.execute((1, 100));
    assert!(
        fired.get(),
        "the drop ran between the table open and the compile"
    );
    insert.execute((2, 200)).unwrap();
    let mut expected = vec![(2, 200, -1)];
    if first.is_ok() {
        expected.insert(0, (1, 100, -1));
    }
    let rows = row3(&db, "SELECT id, a, COALESCE(b, -1) FROM t ORDER BY id");
    assert_eq!(rows, expected, "{first:?}");
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_prepared_insert_compiled_across_a_drop_column_writes_its_columns() {
    prepared_insert_compiled_across_a_drop("memory://compiled_positions_insert_drop");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_insert_compiled_across_a_drop_column_writes_its_columns_on_file() {
    let dir = tempfile::tempdir().unwrap();
    prepared_insert_compiled_across_a_drop(&file_dsn(&dir));
}

#[cfg(feature = "test-failpoints")]
fn join_residual_compiled_across_a_drop(dsn: &str) {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO o VALUES (1, 1), (2, 2)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        "INSERT INTO t VALUES (1, 10, 100, 1000), (2, 20, 200, 100)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let join = db
        .prepare(
            "SELECT o.id, COALESCE(t.id, -1) FROM o LEFT JOIN t ON o.k = t.id AND t.a = 100 \
             ORDER BY o.id",
        )
        .unwrap();
    let pairs = |rows: stoolap::Rows| -> Vec<(i64, i64)> {
        rows.map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap())
        })
        .collect()
    };
    let expected = vec![(1, 1), (2, -1)];
    assert_eq!(pairs(join.query(()).unwrap()), expected);
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_join_inner_opened(move || {
        other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
        hook_fired.set(true);
    });
    let _ = join.query(()).map(pairs);
    assert!(
        fired.get(),
        "the drop ran between the inner open and the residual"
    );
    assert_eq!(pairs(join.query(()).unwrap()), expected);
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_join_residual_compiled_across_a_drop_column_reads_its_columns() {
    join_residual_compiled_across_a_drop("memory://compiled_positions_join_drop");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_join_residual_compiled_across_a_drop_column_reads_its_columns_on_file() {
    let dir = tempfile::tempdir().unwrap();
    join_residual_compiled_across_a_drop(&file_dsn(&dir));
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_update_compiled_across_a_drop_column_writes_its_column() {
    let db = Database::open("memory://compiled_positions_update_compile").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100, 1000)", ())
        .unwrap();
    let update = db.prepare("UPDATE t SET a = $1 WHERE id = $2").unwrap();
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_compile_schema_read(move || {
        other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
        hook_fired.set(true);
    });
    let _ = update.execute((555, 1));
    assert!(fired.get(), "the drop ran inside the compile");
    update.execute((777, 1)).unwrap();
    assert_eq!(row3(&db, "SELECT id, a, b FROM t"), [(1, 777, 1000)]);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_prepared_select_compiled_across_a_drop_column_reads_its_column() {
    let db = Database::open("memory://compiled_positions_select_compile").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100, 1000)", ())
        .unwrap();
    let select = db.prepare("SELECT a FROM t WHERE id = $1").unwrap();
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_compile_schema_read(move || {
        other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
        hook_fired.set(true);
    });
    let _ = select.query((1,)).map(|rows| rows.count());
    assert!(fired.get(), "the drop ran inside the compile");
    let a: Vec<i64> = select
        .query((1,))
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(a, [100]);
}

/// Two handles of one transaction, opened on either side of a column
/// change: the older one may not write rows the newer one reads as its own
#[test]
fn an_older_handle_of_a_transaction_does_not_write_under_a_newer_one() {
    use stoolap::storage::traits::Engine;
    use stoolap::{Row, Value};
    let db = Database::open("memory://compiled_positions_two_handles").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
        (),
    )
    .unwrap();
    let tx = db.engine().begin_transaction().unwrap();
    let mut older = tx.get_table("t").unwrap();
    db.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
    let newer = tx.get_table("t").unwrap();
    let wide = Row::from_values(vec![
        Value::Integer(1),
        Value::Integer(10),
        Value::Integer(100),
    ]);
    match older.insert(wide) {
        Err(Error::SchemaChanged { .. }) => {}
        written => {
            written.unwrap();
            let rows: Vec<Vec<Value>> = newer
                .collect_all_rows(None)
                .unwrap()
                .into_iter()
                .map(|(_, row)| row.iter().cloned().collect())
                .collect();
            assert_eq!(rows, [vec![Value::Integer(1), Value::Integer(100)]]);
        }
    }
}

/// A handle opened inside an update's setter, while the update reads the
/// transaction's own rows, is admitted without waiting for that read
#[test]
fn a_handle_opened_inside_an_update_setter_does_not_wait_for_the_update() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use stoolap::storage::traits::Engine;
        use stoolap::{Row, Value};
        let db = Database::open("memory://compiled_positions_setter_open").unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
            .unwrap();
        let tx = db.engine().begin_transaction().unwrap();
        let mut table = tx.get_table("t").unwrap();
        table
            .insert(Row::from_values(vec![
                Value::Integer(1),
                Value::Integer(10),
            ]))
            .unwrap();
        let updated = table.update_by_row_ids(&[1], &mut |row| {
            tx.get_table("t")?;
            Ok((row, true))
        });
        done.send(updated.map(|_| ())).unwrap();
    });
    let updated = finished
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the update finished");
    updated.unwrap();
}

/// A transaction's first handle on a table takes its schema after the
/// transaction's store is published: an older handle opened through that
/// store meanwhile, then a column drop, leave the first handle newer
#[cfg(feature = "test-failpoints")]
#[test]
fn a_first_handle_newer_than_its_published_store_is_admitted() {
    use std::cell::RefCell;
    use stoolap::storage::traits::{Engine, Table};
    use stoolap::{Row, Value};
    let db = Database::open("memory://compiled_positions_first_handle").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
        (),
    )
    .unwrap();
    let tx = Rc::new(db.engine().begin_transaction().unwrap());
    let older: Rc<RefCell<Option<Box<dyn Table>>>> = Rc::new(RefCell::new(None));
    let (hook_tx, hook_older, other) = (Rc::clone(&tx), Rc::clone(&older), db.clone());
    stoolap::test_failpoints::after_txn_store_published(move || {
        *hook_older.borrow_mut() = Some(hook_tx.get_table("t").unwrap());
        other.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
    });
    let newer = tx.get_table("t").unwrap();
    let mut older = older.borrow_mut().take().expect("the older handle opened");
    let wide = Row::from_values(vec![
        Value::Integer(1),
        Value::Integer(10),
        Value::Integer(100),
    ]);
    match older.insert(wide) {
        Err(Error::SchemaChanged { .. }) => {}
        written => {
            written.unwrap();
            let rows: Vec<Vec<Value>> = newer
                .collect_all_rows(None)
                .unwrap()
                .into_iter()
                .map(|(_, row)| row.iter().cloned().collect())
                .collect();
            assert_eq!(rows, [vec![Value::Integer(1), Value::Integer(100)]]);
        }
    }
}

// --- A transaction's own rows after another session's column change -----

fn local_rows_after_drop(dsn: &str) {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100)", ()).unwrap();
    let other = db.clone();
    let mut txn = other.begin().unwrap();
    txn.execute("INSERT INTO t VALUES (2, 20, 200)", ())
        .unwrap();
    txn.execute("UPDATE t SET a = 101 WHERE id = 1", ())
        .unwrap();
    db.execute("ALTER TABLE t DROP COLUMN x", ()).unwrap();
    for (sql, expected) in [
        ("SELECT id, a FROM t ORDER BY id", vec![(1, 101), (2, 200)]),
        ("SELECT id, a FROM t WHERE id = 2", vec![(2, 200)]),
        (
            "SELECT id, a FROM t WHERE a > 0 ORDER BY id",
            vec![(1, 101), (2, 200)],
        ),
    ] {
        let read: Result<Vec<(i64, i64)>, Error> = txn.query(sql, ()).and_then(|rows| {
            rows.map(|r| r.map(|r| (r.get(0).unwrap(), r.get(1).unwrap())))
                .collect()
        });
        match read {
            Err(Error::SchemaChanged { .. }) => {}
            other => assert_eq!(other.unwrap(), expected, "{sql}"),
        }
    }
}

#[test]
fn a_foreign_key_check_on_a_changed_parent_reports_the_change() {
    let db = Database::open("memory://compiled_positions_fk_parent").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, x INTEGER)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id))",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let other = db.clone();
    let mut txn = other.begin().unwrap();
    txn.execute("INSERT INTO p VALUES (1, 10)", ()).unwrap();
    db.execute("ALTER TABLE p DROP COLUMN x", ()).unwrap();
    let inserted = txn.execute("INSERT INTO c VALUES (1, 1)", ());
    assert!(
        matches!(inserted, Err(Error::SchemaChanged { .. })),
        "{inserted:?}"
    );
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_transaction_reads_its_own_rows_after_another_sessions_drop_column() {
    local_rows_after_drop("memory://compiled_positions_local_rows");
}

#[test]
fn a_transaction_reads_its_own_rows_after_another_sessions_drop_column_on_file() {
    let dir = tempfile::tempdir().unwrap();
    local_rows_after_drop(&file_dsn(&dir));
}
