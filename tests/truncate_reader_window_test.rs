// Copyright 2026 Stoolap Contributors
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

//! A read that meets a TRUNCATE between its two steps sees every row or
//! none, never the hot rows without the sealed ones

#![cfg(feature = "test-failpoints")]

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use stoolap::core::Value;
use stoolap::test_failpoints as fp;
use stoolap::Database;

/// How long a test waits for word from another thread before it fails
const WATCHDOG: Duration = Duration::from_secs(60);

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// Twenty rows in two volumes and three hot ones, `v` indexed and the hot
/// rows' values inside the sealed range, `g = id % 3`
fn twenty_sealed_three_hot(dir: &Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_v ON t(v)", ()).unwrap();
    for volume in 0..2 {
        let values: Vec<String> = (volume * 10 + 1..=volume * 10 + 10)
            .map(|id| format!("({id}, {id}, {})", id % 3))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute(
        "INSERT INTO t VALUES (21, 5, 0), (22, 10, 1), (23, 15, 2)",
        (),
    )
    .unwrap();
    db
}

/// The count a COUNT(*) answers, or the number of rows a query returns
fn measure(db: &Database, sql: &str) -> Result<i64, String> {
    if sql.starts_with("SELECT COUNT(*)") {
        return db.query_one(sql, ()).map_err(|e| e.to_string());
    }
    let mut rows = 0;
    for row in db.query(sql, ()).map_err(|e| e.to_string())? {
        let _: Value = row.map_err(|e| e.to_string())?.get(0).unwrap();
        rows += 1;
    }
    Ok(rows)
}

enum Seen {
    Waiting,
    Done(Result<i64, String>),
}

/// Runs `sql` on another thread while a TRUNCATE holds between clearing the
/// volumes and the hot rows; returns the reader's answer and whether it
/// waited for the truncate instead of answering inside it
fn read_inside_a_truncate(sql: &'static str) -> (i64, bool) {
    let (answer, waited) = read_inside_a_truncate_of(twenty_sealed_three_hot, sql);
    (answer.unwrap(), waited)
}

fn read_inside_a_truncate_of(
    setup: fn(&Path) -> Database,
    sql: &'static str,
) -> (Result<i64, String>, bool) {
    let dir = tempfile::tempdir().unwrap();
    let db = setup(dir.path());
    let (paused_tx, paused) = mpsc::channel();
    let (go, go_rx) = mpsc::channel::<()>();
    let truncator = db.clone();
    let truncate = std::thread::spawn(move || {
        fp::after_truncate_cold_cleared(move || {
            paused_tx.send(()).unwrap();
            go_rx.recv().unwrap();
        });
        truncator.execute("TRUNCATE TABLE t", ()).unwrap();
    });
    paused
        .recv_timeout(WATCHDOG)
        .expect("the truncate reached its pause");

    let (seen_tx, seen) = mpsc::channel();
    let reader = db.clone();
    let read = std::thread::spawn(move || {
        let waiting = seen_tx.clone();
        fp::on_destruction_wait(move || waiting.send(Seen::Waiting).unwrap());
        seen_tx.send(Seen::Done(measure(&reader, sql))).unwrap();
    });
    // The messages order the threads; the bound only keeps a broken build
    // from holding the run forever
    let next = || {
        seen.recv_timeout(WATCHDOG)
            .unwrap_or_else(|_| panic!("{sql}: no word from the reader"))
    };
    let answer = match next() {
        Seen::Done(n) => (n, false),
        Seen::Waiting => {
            go.send(()).unwrap();
            let Seen::Done(n) = next() else {
                panic!("{sql}: a second wait signal");
            };
            (n, true)
        }
    };
    let _ = go.send(());
    truncate.join().unwrap();
    read.join().unwrap();
    answer
}

/// The value of column `column` of each row `sql` returns, added up: the
/// answer of a one-row aggregate, or the sum of a grouped count
fn total(db: &Database, sql: &str, column: usize) -> Value {
    let mut sum: Option<f64> = None;
    for row in db.query(sql, ()).unwrap() {
        let value: Value = row.unwrap().get(column).unwrap();
        if let Some(v) = value.as_float64() {
            *sum.get_or_insert(0.0) += v;
        }
    }
    sum.map_or_else(Value::null_unknown, Value::Float)
}

/// Runs `sql` with a whole TRUNCATE, on another thread, between the read's
/// step over the hot rows and its step over the sealed ones; None when the
/// read never split
fn read_across_a_whole_truncate(sql: &str, column: usize) -> Option<Value> {
    let dir = tempfile::tempdir().unwrap();
    let db = twenty_sealed_three_hot(dir.path());
    let truncator = db.clone();
    let split = std::rc::Rc::new(std::cell::Cell::new(false));
    let noted = std::rc::Rc::clone(&split);
    fp::between_hot_and_cold_reads(move || {
        noted.set(true);
        std::thread::spawn(move || truncator.execute("TRUNCATE TABLE t", ()).unwrap())
            .join()
            .unwrap();
    });
    let answer = total(&db, sql, column);
    split.get().then_some(answer)
}

#[test]
fn reads_split_by_a_whole_truncate_see_every_row_or_none() {
    let mut torn = Vec::new();
    for (sql, column, whole) in [
        ("SELECT COUNT(*) FROM t", 0, 23.0),
        ("SELECT SUM(v) FROM t", 0, 240.0),
        ("SELECT MIN(v) FROM t", 0, 1.0),
        ("SELECT MAX(v) FROM t", 0, 20.0),
        ("SELECT g, COUNT(*) FROM t GROUP BY g", 1, 23.0),
    ] {
        match read_across_a_whole_truncate(sql, column) {
            None => torn.push(format!("{sql}: never split")),
            Some(answer) => {
                let whole_or_none = answer.is_null()
                    || answer.as_float64() == Some(0.0)
                    || answer.as_float64() == Some(whole);
                if !whole_or_none {
                    torn.push(format!("{sql}: {answer:?}"));
                }
            }
        }
    }
    assert!(torn.is_empty(), "{torn:#?}");
}

#[test]
fn reads_that_meet_a_truncate_between_its_steps_wait_for_it() {
    let mut torn = Vec::new();
    for (sql, whole) in [
        ("SELECT COUNT(*) FROM t", 23),
        ("SELECT COUNT(*) FROM t WHERE v >= 0", 23),
        ("SELECT COUNT(*) FROM t WHERE g = 1", 8),
        ("SELECT id FROM t LIMIT 100", 23),
        ("SELECT id FROM t WHERE id IN (1, 5, 15, 21, 22)", 5),
        ("SELECT id FROM t WHERE id = 5", 1),
        ("SELECT id FROM t ORDER BY id LIMIT 100", 23),
        ("SELECT id FROM t ORDER BY v DESC LIMIT 100", 23),
        ("SELECT * FROM t", 23),
    ] {
        let (answer, waited) = read_inside_a_truncate(sql);
        if !waited || (answer != 0 && answer != whole) {
            torn.push(format!("{sql}: {answer} of {whole}, waited {waited}"));
        }
    }
    assert!(torn.is_empty(), "{torn:#?}");
}

#[test]
fn a_read_that_a_truncate_overtakes_before_the_generation_moves_is_taken_again() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let dir = tempfile::tempdir().unwrap();
    let db = twenty_sealed_three_hot(dir.path());
    let release: Rc<RefCell<Option<mpsc::Sender<()>>>> = Rc::default();
    let truncate: Rc<RefCell<Option<std::thread::JoinHandle<()>>>> = Rc::default();
    let (to_release, joined) = (Rc::clone(&release), Rc::clone(&truncate));
    let truncator = db.clone();
    fp::after_merged_read_began(move || {
        let (cleared_tx, cleared) = mpsc::channel();
        let (go, go_rx) = mpsc::channel::<()>();
        *joined.borrow_mut() = Some(std::thread::spawn(move || {
            fp::before_clear_generation_moves(move || {
                cleared_tx.send(()).unwrap();
                go_rx.recv().unwrap();
            });
            truncator.execute("TRUNCATE TABLE t", ()).unwrap();
        }));
        cleared
            .recv_timeout(WATCHDOG)
            .expect("the truncate cleared the volumes");
        *to_release.borrow_mut() = Some(go);
    });
    let releasing = Rc::clone(&release);
    fp::on_destruction_wait(move || {
        if let Some(go) = releasing.borrow_mut().take() {
            go.send(()).unwrap();
        }
    });

    let answer = measure(&db, "SELECT * FROM t").unwrap();
    if let Some(go) = release.borrow_mut().take() {
        go.send(()).unwrap();
    }
    truncate
        .borrow_mut()
        .take()
        .expect("the read began")
        .join()
        .unwrap();
    assert!(answer == 0 || answer == 23, "{answer} rows of 23");
}

#[test]
fn a_batch_primary_key_fetch_split_by_a_whole_truncate_sees_every_row_or_none() {
    use stoolap::storage::traits::Engine;

    let dir = tempfile::tempdir().unwrap();
    let db = twenty_sealed_three_hot(dir.path());
    let engine = db.engine();
    let (_, token) = engine.get_version_store("t").unwrap().schema_and_token();
    let other = db.clone();
    let split = std::rc::Rc::new(std::cell::Cell::new(false));
    let noted = std::rc::Rc::clone(&split);
    fp::after_pk_hot_rows_fetched(move || {
        noted.set(true);
        other.execute("TRUNCATE TABLE t", ()).unwrap();
    });
    let rows = engine.fetch_rows_by_ids("t", &[1, 21], token).unwrap();
    assert!(split.get(), "the fetch went on to the volumes");
    let mut ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    assert!(
        ids.is_empty() || ids == [1, 21],
        "{ids:?}: a sealed row and a hot one, both or neither"
    );
}

/// Three hot rows and no volume
fn three_hot(dir: &Path) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db
}

#[test]
fn a_historical_read_of_a_table_without_volumes_waits_out_a_truncate() {
    let (answer, waited) =
        read_inside_a_truncate_of(three_hot, "SELECT * FROM t AS OF TRANSACTION 1");
    assert!(answer.is_ok(), "{answer:?}");
    assert!(waited, "the read answered inside the truncate");
}

#[test]
fn a_refused_hot_only_fetch_leaves_nothing_in_the_callers_buffer() {
    use std::cell::RefCell;
    use std::rc::Rc;
    use stoolap::core::{Row, RowVec};
    use stoolap::storage::expression::logical::ConstBoolExpr;
    use stoolap::storage::traits::Engine;

    let dir = tempfile::tempdir().unwrap();
    let db = three_hot(dir.path());
    // A claim refuses the truncate once it has marked rows as leaving
    let mut claiming = db.begin().unwrap();
    claiming
        .execute("UPDATE t SET v = 9 WHERE id = 1", ())
        .unwrap();

    let release: Rc<RefCell<Option<mpsc::Sender<()>>>> = Rc::default();
    let truncate: Rc<RefCell<Option<std::thread::JoinHandle<bool>>>> = Rc::default();
    let (to_release, joined) = (Rc::clone(&release), Rc::clone(&truncate));
    let truncator = db.clone();
    fp::after_unsealed_read(move || {
        let (marked_tx, marked) = mpsc::channel();
        let (go, go_rx) = mpsc::channel::<()>();
        *joined.borrow_mut() = Some(std::thread::spawn(move || {
            fp::after_truncate_marked(move || {
                marked_tx.send(()).unwrap();
                go_rx.recv().unwrap();
            });
            truncator.execute("TRUNCATE TABLE t", ()).is_err()
        }));
        marked
            .recv_timeout(WATCHDOG)
            .expect("the truncate marked rows as leaving");
        *to_release.borrow_mut() = Some(go);
    });
    let releasing = Rc::clone(&release);
    fp::on_destruction_wait(move || {
        if let Some(go) = releasing.borrow_mut().take() {
            go.send(()).unwrap();
        }
    });

    let mut txn = db.engine().begin_transaction().unwrap();
    let table = txn.get_table("t").unwrap();
    let mut buffer = RowVec::new();
    buffer.push((
        -1,
        Row::from_values(vec![Value::Integer(-1), Value::Integer(-1)]),
    ));
    table
        .fetch_rows_by_ids_into(&[1, 2, 3], &ConstBoolExpr::true_expr(), &mut buffer)
        .unwrap();
    drop(table);
    txn.rollback().unwrap();
    let refused = truncate.borrow_mut().take().expect("the read was refused");
    assert!(refused.join().unwrap(), "the claim refused the truncate");
    claiming.rollback().unwrap();

    let mut ids: Vec<i64> = buffer.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        [-1, 1, 2, 3],
        "the earlier row kept, each fetched row once"
    );
}
