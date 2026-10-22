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

//! A statement that reads hot rows through its view reads them in steps
//! from the root its scan took: a reader that stops early reads one step,
//! and one that reads on sees the rows as they were when the scan began

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use stoolap::test_failpoints::{
    after_hot_scan_checked, hot_scan_step_rows, on_last_view_lease_released,
};
use stoolap::Database;

const ROWS: i64 = 20_000;

/// The most rows a scan's first step reads: whole leaves of up to 128
/// rows until it holds 64
const FIRST_STEP: usize = 64 + 127;

fn fill(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT g.value, g.value % 10, 'v' || CAST(g.value AS TEXT) \
             FROM generate_series(1, {ROWS}) g"
        ),
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO m VALUES (1, 3), (2, 7)", ())
        .unwrap();
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
#[cfg(not(feature = "test-filedb"))]
fn a_limit_reads_one_step_of_the_hot_rows() {
    let db = Database::open("memory://hot_streaming_scan_limit").unwrap();
    fill(&db);
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE k + 0 > ANY (SELECT k FROM m) LIMIT 10",
            10,
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM m WHERE m.k = t.k) LIMIT 3",
            3,
        ),
        (
            "SELECT id FROM t WHERE s <> 'x' AND k + 0 < ALL (SELECT k FROM m) LIMIT 20",
            20,
        ),
    ] {
        let before = hot_scan_step_rows();
        let got = ids(&db, sql);
        let read = hot_scan_step_rows() - before;
        assert_eq!(got.len(), want, "{sql}: {got:?}");
        // One step of t, and the two rows of m
        assert!(
            read > 2 && read <= FIRST_STEP + 2,
            "{sql}: {read} hot rows read for {want}"
        );
    }
}

#[test]
fn a_scan_read_on_sees_the_rows_its_scan_began_with() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    fill(&db);
    let writer = db.clone();
    let before = hot_scan_step_rows();
    let mut rows = db
        .query("SELECT id, k, s FROM t WHERE k + 0 >= 0", ())
        .unwrap();
    let mut seen = Vec::new();
    for _ in 0..10 {
        seen.push(rows.next().unwrap().unwrap().get::<i64>(0).unwrap());
    }
    assert!(
        (1..=FIRST_STEP).contains(&(hot_scan_step_rows() - before)),
        "the scan reads in steps"
    );
    writer
        .execute("UPDATE t SET k = -1 WHERE id % 2 = 0", ())
        .unwrap();
    writer
        .execute("DELETE FROM t WHERE id % 3 = 0", ())
        .unwrap();
    writer
        .execute(
            &format!("INSERT INTO t VALUES ({}, 1, 'new')", ROWS + 1),
            (),
        )
        .unwrap();
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for row in rows {
        let row = row.unwrap();
        let id = row.get::<i64>(0).unwrap();
        assert_eq!(row.get::<i64>(1).unwrap(), id % 10, "row {id} as it began");
        assert_eq!(row.get::<String>(2).unwrap(), format!("v{id}"));
        seen.push(id);
    }
    assert_eq!(seen, (1..=ROWS).collect::<Vec<_>>());
    let after = ids(&db, "SELECT id FROM t WHERE k + 0 = -1 LIMIT 1000000");
    assert_eq!(
        after.len() as i64,
        (1..=ROWS).filter(|id| id % 2 == 0 && id % 3 != 0).count() as i64
    );
}

#[test]
#[cfg(not(feature = "test-filedb"))]
fn a_scan_read_on_survives_a_truncate() {
    let db = Database::open("memory://hot_streaming_scan_truncate").unwrap();
    fill(&db);
    let writer = db.clone();
    let before = hot_scan_step_rows();
    let mut rows = db.query("SELECT id FROM t WHERE k + 0 >= 0", ()).unwrap();
    assert_eq!(rows.next().unwrap().unwrap().get::<i64>(0).unwrap(), 1);
    assert!(
        (1..=FIRST_STEP).contains(&(hot_scan_step_rows() - before)),
        "the scan reads in steps"
    );
    writer.execute("TRUNCATE TABLE t", ()).unwrap();
    let rest: Vec<i64> = rows
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(rest, (2..=ROWS).collect::<Vec<_>>());
    assert_eq!(ids(&db, "SELECT id FROM t"), Vec::<i64>::new());
}

#[test]
#[cfg(not(feature = "test-filedb"))]
fn a_stepped_scan_answers_as_a_whole_read() {
    let db = Database::open("memory://hot_streaming_scan_answers").unwrap();
    fill(&db);
    db.execute("DELETE FROM t WHERE id % 7 = 0", ()).unwrap();
    db.execute("UPDATE t SET s = 'u' WHERE id % 5 = 0", ())
        .unwrap();
    db.execute("ALTER TABLE t ADD COLUMN d INTEGER DEFAULT 9", ())
        .unwrap();
    let live = |id: &i64| id % 7 != 0;
    let all: Vec<i64> = (1..=ROWS).filter(live).collect();
    let before = hot_scan_step_rows();
    assert_eq!(ids(&db, "SELECT id FROM t WHERE id + 0 > 0"), all);
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE d + 0 = 9"),
        all,
        "a row older than the column reads its default"
    );
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE s = 'u' AND k + 0 >= 0"),
        all.iter()
            .copied()
            .filter(|id| id % 5 == 0)
            .collect::<Vec<_>>()
    );
    assert!(
        hot_scan_step_rows() > before,
        "read through the stepped scan"
    );
    // A transaction's own writes are read with the store's rows
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM t WHERE id <= 10", ()).unwrap();
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE id + 0 > 0 LIMIT 3"),
        vec![11, 12, 13]
    );
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn a_column_dropped_once_the_scan_took_its_rows_leaves_the_filter_on_them() {
    let db = Database::open("memory://hot_streaming_scan_drop_column").unwrap();
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO d VALUES (1, 10, 20), (2, 11, 21)", ())
        .unwrap();
    let writer = db.clone();
    let dropped = Rc::new(Cell::new(false));
    let seen = Rc::clone(&dropped);
    after_hot_scan_checked(move || {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(writer.execute("ALTER TABLE d DROP COLUMN a", ()).is_ok());
        });
        seen.set(rx.recv_timeout(Duration::from_secs(5)) == Ok(true));
    });
    assert_eq!(ids(&db, "SELECT id FROM d WHERE b = 20"), vec![1]);
    assert!(dropped.get(), "the column was dropped inside the scan");
}

#[test]
fn a_closed_scan_lets_go_of_its_view_while_its_result_lives() {
    let db = Database::open("memory://hot_streaming_scan_close").unwrap();
    fill(&db);
    let mut rows = db.query("SELECT id FROM t WHERE k + 0 >= 0", ()).unwrap();
    assert_eq!(rows.next().unwrap().unwrap().get::<i64>(0).unwrap(), 1);
    let released = Rc::new(Cell::new(false));
    let seen = Rc::clone(&released);
    on_last_view_lease_released(move || seen.set(true));
    rows.close();
    assert!(released.get(), "the view was let go on close");
    drop(rows);
}
