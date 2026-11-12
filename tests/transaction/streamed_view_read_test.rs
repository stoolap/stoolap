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

//! An autocommit read committed SELECT whose rows stream after it returns
//! reads them through its statement view, whatever commits, seals and
//! maintenance run meanwhile

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use stoolap::test_failpoints as fp;
use stoolap::{Database, Error, Rows};

const RANGE: &str = "SELECT id, v FROM t WHERE id > $1 AND id < $2";
const CHANGES: &[&str] = &[
    "UPDATE t SET v = v + 100 WHERE id IN (1, 4)",
    "DELETE FROM t WHERE id IN (2, 5)",
    "INSERT INTO t VALUES (7, 70)",
];
const BEFORE: [(i64, i64); 6] = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];
const AFTER: [(i64, i64); 5] = [(1, 110), (3, 30), (4, 140), (6, 60), (7, 70)];

fn open_file(dir: &Path) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap()
}

/// Rows 1..=3 sealed when `sealed`, rows 4..=6 hot
fn fill(db: &Database, sealed: bool) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    if sealed {
        db.engine().seal_for_test().unwrap();
    }
    db.execute("INSERT INTO t VALUES (4, 40), (5, 50), (6, 60)", ())
        .unwrap();
}

fn pair(row: stoolap::Result<stoolap::ResultRow>) -> stoolap::Result<(i64, i64)> {
    let row = row?;
    Ok((row.get::<i64>(0)?, row.get::<i64>(1)?))
}

/// The first row of `rows`, then `between` on another handle, then the rest
fn streamed(mut rows: Rows, between: impl FnOnce()) -> stoolap::Result<Vec<(i64, i64)>> {
    let mut seen = vec![pair(rows.next().expect("a first row"))?];
    between();
    for row in rows {
        seen.push(pair(row)?);
    }
    seen.sort_unstable();
    Ok(seen)
}

#[test]
fn a_streamed_range_keeps_its_view_through_commits_and_a_seal() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    fill(&db, true);
    let range = db.prepare(RANGE).unwrap();
    let registered = || db.engine().registry().min_reader_seq().is_some();
    let rows = range.query((0, 100)).unwrap();
    assert!(registered(), "the rows hold their view");
    let other = db.clone();
    let seen = streamed(rows, || {
        for sql in CHANGES {
            other.execute(sql, ()).unwrap();
        }
        other.engine().seal_for_test().unwrap();
    })
    .unwrap();
    assert_eq!(seen, BEFORE, "the rows read through their view");
    assert!(!registered(), "the view went with the rows");
    let mut now: Vec<(i64, i64)> = range
        .query((0, 100))
        .unwrap()
        .map(|row| pair(row).unwrap())
        .collect();
    now.sort_unstable();
    assert_eq!(now, AFTER, "a later run reads the commits");
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_streamed_range_keeps_the_versions_maintenance_would_reclaim() {
    let db = Database::open("memory://streamed_view_read_maintenance").unwrap();
    fill(&db, false);
    let range = db.prepare(RANGE).unwrap();
    let store = db.engine().get_version_store("t").unwrap();
    let reclaimed = || {
        store.trim_history_past_limit()
            + store.cleanup_old_previous_versions_with_retention(std::time::Duration::ZERO) as usize
    };
    let rows = range.query((0, 100)).unwrap();
    let mut alive = 0;
    let seen = streamed(rows, || {
        for _ in 0..20 {
            db.execute("UPDATE t SET v = v + 1 WHERE id = 6", ())
                .unwrap();
        }
        alive = reclaimed();
    })
    .unwrap();
    assert_eq!(seen, BEFORE, "the rows read through their view");
    assert_eq!(
        (alive, reclaimed(), reclaimed()),
        (2, 0, 0),
        "reclaimed while the rows stream, once they are read, then again"
    );
}

#[test]
fn a_streamed_range_reads_through_its_view_alone() {
    let db = Database::open("memory://streamed_view_read_alone").unwrap();
    fill(&db, false);
    let range = db.prepare(RANGE).unwrap();
    let active_at_open = || {
        let (engine, opened) = (Arc::clone(db.engine()), Rc::new(Cell::new(None)));
        let seen = Rc::clone(&opened);
        fp::after_select_table_opened(move || seen.set(Some(engine.registry().active_count())));
        let rows: Vec<(i64, i64)> = range
            .query((0, 100))
            .unwrap()
            .map(|row| pair(row).unwrap())
            .collect();
        assert_eq!(rows.len(), BEFORE.len());
        opened.get()
    };
    let read_committed = active_at_open();
    db.execute("SET isolation_level = 'SNAPSHOT'", ()).unwrap();
    let snapshot = active_at_open();
    assert_eq!(
        (read_committed, snapshot),
        (Some(0), Some(1)),
        "transactions open as the table opens, under read committed and under snapshot"
    );
}

#[test]
fn a_streamed_range_a_truncate_overtakes_ends_with_the_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    fill(&db, true);
    db.execute("CREATE TABLE other (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let range = db.prepare(RANGE).unwrap();
    let other = db.clone();
    let rows = range.query((0, 100)).unwrap();
    let read = streamed(rows, || {
        other.execute("TRUNCATE TABLE other", ()).unwrap();
    });
    assert!(
        matches!(read, Err(Error::ViewOvertaken)),
        "the end of the rows is not their end: {read:?}"
    );
}

#[test]
fn a_streamed_range_ends_when_its_table_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    fill(&db, true);
    let range = db.prepare(RANGE).unwrap();
    let other = db.clone();
    let rows = range.query((0, 100)).unwrap();
    let read = streamed(rows, || {
        other.execute("DROP TABLE t", ()).unwrap();
    });
    assert!(
        matches!(read, Err(Error::ViewOvertaken)),
        "the end of the rows is not their end: {read:?}"
    );
}
