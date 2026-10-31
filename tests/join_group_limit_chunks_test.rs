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

//! A GROUP BY + LIMIT join without an index on its right key reads its left
//! side in chunks: each chunk's groups follow the ones before, every left
//! row joined once, until the groups the LIMIT and OFFSET want are found

use std::sync::{Mutex, MutexGuard};

use stoolap::Database;

// The programs run are counted process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// t has `rows` rows, s one row for each of t's keys, neither indexed on
/// the key
fn setup(name: &str, rows: i64) -> Database {
    let db = Database::open(&format!("memory://join_group_limit_chunks_{name}")).unwrap();
    setup_on(db, rows)
}

fn setup_on(db: Database, rows: i64) -> Database {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db.execute(
        "INSERT INTO t SELECT g.value, g.value FROM generate_series(1, $1) g",
        (rows,),
    )
    .unwrap();
    db.execute(
        "INSERT INTO s SELECT g.value, g.value, 0 FROM generate_series(1, $1) g",
        (rows,),
    )
    .unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn an_offset_skips_groups_and_the_limit_still_finds_its_own() {
    let _guard = serial();
    let db = setup("offset", 200);
    // One t row in ten forms a group: 20 groups in all
    let groups = "SELECT t.id, COUNT(*) FROM t JOIN s ON s.k = t.k \
                  WHERE (t.id + s.v) % 10 = 0 GROUP BY t.id";
    for (limit, offset, want) in [(10, 5, 10), (10, 15, 5), (5, 0, 5), (30, 0, 20)] {
        let sql = format!("{groups} LIMIT {limit} OFFSET {offset}");
        let got = ids(&db, &sql);
        assert_eq!(got.len(), want, "{sql}: {got:?}");
        let mut distinct = got.clone();
        distinct.dedup();
        assert_eq!(distinct, got, "{sql}: a group came twice");
        assert!(got.iter().all(|id| id % 10 == 0), "{sql}: {got:?}");
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_commit_between_chunks_neither_repeats_nor_skips_a_group() {
    let _guard = serial();
    let _failpoints = stoolap::test_failpoints::FailpointGuard::new();
    let sql = "SELECT t.id, COUNT(*) FROM t JOIN s ON s.k = t.k \
               WHERE (t.id + s.v) % 10 = 0 GROUP BY t.id LIMIT 10";
    let dir = tempfile::tempdir().unwrap();
    // The first chunk's rows move, only rows after it change, or a seal
    // moves the rows the chunks read to a volume
    for (n, change, in_transaction) in [
        (0, "DELETE FROM t WHERE id IN (1, 10)", false),
        (1, "INSERT INTO t VALUES (0, 0)", false),
        (2, "DELETE FROM t WHERE id = 150", false),
        (3, "DELETE FROM t WHERE id IN (1, 10)", true),
        (4, "PRAGMA CHECKPOINT", false),
        (5, "PRAGMA CHECKPOINT", true),
    ] {
        let db = match change {
            "PRAGMA CHECKPOINT" => {
                let path = dir.path().join(format!("db{n}"));
                let dsn = format!("file://{}?checkpoint_interval=0", path.display());
                let db = setup_on(Database::open(&dsn).unwrap(), 200);
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
                db.execute("UPDATE t SET k = k WHERE id > 100", ()).unwrap();
                db
            }
            _ => setup(&format!("commit_{n}"), 200),
        };
        db.execute("INSERT INTO s VALUES (0, 0, 0)", ()).unwrap();
        let before = ids(&db, sql);
        let other = db.clone();
        stoolap::test_failpoints::before_join_next_chunk(move || {
            other.execute("DELETE FROM t WHERE id = 5", ()).unwrap();
            other.execute(change, ()).unwrap();
        });
        let got = match in_transaction {
            false => ids(&db, sql),
            true => {
                let mut tx = db.begin().unwrap();
                let mut got: Vec<i64> = tx
                    .query(sql, ())
                    .unwrap()
                    .map(|row| row.unwrap().get::<i64>(0).unwrap())
                    .collect();
                tx.commit().unwrap();
                got.sort_unstable();
                got
            }
        };
        stoolap::test_failpoints::reset_all();
        let after = ids(&db, sql);
        assert!(
            got == before || got == after,
            "{change}: {got:?}, the answer before {before:?}, after {after:?}"
        );
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn every_left_row_is_joined_once_however_many_chunks_it_takes() {
    let _guard = serial();
    let db = setup("once", 100);
    // The ten groups the LIMIT wants take every one of t's 100 rows, which
    // the statement without a LIMIT joins in one pass
    let groups = "SELECT t.id, COUNT(*) FROM t JOIN s ON s.k = t.k \
                  WHERE (t.id + s.v) % 10 = 0 GROUP BY t.id";
    let asked = |sql: &str| {
        let mut rows = 0;
        let ran = stoolap::test_failpoints::programs_ran(|| {
            rows = db.query(sql, ()).unwrap().count();
        });
        let asked = ran
            .iter()
            .filter(|(listing, _)| listing.contains("Mod"))
            .map(|(_, times)| times)
            .sum::<usize>();
        (rows, asked)
    };
    let (_, whole) = asked(groups);
    let (rows, chunked) = asked(&format!("{groups} LIMIT 10"));
    assert_eq!(rows, 10);
    assert!(whole > 0, "the WHERE ran as no program");
    assert!(
        chunked <= whole,
        "the chunks asked the joined rows {chunked} times, one pass {whole}"
    );
}
