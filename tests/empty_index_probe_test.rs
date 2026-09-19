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

//! An empty secondary-index probe answers "no row" only while the index
//! carries every key the statement can see: local writes, old snapshots
//! and rollbacks keep their rows.

use stoolap::Database;
use tempfile::tempdir;

fn ids(db: &Database, key: i64) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query("SELECT id FROM t WHERE k = $1", (key,))
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn seeded(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k) USING BTREE", ())
        .unwrap();
    for i in 1..=200 {
        db.execute(&format!("INSERT INTO t VALUES ({}, {})", i, i * 10), ())
            .unwrap();
    }
    db
}

fn each_engine(name: &str, test: impl Fn(&Database)) {
    let dir = tempdir().unwrap();
    let file = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    for dsn in [format!("memory://empty_probe_{name}"), file] {
        let db = seeded(&dsn);
        test(&db);
    }
}

#[test]
fn an_absent_key_returns_nothing_and_a_present_one_its_row() {
    each_engine("absent", |db| {
        assert_eq!(ids(db, 555), Vec::<i64>::new());
        assert_eq!(ids(db, 70), vec![7]);
    });
}

#[test]
fn a_local_insert_is_found_under_its_key_before_commit() {
    each_engine("local_insert", |db| {
        db.execute("BEGIN", ()).unwrap();
        db.execute("INSERT INTO t VALUES (1000, 555)", ()).unwrap();
        assert_eq!(ids(db, 555), vec![1000]);
        db.execute("ROLLBACK", ()).unwrap();
        assert_eq!(ids(db, 555), Vec::<i64>::new());
    });
}

#[test]
fn a_local_key_update_moves_the_row_and_a_rollback_moves_it_back() {
    each_engine("local_update", |db| {
        db.execute("BEGIN", ()).unwrap();
        db.execute("UPDATE t SET k = 555 WHERE id = 7", ()).unwrap();
        assert_eq!(ids(db, 70), Vec::<i64>::new());
        assert_eq!(ids(db, 555), vec![7]);
        db.execute("ROLLBACK", ()).unwrap();
        assert_eq!(ids(db, 70), vec![7]);
        assert_eq!(ids(db, 555), Vec::<i64>::new());
    });
}

#[test]
fn a_snapshot_keeps_seeing_the_old_key_after_another_commit_moves_it() {
    each_engine("snapshot", |db| {
        let snap = db.clone();
        let writer = db.clone();
        snap.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
            .unwrap();
        assert_eq!(ids(&snap, 70), vec![7]);
        writer
            .execute("UPDATE t SET k = 555 WHERE id = 7", ())
            .unwrap();
        // The index now carries 555 for row 7 and not 70; the snapshot
        // still sees the row under 70 and nothing under 555
        assert_eq!(ids(&snap, 70), vec![7]);
        assert_eq!(ids(&snap, 555), Vec::<i64>::new());
        assert_eq!(ids(db, 70), Vec::<i64>::new());
        assert_eq!(ids(db, 555), vec![7]);
        snap.execute("COMMIT", ()).unwrap();
        assert_eq!(ids(&snap, 70), Vec::<i64>::new());
        assert_eq!(ids(&snap, 555), vec![7]);
    });
}
