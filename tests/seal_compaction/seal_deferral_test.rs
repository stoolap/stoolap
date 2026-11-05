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

//! A seal whose registration stays stale for every round it tries is put
//! off: nothing it wrote is left behind and its rows stay hot.

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering;

use stoolap::test_failpoints::{FailpointGuard, SEAL_REGISTRATION_STALE_ROUNDS};
use stoolap::Database;

fn dsn(dir: &tempfile::TempDir) -> String {
    format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    )
}

/// 1,000 sealed rows and 1,000 hot ones, with an index whose side files
/// the seal builds
fn table_with_hot_rows(dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let insert = db.prepare("INSERT INTO t VALUES ($1, $1)").unwrap();
    let rows = |from: i64, to: i64| {
        db.execute("BEGIN", ()).unwrap();
        for id in from..=to {
            insert.execute((id,)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
    };
    rows(1, 1_000);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    rows(1_001, 2_000);
    db
}

/// The volume and side files on disk
fn files(dir: &tempfile::TempDir, ext: &str) -> usize {
    std::fs::read_dir(dir.path().join("volumes").join("t"))
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == ext))
                .count()
        })
        .unwrap_or(0)
}

fn count(db: &Database) -> i64 {
    db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
        .unwrap()
}

fn by_k(db: &Database, k: i64) -> Vec<i64> {
    db.query("SELECT id FROM t WHERE k = $1", (k,))
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn a_seal_stale_every_round_leaves_its_rows_hot_and_its_files_gone() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = table_with_hot_rows(&dir);
    assert_eq!((files(&dir, "vol"), files(&dir, "sidx")), (1, 1));

    SEAL_REGISTRATION_STALE_ROUNDS.store(3, Ordering::Release);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        SEAL_REGISTRATION_STALE_ROUNDS.load(Ordering::Acquire),
        0,
        "the seal tried three rounds"
    );

    assert_eq!(db.engine().volume_stats().len(), 1, "no volume published");
    assert_eq!(
        (files(&dir, "vol"), files(&dir, "sidx")),
        (1, 1),
        "the unpublished volume and side file are removed"
    );
    assert_eq!(count(&db), 2_000);
    assert_eq!(by_k(&db, 1_500), vec![1_500], "a hot row through the index");
    assert_eq!(by_k(&db, 500), vec![500], "a sealed row through the index");

    // No checkpoint at close, and no file holds the hot rows: they come back
    // from the log, and the seal after recovery puts them in a volume
    db.close().unwrap();
    drop(db);
    assert_eq!(files(&dir, "vol"), 1, "the hot rows are in the log alone");
    let db = Database::open(&dsn(&dir)).unwrap();
    assert_eq!(count(&db), 2_000, "the hot rows replayed from the log");
    assert_eq!(by_k(&db, 1_500), vec![1_500]);
    assert_eq!(db.engine().volume_stats().len(), 2, "sealed after recovery");
}

#[test]
fn the_next_checkpoint_seals_what_a_stale_seal_left_hot() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = table_with_hot_rows(&dir);

    SEAL_REGISTRATION_STALE_ROUNDS.store(3, Ordering::Release);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 1);

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 2, "the seal landed");
    assert_eq!((files(&dir, "vol"), files(&dir, "sidx")), (2, 2));
    assert_eq!(count(&db), 2_000);
    assert_eq!(by_k(&db, 1_500), vec![1_500]);

    db.close().unwrap();
    drop(db);
    let db = Database::open(&dsn(&dir)).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 2);
    assert_eq!(count(&db), 2_000);
}
