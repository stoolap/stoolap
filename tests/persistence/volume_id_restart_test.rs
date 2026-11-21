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

//! A volume sealed after a restart takes an id above every id the database's
//! files already use, also when the clock reads no later than it did before

#![cfg(feature = "test-failpoints")]

use std::path::Path;

use stoolap::test_failpoints::{self, FailpointGuard};
use stoolap::Database;

/// The clock every seal of these tests reads, in microseconds
const CLOCK: u64 = 1_000_000;

fn open(dir: &Path) -> Database {
    let dsn = format!("file://{}?checkpoint_interval=0", dir.display());
    Database::open(&dsn).unwrap()
}

fn ids(db: &Database) -> Vec<i64> {
    db.query("SELECT id FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

/// The ids the volume files of table `t` are named after, in order
fn volume_ids(dir: &Path) -> Vec<u64> {
    let mut ids: Vec<u64> = std::fs::read_dir(dir.join("volumes").join("t"))
        .unwrap()
        .filter_map(|entry| {
            let name = entry.unwrap().file_name().into_string().unwrap();
            let hex = name.strip_prefix("vol_")?.strip_suffix(".vol")?;
            u64::from_str_radix(hex, 16).ok()
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// Rows 1 and 2 sealed into a volume with the clock held, then the database
/// closed and the process's handed-out ids forgotten, as after a restart
fn sealed_then_restarted(dir: &Path) {
    test_failpoints::forget_volume_ids();
    test_failpoints::hold_volume_clock(Some(CLOCK));
    let db = open(dir);
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    test_failpoints::forget_volume_ids();
}

#[test]
fn a_seal_after_a_restart_with_the_clock_set_back_keeps_the_earlier_volume() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    sealed_then_restarted(dir.path());
    assert_eq!(volume_ids(dir.path()), [CLOCK], "the first volume's id");

    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(ids(&db), [1, 2, 3], "rows after the second seal");
    db.close().unwrap();

    test_failpoints::forget_volume_ids();
    let db = open(dir.path());
    assert_eq!(ids(&db), [1, 2, 3], "rows after reopening");
    db.close().unwrap();
}

#[test]
fn a_seal_after_a_restart_takes_an_id_above_every_file_named_after_one() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    sealed_then_restarted(dir.path());
    // A side file left behind by a volume that is gone
    let left = CLOCK + 7;
    let stray = dir
        .path()
        .join(format!("volumes/t/vol_{left:016x}.g2.sidx"));
    std::fs::write(&stray, b"left behind").unwrap();

    let db = open(dir.path());
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let newest = volume_ids(dir.path()).into_iter().max();
    assert!(
        newest > Some(left),
        "the newest volume id {newest:?} is not above the left file's {left}"
    );
    assert_eq!(ids(&db), [1, 2, 3]);
    db.close().unwrap();
}
