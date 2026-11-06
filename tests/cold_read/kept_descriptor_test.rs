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

//! A volume file read a row group at a time is opened once while it stays
//! among the files read most recently, not once per block.

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::volume_file_opens;
use stoolap::Database;

/// The open counter is process-wide
static OPENS: std::sync::Mutex<()> = std::sync::Mutex::new(());

const ROWS_PER_VOLUME: i64 = 200_000;

#[test]
fn a_scan_opens_each_volume_file_once() {
    let _opens = OPENS.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        for volume in 0..2 {
            db.execute(
                &format!(
                    "INSERT INTO t SELECT g.value, g.value * 2 FROM generate_series({}, {}) g",
                    volume * ROWS_PER_VOLUME + 1,
                    (volume + 1) * ROWS_PER_VOLUME
                ),
                (),
            )
            .unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("PRAGMA GROUP_CACHE_MB = 0", ()).unwrap();
    let want: i64 = (1..=2 * ROWS_PER_VOLUME).map(|i| i * 2).sum();
    let before = volume_file_opens();
    let sum: i64 = db.query_one("SELECT SUM(k + 0) FROM t", ()).unwrap();
    assert_eq!(sum, want);
    let first = volume_file_opens() - before;
    assert!(
        first <= 2,
        "the scan of two volumes of four groups each opened their files {first} times"
    );
    let before = volume_file_opens();
    let sum: i64 = db.query_one("SELECT SUM(k + 1) FROM t", ()).unwrap();
    assert_eq!(sum, want + 2 * ROWS_PER_VOLUME);
    let again = volume_file_opens() - before;
    assert_eq!(
        again, 0,
        "a second scan opened the files again {again} times"
    );
}
