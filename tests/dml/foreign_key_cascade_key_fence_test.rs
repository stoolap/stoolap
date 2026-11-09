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

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    block_file_reads, side_file_reads, volume_file_reads_by_fence, FailpointGuard,
};
use stoolap::Database;

/// The process-wide group cache budget at zero while held, restored on
/// drop; held one at a time, since tests share the process under cargo test
struct NoGroupCache<'a> {
    db: &'a Database,
    budget: i64,
    _serial: MutexGuard<'static, ()>,
}

static GROUP_CACHE_BUDGET: Mutex<()> = Mutex::new(());

impl<'a> NoGroupCache<'a> {
    fn set(db: &'a Database) -> Self {
        let serial = GROUP_CACHE_BUDGET
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let budget: i64 = db.query_one("PRAGMA GROUP_CACHE_MB", ()).unwrap();
        db.execute("PRAGMA GROUP_CACHE_MB = 0", ()).unwrap();
        Self {
            db,
            budget,
            _serial: serial,
        }
    }
}

impl Drop for NoGroupCache<'_> {
    fn drop(&mut self) {
        let _ = self
            .db
            .execute(&format!("PRAGMA GROUP_CACHE_MB = {}", self.budget), ());
    }
}

fn delta(after: (usize, usize), before: (usize, usize)) -> (usize, usize) {
    (after.0 - before.0, after.1 - before.1)
}

#[test]
fn a_hot_parent_reads_its_sealed_children_outside_the_seal_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    ))
    .unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE, x INTEGER)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY REFERENCES p(u) ON UPDATE CASCADE)",
        "INSERT INTO p SELECT value, value * 10, 0 FROM generate_series(1, 2000)",
        "INSERT INTO c SELECT value * 10 FROM generate_series(1, 2000)",
        "PRAGMA CHECKPOINT",
        // Parent 7 gets a hot copy; its child 70 stays sealed
        "UPDATE p SET x = 1 WHERE id = 7",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db.engine().cold_volumes_for_test("p");
    db.engine().cold_volumes_for_test("c");
    let _cache = NoGroupCache::set(&db);
    let blocks = block_file_reads();
    let others = (volume_file_reads_by_fence(), side_file_reads());
    let error = db
        .execute("UPDATE p SET u = 99999 WHERE x = 1", ())
        .unwrap_err()
        .to_string();
    let (outside, under) = delta(block_file_reads(), blocks);
    let volumes_under = delta(volume_file_reads_by_fence(), others.0).1;
    let sides_under = delta(side_file_reads(), others.1).1;
    assert!(error.contains("into the primary key of table"), "{error}");
    assert_eq!(under, 0, "blocks read under the fence ({outside} outside)");
    assert_eq!(
        (volumes_under, sides_under),
        (0, 0),
        "volumes, side files read under the fence"
    );
    assert!(outside > 0, "the sealed child was read, outside the fence");
    assert_eq!(
        db.query_one::<i64, _>("SELECT u FROM p WHERE id = 7", ())
            .unwrap(),
        70
    );
}
