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

//! An aggregate over the volumes answers from one state of them, however
//! a compaction or a seal moves them while it reads

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    after_cold_view_begun, block_file_reads, side_file_reads, volume_file_reads_by_fence,
    FailpointGuard,
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

/// Volume, block and side file reads this thread made under the seal fence
fn under_fence() -> usize {
    volume_file_reads_by_fence().1 + block_file_reads().1 + side_file_reads().1
}

/// Two sealed volumes of ids 1 to 200, `v` equal to the id, the first ten
/// deleted
fn sealed(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=1000",
        dir.display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for first in [1, 101] {
        db.execute(
            &format!(
                "INSERT INTO t SELECT value, value FROM generate_series({first}, {first} + 99)"
            ),
            (),
        )
        .unwrap();
        db.engine().seal_for_test().unwrap();
    }
    db.execute("DELETE FROM t WHERE id <= 10", ()).unwrap();
    db
}

#[test]
fn an_aggregate_whose_volumes_a_compaction_replaces_mid_read_reads_their_columns() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let want: i64 = (11..=200).sum();
    let compacted = Rc::new(Cell::new(false));
    let (other, ran) = (db.clone(), Rc::clone(&compacted));
    after_cold_view_begun(move || {
        other.engine().compact_for_test().unwrap();
        ran.set(true);
    });
    let sum: Option<i64> = db.query_one("SELECT SUM(v) FROM t", ()).unwrap();
    assert!(compacted.get(), "the compaction ran inside the read");
    assert_eq!(sum, Some(want), "the sum read across the compaction");
    let after: Option<i64> = db.query_one("SELECT SUM(v) FROM t", ()).unwrap();
    assert_eq!(after, Some(want));
}

#[test]
fn an_aggregate_a_seal_lands_inside_reads_again_and_counts_each_row_once() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let mut first = 201;
    let mut wrong = Vec::new();
    for sql in ["SELECT SUM(v) FROM t", "SELECT SUM(v) FROM t WHERE v > 0"] {
        // Hot rows the seal inside the read moves into a volume
        db.execute(
            &format!(
                "INSERT INTO t SELECT value, value FROM generate_series({first}, {first} + 49)"
            ),
            (),
        )
        .unwrap();
        let want: i64 = (11..first + 50).sum();
        first += 50;
        let other = db.clone();
        after_cold_view_begun(move || other.engine().seal_for_test().unwrap());
        let retries = stoolap::test_failpoints::seal_retries();
        let got: Option<i64> = db.query_one(sql, ()).unwrap();
        let retried = stoolap::test_failpoints::seal_retries() - retries;
        if got != Some(want) || retried == 0 {
            wrong.push(format!(
                "{sql}: got {got:?}, want {want}, {retried} retries"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_aggregate_over_evicted_volumes_reads_no_file_under_the_seal_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=1000",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE a (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX a_g ON a (g)", ()).unwrap();
    for first in [1, 1001, 2001] {
        db.execute(
            &format!(
                "INSERT INTO a SELECT value, value % 10, value \
                 FROM generate_series({first}, {first} + 999)"
            ),
            (),
        )
        .unwrap();
        db.engine().seal_for_test().unwrap();
    }
    db.execute("DELETE FROM a WHERE id % 7 = 0", ()).unwrap();
    let kept: Vec<i64> = (1..=3000).filter(|id| id % 7 != 0).collect();
    let in_group = |g: i64| kept.iter().filter(|id| *id % 10 == g).count();
    let groups: String = (0..10)
        .map(|g| format!("{g}:{}", in_group(g)))
        .collect::<Vec<_>>()
        .join(",");
    // Each statement, the path it reaches and its answer as text
    let cases: [(&str, String); 8] = [
        ("SELECT SUM(v) FROM a", kept.iter().sum::<i64>().to_string()),
        ("SELECT MIN(v) FROM a", "1".to_string()),
        ("SELECT MAX(v) FROM a", "3000".to_string()),
        (
            "SELECT COUNT(*) FROM a WHERE v > 100",
            kept.iter().filter(|v| **v > 100).count().to_string(),
        ),
        ("SELECT g, COUNT(*) FROM a GROUP BY g ORDER BY g", groups),
        ("SELECT COUNT(DISTINCT g) FROM a", "10".to_string()),
        (
            "SELECT COUNT(*) FROM (SELECT ROW_NUMBER() OVER (PARTITION BY g) FROM a)",
            kept.len().to_string(),
        ),
        (
            "SELECT COUNT(*) FROM (SELECT ROW_NUMBER() OVER (PARTITION BY g) FROM a LIMIT 5)",
            "5".to_string(),
        ),
    ];
    let _cache = NoGroupCache::set(&db);
    let mut wrong = Vec::new();
    for (sql, want) in cases {
        db.engine().cold_volumes_for_test("a");
        let before = under_fence();
        let got: Vec<String> = db
            .query(sql, ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (0..row.len())
                    .map(|i| row.get_value(i).unwrap().to_string())
                    .collect::<Vec<_>>()
                    .join(":")
            })
            .collect();
        let read = under_fence() - before;
        if got.join(",") != want {
            wrong.push(format!("{sql}: got {}, want {want}", got.join(",")));
        }
        if read > 0 {
            wrong.push(format!("{sql}: {read} file reads under the seal fence"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
