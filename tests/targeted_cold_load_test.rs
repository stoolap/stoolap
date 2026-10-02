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

//! A DML statement loads only the cold volumes it reads from, through the
//! file owners its snapshot captured

#![cfg(feature = "test-failpoints")]

use std::path::{Path, PathBuf};

use stoolap::test_failpoints::{volume_loads, FailpointGuard};
use stoolap::Database;

const ROWS: i64 = 1_000;

/// `volumes` sealed volumes of ROWS rows each (ids b*ROWS+1 ..), k = id * 10
/// unique, all made cold
fn cold_table(dir: &Path, volumes: i64) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX t_k ON t(k)", ()).unwrap();
    for b in 0..volumes {
        let values: Vec<String> = (b * ROWS + 1..=(b + 1) * ROWS)
            .map(|id| format!("({id}, {}, 0)", id * 10))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let (count, cold) = db.engine().cold_volumes_for_test("t");
    assert_eq!((count, cold), (volumes as usize, volumes as usize));
    db
}

/// Volume loads (requests, file reads) of running `sql`
fn loads_of(db: &Database, sql: &str) -> (usize, usize) {
    let before = volume_loads();
    db.execute(sql, ()).unwrap();
    let after = volume_loads();
    (after.0 - before.0, after.1 - before.1)
}

fn vol_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "vol") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn an_upsert_on_one_cold_row_loads_only_its_volume() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    let loads = loads_of(
        &db,
        "INSERT INTO t VALUES (5, 50, 0) ON DUPLICATE KEY UPDATE v = 2",
    );
    assert_eq!(loads, (1, 1), "loads (requests, reads)");
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 5", ()).unwrap();
    assert_eq!(v, 2);
}

#[test]
fn a_delete_by_primary_key_loads_only_its_volume() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    assert_eq!(loads_of(&db, "DELETE FROM t WHERE id = 5"), (1, 1));
    let n: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(n, 8 * ROWS - 1);
}

#[test]
fn a_delete_whose_predicate_prunes_to_two_volumes_loads_two() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    let loads = loads_of(&db, "DELETE FROM t WHERE id BETWEEN 1500 AND 2500");
    assert_eq!(loads, (2, 2));
    let n: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(n, 8 * ROWS - 1001);
}

#[test]
fn a_damaged_volume_the_statement_prunes_does_not_fail_it() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    // The newest volume holds ids 7001..8000; the statement reads id 5
    let newest = vol_files(dir.path()).pop().unwrap();
    std::fs::write(&newest, b"damaged").unwrap();
    db.execute("DELETE FROM t WHERE id = 5", ()).unwrap();
}

fn count(db: &Database) -> i64 {
    db.query_one("SELECT COUNT(*) FROM t", ()).unwrap()
}

#[test]
fn an_unreadable_candidate_fails_the_statement_before_any_change() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    db.execute("INSERT INTO t VALUES (9001, 90010, 0)", ())
        .unwrap();
    // Ids 1001..2000 live in the second oldest volume
    let second = vol_files(dir.path()).remove(1);
    let bytes = std::fs::read(&second).unwrap();
    std::fs::write(&second, b"damaged").unwrap();
    assert!(db
        .execute("DELETE FROM t WHERE id IN (5, 1005, 9001)", ())
        .is_err());
    // The file comes back so the checks can read the table
    std::fs::write(&second, &bytes).unwrap();
    let hot: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 9001", ())
        .unwrap();
    assert_eq!(hot, 1, "the hot row stays");
    let healthy: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 5", ())
        .unwrap();
    assert_eq!(healthy, 1, "the healthy candidate's row stays");
}

#[test]
fn a_statement_that_cannot_settle_its_cold_reads_leaves_the_transaction_usable() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (9001, 90010, 0)", ())
        .unwrap();
    stoolap::test_failpoints::COLD_READS_FORGET.store(true, std::sync::atomic::Ordering::Release);
    let error = tx.execute("DELETE FROM t WHERE id = 5", ()).unwrap_err();
    stoolap::test_failpoints::COLD_READS_FORGET.store(false, std::sync::atomic::Ordering::Release);
    assert!(error.to_string().contains("write conflict"), "{error}");
    tx.execute("INSERT INTO t VALUES (9002, 90020, 0)", ())
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(count(&db), 8 * ROWS + 2, "both inserts kept, id 5 kept");
}

#[test]
fn a_compaction_after_capture_does_not_hide_a_unique_conflict_or_a_row() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    let other = db.clone();
    other.execute("PRAGMA compact_threshold = 2", ()).unwrap();
    // k = 10010 belongs to id 1001, in another volume than id 5
    let other_conflict = other.clone();
    stoolap::test_failpoints::after_statement_captured(move || {
        other_conflict.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    let error = db
        .execute("UPDATE t SET k = 10010 WHERE id = 5", ())
        .unwrap_err();
    assert!(
        error.to_string().to_lowercase().contains("unique"),
        "{error}"
    );
    stoolap::test_failpoints::after_statement_captured(move || {
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    db.execute("UPDATE t SET v = 3 WHERE id = 6", ()).unwrap();
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 6", ()).unwrap();
    assert_eq!(v, 3);
}

#[test]
fn a_repeated_round_reuses_the_volumes_it_read() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    let other = db.clone();
    // A seal between the round and the fence makes the round repeat
    stoolap::test_failpoints::after_cold_round_prepared(move || {
        other
            .execute("INSERT INTO t VALUES (9001, 90010, 0)", ())
            .unwrap();
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    let loads = loads_of(&db, "UPDATE t SET v = 4 WHERE id = 5");
    assert_eq!(loads, (1, 1), "loads (requests, reads)");
    let v: i64 = db.query_one("SELECT v FROM t WHERE id = 5", ()).unwrap();
    assert_eq!(v, 4);
}

#[test]
fn publication_keeps_a_view_another_reader_published_meanwhile() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 8);
    let other = db.clone();
    let hook_clones = std::rc::Rc::new(std::cell::Cell::new(0));
    let seen = std::rc::Rc::clone(&hook_clones);
    // After the round read id 5's volume, a unique check of another
    // statement loads and publishes the same volume (k = 50 is id 5's)
    stoolap::test_failpoints::after_cold_round_prepared(move || {
        let before = stoolap::test_failpoints::segment_map_clones();
        assert!(other
            .execute("INSERT INTO t VALUES (99999, 50, 0)", ())
            .is_err());
        seen.set(stoolap::test_failpoints::segment_map_clones() - before);
    });
    let before = stoolap::test_failpoints::segment_map_clones();
    db.execute("UPDATE t SET v = 4 WHERE id = 5", ()).unwrap();
    let total = stoolap::test_failpoints::segment_map_clones() - before;
    assert_eq!(
        total - hook_clones.get(),
        0,
        "the statement found the entry warm and left it"
    );
}

#[test]
fn many_unique_candidates_publish_with_one_map_copy() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = cold_table(dir.path(), 64);
    // Rows 1..64 live in the oldest volume; each new k falls inside another
    // volume's k range and is present nowhere
    let clones = stoolap::test_failpoints::segment_map_clones();
    let loads = loads_of(&db, "UPDATE t SET k = id * 10000 + 505 WHERE id <= 64");
    let clones = stoolap::test_failpoints::segment_map_clones() - clones;
    assert!(clones <= 1, "{clones} map copies");
    assert!(loads.0 <= 4, "loads {loads:?}");
    let moved: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE k % 10 = 5", ())
        .unwrap();
    assert_eq!(moved, 64);
}
