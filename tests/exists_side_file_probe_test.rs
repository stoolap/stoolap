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

//! A limited EXISTS or NOT EXISTS over a table with sealed rows probes the
//! side files of its index per outer row, answers as the semi-join does, and
//! a probe the table refuses reads one semi-join set, not the volumes again.

use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};

use stoolap::storage::volume::secondary::READS;
use stoolap::Database;

// The side file counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `t` with an index on `k` and `u`, the same rows without one; key -1 holds
/// a tenth of the rows, past what a side file answers
fn fixture() -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT value, \
                 CASE WHEN value % 10 = 0 THEN -1 ELSE value % 1000 END \
                 FROM generate_series(1, 5000)"
            ),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO o VALUES (1, 1), (2, 2), (3, 3), (4, 4000), (5, NULL), (6, 7), (7, -1), (8, 999)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO d SELECT value, -1 FROM generate_series(1, 40)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // Key 1's rows move to key 4000 in the hot store; key 3's rows go away
    for table in ["t", "u"] {
        db.execute(&format!("UPDATE {table} SET k = 4000 WHERE k = 1"), ())
            .unwrap();
        db.execute(&format!("DELETE FROM {table} WHERE k = 3"), ())
            .unwrap();
    }
    (dir, db)
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

const SHAPES: [&str; 4] = [
    "SELECT o.id FROM o WHERE EXISTS (SELECT 1 FROM {t} WHERE {t}.k = o.k) ORDER BY o.id LIMIT 10",
    "SELECT o.id FROM o WHERE NOT EXISTS (SELECT 1 FROM {t} WHERE {t}.k = o.k) ORDER BY o.id LIMIT 10",
    "SELECT d.id FROM d WHERE EXISTS (SELECT 1 FROM {t} WHERE {t}.k = d.k) ORDER BY d.id LIMIT 30",
    "SELECT d.id FROM d WHERE NOT EXISTS (SELECT 1 FROM {t} WHERE {t}.k = d.k) ORDER BY d.id LIMIT 30",
];

#[test]
fn a_limited_exists_on_sealed_rows_probes_the_side_files() {
    let _serial = serial();
    let (_dir, db) = fixture();
    let control = ids(&db, &SHAPES[0].replace("{t}", "u"));
    let before = READS.probes.load(Ordering::Relaxed);
    assert_eq!(ids(&db, &SHAPES[0].replace("{t}", "t")), control);
    assert!(
        READS.probes.load(Ordering::Relaxed) > before,
        "the EXISTS read the side files"
    );
}

#[test]
fn every_shape_answers_as_the_unindexed_table() {
    let _serial = serial();
    let (_dir, db) = fixture();
    for shape in SHAPES {
        assert_eq!(
            ids(&db, &shape.replace("{t}", "t")),
            ids(&db, &shape.replace("{t}", "u")),
            "{shape}"
        );
    }
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (9001, 2), (9002, 999)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (9001, 2), (9002, 999)", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE k = 7", ()).unwrap();
    db.execute("DELETE FROM u WHERE k = 7", ()).unwrap();
    for shape in SHAPES {
        assert_eq!(
            ids(&db, &shape.replace("{t}", "t")),
            ids(&db, &shape.replace("{t}", "u")),
            "in a transaction: {shape}"
        );
    }
    db.execute("ROLLBACK", ()).unwrap();
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_refused_probe_reads_one_semi_join_set() {
    let _serial = serial();
    let (_dir, db) = fixture();
    let refusals = READS.cost_scans.load(Ordering::Relaxed);
    let runs = stoolap::test_failpoints::exists_subquery_runs();
    assert_eq!(ids(&db, &SHAPES[3].replace("{t}", "t")), Vec::<i64>::new());
    assert!(
        READS.cost_scans.load(Ordering::Relaxed) > refusals,
        "the dense key was refused"
    );
    assert_eq!(
        stoolap::test_failpoints::exists_subquery_runs() - runs,
        0,
        "no outer row ran the subquery over the volumes"
    );
}
