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

//! An equality on every column of a unique key reads the cold rows a
//! built per-volume unique index names, and answers as the scan does

use std::sync::{Mutex, MutexGuard};
use stoolap::Database;

// The unique read counters are process-wide: every test that builds an
// index or decodes a block takes its turn
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const ROWS: i64 = 200_000;

const CREATE: &str = "CREATE TABLE u (id INTEGER PRIMARY KEY, ex TEXT NOT NULL, \
     sym TEXT NOT NULL, t INTEGER NOT NULL, v INTEGER, UNIQUE(ex, sym, t))";

fn dsn(dir: &std::path::Path) -> String {
    format!("file://{}?checkpoint_interval=3600", dir.display())
}

/// Row g has ex 'e{g % 3}', sym 's{g % 7}', t g: every key is distinct
fn sealed_table(dir: &std::path::Path) {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(CREATE, ()).unwrap();
    db.execute(
        &format!(
            "INSERT INTO u SELECT g.value, 'e' || (g.value % 3), 's' || (g.value % 7), \
             g.value, g.value * 10 FROM generate_series(1, {ROWS}) g"
        ),
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
}

/// A conflicting insert has the unique check build the reopened volume's index
fn build_index(db: &Database) {
    assert!(db
        .execute("INSERT INTO u VALUES (-1, 'e1', 's1', 1, 0)", ())
        .is_err());
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// Row 50000 has ex 'e2', sym 's6'
const HIT: &str = "ex = 'e2' AND sym = 's6' AND t = 50000";
/// Every column's value exists, the key does not
const MISS: &str = "ex = 'e0' AND sym = 's6' AND t = 50000";

const SHAPES: [&str; 4] = [
    "SELECT id FROM u WHERE {}",
    "SELECT id, v FROM u WHERE {} LIMIT 1",
    "SELECT id FROM u WHERE {} ORDER BY id LIMIT 1",
    "SELECT id FROM u WHERE {} LIMIT 5 OFFSET 0",
];

/// The shapes that read through the filtered cold paths; ORDER BY the key
/// with a LIMIT merges by row id instead
#[cfg(feature = "test-failpoints")]
const COUNTED: [&str; 3] = [SHAPES[0], SHAPES[1], SHAPES[3]];

fn shape(template: &str, filter: &str) -> String {
    template.replace("{}", filter)
}

#[test]
fn unique_key_equalities_answer_the_same_with_and_without_a_built_index() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    sealed_table(dir.path());
    let db = Database::open(&dsn(dir.path())).unwrap();
    let mut unbuilt = Vec::new();
    for template in SHAPES {
        for filter in [HIT, MISS] {
            unbuilt.push(ids(&db, &shape(template, filter)));
        }
    }
    build_index(&db);
    let mut built = Vec::new();
    for template in SHAPES {
        for filter in [HIT, MISS] {
            built.push(ids(&db, &shape(template, filter)));
        }
    }
    assert_eq!(built, unbuilt);
    assert_eq!(built[0], vec![50_000]);
    assert!(built[1].is_empty());
}

#[test]
fn an_older_copy_a_hot_version_and_a_delete_hide_the_candidate() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    sealed_table(dir.path());
    let db = Database::open(&dsn(dir.path())).unwrap();
    build_index(&db);
    // A newer volume takes row 50000: the older copy is no longer visible
    db.execute("UPDATE u SET v = 1 WHERE id = 50000", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for template in SHAPES {
        assert_eq!(ids(&db, &shape(template, HIT)), vec![50_000], "{template}");
    }
    let v: i64 = db
        .query_one(&format!("SELECT v FROM u WHERE {HIT}"), ())
        .unwrap();
    assert_eq!(v, 1, "the older copy answered");
    // A hot version shadows the cold one
    db.execute("UPDATE u SET v = 2 WHERE id = 50000", ())
        .unwrap();
    let v: i64 = db
        .query_one(&format!("SELECT v FROM u WHERE {HIT}"), ())
        .unwrap();
    assert_eq!(v, 2, "the cold copy answered past the hot one");
    // A deleted row is gone
    db.execute("DELETE FROM u WHERE id = 50000", ()).unwrap();
    for template in SHAPES {
        assert!(ids(&db, &shape(template, HIT)).is_empty(), "{template}");
    }
}

#[test]
fn a_value_of_another_type_answers_as_sql_equality_does() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    sealed_table(dir.path());
    let db = Database::open(&dsn(dir.path())).unwrap();
    let literals = ["50000.0", "50000.5", "1e20", "'50000'", "'50000.0'", "-0.0"];
    let answers = |db: &Database| -> Vec<Vec<i64>> {
        literals
            .iter()
            .map(|t| {
                ids(
                    db,
                    &format!("SELECT id FROM u WHERE ex = 'e2' AND sym = 's6' AND t = {t}"),
                )
            })
            .collect()
    };
    let bound = |db: &Database| -> Vec<Vec<i64>> {
        let sql = "SELECT id FROM u WHERE ex = ? AND sym = ? AND t = ?";
        let collect = |rows: stoolap::Rows| -> Vec<i64> {
            rows.map(|r| r.unwrap().get::<i64>(0).unwrap()).collect()
        };
        vec![
            collect(db.query(sql, ("e2", "s6", 50_000.0f64)).unwrap()),
            collect(db.query(sql, ("e2", "s6", 50_000.5f64)).unwrap()),
            collect(db.query(sql, ("e2", "s6", "50000")).unwrap()),
            collect(db.query(sql, ("e2", "s6", 50_000i64)).unwrap()),
        ]
    };
    let scanned = answers(&db);
    let scanned_bound = bound(&db);
    build_index(&db);
    assert_eq!(answers(&db), scanned);
    assert_eq!(bound(&db), scanned_bound);
    assert_eq!(scanned_bound[3], vec![50_000]);
    assert_eq!(scanned[0], vec![50_000]);
    assert!(scanned[1].is_empty());
    assert!(scanned[2].is_empty());
}

#[test]
fn a_key_column_added_after_the_seal_answers_from_its_default() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    sealed_table(dir.path());
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("ALTER TABLE u ADD COLUMN w INTEGER DEFAULT 7", ())
        .unwrap();
    db.execute("CREATE UNIQUE INDEX u_t_w ON u (t, w)", ())
        .unwrap();
    assert_eq!(
        ids(&db, "SELECT id FROM u WHERE t = 50000 AND w = 7"),
        vec![50_000]
    );
    assert!(ids(&db, "SELECT id FROM u WHERE t = 50000 AND w = 8").is_empty());
}

#[cfg(feature = "test-failpoints")]
mod counted {
    use super::*;
    use stoolap::test_failpoints::unique_read_counts;

    /// Index builds, candidates and block decodes one statement caused
    fn counted(db: &Database, sql: &str) -> (Vec<i64>, (usize, usize, usize)) {
        let before = unique_read_counts();
        let rows = ids(db, sql);
        let after = unique_read_counts();
        (
            rows,
            (after.0 - before.0, after.1 - before.1, after.2 - before.2),
        )
    }

    #[test]
    fn a_built_index_names_the_one_candidate_and_decodes_its_group_only() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        sealed_table(dir.path());
        let db = Database::open(&dsn(dir.path())).unwrap();
        build_index(&db);
        for template in COUNTED {
            let (rows, (builds, candidates, blocks)) = counted(&db, &shape(template, HIT));
            assert_eq!(rows, vec![50_000], "{template}");
            assert_eq!(builds, 0, "{template}: a read built an index");
            assert_eq!(candidates, 1, "{template}: candidates");
            assert!(blocks <= 5, "{template}: {blocks} blocks decoded");
        }
    }

    #[test]
    fn a_missing_key_on_a_built_index_decodes_nothing() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        sealed_table(dir.path());
        let db = Database::open(&dsn(dir.path())).unwrap();
        build_index(&db);
        for template in COUNTED {
            let (rows, (builds, candidates, blocks)) = counted(&db, &shape(template, MISS));
            assert!(rows.is_empty(), "{template}");
            assert_eq!((builds, candidates, blocks), (0, 0, 0), "{template}");
        }
    }

    #[test]
    fn a_read_never_builds_a_unique_index() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        sealed_table(dir.path());
        let db = Database::open(&dsn(dir.path())).unwrap();
        for template in SHAPES {
            for filter in [HIT, MISS] {
                let (_, (builds, candidates, _)) = counted(&db, &shape(template, filter));
                assert_eq!((builds, candidates), (0, 0), "{template} {filter}");
            }
        }
    }

    #[test]
    fn a_snapshot_reads_without_the_unique_index() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        sealed_table(dir.path());
        let db = Database::open(&dsn(dir.path())).unwrap();
        build_index(&db);
        db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
            .unwrap();
        let (rows, (_, candidates, _)) = counted(&db, &format!("SELECT id FROM u WHERE {HIT}"));
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(rows, vec![50_000]);
        assert_eq!(candidates, 0, "a snapshot read used the unique index");
    }
}
