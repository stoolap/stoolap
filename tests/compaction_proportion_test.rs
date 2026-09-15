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

//! Compaction's normal size merge accepts a rewrite only when the largest
//! input holds at most four times the rows of the other inputs together:
//! a volume just under the target is not rewritten for one seal's worth
//! of rows, and the small volumes merge among themselves while it waits.

use stoolap::Database;

fn volume_files(dir: &std::path::Path, table: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.join("volumes").join(table))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".vol"))
        .collect();
    names.sort();
    names
}

#[test]
fn a_near_target_volume_waits_while_small_volumes_merge_among_themselves() {
    let dir = tempfile::tempdir().unwrap();
    // The target floors at 65,536 rows: the first volume is 60,000 rows,
    // sub-target by 5,536, and each later seal is 1,000 rows
    let db = Database::open(&format!(
        "file://{}?target_volume_rows=65536&compact_threshold=4",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t SELECT g.value, g.value FROM generate_series(1, 60000) g",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let seed = volume_files(dir.path(), "t");
    assert_eq!(seed.len(), 1);

    // One seal of 1,000 rows and a tombstone in it: two sub-target volumes
    // with tombstones, the trigger that rewrote the seed volume before
    db.execute(
        "INSERT INTO t SELECT g.value, g.value FROM generate_series(60001, 61000) g",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("DELETE FROM t WHERE id = 60500", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let files = volume_files(dir.path(), "t");
    assert!(
        files.contains(&seed[0]),
        "the 60,000-row volume was rewritten for 1,000 rows"
    );

    // Four more seals: six sub-target volumes; the five small ones merge,
    // the large one waits
    for k in 1..=4 {
        db.execute(
            &format!(
                "INSERT INTO t SELECT g.value, g.value FROM generate_series({}, {}) g",
                61_000 + (k - 1) * 1_000 + 1,
                61_000 + k * 1_000
            ),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let files = volume_files(dir.path(), "t");
    assert!(files.contains(&seed[0]), "the large volume was rewritten");
    assert_eq!(files.len(), 2, "the small volumes did not merge: {files:?}");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 60_000 + 5_000 - 1);
    let rows = db
        .query("SELECT id FROM t WHERE id IN (60500, 60499, 64999)", ())
        .unwrap()
        .count();
    assert_eq!(rows, 2);
}
