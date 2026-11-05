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
    // sub-target by 5,536, and each later seal is 1,000 rows. Compaction
    // looks at the table once it holds more than two sub-target volumes
    let db = Database::open(&format!(
        "file://{}?target_volume_rows=65536&compact_threshold=2",
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
    // with tombstones, the trigger that rewrote the seed volume before.
    // The small volume sheds its tombstone alone
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

    // Four more seals, each merged with the small volumes before it while
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

#[test]
fn a_small_volume_with_a_tombstone_is_rewritten_alone_while_the_large_one_waits() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?target_volume_rows=65536",
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
    db.execute(
        "INSERT INTO t SELECT g.value, g.value FROM generate_series(60001, 61000) g",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = volume_files(dir.path(), "t");
    assert_eq!(before.len(), 2);

    // The tombstone is the small volume's own reason: it is rewritten
    // alone at the next checkpoint, with no further seal to carry it,
    // and the large one still waits
    db.execute("DELETE FROM t WHERE id = 60500", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let after = volume_files(dir.path(), "t");
    assert_eq!(after.len(), 2, "{after:?}");
    assert!(
        after.contains(&before[0]),
        "the 60,000-row volume was rewritten"
    );
    assert!(
        !after.contains(&before[1]),
        "the small volume kept its tombstone: {after:?}"
    );
    let rows: Vec<(i64, i64)> = db
        .query("PRAGMA VOLUME_STATS", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(3).unwrap(), row.get::<i64>(6).unwrap())
        })
        .collect();
    assert_eq!(
        rows,
        vec![(60_000, 0), (999, 0)],
        "rows and tombstones per volume"
    );
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 60_999);
}

#[test]
fn the_estimate_counts_outputs_the_way_the_writer_splits_them() {
    let dir = tempfile::tempdir().unwrap();
    // A target of 100,000 rows writes outputs of 65,536 rows: three
    // volumes of 40,000 rows merge into two outputs, one volume fewer,
    // so all three are rewritten together
    let db = Database::open(&format!(
        "file://{}?target_volume_rows=100000&compact_threshold=2",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    for k in 0..3 {
        db.execute(
            &format!(
                "INSERT INTO t SELECT g.value, g.value FROM generate_series({}, {}) g",
                k * 40_000 + 1,
                (k + 1) * 40_000
            ),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let mut rows: Vec<i64> = db
        .query("PRAGMA VOLUME_STATS", ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(3).unwrap())
        .collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![54_464, 65_536]);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 120_000);
}
