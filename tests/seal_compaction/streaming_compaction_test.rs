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

//! A compaction output is written to its file as the rows stream and is
//! published over that file: its columns are read a group at a time, and
//! the first unique INSERT after the compaction reads no column whole.

use stoolap::Database;

/// (segment id, tier, rows, resident bytes) of every volume of `t`
fn volumes(db: &Database) -> Vec<(i64, String, i64, i64)> {
    db.query("PRAGMA VOLUME_STATS", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<i64>(1).unwrap(),
                row.get::<String>(2).unwrap(),
                row.get::<i64>(3).unwrap(),
                row.get::<i64>(4).unwrap(),
            )
        })
        .collect()
}

fn open(dir: &std::path::Path) -> Database {
    Database::open(&format!(
        "file://{}?target_volume_rows=65536&compact_threshold=2",
        dir.display()
    ))
    .unwrap()
}

#[test]
fn a_compaction_output_is_published_over_its_file_and_a_unique_insert_reads_no_column_whole() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX uk ON t(k)", ()).unwrap();
    // Three sealed volumes of 40,000 rows: the third takes the table past
    // the threshold and the checkpoint compacts them into 65,536 + 54,464
    for batch in 0..3 {
        db.execute(
            &format!(
                "INSERT INTO t SELECT g.value, g.value * 2, 'v' || (g.value % 50) \
                 FROM generate_series({}, {}) g",
                batch * 40_000 + 1,
                (batch + 1) * 40_000
            ),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let after = volumes(&db);
    let mut rows: Vec<i64> = after.iter().map(|v| v.2).collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![54_464, 65_536], "{after:?}");
    for (_, tier, _, bytes) in &after {
        assert_eq!(
            tier, "file",
            "an output published with its columns decoded: {after:?}"
        );
        // Metadata, dictionary and index only: a decoded integer column of
        // 65,536 rows alone would be 589,824 bytes
        assert!(*bytes < 2_000_000, "{after:?}");
    }
    assert!(!dir
        .path()
        .join("volumes")
        .join("t")
        .read_dir()
        .unwrap()
        .flatten()
        .any(|e| e.path().extension().is_some_and(|x| x == "tmp")));

    // The first unique INSERTs after the compaction: a conflict and a new
    // key, neither decoding a column whole
    let conflict = db.execute("INSERT INTO t VALUES (200000, 60000, 'dup')", ());
    assert!(conflict.is_err(), "the duplicate key was accepted");
    assert_eq!(
        db.execute("INSERT INTO t VALUES (200001, 999999, 'new')", ())
            .unwrap(),
        1
    );
    let later = volumes(&db);
    for ((id, tier, _, bytes), (_, _, _, before)) in later.iter().zip(&after) {
        assert_eq!(tier, "file", "volume {id} turned {tier}: {later:?}");
        assert_eq!(bytes, before, "volume {id} grew: {later:?} from {after:?}");
    }
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 120_001);
    let v: String = db
        .query_one("SELECT v FROM t WHERE k = 120000", ())
        .unwrap();
    assert_eq!(v, "v0");
}

#[test]
fn a_reopened_table_reads_its_volumes_from_their_files_a_group_at_a_time() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v TEXT)",
            (),
        )
        .unwrap();
        db.execute("CREATE UNIQUE INDEX uk ON t(k)", ()).unwrap();
        for batch in 0..3 {
            db.execute(
                &format!(
                    "INSERT INTO t SELECT g.value, g.value * 2, 'v' || (g.value % 50) \
                     FROM generate_series({}, {}) g",
                    batch * 40_000 + 1,
                    (batch + 1) * 40_000
                ),
                (),
            )
            .unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
    }
    let db = open(dir.path());
    let opened = volumes(&db);
    assert_eq!(opened.len(), 2, "{opened:?}");
    for (_, tier, _, bytes) in &opened {
        assert_eq!(
            tier, "file",
            "a volume opened with its blocks in RAM: {opened:?}"
        );
        assert!(*bytes < 2_000_000, "{opened:?}");
    }
    // A point read and a filtered scan decode the groups they need
    // through the cache and leave the volumes as they were opened
    let v: String = db
        .query_one("SELECT v FROM t WHERE id = 70000", ())
        .unwrap();
    assert_eq!(v, "v0");
    let n = db
        .query("SELECT id, k FROM t WHERE v = 'v7' AND k > 200000", ())
        .unwrap()
        .count();
    assert_eq!(n, 400);
    let read = volumes(&db);
    assert_eq!(read, opened);
    // The unique index is not in the file: the first conflict after a
    // reopen is found by building it, and is found
    assert!(db
        .execute("INSERT INTO t VALUES (200000, 60000, 'dup')", ())
        .is_err());
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 120_000);
}
