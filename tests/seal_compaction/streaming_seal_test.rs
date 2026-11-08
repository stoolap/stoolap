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

//! A sealed volume is written to its file a row group at a time and is
//! published over that file, as a compaction output is.

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

#[test]
fn a_sealed_volume_is_published_over_its_file_and_a_unique_insert_reads_no_column_whole() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?target_volume_rows=65536&compact_threshold=100",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX uk ON t(k)", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT g.value, g.value * 2, 'v' || (g.value % 50) \
         FROM generate_series(1, 100000) g",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let sealed = volumes(&db);
    let mut rows: Vec<i64> = sealed.iter().map(|v| v.2).collect();
    rows.sort_unstable();
    assert_eq!(rows, vec![34_464, 65_536], "{sealed:?}");
    for (_, tier, _, bytes) in &sealed {
        assert_eq!(
            tier, "file",
            "a volume sealed with its columns decoded: {sealed:?}"
        );
        // Metadata, dictionary and index only: a decoded integer column of
        // 65,536 rows alone would be 589,824 bytes
        assert!(*bytes < 2_000_000, "{sealed:?}");
    }
    assert!(!dir
        .path()
        .join("volumes")
        .join("t")
        .read_dir()
        .unwrap()
        .flatten()
        .any(|e| e.path().extension().is_some_and(|x| x == "tmp")));

    let conflict = db.execute("INSERT INTO t VALUES (100001, 60000, 'dup')", ());
    assert!(conflict.is_err(), "the duplicate key was accepted");
    assert_eq!(
        db.execute("INSERT INTO t VALUES (100002, 999999, 'new')", ())
            .unwrap(),
        1
    );
    let later = volumes(&db);
    for ((id, tier, _, bytes), (_, _, _, before)) in later.iter().zip(&sealed) {
        assert_eq!(tier, "file", "volume {id} turned {tier}: {later:?}");
        assert_eq!(bytes, before, "volume {id} grew: {later:?} from {sealed:?}");
    }
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 100_001);
    let v: String = db
        .query_one("SELECT v FROM t WHERE k = 140000", ())
        .unwrap();
    assert_eq!(v, "v0");
}

/// (rows, rows per group) of every volume file of table `w`
fn group_sizes(dir: &std::path::Path) -> Vec<(usize, usize)> {
    let mut sizes: Vec<(usize, usize)> = std::fs::read_dir(dir.join("volumes").join("w"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "vol"))
        .map(|p| {
            let volume = stoolap::storage::volume::io::read_volume_from_disk(&p).unwrap();
            (volume.meta.row_count, volume.meta.group_rows)
        })
        .collect();
    sizes.sort_unstable();
    sizes
}

fn doc(i: i64) -> String {
    format!("{{\"i\":{i},\"k\":\"{}\"}}", "x".repeat(3_000))
}

fn check_wide(db: &Database, rows: i64) {
    let count: i64 = db.query_one("SELECT COUNT(*) FROM w", ()).unwrap();
    assert_eq!(count, rows);
    for id in [0, 2_047, 2_048, rows / 2, rows - 1] {
        let read: String = db
            .query_one("SELECT doc FROM w WHERE id = $1", (id,))
            .unwrap();
        assert_eq!(read, doc(id), "row {id}");
    }
}

#[test]
fn a_table_of_wide_rows_is_sealed_and_compacted_in_groups_its_rows_size() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?target_volume_rows=65536&compact_threshold=2",
        dir.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, doc JSON)", ())
        .unwrap();
    let insert = db.prepare("INSERT INTO w VALUES ($1, $2)").unwrap();
    for part in 0..2i64 {
        let mut tx = db.begin().unwrap();
        for id in part * 6_000..(part + 1) * 6_000 {
            insert.execute_in(&mut tx, (id, doc(id))).unwrap();
        }
        tx.commit().unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        if part == 0 {
            // 3,000 bytes a row: the first volume measures 2,048 rows a group
            assert_eq!(group_sizes(dir.path()), vec![(6_000, 2_048)]);
        }
    }
    assert_eq!(
        group_sizes(dir.path()),
        vec![(6_000, 2_048), (6_000, 2_048)]
    );
    check_wide(&db, 12_000);
    drop(db);
    // Reopened, a checkpoint compacts both into one, measured again
    let db = Database::open(&dsn).unwrap();
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(group_sizes(dir.path()), vec![(12_000, 2_048)]);
    check_wide(&db, 12_000);
    drop(db);
    check_wide(&Database::open(&dsn).unwrap(), 12_000);
}
