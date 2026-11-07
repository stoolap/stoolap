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

use std::path::{Path, PathBuf};
use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk_v4};
use stoolap::Database;

const VOLUME_ROWS: i64 = 65_536;
const VOLUMES: i64 = 3;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=2&target_volume_rows=65536",
        dir.display()
    )
}

fn volume_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir.join("volumes").join("t"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "vol"))
        .collect();
    files.sort();
    files
}

fn magics(dir: &Path) -> (usize, usize) {
    let mut counts = (0, 0);
    for path in volume_files(dir) {
        match &std::fs::read(&path).unwrap()[..4] {
            b"STV4" => counts.0 += 1,
            b"STV5" => counts.1 += 1,
            other => panic!("{path:?} starts with {other:?}"),
        }
    }
    counts
}

fn rows(db: &Database) -> Vec<(i64, i64, String)> {
    db.query("SELECT id, v, s FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get(0).unwrap(),
                row.get(1).unwrap(),
                row.get(2).unwrap(),
            )
        })
        .collect()
}

#[test]
fn compaction_rewrites_one_v4_volume_per_cycle_as_v5() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("v4_rewrite");
    let db = Database::open(&dsn(&dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    for n in 0..VOLUMES {
        let (first, last) = (n * VOLUME_ROWS + 1, (n + 1) * VOLUME_ROWS);
        db.execute(
            &format!(
                "INSERT INTO t SELECT g.value, g.value * 3, 'k' || (g.value % 7) \
                 FROM generate_series({first}, {last}) g"
            ),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let want = rows(&db);
    db.close().unwrap();
    drop(db);

    // The same volumes in the format written before V5
    let volumes = dir.join("volumes");
    for path in volume_files(&dir) {
        let name = path.file_stem().unwrap().to_str().unwrap();
        let id = u64::from_str_radix(name.trim_start_matches("vol_"), 16).unwrap();
        let volume = read_volume_from_disk(&path).unwrap();
        write_volume_to_disk_v4(&volumes, "t", id, &volume).unwrap();
    }
    assert_eq!(magics(&dir), (VOLUMES as usize, 0));

    let db = Database::open(&dsn(&dir)).unwrap();
    assert_eq!(rows(&db), want, "the V4 volumes read back");
    let mut left = VOLUMES as usize;
    for cycle in 1..=VOLUMES {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let (v4, _) = magics(&dir);
        assert!(
            v4 + 1 == left,
            "cycle {cycle}: V4 volumes went from {left} to {v4}"
        );
        left = v4;
        if left == 0 {
            break;
        }
    }
    assert_eq!(left, 0, "every V4 volume was rewritten");
    assert_eq!(rows(&db), want, "the rewrite kept every row");
    db.close().unwrap();
    drop(db);

    let db = Database::open(&dsn(&dir)).unwrap();
    assert_eq!(magics(&dir).0, 0);
    assert_eq!(rows(&db), want, "the V5 volumes read back after a reopen");
}

#[test]
fn a_v4_volume_whose_overlap_exceeds_the_limit_is_rewritten_alone() {
    use std::sync::Arc;
    use stoolap::core::{DataType, Row, SchemaBuilder, Value};
    use stoolap::storage::volume::io::write_volume_to_disk;
    use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
    use stoolap::storage::volume::writer::VolumeBuilder;

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("v4_overlap");
    let db = Database::open(&dsn(&dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.close().unwrap();
    drop(db);
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("v", DataType::Integer, true, false)
        .column("s", DataType::Text, true, false)
        .build();
    let volumes = dir.join("volumes");
    let manager = SegmentManager::new("t", Some(volumes.clone()));
    // A to D hold the same rows at the target size, in V4; E, oversized and
    // in V5, is the cycle's own work
    let layout = [
        (1, 1, VOLUME_ROWS, true),
        (2, 1, VOLUME_ROWS, true),
        (3, 1, VOLUME_ROWS, true),
        (4, 1, VOLUME_ROWS, true),
        (5, 200_001, 300_000, false),
    ];
    for (id, first, last, v4) in layout {
        let mut builder = VolumeBuilder::new(&schema);
        for row in first..=last {
            let values = vec![Value::Integer(row), Value::Integer(id), Value::text("x")];
            builder.add_row(row, &Row::from_values(values));
        }
        let volume = builder.finish().unwrap();
        let path = if v4 {
            write_volume_to_disk_v4(&volumes, "t", id as u64, &volume).unwrap()
        } else {
            write_volume_to_disk(&volumes, "t", id as u64, &volume).unwrap()
        };
        manager
            .register_segment(
                id as u64,
                Arc::new(read_volume_from_disk(&path).unwrap()),
                SegmentMeta {
                    segment_id: id as u64,
                    file_path: path.file_name().unwrap().into(),
                    row_count: (last - first + 1) as usize,
                    min_row_id: first,
                    max_row_id: last,
                    creation_lsn: 0,
                    seal_seq: 0,
                    schema_version: 0,
                },
                Some(&schema),
            )
            .unwrap();
    }
    manager.persist().unwrap();
    drop(manager);
    assert_eq!(magics(&dir), (4, 1));

    let db = Database::open(&dsn(&dir)).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        magics(&dir).0,
        3,
        "one cycle rewrote one V4 volume, not the volumes its rows overlap"
    );
    let rows: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(rows, VOLUME_ROWS + 100_000);
}
