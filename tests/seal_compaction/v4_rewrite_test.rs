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
