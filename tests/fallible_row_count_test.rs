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
use std::sync::Arc;

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::mvcc::{MVCCTable, TransactionVersionStore, VersionStore};
use stoolap::storage::traits::Table;
use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};
use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
use stoolap::storage::volume::table::SegmentedTable;
use stoolap::storage::volume::writer::VolumeBuilder;
use stoolap::Database;

fn cold_table(dir: &Path, snapshot: bool) -> (SegmentedTable, PathBuf) {
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("v", DataType::Integer, false, false)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    for id in [1, 2] {
        builder.add_row(
            id,
            &Row::from_values(vec![Value::Integer(id), Value::Integer(id * 10)]),
        );
    }
    let path = write_volume_to_disk(dir, "t", 1, &builder.finish()).unwrap();
    let volume = Arc::new(read_volume_from_disk(&path).unwrap());
    assert!(volume.is_warm());
    volume.mark_accessed();
    let manager = Arc::new(SegmentManager::new("t", Some(dir.to_path_buf())));
    manager.register_segment(
        1,
        volume,
        SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("vol_0000000000000001.vol"),
            row_count: 2,
            min_row_id: 1,
            max_row_id: 2,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        },
        Some(&schema),
    );
    manager.evict_idle_volumes(0);
    manager.evict_idle_volumes(3);
    assert!(manager.segments_raw()[&1].volume.is_cold());
    let store = Arc::new(VersionStore::new(schema.table_name.clone(), schema));
    let mut local = TransactionVersionStore::new(Arc::clone(&store), 1);
    if snapshot {
        local
            .put(
                3,
                Row::from_values(vec![Value::Integer(3), Value::Integer(30)]),
                false,
            )
            .unwrap();
    }
    let hot = Box::new(MVCCTable::new(1, store, local));
    let table = if snapshot {
        SegmentedTable::with_snapshot_seq(hot, manager, 1)
    } else {
        SegmentedTable::new(hot, manager)
    };
    (table, path)
}

#[test]
fn snapshot_count_propagates_reload_failure_and_keeps_local_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (table, path) = cold_table(dir.path(), true);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    for _ in 0..2 {
        let expected = table.collect_all_rows(None).map(|rows| rows.len());
        assert!(expected
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("cold volume reload failed"));
        assert_eq!(format!("{:?}", table.row_count()), format!("{expected:?}"));
    }
    std::fs::write(&path, bytes).unwrap();
    assert_eq!(table.collect_all_rows(None).unwrap().len(), 3);
}

#[test]
fn distinct_values_propagates_cold_reload_failure() {
    let dir = tempfile::tempdir().unwrap();
    let (table, path) = cold_table(dir.path(), false);
    std::fs::remove_file(&path).unwrap();
    let expected = table
        .segment_manager()
        .get_volumes_newest_first()
        .map(|_| Some(Vec::<Value>::new()));
    assert!(expected
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("cold volume reload failed"));
    assert_eq!(
        format!("{:?}", table.compute_distinct_values(1)),
        format!("{expected:?}")
    );
}

#[test]
fn exact_counts_include_transaction_local_rows() {
    let db = Database::open("memory://exact_counts_include_transaction_local_rows").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    assert_eq!(
        tx.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
            .unwrap(),
        2
    );
    tx.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(
        tx.query_one::<i64, _>("SELECT COUNT(*) + 1 FROM t", ())
            .unwrap(),
        2
    );
    tx.rollback().unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
            .unwrap(),
        0
    );
}

#[test]
fn prepared_count_tracks_committed_changes() {
    let db = Database::open("memory://prepared_count_tracks_committed_changes").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let count = db.prepare("SELECT COUNT(*) FROM t").unwrap();
    for id in 1..=3 {
        assert_eq!(count.query_one::<i64, _>(()).unwrap(), id - 1);
        db.execute("INSERT INTO t VALUES ($1)", (id,)).unwrap();
        assert_eq!(count.query_one::<i64, _>(()).unwrap(), id);
    }
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(count.query_one::<i64, _>(()).unwrap(), 2);
}

#[test]
fn cold_distinct_preserves_local_unindexed_values() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    let values: Vec<i64> = tx
        .query("SELECT DISTINCT v FROM t ORDER BY v", ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(values, [10, 20, 30]);
    tx.rollback().unwrap();
}
