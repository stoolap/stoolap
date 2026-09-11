// Copyright 2026 Stoolap Contributors
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

use std::sync::Arc;

use stoolap::core::{DataType, Row, Value};
use stoolap::storage::volume::io::write_volume_to_disk;
use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
use stoolap::storage::volume::table::SegmentedTable;
use stoolap::storage::volume::writer::VolumeBuilder;
use stoolap::storage::{Engine, Table};
use stoolap::{Database, IsolationLevel};

fn setup(name: &str, hot_values: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(&format!("INSERT INTO t VALUES {hot_values}"), ())
        .unwrap();
    db.execute(
        "CREATE TABLE reference (id INTEGER PRIMARY KEY, k INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO reference VALUES (1, NULL), (2, 20), (3, 10), (4, NULL), (5, 30)",
        (),
    )
    .unwrap();
    db
}

fn check_storage_order(db: &Database, table: &dyn Table) {
    for (ascending, direction) in [(true, "ASC"), (false, "DESC")] {
        let expected: Vec<i64> = db
            .query(
                &format!("SELECT id FROM reference ORDER BY k {direction}, id ASC"),
                (),
            )
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        for limit in [0, 1, 3, 10] {
            for offset in [0, 2, 4, 8] {
                let rows = table
                    .collect_rows_sorted_with_limit(1, ascending, limit, offset)
                    .unwrap();
                let actual: Vec<i64> = rows
                    .iter()
                    .map(|row| row.get(0).unwrap().as_int64().unwrap())
                    .collect();
                let wanted: Vec<i64> = expected.iter().skip(offset).take(limit).copied().collect();
                assert_eq!(actual, wanted, "{direction} LIMIT {limit} OFFSET {offset}");
            }
        }
    }
}

#[test]
fn storage_sorted_nulls_read_committed() {
    let db = setup(
        "storage_sorted_nulls_read_committed",
        "(1, NULL), (2, 20), (3, 10), (4, NULL), (5, 30)",
    );
    let tx = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::ReadCommitted)
        .unwrap();
    let table = tx.get_table("t").unwrap();
    check_storage_order(&db, table.as_ref());
}

#[test]
fn storage_sorted_nulls_snapshot_history() {
    let db = setup(
        "storage_sorted_nulls_snapshot_history",
        "(1, NULL), (2, 20), (3, 10), (4, NULL), (5, 30)",
    );
    let tx = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
        .unwrap();
    db.execute("UPDATE t SET k = 99 WHERE id = 3", ()).unwrap();
    let table = tx.get_table("t").unwrap();
    check_storage_order(&db, table.as_ref());
}

#[test]
fn storage_sorted_nulls_local_rows() {
    let db = setup(
        "storage_sorted_nulls_local_rows",
        "(1, NULL), (2, 20), (3, 10), (4, NULL), (5, 30)",
    );
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut table = tx.get_table("t").unwrap();
    table
        .insert(Row::from_values(vec![Value::Integer(6), Value::Integer(5)]))
        .unwrap();
    db.execute("INSERT INTO reference VALUES (6, 5)", ())
        .unwrap();
    check_storage_order(&db, table.as_ref());
    tx.rollback().unwrap();
}

#[test]
fn storage_sorted_nulls_merged_volume() {
    let db = setup(
        "storage_sorted_nulls_merged_volume",
        "(3, 10), (4, NULL), (5, 30)",
    );
    let tx = db
        .engine()
        .begin_transaction_with_level(IsolationLevel::ReadCommitted)
        .unwrap();
    let hot = tx.get_table("t").unwrap();
    let schema = hot.schema().clone();
    let mut builder = VolumeBuilder::new(&schema);
    builder.add_row(
        1,
        &Row::from_values(vec![Value::Integer(1), Value::null(DataType::Integer)]),
    );
    builder.add_row(
        2,
        &Row::from_values(vec![Value::Integer(2), Value::Integer(20)]),
    );
    let volume = builder.finish();
    let dir = tempfile::tempdir().unwrap();
    let path = write_volume_to_disk(dir.path(), "t", 1, &volume).unwrap();
    let manager = Arc::new(SegmentManager::new("t", Some(dir.path().to_path_buf())));
    manager.register_segment(
        1,
        Arc::new(volume),
        SegmentMeta {
            segment_id: 1,
            file_path: path,
            row_count: 2,
            min_row_id: 1,
            max_row_id: 2,
            schema_version: 0,
            creation_lsn: 0,
            seal_seq: 0,
        },
        Some(&schema),
    );
    let table = SegmentedTable::new(hot, manager);
    check_storage_order(&db, &table);
}
