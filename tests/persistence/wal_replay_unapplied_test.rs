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

//! A committed log record that recovery cannot apply stops the open and
//! names its LSN, where it was passed over before

use std::io::Write;
use std::path::Path;

use stoolap::core::{ForeignKeyAction, ForeignKeyConstraint};
use stoolap::storage::mvcc::engine::MVCCEngine;
use stoolap::storage::mvcc::persistence::{serialize_row_version, IndexMetadata, DDL_TXN_ID};
use stoolap::storage::mvcc::version_store::RowVersion;
use stoolap::storage::mvcc::wal_manager::{WALEntry, WALManager, WALOperationType};
use stoolap::storage::SyncMode;
use stoolap::{DataType, Database, IndexType, Row, SchemaBuilder, Value};

const TXN: i64 = 1_000_000;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// A closed database whose table t (id, n) holds one row
fn database() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
        "INSERT INTO t VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    db.close().unwrap();
    dir
}

fn record(txn_id: i64, table: &str, op: WALOperationType, data: Vec<u8>) -> WALEntry {
    WALEntry::new(txn_id, table.to_string(), 0, op, data)
}

/// Appends `entry` to the database's log, committed under its own id
fn append_committed(dir: &Path, entry: WALEntry) {
    let txn_id = entry.txn_id;
    let wal = WALManager::new(dir.join("wal"), SyncMode::Full).unwrap();
    wal.append_unit(vec![entry, WALEntry::commit_marker(txn_id)], true)
        .unwrap();
    wal.close().unwrap();
}

/// The open's error after `entry` was committed to a database's log
fn open_error(entry: WALEntry) -> String {
    let dir = database();
    append_committed(dir.path(), entry);
    match Database::open(&dsn(dir.path())) {
        Ok(_) => panic!("the open passed over the record"),
        Err(e) => e.to_string(),
    }
}

fn assert_stops(entry: WALEntry, names: &[&str]) {
    let error = open_error(entry);
    for name in names.iter().chain(&["LSN"]) {
        assert!(error.contains(name), "{name} missing from: {error}");
    }
}

fn row(values: Vec<Value>) -> Vec<u8> {
    serialize_row_version(&RowVersion {
        txn_id: TXN,
        deleted_at_txn_id: 0,
        data: Row::from_values(values),
        create_time: 0,
    })
    .unwrap()
}

/// A table record for table u, its bytes changed by `corrupt`
fn table_record(schema: stoolap::Schema, corrupt: impl Fn(&mut Vec<u8>)) -> WALEntry {
    let mut data = MVCCEngine::serialize_schema(&schema);
    corrupt(&mut data);
    record(DDL_TXN_ID, "u", WALOperationType::CreateTable, data)
}

/// Replaces the one occurrence of `from` in `data` with `to`
fn replace(data: &mut [u8], from: &[u8], to: &[u8]) {
    let at = data
        .windows(from.len())
        .position(|w| w == from)
        .expect("bytes to replace");
    data[at..at + to.len()].copy_from_slice(to);
}

#[test]
fn an_unreadable_table_record_stops_the_open() {
    assert_stops(
        record(
            DDL_TXN_ID,
            "u",
            WALOperationType::CreateTable,
            vec![1, 2, 3],
        ),
        &["CreateTable"],
    );
}

#[test]
fn a_table_record_with_an_unknown_column_type_stops_the_open() {
    let schema = SchemaBuilder::new("u")
        .column("id", DataType::Integer, false, true)
        .column("qq", DataType::Integer, true, false)
        .build();
    let entry = table_record(schema, |data| {
        let tag = DataType::Integer.as_u8();
        replace(data, &[b'q', b'q', tag], &[b'q', b'q', 0xEE]);
    });
    assert_stops(entry, &["CreateTable", "data type"]);
}

#[test]
fn a_table_record_with_an_unreadable_foreign_key_name_stops_the_open() {
    let mut schema = SchemaBuilder::new("u")
        .column("id", DataType::Integer, false, true)
        .column("p", DataType::Integer, true, false)
        .build();
    schema.foreign_keys.push(ForeignKeyConstraint {
        column_index: 1,
        column_name: "p".to_string(),
        referenced_table: "qq".to_string(),
        referenced_column: "id".to_string(),
        on_delete: ForeignKeyAction::Restrict,
        on_update: ForeignKeyAction::Restrict,
    });
    let entry = table_record(schema, |data| replace(data, b"qq", &[0xFF, 0xFE]));
    assert_stops(entry, &["CreateTable", "foreign key"]);
}

#[test]
fn an_unreadable_row_stops_the_open() {
    assert_stops(
        record(TXN, "t", WALOperationType::Insert, vec![1, 2, 3]),
        &["Insert", "'t'"],
    );
}

#[test]
fn a_row_of_a_table_the_log_never_named_stops_the_open() {
    let data = row(vec![Value::Integer(2), Value::Integer(2)]);
    assert_stops(
        record(TXN, "gone", WALOperationType::Insert, data),
        &["Insert", "'gone'"],
    );
}

#[test]
fn a_delete_in_a_table_the_log_never_named_stops_the_open() {
    let data = row(vec![Value::Integer(1), Value::Integer(1)]);
    assert_stops(
        record(TXN, "gone", WALOperationType::Delete, data),
        &["Delete", "'gone'"],
    );
}

#[test]
fn an_unreadable_index_record_stops_the_open() {
    assert_stops(
        record(
            DDL_TXN_ID,
            "t",
            WALOperationType::CreateIndex,
            vec![1, 2, 3],
        ),
        &["CreateIndex"],
    );
}

#[test]
fn an_index_record_that_cannot_build_its_index_stops_the_open() {
    let meta = IndexMetadata {
        name: "t_n".to_string(),
        table_name: "t".to_string(),
        column_names: Vec::new(),
        column_ids: Vec::new(),
        data_types: Vec::new(),
        is_unique: true,
        index_type: IndexType::BTree,
        hnsw_m: None,
        hnsw_ef_construction: None,
        hnsw_ef_search: None,
        hnsw_distance_metric: None,
        identity: None,
    };
    assert_stops(
        record(
            DDL_TXN_ID,
            "t",
            WALOperationType::CreateIndex,
            meta.serialize(),
        ),
        &["CreateIndex", "column"],
    );
}

#[test]
fn an_unreadable_index_drop_stops_the_open() {
    assert_stops(
        record(
            DDL_TXN_ID,
            "t",
            WALOperationType::DropIndex,
            vec![0xFF, 0xFE],
        ),
        &["DropIndex"],
    );
}

#[test]
fn an_unreadable_view_drop_stops_the_open() {
    assert_stops(
        record(
            DDL_TXN_ID,
            "v",
            WALOperationType::DropView,
            vec![0xFF, 0xFE],
        ),
        &["DropView"],
    );
}

#[test]
fn an_unreadable_view_record_stops_the_open() {
    assert_stops(
        record(DDL_TXN_ID, "v", WALOperationType::CreateView, vec![1, 2, 3]),
        &["CreateView"],
    );
}

#[test]
fn a_record_whose_checksum_holds_but_does_not_decode_stops_the_open() {
    let dir = database();
    let mut entry = record(TXN, "t", WALOperationType::Insert, vec![1, 2, 3]);
    entry.lsn = u64::MAX / 2;
    let mut bytes = entry.encode();
    // The operation byte follows the header, txn id, name and row id
    let op = 32 + 8 + 2 + 1 + 8;
    bytes[op] = 0xEE;
    let end = bytes.len() - 4;
    let crc = crc32fast::hash(&bytes[32..end]);
    bytes[end..].copy_from_slice(&crc.to_le_bytes());
    let newest = std::fs::read_dir(dir.path().join("wal"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "log"))
        .max_by_key(|p| std::fs::metadata(p).unwrap().modified().unwrap())
        .unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(newest)
        .unwrap()
        .write_all(&bytes)
        .unwrap();
    let error = match Database::open(&dsn(dir.path())) {
        Ok(_) => panic!("the open passed over the record"),
        Err(e) => e.to_string(),
    };
    assert!(
        error.contains("cannot decode") && error.contains("LSN"),
        "{error}"
    );
}

/// Opens a crash image taken after one handle committed a write while
/// another ran `ddl` on its table
fn reopens_after(ddl: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
        "INSERT INTO t VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let other = db.clone();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (2, 2)", ()).unwrap();
    for sql in ddl {
        other.execute(sql, ()).unwrap();
    }
    let _ = tx.commit();
    let image = tempfile::tempdir().unwrap();
    copy_dir(dir.path(), image.path());
    if let Err(e) = Database::open(&dsn(image.path())) {
        panic!("the crash image of {ddl:?} does not open: {e}");
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn a_write_committed_across_a_drop_of_its_table_reopens() {
    reopens_after(&["DROP TABLE t"]);
}

#[test]
fn a_write_committed_across_a_rename_of_its_table_reopens() {
    reopens_after(&["ALTER TABLE t RENAME TO u"]);
}

#[test]
fn a_write_committed_across_a_drop_and_create_of_its_table_reopens() {
    reopens_after(&[
        "DROP TABLE t",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)",
    ]);
}

#[test]
fn a_write_committed_across_a_drop_rolled_back_reopens() {
    reopens_after(&["BEGIN", "DROP TABLE t", "ROLLBACK"]);
}
