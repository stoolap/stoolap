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

use std::path::PathBuf;
use std::sync::Arc;
use stoolap::core::{DataType, Error, Row, SchemaBuilder, Value};
use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};
use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
use stoolap::storage::volume::writer::VolumeBuilder;
use stoolap::Database;

struct Fixture {
    _dir: tempfile::TempDir,
    dsn: String,
    damaged_path: PathBuf,
    original: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!(
            "file://{}?checkpoint_on_close=off&checkpoint_interval=3600&cleanup_interval=3600",
            dir.path().display()
        );
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.close().unwrap();
        drop(db);
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("v", DataType::Integer, false, false)
            .build();
        let volume_dir = dir.path().join("volumes");
        let manager = SegmentManager::new("t", Some(volume_dir.clone()));
        let mut damaged_path = PathBuf::new();
        let mut original = Vec::new();
        for id in 1..=2 {
            let mut builder = VolumeBuilder::new(&schema);
            builder.add_row(
                id,
                &Row::from_values(vec![Value::Integer(id), Value::Integer(id * 10)]),
            );
            let volume = builder.finish();
            let path = write_volume_to_disk(&volume_dir, "t", id as u64, &volume).unwrap();
            if id == 2 {
                original = std::fs::read(&path).unwrap();
                let meta_len = u32::from_le_bytes(original[16..20].try_into().unwrap()) as usize;
                let mut bytes = original[..20 + meta_len].to_vec();
                for size in [9u64, 10] {
                    bytes.extend_from_slice(&size.to_le_bytes());
                    bytes.extend_from_slice(&size.to_le_bytes());
                }
                bytes.push(0);
                bytes.extend_from_slice(&id.to_le_bytes());
                bytes.extend_from_slice(&[0; 10]);
                bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
                std::fs::write(&path, bytes).unwrap();
                damaged_path = path.clone();
            }
            manager.register_segment(
                id as u64,
                Arc::new(read_volume_from_disk(&path).unwrap()),
                SegmentMeta {
                    segment_id: id as u64,
                    file_path: path.file_name().unwrap().into(),
                    row_count: 1,
                    min_row_id: id,
                    max_row_id: id,
                    creation_lsn: 0,
                    seal_seq: 0,
                    schema_version: 0,
                },
                Some(&schema),
            );
        }
        manager.persist().unwrap();
        Self {
            _dir: dir,
            dsn,
            damaged_path,
            original,
        }
    }

    fn open(&self) -> Database {
        Database::open(&self.dsn).unwrap()
    }

    fn assert_cold_unchanged(&self, db: Database) {
        db.close().unwrap();
        drop(db);
        std::fs::write(&self.damaged_path, &self.original).unwrap();
        let db = self.open();
        assert_eq!(
            pairs(&db, "SELECT * FROM t ORDER BY id"),
            [(1, 10), (2, 20)]
        );
        db.close().unwrap();
    }
}

fn assert_read_error(db: &Database, sql: &str) {
    let result = db
        .query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
    assert!(
        matches!(result, Err(Error::Internal { ref message })
            if message == "corrupt V4 block: invalid column block length"
                || message == "scan error: corrupt V4 block: invalid column block length"),
        "{sql}: {result:?}"
    );
}

fn add_target(db: &Database) {
    db.execute(
        "CREATE TABLE target (id INTEGER PRIMARY KEY, v INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    begin_target_update(db);
}

fn begin_target_update(db: &Database) {
    db.execute("INSERT INTO target VALUES (1, 100), (2, 200)", ())
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE target SET v = 101 WHERE id = 1", ())
        .unwrap();
}

fn assert_target_unchanged(db: &Database) {
    assert_eq!(
        pairs(db, "SELECT * FROM target ORDER BY id"),
        [(1, 101), (2, 200)]
    );
}

fn pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect()
}

#[test]
fn unread_corrupt_volume_does_not_fail_pruned_query() {
    let fixture = Fixture::new();
    let db = fixture.open();
    assert_eq!(pairs(&db, "SELECT * FROM t WHERE id = 1"), [(1, 10)]);
    fixture.assert_cold_unchanged(db);
}

#[test]
fn parallel_filter_propagates_late_scan_error() {
    let fixture = Fixture::new();
    let db = fixture.open();
    assert_read_error(
        &db,
        "SELECT v FROM t WHERE id > 0 AND RANDOM() >= 0 ORDER BY id",
    );
    fixture.assert_cold_unchanged(db);
}

#[test]
fn correlated_filter_propagates_late_scan_error() {
    let fixture = Fixture::new();
    let db = fixture.open();
    add_target(&db);
    assert_read_error(&db, "SELECT v FROM t WHERE (SELECT target.v FROM target WHERE target.id = t.id) > 0 ORDER BY id");
    db.execute("COMMIT", ()).unwrap();
    fixture.assert_cold_unchanged(db);
}

#[test]
fn delete_returning_does_not_delete_readable_prefix() {
    let fixture = Fixture::new();
    let db = fixture.open();
    assert_read_error(&db, "DELETE FROM t RETURNING id");
    fixture.assert_cold_unchanged(db);
}

#[test]
fn correlated_update_precompute_propagates_late_scan_error() {
    let fixture = Fixture::new();
    let db = fixture.open();
    add_target(&db);
    assert_read_error(
        &db,
        "UPDATE t SET v = (SELECT target.v FROM target WHERE target.id = t.id) WHERE id = 1",
    );
    assert_target_unchanged(&db);
    db.execute("COMMIT", ()).unwrap();
    fixture.assert_cold_unchanged(db);
}

#[test]
fn correlated_set_error_preserves_prior_statement() {
    let fixture = Fixture::new();
    let db = fixture.open();
    add_target(&db);
    assert_read_error(
        &db,
        "UPDATE target SET v = (SELECT t.v FROM t WHERE t.id = target.id)",
    );
    assert_target_unchanged(&db);
    db.execute("COMMIT", ()).unwrap();
    assert_target_unchanged(&db);
    fixture.assert_cold_unchanged(db);
}

#[test]
fn correlated_update_filter_error_preserves_prior_statement() {
    let fixture = Fixture::new();
    let db = fixture.open();
    db.execute("CREATE TABLE target (id INTEGER, v INTEGER UNIQUE)", ())
        .unwrap();
    begin_target_update(&db);
    assert_read_error(
        &db,
        "UPDATE target SET v = v + 1 WHERE (SELECT t.v FROM t WHERE t.id = target.id) > 0",
    );
    assert_target_unchanged(&db);
    db.execute("COMMIT", ()).unwrap();
    assert_target_unchanged(&db);
    fixture.assert_cold_unchanged(db);
}

#[test]
fn correlated_delete_filter_error_preserves_prior_statement() {
    let fixture = Fixture::new();
    let db = fixture.open();
    add_target(&db);
    assert_read_error(
        &db,
        "DELETE FROM target WHERE (SELECT t.v FROM t WHERE t.id = target.id) > 0",
    );
    assert_target_unchanged(&db);
    db.execute("COMMIT", ()).unwrap();
    assert_target_unchanged(&db);
    fixture.assert_cold_unchanged(db);
}

#[test]
fn correlated_precompute_filter_error_preserves_prior_statement() {
    let fixture = Fixture::new();
    let db = fixture.open();
    add_target(&db);
    assert_read_error(&db, "UPDATE target SET v = (SELECT t.id FROM t WHERE t.id = target.id) WHERE (SELECT t.v FROM t WHERE t.id = target.id) > 0");
    assert_target_unchanged(&db);
    db.execute("COMMIT", ()).unwrap();
    fixture.assert_cold_unchanged(db);
}

#[test]
fn correlated_fk_precheck_reports_read_error_before_set_error() {
    let fixture = Fixture::new();
    let db = fixture.open();
    db.execute("CREATE TABLE target (id INTEGER, v INTEGER UNIQUE)", ())
        .unwrap();
    db.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, parent_v INTEGER REFERENCES target(v) ON UPDATE RESTRICT)", ()).unwrap();
    begin_target_update(&db);
    assert_read_error(
        &db,
        "UPDATE target SET v = 'bad' WHERE (SELECT t.v FROM t WHERE t.id = target.id + 0) > 0",
    );
    assert_target_unchanged(&db);
    db.execute("COMMIT", ()).unwrap();
    fixture.assert_cold_unchanged(db);
}

fn assert_exists_optimizer_error(sql: &str) {
    let fixture = Fixture::new();
    let db = fixture.open();
    db.execute(
        "CREATE TABLE target (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO target VALUES (1, 10), (2, 10)", ())
        .unwrap();
    assert_read_error(&db, sql);
    assert_eq!(
        pairs(&db, "SELECT * FROM target ORDER BY id"),
        [(1, 10), (2, 10)]
    );
    fixture.assert_cold_unchanged(db);
}

#[test]
fn select_preserves_exists_optimizer_error() {
    assert_exists_optimizer_error(
        "SELECT id FROM target WHERE EXISTS (SELECT 1 FROM t WHERE t.v = target.v)",
    );
}

#[test]
fn update_preserves_exists_optimizer_error() {
    assert_exists_optimizer_error(
        "UPDATE target SET v = v + 1 WHERE EXISTS (SELECT 1 FROM t WHERE t.v = target.v)",
    );
}

#[test]
fn delete_preserves_exists_optimizer_error() {
    assert_exists_optimizer_error(
        "DELETE FROM target WHERE EXISTS (SELECT 1 FROM t WHERE t.v = target.v)",
    );
}
