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

//! Eviction drops a volume's blocks when it holds them in memory, and
//! leaves a volume that reads its blocks from its file at warm

use std::path::PathBuf;
use std::sync::Arc;

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::volume::io::{
    read_volume_from_disk, serialize_v4_public, write_volume_to_disk,
};
use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
use stoolap::storage::volume::writer::{FrozenVolume, VolumeBuilder};

/// The volume left after eviction runs for every tier a volume can fall
fn evicted(
    volume: FrozenVolume,
    dir: &std::path::Path,
    schema: &stoolap::Schema,
) -> Arc<FrozenVolume> {
    let manager = SegmentManager::new("t", Some(dir.to_path_buf()));
    manager.register_segment(
        1,
        Arc::new(volume),
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
        Some(schema),
    );
    for epoch in [0, 3, 6, 9, 12] {
        manager.evict_idle_volumes(epoch, false);
    }
    Arc::clone(&manager.segments_raw()[&1].volume)
}

fn built(dir: &std::path::Path) -> (FrozenVolume, PathBuf, stoolap::Schema) {
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
    let volume = builder.finish().unwrap();
    let path = write_volume_to_disk(dir, "t", 1, &volume).unwrap();
    (volume, path, schema)
}

#[test]
fn a_volume_holding_its_blocks_in_memory_goes_cold() {
    let dir = tempfile::tempdir().unwrap();
    let (mut volume, _, schema) = built(dir.path());
    let (_, store) = serialize_v4_public(&volume).unwrap();
    volume.columns.attach_compressed_store(store);
    assert!(evicted(volume, dir.path(), &schema).is_cold());
}

#[test]
fn a_volume_reading_its_file_stops_at_warm() {
    let dir = tempfile::tempdir().unwrap();
    let (_, path, schema) = built(dir.path());
    let volume = read_volume_from_disk(&path).unwrap();
    assert!(volume.columns.compressed_store().unwrap().is_file_backed());
    let volume = evicted(volume, dir.path(), &schema);
    assert!(!volume.is_cold(), "a file-backed volume is never made cold");
    assert!(volume.is_warm());
}

#[test]
fn a_file_volume_drops_a_decoded_column_when_idle() {
    let dir = tempfile::tempdir().unwrap();
    let (_, path, schema) = built(dir.path());
    let volume = read_volume_from_disk(&path).unwrap();
    volume.columns.get(1).unwrap();
    assert!(volume.columns.resident(1).is_some());
    assert!(volume.is_warm(), "one decoded column of two leaves it warm");
    let volume = evicted(volume, dir.path(), &schema);
    assert!(
        volume.columns.resident(1).is_none(),
        "the idle decoded column is dropped"
    );
    assert!(volume.columns.compressed_store().unwrap().is_file_backed());
}

fn unique_indexes(volume: &FrozenVolume) -> usize {
    volume.unique_indices.read().len()
}

#[test]
fn an_idle_volume_lets_go_of_its_unique_index() {
    let dir = tempfile::tempdir().unwrap();
    let (volume, _, schema) = built(dir.path());
    volume.prebuild_unique_index(&[1]).unwrap();
    assert_eq!(unique_indexes(&volume), 1);
    let volume = evicted(volume, dir.path(), &schema);
    assert_eq!(unique_indexes(&volume), 0, "the idle volume kept its index");
    let mut found = Vec::new();
    volume
        .unique_lookup_all(&[1], &[&Value::Integer(20)], |row| {
            found.push(row);
            false
        })
        .unwrap();
    assert_eq!(found, vec![1], "the index built again answers");
}

#[test]
fn an_idle_file_volume_lets_go_of_its_unique_index() {
    let dir = tempfile::tempdir().unwrap();
    let (_, path, schema) = built(dir.path());
    let volume = read_volume_from_disk(&path).unwrap();
    volume.prebuild_unique_index(&[1]).unwrap();
    let volume = evicted(volume, dir.path(), &schema);
    assert_eq!(
        unique_indexes(&volume),
        0,
        "the idle file volume kept its index"
    );
}

#[test]
fn a_volume_a_lookup_reaches_keeps_its_unique_index() {
    let dir = tempfile::tempdir().unwrap();
    let (volume, _, schema) = built(dir.path());
    volume.prebuild_unique_index(&[1]).unwrap();
    let manager = SegmentManager::new("t", Some(dir.path().to_path_buf()));
    manager.register_segment(
        1,
        Arc::new(volume),
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
    for epoch in [0, 3, 6, 9, 12] {
        manager.segments_raw()[&1]
            .volume
            .unique_lookup_all(&[1], &[&Value::Integer(20)], |_| true)
            .unwrap();
        manager.evict_idle_volumes(epoch, false);
    }
    assert_eq!(
        unique_indexes(&manager.segments_raw()[&1].volume),
        1,
        "a volume in use lost its index"
    );
}

#[test]
fn a_lookup_through_a_captured_view_keeps_the_shared_index() {
    use std::sync::atomic::Ordering;
    let dir = tempfile::tempdir().unwrap();
    let (_, path, schema) = built(dir.path());
    let volume = Arc::new(read_volume_from_disk(&path).unwrap());
    volume.prebuild_unique_index(&[1]).unwrap();
    let manager = SegmentManager::new("t", Some(dir.path().to_path_buf()));
    manager.register_segment(
        1,
        Arc::clone(&volume),
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
    let snapshot = manager.statement_snapshot();
    let epoch = volume.last_access_epoch.load(Ordering::Relaxed) + 3;
    manager.evict_idle_volumes(epoch, false);
    let replacement = Arc::clone(&manager.segments_raw()[&1].volume);
    assert!(
        !Arc::ptr_eq(&volume, &replacement),
        "the volume was not rewarmed"
    );
    assert!(Arc::ptr_eq(
        &volume.unique_indices,
        &replacement.unique_indices
    ));
    for epoch in epoch + 3..epoch + 9 {
        assert_eq!(
            manager
                .find_row_id_by_values_in(&snapshot, &[1], &[&Value::Integer(20)], &[])
                .unwrap(),
            Some(2)
        );
        manager.evict_idle_volumes(epoch, false);
        assert_eq!(
            unique_indexes(&replacement),
            1,
            "eviction dropped an index a captured view is using"
        );
    }
}

#[test]
fn a_file_volume_forgets_a_failed_read_when_idle() {
    let dir = tempfile::tempdir().unwrap();
    let (_, path, schema) = built(dir.path());
    let volume = read_volume_from_disk(&path).unwrap();
    let away = path.with_extension("away");
    std::fs::rename(&path, &away).unwrap();
    assert!(volume.columns.get(1).is_err());
    std::fs::rename(&away, &path).unwrap();
    assert!(volume.columns.get(1).is_err(), "the failure is cached");
    let volume = evicted(volume, dir.path(), &schema);
    assert!(
        volume.columns.get(1).is_ok(),
        "eviction clears the cached failure"
    );
    assert!(volume.columns.compressed_store().unwrap().is_file_backed());
}
