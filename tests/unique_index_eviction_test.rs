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

//! Eviction releases unique indexes by their own database's idle cycles,
//! and an older pass never releases an index a newer one saw in use

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
use stoolap::storage::volume::writer::{VolumeBuilder, GLOBAL_EVICTION_EPOCH};

// The global epoch stamps every new volume: tests here change it in turn
static GLOBAL_EPOCH: Mutex<()> = Mutex::new(());

struct RestoreEpoch(u64);

impl Drop for RestoreEpoch {
    fn drop(&mut self) {
        GLOBAL_EVICTION_EPOCH.store(self.0, Ordering::Relaxed);
    }
}

fn manager_with_indexes(count: u64) -> Arc<SegmentManager> {
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("v", DataType::Integer, false, false)
        .build();
    let manager = Arc::new(SegmentManager::new("t", None));
    for segment in 1..=count {
        let mut builder = VolumeBuilder::new(&schema);
        for id in [1, 2] {
            builder.add_row(
                id,
                &Row::from_values(vec![Value::Integer(id), Value::Integer(id * 10)]),
            );
        }
        let volume = builder.finish().unwrap();
        volume.prebuild_unique_index(&[1]).unwrap();
        manager.register_segment(
            segment,
            Arc::new(volume),
            SegmentMeta {
                segment_id: segment,
                file_path: PathBuf::from("unused.vol"),
                row_count: 2,
                min_row_id: 1,
                max_row_id: 2,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            Some(&schema),
        );
    }
    manager
}

#[test]
fn unused_indexes_go_after_this_databases_idle_cycles() {
    let _serial = GLOBAL_EPOCH.lock().unwrap();
    let _restore = RestoreEpoch(GLOBAL_EVICTION_EPOCH.swap(100, Ordering::Relaxed));
    let manager = manager_with_indexes(1);
    let volume = Arc::clone(&manager.segments_raw()[&1].volume);
    volume.mark_accessed();
    for epoch in 1..=4 {
        manager.evict_idle_volumes(epoch, false);
    }
    assert_eq!(
        volume.unique_indices.read().len(),
        0,
        "the index waited for another database's epochs"
    );
}

#[test]
fn an_older_pass_keeps_an_index_a_newer_pass_saw_in_use() {
    let _serial = GLOBAL_EPOCH.lock().unwrap();
    let _restore = RestoreEpoch(GLOBAL_EVICTION_EPOCH.swap(0, Ordering::Relaxed));
    let manager = manager_with_indexes(2);
    manager.evict_idle_volumes(1, false);
    let segments = manager.segments_raw();
    let mut volumes = segments.iter();
    let (&blocker_id, blocker) = volumes.next().unwrap();
    let target = Arc::clone(&volumes.next().unwrap().1.volume);
    // The older pass stops at the blocker's release with the target picked
    let blocker_guard = blocker.volume.unique_indices.read();
    let owners = Arc::strong_count(&target.unique_indices);
    let older = Arc::clone(&manager);
    let older = std::thread::spawn(move || older.evict_idle_volumes(4, false));
    let deadline = Instant::now() + Duration::from_secs(5);
    while Arc::strong_count(&target.unique_indices) == owners {
        assert!(
            Instant::now() < deadline,
            "the older pass did not pick the target"
        );
        std::thread::yield_now();
    }
    manager.remove_segments(&[blocker_id]);
    target
        .unique_lookup_all(&[1], &[&Value::Integer(20)], |_| true)
        .unwrap();
    manager.evict_idle_volumes(5, false);
    drop(blocker_guard);
    older.join().unwrap();
    assert_eq!(
        target.unique_indices.read().len(),
        1,
        "the older pass released an index used since it decided"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_release_after_a_read_took_the_index_keeps_it() {
    use stoolap::storage::volume::writer::UniqueCandidates;
    let _serial = GLOBAL_EPOCH.lock().unwrap();
    let _restore = RestoreEpoch(GLOBAL_EVICTION_EPOCH.swap(0, Ordering::Relaxed));
    let manager = manager_with_indexes(1);
    manager.evict_idle_volumes(1, false);
    let volume = Arc::clone(&manager.segments_raw()[&1].volume);
    let evicting = Arc::clone(&manager);
    stoolap::test_failpoints::on_unique_index_taken(move || {
        evicting.evict_idle_volumes(4, false);
    });
    let found = volume.built_unique_candidates(&[1], &[&Value::Integer(20)], 10);
    assert!(
        matches!(found, UniqueCandidates::Positions(ref positions) if positions == &[1]),
        "the read did not find its row"
    );
    assert_eq!(
        volume.unique_indices.read().len(),
        1,
        "a release after the read took the index dropped it"
    );
}
