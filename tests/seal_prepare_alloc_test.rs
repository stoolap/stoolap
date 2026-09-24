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

//! A seal's preparation against a small older volume walks the older side,
//! counted in the bytes it allocates on its thread

// The mimalloc feature sets the library's own global allocator
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};
use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
use stoolap::storage::volume::writer::{FrozenVolume, VolumeBuilder};

struct ThreadCounting;

thread_local! {
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + layout.size()));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + new_size));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

fn volume_of(ids: &[i64]) -> FrozenVolume {
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    for &id in ids {
        builder.add_row(id, &Row::from_values(vec![Value::Integer(id)]));
    }
    builder.finish().unwrap()
}

fn meta_for(id: u64, ids: &[i64]) -> SegmentMeta {
    SegmentMeta {
        segment_id: id,
        file_path: std::path::PathBuf::new(),
        row_count: ids.len(),
        min_row_id: *ids.iter().min().unwrap(),
        max_row_id: *ids.iter().max().unwrap(),
        creation_lsn: 0,
        seal_seq: 0,
        schema_version: 0,
    }
}

#[test]
fn a_preparation_walks_a_small_older_volume_instead_of_collecting_the_new_ids() {
    let dir = tempfile::tempdir().unwrap();
    // Read back from its file, as a reopen loads it, its order undecided
    let old_ids = [20_000i64, 1_000_000];
    let path = write_volume_to_disk(dir.path(), "t", 1, &volume_of(&old_ids)).unwrap();
    let old = Arc::new(read_volume_from_disk(&path).unwrap());
    let mgr = SegmentManager::new("t", None);
    mgr.register_segment(1, old, meta_for(1, &old_ids), None);

    let new_ids: Vec<i64> = (20_001..=120_000).collect();
    let new = Arc::new(volume_of(&new_ids));
    let _ = new.id_bounds();

    let before = ALLOCATED.with(Cell::get);
    let prepared = mgr.prepare_registration(std::slice::from_ref(&new));
    let allocated = ALLOCATED.with(Cell::get) - before;
    drop(prepared);
    assert!(
        allocated < 64 * 1024,
        "preparing 100,000 new rows against 2 older ones allocated {allocated} bytes"
    );
}
