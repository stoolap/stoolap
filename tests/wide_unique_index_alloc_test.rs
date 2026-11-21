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

//! A unique index allocates by the row group, not by the row: a writer
//! indexing a key of many columns, and a build over decoded columns

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::volume::column::ROW_GROUP_SIZE;
use stoolap::storage::volume::output::VolumeFileWriter;
use stoolap::storage::volume::writer::{TypedCells, VolumeBuilder};

struct ThreadCounting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counters are const
    // thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        BYTES.with(|b| b.set(b.get() + layout.size()));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        BYTES.with(|b| b.set(b.get() + new_size));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

const ROWS: usize = 131_072;
const BATCH: usize = 4_096;

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

/// Allocations while `ROWS` rows of six integer columns are appended, two
/// row groups flushed, with the unique column sets `sets` indexed
fn appending(sets: Vec<Vec<usize>>) -> usize {
    let mut schema = SchemaBuilder::new("t").column("id", DataType::Integer, false, true);
    for name in ["a", "b", "c", "d", "e"] {
        schema = schema.column(name, DataType::Integer, false, false);
    }
    let schema = schema.build();
    let ids: Vec<i64> = (0..ROWS as i64).collect();
    let nulls = vec![false; ROWS];
    let dir = tempfile::tempdir().unwrap();
    let mut writer = VolumeFileWriter::new(dir.path(), "t", 1, &schema, ROWS, true).unwrap();
    writer.index_unique_sets(sets);
    let before = allocations();
    for start in (0..ROWS).step_by(BATCH) {
        let end = start + BATCH;
        let columns: Vec<TypedCells<'_>> = (0..6)
            .map(|_| TypedCells::Int64 {
                values: &ids[start..end],
                nulls: &nulls[start..end],
            })
            .collect();
        writer.append_typed(&ids[start..end], &columns).unwrap();
    }
    let allocated = allocations() - before;
    writer.finish().unwrap();
    allocated
}

#[test]
fn a_build_over_decoded_columns_holds_a_group_of_hashes() {
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("a", DataType::Integer, false, false)
        .column("b", DataType::Integer, false, false)
        .column("c", DataType::Integer, false, false)
        .build();
    let rows = 3 * ROW_GROUP_SIZE;
    let mut builder = VolumeBuilder::new(&schema);
    for id in 0..rows as i64 {
        let cells = [id, id, id * 3, id * 7].map(Value::Integer);
        builder.add_row(id, &Row::from_values(cells.to_vec()));
    }
    let volume = builder.finish().unwrap();
    let before = BYTES.with(Cell::get);
    volume.prebuild_unique_index(&[1, 2, 3]).unwrap();
    let bytes = BYTES.with(Cell::get) - before;
    // The entries and the sort's buffer, a group's hashes and NULL flags
    let bound = 2 * rows * 8 + ROW_GROUP_SIZE * (3 * 8 + 1) + 256 * 1024;
    assert!(
        bytes <= bound,
        "{bytes} bytes allocated building over {rows} decoded rows, at most {bound}"
    );
}

#[test]
fn a_wide_unique_key_allocates_by_the_group() {
    let plain = appending(Vec::new());
    let keyed = appending(vec![vec![1, 2, 3, 4, 5]]);
    assert!(
        keyed <= plain + 64,
        "a five-column key: {keyed} allocations for {ROWS} rows, {plain} without it"
    );
}
