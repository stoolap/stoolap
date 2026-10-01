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

//! A volume writer's unique set that indexes no row allocates nothing per
//! batch, however many columns it spans

// The mimalloc and jemalloc-prof features set the library's own global allocator
#![cfg(not(any(feature = "mimalloc", feature = "jemalloc-prof")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, SchemaBuilder};
use stoolap::storage::volume::output::VolumeFileWriter;
use stoolap::storage::volume::writer::TypedCells;

struct ThreadCounting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

const BATCH: usize = 4096;
const GROUP: usize = 65_536;

/// Allocations of the batches inside the second row group, after the
/// first group's flush sized the writer's buffers, over columns 1..=width
/// with the last of them null in every row, indexed as one unique set or not
fn second_group_allocations(width: usize, indexed: bool) -> usize {
    let mut builder = SchemaBuilder::new("t").column("id", DataType::Integer, false, true);
    for c in 1..=width {
        builder = builder.column(&format!("c{c}"), DataType::Integer, true, false);
    }
    let schema = builder.build();
    let rows = 2 * GROUP;
    let ids: Vec<i64> = (0..rows as i64).collect();
    let none = vec![false; rows];
    let all = vec![true; rows];
    let dir = tempfile::tempdir().unwrap();
    let mut writer = VolumeFileWriter::new(dir.path(), "t", 1, &schema, rows, false).unwrap();
    if indexed {
        writer.index_unique_sets(vec![(1..=width).collect()]);
    }
    let mut counted = 0;
    for start in (0..rows - BATCH).step_by(BATCH) {
        let end = start + BATCH;
        let columns: Vec<TypedCells<'_>> = (0..=width)
            .map(|c| TypedCells::Int64 {
                values: &ids[start..end],
                nulls: if c == width {
                    &all[start..end]
                } else {
                    &none[start..end]
                },
            })
            .collect();
        let before = ALLOCATIONS.with(Cell::get);
        writer.append_typed(&ids[start..end], &columns).unwrap();
        if start >= GROUP {
            counted += ALLOCATIONS.with(Cell::get) - before;
        }
    }
    writer.abort();
    counted
}

#[test]
fn a_unique_set_without_indexed_rows_allocates_nothing_per_batch() {
    for width in 1..=6 {
        assert_eq!(
            second_group_allocations(width, true),
            second_group_allocations(width, false),
            "a {width}-column set allocated inside the second group"
        );
    }
}
