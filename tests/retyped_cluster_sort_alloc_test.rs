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

//! A compaction ordering a clustered key whose type changed casts each key
//! cell once, not once per comparison, counted in allocations on its thread

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

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

const ROWS: i64 = 4096;

/// Allocations of the checkpoint that compacts three volumes of a
/// clustered table, the first two sealed with an INTEGER key; `retype`
/// changes the key to TEXT before the third
fn compaction_allocations(retype: bool) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?compact_threshold=2&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL) CLUSTER BY (k)",
        (),
    )
    .unwrap();
    for part in 0..2 {
        let base = part * ROWS;
        db.execute(
            &format!(
                "INSERT INTO t SELECT value, (value * 7919) % 1000003 FROM generate_series({}, {})",
                base + 1,
                base + ROWS
            ),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    if retype {
        db.execute("ALTER TABLE t MODIFY COLUMN k TEXT NOT NULL", ())
            .unwrap();
    }
    db.execute("INSERT INTO t VALUES (100000, 5)", ()).unwrap();
    let before = ALLOCATIONS.with(Cell::get);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let spent = ALLOCATIONS.with(Cell::get) - before;
    let volumes = db.query("PRAGMA VOLUME_STATS", ()).unwrap().count();
    assert_eq!(volumes, 1, "retype {retype}: the volumes did not compact");
    spent
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_retyped_key_is_cast_once_per_cell_when_compaction_sorts() {
    let plain = compaction_allocations(false);
    let retyped = compaction_allocations(true);
    let cells = 2 * ROWS as usize;
    eprintln!("allocations: retyped {retyped}, plain {plain}, {cells} retyped cells");
    // Each retyped cell casts to one string; a cast per comparison would
    // cost about log2(cells) more
    assert!(
        retyped <= plain + 4 * cells,
        "retyped {retyped}, plain {plain}, {cells} retyped cells"
    );
}
