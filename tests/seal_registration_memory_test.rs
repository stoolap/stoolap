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

//! A seal over a large sealed table keeps no memory sized to the rows
//! already sealed, counted in the live bytes of the whole process

// The mimalloc feature sets the library's own global allocator
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering};

use stoolap::Database;

struct Counting;

static LIVE: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the counter is an atomic
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as i64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(new_size as i64 - layout.size() as i64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

#[test]
fn a_seal_over_many_sealed_rows_keeps_no_memory_sized_to_them() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let insert = db.prepare("INSERT INTO t VALUES ($1, $1)").unwrap();
    db.execute("BEGIN", ()).unwrap();
    for id in 1..=300_000i64 {
        insert.execute((id,)).unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    for id in 300_001..=301_000i64 {
        insert.execute((id,)).unwrap();
    }
    db.execute("COMMIT", ()).unwrap();

    let before = LIVE.load(Ordering::Relaxed);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let kept = LIVE.load(Ordering::Relaxed) - before;
    assert_eq!(
        db.engine().volume_stats().len(),
        2,
        "the second seal landed"
    );
    assert!(
        kept < 1024 * 1024,
        "a seal of 1,000 rows over 300,000 sealed ones kept {kept} bytes"
    );
}
