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

//! A sealed aggregate that walks a volume's side file allocates per volume
//! no more than its predicates and its calls need

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use stoolap::Database;

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the counter never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

const PER_VOLUME: i64 = 2000;

/// The side reader, one held group per predicate and call, and the calls
const PER_WALKED_VOLUME: usize = 20;

/// Allocations of each query over `volumes` sealed volumes, every key in every volume
fn allocations(dir: &std::path::Path, volumes: i64, queries: &[&str]) -> Vec<usize> {
    let dsn = format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, p INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_v ON t (v)", ()).unwrap();
    for vol in 0..volumes {
        let values: Vec<String> = (1..=PER_VOLUME)
            .map(|n| {
                let id = vol * PER_VOLUME + n;
                format!("({id}, {}, {})", (n * 7919) % 100, id % 10 + 1)
            })
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.close().unwrap();
    // Reopened, the volumes are read from their files
    let db = Database::open(&dsn).unwrap();
    queries
        .iter()
        .map(|sql| {
            let mut counted = usize::MAX;
            for key in 0..4i64 {
                let before = ALLOCATIONS.load(Ordering::Relaxed);
                let sql = sql.replace("$1", &key.to_string());
                let n: i64 = db.query_one(&sql, ()).unwrap();
                let taken = ALLOCATIONS.load(Ordering::Relaxed) - before;
                assert!(n > 0, "{sql} with {key} answered {n}");
                // The first query opens the side files; the least of the rest counts
                if key > 0 {
                    counted = counted.min(taken);
                }
            }
            counted
        })
        .collect()
}

#[test]
fn a_sealed_side_walk_allocates_per_volume_no_more_than_its_inputs() {
    let queries = [
        "SELECT COUNT(*) FROM t WHERE v = $1",
        "SELECT SUM(p) FROM t WHERE v = $1",
    ];
    let two = allocations(tempfile::tempdir().unwrap().path(), 2, &queries);
    let six = allocations(tempfile::tempdir().unwrap().path(), 6, &queries);
    for ((sql, two), six) in queries.iter().zip(&two).zip(&six) {
        let per_volume = six.saturating_sub(*two) / 4;
        assert!(
            per_volume < PER_WALKED_VOLUME,
            "{sql}: {per_volume} allocations per walked volume (2 volumes {two}, 6 volumes {six})"
        );
    }
}
