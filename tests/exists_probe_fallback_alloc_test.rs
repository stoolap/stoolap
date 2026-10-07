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

//! Clearing every thread-local cache releases the semi-join set a refused
//! EXISTS probe was answered from

// The library's own allocator features set a global allocator of their own
#![cfg(not(any(feature = "mimalloc", feature = "heap-profile", feature = "dhat-heap")))]

use stoolap::Database;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[test]
fn clearing_all_caches_releases_the_probe_fallback_set() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t SELECT value, CASE WHEN value % 10 = 0 THEN -1 ELSE value END \
         FROM generate_series(1, 100000)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, -1)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    let profiler = dhat::Profiler::builder().testing().build();
    let found: Vec<i64> = db
        .query(
            "SELECT o.id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.k = o.k) LIMIT 10",
            (),
        )
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(found, [1]);
    stoolap::executor::clear_all_thread_local_caches();
    let cleared = dhat::HeapStats::get().curr_bytes;
    db.query("SELECT 1", ())
        .unwrap()
        .for_each(|r| drop(r.unwrap()));
    let next_select = dhat::HeapStats::get().curr_bytes;
    drop(profiler);
    assert!(
        cleared <= next_select + 64 * 1024,
        "clear_all kept {} bytes the next statement released",
        cleared.saturating_sub(next_select)
    );
    db.execute("ROLLBACK", ()).unwrap();
}
