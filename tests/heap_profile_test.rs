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

//! PRAGMA heap_profile writes a gzipped pprof heap profile with
//! heap-profile, unless dhat-heap takes the allocator, and refuses in every
//! other build

use stoolap::Database;

#[cfg(all(feature = "heap-profile", not(feature = "dhat-heap")))]
#[test]
fn a_heap_profile_is_written_in_pprof_format() {
    use std::io::Read;

    let db = Database::open("memory://heap_profile_written").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    let values: Vec<String> = (1..=20_000).map(|id| format!("({id},'v{id}')")).collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heap.pb.gz");
    let bytes: i64 = db
        .query_one(&format!("PRAGMA heap_profile = '{}'", path.display()), ())
        .unwrap();
    let file = std::fs::read(&path).unwrap();
    assert_eq!(bytes as usize, file.len());
    assert_eq!(&file[..2], &[0x1f, 0x8b], "a gzipped profile");
    let mut profile = Vec::new();
    flate2::read::GzDecoder::new(&file[..])
        .read_to_end(&mut profile)
        .unwrap();
    let text = String::from_utf8_lossy(&profile);
    assert!(text.contains("inuse_space"), "a pprof heap profile");
}

#[cfg(not(all(feature = "heap-profile", not(feature = "dhat-heap"))))]
#[test]
fn a_heap_profile_needs_the_heap_profile_feature() {
    let db = Database::open("memory://heap_profile_refused").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heap.pb.gz");
    let error = db
        .execute(&format!("PRAGMA heap_profile = '{}'", path.display()), ())
        .unwrap_err();
    assert!(error.to_string().contains("heap-profile"), "{error}");
    assert!(!path.exists());
}
