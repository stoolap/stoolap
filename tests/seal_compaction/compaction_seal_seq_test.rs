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

//! A compacted volume carries the newest seal sequence of its inputs, which
//! bounds the commits of every row it holds

#![cfg(feature = "test-failpoints")]

use std::path::Path;

use stoolap::storage::volume::manifest::SegmentManager;
use stoolap::Database;

fn seal_seqs(dir: &Path) -> Vec<u64> {
    let mgr = SegmentManager::load_from_disk("t", &dir.join("volumes"))
        .unwrap()
        .unwrap();
    let seqs = mgr.manifest().segments.iter().map(|m| m.seal_seq).collect();
    seqs
}

#[test]
fn a_compacted_volume_keeps_the_newest_seal_of_its_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=2",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    // Two volumes stay below the threshold, so the checkpoints compact nothing
    for id in 1..=2 {
        db.execute(&format!("INSERT INTO t VALUES ({id}, {id})"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let sealed = seal_seqs(dir.path());
    assert_eq!(sealed.len(), 2, "{sealed:?}");
    assert!(sealed.iter().all(|&seq| seq > 0), "{sealed:?}");
    db.execute("INSERT INTO t VALUES (3, 3)", ()).unwrap();
    db.engine().seal_for_test().unwrap();
    db.engine().compact_for_test().unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let compacted = seal_seqs(dir.path());
    // The third volume was sealed last, so its sequence is the newest
    assert_eq!(compacted.len(), 1, "{compacted:?}");
    assert!(
        compacted[0] > *sealed.iter().max().unwrap(),
        "{compacted:?} {sealed:?}"
    );
}
