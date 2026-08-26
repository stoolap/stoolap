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

//! Regression test for: warn-and-skip on an unreadable manifest at open.
//! An unsupported manifest version (e.g. a database directory written by a
//! newer stoolap) opened "successfully" with the table silently missing,
//! and the orphan reaper could then delete the table's intact .vol files.
//! Opening must fail closed instead.

use stoolap::Database;

#[test]
fn test_open_fails_closed_on_unsupported_manifest_version() {
    let dir = tempfile::tempdir().unwrap();
    let db_dir = dir.path().join("verdb");
    let dsn = format!("file://{}", db_dir.display());

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for i in 1..=100i64 {
        db.execute("INSERT INTO t (id, v) VALUES ($1, $2)", (i, i))
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();

    // Simulate a manifest written by a newer format: bump the version u32
    // at byte offset 4 (after the 4-byte magic) to 99.
    let manifest = db_dir.join("volumes").join("t").join("manifest.bin");
    let mut bytes = std::fs::read(&manifest).unwrap();
    bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(&manifest, &bytes).unwrap();

    match Database::open(&dsn) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("manifest"),
                "error should name the manifest, got: {}",
                msg
            );
        }
        Ok(db) => {
            let table_visible = db.query("SELECT COUNT(*) FROM t", ()).is_ok();
            panic!(
                "open must fail closed on unsupported manifest version \
                 (opened instead; table visible: {})",
                table_visible
            );
        }
    }
}
