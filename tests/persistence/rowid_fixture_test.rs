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

//! A database an older build left with a column named after the row
//! identity refuses to open. The fixture and how it was written are in
//! tests/fixtures/rowid

use std::path::Path;

use stoolap::Database;

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A copy of the fixture `name` in a directory of its own, and its DSN
fn copied(name: &str) -> (tempfile::TempDir, String) {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rowid")
        .join(name);
    let dir = tempfile::tempdir().unwrap();
    copy_tree(&fixture, dir.path());
    let dsn = format!("file://{}", dir.path().display());
    (dir, dsn)
}

fn pairs(db: &Database) -> Vec<(i64, i64)> {
    db.query("SELECT a, b FROM t ORDER BY a", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect()
}

#[test]
fn a_table_with_a_column_named_after_the_row_identity_refuses_to_open() {
    let (_dir, dsn) = copied("column_named_rowid");
    let message = match Database::open(&dsn) {
        Ok(_) => panic!("the database opened"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains("table t") && message.contains("row's identity"),
        "{message}"
    );
}

#[test]
fn a_snapshot_with_a_column_named_after_the_row_identity_restores_nothing() {
    let (_dir, dsn) = copied("snapshot_named_rowid");
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(pairs(&db), [(1, 10), (2, 20)]);
        let message = db.execute("PRAGMA RESTORE", ()).unwrap_err().to_string();
        assert!(message.contains("row's identity"), "{message}");
        assert_eq!(pairs(&db), [(1, 10), (2, 20)]);
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(pairs(&db), [(1, 10), (2, 20)]);
}
