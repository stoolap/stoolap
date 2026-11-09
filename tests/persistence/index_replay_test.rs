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

//! An UPDATE of an indexed column replayed from the log moves the row's key
//! in every kind of index: the old key no longer holds the row

use std::path::Path;

use stoolap::Database;

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn a_replayed_update_frees_the_old_key_of_a_unique_index() {
    let mut wrong = Vec::new();
    for index in [
        "CREATE UNIQUE INDEX ix ON t (v) USING BTREE",
        "CREATE UNIQUE INDEX ix ON t (v) USING HASH",
        "CREATE UNIQUE INDEX ix ON t (v) USING BITMAP",
        "CREATE UNIQUE INDEX ix ON t (v, w)",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let image = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(dir.path())).unwrap();
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER)",
            index,
            "INSERT INTO t VALUES (1, 10, 1), (2, 20, 1)",
            "UPDATE t SET v = 30 WHERE id = 1",
        ] {
            db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        }
        crash_image(dir.path(), image.path());
        drop(db);

        let db = Database::open(&dsn(image.path())).unwrap();
        if let Err(e) = db.execute("INSERT INTO t VALUES (3, 10, 1)", ()) {
            wrong.push(format!("{index}: the old key still holds row 1: {e}"));
        }
        match db.execute("INSERT INTO t VALUES (4, 30, 1)", ()) {
            Err(e) if e.to_string().contains("unique") => {}
            other => wrong.push(format!(
                "{index}: the new key does not hold row 1: {other:?}"
            )),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
