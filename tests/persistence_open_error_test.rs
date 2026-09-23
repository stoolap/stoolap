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

//! A file database whose WAL cannot start does not open

use stoolap::Database;

fn dsn(dir: &std::path::Path) -> String {
    format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.display()
    )
}

/// A file where the WAL directory should be: the open fails, rather than
/// open a database whose writes nothing keeps
#[test]
fn a_database_whose_wal_cannot_start_does_not_open() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("wal"), b"not a directory").unwrap();
    let opened = Database::open(&dsn(dir.path()));
    assert!(opened.is_err(), "the open fails");
    drop(opened);
    // With the obstacle gone the database opens and keeps its writes
    std::fs::remove_file(dir.path().join("wal")).unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.close().unwrap();
    drop(db);
    let db = Database::open(&dsn(dir.path())).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 1);
    db.close().unwrap();
}
