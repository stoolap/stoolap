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

//! PRAGMA MEMORY_STATS counts the unique indexes a sealed volume holds

use stoolap::Database;
use tempfile::tempdir;

const ROWS: i64 = 10_000;

fn volume_bytes(db: &Database, table: &str) -> i64 {
    for row in db.query("PRAGMA MEMORY_STATS", ()).expect("memory stats") {
        let row = row.expect("row");
        let name: String = row.get(0).expect("table_name");
        if name == table {
            return row.get(6).expect("volume_bytes");
        }
    }
    panic!("PRAGMA MEMORY_STATS has no row for {table}");
}

fn fill(db: &Database, table: &str) {
    let values: Vec<String> = (0..ROWS).map(|i| format!("({i}, {})", i * 7)).collect();
    db.execute(
        &format!("INSERT INTO {table} VALUES {}", values.join(",")),
        (),
    )
    .expect("insert");
}

#[test]
fn a_sealed_unique_index_is_counted_in_volume_bytes() {
    let dir = tempdir().expect("tempdir");
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .expect("open");
    db.execute(
        "CREATE TABLE plain (id INTEGER PRIMARY KEY, code INTEGER NOT NULL)",
        (),
    )
    .expect("create plain");
    db.execute(
        "CREATE TABLE keyed (id INTEGER PRIMARY KEY, code INTEGER NOT NULL UNIQUE)",
        (),
    )
    .expect("create keyed");
    fill(&db, "plain");
    fill(&db, "keyed");
    db.execute("PRAGMA CHECKPOINT", ()).expect("checkpoint");
    let plain = volume_bytes(&db, "plain");
    assert!(plain > 0, "the plain table sealed into a volume");
    assert_eq!(
        volume_bytes(&db, "keyed") - plain,
        ROWS * 16,
        "the unique index adds 16 bytes per row"
    );
}
