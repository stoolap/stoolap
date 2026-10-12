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

//! A plain INSERT refused by a unique key names the key's columns and
//! values, whether the row it collides with is hot or in a volume

use stoolap::Database;

/// The message of the error `insert` gets the second time it runs, with the
/// first row still hot or sealed into a volume
fn conflict_message(cold: bool, ddl: &str, insert: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.path().display()
    ))
    .unwrap();
    db.execute(ddl, ()).unwrap();
    db.execute(insert, ()).unwrap();
    if cold {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(db.engine().volume_stats().len(), 1);
    }
    db.execute(insert, ()).unwrap_err().to_string()
}

fn assert_names(message: &str, columns: &str, values: &str) {
    assert!(
        message.ends_with(&format!("on column {columns} with value {values}")),
        "{message}"
    );
}

#[test]
fn a_single_column_key_is_named_with_its_value() {
    for cold in [false, true] {
        let message = conflict_message(
            cold,
            "CREATE TABLE u (id INTEGER PRIMARY KEY AUTO_INCREMENT, code INTEGER UNIQUE)",
            "INSERT INTO u (code) VALUES (7)",
        );
        assert_names(&message, "code", "7");
    }
}

#[test]
fn a_key_of_several_columns_names_each_value_in_order() {
    for cold in [false, true] {
        let message = conflict_message(
            cold,
            "CREATE TABLE m (id INTEGER PRIMARY KEY AUTO_INCREMENT, ts TIMESTAMP NOT NULL, \
             name TEXT NOT NULL, v FLOAT, UNIQUE(ts, name))",
            "INSERT INTO m (ts, name, v) VALUES ('2026-01-02T03:04:05Z', 'cpu', 1.5)",
        );
        assert_names(&message, "ts, name", "2026-01-02T03:04:05+00:00, cpu");
    }
}

#[test]
fn a_long_text_key_is_named_in_full() {
    let long = "x".repeat(200);
    for cold in [false, true] {
        let message = conflict_message(
            cold,
            "CREATE TABLE l (id INTEGER PRIMARY KEY AUTO_INCREMENT, name TEXT UNIQUE)",
            &format!("INSERT INTO l (name) VALUES ('{long}')"),
        );
        assert_names(&message, "name", &long);
    }
}
