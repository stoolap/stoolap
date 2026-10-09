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

//! A value written before MODIFY COLUMN changed its column's type is stored
//! by a later seal or compaction as CAST to the new type gives it, and reads
//! so after a reopen.

use stoolap::core::{DataType, Value};
use stoolap::Database;

fn open(dir: &std::path::Path) -> Database {
    Database::open(&format!("file://{}?compact_threshold=2", dir.display())).unwrap()
}

fn value(db: &Database, sql: &str) -> Value {
    let row = db.query(sql, ()).unwrap().next().unwrap().unwrap();
    row.get_value(0)
        .cloned()
        .unwrap_or(Value::Null(DataType::Null))
}

fn x_of(db: &Database, id: i64) -> Value {
    value(db, &format!("SELECT x FROM t WHERE id = {id}"))
}

#[test]
fn a_sealed_or_compacted_value_reads_as_its_cast_to_the_new_type() {
    // (old type, new type, value sealed before the change, value hot at the change)
    let cases = [
        ("INTEGER", "FLOAT", "7", "8"),
        ("FLOAT", "INTEGER", "7.5", "-8.25"),
        ("INTEGER", "TEXT", "7", "8"),
        ("FLOAT", "TEXT", "7.5", "0.25"),
        ("TEXT", "INTEGER", "'7'", "'x'"),
        ("TEXT", "FLOAT", "'7.5'", "'x'"),
        ("INTEGER", "BOOLEAN", "1", "0"),
        ("BOOLEAN", "INTEGER", "true", "false"),
        (
            "TIMESTAMP",
            "TEXT",
            "'2024-01-02 03:04:05'",
            "'2024-01-03 00:00:00'",
        ),
        ("TEXT", "TIMESTAMP", "'2024-01-02 03:04:05'", "'not a time'"),
    ];
    let mut wrong = Vec::new();
    for (from, to, sealed, hot) in cases {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path());
        db.execute(
            &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, x {from})"),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO t VALUES (1, {sealed})"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute(&format!("INSERT INTO t VALUES (2, {hot})"), ())
            .unwrap();
        db.execute(&format!("ALTER TABLE t MODIFY COLUMN x {to}"), ())
            .unwrap();
        let want = [
            value(
                &db,
                &format!("SELECT CAST(CAST({sealed} AS {from}) AS {to})"),
            ),
            value(&db, &format!("SELECT CAST(CAST({hot} AS {from}) AS {to})")),
        ];
        // Row 2 is sealed under the new type
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        if x_of(&db, 2) != want[1] {
            wrong.push(format!("{from} -> {to}, sealed after: {:?}", x_of(&db, 2)));
        }
        // A third volume makes the three compact into one
        db.execute("INSERT INTO t VALUES (3, NULL)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let volumes = db.query("PRAGMA VOLUME_STATS", ()).unwrap().count();
        assert_eq!(volumes, 1, "{from} -> {to}: the volumes did not compact");
        for (id, want) in [(1, &want[0]), (2, &want[1])] {
            let got = x_of(&db, id);
            if &got != want {
                wrong.push(format!(
                    "{from} -> {to}, row {id} compacted: {got:?}, want {want:?}"
                ));
            }
        }
        drop(db);
        let db = open(dir.path());
        for (id, want) in [(1, &want[0]), (2, &want[1])] {
            let got = x_of(&db, id);
            if &got != want {
                wrong.push(format!(
                    "{from} -> {to}, row {id} reopened: {got:?}, want {want:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_default_of_a_column_added_later_compacts_as_its_cast() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // Row 1's volume never held x: it reads the default
    db.execute("ALTER TABLE t ADD COLUMN x INTEGER DEFAULT 5", ())
        .unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN x TEXT", ())
        .unwrap();
    let want = value(&db, "SELECT CAST(5 AS TEXT)");
    db.execute("INSERT INTO t VALUES (2, 'b')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, 'c')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let volumes = db.query("PRAGMA VOLUME_STATS", ()).unwrap().count();
    assert_eq!(volumes, 1, "the volumes did not compact");
    assert_eq!(x_of(&db, 1), want, "compacted");
    drop(db);
    let db = open(dir.path());
    assert_eq!(x_of(&db, 1), want, "reopened");
}
