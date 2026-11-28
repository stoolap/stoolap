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

//! ALTER TABLE ADD COLUMN and MODIFY COLUMN refuse a constraint they cannot
//! keep and leave the table as it was: neither accepts one and ignores it

use stoolap::Database;

/// Each statement that names a constraint its operation does not keep, and
/// the constraint the error names
const REFUSED: &[(&str, &str)] = &[
    (
        "ADD COLUMN z TEXT UNIQUE",
        "ADD COLUMN does not take a UNIQUE",
    ),
    (
        "ADD COLUMN z INTEGER PRIMARY KEY",
        "ADD COLUMN does not take a PRIMARY KEY",
    ),
    (
        "ADD COLUMN z INTEGER AUTO_INCREMENT",
        "ADD COLUMN does not take an AUTO_INCREMENT",
    ),
    (
        "ADD COLUMN z INTEGER REFERENCES p(id)",
        "ADD COLUMN does not take a REFERENCES",
    ),
    (
        "ADD COLUMN z INTEGER DEFAULT 1 CHECK (z < 100) UNIQUE",
        "ADD COLUMN does not take a UNIQUE",
    ),
    (
        "MODIFY COLUMN v INTEGER UNIQUE",
        "MODIFY COLUMN does not take a UNIQUE",
    ),
    (
        "MODIFY COLUMN v INTEGER PRIMARY KEY",
        "MODIFY COLUMN does not take a PRIMARY KEY",
    ),
    (
        "MODIFY COLUMN v INTEGER AUTO_INCREMENT",
        "MODIFY COLUMN does not take an AUTO_INCREMENT",
    ),
    (
        "MODIFY COLUMN v INTEGER REFERENCES p(id)",
        "MODIFY COLUMN does not take a REFERENCES",
    ),
    (
        "MODIFY COLUMN v INTEGER DEFAULT 7 CHECK (v < 2000) UNIQUE",
        "MODIFY COLUMN does not take a UNIQUE",
    ),
];

fn text(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER DEFAULT 3 CHECK (v < 1000))",
        "INSERT INTO t VALUES (1, 500)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

#[test]
fn an_alter_column_constraint_the_operation_cannot_keep_is_refused() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        let before = text(&db, "SHOW CREATE TABLE t");
        for (operation, names) in REFUSED {
            let sql = format!("ALTER TABLE t {operation}");
            match db.execute(&sql, ()) {
                Err(e) if e.to_string().contains(names) => {}
                got => wrong.push(format!("sealed {sealed}: {sql}: {got:?}, want '{names}'")),
            }
            let after = text(&db, "SHOW CREATE TABLE t");
            if after != before {
                wrong.push(format!("sealed {sealed}: {sql} left {after:?}"));
            }
        }
        // The column keeps the DEFAULT and CHECK it was created with
        db.execute("INSERT INTO t (id) VALUES (2)", ()).unwrap();
        if db.execute("INSERT INTO t VALUES (3, 5000)", ()).is_ok() {
            wrong.push(format!("sealed {sealed}: the CHECK of v let 5000 in"));
        }
        let rows = text(&db, "SELECT id, v FROM t ORDER BY id");
        if rows != ["1|500", "2|3"] {
            wrong.push(format!("sealed {sealed}: rows {rows:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_alter_column_keeps_not_null_and_an_added_columns_default() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        db.execute("ALTER TABLE t ADD COLUMN z INTEGER NOT NULL DEFAULT 4", ())
            .unwrap();
        db.execute("ALTER TABLE t MODIFY COLUMN v TEXT NOT NULL", ())
            .unwrap();
        assert_eq!(
            text(&db, "SELECT id, v, z FROM t"),
            ["1|500|4"],
            "sealed {sealed}"
        );
        assert!(
            db.execute("INSERT INTO t (id, v, z) VALUES (2, NULL, 1)", ())
                .is_err(),
            "sealed {sealed}: v took NULL after MODIFY COLUMN ... NOT NULL"
        );
    }
}
