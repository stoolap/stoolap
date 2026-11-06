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

//! A read of a sealed row by its primary key or a unique key decodes the
//! row groups of the columns the statement names, not of every column.

use stoolap::{Database, Value};

/// The decoded group cache is process-wide; its counters are read around
/// one statement at a time
static DECODES: std::sync::Mutex<()> = std::sync::Mutex::new(());

const ROWS_PER_VOLUME: i64 = 5_000;

/// A file database with four sealed volumes of t (id, k, u UNIQUE, name,
/// doc), the last two wide, reopened so every read decodes from the file
fn sealed(dir: &std::path::Path) -> Database {
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, u INTEGER UNIQUE, name TEXT, doc JSON)",
            (),
        )
        .unwrap();
        for volume in 0..4 {
            let values = (1..=ROWS_PER_VOLUME)
                .map(|i| {
                    let id = volume * ROWS_PER_VOLUME + i;
                    format!(
                        "({id}, {}, {}, 'n{id}{}', '{{\"a\": {id}, \"pad\": \"{}\"}}')",
                        id * 3,
                        id + 1_000_000,
                        "x".repeat(64),
                        "y".repeat(256)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            db.execute(&format!("INSERT INTO t VALUES {values}"), ())
                .unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db.close().unwrap();
    }
    Database::open(&dsn).unwrap()
}

fn misses(db: &Database) -> i64 {
    let rows = db.query("PRAGMA GROUP_CACHE_STATS", ()).unwrap();
    let at = rows.columns().iter().position(|c| c == "misses").unwrap();
    rows.into_iter().next().unwrap().unwrap().get(at).unwrap()
}

/// The rows `sql` answers with `params`, and the groups it decoded
fn read(db: &Database, sql: &str, params: Vec<Value>) -> (Vec<Vec<Value>>, i64) {
    let before = misses(db);
    let rows = db
        .query(sql, params)
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| r.get_value(i).cloned().unwrap())
                .collect()
        })
        .collect();
    (rows, misses(db) - before)
}

#[test]
fn a_point_read_decodes_only_the_columns_it_names() {
    let _decodes = DECODES.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    // Each read is of a row in another volume, so no group is cached yet;
    // the groups are those of the columns the SELECT list and WHERE name
    for (sql, params, want, groups) in [
        (
            "SELECT k FROM t WHERE id = 7",
            vec![],
            vec![Value::Integer(21)],
            2,
        ),
        (
            "SELECT k FROM t WHERE id = $1",
            vec![Value::Integer(5_007)],
            vec![Value::Integer(15_021)],
            2,
        ),
        (
            "SELECT k * 2 FROM t WHERE id = $1 AND u = $2",
            vec![Value::Integer(10_007), Value::Integer(1_010_007)],
            vec![Value::Integer(60_042)],
            3,
        ),
        (
            "SELECT id, k FROM t WHERE u = $1",
            vec![Value::Integer(1_015_007)],
            vec![Value::Integer(15_007), Value::Integer(45_021)],
            3,
        ),
    ] {
        let (rows, decoded) = read(&db, sql, params);
        assert_eq!(rows, vec![want], "{sql}");
        assert!(
            decoded <= groups,
            "{sql}: decoded {decoded} groups, the named columns have {groups}"
        );
    }
}

#[test]
fn a_point_read_of_every_column_answers_them_all() {
    let _decodes = DECODES.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let (rows, _) = read(
        &db,
        "SELECT * FROM t WHERE id = $1",
        vec![Value::Integer(42)],
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::Integer(42));
    assert_eq!(rows[0][1], Value::Integer(126));
    assert_eq!(rows[0][2], Value::Integer(1_000_042));
    assert_eq!(rows[0][3], Value::text(format!("n42{}", "x".repeat(64))));
    assert!(!rows[0][4].is_null(), "the JSON column is read");
    db.execute("BEGIN", ()).unwrap();
    let (rows, _) = read(
        &db,
        "SELECT name FROM t WHERE id = $1",
        vec![Value::Integer(43)],
    );
    assert_eq!(
        rows,
        vec![vec![Value::text(format!("n43{}", "x".repeat(64)))]]
    );
    db.execute("COMMIT", ()).unwrap();
}
