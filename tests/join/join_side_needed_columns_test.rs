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

//! A join reads from a sealed table side only the columns the statement
//! names, as a statement over that table alone does.

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::take_group_column_decodes;
use stoolap::Database;

/// The column of `e` no statement below names but the last
const PAYLOAD: usize = 3;

static GROUP_CACHE_BUDGET: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn sealed_tables(dir: &std::path::Path) -> Database {
    let dsn = format!("file://{}/join_side_needed_columns", dir.display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE e (id INTEGER PRIMARY KEY, sid INTEGER, note TEXT, payload TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    for i in 0..20 {
        db.execute(&format!("INSERT INTO s VALUES ({i}, 's{i}')"), ())
            .unwrap();
    }
    db.execute("BEGIN", ()).unwrap();
    for i in 0..2_000 {
        db.execute(
            &format!(
                "INSERT INTO e VALUES ({i}, {}, 'n{}', 'payload {i}')",
                i % 20,
                i % 7
            ),
            (),
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    drop(db);
    // Reopened, the volumes are read group by group through the group cache
    Database::open(&dsn).unwrap()
}

fn count(db: &Database, sql: &str) -> usize {
    db.query(sql, ())
        .unwrap()
        .inspect(|r| assert!(r.is_ok()))
        .count()
}

#[test]
fn test_join_decodes_no_column_of_a_sealed_side_it_does_not_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_tables(dir.path());
    let _serial = GROUP_CACHE_BUDGET
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let budget: i64 = db.query_one("PRAGMA GROUP_CACHE_MB", ()).unwrap();
    // With no budget every block a read asks for decodes again
    db.execute("PRAGMA GROUP_CACHE_MB = 0", ()).unwrap();
    let shapes = [
        ("SELECT e.id, s.name FROM e JOIN s ON s.id = e.sid", 2_000),
        (
            "SELECT e.id, s.name FROM e JOIN s ON s.id = e.sid WHERE e.note = 'n3'",
            286,
        ),
        (
            "SELECT e.id, s.name FROM e LEFT JOIN s ON s.id = e.sid",
            2_000,
        ),
        ("SELECT s.name, e.note FROM s JOIN e ON e.sid = s.id", 2_000),
        (
            "SELECT s.name, COUNT(*) FROM e JOIN s ON s.id = e.sid GROUP BY s.name",
            20,
        ),
    ];
    for (sql, rows) in shapes {
        take_group_column_decodes();
        assert_eq!(count(&db, sql), rows, "{sql}");
        let decodes = take_group_column_decodes();
        assert!(
            decodes.iter().all(|&(column, _)| column != PAYLOAD),
            "{sql}: decoded e.payload: {decodes:?}"
        );
    }
    let rows: Vec<(i64, String, String)> = db
        .query(
            "SELECT e.id, e.note, s.name FROM e JOIN s ON s.id = e.sid ORDER BY e.id",
            (),
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap(), r.get(2).unwrap())
        })
        .collect();
    let expected: Vec<(i64, String, String)> = (0..2_000)
        .map(|i| (i, format!("n{}", i % 7), format!("s{}", i % 20)))
        .collect();
    assert_eq!(rows, expected, "the columns a join names read their values");
    take_group_column_decodes();
    let sql = "SELECT e.payload FROM e JOIN s ON s.id = e.sid";
    assert_eq!(count(&db, sql), 2_000);
    assert!(
        take_group_column_decodes()
            .iter()
            .any(|&(column, _)| column == PAYLOAD),
        "{sql}: the read that names e.payload decoded no block of it"
    );
    db.execute(&format!("PRAGMA GROUP_CACHE_MB = {budget}"), ())
        .unwrap();
}
