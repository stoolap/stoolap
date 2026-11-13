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

//! A planned join's table side filtered by literals is answered by the
//! semantic cache as a `SELECT *` of the table by that filter is

use stoolap::Database;

const JOIN: &str =
    "SELECT p.id, o.id FROM p JOIN o ON o.qty = p.id WHERE o.id < 10 ORDER BY p.id, o.id";

fn tables(db: &Database) {
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE o (id INTEGER PRIMARY KEY, qty INTEGER, note TEXT)",
        "INSERT INTO p SELECT value FROM generate_series(1, 5)",
        "INSERT INTO o SELECT value, value % 5 + 1, 'n' FROM generate_series(1, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn pairs(db: &Database) -> Vec<(i64, i64)> {
    db.query(JOIN, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect()
}

#[test]
fn a_join_side_is_answered_by_the_cache_until_its_table_changes() {
    let db = Database::open("memory://semantic_join_side_hits").unwrap();
    tables(&db);
    let first = [
        (1, 5),
        (2, 1),
        (2, 6),
        (3, 2),
        (3, 7),
        (4, 3),
        (4, 8),
        (5, 4),
        (5, 9),
    ];
    assert_eq!(pairs(&db), first, "first run");
    let before = db.semantic_cache_stats().unwrap().hits;
    assert_eq!(pairs(&db), first, "second run");
    let hits = db.semantic_cache_stats().unwrap().hits;
    assert!(
        hits > before,
        "the side was answered by the cache: {before} -> {hits}"
    );
    db.execute("UPDATE o SET qty = 1 WHERE id = 2", ()).unwrap();
    let moved = [
        (1, 2),
        (1, 5),
        (2, 1),
        (2, 6),
        (3, 7),
        (4, 3),
        (4, 8),
        (5, 4),
        (5, 9),
    ];
    assert_eq!(pairs(&db), moved, "after the update");
}

#[test]
fn a_narrower_side_filter_is_answered_from_a_wider_one() {
    let db = Database::open("memory://semantic_join_side_subsumed").unwrap();
    tables(&db);
    assert_eq!(pairs(&db).len(), 9, "the wider join");
    let before = db.semantic_cache_stats().unwrap().subsumption_hits;
    let narrower = JOIN.replace("o.id < 10", "o.id < 5");
    let got: Vec<(i64, i64)> = db
        .query(&narrower, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect();
    assert_eq!(got, [(2, 1), (3, 2), (4, 3), (5, 4)], "the narrower join");
    let after = db.semantic_cache_stats().unwrap().subsumption_hits;
    assert!(
        after > before,
        "answered from the wider entry: {before} -> {after}"
    );
}

#[test]
fn a_sealed_side_read_by_need_leaves_no_partial_rows_in_the_cache() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    tables(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(pairs(&db).len(), 9, "the join");
    let notes: Vec<Option<String>> = db
        .query("SELECT * FROM o WHERE o.id < 10", ())
        .unwrap()
        .map(|row| row.unwrap().get::<Option<String>>(2).unwrap())
        .collect();
    assert_eq!(notes, vec![Some("n".to_string()); 9], "notes of the rows");
}

#[test]
fn a_filter_with_an_escape_parameter_is_read_from_its_table() {
    let mut wrong = Vec::new();
    for (at, (query, want)) in [
        (
            "SELECT p.id, o.id FROM p JOIN o ON o.qty = p.id \
             WHERE o.id < o.qty AND o.note LIKE o.pattern ESCAPE $1",
            1,
        ),
        (
            "SELECT * FROM o WHERE id < qty AND note LIKE pattern ESCAPE $1",
            2,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let db = Database::open(&format!("memory://semantic_join_side_escape_{at}")).unwrap();
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY)",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, qty INTEGER, note TEXT, pattern TEXT)",
            "INSERT INTO p VALUES (2)",
            "INSERT INTO o VALUES (1, 2, 'a_b', 'a!_b'), (3, 4, 'axb', 'a!_b')",
        ] {
            db.execute(sql, ()).unwrap();
        }
        // The rows of the filter without the LIKE, cached first
        db.query("SELECT * FROM o WHERE id < qty", ())
            .unwrap()
            .count();
        db.query("SELECT * FROM o WHERE id < qty", ())
            .unwrap()
            .count();
        let found: Result<Vec<i64>, String> = db
            .query(query, ("!",))
            .map_err(|e| e.to_string())
            .and_then(|rows| {
                rows.map(|row| {
                    row.map_err(|e| e.to_string())?
                        .get::<i64>(1)
                        .map_err(|e| e.to_string())
                })
                .collect()
            });
        if found.as_deref() != Ok(&[want][..]) {
            wrong.push(format!("{query}: {found:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
