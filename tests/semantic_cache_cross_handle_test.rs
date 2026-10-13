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

//! A result one handle cached is not served after another handle changed
//! the table: through a clone, the Transaction API, DDL or TRUNCATE

use stoolap::Database;

const CACHED: &str = "SELECT * FROM t WHERE v > 5";

fn table(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db
}

/// Runs the cacheable query twice on `reader`, so its result is cached
fn cache(reader: &Database) -> usize {
    reader.query(CACHED, ()).unwrap().count();
    reader.query(CACHED, ()).unwrap().count()
}

fn rows(reader: &Database) -> usize {
    reader.query(CACHED, ()).unwrap().count()
}

fn columns(reader: &Database) -> usize {
    reader.query(CACHED, ()).unwrap().columns().len()
}

#[test]
fn test_same_handle_still_hits_the_cache() {
    let db = table("memory://semantic_cross_same_handle");
    cache(&db);
    let before = db.semantic_cache_stats().unwrap().exact_hits;
    assert_eq!(rows(&db), 2);
    assert_eq!(db.semantic_cache_stats().unwrap().exact_hits, before + 1);
}

#[test]
fn test_writes_through_a_clone_are_seen() {
    let writer = table("memory://semantic_cross_clone");
    let reader = writer.clone();
    assert_eq!(cache(&reader), 2);
    writer
        .execute("INSERT INTO t VALUES (3, 30), (4, 40)", ())
        .unwrap();
    assert_eq!(rows(&reader), 4, "insert");
    writer
        .execute("UPDATE t SET v = 1 WHERE id = 1", ())
        .unwrap();
    assert_eq!(rows(&reader), 3, "update");
    writer.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(rows(&reader), 2, "delete");
}

#[test]
fn test_writes_through_the_transaction_api_are_seen() {
    let writer = table("memory://semantic_cross_tx_api");
    let reader = writer.clone();
    assert_eq!(cache(&reader), 2);
    let mut tx = writer.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    tx.commit().unwrap();
    assert_eq!(rows(&reader), 3);
}

#[test]
fn test_ddl_through_another_handle_is_seen() {
    let writer = table("memory://semantic_cross_ddl");
    let reader = writer.clone();
    cache(&reader);
    assert_eq!(columns(&reader), 2);
    writer
        .execute("ALTER TABLE t ADD COLUMN w INTEGER DEFAULT 7", ())
        .unwrap();
    assert_eq!(columns(&reader), 3);
}

#[test]
fn test_truncate_through_another_handle_is_seen() {
    let writer = table("memory://semantic_cross_truncate");
    let reader = writer.clone();
    assert_eq!(cache(&reader), 2);
    writer.execute("TRUNCATE TABLE t", ()).unwrap();
    assert_eq!(rows(&reader), 0);
}

#[test]
fn test_delete_of_a_sealed_row_through_another_handle_is_seen() {
    let dir = tempfile::tempdir().unwrap();
    let writer = table(&format!("file://{}", dir.path().display()));
    writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let reader = writer.clone();
    assert_eq!(cache(&reader), 2);
    writer.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(rows(&reader), 1);
}
