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

use stoolap::Database;
use tempfile::tempdir;

fn next_id_after_insert(db: &Database, table: &str) -> i64 {
    db.execute(&format!("INSERT INTO {table} (v) VALUES ('new')"), ())
        .unwrap();
    db.query_one(&format!("SELECT id FROM {table} WHERE v = 'new'"), ())
        .unwrap()
}

#[test]
fn test_deleted_id_not_reused_after_reopen() {
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().join("db").display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
            .unwrap();
        db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        next_id_after_insert(&db, "t"),
        4,
        "id 3 was handed out before the reopen"
    );
}

#[test]
fn test_deleted_id_not_reused_after_checkpoint_and_crash() {
    let dir = tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().join("db").display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
            .unwrap();
        db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        next_id_after_insert(&db, "t"),
        4,
        "id 3 was handed out before the checkpoint"
    );
}

#[test]
fn test_deleted_rowid_not_reused_after_reopen() {
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().join("db").display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE n (v TEXT)", ()).unwrap();
        db.execute("INSERT INTO n (v) VALUES ('a'), ('b'), ('c')", ())
            .unwrap();
        db.execute("DELETE FROM n WHERE v = 'c'", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("INSERT INTO n (v) VALUES ('d')", ()).unwrap();
    let rowid: i64 = db
        .query_one("SELECT _rowid_ FROM n WHERE v = 'd'", ())
        .unwrap();
    assert_eq!(rowid, 4, "row id 3 was handed out before the reopen");
}

#[test]
fn test_truncate_restarts_ids_after_reopen() {
    let dir = tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().join("db").display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
            .unwrap();
        db.execute("TRUNCATE TABLE t", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(next_id_after_insert(&db, "t"), 1);
}
