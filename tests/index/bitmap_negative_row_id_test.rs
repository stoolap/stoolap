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

//! A bitmap index answers for rows whose INTEGER primary key is negative as
//! a scan does, however the rows were added, updated or deleted

use stoolap::Database;

const SCHEMA: &str = "(id INTEGER PRIMARY KEY, flag BOOLEAN NOT NULL, k INTEGER NOT NULL)";

/// `t` with bitmap indexes on flag (chosen for BOOLEAN) and k, `plain` with
/// no index; both hold the same rows
fn tables(name: &str, index_first: bool) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(&format!("CREATE TABLE t {SCHEMA}"), ()).unwrap();
    db.execute(&format!("CREATE TABLE plain {SCHEMA}"), ())
        .unwrap();
    let index = || {
        db.execute("CREATE INDEX idx_t_flag ON t(flag)", ())
            .unwrap();
        db.execute("CREATE INDEX idx_t_k ON t(k) USING BITMAP", ())
            .unwrap();
    };
    if index_first {
        index();
    }
    for table in ["t", "plain"] {
        // One row by itself, then a batch in one transaction
        db.execute(&format!("INSERT INTO {table} VALUES (-1, true, 1)"), ())
            .unwrap();
        let insert = db
            .prepare(&format!("INSERT INTO {table} VALUES ($1, $2, $3)"))
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        for id in (-40..=40i64).filter(|id| *id != -1) {
            insert.execute((id, id % 2 == 0, id.rem_euclid(3))).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
    }
    if !index_first {
        index();
    }
    db
}

fn ids(db: &Database, table: &str, filter: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(&format!("SELECT id FROM {table} WHERE {filter}"), ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn assert_same(db: &Database, when: &str) {
    for filter in [
        "flag = true",
        "flag = false",
        "k = 1",
        "k = 2 AND flag = true",
    ] {
        let scanned = ids(db, "plain", filter);
        assert!(
            scanned.iter().any(|id| *id < 0),
            "{when}: {filter} has negative ids"
        );
        assert_eq!(ids(db, "t", filter), scanned, "{when}: {filter}");
    }
}

fn update_and_delete(db: &Database) {
    for table in ["t", "plain"] {
        db.execute(
            &format!("UPDATE {table} SET k = 1, flag = true WHERE id = -7"),
            (),
        )
        .unwrap();
        db.execute(&format!("UPDATE {table} SET k = 2 WHERE id < -30"), ())
            .unwrap();
        db.execute(&format!("DELETE FROM {table} WHERE id = -8 OR id = -1"), ())
            .unwrap();
    }
}

#[test]
fn a_bitmap_index_built_before_the_rows_answers_for_negative_ids() {
    let db = tables("bitmap_negative_before", true);
    assert_same(&db, "after the inserts");
    update_and_delete(&db);
    assert_same(&db, "after updates and deletes");
}

#[test]
fn a_bitmap_index_built_over_the_rows_answers_for_negative_ids() {
    let db = tables("bitmap_negative_after", false);
    assert_same(&db, "after the build");
    update_and_delete(&db);
    assert_same(&db, "after updates and deletes");
}
