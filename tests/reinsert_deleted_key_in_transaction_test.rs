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

//! A key a transaction has deleted may be taken again by the same
//! transaction: the primary key check and the unique index check look past
//! rows the transaction itself has deleted.

use stoolap::Database;

fn setup(name: &str, index: bool) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, u INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, 7, 10), (2, 8, 20), (3, 7, 30)",
        (),
    )
    .unwrap();
    if index {
        db.execute("CREATE INDEX ix_a ON x (a)", ()).unwrap();
    }
    db
}

fn rows(db: &Database) -> Vec<(i64, i64, i64)> {
    db.query("SELECT id, a, u FROM x ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<i64>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect()
}

#[test]
fn test_a_deleted_key_is_taken_again_and_committed() {
    for index in [false, true] {
        let db = setup(&format!("reinsert_commit_{index}"), index);
        db.execute("BEGIN", ()).unwrap();
        assert_eq!(db.execute("DELETE FROM x WHERE a = 7", ()).unwrap(), 2);
        assert_eq!(
            db.execute("INSERT INTO x VALUES (1, 7, 100), (3, 9, 30)", ())
                .unwrap(),
            2
        );
        assert_eq!(
            rows(&db),
            [(1, 7, 100), (2, 8, 20), (3, 9, 30)],
            "index {index}"
        );
        let by_index: i64 = db
            .query("SELECT COUNT(*) FROM x WHERE a = 7", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(by_index, 1, "index {index}");
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(
            rows(&db),
            [(1, 7, 100), (2, 8, 20), (3, 9, 30)],
            "index {index}"
        );
        let by_index: i64 = db
            .query("SELECT COUNT(*) FROM x WHERE a = 9", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(by_index, 1, "index {index}");
    }
}

#[test]
fn test_a_rolled_back_reinsert_leaves_the_old_row() {
    let db = setup("reinsert_rollback", true);
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM x WHERE id = 1", ()).unwrap();
    db.execute("INSERT INTO x VALUES (1, 7, 100)", ()).unwrap();
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(rows(&db), [(1, 7, 10), (2, 8, 20), (3, 7, 30)]);
}

#[test]
fn test_a_key_still_held_is_refused() {
    let db = setup("reinsert_refused", false);
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM x WHERE id = 1", ()).unwrap();
    assert!(db.execute("INSERT INTO x VALUES (2, 1, 1)", ()).is_err());
    // the unique value of the deleted row is free, another row's is not
    db.execute("INSERT INTO x VALUES (9, 1, 10)", ()).unwrap();
    assert!(db.execute("INSERT INTO x VALUES (8, 1, 20)", ()).is_err());
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(rows(&db), [(2, 8, 20), (3, 7, 30), (9, 1, 10)]);
}
