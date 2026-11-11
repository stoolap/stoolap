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

//! Two foreign keys on one column

use stoolap::Database;

const PARENTS: &[&str] = &[
    "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
    "CREATE TABLE q (id INTEGER PRIMARY KEY)",
    "INSERT INTO p VALUES (1, 1), (2, 20), (3, 3)",
    "INSERT INTO q VALUES (1), (3)",
];

/// One key into each of two columns of `p`, and one into `p` and one into `q`
const CHILDREN: &[&str] = &[
    "CREATE TABLE c (id INTEGER PRIMARY KEY, x INTEGER REFERENCES p(u) ON UPDATE CASCADE, FOREIGN KEY (x) REFERENCES p(id))",
    "CREATE TABLE d (id INTEGER PRIMARY KEY, x INTEGER REFERENCES p(id) ON DELETE CASCADE, FOREIGN KEY (x) REFERENCES q(id))",
];

fn index_names(db: &Database, table: &str) -> Vec<String> {
    db.query(&format!("SHOW INDEXES FROM {table}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(1).unwrap())
        .collect()
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}

fn check_keys(db: &Database) {
    assert_eq!(index_names(db, "c"), ["fk_c_x"]);
    assert_eq!(index_names(db, "d"), ["fk_d_x"]);
    for (table, values) in [("c", (1, 2)), ("d", (1, 2))] {
        db.execute(&format!("INSERT INTO {table} VALUES (1, {})", values.0), ())
            .unwrap_or_else(|e| panic!("{table}: a value both parents hold was refused: {e}"));
        assert!(
            db.execute(&format!("INSERT INTO {table} VALUES (2, {})", values.1), ())
                .is_err(),
            "{table}: a value only one parent holds was taken"
        );
    }
    assert!(db.execute("UPDATE p SET u = 5 WHERE id = 1", ()).is_err());
    assert!(db.execute("DELETE FROM q WHERE id = 1", ()).is_err());
    db.execute("DELETE FROM p WHERE id = 1", ()).unwrap_err();
    db.execute("DELETE FROM c", ()).unwrap();
    db.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    assert_eq!(
        count(db, "SELECT COUNT(*) FROM d"),
        0,
        "the ON DELETE CASCADE did not run"
    );
}

#[test]
fn a_column_with_two_foreign_keys_gets_one_index() {
    let db = Database::open("memory://fk_one_column").unwrap();
    for sql in PARENTS.iter().chain(CHILDREN) {
        db.execute(sql, ())
            .unwrap_or_else(|e| panic!("{sql} failed: {e}"));
    }
    check_keys(&db);
    db.execute("ALTER TABLE c DROP COLUMN x", ()).unwrap();
    assert!(index_names(&db, "c").is_empty());
    db.execute("DELETE FROM p", ()).unwrap();
}

#[test]
fn a_column_with_two_foreign_keys_gets_one_index_inside_a_transaction() {
    let db = Database::open("memory://fk_one_column_tx").unwrap();
    for sql in PARENTS {
        db.execute(sql, ()).unwrap();
    }
    db.execute("BEGIN", ()).unwrap();
    for sql in CHILDREN {
        db.execute(sql, ())
            .unwrap_or_else(|e| panic!("{sql} failed: {e}"));
    }
    db.execute("COMMIT", ()).unwrap();
    check_keys(&db);
}

#[test]
fn a_column_with_two_foreign_keys_keeps_one_index_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        for sql in PARENTS.iter().chain(CHILDREN) {
            db.execute(sql, ())
                .unwrap_or_else(|e| panic!("{sql} failed: {e}"));
        }
        db.close().unwrap();
    }
    check_keys(&Database::open(&dsn).unwrap());
}
