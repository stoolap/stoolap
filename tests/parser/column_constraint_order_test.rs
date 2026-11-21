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

//! A column definition takes its constraints in any order, in CREATE TABLE
//! and in ALTER TABLE ADD COLUMN alike: a DEFAULT value ends before NOT, AND
//! and OR, so NOT NULL or NULL after it is the column's own

use stoolap::Database;

fn create_table(db: &Database, table: &str) -> String {
    db.query(&format!("SHOW CREATE TABLE {table}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(1).unwrap())
        .next()
        .unwrap()
}

fn values(db: &Database, sql: &str) -> String {
    let row = db.query(sql, ()).unwrap().next().unwrap().unwrap();
    (0..row.len())
        .map(|i| row.get_value(i).unwrap().to_string())
        .collect::<Vec<_>>()
        .join("|")
}

fn error(db: &Database, sql: &str) -> String {
    db.execute(sql, ()).unwrap_err().to_string()
}

#[test]
fn a_default_followed_by_null_or_not_null_is_two_constraints() {
    let db = Database::open("memory://column_constraint_order").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER DEFAULT 0 NOT NULL, \
         b INTEGER DEFAULT 1 + 2 NULL, c INTEGER NULL, d BOOLEAN DEFAULT 1 > 0 NOT NULL, \
         e BOOLEAN DEFAULT (TRUE AND FALSE))",
        (),
    )
    .unwrap();
    assert_eq!(
        create_table(&db, "t"),
        "CREATE TABLE t (id Integer PRIMARY KEY, a Integer NOT NULL DEFAULT 0, \
         b Integer DEFAULT (1 + 2), c Integer, d Boolean NOT NULL DEFAULT (1 > 0), \
         e Boolean DEFAULT (TRUE AND FALSE))"
    );
    db.execute("INSERT INTO t (id) VALUES (1)", ()).unwrap();
    assert_eq!(
        values(&db, "SELECT a, b, c, d, e FROM t"),
        "0|3|NULL|true|false"
    );
    assert_eq!(
        error(&db, "INSERT INTO t (id, a) VALUES (2, NULL)"),
        "NULL value in non-nullable column 'a'"
    );

    db.execute("CREATE TABLE z (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO z VALUES (1)", ()).unwrap();
    db.execute("ALTER TABLE z ADD COLUMN m INTEGER DEFAULT 0 NOT NULL", ())
        .unwrap();
    db.execute("ALTER TABLE z ADD COLUMN n INTEGER DEFAULT 7 NULL", ())
        .unwrap();
    db.execute("ALTER TABLE z ADD COLUMN o TEXT NULL", ())
        .unwrap();
    assert_eq!(
        create_table(&db, "z"),
        "CREATE TABLE z (id Integer PRIMARY KEY, m Integer NOT NULL DEFAULT 0, \
         n Integer DEFAULT 7, o Text)"
    );
    assert_eq!(
        values(&db, "SELECT m, n, o FROM z WHERE id = 1"),
        "0|7|NULL"
    );
    assert_eq!(
        error(&db, "UPDATE z SET m = NULL"),
        "NULL value in non-nullable column 'm'"
    );
}

#[test]
fn null_and_not_null_on_one_column_are_refused() {
    let db = Database::open("memory://column_constraint_conflict").unwrap();
    db.execute("CREATE TABLE z (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, m INTEGER NULL NOT NULL)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, m INTEGER NOT NULL DEFAULT 0 NULL)",
        "ALTER TABLE z ADD COLUMN m INTEGER DEFAULT 0 NOT NULL NULL",
    ] {
        let err = error(&db, sql);
        assert!(
            err.contains("conflicting NULL and NOT NULL for column 'm'"),
            "{sql}: {err}"
        );
    }
    let err = error(
        &db,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b BOOLEAN DEFAULT TRUE AND FALSE)",
    );
    assert!(err.contains("got 'AND'"), "{err}");
}
