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

#[test]
fn a_like_default_with_escape_ends_before_not_null() {
    let db = Database::open("memory://column_constraint_escape").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b BOOLEAN DEFAULT 'a' LIKE 'a!%' ESCAPE '!' NOT NULL)",
        (),
    )
    .unwrap();
    assert_eq!(
        create_table(&db, "t"),
        "CREATE TABLE t (id Integer PRIMARY KEY, b Boolean NOT NULL DEFAULT 'a' LIKE 'a!%' ESCAPE '!')"
    );
    assert_eq!(
        error(&db, "INSERT INTO t (id, b) VALUES (1, NULL)"),
        "NULL value in non-nullable column 'b'"
    );
    db.execute("CREATE TABLE z (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO z VALUES (1)", ()).unwrap();
    db.execute(
        "ALTER TABLE z ADD COLUMN b BOOLEAN DEFAULT 'a' LIKE 'a' ESCAPE '!' NOT NULL",
        (),
    )
    .unwrap();
    assert_eq!(values(&db, "SELECT b FROM z"), "true");
    assert_eq!(
        error(&db, "UPDATE z SET b = NULL"),
        "NULL value in non-nullable column 'b'"
    );
    assert_eq!(
        values(
            &db,
            "SELECT COUNT(*) FROM z WHERE 'a' LIKE 'a' ESCAPE '!' AND id = 2"
        ),
        "0"
    );
}

#[test]
fn a_negated_predicate_default_prints_ddl_that_runs_again() {
    let db = Database::open("memory://column_constraint_negated").unwrap();
    let again = Database::open("memory://column_constraint_negated_again").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b BOOLEAN DEFAULT ('a' NOT LIKE 'b'), \
         c BOOLEAN DEFAULT (1 NOT IN (2, 3)) NOT NULL, d BOOLEAN DEFAULT (1 NOT BETWEEN 2 AND 3), \
         e BOOLEAN DEFAULT 'a' NOT LIKE 'a' ESCAPE '!' NULL)",
        (),
    )
    .unwrap();
    let shown = create_table(&db, "t");
    assert_eq!(
        shown,
        "CREATE TABLE t (id Integer PRIMARY KEY, b Boolean DEFAULT 'a' NOT LIKE 'b', \
         c Boolean NOT NULL DEFAULT 1 NOT IN (2, 3), d Boolean DEFAULT 1 NOT BETWEEN 2 AND 3, \
         e Boolean DEFAULT 'a' NOT LIKE 'a' ESCAPE '!')"
    );
    again
        .execute(&shown, ())
        .unwrap_or_else(|e| panic!("{shown}: {e}"));
    assert_eq!(create_table(&again, "t"), shown);
    again.execute("INSERT INTO t (id) VALUES (1)", ()).unwrap();
    assert_eq!(
        values(&again, "SELECT b, c, d, e FROM t"),
        "true|true|true|false"
    );
    let err = error(
        &db,
        "CREATE TABLE u (id INTEGER PRIMARY KEY, b BOOLEAN DEFAULT 0 NOT UNIQUE)",
    );
    assert!(
        err.contains("expected NULL, IN, BETWEEN or LIKE after NOT"),
        "{err}"
    );
}
