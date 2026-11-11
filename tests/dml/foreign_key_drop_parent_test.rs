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

//! DROP TABLE of a table other tables' foreign keys name

use stoolap::Database;

fn tables(db: &Database, children: &[&str]) {
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON DELETE CASCADE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO p VALUES (1), (2)", ()).unwrap();
    for row in children {
        db.execute(&format!("INSERT INTO c VALUES {row}"), ())
            .unwrap();
    }
}

fn create_c(db: &Database) -> String {
    let row = db
        .query("SHOW CREATE TABLE c", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    row.get::<String>(1).unwrap()
}

fn assert_refused(db: &Database, sql: &str) {
    let error = db.execute(sql, ()).unwrap_err();
    assert!(
        error.to_string().contains("table 'c' references it")
            && error.to_string().contains("CASCADE"),
        "{sql} failed with {error}"
    );
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM p", ())
            .unwrap(),
        2,
        "{sql} dropped the table"
    );
    assert!(
        create_c(db).contains("REFERENCES p(id)"),
        "{sql} took the key"
    );
}

#[test]
fn a_table_another_key_names_is_not_dropped_whatever_rows_the_child_holds() {
    for (name, children) in [
        ("no_rows", &[][..]),
        ("null_rows", &["(1, NULL)"][..]),
        ("referencing_rows", &["(1, 1)", "(2, NULL)"][..]),
    ] {
        let db = Database::open(&format!("memory://fk_drop_parent_{name}")).unwrap();
        tables(&db, children);
        assert_refused(&db, "DROP TABLE p");
        assert_refused(&db, "DROP TABLE IF EXISTS p");
        db.execute("BEGIN", ()).unwrap();
        assert_refused(&db, "DROP TABLE p");
        db.execute("ROLLBACK", ()).unwrap();
        assert!(
            db.execute("INSERT INTO c VALUES (9, 7)", ()).is_err(),
            "{name}"
        );
        db.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM c WHERE pid = 1", ())
                .unwrap(),
            0,
            "{name}: the key lost its action"
        );
        db.execute("DROP TABLE c", ()).unwrap();
        db.execute("DROP TABLE p", ()).unwrap();
    }
}

#[test]
fn drop_table_cascade_drops_the_keys_that_name_the_table_and_keeps_the_rows() {
    let db = Database::open("memory://fk_drop_parent_cascade").unwrap();
    tables(&db, &["(1, 1)", "(2, NULL)"]);
    db.execute("DROP TABLE p CASCADE", ()).unwrap();
    assert!(!create_c(&db).contains("REFERENCES"));
    db.execute("INSERT INTO c VALUES (3, 7)", ()).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM c", ())
            .unwrap(),
        3
    );
    db.execute("DROP TABLE IF EXISTS p CASCADE", ()).unwrap();
}

#[test]
fn drop_table_cascade_inside_a_transaction_drops_the_keys() {
    let db = Database::open("memory://fk_drop_parent_cascade_tx").unwrap();
    tables(&db, &["(1, 1)"]);
    db.execute("BEGIN", ()).unwrap();
    db.execute("DROP TABLE p CASCADE", ()).unwrap();
    db.execute("COMMIT", ()).unwrap();
    assert!(!create_c(&db).contains("REFERENCES"));
    db.execute("INSERT INTO c VALUES (2, 7)", ()).unwrap();
}

#[test]
fn a_refused_drop_and_a_cascade_hold_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        tables(&db, &[]);
        assert_refused(&db, "DROP TABLE p");
        db.execute("CREATE TABLE q (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute(
            "CREATE TABLE d (id INTEGER PRIMARY KEY, qid INTEGER REFERENCES q(id))",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO q VALUES (1)", ()).unwrap();
        db.execute("INSERT INTO d VALUES (1, 1)", ()).unwrap();
        db.execute("DROP TABLE q CASCADE", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert!(create_c(&db).contains("REFERENCES p(id)"));
    assert!(db.execute("INSERT INTO c VALUES (5, 7)", ()).is_err());
    db.execute("INSERT INTO d VALUES (2, 7)", ()).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM d", ())
            .unwrap(),
        2
    );
}

#[test]
fn drop_table_cascade_is_written_back_as_it_was_parsed() {
    let statement = stoolap::parser::parse_sql("DROP TABLE IF EXISTS p CASCADE").unwrap();
    assert_eq!(statement[0].to_string(), "DROP TABLE IF EXISTS p CASCADE");
    let statement = stoolap::parser::parse_sql("DROP TABLE p").unwrap();
    assert_eq!(statement[0].to_string(), "DROP TABLE p");
}
