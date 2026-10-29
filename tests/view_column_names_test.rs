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

fn columns(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ()).unwrap().columns().to_vec()
}

fn error(db: &Database, sql: &str) -> String {
    db.execute(sql, ()).expect_err(sql).to_string()
}

fn query_error(db: &Database, sql: &str) -> String {
    match db.query(sql, ()) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("{sql} ran"),
    }
}

fn setup(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 100)", ()).unwrap();
}

#[test]
fn a_view_is_refused_when_its_query_cannot_run() {
    let db = Database::open("memory://view_names_refused").unwrap();
    setup(&db);
    let missing = error(&db, "CREATE VIEW v AS SELECT x FROM missing");
    assert!(missing.contains("missing"), "{missing}");
    let itself = error(&db, "CREATE VIEW v AS SELECT id FROM v");
    assert!(itself.contains('v'), "{itself}");
    let column = error(&db, "CREATE VIEW v AS SELECT z FROM t");
    assert!(column.contains('z'), "{column}");
    assert!(
        db.query("SELECT * FROM v", ()).is_err(),
        "no view was created"
    );
}

#[test]
fn a_view_is_refused_when_two_columns_share_a_name() {
    let db = Database::open("memory://view_names_twice").unwrap();
    setup(&db);
    for sql in [
        "CREATE VIEW v AS SELECT a, a FROM t",
        "CREATE VIEW v AS SELECT a, b AS A FROM t",
        "CREATE VIEW v AS SELECT id, a AS id FROM t",
        "CREATE VIEW v AS SELECT 1 AS \"Ä\", 2 AS \"ä\"",
    ] {
        let message = error(&db, sql);
        assert!(message.contains("more than once"), "{sql}: {message}");
    }
    db.execute("CREATE VIEW v AS SELECT a, a AS c FROM t", ())
        .unwrap();
    assert_eq!(columns(&db, "SELECT * FROM v"), ["a", "c"]);
}

#[test]
fn a_view_keeps_the_names_it_was_created_with() {
    let db = Database::open("memory://view_names_kept").unwrap();
    setup(&db);
    db.execute("CREATE VIEW v AS SELECT id, a + 1, b AS k FROM t", ())
        .unwrap();
    assert_eq!(columns(&db, "SELECT * FROM v"), ["id", "a + 1", "k"]);
    db.execute("ALTER TABLE t RENAME COLUMN id TO n", ())
        .unwrap();
    db.execute("CREATE VIEW w AS SELECT * FROM t", ()).unwrap();
    db.execute("ALTER TABLE t RENAME COLUMN a TO z", ())
        .unwrap();
    assert_eq!(columns(&db, "SELECT * FROM w"), ["n", "a", "b"]);
    assert_eq!(columns(&db, "SELECT a FROM w"), ["a"]);
    let rows: Vec<i64> = db
        .query("SELECT a FROM w", ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(rows, [10]);
}

#[test]
fn a_view_whose_query_gives_other_columns_is_an_error() {
    let db = Database::open("memory://view_names_width").unwrap();
    setup(&db);
    db.execute("CREATE VIEW w AS SELECT * FROM t", ()).unwrap();
    db.execute("ALTER TABLE t ADD COLUMN c INTEGER", ())
        .unwrap();
    let wider = query_error(&db, "SELECT * FROM w");
    assert!(wider.contains("create the view again"), "{wider}");
    db.execute("DROP VIEW w", ()).unwrap();
    db.execute("CREATE VIEW w AS SELECT * FROM t", ()).unwrap();
    db.execute("ALTER TABLE t DROP COLUMN c", ()).unwrap();
    let narrower = query_error(&db, "SELECT id FROM w");
    assert!(narrower.contains("create the view again"), "{narrower}");
}

#[test]
fn a_view_recorded_without_names_takes_the_ones_it_first_gives() {
    let db = Database::open("memory://view_names_old").unwrap();
    setup(&db);
    db.engine()
        .create_view("old", "SELECT * FROM t".to_string(), None, false)
        .unwrap();
    assert_eq!(columns(&db, "SELECT * FROM old"), ["id", "a", "b"]);
    db.execute("ALTER TABLE t RENAME COLUMN a TO z", ())
        .unwrap();
    assert_eq!(columns(&db, "SELECT * FROM old"), ["id", "a", "b"]);
}

#[test]
fn explain_of_a_view_recorded_without_names_runs_nothing() {
    let db = Database::open("memory://view_names_old_explain").unwrap();
    setup(&db);
    db.engine()
        .create_view("old", "SELECT * FROM t".to_string(), None, false)
        .unwrap();
    let plan: Vec<String> = db
        .query("EXPLAIN SELECT * FROM (SELECT a FROM old) AS d", ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert!(
        plan.iter().any(|line| line.contains("View old")),
        "{plan:?}"
    );
    db.execute("ALTER TABLE t RENAME COLUMN a TO z", ())
        .unwrap();
    assert_eq!(
        columns(&db, "SELECT * FROM old"),
        ["id", "z", "b"],
        "the EXPLAIN ran the view's query, which recorded its names"
    );
}

#[test]
fn a_column_name_longer_than_a_u16_survives_the_record() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/db", dir.path().display());
    let long = "x".repeat(70_000);
    for checkpoint in [false, true] {
        {
            let db = Database::open(&dsn).unwrap();
            db.execute("DROP VIEW IF EXISTS w", ()).unwrap();
            db.execute("DROP TABLE IF EXISTS t", ()).unwrap();
            setup(&db);
            db.execute(
                &format!("CREATE VIEW w AS SELECT a AS {long}, b FROM t"),
                (),
            )
            .unwrap();
            if checkpoint {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            db.close().unwrap();
        }
        let db = Database::open(&dsn).unwrap();
        let names = columns(&db, "SELECT * FROM w");
        assert_eq!(names.len(), 2, "checkpoint {checkpoint}");
        assert!(
            names[0] == long && names[1] == "b",
            "checkpoint {checkpoint}"
        );
        db.close().unwrap();
    }
}

#[test]
fn a_view_keeps_its_names_when_the_database_opens_again() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/db", dir.path().display());
    for checkpoint in [false, true] {
        {
            let db = Database::open(&dsn).unwrap();
            db.execute("DROP VIEW IF EXISTS w", ()).unwrap();
            db.execute("DROP TABLE IF EXISTS t", ()).unwrap();
            setup(&db);
            db.execute("CREATE VIEW w AS SELECT * FROM t", ()).unwrap();
            if checkpoint {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            db.close().unwrap();
        }
        let db = Database::open(&dsn).unwrap();
        db.execute("ALTER TABLE t RENAME COLUMN a TO z", ())
            .unwrap();
        assert_eq!(
            columns(&db, "SELECT * FROM w"),
            ["id", "a", "b"],
            "checkpoint {checkpoint}"
        );
        db.close().unwrap();
    }
}

#[test]
fn a_view_keeps_its_names_through_a_restore() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}/db", dir.path().display())).unwrap();
    setup(&db);
    db.execute("CREATE VIEW w AS SELECT * FROM t", ()).unwrap();
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();
    db.execute("PRAGMA RESTORE", ()).unwrap();
    db.execute("ALTER TABLE t RENAME COLUMN a TO z", ())
        .unwrap();
    assert_eq!(columns(&db, "SELECT * FROM w"), ["id", "a", "b"]);
    db.close().unwrap();
}
