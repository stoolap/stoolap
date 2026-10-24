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

//! A write never stores a row a constraint would refuse: a constraint is
//! kept and enforced or refused when it is declared, and an evaluation
//! error is the statement's error on every write path

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}

#[test]
fn add_column_refuses_the_constraints_it_cannot_keep() {
    let db = Database::open("memory://write_path_add_column").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap();
    let mut wrong = Vec::new();
    for constraint in [
        "CHECK (w < 3)",
        "UNIQUE",
        "PRIMARY KEY",
        "AUTO_INCREMENT",
        "REFERENCES p(id)",
    ] {
        let sql = format!("ALTER TABLE c ADD COLUMN w INTEGER {constraint}");
        if db.execute(&sql, ()).is_ok() {
            wrong.push(format!("{sql}: accepted"));
            db.execute("ALTER TABLE c DROP COLUMN w", ()).unwrap();
        }
    }
    // Nothing changed: the table still takes rows of two columns
    db.execute("INSERT INTO c VALUES (2, 1)", ()).unwrap();
    db.execute("ALTER TABLE c ADD COLUMN w INTEGER NOT NULL DEFAULT 2", ())
        .unwrap();
    assert_eq!(count(&db, "SELECT SUM(w) FROM c"), 4);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_check_that_fails_to_evaluate_fails_the_update() {
    let db = Database::open("memory://write_path_check_error").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v * 9223372036854775807 < 10))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();
    assert!(
        db.execute("INSERT INTO t VALUES (2, 2)", ()).is_err(),
        "INSERT raises the CHECK's overflow"
    );
    assert!(
        db.execute("UPDATE t SET v = 2 WHERE id = 1", ()).is_err(),
        "UPDATE raises the CHECK's overflow"
    );
    assert_eq!(count(&db, "SELECT SUM(v) FROM t"), 0, "no row took 2");
}

#[test]
fn copy_raises_a_default_error_only_for_rows_that_take_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open("memory://write_path_copy_default").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER DEFAULT (9223372036854775807 + 1))",
        (),
    )
    .unwrap();
    let with_v = dir.path().join("with_v.csv");
    std::fs::write(&with_v, "id,v\n1,5\n").unwrap();
    db.execute(
        &format!(
            "COPY t FROM '{}' WITH (FORMAT CSV, HEADER TRUE)",
            with_v.display()
        ),
        (),
    )
    .unwrap();
    let without_v = dir.path().join("without_v.csv");
    std::fs::write(&without_v, "id\n2\n").unwrap();
    let result = db.execute(
        &format!(
            "COPY t (id) FROM '{}' WITH (FORMAT CSV, HEADER TRUE)",
            without_v.display()
        ),
        (),
    );
    assert!(result.is_err(), "a row taking the failing default fails");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 1);
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM t WHERE v IS NULL"),
        0,
        "no row stored the failing default as NULL"
    );
}

#[test]
fn copy_runs_a_volatile_default_for_each_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open("memory://write_path_copy_volatile").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, r FLOAT DEFAULT (RANDOM()))",
        (),
    )
    .unwrap();
    let path = dir.path().join("ids.csv");
    let ids: Vec<String> = (1..=50).map(|i| i.to_string()).collect();
    std::fs::write(&path, format!("id\n{}\n", ids.join("\n"))).unwrap();
    db.execute(
        &format!(
            "COPY t (id) FROM '{}' WITH (FORMAT CSV, HEADER TRUE)",
            path.display()
        ),
        (),
    )
    .unwrap();
    assert!(
        count(&db, "SELECT COUNT(DISTINCT r) FROM t") > 1,
        "each row runs RANDOM() as INSERT does"
    );
}

#[test]
fn a_check_runs_on_a_null_value_and_only_a_null_result_passes() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open("memory://write_path_check_null").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v IS NOT NULL), \
         w INTEGER CHECK (w > 0))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 1)", ()).unwrap();
    let mut wrong = Vec::new();
    for sql in [
        "INSERT INTO t VALUES (2, NULL, 1)",
        "INSERT INTO t (id, w) VALUES (3, 1)",
        "UPDATE t SET v = NULL WHERE id = 1",
        "INSERT INTO t VALUES (1, 1, 1) ON DUPLICATE KEY UPDATE v = NULL",
        "INSERT INTO t VALUES (6, NULL, 1) ON DUPLICATE KEY UPDATE v = 1",
    ] {
        if db.execute(sql, ()).is_ok() {
            wrong.push(format!("{sql}: accepted a NULL v"));
        }
    }
    let path = dir.path().join("null_v.csv");
    std::fs::write(&path, "id,v,w\n4,,1\n").unwrap();
    let copy = format!(
        "COPY t FROM '{}' WITH (FORMAT CSV, HEADER TRUE)",
        path.display()
    );
    if db.execute(&copy, ()).is_ok() {
        wrong.push("COPY: accepted a NULL v".to_string());
    }
    // A NULL w makes w > 0 NULL, which passes
    db.execute("INSERT INTO t VALUES (5, 5, NULL)", ()).unwrap();
    db.execute("UPDATE t SET w = NULL WHERE id = 1", ())
        .unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE v IS NULL"), 0);
}

#[test]
fn not_null_refuses_an_update_on_hot_and_sealed_rows() {
    let statements = [
        "UPDATE t SET v = NULL WHERE id = 1",
        "UPDATE t SET v = NULL WHERE k = 1",
        "UPDATE t SET v = NULLIF(v, 9) WHERE id = 1",
        "UPDATE t SET v = NULL",
        "UPDATE t SET v = CASE WHEN id = 2 THEN NULL ELSE 5 END",
        "UPDATE t SET v = NULL WHERE id IN (SELECT id FROM t WHERE k = 2)",
        "INSERT INTO t VALUES (1, 1, 1) ON CONFLICT (id) DO UPDATE SET v = NULL",
    ];
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 1, 9), (2, 2, 9)", ())
            .unwrap();
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for sql in statements {
            if db.execute(sql, ()).is_ok() {
                wrong.push(format!("sealed={sealed} {sql}: accepted"));
            }
        }
        let mut tx = db.begin().unwrap();
        tx.execute("UPDATE t SET v = 7 WHERE id = 1", ()).unwrap();
        for sql in [
            "UPDATE t SET v = NULL WHERE k = 1",
            "UPDATE t SET v = NULL WHERE k = 2",
        ] {
            if tx.execute(sql, ()).is_ok() {
                wrong.push(format!("sealed={sealed} in a transaction {sql}: accepted"));
            }
        }
        tx.commit().unwrap();
        assert_eq!(
            count(&db, "SELECT SUM(v) FROM t"),
            16,
            "sealed={sealed}: a refused statement changed a row"
        );
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
