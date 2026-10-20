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

//! A CHECK, on a column or on the table, is an expression over the whole
//! row, enforced on every write path. DEFAULT and CHECK may not hold a
//! subquery, an aggregate or a window function, and CREATE refuses them.

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

fn show_create(db: &Database) -> String {
    let mut rows = db.query("SHOW CREATE TABLE c", ()).unwrap();
    rows.next().unwrap().unwrap().get::<String>(1).unwrap()
}

/// c: CHECK (w > v) on w, CHECK (w + v < 100) on the table, row (1, 5, 1)
fn table(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, w INTEGER CHECK (w > v), v INTEGER, CHECK (w + v < 100))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, 5, 1)", ()).unwrap();
    db
}

/// Each statement must fail and leave c as it was
fn refused(db: &Database, statements: &[&str]) {
    let before: Vec<String> = db
        .query("SELECT * FROM c ORDER BY id", ())
        .unwrap()
        .map(|r| format!("{:?}", r.unwrap()))
        .collect();
    let mut wrong = Vec::new();
    for sql in statements {
        if db.execute(sql, ()).is_ok() {
            wrong.push(format!("{sql}: accepted"));
        }
    }
    let after: Vec<String> = db
        .query("SELECT * FROM c ORDER BY id", ())
        .unwrap()
        .map(|r| format!("{:?}", r.unwrap()))
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(before, after, "a refused write changed c");
}

#[test]
fn a_column_check_reads_the_other_columns_of_the_row() {
    let db = table("check_cross_column");
    db.execute("INSERT INTO c VALUES (2, 9, 3)", ()).unwrap();
    db.execute("UPDATE c SET v = 4 WHERE id = 2", ()).unwrap();
    refused(
        &db,
        &[
            "INSERT INTO c VALUES (3, 1, 5)",
            "UPDATE c SET v = 20 WHERE id = 1",
            "UPDATE c SET w = 0 WHERE v > 0",
        ],
    );
    assert_eq!(ids(&db, "SELECT id FROM c ORDER BY id"), vec![1, 2]);
}

#[test]
fn a_table_check_is_enforced_on_every_write_path() {
    let db = table("check_table_paths");
    db.execute("INSERT INTO c VALUES (2, 50, 40)", ()).unwrap();
    refused(
        &db,
        &[
            "INSERT INTO c VALUES (3, 60, 50)",
            "INSERT INTO c (id, w, v) VALUES (3, 60, 50)",
            "INSERT INTO c SELECT id + 10, w + 50, v FROM c",
            "INSERT INTO c VALUES (3, 60, 50) ON CONFLICT (id) DO UPDATE SET w = 2",
            "INSERT INTO c VALUES (1, 6, 1) ON CONFLICT (id) DO UPDATE SET w = 99",
            "INSERT INTO c VALUES (1, 6, 1) ON DUPLICATE KEY UPDATE w = 99",
            "UPDATE c SET w = 99 WHERE id = 1",
            "UPDATE c SET w = w + 50 WHERE v > 0",
        ],
    );
    let mut wrong = Vec::new();
    for (sql, params) in [
        ("UPDATE c SET w = $1 WHERE id = $2", (99i64, 1i64)),
        ("INSERT INTO c VALUES ($2 + 2, $1, $1)", (60, 1)),
    ] {
        if db.execute(sql, params).is_ok() {
            wrong.push(format!("{sql} {params:?}: accepted"));
        }
    }
    let update = db.prepare("UPDATE c SET w = $1 WHERE id = $2").unwrap();
    if update.execute((99i64, 1i64)).is_ok() {
        wrong.push("prepared UPDATE: accepted".to_string());
    }
    let mut tx = db.begin().unwrap();
    if tx.execute("UPDATE c SET w = 99 WHERE id = 1", ()).is_ok() {
        wrong.push("UPDATE in a transaction: accepted".to_string());
    }
    if tx.execute("INSERT INTO c VALUES (3, 60, 50)", ()).is_ok() {
        wrong.push("INSERT in a transaction: accepted".to_string());
    }
    tx.commit().unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(
        db.query_one::<i64, _>("SELECT SUM(w) FROM c", ()).unwrap(),
        55,
        "no row broke CHECK (w + v < 100)"
    );
}

#[test]
fn a_correlated_update_runs_the_checks() {
    let db = table("check_correlated_update");
    refused(
        &db,
        &[
            "UPDATE c SET w = (SELECT c.v - 1) WHERE id = 1",
            "UPDATE c SET w = (SELECT c.v + 100) WHERE id = 1",
            "UPDATE c SET v = (SELECT c.w + 1) WHERE v > 0",
        ],
    );
}

#[test]
fn a_check_reads_a_column_whose_name_holds_a_dot() {
    let db = Database::open("memory://check_dotted_name").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, \"a.b\" INTEGER CHECK (\"a.b\" > 0), w INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, 1, 1)", ()).unwrap();
    refused(
        &db,
        &[
            "INSERT INTO c VALUES (2, -1, 1)",
            "UPDATE c SET \"a.b\" = -1 WHERE id = 1",
            "UPDATE c SET \"a.b\" = -1 WHERE w > 0",
        ],
    );
    assert!(
        db.execute("ALTER TABLE c DROP COLUMN w", ()).is_ok(),
        "a CHECK reading only \"a.b\" keeps w droppable"
    );
}

#[test]
fn rename_follows_the_column_a_check_binds_to() {
    for (sql, name) in [
        ("ALTER TABLE c RENAME COLUMN b TO z", "rename_last_part"),
        ("ALTER TABLE c RENAME COLUMN \"a.b\" TO x", "rename_dotted"),
    ] {
        let db = Database::open(&format!("memory://check_{name}")).unwrap();
        db.execute(
            "CREATE TABLE c (id INTEGER PRIMARY KEY, \"a.b\" INTEGER CHECK (\"a.b\" > 0), b INTEGER, CHECK (b < 10))",
            (),
        )
        .unwrap();
        db.execute(sql, ()).unwrap();
        db.execute("INSERT INTO c VALUES (1, 1, 1)", ()).unwrap();
        for row in ["(2, -1, 1)", "(2, 1, 20)"] {
            assert!(
                db.execute(&format!("INSERT INTO c VALUES {row}"), ())
                    .is_err(),
                "after {sql}, {row} was stored"
            );
        }
    }
}

#[test]
fn rename_refuses_a_name_that_would_rebind_a_check() {
    let db = Database::open("memory://check_rename_rebind").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, b INTEGER, z INTEGER, CHECK (c.b > 0))",
        (),
    )
    .unwrap();
    let sql = "ALTER TABLE c RENAME COLUMN z TO \"c.b\"";
    assert!(
        db.execute(sql, ()).is_err(),
        "{sql} moved CHECK (c.b > 0) onto z"
    );
    refused(&db, &["INSERT INTO c VALUES (1, -1, 1)"]);
}

#[test]
fn an_update_runs_a_volatile_check_whatever_it_sets() {
    let db = Database::open("memory://check_volatile").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER, deadline TIMESTAMP CHECK (deadline > NOW()))",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO c VALUES (1, 1, NOW() + INTERVAL '1 second')",
        (),
    )
    .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(1200));
    refused(
        &db,
        &[
            "UPDATE c SET v = 2 WHERE id = 1",
            "UPDATE c SET v = 2 WHERE v > 0",
        ],
    );
    assert!(
        db.execute("UPDATE c SET v = $1 WHERE id = $2", (2i64, 1i64))
            .is_err(),
        "the PK UPDATE with params was accepted past the deadline"
    );
}

#[test]
fn a_foreign_key_action_runs_the_child_checks() {
    let db = Database::open("memory://check_fk_action").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE k (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON DELETE SET NULL, CHECK (pid IS NOT NULL))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO p VALUES (1)", ()).unwrap();
    db.execute("INSERT INTO k VALUES (1, 1)", ()).unwrap();
    assert!(
        db.execute("DELETE FROM p WHERE id = 1", ()).is_err(),
        "ON DELETE SET NULL broke CHECK (pid IS NOT NULL)"
    );
    assert_eq!(ids(&db, "SELECT pid FROM k"), vec![1]);

    db.execute(
        "CREATE TABLE q (id INTEGER PRIMARY KEY, code INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY, code INTEGER REFERENCES q(code) ON UPDATE CASCADE, CHECK (code < 10))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO q VALUES (1, 1)", ()).unwrap();
    db.execute("INSERT INTO m VALUES (1, 1)", ()).unwrap();
    assert!(
        db.execute("UPDATE q SET code = 20 WHERE id = 1", ())
            .is_err(),
        "ON UPDATE CASCADE broke CHECK (code < 10)"
    );
    assert_eq!(ids(&db, "SELECT code FROM m"), vec![1]);
}

#[test]
fn copy_keeps_a_table_check() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("check_copy");
    let path = dir.path().join("rows.csv");
    std::fs::write(&path, "id,w,v\n2,9,3\n3,60,50\n").unwrap();
    let sql = format!(
        "COPY c FROM '{}' WITH (FORMAT CSV, HEADER TRUE)",
        path.display()
    );
    assert!(db.execute(&sql, ()).is_err(), "COPY stored (3, 60, 50)");
    assert!(
        !ids(&db, "SELECT id FROM c ORDER BY id").contains(&3),
        "COPY stored (3, 60, 50)"
    );
}

#[test]
fn rename_column_rewrites_the_checks_that_read_it() {
    let db = table("check_rename");
    db.execute("ALTER TABLE c RENAME COLUMN v TO z", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (2, 9, 3)", ()).unwrap();
    refused(
        &db,
        &[
            "INSERT INTO c VALUES (3, 3, 9)",
            "INSERT INTO c VALUES (3, 60, 50)",
        ],
    );
    let create = show_create(&db);
    assert!(
        create.contains("CHECK ((w > z))") && create.contains("CHECK (((w + z) < 100))"),
        "{create}"
    );
}

#[test]
fn drop_column_refuses_a_column_another_check_reads() {
    let db = table("check_drop");
    for sql in ["ALTER TABLE c DROP COLUMN v", "ALTER TABLE c DROP COLUMN w"] {
        assert!(db.execute(sql, ()).is_err(), "{sql} was accepted");
    }
    db.execute("INSERT INTO c VALUES (2, 9, 3)", ()).unwrap();
    assert_eq!(ids(&db, "SELECT v FROM c ORDER BY id"), vec![1, 3]);
}

#[test]
fn dropping_a_parent_table_keeps_the_child_checks() {
    let db = Database::open("memory://check_drop_parent").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, w INTEGER, v INTEGER REFERENCES p(id), CHECK (w + v < 100))",
        (),
    )
    .unwrap();
    db.execute("DROP TABLE p", ()).unwrap();
    db.execute("INSERT INTO c VALUES (1, 5, 1)", ()).unwrap();
    refused(&db, &["INSERT INTO c VALUES (2, 60, 50)"]);
}

#[test]
fn create_refuses_a_default_or_check_it_cannot_keep() {
    let db = Database::open("memory://check_create_refuses").unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    let mut wrong = Vec::new();
    for (name, columns) in [
        ("d1", "v INTEGER DEFAULT (SELECT MAX(x) FROM u)"),
        ("d2", "v INTEGER DEFAULT (1 + (SELECT 1))"),
        ("d3", "v INTEGER DEFAULT COUNT(*)"),
        ("c1", "v INTEGER CHECK (v < (SELECT MAX(x) FROM u))"),
        ("c2", "v INTEGER CHECK (EXISTS (SELECT 1 FROM u))"),
        ("c3", "v INTEGER CHECK (v IN (SELECT x FROM u))"),
        ("c4", "v INTEGER CHECK (SUM(v) > 0)"),
        ("c5", "v INTEGER CHECK (ROW_NUMBER() OVER () > 0)"),
        ("c6", "v INTEGER CHECK (nope > 0)"),
        ("t1", "v INTEGER, CHECK (v < (SELECT MAX(x) FROM u))"),
        ("t2", "v INTEGER, CHECK (COUNT(*) > 0)"),
        ("t3", "v INTEGER, CHECK (nope > 0)"),
    ] {
        let sql = format!("CREATE TABLE {name} (id INTEGER PRIMARY KEY, {columns})");
        if db.execute(&sql, ()).is_ok() {
            wrong.push(format!("{sql}: accepted"));
        } else if db.execute(&format!("SELECT * FROM {name}"), ()).is_ok() {
            wrong.push(format!("{sql}: refused but created {name}"));
        }
    }
    let sql = "ALTER TABLE u ADD COLUMN y INTEGER DEFAULT (SELECT 1)";
    if db.execute(sql, ()).is_ok() {
        wrong.push(format!("{sql}: accepted"));
    }
    db.execute("INSERT INTO u VALUES (1, 1)", ()).unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn show_create_table_carries_a_table_check() {
    let db = table("check_show_create");
    let create = show_create(&db);
    assert!(create.contains(", CHECK (((w + v) < 100)))"), "{create}");
}

fn reopened(name: &str, keep: impl Fn(&Database)) {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}/{name}?checkpoint_on_close=off",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE c (id INTEGER PRIMARY KEY, w INTEGER CHECK (w > v), v INTEGER, CHECK (w + v < 100))",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO c VALUES (1, 5, 1)", ()).unwrap();
        keep(&db);
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    refused(
        &db,
        &[
            "INSERT INTO c VALUES (2, 1, 5)",
            "INSERT INTO c VALUES (2, 60, 50)",
        ],
    );
}

#[test]
fn a_table_check_survives_wal_replay() {
    reopened("check_wal", |_| {});
}

#[test]
fn a_table_check_survives_a_snapshot() {
    reopened("check_snapshot", |db| {
        db.create_snapshot().unwrap();
        db.restore_snapshot(None).unwrap();
        refused(db, &["INSERT INTO c VALUES (2, 60, 50)"]);
    });
}

#[test]
fn a_table_check_survives_a_checkpoint() {
    reopened("check_checkpoint", |db| {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
}
