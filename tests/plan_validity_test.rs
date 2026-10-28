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

//! A cached plan holds across a schema change to a table or view it does
//! not read, and answers as a plan made anew after a change to one it reads

use stoolap::Database;

const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 0, 30), (4, 1, 40), (5, 2, 50), (6, 0, 60)",
    "INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)",
    "CREATE VIEW vj AS SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
     WHERE t.b > (SELECT MAX(x) FROM u)",
    "CREATE VIEW vu AS SELECT * FROM u",
];

/// Each read shape, as each plan kind holds it
const READS: &[(&str, &str)] = &[
    ("PK lookup", "SELECT * FROM u WHERE id = 2"),
    ("COUNT(*)", "SELECT COUNT(*) FROM u"),
    ("COUNT(DISTINCT)", "SELECT COUNT(DISTINCT x) FROM u"),
    (
        "prepared select",
        "SELECT * FROM t WHERE b > (SELECT MIN(x) FROM u) * 20",
    ),
    (
        "prepared grouped",
        "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) >= (SELECT MIN(x) FROM u)",
    ),
    (
        "prepared join",
        "SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > (SELECT MAX(x) FROM u)",
    ),
    ("view of a join", "SELECT * FROM vj"),
    (
        "join over a view of a star",
        "SELECT t.id, vu.x FROM t JOIN vu ON vu.id = t.a WHERE t.b > (SELECT MAX(x) FROM u)",
    ),
    (
        "block over a view of a star",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM vu WHERE vu.id = t.a AND vu.x > 1)",
    ),
    ("view of a star", "SELECT * FROM vu WHERE id > (SELECT 0)"),
    ("plain", "SELECT * FROM u"),
];

// Every test here builds plans, which the compile counters count process-wide
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://plan_validity_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = result.columns().join("|");
    let mut out = result
        .map(|row| {
            row.map_err(|e| e.to_string()).map(|row| {
                (0..row.len())
                    .map(|i| {
                        row.get::<Option<String>>(i)
                            .unwrap()
                            .unwrap_or_else(|| "NULL".into())
                    })
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    out.sort();
    out.insert(0, columns);
    Ok(out)
}

/// Each change to a table or view the reads read, as statements
const CHANGES: &[(&str, &[&str])] = &[
    ("RENAME COLUMN alone", &["ALTER TABLE u RENAME COLUMN x TO z"]),
    (
        "a table made a view of its name",
        &[
            "DROP VIEW vj",
            "DROP VIEW vu",
            "CREATE TABLE u2 (id INTEGER PRIMARY KEY, x INTEGER)",
            "INSERT INTO u2 VALUES (1, 2), (2, 3), (3, 1)",
            "DROP TABLE u",
            "CREATE VIEW u AS SELECT id, x FROM u2",
            "CREATE VIEW vj AS SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
             WHERE t.b > (SELECT MAX(x) FROM u)",
            "CREATE VIEW vu AS SELECT * FROM u",
        ],
    ),
    ("ADD COLUMN", &["ALTER TABLE u ADD COLUMN y INTEGER DEFAULT 7"]),
    ("RENAME COLUMN", &["ALTER TABLE u RENAME COLUMN x TO z", "ALTER TABLE u ADD COLUMN x INTEGER DEFAULT 1"]),
    (
        "made again with its columns moved",
        &[
            "DROP VIEW vj",
            "DROP VIEW vu",
            "DROP TABLE u",
            "CREATE TABLE u (x INTEGER, id INTEGER PRIMARY KEY)",
            "INSERT INTO u (id, x) VALUES (1, 3), (2, 1), (3, 2), (4, 9)",
            "CREATE VIEW vj AS SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
             WHERE t.b > (SELECT MAX(x) FROM u)",
            "CREATE VIEW vu AS SELECT * FROM u",
        ],
    ),
    (
        "views defined again",
        &[
            "DROP VIEW vj",
            "CREATE VIEW vj AS SELECT t.id, t.b AS x FROM t WHERE t.b < (SELECT MAX(x) FROM u) * 20",
            "DROP VIEW vu",
            "CREATE VIEW vu AS SELECT id, x * 10 AS x FROM u",
        ],
    ),
    (
        "renamed away and made again",
        &[
            "DROP VIEW vj",
            "DROP VIEW vu",
            "ALTER TABLE u RENAME TO u_old",
            "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, w INTEGER)",
            "INSERT INTO u VALUES (1, 5, 0), (2, 6, 0)",
            "CREATE VIEW vj AS SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
             WHERE t.b > (SELECT MAX(x) FROM u)",
            "CREATE VIEW vu AS SELECT * FROM u",
        ],
    ),
];

#[test]
fn a_plan_answers_as_a_new_one_after_a_change_to_what_it_reads() {
    let _serial = serial();
    let mut wrong = Vec::new();
    for (change_index, (change, statements)) in CHANGES.iter().enumerate() {
        let db = open(&format!("changed_{change_index}"));
        for (_, sql) in READS {
            rows(&db, sql).unwrap();
            rows(&db, sql).unwrap();
        }
        for statement in *statements {
            db.execute(statement, ())
                .unwrap_or_else(|e| panic!("{statement}: {e}"));
        }
        for (name, sql) in READS {
            let held = rows(&db, sql);
            let fresh = rows(&db, &format!("{sql} "));
            if held != fresh {
                wrong.push(format!(
                    "{name} after {change}: {held:?}, a new plan gives {fresh:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Each write, the statement reading what it left, as each plan kind holds it
const WRITES: &[(&str, &str, &str)] = &[
    (
        "PK update",
        "UPDATE u SET x = x + 1 WHERE id = 2",
        "SELECT * FROM u",
    ),
    ("PK delete", "DELETE FROM u WHERE id = 3", "SELECT * FROM u"),
    (
        "insert",
        "INSERT INTO u (id, x) SELECT MAX(id) + 1, 1 FROM u",
        "SELECT * FROM u",
    ),
    (
        "conflict insert",
        "INSERT INTO u (id, x) VALUES (1, 5) ON DUPLICATE KEY UPDATE x = x + 100",
        "SELECT * FROM u",
    ),
    (
        "prepared update",
        "UPDATE t SET b = b + 1 WHERE a = (SELECT MIN(x) FROM u)",
        "SELECT * FROM t",
    ),
];

#[test]
fn a_write_plan_writes_as_a_new_one_after_a_change_to_its_target() {
    let _serial = serial();
    let mut wrong = Vec::new();
    for (change_index, (change, statements)) in CHANGES.iter().enumerate() {
        for (write_index, (name, write, read)) in WRITES.iter().enumerate() {
            let held = open(&format!("write_held_{change_index}_{write_index}"));
            let fresh = open(&format!("write_fresh_{change_index}_{write_index}"));
            held.execute(write, ()).unwrap();
            held.execute(write, ()).unwrap();
            fresh.execute(&format!("{write} "), ()).unwrap();
            fresh.execute(&format!("{write} "), ()).unwrap();
            for db in [&held, &fresh] {
                for statement in *statements {
                    db.execute(statement, ()).unwrap();
                }
            }
            let held_write = held.execute(write, ()).map_err(|e| e.to_string());
            let fresh_write = fresh
                .execute(&format!("{write}  "), ())
                .map_err(|e| e.to_string());
            let (got, want) = (rows(&held, read), rows(&fresh, read));
            if held_write.is_ok() != fresh_write.is_ok() || got != want {
                wrong.push(format!(
                    "{name} after {change}: wrote {held_write:?} and left {got:?}, \
                     a new plan wrote {fresh_write:?} and left {want:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_parent_s_compiled_writes_keep_a_foreign_key_made_after_them() {
    let _serial = serial();
    let db = Database::open("memory://plan_validity_foreign_key").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO p VALUES (1, 0), (2, 0), (3, 0)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let delete = "DELETE FROM p WHERE id = $1";
    let update = "UPDATE p SET v = v + 1 WHERE id = $1";
    for id in [8, 9] {
        db.execute(delete, (id,)).unwrap();
        db.execute(update, (id,)).unwrap();
    }
    for sql in [
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id))",
        "INSERT INTO c VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert!(
        db.execute(delete, (1,)).is_err(),
        "a parent row a child references is not deleted"
    );
    assert_eq!(
        rows(&db, "SELECT id FROM p").unwrap(),
        ["id", "1", "2", "3"],
        "the parent rows stay"
    );
}

#[cfg(feature = "test-failpoints")]
mod held {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    use stoolap::test_failpoints::{
        after_compiled_epoch_checked, bridge_compiles, prepared_dml_built, prepared_selects_built,
        program_compiles,
    };

    fn compiles() -> [usize; 4] {
        [
            bridge_compiles(),
            program_compiles(),
            prepared_selects_built(),
            prepared_dml_built(),
        ]
    }

    /// Changes to a table and a view no shape reads
    const ELSEWHERE: &[&str] = &[
        "CREATE TABLE z (id INTEGER PRIMARY KEY)",
        "ALTER TABLE z ADD COLUMN y INTEGER",
        "ALTER TABLE z RENAME COLUMN y TO w",
        "CREATE VIEW vz AS SELECT id FROM z",
        "DROP VIEW vz",
        "DROP TABLE z",
    ];

    /// Each shape: its statements, and the read that shows what it did
    const SHAPES: &[(&str, &[&str], &str)] = &[
        (
            "prepared select",
            &["SELECT * FROM t WHERE b > (SELECT MIN(x) FROM u) * 20"],
            "SELECT * FROM t WHERE b > (SELECT MIN(x) FROM u) * 20",
        ),
        (
            "prepared grouped",
            &["SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) >= (SELECT MIN(x) FROM u)"],
            "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) >= (SELECT MIN(x) FROM u)",
        ),
        (
            "prepared join",
            &["SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > (SELECT MAX(x) FROM u)"],
            "SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > (SELECT MAX(x) FROM u)",
        ),
        ("view of a join", &["SELECT * FROM vj"], "SELECT * FROM vj"),
        (
            "prepared update",
            &["UPDATE t SET b = b WHERE a = (SELECT MIN(x) FROM u)"],
            "SELECT * FROM t",
        ),
        (
            "CREATE TABLE ... AS SELECT of a join",
            &[
                "DROP TABLE IF EXISTS c",
                "CREATE TABLE c AS SELECT t.id, u.x FROM t JOIN u ON u.id = t.a \
                 WHERE t.b > (SELECT MAX(x) FROM u)",
            ],
            "SELECT * FROM c",
        ),
    ];

    fn run(db: &Database, statements: &[&str]) {
        for sql in statements {
            db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        }
    }

    #[test]
    fn a_plan_holds_across_a_change_to_what_it_does_not_read() {
        let _serial = serial();
        let mut wrong = Vec::new();
        for (index, (name, statements, read)) in SHAPES.iter().enumerate() {
            let db = open(&format!("held_{index}"));
            run(&db, statements);
            run(&db, statements);
            let before = rows(&db, read);
            let delta = |db: &Database| {
                let compiled = compiles();
                run(db, statements);
                compiles()
                    .iter()
                    .zip(compiled)
                    .map(|(now, then)| now - then)
                    .collect::<Vec<_>>()
            };
            let held = delta(&db);
            run(&db, ELSEWHERE);
            let after = delta(&db);
            if after != held {
                wrong.push(format!(
                    "{name}: a run compiles {held:?}, after a change elsewhere {after:?}"
                ));
            }
            if rows(&db, read) != before {
                wrong.push(format!("{name}: answered otherwise"));
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    #[test]
    fn a_compiled_pk_lookup_holds_across_a_change_to_another_table() {
        let _serial = serial();
        let db = open("held_pk_lookup");
        let lookup = "SELECT * FROM u WHERE id = $1";
        for id in [1, 2] {
            db.query(lookup, (id,)).unwrap().count();
        }
        run(&db, ELSEWHERE);
        let checked = Rc::new(Cell::new(false));
        let seen = Rc::clone(&checked);
        after_compiled_epoch_checked(move || seen.set(true));
        db.query(lookup, (3,)).unwrap().count();
        assert!(
            checked.get(),
            "the compiled lookup runs without compiling again"
        );
    }
}
