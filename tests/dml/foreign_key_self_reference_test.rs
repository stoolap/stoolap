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

//! A foreign key from a table into itself

use stoolap::Database;

/// The rows of `s` as `[id, up]`, a NULL `up` read as 0
fn rows(db: &Database) -> Vec<[i64; 2]> {
    db.query("SELECT id, COALESCE(up, 0) FROM s ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            [row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()]
        })
        .collect()
}

/// `s` keyed as `key` with `inserts` run, on the hot store and then sealed
fn each_layout(name: &str, key: &str, inserts: &[&str], check: impl Fn(&Database)) {
    let tables = |dsn: &str, sealed: bool| {
        let db = Database::open(dsn).unwrap();
        db.execute(
            &format!("CREATE TABLE s (id INTEGER PRIMARY KEY, up INTEGER {key}, n INTEGER)"),
            (),
        )
        .unwrap();
        for sql in inserts {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db
    };
    check(&tables(
        &format!("memory://fk_self_reference_{name}"),
        false,
    ));
    let dir = tempfile::tempdir().unwrap();
    check(&tables(&format!("file://{}", dir.path().display()), true));
}

/// A chain of `len` rows, each pointing at the one before it
fn chain(len: i64) -> String {
    let values: Vec<String> = (1..=len)
        .map(|id| match id {
            1 => "(1, NULL, 0)".to_string(),
            _ => format!("({id}, {}, 0)", id - 1),
        })
        .collect();
    format!("INSERT INTO s VALUES {}", values.join(", "))
}

#[test]
fn a_table_can_reference_its_own_key() {
    for (n, create) in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, up INTEGER REFERENCES s(id))",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, up INTEGER REFERENCES s)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, up INTEGER, FOREIGN KEY (up) REFERENCES s(id))",
        "CREATE TABLE S (id INTEGER PRIMARY KEY, up INTEGER REFERENCES s(id))",
    ]
    .iter()
    .enumerate()
    {
        let db = Database::open(&format!("memory://fk_self_reference_create_{n}")).unwrap();
        db.execute(create, ()).unwrap();
        db.execute("INSERT INTO s VALUES (1, NULL), (2, 1)", ())
            .unwrap();
        assert!(
            db.execute("INSERT INTO s VALUES (3, 9)", ()).is_err(),
            "{create}: a missing parent is refused"
        );
        assert_eq!(rows(&db), [[1, 0], [2, 1]], "{create}");
    }
}

#[test]
fn a_self_reference_names_a_unique_column_of_the_table_being_created() {
    for (n, create) in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, code INTEGER UNIQUE, up INTEGER REFERENCES s(code))",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, code INTEGER, up INTEGER REFERENCES s(code), UNIQUE (code))",
    ]
    .iter()
    .enumerate()
    {
        let db = Database::open(&format!("memory://fk_self_reference_unique_{n}")).unwrap();
        db.execute(create, ()).unwrap();
        db.execute("INSERT INTO s VALUES (1, 10, NULL), (2, 20, 10)", ())
            .unwrap();
        assert!(db.execute("INSERT INTO s VALUES (3, 30, 1)", ()).is_err());
    }
    let db = Database::open("memory://fk_self_reference_unique_refused").unwrap();
    for create in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, code INTEGER, up INTEGER REFERENCES s(code))",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, up INTEGER REFERENCES s(nope))",
        "CREATE TABLE s (id INTEGER, up INTEGER REFERENCES s)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY REFERENCES s(id))",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, code INTEGER UNIQUE REFERENCES s(code))",
    ] {
        assert!(db.execute(create, ()).is_err(), "{create}");
    }
}

#[test]
fn an_insert_finds_parents_among_its_own_rows() {
    each_layout("insert", "REFERENCES s(id)", &[], |db| {
        for sql in [
            "INSERT INTO s VALUES (1, 1, 0)",
            "INSERT INTO s VALUES (2, 3, 0), (3, 2, 0)",
            "INSERT INTO s (id, up) VALUES (4, 5), (5, 4)",
            "INSERT INTO s SELECT id + 10, up + 10, 0 FROM s",
        ] {
            db.execute(sql, ()).unwrap();
        }
        db.execute("INSERT INTO s VALUES ($1, $2, 0), ($2, $1, 0)", (21, 22))
            .unwrap();
        let insert = db.prepare("INSERT INTO s VALUES ($1, $2, 0)").unwrap();
        insert.execute((31, 31)).unwrap();
        let want = [
            [1, 1],
            [2, 3],
            [3, 2],
            [4, 5],
            [5, 4],
            [11, 11],
            [12, 13],
            [13, 12],
            [14, 15],
            [15, 14],
            [21, 22],
            [22, 21],
            [31, 31],
        ];
        assert_eq!(rows(db), want);
        for sql in [
            "INSERT INTO s VALUES (40, 41, 0)",
            "INSERT INTO s VALUES (40, NULL, 0), (41, 42, 0)",
            "INSERT INTO s SELECT id + 100, up + 200, 0 FROM s",
        ] {
            assert!(db.execute(sql, ()).is_err(), "{sql}");
            assert_eq!(rows(db), want, "{sql} writes nothing");
        }
        assert!(insert.execute((41, 42)).is_err());
        db.execute("BEGIN", ()).unwrap();
        assert!(db.execute("INSERT INTO s VALUES (40, 41, 0)", ()).is_err());
        db.execute("INSERT INTO s VALUES (42, 42, 0)", ()).unwrap();
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(rows(db).len(), want.len() + 1);
    });
}

#[test]
fn an_upsert_finds_a_parent_its_own_update_writes() {
    each_layout(
        "upsert",
        "REFERENCES s(id)",
        &["INSERT INTO s VALUES (1, NULL, 0)"],
        |db| {
            db.execute(
                "INSERT INTO s VALUES (1, 1, 0) ON CONFLICT (id) DO UPDATE SET up = 1",
                (),
            )
            .unwrap();
            db.execute(
                "INSERT INTO s VALUES (2, 2, 0) ON CONFLICT (id) DO UPDATE SET up = 1",
                (),
            )
            .unwrap();
            assert!(db
                .execute(
                    "INSERT INTO s VALUES (2, 2, 0) ON CONFLICT (id) DO UPDATE SET up = 9",
                    ()
                )
                .is_err());
            assert_eq!(rows(db), [[1, 1], [2, 2]]);
        },
    );
}

#[test]
fn a_cascade_delete_reaches_every_row_beneath_whatever_the_depth() {
    each_layout(
        "delete_cascade",
        "REFERENCES s(id) ON DELETE CASCADE",
        &[
            &chain(40),
            "INSERT INTO s VALUES (100, NULL, 0), (101, 100, 0)",
        ],
        |db| {
            db.execute("DELETE FROM s WHERE id = 30", ()).unwrap();
            assert_eq!(rows(db).len(), 29 + 2);
            db.execute("DELETE FROM s WHERE id = 1", ()).unwrap();
            assert_eq!(rows(db), [[100, 0], [101, 100]]);
        },
    );
}

#[test]
fn a_cascade_delete_round_a_cycle_deletes_each_row_once() {
    each_layout(
        "delete_cycle",
        "REFERENCES s(id) ON DELETE CASCADE",
        &["INSERT INTO s VALUES (1, 3, 0), (2, 1, 0), (3, 2, 0), (4, 4, 0), (5, NULL, 0)"],
        |db| {
            assert_eq!(db.execute("DELETE FROM s WHERE id = 2", ()).unwrap(), 1);
            assert_eq!(rows(db), [[4, 4], [5, 0]]);
            db.execute("DELETE FROM s WHERE id = 4", ()).unwrap();
            assert_eq!(rows(db), [[5, 0]]);
        },
    );
}

#[test]
fn a_set_null_delete_clears_the_rows_that_stay() {
    each_layout(
        "delete_set_null",
        "REFERENCES s(id) ON DELETE SET NULL",
        &[&chain(4), "INSERT INTO s VALUES (5, 5, 0), (6, 5, 0)"],
        |db| {
            db.execute("DELETE FROM s WHERE id = 2", ()).unwrap();
            assert_eq!(rows(db), [[1, 0], [3, 0], [4, 3], [5, 5], [6, 5]]);
            db.execute("DELETE FROM s WHERE id IN (3, 4, 5)", ())
                .unwrap();
            assert_eq!(rows(db), [[1, 0], [6, 0]]);
            db.execute("DELETE FROM s", ()).unwrap();
            assert!(rows(db).is_empty());
        },
    );
}

#[test]
fn a_restricting_key_counts_only_the_rows_the_delete_leaves() {
    for (n, action) in ["ON DELETE RESTRICT", "ON DELETE NO ACTION", ""]
        .iter()
        .enumerate()
    {
        each_layout(
            &format!("delete_restrict_{n}"),
            &format!("REFERENCES s(id) {action}"),
            &[&chain(4), "INSERT INTO s VALUES (5, 5, 0)"],
            |db| {
                let all = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 5]];
                for sql in [
                    "DELETE FROM s WHERE id = 1",
                    "DELETE FROM s WHERE id IN (1, 2)",
                    "DELETE FROM s WHERE id <= 3",
                ] {
                    assert!(db.execute(sql, ()).is_err(), "{action}: {sql}");
                    assert_eq!(rows(db), all, "{action}: {sql} writes nothing");
                }
                db.execute("DELETE FROM s WHERE id = 5", ()).unwrap();
                db.execute("DELETE FROM s WHERE id >= 3", ()).unwrap();
                assert_eq!(rows(db), [[1, 0], [2, 1]]);
                db.execute("DELETE FROM s", ()).unwrap();
                assert!(rows(db).is_empty());
            },
        );
    }
}

#[test]
fn a_cascade_delete_is_refused_by_a_restricting_key_into_a_row_it_removes() {
    each_layout(
        "delete_cascade_restrict",
        "REFERENCES s(id) ON DELETE CASCADE",
        &[&chain(3)],
        |db| {
            db.execute(
                "CREATE TABLE r (id INTEGER PRIMARY KEY, s_id INTEGER REFERENCES s(id) ON DELETE RESTRICT)",
                (),
            )
            .unwrap();
            db.execute("INSERT INTO r VALUES (1, 3)", ()).unwrap();
            assert!(db.execute("DELETE FROM s WHERE id = 1", ()).is_err());
            assert_eq!(rows(db), [[1, 0], [2, 1], [3, 2]]);
            db.execute("DELETE FROM r", ()).unwrap();
            db.execute("DELETE FROM s WHERE id = 1", ()).unwrap();
            assert!(rows(db).is_empty());
        },
    );
}

#[test]
fn a_restricting_key_beside_a_cascade_into_one_row_lets_the_row_go() {
    let db = Database::open("memory://fk_self_reference_two_keys").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, a INTEGER REFERENCES p(id) ON DELETE CASCADE, b INTEGER REFERENCES p(id) ON DELETE RESTRICT, k INTEGER REFERENCES p(id) ON DELETE SET NULL)",
        "INSERT INTO p VALUES (1), (2)",
        "INSERT INTO c VALUES (1, 1, 1, 1), (2, 2, 1, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert!(db.execute("DELETE FROM p WHERE id = 1", ()).is_err());
    db.execute("DELETE FROM p WHERE id = 2", ()).unwrap();
    db.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM c", ())
            .unwrap(),
        0
    );
}

#[test]
fn a_set_null_into_a_row_a_cascade_removes_is_not_asked() {
    each_layout(
        "null_and_gone",
        "REFERENCES s(id) ON DELETE SET NULL",
        &["INSERT INTO s VALUES (1, NULL, 0), (2, 1, 0)"],
        |db| {
            db.execute(
                "CREATE TABLE g (id INTEGER PRIMARY KEY, s_id INTEGER REFERENCES s(id) ON DELETE CASCADE)",
                (),
            )
            .unwrap();
            db.execute("INSERT INTO g VALUES (1, 2)", ()).unwrap();
            db.execute("DELETE FROM s WHERE id IN (1, 2)", ()).unwrap();
            assert!(rows(db).is_empty());
        },
    );
}

#[test]
fn a_set_null_delete_refuses_a_column_that_cannot_be_null_only_for_a_row_that_stays() {
    let db = Database::open("memory://fk_self_reference_not_null").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, a INTEGER REFERENCES p(id) ON DELETE CASCADE, k INTEGER REFERENCES p(id) ON DELETE SET NULL, CHECK (k IS NOT NULL))",
        "INSERT INTO p VALUES (1), (2)",
        "INSERT INTO c VALUES (1, 1, 1), (2, 2, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert!(db.execute("DELETE FROM p WHERE id = 1", ()).is_err());
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM c", ())
            .unwrap(),
        2
    );
    db.execute("DELETE FROM p", ()).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM c", ())
            .unwrap(),
        0
    );
}

#[test]
fn a_delete_inside_a_transaction_sees_its_own_rows() {
    each_layout(
        "delete_txn",
        "REFERENCES s(id) ON DELETE CASCADE",
        &[&chain(3)],
        |db| {
            db.execute("BEGIN", ()).unwrap();
            db.execute("INSERT INTO s VALUES (4, 3, 0), (5, 4, 0)", ())
                .unwrap();
            db.execute("DELETE FROM s WHERE id = 2", ()).unwrap();
            assert_eq!(rows(db), [[1, 0]]);
            db.execute("ROLLBACK", ()).unwrap();
            assert_eq!(rows(db), [[1, 0], [2, 1], [3, 2]]);
        },
    );
}

#[test]
fn a_table_whose_rows_reference_only_itself_is_dropped_and_truncated() {
    each_layout("drop", "REFERENCES s(id)", &[&chain(3)], |db| {
        db.execute("TRUNCATE TABLE s", ()).unwrap();
        assert!(rows(db).is_empty());
        db.execute(&chain(3), ()).unwrap();
        db.execute(
            "CREATE TABLE r (id INTEGER PRIMARY KEY, s_id INTEGER REFERENCES s(id))",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO r VALUES (1, 2)", ()).unwrap();
        assert!(db.execute("TRUNCATE TABLE s", ()).is_err());
        assert!(db.execute("DROP TABLE s", ()).is_err());
        db.execute("DROP TABLE r", ()).unwrap();
        db.execute("DROP TABLE s", ()).unwrap();
    });
}

#[test]
fn an_update_of_the_key_carries_the_rows_that_reference_it() {
    each_layout(
        "update_cascade",
        "REFERENCES s(id) ON UPDATE CASCADE",
        &[&chain(3), "INSERT INTO s VALUES (4, 4, 0)"],
        |db| {
            db.execute("UPDATE s SET id = id + 10", ()).unwrap();
            assert_eq!(rows(db), [[11, 0], [12, 11], [13, 12], [14, 14]]);
            db.execute("UPDATE s SET id = 20 WHERE id = 12", ())
                .unwrap();
            assert_eq!(rows(db), [[11, 0], [13, 20], [14, 14], [20, 11]]);
            db.execute("UPDATE s SET id = id + 100, up = NULL WHERE id = 14", ())
                .unwrap();
            assert_eq!(rows(db), [[11, 0], [13, 20], [20, 11], [114, 0]]);
        },
    );
    each_layout(
        "update_set_null",
        "REFERENCES s(id) ON UPDATE SET NULL",
        &[&chain(3)],
        |db| {
            db.execute("UPDATE s SET id = 10 WHERE id = 1", ()).unwrap();
            assert_eq!(rows(db), [[2, 0], [3, 2], [10, 0]]);
        },
    );
}

#[test]
fn an_update_finds_a_parent_its_own_rows_write() {
    each_layout(
        "update_parent",
        "REFERENCES s(id)",
        &["INSERT INTO s VALUES (1, NULL, 0), (2, NULL, 0)"],
        |db| {
            db.execute("BEGIN", ()).unwrap();
            db.execute("UPDATE s SET id = 5, up = 5 WHERE id = 1", ())
                .unwrap();
            db.execute("COMMIT", ()).unwrap();
            db.execute("UPDATE s SET up = 5 WHERE id = 2", ()).unwrap();
            assert!(db.execute("UPDATE s SET up = 9 WHERE id = 2", ()).is_err());
            db.execute("BEGIN", ()).unwrap();
            assert!(db.execute("UPDATE s SET up = 9 WHERE id = 2", ()).is_err());
            db.execute("COMMIT", ()).unwrap();
            assert_eq!(rows(db), [[2, 5], [5, 5]]);
        },
    );
}

#[test]
fn a_renamed_table_keeps_its_key_into_itself() {
    let db = Database::open("memory://fk_self_reference_rename").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, up INTEGER REFERENCES t(id) ON DELETE CASCADE)",
        "INSERT INTO t VALUES (1, NULL), (2, 1)",
        "ALTER TABLE t RENAME TO s",
        "INSERT INTO s VALUES (3, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert!(db.execute("INSERT INTO s VALUES (4, 9)", ()).is_err());
    db.execute("DELETE FROM s WHERE id = 1", ()).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM s", ())
            .unwrap(),
        0
    );
}

#[test]
fn a_key_into_its_own_table_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE s (id INTEGER PRIMARY KEY, up INTEGER REFERENCES s(id) ON DELETE CASCADE, n INTEGER)",
            (),
        )
        .unwrap();
        db.execute(&chain(3), ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert!(db.execute("INSERT INTO s VALUES (4, 9, 0)", ()).is_err());
    db.execute("DELETE FROM s WHERE id = 2", ()).unwrap();
    assert_eq!(rows(&db), [[1, 0]]);
}

#[test]
fn a_copy_finds_parents_among_its_own_rows() {
    use std::io::Write;
    let db = Database::open("memory://fk_self_reference_copy").unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, up INTEGER REFERENCES s(id), n INTEGER)",
        (),
    )
    .unwrap();
    let copy = |lines: &[&str], format: &str| {
        let file = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(file.as_file(), "{line}").unwrap();
        }
        db.execute(
            &format!(
                "COPY s FROM '{}' WITH (FORMAT {format})",
                file.path().display()
            ),
            (),
        )
    };
    copy(&["1,2,0", "2,1,0", "3,3,0"], "CSV, HEADER false").unwrap();
    copy(
        &[
            r#"{"id": 4, "up": 5, "n": 0}"#,
            r#"{"id": 5, "up": 4, "n": 0}"#,
        ],
        "JSON",
    )
    .unwrap();
    let want = [[1, 2], [2, 1], [3, 3], [4, 5], [5, 4]];
    assert_eq!(rows(&db), want);
    assert!(copy(&["6,,0", "7,8,0"], "CSV, HEADER false").is_err());
    assert!(copy(&[r#"{"id": 6, "up": 9, "n": 0}"#], "JSON").is_err());
    assert_eq!(rows(&db), want);
}

#[test]
fn a_restricting_key_beside_a_cascade_into_a_table_no_key_references_lets_the_row_go() {
    let db = Database::open("memory://fk_self_reference_leaf_restrict").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, a INTEGER REFERENCES p(id) ON DELETE CASCADE, b INTEGER REFERENCES p(id) ON DELETE RESTRICT)",
        "INSERT INTO p VALUES (1), (2)",
        "INSERT INTO c VALUES (1, 1, 1), (2, 2, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert!(db.execute("DELETE FROM p WHERE id = 1", ()).is_err());
    db.execute("DELETE FROM p WHERE id = 2", ()).unwrap();
    db.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM c", ())
            .unwrap(),
        0
    );
}

#[test]
fn an_upsert_checks_its_keys_into_its_own_table_as_the_statement_leaves_them() {
    each_layout(
        "upsert_end",
        "REFERENCES s(id)",
        &["INSERT INTO s VALUES (1, NULL, 0)"],
        |db| {
            let upsert = "ON CONFLICT (id) DO UPDATE SET up = EXCLUDED.up";
            for values in ["(1, 2, 0), (2, NULL, 0)", "(3, 99, 0), (3, NULL, 0)"] {
                db.execute(&format!("INSERT INTO s VALUES {values} {upsert}"), ())
                    .unwrap();
            }
            assert_eq!(rows(db), [[1, 2], [2, 0], [3, 0]]);
            for values in ["(1, 98, 0), (4, NULL, 0)", "(4, NULL, 0), (4, 97, 0)"] {
                let sql = format!("INSERT INTO s VALUES {values} {upsert}");
                assert!(db.execute(&sql, ()).is_err(), "{sql}");
                assert_eq!(rows(db), [[1, 2], [2, 0], [3, 0]], "{sql} writes nothing");
            }
        },
    );
}
