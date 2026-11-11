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

//! An UPDATE that writes a referenced column's own value back

use stoolap::Database;

/// `tables` run, then `check` on the hot store and on sealed volumes
fn each_layout(name: &str, tables: &[&str], check: impl Fn(&Database)) {
    let open = |dsn: &str, sealed: bool| {
        let db = Database::open(dsn).unwrap();
        for sql in tables {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db
    };
    check(&open(&format!("memory://fk_same_value_{name}"), false));
    let dir = tempfile::tempdir().unwrap();
    check(&open(&format!("file://{}", dir.path().display()), true));
}

fn pairs(db: &Database, sql: &str) -> Vec<[i64; 2]> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            [row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()]
        })
        .collect()
}

const PARENT_AND_CHILD: &[&str] = &[
    "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
    "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id), pu INTEGER REFERENCES p(u))",
    "INSERT INTO p VALUES (1, 10), (2, 20)",
    "INSERT INTO c VALUES (1, 1, 10)",
];

#[test]
fn an_update_writing_a_referenced_value_back_is_not_refused() {
    each_layout("restrict", PARENT_AND_CHILD, |db| {
        for sql in [
            "UPDATE p SET id = id",
            "UPDATE p SET u = u",
            "UPDATE p SET u = u WHERE id = 1",
            "UPDATE p SET u = 10 WHERE id = 1",
            "UPDATE p SET id = 1, u = 10 WHERE u = 10",
            "UPDATE p SET u = CASE WHEN id = 1 THEN 10 ELSE 21 END",
        ] {
            db.execute(sql, ())
                .unwrap_or_else(|e| panic!("{sql} was refused: {e}"));
        }
        db.execute("UPDATE p SET u = $1 WHERE id = $2", (10, 1))
            .unwrap();
        let prepared = db.prepare("UPDATE p SET u = $1 WHERE id = $2").unwrap();
        prepared.execute((10, 1)).unwrap();
        let mut tx = db.begin().unwrap();
        tx.execute("UPDATE p SET id = id, u = u", ()).unwrap();
        tx.commit().unwrap();
        assert_eq!(
            pairs(db, "SELECT id, u FROM p ORDER BY id"),
            [[1, 10], [2, 21]]
        );
    });
}

#[test]
fn an_update_changing_a_referenced_value_is_still_refused() {
    each_layout("still_refused", PARENT_AND_CHILD, |db| {
        for sql in [
            "UPDATE p SET u = u + 1",
            "UPDATE p SET id = id + 10",
            "UPDATE p SET u = CASE WHEN id = 2 THEN 20 ELSE 11 END",
        ] {
            let error = db.execute(sql, ()).unwrap_err();
            assert!(
                error.to_string().contains("still referenced"),
                "{sql} failed with {error}"
            );
        }
        let mut tx = db.begin().unwrap();
        assert!(tx.execute("UPDATE p SET u = u + 1", ()).is_err());
        tx.execute("UPDATE p SET u = 20 WHERE id = 2", ()).unwrap();
        tx.commit().unwrap();
        assert_eq!(
            pairs(db, "SELECT id, u FROM p ORDER BY id"),
            [[1, 10], [2, 20]]
        );
    });
}

#[test]
fn a_key_into_its_own_table_writing_its_value_back_is_not_refused() {
    each_layout(
        "self",
        &[
            "CREATE TABLE t (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES t(id))",
            "INSERT INTO t VALUES (1, NULL), (2, 1)",
        ],
        |db| {
            db.execute("UPDATE t SET id = id", ()).unwrap();
            db.execute("UPDATE t SET id = 1 WHERE id = 1", ()).unwrap();
            assert!(db.execute("UPDATE t SET id = 5 WHERE id = 1", ()).is_err());
            assert_eq!(
                pairs(db, "SELECT id, COALESCE(pid, 0) FROM t ORDER BY id"),
                [[1, 0], [2, 1]]
            );
        },
    );
}

#[test]
fn a_restricting_key_asks_the_rows_as_the_update_leaves_them() {
    each_layout(
        "self_both",
        &[
            "CREATE TABLE t (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES t(id))",
            "INSERT INTO t VALUES (1, NULL), (2, 1)",
        ],
        |db| {
            db.execute("UPDATE t SET id = id + 10, pid = pid + 10", ())
                .unwrap();
            assert_eq!(
                pairs(db, "SELECT id, COALESCE(pid, 0) FROM t ORDER BY id"),
                [[11, 0], [12, 11]]
            );
            assert!(db.execute("UPDATE t SET id = id + 10", ()).is_err());
        },
    );
}

#[test]
fn a_cascade_writing_its_value_back_does_not_ask_the_keys_beneath_it() {
    each_layout(
        "cascade",
        &[
            "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, pu INTEGER UNIQUE REFERENCES p(u) ON UPDATE CASCADE)",
            "CREATE TABLE g (id INTEGER PRIMARY KEY, cu INTEGER REFERENCES c(pu))",
            "INSERT INTO p VALUES (1, 10)",
            "INSERT INTO c VALUES (1, 10)",
            "INSERT INTO g VALUES (1, 10)",
        ],
        |db| {
            db.execute("UPDATE p SET u = u", ()).unwrap();
            db.execute("UPDATE p SET u = 10 WHERE id = 1", ()).unwrap();
            assert!(db.execute("UPDATE p SET u = 11", ()).is_err());
            assert_eq!(pairs(db, "SELECT id, pu FROM c"), [[1, 10]]);
        },
    );
}

#[test]
fn a_referenced_value_written_back_is_not_held() {
    let db = Database::open("memory://fk_same_value_hold").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, code TEXT UNIQUE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, code TEXT REFERENCES p(code))",
        "INSERT INTO p VALUES (1, 'a')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let other = db.clone();
    let mut inserting = other.begin().unwrap();
    inserting
        .execute("INSERT INTO c VALUES (1, 'a')", ())
        .unwrap();
    db.execute("UPDATE p SET code = 'a' WHERE id = 1", ())
        .unwrap();
    inserting.commit().unwrap();

    let mut updating = other.begin().unwrap();
    updating.execute("UPDATE p SET code = code", ()).unwrap();
    db.execute("INSERT INTO c VALUES (2, 'a')", ()).unwrap();
    updating.commit().unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM c WHERE code = 'a'", ())
            .unwrap(),
        2
    );
}
