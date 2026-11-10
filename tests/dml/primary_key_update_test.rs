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

fn rows(db: &Database, sql: &str) -> Vec<Vec<Option<i64>>> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get::<Option<i64>>(i).unwrap())
                .collect()
        })
        .collect()
}

/// Each layout the rows are read from: the hot store, then volumes
fn each_layout(name: &str, check: impl Fn(&Database)) {
    let tables = |dsn: &str, sealed: bool| {
        let db = Database::open(dsn).unwrap();
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, u INTEGER UNIQUE, v INTEGER)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, tid INTEGER REFERENCES t(id) ON UPDATE CASCADE)",
            "CREATE TABLE n (id INTEGER PRIMARY KEY, tid INTEGER REFERENCES t(id) ON UPDATE SET NULL)",
            "CREATE TABLE r (id INTEGER PRIMARY KEY, tid INTEGER REFERENCES t(id))",
            "INSERT INTO t VALUES (1, 10, 100), (2, 20, 200), (3, 30, 300), (4, 40, 400)",
            "INSERT INTO c VALUES (1, 1), (2, 2)",
            "INSERT INTO n VALUES (1, 1)",
            "INSERT INTO r VALUES (1, 4)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db
    };
    check(&tables(&format!("memory://pk_update_{name}"), false));
    let dir = tempfile::tempdir().unwrap();
    check(&tables(&format!("file://{}", dir.path().display()), true));
}

fn table(db: &Database) -> Vec<Vec<Option<i64>>> {
    rows(db, "SELECT id, u, v FROM t ORDER BY id")
}

fn t(rows: &[[i64; 3]]) -> Vec<Vec<Option<i64>>> {
    rows.iter()
        .map(|row| row.iter().map(|&value| Some(value)).collect())
        .collect()
}

const START: [[i64; 3]; 4] = [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]];

#[test]
fn a_key_update_moves_each_row_to_its_new_key() {
    each_layout("moves", |db| {
        db.execute("UPDATE t SET id = id + 10 WHERE id <= 3", ())
            .unwrap();
        let moved = t(&[[4, 40, 400], [11, 10, 100], [12, 20, 200], [13, 30, 300]]);
        assert_eq!(table(db), moved);
        assert_eq!(rows(db, "SELECT v FROM t WHERE id = 12"), [[Some(200)]]);
        assert!(rows(db, "SELECT v FROM t WHERE id = 2").is_empty());
        assert_eq!(rows(db, "SELECT id FROM t WHERE u = 30"), [[Some(13)]]);
        assert_eq!(
            rows(
                db,
                "SELECT id FROM t WHERE id BETWEEN 11 AND 12 ORDER BY id"
            ),
            [[Some(11)], [Some(12)]]
        );
        assert_eq!(
            rows(
                db,
                "SELECT c.id, t.v FROM c JOIN t ON t.id = c.tid ORDER BY c.id"
            ),
            [[Some(1), Some(100)], [Some(2), Some(200)]]
        );
        assert!(db.execute("INSERT INTO t VALUES (13, 99, 0)", ()).is_err());
        db.execute("INSERT INTO t VALUES (2, 21, 0)", ()).unwrap();
    });
}

#[test]
fn keys_collide_only_with_the_keys_the_statement_leaves() {
    each_layout("statement_end", |db| {
        db.execute("UPDATE t SET id = id + 1 WHERE id <= 3", ())
            .unwrap_err();
        assert_eq!(table(db), t(&START), "4 stays, so 3 cannot become 4");
        db.execute("DELETE FROM r", ()).unwrap();
        db.execute("UPDATE t SET id = id + 1", ()).unwrap();
        assert_eq!(
            table(db),
            t(&[[2, 10, 100], [3, 20, 200], [4, 30, 300], [5, 40, 400]])
        );
        db.execute(
            "UPDATE t SET id = CASE id WHEN 2 THEN 3 WHEN 3 THEN 2 END WHERE id IN (2, 3)",
            (),
        )
        .unwrap();
        assert_eq!(
            rows(db, "SELECT id, v FROM t WHERE id <= 3 ORDER BY id"),
            [[Some(2), Some(200)], [Some(3), Some(100)]]
        );
        db.execute("UPDATE t SET id = 9", ()).unwrap_err();
        assert_eq!(rows(db, "SELECT COUNT(*) FROM t WHERE id = 9"), [[Some(0)]]);
    });
}

#[test]
fn a_key_update_answers_returning_parameters_and_prepared_runs() {
    each_layout("prepared", |db| {
        let returned = rows(db, "UPDATE t SET id = 21 WHERE id = 1 RETURNING id, v");
        assert_eq!(returned, [[Some(21), Some(100)]]);
        db.execute("UPDATE t SET id = $1 WHERE id = $2", (22, 2))
            .unwrap();
        let stmt = db.prepare("UPDATE t SET id = $1 WHERE id = $2").unwrap();
        stmt.execute((23, 3)).unwrap();
        stmt.execute((24, 23)).unwrap();
        assert_eq!(
            table(db),
            t(&[[4, 40, 400], [21, 10, 100], [22, 20, 200], [24, 30, 300]])
        );
        assert_eq!(
            rows(db, "SELECT tid FROM c ORDER BY id"),
            [[Some(21)], [Some(22)]]
        );
        let returned = rows(
            db,
            "UPDATE t SET id = _rowid_ + 100 WHERE id IN (SELECT tid FROM c) RETURNING _rowid_, id",
        );
        assert_eq!(returned, [[Some(121), Some(121)], [Some(122), Some(122)]]);
        assert_eq!(
            rows(db, "SELECT _rowid_, v FROM t WHERE id >= 100 ORDER BY id"),
            [[Some(121), Some(100)], [Some(122), Some(200)]]
        );
        assert_eq!(
            rows(db, "SELECT tid FROM c ORDER BY id"),
            [[Some(121)], [Some(122)]]
        );
    });
}

#[test]
fn a_key_that_is_null_or_not_an_integer_is_refused() {
    each_layout("refused_value", |db| {
        db.execute("UPDATE t SET id = NULL WHERE id = 1", ())
            .unwrap_err();
        db.execute("UPDATE t SET id = 'x' WHERE id = 1", ())
            .unwrap_err();
        assert_eq!(table(db), t(&START));
    });
}

#[test]
fn a_key_update_runs_the_update_actions_of_the_rows_naming_it() {
    each_layout("actions", |db| {
        let error = db
            .execute("UPDATE t SET id = 14 WHERE id = 4", ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("foreign key"), "{error}");
        db.execute("UPDATE t SET id = id + 10 WHERE id <= 2", ())
            .unwrap();
        assert_eq!(
            rows(db, "SELECT tid FROM c ORDER BY id"),
            [[Some(11)], [Some(12)]]
        );
        assert_eq!(rows(db, "SELECT tid FROM n"), [[None]]);
        assert_eq!(rows(db, "SELECT tid FROM r"), [[Some(4)]]);
    });
}

#[test]
fn a_refused_key_update_leaves_an_open_transaction_as_it_was() {
    each_layout("txn", |db| {
        db.execute("BEGIN", ()).unwrap();
        db.execute("UPDATE t SET id = 11 WHERE id = 1", ()).unwrap();
        db.execute("UPDATE t SET id = 2 WHERE id = 3", ())
            .unwrap_err();
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(
            table(db),
            t(&[[2, 20, 200], [3, 30, 300], [4, 40, 400], [11, 10, 100]])
        );
        db.execute("BEGIN", ()).unwrap();
        db.execute("UPDATE t SET id = 12 WHERE id = 2", ()).unwrap();
        db.execute("ROLLBACK", ()).unwrap();
        assert_eq!(rows(db, "SELECT v FROM t WHERE id = 2"), [[Some(200)]]);
        assert_eq!(
            rows(db, "SELECT tid FROM c ORDER BY id"),
            [[Some(11)], [Some(2)]]
        );
    });
}

#[test]
fn an_upsert_moves_the_row_whose_key_it_sets() {
    each_layout("upsert", |db| {
        db.execute(
            "INSERT INTO t VALUES (1, 0, 0) ON CONFLICT (id) DO UPDATE SET id = 11",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO t VALUES (12, 20, 0) ON CONFLICT (u) DO UPDATE SET id = excluded.id",
            (),
        )
        .unwrap();
        let returned = rows(
            db,
            "INSERT INTO t VALUES (3, 0, 0) ON CONFLICT (id) DO UPDATE SET id = 13 RETURNING id, v",
        );
        assert_eq!(returned, [[Some(13), Some(300)]]);
        db.execute(
            "INSERT INTO t VALUES (13, 0, 0) ON CONFLICT (id) DO UPDATE SET id = 11",
            (),
        )
        .unwrap_err();
        assert_eq!(
            table(db),
            t(&[[4, 40, 400], [11, 10, 100], [12, 20, 200], [13, 30, 300]])
        );
        assert_eq!(rows(db, "SELECT v FROM t WHERE id = 12"), [[Some(200)]]);
        assert_eq!(
            rows(db, "SELECT tid FROM c ORDER BY id"),
            [[Some(11)], [Some(12)]]
        );
        assert_eq!(rows(db, "SELECT tid FROM n"), [[None]]);
    });
}
