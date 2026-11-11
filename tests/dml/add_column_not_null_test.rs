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

//! ALTER TABLE ADD COLUMN of a NOT NULL column with no default

use stoolap::Database;

fn columns(db: &Database) -> Vec<String> {
    db.query("SELECT * FROM t", ()).unwrap().columns().to_vec()
}

fn assert_refused(db: &Database, sql: &str) {
    let error = db.execute(sql, ()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("not null constraint failed for column z"),
        "{sql} failed with {error}"
    );
    assert_eq!(columns(db), ["id", "v"], "{sql} left the column");
}

/// `t (id, v)` holding `rows`, on the hot store and then sealed
fn each_layout(name: &str, rows: &str, check: impl Fn(&Database)) {
    let open = |dsn: &str, sealed: bool| {
        let db = Database::open(dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        if !rows.is_empty() {
            db.execute(&format!("INSERT INTO t VALUES {rows}"), ())
                .unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db
    };
    check(&open(&format!("memory://add_not_null_{name}"), false));
    let dir = tempfile::tempdir().unwrap();
    check(&open(&format!("file://{}", dir.path().display()), true));
}

#[test]
fn a_not_null_column_with_no_default_is_refused_over_rows() {
    each_layout("rows", "(1, 10), (2, 20)", |db| {
        assert_refused(db, "ALTER TABLE t ADD COLUMN z INTEGER NOT NULL");
        assert_refused(
            db,
            "ALTER TABLE t ADD COLUMN z INTEGER NOT NULL DEFAULT NULL",
        );
        assert_refused(db, "ALTER TABLE t ADD COLUMN z TEXT NOT NULL");
        db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
                .unwrap(),
            3
        );
        db.execute("ALTER TABLE t ADD COLUMN z INTEGER NOT NULL DEFAULT 7", ())
            .unwrap();
        db.execute("ALTER TABLE t ADD COLUMN n INTEGER", ())
            .unwrap();
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM t WHERE z = 7 AND n IS NULL", ())
                .unwrap(),
            3
        );
    });
}

#[test]
fn a_not_null_column_with_no_default_is_added_to_a_table_without_rows() {
    each_layout("empty", "", |db| {
        db.execute("ALTER TABLE t ADD COLUMN z INTEGER NOT NULL", ())
            .unwrap();
        assert!(db
            .execute("INSERT INTO t (id, v) VALUES (1, 10)", ())
            .is_err());
        db.execute("INSERT INTO t VALUES (1, 10, 5)", ()).unwrap();
        assert_eq!(db.query_one::<i64, _>("SELECT z FROM t", ()).unwrap(), 5);
    });
    each_layout("emptied", "(1, 10), (2, 20)", |db| {
        db.execute("DELETE FROM t", ()).unwrap();
        db.execute("ALTER TABLE t ADD COLUMN z INTEGER NOT NULL", ())
            .unwrap();
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
                .unwrap(),
            0
        );
    });
}

#[test]
fn a_refused_column_is_gone_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
        assert_refused(&db, "ALTER TABLE t ADD COLUMN z INTEGER NOT NULL");
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(columns(&db), ["id", "v"]);
    db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
}

#[test]
fn an_uncommitted_row_does_not_keep_the_column_out_and_cannot_commit_after_it() {
    let db = Database::open("memory://add_not_null_uncommitted").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let other = db.clone();
    let mut inserting = other.begin().unwrap();
    inserting
        .execute("INSERT INTO t VALUES (1, 10)", ())
        .unwrap();
    db.execute("ALTER TABLE t ADD COLUMN z INTEGER NOT NULL", ())
        .unwrap();
    assert!(inserting.commit().is_err());
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM t WHERE z IS NULL", ())
            .unwrap(),
        0
    );
}
