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

//! ALTER TABLE MODIFY COLUMN takes a DEFAULT and a CHECK: the column keeps
//! the ones it has unless the statement writes one, and a CHECK is taken
//! only when every row the table holds passes it

use stoolap::Database;

fn text(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

/// No checkpoint but the ones a test asks for: a close leaves the log to
/// be replayed
fn dsn(dir: &tempfile::TempDir) -> String {
    format!(
        "file://{}?checkpoint_interval=0&checkpoint_on_close=off",
        dir.path().display()
    )
}

/// `t (id, v DEFAULT 3 CHECK (v < 1000), w)` holding (1, 5, 1) and
/// (2, 50, 2), on the hot store or sealed
fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER DEFAULT 3 CHECK (v < 1000), w INTEGER)",
        "INSERT INTO t VALUES (1, 5, 1), (2, 50, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn each_layout(check: impl Fn(&Database, &str)) {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        check(&db, if sealed { "sealed" } else { "hot" });
    }
}

fn error(db: &Database, sql: &str) -> String {
    match db.execute(sql, ()) {
        Ok(rows) => format!("accepted, {rows} rows"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn modify_column_takes_a_default_and_keeps_the_rows() {
    each_layout(|db, layout| {
        db.execute("ALTER TABLE t MODIFY COLUMN v INTEGER DEFAULT 7", ())
            .unwrap();
        db.execute("ALTER TABLE t MODIFY COLUMN w INTEGER DEFAULT 2 + 2", ())
            .unwrap();
        db.execute("INSERT INTO t (id) VALUES (3)", ()).unwrap();
        assert_eq!(
            text(db, "SELECT id, v, w FROM t ORDER BY id"),
            ["1|5|1", "2|50|2", "3|7|4"],
            "{layout}"
        );
        assert_eq!(
            text(db, "SHOW CREATE TABLE t"),
            ["t|CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER DEFAULT 7 CHECK (v < 1000), w INTEGER DEFAULT (2 + 2))"],
            "{layout}"
        );
        // DEFAULT NULL is the way back to no value
        db.execute("ALTER TABLE t MODIFY COLUMN v INTEGER DEFAULT NULL", ())
            .unwrap();
        db.execute("INSERT INTO t (id) VALUES (4)", ()).unwrap();
        assert_eq!(
            text(db, "SELECT v FROM t WHERE id = 4"),
            ["NULL"],
            "{layout}"
        );
    });
}

#[test]
fn modify_column_without_a_constraint_keeps_the_default_and_the_check() {
    each_layout(|db, layout| {
        db.execute("ALTER TABLE t MODIFY COLUMN v FLOAT", ())
            .unwrap();
        db.execute("INSERT INTO t (id) VALUES (3)", ()).unwrap();
        assert_eq!(
            text(db, "SELECT TYPEOF(v), v FROM t WHERE id = 3"),
            ["FLOAT|3"],
            "{layout}"
        );
        assert!(
            error(db, "INSERT INTO t VALUES (4, 5000, 1)").contains("CHECK constraint failed"),
            "{layout}: the CHECK of v went with its type"
        );
    });
}

#[test]
fn modify_column_refuses_a_check_a_row_fails_and_changes_nothing() {
    each_layout(|db, layout| {
        let before = text(db, "SHOW CREATE TABLE t");
        for sql in [
            "ALTER TABLE t MODIFY COLUMN v INTEGER CHECK (v < 10)",
            "ALTER TABLE t MODIFY COLUMN v INTEGER DEFAULT 1 CHECK (v < 10)",
            "ALTER TABLE t MODIFY COLUMN w INTEGER CHECK (w * 30 < v)",
            "ALTER TABLE t MODIFY COLUMN v TEXT CHECK (LENGTH(v) < 2)",
        ] {
            let got = error(db, sql);
            assert!(
                got.contains("CHECK constraint failed"),
                "{layout}: {sql}: {got}"
            );
            assert_eq!(text(db, "SHOW CREATE TABLE t"), before, "{layout}: {sql}");
        }
        assert_eq!(
            text(db, "SELECT id, v, w FROM t ORDER BY id"),
            ["1|5|1", "2|50|2"],
            "{layout}"
        );
        // The CHECK the column had still judges
        assert!(
            error(db, "INSERT INTO t VALUES (3, 5000, 1)").contains("CHECK constraint failed"),
            "{layout}"
        );
    });
}

#[test]
fn modify_column_judges_each_row_by_its_newest_copy() {
    each_layout(|db, layout| {
        db.execute("UPDATE t SET v = 6 WHERE id = 2", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3, 70, 3)", ()).unwrap();
        db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
        db.execute("ALTER TABLE t MODIFY COLUMN v INTEGER CHECK (v < 10)", ())
            .unwrap_or_else(|e| panic!("{layout}: a row's older copy was judged: {e}"));
        assert_eq!(
            text(db, "SELECT id, v FROM t ORDER BY id"),
            ["1|5", "2|6"],
            "{layout}"
        );
    });
}

#[test]
fn modify_column_takes_a_check_that_then_judges_every_write() {
    each_layout(|db, layout| {
        let insert = db.prepare("INSERT INTO t VALUES ($1, $2, $3)").unwrap();
        let update = db.prepare("UPDATE t SET w = $1 WHERE id = $2").unwrap();
        insert.execute((3, 60, 3)).unwrap();
        // The second CHECK of v takes the place of the first
        for sql in [
            "ALTER TABLE t MODIFY COLUMN w INTEGER CHECK (w < v)",
            "ALTER TABLE t MODIFY COLUMN v TEXT NOT NULL CHECK (LENGTH(v) < 3)",
        ] {
            db.execute(sql, ())
                .unwrap_or_else(|e| panic!("{layout}: {sql}: {e}"));
        }
        assert_eq!(
            text(db, "SHOW CREATE TABLE t"),
            ["t|CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT NOT NULL DEFAULT 3 CHECK (LENGTH(v) < 3), w INTEGER CHECK (w < v))"],
            "{layout}"
        );
        let got = [
            insert.execute((4, "7", 9)),
            insert.execute((5, "777", 1)),
            insert.execute((6, "77", 1)),
            update.execute((9, 1)),
            update.execute((4, 1)),
            db.execute("UPDATE t SET v = '1000' WHERE id = 1", ()),
        ];
        let got: Vec<String> = got
            .into_iter()
            .map(|result| match result.map_err(|e| e.to_string()) {
                Ok(rows) => rows.to_string(),
                Err(e) if e.contains("CHECK constraint failed for column w") => "w".to_string(),
                Err(e) if e.contains("CHECK constraint failed for column v") => "v".to_string(),
                Err(e) => e,
            })
            .collect();
        assert_eq!(got, ["w", "v", "1", "w", "1", "v"], "{layout}");
        assert_eq!(
            text(db, "SELECT id, v, w FROM t ORDER BY id"),
            ["1|5|4", "2|50|2", "3|60|3", "6|77|1"],
            "{layout}"
        );
    });
}

#[test]
fn modify_column_refuses_a_default_or_check_that_is_not_one_rows_value() {
    each_layout(|db, layout| {
        let before = text(db, "SHOW CREATE TABLE t");
        for (sql, names) in [
            (
                "ALTER TABLE t MODIFY COLUMN v INTEGER CHECK ((SELECT 1) = 1)",
                "the CHECK of v may not hold a subquery",
            ),
            (
                "ALTER TABLE t MODIFY COLUMN v INTEGER CHECK (SUM(v) > 1)",
                "the CHECK of v may not hold a subquery",
            ),
            (
                "ALTER TABLE t MODIFY COLUMN v INTEGER DEFAULT (SELECT 1)",
                "the DEFAULT of v may not hold a subquery",
            ),
            (
                "ALTER TABLE t MODIFY COLUMN v INTEGER CHECK (nope > 1)",
                "the CHECK of v",
            ),
            ("ALTER TABLE t MODIFY COLUMN nope INTEGER DEFAULT 1", "nope"),
        ] {
            let got = error(db, sql);
            assert!(got.contains(names), "{layout}: {sql}: {got}");
            assert_eq!(text(db, "SHOW CREATE TABLE t"), before, "{layout}: {sql}");
        }
    });
}

#[test]
fn a_modified_default_and_check_survive_a_reopen() {
    let want = "t|CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT DEFAULT 'x' CHECK (LENGTH(v) < 3), w INTEGER DEFAULT 9 CHECK (w >= 0))";
    for (sealed, checkpoint_after) in [(false, false), (true, false), (false, true), (true, true)] {
        let layout = format!("sealed {sealed}, checkpoint after the change {checkpoint_after}");
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(&dir, sealed);
            for sql in [
                "ALTER TABLE t MODIFY COLUMN v TEXT DEFAULT 'x' CHECK (LENGTH(v) < 3)",
                "ALTER TABLE t MODIFY COLUMN w INTEGER DEFAULT 9",
                "ALTER TABLE t MODIFY COLUMN w INTEGER CHECK (w >= 0)",
            ] {
                db.execute(sql, ())
                    .unwrap_or_else(|e| panic!("{layout}: {sql}: {e}"));
            }
            if checkpoint_after {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            db.close().unwrap();
        }
        let db = Database::open(&dsn(&dir)).unwrap();
        assert_eq!(text(&db, "SHOW CREATE TABLE t"), [want], "{layout}");
        db.execute("INSERT INTO t (id) VALUES (3)", ()).unwrap();
        assert!(
            error(&db, "INSERT INTO t VALUES (4, 'long', 1)")
                .contains("CHECK constraint failed for column v"),
            "{layout}"
        );
        assert!(
            error(&db, "INSERT INTO t VALUES (5, 'a', -1)")
                .contains("CHECK constraint failed for column w"),
            "{layout}"
        );
        assert_eq!(
            text(&db, "SELECT id, v, w FROM t ORDER BY id"),
            ["1|5|1", "2|50|2", "3|x|9"],
            "{layout}"
        );
    }
}
