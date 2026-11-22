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

//! ALTER TABLE ALTER COLUMN DROP DEFAULT and DROP CHECK take a column's
//! default and its own CHECK away, and leave its type, its NOT NULL, its
//! rows and the table's other CHECKs as they were

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

const CREATE: &str = "CREATE TABLE t (id INTEGER PRIMARY KEY, \
     name TEXT NOT NULL DEFAULT 'x' CHECK (LENGTH(name) < 3), \
     v INTEGER DEFAULT 3 CHECK (v < 10), CHECK (v <> 7))";

/// The table of `CREATE` holding (1, 'x', 3), on the hot store or sealed
fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(CREATE, ()).unwrap();
    db.execute("INSERT INTO t (id) VALUES (1)", ()).unwrap();
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

fn outcome(db: &Database, sql: &str) -> String {
    match db.execute(sql, ()) {
        Ok(rows) => format!("{rows} rows"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn drop_check_takes_the_columns_check_and_no_other() {
    each_layout(|db, layout| {
        let insert = db.prepare("INSERT INTO t VALUES ($1, $2, $3)").unwrap();
        db.execute("ALTER TABLE t ALTER COLUMN v DROP CHECK", ())
            .unwrap();
        assert_eq!(
            text(db, "SHOW CREATE TABLE t"),
            ["t|CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL DEFAULT 'x' CHECK (LENGTH(name) < 3), v INTEGER DEFAULT 3, CHECK (v <> 7))"],
            "{layout}"
        );
        let got = [
            insert.execute((2, "a", 500)).map_err(|e| e.to_string()),
            insert.execute((3, "a", 7)).map_err(|e| e.to_string()),
            insert.execute((4, "long", 1)).map_err(|e| e.to_string()),
            db.execute("UPDATE t SET v = 9000 WHERE id = 1", ())
                .map_err(|e| e.to_string()),
        ]
        .map(|got| match got {
            Ok(rows) => rows.to_string(),
            Err(e) if e.contains("CHECK constraint failed for table t") => "table".to_string(),
            Err(e) if e.contains("CHECK constraint failed for column name") => "name".to_string(),
            Err(e) => e,
        });
        assert_eq!(got, ["1", "table", "name", "1"], "{layout}");
        assert_eq!(
            text(db, "SELECT * FROM t ORDER BY id"),
            ["1|x|9000", "2|a|500"],
            "{layout}"
        );
    });
}

#[test]
fn drop_default_leaves_null_for_new_rows_and_the_rows_there_untouched() {
    each_layout(|db, layout| {
        db.execute("ALTER TABLE t ALTER v DROP DEFAULT", ())
            .unwrap();
        db.execute("INSERT INTO t (id) VALUES (2)", ()).unwrap();
        assert_eq!(
            text(db, "SELECT * FROM t ORDER BY id"),
            ["1|x|3", "2|x|NULL"],
            "{layout}"
        );
        assert_eq!(
            text(db, "SHOW CREATE TABLE t"),
            ["t|CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL DEFAULT 'x' CHECK (LENGTH(name) < 3), v INTEGER DEFAULT NULL CHECK (v < 10), CHECK (v <> 7))"],
            "{layout}"
        );
        // A NOT NULL column stays NOT NULL and now needs its value written
        db.execute("ALTER TABLE t ALTER COLUMN name DROP DEFAULT", ())
            .unwrap();
        let got = outcome(db, "INSERT INTO t (id) VALUES (3)");
        assert!(
            got.contains("non-nullable column 'name'"),
            "{layout}: {got}"
        );
        db.execute("INSERT INTO t (id, name) VALUES (3, 'b')", ())
            .unwrap();
    });
}

#[test]
fn a_column_added_with_a_default_keeps_it_in_the_rows_that_took_it() {
    each_layout(|db, layout| {
        db.execute(
            "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 4 CHECK (z < 5)",
            (),
        )
        .unwrap();
        db.execute("ALTER TABLE t ALTER COLUMN z DROP DEFAULT", ())
            .unwrap();
        db.execute("ALTER TABLE t ALTER COLUMN z DROP CHECK", ())
            .unwrap();
        db.execute("INSERT INTO t (id) VALUES (2)", ()).unwrap();
        db.execute("INSERT INTO t (id, z) VALUES (3, 50)", ())
            .unwrap();
        assert_eq!(
            text(db, "SELECT id, z FROM t ORDER BY id"),
            ["1|4", "2|NULL", "3|50"],
            "{layout}"
        );
    });
}

#[test]
fn dropping_what_a_column_does_not_have_changes_nothing() {
    each_layout(|db, layout| {
        let before = text(db, "SHOW CREATE TABLE t");
        for sql in [
            "ALTER TABLE t ALTER COLUMN id DROP DEFAULT",
            "ALTER TABLE t ALTER COLUMN id DROP CHECK",
        ] {
            assert_eq!(outcome(db, sql), "0 rows", "{layout}: {sql}");
        }
        for sql in [
            "ALTER TABLE t ALTER COLUMN v DROP CHECK",
            "ALTER TABLE t ALTER COLUMN v DROP DEFAULT",
        ] {
            db.execute(sql, ()).unwrap();
            let once = text(db, "SHOW CREATE TABLE t");
            assert_eq!(outcome(db, sql), "0 rows", "{layout}: {sql} again");
            assert_eq!(text(db, "SHOW CREATE TABLE t"), once, "{layout}: {sql}");
        }
        assert_ne!(text(db, "SHOW CREATE TABLE t"), before, "{layout}");
        let got = outcome(db, "ALTER TABLE t ALTER COLUMN nope DROP CHECK");
        assert!(got.contains("'nope' not found"), "{layout}: {got}");
    });
}

#[test]
fn only_drop_default_and_drop_check_follow_alter_column() {
    let db = Database::open("memory://alter_column_drop_parse").unwrap();
    db.execute(CREATE, ()).unwrap();
    for (sql, names) in [
        (
            "ALTER TABLE t ALTER COLUMN v DROP NOT NULL",
            "expected DEFAULT or CHECK after DROP",
        ),
        (
            "ALTER TABLE t ALTER COLUMN v SET DEFAULT 1",
            "expected DROP DEFAULT or DROP CHECK after ALTER COLUMN v",
        ),
    ] {
        let got = outcome(&db, sql);
        assert!(got.contains(names), "{sql}: {got}");
    }
}

#[test]
fn a_dropped_default_and_check_stay_dropped_after_a_reopen() {
    let want = "t|CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL DEFAULT 'x', v INTEGER DEFAULT NULL, CHECK (v <> 7))";
    for (sealed, checkpoint_after) in [(false, false), (true, false), (false, true), (true, true)] {
        let layout = format!("sealed {sealed}, checkpoint after the change {checkpoint_after}");
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(&dir, sealed);
            for sql in [
                "ALTER TABLE t ALTER COLUMN v DROP CHECK",
                "ALTER TABLE t ALTER COLUMN v DROP DEFAULT",
                "ALTER TABLE t ALTER COLUMN name DROP CHECK",
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
        db.execute(
            "INSERT INTO t (id, name, v) VALUES (2, 'a long name', 500)",
            (),
        )
        .unwrap_or_else(|e| panic!("{layout}: a dropped CHECK came back: {e}"));
        db.execute("INSERT INTO t (id) VALUES (3)", ()).unwrap();
        assert_eq!(
            text(&db, "SELECT * FROM t ORDER BY id"),
            ["1|x|3", "2|a long name|500", "3|x|NULL"],
            "{layout}"
        );
    }
}
