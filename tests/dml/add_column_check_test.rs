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

//! ALTER TABLE ADD COLUMN takes a CHECK, once every row the table holds
//! passes it with the column holding the value the row takes for it

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

/// `t (id, v, w)` holding (1, 5, 1) and (2, 50, 2), on the hot store or
/// sealed
fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER)",
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

fn outcome(db: &Database, sql: &str) -> String {
    match db.execute(sql, ()) {
        Ok(rows) => format!("accepted, {rows} rows"),
        Err(error) => error.to_string(),
    }
}

const FAILED: &str = "CHECK constraint failed for column z";

#[test]
fn add_column_takes_a_check_the_rows_pass_and_it_judges_every_write() {
    each_layout(|db, layout| {
        let update = db.prepare("UPDATE t SET v = $1 WHERE id = $2").unwrap();
        for sql in [
            "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 40 CHECK (z < 100 AND z > v - 20)",
            "ALTER TABLE t ADD COLUMN n INTEGER CHECK (n > 0)",
        ] {
            db.execute(sql, ())
                .unwrap_or_else(|e| panic!("{layout}: {sql}: {e}"));
        }
        assert_eq!(
            text(db, "SHOW CREATE TABLE t"),
            ["t|CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER, z INTEGER DEFAULT 40 CHECK ((z < 100) AND (z > (v - 20))), n INTEGER CHECK (n > 0))"],
            "{layout}"
        );
        let got = [
            outcome(db, "INSERT INTO t (id, v) VALUES (3, 1)"),
            outcome(db, "INSERT INTO t (id, v, z) VALUES (4, 1, 100)"),
            outcome(db, "INSERT INTO t (id, v) VALUES (5, 60)"),
            outcome(db, "INSERT INTO t (id, v, n) VALUES (6, 1, 0)"),
            outcome(db, "UPDATE t SET z = 500 WHERE id = 1"),
            update
                .execute((70, 2))
                .map_or_else(|e| e.to_string(), |rows| format!("accepted, {rows} rows")),
        ]
        .map(|got| match got {
            got if got.contains(FAILED) => "z".to_string(),
            got if got.contains("CHECK constraint failed for column n") => "n".to_string(),
            got => got,
        });
        assert_eq!(
            got,
            ["accepted, 1 rows", "z", "z", "n", "z", "z"],
            "{layout}"
        );
        assert_eq!(
            text(db, "SELECT id, v, z, n FROM t ORDER BY id"),
            ["1|5|40|NULL", "2|50|40|NULL", "3|1|40|NULL"],
            "{layout}"
        );
    });
}

#[test]
fn add_column_refuses_a_check_a_row_fails_and_leaves_no_column() {
    each_layout(|db, layout| {
        let before = text(db, "SHOW CREATE TABLE t");
        for sql in [
            "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 400 CHECK (z < 100)",
            "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 10 CHECK (z > v)",
            "ALTER TABLE t ADD COLUMN z INTEGER NOT NULL DEFAULT 0 CHECK (z + w <> 2)",
            "ALTER TABLE t ADD COLUMN z INTEGER CHECK (z IS NOT NULL)",
        ] {
            let got = outcome(db, sql);
            assert!(got.contains(FAILED), "{layout}: {sql}: {got}");
            assert_eq!(text(db, "SHOW CREATE TABLE t"), before, "{layout}: {sql}");
        }
        db.execute("INSERT INTO t VALUES (3, 7, 7)", ()).unwrap();
        assert_eq!(
            text(db, "SELECT * FROM t ORDER BY id"),
            ["1|5|1", "2|50|2", "3|7|7"],
            "{layout}"
        );
    });
}

#[test]
fn add_column_judges_each_row_by_its_newest_copy() {
    each_layout(|db, layout| {
        db.execute("UPDATE t SET v = 6 WHERE id = 2", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3, 70, 3)", ()).unwrap();
        db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
        let sql = "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 10 CHECK (z > v)";
        db.execute(sql, ())
            .unwrap_or_else(|e| panic!("{layout}: a row's older copy was judged: {e}"));
        assert_eq!(
            text(db, "SELECT id, v, z FROM t ORDER BY id"),
            ["1|5|10", "2|6|10"],
            "{layout}"
        );
    });
}

#[test]
fn add_column_judges_no_row_of_an_empty_table() {
    let db = Database::open("memory://add_column_check_empty").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 400 CHECK (z < 100)",
        (),
    )
    .unwrap();
    let got = outcome(&db, "INSERT INTO t (id) VALUES (1)");
    assert!(got.contains(FAILED), "{got}");
}

#[test]
fn a_column_added_again_is_judged_by_its_default_not_by_the_dropped_values() {
    each_layout(|db, layout| {
        db.execute("ALTER TABLE t DROP COLUMN w", ()).unwrap();
        let got = outcome(
            db,
            "ALTER TABLE t ADD COLUMN w INTEGER DEFAULT 9 CHECK (w = 2)",
        );
        assert!(
            got.contains("CHECK constraint failed for column w"),
            "{layout}: {got}"
        );
        db.execute(
            "ALTER TABLE t ADD COLUMN w INTEGER DEFAULT 9 CHECK (w = 9)",
            (),
        )
        .unwrap_or_else(|e| panic!("{layout}: the dropped values were judged: {e}"));
        assert_eq!(
            text(db, "SELECT id, w FROM t ORDER BY id"),
            ["1|9", "2|9"],
            "{layout}"
        );
    });
}

#[test]
fn add_column_refuses_a_check_that_is_not_one_rows_value() {
    each_layout(|db, layout| {
        let before = text(db, "SHOW CREATE TABLE t");
        for (sql, names) in [
            (
                "ALTER TABLE t ADD COLUMN z INTEGER CHECK ((SELECT 1) = 1)",
                "the CHECK of z may not hold a subquery",
            ),
            (
                "ALTER TABLE t ADD COLUMN z INTEGER CHECK (SUM(z) > 1)",
                "the CHECK of z may not hold a subquery",
            ),
            (
                "ALTER TABLE t ADD COLUMN z INTEGER CHECK (nope > 1)",
                "the CHECK of z",
            ),
        ] {
            let got = outcome(db, sql);
            assert!(got.contains(names), "{layout}: {sql}: {got}");
            assert_eq!(text(db, "SHOW CREATE TABLE t"), before, "{layout}: {sql}");
        }
    });
}

#[test]
fn a_row_an_open_transaction_holds_does_not_commit_past_the_added_check() {
    let db = Database::open("memory://add_column_check_open_tx").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 5)", ()).unwrap();
    let other = db.clone();
    other.execute("BEGIN", ()).unwrap();
    other.execute("INSERT INTO t VALUES (2, 500)", ()).unwrap();
    db.execute(
        "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 10 CHECK (z > v)",
        (),
    )
    .unwrap();
    let commit = outcome(&other, "COMMIT");
    assert!(commit.contains("schema changed"), "{commit}");
    assert_eq!(text(&db, "SELECT id, v, z FROM t ORDER BY id"), ["1|5|10"]);
}

#[test]
fn an_added_check_survives_a_reopen() {
    let want = "t|CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER, z INTEGER DEFAULT 40 CHECK (z > w), n TEXT CHECK (LENGTH(n) < 3))";
    for (sealed, checkpoint_after) in [(false, false), (true, false), (false, true), (true, true)] {
        let layout = format!("sealed {sealed}, checkpoint after the change {checkpoint_after}");
        let dir = tempfile::tempdir().unwrap();
        {
            let db = open(&dir, sealed);
            for sql in [
                "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 40 CHECK (z > w)",
                "ALTER TABLE t ADD COLUMN n TEXT CHECK (LENGTH(n) < 3)",
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
        db.execute("INSERT INTO t (id, w) VALUES (3, 3)", ())
            .unwrap();
        let got = outcome(&db, "INSERT INTO t (id, w) VALUES (4, 40)");
        assert!(got.contains(FAILED), "{layout}: {got}");
        let got = outcome(&db, "INSERT INTO t (id, w, n) VALUES (5, 1, 'long')");
        assert!(
            got.contains("CHECK constraint failed for column n"),
            "{layout}: {got}"
        );
        assert_eq!(
            text(&db, "SELECT id, w, z, n FROM t ORDER BY id"),
            ["1|1|40|NULL", "2|2|40|NULL", "3|3|40|NULL"],
            "{layout}"
        );
    }
}
