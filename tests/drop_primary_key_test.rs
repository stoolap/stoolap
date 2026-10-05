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

//! Dropping the primary key column is refused on every path before the
//! schema or the log changes: the table, its indexes and its rows stay as
//! they were, and so does a reopen that replays the log.

use stoolap::core::Error;
use stoolap::Database;

type Snapshot = (Vec<String>, Vec<(String, String)>, Vec<(i64, i64, String)>);

fn setup(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10, 'a'), (2, 20, 'b'), (3, 10, 'c')",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX ik ON t(k)", ()).unwrap();
    db.execute("CREATE UNIQUE INDEX iu ON t(s)", ()).unwrap();
}

fn snapshot(db: &Database) -> Snapshot {
    let columns = db
        .query("DESCRIBE t", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    let mut indexes: Vec<(String, String)> = db
        .query("SHOW INDEXES FROM t", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(1).unwrap(), r.get(2).unwrap())
        })
        .collect();
    indexes.sort();
    let rows = db
        .query("SELECT id, k, s FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap(), r.get(2).unwrap())
        })
        .collect();
    (columns, indexes, rows)
}

fn assert_refused(result: stoolap::core::Result<i64>, how: &str) {
    assert!(
        matches!(result, Err(Error::CannotDropPrimaryKey)),
        "{how}: {result:?}"
    );
}

fn log_bytes(dir: &std::path::Path) -> u64 {
    std::fs::read_dir(dir.join("wal"))
        .unwrap()
        .map(|entry| entry.unwrap().metadata().unwrap().len())
        .sum()
}

fn assert_key_still_holds(db: &Database, stage: &str) {
    assert!(
        db.execute("INSERT INTO t VALUES (2, 1, 'x')", ()).is_err(),
        "{stage}: the primary key refuses a duplicate"
    );
    let found: Vec<i64> = db
        .query("SELECT k FROM t WHERE id = 2", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(found, [20], "{stage}: a lookup by the key");
}

#[test]
fn sql_drop_of_the_primary_key_column_changes_nothing() {
    let db = Database::open("memory://drop_primary_key_sql").unwrap();
    setup(&db);
    let before = snapshot(&db);
    for column in ["id", "ID"] {
        assert_refused(
            db.execute(&format!("ALTER TABLE t DROP COLUMN {column}"), ()),
            column,
        );
        assert_eq!(snapshot(&db), before, "after DROP COLUMN {column}");
    }
    assert_key_still_holds(&db, "after the refusal");
}

#[test]
fn a_refused_primary_key_drop_leaves_nothing_for_replay() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    );
    let before = {
        let db = Database::open(&dsn).unwrap();
        setup(&db);
        let before = snapshot(&db);
        let logged = log_bytes(dir.path());
        assert_refused(db.execute("ALTER TABLE t DROP COLUMN id", ()), "sql");
        assert_eq!(log_bytes(dir.path()), logged, "the refusal wrote no record");
        db.close().unwrap();
        before
    };
    let db = Database::open(&dsn).unwrap();
    assert_eq!(snapshot(&db), before, "after the log is replayed");
    assert_key_still_holds(&db, "after reopen");
    db.execute("INSERT INTO t VALUES (4, 40, 'd')", ()).unwrap();
}

#[test]
fn engine_drop_of_the_primary_key_column_changes_nothing() {
    let db = Database::open("memory://drop_primary_key_engine").unwrap();
    setup(&db);
    let before = snapshot(&db);
    assert!(matches!(
        db.engine().drop_column("t", "id"),
        Err(Error::CannotDropPrimaryKey)
    ));
    assert_eq!(snapshot(&db), before);
    assert_key_still_holds(&db, "after the engine refusal");
}
