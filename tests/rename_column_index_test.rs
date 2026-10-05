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

//! An index follows its column through RENAME COLUMN: the column answers by
//! its new name, a new column under the old name is not the index's, and
//! writes and unique checks keep to the column, across seal, reopen and log
//! replay. Every query is compared with `c`, the same table without indexes.

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

fn assert_answers_as_unindexed(db: &Database, filters: &[&str], stage: &str) {
    for filter in filters {
        assert_eq!(
            ids(db, &format!("SELECT id FROM t WHERE {filter}")),
            ids(db, &format!("SELECT id FROM c WHERE {filter}")),
            "{stage}: WHERE {filter}"
        );
    }
}

/// `(index, column)` pairs of `t`, as SHOW INDEXES lists them
fn indexes(db: &Database) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = db
        .query("SHOW INDEXES FROM t", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(1).unwrap(), r.get(2).unwrap())
        })
        .collect();
    out.sort();
    out
}

fn both(db: &Database, statements: &[&str]) {
    for table in ["t", "c"] {
        for sql in statements {
            db.execute(&sql.replace("{t}", table), ()).unwrap();
        }
    }
}

// --- RENAME COLUMN ---------------------------------------------------------

fn rename_setup(db: &Database) {
    both(
        db,
        &[
            "CREATE TABLE {t} (id INTEGER PRIMARY KEY, a INTEGER, u INTEGER)",
            "INSERT INTO {t} VALUES (1, 10, 100), (2, 20, 200), (3, 10, 300)",
        ],
    );
    db.execute("CREATE INDEX ia ON t(a)", ()).unwrap();
    db.execute("CREATE UNIQUE INDEX iu ON t(u)", ()).unwrap();
}

/// Renames the indexed columns, gives their old names to new columns, then
/// writes to both
fn rename_and_write(db: &Database) {
    both(
        db,
        &[
            "ALTER TABLE {t} RENAME COLUMN a TO b",
            "ALTER TABLE {t} RENAME COLUMN u TO w",
            "ALTER TABLE {t} ADD COLUMN a INTEGER",
            "ALTER TABLE {t} ADD COLUMN u INTEGER",
            "UPDATE {t} SET a = 99 WHERE id = 1",
            "INSERT INTO {t} VALUES (4, 40, 400, 10, 500)",
            "UPDATE {t} SET b = 50 WHERE id = 2",
            "DELETE FROM {t} WHERE id = 3",
        ],
    );
}

const RENAME_FILTERS: [&str; 12] = [
    "b = 10",
    "b = 20",
    "b = 50",
    "b BETWEEN 10 AND 45",
    "a = 10",
    "a = 99",
    "a + 0 = 10",
    "a IS NULL",
    "w = 200",
    "w = 400",
    "u = 500",
    "u IS NULL",
];

fn assert_renamed_unique_follows_its_column(db: &Database, stage: &str) {
    assert!(
        db.execute("INSERT INTO t VALUES (5, 1, 400, 1, 1)", ())
            .is_err(),
        "{stage}: the renamed column w keeps its unique index"
    );
    db.execute("INSERT INTO t VALUES (6, 1, 600, 1, 500)", ())
        .unwrap_or_else(|e| panic!("{stage}: the new column u has no unique index: {e}"));
    db.execute("DELETE FROM t WHERE id = 6", ()).unwrap();
}

#[test]
fn a_renamed_index_answers_for_its_column() {
    let db = Database::open("memory://alter_rename_index").unwrap();
    rename_setup(&db);
    rename_and_write(&db);
    assert_answers_as_unindexed(&db, &RENAME_FILTERS, "after rename");
    assert_eq!(
        indexes(&db),
        [
            ("ia".to_string(), "b".to_string()),
            ("iu".to_string(), "w".to_string()),
        ]
    );
    assert_renamed_unique_follows_its_column(&db, "after rename");
}

#[test]
fn a_renamed_index_answers_across_seal_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        rename_setup(&db);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        rename_and_write(&db);
        assert_answers_as_unindexed(&db, &RENAME_FILTERS, "rename over sealed rows");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_answers_as_unindexed(&db, &RENAME_FILTERS, "after a second seal");
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_answers_as_unindexed(&db, &RENAME_FILTERS, "after reopen");
    assert_renamed_unique_follows_its_column(&db, "after reopen");
}

#[test]
fn a_renamed_index_answers_after_log_replay() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        rename_setup(&db);
        rename_and_write(&db);
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_answers_as_unindexed(&db, &RENAME_FILTERS, "after replay");
    assert_renamed_unique_follows_its_column(&db, "after replay");
}

#[test]
fn a_renamed_primary_key_answers_by_its_new_name() {
    let db = Database::open("memory://alter_rename_pk").unwrap();
    rename_setup(&db);
    both(
        &db,
        &[
            "ALTER TABLE {t} RENAME COLUMN id TO pk",
            "INSERT INTO {t} VALUES (7, 70, 700)",
        ],
    );
    for filter in ["pk = 7", "pk = 2", "pk BETWEEN 2 AND 7"] {
        assert_eq!(
            ids(&db, &format!("SELECT pk FROM t WHERE {filter}")),
            ids(&db, &format!("SELECT pk FROM c WHERE {filter}")),
            "WHERE {filter}"
        );
    }
    assert!(db.execute("INSERT INTO t VALUES (7, 1, 1)", ()).is_err());
}

// --- With failpoints -------------------------------------------------------

// The failpoints and their counters are process-wide
#[cfg(feature = "test-failpoints")]
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "test-failpoints")]
#[test]
fn a_renamed_vector_index_keeps_its_graph_over_sealed_rows() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
        .unwrap();
    db.execute("CREATE INDEX iv ON t(v) USING HNSW", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, '[1,0]'), (2, '[2,0]'), (3, '[9,0]'), (4, '[3,0]')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let nearest = |column: &str| -> Vec<i64> {
        db.query(
            &format!("SELECT id FROM t ORDER BY VEC_DISTANCE_L2({column}, '[1,0]') LIMIT 3"),
            (),
        )
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
    };
    assert_eq!(nearest("v"), [1, 2, 4]);
    let filled = stoolap::test_failpoints::indexes_filled_from_cold();
    db.execute("ALTER TABLE t RENAME COLUMN v TO w", ())
        .unwrap();
    assert_eq!(nearest("w"), [1, 2, 4]);
    assert_eq!(
        stoolap::test_failpoints::indexes_filled_from_cold(),
        filled,
        "the rename read no sealed row into the graph"
    );
    assert_eq!(indexes(&db), [("iv".to_string(), "w".to_string())]);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_rename_the_log_refused_keeps_the_index_bindings() {
    use std::sync::atomic::Ordering;
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    rename_setup(&db);
    let identities =
        |db: &Database| ["ia", "iu"].map(|index| db.engine().index_identity("t", index));
    let before = identities(&db);
    stoolap::test_failpoints::WAL_WRITE_FAIL.store(true, Ordering::Release);
    let renamed = db.execute("ALTER TABLE t RENAME COLUMN a TO b", ());
    stoolap::test_failpoints::WAL_WRITE_FAIL.store(false, Ordering::Release);
    assert!(renamed.is_err());
    let unchanged = |db: &Database, stage: &str| {
        assert_eq!(
            indexes(db),
            [
                ("ia".to_string(), "a".to_string()),
                ("iu".to_string(), "u".to_string()),
            ],
            "{stage}"
        );
        assert_eq!(identities(db), before, "{stage}");
        assert_eq!(ids(db, "SELECT id FROM t WHERE a = 10"), [1, 3], "{stage}");
        assert_eq!(ids(db, "SELECT id FROM t WHERE u = 200"), [2], "{stage}");
    };
    unchanged(&db, "after the refused rename");
    let _ = db.close();
    let db = Database::open(&dsn).unwrap();
    unchanged(&db, "after reopen");
    db.execute("INSERT INTO t VALUES (9, 10, 900)", ()).unwrap();
    assert_eq!(ids(&db, "SELECT id FROM t WHERE a = 10"), [1, 3, 9]);
    assert!(db.execute("INSERT INTO t VALUES (10, 1, 900)", ()).is_err());
}
