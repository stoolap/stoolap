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

//! Every read path that names a column of a sealed volume resolves it
//! through the volume's column mapping, so a column renamed since the
//! seal, once or in a cycle, is the column the schema means and not the
//! volume column that used to carry the name.

use stoolap::Database;

/// 70,000 sealed rows with x = 100000 + id and y = 70001 - id, reopened,
/// then x renamed to z and y to x: logical x is the old y
fn swapped(dir: &std::path::Path) -> Database {
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE q (id INTEGER PRIMARY KEY, x INTEGER NOT NULL, y INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO q SELECT g.value, 100000 + g.value, 70001 - g.value FROM generate_series(1, 70000) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN x TO z", ())
        .unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN y TO x", ())
        .unwrap();
    db
}

#[test]
fn an_ordered_limit_takes_its_bounds_from_the_renamed_column() {
    let dir = tempfile::tempdir().unwrap();
    let db = swapped(dir.path());
    let row = db
        .query("SELECT id, x FROM q ORDER BY x ASC LIMIT 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(
        (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()),
        (70_000, 1),
        "the order bound came from the old x column"
    );
    let row = db
        .query("SELECT id, x FROM q ORDER BY x DESC LIMIT 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(
        (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()),
        (1, 70_000)
    );
}

#[test]
fn the_min_max_shortcut_reads_the_renamed_column() {
    let dir = tempfile::tempdir().unwrap();
    let db = swapped(dir.path());
    // Local writes and a LIMIT steer the aggregate onto the index shortcut
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO q (id, z, x) VALUES (70001, 500000, 80000)", ())
        .unwrap();
    let min: i64 = tx.query_one("SELECT MIN(x) FROM q LIMIT 1", ()).unwrap();
    let max: i64 = tx.query_one("SELECT MAX(x) FROM q LIMIT 1", ()).unwrap();
    tx.rollback().unwrap();
    assert_eq!(
        (min, max),
        (1, 80_000),
        "the extrema came from the old x column"
    );
}

#[test]
fn an_anti_join_dictionary_filter_reads_the_renamed_column() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO p VALUES (1)", ()).unwrap();
        db.execute(
            "CREATE TABLE r (id INTEGER PRIMARY KEY, x TEXT NOT NULL, y TEXT NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO r SELECT g.value, 'old', 'needle' FROM generate_series(1, 1000) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("ALTER TABLE r RENAME COLUMN x TO z", ())
        .unwrap();
    db.execute("ALTER TABLE r RENAME COLUMN y TO x", ())
        .unwrap();
    let rows = db
        .query(
            "SELECT p.id FROM p WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.id = p.id AND r.x = 'needle')",
            (),
        )
        .unwrap()
        .count();
    assert_eq!(
        rows, 0,
        "the dictionary of the old x column answered for the new one"
    );
    let rows = db
        .query(
            "SELECT p.id FROM p WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.id = p.id AND r.x = 'old')",
            (),
        )
        .unwrap()
        .count();
    assert_eq!(rows, 1);
}

#[test]
fn a_cycle_of_renames_is_followed_when_the_mapping_is_computed_again() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE c (id INTEGER PRIMARY KEY, x INTEGER NOT NULL, y INTEGER NOT NULL, z INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO c SELECT g.value, 1, 2, 3 FROM generate_series(1, 1000) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "ALTER TABLE c RENAME COLUMN x TO t",
        "ALTER TABLE c RENAME COLUMN y TO x",
        "ALTER TABLE c RENAME COLUMN z TO y",
        "ALTER TABLE c RENAME COLUMN t TO z",
        // A new column has the mapping computed again from the whole history
        "ALTER TABLE c ADD COLUMN marker INTEGER DEFAULT 0",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let row = db
        .query("SELECT x, y, z, marker FROM c WHERE id = 7", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let values: Vec<Option<i64>> = (0..4).map(|i| row.get::<Option<i64>>(i).unwrap()).collect();
    assert_eq!(values, vec![Some(2), Some(3), Some(1), Some(0)]);
    let rows = db
        .query("SELECT id FROM c WHERE x = 2 AND y = 3 AND z = 1", ())
        .unwrap()
        .count();
    assert_eq!(rows, 1_000);
}

/// The anti-join hands the inner table its non-correlated filter as
/// written; the cold collector must prepare it for the schema, or it
/// rejects every sealed row and the outer rows all pass
#[test]
fn an_anti_join_filter_reaches_the_sealed_rows_of_the_inner_table() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1), (2000)", ()).unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, x TEXT NOT NULL)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO r SELECT g.value, 'old' FROM generate_series(1, 1000) g",
        (),
    )
    .unwrap();
    // A plain projection without ORDER BY or LIMIT: the shape that goes
    // through the anti-join
    let sql =
        "SELECT p.id FROM p WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.id = p.id AND r.x = 'old')";
    let ids = |db: &Database| -> Vec<i64> {
        let mut ids: Vec<i64> = db
            .query(sql, ())
            .unwrap()
            .map(|r| r.unwrap().get::<i64>(0).unwrap())
            .collect();
        ids.sort_unstable();
        ids
    };
    assert_eq!(ids(&db), vec![2000]);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        ids(&db),
        vec![2000],
        "the sealed inner rows were all rejected"
    );
}

/// Sealed rows, then a history of renames and drops that can only be
/// resolved by the order of the changes; each case reads the sealed rows
/// after the history, and once more after a reopen so the order is read
/// back from the manifest
fn sealed_then(dir: &std::path::Path, create: &str, insert: &str, history: &[&str]) -> String {
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(create, ()).unwrap();
        db.execute(insert, ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    for sql in history {
        db.execute(sql, ()).unwrap();
    }
    dsn
}

fn values(db: &Database, sql: &str) -> Vec<Option<i64>> {
    let row = db.query(sql, ()).unwrap().next().unwrap().unwrap();
    (0..row.len())
        .map(|i| row.get::<Option<i64>>(i).unwrap())
        .collect()
}

#[test]
fn a_temporary_name_used_twice_resolves_each_column_to_its_own_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = sealed_then(
        dir.path(),
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b INTEGER NOT NULL)",
        "INSERT INTO t SELECT g.value, 1, 2 FROM generate_series(1, 100) g",
        &[
            "ALTER TABLE t RENAME COLUMN a TO tmp",
            "ALTER TABLE t RENAME COLUMN tmp TO a",
            "ALTER TABLE t RENAME COLUMN b TO tmp",
            "ALTER TABLE t RENAME COLUMN tmp TO b",
            "ALTER TABLE t ADD COLUMN marker INTEGER DEFAULT 0",
        ],
    );
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(
            values(&db, "SELECT a, b, marker FROM t WHERE id = 5"),
            vec![Some(1), Some(2), Some(0)]
        );
    }
}

#[test]
fn a_name_dropped_and_added_back_then_renamed_carries_the_default_not_the_old_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = sealed_then(
        dir.path(),
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL)",
        "INSERT INTO t SELECT g.value, 1 FROM generate_series(1, 100) g",
        &[
            "ALTER TABLE t RENAME COLUMN a TO b",
            "ALTER TABLE t DROP COLUMN b",
            "ALTER TABLE t ADD COLUMN b INTEGER DEFAULT 4",
            "ALTER TABLE t RENAME COLUMN b TO c",
        ],
    );
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(values(&db, "SELECT c FROM t WHERE id = 5"), vec![Some(4)]);
        let rows = db
            .query("SELECT id FROM t WHERE c = 4", ())
            .unwrap()
            .count();
        assert_eq!(rows, 100);
    }
}

#[test]
fn a_name_freed_by_a_drop_and_taken_by_a_rename_reads_the_renamed_column() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = sealed_then(
        dir.path(),
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, c INTEGER NOT NULL)",
        "INSERT INTO t SELECT g.value, 1, 9 FROM generate_series(1, 100) g",
        &[
            "ALTER TABLE t RENAME COLUMN a TO b",
            "ALTER TABLE t DROP COLUMN b",
            "ALTER TABLE t RENAME COLUMN c TO b",
        ],
    );
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(values(&db, "SELECT b FROM t WHERE id = 5"), vec![Some(9)]);
        let rows = db
            .query("SELECT id FROM t WHERE b = 9", ())
            .unwrap()
            .count();
        assert_eq!(rows, 100);
    }
}

#[test]
fn a_name_dropped_and_added_back_reads_the_default_not_the_dropped_data() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = sealed_then(
        dir.path(),
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL)",
        "INSERT INTO t SELECT g.value, 1 FROM generate_series(1, 100) g",
        &[
            "ALTER TABLE t RENAME COLUMN a TO b",
            "ALTER TABLE t DROP COLUMN b",
            "ALTER TABLE t ADD COLUMN b INTEGER DEFAULT 4",
        ],
    );
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(values(&db, "SELECT b FROM t WHERE id = 5"), vec![Some(4)]);
        let rows = db
            .query("SELECT id FROM t WHERE b = 1", ())
            .unwrap()
            .count();
        assert_eq!(rows, 0);
    }
}

/// The engine's schema epoch starts over at an open while the versions a
/// manifest carries persist; the epoch must restart above them, or a
/// change made after the open orders before the seal
#[test]
fn a_rename_after_a_reopen_orders_after_a_seal_made_at_a_high_version() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE q (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        // Raise the schema version well above what a fresh open starts at
        for i in 0..8 {
            db.execute(&format!("ALTER TABLE q ADD COLUMN pad{i} INTEGER"), ())
                .unwrap();
            db.execute(&format!("ALTER TABLE q DROP COLUMN pad{i}"), ())
                .unwrap();
        }
        db.execute(
            "INSERT INTO q SELECT g.value, 1, 2 FROM generate_series(1, 100) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN a TO z", ())
        .unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN b TO a", ())
        .unwrap();
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(
            values(&db, "SELECT z, a FROM q WHERE id = 5"),
            vec![Some(1), Some(2)]
        );
    }
    assert_eq!(
        values(&db, "SELECT z, a FROM q WHERE id = 5"),
        vec![Some(1), Some(2)]
    );
}

/// A rename made in the same schema version a later seal carries happened
/// before that seal: the sealed rows already carry the new name and the
/// rename must not be walked again
#[test]
fn a_rename_made_just_before_a_seal_is_not_applied_to_the_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE q (id INTEGER PRIMARY KEY, x INTEGER NOT NULL, y INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO q SELECT g.value, 11, 22 FROM generate_series(1, 100) g",
        (),
    )
    .unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN x TO z", ())
        .unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN y TO x", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN z TO y", ())
        .unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN x TO z", ())
        .unwrap();
    assert_eq!(
        values(&db, "SELECT y, z FROM q WHERE id = 5"),
        vec![Some(11), Some(22)]
    );
}

/// Renames replayed from the log after an open without a checkpoint take
/// the same versions the statements took, so a volume sealed between them
/// sees the same order either way
#[test]
fn renames_replayed_from_the_log_keep_their_order_against_the_seal() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE q (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO q SELECT g.value, 1, 2 FROM generate_series(1, 100) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("ALTER TABLE q RENAME COLUMN a TO tmp", ())
            .unwrap();
        db.execute("ALTER TABLE q RENAME COLUMN b TO a", ())
            .unwrap();
        db.execute("ALTER TABLE q RENAME COLUMN tmp TO b", ())
            .unwrap();
        assert_eq!(
            values(&db, "SELECT a, b FROM q WHERE id = 5"),
            vec![Some(2), Some(1)]
        );
    }
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(
            values(&db, "SELECT a, b FROM q WHERE id = 5"),
            vec![Some(2), Some(1)]
        );
    }
}

/// A rename the manifest holds already is replayed from the log when the
/// log was kept past it (a snapshot held the truncation back): the
/// replay must not record it a second time, or the volume sealed after
/// the rename is read as if it predated it
#[test]
fn a_rename_replayed_over_a_manifest_that_holds_it_is_not_recorded_twice() {
    use stoolap::IsolationLevel;
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("ALTER TABLE t RENAME COLUMN a TO b", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (2, 2)", ()).unwrap();
        // A snapshot begun here keeps row 3 in the hot store through the
        // checkpoint, so the log is not truncated while the manifest,
        // holding the rename, is written out with the second volume
        let snapshot = db
            .begin_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        db.execute("INSERT INTO t VALUES (3, 3)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        drop(snapshot);
        assert_eq!(
            values(&db, "SELECT id, b FROM t WHERE id = 2"),
            vec![Some(2), Some(2)]
        );
    }
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        for id in 1..=3 {
            assert_eq!(
                values(&db, &format!("SELECT id, b FROM t WHERE id = {id}")),
                vec![Some(id), Some(id)],
                "row {id} after a reopen"
            );
        }
    }
}

/// After a full checkpoint the log holds no renames and the manifest is
/// the only history: a volume read back from disk keeps its own column
/// names, and the renamed names resolve through the mapping alone
#[test]
fn renames_survive_a_full_checkpoint_and_a_reopen_through_the_manifest_alone() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 1, 2)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("ALTER TABLE t RENAME COLUMN a TO t", ())
            .unwrap();
        db.execute("ALTER TABLE t RENAME COLUMN b TO a", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(
            values(&db, "SELECT t, a FROM t WHERE id = 1"),
            vec![Some(1), Some(2)]
        );
    }
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(
            values(&db, "SELECT t, a FROM t WHERE id = 1"),
            vec![Some(1), Some(2)]
        );
        let rows = db
            .query("SELECT id FROM t WHERE a = 2 AND t = 1", ())
            .unwrap()
            .count();
        assert_eq!(rows, 1);
    }
}

/// A checkpoint that runs while a rename statement is in progress seals
/// whatever is sealable and writes the manifest out only once the
/// statement is complete, under the DDL guard, so the manifest never
/// holds the rename without its log position or the position without the
/// rename, whichever way the seal and the statement interleave
#[test]
fn a_checkpoint_during_a_rename_statement_persists_the_whole_statement_or_none() {
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("INSERT INTO t VALUES (2, 2)", ()).unwrap();
        // The statement's steps, as the executor runs them, under the guard
        // it holds; a checkpoint started meanwhile seals row 2 before or
        // after the rename and waits for the guard to write the manifest
        let ddl = db.engine().ddl_guard();
        let checkpointer = db.clone();
        let checkpoint = std::thread::spawn(move || {
            checkpointer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        {
            let txn = db.engine().begin_transaction().unwrap();
            let mut table = txn.get_table("t").unwrap();
            table.rename_column("a", "b").unwrap();
        }
        db.engine().refresh_schema_cache("t").unwrap();
        db.engine()
            .record_alter_table_rename_column("t", "a", "b")
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        db.engine().propagate_column_alias("t", "b", "a");
        drop(ddl);
        checkpoint.join().unwrap();
        db.execute("INSERT INTO t VALUES (3, 3)", ()).unwrap();
        for id in 1..=3 {
            assert_eq!(
                values(&db, &format!("SELECT id, b FROM t WHERE id = {id}")),
                vec![Some(id), Some(id)],
                "row {id} live"
            );
        }
    }
    for _ in 0..2 {
        let db = Database::open(&dsn).unwrap();
        for id in 1..=3 {
            assert_eq!(
                values(&db, &format!("SELECT id, b FROM t WHERE id = {id}")),
                vec![Some(id), Some(id)],
                "row {id} after a reopen"
            );
        }
    }
}
