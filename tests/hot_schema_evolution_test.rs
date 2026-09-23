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

//! Rows still in the hot store after the table's columns were renamed,
//! dropped and added read as the schema says, as the sealed rows do

use stoolap::Database;

fn pairs(db: &Database, sql: &str) -> Vec<(i64, String)> {
    let mut out: Vec<(i64, String)> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get::<String>(1).unwrap_or_default())
        })
        .collect();
    out.sort();
    out
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn load(db: &Database, seal: bool) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, a TEXT, b INTEGER, spare TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1,0,'alpha',2,'one'),(2,1,'beta',NULL,'two'),(3,2,NULL,4,'three'),(4,1,'delta',1,'four')",
        (),
    )
    .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("INSERT INTO t VALUES (5,3,'epsilon',7,'five')", ())
        .unwrap();
}

fn file_db(dir: &tempfile::TempDir) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap()
}

/// A column added after a drop reads its default in every row, not the
/// dropped column's value the hot row still carries in that position
#[test]
fn a_column_added_after_a_drop_reads_its_default_in_hot_rows() {
    let dir = tempfile::tempdir().unwrap();
    for (name, db, seal) in [
        (
            "memory",
            Database::open("memory://hot_schema_drop_add").unwrap(),
            false,
        ),
        ("file", file_db(&dir), true),
    ] {
        load(&db, seal);
        db.execute("ALTER TABLE t RENAME COLUMN a TO label", ())
            .unwrap();
        db.execute("ALTER TABLE t DROP COLUMN spare", ()).unwrap();
        db.execute("ALTER TABLE t ADD COLUMN a TEXT DEFAULT 'fresh'", ())
            .unwrap();
        let fresh: Vec<(i64, String)> = (1..=5).map(|id| (id, "fresh".to_string())).collect();
        assert_eq!(
            pairs(&db, "SELECT id, a FROM t"),
            fresh,
            "{name}: projection"
        );
        assert_eq!(
            pairs(&db, "SELECT id, a FROM t WHERE k >= 0"),
            fresh,
            "{name}: filtered projection"
        );
        assert_eq!(
            ids(&db, "SELECT id FROM t WHERE a = 'fresh'"),
            vec![1, 2, 3, 4, 5],
            "{name}: filter on the added column"
        );
        assert_eq!(
            pairs(&db, "SELECT id, label FROM t WHERE k = 1"),
            vec![(2, "beta".to_string()), (4, "delta".to_string())],
            "{name}: the renamed column"
        );
        assert_eq!(
            ids(&db, "SELECT id FROM t WHERE LENGTH(label) > 4"),
            vec![1, 4, 5],
            "{name}: expression on the renamed column"
        );
    }
}

/// A filter on a column added with a default admits the hot rows that
/// do not physically carry it
#[test]
fn a_filter_on_an_added_default_column_admits_hot_rows() {
    let dir = tempfile::tempdir().unwrap();
    for (name, db, seal) in [
        (
            "memory",
            Database::open("memory://hot_schema_added_default").unwrap(),
            false,
        ),
        ("file", file_db(&dir), true),
    ] {
        load(&db, seal);
        db.execute("ALTER TABLE t RENAME COLUMN a TO label", ())
            .unwrap();
        db.execute("ALTER TABLE t ADD COLUMN a TEXT DEFAULT 'fresh'", ())
            .unwrap();
        assert_eq!(
            ids(&db, "SELECT id FROM t WHERE a = 'fresh'"),
            vec![1, 2, 3, 4, 5],
            "{name}: equality on the added column"
        );
        assert_eq!(
            pairs(&db, "SELECT id, label FROM t WHERE a = 'fresh' AND k >= 1"),
            vec![
                (2, "beta".to_string()),
                (3, String::new()),
                (4, "delta".to_string()),
                (5, "epsilon".to_string())
            ],
            "{name}: added column beside another filter"
        );
        assert_eq!(
            ids(&db, "SELECT id FROM t WHERE a IS NULL"),
            Vec::<i64>::new(),
            "{name}: the default is not NULL"
        );
        assert_eq!(
            ids(&db, "SELECT DISTINCT k FROM t WHERE a = 'fresh' ORDER BY k"),
            vec![0, 1, 2, 3],
            "{name}: distinct through the filter"
        );
    }
}

/// The columns after a dropped one are read at their new positions in
/// hot rows, as they are in sealed rows
#[test]
fn dropping_a_middle_column_moves_the_hot_rows_columns_down() {
    let dir = tempfile::tempdir().unwrap();
    for (name, db, seal) in [
        (
            "memory",
            Database::open("memory://hot_schema_drop_middle").unwrap(),
            false,
        ),
        ("file", file_db(&dir), true),
    ] {
        load(&db, seal);
        db.execute("ALTER TABLE t DROP COLUMN b", ()).unwrap();
        assert_eq!(
            pairs(&db, "SELECT id, spare FROM t"),
            vec![
                (1, "one".to_string()),
                (2, "two".to_string()),
                (3, "three".to_string()),
                (4, "four".to_string()),
                (5, "five".to_string())
            ],
            "{name}: the column after the dropped one"
        );
        assert_eq!(
            ids(&db, "SELECT id FROM t WHERE spare = 'five'"),
            vec![5],
            "{name}: filter on the moved column"
        );
        assert_eq!(
            ids(&db, "SELECT id FROM t WHERE a IS NULL"),
            vec![3],
            "{name}: the column before it"
        );
        db.execute("UPDATE t SET spare = 'six' WHERE id = 5", ())
            .unwrap();
        assert_eq!(
            ids(&db, "SELECT id FROM t WHERE spare = 'six'"),
            vec![5],
            "{name}: an update after the drop"
        );
    }
}

/// The hot rows keep the layout the columns were moved to across a reopen
/// that replays the column changes from the log
#[test]
fn a_reopen_replays_the_column_changes_onto_the_hot_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(&dir);
    load(&db, false);
    db.execute("ALTER TABLE t RENAME COLUMN a TO label", ())
        .unwrap();
    db.execute("ALTER TABLE t DROP COLUMN b", ()).unwrap();
    db.execute("ALTER TABLE t ADD COLUMN a TEXT DEFAULT 'fresh'", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (6, 4, 'zeta', 'six', 'stale')", ())
        .unwrap();
    db.close().unwrap();
    drop(db);

    let db = file_db(&dir);
    let mut expected: Vec<(i64, String)> = (1..=5).map(|id| (id, "fresh".to_string())).collect();
    expected.push((6, "stale".to_string()));
    assert_eq!(pairs(&db, "SELECT id, a FROM t"), expected);
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE a = 'fresh'"),
        vec![1, 2, 3, 4, 5]
    );
    assert_eq!(
        pairs(&db, "SELECT id, spare FROM t WHERE k = 1"),
        vec![(2, "two".to_string()), (4, "four".to_string())]
    );
}

/// A transaction that wrote a row under the columns as they were cannot
/// commit it under the columns as they are
#[test]
fn a_transaction_that_wrote_before_a_column_change_cannot_commit() {
    let db = Database::open("memory://hot_schema_layout_conflict").unwrap();
    load(&db, false);
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (6, 4, 'zeta', 8, 'six')", ())
        .unwrap();
    db.execute("ALTER TABLE t DROP COLUMN b", ()).unwrap();
    assert!(
        matches!(tx.commit(), Err(stoolap::Error::SchemaChanged { .. })),
        "a commit under the columns as they were"
    );
    assert_eq!(ids(&db, "SELECT id FROM t"), vec![1, 2, 3, 4, 5]);
    assert_eq!(
        pairs(&db, "SELECT id, spare FROM t WHERE id = 1"),
        vec![(1, "one".to_string())]
    );
    db.execute("INSERT INTO t VALUES (7, 5, 'eta', 'seven')", ())
        .unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, spare FROM t WHERE k = 5"),
        vec![(7, "seven".to_string())]
    );
}

/// A snapshot transaction that reads the version of a row an update
/// replaced after its snapshot sees that version's columns where the
/// schema now puts them
#[test]
fn a_snapshot_reads_the_older_version_in_the_new_layout() {
    use stoolap::IsolationLevel;
    let db = Database::open("memory://hot_schema_snapshot_history").unwrap();
    load(&db, false);
    let mut snapshot = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(
        snapshot
            .query("SELECT id, spare FROM t WHERE id = 2", ())
            .unwrap()
            .map(|r| r.unwrap().get::<String>(1).unwrap())
            .collect::<Vec<_>>(),
        vec!["two".to_string()]
    );
    db.execute("UPDATE t SET spare = 'deux' WHERE id = 2", ())
        .unwrap();
    db.execute("ALTER TABLE t DROP COLUMN b", ()).unwrap();
    // The snapshot still sees 'two', at the column's new position
    assert_eq!(
        snapshot
            .query("SELECT id, spare FROM t WHERE id = 2", ())
            .unwrap()
            .map(|r| r.unwrap().get::<String>(1).unwrap())
            .collect::<Vec<_>>(),
        vec!["two".to_string()]
    );
    snapshot.rollback().unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, spare FROM t WHERE id = 2"),
        vec![(2, "deux".to_string())]
    );
}

fn hot_bytes(db: &Database, table: &str) -> i64 {
    let rows = db.query("PRAGMA MEMORY_STATS", ()).unwrap();
    let at = rows
        .columns()
        .iter()
        .position(|c| c == "hot_bytes")
        .unwrap();
    for row in rows {
        let row = row.unwrap();
        if row.get::<String>(0).unwrap() == table {
            return row.get(at).unwrap();
        }
    }
    panic!("no row for {table}")
}

/// A column added to a table whose rows were all sealed fills no arena
/// slot: the cleared slots hold no row, and the hot bytes stay at zero
#[test]
fn a_column_added_after_a_seal_fills_no_cleared_slot() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(&dir);
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let insert = db.prepare("INSERT INTO t VALUES (?, ?)").unwrap();
    for id in 1..=100 {
        insert.execute((id, id)).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(hot_bytes(&db, "t"), 0, "sealed rows leave the arena");
    db.execute("ALTER TABLE t ADD COLUMN w INTEGER DEFAULT 7", ())
        .unwrap();
    assert_eq!(hot_bytes(&db, "t"), 0, "no row to lay out");
    db.execute("INSERT INTO t VALUES (101, 1, 1)", ()).unwrap();
    let one = hot_bytes(&db, "t");
    assert!(one > 0 && one < 200, "one hot row: {one} bytes");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(hot_bytes(&db, "t"), 0, "sealed again");
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE w = 7"),
        (1..=100).collect::<Vec<_>>()
    );
}

/// A write rolled back to a savepoint leaves no row behind, so a column
/// change by another connection after it does not stand in the way of
/// the rows the transaction writes under the new columns
#[test]
fn a_savepoint_rollback_leaves_no_layout_behind() {
    let db = Database::open("memory://hot_schema_savepoint_layout").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a1', 'b1')", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("SAVEPOINT s", ()).unwrap();
    tx.execute("INSERT INTO t VALUES (2, 'rolled-a', 'rolled-b')", ())
        .unwrap();
    tx.execute("ROLLBACK TO SAVEPOINT s", ()).unwrap();
    db.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
    tx.execute("INSERT INTO t VALUES (3, 'new-b')", ()).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, b FROM t"),
        vec![(1, "b1".to_string()), (3, "new-b".to_string())]
    );
}

/// An UPDATE of an existing row writes under the columns as they are as
/// an INSERT does: a column dropped before its commit refuses it
#[test]
fn an_update_written_before_a_column_change_cannot_commit() {
    use stoolap::IsolationLevel;
    let db = Database::open("memory://hot_schema_update_layout").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'old-a', 'old-b')", ())
        .unwrap();
    for isolation in [
        IsolationLevel::ReadCommitted,
        IsolationLevel::SnapshotIsolation,
    ] {
        let mut tx = db.begin_with_isolation(isolation).unwrap();
        tx.execute("UPDATE t SET b = 'new-b' WHERE id = 1", ())
            .unwrap();
        db.execute("ALTER TABLE t ADD COLUMN c TEXT DEFAULT 'c'", ())
            .unwrap();
        assert!(
            matches!(tx.commit(), Err(stoolap::Error::SchemaChanged { .. })),
            "{isolation:?}: the update was written under the old columns"
        );
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "old-b".to_string())],
            "{isolation:?}"
        );
        db.execute("ALTER TABLE t DROP COLUMN c", ()).unwrap();
    }
    // The key-equality update of the storage API, the path the statement
    // above takes, written through an older handle after the drop
    {
        use stoolap::core::{Operator, Row, Value};
        use stoolap::storage::expression::{ComparisonExpr, Expression};
        use stoolap::storage::traits::Engine;
        let mut tx = db.engine().begin_transaction().unwrap();
        let mut older = tx.get_table("t").unwrap();
        let mut by_key = ComparisonExpr::new("id", Operator::Eq, Value::Integer(1));
        by_key.prepare_for_schema(older.schema());
        db.execute("ALTER TABLE t ADD COLUMN c TEXT DEFAULT 'c'", ())
            .unwrap();
        let updated = older
            .update(Some(&by_key), &mut |row| {
                let mut values: Vec<Value> = row.iter().cloned().collect();
                values[2] = Value::text("new-b");
                Ok((Row::from_values(values), true))
            })
            .unwrap();
        assert_eq!(updated, 1);
        drop(older);
        assert!(
            matches!(tx.commit(), Err(stoolap::Error::SchemaChanged { .. })),
            "the key-equality update was written under the old columns"
        );
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "old-b".to_string())]
        );
    }
}

/// An UPDATE that matched no row wrote nothing, so a column change by
/// another connection after it does not stand in the way of the rows the
/// transaction writes under the new columns
#[test]
fn an_empty_update_binds_the_transaction_to_no_layout() {
    let db = Database::open("memory://hot_schema_empty_update").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a1', 'b1')", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    assert_eq!(
        tx.execute("UPDATE t SET b = 'right-b' WHERE a = 'absent'", ())
            .unwrap(),
        0
    );
    db.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
    tx.execute("INSERT INTO t VALUES (2, 'new-b')", ()).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, b FROM t"),
        vec![(1, "b1".to_string()), (2, "new-b".to_string())]
    );
}

/// A column dropped through a table handle of the storage API moves the
/// rows as the statement does: the handle's change is whole at once
#[test]
fn a_column_dropped_through_a_table_handle_moves_the_rows() {
    use stoolap::storage::traits::Engine;
    let db = Database::open("memory://hot_schema_handle_drop").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'wrong-a', 'old-b')", ())
        .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    tx.drop_table_column("t", "a").unwrap();
    tx.commit().unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, b FROM t"),
        vec![(1, "old-b".to_string())]
    );
    let mut tx = db.engine().begin_transaction().unwrap();
    tx.add_table_column(
        "t",
        stoolap::core::SchemaColumn::new(2, "c", stoolap::core::DataType::Text, true, false),
    )
    .unwrap();
    tx.commit().unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, b FROM t WHERE c IS NULL"),
        vec![(1, "old-b".to_string())]
    );
}

/// A handle writes under the layout it took its schema with, whatever
/// handles the transaction opened since: a row written through the
/// older handle after a column drop cannot commit
#[test]
fn a_write_through_an_older_handle_cannot_commit_after_a_column_drop() {
    use stoolap::core::{Row, Value};
    use stoolap::storage::traits::Engine;
    let db = Database::open("memory://hot_schema_older_handle").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a1', 'b1')", ())
        .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut older = tx.get_table("t").unwrap();
    db.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
    let newer = tx.get_table("t").unwrap();
    assert_eq!(newer.schema().columns.len(), 2);
    drop(newer);
    older
        .insert(Row::from_values(vec![
            Value::Integer(2),
            Value::text("wrong-a"),
            Value::text("right-b"),
        ]))
        .unwrap();
    drop(older);
    assert!(
        matches!(tx.commit(), Err(stoolap::Error::SchemaChanged { .. })),
        "the row was written under the older handle's columns"
    );
    assert_eq!(
        pairs(&db, "SELECT id, b FROM t"),
        vec![(1, "b1".to_string())]
    );
}

/// A handle that drops a column itself writes under the columns as they
/// are then: its rows commit
#[test]
fn a_handle_that_changed_the_columns_itself_writes_under_them() {
    use stoolap::core::{Row, Value};
    use stoolap::storage::traits::Engine;
    let db = Database::open("memory://hot_schema_own_change").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a1', 'b1')", ())
        .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut table = tx.get_table("t").unwrap();
    table.drop_column("a").unwrap();
    table
        .insert(Row::from_values(vec![
            Value::Integer(2),
            Value::text("new-b"),
        ]))
        .unwrap();
    drop(table);
    tx.commit().unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, b FROM t"),
        vec![(1, "b1".to_string()), (2, "new-b".to_string())]
    );
}

/// A write of any kind through an older handle after a column drop
/// carries the handle's layout: an update of a row the transaction wrote
/// and a delete by a predicate bound to the old columns cannot commit
#[test]
fn every_write_through_an_older_handle_carries_its_layout() {
    use stoolap::core::{Operator, Row, Value};
    use stoolap::storage::expression::{ComparisonExpr, Expression};
    use stoolap::storage::traits::Engine;
    for write in ["update", "delete"] {
        let db = Database::open(&format!("memory://hot_schema_older_{write}")).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'wrong-a', 'old-b')", ())
            .unwrap();
        let mut tx = db.engine().begin_transaction().unwrap();
        let mut older = tx.get_table("t").unwrap();
        db.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
        if write == "update" {
            // A row the transaction wrote under the new columns, updated
            // through the older handle
            let mut newer = tx.get_table("t").unwrap();
            newer
                .insert(Row::from_values(vec![
                    Value::Integer(2),
                    Value::text("new-b"),
                ]))
                .unwrap();
            drop(newer);
            older
                .update_by_row_ids(&[2], &mut |_| {
                    Ok((
                        Row::from_values(vec![
                            Value::Integer(2),
                            Value::text("wrong-a"),
                            Value::text("right-b"),
                        ]),
                        true,
                    ))
                })
                .unwrap();
        } else {
            // Bound to the old columns, the predicate reads b where a was:
            // it matches the row now and must not delete it
            let mut predicate = ComparisonExpr::new("a", Operator::Eq, Value::text("old-b"));
            predicate.prepare_for_schema(older.schema());
            older.delete(Some(&predicate)).unwrap();
        }
        drop(older);
        assert!(
            matches!(tx.commit(), Err(stoolap::Error::SchemaChanged { .. })),
            "{write}: written under the older handle's columns"
        );
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "old-b".to_string())],
            "{write}"
        );
    }
}

/// A handle that renames a column takes the schema and the layout as
/// they are, so a row it writes under the current columns commits
#[test]
fn a_handle_that_renamed_a_column_writes_under_the_current_columns() {
    use stoolap::core::{Row, Value};
    use stoolap::storage::traits::Engine;
    let db = Database::open("memory://hot_schema_rename_handle").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a1', 'old-b')", ())
        .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut table = tx.get_table("t").unwrap();
    db.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
    table.rename_column("b", "c").unwrap();
    table
        .insert(Row::from_values(vec![
            Value::Integer(2),
            Value::text("right-c"),
        ]))
        .unwrap();
    drop(table);
    tx.commit().unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, c FROM t"),
        vec![(1, "old-b".to_string()), (2, "right-c".to_string())]
    );
}

/// A rollback to a savepoint takes the layout of the rows it discards
/// with them: the rows that survive were written under the current
/// columns and commit
#[test]
fn a_savepoint_rollback_takes_the_discarded_rows_layout_with_them() {
    use stoolap::core::{Row, Value};
    use stoolap::storage::traits::Engine;
    let db = Database::open("memory://hot_schema_savepoint_layout_rows").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a1', 'old-b')", ())
        .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut older = tx.get_table("t").unwrap();
    db.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
    let mut newer = tx.get_table("t").unwrap();
    newer
        .insert(Row::from_values(vec![
            Value::Integer(2),
            Value::text("right-b"),
        ]))
        .unwrap();
    drop(newer);
    tx.create_savepoint("s").unwrap();
    older
        .insert(Row::from_values(vec![
            Value::Integer(3),
            Value::text("wrong-a"),
            Value::text("wrong-b"),
        ]))
        .unwrap();
    drop(older);
    tx.rollback_to_savepoint("s").unwrap();
    tx.commit().unwrap();
    assert_eq!(
        pairs(&db, "SELECT id, b FROM t"),
        vec![(1, "old-b".to_string()), (2, "right-b".to_string())]
    );
}

#[cfg(feature = "test-failpoints")]
mod failpoints {
    use super::*;
    use stoolap::test_failpoints;

    /// A commit that publishes one table and is refused on the next takes
    /// the first back: the version the publication displaced at the
    /// history limit comes back in the layout the rows have now, not the
    /// one it was displaced in
    #[test]
    fn an_undo_restores_a_displaced_version_in_the_current_layout() {
        let _guard = test_failpoints::FailpointGuard::new();
        let db = Database::open("memory://hot_schema_undo_displaced").unwrap();
        for table in ["a", "z"] {
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, x TEXT, b TEXT)"),
                (),
            )
            .unwrap();
        }
        db.execute("INSERT INTO a VALUES (1, 'wrong-x', 'b0')", ())
            .unwrap();
        // The chain at its history limit, so the next publication drops the
        // history below the version it displaces
        for n in 1..=9 {
            db.execute(&format!("UPDATE a SET b = 'b{n}' WHERE id = 1"), ())
                .unwrap();
        }
        let other = db.clone();
        // The first table's commit publishes; at the second's, before its
        // layout check, the column goes from both
        test_failpoints::after_indexes_published(move || {
            test_failpoints::after_indexes_published(move || {
                other.execute("ALTER TABLE a DROP COLUMN x", ()).unwrap();
                other.execute("ALTER TABLE z DROP COLUMN x", ()).unwrap();
            });
        });
        let mut tx = db.begin().unwrap();
        tx.execute("UPDATE a SET b = 'new-b' WHERE id = 1", ())
            .unwrap();
        tx.execute("INSERT INTO z VALUES (2, 'x', 'zb')", ())
            .unwrap();
        assert!(
            matches!(tx.commit(), Err(stoolap::Error::SchemaChanged { .. })),
            "the second table refuses the commit"
        );
        assert_eq!(
            pairs(&db, "SELECT id, b FROM a"),
            vec![(1, "b9".to_string())],
            "the displaced version came back in the new layout"
        );
        assert_eq!(
            db.engine().get_version_store("a").unwrap().chain_entries(),
            0,
            "the undo put the displaced version back at the head"
        );
        assert_eq!(ids(&db, "SELECT id FROM z"), Vec::<i64>::new());
    }

    /// A commit at the history limit keeps the version it displaced only
    /// as the head's previous version, and one refused before publishing
    /// leaves the chain as it was
    #[test]
    fn a_pruning_commit_keeps_only_the_version_it_displaced() {
        let _guard = test_failpoints::FailpointGuard::new();
        let db = Database::open("memory://hot_schema_displaced_released").unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, b TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'b0')", ()).unwrap();
        for n in 1..=9 {
            db.execute(&format!("UPDATE t SET b = 'b{n}' WHERE id = 1"), ())
                .unwrap();
        }
        let store = db.engine().get_version_store("t").unwrap();
        // A commit that goes through: the transaction and its store go
        {
            let mut tx = db.begin().unwrap();
            tx.execute("UPDATE t SET b = 'b10' WHERE id = 1", ())
                .unwrap();
            tx.commit().unwrap();
        }
        assert_eq!(store.chain_entries(), 1, "after a commit");
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "b10".to_string())]
        );
        // A commit taken back: its previous version is the head again
        for n in 11..=18 {
            db.execute(&format!("UPDATE t SET b = 'b{n}' WHERE id = 1"), ())
                .unwrap();
        }
        assert_eq!(store.chain_entries(), 9, "the next commit prunes");
        let other = db.clone();
        test_failpoints::after_indexes_published(move || {
            other
                .execute("ALTER TABLE t ADD COLUMN c TEXT", ())
                .unwrap();
        });
        let mut tx = db.begin().unwrap();
        tx.execute("UPDATE t SET b = 'b19' WHERE id = 1", ())
            .unwrap();
        assert!(matches!(
            tx.commit(),
            Err(stoolap::Error::SchemaChanged { .. })
        ));
        assert_eq!(
            store.chain_entries(),
            9,
            "a commit refused before publishing leaves the history as it was"
        );
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "b18".to_string())]
        );
    }

    /// A handle kept from a commit long visible does not own what a later
    /// commit displaced from the same row: dropping it while the later
    /// commit is in flight leaves that commit's undo whole
    #[test]
    fn a_handle_kept_from_an_older_commit_does_not_take_a_newer_commits_undo() {
        use stoolap::core::{Row, Value};
        use stoolap::storage::traits::Engine;
        let _guard = test_failpoints::FailpointGuard::new();
        let db = Database::open("memory://hot_schema_older_handle_undo").unwrap();
        for table in ["a", "z"] {
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, b TEXT)"),
                (),
            )
            .unwrap();
        }
        db.execute("INSERT INTO a VALUES (1, 'b0')", ()).unwrap();
        for n in 1..=9 {
            db.execute(&format!("UPDATE a SET b = 'b{n}' WHERE id = 1"), ())
                .unwrap();
        }
        // A commit at the history limit, its table handle kept
        let mut older = db.engine().begin_transaction().unwrap();
        let mut kept = older.get_table("a").unwrap();
        kept.update_by_row_ids(&[1], &mut |_| {
            Ok((
                Row::from_values(vec![Value::Integer(1), Value::text("b10")]),
                true,
            ))
        })
        .unwrap();
        older.commit().unwrap();
        for n in 11..=18 {
            db.execute(&format!("UPDATE a SET b = 'b{n}' WHERE id = 1"), ())
                .unwrap();
        }
        assert_eq!(
            db.engine().get_version_store("a").unwrap().chain_entries(),
            9,
            "the next commit prunes"
        );
        // The later commit publishes a (displacing b18), then at z's
        // commit the kept handle goes and z's columns change
        let other = db.clone();
        let mut kept = Some(kept);
        test_failpoints::after_indexes_published(move || {
            test_failpoints::after_indexes_published(move || {
                drop(kept.take());
                other
                    .execute("ALTER TABLE z ADD COLUMN c TEXT", ())
                    .unwrap();
            });
        });
        let mut tx = db.begin().unwrap();
        tx.execute("UPDATE a SET b = 'b19' WHERE id = 1", ())
            .unwrap();
        tx.execute("INSERT INTO z VALUES (2, 'zb')", ()).unwrap();
        assert!(matches!(
            tx.commit(),
            Err(stoolap::Error::SchemaChanged { .. })
        ));
        assert_eq!(
            pairs(&db, "SELECT id, b FROM a"),
            vec![(1, "b18".to_string())],
            "the later commit's undo found its displaced version"
        );
        assert_eq!(ids(&db, "SELECT id FROM z"), Vec::<i64>::new());
    }

    /// A commit taken back after its table was truncated has no row to
    /// restore, and leaves no version behind
    #[test]
    fn an_undo_after_a_truncate_leaves_no_version() {
        let _guard = test_failpoints::FailpointGuard::new();
        let db = Database::open("memory://hot_schema_undo_after_truncate").unwrap();
        for table in ["a", "z"] {
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, b TEXT)"),
                (),
            )
            .unwrap();
        }
        db.execute("INSERT INTO a VALUES (1, 'b0')", ()).unwrap();
        for n in 1..=9 {
            db.execute(&format!("UPDATE a SET b = 'b{n}' WHERE id = 1"), ())
                .unwrap();
        }
        let store = db.engine().get_version_store("a").unwrap();
        // The first table's commit displaces b9; at the second's, the
        // first is truncated and the second's columns change
        let other = db.clone();
        test_failpoints::after_indexes_published(move || {
            test_failpoints::after_indexes_published(move || {
                other.execute("TRUNCATE TABLE a", ()).unwrap();
                other
                    .execute("ALTER TABLE z ADD COLUMN c TEXT", ())
                    .unwrap();
            });
        });
        let mut tx = db.begin().unwrap();
        tx.execute("UPDATE a SET b = 'b10' WHERE id = 1", ())
            .unwrap();
        tx.execute("INSERT INTO z VALUES (2, 'zb')", ()).unwrap();
        assert!(matches!(
            tx.commit(),
            Err(stoolap::Error::SchemaChanged { .. })
        ));
        drop(tx);
        assert_eq!(store.chain_entries(), 0, "nothing of the undo stays");
        assert_eq!(ids(&db, "SELECT id FROM a"), Vec::<i64>::new());
        assert_eq!(ids(&db, "SELECT id FROM z"), Vec::<i64>::new());
    }

    /// A column change that lands after a commit's layout check and before
    /// its rows are published refuses the commit: the check and the
    /// publication share the lock the change moves the rows under
    #[test]
    fn a_column_dropped_between_the_check_and_the_publication_refuses_the_commit() {
        let _guard = test_failpoints::FailpointGuard::new();
        let db = Database::open("memory://hot_schema_publish_race").unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a1', 'b1')", ())
            .unwrap();
        let other = db.clone();
        test_failpoints::after_indexes_published(move || {
            other.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
        });
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO t VALUES (2, 'wrong-a', 'right-b')", ())
            .unwrap();
        assert!(
            matches!(tx.commit(), Err(stoolap::Error::SchemaChanged { .. })),
            "the row was written under the columns as they were"
        );
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "b1".to_string())]
        );
        db.execute("INSERT INTO t VALUES (2, 'right-b')", ())
            .unwrap();
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "b1".to_string()), (2, "right-b".to_string())]
        );
    }

    /// An UPDATE whose input was read under the columns as they were, a
    /// column dropped before it wrote, cannot go through: the statement's
    /// handle bound the layout it read under
    #[test]
    fn an_update_whose_input_was_read_before_a_column_drop_is_refused() {
        let _guard = test_failpoints::FailpointGuard::new();
        let db = Database::open("memory://hot_schema_stale_input").unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'wrong-a', 'old-b')", ())
            .unwrap();
        let other = db.clone();
        test_failpoints::after_version_root(move || {
            other.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
        });
        assert!(
            matches!(
                db.execute("UPDATE t SET b = 'right-b'", ()),
                Err(stoolap::Error::SchemaChanged { .. })
            ),
            "the input was read under the old columns"
        );
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "old-b".to_string())]
        );
        db.execute("UPDATE t SET b = 'right-b'", ()).unwrap();
        assert_eq!(
            pairs(&db, "SELECT id, b FROM t"),
            vec![(1, "right-b".to_string())]
        );
    }
}
