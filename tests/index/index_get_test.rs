// Copyright 2026 Stoolap Contributors
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

//! An equality on a column with a non-unique index of its own reads a hot
//! table's rows by that index, and answers as the scan does

use stoolap::Database;

const CREATE: &str = "CREATE TABLE o (id INTEGER PRIMARY KEY, g INTEGER, name TEXT, v FLOAT)";
const BY_G: &str = "SELECT id, name FROM o WHERE g = $1";

/// Rows 1 to 20 with g = id % 4 and name n<id>, indexed on g; sealed into a
/// volume when `cold`
fn table(name: &str, cold: bool, dir: &tempfile::TempDir) -> Database {
    let db = match cold {
        true => Database::open(&format!(
            "file://{}?checkpoint_interval=3600&compact_threshold=1000",
            dir.path().display()
        )),
        false => Database::open(&format!("memory://{name}")),
    }
    .unwrap();
    fill(&db);
    if cold {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(db.engine().volume_stats().len(), 1, "the rows are sealed");
    }
    db
}

fn fill(db: &Database) {
    db.execute(CREATE, ()).unwrap();
    db.execute("CREATE INDEX o_g ON o(g)", ()).unwrap();
    for id in 1..=20i64 {
        db.execute(
            "INSERT INTO o VALUES ($1, $2, $3, $4)",
            (id, id % 4, format!("n{id}"), id as f64),
        )
        .unwrap();
    }
}

fn rows(db: &Database, sql: &str, params: impl stoolap::Params) -> Vec<(i64, String)> {
    db.query(sql, params)
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap())
        })
        .collect()
}

/// `rows` of a statement, and whether the index key read answered it
fn read(db: &Database, sql: &str, params: impl stoolap::Params) -> (Vec<(i64, String)>, bool) {
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::take_read_paths();
    let found = rows(db, sql, params);
    #[cfg(feature = "test-failpoints")]
    let keyed = stoolap::test_failpoints::take_read_paths().contains(&"exec.index_get");
    #[cfg(not(feature = "test-failpoints"))]
    let keyed = true;
    (found, keyed)
}

fn plan(db: &Database, sql: &str) -> Vec<String> {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect()
}

fn n(ids: &[i64]) -> Vec<(i64, String)> {
    ids.iter().map(|&id| (id, format!("n{id}"))).collect()
}

#[test]
fn an_indexed_equality_reads_its_rows_hot_and_answers_sealed() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = table("index_get_rows", cold, &dir);
        // The first run plans the statement, the second runs the plan held
        rows(&db, BY_G, (1,));
        let (found, keyed) = read(&db, BY_G, (1,));
        assert_eq!(found, n(&[1, 5, 9, 13, 17]), "cold {cold}");
        #[cfg(feature = "test-failpoints")]
        assert_eq!(keyed, !cold, "the key read answers a hot table alone");
        let _ = keyed;
        assert_eq!(read(&db, BY_G, (7,)).0, vec![], "cold {cold}");
        assert_eq!(
            read(&db, "SELECT id, name FROM o WHERE g = $1 AND v > 10", (2,)).0,
            n(&[14, 18]),
            "cold {cold}"
        );
    }
}

#[test]
fn an_update_and_a_delete_move_rows_out_of_a_key() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("index_get_changes", false, &dir);
    rows(&db, BY_G, (3,));
    db.execute("UPDATE o SET g = 9 WHERE id = 7", ()).unwrap();
    db.execute("DELETE FROM o WHERE id = 11", ()).unwrap();
    let (found, keyed) = read(&db, BY_G, (3,));
    assert_eq!(found, n(&[3, 15, 19]));
    assert!(keyed);
    assert_eq!(read(&db, BY_G, (9,)).0, n(&[7]));
}

#[test]
fn an_ordered_statement_reads_in_its_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("index_get_ordered", false, &dir);
    for _ in 0..2 {
        assert_eq!(
            rows(
                &db,
                "SELECT id, name, v FROM o WHERE g = $1 ORDER BY v DESC LIMIT 2",
                (1,)
            ),
            n(&[17, 13])
        );
        assert_eq!(
            rows(
                &db,
                "SELECT id, name FROM o WHERE g = $1 ORDER BY name DESC",
                (2,)
            ),
            n(&[6, 2, 18, 14, 10])
        );
        assert_eq!(
            rows(
                &db,
                "SELECT DISTINCT ON (g) id, name, g, v FROM o WHERE g = $1 ORDER BY g, v DESC",
                (2,)
            ),
            n(&[18])
        );
    }
}

#[test]
fn limit_and_offset_count_the_rows_the_where_keeps() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("index_get_limit", false, &dir);
    let sql = "SELECT id, name FROM o WHERE g = $1 AND v > 2 LIMIT 2 OFFSET 1";
    rows(&db, sql, (1,));
    assert_eq!(read(&db, sql, (1,)).0, n(&[9, 13]));
}

#[test]
fn a_value_of_another_type_answers_as_sql_equality_does() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = table("index_get_types", cold, &dir);
        rows(&db, BY_G, (2,));
        assert_eq!(
            rows(&db, BY_G, (2.0,)),
            n(&[2, 6, 10, 14, 18]),
            "cold {cold}"
        );
        assert_eq!(rows(&db, BY_G, (2.5,)), vec![], "cold {cold}");
        assert_eq!(
            rows(&db, BY_G, ("2",)),
            n(&[2, 6, 10, 14, 18]),
            "cold {cold}"
        );
        assert_eq!(
            rows(&db, BY_G, (stoolap::Value::null_unknown(),)),
            vec![],
            "cold {cold}"
        );
    }
}

#[test]
fn a_number_asked_of_a_text_column_answers_as_sql_equality_does() {
    let db = Database::open("memory://index_get_text_number").unwrap();
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, code TEXT)", ())
        .unwrap();
    db.execute("CREATE INDEX c_code ON c(code)", ()).unwrap();
    db.execute(
        "INSERT INTO c VALUES (1, '7'), (2, '07'), (3, '7.0'), (4, 'x'), (5, ' 7'), (6, '8')",
        (),
    )
    .unwrap();
    let ids = |key: stoolap::Value| {
        let mut ids: Vec<i64> = db
            .query("SELECT id FROM c WHERE code = $1", (key,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        ids.sort();
        ids
    };
    for _ in 0..2 {
        assert_eq!(ids(stoolap::Value::Integer(7)), vec![1, 2, 3]);
        assert_eq!(ids(stoolap::Value::Float(7.0)), vec![1, 2, 3]);
        assert_eq!(ids(stoolap::Value::text("7")), vec![1]);
    }
}

#[test]
fn a_hash_index_reads_its_rows() {
    let db = Database::open("memory://index_get_hash").unwrap();
    db.execute("CREATE TABLE h (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute("CREATE INDEX h_name ON h(name) USING HASH", ())
        .unwrap();
    db.execute("INSERT INTO h VALUES (1, 'a'), (2, 'b'), (3, 'a')", ())
        .unwrap();
    let sql = "SELECT id, name FROM h WHERE name = $1";
    rows(&db, sql, ("a",));
    let (found, keyed) = read(&db, sql, ("a",));
    assert_eq!(found, vec![(1, "a".to_string()), (3, "a".to_string())]);
    assert!(keyed);
}

#[test]
fn a_transaction_reads_its_own_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("index_get_transaction", false, &dir);
    rows(&db, BY_G, (0,));
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO o VALUES (99, 0, 'n99', 0.0)", ())
        .unwrap();
    db.execute("UPDATE o SET g = 5 WHERE id = 4", ()).unwrap();
    let mut found = rows(&db, BY_G, (0,));
    found.sort();
    assert_eq!(found, n(&[8, 12, 16, 20, 99]));
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(read(&db, BY_G, (0,)).0, n(&[4, 8, 12, 16, 20]));
}

#[test]
fn a_key_holding_more_rows_than_the_read_takes_reads_as_the_plan_does() {
    let db = Database::open("memory://index_get_many").unwrap();
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX m_g ON m(g)", ()).unwrap();
    db.execute(
        "INSERT INTO m SELECT value, value % 2 FROM generate_series(1, 1000)",
        (),
    )
    .unwrap();
    let sql = "SELECT COUNT(*) FROM (SELECT id FROM m WHERE g = $1) s";
    let count: i64 = db.query_one(sql, (1,)).unwrap();
    assert_eq!(count, 500);
    let ids = |db: &Database| {
        db.query("SELECT id FROM m WHERE g = $1", (0,))
            .unwrap()
            .count()
    };
    ids(&db);
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::take_read_paths();
    assert_eq!(ids(&db), 500);
    #[cfg(feature = "test-failpoints")]
    assert!(
        !stoolap::test_failpoints::take_read_paths().contains(&"exec.index_get"),
        "a key past the cap is the plan's"
    );
}

#[test]
fn a_snapshot_isolation_handle_reads_as_the_plan_does() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("index_get_snapshot", false, &dir);
    db.set_default_isolation_level(stoolap::IsolationLevel::SnapshotIsolation)
        .unwrap();
    rows(&db, BY_G, (1,));
    let (found, keyed) = read(&db, BY_G, (1,));
    assert_eq!(found, n(&[1, 5, 9, 13, 17]));
    #[cfg(feature = "test-failpoints")]
    assert!(!keyed, "the key read reads read committed alone");
    let _ = keyed;
}

#[test]
fn a_prepared_statement_follows_its_index_being_dropped_and_made_again() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("index_get_index_ddl", false, &dir);
    let stmt = db.prepare(BY_G).unwrap();
    let ids = |stmt: &stoolap::Statement| {
        #[cfg(feature = "test-failpoints")]
        stoolap::test_failpoints::take_read_paths();
        let ids: Vec<i64> = stmt
            .query((1,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        #[cfg(feature = "test-failpoints")]
        let keyed = stoolap::test_failpoints::take_read_paths().contains(&"exec.index_get");
        #[cfg(not(feature = "test-failpoints"))]
        let keyed = false;
        (ids, keyed)
    };
    ids(&stmt);
    assert_eq!(ids(&stmt).0, vec![1, 5, 9, 13, 17]);
    db.execute("DROP INDEX o_g ON o", ()).unwrap();
    let (found, keyed) = ids(&stmt);
    assert_eq!(found, vec![1, 5, 9, 13, 17]);
    assert!(!keyed, "no index, no key read");
    db.execute("INSERT INTO o VALUES (21, 1, 'n21', 21.0)", ())
        .unwrap();
    db.execute("CREATE INDEX o_g2 ON o(g)", ()).unwrap();
    ids(&stmt);
    let (found, keyed) = ids(&stmt);
    assert_eq!(found, vec![1, 5, 9, 13, 17, 21]);
    #[cfg(feature = "test-failpoints")]
    assert!(keyed, "the new index reads the key");
    let _ = keyed;
}

#[test]
fn a_bitmap_index_reads_its_rows() {
    let db = Database::open("memory://index_get_bitmap").unwrap();
    db.execute(
        "CREATE TABLE b (id INTEGER PRIMARY KEY, on_sale BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX b_on_sale ON b(on_sale) USING BITMAP", ())
        .unwrap();
    db.execute(
        "INSERT INTO b VALUES (1, true), (2, false), (3, true), (4, NULL)",
        (),
    )
    .unwrap();
    let sql = "SELECT id FROM b WHERE on_sale = $1";
    let ids = |key: bool| -> Vec<i64> {
        db.query(sql, (key,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    };
    ids(true);
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::take_read_paths();
    assert_eq!(ids(true), vec![1, 3]);
    #[cfg(feature = "test-failpoints")]
    assert!(stoolap::test_failpoints::take_read_paths().contains(&"exec.index_get"));
    assert_eq!(ids(false), vec![2]);
}

#[test]
fn a_full_limit_asks_no_further_row_of_its_filter() {
    let db = Database::open("memory://index_get_limit_filter").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t(g)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 0), (2, 1, 1)", ())
        .unwrap();
    // The second row's term overflows
    let sql = "SELECT id FROM t WHERE g = $1 AND 9223372036854775807 + n > 0 LIMIT 1";
    for _ in 0..2 {
        let found: stoolap::Result<Vec<i64>> = db.query(sql, (1,)).and_then(|rows| {
            rows.map(|row| row.map(|r| r.get::<i64>(0).unwrap()))
                .collect()
        });
        assert_eq!(found.map_err(|e| e.to_string()), Ok(vec![1]));
    }
}

#[test]
fn a_snapshot_the_caller_gives_is_read_at_its_own_point() {
    use stoolap::core::IsolationLevel;
    use stoolap::executor::context::StatementSnapshot;
    use stoolap::storage::traits::Engine;
    let db = Database::open("memory://index_get_given_snapshot").unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, g INTEGER, u INTEGER)",
        "CREATE INDEX s_g ON s(g)",
        "CREATE UNIQUE INDEX s_u ON s(u)",
        "INSERT INTO s VALUES (1, 1, 10), (2, 1, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let engine = db.engine();
    let executor = stoolap::Executor::new(std::sync::Arc::clone(engine));
    let first = |ctx: &stoolap::ExecutionContext, sql: &str| -> Vec<i64> {
        let mut result = executor.execute_with_context(sql, ctx).unwrap();
        let mut out = Vec::new();
        while result.next() {
            out.push(result.row().get(0).and_then(|v| v.as_int64()).unwrap());
        }
        out.sort_unstable();
        out
    };
    // Every key read and the table's count, each planned before the snapshot
    let reads: [(&str, &[i64]); 4] = [
        ("SELECT id FROM s WHERE g = 1", &[1, 2]),
        ("SELECT id FROM s WHERE u = 30", &[]),
        ("SELECT id FROM s WHERE id = 3", &[]),
        ("SELECT COUNT(*) FROM s", &[2]),
    ];
    for (sql, _) in reads {
        first(&stoolap::ExecutionContext::new(), sql);
        first(&stoolap::ExecutionContext::new(), sql);
    }
    let snapshot = StatementSnapshot::new(
        engine
            .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
            .unwrap(),
        IsolationLevel::SnapshotIsolation,
        None,
    );
    db.execute("INSERT INTO s VALUES (3, 1, 30)", ()).unwrap();
    let ctx = stoolap::ExecutionContext::new().with_statement_snapshot(snapshot);
    let wrong: Vec<String> = reads
        .iter()
        .filter_map(|(sql, expected)| {
            let seen = first(&ctx, sql);
            (seen != *expected).then(|| format!("{sql}: {seen:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "the rows at the snapshot: {wrong:?}");
}

#[test]
fn a_read_committed_snapshot_the_caller_gives_reads_its_own_writes() {
    use stoolap::core::{IsolationLevel, Row, Value};
    use stoolap::executor::context::StatementSnapshot;
    use stoolap::storage::traits::Engine;
    let db = Database::open("memory://index_get_snapshot_writes").unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, g INTEGER)",
        "CREATE INDEX s_g ON s(g)",
        "INSERT INTO s VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let engine = db.engine();
    let executor = stoolap::Executor::new(std::sync::Arc::clone(engine));
    let first = |ctx: &stoolap::ExecutionContext, sql: &str| -> Vec<i64> {
        let mut result = executor.execute_with_context(sql, ctx).unwrap();
        let mut out = Vec::new();
        while result.next() {
            out.push(result.row().get(0).and_then(|v| v.as_int64()).unwrap());
        }
        out.sort_unstable();
        out
    };
    let reads: [(&str, &[i64]); 2] = [
        ("SELECT id FROM s WHERE g = 1", &[1, 2]),
        ("SELECT id FROM s WHERE id = 2", &[2]),
    ];
    for (sql, _) in reads {
        first(&stoolap::ExecutionContext::new(), sql);
        first(&stoolap::ExecutionContext::new(), sql);
    }
    let snapshot = StatementSnapshot::new(
        engine.begin_transaction().unwrap(),
        IsolationLevel::ReadCommitted,
        None,
    );
    snapshot
        .get_table("s")
        .unwrap()
        .insert(Row::from_values(vec![Value::Integer(2), Value::Integer(1)]))
        .unwrap();
    let ctx = stoolap::ExecutionContext::new().with_statement_snapshot(snapshot);
    let wrong: Vec<String> = reads
        .iter()
        .filter_map(|(sql, expected)| {
            let seen = first(&ctx, sql);
            (seen != *expected).then(|| format!("{sql}: {seen:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "the snapshot's own row: {wrong:?}");
}

#[test]
fn explain_names_the_index_read() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("index_get_explain", false, &dir);
    assert_eq!(
        plan(&db, "SELECT id, name FROM o WHERE g = 3"),
        vec![
            "Project id, name",
            "  Filter",
            "    IndexGet o g = 3",
            "    else IndexScan o (g = 3)",
            "      Index o_g (g =)",
        ]
    );
}

#[cfg(feature = "test-failpoints")]
mod windows {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::mpsc;
    use std::time::Duration;
    use stoolap::test_failpoints::{
        after_commit_index_removed, after_compiled_epoch_checked, after_index_get_ids_taken,
        after_indexes_published, before_index_get_manager_taken, fail_wal_syncs_under,
        programs_ran, take_read_paths, FailpointGuard,
    };

    #[test]
    fn a_sealed_table_renamed_before_its_manager_is_taken_never_reads_as_empty() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = file_table(&dir, "checkpoint_interval=3600&compact_threshold=1000");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
        let writer = db.clone();
        before_index_get_manager_taken(move || {
            writer.execute("ALTER TABLE o RENAME TO o_old", ()).unwrap();
        });
        let found: Result<Vec<i64>, String> = db
            .query("SELECT id FROM o WHERE g = $1", (1,))
            .map_err(|e| e.to_string())
            .map(|rows| {
                rows.map(|row| row.unwrap().get::<i64>(0).unwrap())
                    .collect()
            });
        assert!(
            matches!(found.as_deref(), Ok([1, 5, 9, 13, 17]) | Err(_)),
            "the sealed rows or a refusal, never no row: {found:?}"
        );
    }

    #[test]
    fn a_table_renamed_and_sealed_between_the_ids_and_the_rows_keeps_its_old_name_free() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = file_table(&dir, "checkpoint_interval=3600&compact_threshold=1000");
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
        let writer = db.clone();
        after_index_get_ids_taken(move || {
            writer.execute("ALTER TABLE o RENAME TO o_old", ()).unwrap();
            writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        let found: Result<Vec<i64>, String> = db
            .query("SELECT id FROM o WHERE g = $1", (1,))
            .map_err(|e| e.to_string())
            .map(|rows| {
                rows.map(|row| row.unwrap().get::<i64>(0).unwrap())
                    .collect()
            });
        assert!(
            matches!(found.as_deref(), Ok([1, 5, 9, 13, 17]) | Err(_)),
            "{found:?}"
        );
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert!(
            !dir.path().join("volumes").join("o").exists(),
            "no files under the old name"
        );
        db.execute("ALTER TABLE o_old RENAME TO o", ())
            .expect("the old name is free to take back");
    }

    #[test]
    fn a_run_binds_its_parameters_once() {
        let _guard = FailpointGuard::new();
        let db = Database::open("memory://index_get_binds").unwrap();
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v VECTOR(3))",
            "CREATE INDEX t_g ON t(g)",
            "INSERT INTO t SELECT value, 1, '[1,2,3]' FROM generate_series(1, 5)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        let sql =
            "SELECT VEC_DISTANCE_L2(v, $2) FROM t WHERE g = $1 AND VEC_DISTANCE_L2(v, $2) >= 0";
        let params = || vec![stoolap::Value::Integer(1), stoolap::Value::text("[1,2,4]")];
        db.query(sql, params()).unwrap().count();
        let mut returned = 0;
        take_read_paths();
        let ran = programs_ran(|| {
            returned = db.query(sql, params()).unwrap().count();
        });
        assert!(take_read_paths().contains(&"exec.index_get"));
        let per_row: Vec<String> = ran
            .into_iter()
            .filter(|(listing, times)| *times > 1 && listing.contains("LoadParam"))
            .map(|(listing, times)| format!("{times} runs of {listing}"))
            .collect();
        assert_eq!(returned, 5);
        assert!(per_row.is_empty(), "{per_row:?}");
    }

    const WAIT: Duration = Duration::from_secs(10);

    fn ids(db: &Database, g: i64) -> Vec<i64> {
        db.query("SELECT id FROM o WHERE g = $1", (g,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    }

    /// `ids`, and whether the index key read answered
    fn keyed_ids(db: &Database, g: i64) -> (Vec<i64>, bool) {
        take_read_paths();
        let found = ids(db, g);
        (found, take_read_paths().contains(&"exec.index_get"))
    }

    /// The table on a file database that has sealed nothing
    fn file_table(dir: &tempfile::TempDir, params: &str) -> Database {
        let db = Database::open(&format!("file://{}?{params}", dir.path().display())).unwrap();
        fill(&db);
        db
    }

    #[test]
    fn a_commit_failing_on_its_wal_marker_leaves_its_rows_readable() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = file_table(
            &dir,
            "sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        );
        assert_eq!(ids(&db, 2), vec![2, 6, 10, 14, 18]);
        let reader = db.clone();
        let seen = Rc::new(RefCell::new(None));
        let slot = Rc::clone(&seen);
        // Only this database's WAL fails its sync; the tests beside it keep theirs
        let own = dir.path().to_path_buf();
        after_indexes_published(move || {
            *slot.borrow_mut() = Some(ids(&reader, 2));
            fail_wal_syncs_under(Some(&own));
        });
        let failed = db.execute("DELETE FROM o WHERE id = 6", ()).is_err();
        fail_wal_syncs_under(None);
        assert!(failed, "the commit failed");
        assert_eq!(
            seen.borrow_mut().take(),
            Some(vec![2, 6, 10, 14, 18]),
            "the rows while the commit publishes"
        );
        assert_eq!(
            keyed_ids(&db, 2),
            (vec![2, 6, 10, 14, 18], true),
            "the rows the rollback put back, by the key read"
        );
    }

    #[test]
    fn a_commit_failing_on_a_unique_key_leaves_its_rows_readable() {
        let _guard = FailpointGuard::new();
        let db = Database::open("memory://index_get_unique_rollback").unwrap();
        for sql in [
            "CREATE TABLE x (id INTEGER PRIMARY KEY, u INTEGER NOT NULL, g INTEGER)",
            "CREATE UNIQUE INDEX x_u ON x (u)",
            "CREATE INDEX x_g ON x (g)",
            "INSERT INTO x VALUES (1, 10, 1), (2, 20, 1)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        let by_g = |db: &Database| -> Vec<i64> {
            db.query("SELECT id FROM x WHERE g = $1", (1,))
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect()
        };
        assert_eq!(by_g(&db), vec![1, 2]);
        let (ready_tx, ready) = mpsc::channel();
        let (commit_tx, commit) = mpsc::channel::<()>();
        let writer = {
            let (other, reader) = (db.clone(), db.clone());
            std::thread::spawn(move || {
                let mut b = other.begin().unwrap();
                b.execute("UPDATE x SET u = 30, g = 5 WHERE id = 2", ())
                    .unwrap();
                ready_tx.send(()).unwrap();
                commit.recv_timeout(WAIT).unwrap();
                let seen = Rc::new(RefCell::new(None));
                let slot = Rc::clone(&seen);
                after_commit_index_removed(move || {
                    *slot.borrow_mut() = Some(by_g(&reader));
                });
                let failed = b.commit().is_err();
                let seen = seen.borrow_mut().take();
                (failed, seen)
            })
        };
        ready.recv_timeout(WAIT).unwrap();
        db.execute("UPDATE x SET u = 30 WHERE id = 1", ()).unwrap();
        commit_tx.send(()).unwrap();
        let (failed, seen) = writer.join().unwrap();
        assert!(failed, "the commit failed on the unique key");
        assert_eq!(seen, Some(vec![1, 2]), "the rows while the commit undoes");
        assert_eq!(by_g(&db), vec![1, 2], "the rows the rollback put back");
    }

    #[test]
    fn a_seal_between_the_ids_and_the_rows_leaves_the_read_to_the_plan() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = file_table(&dir, "checkpoint_interval=3600&compact_threshold=1000");
        assert_eq!(keyed_ids(&db, 1), (vec![1, 5, 9, 13, 17], true));
        let writer = db.clone();
        after_index_get_ids_taken(move || {
            writer.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        assert_eq!(keyed_ids(&db, 1), (vec![1, 5, 9, 13, 17], false));
        assert_eq!(db.engine().volume_stats().len(), 1, "the rows were sealed");
    }

    #[test]
    fn a_truncate_between_the_ids_and_the_rows_reads_the_rows_or_none() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = table("index_get_truncate", false, &dir);
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
        let writer = db.clone();
        after_index_get_ids_taken(move || {
            writer.execute("TRUNCATE TABLE o", ()).unwrap();
            // A row taking a truncated row's id under another key
            writer
                .execute("INSERT INTO o VALUES (5, 3, 'new', 0.0)", ())
                .unwrap();
        });
        let found = ids(&db, 1);
        assert!(
            found == vec![1, 5, 9, 13, 17] || found.is_empty(),
            "the rows before the truncate or after it: {found:?}"
        );
        assert_eq!(ids(&db, 1), Vec::<i64>::new());
    }

    #[test]
    fn a_table_dropped_between_the_ids_and_the_rows_reads_its_rows_or_refuses() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = table("index_get_dropped_table", false, &dir);
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
        let writer = db.clone();
        after_index_get_ids_taken(move || {
            writer.execute("DROP TABLE o", ()).unwrap();
        });
        let found: Result<Vec<i64>, String> = db
            .query("SELECT id FROM o WHERE g = $1", (1,))
            .map_err(|e| e.to_string())
            .and_then(|rows| {
                rows.map(|row| row.map(|r| r.get::<i64>(0).unwrap()))
                    .collect::<stoolap::Result<Vec<i64>>>()
                    .map_err(|e| e.to_string())
            });
        assert!(
            matches!(found.as_deref(), Ok([1, 5, 9, 13, 17]) | Err(_)),
            "the rows or a refusal, never no row: {found:?}"
        );
    }

    #[test]
    fn a_table_renamed_after_the_plan_was_checked_gets_no_manager_under_its_old_name() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = file_table(&dir, "checkpoint_interval=3600&compact_threshold=1000");
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
        let writer = db.clone();
        after_compiled_epoch_checked(move || {
            writer.execute("ALTER TABLE o RENAME TO o_old", ()).unwrap();
        });
        // The table the checked plan read, or a refusal; never no row
        let found: Result<Vec<i64>, String> = db
            .query("SELECT id FROM o WHERE g = $1", (1,))
            .map_err(|e| e.to_string())
            .map(|rows| {
                rows.map(|row| row.unwrap().get::<i64>(0).unwrap())
                    .collect()
            });
        assert!(
            matches!(found.as_deref(), Ok([1, 5, 9, 13, 17]) | Err(_)),
            "{found:?}"
        );
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert!(
            !dir.path().join("volumes").join("o").exists(),
            "no files under the old name"
        );
    }

    #[test]
    fn a_commit_between_the_ids_and_the_rows_is_not_read_half_done() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = table("index_get_ids_window", false, &dir);
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
        let writer = db.clone();
        after_index_get_ids_taken(move || {
            writer
                .execute("UPDATE o SET g = 3 - g WHERE g IN (1, 2)", ())
                .unwrap();
        });
        assert_eq!(
            ids(&db, 1),
            vec![2, 6, 10, 14, 18],
            "the rows of one committed state"
        );
    }

    #[test]
    fn a_column_dropped_between_the_ids_and_the_rows_leaves_the_read_to_the_plan() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = table("index_get_dropped_column", false, &dir);
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
        let writer = db.clone();
        after_index_get_ids_taken(move || {
            writer.execute("ALTER TABLE o DROP COLUMN v", ()).unwrap();
        });
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
    }

    #[test]
    fn a_commit_moving_a_value_between_rows_is_not_read_half_done() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = table("index_get_moving_value", false, &dir);
        assert_eq!(ids(&db, 1), vec![1, 5, 9, 13, 17]);
        let reader = db.clone();
        let seen = Rc::new(RefCell::new(None));
        let slot = Rc::clone(&seen);
        after_indexes_published(move || {
            *slot.borrow_mut() = Some((ids(&reader, 1), ids(&reader, 2)));
        });
        db.execute("UPDATE o SET g = 3 - g WHERE g IN (1, 2)", ())
            .unwrap();
        assert_eq!(
            seen.borrow_mut().take(),
            Some((vec![1, 5, 9, 13, 17], vec![2, 6, 10, 14, 18])),
            "the values as they were before the commit"
        );
        assert_eq!(ids(&db, 1), vec![2, 6, 10, 14, 18]);
    }
}
