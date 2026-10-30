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

//! An equality on a column with a unique index of its own reads the row by
//! that index, hot or sealed, and answers as the scan does

use stoolap::Database;

const CREATE: &str =
    "CREATE TABLE u (id INTEGER PRIMARY KEY, code INTEGER UNIQUE, name TEXT, v FLOAT)";
const BY_CODE: &str = "SELECT id, name FROM u WHERE code = $1";

/// Rows 1 to 20, code 7 * id; sealed into a volume when `cold`
fn table(name: &str, cold: bool, dir: &tempfile::TempDir) -> Database {
    let db = match cold {
        true => Database::open(&format!(
            "file://{}?checkpoint_interval=3600&compact_threshold=1000",
            dir.path().display()
        )),
        false => Database::open(&format!("memory://{name}")),
    }
    .unwrap();
    db.execute(CREATE, ()).unwrap();
    for id in 1..=20i64 {
        db.execute(
            "INSERT INTO u VALUES ($1, $2, $3, $4)",
            (id, id * 7, format!("n{id}"), id as f64),
        )
        .unwrap();
    }
    if cold {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(db.engine().volume_stats().len(), 1, "the rows are sealed");
    }
    db
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

/// `rows` of a statement the unique read answers
fn read(db: &Database, sql: &str, params: impl stoolap::Params) -> Vec<(i64, String)> {
    #[cfg(feature = "test-failpoints")]
    stoolap::test_failpoints::take_read_paths();
    let found = rows(db, sql, params);
    #[cfg(feature = "test-failpoints")]
    assert!(
        stoolap::test_failpoints::take_read_paths().contains(&"exec.unique_get"),
        "{sql} read by the unique index"
    );
    found
}

fn plan(db: &Database, sql: &str) -> Vec<String> {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect()
}

fn n(id: i64) -> Vec<(i64, String)> {
    vec![(id, format!("n{id}"))]
}

#[test]
fn a_unique_equality_reads_its_row_hot_and_sealed() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = table("unique_get_reads", cold, &dir);
        assert_eq!(read(&db, BY_CODE, (14,)), n(2), "cold {cold}");
        assert_eq!(read(&db, BY_CODE, (15,)), vec![], "cold {cold}");
        let literal = "SELECT id, name FROM u WHERE code = 21";
        assert_eq!(read(&db, literal, ()), n(3), "cold {cold}");
        let both = "SELECT id, name FROM u WHERE code = $1 AND name = $2";
        assert_eq!(read(&db, both, (28, "n4")), n(4), "cold {cold}");
        assert_eq!(read(&db, both, (28, "n5")), vec![], "cold {cold}");
        assert_eq!(read(&db, both, (28, "n4")), n(4), "cold {cold}");
    }
}

#[test]
fn an_update_and_a_delete_over_a_sealed_row_hide_its_old_value() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = table("unique_get_rewritten", cold, &dir);
        db.execute("UPDATE u SET code = 1000 WHERE id = 2", ())
            .unwrap();
        db.execute("DELETE FROM u WHERE id = 3", ()).unwrap();
        assert_eq!(read(&db, BY_CODE, (14,)), vec![], "cold {cold}");
        assert_eq!(read(&db, BY_CODE, (1000,)), n(2), "cold {cold}");
        assert_eq!(read(&db, BY_CODE, (21,)), vec![], "cold {cold}");
        // A value an updated row gave up, taken by another row
        db.execute("UPDATE u SET code = 14 WHERE id = 5", ())
            .unwrap();
        assert_eq!(read(&db, BY_CODE, (14,)), n(5), "cold {cold}");
    }
}

#[test]
fn a_value_of_another_type_answers_as_sql_equality_does() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = table("unique_get_types", cold, &dir);
        assert_eq!(rows(&db, BY_CODE, (14.0,)), n(2), "cold {cold}");
        assert_eq!(rows(&db, BY_CODE, (14.5,)), vec![], "cold {cold}");
        assert_eq!(rows(&db, BY_CODE, ("14",)), n(2), "cold {cold}");
        assert_eq!(
            rows(&db, BY_CODE, (stoolap::Value::null_unknown(),)),
            vec![],
            "cold {cold}"
        );
    }
}

#[test]
fn a_transaction_reads_its_own_writes() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = table("unique_get_transaction", cold, &dir);
        db.execute("BEGIN", ()).unwrap();
        db.execute("INSERT INTO u VALUES (99, 693, 'n99', 0.0)", ())
            .unwrap();
        db.execute("UPDATE u SET code = 2000 WHERE id = 6", ())
            .unwrap();
        assert_eq!(rows(&db, BY_CODE, (693,)), n(99), "cold {cold}");
        assert_eq!(rows(&db, BY_CODE, (2000,)), n(6), "cold {cold}");
        assert_eq!(rows(&db, BY_CODE, (42,)), vec![], "cold {cold}");
        db.execute("ROLLBACK", ()).unwrap();
        assert_eq!(read(&db, BY_CODE, (42,)), n(6), "cold {cold}");
        assert_eq!(read(&db, BY_CODE, (693,)), vec![], "cold {cold}");
    }
}

#[test]
fn explain_names_the_unique_read() {
    let dir = tempfile::tempdir().unwrap();
    let db = table("unique_get_explain", false, &dir);
    assert_eq!(
        plan(&db, "SELECT id, name FROM u WHERE code = 21"),
        vec![
            "Project id, name",
            "  Filter",
            "    UniqueGet u code = 21",
            "    else IndexScan u (code = 21)",
            "      Index unique_u_code (code =)",
        ]
    );
}

#[test]
fn an_index_that_is_not_unique_or_not_one_column_reads_no_row_by_it() {
    let db = Database::open("memory://unique_get_other_indexes").unwrap();
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER, UNIQUE(a, b))",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX w_c ON w(c)", ()).unwrap();
    db.execute("INSERT INTO w VALUES (1, 1, 2, 3)", ()).unwrap();
    for sql in [
        "SELECT id FROM w WHERE a = 1",
        "SELECT id FROM w WHERE c = 3",
    ] {
        assert!(
            plan(&db, sql)
                .iter()
                .all(|line| !line.contains("UniqueGet")),
            "{sql}: {:?}",
            plan(&db, sql)
        );
    }
}

#[test]
fn a_dropped_unique_index_leaves_the_statement_to_the_scan() {
    let db = Database::open("memory://unique_get_dropped").unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, code INTEGER)", ())
        .unwrap();
    db.execute("CREATE UNIQUE INDEX d_code ON d(code)", ())
        .unwrap();
    db.execute("INSERT INTO d VALUES (1, 7), (2, 14)", ())
        .unwrap();
    let stmt = db.prepare("SELECT id FROM d WHERE code = $1").unwrap();
    let ids = |stmt: &stoolap::Statement| {
        stmt.query((14,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&stmt), vec![2]);
    db.execute("DROP INDEX d_code ON d", ()).unwrap();
    assert_eq!(ids(&stmt), vec![2]);
    db.execute("INSERT INTO d VALUES (3, 14)", ()).unwrap();
    assert_eq!(ids(&stmt), vec![2, 3]);
}

#[test]
fn a_false_constant_term_keeps_every_row_out_before_a_key_read() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = table("unique_get_constant", cold, &dir);
        for sql in [
            "SELECT id, name FROM u WHERE code = $1 AND ABS(name) > 0 AND $2 = 0",
            "SELECT id, name FROM u WHERE id = $1 AND ABS(name) > 0 AND $2 = 0",
        ] {
            assert_eq!(rows(&db, sql, (7, 1)), vec![], "{sql}, cold {cold}");
            let stmt = db.prepare(sql).unwrap();
            assert_eq!(stmt.query((7, 1)).unwrap().count(), 0, "{sql}, cold {cold}");
        }
        assert_eq!(
            rows(
                &db,
                "SELECT id, name FROM u WHERE code = 7 AND ABS(name) > 0 AND 1 = 0",
                ()
            ),
            vec![],
            "cold {cold}"
        );
        // A constant term that holds leaves the key read to answer
        assert_eq!(
            read(
                &db,
                "SELECT id, name FROM u WHERE 1 = 1 AND code = $1",
                (7,)
            ),
            n(1),
            "cold {cold}"
        );
    }
}

/// ABS answering -1 for every argument, on a registry of its own
#[derive(Default)]
struct MinusOne;

impl stoolap::functions::ScalarFunction for MinusOne {
    fn name(&self) -> &str {
        "ABS"
    }
    fn info(&self) -> stoolap::functions::FunctionInfo {
        use stoolap::functions::{FunctionDataType, FunctionInfo, FunctionSignature, FunctionType};
        FunctionInfo::new(
            "ABS",
            FunctionType::Scalar,
            "test override",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, _args: &[stoolap::Value]) -> stoolap::core::Result<stoolap::Value> {
        Ok(stoolap::Value::Integer(-1))
    }
    fn clone_box(&self) -> Box<dyn stoolap::functions::ScalarFunction> {
        Box::new(MinusOne)
    }
}

#[test]
fn a_constant_term_is_asked_with_the_statements_own_functions() {
    let registry = stoolap::functions::registry::FunctionRegistry::new();
    registry.register_scalar::<MinusOne>();
    let mut engine = stoolap::storage::mvcc::engine::MVCCEngine::in_memory();
    stoolap::storage::traits::Engine::open(&mut engine).unwrap();
    let executor = stoolap::executor::Executor::with_function_registry(
        std::sync::Arc::new(engine),
        std::sync::Arc::new(registry),
    );
    for sql in [CREATE, "INSERT INTO u VALUES (1, 7, 'n1', 1.0)"] {
        executor.execute(sql).unwrap();
    }
    let count = |sql: &str| {
        let mut rows = executor.execute(sql).unwrap();
        let mut n = 0;
        while rows.next() {
            n += 1;
        }
        n
    };
    for sql in [
        "SELECT id FROM u WHERE id = 1 AND ABS(1) < 0",
        "SELECT id FROM u WHERE code = 7 AND ABS(1) < 0",
        "SELECT id FROM u WHERE name = 'n1' AND ABS(1) < 0",
    ] {
        assert_eq!(count(sql), 1, "{sql}: the override holds for the row");
    }
}

#[cfg(feature = "test-failpoints")]
mod windows {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use stoolap::test_failpoints::{
        after_indexes_published, after_unique_read_cold_snapshot, in_seal_after_indexes_cleaned,
        FailpointGuard,
    };

    fn ids(db: &Database, code: i64) -> Vec<i64> {
        db.query("SELECT id FROM u WHERE code = $1", (code,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect()
    }

    #[test]
    fn a_commit_moving_a_value_between_rows_is_not_read_half_done() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = table("unique_get_moving_value", false, &dir);
        assert_eq!(ids(&db, 14), vec![2]);
        let reader = db.clone();
        let seen = Rc::new(RefCell::new(None));
        let slot = Rc::clone(&seen);
        after_indexes_published(move || {
            *slot.borrow_mut() = Some((ids(&reader, 14), ids(&reader, 21)));
        });
        db.execute("BEGIN", ()).unwrap();
        db.execute("UPDATE u SET code = 1000 WHERE id = 2", ())
            .unwrap();
        db.execute("UPDATE u SET code = 14 WHERE id = 3", ())
            .unwrap();
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(
            seen.borrow_mut().take(),
            Some((vec![2], vec![3])),
            "the values as they were before the commit"
        );
        assert_eq!(ids(&db, 14), vec![3]);
        assert_eq!(ids(&db, 21), Vec::<i64>::new());
    }

    #[test]
    fn a_seal_moving_a_row_again_does_not_hide_it_halfway() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = table("unique_get_resealed", true, &dir);
        db.execute("UPDATE u SET v = 99.0 WHERE id = 2", ())
            .unwrap();
        let reader = db.clone();
        let read = Rc::new(RefCell::new(None));
        let slot = Rc::clone(&read);
        in_seal_after_indexes_cleaned(move || {
            let (done, finished) = std::sync::mpsc::channel();
            let handle = std::thread::spawn(move || {
                let found = (
                    rows(&reader, BY_CODE, (14,)),
                    rows(&reader, "SELECT id, name FROM u WHERE id = $1", (2,)),
                );
                let _ = done.send(());
                found
            });
            // A read that waits on the seal is left to finish after it
            let _ = finished.recv_timeout(std::time::Duration::from_millis(500));
            *slot.borrow_mut() = Some(handle);
        });
        db.engine().seal_for_test().unwrap();
        let handle = read.borrow_mut().take().expect("the seal moved the row");
        assert_eq!(handle.join().unwrap(), (n(2), n(2)));
    }

    #[test]
    fn a_volume_compacted_after_the_snapshot_still_holds_the_row() {
        let _guard = FailpointGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=3600&compact_threshold=1000",
            dir.path().display()
        ))
        .unwrap();
        db.execute(CREATE, ()).unwrap();
        db.execute("INSERT INTO u VALUES (1, 10, 'n1', 1.0)", ())
            .unwrap();
        db.engine().seal_for_test().unwrap();
        db.execute("INSERT INTO u VALUES (3, 30, 'n3', 3.0)", ())
            .unwrap();
        db.engine().seal_for_test().unwrap();
        assert_eq!(db.engine().volume_stats().len(), 2);
        db.engine().cold_volumes_for_test("u");
        db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        let compactor = db.clone();
        after_unique_read_cold_snapshot(move || {
            let (done, finished) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = done.send(compactor.engine().compact_for_test().is_ok());
            });
            let compacted = finished
                .recv_timeout(std::time::Duration::from_secs(10))
                .ok();
            assert_eq!(compacted, Some(true), "the compaction did not finish");
        });
        assert_eq!(read(&db, BY_CODE, (30,)), n(3));
        assert_eq!(
            db.engine().volume_stats().len(),
            1,
            "the volumes were not merged"
        );
    }
}
