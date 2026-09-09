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

//! Fixed statement visibility across compiled, generic, and correlated reads.

use std::sync::Arc;
use stoolap::executor::{ExecutionContext, Executor};
use stoolap::storage::mvcc::engine::MVCCEngine;
use stoolap::storage::traits::QueryResult;
use stoolap::Value;

fn setup() -> (Arc<MVCCEngine>, Executor) {
    let engine = Arc::new(MVCCEngine::in_memory());
    engine.open_engine().unwrap();
    let executor = Executor::new(engine.clone());
    executor
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)")
        .unwrap();
    executor.execute("CREATE INDEX t_g ON t(g)").unwrap();
    executor
        .execute("CREATE TABLE outer_keys (id INTEGER PRIMARY KEY, g INTEGER)")
        .unwrap();
    executor
        .execute("INSERT INTO t VALUES (1, 10), (2, 10)")
        .unwrap();
    executor
        .execute("INSERT INTO outer_keys VALUES (1, 10), (2, 20)")
        .unwrap();
    (engine, executor)
}

fn rows(mut result: Box<dyn QueryResult>) -> Vec<Vec<Value>> {
    let mut rows = Vec::new();
    while result.next() {
        rows.push(result.take_row().as_slice().to_vec());
    }
    assert!(result.last_error().is_none());
    rows
}

#[test]
fn fixed_epoch_covers_cached_plan_compiled_pk_and_aggregate() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let (engine, executor) = setup();
    let plan = executor
        .get_or_create_plan("SELECT * FROM t WHERE id = 1")
        .unwrap();
    drop(
        executor
            .execute_with_cached_plan(&plan, &ExecutionContext::new())
            .unwrap(),
    );
    drop(executor.execute("SELECT COUNT(*) FROM t").unwrap());
    let ctx = ExecutionContext::new().with_read_epoch(engine.registry().capture_read_epoch());
    executor
        .execute("UPDATE t SET g = 20 WHERE id = 1")
        .unwrap();
    executor.execute("DELETE FROM t WHERE id = 2").unwrap();
    executor
        .execute("INSERT INTO t VALUES (3, 30), (4, 40)")
        .unwrap();
    assert_eq!(
        rows(executor.execute_with_cached_plan(&plan, &ctx).unwrap()),
        vec![vec![Value::Integer(1), Value::Integer(10)]]
    );
    assert_eq!(
        rows(
            executor
                .execute_with_context("SELECT COUNT(*) FROM t", &ctx)
                .unwrap()
        ),
        vec![vec![Value::Integer(2)]]
    );
    assert_eq!(
        rows(
            executor
                .execute_with_context("SELECT id, g FROM t ORDER BY id", &ctx)
                .unwrap()
        ),
        vec![
            vec![Value::Integer(1), Value::Integer(10)],
            vec![Value::Integer(2), Value::Integer(10)]
        ]
    );
    assert_eq!(
        rows(executor.execute("SELECT COUNT(*) FROM t").unwrap()),
        vec![vec![Value::Integer(3)]]
    );
}

#[test]
fn fixed_epoch_keeps_old_secondary_keys_for_correlated_and_anti_reads() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let (engine, executor) = setup();
    let ctx = ExecutionContext::new().with_read_epoch(engine.registry().capture_read_epoch());
    executor.execute("UPDATE t SET g = 20").unwrap();
    assert_eq!(
        rows(
            executor
                .execute_with_context("SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g", &ctx)
                .unwrap()
        ),
        vec![vec![Value::Integer(10), Value::Integer(2)]]
    );
    assert_eq!(
        rows(
            executor
                .execute_with_context(
                    "SELECT o.id, t.id FROM outer_keys o JOIN t ON o.g = t.g ORDER BY o.id, t.id",
                    &ctx
                )
                .unwrap()
        ),
        vec![
            vec![Value::Integer(1), Value::Integer(1)],
            vec![Value::Integer(1), Value::Integer(2)]
        ]
    );

    assert_eq!(rows(executor.execute_with_context("SELECT id FROM outer_keys o WHERE EXISTS (SELECT 1 FROM t WHERE t.g = o.g) ORDER BY id", &ctx).unwrap()), vec![vec![Value::Integer(1)]]);
    assert_eq!(rows(executor.execute_with_context("SELECT o.id, (SELECT COUNT(*) FROM t WHERE t.g = o.g) AS n FROM outer_keys o ORDER BY o.id", &ctx).unwrap()), vec![vec![Value::Integer(1), Value::Integer(2)], vec![Value::Integer(2), Value::Integer(0)]]);
    assert_eq!(rows(executor.execute_with_context("SELECT id FROM outer_keys o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.g = o.g) ORDER BY id", &ctx).unwrap()), vec![vec![Value::Integer(2)]]);
}

#[test]
fn fixed_epoch_dml_does_not_select_later_matching_rows() {
    #[cfg(feature = "test-failpoints")]
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let (engine, executor) = setup();
    let ctx = ExecutionContext::new().with_read_epoch(engine.registry().capture_read_epoch());
    executor.execute("UPDATE t SET g = 20").unwrap();
    let updated = executor
        .execute_with_context("UPDATE t SET g = 99 WHERE g = 20", &ctx)
        .unwrap();
    assert_eq!(updated.rows_affected(), 0);
    let deleted = executor
        .execute_with_context("DELETE FROM t WHERE g = 20", &ctx)
        .unwrap();
    assert_eq!(deleted.rows_affected(), 0);
    assert_eq!(
        rows(executor.execute("SELECT g FROM t ORDER BY id").unwrap()),
        vec![vec![Value::Integer(20)], vec![Value::Integer(20)]]
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn failed_commit_acknowledges_only_after_all_table_undo_and_retained_epoch_release() {
    use stoolap::storage::traits::Engine;
    let _guard = stoolap::test_failpoints::FailpointGuard::new();
    let (engine, executor) = setup();
    let transaction = engine.begin_transaction().unwrap();
    let txn_id = transaction.id();
    executor.install_transaction(transaction);
    executor.execute("UPDATE t SET g = 99").unwrap();
    executor.execute("UPDATE outer_keys SET g = 99").unwrap();
    // This root could have been captured during failed publication. Its
    // lease must keep the aborted marker even after complete successful undo.
    let lease = engine.registry().capture_read_epoch();
    stoolap::test_failpoints::fail_table_publish_on(2);
    assert!(executor.execute("COMMIT").is_err());
    engine.registry().run_gc();
    assert_eq!(engine.registry().get_commit_sequence(txn_id), None);
    assert_eq!(
        rows(executor.execute("SELECT SUM(g) FROM t").unwrap()),
        vec![vec![Value::Integer(20)]]
    );
    assert_eq!(
        rows(executor.execute("SELECT SUM(g) FROM outer_keys").unwrap()),
        vec![vec![Value::Integer(30)]]
    );
    drop(lease);
    engine.registry().run_gc();
    // Retired outcome metadata becomes implicit only once no pre-undo root
    // can refer to the aborted publication.
    assert_eq!(engine.registry().get_commit_sequence(txn_id), Some(0));
}
