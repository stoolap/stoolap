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

#![cfg(feature = "test-failpoints")]

use stoolap::{test_failpoints, Database};

fn fixture(cold_target: bool) -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&sync_mode=full",
        dir.path().display()
    ))
    .unwrap();
    let order = if cold_target {
        ["target", "source"]
    } else {
        ["source", "target"]
    };
    for (position, name) in order.iter().enumerate() {
        db.execute(
            &format!("CREATE TABLE {name} (id INTEGER PRIMARY KEY, amount INTEGER)"),
            (),
        )
        .unwrap();
        let values = if *name == "source" {
            "(1, 10), (2, 20), (3, 30)"
        } else {
            "(1, 0), (2, 0), (3, 0)"
        };
        db.execute(&format!("INSERT INTO {name} VALUES {values}"), ())
            .unwrap();
        if position == 0 {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
    }
    (dir, db)
}

fn assert_dml_cold_error(cold_target: bool, sql: &str) {
    let _guard = test_failpoints::FailpointGuard::new();
    let (_dir, db) = fixture(cold_target);
    let mut failures = 0;
    for nth in 1..=16 {
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO target VALUES (99, 99)", ())
            .unwrap();
        test_failpoints::fail_cold_read_on(nth);
        let result = tx.execute(sql, ());
        test_failpoints::fail_cold_read_on(0);
        match result {
            Err(error) => {
                failures += 1;
                assert!(
                    error.to_string().contains("injected cold read failure"),
                    "read {nth}: {sql}: {error}"
                );
                assert_eq!(
                    tx.query_one::<i64, _>("SELECT SUM(amount) FROM target WHERE id <= 3", ())
                        .unwrap(),
                    0
                );
                assert_eq!(
                    tx.query_one::<i64, _>("SELECT COUNT(*) FROM target WHERE id <= 3", ())
                        .unwrap(),
                    3
                );
                // The failed statement must preserve earlier successful work.
                tx.commit().unwrap();
                assert_eq!(
                    db.query_one::<i64, _>("SELECT amount FROM target WHERE id = 99", ())
                        .unwrap(),
                    99
                );
                db.execute("DELETE FROM target WHERE id = 99", ()).unwrap();
            }
            Ok(count) => {
                assert_eq!(
                    count, 3,
                    "read {nth}: subquery failure reported a successful partial statement: {sql}"
                );
                tx.rollback().unwrap();
            }
        }
    }
    assert!(
        failures >= 2,
        "must exercise errors after scan progress: {sql}"
    );
}

const SCALAR_UPDATE: &str = "UPDATE target SET amount = (SELECT amount FROM source WHERE source.id = target.id LIMIT 1) WHERE target.id <= 3";

#[test]
fn scalar_update_propagates_inner_cold_error() {
    assert_dml_cold_error(false, SCALAR_UPDATE);
}

#[test]
fn scalar_update_propagates_outer_cold_error() {
    assert_dml_cold_error(true, SCALAR_UPDATE);
}

#[test]
fn update_correlated_predicate_preserves_cold_errors() {
    for cold_target in [false, true] {
        assert_dml_cold_error(cold_target, "UPDATE target SET amount = 5 WHERE target.id <= 3 AND (SELECT amount FROM source WHERE source.id = target.id LIMIT 1) > 0");
    }
}

#[test]
fn delete_correlated_predicate_preserves_cold_errors() {
    for cold_target in [false, true] {
        assert_dml_cold_error(cold_target, "DELETE FROM target WHERE target.id <= 3 AND (SELECT amount FROM source WHERE source.id = target.id LIMIT 1) > 0");
    }
}

#[test]
fn delete_returning_propagates_outer_cold_scan_error() {
    assert_dml_cold_error(true, "DELETE FROM target WHERE target.id <= 3 RETURNING id");
}

#[test]
fn correlated_select_propagates_outer_scan_error() {
    let _guard = test_failpoints::FailpointGuard::new();
    let (_dir, db) = fixture(true);
    test_failpoints::fail_cold_read_on(1);
    let result: stoolap::Result<Vec<_>> = db.query(
        "SELECT id FROM target WHERE EXISTS (SELECT 1 FROM source WHERE source.id = target.id LIMIT 1)", (),
    ).and_then(|rows| rows.collect());
    test_failpoints::fail_cold_read_on(0);
    let error = result.expect_err("correlated SELECT swallowed its cold outer scan error");
    assert!(
        error.to_string().contains("injected cold read failure"),
        "{error}"
    );
}
