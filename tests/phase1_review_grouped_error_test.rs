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

#[test]
fn grouped_correlated_projection_propagates_cold_subquery_error() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600&sync_mode=full",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE source (id INTEGER PRIMARY KEY, amount INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO source VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute(
        "CREATE TABLE target (id INTEGER PRIMARY KEY, amount INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO target VALUES (1, 0), (2, 0), (3, 0)", ())
        .unwrap();
    test_failpoints::fail_cold_read_on(1);
    let result = db.query("SELECT id, SUM(amount), (SELECT MAX(amount) FROM source WHERE source.id = target.id) FROM target GROUP BY id", ())
        .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>());
    test_failpoints::fail_cold_read_on(0);
    assert!(
        result.is_err(),
        "grouped projection swallowed cold subquery error"
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("injected cold read failure"));
}
