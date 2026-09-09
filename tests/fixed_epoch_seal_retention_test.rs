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

use stoolap::executor::{ExecutionContext, Executor};
use stoolap::storage::traits::QueryResult;
use stoolap::{Database, Value};

fn collect(mut result: Box<dyn QueryResult>) -> Vec<Vec<Value>> {
    let mut values = Vec::new();
    while result.next() {
        values.push(result.take_row().as_slice().to_vec());
    }
    assert!(result.last_error().is_none());
    values
}

#[test]
fn delayed_table_binding_keeps_epoch_across_a_later_seal() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (3, 30)", ())
        .unwrap();
    let executor = Executor::new(db.engine().clone());
    // No Table or hot root has been captured yet; the external epoch alone
    // owns the right to bind this table later at the earlier cutoff.
    let context =
        ExecutionContext::new().with_read_epoch(db.engine().registry().capture_read_epoch());
    db.execute("UPDATE t SET g = 20 WHERE id = 1", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        collect(
            executor
                .execute_with_context("SELECT id, g FROM t ORDER BY id", &context)
                .unwrap()
        ),
        vec![
            vec![Value::Integer(1), Value::Integer(10)],
            vec![Value::Integer(3), Value::Integer(30)],
        ]
    );
    assert!(db
        .engine()
        .volume_stats()
        .iter()
        .any(|volume| volume.0 == "t"));
}
