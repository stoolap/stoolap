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

//! A scalar subquery keyed by a parameter keeps no answer across runs when
//! the call reads its row by the key, and keeps one when the call scans

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{subquery_cache_hits, FailpointGuard};
use stoolap::{Database, Value};

#[test]
fn a_parameter_keyed_block_keeps_its_answer_only_for_a_call_that_scans() {
    let _guard = FailpointGuard::new();
    let db = Database::open("memory://point_block_answer").unwrap();
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO w VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let insert = db
        .prepare("INSERT INTO t VALUES ($1, (SELECT x FROM w WHERE id = $2))")
        .unwrap();
    // The same key five times: an integer is read by key, a text is scanned
    let mut hits = Vec::new();
    for (first, key) in [(1, Value::Integer(2)), (11, Value::text("bad"))] {
        let before = subquery_cache_hits();
        for id in first..first + 5 {
            insert.execute((Value::Integer(id), key.clone())).unwrap();
        }
        hits.push(subquery_cache_hits() - before);
    }
    let written: Vec<Option<i64>> = db
        .query("SELECT v FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    let mut want = vec![Some(20); 5];
    want.extend([None; 5]);
    assert_eq!(
        (hits, written),
        (vec![0, 4], want),
        "(kept answers read for the integer key and for the text key, values written)"
    );
}
