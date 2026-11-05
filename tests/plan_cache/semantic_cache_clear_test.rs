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

//! Rows a SELECT caches right after a clear of the semantic cache stay
//! counted, so a later invalidation does not take the count below zero

#![cfg(feature = "test-failpoints")]

use std::sync::Arc;

use stoolap::Database;

const CACHED: &str = "SELECT * FROM t WHERE v > 5";

#[test]
fn test_rows_cached_as_a_clear_finishes_stay_counted() {
    let db = Arc::new(Database::open("memory://semantic_cache_clear_count").unwrap());
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let hook_db = Arc::clone(&db);
    stoolap::test_failpoints::after_semantic_cache_cleared(move || {
        assert_eq!(hook_db.query(CACHED, ()).unwrap().count(), 2);
    });
    db.clear_semantic_cache().unwrap();
    db.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    assert_eq!(db.query(CACHED, ()).unwrap().count(), 3);
    assert_eq!(db.query(CACHED, ()).unwrap().count(), 3);
}
