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

//! A prepared statement's LIMIT subquery is prepared once, and read again
//! on every execution

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::prepared_dml_built;
use stoolap::Database;

#[test]
fn a_prepared_limit_subquery_is_prepared_once() {
    let db = Database::open("memory://limit_subquery_prepare").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3), (4)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1), (2)", ()).unwrap();
    let statement = db
        .prepare("SELECT id FROM t ORDER BY id LIMIT (SELECT COUNT(*) FROM u)")
        .unwrap();
    let before = prepared_dml_built();
    let mut sizes = Vec::new();
    for id in 3..8 {
        sizes.push(statement.query(()).unwrap().count());
        db.execute("INSERT INTO u VALUES ($1)", (id,)).unwrap();
    }
    assert_eq!(sizes, [2, 3, 4, 4, 4], "each execution reads u again");
    assert_eq!(prepared_dml_built() - before, 1, "prepared again");
}
