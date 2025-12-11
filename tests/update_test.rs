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

//! UPDATE Statement Integration Tests

use stoolap::Database;

#[test]
fn test_simple_update() {
    let db = Database::open("memory://update_test_1").expect("Failed to create database");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t1 VALUES (1, 'a')", ())
        .expect("Failed to insert");

    println!("Before UPDATE");

    // This is where it deadlocks
    db.execute("UPDATE t1 SET val = 'b' WHERE id = 1", ())
        .expect("Failed to update");

    println!("After UPDATE");

    let val: String = db.query_one("SELECT val FROM t1 WHERE id = 1", ()).unwrap();
    assert_eq!(val, "b");
}
