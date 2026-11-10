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
fn a_cascade_leaves_a_child_that_moved_to_another_parent_after_it_was_read() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://fk_cascade_race_moved_child").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON UPDATE CASCADE)",
        "INSERT INTO p VALUES (1), (3)",
        "INSERT INTO c VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let writer = db.clone();
    test_failpoints::after_cascade_targets_read(move || {
        std::thread::spawn(move || writer.execute("UPDATE c SET pid = 3 WHERE id = 1", ()))
            .join()
            .unwrap()
            .unwrap();
    });
    db.execute("UPDATE p SET id = 2 WHERE id = 1", ()).unwrap();
    let pid: i64 = db.query_one("SELECT pid FROM c WHERE id = 1", ()).unwrap();
    assert_eq!(pid, 3, "the cascade rewrote a child that no longer named 1");
}
