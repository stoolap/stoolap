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

use stoolap::{Database, Params};

/// Twenty outer rows; every third id has an inner row, so 14 have none
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    for i in 1..=20 {
        db.execute("INSERT INTO t VALUES ($1)", (i,)).unwrap();
        if i % 3 == 0 {
            db.execute("INSERT INTO u VALUES ($1, $1)", (i,)).unwrap();
        }
    }
    db
}

fn count<P: Params>(db: &Database, sql: &str, params: P) -> usize {
    db.query(sql, params).unwrap().count()
}

const BASE: &str = "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.id)";

#[test]
fn test_anti_join_takes_a_parameter_limit() {
    let db = setup("anti_join_param_limit");
    assert_eq!(count(&db, BASE, ()), 14, "every unmatched row");
    assert_eq!(count(&db, &format!("{BASE} LIMIT $1"), (3,)), 3);
    assert_eq!(count(&db, &format!("{BASE} LIMIT 1 + 1"), ()), 2);
    assert_eq!(
        count(&db, &format!("{BASE} LIMIT $1 OFFSET $2"), (5, 11)),
        3
    );
}

#[test]
fn test_anti_join_takes_an_offset_without_a_limit() {
    let db = setup("anti_join_offset_only");
    assert_eq!(count(&db, &format!("{BASE} OFFSET 10"), ()), 4);
    assert_eq!(count(&db, &format!("{BASE} OFFSET $1"), (12,)), 2);
}

#[test]
fn test_anti_join_keeps_literal_limits() {
    let db = setup("anti_join_literal_limit");
    assert_eq!(count(&db, &format!("{BASE} LIMIT 20000"), ()), 14);
    assert_eq!(count(&db, &format!("{BASE} LIMIT 20000 OFFSET 12"), ()), 2);
}
