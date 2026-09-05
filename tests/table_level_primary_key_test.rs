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

//! A PRIMARY KEY given at table level is a key: over one column it is the
//! column's own primary key, over several it is kept unique the way a
//! composite UNIQUE is, and ON CONFLICT finds it either way.

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| r.get::<i64>(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

#[test]
fn test_a_composite_primary_key_is_unique() {
    let db = Database::open("memory://composite_primary_key").unwrap();
    db.execute(
        "CREATE TABLE cu (k1 INTEGER, k2 INTEGER, v INTEGER, PRIMARY KEY (k1, k2))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO cu VALUES (1, 1, 10), (1, 2, 20)", ())
        .unwrap();
    assert!(db.execute("INSERT INTO cu VALUES (1, 1, 99)", ()).is_err());
    db.execute(
        "INSERT INTO cu VALUES (1, 1, 100), (3, 3, 33) ON CONFLICT (k1, k2) DO UPDATE SET v = cu.v + excluded.v",
        (),
    )
    .unwrap();
    let skipped = db
        .execute(
            "INSERT INTO cu VALUES (1, 2, 999) ON CONFLICT (k1, k2) DO NOTHING",
            (),
        )
        .unwrap();
    assert_eq!(skipped, 0);
    assert_eq!(
        rows(&db, "SELECT k1, k2, v FROM cu ORDER BY k1, k2"),
        ["1,1,110", "1,2,20", "3,3,33"]
    );
}

#[test]
fn test_a_table_level_primary_key_over_one_column_is_the_column_key() {
    let db = Database::open("memory://table_level_primary_key").unwrap();
    db.execute(
        "CREATE TABLE t1 (id INTEGER, v INTEGER, PRIMARY KEY (id))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t1 VALUES (1, 10)", ()).unwrap();
    assert!(db.execute("INSERT INTO t1 VALUES (1, 11)", ()).is_err());
    db.execute(
        "INSERT INTO t1 VALUES (1, 12) ON CONFLICT (id) DO UPDATE SET v = excluded.v",
        (),
    )
    .unwrap();
    assert_eq!(rows(&db, "SELECT id, v FROM t1 ORDER BY id"), ["1,12"]);
    assert_eq!(rows(&db, "SELECT v FROM t1 WHERE id = 1"), ["12"]);
}
