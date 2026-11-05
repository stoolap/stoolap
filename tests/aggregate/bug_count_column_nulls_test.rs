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

//! Regression tests for: COUNT(col) counting NULL values in the GROUP BY
//! fast path. COUNT(col) must count only non-NULL values; COUNT(*) counts
//! all rows.

use stoolap::Database;

fn setup_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://count_nulls_{}", name))
        .expect("Failed to create database");
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, c INTEGER)",
        (),
    )
    .unwrap();
    // g=1: c values [10, NULL]; g=2: c values [NULL, NULL]
    db.execute("INSERT INTO t (id, g, c) VALUES (1, 1, 10)", ())
        .unwrap();
    db.execute("INSERT INTO t (id, g, c) VALUES (2, 1, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO t (id, g, c) VALUES (3, 2, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO t (id, g, c) VALUES (4, 2, NULL)", ())
        .unwrap();
    db
}

#[test]
fn test_count_column_skips_nulls_in_group_by_fast_path() {
    let db = setup_db("group_by");
    let rows = db
        .query("SELECT g, COUNT(c) FROM t GROUP BY g", ())
        .unwrap();

    let mut counts = std::collections::HashMap::new();
    for r in rows {
        let r = r.unwrap();
        counts.insert(r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap());
    }
    assert_eq!(counts.get(&1), Some(&1), "COUNT(c) must skip the NULL");
    assert_eq!(counts.get(&2), Some(&0), "all-NULL group counts 0");
}

#[test]
fn test_count_column_in_having() {
    let db = setup_db("having");
    let mut rows = db
        .query("SELECT g FROM t GROUP BY g HAVING COUNT(c) = 0", ())
        .unwrap();
    let r = rows
        .next()
        .expect("one group with zero non-NULL c")
        .unwrap();
    assert_eq!(r.get::<i64>(0).unwrap(), 2);
    assert!(rows.next().is_none(), "only group 2 has COUNT(c) = 0");
}

#[test]
fn test_having_count_star_binds_to_count_star() {
    let db = setup_db("having_star");
    // Both groups have 2 rows, so HAVING COUNT(*) > 1 must keep both,
    // even though the SELECT-list COUNT(c) values are 1 and 0.
    let rows = db
        .query(
            "SELECT g, COUNT(c) FROM t GROUP BY g HAVING COUNT(*) > 1",
            (),
        )
        .unwrap();
    let mut counts = std::collections::HashMap::new();
    for r in rows {
        let r = r.unwrap();
        counts.insert(r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap());
    }
    assert_eq!(counts.len(), 2, "HAVING COUNT(*) must not read COUNT(c)");
    assert_eq!(counts.get(&1), Some(&1));
    assert_eq!(counts.get(&2), Some(&0));
}

#[test]
fn test_count_column_with_index_on_group_column() {
    // An index on the group column routes into the streaming GROUP BY
    // path, which previously treated COUNT(col) as COUNT(*).
    let db = setup_db("indexed_group");
    db.execute("CREATE INDEX idx_t_g ON t (g)", ()).unwrap();
    let rows = db
        .query("SELECT g, COUNT(c) FROM t GROUP BY g", ())
        .unwrap();
    let mut counts = std::collections::HashMap::new();
    for r in rows {
        let r = r.unwrap();
        counts.insert(r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap());
    }
    assert_eq!(counts.get(&1), Some(&1), "COUNT(c) must skip NULLs");
    assert_eq!(counts.get(&2), Some(&0), "all-NULL group counts 0");
}

#[test]
fn test_count_distinct_in_derived_table() {
    let db = setup_db("distinct_derived");
    // g=3 gets duplicate values: c in [7, 7, NULL].
    db.execute("INSERT INTO t (id, g, c) VALUES (5, 3, 7)", ())
        .unwrap();
    db.execute("INSERT INTO t (id, g, c) VALUES (6, 3, 7)", ())
        .unwrap();
    db.execute("INSERT INTO t (id, g, c) VALUES (7, 3, NULL)", ())
        .unwrap();
    let rows = db
        .query(
            "SELECT g, COUNT(DISTINCT c) FROM (SELECT g, c FROM t) s GROUP BY g",
            (),
        )
        .unwrap();
    let mut counts = std::collections::HashMap::new();
    for r in rows {
        let r = r.unwrap();
        counts.insert(r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap());
    }
    assert_eq!(
        counts.get(&3),
        Some(&1),
        "COUNT(DISTINCT c) must dedup and skip NULLs"
    );
}

#[test]
fn test_count_column_without_group_by() {
    let db = setup_db("plain");
    let mut rows = db.query("SELECT COUNT(c), COUNT(*) FROM t", ()).unwrap();
    let r = rows.next().unwrap().unwrap();
    assert_eq!(r.get::<i64>(0).unwrap(), 1, "COUNT(c): one non-NULL value");
    assert_eq!(r.get::<i64>(1).unwrap(), 4);
}
