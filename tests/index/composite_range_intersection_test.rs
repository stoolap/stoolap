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

//! A range on a multi-column index gives its rows in key order; intersected
//! with another index's rows it answers as a scan does

use stoolap::Database;

/// `t` with indexes on (a, b) and c, `plain` with the same rows and none.
/// b falls as the id rises, so the range's key order is the reverse of id
/// order. The equality on the BOOLEAN a is what leads the lookup to the
/// multi-column index.
fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for table in ["t", "plain"] {
        db.execute(
            &format!(
                "CREATE TABLE {table} (id INTEGER PRIMARY KEY, a BOOLEAN NOT NULL, \
                 b INTEGER NOT NULL, c BOOLEAN NOT NULL)"
            ),
            (),
        )
        .unwrap();
        let insert = db
            .prepare(&format!("INSERT INTO {table} VALUES ($1, true, $2, $3)"))
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        for id in 1..=100i64 {
            insert.execute((id, 101 - id, id % 2 == 0)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
    }
    db.execute("CREATE INDEX idx_t_ab ON t(a, b)", ()).unwrap();
    db.execute("CREATE INDEX idx_t_c ON t(c)", ()).unwrap();
    db
}

fn ids(db: &Database, table: &str, filter: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(&format!("SELECT id FROM {table} WHERE {filter}"), ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn assert_as_scan(db: &Database, filters: &[&str]) {
    for filter in filters {
        let scanned = ids(db, "plain", filter);
        assert!(!scanned.is_empty(), "{filter}: the fixture matches rows");
        assert_eq!(ids(db, "t", filter), scanned, "{filter}");
    }
}

#[test]
fn a_multi_column_range_and_a_boolean_index_answer_as_a_scan() {
    let db = tables("composite_range_intersection");
    assert_as_scan(
        &db,
        &[
            "a = true AND (b <= 80 AND c = true)",
            "(a = true AND b <= 80) AND c = true",
            "a = true AND b <= 80 AND c = true",
            "a = true AND (b >= 10 AND b <= 60 AND c = false)",
        ],
    );
}
