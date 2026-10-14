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

//! The scalar, IN and semi-join caches serve a repeated subquery while its
//! tables are unchanged, and a commit landing after a subquery began reading
//! does not leave its older answer cached as current

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::{Database, Value};

const QUERIES: [&str; 3] = [
    "SELECT (SELECT bv FROM b WHERE bid = 1)",
    "SELECT COUNT(*) FROM a WHERE x IN (SELECT bv FROM b WHERE bid = 1)",
    "SELECT COUNT(*) FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.bv = a.x AND b.bid = 1)",
];

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE b (bid INTEGER PRIMARY KEY, bv INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO b VALUES (1, 10)", ()).unwrap();
    db.execute(
        "INSERT INTO a SELECT g.value, CASE WHEN g.value <= 2 THEN 10 ELSE 20 END \
         FROM generate_series(1, 5) g",
        (),
    )
    .unwrap();
    db
}

fn one(db: &Database, sql: &str) -> String {
    let mut rows = db.query(sql, ()).unwrap();
    match rows.next().unwrap().unwrap().get_value(0) {
        Some(Value::Integer(n)) => n.to_string(),
        other => format!("{other:?}"),
    }
}

#[test]
fn test_an_unchanged_subquery_is_served_from_its_cache() {
    let db = open("subquery_stamp_hits");
    for sql in QUERIES {
        let first = one(&db, sql);
        let hits = stoolap::test_failpoints::subquery_cache_hits();
        assert_eq!(one(&db, sql), first, "{sql}");
        assert!(
            stoolap::test_failpoints::subquery_cache_hits() > hits,
            "{sql} ran again over unchanged tables"
        );
    }
}

#[test]
fn test_a_commit_after_a_subquery_began_reading_is_not_cached_away() {
    for (i, sql) in QUERIES.iter().enumerate() {
        let reader = open(&format!("subquery_stamp_order_{i}"));
        reader
            .execute("SET isolation_level = 'SNAPSHOT'", ())
            .unwrap();
        let before = one(&reader, "SELECT COUNT(*) FROM a WHERE x = 20");
        assert_eq!(before, "3");
        // Kept alive past the hook: dropping a handle clears this thread's caches
        let writer = Rc::new(reader.clone());
        let hook_writer = Rc::clone(&writer);
        let fired = Rc::new(Cell::new(false));
        let hook_fired = Rc::clone(&fired);
        stoolap::test_failpoints::after_select_table_opened(move || {
            hook_writer
                .execute("UPDATE b SET bv = 20 WHERE bid = 1", ())
                .unwrap();
            hook_fired.set(true);
        });
        let first = one(&reader, sql);
        assert!(
            fired.get(),
            "{sql}: the update committed after the read began"
        );
        let expected = ["20", "3", "3"][i];
        assert_eq!(
            one(&reader, sql),
            expected,
            "{sql} after the update (first read {first})"
        );
    }
}

#[test]
fn test_a_table_replaced_while_the_stamp_is_read_is_not_answered_from_the_cache() {
    for (i, sql) in QUERIES.iter().enumerate() {
        let reader = open(&format!("subquery_stamp_replaced_{i}"));
        let first = one(&reader, sql);
        // Kept alive past the hook: dropping a handle clears this thread's caches
        let writer = Rc::new(reader.clone());
        let hook_writer = Rc::clone(&writer);
        let fired = Rc::new(Cell::new(false));
        let hook_fired = Rc::clone(&fired);
        stoolap::test_failpoints::after_cache_stamp_publish_read(move || {
            hook_writer.execute("DROP TABLE b", ()).unwrap();
            hook_writer
                .execute("CREATE TABLE b (bid INTEGER PRIMARY KEY, bv INTEGER)", ())
                .unwrap();
            hook_writer
                .execute("INSERT INTO b VALUES (1, 20)", ())
                .unwrap();
            hook_fired.set(true);
        });
        let expected = ["20", "3", "3"][i];
        assert_eq!(one(&reader, sql), expected, "{sql} (first read {first})");
        assert!(
            fired.get(),
            "{sql}: the table was replaced inside the stamp read"
        );
    }
}
