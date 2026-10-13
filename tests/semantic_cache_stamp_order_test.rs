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

//! A snapshot scan that misses a commit landing after its transaction began
//! does not cache its rows as current: the stamp predates the transaction

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::Database;

const CACHED: &str = "SELECT * FROM t WHERE v > 5";

#[test]
fn test_commit_after_a_snapshot_scan_began_is_not_cached_away() {
    let reader = Database::open("memory://semantic_stamp_order").unwrap();
    reader
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    reader
        .execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    reader
        .execute("SET isolation_level = 'SNAPSHOT'", ())
        .unwrap();
    let writer = reader.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_select_table_opened(move || {
        writer.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
        hook_fired.set(true);
    });
    assert_eq!(reader.query(CACHED, ()).unwrap().count(), 2);
    assert!(fired.get(), "the insert committed after the scan began");
    assert_eq!(reader.query(CACHED, ()).unwrap().count(), 3);
    reader
        .execute("SET isolation_level = 'READ COMMITTED'", ())
        .unwrap();
}

#[test]
fn test_uncacheable_selects_take_no_stamp() {
    let db = Database::open("memory://semantic_stamp_uncacheable").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let stamped = Rc::new(Cell::new(false));
    let hook_stamped = Rc::clone(&stamped);
    stoolap::test_failpoints::after_cache_stamp_publish_read(move || hook_stamped.set(true));
    db.query("SELECT * FROM t WHERE v > $1", (5,))
        .unwrap()
        .count();
    db.query("SELECT * FROM t WHERE v > 5 ORDER BY v", ())
        .unwrap()
        .count();
    db.query("SELECT * FROM t WHERE v > 5 LIMIT 1", ())
        .unwrap()
        .count();
    db.query("SELECT id FROM t WHERE v > 5", ())
        .unwrap()
        .count();
    db.query("SELECT * FROM t AS OF TRANSACTION 1 WHERE v > 5", ())
        .unwrap()
        .count();
    assert!(!stamped.get(), "an uncacheable SELECT took a stamp");
    db.query(CACHED, ()).unwrap().count();
    assert!(stamped.get(), "a cacheable SELECT takes a stamp");
}

#[test]
fn test_table_replaced_while_the_stamp_is_read_is_not_cached_away() {
    let reader = Database::open("memory://semantic_stamp_replaced").unwrap();
    reader
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    reader.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    reader.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    let writer = reader.clone();
    let hook_writer = writer.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_cache_stamp_publish_read(move || {
        hook_writer.execute("DROP TABLE t", ()).unwrap();
        hook_writer
            .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        hook_writer
            .execute("INSERT INTO t VALUES (1, 10)", ())
            .unwrap();
        hook_fired.set(true);
    });
    assert_eq!(reader.query(CACHED, ()).unwrap().count(), 1);
    assert!(fired.get(), "the table was replaced inside the stamp read");
    writer.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    assert_eq!(reader.query(CACHED, ()).unwrap().count(), 2);
}
