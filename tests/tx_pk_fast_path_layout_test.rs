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

//! A primary key SELECT inside a transaction whose table drops a column
//! between the plan's epoch check and the transaction's table handle reads
//! the new layout through the normal path, not the plan's stale positions

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::Database;

const SELECT: &str = "SELECT v FROM t WHERE id = $1";

fn table(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'x', 10)", ()).unwrap();
    db
}

#[test]
fn test_cached_lookup_in_begin_rereads_a_dropped_column_layout() {
    let db = table("tx_pk_layout_cached");
    assert_eq!(db.query_one::<i64, _>(SELECT, (1,)).unwrap(), 10);
    db.execute("BEGIN", ()).unwrap();
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_compiled_epoch_checked(move || {
        other.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
        hook_fired.set(true);
    });
    assert_eq!(db.query_one::<i64, _>(SELECT, (1,)).unwrap(), 10);
    assert!(fired.get(), "the column dropped after the epoch check");
    db.execute("COMMIT", ()).unwrap();
}

#[test]
fn test_transaction_api_lookup_rereads_a_dropped_column_layout() {
    let db = table("tx_pk_layout_api");
    let ast = stoolap::parser::Parser::new(SELECT)
        .parse_program()
        .unwrap()
        .statements
        .remove(0);
    let mut tx = db.begin().unwrap();
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_pk_lookup_resolved(move || {
        other.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
        hook_fired.set(true);
    });
    let mut rows = tx.query_prepared(&ast, (1,)).unwrap();
    assert!(rows.advance());
    assert_eq!(rows.current_row().as_slice(), [stoolap::Value::Integer(10)]);
    drop(rows);
    assert!(
        fired.get(),
        "the column dropped after the positions resolved"
    );
    tx.commit().unwrap();
}

#[test]
fn test_transaction_api_cached_lookup_rereads_a_dropped_column_layout() {
    let db = table("tx_pk_layout_api_cached");
    assert_eq!(db.query_one::<i64, _>(SELECT, (1,)).unwrap(), 10);
    let mut tx = db.begin().unwrap();
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_compiled_epoch_checked(move || {
        other.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
        hook_fired.set(true);
    });
    assert_eq!(tx.query_one::<i64, _>(SELECT, (1,)).unwrap(), 10);
    assert!(fired.get(), "the column dropped after the epoch check");
    tx.commit().unwrap();
}
