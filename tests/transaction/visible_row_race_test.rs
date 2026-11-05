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

//! A row that exists throughout a lookup is visible to it, even when an
//! update committed between the lookup's hot check and its volume read moves
//! the row's authority to the hot store

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::{Database, Engine};

#[test]
fn test_has_visible_row_holds_across_an_update_after_the_hot_check() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'orig')", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(db.engine().volume_stats().len(), 1, "row 1 is sealed");

    let tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let writer = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook_fired = Rc::clone(&fired);
    stoolap::test_failpoints::after_segment_row_hot_checked(move || {
        writer
            .execute("UPDATE t SET val = 'new' WHERE id = 1", ())
            .unwrap();
        hook_fired.set(true);
    });
    assert!(table.has_visible_row(1).unwrap());
    assert!(fired.get(), "the update landed after the hot check");
}
