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

//! An index created inside a transaction holds that transaction's own rows
//! by the keys they commit with, and refuses a unique key they break

use stoolap::storage::traits::{Engine, ScanAccess};
use stoolap::{Database, IndexType, Row, Value};

const KINDS: [(&str, Option<IndexType>, &[&str]); 4] = [
    ("btree", Some(IndexType::BTree), &["k"]),
    ("hash", Some(IndexType::Hash), &["k"]),
    ("bitmap", Some(IndexType::Bitmap), &["k"]),
    ("multi_column", None, &["k", "w"]),
];

fn table(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        (),
    )
    .unwrap();
    db
}

fn row(id: i64, k: i64) -> Row {
    Row::from_values(vec![
        Value::Integer(id),
        Value::Integer(k),
        Value::Integer(1),
    ])
}

#[test]
fn a_row_rekeyed_after_the_index_was_created_in_its_transaction_frees_the_old_key() {
    let mut wrong = Vec::new();
    for (kind, index_type, columns) in KINDS {
        let db = table(&format!("idx_in_txn_rekeyed_{kind}"));
        let mut tx = db.engine().begin_transaction().unwrap();
        let mut t = tx.get_table("t").unwrap();
        t.insert(row(1, 10)).unwrap();
        t.create_index_with_type("ix", columns, true, index_type)
            .unwrap();
        t.update(ScanAccess::Seq, None, &mut |_, mut r| {
            r.set(1, Value::Integer(20)).unwrap();
            Ok((r, true))
        })
        .unwrap();
        drop(t);
        tx.commit().unwrap();

        if let Err(e) = db.execute("INSERT INTO t VALUES (2, 10, 1)", ()) {
            wrong.push(format!("{kind}: the old key still holds row 1: {e}"));
        }
        match db.execute("INSERT INTO t VALUES (3, 20, 1)", ()) {
            Err(e) if e.to_string().contains("unique") => {}
            other => wrong.push(format!(
                "{kind}: the new key does not hold row 1: {other:?}"
            )),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_rolled_back_row_leaves_no_key_in_an_index_created_in_its_transaction() {
    let mut wrong = Vec::new();
    for (kind, index_type, columns) in KINDS {
        let db = table(&format!("idx_in_txn_rolled_back_{kind}"));
        let mut tx = db.engine().begin_transaction().unwrap();
        let mut t = tx.get_table("t").unwrap();
        t.insert(row(1, 10)).unwrap();
        t.create_index_with_type("ix", columns, true, index_type)
            .unwrap();
        drop(t);
        tx.rollback().unwrap();

        if let Err(e) = db.execute("INSERT INTO t VALUES (2, 10, 1)", ()) {
            wrong.push(format!("{kind}: the rolled back row holds the key: {e}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_unique_index_created_in_a_transaction_refuses_its_own_duplicate_rows() {
    for (kind, index_type, columns) in KINDS {
        let db = table(&format!("idx_in_txn_duplicates_{kind}"));
        db.execute("INSERT INTO t VALUES (1, 10, 1)", ()).unwrap();
        for duplicates in [vec![row(2, 10)], vec![row(2, 30), row(3, 30)]] {
            let mut tx = db.engine().begin_transaction().unwrap();
            let mut t = tx.get_table("t").unwrap();
            for r in duplicates {
                t.insert(r).unwrap();
            }
            let error = t
                .create_index_with_type("ix", columns, true, index_type)
                .unwrap_err();
            assert!(error.to_string().contains("unique"), "{kind}: {error}");
            drop(t);
            tx.rollback().unwrap();
        }
    }
}
