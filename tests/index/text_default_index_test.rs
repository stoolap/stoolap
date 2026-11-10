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

//! An index on a TEXT column takes a B-tree unless another type is asked
//! for; a JSON column keeps a hash

use stoolap::storage::traits::Engine;
use stoolap::{Database, IndexType};

#[test]
fn a_text_column_index_defaults_to_a_btree() {
    let db = Database::open("memory://text_default_index").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT UNIQUE, c TEXT, j JSON)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX ix_a ON t (a)", ()).unwrap();
    db.execute("CREATE INDEX ix_c ON t (c) USING HASH", ())
        .unwrap();
    db.execute("CREATE INDEX ix_j ON t (j)", ()).unwrap();

    let tx = db.engine().begin_transaction().unwrap();
    let t = tx.get_table("t").unwrap();
    let kinds: Vec<(&str, Option<IndexType>)> = ["a", "b", "c", "j"]
        .into_iter()
        .map(|c| (c, t.get_index_on_column(c).map(|ix| ix.index_type())))
        .collect();
    assert_eq!(
        kinds,
        [
            ("a", Some(IndexType::BTree)),
            ("b", Some(IndexType::BTree)),
            ("c", Some(IndexType::Hash)),
            ("j", Some(IndexType::Hash)),
        ]
    );
}
