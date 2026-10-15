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

//! A hash index's equality candidates are capped before they are copied:
//! a key holding more rows than the cap is refused without a copy

use stoolap::core::Value;
use stoolap::storage::traits::{CappedEqual, Engine, ProbeScratch};
use stoolap::Database;

fn keyed(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT)", ())
        .unwrap();
    db.execute("CREATE INDEX t_k ON t(k) USING HASH", ())
        .unwrap();
    db.execute(
        "INSERT INTO t SELECT value, CASE WHEN value <= 50 THEN 'many' ELSE 'one' || value END \
         FROM generate_series(1, 60)",
        (),
    )
    .unwrap();
    db
}

fn candidates(db: &Database, key: &str, max: usize) -> (Option<CappedEqual>, Vec<i64>) {
    let tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let mut ids = Vec::new();
    let found = table
        .equality_candidates(
            "k",
            &Value::text(key),
            max,
            &mut ids,
            &mut ProbeScratch::default(),
        )
        .unwrap();
    (found, ids)
}

#[test]
fn a_hash_key_over_the_cap_is_refused_before_its_rows_are_copied() {
    let db = keyed("hash_cap_over");
    let (found, ids) = candidates(&db, "many", 10);
    assert_eq!(found, Some(CappedEqual::OverCap));
    assert_eq!(ids.capacity(), 0, "no row was copied");
}

#[test]
fn a_hash_key_within_the_cap_is_copied_whole() {
    let db = keyed("hash_cap_within");
    let (found, mut ids) = candidates(&db, "many", 50);
    assert_eq!(found, Some(CappedEqual::Copied));
    ids.sort_unstable();
    assert_eq!(ids, (1..=50).collect::<Vec<i64>>());
    let (found, ids) = candidates(&db, "one55", 1);
    assert_eq!((found, ids), (Some(CappedEqual::Copied), vec![55]));
    let (found, ids) = candidates(&db, "absent", 1);
    assert_eq!((found, ids), (Some(CappedEqual::Copied), vec![]));
}
