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

//! The primary-key fast path must give the same answer as the general
//! planner for every statement shape it accepts.

use stoolap::Database;

fn one_row(db: &Database, sql: &str) -> (Vec<String>, Vec<String>) {
    let mut rows = db.query(sql, ()).unwrap();
    let columns: Vec<String> = rows.columns().iter().map(|c| c.to_string()).collect();
    let row = rows.next().unwrap().unwrap();
    let values = (0..columns.len())
        .map(|i| format!("{:?}", row.get::<stoolap::Value>(i).unwrap()))
        .collect();
    assert!(rows.next().is_none(), "one row expected for {sql}");
    (columns, values)
}

fn seed_with_history(db: &Database) {
    db.execute(
        "CREATE TABLE h (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO h VALUES (1, 'ann', 31)", ())
        .unwrap();
    db.execute("UPDATE h SET age = 99 WHERE id = 1", ())
        .unwrap();
}

#[test]
fn star_pk_lookup_with_as_of_reads_the_historical_row() {
    let db = Database::open("memory://pk_as_of_star").unwrap();
    seed_with_history(&db);
    let (_, now) = one_row(&db, "SELECT * FROM h WHERE id = 1");
    assert_eq!(now[2], "Integer(99)");
    let (_, then) = one_row(&db, "SELECT * FROM h AS OF TRANSACTION 1 WHERE id = 1");
    assert_eq!(
        then[2], "Integer(31)",
        "AS OF must not take the PK fast path"
    );
}
