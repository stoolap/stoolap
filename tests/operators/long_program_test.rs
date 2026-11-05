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

//! An expression compiled into more ops than 16 bits count jumps to where
//! it was compiled to

use stoolap::Database;

const BRANCHES: i64 = 30_000;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 5), (2, 29999), (3, 40000)", ())
        .unwrap();
    db
}

fn first_column(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn a_case_of_many_branches_answers() {
    let db = setup("long_program_case");
    let branches: Vec<String> = (1..=BRANCHES)
        .map(|i| format!("WHEN a = {i} THEN {i}"))
        .collect();
    let case = format!("CASE {} ELSE -1 END", branches.join(" "));
    assert_eq!(
        first_column(&db, &format!("SELECT {case} FROM t ORDER BY id")),
        vec![5, 29999, -1],
        "select list"
    );
    assert_eq!(
        first_column(
            &db,
            &format!("SELECT id FROM t WHERE {case} > 10 ORDER BY id")
        ),
        vec![2],
        "WHERE"
    );
}
