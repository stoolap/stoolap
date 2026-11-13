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

//! A grouped statement's output blocks know how many groups ask them, so
//! a correlated block over a large indexed table reads only each group's
//! rows, not the table whole

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::inner_table_reads;
use stoolap::Database;

/// Five ages over 1,000 users, 2,000 orders indexed by user
fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER)",
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)",
        "CREATE INDEX orders_user ON orders (user_id)",
        "INSERT INTO users SELECT value, value % 5 + 18 FROM generate_series(1, 1000)",
        "INSERT INTO orders SELECT value, value % 1000 + 1 FROM generate_series(1, 2000)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

#[test]
fn a_grouped_output_block_reads_only_its_groups_rows() {
    let block = "(SELECT COUNT(*) FROM orders o WHERE o.user_id = u.age)";
    let mut wrong = Vec::new();
    for (at, sql) in [
        format!("SELECT u.age, {block} FROM users u GROUP BY u.age ORDER BY u.age"),
        format!("SELECT u.age, 2 FROM users u GROUP BY u.age HAVING {block} = 2 ORDER BY u.age"),
        format!("SELECT u.age, {block} FROM users u GROUP BY u.age ORDER BY {block}, u.age"),
    ]
    .into_iter()
    .enumerate()
    {
        let db = tables(&format!("grouped_output_rows_{at}"));
        let before = inner_table_reads();
        let rows: Vec<(i64, i64)> = db
            .query(&sql, ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
            })
            .collect();
        let whole = inner_table_reads() - before;
        let want = [(18, 2), (19, 2), (20, 2), (21, 2), (22, 2)];
        if rows != want || whole != 0 {
            wrong.push(format!("{sql}: {rows:?}, {whole} whole reads"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
