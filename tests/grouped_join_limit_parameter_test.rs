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

//! A GROUP BY join whose LIMIT is a parameter reads its left side in
//! chunks as one whose LIMIT is the same count written out does

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{join_paths_taken, reset_join_paths};
use stoolap::Database;

fn rows(db: &Database, sql: &str, limit: Option<i64>) -> Vec<String> {
    let result = match limit {
        Some(limit) => db.query(sql, (limit,)),
        None => db.query(sql, ()),
    };
    result
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            format!(
                "{}|{}",
                row.get::<i64>(0).unwrap(),
                row.get::<i64>(1).unwrap()
            )
        })
        .collect()
}

#[test]
fn a_parameter_limit_reads_a_grouped_join_in_chunks_as_a_written_one() {
    let db = Database::open("memory://grouped_join_limit_parameter").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 50 FROM generate_series(1, 200) g",
        "INSERT INTO u SELECT g.value, g.value FROM generate_series(1, 50) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let grouped = "SELECT t.id, COUNT(u.x) FROM t JOIN u ON u.id = t.a GROUP BY t.id LIMIT";
    reset_join_paths();
    let written = rows(&db, &format!("{grouped} 3"), None);
    let written_paths = join_paths_taken();
    reset_join_paths();
    let parameter = rows(&db, &format!("{grouped} $1"), Some(3));
    let parameter_paths = join_paths_taken();
    assert!(
        written_paths.contains(&"grouped_index_join"),
        "the written LIMIT reads in chunks: {written_paths:?}"
    );
    assert_eq!(
        parameter_paths, written_paths,
        "the parameter LIMIT takes the same path"
    );
    assert_eq!(parameter.len(), 3);
    assert_eq!(parameter.len(), written.len());
}
