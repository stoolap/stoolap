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

//! A subquery block computing a window function runs as one statement for
//! each enclosing row, its windows over the rows that row lets it read

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO u VALUES (1, 5), (2, 25), (3, 15)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<i64>> {
    let read = |rows: stoolap::Rows| -> Vec<Vec<i64>> {
        rows.map(|row| {
            let row = row.unwrap();
            (0..row.len()).map(|i| row.get::<i64>(i).unwrap()).collect()
        })
        .collect()
    };
    let direct = read(db.query(sql, ()).unwrap());
    let stmt = db.prepare(sql).unwrap();
    for run in 0..2 {
        assert_eq!(
            read(stmt.query(()).unwrap()),
            direct,
            "{sql}: prepared run {run}"
        );
    }
    direct
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

#[test]
fn a_windowed_block_answers_over_the_rows_its_enclosing_row_lets_it_read() {
    let db = setup("windowed_blocks_answers");
    let cases: [(&str, Vec<Vec<i64>>); 4] = [
        (
            "SELECT t.id, (SELECT ROW_NUMBER() OVER (ORDER BY u.id) * 100 + u.v FROM u \
             WHERE u.id <= t.id ORDER BY u.id DESC LIMIT 1) FROM t ORDER BY t.id",
            vec![vec![1, 105], vec![2, 225], vec![3, 315]],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT ROW_NUMBER() OVER () FROM u WHERE u.v > t.a) \
             ORDER BY id",
            vec![vec![1], vec![2]],
        ),
        (
            "SELECT id FROM t WHERE t.a + 3 IN \
             (SELECT ROW_NUMBER() OVER (ORDER BY u.v) * 10 + u.id FROM u WHERE u.v < t.a) \
             ORDER BY id",
            vec![vec![2]],
        ),
        (
            "SELECT t.id, (SELECT SUM(u.v) OVER () FROM (SELECT * FROM u) AS u \
             WHERE u.v < t.a AND u.id > (SELECT MIN(id) FROM u) - 1 LIMIT 1) FROM t ORDER BY t.id",
            vec![vec![1, 5], vec![2, 20], vec![3, 45]],
        ),
    ];
    for (sql, want) in cases {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        assert_eq!(rows(&db, sql), want, "{sql}");
    }
}
