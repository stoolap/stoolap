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

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t VALUES (1, 5), (2, 1), (3, 7)",
        "INSERT INTO u VALUES (1, 2), (2, 6)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap().trim().to_string())
        .find(|line| line.starts_with("Prepared:"))
        .unwrap_or_default()
}

#[test]
fn a_window_no_function_reads_is_never_read() {
    let db = setup("unused_window_never_read");
    for window in [
        "WINDOW w AS ()",
        "WINDOW w AS (ORDER BY nosuch)",
        "WINDOW w AS (PARTITION BY 1 / 0)",
    ] {
        let sql = format!("SELECT id FROM t WHERE k > (SELECT MIN(v) FROM u) {window} ORDER BY id");
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        assert_eq!(ids(&db, &sql), vec![1, 3], "{sql}");
        let plain = format!("SELECT id FROM t WHERE k > 2 {window} ORDER BY id");
        assert_eq!(ids(&db, &plain), vec![1, 3], "{plain}");
    }
}

#[test]
fn a_window_a_function_reads_is_read() {
    let db = setup("unused_window_read");
    let sql = "SELECT id, ROW_NUMBER() OVER w FROM t WHERE k > (SELECT MIN(v) FROM u) \
               WINDOW w AS (ORDER BY id DESC) ORDER BY id";
    let rows: Vec<(i64, i64)> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect();
    assert_eq!(rows, vec![(1, 2), (3, 1)], "{sql}");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_statement_with_an_unused_window_runs_no_general_path_mechanism() {
    use stoolap::test_failpoints::general_path_total;
    let db = setup("unused_window_general_path");
    for sql in [
        "SELECT id FROM t WHERE k > (SELECT MIN(v) FROM u) WINDOW w AS () ORDER BY id",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.v = t.k + 1) WINDOW w AS () \
         ORDER BY id",
    ] {
        let before = general_path_total();
        ids(&db, sql);
        assert_eq!(general_path_total(), before, "{sql}");
    }
}
