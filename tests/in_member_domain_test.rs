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

//! An IN or NOT IN over a subquery answers on a primary key or an indexed
//! column as on a plain column, whatever kind of value the subquery gives

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| e.to_string())?.get::<i64>(0).unwrap());
    }
    out.sort_unstable();
    Ok(out)
}

fn create(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, x TEXT, b BOOLEAN, ts TIMESTAMP, s TEXT)",
        "CREATE TABLE p (rid INTEGER PRIMARY KEY, id INTEGER, k INTEGER, x TEXT, b BOOLEAN, ts TIMESTAMP, s TEXT)",
        "INSERT INTO t VALUES (1, 1, '1', true, '2024-01-01 00:00:00', 'a'), \
         (2, 2, 'abc', false, '2024-01-02 00:00:00', 'b'), (3, 3, '3', true, '2024-01-03 00:00:00', 'c')",
        "INSERT INTO p SELECT id + 10, id, k, x, b, ts, s FROM t",
        "CREATE INDEX t_k ON t(k)",
        "CREATE INDEX t_x ON t(x)",
        "CREATE INDEX t_b ON t(b)",
        "CREATE INDEX t_ts ON t(ts)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, i INTEGER, t TEXT, tn TEXT, tb TEXT, tt TEXT, \
         f FLOAT, g FLOAT, b BOOLEAN, ts TIMESTAMP, j JSON)",
        "INSERT INTO m VALUES (1, 2, 'abc', '2', 'true', '2024-01-02 00:00:00', 2.0, 2.5, true, \
         '2024-01-02 00:00:00', '{\"a\":1}')",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn check(db: &Database, phase: &str) {
    let mut failed = Vec::new();
    for column in ["id", "k", "x", "b", "ts"] {
        for member in ["i", "t", "tn", "tb", "tt", "f", "g", "b", "ts", "j"] {
            for not in ["", "NOT "] {
                for tail in ["", " AND s || 'x' <> 'zx'", " LIMIT 5"] {
                    let sql = format!(
                        "SELECT id FROM {{t}} WHERE {column} {not}IN (SELECT {member} FROM m){tail}"
                    );
                    let indexed = ids(db, &sql.replace("{t}", "t"));
                    let plain = ids(db, &sql.replace("{t}", "p"));
                    if indexed != plain {
                        failed.push(format!("{phase}: {sql}: {indexed:?}, plain {plain:?}"));
                    }
                }
            }
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn a_subquery_member_of_any_kind_answers_as_on_a_plain_column_in_memory() {
    let db = Database::open("memory://a_subquery_member_of_any_kind_answers").unwrap();
    create(&db);
    check(&db, "memory");
}

#[test]
fn a_subquery_member_of_any_kind_answers_as_on_a_plain_column_hot_and_sealed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    create(&db);
    check(&db, "hot");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_subquery_whose_members_are_read_runs_once() {
    let db = Database::open("memory://a_subquery_whose_members_are_read_runs_once").unwrap();
    create(&db);
    let mut failed = Vec::new();
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE id NOT IN \
             (SELECT CASE WHEN RANDOM() < 2 THEN TRUE END FROM m) AND s || 'x' <> 'zx'",
            vec![1, 2, 3],
        ),
        (
            "SELECT id FROM t WHERE id NOT IN \
             (SELECT i FROM m WHERE RANDOM() < 0) AND s || 'x' <> 'zx'",
            vec![1, 2, 3],
        ),
        (
            "SELECT id FROM t WHERE k NOT IN \
             (SELECT i FROM m WHERE RANDOM() < 2) AND s || 'x' <> 'zx'",
            vec![1, 3],
        ),
        (
            "SELECT id FROM t WHERE id IN \
             (SELECT CASE WHEN RANDOM() < 2 THEN TRUE END FROM m) AND s || 'x' <> 'zx'",
            vec![],
        ),
    ] {
        let runs = stoolap::test_failpoints::in_subquery_runs();
        assert_eq!(ids(&db, sql), Ok(want), "{sql}");
        let ran = stoolap::test_failpoints::in_subquery_runs() - runs;
        if ran != 1 {
            failed.push(format!("{sql}: the subquery ran {ran} times"));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}
