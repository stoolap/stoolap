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

//! A join without a subquery runs on its prepared blocks and answers as a
//! join did before it ran there

use stoolap::Database;

const SETUP: &[&str] = &[
    "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER)",
    "CREATE TABLE r (id INTEGER PRIMARY KEY, v INTEGER)",
    "INSERT INTO l VALUES (1, 1), (2, 2), (3, 3)",
    "INSERT INTO r VALUES (1, 2), (2, 3), (3, 4)",
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, b INTEGER)",
    "INSERT INTO t VALUES (1, 10), (2, 20)",
    "INSERT INTO u VALUES (5, 30), (6, 40)",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://plain_join_blocks_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn ids(db: &Database, sql: &str, param: i64) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, (param,)).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| e.to_string())?.get::<i64>(0).unwrap());
    }
    out.sort_unstable();
    Ok(out)
}

#[test]
fn a_value_the_run_computes_is_read_by_a_condition_the_scan_does_not_take() {
    let dir = tempfile::tempdir().unwrap();
    for (context, db) in [("hot", hot("computed")), ("sealed", sealed(&dir))] {
        let sql = "SELECT l.id FROM l JOIN r ON l.k = r.id WHERE r.v > ABS($1) AND r.v % 2 = 0";
        let stmt = db.prepare(sql).unwrap();
        for _ in 0..2 {
            let got: Vec<i64> = stmt
                .query((-1,))
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(0).unwrap())
                .collect();
            assert_eq!(got, [1, 3], "{context}: prepared");
            assert_eq!(ids(&db, sql, -3), Ok(vec![3]), "{context}: text");
        }
    }
}

#[test]
fn a_computed_value_answers_as_its_literal_in_every_clause_and_join() {
    let shapes = [
        "SELECT l.id FROM l JOIN r ON l.k = r.id AND r.v > ABS($1)",
        "SELECT l.id FROM l LEFT JOIN r ON l.k = r.id AND r.v > ABS($1) WHERE l.id > 0",
        "SELECT l.id FROM l JOIN r ON l.k = r.id WHERE l.k + r.v > ABS($1) + 3",
        "SELECT l.id FROM l JOIN r ON l.k = r.id WHERE r.v % 2 = 0 AND r.v > ABS($1) LIMIT 5",
        "SELECT l.id FROM l JOIN r ON l.k = r.v WHERE r.v % 2 = 0 AND r.v > ABS($1)",
        "SELECT l.id FROM l JOIN r ON l.k < r.v WHERE r.v - l.k > ABS($1)",
        "SELECT l.id FROM l LEFT JOIN r ON l.k = r.v WHERE COALESCE(r.v, 9) % 2 = 1 AND l.k >= ABS($1)",
    ];
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (context, db) in [("hot", hot("matrix")), ("sealed", sealed(&dir))] {
        for sql in shapes {
            let control = sql.replace("ABS($1)", "1");
            let (got, want) = (ids(&db, sql, -1), ids(&db, &control, 0));
            if got.is_err() || got != want {
                wrong.push(format!("{context}: {sql}: {got:?}, want {want:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_select_alias_for_a_name_both_sources_have_reads_the_left_one() {
    let db = hot("alias");
    let sql = "SELECT id AS x FROM t JOIN u ON t.id = u.b - 29 WHERE x > $1";
    for _ in 0..2 {
        assert_eq!(ids(&db, sql, 0), Ok(vec![1]));
    }
}
