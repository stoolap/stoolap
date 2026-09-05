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

//! A WITH clause may stand before UPDATE and DELETE; the statement's
//! subqueries read the CTEs it defines.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE y (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, 1, ''), (2, 2, ''), (3, 1, ''), (4, 3, '')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO y VALUES (1, 1, 5), (2, 1, NULL), (3, 2, 7), (4, 1, 9)",
        (),
    )
    .unwrap();
    db
}

fn texts(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect()
}

#[test]
fn test_with_before_update() {
    let db = setup("with_before_update");
    let changed = db
        .execute(
            "WITH hot AS (SELECT a FROM y GROUP BY a HAVING COUNT(*) > 1) UPDATE x SET s = 'hot' WHERE a IN (SELECT a FROM hot)",
            (),
        )
        .unwrap();
    assert_eq!(changed, 2);
    assert_eq!(
        texts(&db, "SELECT s FROM x ORDER BY id"),
        ["hot", "", "hot", ""]
    );
    let changed = db
        .execute(
            "WITH RECURSIVE n(v) AS (SELECT 1 UNION ALL SELECT v + 1 FROM n WHERE v < 2) UPDATE x SET s = 'n' WHERE id IN (SELECT v FROM n)",
            (),
        )
        .unwrap();
    assert_eq!(changed, 2);
    assert_eq!(
        texts(&db, "SELECT s FROM x ORDER BY id"),
        ["n", "n", "hot", ""]
    );
}

#[test]
fn test_with_before_delete() {
    let db = setup("with_before_delete");
    let removed = db
        .execute(
            "WITH gone AS (SELECT id FROM y WHERE b IS NULL) DELETE FROM y WHERE id IN (SELECT id FROM gone)",
            (),
        )
        .unwrap();
    assert_eq!(removed, 1);
    let ids: Vec<i64> = db
        .query("SELECT id FROM y ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(ids, [1, 3, 4]);
}
