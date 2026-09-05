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

//! A column named in JOIN ... USING is one column of the joined row, found
//! by its bare name or through either table, in a select list, an
//! aggregate and a GROUP BY alike.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE ux (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE uy (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO ux VALUES (1, 1), (2, 2), (3, 3), (4, NULL), (5, 2)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO uy VALUES (1, 2), (2, 5), (3, 3), (4, 2), (5, NULL)",
        (),
    )
    .unwrap();
    db
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| {
                    r.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

#[test]
fn test_the_using_column_is_found_by_every_name() {
    let db = setup("using_every_name");
    assert_eq!(
        rows(
            &db,
            "SELECT a, ux.a, uy.a FROM ux JOIN uy USING (a) ORDER BY ux.id, uy.id"
        ),
        ["2,2,2", "2,2,2", "3,3,3", "2,2,2", "2,2,2"]
    );
}

#[test]
fn test_the_using_column_is_aggregated_and_grouped() {
    let db = setup("using_aggregated");
    assert_eq!(
        rows(
            &db,
            "SELECT SUM(a), SUM(ux.a), MAX(a), COUNT(*) FROM ux JOIN uy USING (a)"
        ),
        ["11,11,3,5"]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT a, COUNT(*) FROM ux JOIN uy USING (a) GROUP BY a ORDER BY a"
        ),
        ["2,4", "3,1"]
    );
    assert_eq!(
        rows(&db, "SELECT SUM(a) FROM ux LEFT JOIN uy USING (a)"),
        ["12"]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT ux.id, uy.id FROM ux JOIN uy USING (a) WHERE a = 3"
        ),
        ["3,3"]
    );
}
