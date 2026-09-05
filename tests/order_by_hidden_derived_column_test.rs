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

//! ORDER BY may name a column of a view, a derived table or a CTE that the
//! select list leaves out; the sort still reads it.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE hx (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO hx VALUES (1, 1, 10), (2, 1, 11), (3, 2, 50), (4, 3, 30), (5, NULL, 40), (6, 3, 5)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE VIEW va AS SELECT a, COUNT(*) AS c, SUM(b) AS sb FROM hx GROUP BY a",
        (),
    )
    .unwrap();
    db.execute("CREATE VIEW vp AS SELECT id, a, b * 2 AS d FROM hx", ())
        .unwrap();
    db
}

fn ints(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn test_a_view_column_outside_the_select_list_sorts() {
    let db = setup("hidden_view_column");
    // sums: a = 2 gives 50, a = 3 gives 35, a = 1 gives 21
    assert_eq!(
        ints(
            &db,
            "SELECT a FROM va WHERE a IS NOT NULL ORDER BY sb DESC, a LIMIT 3"
        ),
        [2, 3, 1]
    );
    assert_eq!(
        ints(&db, "SELECT id FROM vp ORDER BY d DESC, id"),
        [3, 5, 4, 2, 1, 6]
    );
    assert_eq!(
        ints(
            &db,
            "SELECT id FROM vp WHERE a IS NOT NULL ORDER BY d, id LIMIT 2"
        ),
        [6, 1]
    );
    let columns: Vec<String> = db
        .query("SELECT a FROM va ORDER BY sb DESC", ())
        .unwrap()
        .columns()
        .to_vec();
    assert_eq!(columns, ["a"]);
}

#[test]
fn test_a_derived_table_or_cte_column_outside_the_select_list_sorts() {
    let db = setup("hidden_derived_column");
    assert_eq!(
        ints(
            &db,
            "SELECT a FROM (SELECT a, SUM(b) AS sb FROM hx GROUP BY a) t WHERE a IS NOT NULL ORDER BY sb DESC, a"
        ),
        [2, 3, 1]
    );
    assert_eq!(
        ints(
            &db,
            "WITH t AS (SELECT a, SUM(b) AS sb FROM hx GROUP BY a) SELECT a FROM t WHERE a IS NOT NULL ORDER BY sb DESC, a"
        ),
        [2, 3, 1]
    );
    assert_eq!(
        ints(
            &db,
            "SELECT id FROM (SELECT id, a, b FROM hx) t ORDER BY t.b DESC, id LIMIT 2"
        ),
        [3, 5]
    );
    assert_eq!(
        ints(
            &db,
            "SELECT DISTINCT a FROM (SELECT id, a, b FROM hx) t WHERE a IS NOT NULL ORDER BY a DESC"
        ),
        [3, 2, 1]
    );
}
