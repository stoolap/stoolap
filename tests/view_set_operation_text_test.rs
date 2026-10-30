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

//! A set operation's ORDER BY, LIMIT and OFFSET belong to the whole
//! operation in the text a view keeps and in a select item's name

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 3), (2, 1), (3, 2), (4, 5)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 5), (2, 4), (3, 2)", ())
        .unwrap();
    db
}

fn values(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn a_view_over_a_set_operation_with_order_by_limit_or_offset_answers_as_its_query() {
    let db = open("view_set_operation_order");
    for (view, query, expected) in [
        (
            "w_union",
            "SELECT a FROM t UNION SELECT x FROM u ORDER BY 1",
            vec![1, 2, 3, 4, 5],
        ),
        (
            "w_union_all",
            "SELECT a FROM t UNION ALL SELECT x FROM u ORDER BY 1 DESC LIMIT 3",
            vec![5, 5, 4],
        ),
        (
            "w_intersect",
            "SELECT a FROM t INTERSECT SELECT x FROM u ORDER BY 1 LIMIT 1 OFFSET 1",
            vec![5],
        ),
        (
            "w_except",
            "SELECT a FROM t EXCEPT SELECT x FROM u ORDER BY 1 OFFSET 1",
            vec![3],
        ),
        (
            "w_precedence",
            "SELECT a FROM t UNION SELECT x FROM u INTERSECT SELECT a FROM t ORDER BY 1 DESC",
            vec![5, 3, 2, 1],
        ),
    ] {
        db.execute(&format!("CREATE VIEW {view} AS {query}"), ())
            .unwrap_or_else(|e| panic!("CREATE VIEW {view} AS {query}: {e}"));
        assert_eq!(values(&db, query), expected, "{query}");
        assert_eq!(
            values(&db, &format!("SELECT * FROM {view}")),
            expected,
            "the view answers as its query: {query}"
        );
    }
}

#[test]
fn a_select_item_holding_a_set_operation_is_named_with_its_order_by_and_limit_last() {
    let db = open("select_item_set_operation_name");
    let rows = db
        .query(
            "SELECT (SELECT MAX(z) FROM (SELECT a AS z FROM t UNION SELECT x FROM u ORDER BY 1 LIMIT 2) s)",
            (),
        )
        .unwrap();
    assert_eq!(
        rows.columns(),
        ["(SELECT MAX(z) FROM (SELECT a AS z FROM t UNION SELECT x FROM u ORDER BY 1 ASC LIMIT 2) AS s)"]
    );
    let answers: Vec<i64> = rows
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(answers, vec![2]);
}
