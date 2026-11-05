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

//! An index join looks up one key equality and applies the rest of the ON
//! clause to every pair, and it reads the table the statement names: the
//! table as of the time asked, never a table a CTE of the same name hides

use stoolap::Database;

type Pairs = Vec<(i64, Option<i64>)>;

fn pairs(db: &Database, sql: &str) -> Pairs {
    let mut rows: Pairs = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect();
    rows.sort();
    rows
}

fn check(db: &Database, cases: &[(&str, Pairs)]) {
    let wrong: Vec<String> = cases
        .iter()
        .filter_map(|(sql, want)| {
            let got = pairs(db, sql);
            (&got != want).then(|| format!("{sql}\n  got {got:?}, want {want:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// t has a PK on id and an index on y; r drives the CTEs
fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, flag INTEGER, y INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_y ON t(y)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 0, 10), (2, 1, 20)", ())
        .unwrap();
    db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO r VALUES (1, 10), (2, 99)", ())
        .unwrap();
    db
}

#[test]
fn an_index_join_applies_the_whole_on_clause() {
    let db = tables("index_join_rule_on");
    check(
        &db,
        &[
            (
                "SELECT r.id, t.y FROM r JOIN t ON r.id = t.id AND t.flag = 1",
                vec![(2, Some(20))],
            ),
            (
                "WITH c(id) AS (SELECT 1 UNION ALL SELECT 2) \
                 SELECT c.id, t.y FROM c JOIN t ON c.id = t.id AND t.flag = 1",
                vec![(2, Some(20))],
            ),
            (
                "WITH c(id) AS (SELECT id FROM r) \
                 SELECT c.id, t.y FROM c JOIN t ON c.id = t.id AND t.flag = 1",
                vec![(2, Some(20))],
            ),
            (
                "WITH c(id, x) AS (SELECT id, x FROM r) \
                 SELECT c.id, t.y FROM c JOIN t ON c.id = t.id AND c.x = t.y",
                vec![(1, Some(10))],
            ),
            (
                "WITH c(x) AS (SELECT x FROM r) \
                 SELECT c.x, t.y FROM c JOIN t ON c.x = t.y AND t.flag = 1",
                vec![],
            ),
            (
                "WITH c(id) AS (SELECT id FROM r) \
                 SELECT c.id, t.y FROM t JOIN c ON c.id = t.id AND t.flag = 1",
                vec![(2, Some(20))],
            ),
            (
                "WITH c AS (SELECT id FROM r), d AS (SELECT * FROM c) \
                 SELECT d.id, t.y FROM d JOIN t ON d.id = t.id AND t.flag = 1",
                vec![(2, Some(20))],
            ),
        ],
    );
}

#[test]
fn a_grouped_index_join_applies_the_whole_on_clause() {
    let db = Database::open("memory://index_join_rule_grouped").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE b (id INTEGER PRIMARY KEY, aid INTEGER, flag INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX b_aid ON b(aid)", ()).unwrap();
    db.execute("INSERT INTO a VALUES (1), (2), (3)", ())
        .unwrap();
    db.execute(
        "INSERT INTO b VALUES (1, 1, 1), (2, 1, 0), (3, 2, 0), (4, 3, 1), (5, 3, 1)",
        (),
    )
    .unwrap();
    check(
        &db,
        &[
            (
                "SELECT a.id, COUNT(*) FROM a JOIN b ON a.id = b.aid AND b.flag = 1 \
                 GROUP BY a.id LIMIT 5",
                vec![(1, Some(1)), (3, Some(2))],
            ),
            (
                "SELECT a.id, COUNT(b.id) FROM a LEFT JOIN b ON a.id = b.aid AND b.flag = 1 \
                 GROUP BY a.id LIMIT 5",
                vec![(1, Some(1)), (2, Some(0)), (3, Some(2))],
            ),
        ],
    );
}

#[test]
fn a_cte_named_like_a_table_is_joined_instead_of_the_table() {
    let db = tables("index_join_rule_shadow");
    let cte = "WITH t(id, y) AS (SELECT 1, 7) ";
    check(
        &db,
        &[
            (
                &format!("{cte}SELECT r.id, t.y FROM r JOIN t ON r.id = t.id ORDER BY r.id"),
                vec![(1, Some(7))],
            ),
            (
                &format!("{cte}SELECT r.id, t.y FROM r LEFT JOIN t ON r.id = t.id ORDER BY r.id"),
                vec![(1, Some(7)), (2, None)],
            ),
            (
                &format!("{cte}SELECT DISTINCT r.id, t.y FROM r JOIN t ON r.id = t.id"),
                vec![(1, Some(7))],
            ),
        ],
    );
}

#[test]
fn an_index_join_reads_the_table_as_of_the_time_asked() {
    let db = tables("index_join_rule_as_of");
    std::thread::sleep(std::time::Duration::from_millis(20));
    let before: String = db.query_one("SELECT CAST(NOW() AS TEXT)", ()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    db.execute("UPDATE t SET y = 99 WHERE id = 1", ()).unwrap();
    let as_of = format!("AS OF TIMESTAMP '{before}'");
    check(
        &db,
        &[
            (
                &format!("SELECT r.id, t.y FROM r JOIN t {as_of} ON r.id = t.id WHERE r.id = 1"),
                vec![(1, Some(10))],
            ),
            (
                &format!(
                    "WITH c(id) AS (SELECT 1) SELECT c.id, t.y FROM c JOIN t {as_of} ON c.id = t.id"
                ),
                vec![(1, Some(10))],
            ),
            (
                &format!(
                    "SELECT r.id, x.y FROM r JOIN (SELECT * FROM t {as_of}) x ON r.id = x.id \
                     WHERE r.id = 1"
                ),
                vec![(1, Some(10))],
            ),
        ],
    );
}
