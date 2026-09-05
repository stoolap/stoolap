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

//! An outer join keeps a row of the side it preserves only where the whole
//! ON condition matched nothing for it, and one such row however many
//! candidates the condition turned away.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, t_id INTEGER, w INTEGER)",
        (),
    )
    .unwrap();
    for (id, g) in [
        (1i64, "a"),
        (2, "a"),
        (3, "b"),
        (4, "b"),
        (5, "c"),
        (6, "c"),
    ] {
        db.execute("INSERT INTO t VALUES ($1, $2)", (id, g))
            .unwrap();
    }
    // t 1 has two, t 2 and t 5 have one, the last belongs to nobody
    for (id, t_id, w) in [
        (1i64, Some(1i64), 100i64),
        (2, Some(1), 200),
        (3, Some(2), 300),
        (4, Some(5), 400),
        (5, None, 500),
    ] {
        db.execute("INSERT INTO u VALUES ($1, $2, $3)", (id, t_id, w))
            .unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

/// A left row that the condition matched keeps its matches and gains no
/// row of its own, however many candidates it turned away
#[test]
fn test_left_join_with_a_predicate_beside_the_key() {
    let db = setup("join_on_condition_left");
    assert_eq!(
        rows(
            &db,
            "SELECT t.id, u.id FROM t LEFT JOIN u ON u.t_id = t.id AND u.w > 150 ORDER BY t.id, u.id"
        ),
        [
            "1,2", "2,3", "3,NULL", "4,NULL", "5,4", "6,NULL"
        ],
        "t 1 turned one candidate away and keeps only the match"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t LEFT JOIN u ON u.t_id = t.id AND u.w > 150"
        ),
        6
    );
}

/// A condition nothing satisfies leaves one row per left row, not one per
/// candidate it turned away
#[test]
fn test_left_join_with_a_condition_nothing_meets() {
    let db = setup("join_on_condition_false");
    assert_eq!(
        rows(
            &db,
            "SELECT t.id, u.id FROM t LEFT JOIN u ON u.t_id = t.id AND 1 = 0 ORDER BY t.id"
        ),
        ["1,NULL", "2,NULL", "3,NULL", "4,NULL", "5,NULL", "6,NULL"]
    );
}

/// A predicate on the preserved side reads the same way
#[test]
fn test_left_join_with_a_predicate_on_the_left() {
    let db = setup("join_on_condition_left_side");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t LEFT JOIN u ON u.t_id = t.id AND t.id > 3"
        ),
        6
    );
}

/// What the condition keeps is what an inner join would have kept
#[test]
fn test_inner_join_keeps_only_the_matches() {
    let db = setup("join_on_condition_inner");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t JOIN u ON u.t_id = t.id AND u.w > 150"
        ),
        3
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM t LEFT JOIN u ON u.t_id = t.id AND u.w > 50"
        ),
        7,
        "a condition every candidate meets changes nothing"
    );
}

/// The right side of a right join is preserved the same way
#[test]
fn test_right_join_with_a_predicate_beside_the_key() {
    let db = setup("join_on_condition_right");
    assert_eq!(
        rows(
            &db,
            "SELECT t.id, u.id FROM u RIGHT JOIN t ON u.t_id = t.id AND u.w > 150 ORDER BY t.id, u.id"
        ),
        ["1,2", "2,3", "3,NULL", "4,NULL", "5,4", "6,NULL"]
    );
}
