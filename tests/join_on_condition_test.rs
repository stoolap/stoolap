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

/// Two left rows holding the same values are two rows, and each keeps a
/// row of its own when the condition turns its candidates away
#[test]
fn test_left_rows_that_hold_the_same_values_stay_apart() {
    let db = Database::open("memory://join_on_condition_twins").unwrap();
    db.execute("CREATE TABLE l (k INTEGER)", ()).unwrap();
    db.execute("CREATE TABLE r (k INTEGER, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO l VALUES (1), (1), (2)", ())
        .unwrap();
    db.execute("INSERT INTO r VALUES (1, 0), (1, 9), (3, 7)", ())
        .unwrap();

    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM l LEFT JOIN r ON l.k = r.k AND r.v > 900"
        ),
        3,
        "both rows holding a 1 are turned away and both are kept"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT l.k, r.v FROM l LEFT JOIN r ON l.k = r.k AND r.v > 100 ORDER BY l.k"
        ),
        ["1,NULL", "1,NULL", "2,NULL"]
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM l LEFT JOIN r ON l.k = r.k AND r.v > 1"
        ),
        3,
        "each 1 keeps its one match and the 2 keeps a row of its own"
    );
}

/// A full join returns the rows of both sides and nothing besides
#[test]
fn test_full_join_adds_no_row_of_its_own() {
    let db = Database::open("memory://join_on_condition_full").unwrap();
    db.execute("CREATE TABLE l (k INTEGER)", ()).unwrap();
    db.execute("CREATE TABLE r (k INTEGER, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO l VALUES (1), (2)", ()).unwrap();
    db.execute("INSERT INTO r VALUES (1, 0), (3, 7)", ())
        .unwrap();

    let out = rows(
        &db,
        "SELECT l.k, r.k, r.v FROM l FULL OUTER JOIN r ON l.k = r.k AND r.v > 0",
    );
    assert_eq!(out.len(), 4, "two left rows and two right rows: {out:?}");
    assert!(
        !out.contains(&"NULL,NULL,NULL".to_string()),
        "a row of nothing but NULLs belongs to neither side: {out:?}"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM l FULL OUTER JOIN r ON l.k = r.k AND 1 = 0"
        ),
        4
    );
}

/// Sorted inputs take a different path through the join and honour the
/// condition on it just the same
#[test]
fn test_sorted_inputs_honour_the_condition() {
    let db = Database::open("memory://join_on_condition_sorted").unwrap();
    db.execute("CREATE TABLE a (k INTEGER, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (k INTEGER, v INTEGER)", ())
        .unwrap();
    for i in 1i64..=400 {
        db.execute("INSERT INTO a VALUES ($1, $2)", (i, i * 10))
            .unwrap();
        db.execute("INSERT INTO b VALUES ($1, $2)", (i, i % 4))
            .unwrap();
    }

    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM a JOIN b ON a.k = b.k AND b.v = 2"
        ),
        100,
        "every fourth row meets the condition"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM a LEFT JOIN b ON a.k = b.k AND b.v = 2"
        ),
        400
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(b.v) FROM a LEFT JOIN b ON a.k = b.k AND b.v = 2"
        ),
        100,
        "the rows the condition turned away are kept without a right side"
    );
}

/// A subquery in the ON clause is read once and the join compares each pair
/// against what it returned
#[test]
fn test_a_subquery_in_the_join_condition() {
    let db = Database::open("memory://join_on_subquery").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, p_id INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO p VALUES (1), (2), (3)", ())
        .unwrap();
    db.execute(
        "INSERT INTO r VALUES (1,1,100), (2,1,200), (3,2,300), (4,3,400), (5,3,500)",
        (),
    )
    .unwrap();

    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM p JOIN r ON r.p_id = p.id AND r.v > (SELECT AVG(v) FROM r)"
        ),
        2
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM p LEFT JOIN r ON r.p_id = p.id AND r.v > (SELECT AVG(v) FROM r)"
        ),
        4,
        "two matches and two rows the condition turned away"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM p JOIN r ON r.p_id = p.id WHERE r.v > (SELECT AVG(v) FROM r)"
        ),
        2,
        "the same condition in the WHERE clause"
    );
}

/// A LEFT JOIN keeps its left rows whatever the right side is read from
#[test]
fn test_left_join_against_a_derived_table() {
    let db = Database::open("memory://join_left_derived").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, p_id INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO p VALUES (1,10),(2,20),(3,30)", ())
        .unwrap();
    db.execute("INSERT INTO r VALUES (1,1,100),(2,2,200)", ())
        .unwrap();

    assert_eq!(
        rows(
            &db,
            "SELECT p.id, s.t FROM p LEFT JOIN (SELECT p_id, SUM(v) AS t FROM r GROUP BY p_id) s ON s.p_id = p.id ORDER BY p.id"
        ),
        ["1,100", "2,200", "3,NULL"]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT p.id, s.v FROM p LEFT JOIN (SELECT p_id, v FROM r) s ON s.p_id = p.id ORDER BY p.id"
        ),
        ["1,100", "2,200", "3,NULL"]
    );
    assert_eq!(
        rows(
            &db,
            "WITH s AS (SELECT p_id, SUM(v) AS t FROM r GROUP BY p_id) SELECT p.id, s.t FROM p LEFT JOIN s ON s.p_id = p.id ORDER BY p.id"
        ),
        ["1,100", "2,200", "3,NULL"]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT * FROM p LEFT JOIN (SELECT p_id, v FROM r) s ON s.p_id = p.id ORDER BY p.id"
        ),
        ["1,10,1,100", "2,20,2,200", "3,30,NULL,NULL"],
        "the left side comes first"
    );
}

/// An index on the key offers a shorter road to the pairs, and the rest of
/// the ON clause is still asked of each one it returns
#[test]
fn test_the_condition_holds_when_an_index_answers_the_key() {
    for indexed in [false, true] {
        let db = Database::open(&format!(
            "memory://join_on_indexed_{}",
            if indexed { "yes" } else { "no" }
        ))
        .unwrap();
        db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute(
            "CREATE TABLE r (id INTEGER PRIMARY KEY, p_id INTEGER, v INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO p VALUES (1), (2), (3)", ())
            .unwrap();
        db.execute(
            "INSERT INTO r VALUES (1,1,100), (2,1,200), (3,2,300), (4,3,400), (5,3,500)",
            (),
        )
        .unwrap();
        if indexed {
            db.execute("CREATE INDEX rp ON r (p_id)", ()).unwrap();
        }

        assert_eq!(
            rows(
                &db,
                "SELECT p.id FROM p JOIN r ON r.p_id = p.id AND r.v > 300 ORDER BY p.id"
            ),
            ["3", "3"],
            "indexed: {indexed}"
        );
        assert_eq!(
            rows(
                &db,
                "SELECT p.id FROM p JOIN r ON r.p_id = p.id AND r.v > (SELECT AVG(v) FROM r) ORDER BY p.id"
            ),
            ["3", "3"],
            "indexed: {indexed}"
        );
        assert_eq!(
            rows(
                &db,
                "SELECT p.id, r.v FROM p LEFT JOIN r ON r.p_id = p.id AND r.v > 300 ORDER BY p.id, r.v"
            ),
            ["1,NULL", "2,NULL", "3,400", "3,500"],
            "indexed: {indexed}"
        );
        assert_eq!(
            rows(
                &db,
                "SELECT p.id, (SELECT COUNT(*) FROM r WHERE r.p_id = p.id) FROM p ORDER BY p.id"
            ),
            ["1,2", "2,1", "3,2"],
            "a correlated count still reads the same, indexed: {indexed}"
        );
    }
}

/// The ON clause reads the transaction it runs in, whether or not an index
/// answers the key
#[test]
fn test_the_condition_sees_the_transaction() {
    for indexed in [false, true] {
        let db = Database::open(&format!(
            "memory://join_on_txn_{}",
            if indexed { "yes" } else { "no" }
        ))
        .unwrap();
        db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY, p_id INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO p VALUES (1), (2)", ()).unwrap();
        db.execute("INSERT INTO r VALUES (1, 1), (2, 2)", ())
            .unwrap();
        if indexed {
            db.execute("CREATE INDEX rp ON r (p_id)", ()).unwrap();
        }

        db.execute("BEGIN", ()).unwrap();
        assert_eq!(
            rows(
                &db,
                "SELECT p.id FROM p JOIN r ON r.p_id = p.id AND CURRENT_TRANSACTION_ID() IS NOT NULL ORDER BY p.id"
            ),
            ["1", "2"],
            "indexed: {indexed}"
        );
        db.execute("ROLLBACK", ()).unwrap();
    }
}
