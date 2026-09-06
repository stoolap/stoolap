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

//! ORDER BY, LIMIT and OFFSET at the end of a set operation apply to the
//! whole of it, not to its first branch.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE y (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, 1, 10), (2, 2, 20), (3, 1, 30)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO y VALUES (1, 1, 5), (2, 1, 50), (3, 2, 7), (4, 3, 9)",
        (),
    )
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
fn test_order_by_and_limit_apply_to_the_whole_union() {
    let db = setup("set_op_order_limit");
    assert_eq!(
        ints(
            &db,
            "SELECT a FROM x WHERE a > 0 UNION SELECT a FROM y WHERE a > 0 ORDER BY a DESC LIMIT 2"
        ),
        [3, 2]
    );
    assert_eq!(
        ints(
            &db,
            "SELECT a FROM x WHERE a > 0 INTERSECT SELECT a FROM y WHERE a > 0 ORDER BY a DESC LIMIT 3"
        ),
        [2, 1]
    );
    assert_eq!(
        ints(
            &db,
            "SELECT id FROM x WHERE a = 1 UNION ALL SELECT id FROM y WHERE a = 1 ORDER BY id LIMIT 3"
        ),
        [1, 1, 2]
    );
}

#[test]
fn test_offset_skips_rows_of_the_whole_union() {
    let db = setup("set_op_offset");
    assert_eq!(
        ints(
            &db,
            "SELECT id FROM x UNION ALL SELECT id FROM y ORDER BY id LIMIT 10 OFFSET 5"
        ),
        [3, 4]
    );
    assert_eq!(
        ints(
            &db,
            "SELECT COUNT(*) FROM (SELECT id FROM x UNION ALL SELECT id FROM y LIMIT 10 OFFSET 5)"
        ),
        [2]
    );
    assert_eq!(
        ints(
            &db,
            "SELECT COUNT(*) FROM (SELECT id FROM x UNION ALL SELECT id FROM y ORDER BY id LIMIT 2)"
        ),
        [2]
    );
}

#[test]
fn test_union_all_limit_and_offset_without_order_by() {
    let db = Database::open("memory://set_operation_union_all_bound").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let rows: Vec<String> = (1..=1000).map(|i| format!("({i}, {i})")).collect();
    db.execute(&format!("INSERT INTO a VALUES {}", rows.join(", ")), ())
        .unwrap();
    db.execute("INSERT INTO b VALUES (1, -1), (2, -2), (3, -3)", ())
        .unwrap();
    // The first branch alone covers LIMIT + OFFSET, so its rows are the answer
    assert_eq!(
        ints(
            &db,
            "SELECT v FROM a UNION ALL SELECT v FROM b LIMIT 3 OFFSET 2"
        ),
        [3, 4, 5]
    );
    // The first branch runs out and the second one fills the rest
    assert_eq!(
        ints(
            &db,
            "SELECT v FROM a WHERE id <= 2 UNION ALL SELECT v FROM b LIMIT 3 OFFSET 1"
        ),
        [2, -1, -2]
    );
    assert_eq!(
        ints(
            &db,
            "SELECT COUNT(*) FROM (SELECT v FROM a UNION ALL SELECT v FROM b LIMIT 1500)"
        ),
        [1003]
    );
}

#[test]
fn test_a_limit_and_offset_past_i64_do_not_fail_the_first_branch() {
    let db = Database::open("memory://set_operation_limit_overflow").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO a VALUES (1)", ()).unwrap();
    db.execute("INSERT INTO b VALUES (2)", ()).unwrap();
    assert_eq!(
        ints(
            &db,
            "SELECT id FROM a UNION ALL SELECT id FROM b LIMIT 9223372036854775807 OFFSET 1"
        ),
        [2]
    );
}
