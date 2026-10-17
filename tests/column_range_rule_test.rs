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

use stoolap::Database;

fn ints(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// `t (id, a)` with ids 1..=20 and `a = id`, and `u (id, a)` with `a = 21 - id`
fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    for i in 1..=20 {
        db.execute(&format!("INSERT INTO t VALUES ({i}, {i})"), ())
            .unwrap();
        db.execute(&format!("INSERT INTO u VALUES ({i}, {})", 21 - i), ())
            .unwrap();
    }
    db
}

#[test]
fn test_a_keyset_walk_starts_at_the_tightest_bound() {
    let db = tables("range_keyset");
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE id > 5 AND 10 < id ORDER BY id LIMIT 3",
            vec![11, 12, 13],
        ),
        (
            "SELECT id FROM t WHERE id > 5 AND id >= 8 ORDER BY id LIMIT 2",
            vec![8, 9],
        ),
        (
            "SELECT id FROM t WHERE id >= 2.5 ORDER BY id LIMIT 2",
            vec![3, 4],
        ),
        (
            "SELECT id FROM t WHERE id > 5 AND id > 10 ORDER BY id DESC LIMIT 2",
            vec![20, 19],
        ),
    ] {
        assert_eq!(ints(&db, sql), want, "{sql}");
    }
}

#[test]
fn test_no_key_lies_above_the_largest_one() {
    let db = Database::open("memory://range_pk_max").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (9223372036854775807, 1), (1, 1)", ())
        .unwrap();
    let sql = "DELETE FROM t WHERE id > 9223372036854775807 AND id <= 9223372036854775807";
    assert_eq!(db.execute(sql, ()).unwrap(), 0, "{sql}");
    assert_eq!(ints(&db, "SELECT COUNT(*) FROM t"), vec![2]);
}

#[test]
fn test_an_outer_conjunct_beside_an_inner_one_of_the_same_name_is_kept() {
    let db = tables("range_outer_name");
    assert_eq!(
        ints(
            &db,
            "SELECT t.id FROM t WHERE t.id IN (SELECT u.id FROM u WHERE a > 15 AND t.a > 5) \
             ORDER BY t.id"
        ),
        Vec::<i64>::new(),
        "u.a > 15 holds for ids 1..5, t.a > 5 for none of them"
    );
    assert_eq!(
        ints(
            &db,
            "SELECT t.id FROM t WHERE (SELECT COUNT(*) FROM u WHERE u.a > 15 AND t.a > 5) > 0 \
             ORDER BY t.id"
        ),
        (6..=20).collect::<Vec<i64>>(),
    );
}
