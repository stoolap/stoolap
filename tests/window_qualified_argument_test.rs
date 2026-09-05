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

//! A window function reads the same column whether its argument names the
//! column on its own or through the table it comes from. On one table the
//! rows carry the bare name, so a qualified argument has to fall back to it.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, city TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)",
        (),
    )
    .unwrap();
    for id in 1..=4i64 {
        db.execute(
            "INSERT INTO users VALUES ($1, $2)",
            (id, if id % 2 == 0 { "a" } else { "b" }),
        )
        .unwrap();
        db.execute("INSERT INTO orders VALUES ($1, $2, $3)", (id, id, id * 10))
            .unwrap();
    }
    db
}

fn column(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            row.get::<Option<String>>(0)
                .unwrap()
                .unwrap_or_else(|| "NULL".into())
        })
        .collect()
}

/// Every window function that reads a column reads it through an alias too
#[test]
fn test_qualified_argument_matches_the_bare_one() {
    let db = setup("window_qualified_argument");
    for (bare, qualified) in [
        (
            "SELECT LAG(id, 1) OVER (ORDER BY id) FROM users ORDER BY id",
            "SELECT LAG(u.id, 1) OVER (ORDER BY u.id) FROM users u ORDER BY u.id",
        ),
        (
            "SELECT LAG(id) OVER (ORDER BY id) FROM users ORDER BY id",
            "SELECT LAG(u.id) OVER (ORDER BY u.id) FROM users u ORDER BY u.id",
        ),
        (
            "SELECT LEAD(id, 1) OVER (ORDER BY id) FROM users ORDER BY id",
            "SELECT LEAD(u.id, 1) OVER (ORDER BY u.id) FROM users u ORDER BY u.id",
        ),
        (
            "SELECT FIRST_VALUE(id) OVER (ORDER BY id) FROM users ORDER BY id",
            "SELECT FIRST_VALUE(u.id) OVER (ORDER BY u.id) FROM users u ORDER BY u.id",
        ),
        (
            "SELECT LAST_VALUE(id) OVER (ORDER BY id) FROM users ORDER BY id",
            "SELECT LAST_VALUE(u.id) OVER (ORDER BY u.id) FROM users u ORDER BY u.id",
        ),
        (
            "SELECT NTH_VALUE(id, 2) OVER (ORDER BY id) FROM users ORDER BY id",
            "SELECT NTH_VALUE(u.id, 2) OVER (ORDER BY u.id) FROM users u ORDER BY u.id",
        ),
        (
            "SELECT LAG(id, 1) OVER (PARTITION BY city ORDER BY id) FROM users ORDER BY id",
            "SELECT LAG(u.id, 1) OVER (PARTITION BY u.city ORDER BY u.id) FROM users u ORDER BY u.id",
        ),
    ] {
        let expected = column(&db, bare);
        assert_eq!(column(&db, qualified), expected, "{qualified}");
        assert!(
            expected.iter().any(|value| value != "NULL"),
            "the bare form itself reads nothing: {bare}"
        );
    }
}

/// The table's own name qualifies a column the same way an alias does
#[test]
fn test_table_name_qualifies_a_window_argument() {
    let db = setup("window_qualified_by_table");
    assert_eq!(
        column(
            &db,
            "SELECT LAG(users.id, 1) OVER (ORDER BY users.id) FROM users ORDER BY users.id"
        ),
        ["NULL", "1", "2", "3"]
    );
}

/// A join carries qualified column names, which still resolve
#[test]
fn test_qualified_argument_over_a_join() {
    let db = setup("window_qualified_join");
    assert_eq!(
        column(
            &db,
            "SELECT LAG(o.amount, 1) OVER (ORDER BY o.id) FROM users u JOIN orders o ON o.user_id = u.id ORDER BY o.id"
        ),
        ["NULL", "10", "20", "30"]
    );
}
