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

//! DISTINCT reads the columns the SELECT asked for, and a LIMIT counts the
//! rows it left behind.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT, n INTEGER)",
        (),
    )
    .unwrap();
    for (id, g, n) in [
        (1i64, "a", 10i64),
        (2, "a", 20),
        (3, "b", 30),
        (4, "b", 40),
        (5, "c", 50),
    ] {
        db.execute("INSERT INTO p VALUES ($1, $2, $3)", (id, g, n))
            .unwrap();
    }
    db
}

fn values(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            row.unwrap()
                .get::<Option<String>>(0)
                .unwrap()
                .unwrap_or_else(|| "NULL".into())
        })
        .collect()
}

#[test]
fn test_distinct_with_a_limit() {
    let db = setup("distinct_limit");
    assert_eq!(
        values(&db, "SELECT DISTINCT g FROM p ORDER BY g LIMIT 2"),
        ["a", "b"]
    );
    assert_eq!(
        values(&db, "SELECT DISTINCT g FROM p ORDER BY g LIMIT 10"),
        ["a", "b", "c"]
    );
    assert_eq!(
        values(&db, "SELECT DISTINCT g FROM p ORDER BY g LIMIT 1"),
        ["a"]
    );
    assert_eq!(
        values(&db, "SELECT DISTINCT g FROM p ORDER BY g DESC LIMIT 2"),
        ["c", "b"]
    );
}

#[test]
fn test_distinct_with_a_limit_and_an_offset() {
    let db = setup("distinct_limit_offset");
    assert_eq!(
        values(&db, "SELECT DISTINCT g FROM p ORDER BY g LIMIT 2 OFFSET 1"),
        ["b", "c"]
    );
    assert_eq!(
        values(&db, "SELECT DISTINCT g FROM p ORDER BY g OFFSET 1"),
        ["b", "c"]
    );
}

/// Sorting by a column the SELECT leaves out does not widen what DISTINCT
/// reads
#[test]
fn test_distinct_sorted_by_a_column_it_does_not_read() {
    let db = setup("distinct_limit_hidden");
    assert_eq!(
        values(&db, "SELECT DISTINCT g FROM p ORDER BY id LIMIT 2"),
        ["a", "b"]
    );
    assert_eq!(
        values(&db, "SELECT DISTINCT UPPER(g) FROM p ORDER BY 1 LIMIT 2"),
        ["A", "B"]
    );
}
