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

//! A boolean column standing on its own is a condition, and it holds
//! wherever the column is true.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, b BOOLEAN, n INTEGER)",
        (),
    )
    .unwrap();
    for (id, b, n) in [(1i64, true, 1i64), (2, false, 2), (3, true, 3)] {
        db.execute("INSERT INTO c VALUES ($1, $2, $3)", (id, b, n))
            .unwrap();
    }
    db
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

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

#[test]
fn test_a_boolean_column_reads_as_a_condition() {
    let db = setup("boolean_predicate_where");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE b"), 2);
    assert_eq!(ids(&db, "SELECT id FROM c WHERE b ORDER BY id"), [1, 3]);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE (b)"), 2);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE c.b"), 2);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE NOT b"), 1);
}

/// The same column beside another condition
#[test]
fn test_a_boolean_column_beside_another_condition() {
    let db = setup("boolean_predicate_combined");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE b AND 1 = 1"), 2);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE b OR FALSE"), 2);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE b AND n > 1"), 1);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE b = TRUE"), 2);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c WHERE b IS TRUE"), 2);
}

/// A condition is a condition wherever the query puts one
#[test]
fn test_a_boolean_column_in_the_other_clauses() {
    let db = setup("boolean_predicate_clauses");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM c x JOIN c y ON x.id = y.id AND x.b"
        ),
        2
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c GROUP BY b HAVING b"), 2);
}
