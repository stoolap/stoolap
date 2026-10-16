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

/// Every parent has a child now; before the children were written, none did
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, p_id INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1), (2), (3)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

const BEFORE: &str = "AS OF TIMESTAMP '1970-01-01 00:00:00'";

#[test]
fn test_exists_reads_the_inner_table_as_of() {
    let db = setup("semi_join_as_of_exists");
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE EXISTS (SELECT 1 FROM c WHERE c.p_id = p.id) ORDER BY id"
        ),
        vec![1, 2, 3],
        "now every parent has a child"
    );
    assert_eq!(
        ids(
            &db,
            &format!(
                "SELECT id FROM p WHERE EXISTS (SELECT 1 FROM c {BEFORE} WHERE c.p_id = p.id) ORDER BY id"
            )
        ),
        Vec::<i64>::new(),
        "before the children were written, none had one"
    );
}

#[test]
fn test_not_exists_reads_the_inner_table_as_of() {
    let db = setup("semi_join_as_of_not_exists");
    assert_eq!(
        ids(
            &db,
            &format!(
                "SELECT id FROM p WHERE NOT EXISTS (SELECT 1 FROM c {BEFORE} WHERE c.p_id = p.id) ORDER BY id"
            )
        ),
        vec![1, 2, 3],
        "the anti-join reads the children as of then"
    );
    assert_eq!(
        ids(
            &db,
            &format!(
                "SELECT id FROM p WHERE id > 0 AND NOT EXISTS (SELECT 1 FROM c {BEFORE} WHERE c.p_id = p.id) ORDER BY id"
            )
        ),
        vec![1, 2, 3],
        "NOT EXISTS beside another conjunct, the semi-join form"
    );
}
