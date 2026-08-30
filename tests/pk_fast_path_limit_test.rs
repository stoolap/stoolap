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

//! The PK-lookup fast path must respect LIMIT 0 and OFFSET: a fast-path
//! answer that ignores them returns a row where correct execution
//! returns an empty set.

use stoolap::api::Database;

/// One database per test: under `cargo test` all tests share a process,
/// and `memory://` DSNs are registry-shared within it.
fn test_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://pk_fast_path_limit_{}", name)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (5, 'x')", ()).unwrap();
    db
}

fn row_count(db: &Database, sql: &str) -> usize {
    db.query(sql, ()).unwrap().collect_vec().unwrap().len()
}

#[test]
fn pk_lookup_respects_limit_zero() {
    let db = test_db("pk_lookup_respects_limit_zero");

    // Run twice: the second execution takes the cached/compiled path.
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id = 5 LIMIT 0"), 0);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id = 5 LIMIT 0"), 0);
}

#[test]
fn pk_lookup_respects_offset() {
    let db = test_db("pk_lookup_respects_offset");

    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE id = 5 LIMIT 10 OFFSET 1"),
        0
    );
    assert_eq!(
        row_count(&db, "SELECT * FROM t WHERE id = 5 LIMIT 10 OFFSET 1"),
        0
    );
}

#[test]
fn pk_lookup_with_limit_one_still_returns_the_row() {
    let db = test_db("pk_lookup_with_limit_one_still_returns_the_row");

    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id = 5 LIMIT 1"), 1);
    assert_eq!(row_count(&db, "SELECT * FROM t WHERE id = 5 LIMIT 1"), 1);
}

#[test]
fn prepared_pk_lookup_respects_limit_zero() {
    let db = test_db("prepared_pk_lookup_respects_limit_zero");

    let stmt = db.prepare("SELECT * FROM t WHERE id = $1 LIMIT 0").unwrap();
    for _ in 0..3 {
        assert_eq!(stmt.query((5,)).unwrap().collect_vec().unwrap().len(), 0);
    }

    let stmt = db
        .prepare("SELECT * FROM t WHERE id = $1 LIMIT 5 OFFSET 2")
        .unwrap();
    for _ in 0..3 {
        assert_eq!(stmt.query((5,)).unwrap().collect_vec().unwrap().len(), 0);
    }
}

#[test]
fn prepared_pk_lookup_survives_wrongly_typed_first_execution() {
    let db = test_db("prepared_pk_lookup_survives_wrongly_typed_first_execution");

    let stmt = db.prepare("SELECT * FROM t WHERE id = $1").unwrap();

    // First execution with a text parameter: no match, standard path
    assert_eq!(
        stmt.query(("abc",)).unwrap().collect_vec().unwrap().len(),
        0
    );
    // Later integer executions must still return the row (and may fast-path)
    for _ in 0..3 {
        assert_eq!(stmt.query((5,)).unwrap().collect_vec().unwrap().len(), 1);
    }
    // Interleave wrong and right types
    assert_eq!(
        stmt.query(("abc",)).unwrap().collect_vec().unwrap().len(),
        0
    );
    assert_eq!(stmt.query((5,)).unwrap().collect_vec().unwrap().len(), 1);
}
