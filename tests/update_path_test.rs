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

//! Expression SET values on the PK fast-update path: compiled once per
//! plan, executed per update against the current row.

use stoolap::api::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://update_path_{}", name)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 20)", ()).unwrap();
    db
}

#[test]
fn prepared_expression_update_reads_current_row() {
    let db = setup("expr");
    let stmt = db.prepare("UPDATE t SET a = a + 1 WHERE id = $1").unwrap();
    for _ in 0..5 {
        assert_eq!(stmt.execute((1i64,)).unwrap(), 1);
    }
    let a: i64 = db.query_one("SELECT a FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(a, 15, "each execution must read the then-current row");
}

#[test]
fn multi_assignment_reads_original_row() {
    let db = setup("swap");
    // All SET expressions read the ORIGINAL row: this swaps
    db.execute("UPDATE t SET a = b, b = a WHERE id = 1", ())
        .unwrap();
    let r = db
        .query("SELECT a, b FROM t WHERE id = 1", ())
        .unwrap()
        .collect_vec()
        .unwrap();
    let (a, b): (i64, i64) = (r[0].get(0).unwrap(), r[0].get(1).unwrap());
    assert_eq!((a, b), (20, 10));
}

#[test]
fn expression_update_with_parameter_inside() {
    let db = setup("exprparam");
    let stmt = db.prepare("UPDATE t SET a = a + $1 WHERE id = $2").unwrap();
    stmt.execute((5i64, 1i64)).unwrap();
    stmt.execute((7i64, 1i64)).unwrap();
    let a: i64 = db.query_one("SELECT a FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(a, 22);
}

#[test]
fn expression_update_null_propagates() {
    let db = setup("exprnull");
    db.execute("INSERT INTO t VALUES (2, NULL, 1)", ()).unwrap();
    db.execute("UPDATE t SET a = a + 1 WHERE id = 2", ())
        .unwrap();
    let n: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE id = 2 AND a IS NULL", ())
        .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn expression_update_still_respects_check_constraints() {
    let db = Database::open("memory://update_path_check").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, age INTEGER CHECK (age >= 0))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, 0)", ()).unwrap();
    // Fast path bails for CHECK-constrained columns; the full path must
    // reject the violating decrement
    let stmt = db
        .prepare("UPDATE c SET age = age - 1 WHERE id = $1")
        .unwrap();
    assert!(stmt.execute((1i64,)).is_err());
    let age: i64 = db.query_one("SELECT age FROM c WHERE id = 1", ()).unwrap();
    assert_eq!(age, 0);
}

#[test]
fn scalar_subquery_set_takes_the_full_path() {
    let db = setup("subq");
    db.execute("INSERT INTO t VALUES (2, 99, 0)", ()).unwrap();
    let stmt = db
        .prepare("UPDATE t SET a = (SELECT MAX(a) FROM t) WHERE id = $1")
        .unwrap();
    stmt.execute((1i64,)).unwrap();
    let a: i64 = db.query_one("SELECT a FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(a, 99);
    // Repeat through the cached plan
    stmt.execute((1i64,)).unwrap();
    let a: i64 = db.query_one("SELECT a FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(a, 99);
}

#[test]
fn vector_dimension_validation_on_fast_update_path() {
    let db = Database::open("memory://update_path_vec").unwrap();
    db.execute("CREATE TABLE v (id INTEGER PRIMARY KEY, emb VECTOR(3))", ())
        .unwrap();
    db.execute("INSERT INTO v VALUES (1, CAST('[1,2,3]' AS VECTOR))", ())
        .unwrap();

    // Expression SET with a wrong-dimension vector must be rejected like
    // the full path (VectorDimensionMismatch), not silently stored
    let stmt = db
        .prepare("UPDATE v SET emb = CAST('[1,2]' AS VECTOR) WHERE id = $1")
        .unwrap();
    assert!(stmt.execute((1i64,)).is_err());

    // Same through a parameter source
    let two_dim: Vec<f32> = vec![1.0, 2.0];
    let stmt2 = db.prepare("UPDATE v SET emb = $1 WHERE id = $2").unwrap();
    let _ = two_dim;
    let res = stmt2.execute(("[1,2]", 1i64));
    assert!(res.is_err(), "wrong-dimension vector text must not store");

    // The stored row is unchanged and correct updates still work
    let stmt3 = db
        .prepare("UPDATE v SET emb = CAST('[4,5,6]' AS VECTOR) WHERE id = $1")
        .unwrap();
    stmt3.execute((1i64,)).unwrap();
}

#[test]
fn set_errors_are_consistent_across_paths() {
    let db = Database::open("memory://update_path_err").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 9223372036854775807, 0)", ())
        .unwrap();

    // Overflow must fail the statement on BOTH paths, applying nothing
    assert!(db
        .execute("UPDATE t SET a = a + 1, b = 5 WHERE id = 1", ())
        .is_err());
    let mut tx = db.begin().unwrap();
    assert!(tx
        .execute("UPDATE t SET a = a + 1, b = 5 WHERE id = 1", ())
        .is_err());
    drop(tx);
    let b: i64 = db.query_one("SELECT b FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(b, 0, "no partial application on either path");

    // Non-null-to-NULL coercion must error on BOTH paths
    assert!(db
        .execute("UPDATE t SET a = 'abc' WHERE id = 1", ())
        .is_err());
    let mut tx = db.begin().unwrap();
    assert!(tx
        .execute("UPDATE t SET a = 'abc' WHERE id = 1", ())
        .is_err());
    drop(tx);
    let a: i64 = db.query_one("SELECT a FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(a, 9223372036854775807);
}
