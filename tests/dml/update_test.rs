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

//! UPDATE Statement Integration Tests

use stoolap::Database;

#[test]
fn test_simple_update() {
    let db = Database::open("memory://update_test_1").expect("Failed to create database");

    db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t1 VALUES (1, 'a')", ())
        .expect("Failed to insert");

    println!("Before UPDATE");

    // This is where it deadlocks
    db.execute("UPDATE t1 SET val = 'b' WHERE id = 1", ())
        .expect("Failed to update");

    println!("After UPDATE");

    let val: String = db.query_one("SELECT val FROM t1 WHERE id = 1", ()).unwrap();
    assert_eq!(val, "b");
}

/// UPDATE on a primary key column must be rejected
#[test]
fn test_pk_update_rejected() {
    let db = Database::open("memory://update_pk_reject").expect("Failed to create database");

    db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");

    db.execute("INSERT INTO t2 VALUES (1, 'a')", ())
        .expect("Failed to insert");

    let result = db.execute("UPDATE t2 SET id = 2 WHERE id = 1", ());
    assert!(result.is_err(), "UPDATE on PK should be rejected");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("primary key"),
        "Error should mention primary key, got: {}",
        err
    );

    // Original row should be untouched
    let val: String = db.query_one("SELECT val FROM t2 WHERE id = 1", ()).unwrap();
    assert_eq!(val, "a");
}

/// Non-PK column updates still work fine
#[test]
fn test_non_pk_update_works() {
    let db = Database::open("memory://update_non_pk").expect("Failed to create database");

    db.execute(
        "CREATE TABLE t3 (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
        (),
    )
    .expect("Failed to create table");

    db.execute("INSERT INTO t3 VALUES (1, 'Alice', 100)", ())
        .expect("Failed to insert");

    db.execute("UPDATE t3 SET name = 'Bob', score = 200 WHERE id = 1", ())
        .expect("Non-PK update should succeed");

    let name: String = db
        .query_one("SELECT name FROM t3 WHERE id = 1", ())
        .unwrap();
    assert_eq!(name, "Bob");

    let score: i64 = db
        .query_one("SELECT score FROM t3 WHERE id = 1", ())
        .unwrap();
    assert_eq!(score, 200);
}

/// PK update must also be rejected through the Transaction API
#[test]
fn test_pk_update_rejected_via_transaction_api() {
    let db = Database::open("memory://update_pk_tx_api").expect("Failed to create database");

    db.execute("CREATE TABLE t4 (id INTEGER PRIMARY KEY, val TEXT)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t4 VALUES (1, 'a')", ())
        .expect("Failed to insert");

    let mut tx = db.begin().unwrap();
    let result = tx.execute("UPDATE t4 SET id = 2 WHERE id = 1", ());
    assert!(
        result.is_err(),
        "PK update via Transaction API should be rejected"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("primary key"),
        "Error should mention primary key, got: {}",
        err
    );
    tx.rollback().unwrap();

    // Row should be untouched
    let val: String = db.query_one("SELECT val FROM t4 WHERE id = 1", ()).unwrap();
    assert_eq!(val, "a");
}
