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

//! Valid SQL that used to panic: i64::MIN primary keys (the I64Map
//! empty sentinel) and the unbounded-LIMIT batch size overflow on the
//! presorted window path.

use stoolap::Database;

#[test]
fn i64_min_primary_key_reports_instead_of_panicking() {
    // i64::MIN is the reserved empty sentinel of the row-id maps and an
    // INTEGER PRIMARY KEY doubles as the row id, so the value cannot
    // address a row. It must surface as an error, never a panic, and
    // the connection must stay usable.
    let db = Database::open("memory://panic_i64min_pk").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();

    let err = db
        .execute("INSERT INTO t VALUES ($1, $2)", (i64::MIN, 7i64))
        .expect_err("i64::MIN primary key must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("-9223372036854775808"),
        "error should name the offending value, got: {msg}"
    );

    // Neighbouring values still work and the table is intact
    db.execute("INSERT INTO t VALUES ($1, $2)", (i64::MIN + 1, 8i64))
        .unwrap();
    db.execute("INSERT INTO t VALUES ($1, $2)", (5i64, 9i64))
        .unwrap();
    let c: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(c, 2);
    let v: i64 = db
        .query_one("SELECT v FROM t WHERE id = $1", (i64::MIN + 1,))
        .unwrap();
    assert_eq!(v, 8);
}

#[test]
fn presorted_window_without_limit_does_not_overflow() {
    let db = Database::open("memory://panic_window_limit").unwrap();
    db.execute(
        "CREATE TABLE ps (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO ps VALUES (1,30,1), (2,10,2), (3,20,3), (4,10,4)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_psv ON ps(v)", ()).unwrap();

    let rows = db
        .query(
            "SELECT id, v, ROW_NUMBER() OVER (ORDER BY v) FROM ps ORDER BY id",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 4);
}

#[cfg(feature = "test-filedb")]
#[test]
fn i64_min_primary_key_rejected_on_persistent_tables() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let err = db
        .execute("INSERT INTO t VALUES ($1, $2)", (i64::MIN, 7i64))
        .expect_err("i64::MIN primary key must be rejected on file DBs too");
    assert!(err.to_string().contains("-9223372036854775808"));
    db.execute("INSERT INTO t VALUES ($1, $2)", (1i64, 2i64))
        .unwrap();
    let c: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(c, 1);
}
