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
    // More than 98 rows: in release the old code wrapped the batch size
    // to 98 and silently truncated instead of panicking
    let mut tx = db.begin().unwrap();
    for i in 1..=200i64 {
        tx.execute("INSERT INTO ps VALUES ($1, $2, $3)", (i, (i * 7) % 200, i))
            .unwrap();
    }
    tx.commit().unwrap();
    db.execute("CREATE INDEX idx_psv ON ps(v)", ()).unwrap();

    let rows = db
        .query(
            "SELECT id, v, ROW_NUMBER() OVER (ORDER BY v) FROM ps ORDER BY id",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(
        rows.len(),
        200,
        "every row must come back, not the wrapped 98"
    );
}

#[cfg(feature = "test-filedb")]
#[test]
fn i64_min_primary_key_rejected_on_persistent_tables() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    // Seal rows into cold segments first, so the rejected insert really
    // crosses the segmented-table path
    let mut tx = db.begin().unwrap();
    for i in 1..=60i64 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let err = db
        .execute("INSERT INTO t VALUES ($1, $2)", (i64::MIN, 7i64))
        .expect_err("i64::MIN primary key must be rejected on file DBs too");
    assert!(err.to_string().contains("-9223372036854775808"));
    db.execute("INSERT INTO t VALUES ($1, $2)", (1000i64, 2i64))
        .unwrap();
    let c: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(c, 61);
}

#[test]
fn i64_min_lookup_in_transaction_finds_nothing() {
    // No row can carry the sentinel id, so every lookup shape must
    // report "not found" instead of reaching the row-id map
    let db = Database::open("memory://panic_i64min_lookup").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();

    // Autocommit
    let rows = db
        .query("SELECT v FROM t WHERE id = $1", (i64::MIN,))
        .unwrap()
        .collect_vec()
        .unwrap();
    assert!(rows.is_empty());

    // Inside a transaction with uncommitted changes to the same table
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (2, 20)", ()).unwrap();
    let rows = tx
        .query("SELECT v FROM t WHERE id = $1", (i64::MIN,))
        .unwrap()
        .collect_vec()
        .unwrap();
    assert!(rows.is_empty(), "sentinel lookup must not match a row");
    let n = tx
        .execute("UPDATE t SET v = 99 WHERE id = $1", (i64::MIN,))
        .unwrap();
    assert_eq!(n, 0);
    let n = tx
        .execute("DELETE FROM t WHERE id = $1", (i64::MIN,))
        .unwrap();
    assert_eq!(n, 0);
    tx.commit().unwrap();

    let c: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(c, 2);
}

#[cfg(feature = "test-filedb")]
#[test]
fn i64_min_lookup_on_persistent_table_finds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 1..=50i64 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, i * 2))
            .unwrap();
    }
    tx.commit().unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // After sealing, the lookup crosses the cold-segment path too
    let rows = db
        .query("SELECT v FROM t WHERE id = $1", (i64::MIN,))
        .unwrap()
        .collect_vec()
        .unwrap();
    assert!(rows.is_empty());
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (100, 200)", ()).unwrap();
    let rows = tx
        .query("SELECT v FROM t WHERE id = $1", (i64::MIN,))
        .unwrap()
        .collect_vec()
        .unwrap();
    assert!(rows.is_empty());
    tx.rollback().unwrap();
}

#[test]
fn i64_min_in_list_is_matchable() {
    let db = Database::open("memory://panic_i64min_in").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, $1)", (i64::MIN,))
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, 3)", ()).unwrap();

    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE v IN (-9223372036854775808)",
            (),
        )
        .unwrap();
    assert_eq!(c, 1);
    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE v IN (1, 2, -9223372036854775808)",
            (),
        )
        .unwrap();
    assert_eq!(c, 1);
    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE v NOT IN (-9223372036854775808)",
            (),
        )
        .unwrap();
    assert_eq!(c, 1);
    // PK column IN-list takes the index probe path
    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE id IN (-9223372036854775808, 1)",
            (),
        )
        .unwrap();
    assert_eq!(c, 1);
}

#[test]
fn i64_min_group_by_value() {
    let db = Database::open("memory://panic_i64min_group").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, $1)", (i64::MIN,))
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, 3)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (3, $1)", (i64::MIN,))
        .unwrap();

    let rows = db
        .query("SELECT v, COUNT(*) FROM t GROUP BY v ORDER BY v", ())
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 2);
    let first: i64 = rows[0].get(0).unwrap();
    let first_count: i64 = rows[0].get(1).unwrap();
    assert_eq!(first, i64::MIN);
    assert_eq!(first_count, 2);
    let s: i64 = db
        .query_one("SELECT SUM(id) FROM t GROUP BY v ORDER BY v LIMIT 1", ())
        .unwrap();
    assert_eq!(s, 4);
}

#[test]
fn huge_limit_does_not_abort() {
    let db = Database::open("memory://panic_huge_limit").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 1..=10i64 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, i % 3))
            .unwrap();
    }
    tx.commit().unwrap();

    let n = db
        .query("SELECT * FROM t LIMIT 9223372036854775807", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .len();
    assert_eq!(n, 10);
    let n = db
        .query(
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY v) FROM t LIMIT 9223372036854775807",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap()
        .len();
    assert_eq!(n, 10);
    let n = db
        .query(
            "SELECT id, ROW_NUMBER() OVER (ORDER BY v) FROM t ORDER BY id \
             LIMIT 9223372036854775807 OFFSET 3",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap()
        .len();
    assert_eq!(n, 7);
}

#[test]
fn auto_increment_exhaustion_reports_cleanly() {
    let db = Database::open("memory://panic_autoinc").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (9223372036854775807, 1)", ())
        .unwrap();
    let err = db
        .execute("INSERT INTO t (v) VALUES (2)", ())
        .expect_err("exhausted auto-increment must report, not wrap");
    assert!(
        err.to_string().to_lowercase().contains("auto"),
        "error should mention auto-increment, got: {err}"
    );
    // The id space must stay intact
    let c: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(c, 1);
}

#[test]
fn huge_limit_inside_transaction() {
    // The local-changes path builds its own result buffer
    let db = Database::open("memory://panic_limit_tx").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2)", ())
        .unwrap();

    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (3, 3)", ()).unwrap();
    let n = tx
        .query("SELECT * FROM t LIMIT 9223372036854775807", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .len();
    assert_eq!(n, 3);
    tx.commit().unwrap();
}

#[test]
fn huge_limit_with_cte_join() {
    let db = Database::open("memory://panic_limit_cte").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 1..=20i64 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();

    let n = db
        .query(
            "WITH c AS (SELECT id, v FROM t WHERE v > 5) \
             SELECT c.id FROM c JOIN t ON t.id = c.id LIMIT 9223372036854775807",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap()
        .len();
    assert_eq!(n, 15);
}

#[test]
fn i64_min_join_key_against_pk() {
    // A non-PK column may legally hold i64::MIN; joining it against a PK
    // can never match, but it must not panic on the way there
    let db = Database::open("memory://panic_join_sentinel").unwrap();
    db.execute(
        "CREATE TABLE outer_t (id INTEGER PRIMARY KEY, k INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE inner_t (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 1..=50i64 {
        tx.execute("INSERT INTO inner_t VALUES ($1, $2)", (i, i * 2))
            .unwrap();
        tx.execute("INSERT INTO outer_t VALUES ($1, $2)", (i, i))
            .unwrap();
    }
    tx.execute("INSERT INTO outer_t VALUES ($1, $2)", (100i64, i64::MIN))
        .unwrap();
    tx.commit().unwrap();

    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM outer_t o JOIN inner_t i ON i.id = o.k",
            (),
        )
        .unwrap();
    assert_eq!(c, 50, "the sentinel key matches nothing");
}

#[test]
fn i64_min_not_in_subquery_on_pk() {
    let db = Database::open("memory://panic_not_in_sub").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE ex (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 1..=30i64 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.execute("INSERT INTO ex VALUES (1, $1)", (i64::MIN,))
        .unwrap();
    tx.execute("INSERT INTO ex VALUES (2, 5)", ()).unwrap();
    tx.commit().unwrap();

    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE id NOT IN (SELECT k FROM ex)",
            (),
        )
        .unwrap();
    assert_eq!(c, 29, "only id = 5 is excluded");
    let n = db
        .execute("DELETE FROM t WHERE id IN (SELECT k FROM ex)", ())
        .unwrap();
    assert_eq!(n, 1);
}
