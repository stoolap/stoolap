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

//! Tests for ON CONFLICT (cols) target enforcement.
//! Verifies that conflict handling only triggers when the violation matches
//! the specified conflict target columns (PostgreSQL semantics).

use stoolap::Database;

/// Reproduction case: conflict on column b should NOT trigger DO UPDATE targeting column a.
#[test]
fn test_conflict_target_rejects_wrong_column() {
    let db = Database::open("memory://conflict_target_wrong").unwrap();

    db.execute(
        "CREATE TABLE t (a INTEGER UNIQUE, b INTEGER UNIQUE, v TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 10, 'x')", ()).unwrap();

    // Conflict is on b=10, but target is (a) — should error
    let result = db.execute(
        "INSERT INTO t VALUES (2, 10, 'y') ON CONFLICT (a) DO UPDATE SET v = 'updated'",
        (),
    );
    assert!(
        result.is_err(),
        "Should error when conflict is on b but target is (a)"
    );

    // Verify original row is untouched
    let rows: Vec<_> = db
        .query("SELECT a, b, v FROM t ORDER BY a", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<i64>(1).unwrap(),
                r.get::<String>(2).unwrap(),
            )
        })
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0], (1, 10, "x".to_string()));
}

/// Conflict on column a should trigger DO UPDATE targeting column a.
#[test]
fn test_conflict_target_matches_correct_column() {
    let db = Database::open("memory://conflict_target_match").unwrap();

    db.execute(
        "CREATE TABLE t (a INTEGER UNIQUE, b INTEGER UNIQUE, v TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 10, 'x')", ()).unwrap();

    // Conflict is on a=1, target is (a) — should update
    db.execute(
        "INSERT INTO t VALUES (1, 20, 'y') ON CONFLICT (a) DO UPDATE SET v = 'updated'",
        (),
    )
    .unwrap();

    let v: String = db
        .query("SELECT v FROM t", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(v, "updated");
}

/// DO NOTHING with conflict target should only skip matching conflicts.
#[test]
fn test_do_nothing_target_rejects_wrong_column() {
    let db = Database::open("memory://do_nothing_target_wrong").unwrap();

    db.execute(
        "CREATE TABLE t (a INTEGER UNIQUE, b INTEGER UNIQUE, v TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 10, 'x')", ()).unwrap();

    // Conflict is on b=10, but target is (a) — should error, not silently skip
    let result = db.execute(
        "INSERT INTO t VALUES (2, 10, 'y') ON CONFLICT (a) DO NOTHING",
        (),
    );
    assert!(
        result.is_err(),
        "Should error when conflict is on b but target is (a)"
    );
}

/// DO NOTHING with matching conflict target should skip silently.
#[test]
fn test_do_nothing_target_skips_matching_conflict() {
    let db = Database::open("memory://do_nothing_target_match").unwrap();

    db.execute(
        "CREATE TABLE t (a INTEGER UNIQUE, b INTEGER UNIQUE, v TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 10, 'x')", ()).unwrap();

    // Conflict is on a=1, target is (a) — should skip silently
    db.execute(
        "INSERT INTO t VALUES (1, 99, 'y') ON CONFLICT (a) DO NOTHING",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query("SELECT COUNT(*) FROM t", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 1);
}

/// Empty conflict target (MySQL semantics) should match any conflict.
#[test]
fn test_empty_conflict_target_matches_all() {
    let db = Database::open("memory://empty_conflict_target").unwrap();

    db.execute(
        "CREATE TABLE t (a INTEGER UNIQUE, b INTEGER UNIQUE, v TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 10, 'x')", ()).unwrap();

    // ON DUPLICATE KEY UPDATE (no target) — should match any conflict
    db.execute(
        "INSERT INTO t VALUES (1, 20, 'y') ON DUPLICATE KEY UPDATE v = 'updated'",
        (),
    )
    .unwrap();

    let v: String = db
        .query("SELECT v FROM t", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(v, "updated");
}

/// PK conflict with PK as conflict target should work.
#[test]
fn test_conflict_target_pk() {
    let db = Database::open("memory://conflict_target_pk").unwrap();

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, v TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 10, 'x')", ()).unwrap();

    // PK conflict, target is (id) — should update
    db.execute(
        "INSERT INTO t VALUES (1, 20, 'y') ON CONFLICT (id) DO UPDATE SET v = 'updated'",
        (),
    )
    .unwrap();

    let v: String = db
        .query("SELECT v FROM t WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(v, "updated");
}

/// PK conflict but conflict target is a different unique column — should error.
#[test]
fn test_conflict_target_pk_wrong_target() {
    let db = Database::open("memory://conflict_target_pk_wrong").unwrap();

    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, v TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO t VALUES (1, 10, 'x')", ()).unwrap();

    // PK conflict (id=1), but target is (a) — should error
    let result = db.execute(
        "INSERT INTO t VALUES (1, 20, 'y') ON CONFLICT (a) DO UPDATE SET v = 'updated'",
        (),
    );
    assert!(
        result.is_err(),
        "Should error when PK conflict but target is (a)"
    );
}

/// INSERT...SELECT with conflict target enforcement.
#[test]
fn test_conflict_target_insert_select() {
    let db = Database::open("memory://conflict_target_insert_select").unwrap();

    db.execute(
        "CREATE TABLE t (a INTEGER UNIQUE, b INTEGER UNIQUE, v TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE src (a INTEGER, b INTEGER, v TEXT)", ())
        .unwrap();

    db.execute("INSERT INTO t VALUES (1, 10, 'x')", ()).unwrap();
    db.execute("INSERT INTO src VALUES (2, 10, 'y')", ())
        .unwrap();

    // INSERT...SELECT: conflict on b=10 but target is (a) — should error
    let result = db.execute(
        "INSERT INTO t SELECT * FROM src ON CONFLICT (a) DO UPDATE SET v = 'updated'",
        (),
    );
    assert!(
        result.is_err(),
        "INSERT...SELECT should error when conflict is on b but target is (a)"
    );
}
