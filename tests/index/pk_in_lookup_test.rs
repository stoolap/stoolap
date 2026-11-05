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

//! A primary key `IN` list is answered by row id: the members must be read
//! as the comparison reads them, and a LIMIT must count the rows found, not
//! the members asked for.

use stoolap::Database;

/// The decoded-group counter is process-wide, so the tests that measure
/// it run one at a time, their fixtures included, whatever the runner does
static DECODES: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn measuring() -> std::sync::MutexGuard<'static, ()> {
    DECODES.lock().unwrap_or_else(|e| e.into_inner())
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
}

fn fixture(name: &str) -> Database {
    let db = Database::open(&format!("memory://pk_in_lookup_{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db
}

/// A member equal to a key matches whatever its type, as `=` does; one no
/// key can equal matches nothing.
#[test]
fn a_numeric_member_matches_as_the_comparison_does() {
    let db = fixture("float");
    assert_eq!(ids(&db, "SELECT id FROM t WHERE id = 1.0"), vec![1]);
    assert_eq!(ids(&db, "SELECT id FROM t WHERE id IN (1.0)"), vec![1]);
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE id IN (1.0, 2) ORDER BY id"),
        vec![1, 2]
    );
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE id IN (1.5)"),
        Vec::<i64>::new()
    );
    assert_eq!(ids(&db, "SELECT id FROM t WHERE id IN (1.5, 3)"), vec![3]);
}

/// A LIMIT counts the rows found: a member with no row does not use up
/// the limit.
#[test]
fn a_limit_counts_the_rows_found_not_the_members() {
    let db = fixture("limit");
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE id IN (0, 1) LIMIT 1"),
        vec![1]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE id IN (0, 1, 2) LIMIT 1 OFFSET 1"
        ),
        vec![2]
    );
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE id IN (0, 9, 1, 8, 2) LIMIT 2"),
        vec![1, 2]
    );
    // A deleted key is no row either
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(
        ids(&db, "SELECT id FROM t WHERE id IN (1, 2) LIMIT 1"),
        vec![2]
    );
}

/// The same rule when the members come from a subquery.
#[test]
fn a_limit_over_a_subquerys_members_counts_the_rows_found() {
    let db = fixture("subquery");
    db.execute("CREATE TABLE u (k INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (0), (1)", ()).unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE id IN (SELECT k FROM u) LIMIT 1"
        ),
        vec![1]
    );
    db.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE id IN (SELECT k FROM u) LIMIT 1"
        ),
        Vec::<i64>::new()
    );
}

/// The same rule when an EXISTS is turned into a set of members.
#[test]
fn a_limit_over_an_exists_set_counts_the_rows_found() {
    let db = fixture("exists");
    db.execute("CREATE TABLE u (k INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (0), (1)", ()).unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id) LIMIT 1"
        ),
        vec![1]
    );
}

/// A float member that is exactly a key matches whatever its size, as `=`
/// does: the conversion is the comparison's.
#[test]
fn a_large_float_member_matches_as_the_comparison_does() {
    let db = Database::open("memory://pk_in_lookup_large").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10), (9007199254740992, 20), (-9007199254740992, 30)",
        (),
    )
    .unwrap();
    for (member, want) in [
        ("9007199254740992.0", 9007199254740992i64),
        ("-9007199254740992.0", -9007199254740992i64),
    ] {
        assert_eq!(
            ids(&db, &format!("SELECT id FROM t WHERE id = {member}")),
            vec![want]
        );
        assert_eq!(
            ids(&db, &format!("SELECT id FROM t WHERE id IN ({member})")),
            vec![want],
            "IN ({member})"
        );
    }
    // The subquery route reads its members the same way
    db.execute("CREATE TABLE f (x FLOAT)", ()).unwrap();
    db.execute("INSERT INTO f VALUES (1.0), (9007199254740992.0)", ())
        .unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE id IN (SELECT x FROM f) ORDER BY id"
        ),
        vec![1, 9007199254740992]
    );
}

/// The EXISTS set path in a file database, with members no row answers
/// to: a LIMIT counts the rows found, whatever the set's order.
#[test]
fn a_limit_over_an_exists_set_with_absent_members_counts_the_rows_found() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("CREATE TABLE u (k INTEGER, g INTEGER)", ())
        .unwrap();
    // Two hundred members without a row beside the one with
    let values = (0..200)
        .map(|i| format!("({}, 1)", -i))
        .chain(std::iter::once("(1, 1)".to_string()))
        .collect::<Vec<_>>()
        .join(",");
    db.execute(&format!("INSERT INTO u VALUES {values}"), ())
        .unwrap();
    for _ in 0..5 {
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id) LIMIT 1"
            ),
            vec![1]
        );
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id AND u.g = 1) LIMIT 1"
            ),
            vec![1]
        );
    }
}

/// A small LIMIT over a long list reads a few rows, not every member's:
/// the fetch stops once the rows found satisfy it.
#[test]
fn a_small_limit_over_a_long_list_reads_a_few_rows() {
    let _measuring = measuring();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, name TEXT)",
            (),
        )
        .unwrap();
        // Four sealed volumes of five thousand rows with a long text column
        for volume in 0..4 {
            for chunk in 0..5 {
                let values = (0..1000)
                    .map(|i| {
                        let id = volume * 5000 + chunk * 1000 + i + 1;
                        format!("({id}, {}, '{}')", id * 3, "x".repeat(128))
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                db.execute(&format!("INSERT INTO t VALUES {values}"), ())
                    .unwrap();
            }
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    let members = (1..=20_000)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let misses = |db: &Database| -> i64 {
        let rows = db.query("PRAGMA GROUP_CACHE_STATS", ()).unwrap();
        let columns: Vec<String> = rows.columns().to_vec();
        let at = columns.iter().position(|c| c == "misses").unwrap();
        rows.into_iter().next().unwrap().unwrap().get(at).unwrap()
    };
    let before = misses(&db);
    assert_eq!(
        ids(
            &db,
            &format!("SELECT id FROM t WHERE id IN ({members}) LIMIT 1")
        ),
        vec![1]
    );
    let decoded = misses(&db) - before;
    assert!(
        decoded <= 3,
        "the first volume's groups alone: {decoded} decoded"
    );
}

/// A run of members without a row ahead of the ones with: the fetch after
/// it is still a bounded batch, so a LIMIT 1 reads a few rows, not every
/// remaining member's.
#[test]
fn a_run_of_absent_members_does_not_make_the_next_fetch_read_everything() {
    let _measuring = measuring();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, name TEXT)",
            (),
        )
        .unwrap();
        for volume in 0..4 {
            for chunk in 0..5 {
                let values = (0..1000)
                    .map(|i| {
                        let id = volume * 5000 + chunk * 1000 + i + 1;
                        format!("({id}, {}, '{}')", id * 3, "x".repeat(128))
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                db.execute(&format!("INSERT INTO t VALUES {values}"), ())
                    .unwrap();
            }
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    // 32,752 members without a row, then every row's
    let members = (-32_751..=20_000)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let misses = |db: &Database| -> i64 {
        let rows = db.query("PRAGMA GROUP_CACHE_STATS", ()).unwrap();
        let columns: Vec<String> = rows.columns().to_vec();
        let at = columns.iter().position(|c| c == "misses").unwrap();
        rows.into_iter().next().unwrap().unwrap().get(at).unwrap()
    };
    let before = misses(&db);
    assert_eq!(
        ids(
            &db,
            &format!("SELECT id FROM t WHERE id IN ({members}) LIMIT 1")
        ),
        vec![1]
    );
    let decoded = misses(&db) - before;
    assert!(
        decoded <= 3,
        "the first volume's groups alone: {decoded} decoded"
    );
}

/// A float member of a subquery's set names the key it equals in every
/// statement, IN and NOT IN alike: the UPDATE and DELETE candidate paths
/// and the negated SELECT read it as the comparison does
#[test]
fn a_float_member_names_the_key_in_every_statement() {
    for (name, statement, expected) in [
        (
            "select_in",
            "SELECT id FROM t WHERE id IN (SELECT k FROM keys) ORDER BY id",
            vec![1],
        ),
        (
            "update_in",
            "UPDATE t SET k = 1 WHERE id IN (SELECT k FROM keys) RETURNING id",
            vec![1],
        ),
        (
            "delete_in",
            "DELETE FROM t WHERE id IN (SELECT k FROM keys) RETURNING id",
            vec![1],
        ),
        (
            "select_not_in",
            "SELECT id FROM t WHERE id NOT IN (SELECT k FROM keys) ORDER BY id",
            vec![2, 3],
        ),
        (
            "update_not_in",
            "UPDATE t SET k = 1 WHERE id NOT IN (SELECT k FROM keys) RETURNING id",
            vec![2, 3],
        ),
        (
            "delete_not_in",
            "DELETE FROM t WHERE id NOT IN (SELECT k FROM keys) RETURNING id",
            vec![2, 3],
        ),
    ] {
        let db = fixture(&format!("subquery_{name}"));
        db.execute("CREATE TABLE keys (k FLOAT)", ()).unwrap();
        db.execute("INSERT INTO keys VALUES (1.0), (2.5)", ())
            .unwrap();
        let mut got = ids(&db, statement);
        got.sort_unstable();
        assert_eq!(got, expected, "{statement}");
        if statement.starts_with("UPDATE") {
            assert_eq!(
                ids(&db, "SELECT id FROM t WHERE k = 1 ORDER BY id"),
                expected,
                "{statement}: the rows updated"
            );
        }
    }
}

/// NOT IN on the key walks the keys the transaction sees, whatever they
/// are: sparse, negative, with a member gone since, under LIMIT and
/// OFFSET, and inside a transaction with keys of its own
#[test]
fn a_negated_key_set_walks_the_keys_that_are() {
    let db = Database::open("memory://pk_in_lookup_negated_keys").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (10, 1), (20, 2), (30, 3), (-5, 4)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE keys (k FLOAT)", ()).unwrap();
    db.execute("INSERT INTO keys VALUES (10.0), (20.5)", ())
        .unwrap();
    let sql = "SELECT id FROM t WHERE id NOT IN (SELECT k FROM keys) ORDER BY id";
    assert_eq!(ids(&db, sql), vec![-5, 20, 30]);
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE id NOT IN (SELECT k FROM keys) ORDER BY id LIMIT 1 OFFSET 1"
        ),
        vec![20]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE id NOT IN (SELECT k FROM keys) LIMIT 2"
        ),
        vec![-5, 20]
    );
    db.execute("DELETE FROM t WHERE id = 20", ()).unwrap();
    assert_eq!(ids(&db, sql), vec![-5, 30]);
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (40, 5)", ()).unwrap();
    tx.execute("DELETE FROM t WHERE id = -5", ()).unwrap();
    let mut seen: Vec<i64> = tx
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![30, 40], "the transaction's own keys");
    tx.rollback().unwrap();
    assert_eq!(ids(&db, sql), vec![-5, 30]);
}
