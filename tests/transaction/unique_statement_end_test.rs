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

//! Inside a transaction, each statement's new unique keys are judged when
//! the statement ends, against the rows as the transaction then sees them:
//! a statement that breaks a key fails and is undone, and the transaction
//! goes on

use stoolap::Database;

/// Where the rows (1, 10) and (2, 20) of t live
#[derive(Clone, Copy, Debug)]
enum Rows {
    Hot,
    Sealed,
    /// Row 1 sealed, row 2 hot
    Mixed,
}

/// t(id, u UNIQUE, w) and s(id, a, b, UNIQUE (a, b)), laid out as `rows` says
fn database(rows: Rows, dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, u INTEGER UNIQUE, w INTEGER)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, UNIQUE (a, b))",
        "INSERT INTO t VALUES (1, 10, 0)",
        "INSERT INTO s VALUES (1, 1, 'x')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if matches!(rows, Rows::Mixed) {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    for sql in [
        "INSERT INTO t VALUES (2, 20, 0)",
        "INSERT INTO s VALUES (2, 1, 'y')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if matches!(rows, Rows::Sealed) {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each statement's outcome, `ok`, `taken` for a unique error or `twice`
/// for an upsert that meets a row it wrote, then the rows of t
fn run(db: &Database, steps: &[&str]) -> String {
    let outcomes: Vec<&str> = steps
        .iter()
        .map(|sql| match db.execute(sql, ()) {
            Ok(_) => "ok",
            Err(e) if e.to_string().starts_with("unique constraint failed") => "taken",
            Err(e) if e.to_string().contains("cannot affect a row a second time") => "twice",
            Err(e) => panic!("{sql}: {e}"),
        })
        .collect();
    let rows: Vec<String> = db
        .query("SELECT id, u FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            let u = row.get::<Option<i64>>(1).unwrap();
            let u = u.map_or("NULL".to_string(), |u| u.to_string());
            format!("{}:{u}", row.get::<i64>(0).unwrap())
        })
        .collect();
    format!("{} => {}", outcomes.join(" "), rows.join(","))
}

const CASES: &[(&[&str], &str)] = &[
    (
        &[
            "BEGIN",
            "UPDATE t SET u = 20 WHERE id = 1",
            "INSERT INTO t VALUES (3, 30, 0)",
            "COMMIT",
        ],
        "ok taken ok ok => 1:10,2:20,3:30",
    ),
    (
        &[
            "BEGIN",
            "UPDATE t SET u = 20 WHERE id = 1",
            "UPDATE t SET u = 10 WHERE id = 2",
            "COMMIT",
        ],
        "ok taken taken ok => 1:10,2:20",
    ),
    (
        &[
            "BEGIN",
            "UPDATE t SET u = CASE id WHEN 1 THEN 20 WHEN 2 THEN 10 END",
            "COMMIT",
        ],
        "ok ok ok => 1:20,2:10",
    ),
    (
        &[
            "BEGIN",
            "UPDATE t SET u = u + 10",
            "UPDATE t SET u = u - 10",
            "COMMIT",
        ],
        "ok ok ok ok => 1:10,2:20",
    ),
    (
        &["BEGIN", "UPDATE t SET u = 30", "COMMIT"],
        "ok taken ok => 1:10,2:20",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (3, 30, 0)",
            "INSERT INTO t VALUES (4, 30, 0)",
            "COMMIT",
        ],
        "ok ok taken ok => 1:10,2:20,3:30",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (3, 30, 0), (4, 30, 0)",
            "INSERT INTO t VALUES (4, 30, 0)",
            "COMMIT",
        ],
        "ok taken ok ok => 1:10,2:20,4:30",
    ),
    (
        &[
            "BEGIN",
            "UPDATE t SET u = 30 WHERE id = 1",
            "INSERT INTO t VALUES (3, 30, 0)",
            "COMMIT",
        ],
        "ok ok taken ok => 1:30,2:20",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (3, 30, 0)",
            "UPDATE t SET u = 30 WHERE id = 1",
            "COMMIT",
        ],
        "ok ok taken ok => 1:10,2:20,3:30",
    ),
    (
        &[
            "BEGIN",
            "UPDATE t SET u = 30 WHERE id = 1",
            "INSERT INTO t VALUES (3, 10, 0)",
            "COMMIT",
        ],
        "ok ok ok ok => 1:30,2:20,3:10",
    ),
    (
        &[
            "BEGIN",
            "DELETE FROM t WHERE id = 2",
            "UPDATE t SET u = 20 WHERE id = 1",
            "COMMIT",
        ],
        "ok ok ok ok => 1:20",
    ),
    (
        &[
            "BEGIN",
            "UPDATE t SET w = 1",
            "UPDATE t SET u = 20 WHERE id = 1",
            "COMMIT",
        ],
        "ok ok taken ok => 1:10,2:20",
    ),
    (
        &[
            "BEGIN",
            "SAVEPOINT a",
            "INSERT INTO t VALUES (3, 30, 0)",
            "ROLLBACK TO SAVEPOINT a",
            "INSERT INTO t VALUES (4, 30, 0)",
            "COMMIT",
        ],
        "ok ok ok ok ok ok => 1:10,2:20,4:30",
    ),
    (
        &[
            "BEGIN",
            "SAVEPOINT a",
            "UPDATE t SET u = 30 WHERE id = 1",
            "ROLLBACK TO SAVEPOINT a",
            "INSERT INTO t VALUES (3, 30, 0)",
            "INSERT INTO t VALUES (4, 10, 0)",
            "COMMIT",
        ],
        "ok ok ok ok ok taken ok => 1:10,2:20,3:30",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (3, NULL, 0), (4, NULL, 0)",
            "UPDATE t SET u = NULL WHERE id = 1",
            "COMMIT",
        ],
        "ok ok ok ok => 1:NULL,2:20,3:NULL,4:NULL",
    ),
    (
        &["BEGIN", "UPDATE t SET u = 20 WHERE id = 1", "ROLLBACK"],
        "ok taken ok => 1:10,2:20",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (3, 30, 0)",
            "INSERT INTO t VALUES (4, 30, 0) ON CONFLICT (u) DO UPDATE SET u = 31",
            "COMMIT",
        ],
        "ok ok ok ok => 1:10,2:20,3:31",
    ),
    (
        &["INSERT INTO t VALUES (20, 200, 0), (21, 200, 0) ON CONFLICT (u) DO NOTHING"],
        "ok => 1:10,2:20,20:200",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (20, 200, 0), (21, 200, 0) ON CONFLICT (u) DO NOTHING",
            "COMMIT",
        ],
        "ok ok ok => 1:10,2:20,20:200",
    ),
    (
        &["INSERT INTO t VALUES (11, 110, 0), (11, 111, 0) ON CONFLICT (id) DO UPDATE SET u = 112"],
        "twice => 1:10,2:20",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (5, 50, 0), (5, 51, 0) ON CONFLICT (id) DO UPDATE SET u = 10",
            "COMMIT",
        ],
        "ok twice ok => 1:10,2:20",
    ),
    (
        &["INSERT INTO t VALUES (11, 110, 0), (11, 111, 0) ON DUPLICATE KEY UPDATE u = 112"],
        "ok => 1:10,2:20,11:112",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (5, 50, 0)",
            "SAVEPOINT s",
            "DELETE FROM t WHERE id = 5",
            "INSERT INTO t VALUES (6, 50, 0)",
            "ROLLBACK TO SAVEPOINT s",
            "INSERT INTO t VALUES (7, 50, 0)",
            "COMMIT",
        ],
        "ok ok ok ok ok ok taken ok => 1:10,2:20,5:50",
    ),
    (
        &[
            "BEGIN",
            "INSERT INTO t VALUES (1, 30, 0), (5, 30, 0) ON DUPLICATE KEY UPDATE u = excluded.u",
            "COMMIT",
        ],
        "ok ok ok => 1:30,2:20",
    ),
    (
        &["INSERT INTO t VALUES (1, 30, 0), (5, 30, 0) ON DUPLICATE KEY UPDATE u = excluded.u"],
        "ok => 1:30,2:20",
    ),
];

#[test]
fn a_transaction_s_statement_is_judged_by_the_keys_it_leaves() {
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        for (steps, want) in CASES {
            let dir = tempfile::tempdir().unwrap();
            let db = database(rows, &dir);
            let got = run(&db, steps);
            if got != *want {
                wrong.push(format!("{rows:?} {steps:?}\n  got  {got}\n  want {want}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_installed_transaction_is_judged_at_the_statement_end() {
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
        executor.install_transaction(executor.begin_transaction().unwrap());
        let updated = executor.execute("UPDATE t SET u = 20 WHERE id = 1");
        if !matches!(updated, Err(stoolap::Error::UniqueConstraint { .. })) {
            wrong.push(format!("{rows:?}: {:?}", updated.map(|_| ())));
        }
        if let Some(mut tx) = executor.take_transaction() {
            tx.rollback().unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_installed_transaction_judges_by_the_rows_it_wrote_before() {
    use stoolap::core::{Row, Value};
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
        let tx = executor.begin_transaction().unwrap();
        let mut table = tx.get_table("t").unwrap();
        let row = [3, 30, 0].map(Value::Integer).to_vec();
        table.insert(Row::from(row)).unwrap();
        drop(table);
        executor.install_transaction(tx);
        let inserted = executor.execute("INSERT INTO t VALUES (4, 30, 0)");
        if !matches!(inserted, Err(stoolap::Error::UniqueConstraint { .. })) {
            wrong.push(format!("{rows:?}: {:?}", inserted.map(|_| ())));
        }
        if let Some(mut tx) = executor.take_transaction() {
            tx.rollback().unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_upsert_updates_the_negative_row_holding_its_key() {
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        for sql in [
            "BEGIN",
            "UPDATE t SET u = 30 WHERE id = 1",
            "INSERT INTO t VALUES (-2, 10, 0)",
            "INSERT INTO t VALUES (-3, 10, 7) ON CONFLICT (u) DO UPDATE SET w = excluded.w",
            "COMMIT",
        ] {
            db.execute(sql, ()).unwrap();
        }
        let left: Vec<(i64, i64, i64)> = db
            .query("SELECT id, u, w FROM t ORDER BY id", ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                let get = |at| row.get::<i64>(at).unwrap();
                (get(0), get(1), get(2))
            })
            .collect();
        if left != [(-2, 10, 7), (1, 30, 0), (2, 20, 0)] {
            wrong.push(format!("{rows:?}: {left:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_installed_transaction_judges_the_keys_of_an_index_built_before() {
    use stoolap::core::{Row, Value};
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        db.execute("UPDATE t SET w = id", ()).unwrap();
        let executor = stoolap::Executor::new(std::sync::Arc::clone(db.engine()));
        let tx = executor.begin_transaction().unwrap();
        let mut table = tx.get_table("t").unwrap();
        let row = [3, 30, 100].map(Value::Integer).to_vec();
        table.insert(Row::from(row)).unwrap();
        table.create_index("t_w", &["w"], true).unwrap();
        drop(table);
        executor.install_transaction(tx);
        let inserted = executor.execute("INSERT INTO t VALUES (4, 30, 200)");
        if !matches!(inserted, Err(stoolap::Error::UniqueConstraint { .. })) {
            wrong.push(format!("{rows:?}: {:?}", inserted.map(|_| ())));
        }
        if let Some(mut tx) = executor.take_transaction() {
            tx.rollback().unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_unique_index_built_beside_a_transaction_judges_its_earlier_rows() {
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        db.execute("UPDATE t SET w = id", ()).unwrap();
        let other = db.clone();
        db.execute("BEGIN", ()).unwrap();
        db.execute("INSERT INTO t VALUES (3, 30, 7)", ()).unwrap();
        other
            .execute("CREATE UNIQUE INDEX t_w ON t (w)", ())
            .unwrap();
        let inserted = db.execute("INSERT INTO t VALUES (4, 40, 7)", ());
        if !matches!(inserted, Err(stoolap::Error::UniqueConstraint { .. })) {
            wrong.push(format!("{rows:?}: {:?}", inserted.map(|_| ())));
        }
        db.execute("ROLLBACK", ()).unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_composite_key_is_judged_at_the_statement_end() {
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        let steps = [
            "BEGIN",
            "UPDATE s SET b = CASE b WHEN 'x' THEN 'y' ELSE 'x' END",
            "UPDATE s SET b = 'x' WHERE id = 1",
            "INSERT INTO s VALUES (3, 1, 'z')",
            "INSERT INTO s VALUES (4, 1, 'z')",
            "INSERT INTO s VALUES (5, 2, 'z')",
            "COMMIT",
        ];
        let got: Vec<&str> = steps
            .iter()
            .map(|sql| match db.execute(sql, ()) {
                Ok(_) => "ok",
                Err(_) => "refused",
            })
            .collect();
        let left: Vec<String> = db
            .query("SELECT id, a, b FROM s ORDER BY id", ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                let (id, a) = (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap());
                format!("{id}:{a}{}", row.get::<String>(2).unwrap())
            })
            .collect();
        let want_steps = ["ok", "ok", "refused", "ok", "refused", "ok", "ok"];
        let want_rows = ["1:1y", "2:1x", "3:1z", "5:2z"];
        if got != want_steps || left != want_rows {
            wrong.push(format!("{rows:?}: {got:?} left {left:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_unique_index_a_transaction_creates_is_judged_by_its_own_rows() {
    use stoolap::core::Value;
    use stoolap::storage::traits::{Engine, ScanAccess};
    let mut wrong = Vec::new();
    for rows in [Rows::Hot, Rows::Sealed, Rows::Mixed] {
        let dir = tempfile::tempdir().unwrap();
        let db = database(rows, &dir);
        db.execute("UPDATE t SET w = id", ()).unwrap();
        let mut txn = db.engine().begin_transaction().unwrap();
        let mut table = txn.get_table("t").unwrap();
        // Rows committed with distinct values the transaction gives one value
        table
            .update(ScanAccess::Seq, None, &mut |_, mut row| {
                row.set(2, Value::Integer(7)).unwrap();
                Ok((row, true))
            })
            .unwrap();
        let created = table.create_index("t_w", &["w"], true);
        drop(table);
        txn.rollback().unwrap();
        if !matches!(created, Err(stoolap::core::Error::UniqueConstraint { .. })) {
            wrong.push(format!("{rows:?}: {created:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
