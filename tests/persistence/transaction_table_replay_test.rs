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

//! A table a transaction creates comes back from the log with the commit
//! that made it and with nothing else, and a rolled-back DROP keeps its table

use std::path::Path;

use stoolap::Database;

fn dsn(path: &Path) -> String {
    format!("file://{}?checkpoint_interval=3600", path.display())
}

/// The database files as they stand, as a crash would leave them
fn crash_image(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            crash_image(&entry.path(), &target);
        } else if entry.file_name() != "db.lock" {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

const CREATE: &str = "CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER UNIQUE, c INTEGER)";

/// Statements, then the rows of t after a crash (None: no table t), then
/// statements the replayed t must refuse
const CASES: &[(&[&str], Option<i64>, &[&str])] = &[
    (
        &["BEGIN", CREATE, "INSERT INTO t VALUES (1, 1, 1)", "COMMIT"],
        Some(1),
        &["INSERT INTO t VALUES (2, 1, 2)"],
    ),
    (
        &[
            "BEGIN",
            CREATE,
            "CREATE UNIQUE INDEX tc ON t (c)",
            "INSERT INTO t VALUES (1, 1, 1)",
            "COMMIT",
        ],
        Some(1),
        &["INSERT INTO t VALUES (2, 2, 1)"],
    ),
    (
        &["BEGIN", CREATE, "INSERT INTO t VALUES (1, 1, 1)"],
        None,
        &[],
    ),
    (
        &[
            "BEGIN",
            CREATE,
            "INSERT INTO t VALUES (1, 1, 1)",
            "ROLLBACK",
        ],
        None,
        &[],
    ),
    (
        &[
            "BEGIN",
            "SAVEPOINT s",
            CREATE,
            "ROLLBACK TO SAVEPOINT s",
            "COMMIT",
        ],
        None,
        &[],
    ),
    (
        &[
            CREATE,
            "INSERT INTO t VALUES (1, 1, 1)",
            "BEGIN",
            "DROP TABLE t",
            "ROLLBACK",
        ],
        Some(0),
        &[],
    ),
    (
        &[
            "BEGIN",
            CREATE,
            "SAVEPOINT s",
            "DROP TABLE t",
            "ROLLBACK TO SAVEPOINT s",
        ],
        None,
        &[],
    ),
    (
        &[
            "BEGIN",
            CREATE,
            "SAVEPOINT s",
            "DROP TABLE t",
            "ROLLBACK TO SAVEPOINT s",
            "COMMIT",
        ],
        Some(0),
        &[],
    ),
    (&["BEGIN", CREATE, "DROP TABLE t", "ROLLBACK"], None, &[]),
    (
        &[
            "BEGIN",
            "SAVEPOINT s",
            CREATE,
            "DROP TABLE t",
            "ROLLBACK TO SAVEPOINT s",
            "COMMIT",
        ],
        None,
        &[],
    ),
];

#[test]
fn a_transactions_table_is_replayed_as_the_transaction_ended() {
    let mut wrong = Vec::new();
    for (statements, rows, refused) in CASES {
        let dir = tempfile::tempdir().unwrap();
        let image = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(dir.path())).unwrap();
        for sql in *statements {
            db.execute(sql, ()).unwrap();
        }
        let ended = statements
            .last()
            .is_some_and(|sql| *sql == "COMMIT" || *sql == "ROLLBACK")
            || !statements.contains(&"BEGIN");
        let live = db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ()).ok();
        if ended && live != *rows {
            wrong.push(format!("{statements:?}: live rows {live:?}, want {rows:?}"));
        }
        crash_image(dir.path(), image.path());
        drop(db);

        let db = Database::open(&dsn(image.path())).unwrap();
        let replayed = db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ()).ok();
        if replayed != *rows {
            wrong.push(format!("{statements:?}: rows {replayed:?}, want {rows:?}"));
        }
        for sql in *refused {
            if db.execute(sql, ()).is_ok() {
                wrong.push(format!("{statements:?}: accepted {sql}"));
            }
        }
        db.close().unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_checkpoint_leaves_an_open_transactions_table_to_its_end() {
    let insert = "INSERT INTO t VALUES (1, 1, 1)";
    // Statements before the checkpoint, after it, and the rows of t after a crash
    let cases: &[(&[&str], &[&str], Option<i64>)] = &[
        (&["BEGIN", CREATE, insert], &[], None),
        (&["BEGIN", CREATE, insert], &["COMMIT"], Some(1)),
        (&["BEGIN", CREATE, insert], &["ROLLBACK"], None),
        (
            &["BEGIN", CREATE],
            &["COMMIT", "PRAGMA CHECKPOINT"],
            Some(0),
        ),
        (
            &["BEGIN", "SAVEPOINT s", CREATE, "ROLLBACK TO SAVEPOINT s"],
            &[],
            None,
        ),
    ];
    let mut wrong = Vec::new();
    for (before, after, rows) in cases {
        let dir = tempfile::tempdir().unwrap();
        let image = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(dir.path())).unwrap();
        let other = db.clone();
        for sql in *before {
            db.execute(sql, ()).unwrap();
        }
        other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        for sql in *after {
            db.execute(sql, ()).unwrap();
        }
        crash_image(dir.path(), image.path());
        drop(other);
        drop(db);

        let db = Database::open(&dsn(image.path())).unwrap();
        let replayed = db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ()).ok();
        if replayed != *rows {
            wrong.push(format!(
                "{before:?} checkpoint {after:?}: rows {replayed:?}, want {rows:?}"
            ));
        }
        db.close().unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Whether a checkpoint on `other` publishes a new recovery boundary
fn publishes_boundary(dir: &Path, other: &Database) -> bool {
    let meta = dir.join("wal").join("checkpoint.meta");
    let before = std::fs::read(&meta).ok();
    other.execute("PRAGMA CHECKPOINT", ()).unwrap();
    std::fs::read(&meta).ok() != before
}

#[test]
fn a_checkpoint_moves_the_boundary_once_a_creating_transaction_ends() {
    let ends: &[&[&str]] = &[
        &["COMMIT"],
        &["ROLLBACK"],
        &[
            "SAVEPOINT s",
            "CREATE TABLE u (id INTEGER PRIMARY KEY)",
            "ROLLBACK TO SAVEPOINT s",
            "COMMIT",
        ],
    ];
    let mut wrong = Vec::new();
    for end in ends {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(dir.path())).unwrap();
        let other = db.clone();
        db.execute("BEGIN", ()).unwrap();
        db.execute(CREATE, ()).unwrap();
        if publishes_boundary(dir.path(), &other) {
            wrong.push(format!(
                "{end:?}: boundary moved while the transaction was open"
            ));
        }
        for sql in *end {
            db.execute(sql, ()).unwrap();
        }
        if !publishes_boundary(dir.path(), &other) {
            wrong.push(format!(
                "{end:?}: boundary held after the transaction ended"
            ));
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    let other = db.clone();
    db.execute("BEGIN", ()).unwrap();
    db.execute(CREATE, ()).unwrap();
    drop(db);
    if !publishes_boundary(dir.path(), &other) {
        wrong.push("dropped transaction: boundary held".to_string());
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
