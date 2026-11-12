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
