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

//! The last handle of a database lets other databases open while its
//! engine closes

#![cfg(feature = "test-failpoints")]

use std::sync::mpsc;
use std::time::Duration;

use stoolap::test_failpoints::{self as fp, FailpointGuard};
use stoolap::Database;

#[test]
fn another_database_opens_while_the_last_handle_of_one_closes() {
    let _guard = FailpointGuard::new();
    let closing = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", closing.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    let dsn = format!("file://{}", other.path().display());
    let (seen, outcome) = mpsc::channel();
    fp::after_checkpoint_managers_taken(move || {
        let (opened, open) = mpsc::channel();
        std::thread::spawn(move || opened.send(Database::open(&dsn).is_ok()));
        let _ = seen.send(open.recv_timeout(Duration::from_secs(5)));
    });
    drop(db);
    assert_eq!(
        outcome.recv().unwrap(),
        Ok(true),
        "another database opens while the closing checkpoint runs"
    );
}
