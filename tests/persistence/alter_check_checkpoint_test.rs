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

//! A function an ALTER TABLE's CHECK calls starts no checkpoint, snapshot
//! or restore, so the catalog is never written as the statement has it
//! before its record: a crash after the statement recovers what it answered

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::Database;

// One test runs at a time
static SERIAL: Mutex<()> = Mutex::new(());

/// The database the CHECK's function writes through, a handle of its own
static NESTED: Mutex<Option<Database>> = Mutex::new(None);
/// What each statement the function ran answered
static ANSWERS: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Whether the function lets the row pass
static PASSES: AtomicBool = AtomicBool::new(true);

const WRITES: [&str; 3] = ["PRAGMA CHECKPOINT", "PRAGMA SNAPSHOT", "PRAGMA RESTORE"];

/// Tries a checkpoint, a snapshot and a restore, then judges its row
#[derive(Default)]
struct WritesTheCatalog;

impl ScalarFunction for WritesTheCatalog {
    fn name(&self) -> &str {
        "ALTER_CHECK_WRITES_THE_CATALOG"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ALTER_CHECK_WRITES_THE_CATALOG",
            FunctionType::Scalar,
            "tries a checkpoint, a snapshot and a restore, then judges its row",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, _args: &[Value]) -> stoolap::core::Result<Value> {
        let nested = NESTED.lock().unwrap_or_else(|e| e.into_inner());
        let mut answers = ANSWERS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(db) = nested.as_ref() {
            for sql in WRITES {
                answers.push(match db.execute(sql, ()) {
                    Ok(_) => format!("{sql} ran"),
                    Err(error) => error.to_string(),
                });
            }
        }
        Ok(Value::Boolean(PASSES.load(Ordering::SeqCst)))
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(WritesTheCatalog)
    }
}

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
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

fn text(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

/// Runs `alter` over a sealed table whose CHECK function lets the rows pass
/// or not, and returns what the statement answered and what a crash right
/// after it recovers: the table's definition and its rows
fn after_a_crash(alter: &str, passes: bool) -> (bool, Vec<String>, Vec<String>) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = global_registry().register_scalar::<WritesTheCatalog>();
    let root = tempfile::tempdir().unwrap();
    let (first, crashed) = (root.path().join("0"), root.path().join("1"));
    let db = Database::open(&dsn(&first)).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t VALUES (1, 5), (2, 50)",
        "PRAGMA CHECKPOINT",
        "PRAGMA SNAPSHOT",
    ] {
        db.execute(sql, ()).unwrap();
    }
    PASSES.store(passes, Ordering::SeqCst);
    ANSWERS.lock().unwrap_or_else(|e| e.into_inner()).clear();
    *NESTED.lock().unwrap_or_else(|e| e.into_inner()) = Some(db.clone());
    let answered = db.execute(alter, ());
    NESTED.lock().unwrap_or_else(|e| e.into_inner()).take();
    let answers = std::mem::take(&mut *ANSWERS.lock().unwrap_or_else(|e| e.into_inner()));
    assert!(
        !answers.is_empty(),
        "{alter}: the CHECK's function never ran"
    );
    for (answer, sql) in answers.iter().zip(WRITES.iter().cycle()) {
        assert!(
            answer.contains("inside a function another DDL statement calls"),
            "{alter}: {sql} inside the CHECK: {answer}"
        );
    }
    assert_eq!(answered.is_ok(), passes, "{alter}: {answered:?}");

    crash_image(&first, &crashed);
    let recovered = Database::open(&dsn(&crashed))
        .unwrap_or_else(|e| panic!("{alter}, passes {passes}: the crash does not recover: {e}"));
    (
        passes,
        text(&recovered, "SHOW CREATE TABLE t"),
        text(&recovered, "SELECT * FROM t ORDER BY id"),
    )
}

const ADD: &str =
    "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 4 CHECK (ALTER_CHECK_WRITES_THE_CATALOG(z))";
const MODIFY: &str = "ALTER TABLE t MODIFY COLUMN v TEXT CHECK (ALTER_CHECK_WRITES_THE_CATALOG(v))";

#[test]
fn an_added_column_whose_check_tried_a_checkpoint_recovers_once() {
    assert_eq!(
        after_a_crash(ADD, true),
        (
            true,
            vec!["t|CREATE TABLE t (id Integer PRIMARY KEY, v Integer, z Integer DEFAULT 4 CHECK (ALTER_CHECK_WRITES_THE_CATALOG(z)))".to_string()],
            vec!["1|5|4".to_string(), "2|50|4".to_string()],
        )
    );
}

#[test]
fn a_refused_column_whose_check_tried_a_checkpoint_does_not_come_back() {
    assert_eq!(
        after_a_crash(ADD, false),
        (
            false,
            vec!["t|CREATE TABLE t (id Integer PRIMARY KEY, v Integer)".to_string()],
            vec!["1|5".to_string(), "2|50".to_string()],
        )
    );
}

#[test]
fn a_modified_column_whose_check_tried_a_checkpoint_recovers_as_changed() {
    assert_eq!(
        after_a_crash(MODIFY, true),
        (
            true,
            vec!["t|CREATE TABLE t (id Integer PRIMARY KEY, v Text CHECK (ALTER_CHECK_WRITES_THE_CATALOG(v)))".to_string()],
            vec!["1|5".to_string(), "2|50".to_string()],
        )
    );
}

#[test]
fn a_refused_modify_whose_check_tried_a_checkpoint_recovers_unchanged() {
    assert_eq!(
        after_a_crash(MODIFY, false),
        (
            false,
            vec!["t|CREATE TABLE t (id Integer PRIMARY KEY, v Integer)".to_string()],
            vec!["1|5".to_string(), "2|50".to_string()],
        )
    );
}
