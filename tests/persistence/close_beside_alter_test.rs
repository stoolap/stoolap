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

//! A close from another thread waits for a DDL statement in flight, so the
//! next open finds what the statement answered

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Duration;

use stoolap::core::Value;
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::Database;

// One test runs at a time
static SERIAL: Mutex<()> = Mutex::new(());

/// The handle another thread closes while the CHECK judges
static CLOSED_BESIDE: Mutex<Option<Database>> = Mutex::new(None);
/// The closing thread, once the CHECK's function has started it
static CLOSER: Mutex<Option<JoinHandle<stoolap::Result<()>>>> = Mutex::new(None);
/// Whether the function lets the row pass
static PASSES: AtomicBool = AtomicBool::new(true);

/// Starts a close on another thread, gives it time to begin, then judges
#[derive(Default)]
struct ClosedBeside;

impl ScalarFunction for ClosedBeside {
    fn name(&self) -> &str {
        "ALTER_CHECK_CLOSED_BESIDE"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ALTER_CHECK_CLOSED_BESIDE",
            FunctionType::Scalar,
            "starts a close on another thread, then judges its row",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, _args: &[Value]) -> stoolap::core::Result<Value> {
        let handle = CLOSED_BESIDE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(db) = handle {
            let closer = std::thread::spawn(move || db.close());
            *CLOSER.lock().unwrap_or_else(|e| e.into_inner()) = Some(closer);
            std::thread::sleep(Duration::from_millis(300));
        }
        Ok(Value::Boolean(PASSES.load(Ordering::SeqCst)))
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ClosedBeside)
    }
}

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?checkpoint_interval=3600&compact_threshold=1000",
        dir.display()
    )
}

/// Runs `alter` over a sealed table while another thread closes the
/// database, and returns whether the statement was taken and the table's
/// definition at the next open
fn reopened_after(alter: &str, passes: bool) -> (bool, String) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = global_registry().register_scalar::<ClosedBeside>();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t VALUES (1, 5), (2, 50)",
        "PRAGMA CHECKPOINT",
    ] {
        db.execute(sql, ()).unwrap();
    }
    PASSES.store(passes, Ordering::SeqCst);
    *CLOSED_BESIDE.lock().unwrap_or_else(|e| e.into_inner()) = Some(db.clone());
    let taken = db.execute(alter, ()).is_ok();
    let closer = CLOSER.lock().unwrap_or_else(|e| e.into_inner()).take();
    closer
        .expect("the CHECK's function never ran")
        .join()
        .unwrap()
        .unwrap();
    drop(db);

    let reopened = Database::open(&dsn(dir.path())).unwrap();
    let definition = reopened
        .query("SHOW CREATE TABLE t", ())
        .unwrap()
        .map(|row| row.unwrap().get_value(1).unwrap().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    (taken, definition)
}

const ADD: &str =
    "ALTER TABLE t ADD COLUMN z INTEGER DEFAULT 4 CHECK (ALTER_CHECK_CLOSED_BESIDE(z))";

#[test]
fn a_column_refused_while_another_thread_closes_does_not_come_back() {
    assert_eq!(
        reopened_after(ADD, false),
        (
            false,
            "CREATE TABLE t (id Integer PRIMARY KEY, v Integer)".to_string()
        )
    );
}

#[test]
fn a_column_added_while_another_thread_closes_is_there_at_the_next_open() {
    assert_eq!(
        reopened_after(ADD, true),
        (
            true,
            "CREATE TABLE t (id Integer PRIMARY KEY, v Integer, z Integer DEFAULT 4 CHECK (ALTER_CHECK_CLOSED_BESIDE(z)))".to_string()
        )
    );
}
