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

//! An UPDATE or DELETE without a subquery runs in the prepared family and
//! changes the rows, reports the count and fails as the general path does,
//! hot and sealed, in a transaction or not

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering::Relaxed;
use std::sync::Mutex;

use stoolap::test_failpoints::{prepared_dml_site_runs, PreparedDmlSite, PREPARED_FAMILIES_OFF};
use stoolap::{Database, Value};

/// The switch is process-wide: one test runs with it at a time
static SWITCH: Mutex<()> = Mutex::new(());

/// Each statement with its parameters
const RUN: &[(&str, &[i64])] = &[
    ("UPDATE t SET b = 1 WHERE id = 5", &[]),
    ("UPDATE t SET b = b + 1 WHERE a = 3", &[]),
    (
        "UPDATE t SET b = b * 2, s = 'r' WHERE id > 10 AND id < 20",
        &[],
    ),
    ("UPDATE t SET f = f + 0.5 WHERE s = 'x7'", &[]),
    ("UPDATE t SET s = s || '!' WHERE s LIKE 'x1%'", &[]),
    ("UPDATE t SET n = 1 WHERE n IS NULL", &[]),
    ("UPDATE t SET b = 0 WHERE a IN (1, 2) OR f > 30", &[]),
    ("UPDATE t SET b = -b WHERE NOT (a = 4)", &[]),
    (
        "UPDATE t SET a = a + $1 WHERE id BETWEEN $2 AND 30",
        &[10, 25],
    ),
    ("UPDATE t SET b = NULL WHERE id = 40", &[]),
    ("UPDATE t SET f = 1.5", &[]),
    ("UPDATE t SET b = 9 WHERE 1 = 0", &[]),
    ("UPDATE t SET b = 9 WHERE a = 99", &[]),
    ("UPDATE t SET u = 7 WHERE id = 8", &[]),
    ("UPDATE t SET u = 1 WHERE id = 3", &[]),
    ("UPDATE t SET c = NULL WHERE id = 6", &[]),
    ("UPDATE t SET b = 9223372036854775807 + b WHERE id = 2", &[]),
    ("UPDATE t SET b = 9223372036854775807 + 1 WHERE a = 99", &[]),
    ("UPDATE t SET b = 9223372036854775807 + 1 WHERE id = 99", &[]),
    ("UPDATE t SET b = 1 / 0 WHERE s = 'none'", &[]),
    ("UPDATE t SET b = 9223372036854775807 + 1 WHERE a = 1", &[]),
    (
        "INSERT INTO t VALUES (99, 1, 1, 1.0, 'n', 1, 99, 1) ON CONFLICT (id) DO UPDATE SET b = 9223372036854775807 + 1",
        &[],
    ),
    (
        "INSERT INTO t VALUES (1, 1, 1, 1.0, 'n', 1, 98, 1) ON CONFLICT (id) DO UPDATE SET b = 9223372036854775807 + 1",
        &[],
    ),
    ("DELETE FROM t WHERE id = 45", &[]),
    ("DELETE FROM t WHERE a = 6", &[]),
    ("DELETE FROM t WHERE s = 'x9' OR f < 1", &[]),
    ("DELETE FROM t WHERE id > $1 AND b IS NOT NULL", &[46]),
    ("DELETE FROM t WHERE a = 99", &[]),
];

/// t with 50 rows; sealed into a volume when `cold`
fn table(name: &str, cold: bool, dir: &tempfile::TempDir) -> Database {
    let db = match cold {
        true => Database::open(&format!(
            "file://{}?checkpoint_interval=3600&compact_threshold=1000",
            dir.path().display()
        )),
        false => Database::open(&format!("memory://{name}")),
    }
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, f FLOAT, s TEXT, \
         n INTEGER, u INTEGER UNIQUE, c INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_a ON t(a)", ()).unwrap();
    for id in 1..=50i64 {
        let n = if id % 4 == 0 {
            Value::null_unknown()
        } else {
            Value::Integer(id)
        };
        db.execute(
            "INSERT INTO t VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            vec![
                Value::Integer(id),
                Value::Integer(id % 10),
                Value::Integer(id * 3),
                Value::Float(id as f64 * 0.75),
                Value::text(format!("x{id}")),
                n,
                Value::Integer(id),
                Value::Integer(1),
            ],
        )
        .unwrap();
    }
    if cold {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each statement's count or error, then the table it leaves
fn answers(name: &str, family: bool, cold: bool, in_transaction: bool) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    let db = table(name, cold, &dir);
    PREPARED_FAMILIES_OFF.store(!family, Relaxed);
    let mut answers = Vec::new();
    for (sql, params) in RUN {
        if in_transaction {
            db.execute("BEGIN", ()).unwrap();
        }
        let params: Vec<Value> = params.iter().map(|v| Value::Integer(*v)).collect();
        answers.push(match db.execute(sql, params) {
            Ok(changed) => format!("{sql}: changed {changed}"),
            Err(error) => format!("{sql}: error {error}"),
        });
        if in_transaction {
            if let Err(error) = db.execute("COMMIT", ()) {
                answers.push(format!("{sql}: commit error {error}"));
                let _ = db.execute("ROLLBACK", ());
            }
        }
    }
    PREPARED_FAMILIES_OFF.store(false, Relaxed);
    let rows = db.query("SELECT * FROM t ORDER BY id", ()).unwrap();
    for row in rows {
        answers.push(format!("{:?}", row.unwrap().as_row()));
    }
    answers
}

#[test]
fn a_plain_update_or_delete_answers_in_the_family_as_on_the_general_path() {
    let _serial = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let set_before = prepared_dml_site_runs(PreparedDmlSite::UpdateSet);
    for (cold, in_transaction) in [(false, false), (true, false), (false, true), (true, true)] {
        let name = format!("plain_dml_family_{cold}_{in_transaction}");
        let general_before = prepared_dml_site_runs(PreparedDmlSite::UpdateSet);
        let general = answers(&format!("{name}_general"), false, cold, in_transaction);
        assert_eq!(
            prepared_dml_site_runs(PreparedDmlSite::UpdateSet),
            general_before,
            "the general path runs no prepared SET"
        );
        let family = answers(&format!("{name}_family"), true, cold, in_transaction);
        let differ: Vec<String> = general
            .iter()
            .zip(&family)
            .filter(|(g, f)| g != f)
            .map(|(g, f)| format!("  general {g}\n  family  {f}"))
            .collect();
        assert!(
            differ.is_empty() && general.len() == family.len(),
            "hot/sealed {cold}, in a transaction {in_transaction}:\n{}",
            differ.join("\n")
        );
    }
    assert!(
        prepared_dml_site_runs(PreparedDmlSite::UpdateSet) > set_before,
        "a plain UPDATE ran its SET in the family"
    );
}
