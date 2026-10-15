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

//! An equality on a column a volume holds sorted scans only that value's
//! rows, found by binary search, and answers as on an unsorted volume

use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::Database;

/// The pruning counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

/// Seven rows per key, so key 9362 spans the first two row groups
const ROWS: i64 = 140_000;

fn open(dir: &std::path::Path) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap()
}

/// `t` holds k and ts in row order, with an index on k; `u` the same rows
/// in a scattered order
fn create(db: &Database) {
    for (table, id) in [("t", "value"), ("u", "(value * 7919) % 140000")] {
        db.execute(
            &format!(
                "CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, \
                 ts TIMESTAMP NOT NULL, v INTEGER)"
            ),
            (),
        )
        .unwrap();
        if table == "t" {
            db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
        }
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT {id}, value / 7, \
                 DATE_ADD('2024-01-01 00:00:00', value / 7, 'minute'), value \
                 FROM generate_series(0, {}) ",
                ROWS - 1
            ),
            (),
        )
        .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
}

fn ids(db: &Database, sql: &str, params: Vec<Value>) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, params)
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn minute(key: i64) -> Value {
    let base = chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z").unwrap();
    Value::Timestamp((base + chrono::Duration::minutes(key)).with_timezone(&chrono::Utc))
}

const KEYS: [i64; 7] = [-1, 0, 5000, 9362, 9363, 19_999, 20_000];

fn check(db: &Database, phase: &str) {
    for key in KEYS {
        for (column, value) in [("k", Value::Integer(key)), ("ts", minute(key))] {
            let sql = format!("SELECT v FROM t WHERE {column} = $1");
            let want = ids(db, &sql.replace(" t ", " u "), vec![value.clone()]);
            assert_eq!(
                want.len(),
                if (0..20_000).contains(&key) { 7 } else { 0 },
                "{column} = {key}"
            );
            assert_eq!(
                ids(db, &sql, vec![value]),
                want,
                "{phase}: {column} = {key}"
            );
        }
    }
}

#[test]
fn an_equality_on_a_sorted_column_answers_as_on_an_unsorted_one() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db);
    check(&db, "sealed");
    drop(db);
    let db = open(dir.path());
    check(&db, "reopened");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn an_equality_on_a_sorted_column_scans_only_its_rows() {
    use stoolap::test_failpoints::volume_rows_left;
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db);
    for (column, value) in [("k", Value::Integer(9362)), ("ts", minute(9362))] {
        let (before, probed) = (volume_rows_left(), probes(&db));
        let got = ids(
            &db,
            &format!("SELECT v FROM t WHERE {column} = $1"),
            vec![value],
        );
        assert_eq!(got.len(), 7);
        assert_eq!(
            volume_rows_left() - before,
            7,
            "{column} = 9362 scans its seven rows, across the group boundary"
        );
        assert_eq!(
            probes(&db),
            probed,
            "{column}: the sorted range is read, not the side file"
        );
    }
}

#[cfg(feature = "test-failpoints")]
fn probes(db: &Database) -> i64 {
    let rows = db.query("PRAGMA INDEX_READ_STATS", ()).unwrap();
    let at = rows.columns().iter().position(|c| c == "probes").unwrap();
    rows.into_iter().next().unwrap().unwrap().get(at).unwrap()
}
