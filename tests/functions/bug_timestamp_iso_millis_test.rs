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

//! Regression test: ISO timestamp with milliseconds + Z suffix was rejected.
//! Format: 2024-01-15T14:30:00.000Z

use stoolap::Database;

#[test]
fn test_timestamp_iso_millis_z() {
    let db = Database::open("memory://ts_millis_z").unwrap();

    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts TIMESTAMP NOT NULL)",
        (),
    )
    .unwrap();

    // This format was previously rejected
    db.execute(
        "INSERT INTO events VALUES (1, '2024-01-15T14:30:00.000Z')",
        (),
    )
    .unwrap();

    let rows = db.query("SELECT ts FROM events WHERE id = 1", ()).unwrap();
    let ts: String = rows.into_iter().next().unwrap().unwrap().get(0).unwrap();
    assert!(ts.contains("2024-01-15"), "Expected date part, got: {}", ts);
    assert!(ts.contains("14:30:00"), "Expected time part, got: {}", ts);
}

#[test]
fn test_timestamp_iso_millis_no_tz() {
    let db = Database::open("memory://ts_millis_no_tz").unwrap();

    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts TIMESTAMP NOT NULL)",
        (),
    )
    .unwrap();

    // ISO with T separator + fractional seconds, no timezone
    db.execute(
        "INSERT INTO events VALUES (1, '2024-01-15T14:30:00.000')",
        (),
    )
    .unwrap();

    let rows = db.query("SELECT ts FROM events WHERE id = 1", ()).unwrap();
    let ts: String = rows.into_iter().next().unwrap().unwrap().get(0).unwrap();
    assert!(ts.contains("2024-01-15"), "Expected date part, got: {}", ts);
    assert!(ts.contains("14:30:00"), "Expected time part, got: {}", ts);
}

#[test]
fn test_timestamp_iso_micros_z() {
    let db = Database::open("memory://ts_micros_z").unwrap();

    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts TIMESTAMP NOT NULL)",
        (),
    )
    .unwrap();

    // Microsecond precision with Z
    db.execute(
        "INSERT INTO events VALUES (1, '2024-01-15T14:30:00.123456Z')",
        (),
    )
    .unwrap();

    let rows = db.query("SELECT ts FROM events WHERE id = 1", ()).unwrap();
    let ts: String = rows.into_iter().next().unwrap().unwrap().get(0).unwrap();
    assert!(ts.contains("2024-01-15"), "Expected date part, got: {}", ts);
}

#[test]
fn test_timestamp_all_formats_roundtrip() {
    let db = Database::open("memory://ts_all_formats").unwrap();

    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts TIMESTAMP NOT NULL)",
        (),
    )
    .unwrap();

    let formats = [
        (1, "2024-01-15T14:30:00.000Z"), // ms + Z (was broken)
        (2, "2024-01-15T14:30:00.000"),  // ms + T, no tz (was broken)
        (3, "2024-01-15T14:30:00Z"),     // no ms + Z (worked)
        (4, "2024-01-15 14:30:00.000"),  // ms, space separator (worked)
        (5, "2024-01-15T14:30:00"),      // no ms, no tz (worked)
        (6, "2024-01-15 14:30:00"),      // SQL style (worked)
    ];

    let stmt = db.prepare("INSERT INTO events VALUES ($1, $2)").unwrap();

    for (id, ts) in &formats {
        stmt.execute((*id as i64, *ts)).unwrap();
    }

    let rows = db.query("SELECT COUNT(*) FROM events", ()).unwrap();
    let count: i64 = rows.into_iter().next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(count, formats.len() as i64);
}
