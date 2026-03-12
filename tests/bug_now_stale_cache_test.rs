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

//! Tests for NOW()/CURRENT_TIMESTAMP staleness with semantic cache and constant folding.

use stoolap::Database;

/// Two sequential NOW() calls must return different timestamps when time has elapsed.
/// Uses EXTRACT(MICROSECOND ...) for sub-second precision comparison.
#[test]
fn test_now_not_stale_across_queries() {
    let db = Database::open("memory://now_stale").unwrap();

    // Extract microsecond component — changes within the same second
    let us1: i64 = db
        .query("SELECT EXTRACT(MICROSECOND FROM NOW())", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));

    let us2: i64 = db
        .query("SELECT EXTRACT(MICROSECOND FROM NOW())", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    // Both epoch (seconds) and microsecond together should differ across 50ms
    let epoch1: i64 = db
        .query("SELECT EXTRACT(EPOCH FROM NOW())", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    // Use combined epoch*1000000 + microsecond to get unique values
    // If they were stale-cached, us1==us2 AND epoch1 would be the same
    // Since we've slept 50ms, at least the microsecond part should differ
    // (unless we happen to hit exact microsecond, which is astronomically unlikely)
    assert_ne!(
        us1, us2,
        "NOW() returned identical microsecond values {} after 50ms sleep (stale cache?)",
        us1,
    );
    // Also verify epoch is reasonable (not zero/garbage)
    assert!(epoch1 > 1_700_000_000, "Epoch should be recent");
}

/// CURRENT_TIMESTAMP should not be stale across queries.
#[test]
fn test_current_timestamp_not_stale() {
    let db = Database::open("memory://current_ts_stale").unwrap();

    let us1: i64 = db
        .query("SELECT EXTRACT(MICROSECOND FROM CURRENT_TIMESTAMP)", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));

    let us2: i64 = db
        .query("SELECT EXTRACT(MICROSECOND FROM CURRENT_TIMESTAMP)", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    assert_ne!(
        us1, us2,
        "CURRENT_TIMESTAMP returned identical microsecond values"
    );
}

/// WHERE clause with NOW() should not be served from semantic cache.
#[test]
fn test_now_in_where_not_cached() {
    let db = Database::open("memory://now_where_cache").unwrap();

    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO events VALUES (1, '2020-01-01 00:00:00')", ())
        .unwrap();

    // First query: SELECT * with NOW() in WHERE
    let count1: usize = db
        .query("SELECT * FROM events WHERE ts < NOW()", ())
        .unwrap()
        .count();
    assert_eq!(count1, 1);

    // Insert a new row with a past timestamp
    db.execute("INSERT INTO events VALUES (2, '2020-06-01 00:00:00')", ())
        .unwrap();

    // Second query: same WHERE but new data — must NOT return cached result
    let count2: usize = db
        .query("SELECT * FROM events WHERE ts < NOW()", ())
        .unwrap()
        .count();
    assert_eq!(
        count2, 2,
        "Second query should see newly inserted row (not cached result)"
    );
}

/// NOW() - INTERVAL arithmetic should not be folded to a stale constant.
#[test]
fn test_now_minus_interval_not_folded_stale() {
    let db = Database::open("memory://now_interval_fold").unwrap();

    let us1: i64 = db
        .query(
            "SELECT EXTRACT(MICROSECOND FROM NOW() - INTERVAL '1 second')",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));

    let us2: i64 = db
        .query(
            "SELECT EXTRACT(MICROSECOND FROM NOW() - INTERVAL '1 second')",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();

    assert_ne!(
        us1, us2,
        "NOW() - INTERVAL returned identical microsecond values: {} (stale fold?)",
        us1,
    );
}
