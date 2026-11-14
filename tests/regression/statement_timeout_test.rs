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

//! Statements that run for seconds stop near a 20 ms timeout, on the clock

use std::time::{Duration, Instant};

use stoolap::{Database, Error};

const TIMEOUT_MS: u64 = 20;

/// Far past the timeout and far below what each statement takes untimed
const BOUND: Duration = Duration::from_secs(1);

fn database() -> Database {
    let db = Database::open("memory://statement_timeout_shapes").unwrap();
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        "INSERT INTO u SELECT value, value, REPEAT('x', 50) FROM generate_series(1, 400000)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t SELECT value, value % 97 FROM generate_series(1, 4000)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Every row of `sql` read under the timeout, and how long the call took
fn read_all(db: &Database, sql: &str) -> (stoolap::Result<usize>, Duration) {
    let started = Instant::now();
    let read = db
        .query_with_timeout(sql, (), TIMEOUT_MS)
        .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>())
        .map(|rows| rows.len());
    (read, started.elapsed())
}

#[test]
fn every_measured_shape_stops_near_its_timeout() {
    let db = database();
    let mut wrong = Vec::new();

    let update = "UPDATE u SET v = v + 1, s = UPPER(s) WHERE v > 0";
    let before: i64 = db.query_one("SELECT SUM(v) FROM u", ()).unwrap();
    let started = Instant::now();
    let updated = db.execute_with_timeout(update, (), TIMEOUT_MS);
    let took = started.elapsed();
    if !matches!(updated, Err(Error::QueryCancelled)) || took > BOUND {
        wrong.push(format!("{update} gave {updated:?} after {took:?}"));
    }
    let after: i64 = db.query_one("SELECT SUM(v) FROM u", ()).unwrap();
    if after != before {
        wrong.push(format!("{update} stopped and wrote rows"));
    }

    for sql in [
        "SELECT id, UPPER(REPEAT(s, 20)) FROM u",
        "SELECT COUNT(*) FROM t a, t b WHERE a.v + b.v > 0",
        "SELECT a.id FROM t a, t b WHERE a.v = b.v ORDER BY a.id LIMIT 1",
        "SELECT COUNT(*) FROM generate_series(1, 3000) a, generate_series(1, 3000) b \
         WHERE a.value + b.value > 0",
        "SELECT COUNT(*) FROM t a WHERE EXISTS (SELECT 1 FROM t b WHERE b.v = a.v AND b.id > a.id * 0)",
    ] {
        let (read, took) = read_all(&db, sql);
        if !matches!(read, Err(Error::QueryCancelled)) || took > BOUND {
            wrong.push(format!("{sql} gave {read:?} after {took:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
