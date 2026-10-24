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

//! A time of day read as a TIMESTAMP takes the date it is read on, so an
//! answer that read one is never kept for a later statement, as a clock's.

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::subquery_cache_hits;
use stoolap::Database;

// The cache hit counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// t holds today's noon and another day's; u holds `text` once
fn setup(name: &str, text: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, txt TEXT)",
        "INSERT INTO t VALUES (1, '12:00:00'), (2, '2024-01-15 12:00:00')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db.execute("INSERT INTO u VALUES (1, $1)", (text,)).unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort();
    out
}

/// The rows of `sql` run twice, and whether the second run took a kept answer
fn twice(db: &Database, sql: &str) -> (Vec<i64>, Vec<i64>, bool) {
    let first = ids(db, sql);
    let before = subquery_cache_hits();
    let second = ids(db, sql);
    (first, second, subquery_cache_hits() > before)
}

#[test]
fn an_answer_that_read_the_clock_is_not_kept() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for (name, text, sql, want, kept) in [
        (
            "clock_literal",
            "x",
            "SELECT id FROM t WHERE ts IN (SELECT CAST('12:00:00' AS TIMESTAMP) FROM u)",
            vec![1],
            false,
        ),
        (
            "clock_member",
            "12:00:00",
            "SELECT id FROM t WHERE ts IN (SELECT txt FROM u)",
            vec![1],
            false,
        ),
        (
            "dated_literal",
            "x",
            "SELECT id FROM t WHERE ts IN (SELECT CAST('2024-01-15 12:00:00' AS TIMESTAMP) FROM u)",
            vec![2],
            true,
        ),
        (
            "dated_member",
            "2024-01-15 12:00:00",
            "SELECT id FROM t WHERE ts IN (SELECT txt FROM u)",
            vec![2],
            true,
        ),
    ] {
        let db = setup(&format!("clock_reading_{name}"), text);
        let (first, second, hit) = twice(&db, sql);
        if first != want || second != want || hit != kept {
            wrong.push(format!(
                "{name}: {sql}\n  rows {first:?} then {second:?}, want {want:?}; \
                 kept {hit}, want {kept}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
