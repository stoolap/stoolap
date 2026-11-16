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

//! Whether zone maps rule a whole scan out is asked of the segments only
//! until one may hold a match: a statement some segment answers asks one
//! segment a comparison, and one no segment answers is still skipped

#![cfg(feature = "test-failpoints")]

use stoolap::{test_failpoints, Database};

const ROWS: i64 = 20_000;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn zone_maps_ask_the_segments_only_until_one_may_match() {
    let db = Database::open("memory://zone_map_any_segment").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, a INTEGER, b INTEGER)".to_string(),
        format!(
            "INSERT INTO t SELECT value, value, value % 2, value * 7919 % {ROWS} \
             FROM generate_series(1, {ROWS})"
        ),
        "CREATE INDEX t_b ON t (b)".to_string(),
        "CREATE INDEX t_v ON t (v)".to_string(),
        "ANALYZE t".to_string(),
    ] {
        db.execute(&sql, ()).unwrap();
    }
    let key = 1001 * 7919 % ROWS;
    // Every segment holds a = 1 and spans b's values; v > N - 5 only the last
    for (sql, expected, most) in [
        (
            format!("SELECT id FROM t WHERE a = 1 AND b = {key} + 0"),
            vec![1001],
            2,
        ),
        (
            format!("SELECT id FROM t WHERE v > {}", ROWS - 5),
            vec![19996, 19997, 19998, 19999, 20000],
            1,
        ),
    ] {
        let checked = test_failpoints::zone_map_entries_checked();
        assert_eq!(ids(&db, &sql), expected, "{sql}");
        let checked = test_failpoints::zone_map_entries_checked() - checked;
        assert!(
            checked <= most,
            "{sql}: {checked} zone map segments asked, at most {most}"
        );
    }
    // No segment holds a past 1: the scan is skipped all the same
    let pruned = test_failpoints::zone_map_prunes();
    let sql = "SELECT id FROM t WHERE a > 5";
    assert_eq!(ids(&db, sql), Vec::<i64>::new());
    assert_eq!(test_failpoints::zone_map_prunes(), pruned + 1, "{sql}");
}
