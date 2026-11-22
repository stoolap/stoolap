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

//! An aggregate over a large sealed scan with a WHERE reads the scan in
//! parts, holding a part of it at a time, and answers as over the whole

use stoolap::Database;

const ROWS: i64 = 120_000;

/// Parts far smaller than the scan, so groups and passing rows fall across them
#[cfg(feature = "test-failpoints")]
const PART_ROWS: usize = 5_000;

/// t: id 1..=ROWS, a = id % 1000, b = (id % 97) / 97, g = id % 3
fn passes(id: i64) -> bool {
    (id % 97) as f64 / 97.0 > 0.25 && (id % 1000) % 7 == 3
}

fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b FLOAT, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT value, value % 1000, (value % 97) / 97.0, value % 3 FROM generate_series(1, {ROWS})"
        ),
        (),
    )
    .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
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

/// The rows of `sql`, read with parts of `PART_ROWS` where a test sets the
/// size, checked to have held fewer than two parts of the scan at once
fn parted(db: &Database, sql: &str, sealed: bool) -> Vec<String> {
    #[cfg(feature = "test-failpoints")]
    {
        stoolap::test_failpoints::set_scan_part_rows(PART_ROWS);
        stoolap::test_failpoints::reset_scan_part_largest();
    }
    let out = rows(db, sql);
    #[cfg(feature = "test-failpoints")]
    {
        stoolap::test_failpoints::set_scan_part_rows(0);
        let largest = stoolap::test_failpoints::scan_part_largest();
        if sealed {
            assert!(
                (1..2 * PART_ROWS).contains(&largest),
                "{sql}: the aggregation held {largest} scanned rows at once"
            );
        }
    }
    let _ = sealed;
    out
}

#[test]
fn an_aggregate_over_a_sealed_scan_with_a_where_reads_it_in_parts() {
    let ids: Vec<i64> = (1..=ROWS).filter(|&id| passes(id)).collect();
    let count = ids.len();
    let sum_a: i64 = ids.iter().map(|id| id % 1000).sum();
    let whole = vec![format!(
        "{count}|{sum_a}|{}|{}",
        ids.first().unwrap(),
        ids.last().unwrap()
    )];
    let grouped: Vec<String> = (0..3)
        .map(|g| {
            let of: Vec<i64> = ids.iter().copied().filter(|id| id % 3 == g).collect();
            format!("{g}|{}|{}", of.len(), of.iter().sum::<i64>())
        })
        .collect();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        let sql = "SELECT COUNT(*), SUM(a), MIN(id), MAX(id) FROM t WHERE b > 0.25 AND a % 7 = 3";
        assert_eq!(parted(&db, sql, sealed), whole, "sealed {sealed}: {sql}");
        let sql = "SELECT g, COUNT(*), SUM(id) FROM t WHERE b > 0.25 AND a % 7 = 3 \
                   GROUP BY g ORDER BY g";
        assert_eq!(parted(&db, sql, sealed), grouped, "sealed {sealed}: {sql}");
    }
}

/// Rows of f, and the part size under `test-failpoints`: parts large enough
/// that the aggregation splits each over the thread pool
const FLOAT_ROWS: i64 = 210_000;
#[cfg(feature = "test-failpoints")]
const FLOAT_PART_ROWS: usize = 100_000;

/// A part boundary between a negative and two positive values near the
/// FLOAT limit: in order the sum stays finite, the second part alone would
/// not
#[test]
fn a_float_sum_split_in_parts_adds_as_the_rows_come_in_order() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE f (id INTEGER PRIMARY KEY, a INTEGER, x FLOAT)",
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO f SELECT value, value, CASE value WHEN 1 THEN -1e308 \
                 WHEN 100001 THEN 1e308 WHEN 100002 THEN 1e308 ELSE 0.0 END \
                 FROM generate_series(1, {FLOAT_ROWS})"
            ),
            (),
        )
        .unwrap();
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let sql = "SELECT SUM(x), AVG(x) FROM f WHERE a % 2 >= 0";
        #[cfg(feature = "test-failpoints")]
        stoolap::test_failpoints::set_scan_part_rows(FLOAT_PART_ROWS);
        let got: Result<Vec<(f64, f64)>, String> = db
            .query(sql, ())
            .map_err(|e| e.to_string())
            .and_then(|rows| {
                rows.map(|row| {
                    let row = row.map_err(|e| e.to_string())?;
                    let float = |i| match row.get_value(i).unwrap() {
                        stoolap::Value::Float(f) => Ok(*f),
                        other => Err(format!("{other:?}")),
                    };
                    Ok((float(0)?, float(1)?))
                })
                .collect()
            });
        #[cfg(feature = "test-failpoints")]
        stoolap::test_failpoints::set_scan_part_rows(0);
        assert_eq!(
            got,
            Ok(vec![(1e308, 1e308 / FLOAT_ROWS as f64)]),
            "sealed {sealed}: {sql}"
        );
    }
}

/// A scan the WHERE narrows to a few rows of a large sealed table keeps
/// room for about the rows it read, not for a whole part
#[test]
fn a_split_scan_that_reads_few_rows_keeps_little_room() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir, true);
    let sql = "SELECT COUNT(*), SUM(a) FROM t WHERE id <= 50 AND a % 7 = 3";
    let ids: Vec<i64> = (1..=50).filter(|id| (id % 1000) % 7 == 3).collect();
    let want = vec![format!("{}|{}", ids.len(), ids.iter().sum::<i64>())];
    assert_eq!(rows(&db, sql), want, "{sql}");
    #[cfg(feature = "test-failpoints")]
    {
        let room = stoolap::test_failpoints::scan_part_reserved_rows();
        assert!(
            (1..=1024).contains(&room),
            "{sql}: the scan kept room for {room} rows"
        );
    }
}
