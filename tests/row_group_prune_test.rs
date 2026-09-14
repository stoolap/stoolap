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

//! A clustered volume's row groups are ruled out by their zone maps before
//! the dictionary filters decode a block per group: a query for one series
//! decodes the groups that can hold it and nothing else, and answers the
//! same as before on series that straddle a group, on groups whose
//! independent column bounds admit a key they do not hold, on OR, on NULL
//! keys and on the high-match fallback.

use std::sync::Mutex;
use stoolap::storage::volume::group_cache::DECODED_GROUPS;
use stoolap::Database;

/// The decode counter is process-wide, so the tests of this file run one
/// at a time
static SERIAL: Mutex<()> = Mutex::new(());

const ROWS: i64 = 360_000;

/// Twelve series (two exchanges, six symbols), 30,000 rows each, arriving
/// interleaved in id order; the clustered volume holds them in key order,
/// 65,536 rows per group, so a series covers one group or straddles two
fn seeded(dir: &std::path::Path) -> Database {
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, exchange TEXT, symbol TEXT, t INTEGER NOT NULL, price FLOAT) CLUSTER BY (exchange, symbol, t)",
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO t SELECT g.value, 'e' || CAST((g.value - 1) % 2 AS TEXT), \
                 'S' || CAST(((g.value - 1) / 2) % 6 AS TEXT), g.value, g.value * 0.5 \
                 FROM generate_series(1, {ROWS}) g"
            ),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    // Reopened, the volume's columns are read from disk a group at a time
    Database::open(&dsn).unwrap()
}

fn series(exchange: i64, symbol: i64) -> (i64, f64) {
    (1..=ROWS)
        .filter(|id| (id - 1) % 2 == exchange && ((id - 1) / 2) % 6 == symbol)
        .fold((0, 0.0), |(n, sum), id| (n + 1, sum + id as f64 * 0.5))
}

/// Rows and price sum of a projection through the scanner, and the
/// number of column blocks decoded to answer it
fn project(db: &Database, sql: &str) -> ((i64, f64), u64) {
    let before = DECODED_GROUPS.stats().misses;
    let got = db
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<f64>(1).unwrap())
        .fold((0, 0.0), |(n, sum), price| (n + 1, sum + price));
    (got, DECODED_GROUPS.stats().misses - before)
}

#[test]
fn a_series_held_by_one_group_decodes_that_group_alone() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = seeded(dir.path());
    // (e1, S3) is the tenth series in key order: rows 270,000 to 300,000
    // of the volume, inside group 4
    let (got, decoded) = project(
        &db,
        "SELECT id, price FROM t WHERE exchange = 'e1' AND symbol = 'S3'",
    );
    assert_eq!(got, series(1, 3));
    // 7 today: the two key columns of the group for the dictionary pass
    // and the group's columns for the rows; 15 when every group's key
    // columns are decoded before the zone maps are consulted
    assert!(
        decoded <= 8,
        "{decoded} blocks decoded for a series that one group holds"
    );
}

#[test]
fn a_series_straddling_two_groups_is_read_whole() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = seeded(dir.path());
    // (e0, S2) is the third series: rows 60,000 to 90,000, across the
    // boundary at 65,536
    let (got, decoded) = project(
        &db,
        "SELECT id, price FROM t WHERE exchange = 'e0' AND symbol = 'S2'",
    );
    assert_eq!(got, series(0, 2));
    // 12 today, 18 without pruning
    assert!(decoded <= 13, "{decoded} blocks decoded for two groups");
}

#[test]
fn independent_column_bounds_admit_a_group_without_the_key_and_still_answer_right() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = seeded(dir.path());
    // Group 2 runs from the end of e0 (S4, S5) into e1 (S0): its exchange
    // bound admits e0 and its symbol bound admits S0, though no row of it
    // is (e0, S0). It is visited, finds nothing, and the answer stands
    let (got, decoded) = project(
        &db,
        "SELECT id, price FROM t WHERE exchange = 'e0' AND symbol = 'S0'",
    );
    assert_eq!(got, series(0, 0));
    // 15 without pruning
    assert!(decoded <= 10, "{decoded} blocks decoded");
}

#[test]
fn an_or_filter_prunes_nothing_and_answers_the_union() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = seeded(dir.path());
    let count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM t WHERE exchange = 'e0' OR symbol = 'S5'",
            (),
        )
        .unwrap();
    assert_eq!(count, 180_000 + 30_000);
}

#[test]
fn a_filter_matching_most_rows_falls_back_and_still_skips_the_other_exchange() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db = seeded(dir.path());
    // Half the rows match, above the cap of the pre-computed match list
    let (got, decoded) = project(&db, "SELECT id, price FROM t WHERE exchange = 'e1'");
    assert_eq!(got.0, 180_000);
    let expected: f64 = (1..=ROWS)
        .filter(|id| (id - 1) % 2 == 1)
        .map(|id| id as f64 * 0.5)
        .sum();
    assert!((got.1 - expected).abs() < 1e-3);
    // e1 lives in groups 2 to 5; 20 today, 22 when the dictionary pass
    // decodes the groups before it that hold no e1
    assert!(decoded <= 21, "{decoded} blocks decoded for four groups");
}

#[test]
fn a_group_of_null_keys_is_left_out_of_an_equality_and_kept_for_is_null() {
    let _serial = SERIAL.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, exchange TEXT, symbol TEXT, t INTEGER NOT NULL, price FLOAT) CLUSTER BY (exchange, symbol, t)",
            (),
        )
        .unwrap();
        // NULL keys sort first: the first group holds nothing but NULLs
        db.execute(
            "INSERT INTO t SELECT g.value, CASE WHEN g.value <= 70000 THEN NULL ELSE 'e0' END, 'S0', g.value, g.value * 0.5 FROM generate_series(1, 140000) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    let (got, _) = project(
        &db,
        "SELECT id, price FROM t WHERE exchange = 'e0' AND symbol = 'S0'",
    );
    assert_eq!(got.0, 70_000);
    let nulls: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE exchange IS NULL", ())
        .unwrap();
    assert_eq!(nulls, 70_000);
}
