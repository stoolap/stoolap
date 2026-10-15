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

//! MIN, MAX, SUM and AVG over FLOAT give every path the same answer: NULLs
//! are left out, NaN is greater than every number and poisons SUM and AVG

use std::path::Path;

use stoolap::core::Value;
use stoolap::Database;

const NAN_PAYLOAD: u64 = 0x7ff8_0000_0000_0001;

#[derive(Clone, Copy, Debug)]
enum Layout {
    Hot,
    Sealed,
    Reopened,
    /// t's rows up to this id sealed, the rest hot
    Mixed(i64),
}

fn dsn(dir: &Path) -> String {
    format!("file://{}?checkpoint_interval=3600", dir.display())
}

/// t: groups g 1 (NaN first, then numbers), 2 (only NaN), 3 (a number and
/// NULL), 4 (only NULL), and h 1 (rows 2-4, NaN last); u: only NaN and NULL.
/// With `tombstone`, one more row of each is sealed and then deleted, so the
/// reads cannot use stats
fn open(dir: &Path, indexed: bool, layout: Layout, tombstone: bool) -> Database {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, f FLOAT, h INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
    if indexed {
        db.execute("CREATE INDEX t_f ON t(f)", ()).unwrap();
        db.execute("CREATE INDEX t_g ON t(g)", ()).unwrap();
        db.execute("CREATE INDEX u_f ON u(f)", ()).unwrap();
    }
    let nan = Value::Float(f64::NAN);
    let nanp = Value::Float(f64::from_bits(NAN_PAYLOAD));
    let null = Value::Null(stoolap::core::DataType::Float);
    let mut t_rows = vec![
        (1, 1, nan.clone()),
        (2, 1, Value::Float(1.5)),
        (3, 1, Value::Float(-2.0)),
        (4, 2, nan.clone()),
        (5, 2, nanp.clone()),
        (6, 3, Value::Float(4.0)),
        (7, 3, null.clone()),
        (8, 4, null.clone()),
    ];
    let mut u_rows = vec![(1, nan), (2, nanp), (3, null)];
    if tombstone {
        t_rows.push((9, 9, Value::Float(-100.0)));
        u_rows.push((9, Value::Float(-100.0)));
    }
    // Mixed(5) seals every NaN below the hot maximum 4.0, Mixed(3) leaves
    // group 2's NaNs hot only, Mixed(0) seals only the deleted row; u's row 1
    // is sealed either way
    let (sealed_up_to, u_sealed_up_to) = match layout {
        Layout::Mixed(last) => (last, 1),
        _ => (i64::MAX, i64::MAX),
    };
    let first = |id: i64, last: i64| id <= last || id == 9;
    for pass in [true, false] {
        for (id, g, f) in t_rows.iter().filter(|r| first(r.0, sealed_up_to) == pass) {
            let h = Value::Integer((2..=4).contains(id) as i64);
            db.execute(
                "INSERT INTO t VALUES ($1, $2, $3, $4)",
                vec![Value::Integer(*id), Value::Integer(*g), f.clone(), h],
            )
            .unwrap();
        }
        for (id, f) in u_rows.iter().filter(|r| first(r.0, u_sealed_up_to) == pass) {
            db.execute(
                "INSERT INTO u VALUES ($1, $2)",
                vec![Value::Integer(*id), f.clone()],
            )
            .unwrap();
        }
        if pass && !matches!(layout, Layout::Hot) {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
    }
    if tombstone {
        db.execute("DELETE FROM t WHERE id = 9", ()).unwrap();
        db.execute("DELETE FROM u WHERE id = 9", ()).unwrap();
    }
    if matches!(layout, Layout::Reopened) {
        db.close().unwrap();
        return Database::open(&dsn(dir)).unwrap();
    }
    db
}

fn render(v: &Value) -> String {
    match v {
        Value::Null(_) => "NULL".into(),
        Value::Float(x) if x.is_nan() => "NaN".into(),
        Value::Float(x) => format!("{x:?}"),
        Value::Integer(x) => x.to_string(),
        other => format!("{other:?}"),
    }
}

/// Every row as `a|b|..`, sorted
fn rows(db: &Database, sql: &str) -> Vec<String> {
    let mut out: Vec<String> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            let mut cells = Vec::new();
            let mut i = 0;
            while let Some(v) = row.get_value(i) {
                cells.push(render(v));
                i += 1;
            }
            cells.join("|")
        })
        .collect();
    out.sort();
    out
}

fn check(cases: &[(&str, &[&str])]) {
    check_after("", cases);
}

/// `check`, running `setup` on each layout first
fn check_after(setup: &str, cases: &[(&str, &[&str])]) {
    for indexed in [false, true] {
        for layout in [
            Layout::Hot,
            Layout::Sealed,
            Layout::Reopened,
            Layout::Mixed(5),
            Layout::Mixed(3),
            Layout::Mixed(0),
        ] {
            for tombstone in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let db = open(dir.path(), indexed, layout, tombstone);
                if !setup.is_empty() {
                    db.execute(setup, ()).unwrap();
                }
                for (sql, want) in cases {
                    assert_eq!(
                        rows(&db, sql),
                        *want,
                        "indexed={indexed} {layout:?} tombstone={tombstone}: {sql}"
                    );
                }
            }
        }
    }
}

#[test]
fn whole_table_aggregates_order_nan_last_and_propagate_it() {
    check(&[
        (
            "SELECT MIN(f), MAX(f), SUM(f), AVG(f), COUNT(f) FROM t",
            &["-2.0|NaN|NaN|NaN|6"],
        ),
        ("SELECT MIN(f) FROM t", &["-2.0"]),
        ("SELECT MAX(f) FROM t", &["NaN"]),
        ("SELECT SUM(f) FROM t", &["NaN"]),
        ("SELECT AVG(f) FROM t", &["NaN"]),
    ]);
}

#[test]
fn a_column_of_only_nan_has_nan_for_every_aggregate() {
    check(&[
        (
            "SELECT MIN(f), MAX(f), SUM(f), AVG(f), COUNT(f) FROM u",
            &["NaN|NaN|NaN|NaN|2"],
        ),
        ("SELECT MIN(f) FROM u", &["NaN"]),
        ("SELECT MAX(f) FROM u", &["NaN"]),
        ("SELECT SUM(f) FROM u", &["NaN"]),
        ("SELECT AVG(f) FROM u", &["NaN"]),
    ]);
}

#[test]
fn filtered_aggregates_see_nan_wherever_it_comes() {
    check(&[
        (
            "SELECT MIN(f), MAX(f), SUM(f), AVG(f) FROM t WHERE id > 0",
            &["-2.0|NaN|NaN|NaN"],
        ),
        (
            "SELECT MIN(f), MAX(f), SUM(f), AVG(f) FROM t WHERE id BETWEEN 2 AND 4",
            &["-2.0|NaN|NaN|NaN"],
        ),
        (
            "SELECT MIN(f), MAX(f), SUM(f), AVG(f) FROM t WHERE g = 2",
            &["NaN|NaN|NaN|NaN"],
        ),
        (
            "SELECT MIN(f), MAX(f), SUM(f), AVG(f) FROM t WHERE g = 4",
            &["NULL|NULL|NULL|NULL"],
        ),
    ]);
}

#[test]
fn grouped_aggregates_see_nan_in_each_group() {
    check(&[
        (
            "SELECT g, MIN(f), MAX(f), SUM(f), AVG(f), COUNT(f) FROM t GROUP BY g",
            &[
                "1|-2.0|NaN|NaN|NaN|3",
                "2|NaN|NaN|NaN|NaN|2",
                "3|4.0|4.0|4.0|4.0|1",
                "4|NULL|NULL|NULL|NULL|0",
            ],
        ),
        (
            "SELECT h, MIN(f), MAX(f), SUM(f), AVG(f) FROM t GROUP BY h",
            &["0|4.0|NaN|NaN|NaN", "1|-2.0|NaN|NaN|NaN"],
        ),
    ]);
}

#[test]
fn aggregates_of_an_expression_order_nan_last() {
    check(&[
        ("SELECT MIN(f * 1.0), MAX(f * 1.0) FROM t", &["-2.0|NaN"]),
        (
            "SELECT MIN(f * 1.0), MAX(f * 1.0) FROM t WHERE id BETWEEN 2 AND 4",
            &["-2.0|NaN"],
        ),
        (
            "SELECT g, MIN(f * 1.0), MAX(f * 1.0) FROM t GROUP BY g",
            &["1|-2.0|NaN", "2|NaN|NaN", "3|4.0|4.0", "4|NULL|NULL"],
        ),
    ]);
}

#[test]
fn an_added_column_with_a_nan_default_is_nan_everywhere() {
    check_after(
        "ALTER TABLE t ADD COLUMN x FLOAT DEFAULT CAST('NaN' AS FLOAT)",
        &[
            (
                "SELECT MIN(x), MAX(x), SUM(x), AVG(x), COUNT(x) FROM t",
                &["NaN|NaN|NaN|NaN|8"],
            ),
            (
                "SELECT MIN(x), MAX(x), SUM(x), AVG(x) FROM t WHERE id > 0",
                &["NaN|NaN|NaN|NaN"],
            ),
            (
                "SELECT g, MIN(x), MAX(x), SUM(x), AVG(x) FROM t GROUP BY g",
                &[
                    "1|NaN|NaN|NaN|NaN",
                    "2|NaN|NaN|NaN|NaN",
                    "3|NaN|NaN|NaN|NaN",
                    "4|NaN|NaN|NaN|NaN",
                ],
            ),
        ],
    );
}

#[test]
fn integers_and_floats_compare_exactly_with_nan_last() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    let values = [
        Value::Integer(9_007_199_254_740_993),
        Value::Float(9_007_199_254_740_992.0),
        Value::Integer(i64::MAX),
        Value::Float(9_223_372_036_854_775_808.0),
        Value::Float(f64::NAN),
    ];
    let null_i = Value::Null(stoolap::core::DataType::Integer);
    let null_f = Value::Null(stoolap::core::DataType::Float);
    for (table, reversed) in [("m_asc", false), ("m_desc", true)] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT)"),
            (),
        )
        .unwrap();
        let mut ordered = values.to_vec();
        if reversed {
            ordered.reverse();
        }
        for (id, v) in ordered.into_iter().enumerate() {
            let (i, f) = match v {
                Value::Integer(_) => (v, null_f.clone()),
                _ => (null_i.clone(), v),
            };
            db.execute(
                &format!("INSERT INTO {table} VALUES ($1, $2, $3)"),
                vec![Value::Integer(id as i64), i, f],
            )
            .unwrap();
        }
        let sql = format!("SELECT MIN(COALESCE(i, f)), MAX(COALESCE(i, f)) FROM {table}");
        assert_eq!(rows(&db, &sql), ["9007199254740992.0|NaN"], "{sql}");
        let sql = format!("{sql} WHERE f IS NULL OR f < 1e300");
        assert_eq!(
            rows(&db, &sql),
            ["9007199254740992.0|9.223372036854776e18"],
            "{sql}"
        );
    }
}
