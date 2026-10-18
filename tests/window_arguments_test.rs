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

//! A window argument that counts rows (a ROWS frame offset, the offset of
//! LAG and LEAD, the n of NTILE and NTH_VALUE) is an INTEGER of at least 0,
//! 1 for NTILE and NTH_VALUE, and a RANGE frame offset is a number of at
//! least 0; any other value is an error. Every window path computes the
//! frame of a row the same way.

use std::collections::HashMap;

use stoolap::{Database, Params, Value};

const GROUPS: i64 = 12;
const PER_GROUP: i64 = 100;

/// `t` and `u` hold the same rows: GROUPS partitions of PER_GROUP rows,
/// id = g * PER_GROUP + k, x = k, f = k / 2 * 0.5 (two peers per key);
/// `u` indexes g, so a PARTITION BY g with a LIMIT fetches it lazily
fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://window_arguments_{name}")).unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!(
                "CREATE TABLE {table} (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER, f FLOAT)"
            ),
            (),
        )
        .unwrap();
        let rows: Vec<String> = (0..GROUPS * PER_GROUP)
            .map(|id| {
                let k = id % PER_GROUP;
                format!("({id}, {}, {k}, {})", id / PER_GROUP, (k / 2) as f64 * 0.5)
            })
            .collect();
        db.execute(
            &format!("INSERT INTO {table} VALUES {}", rows.join(", ")),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX u_g ON u (g)", ()).unwrap();
    db
}

fn query(db: &Database, sql: &str, params: impl Params) -> stoolap::Result<Vec<(i64, Value)>> {
    db.query(sql, params)?
        .map(|row| {
            let row = row?;
            Ok((
                row.get::<i64>(0)?,
                row.get_value(1)
                    .cloned()
                    .unwrap_or(Value::Null(stoolap::DataType::Null)),
            ))
        })
        .collect()
}

/// The statements that run `call` over each window path: one partition,
/// a few partitions, many partitions (in parallel for a function that is
/// not an aggregate), a LIMIT streamed per partition and a LIMIT fetched
/// per partition from an index
fn shapes(call: &str, over: &str) -> Vec<String> {
    let partitioned = format!("{call} OVER (PARTITION BY g {over})");
    vec![
        format!("SELECT id, {call} OVER ({over}) FROM t WHERE g = 0 ORDER BY id"),
        format!("SELECT id, {partitioned} FROM t WHERE g < 3 ORDER BY id"),
        format!("SELECT id, {partitioned} FROM t ORDER BY id"),
        format!("SELECT id, {partitioned} FROM t LIMIT 250"),
        format!("SELECT id, {partitioned} FROM u LIMIT 250"),
    ]
}

/// The answer of `call` for each k, the same on every window path
fn answers(db: &Database, call: &str, over: &str) -> Vec<Value> {
    let mut by_k: Vec<Option<Value>> = vec![None; PER_GROUP as usize];
    for sql in shapes(call, over) {
        let rows = query(db, &sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert!(!rows.is_empty(), "{sql} returned no row");
        for (id, value) in rows {
            let k = (id % PER_GROUP) as usize;
            match &by_k[k] {
                None => by_k[k] = Some(value),
                Some(expected) => assert_eq!(&value, expected, "{sql}: id {id}"),
            }
        }
    }
    by_k.into_iter()
        .map(|value| value.expect("every k answered"))
        .collect()
}

fn fails_everywhere(db: &Database, call: &str, over: &str) {
    for sql in shapes(call, over) {
        assert!(query(db, &sql, ()).is_err(), "{sql} should fail");
    }
}

fn ints(values: &[i64]) -> Vec<Value> {
    values.iter().map(|&v| Value::Integer(v)).collect()
}

fn null() -> Value {
    Value::Null(stoolap::DataType::Null)
}

#[test]
fn test_a_frame_offset_outside_its_domain_is_an_error() {
    let db = open("frame_errors");
    for call in ["SUM(x)", "FIRST_VALUE(x)", "ROW_NUMBER()"] {
        for over in [
            "ORDER BY id ROWS BETWEEN CURRENT ROW AND -1 FOLLOWING",
            "ORDER BY id ROWS -1 PRECEDING",
            "ORDER BY id ROWS BETWEEN NULL PRECEDING AND CURRENT ROW",
            "ORDER BY id ROWS BETWEEN 1e30 PRECEDING AND CURRENT ROW",
            "ORDER BY id ROWS BETWEEN 'x' PRECEDING AND CURRENT ROW",
            "ORDER BY id ROWS BETWEEN x PRECEDING AND CURRENT ROW",
            "ORDER BY f RANGE BETWEEN -0.5 PRECEDING AND CURRENT ROW",
            "ORDER BY f RANGE BETWEEN NULL PRECEDING AND CURRENT ROW",
            "ORDER BY f RANGE BETWEEN '1' PRECEDING AND CURRENT ROW",
        ] {
            fails_everywhere(&db, call, over);
        }
    }
}

#[test]
fn test_a_function_argument_outside_its_domain_is_an_error() {
    let db = open("argument_errors");
    for call in [
        "NTILE(0)",
        "NTILE(-1)",
        "NTILE(NULL)",
        "NTILE(1e30)",
        "LAG(x, -1)",
        "LEAD(x, -1)",
        "LAG(x, NULL)",
        "LAG(x, 'x')",
        "NTH_VALUE(x, 0)",
        "NTH_VALUE(x, -1)",
    ] {
        fails_everywhere(&db, call, "ORDER BY id");
    }
    let ordered = "SELECT id, x FROM t ORDER BY LAG(x, -1) OVER (ORDER BY id), id";
    assert!(query(&db, ordered, ()).is_err(), "{ordered} should fail");
}

#[test]
fn test_an_argument_is_read_as_an_integer_argument() {
    let db = open("argument_reading");
    let lag2: Vec<Value> = std::iter::repeat_n(null(), 2)
        .chain((0..PER_GROUP - 2).map(Value::Integer))
        .collect();
    for call in [
        "LAG(x, 2)",
        "LAG(x, 2.0)",
        "LAG(x, 2.9)",
        "LAG(x, '2')",
        "LAG(x, 1 + 1)",
    ] {
        assert_eq!(answers(&db, call, "ORDER BY id"), lag2, "{call}");
    }
    let lead0 = ints(&(0..PER_GROUP).collect::<Vec<_>>());
    assert_eq!(answers(&db, "LEAD(x, 0)", "ORDER BY id"), lead0);
    let halves: Vec<Value> = (0..PER_GROUP).map(|k| Value::Integer(k / 50 + 1)).collect();
    assert_eq!(answers(&db, "NTILE(2.5)", "ORDER BY id"), halves);
    let second: Vec<Value> = (0..PER_GROUP)
        .map(|k| if k == 0 { null() } else { Value::Integer(1) })
        .collect();
    assert_eq!(answers(&db, "NTH_VALUE(x, 2.0)", "ORDER BY id"), second);
}

#[test]
fn test_a_parameter_is_read_once_as_a_literal_is() {
    let db = open("parameters");
    let sql = "SELECT id, SUM(x) OVER (ORDER BY id ROWS BETWEEN $1 PRECEDING AND CURRENT ROW) FROM t WHERE g = 0 ORDER BY id";
    let rows = query(&db, sql, (1i64,)).unwrap();
    assert_eq!(rows[3].1, Value::Integer(5));
    assert!(query(&db, sql, (-1i64,)).is_err());
    assert!(query(&db, sql, (Value::Null(stoolap::DataType::Integer),)).is_err());

    let sql = "SELECT id, LAG(x, $1) OVER (ORDER BY id) FROM t WHERE g = 0 ORDER BY id";
    assert_eq!(query(&db, sql, (3i64,)).unwrap()[5].1, Value::Integer(2));
    assert!(query(&db, sql, (-1i64,)).is_err());
}

#[test]
fn test_every_path_frames_a_row_the_same_way() {
    let db = open("frames");
    let sum = |range: std::ops::Range<i64>| {
        if range.is_empty() {
            null()
        } else {
            Value::Integer(range.sum())
        }
    };
    // f is equal across peers, so the nth f of a frame does not depend on
    // how peers are ordered
    let nth_f = |range: std::ops::Range<i64>, nth: i64, desc: bool| {
        if range.end - range.start < nth {
            null()
        } else {
            let k = if desc {
                range.end - nth
            } else {
                range.start + nth - 1
            };
            Value::Float((k / 2) as f64 * 0.5)
        }
    };
    const N: i64 = PER_GROUP;
    /// The range of k a frame holds for row k
    type Holds = fn(i64) -> std::ops::Range<i64>;
    let frames: [(&str, Holds); 10] = [
        (
            "ORDER BY id ROWS BETWEEN 1 + 1 PRECEDING AND CURRENT ROW",
            |k| (k - 2).max(0)..k + 1,
        ),
        (
            "ORDER BY id ROWS BETWEEN 5 FOLLOWING AND UNBOUNDED FOLLOWING",
            |k| (k + 5).min(N)..N,
        ),
        (
            "ORDER BY id ROWS BETWEEN UNBOUNDED FOLLOWING AND UNBOUNDED FOLLOWING",
            |_| N..N,
        ),
        (
            "ORDER BY id ROWS BETWEEN 1 FOLLOWING AND 1 PRECEDING",
            |k| k + 1..k + 1,
        ),
        (
            "ORDER BY id ROWS BETWEEN CURRENT ROW AND 9223372036854775807 FOLLOWING",
            |k| k..N,
        ),
        (
            "ORDER BY id ROWS BETWEEN 9223372036854775807 PRECEDING AND CURRENT ROW",
            |k| 0..k + 1,
        ),
        // f = k / 2 * 0.5: the keys within 0.5 below are the two peers below
        (
            "ORDER BY f RANGE BETWEEN 0.5 PRECEDING AND CURRENT ROW",
            |k| (k / 2 * 2 - 2).max(0)..k / 2 * 2 + 2,
        ),
        (
            "ORDER BY f DESC RANGE BETWEEN 0.5 PRECEDING AND CURRENT ROW",
            |k| k / 2 * 2..(k / 2 * 2 + 4).min(N),
        ),
        (
            "ORDER BY f RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
            |k| k / 2 * 2..N,
        ),
        (
            "ORDER BY f RANGE BETWEEN 1.5 FOLLOWING AND 2 FOLLOWING",
            |k| (k / 2 * 2 + 6).min(N)..(k / 2 * 2 + 10).min(N),
        ),
    ];
    for (over, range) in frames {
        let expected: Vec<Value> = (0..N).map(|k| sum(range(k))).collect();
        assert_eq!(answers(&db, "SUM(x)", over), expected, "SUM over {over}");
        let desc = over.contains("DESC");
        for (call, nth) in [("FIRST_VALUE(f)", 1), ("NTH_VALUE(f, 2)", 2)] {
            let expected: Vec<Value> = (0..N).map(|k| nth_f(range(k), nth, desc)).collect();
            assert_eq!(answers(&db, call, over), expected, "{call} over {over}");
        }
    }
}

#[test]
fn test_a_frame_applies_without_an_order_by() {
    let db = open("no_order_by");
    let rows = query(
        &db,
        "SELECT id, SUM(x) OVER (ROWS BETWEEN CURRENT ROW AND CURRENT ROW) FROM t WHERE g = 0",
        (),
    )
    .unwrap();
    let by_id: HashMap<i64, Value> = rows.into_iter().collect();
    for k in 0..PER_GROUP {
        assert_eq!(by_id[&k], Value::Integer(k), "row {k}");
    }
}

#[test]
fn test_a_range_key_that_is_not_a_number_frames_to_its_peers() {
    let db = Database::open("memory://window_arguments_not_a_number").unwrap();
    db.execute(
        "CREATE TABLE n (id INTEGER PRIMARY KEY, g INTEGER, f FLOAT, x INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO n VALUES (1, 0, 1.0, 1), (2, 0, 2.0, 2), (3, 0, CAST('inf' AS FLOAT), 4), \
         (4, 0, CAST('NaN' AS FLOAT), 10), (5, 0, CAST('NaN' AS FLOAT), 10), (6, 0, NULL, 100)",
        (),
    )
    .unwrap();
    // (frame, SUM(x), COUNT(*), FIRST_VALUE(x)) for ids 1 to 6
    let cases = [
        (
            "RANGE BETWEEN 1 PRECEDING AND CURRENT ROW",
            [1, 3, 4, 20, 20, 100],
            [1, 2, 1, 2, 2, 1],
            [1, 1, 4, 10, 10, 100],
        ),
        (
            "RANGE BETWEEN CAST('inf' AS FLOAT) PRECEDING AND CURRENT ROW",
            [1, 3, 7, 20, 20, 100],
            [1, 2, 3, 2, 2, 1],
            [1, 1, 1, 10, 10, 100],
        ),
    ];
    for (frame, sum, count, first) in cases {
        for (call, expected) in [
            ("SUM(x)", sum),
            ("COUNT(*)", count),
            ("FIRST_VALUE(x)", first),
        ] {
            for partition in ["", "PARTITION BY g"] {
                let sql = format!(
                    "SELECT id, {call} OVER ({partition} ORDER BY f {frame}) FROM n ORDER BY id"
                );
                let rows = query(&db, &sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
                let values: Vec<Value> = rows.into_iter().map(|(_, value)| value).collect();
                assert_eq!(values, ints(&expected), "{sql}");
            }
        }
    }
}
