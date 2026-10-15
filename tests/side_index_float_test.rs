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

//! Sealed reads of FLOAT keys, and of INTEGER keys compared with FLOAT
//! values, through the side file: each answer matches the scan of a twin
//! table without indexes

use std::sync::Mutex;

use stoolap::core::{DataType, Value};
use stoolap::Database;

/// The read counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

const ROWS: i64 = 3000;
const P53: f64 = 9_007_199_254_740_992.0;
const P63: f64 = 9_223_372_036_854_775_808.0;

fn f_of(id: i64) -> Value {
    match id {
        1 => Value::Float(-0.0),
        2 => Value::Float(f64::NAN),
        3 => Value::Float(f64::from_bits(0x7ff8_0000_0000_0001)),
        4 => Value::Float(f64::INFINITY),
        5 => Value::Float(f64::NEG_INFINITY),
        6 => Value::Float(P53),
        7 => Value::Float(P53 + 2.0),
        8 => Value::Null(DataType::Float),
        _ => Value::Float(id as f64 * 0.25 - 300.0),
    }
}

fn i_of(id: i64) -> Value {
    match id {
        10 => Value::Integer(9_007_199_254_740_992),
        11 => Value::Integer(9_007_199_254_740_993),
        12 => Value::Integer(i64::MAX),
        13 => Value::Integer(i64::MIN),
        14 => Value::Null(DataType::Integer),
        _ => Value::Integer(id * 3 - 4000),
    }
}

/// `t` with B-tree indexes on f and i, `u` the same rows without, `w` with
/// an indexed FLOAT column of plain values only and `x` its twin, and `a` a
/// small table of join keys, sealed when `seal`
fn open(dir: &std::path::Path, seal: bool) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    for table in ["t", "u", "w", "x"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, f FLOAT, i INTEGER)"),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_f ON t(f)", ()).unwrap();
    db.execute("CREATE INDEX t_i ON t(i)", ()).unwrap();
    db.execute("CREATE INDEX w_f ON w(f)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    for table in ["t", "u"] {
        let insert = format!("INSERT INTO {table} VALUES ($1, $2, $3)");
        for id in 0..ROWS {
            db.execute(&insert, vec![Value::Integer(id), f_of(id), i_of(id)])
                .unwrap();
        }
    }
    for table in ["w", "x"] {
        let insert = format!("INSERT INTO {table} VALUES ($1, $2, $1)");
        for id in 0..ROWS {
            db.execute(
                &insert,
                vec![Value::Integer(id), Value::Float(id as f64 * 0.25 - 300.0)],
            )
            .unwrap();
        }
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x FLOAT)", ())
        .unwrap();
    for (id, x) in [
        (1, Value::Float(f64::NAN)),
        (2, Value::Float(0.0)),
        (3, Value::Float(-0.0)),
        (4, Value::Float(10.5)),
        (5, Value::Float(P53)),
    ] {
        db.execute("INSERT INTO a VALUES ($1, $2)", vec![Value::Integer(id), x])
            .unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn ids(db: &Database, sql: &str, params: &[Value]) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, params.to_vec())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// Side file probes plus volumes the directory alone answered empty
fn side_reads(db: &Database) -> i64 {
    let rows = db.query("PRAGMA INDEX_READ_STATS", ()).unwrap();
    let columns = rows.columns().to_vec();
    let at = |name: &str| columns.iter().position(|c| c == name).unwrap();
    let (probes, misses) = (at("probes"), at("misses"));
    let row = rows.into_iter().next().unwrap().unwrap();
    row.get::<i64>(probes).unwrap() + row.get::<i64>(misses).unwrap()
}

/// Each query answers as the same query on the twin; one with `Some(served)`
/// is read through the side file or not as `served` says. None checks the
/// answer alone: a query no row meets may be settled by the volume's bloom
/// filter or zone map, and a range over keys that stretch a page to the
/// ends of the domain is left to the scan by the cost estimate
fn check(db: &Database, cases: &[(&str, Vec<Value>, Option<bool>)]) {
    for (sql, params, served) in cases {
        let twin = sql.replace(" t ", " u ").replace(" w ", " x ");
        let want = ids(db, &twin, params);
        let before = side_reads(db);
        let got = ids(db, sql, params);
        let read = side_reads(db) > before;
        assert_eq!(got, want, "{sql} {params:?}");
        if let Some(served) = served {
            assert_eq!(read, *served, "side file read: {sql} {params:?}");
        }
    }
}

#[test]
fn float_keys_answer_as_the_scan() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), true);
    let f = Value::Float;
    let int = Value::Integer;
    let eq = "SELECT id FROM t WHERE f = $1";
    let (yes, no, any) = (Some(true), Some(false), None);
    check(
        &db,
        &[
            (eq, vec![f(1.5)], yes),
            (eq, vec![f(0.0)], yes),
            (eq, vec![f(-0.0)], yes),
            (eq, vec![f(f64::INFINITY)], yes),
            (eq, vec![f(f64::NEG_INFINITY)], yes),
            (eq, vec![f(P53)], yes),
            (eq, vec![int(5)], yes),
            (eq, vec![f(7.3)], any),
            (eq, vec![int(9_007_199_254_740_993)], any),
            (eq, vec![f(f64::NAN)], no),
            (
                "SELECT id FROM t WHERE f BETWEEN $1 AND $2",
                vec![f(10.0), f(12.0)],
                any,
            ),
            (
                "SELECT id FROM t WHERE f > $1 AND f < $2",
                vec![f(-1.0), f(1.0)],
                any,
            ),
            (
                "SELECT id FROM t WHERE f >= $1 AND f <= $2",
                vec![f(-0.0), f(0.0)],
                any,
            ),
            (
                "SELECT id FROM t WHERE f > $1 AND f <= $2",
                vec![int(9_007_199_254_740_991), int(9_007_199_254_740_993)],
                any,
            ),
            (
                "SELECT id FROM t WHERE f > $1",
                vec![int(9_007_199_254_740_993)],
                any,
            ),
            ("SELECT id FROM t WHERE f > $1", vec![f(1e300)], any),
            ("SELECT id FROM t WHERE f < $1", vec![f(-1e300)], any),
            (
                "SELECT id FROM t WHERE f IN ($1, $2)",
                vec![f(10.5), f(-0.0)],
                any,
            ),
        ],
    );
}

#[test]
fn float_ranges_are_estimated_over_their_values() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), true);
    let f = Value::Float;
    check(
        &db,
        &[
            (
                "SELECT id FROM w WHERE f > $1 AND f < $2",
                vec![f(-1.0), f(1.0)],
                Some(true),
            ),
            (
                "SELECT id FROM w WHERE f BETWEEN $1 AND $2",
                vec![f(10.0), f(12.0)],
                Some(true),
            ),
            (
                "SELECT id FROM w WHERE f >= $1 AND f < $2",
                vec![f(-300.0), f(-298.0)],
                Some(true),
            ),
        ],
    );
}

#[test]
fn integer_keys_compared_with_floats_answer_as_the_scan() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), true);
    let f = Value::Float;
    let eq = "SELECT id FROM t WHERE i = $1";
    let (yes, no, any) = (Some(true), Some(false), None);
    check(
        &db,
        &[
            (eq, vec![f(5.0)], yes),
            (eq, vec![f(P53)], yes),
            (eq, vec![f(-P63)], yes),
            (eq, vec![Value::Integer(i64::MAX)], yes),
            (eq, vec![f(5.5)], any),
            (eq, vec![f(P63)], any),
            (eq, vec![f(f64::INFINITY)], any),
            (eq, vec![f(f64::NAN)], no),
            (
                "SELECT id FROM t WHERE i > $1 AND i < $2",
                vec![f(4.5), f(20.5)],
                yes,
            ),
            (
                "SELECT id FROM t WHERE i >= $1 AND i <= $2",
                vec![f(4.5), f(4.9)],
                any,
            ),
            ("SELECT id FROM t WHERE i > $1", vec![f(P63)], any),
            ("SELECT id FROM t WHERE i >= $1", vec![f(P53)], any),
            (
                "SELECT id FROM t WHERE i > $1",
                vec![Value::Integer(i64::MAX)],
                any,
            ),
        ],
    );
}

#[test]
fn nan_keys_keep_the_join_and_subquery_contracts() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), false);
    let members = vec![Value::Float(f64::NAN), Value::Float(10.5)];
    let probe_join = "SELECT t.id FROM a JOIN t ON t.f = a.x LIMIT 100";
    let plan: String = db
        .query(&format!("EXPLAIN {probe_join}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert!(plan.contains("Index Nested Loop"), "{plan}");
    let cases: [(&str, &[Value]); 5] = [
        ("SELECT t.id FROM a JOIN t ON t.f = a.x", &[]),
        (probe_join, &[]),
        ("SELECT id FROM t WHERE f IN (SELECT x FROM a)", &[]),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM a WHERE a.x = t.f)",
            &[],
        ),
        ("SELECT id FROM t WHERE f IN ($1, $2)", &members),
    ];
    // The same table and indexes before and after the seal: only the reads
    // of sealed rows differ
    let hot: Vec<Vec<i64>> = cases.iter().map(|(sql, p)| ids(&db, sql, p)).collect();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for ((sql, params), want) in cases.iter().zip(hot) {
        assert_eq!(ids(&db, sql, params), want, "{sql}");
    }
}

#[test]
fn indexes_the_side_file_does_not_hold_are_scanned() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE v (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, b BOOLEAN, c BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX v_i ON v(i) USING HASH", ())
        .unwrap();
    db.execute("CREATE INDEX v_f ON v(f) USING HASH", ())
        .unwrap();
    db.execute("CREATE INDEX v_b ON v(b) USING BTREE", ())
        .unwrap();
    db.execute("CREATE INDEX v_c ON v(c)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    for id in 0..ROWS {
        db.execute(
            "INSERT INTO v VALUES ($1, $2, $3, $4, $5)",
            vec![
                Value::Integer(id),
                Value::Integer(id),
                Value::Float(id as f64),
                Value::Boolean(id == 7),
                Value::Boolean(id == 7),
            ],
        )
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for (sql, value) in [
        ("SELECT id FROM v WHERE i = $1", Value::Integer(7)),
        ("SELECT id FROM v WHERE f = $1", Value::Float(7.0)),
        ("SELECT id FROM v WHERE b = $1", Value::Boolean(true)),
        ("SELECT id FROM v WHERE c = $1", Value::Boolean(true)),
    ] {
        let before = side_reads(&db);
        assert_eq!(ids(&db, sql, std::slice::from_ref(&value)), [7], "{sql}");
        assert_eq!(side_reads(&db), before, "no side file read: {sql}");
    }
}

#[test]
fn a_float_entry_holds_no_nan_and_records_its_domain() {
    use stoolap::storage::volume::secondary::IndexFile;
    use stoolap::storage::volume::side_index::Domain;
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), true);
    let identity = db.engine().index_identity("t", "t_f").unwrap();
    let sides: Vec<_> = std::fs::read_dir(dir.path().join("volumes").join("t"))
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("sidx"))
        .collect();
    assert_eq!(sides.len(), 1, "{sides:?}");
    let side = IndexFile::open(&sides[0], 1).unwrap();
    let entry = side.directory().entry(identity).expect("an entry for t_f");
    assert_eq!(entry.domain, Some(Domain::Float));
    // Every row but the NULL and the two NaNs
    assert_eq!(entry.n_positions, ROWS as u64 - 3);
}

#[test]
fn probes_find_sealed_keys_stored_another_way() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    // One zero of one sign only, and keys the probes ask for in the
    // other numeric type
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, f FLOAT, i INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX s_f ON s(f)", ()).unwrap();
    db.execute("CREATE INDEX s_i ON s(i)", ()).unwrap();
    db.execute("INSERT INTO s VALUES (0, -0.0, 5)", ()).unwrap();
    db.execute("INSERT INTO s VALUES (1, 7.0, 6)", ()).unwrap();
    for id in 2..300 {
        db.execute(&format!("INSERT INTO s VALUES ({id}, 10.0, 10)"), ())
            .unwrap();
    }
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, f FLOAT, x FLOAT, k INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO p VALUES (1, 0.0, 5.0, 7)", ())
        .unwrap();
    let joins = [
        "SELECT s.id FROM p JOIN s ON p.f = s.f LIMIT 10",
        "SELECT s.id FROM p JOIN s ON p.x = s.i LIMIT 10",
        "SELECT s.id FROM p JOIN s ON p.k = s.f LIMIT 10",
    ];
    for sql in joins {
        let plan: String = db
            .query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect();
        assert!(plan.contains("Index Nested Loop"), "{sql}: {plan}");
    }
    let cases = [
        joins[0],
        joins[1],
        joins[2],
        "SELECT p.id FROM p WHERE EXISTS (SELECT 1 FROM s WHERE s.f = p.f)",
        "SELECT p.id FROM p WHERE EXISTS (SELECT 1 FROM s WHERE s.i = p.x)",
        "SELECT p.id FROM p WHERE EXISTS (SELECT 1 FROM s WHERE s.f = p.k)",
    ];
    let hot: Vec<Vec<i64>> = cases.iter().map(|sql| ids(&db, sql, &[])).collect();
    assert!(hot.iter().all(|rows| !rows.is_empty()), "{hot:?}");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for (sql, want) in cases.iter().zip(hot) {
        assert_eq!(ids(&db, sql, &[]), want, "sealed: {sql}");
    }
}
