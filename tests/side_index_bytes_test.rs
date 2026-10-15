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

//! Sealed reads of TEXT and composite keys through the side file: each
//! answer matches the scan of a twin table without indexes

use std::sync::Mutex;

use stoolap::core::{DataType, Value};
use stoolap::Database;

/// The read counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

const ROWS: i64 = 3000;

fn s_of(id: i64) -> Value {
    let text = match id {
        1 => String::new(),
        2 => "a\0b".to_string(),
        3 => "a".to_string(),
        4 => "a\0".to_string(),
        5 => return Value::Null(DataType::Text),
        6 => "ab".to_string(),
        // Long common prefixes, keys of many lengths across many pages
        _ => format!("key-{:06}-{}", id, "x".repeat((id % 50) as usize)),
    };
    Value::text(text)
}

fn f_of(id: i64) -> Value {
    if id % 97 == 0 {
        Value::Float(f64::NAN)
    } else if id % 89 == 0 {
        Value::Null(DataType::Float)
    } else {
        Value::Float(id as f64)
    }
}

/// `t` with a TEXT B-tree on s, a TEXT hash on h and composites on (a, s)
/// and (a, f); `u` the same rows without indexes
fn open(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!(
                "CREATE TABLE {table} (id INTEGER PRIMARY KEY, s TEXT, h TEXT, a INTEGER, f FLOAT)"
            ),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_s ON t(s)", ()).unwrap();
    db.execute("CREATE INDEX t_h ON t(h) USING HASH", ())
        .unwrap();
    db.execute("CREATE INDEX t_as ON t(a, s)", ()).unwrap();
    db.execute("CREATE INDEX t_af ON t(a, f)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    for table in ["t", "u"] {
        let insert = format!("INSERT INTO {table} VALUES ($1, $2, $3, $4, $5)");
        for id in 0..ROWS {
            db.execute(
                &insert,
                vec![
                    Value::Integer(id),
                    s_of(id),
                    Value::text(format!("h{}", id % 500)),
                    Value::Integer(id % 400),
                    f_of(id),
                ],
            )
            .unwrap();
        }
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
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
/// is read through the side file or not as `served` says
fn check(db: &Database, cases: &[(&str, Vec<Value>, Option<bool>)]) {
    for (sql, params, served) in cases {
        let want = ids(db, &sql.replace(" t ", " u "), params);
        assert!(!want.is_empty() || served.is_none(), "rows for {sql}");
        let before = side_reads(db);
        let got = ids(db, sql, params);
        let read = side_reads(db) > before;
        assert_eq!(got, want, "{sql} {params:?}");
        if let Some(served) = served {
            assert_eq!(read, *served, "side file read: {sql} {params:?}");
        }
    }
}

fn text(s: &str) -> Value {
    Value::text(s)
}

#[test]
fn text_keys_answer_as_the_scan() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let eq = "SELECT id FROM t WHERE s = $1";
    let (yes, any) = (Some(true), None);
    check(
        &db,
        &[
            (eq, vec![text("")], yes),
            (eq, vec![text("a")], yes),
            (eq, vec![text("a\0")], yes),
            (eq, vec![text("a\0b")], yes),
            (eq, vec![text("ab")], yes),
            (
                eq,
                vec![text("key-001234-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")],
                yes,
            ),
            (eq, vec![text("key-001234-")], any),
            (eq, vec![text("zzz")], any),
            (eq, vec![Value::Integer(3)], any),
            (
                "SELECT id FROM t WHERE s >= $1 AND s < $2",
                vec![text("a"), text("b")],
                yes,
            ),
            (
                "SELECT id FROM t WHERE s > $1 AND s <= $2",
                vec![text("a"), text("a\0b")],
                yes,
            ),
            (
                "SELECT id FROM t WHERE s BETWEEN $1 AND $2",
                vec![text("key-000100"), text("key-000130")],
                yes,
            ),
            (
                "SELECT id FROM t WHERE s > $1",
                vec![text("key-002990")],
                yes,
            ),
            ("SELECT id FROM t WHERE s < $1", vec![text("a\0")], yes),
            ("SELECT id FROM t WHERE h = $1", vec![text("h17")], yes),
            ("SELECT id FROM t WHERE h = $1", vec![text("h")], any),
        ],
    );
}

#[test]
fn composite_keys_answer_as_the_scan() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let int = Value::Integer;
    let (yes, any) = (Some(true), None);
    check(
        &db,
        &[
            (
                "SELECT id FROM t WHERE a = $1 AND s = $2",
                vec![int(17), s_of(17)],
                yes,
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND s = $2",
                vec![int(1), text("")],
                yes,
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND s = $2",
                vec![int(2), text("a\0b")],
                yes,
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND s = $2",
                vec![int(3), text("a\0b")],
                any,
            ),
            ("SELECT id FROM t WHERE a = $1", vec![int(17)], yes),
            (
                "SELECT id FROM t WHERE a = $1",
                vec![Value::Float(17.0)],
                yes,
            ),
            (
                "SELECT id FROM t WHERE a = $1",
                vec![Value::Float(17.5)],
                any,
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND s > $2",
                vec![int(17), text("key-001000")],
                yes,
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND f >= $2 AND f < $3",
                vec![int(17), Value::Float(400.0), Value::Float(1300.0)],
                yes,
            ),
            // NULL and NaN components stay in the key and are found by a
            // shorter prefix, never by a comparison on them
            (
                "SELECT id FROM t WHERE a = $1 AND f = $2",
                vec![int(0), Value::Float(f64::NAN)],
                any,
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND f IS NULL",
                vec![int(89)],
                yes,
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND f > $2",
                vec![int(194), Value::Float(0.0)],
                yes,
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND f = $2",
                vec![int(17), Value::Float(817.0)],
                yes,
            ),
        ],
    );
}

#[test]
fn text_and_composite_entries_record_their_kind_and_domains() {
    use stoolap::storage::volume::secondary::IndexFile;
    use stoolap::storage::volume::side_index::{Domain, KeyKind, Outcome};
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let sides: Vec<_> = std::fs::read_dir(dir.path().join("volumes").join("t"))
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("sidx"))
        .collect();
    assert_eq!(sides.len(), 1, "{sides:?}");
    let side = IndexFile::open(&sides[0], 1).unwrap();
    let entry = |name: &str| {
        let identity = db.engine().index_identity("t", name).unwrap();
        side.directory().entry(identity).expect(name).clone()
    };
    let s = entry("t_s");
    assert_eq!((s.kind, s.outcome), (KeyKind::OrderedBytes, Outcome::Built));
    assert_eq!(s.domain, Some(Domain::Text));
    // Every row but the NULL
    assert_eq!(s.n_positions, ROWS as u64 - 1);
    let af = entry("t_af");
    assert_eq!(af.kind, KeyKind::OrderedBytes);
    assert_eq!(
        af.components.iter().map(|c| c.1).collect::<Vec<_>>(),
        [Domain::Integer, Domain::Float]
    );
    // NULL and NaN components are kept
    assert_eq!(af.n_positions, ROWS as u64);
    assert_eq!(entry("t_h").kind, KeyKind::OrderedBytes);
}

#[test]
fn text_reads_survive_a_reopen() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    drop(open(dir.path()));
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    check(
        &db,
        &[
            (
                "SELECT id FROM t WHERE s = $1",
                vec![text("a\0b")],
                Some(true),
            ),
            (
                "SELECT id FROM t WHERE a = $1 AND s = $2",
                vec![Value::Integer(17), s_of(17)],
                Some(true),
            ),
        ],
    );
}

/// The row of `PRAGMA INDEX_BACKFILL`: volumes examined and built
fn backfill(db: &Database) -> (i64, i64) {
    let rows = db.query("PRAGMA INDEX_BACKFILL", ()).unwrap();
    let columns = rows.columns().to_vec();
    let at = |name: &str| columns.iter().position(|c| c == name).unwrap();
    let (examined, built) = (at("examined"), at("built"));
    let row = rows.into_iter().next().unwrap().unwrap();
    (row.get(examined).unwrap(), row.get(built).unwrap())
}

#[test]
fn a_text_index_created_after_the_seal_is_backfilled_beside_the_integer_one() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, a INTEGER, s TEXT)"),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_a ON t(a)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    for table in ["t", "u"] {
        let insert = format!("INSERT INTO {table} VALUES ($1, $2, $3)");
        for id in 0..ROWS {
            db.execute(
                &insert,
                vec![Value::Integer(id), Value::Integer(id), s_of(id)],
            )
            .unwrap();
        }
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("CREATE INDEX t_s ON t(s)", ()).unwrap();
    let on_s = ("SELECT id FROM t WHERE s = $1", vec![text("a\0b")]);
    let on_a = ("SELECT id FROM t WHERE a = $1", vec![Value::Integer(17)]);
    check(&db, &[(on_s.0, on_s.1.clone(), Some(false))]);
    assert_eq!(backfill(&db), (1, 1));
    let served = [
        (on_s.0, on_s.1.clone(), Some(true)),
        (on_a.0, on_a.1.clone(), Some(true)),
    ];
    check(&db, &served);
    assert_eq!(backfill(&db), (0, 0), "nothing left to build");
    drop(db);
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    check(&db, &served);
}

#[test]
fn a_key_longer_than_a_page_leaves_its_index_unsupported_and_not_rebuilt() {
    use stoolap::storage::volume::secondary::IndexFile;
    use stoolap::storage::volume::side_index::Outcome;
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, a INTEGER, s TEXT)"),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_a ON t(a)", ()).unwrap();
    db.execute("CREATE INDEX t_s ON t(s)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    for table in ["t", "u"] {
        let insert = format!("INSERT INTO {table} VALUES ($1, $2, $3)");
        for id in 0..ROWS {
            let s = match id {
                9 => Value::text("x".repeat(40_000)),
                _ => s_of(id),
            };
            db.execute(&insert, vec![Value::Integer(id), Value::Integer(id), s])
                .unwrap();
        }
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let side = std::fs::read_dir(dir.path().join("volumes").join("t"))
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.extension().and_then(|e| e.to_str()) == Some("sidx"))
        .expect("a side file");
    let side = IndexFile::open(&side, 1).unwrap();
    let entry = |name: &str| {
        let identity = db.engine().index_identity("t", name).unwrap();
        side.directory().entry(identity).expect(name).outcome
    };
    assert_eq!(entry("t_s"), Outcome::Unsupported);
    assert_eq!(entry("t_a"), Outcome::Built);
    check(
        &db,
        &[
            (
                "SELECT id FROM t WHERE s = $1",
                vec![text("a\0b")],
                Some(false),
            ),
            (
                "SELECT id FROM t WHERE s = $1",
                vec![Value::text("x".repeat(40_000))],
                Some(false),
            ),
            (
                "SELECT id FROM t WHERE a = $1",
                vec![Value::Integer(17)],
                Some(true),
            ),
        ],
    );
    assert_eq!(
        backfill(&db),
        (0, 0),
        "an unsupported entry is not built again"
    );
}

#[test]
fn text_key_probes_answer_as_the_scan() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for form in ["BTREE", "HASH"] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=3600",
            dir.path().display()
        ))
        .unwrap();
        for table in ["t", "u", "o"] {
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k TEXT)"),
                (),
            )
            .unwrap();
        }
        db.execute(&format!("CREATE INDEX t_k ON t(k) USING {form}"), ())
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        for table in ["t", "u"] {
            let insert = format!("INSERT INTO {table} VALUES ($1, $2)");
            for id in 0..ROWS {
                let k = match id % 2 {
                    0 => s_of(id),
                    _ => text(&format!("h{}", id % 500)),
                };
                db.execute(&insert, vec![Value::Integer(id), k]).unwrap();
            }
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        for (id, k) in [
            (1, text("a\0b")),
            (2, text("")),
            (3, s_of(1234)),
            (4, text("absent")),
            (5, text("h17")),
            (6, Value::Null(DataType::Text)),
        ] {
            db.execute("INSERT INTO o VALUES ($1, $2)", vec![Value::Integer(id), k])
                .unwrap();
        }
        let join = "SELECT t.id FROM o JOIN t ON t.k = o.k LIMIT 100";
        let cases = [
            (
                "SELECT id FROM t WHERE k IN ($1, $2, $3)",
                vec![text("a\0b"), text("h17"), s_of(1234)],
            ),
            (join, vec![]),
            (
                "SELECT id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.k = o.k) LIMIT 5",
                vec![],
            ),
        ];
        for (sql, params) in &cases {
            let want = ids(&db, &sql.replace(" t ", " u ").replace("t.", "u."), params);
            assert!(!want.is_empty(), "rows for {sql}");
            assert_eq!(ids(&db, sql, params), want, "{form}: {sql}");
        }
    }
}
