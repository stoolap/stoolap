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

//! A literal with a TIMESTAMP or DATE hint is a TIMESTAMP wherever it is
//! read: written into a TEXT column it is stored as `CAST(... AS TEXT)`
//! gives it, by every write path, and a projection returns it as a TIMESTAMP.

use stoolap::{Database, Value};

/// A hinted literal and the TEXT a TEXT column stores for it
const LITERALS: [(&str, &str); 5] = [
    (
        "TIMESTAMP '2024-01-01 12:00:00'",
        "2024-01-01T12:00:00+00:00",
    ),
    (
        "TIMESTAMP '2024-01-01 12:00:00.123'",
        "2024-01-01T12:00:00.123+00:00",
    ),
    (
        "TIMESTAMP '2024-01-01 12:00:00.123456'",
        "2024-01-01T12:00:00.123456+00:00",
    ),
    (
        "TIMESTAMP '2024-01-01 12:00:00.123456789'",
        "2024-01-01T12:00:00.123456789+00:00",
    ),
    ("DATE '2024-01-01'", "2024-01-01T00:00:00+00:00"),
];

fn text_of(db: &Database, id: i64) -> Result<Value, String> {
    db.query(&format!("SELECT s FROM t WHERE id = {id}"), ())
        .map_err(|e| e.to_string())?
        .next()
        .ok_or_else(|| "no row".to_string())?
        .map(|row| row.get_value(0).cloned().unwrap())
        .map_err(|e| e.to_string())
}

/// Every wrong answer for `literal`, which a TEXT column stores as `stored`
fn wrong_answers(n: usize, literal: &str, stored: &str, wrong: &mut Vec<String>) {
    let db = Database::open(&format!("memory://literal_write_{n}")).unwrap();
    db.execute(
        &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, d TEXT DEFAULT {literal})"),
        (),
    )
    .unwrap();
    let want = Value::text(stored);
    let mut check = |context: &str, id: i64, write: &[String]| {
        for sql in write {
            if let Err(e) = db.execute(sql, ()) {
                wrong.push(format!("{literal} {context}: {sql} failed: {e}"));
                return;
            }
        }
        let got = text_of(&db, id);
        if got.as_ref() != Ok(&want) {
            wrong.push(format!(
                "{literal} {context}: stored {got:?}, want {want:?}"
            ));
        }
        let read_back = format!(
            "SELECT CAST(s AS TIMESTAMP) = {literal}, s = {literal} FROM t WHERE id = {id}"
        );
        let answers: Result<Vec<Value>, String> = db
            .query(&read_back, ())
            .map_err(|e| e.to_string())
            .and_then(|mut rows| {
                let row = rows.next().ok_or("no row")?.map_err(|e| e.to_string())?;
                Ok((0..2).map(|i| row.get_value(i).cloned().unwrap()).collect())
            });
        let both_true = vec![Value::Boolean(true), Value::Boolean(true)];
        if answers.as_ref() != Ok(&both_true) {
            wrong.push(format!("{literal} {context}: read back {answers:?}"));
        }
    };
    check(
        "INSERT VALUES",
        1,
        &[format!("INSERT INTO t (id, s) VALUES (1, {literal})")],
    );
    check(
        "INSERT VALUES, two rows",
        2,
        &[format!(
            "INSERT INTO t (id, s) VALUES (2, {literal}), (3, {literal})"
        )],
    );
    check(
        "INSERT SELECT",
        4,
        &[format!("INSERT INTO t (id, s) SELECT 4, {literal}")],
    );
    check(
        "INSERT VALUES, COALESCE",
        5,
        &[format!(
            "INSERT INTO t (id, s) VALUES (5, COALESCE({literal}, NULL))"
        )],
    );
    check(
        "UPDATE SET by primary key",
        6,
        &[
            "INSERT INTO t (id, s) VALUES (6, 'x')".to_string(),
            format!("UPDATE t SET s = {literal} WHERE id = 6"),
        ],
    );
    check(
        "UPDATE SET by a scan",
        7,
        &[
            "INSERT INTO t (id, s) VALUES (7, 'y')".to_string(),
            format!("UPDATE t SET s = {literal} WHERE s = 'y'"),
        ],
    );
    check(
        "upsert",
        8,
        &[
            "INSERT INTO t (id, s) VALUES (8, 'x')".to_string(),
            format!("INSERT INTO t (id, s) VALUES (8, 'z') ON CONFLICT (id) DO UPDATE SET s = {literal}"),
        ],
    );
    check(
        "explicit transaction",
        9,
        &[
            "BEGIN".to_string(),
            format!("INSERT INTO t (id, s) VALUES (9, {literal})"),
            "COMMIT".to_string(),
        ],
    );
    check(
        "DEFAULT",
        10,
        &[
            "INSERT INTO t (id) VALUES (10)".to_string(),
            "UPDATE t SET s = d WHERE id = 10".to_string(),
        ],
    );
}

#[test]
fn test_a_hinted_literal_is_written_as_its_cast_by_every_path() {
    let mut wrong = Vec::new();
    for (n, (literal, stored)) in LITERALS.iter().enumerate() {
        wrong_answers(n, literal, stored, &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_a_hinted_literal_is_projected_as_a_timestamp() {
    let db = Database::open("memory://literal_projection").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1, 1), (2, 2)", ())
        .unwrap();
    let mut wrong = Vec::new();
    for (literal, _) in LITERALS {
        let want = db
            .query(&format!("SELECT {literal}"), ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get_value(0)
            .cloned()
            .unwrap();
        if !matches!(want, Value::Timestamp(_)) {
            wrong.push(format!("SELECT {literal} gave {want:?}"));
            continue;
        }
        for sql in [
            format!("SELECT COALESCE({literal}, NULL) FROM p ORDER BY id LIMIT 1"),
            format!("SELECT COALESCE(NULL, {literal}) FROM p ORDER BY id LIMIT 1"),
            format!("SELECT CASE WHEN v = 1 THEN {literal} END FROM p ORDER BY id LIMIT 1"),
            format!("SELECT CASE WHEN v = 2 THEN 0 ELSE {literal} END FROM p ORDER BY id LIMIT 1"),
            format!("SELECT COALESCE({literal}, NULL) FROM p WHERE id = 1"),
        ] {
            let got = db
                .query(&sql, ())
                .map_err(|e| e.to_string())
                .and_then(|mut rows| {
                    let row = rows.next().ok_or("no row")?.map_err(|e| e.to_string())?;
                    Ok(row.get_value(0).cloned().unwrap())
                });
            if got.as_ref() != Ok(&want) {
                wrong.push(format!("{sql} gave {got:?}, want {want:?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_a_stored_expression_keeps_the_literal_hint() {
    let mut wrong = Vec::new();
    for (n, (literal, stored)) in LITERALS.iter().enumerate() {
        let db = Database::open(&format!("memory://literal_check_{n}")).unwrap();
        db.execute(
            &format!("CREATE TABLE c (id INTEGER PRIMARY KEY, s TEXT CHECK (s <> {literal}))"),
            (),
        )
        .unwrap();
        if db
            .execute(&format!("INSERT INTO c VALUES (1, '{stored}')"), ())
            .is_ok()
        {
            wrong.push(format!("CHECK (s <> {literal}) let '{stored}' in"));
        }
        let shown: String = db
            .query("SHOW CREATE TABLE c", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(1)
            .unwrap();
        if !shown.contains(literal) {
            wrong.push(format!("SHOW CREATE TABLE for {literal} gave {shown}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_a_typed_default_survives_wal_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (n, (literal, stored)) in LITERALS.iter().enumerate() {
        let dsn = format!(
            "file://{}/wal_{n}?checkpoint_on_close=off&checkpoint_interval=0",
            dir.path().display()
        );
        let want_ts = {
            let db = Database::open(&dsn).unwrap();
            db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
                .unwrap();
            db.execute("INSERT INTO t VALUES (1), (2)", ()).unwrap();
            db.execute(
                &format!("ALTER TABLE t ADD COLUMN d TIMESTAMP DEFAULT {literal}"),
                (),
            )
            .unwrap();
            db.execute(
                &format!("ALTER TABLE t ADD COLUMN s TEXT DEFAULT {literal}"),
                (),
            )
            .unwrap();
            let want: Value = db
                .query(&format!("SELECT {literal}"), ())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .get_value(0)
                .cloned()
                .unwrap();
            let _ = db.close();
            want
        };
        let db = Database::open(&dsn).unwrap();
        let want = vec![
            vec![Value::Integer(1), want_ts.clone(), Value::text(*stored)],
            vec![Value::Integer(2), want_ts.clone(), Value::text(*stored)],
        ];
        let got: Result<Vec<Vec<Value>>, String> = db
            .query("SELECT id, d, s FROM t ORDER BY id", ())
            .map_err(|e| e.to_string())
            .and_then(|rows| {
                rows.map(|row| {
                    row.map(|row| (0..3).map(|i| row.get_value(i).cloned().unwrap()).collect())
                        .map_err(|e| e.to_string())
                })
                .collect()
            });
        if got.as_ref() != Ok(&want) {
            wrong.push(format!(
                "{literal} after recovery gave {got:?}, want {want:?}"
            ));
        }
        let _ = db.close();
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_an_inline_concat_answers_as_the_vm() {
    let db = Database::open("memory://literal_concat").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, i INTEGER, f FLOAT, b BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 'a', 7, 1e20, TRUE), (2, NULL, 8, -0.5, FALSE), (3, 'c', NULL, 2.5, NULL)",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for expr in [
        "TIMESTAMP '2024-01-01 12:00:00' || s",
        "s || TIMESTAMP '2024-01-01 12:00:00'",
        "i || s",
        "s || i",
        "f || s",
        "s || f",
        "i || '-' || s",
        "b || s",
        "s || '-' || b",
    ] {
        let inline: Vec<Value> = db
            .query(&format!("SELECT {expr} FROM t ORDER BY id LIMIT 3"), ())
            .unwrap()
            .map(|row| row.unwrap().get_value(0).cloned().unwrap())
            .collect();
        for (k, got) in (1..=3).zip(&inline) {
            let vm = db
                .query(&format!("SELECT {expr} FROM t WHERE id = {k}"), ())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .get_value(0)
                .cloned()
                .unwrap();
            let same = match (got, &vm) {
                (Value::Null(_), Value::Null(_)) => true,
                (a, b) => a == b,
            };
            if !same {
                wrong.push(format!("{expr} on row {k}: inline {got:?}, VM {vm:?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn test_a_default_computed_at_alter_survives_wal_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (n, default) in ["NOW()", "TIMESTAMP '12:00:00'", "CURRENT_TIMESTAMP"]
        .iter()
        .enumerate()
    {
        let dsn = format!(
            "file://{}/alter_{n}?checkpoint_on_close=off&checkpoint_interval=0",
            dir.path().display()
        );
        let read = |db: &Database| -> Result<Vec<Vec<Value>>, String> {
            db.query("SELECT id, d, s FROM t ORDER BY id", ())
                .map_err(|e| e.to_string())
                .and_then(|rows| {
                    rows.map(|row| {
                        row.map(|row| (0..3).map(|i| row.get_value(i).cloned().unwrap()).collect())
                            .map_err(|e| e.to_string())
                    })
                    .collect()
                })
        };
        let before = {
            let db = Database::open(&dsn).unwrap();
            db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
                .unwrap();
            db.execute("INSERT INTO t VALUES (1), (2)", ()).unwrap();
            db.execute(
                &format!("ALTER TABLE t ADD COLUMN d TIMESTAMP DEFAULT {default}"),
                (),
            )
            .unwrap();
            db.execute(
                &format!("ALTER TABLE t ADD COLUMN s TEXT DEFAULT {default}"),
                (),
            )
            .unwrap();
            let before = read(&db);
            let _ = db.close();
            before
        };
        std::thread::sleep(std::time::Duration::from_millis(5));
        let db = Database::open(&dsn).unwrap();
        let after = read(&db);
        let filled = before.as_ref().is_ok_and(|rows| {
            rows.iter()
                .all(|row| !row[1].is_null() && !row[2].is_null())
        });
        if !filled || after != before {
            wrong.push(format!(
                "DEFAULT {default}: before {before:?}, after recovery {after:?}"
            ));
        }
        let _ = db.close();
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
