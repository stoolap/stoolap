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

//! A value read as a condition is BOOLEAN, an INTEGER (true when not zero)
//! or NULL; any other type fails the statement, in every place a condition
//! stands, when the statement is parsed if it writes the type and else on
//! the first row that reads it

use stoolap::{Database, Value};

/// t holds one row: b TRUE, i 2, z 0, n NULL, then a FLOAT, a TEXT, a
/// TIMESTAMP and a JSON value; e is empty
fn fill(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b BOOLEAN, i INTEGER, z INTEGER, n INTEGER, \
         f FLOAT, s TEXT, ts TIMESTAMP, j JSON)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, TRUE, 2, 0, NULL, 0.5, 'x', '2024-01-01 00:00:00', '{}')",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE e (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
}

/// Each place a condition stands, and its answer for a condition that is
/// TRUE, FALSE and NULL
const PLACES: &[(&str, [&str; 3])] = &[
    ("SELECT COUNT(*) FROM t WHERE {c}", ["1", "0", "0"]),
    (
        "SELECT COUNT(*) FROM (SELECT 1 AS one FROM t GROUP BY {c} HAVING {c}) d",
        ["1", "0", "0"],
    ),
    ("SELECT COUNT(*) FROM t JOIN t AS u ON {c}", ["1", "0", "0"]),
    (
        "SELECT COUNT(*) FROM t WHERE {c} AND id = 1",
        ["1", "0", "0"],
    ),
    (
        "SELECT COUNT(*) FROM t WHERE {c} OR id = 0",
        ["1", "0", "0"],
    ),
    ("SELECT COUNT(*) FROM t WHERE NOT {c}", ["0", "1", "0"]),
    (
        "SELECT COUNT(*) FROM t WHERE {c} XOR id = 0",
        ["1", "0", "0"],
    ),
    (
        "SELECT CASE WHEN {c} THEN 'y' ELSE 'n' END FROM t",
        ["y", "n", "n"],
    ),
    ("SELECT COUNT(*) FILTER (WHERE {c}) FROM t", ["1", "0", "0"]),
    (
        "SELECT COUNT(*) FILTER (WHERE {c}) OVER () FROM t",
        ["1", "0", "0"],
    ),
    ("SELECT IIF({c}, 'y', 'n') FROM t", ["y", "n", "n"]),
    ("UPDATE t SET id = id WHERE {c}", ["1", "0", "0"]),
];

/// The answer of `sql`: its first value, or the rows an UPDATE changed
fn answer(db: &Database, sql: &str, params: Vec<Value>) -> Result<String, String> {
    if sql.starts_with("UPDATE") {
        return db
            .execute(sql, params)
            .map(|n| n.to_string())
            .map_err(|e| e.to_string());
    }
    let mut rows = db.query(sql, params).map_err(|e| e.to_string())?;
    let row = rows.next().ok_or("no row")?.map_err(|e| e.to_string())?;
    Ok(row.get_value(0).unwrap().to_string())
}

fn refusal(kind: &str) -> String {
    format!("a condition must be BOOLEAN or INTEGER, got {kind}")
}

/// What every place answers for each column of t; one line per mismatch
fn wrong_answers(db: &Database) -> Vec<String> {
    let mut wrong = Vec::new();
    for (place, [yes, no, unknown]) in PLACES {
        let columns = [
            ("t.b", Ok(*yes)),
            ("t.i", Ok(*yes)),
            ("t.z", Ok(*no)),
            ("t.n", Ok(*unknown)),
            ("t.f", Err("FLOAT")),
            ("t.s", Err("TEXT")),
            ("t.ts", Err("TIMESTAMP")),
            ("t.j", Err("JSON")),
        ];
        for (column, want) in columns {
            let sql = place.replace("{c}", column);
            let got = answer(db, &sql, vec![]);
            let fits = match (&got, want) {
                (Ok(got), Ok(want)) => got == want,
                (Err(got), Err(kind)) => got.contains(&refusal(kind)),
                _ => false,
            };
            if !fits {
                wrong.push(format!("{sql}: {got:?}, want {want:?}"));
            }
        }
    }
    wrong
}

#[test]
fn every_condition_takes_boolean_integer_and_null_and_refuses_the_rest() {
    let db = Database::open("memory://condition_rule_hot").unwrap();
    fill(&db);
    let wrong = wrong_answers(&db);
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn every_condition_answers_alike_on_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    fill(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert!(!db.engine().volume_stats().is_empty(), "sealed");
    let wrong = wrong_answers(&db);
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn a_condition_whose_type_the_statement_writes_fails_when_it_is_parsed() {
    let db = Database::open("memory://condition_rule_parsed").unwrap();
    fill(&db);
    let mut wrong = Vec::new();
    let places = PLACES.iter().map(|(place, _)| *place).chain([
        "INSERT INTO t (id) VALUES (1) ON CONFLICT (id) DO UPDATE SET i = 1 WHERE {c}",
        "DELETE FROM t WHERE {c}",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, x INTEGER CHECK ({c}))",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, CHECK ({c}))",
    ]);
    for place in places {
        for (literal, kind) in [
            ("0.5", "FLOAT"),
            ("'true'", "TEXT"),
            ("CAST(1 AS TIMESTAMP)", "TIMESTAMP"),
            ("INTERVAL '1 day'", "TEXT"),
        ] {
            let sql = place.replace("{c}", literal);
            let got = db.execute(&sql, ()).map(|_| ()).map_err(|e| e.to_string());
            match &got {
                Err(e) if e.contains("parse error") && e.contains(&refusal(kind)) => {}
                _ => wrong.push(format!("{sql}: {got:?}")),
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn a_parameter_read_as_a_condition_follows_the_rule_on_each_execution() {
    let db = Database::open("memory://condition_rule_params").unwrap();
    fill(&db);
    let sql = "SELECT COUNT(*) FROM t WHERE $1 AND id = 1";
    let statement = db.prepare(sql).unwrap();
    for (param, want) in [
        (Value::Boolean(true), Ok("1".to_string())),
        (Value::Integer(-1), Ok("1".to_string())),
        (Value::Integer(0), Ok("0".to_string())),
        (Value::Float(1.0), Err(refusal("FLOAT"))),
        (Value::text("1"), Err(refusal("TEXT"))),
        (Value::Boolean(false), Ok("0".to_string())),
    ] {
        let got = statement
            .query((param.clone(),))
            .and_then(|mut rows| {
                rows.next()
                    .expect("a row")
                    .map(|row| row.get_value(0).unwrap().to_string())
            })
            .map_err(|e| e.to_string());
        match (&got, &want) {
            (Ok(got), Ok(want)) if got == want => {}
            (Err(got), Err(want)) if got.contains(want) => {}
            _ => panic!("{sql} with {param:?}: {got:?}, want {want:?}"),
        }
    }
}

#[test]
fn a_column_condition_over_an_empty_table_reads_no_row_so_refuses_nothing() {
    let db = Database::open("memory://condition_rule_empty").unwrap();
    fill(&db);
    assert_eq!(
        answer(&db, "SELECT COUNT(*) FROM e WHERE f", vec![]),
        Ok("0".into())
    );
}

#[test]
fn a_check_on_a_column_a_condition_does_not_take_is_refused_when_declared() {
    let db = Database::open("memory://condition_rule_check").unwrap();
    for (sql, kind) in [
        (
            "CREATE TABLE c (id INTEGER PRIMARY KEY, s TEXT CHECK (s))",
            "TEXT",
        ),
        (
            "CREATE TABLE c (id INTEGER PRIMARY KEY, f FLOAT, b BOOLEAN, CHECK (b AND NOT f))",
            "FLOAT",
        ),
        (
            "CREATE TABLE c (id INTEGER PRIMARY KEY, t TIMESTAMP, CHECK (t))",
            "TIMESTAMP",
        ),
    ] {
        let err = db.execute(sql, ()).unwrap_err().to_string();
        assert!(err.contains(&refusal(kind)), "{sql}: {err}");
    }
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, b BOOLEAN CHECK (b), i INTEGER, s TEXT, \
         CHECK (b OR i))",
        (),
    )
    .unwrap();
    for (sql, kind) in [
        ("ALTER TABLE c ADD COLUMN x FLOAT CHECK (x)", "FLOAT"),
        ("ALTER TABLE c MODIFY COLUMN i TEXT CHECK (i)", "TEXT"),
    ] {
        let err = db.execute(sql, ()).unwrap_err().to_string();
        assert!(err.contains(&refusal(kind)), "{sql}: {err}");
    }
}

#[test]
fn a_check_whose_value_is_computed_refuses_the_row_that_reads_another_type() {
    let db = Database::open("memory://condition_rule_check_rows").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, s TEXT CHECK (s || ''))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, NULL)", ()).unwrap();
    let err = db
        .execute("INSERT INTO c VALUES (2, 'x')", ())
        .unwrap_err()
        .to_string();
    assert!(err.contains(&refusal("TEXT")), "{err}");
}

/// Every row of `sql`, or the error that ends it
fn all_rows(db: &Database, sql: &str) -> Result<usize, String> {
    db.query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map(|rows| rows.len())
        .map_err(|e| e.to_string())
}

#[test]
fn a_cached_answer_reads_a_condition_before_the_constant_beside_it() {
    for warm in [false, true] {
        let db = Database::open(&format!("memory://condition_rule_cache_{warm}")).unwrap();
        fill(&db);
        if warm {
            assert_eq!(all_rows(&db, "SELECT * FROM t WHERE TRUE"), Ok(1));
            assert_eq!(all_rows(&db, "SELECT * FROM t WHERE FALSE"), Ok(0));
        }
        let sql = "SELECT * FROM t WHERE t.f OR TRUE";
        let got = all_rows(&db, sql);
        assert!(
            matches!(&got, Err(e) if e.contains(&refusal("FLOAT"))),
            "{sql}, cache warm {warm}: {got:?}"
        );
    }
}

#[test]
fn a_prepared_conjunction_reads_each_step_before_the_next_runs() {
    let db = Database::open("memory://condition_rule_conjunction").unwrap();
    fill(&db);
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, f FLOAT, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1, 0.5, 2)", ())
        .unwrap();
    let sql = "SELECT COUNT(*) FROM t WHERE EXISTS \
               (SELECT 1 FROM u WHERE u.k = t.id AND u.f AND u.v > $1 * 9223372036854775807)";
    // With 2 the last step overflows before any row is read; u.f, read
    // before it, is the statement's error
    let got = db
        .prepare(sql)
        .unwrap()
        .query((2i64,))
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map(|rows| rows.len())
        .map_err(|e| e.to_string());
    assert!(
        matches!(&got, Err(e) if e.contains(&refusal("FLOAT"))),
        "{sql} with 2: {got:?}"
    );
}
