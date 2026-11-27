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

//! A cosine distance with a NaN or Infinity component is NaN, as the L2
//! and inner product distances are, and a VECTOR column holds only finite
//! components, whichever statement writes it

use std::io::Write;

use stoolap::core::Value;
use stoolap::Database;

/// The one float `sql` answers
fn distance(db: &Database, sql: &str) -> f64 {
    let row = db.query(sql, ()).unwrap().next().unwrap().unwrap();
    match row.get_value(0).unwrap() {
        Value::Float(value) => *value,
        other => panic!("{sql}: {other:?}"),
    }
}

#[test]
fn a_cosine_distance_with_a_non_finite_component_is_nan() {
    let db = Database::open("memory://vector_non_finite_distance").unwrap();
    for sql in [
        "SELECT VEC_DISTANCE_COSINE('[NaN,1,1]', '[1,2,3]')",
        "SELECT VEC_DISTANCE_COSINE('[inf,1,1]', '[1,2,3]')",
        "SELECT VEC_DISTANCE_COSINE('[1,2,3]', '[1,-inf,3]')",
        "SELECT VEC_DISTANCE_L2('[NaN,NaN,NaN]', '[1,2,3]')",
    ] {
        assert!(distance(&db, sql).is_nan(), "{sql}");
    }
    // A zero vector is the farthest, a vector from itself at 0
    assert_eq!(
        distance(&db, "SELECT VEC_DISTANCE_COSINE('[0,0,0]', '[1,2,3]')"),
        1.0
    );
    assert_eq!(
        distance(
            &db,
            "SELECT VEC_DISTANCE_COSINE('[0.1,0.2,0.3]', '[0.1,0.2,0.3]')"
        ),
        0.0
    );
    // Large finite components are not lost: 1 - 6 / (sqrt(3) * sqrt(14))
    let large = distance(
        &db,
        "SELECT VEC_DISTANCE_COSINE('[1e30,1e30,1e30]', '[1,2,3]')",
    );
    assert!((large - 0.074_179_9).abs() < 1e-6, "{large}");
}

/// The error a statement writing a non-finite component to column `e` of
/// table `v` must raise
const REFUSED: &str = "VECTOR column 'e' cannot hold NaN or Infinity";

#[test]
fn a_vector_column_refuses_a_non_finite_component() {
    let db = Database::open("memory://vector_non_finite_column").unwrap();
    for sql in [
        "CREATE TABLE v (id INTEGER PRIMARY KEY, e VECTOR(3))",
        "INSERT INTO v VALUES (1, '[0,1,0]')",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, e TEXT)",
        "INSERT INTO w VALUES (1, '[1,NaN,1]')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let refused = |sql: &str, result: Result<i64, stoolap::Error>| {
        let error = result.expect_err(sql).to_string();
        assert!(error.contains(REFUSED), "{sql}: {error}");
    };
    for sql in [
        "INSERT INTO v VALUES (2, '[NaN,0,0]')",
        "INSERT INTO v VALUES (2, '[1,inf,0]')",
        "UPDATE v SET e = '[0,0,-inf]' WHERE id = 1",
        "INSERT INTO v VALUES (1, '[1,1,1]') ON CONFLICT (id) DO UPDATE SET e = '[NaN,1,1]'",
        "ALTER TABLE w MODIFY COLUMN e VECTOR(3)",
    ] {
        refused(sql, db.execute(sql, ()));
    }
    let sql = "INSERT INTO v VALUES ($1, $2)";
    refused(sql, db.execute(sql, (3, "[1,2,NaN]")));
    let csv = tempfile::NamedTempFile::new().unwrap();
    writeln!(csv.as_file(), "4,\"[1,inf,1]\"").unwrap();
    let sql = format!(
        "COPY v FROM '{}' WITH (FORMAT CSV, HEADER false)",
        csv.path().display()
    );
    refused(&sql, db.execute(&sql, ()));
    // Nothing was written
    let rows: Vec<String> = db
        .query("SELECT id, e FROM v ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            format!(
                "{}|{}",
                row.get_value(0).unwrap(),
                row.get_value(1).unwrap()
            )
        })
        .collect();
    assert_eq!(rows, ["1|[0.0, 1.0, 0.0]"]);
}
