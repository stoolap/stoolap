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

//! A cosine distance with a NaN or Infinity component is NaN, and a VECTOR
//! column holds only finite components, whichever statement writes it

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
        "CREATE TABLE d (id INTEGER PRIMARY KEY, e VECTOR(2) DEFAULT '[NaN,1]')",
        "CREATE TABLE x (id INTEGER PRIMARY KEY)",
        "INSERT INTO x VALUES (1), (2)",
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
        // A default and a table made from a query write the value too
        "INSERT INTO d (id) VALUES (1)",
        "CREATE TABLE c AS SELECT 1 AS id, CAST('[1,NaN]' AS VECTOR) AS e",
        // The rows a table has take a column's default when it is added
        "ALTER TABLE x ADD COLUMN e VECTOR(2) DEFAULT '[NaN,1]'",
    ] {
        refused(sql, db.execute(sql, ()));
    }
    // The refused ADD COLUMN left the table as it was
    let columns = db.query("SELECT * FROM x", ()).unwrap().columns().len();
    assert_eq!(columns, 1);
    // An added column holds its dimension count from the statement on
    let sql = "ALTER TABLE x ADD COLUMN e VECTOR(2) DEFAULT '[1]'";
    let error = db.execute(sql, ()).expect_err(sql).to_string();
    assert!(error.contains("dimension mismatch"), "{sql}: {error}");
    db.execute("ALTER TABLE x ADD COLUMN f VECTOR(2)", ())
        .unwrap();
    let sql = "INSERT INTO x VALUES (3, '[1]')";
    let error = db.execute(sql, ()).expect_err(sql).to_string();
    assert!(error.contains("dimension mismatch"), "{sql}: {error}");
    db.execute("ALTER TABLE x DROP COLUMN f", ()).unwrap();
    // The engine's own call adds a column as the statement does
    let added = db.engine().create_column_with_default(
        "x",
        "e",
        stoolap::core::DataType::Vector,
        true,
        (None, Some(Value::vector(vec![f32::NAN, 1.0])), None),
        2,
    );
    let error = added.expect_err("create_column_with_default").to_string();
    assert!(error.contains(REFUSED), "{error}");
    // A JSON value is no vector, whatever its components
    let sql = "CREATE TABLE j AS SELECT id, CASE WHEN id = 1 THEN CAST('[1,2]' AS VECTOR) \
               ELSE CAST('{\"a\":1}' AS JSON) END AS e FROM x";
    let error = db.execute(sql, ()).expect_err(sql).to_string();
    assert!(error.contains("type mismatch in column 'e'"), "{error}");
    let defaulted = tempfile::NamedTempFile::new().unwrap();
    writeln!(defaulted.as_file(), "5").unwrap();
    let sql = format!(
        "COPY d (id) FROM '{}' WITH (FORMAT CSV, HEADER false)",
        defaulted.path().display()
    );
    refused(&sql, db.execute(&sql, ()));
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

/// The ids `ORDER BY VEC_DISTANCE_IP(e, '[inf,0]') LIMIT 3` gives over `rows`
/// rows, all at distance NaN (`inf * 0`) but the last three, at minus infinity
fn nearest_past_nan(name: &str, rows: i64) -> Vec<i64> {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE v (id INTEGER PRIMARY KEY, e VECTOR(2))", ())
        .unwrap();
    let ids: Vec<i64> = (1..=rows).collect();
    for batch in ids.chunks(1000) {
        let values: Vec<String> = batch
            .iter()
            .map(|&id| match id > rows - 3 {
                true => format!("({id}, '[1,0]')"),
                false => format!("({id}, '[0,1]')"),
            })
            .collect();
        let sql = format!("INSERT INTO v VALUES {}", values.join(", "));
        db.execute(&sql, ()).unwrap();
    }
    let sql = "SELECT id FROM v ORDER BY VEC_DISTANCE_IP(e, '[inf,0]') LIMIT 3";
    let mut ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn a_nan_distance_gives_its_place_in_a_nearest_neighbor_limit() {
    assert_eq!(
        nearest_past_nan("vector_non_finite_limit", 6),
        [4, 5, 6],
        "one scan"
    );
    assert_eq!(
        nearest_past_nan("vector_non_finite_limit_parallel", 20_000),
        [19_998, 19_999, 20_000],
        "scan in parallel chunks"
    );
}

/// The id an HNSW cosine index finds nearest to `[1,1,1]`, among `nearest`
/// as row 1 and 399 rows pointing elsewhere
fn indexed_nearest(name: &str, nearest: &str) -> i64 {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE h (id INTEGER PRIMARY KEY, e VECTOR(3))", ())
        .unwrap();
    let mut values = vec![format!("(1, '{nearest}')")];
    values.extend((2..=400).map(|id| format!("({id}, '[1,0,{}.{id}]')", id % 7)));
    let sql = format!("INSERT INTO h VALUES {}", values.join(", "));
    db.execute(&sql, ()).unwrap();
    db.execute(
        "CREATE INDEX hc ON h(e) USING HNSW WITH (metric = 'cosine')",
        (),
    )
    .unwrap();
    let sql = "SELECT id FROM h ORDER BY VEC_DISTANCE_COSINE(e, '[1,1,1]') LIMIT 1";
    let row = db.query(sql, ()).unwrap().next().unwrap().unwrap();
    row.get::<i64>(0).unwrap()
}

#[test]
fn an_hnsw_cosine_index_ranks_components_past_the_f32_square_range() {
    for (name, nearest) in [
        ("vector_cosine_index_overflow", "[1e30,1e30,1e30]"),
        ("vector_cosine_index_underflow", "[1e-25,1e-25,1e-25]"),
    ] {
        assert_eq!(indexed_nearest(name, nearest), 1, "{nearest}");
    }
}
