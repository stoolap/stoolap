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

//! A nearest-vectors search whose query vector an uncorrelated scalar
//! subquery gives ("rows like this one") is answered by the HNSW index, as
//! one with a literal or a parameter is.

use stoolap::Database;

/// Ten rows on a line: row k at x = 0, 2, 5, 9, 14, 20, 27, 35, 44, 55. Row
/// 7 (x = 27) is at L2 distance 0, 7, 8, 13, 17 from rows 7, 6, 8, 5, 9, and
/// at 18 or more from every other row: no two rows tie
fn docs(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .unwrap();
    for (id, x) in [0, 2, 5, 9, 14, 20, 27, 35, 44, 55].into_iter().enumerate() {
        db.execute(
            &format!("INSERT INTO docs VALUES ({}, '[{x}.0, 0.0, 0.0]')", id + 1),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX idx_emb ON docs(emb) USING HNSW", ())
        .unwrap();
    db
}

/// The five rows nearest row 7, nearest first, and their distances
const NEAREST: [(i64, f64); 5] = [(7, 0.0), (6, 7.0), (8, 8.0), (5, 13.0), (9, 17.0)];

fn plan(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

const BY_ALIAS: &str = "SELECT id, VEC_DISTANCE_L2(emb, (SELECT emb FROM docs WHERE id = 7)) AS d \
                        FROM docs ORDER BY d LIMIT 5";
const BY_KEY: &str = "SELECT id FROM docs \
                      ORDER BY VEC_DISTANCE_L2(emb, (SELECT emb FROM docs WHERE id = 7)) LIMIT 5";

#[test]
fn a_scalar_subquery_query_vector_is_planned_through_the_index() {
    let db = docs("nearest_subquery_plan");
    for sql in [BY_ALIAS, BY_KEY] {
        let plan = plan(&db, sql);
        assert!(
            plan.contains("NearestHnsw docs emb L2"),
            "{sql}\nExpected NearestHnsw, got:\n{plan}"
        );
    }
}

#[test]
fn a_scalar_subquery_query_vector_finds_the_nearest_rows() {
    let db = docs("nearest_subquery_rows");
    let found: Vec<(i64, f64)> = db
        .query(BY_ALIAS, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<f64>(1).unwrap())
        })
        .collect();
    assert_eq!(found, NEAREST, "{BY_ALIAS}");
    let expected: Vec<i64> = NEAREST.iter().map(|&(id, _)| id).collect();
    assert_eq!(ids(&db, BY_KEY), expected, "{BY_KEY}");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn the_index_answers_a_scalar_subquery_query_vector_when_it_runs() {
    let db = docs("nearest_subquery_answered");
    for sql in [BY_ALIAS, BY_KEY] {
        let before = stoolap::test_failpoints::nearest_index_answers();
        assert_eq!(ids(&db, sql).len(), 5);
        assert_eq!(
            stoolap::test_failpoints::nearest_index_answers() - before,
            1,
            "{sql}: the HNSW index answers the run, not a scan of every row"
        );
    }
}

#[test]
fn a_positional_parameter_query_vector_still_finds_its_rows() {
    let db = docs("nearest_subquery_param");
    let found: Vec<i64> = db
        .query(
            "SELECT id FROM docs ORDER BY VEC_DISTANCE_L2(emb, $1) LIMIT 3",
            ("[27.0, 0.0, 0.0]",),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(found, vec![7, 6, 8]);
}

#[test]
fn a_correlated_query_vector_is_answered_row_by_row() {
    let db = docs("nearest_subquery_correlated");
    // Each row at distance 0 from itself
    let found: Vec<f64> = db
        .query(
            "SELECT VEC_DISTANCE_L2(emb, (SELECT emb FROM docs x WHERE x.id = d.id)) AS dist \
             FROM docs d ORDER BY dist LIMIT 3",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<f64>(0).unwrap())
        .collect();
    assert_eq!(found, vec![0.0, 0.0, 0.0]);
}

#[test]
fn a_query_vector_subquery_with_no_row_leaves_the_distance_null() {
    let db = docs("nearest_subquery_none");
    let found: Vec<Option<f64>> = db
        .query(
            "SELECT VEC_DISTANCE_L2(emb, (SELECT emb FROM docs WHERE id = 0)) AS d \
             FROM docs ORDER BY d LIMIT 3",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<Option<f64>>(0).unwrap())
        .collect();
    assert_eq!(found, vec![None, None, None]);
}

#[test]
fn a_query_vector_subquery_with_more_than_one_row_is_an_error() {
    let db = docs("nearest_subquery_many");
    let result = db
        .query(
            "SELECT id FROM docs ORDER BY \
             VEC_DISTANCE_L2(emb, (SELECT emb FROM docs WHERE id < 3)) LIMIT 3",
            (),
        )
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
    assert!(result.is_err(), "a scalar subquery of two rows: {result:?}");
}
