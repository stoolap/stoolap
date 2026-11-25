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

/// 500 rows on a curve, each at its own distance from any other
fn docs(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO docs SELECT value, '[' || CAST(SIN(value) AS TEXT) || ',' || \
         CAST(COS(value) AS TEXT) || ',' || CAST(value / 500.0 AS TEXT) || ']' \
         FROM GENERATE_SERIES(1, 500)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_emb ON docs(emb) USING HNSW", ())
        .unwrap();
    db
}

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

/// Row 7's vector, as a literal
fn literal(db: &Database) -> String {
    db.query_one("SELECT VEC_TO_TEXT(emb) FROM docs WHERE id = 7", ())
        .unwrap()
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
fn a_scalar_subquery_query_vector_finds_what_the_literal_finds() {
    let db = docs("nearest_subquery_rows");
    let vector = literal(&db);
    let expected = ids(
        &db,
        &format!("SELECT id FROM docs ORDER BY VEC_DISTANCE_L2(emb, '{vector}') LIMIT 5"),
    );
    assert_eq!(expected[0], 7, "the row itself is nearest");
    assert_eq!(ids(&db, BY_ALIAS), expected);
    assert_eq!(ids(&db, BY_KEY), expected);
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
    let vector = literal(&db);
    let found: Vec<i64> = db
        .query(
            "SELECT id FROM docs ORDER BY VEC_DISTANCE_L2(emb, $1) LIMIT 3",
            (vector.as_str(),),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(found[0], 7);
    assert_eq!(found.len(), 3);
}

#[test]
fn a_correlated_query_vector_is_answered_row_by_row() {
    let db = docs("nearest_subquery_correlated");
    // Each row's distance from itself: every one is 0, the order is the key's
    let rows = ids(
        &db,
        "SELECT id FROM docs d ORDER BY \
         VEC_DISTANCE_L2(emb, (SELECT emb FROM docs x WHERE x.id = d.id)) LIMIT 3",
    );
    assert_eq!(rows.len(), 3);
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
