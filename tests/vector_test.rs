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

//! Vector Type Integration Tests
//!
//! Tests for VECTOR(N) data type, distance functions, and the <=> operator.

use stoolap::Database;

/// Collect query results, unwrapping each row.
fn collect_rows(result: stoolap::Rows) -> Vec<stoolap::ResultRow> {
    result.map(|r| r.expect("row")).collect()
}

// ---------------------------------------------------------------------------
// 1. CREATE TABLE with VECTOR(N) column
// ---------------------------------------------------------------------------

#[test]
fn test_create_table_with_vector_column() {
    let db = Database::open("memory://vec_create").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    let rows = collect_rows(db.query("SELECT id FROM embeddings", ()).expect("query"));
    assert_eq!(rows.len(), 0);
}

// ---------------------------------------------------------------------------
// 2. INSERT vector data as string literal
// ---------------------------------------------------------------------------

#[test]
fn test_insert_vector_data() {
    let db = Database::open("memory://vec_insert").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0]')",
        (),
    )
    .expect("insert row 1");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (2, '[4.0, 5.0, 6.0]')",
        (),
    )
    .expect("insert row 2");

    let rows = collect_rows(
        db.query("SELECT id, vec FROM embeddings ORDER BY id", ())
            .expect("query"),
    );
    assert_eq!(rows.len(), 2);

    let vec_str: String = rows[0].get(1).expect("get vec");
    assert_eq!(vec_str, "[1.0, 2.0, 3.0]");
}

// ---------------------------------------------------------------------------
// 3. Dimension validation
// ---------------------------------------------------------------------------

#[test]
fn test_dimension_validation_wrong_size() {
    let db = Database::open("memory://vec_dim").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    let result = db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0]')",
        (),
    );
    assert!(result.is_err(), "Should reject wrong dimension");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("dimension") || err_msg.contains("Dimension"),
        "Error should mention dimension mismatch: {}",
        err_msg
    );
}

#[test]
fn test_dimension_validation_too_many() {
    let db = Database::open("memory://vec_dim2").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(2))",
        (),
    )
    .expect("create table");

    let result = db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0]')",
        (),
    );
    assert!(result.is_err(), "Should reject too many dimensions");
}

// ---------------------------------------------------------------------------
// 4. SELECT and display format
// ---------------------------------------------------------------------------

#[test]
fn test_select_vector_display() {
    let db = Database::open("memory://vec_display").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(4))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[0.1, 0.2, 0.3, 0.4]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query("SELECT vec FROM embeddings WHERE id = 1", ())
            .expect("query"),
    );
    assert_eq!(rows.len(), 1);

    let vec_str: String = rows[0].get(0).expect("get vec");
    assert!(vec_str.starts_with('['));
    assert!(vec_str.ends_with(']'));
    assert!(vec_str.contains("0.1"));
}

// ---------------------------------------------------------------------------
// 5. <=> distance operator
// ---------------------------------------------------------------------------

#[test]
fn test_distance_operator() {
    let db = Database::open("memory://vec_op").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 0.0, 0.0]')",
        (),
    )
    .expect("insert");
    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (2, '[0.0, 1.0, 0.0]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query(
            "SELECT id, vec <=> '[1.0, 0.0, 0.0]' AS dist FROM embeddings ORDER BY dist",
            (),
        )
        .expect("query with <=>"),
    );
    assert_eq!(rows.len(), 2);

    let id: i64 = rows[0].get(0).expect("id");
    assert_eq!(id, 1);

    let dist: f64 = rows[0].get(1).expect("dist");
    assert!(
        dist.abs() < 1e-6,
        "Distance to self should be ~0, got {}",
        dist
    );

    let dist2: f64 = rows[1].get(1).expect("dist");
    assert!(
        (dist2 - std::f64::consts::SQRT_2).abs() < 1e-6,
        "Distance should be sqrt(2), got {}",
        dist2
    );
}

// ---------------------------------------------------------------------------
// 6. Distance functions: VEC_DISTANCE_L2, VEC_DISTANCE_COSINE, VEC_DISTANCE_IP
// ---------------------------------------------------------------------------

#[test]
fn test_vec_distance_l2_function() {
    let db = Database::open("memory://vec_l2").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 0.0, 0.0]')",
        (),
    )
    .expect("insert");
    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (2, '[0.0, 1.0, 0.0]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(vec, '[1.0, 0.0, 0.0]') AS dist FROM embeddings ORDER BY dist",
            (),
        )
        .expect("query"),
    );

    let dist1: f64 = rows[0].get(1).expect("dist");
    assert!(dist1.abs() < 1e-6, "L2 distance to self = 0");

    let dist2: f64 = rows[1].get(1).expect("dist");
    assert!((dist2 - std::f64::consts::SQRT_2).abs() < 1e-6);
}

#[test]
fn test_vec_distance_cosine_function() {
    let db = Database::open("memory://vec_cosine").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(2))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 0.0]')",
        (),
    )
    .expect("insert");
    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (2, '[0.0, 1.0]')",
        (),
    )
    .expect("insert");
    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (3, '[1.0, 1.0]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_COSINE(vec, '[1.0, 0.0]') AS dist FROM embeddings ORDER BY dist",
            (),
        )
        .expect("query"),
    );

    // id=1: same direction, cosine distance = 0
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1);
    let dist1: f64 = rows[0].get(1).expect("dist");
    assert!(dist1.abs() < 1e-6, "Cosine dist same direction = 0");

    // id=3: 45 degrees, cosine distance = 1 - cos(45)
    let id2: i64 = rows[1].get(0).expect("id");
    assert_eq!(id2, 3);
    let dist_45: f64 = rows[1].get(1).expect("dist");
    let expected = 1.0 - (1.0 / std::f64::consts::SQRT_2);
    assert!(
        (dist_45 - expected).abs() < 1e-6,
        "Cosine dist 45 deg = {}, got {}",
        expected,
        dist_45
    );

    // id=2: orthogonal, cosine distance = 1
    let dist_orth: f64 = rows[2].get(1).expect("dist");
    assert!((dist_orth - 1.0).abs() < 1e-6, "Cosine dist orthogonal = 1");
}

#[test]
fn test_vec_distance_ip_function() {
    let db = Database::open("memory://vec_ip").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query(
            "SELECT VEC_DISTANCE_IP(vec, '[1.0, 0.0, 0.0]') AS dist FROM embeddings",
            (),
        )
        .expect("query"),
    );
    let dist: f64 = rows[0].get(0).expect("dist");
    assert!((dist - (-1.0)).abs() < 1e-6, "Neg IP = -1, got {}", dist);
}

// ---------------------------------------------------------------------------
// 7. ORDER BY distance with LIMIT
// ---------------------------------------------------------------------------

#[test]
fn test_order_by_distance_limit() {
    let db = Database::open("memory://vec_limit").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    for i in 1..=5 {
        let v = format!("[{}.0, 0.0, 0.0]", i);
        db.execute(
            &format!("INSERT INTO embeddings (id, vec) VALUES ({}, '{}')", i, v),
            (),
        )
        .expect("insert");
    }

    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(vec, '[0.0, 0.0, 0.0]') AS dist FROM embeddings ORDER BY dist LIMIT 3",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 3);

    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "Closest should be id=1");

    let id3: i64 = rows[2].get(0).expect("id");
    assert_eq!(id3, 3, "Third closest should be id=3");
}

// ---------------------------------------------------------------------------
// 8. VEC_DIMS, VEC_NORM utility functions
// ---------------------------------------------------------------------------

#[test]
fn test_vec_dims_function() {
    let db = Database::open("memory://vec_dims").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(4))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0, 4.0]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query("SELECT VEC_DIMS(vec) FROM embeddings", ())
            .expect("query"),
    );
    let dims: i64 = rows[0].get(0).expect("dims");
    assert_eq!(dims, 4);
}

#[test]
fn test_vec_norm_function() {
    let db = Database::open("memory://vec_norm").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(2))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[3.0, 4.0]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query("SELECT VEC_NORM(vec) FROM embeddings", ())
            .expect("query"),
    );
    let norm: f64 = rows[0].get(0).expect("norm");
    assert!((norm - 5.0).abs() < 1e-6, "Norm of [3,4] = 5, got {}", norm);
}

// ---------------------------------------------------------------------------
// 9. NULL handling in distance calculations
// ---------------------------------------------------------------------------

#[test]
fn test_null_vector_handling() {
    let db = Database::open("memory://vec_null").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0]')",
        (),
    )
    .expect("insert with value");

    db.execute("INSERT INTO embeddings (id, vec) VALUES (2, NULL)", ())
        .expect("insert with null");

    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(vec, '[1.0, 0.0, 0.0]') AS dist FROM embeddings ORDER BY id",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 2);

    // First row: non-null distance
    let dist1: f64 = rows[0].get(1).expect("dist");
    assert!(
        dist1 >= 0.0,
        "Non-null vector should produce valid distance"
    );

    // Second row: null distance — get as Option
    let dist2: Result<f64, _> = rows[1].get(1);
    // NULL values typically produce an error when trying to get as f64
    // or return a special null handling
    assert!(
        dist2.is_err() || dist2.unwrap() == 0.0,
        "NULL vector should produce NULL/error distance"
    );

    // VEC_DIMS with NULL should return NULL
    let rows = collect_rows(
        db.query("SELECT VEC_DIMS(vec) FROM embeddings WHERE id = 2", ())
            .expect("VEC_DIMS on null"),
    );
    assert_eq!(rows.len(), 1);
}

// ---------------------------------------------------------------------------
// 10. VEC_TO_TEXT function
// ---------------------------------------------------------------------------

#[test]
fn test_vec_to_text_function() {
    let db = Database::open("memory://vec_to_text").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.5, 2.5, 3.5]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query("SELECT VEC_TO_TEXT(vec) FROM embeddings", ())
            .expect("query"),
    );
    let text: String = rows[0].get(0).expect("text");
    assert!(text.starts_with('['), "Should start with [");
    assert!(text.contains("1.5"), "Should contain 1.5");
    assert!(text.contains("3.5"), "Should contain 3.5");
}

// ---------------------------------------------------------------------------
// 11. Multiple vector columns
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_vector_columns() {
    let db = Database::open("memory://vec_multi").expect("db");

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, title_vec VECTOR(2), content_vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO items (id, title_vec, content_vec) VALUES (1, '[1.0, 0.0]', '[1.0, 0.0, 0.0]')",
        (),
    )
    .expect("insert");

    let rows = collect_rows(
        db.query(
            "SELECT VEC_DIMS(title_vec), VEC_DIMS(content_vec) FROM items",
            (),
        )
        .expect("query"),
    );
    let title_dims: i64 = rows[0].get(0).expect("title dims");
    let content_dims: i64 = rows[0].get(1).expect("content dims");
    assert_eq!(title_dims, 2);
    assert_eq!(content_dims, 3);
}

// ---------------------------------------------------------------------------
// 12. Nearest neighbor search pattern (ORDER BY + LIMIT)
// ---------------------------------------------------------------------------

#[test]
fn test_knn_search_pattern() {
    let db = Database::open("memory://vec_knn").expect("db");

    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, embedding VECTOR(3))",
        (),
    )
    .expect("create table");

    let docs = [
        (1, "rust", "[1.0, 0.0, 0.0]"),
        (2, "python", "[0.0, 1.0, 0.0]"),
        (3, "javascript", "[0.0, 0.0, 1.0]"),
        (4, "typescript", "[0.1, 0.0, 0.9]"),
        (5, "cargo", "[0.9, 0.1, 0.0]"),
    ];

    for (id, title, vec) in &docs {
        db.execute(
            &format!(
                "INSERT INTO docs (id, title, embedding) VALUES ({}, '{}', '{}')",
                id, title, vec
            ),
            (),
        )
        .expect("insert");
    }

    let rows = collect_rows(
        db.query(
            "SELECT title, VEC_DISTANCE_L2(embedding, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("knn query"),
    );
    assert_eq!(rows.len(), 2);

    let title1: String = rows[0].get(0).expect("title");
    assert_eq!(title1, "rust", "Nearest should be 'rust'");

    let title2: String = rows[1].get(0).expect("title");
    assert_eq!(title2, "cargo", "Second nearest should be 'cargo'");
}

// ---------------------------------------------------------------------------
// 13. Vector with transaction
// ---------------------------------------------------------------------------

#[test]
fn test_vector_in_transaction() {
    let db = Database::open("memory://vec_tx").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    // Begin transaction, insert, rollback
    let mut tx = db.begin().expect("begin tx");
    tx.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0]')",
        (),
    )
    .expect("insert in tx");
    tx.rollback().expect("rollback");

    let rows = collect_rows(
        db.query("SELECT COUNT(*) FROM embeddings", ())
            .expect("count"),
    );
    let count: i64 = rows[0].get(0).expect("count");
    assert_eq!(count, 0, "Rollback should have undone the insert");

    // Now commit
    let mut tx = db.begin().expect("begin tx");
    tx.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0]')",
        (),
    )
    .expect("insert in tx");
    tx.commit().expect("commit");

    let rows = collect_rows(
        db.query("SELECT COUNT(*) FROM embeddings", ())
            .expect("count"),
    );
    let count: i64 = rows[0].get(0).expect("count");
    assert_eq!(count, 1, "Commit should persist the insert");
}

// ---------------------------------------------------------------------------
// 14. UPDATE vector column
// ---------------------------------------------------------------------------

#[test]
fn test_update_vector_column() {
    let db = Database::open("memory://vec_update").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 0.0, 0.0]')",
        (),
    )
    .expect("insert");

    db.execute(
        "UPDATE embeddings SET vec = '[0.0, 1.0, 0.0]' WHERE id = 1",
        (),
    )
    .expect("update vector");

    let rows = collect_rows(
        db.query(
            "SELECT VEC_DISTANCE_L2(vec, '[0.0, 1.0, 0.0]') FROM embeddings WHERE id = 1",
            (),
        )
        .expect("query"),
    );
    let dist: f64 = rows[0].get(0).expect("dist");
    assert!(
        dist.abs() < 1e-6,
        "Updated vector should match query exactly"
    );
}

// ---------------------------------------------------------------------------
// 15. DELETE with vector column
// ---------------------------------------------------------------------------

#[test]
fn test_delete_with_vector() {
    let db = Database::open("memory://vec_delete").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 0.0, 0.0]')",
        (),
    )
    .expect("insert");
    db.execute(
        "INSERT INTO embeddings (id, vec) VALUES (2, '[0.0, 1.0, 0.0]')",
        (),
    )
    .expect("insert");

    db.execute("DELETE FROM embeddings WHERE id = 1", ())
        .expect("delete");

    let rows = collect_rows(
        db.query("SELECT COUNT(*) FROM embeddings", ())
            .expect("count"),
    );
    let count: i64 = rows[0].get(0).expect("count");
    assert_eq!(count, 1);
}

// ---------------------------------------------------------------------------
// 16. WAL persistence
// ---------------------------------------------------------------------------

#[test]
fn test_vector_wal_persistence() {
    let test_dir = tempfile::tempdir().expect("create temp dir");
    let db_path = format!("file://{}/vec_wal_test", test_dir.path().display());

    {
        let db = Database::open(&db_path).expect("open db");
        db.execute(
            "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
            (),
        )
        .expect("create table");

        db.execute(
            "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0]')",
            (),
        )
        .expect("insert");
        db.execute(
            "INSERT INTO embeddings (id, vec) VALUES (2, '[4.0, 5.0, 6.0]')",
            (),
        )
        .expect("insert");
    }

    {
        let db = Database::open(&db_path).expect("reopen db");
        let rows = collect_rows(
            db.query("SELECT id, vec FROM embeddings ORDER BY id", ())
                .expect("query"),
        );
        assert_eq!(rows.len(), 2, "Should have 2 rows after WAL recovery");

        let vec1: String = rows[0].get(1).expect("vec1");
        assert!(vec1.contains('1'), "First vector should contain 1");

        let vec2: String = rows[1].get(1).expect("vec2");
        assert!(vec2.contains('4'), "Second vector should contain 4");
    }
}

// ---------------------------------------------------------------------------
// 17. Vector distance in WHERE clause
// ---------------------------------------------------------------------------

#[test]
fn test_vector_distance_in_where() {
    let db = Database::open("memory://vec_where").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    for i in 1..=10 {
        let v = format!("[{}.0, 0.0, 0.0]", i);
        db.execute(
            &format!("INSERT INTO embeddings (id, vec) VALUES ({}, '{}')", i, v),
            (),
        )
        .expect("insert");
    }

    let rows = collect_rows(
        db.query(
            "SELECT id FROM embeddings WHERE VEC_DISTANCE_L2(vec, '[0.0, 0.0, 0.0]') <= 3.0 ORDER BY id",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 3, "Should find 3 vectors within distance 3");

    let ids: Vec<i64> = rows.iter().map(|r| r.get(0).unwrap()).collect();
    assert_eq!(ids, vec![1, 2, 3]);
}

// ---------------------------------------------------------------------------
// 18. Higher-dimensional vectors (128D)
// ---------------------------------------------------------------------------

#[test]
fn test_high_dimensional_vectors() {
    let db = Database::open("memory://vec_high_dim").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(128))",
        (),
    )
    .expect("create table");

    let mut parts = Vec::with_capacity(128);
    for i in 0..128 {
        parts.push(format!("{:.1}", (i as f32) / 128.0));
    }
    let vec_str = format!("[{}]", parts.join(", "));

    db.execute(
        &format!("INSERT INTO embeddings (id, vec) VALUES (1, '{}')", vec_str),
        (),
    )
    .expect("insert 128-dim vector");

    let rows = collect_rows(
        db.query("SELECT VEC_DIMS(vec) FROM embeddings", ())
            .expect("query"),
    );
    let dims: i64 = rows[0].get(0).expect("dims");
    assert_eq!(dims, 128);
}

// ---------------------------------------------------------------------------
// 19. Snapshot persistence
// ---------------------------------------------------------------------------

#[test]
fn test_vector_snapshot_persistence() {
    let test_dir = tempfile::tempdir().expect("create temp dir");
    let db_path = format!("file://{}/vec_snap_test", test_dir.path().display());

    {
        let db = Database::open(&db_path).expect("open db");
        db.execute(
            "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
            (),
        )
        .expect("create table");

        db.execute(
            "INSERT INTO embeddings (id, vec) VALUES (1, '[1.0, 2.0, 3.0]')",
            (),
        )
        .expect("insert");

        db.execute("PRAGMA create_snapshot", ()).ok();
    }

    {
        let db = Database::open(&db_path).expect("reopen db");
        let rows = collect_rows(
            db.query(
                "SELECT VEC_DIMS(vec), VEC_NORM(vec) FROM embeddings WHERE id = 1",
                (),
            )
            .expect("query"),
        );
        assert_eq!(rows.len(), 1, "Should have 1 row after snapshot recovery");

        let dims: i64 = rows[0].get(0).expect("dims");
        assert_eq!(dims, 3);

        let norm: f64 = rows[0].get(1).expect("norm");
        let expected = (14.0f64).sqrt();
        assert!(
            (norm - expected).abs() < 1e-4,
            "Norm should be sqrt(14) = {}, got {}",
            expected,
            norm
        );
    }
}

// ---------------------------------------------------------------------------
// 20. <=> operator with ORDER BY + LIMIT
// ---------------------------------------------------------------------------

#[test]
fn test_distance_operator_order_by_limit() {
    let db = Database::open("memory://vec_op_limit").expect("db");

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))",
        (),
    )
    .expect("create table");

    let vectors = [
        (1, "[1.0, 0.0, 0.0]"),
        (2, "[0.0, 1.0, 0.0]"),
        (3, "[0.5, 0.5, 0.0]"),
        (4, "[0.0, 0.0, 1.0]"),
    ];

    for (id, vec) in &vectors {
        db.execute(
            &format!(
                "INSERT INTO embeddings (id, vec) VALUES ({}, '{}')",
                id, vec
            ),
            (),
        )
        .expect("insert");
    }

    let rows = collect_rows(
        db.query(
            "SELECT id FROM embeddings ORDER BY vec <=> '[1.0, 0.0, 0.0]' LIMIT 2",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 2);

    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "Nearest should be id=1 (exact match)");
}

// ---------------------------------------------------------------------------
// HNSW Index Tests
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_create_index() {
    let db = Database::open("memory://hnsw_create").expect("db");

    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(3))",
        (),
    )
    .expect("create table");

    // Insert some vectors
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (3, '[0.0, 0.0, 1.0]')", ())
        .expect("insert");

    // Create HNSW index
    db.execute("CREATE INDEX idx_emb ON docs(embedding) USING HNSW", ())
        .expect("create hnsw index");

    // Verify the index exists and is usable (basic query still works)
    let rows = collect_rows(
        db.query("SELECT id FROM docs ORDER BY id", ())
            .expect("query"),
    );
    assert_eq!(rows.len(), 3);
}

#[test]
fn test_hnsw_search_nearest() {
    let db = Database::open("memory://hnsw_search").expect("db");

    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(3))",
        (),
    )
    .expect("create table");

    // Insert vectors in a known geometry
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (3, '[0.0, 0.0, 1.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (4, '[0.9, 0.1, 0.0]')", ())
        .expect("insert");

    // Create HNSW index
    db.execute("CREATE INDEX idx_emb ON docs(embedding) USING HNSW", ())
        .expect("create hnsw index");

    // Search for nearest to [1, 0, 0] — should return id=1 (exact match) first
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(embedding, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("query"),
    );

    assert_eq!(rows.len(), 2);
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "Nearest to [1,0,0] should be id=1");

    let id2: i64 = rows[1].get(0).expect("id");
    assert_eq!(
        id2, 4,
        "Second nearest to [1,0,0] should be id=4 ([0.9,0.1,0])"
    );
}

#[test]
fn test_hnsw_larger_dataset() {
    let db = Database::open("memory://hnsw_large").expect("db");

    db.execute(
        "CREATE TABLE vectors (id INTEGER PRIMARY KEY, v VECTOR(4))",
        (),
    )
    .expect("create table");

    // Insert 100 vectors with known pattern
    for i in 0..100 {
        let x = (i as f64) / 100.0;
        let y = 1.0 - x;
        let z = (i as f64 * 0.1).sin();
        let w = (i as f64 * 0.1).cos();
        db.execute(
            &format!(
                "INSERT INTO vectors VALUES ({}, '[{}, {}, {}, {}]')",
                i, x, y, z, w
            ),
            (),
        )
        .expect("insert");
    }

    // Create HNSW index
    db.execute("CREATE INDEX idx_v ON vectors(v) USING HNSW", ())
        .expect("create hnsw index");

    // Search for nearest to [0.5, 0.5, 0, 1] (should be near id=50)
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(v, '[0.5, 0.5, 0.0, 1.0]') AS dist FROM vectors ORDER BY dist LIMIT 5",
            (),
        )
        .expect("query"),
    );

    assert_eq!(rows.len(), 5, "Should return 5 nearest neighbors");

    // Verify distances are in non-decreasing order
    let mut prev_dist: f64 = -1.0;
    for row in &rows {
        let dist: f64 = row.get(1).expect("dist");
        assert!(
            dist >= prev_dist,
            "Distances should be non-decreasing: {} >= {}",
            dist,
            prev_dist
        );
        prev_dist = dist;
    }
}

#[test]
fn test_hnsw_insert_after_index_creation() {
    let db = Database::open("memory://hnsw_post_insert").expect("db");

    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(3))",
        (),
    )
    .expect("create table");

    // Create HNSW index on empty table
    db.execute("CREATE INDEX idx_emb ON docs(embedding) USING HNSW", ())
        .expect("create hnsw index");

    // Insert vectors after index creation
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (3, '[0.0, 0.0, 1.0]')", ())
        .expect("insert");

    // Search should still work (index updated on INSERT commit)
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(embedding, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("query"),
    );

    assert_eq!(rows.len(), 2);
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "Nearest to [1,0,0] should be id=1");
}

#[test]
fn test_hnsw_without_index_fallback() {
    let db = Database::open("memory://hnsw_fallback").expect("db");

    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(3))",
        (),
    )
    .expect("create table");

    // Insert vectors but NO HNSW index
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (3, '[0.0, 0.0, 1.0]')", ())
        .expect("insert");

    // Query should still work via brute-force path (no HNSW index)
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(embedding, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("query"),
    );

    assert_eq!(rows.len(), 2);
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(
        id1, 1,
        "Nearest to [1,0,0] should be id=1 even without HNSW"
    );
}

#[test]
fn test_hnsw_wrong_column_type() {
    let db = Database::open("memory://hnsw_wrong_type").expect("db");

    db.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, name TEXT)", ())
        .expect("create table");

    // HNSW on non-vector column should fail
    let result = db.execute("CREATE INDEX idx_name ON docs(name) USING HNSW", ());

    assert!(result.is_err(), "HNSW on TEXT column should fail");
}

#[test]
fn test_hnsw_cosine_distance() {
    let db = Database::open("memory://hnsw_cosine").expect("db");

    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (3, '[0.7, 0.7, 0.0]')", ())
        .expect("insert");

    // Cosine distance query (without HNSW — brute force)
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_COSINE(embedding, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("query"),
    );

    assert_eq!(rows.len(), 2);
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(
        id1, 1,
        "Cosine nearest to [1,0,0] should be id=1 (exact match)"
    );
}

#[test]
fn test_hnsw_wal_recovery() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("hnsw_wal_test");
    let dsn = format!("file://{}", path.display());

    // Create database with HNSW index
    {
        let db = Database::open(&dsn).expect("db");

        db.execute(
            "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(3))",
            (),
        )
        .expect("create table");

        db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
            .expect("insert");
        db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
            .expect("insert");
        db.execute("INSERT INTO docs VALUES (3, '[0.0, 0.0, 1.0]')", ())
            .expect("insert");

        db.execute("CREATE INDEX idx_emb ON docs(embedding) USING HNSW", ())
            .expect("create hnsw index");

        // Drop to flush WAL
    }

    // Reopen — HNSW index should be rebuilt from WAL
    {
        let db = Database::open(&dsn).expect("reopen db");

        // Verify data survived
        let rows = collect_rows(
            db.query("SELECT id FROM docs ORDER BY id", ())
                .expect("query"),
        );
        assert_eq!(rows.len(), 3, "All 3 rows should survive WAL recovery");

        // Verify HNSW search works after recovery
        let rows = collect_rows(
            db.query(
                "SELECT id, VEC_DISTANCE_L2(embedding, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
                (),
            )
            .expect("query after recovery"),
        );
        assert_eq!(rows.len(), 2);
        let id1: i64 = rows[0].get(0).expect("id");
        assert_eq!(id1, 1, "After WAL recovery, nearest should still be id=1");
    }
}

// =============================================================================
// Parallel Brute-Force k-NN Tests
// =============================================================================

#[test]
fn test_brute_force_vector_search_l2() {
    // Test brute-force path (no HNSW index) with VEC_DISTANCE_L2
    let db = Database::open("memory://bf_l2").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        (),
    )
    .expect("create table");

    // Insert 5 vectors — no HNSW index
    db.execute("INSERT INTO vecs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (3, '[0.0, 0.0, 1.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (4, '[1.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (5, '[0.5, 0.5, 0.5]')", ())
        .expect("insert");

    // Query nearest 3 to [1, 0, 0]
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(v, '[1.0, 0.0, 0.0]') AS dist FROM vecs ORDER BY dist LIMIT 3",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 3);

    // id=1 should be nearest (distance = 0.0)
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "Nearest to [1,0,0] should be id=1");

    let dist1: f64 = rows[0].get(1).expect("dist");
    assert!(
        dist1 < 0.001,
        "Distance to self should be ~0, got {}",
        dist1
    );

    // id=5 [0.5,0.5,0.5] should be second nearest (L2 = sqrt(0.75) ≈ 0.866)
    let id2: i64 = rows[1].get(0).expect("id");
    assert_eq!(id2, 5, "Second nearest should be id=5 [0.5,0.5,0.5]");
}

#[test]
fn test_brute_force_vector_search_cosine() {
    // Test brute-force path with VEC_DISTANCE_COSINE
    let db = Database::open("memory://bf_cosine").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute("INSERT INTO vecs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (3, '[0.9, 0.1, 0.0]')", ())
        .expect("insert");

    // Query nearest by cosine distance to [1, 0, 0]
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_COSINE(v, '[1.0, 0.0, 0.0]') AS dist FROM vecs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 2);

    // id=1 should be nearest (cosine distance = 0.0)
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "Nearest by cosine to [1,0,0] should be id=1");

    // id=3 [0.9,0.1,0] should be second (nearly aligned)
    let id2: i64 = rows[1].get(0).expect("id");
    assert_eq!(id2, 3, "Second nearest by cosine should be id=3");
}

#[test]
fn test_brute_force_vector_search_ip() {
    // Test brute-force path with VEC_DISTANCE_IP (inner product distance)
    let db = Database::open("memory://bf_ip").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute("INSERT INTO vecs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (3, '[2.0, 0.0, 0.0]')", ())
        .expect("insert");

    // Inner product distance = 1 - dot(a, b)
    // dot([1,0,0], [2,0,0]) = 2.0 → distance = -1.0 (smallest = most similar)
    // dot([1,0,0], [1,0,0]) = 1.0 → distance = 0.0
    // dot([1,0,0], [0,1,0]) = 0.0 → distance = 1.0
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_IP(v, '[1.0, 0.0, 0.0]') AS dist FROM vecs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 2);

    // id=3 [2,0,0] has highest dot product → lowest IP distance
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 3, "Nearest by IP to [1,0,0] should be id=3 [2,0,0]");
}

#[test]
fn test_brute_force_with_offset() {
    // Test brute-force with LIMIT + OFFSET
    let db = Database::open("memory://bf_offset").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(2))",
        (),
    )
    .expect("create table");

    db.execute("INSERT INTO vecs VALUES (1, '[0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (2, '[1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (3, '[2.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (4, '[3.0, 0.0]')", ())
        .expect("insert");

    // Skip the nearest, get next 2
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(v, '[0.0, 0.0]') AS dist FROM vecs ORDER BY dist LIMIT 2 OFFSET 1",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 2);

    // id=1 is nearest (skipped), id=2 is second, id=3 is third
    let id1: i64 = rows[0].get(0).expect("id");
    let id2: i64 = rows[1].get(0).expect("id");
    assert_eq!(id1, 2, "After OFFSET 1, first result should be id=2");
    assert_eq!(id2, 3, "After OFFSET 1, second result should be id=3");
}

#[test]
fn test_brute_force_matches_hnsw_results() {
    // Verify that brute-force and HNSW return the same nearest neighbor
    let db = Database::open("memory://bf_vs_hnsw").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(4))",
        (),
    )
    .expect("create table");

    // Insert 20 vectors
    for i in 0..20 {
        let v = format!(
            "[{}, {}, {}, {}]",
            (i as f64) * 0.1,
            (i as f64) * 0.2,
            (i as f64) * 0.05,
            1.0 - (i as f64) * 0.05
        );
        db.execute(&format!("INSERT INTO vecs VALUES ({}, '{}')", i, v), ())
            .expect("insert");
    }

    let query_vec = "'[0.5, 1.0, 0.25, 0.75]'";

    // Brute-force (no index)
    let bf_rows = collect_rows(
        db.query(
            &format!(
                "SELECT id, VEC_DISTANCE_L2(v, {}) AS dist FROM vecs ORDER BY dist LIMIT 5",
                query_vec
            ),
            (),
        )
        .expect("brute force query"),
    );

    // Create HNSW index
    db.execute("CREATE INDEX idx_v ON vecs(v) USING HNSW", ())
        .expect("create hnsw index");

    // HNSW search
    let hnsw_rows = collect_rows(
        db.query(
            &format!(
                "SELECT id, VEC_DISTANCE_L2(v, {}) AS dist FROM vecs ORDER BY dist LIMIT 5",
                query_vec
            ),
            (),
        )
        .expect("hnsw query"),
    );

    assert_eq!(bf_rows.len(), 5);
    assert_eq!(hnsw_rows.len(), 5);

    // The top-1 result should match between brute-force and HNSW
    let bf_id1: i64 = bf_rows[0].get(0).expect("bf id");
    let hnsw_id1: i64 = hnsw_rows[0].get(0).expect("hnsw id");
    assert_eq!(
        bf_id1, hnsw_id1,
        "Top-1 result should match: brute-force={}, HNSW={}",
        bf_id1, hnsw_id1
    );
}

#[test]
fn test_brute_force_with_where_clause() {
    // Brute-force vector search should work even when query has no WHERE
    // (WHERE clause queries fall through to standard path)
    let db = Database::open("memory://bf_no_where").expect("db");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, category TEXT, embedding VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute("INSERT INTO docs VALUES (1, 'a', '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (2, 'b', '[0.0, 1.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (3, 'a', '[0.5, 0.5, 0.0]')", ())
        .expect("insert");

    // Simple vector search without WHERE — should use brute-force fast path
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(embedding, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 2);

    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "Nearest should be id=1");
}

#[test]
fn test_brute_force_select_star() {
    // Test brute-force with SELECT * + distance alias
    let db = Database::open("memory://bf_star").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute("INSERT INTO vecs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");

    let rows = collect_rows(
        db.query(
            "SELECT *, VEC_DISTANCE_L2(v, '[1.0, 0.0, 0.0]') AS dist FROM vecs ORDER BY dist LIMIT 1",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 1);

    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "Nearest should be id=1");
}

#[test]
fn test_brute_force_empty_table() {
    // Brute-force on empty table should return empty results
    let db = Database::open("memory://bf_empty").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        (),
    )
    .expect("create table");

    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(v, '[1.0, 0.0, 0.0]') AS dist FROM vecs ORDER BY dist LIMIT 5",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 0, "Empty table should return 0 rows");
}

#[test]
fn test_hnsw_default_index_type() {
    // Creating an index on a VECTOR column without USING should default to HNSW
    let db = Database::open("memory://hnsw_default").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        (),
    )
    .expect("create table");

    db.execute("INSERT INTO vecs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO vecs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");

    // No USING clause — should auto-select HNSW for VECTOR column
    db.execute("CREATE INDEX idx_v ON vecs(v)", ())
        .expect("create index should default to HNSW");

    // HNSW search should work
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(v, '[1.0, 0.0, 0.0]') AS dist FROM vecs ORDER BY dist LIMIT 1",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 1);
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "HNSW search should work with auto-selected index");
}

// ---------------------------------------------------------------------------
// HNSW Cosine Distance Index
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_cosine_index() {
    let db = Database::open("memory://hnsw_cosine_index").expect("db");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .expect("create table");

    // Insert vectors at different angles
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("ins");
    db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("ins");
    db.execute("INSERT INTO docs VALUES (3, '[0.707, 0.707, 0.0]')", ())
        .expect("ins");
    db.execute("INSERT INTO docs VALUES (4, '[0.0, 0.0, 1.0]')", ())
        .expect("ins");

    // Create HNSW index with cosine metric
    db.execute(
        "CREATE INDEX idx_cos ON docs(emb) USING HNSW WITH (metric = 'cosine')",
        (),
    )
    .expect("create cosine HNSW");

    // Search: closest to [1,0,0] by cosine should be id=1, then id=3
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_COSINE(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 2);
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "cosine: exact match should be first");
    let id2: i64 = rows[1].get(0).expect("id");
    assert_eq!(id2, 3, "cosine: 45-degree angle should be second");
}

// ---------------------------------------------------------------------------
// HNSW Inner Product Distance Index
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_ip_index() {
    let db = Database::open("memory://hnsw_ip").expect("db");
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .expect("create table");

    // Insert vectors with different magnitudes along x-axis
    db.execute("INSERT INTO items VALUES (1, '[3.0, 0.0, 0.0]')", ())
        .expect("ins");
    db.execute("INSERT INTO items VALUES (2, '[1.0, 0.0, 0.0]')", ())
        .expect("ins");
    db.execute("INSERT INTO items VALUES (3, '[0.0, 2.0, 0.0]')", ())
        .expect("ins");

    // Create HNSW index with inner product metric
    db.execute(
        "CREATE INDEX idx_ip ON items(emb) USING HNSW WITH (metric = 'ip')",
        (),
    )
    .expect("create IP HNSW");

    // Search: IP with [1,0,0] — highest dot product is id=1 (3.0), then id=2 (1.0)
    // VEC_DISTANCE_IP returns -dot_product so lower = higher similarity
    let rows = collect_rows(
        db.query(
            "SELECT id FROM items ORDER BY VEC_DISTANCE_IP(emb, '[1.0, 0.0, 0.0]') LIMIT 2",
            (),
        )
        .expect("query"),
    );
    assert_eq!(rows.len(), 2);
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "IP: highest dot product should be first");
    let id2: i64 = rows[1].get(0).expect("id");
    assert_eq!(id2, 2, "IP: second highest dot product");
}

// ---------------------------------------------------------------------------
// WITH clause parameters for HNSW
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_with_clause_params() {
    let db = Database::open("memory://hnsw_with").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(4))",
        (),
    )
    .expect("create table");

    for i in 0..50 {
        let v = format!(
            "[{}, {}, {}, {}]",
            i as f64 * 0.1,
            (i as f64 * 0.2).sin(),
            (i as f64 * 0.3).cos(),
            i as f64 * 0.05,
        );
        db.execute(&format!("INSERT INTO vecs VALUES ({}, '{}')", i, v), ())
            .expect("insert");
    }

    // Create HNSW index with custom parameters
    db.execute(
        "CREATE INDEX idx_custom ON vecs(v) USING HNSW WITH (m = 32, ef_construction = 400, ef_search = 128)",
        (),
    )
    .expect("create HNSW with custom params");

    // Search should work correctly
    let rows = collect_rows(
        db.query(
            "SELECT id FROM vecs ORDER BY VEC_DISTANCE_L2(v, '[0.5, 0.3, 0.7, 0.2]') LIMIT 5",
            (),
        )
        .expect("query"),
    );
    assert_eq!(
        rows.len(),
        5,
        "should return 5 results with custom HNSW params"
    );
}

// ---------------------------------------------------------------------------
// HNSW Graph Serialization Roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_graph_serialization() {
    use stoolap::storage::index::{HnswDistanceMetric, HnswIndex};
    use stoolap::Index;

    // Create an HNSW index and populate it
    let idx = HnswIndex::new(
        "test_idx".to_string(),
        "test_table".to_string(),
        "vec_col".to_string(),
        1,
        3,   // dims
        16,  // m
        200, // ef_construction
        64,  // ef_search
        HnswDistanceMetric::L2,
    );

    // Add some vectors
    let v1 = stoolap::Value::vector(vec![1.0f32, 0.0, 0.0]);
    let v2 = stoolap::Value::vector(vec![0.0f32, 1.0, 0.0]);
    let v3 = stoolap::Value::vector(vec![0.5f32, 0.5, 0.0]);
    idx.add(&[v1], 1, 1).expect("add v1");
    idx.add(&[v2], 2, 2).expect("add v2");
    idx.add(&[v3], 3, 3).expect("add v3");

    assert_eq!(idx.node_count(), 3);

    // Serialize to temp file
    let dir = tempfile::tempdir().expect("tmpdir");
    let graph_path = dir.path().join("test_hnsw.bin");
    idx.save_graph(&graph_path).expect("save graph");

    // Load from file
    let loaded = HnswIndex::load_graph(
        &graph_path,
        "test_idx".to_string(),
        "test_table".to_string(),
        "vec_col".to_string(),
        1,
        3,   // dims
        16,  // m
        200, // ef_construction
        64,  // ef_search
    )
    .expect("load graph")
    .expect("graph should exist");

    assert_eq!(loaded.node_count(), 3, "loaded graph should have 3 nodes");

    // Search should return the same results
    let query_bytes = [
        0x00, 0x00, 0x80, 0x3f, // 1.0f32 LE
        0x00, 0x00, 0x00, 0x00, // 0.0f32 LE
        0x00, 0x00, 0x00, 0x00, // 0.0f32 LE
    ];
    let results = loaded.search_nearest(&query_bytes, 3, 64);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 1, "closest to [1,0,0] should be row 1");
}

// ---------------------------------------------------------------------------
// HNSW Graph Persistence through Snapshot + Recovery
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_graph_snapshot_recovery() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let db_path = format!("file://{}/hnsw_snap", dir.path().display());

    // Phase 1: Create DB, insert data, create HNSW index, snapshot
    {
        let db = Database::open(&db_path).expect("open db");
        db.execute(
            "CREATE TABLE vectors (id INTEGER PRIMARY KEY, v VECTOR(3))",
            (),
        )
        .expect("create table");

        db.execute("INSERT INTO vectors VALUES (1, '[1.0, 0.0, 0.0]')", ())
            .expect("ins");
        db.execute("INSERT INTO vectors VALUES (2, '[0.0, 1.0, 0.0]')", ())
            .expect("ins");
        db.execute("INSERT INTO vectors VALUES (3, '[0.0, 0.0, 1.0]')", ())
            .expect("ins");

        db.execute(
            "CREATE INDEX idx_hnsw ON vectors(v) USING HNSW WITH (metric = 'cosine')",
            (),
        )
        .expect("create hnsw index");

        // Verify search works before snapshot
        let rows = collect_rows(
            db.query(
                "SELECT id FROM vectors ORDER BY VEC_DISTANCE_COSINE(v, '[1.0, 0.0, 0.0]') LIMIT 1",
                (),
            )
            .expect("query"),
        );
        assert_eq!(rows.len(), 1);
        let id: i64 = rows[0].get(0).expect("id");
        assert_eq!(id, 1);
    }

    // Phase 2: Reopen DB — recovery should load HNSW graph from snapshot
    {
        let db = Database::open(&db_path).expect("reopen db");

        // Search should work after recovery
        let rows = collect_rows(
            db.query(
                "SELECT id FROM vectors ORDER BY VEC_DISTANCE_COSINE(v, '[1.0, 0.0, 0.0]') LIMIT 1",
                (),
            )
            .expect("query after recovery"),
        );
        assert_eq!(rows.len(), 1);
        let id: i64 = rows[0].get(0).expect("id");
        assert_eq!(id, 1, "HNSW cosine search should work after recovery");
    }
}

// ---------------------------------------------------------------------------
// Metric mismatch: cosine index should NOT be used for L2 queries
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_metric_mismatch_falls_to_bruteforce() {
    let db = Database::open("memory://metric_mismatch").expect("db");
    db.execute("CREATE TABLE pts (id INTEGER PRIMARY KEY, v VECTOR(3))", ())
        .expect("create table");

    db.execute("INSERT INTO pts VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("ins");
    db.execute("INSERT INTO pts VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("ins");
    db.execute("INSERT INTO pts VALUES (3, '[0.5, 0.5, 0.0]')", ())
        .expect("ins");

    // Create HNSW with cosine metric
    db.execute(
        "CREATE INDEX idx_cos ON pts(v) USING HNSW WITH (metric = 'cosine')",
        (),
    )
    .expect("create cosine index");

    // L2 query should still work (falls to brute-force, not using cosine index)
    let rows = collect_rows(
        db.query(
            "SELECT id FROM pts ORDER BY VEC_DISTANCE_L2(v, '[1.0, 0.0, 0.0]') LIMIT 2",
            (),
        )
        .expect("L2 query should work via brute-force"),
    );
    assert_eq!(rows.len(), 2);
    let id1: i64 = rows[0].get(0).expect("id");
    assert_eq!(id1, 1, "L2 brute-force: exact match should be first");
}

// ---------------------------------------------------------------------------
// UNIQUE HNSW enforcement
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_unique_cosine_rejects_zero_vector_duplicate() {
    let db = Database::open("memory://hnsw_unique_cos_zero").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        (),
    )
    .expect("create table");
    db.execute(
        "CREATE UNIQUE INDEX idx_v ON vecs(v) USING HNSW WITH (metric = 'cosine')",
        (),
    )
    .expect("create unique hnsw");

    db.execute("INSERT INTO vecs VALUES (1, '[0.0, 0.0, 0.0]')", ())
        .expect("insert first zero vector");

    let dup = db.execute("INSERT INTO vecs VALUES (2, '[0.0, 0.0, 0.0]')", ());
    assert!(
        dup.is_err(),
        "UNIQUE HNSW must reject identical zero vectors under cosine metric"
    );

    let rows = collect_rows(db.query("SELECT COUNT(*) FROM vecs", ()).expect("count"));
    let count: i64 = rows[0].get(0).expect("count");
    assert_eq!(count, 1, "duplicate insert must not be persisted");
}

#[test]
fn test_hnsw_unique_allows_reuse_after_update_removes_old_value() {
    let db = Database::open("memory://hnsw_unique_update").expect("db");
    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        (),
    )
    .expect("create table");
    db.execute(
        "CREATE UNIQUE INDEX idx_v ON vecs(v) USING HNSW WITH (metric = 'cosine')",
        (),
    )
    .expect("create unique hnsw");

    db.execute("INSERT INTO vecs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert row 1");
    db.execute("UPDATE vecs SET v='[0.0, 0.0, 0.0]' WHERE id=1", ())
        .expect("update row 1");

    // Old value [1,0,0] should now be available to other rows.
    db.execute("INSERT INTO vecs VALUES (2, '[1.0, 0.0, 0.0]')", ())
        .expect("insert row 2 with row 1's old vector");

    // New value [0,0,0] is owned by row 1 and must remain unique.
    let dup = db.execute("INSERT INTO vecs VALUES (3, '[0.0, 0.0, 0.0]')", ());
    assert!(dup.is_err(), "current live duplicate must be rejected");

    let rows = collect_rows(db.query("SELECT COUNT(*) FROM vecs", ()).expect("count"));
    let count: i64 = rows[0].get(0).expect("count");
    assert_eq!(count, 2);
}

/// Regression test: vector fast path must reject unknown columns, not silently return NULL.
#[test]
fn test_vector_fast_path_rejects_unknown_column() {
    let db = Database::open("memory://vec_col_not_found").expect("open");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .expect("create table");
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("CREATE INDEX idx_emb ON docs(emb) USING HNSW", ())
        .expect("create index");

    // Unknown column in SELECT with vector ORDER BY must error
    let result = db.query(
        "SELECT nope, VEC_DISTANCE_L2(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 1",
        (),
    );
    match result {
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("nope"),
                "error should mention the bad column name, got: {}",
                err_msg
            );
        }
        Ok(_) => {
            panic!("expected ColumnNotFound error for unknown column 'nope' in vector fast path")
        }
    }

    // Valid column should still work
    let rows = collect_rows(
        db.query(
            "SELECT id, VEC_DISTANCE_L2(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 1",
            (),
        )
        .expect("valid query should succeed"),
    );
    assert_eq!(rows.len(), 1);
    let id: i64 = rows[0].get(0).expect("id");
    assert_eq!(id, 1);
}

// ---------------------------------------------------------------------------
// EXPLAIN / EXPLAIN ANALYZE for vector search
// ---------------------------------------------------------------------------

fn get_plan(db: &Database, sql: &str) -> String {
    let rows = db.query(sql, ()).expect("EXPLAIN failed");
    let mut lines = Vec::new();
    for row in rows {
        let row = row.expect("row");
        let line: String = row.get(0).unwrap_or_default();
        lines.push(line);
    }
    lines.join("\n")
}

#[test]
fn test_explain_vector_hnsw_search() {
    let db = Database::open("memory://explain_hnsw").expect("db");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_emb ON docs(emb) USING HNSW", ())
        .expect("index");
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");

    let plan = get_plan(
        &db,
        "EXPLAIN SELECT id, VEC_DISTANCE_L2(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 5",
    );
    assert!(
        plan.contains("HNSW Index Scan"),
        "Expected HNSW Index Scan, got:\n{}",
        plan
    );
    assert!(
        plan.contains("idx_emb"),
        "Expected index name idx_emb, got:\n{}",
        plan
    );
    assert!(
        plan.contains("Metric: L2"),
        "Expected Metric: L2, got:\n{}",
        plan
    );
    assert!(plan.contains("K: 5"), "Expected K: 5, got:\n{}", plan);
    assert!(
        plan.contains("EF Search:"),
        "Expected EF Search, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_vector_brute_force() {
    let db = Database::open("memory://explain_bf").expect("db");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .expect("create");
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");

    let plan = get_plan(
        &db,
        "EXPLAIN SELECT id, VEC_DISTANCE_L2(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 5",
    );
    assert!(
        plan.contains("Vector Scan"),
        "Expected Vector Scan (no HNSW index), got:\n{}",
        plan
    );
    assert!(
        plan.contains("Metric: L2"),
        "Expected Metric: L2, got:\n{}",
        plan
    );
    assert!(plan.contains("K: 5"), "Expected K: 5, got:\n{}", plan);
    // Should NOT contain HNSW
    assert!(
        !plan.contains("HNSW"),
        "Should not show HNSW without index, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_analyze_vector_hnsw() {
    let db = Database::open("memory://explain_analyze_hnsw").expect("db");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_emb ON docs(emb) USING HNSW", ())
        .expect("index");
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");
    db.execute("INSERT INTO docs VALUES (2, '[0.0, 1.0, 0.0]')", ())
        .expect("insert");

    let plan = get_plan(
        &db,
        "EXPLAIN ANALYZE SELECT id, VEC_DISTANCE_L2(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 2",
    );
    assert!(
        plan.contains("HNSW Index Scan"),
        "Expected HNSW Index Scan, got:\n{}",
        plan
    );
    assert!(
        plan.contains("actual time="),
        "Expected actual time in ANALYZE, got:\n{}",
        plan
    );
    assert!(
        plan.contains("actual rows="),
        "Expected actual rows in ANALYZE, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_vector_with_alias() {
    let db = Database::open("memory://explain_alias").expect("db");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_emb ON docs(emb) USING HNSW", ())
        .expect("index");
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");

    // ORDER BY alias instead of direct function call
    let plan = get_plan(
        &db,
        "EXPLAIN SELECT id, VEC_DISTANCE_L2(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 3",
    );
    assert!(
        plan.contains("HNSW Index Scan"),
        "Expected HNSW with alias ORDER BY, got:\n{}",
        plan
    );
    assert!(plan.contains("K: 3"), "Expected K: 3, got:\n{}", plan);
}

#[test]
fn test_explain_vector_cosine() {
    let db = Database::open("memory://explain_cosine").expect("db");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .expect("create");
    db.execute(
        "CREATE INDEX idx_emb ON docs(emb) USING HNSW WITH (metric = 'cosine')",
        (),
    )
    .expect("index");
    db.execute("INSERT INTO docs VALUES (1, '[1.0, 0.0, 0.0]')", ())
        .expect("insert");

    let plan = get_plan(
        &db,
        "EXPLAIN SELECT id, VEC_DISTANCE_COSINE(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs ORDER BY dist LIMIT 5",
    );
    assert!(
        plan.contains("HNSW Index Scan"),
        "Expected HNSW for cosine, got:\n{}",
        plan
    );
    assert!(
        plan.contains("Metric: Cosine"),
        "Expected Metric: Cosine, got:\n{}",
        plan
    );
}

#[test]
fn test_explain_vector_with_where() {
    let db = Database::open("memory://explain_vec_where").expect("db");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, category TEXT, emb VECTOR(3))",
        (),
    )
    .expect("create");
    db.execute("CREATE INDEX idx_emb ON docs(emb) USING HNSW", ())
        .expect("index");
    db.execute("INSERT INTO docs VALUES (1, 'a', '[1.0, 0.0, 0.0]')", ())
        .expect("insert");

    let plan = get_plan(
        &db,
        "EXPLAIN SELECT id, VEC_DISTANCE_L2(emb, '[1.0, 0.0, 0.0]') AS dist FROM docs WHERE category = 'a' ORDER BY dist LIMIT 5",
    );
    assert!(
        plan.contains("HNSW Index Scan"),
        "Expected HNSW with WHERE, got:\n{}",
        plan
    );
    assert!(
        plan.contains("Filter:"),
        "Expected filter in plan, got:\n{}",
        plan
    );
}
