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

//! COLLATE Function Tests
//!
//! Tests COLLATE function for case-insensitive and accent-insensitive comparisons

use stoolap::Database;

fn setup_collate_test_table(db: &Database) {
    db.execute(
        "CREATE TABLE collate_test (
            id INTEGER,
            text_value TEXT
        )",
        (),
    )
    .expect("Failed to create collate_test table");

    // Insert test data
    db.execute(
        "INSERT INTO collate_test (id, text_value) VALUES
        (1, 'Apple'),
        (2, 'apple'),
        (3, 'APPLE'),
        (4, 'Banana'),
        (5, 'banana'),
        (6, 'BANANA'),
        (7, 'Café'),
        (8, 'cafe'),
        (9, 'CAFE'),
        (10, 'Nação'),
        (11, 'nacao'),
        (12, 'NACAO')",
        (),
    )
    .expect("Failed to insert test data");
}

/// Test BINARY collation (case-sensitive)
#[test]
fn test_collate_binary() {
    let db = Database::open("memory://collate_binary").expect("Failed to create database");
    setup_collate_test_table(&db);

    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             WHERE COLLATE(text_value, 'BINARY') = 'Apple'
             ORDER BY id",
            (),
        )
        .expect("Failed to query with BINARY collation");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();
        let value: String = row.get(1).unwrap();

        assert_eq!(id, 1, "Expected id=1");
        assert_eq!(value, "Apple", "Expected value='Apple'");
        count += 1;
    }

    assert_eq!(count, 1, "Expected 1 row from BINARY collation");
}

/// Test NOCASE collation (case-insensitive)
#[test]
fn test_collate_nocase() {
    let db = Database::open("memory://collate_nocase").expect("Failed to create database");
    setup_collate_test_table(&db);

    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             WHERE COLLATE(text_value, 'NOCASE') = COLLATE('apple', 'NOCASE')
             ORDER BY id",
            (),
        )
        .expect("Failed to query with NOCASE collation");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();

        // Should match Apple, apple, and APPLE (ids 1, 2, 3)
        assert!((1..=3).contains(&id), "Unexpected id: {}", id);
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 rows from NOCASE collation");
}

/// Test NOACCENT collation (accent-insensitive)
#[test]
fn test_collate_noaccent() {
    let db = Database::open("memory://collate_noaccent").expect("Failed to create database");
    setup_collate_test_table(&db);

    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             WHERE COLLATE(LOWER(text_value), 'NOACCENT') = COLLATE('cafe', 'NOACCENT')
             ORDER BY id",
            (),
        )
        .expect("Failed to query with NOACCENT collation");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();

        // Should match Café, cafe, CAFE (ids 7, 8, 9)
        assert!((7..=9).contains(&id), "Unexpected id: {}", id);
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 rows from NOACCENT collation");
}

/// Test combined NOCASE and NOACCENT collation
#[test]
fn test_collate_combined_nocase_noaccent() {
    let db = Database::open("memory://collate_combined").expect("Failed to create database");
    setup_collate_test_table(&db);

    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             WHERE COLLATE(COLLATE(text_value, 'NOCASE'), 'NOACCENT') = COLLATE(COLLATE('nação', 'NOCASE'), 'NOACCENT')
             ORDER BY id", ())
        .expect("Failed to query with combined collation");

    let mut count = 0;
    for row in result {
        let row = row.expect("Failed to get row");
        let id: i64 = row.get(0).unwrap();

        // Should match Nação, nacao, NACAO (ids 10, 11, 12)
        assert!((10..=12).contains(&id), "Unexpected id: {}", id);
        count += 1;
    }

    assert_eq!(count, 3, "Expected 3 rows from combined collation");
}

/// Test ordering with COLLATE
#[test]
fn test_collate_order_by() {
    let db = Database::open("memory://collate_orderby").expect("Failed to create database");
    setup_collate_test_table(&db);

    let result = db
        .query(
            "SELECT id, text_value
             FROM collate_test
             ORDER BY COLLATE(text_value, 'NOCASE')
             LIMIT 6",
            (),
        )
        .expect("Failed to query with ORDER BY COLLATE");

    // First 6 should be all forms of "Apple" and "Banana"
    let mut apple_count = 0;
    let mut banana_count = 0;

    for row in result {
        let row = row.expect("Failed to get row");
        let value: String = row.get(1).unwrap();
        let lower_value = value.to_lowercase();

        if lower_value == "apple" {
            apple_count += 1;
        } else if lower_value == "banana" {
            banana_count += 1;
        } else {
            panic!("Unexpected value: {}", value);
        }
    }

    assert_eq!(apple_count, 3, "Expected 3 'Apple' entries");
    assert_eq!(banana_count, 3, "Expected 3 'Banana' entries");
}

/// Test NULL handling with COLLATE
#[test]
fn test_collate_null_handling() {
    let db = Database::open("memory://collate_null").expect("Failed to create database");
    setup_collate_test_table(&db);

    // Insert NULL value
    db.execute(
        "INSERT INTO collate_test (id, text_value) VALUES (13, NULL)",
        (),
    )
    .expect("Failed to insert NULL value");

    // Query with COLLATE on NULL - should not crash
    let result = db
        .query(
            "SELECT COLLATE(text_value, 'BINARY')
             FROM collate_test
             WHERE id = 13",
            (),
        )
        .expect("Failed to query NULL value");

    let mut found = false;
    for row in result {
        let row = row.expect("Failed to get row");
        // The result should be NULL - use Option<String> to properly detect NULL
        let value: Option<String> = row.get(0).unwrap();
        assert!(value.is_none(), "Expected NULL result");
        found = true;
    }

    assert!(found, "Expected to find the NULL row");
}
