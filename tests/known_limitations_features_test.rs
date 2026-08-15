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

//! Comprehensive integration tests for features implemented to resolve known limitations:
//! - JSON manipulation functions (JSON_SET, JSON_INSERT, JSON_REPLACE, JSON_REMOVE, JSON_CONTAINS, JSON_CONTAINS_PATH, JSON_QUOTE, JSON_UNQUOTE, ARRAY_LENGTH, ARRAY_CONTAINS)
//! - Geospatial GIS functions (ST_Point, ST_X, ST_Y, ST_Distance, ST_Distance_Sphere, ST_DWithin, ST_AsText, ST_GeomFromText, ST_Contains, ST_Area, ST_Centroid)
//! - Regular Expression & String functions (SUBSTRING_INDEX, REGEXP_LIKE, REGEXP_REPLACE, REGEXP_SUBSTR, HEX, UNHEX)
//! - UUID Generation (GEN_RANDOM_UUID, UUID)
//! - Vector & AI distance & normalization functions (VEC_NORM, VECTOR_NORM, VEC_NORMALIZE, COSINE_DISTANCE, L2_DISTANCE, INNER_PRODUCT)
//! - Timezone conversion function (CONVERT_TZ)
//! - Conditional Upsert (ON CONFLICT ... DO UPDATE ... WHERE ...)
//! - MySQL-style VALUES(column) in ON DUPLICATE KEY UPDATE

use std::error::Error;
use stoolap::Database;

#[test]
fn test_sql_json_set_and_modify_functions() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, doc JSON);",
        (),
    )?;

    db.execute(
        "INSERT INTO documents VALUES (1, '{\"name\": \"Alice\", \"stats\": {\"views\": 10}}');",
        (),
    )?;

    // Test JSON_SET: update existing and add new
    let rows = db.query(
        "SELECT JSON_SET(doc, '$.stats.views', 11, '$.stats.likes', 5, '$.active', true) FROM documents WHERE id = 1;",
        (),
    )?;
    let mut rows_iter = rows.into_iter();
    let row = rows_iter.next().ok_or("expected row")??;
    let json_str: String = row.get(0)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    assert_eq!(parsed["stats"]["views"], 11);
    assert_eq!(parsed["stats"]["likes"], 5);
    assert_eq!(parsed["active"], true);
    assert_eq!(parsed["name"], "Alice");

    // Test JSON_INSERT: does not overwrite existing
    let rows = db.query(
        "SELECT JSON_INSERT(doc, '$.name', 'Bob', '$.tag', 'admin') FROM documents WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let json_str: String = row.get(0)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    assert_eq!(parsed["name"], "Alice"); // unchanged
    assert_eq!(parsed["tag"], "admin"); // inserted

    // Test JSON_REPLACE: only replaces existing
    let rows = db.query(
        "SELECT JSON_REPLACE(doc, '$.name', 'Alicia', '$.nonexistent', 99) FROM documents WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let json_str: String = row.get(0)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    assert_eq!(parsed["name"], "Alicia"); // replaced
    assert!(parsed.get("nonexistent").is_none()); // not added

    // Test JSON_REMOVE: removes property
    let rows = db.query(
        "SELECT JSON_REMOVE(doc, '$.stats.views') FROM documents WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let json_str: String = row.get(0)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    assert!(parsed["stats"].get("views").is_none());
    assert_eq!(parsed["name"], "Alice");

    Ok(())
}

#[test]
fn test_sql_json_contains_and_path_queries() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, details JSON);",
        (),
    )?;

    db.execute(
        "INSERT INTO items VALUES (1, '{\"user\": {\"id\": 42, \"roles\": [\"admin\", \"editor\"]}, \"active\": true}');",
        (),
    )?;

    // Test JSON_CONTAINS
    let rows = db.query(
        "SELECT JSON_CONTAINS(details, '{\"id\": 42}', '$.user') FROM items WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let matched: bool = row.get(0)?;
    assert!(matched);

    let rows = db.query(
        "SELECT JSON_CONTAINS(details, '\"admin\"', '$.user.roles') FROM items WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let matched: bool = row.get(0)?;
    assert!(matched);

    // Test JSON_CONTAINS_PATH
    let rows = db.query(
        "SELECT JSON_CONTAINS_PATH(details, 'one', '$.user.id', '$.missing') FROM items WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let matched: bool = row.get(0)?;
    assert!(matched);

    let rows = db.query(
        "SELECT JSON_CONTAINS_PATH(details, 'all', '$.user.id', '$.missing') FROM items WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let matched: bool = row.get(0)?;
    assert!(!matched);

    // Test JSON_QUOTE and JSON_UNQUOTE
    let rows = db.query("SELECT JSON_UNQUOTE(JSON_QUOTE('test_string'));", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let unquoted: String = row.get(0)?;
    assert_eq!(unquoted, "test_string");

    // Test ARRAY_CONTAINS and ARRAY_LENGTH
    let rows = db.query(
        "SELECT ARRAY_CONTAINS('[10, 20, 30]', 20), ARRAY_LENGTH('[10, 20, 30]');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let contains_20: bool = row.get(0)?;
    let len: i64 = row.get(1)?;
    assert!(contains_20);
    assert_eq!(len, 3);

    Ok(())
}

#[test]
fn test_sql_geospatial_gis_functions() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE places (id INTEGER PRIMARY KEY, name TEXT, location TEXT);",
        (),
    )?;

    db.execute(
        "INSERT INTO places VALUES (1, 'Origin', ST_POINT(0, 0)), (2, 'Target', ST_POINT(3, 4));",
        (),
    )?;

    // Test ST_X, ST_Y, ST_DISTANCE
    let rows = db.query(
        "SELECT p1.name, ST_X(p1.location), ST_Y(p1.location), ST_DISTANCE(p1.location, p2.location), ST_DWITHIN(p1.location, p2.location, 5.0) \
         FROM places p1, places p2 WHERE p1.id = 1 AND p2.id = 2;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let name: String = row.get(0)?;
    let x: f64 = row.get(1)?;
    let y: f64 = row.get(2)?;
    let dist: f64 = row.get(3)?;
    let dwithin: bool = row.get(4)?;
    assert_eq!(name, "Origin");
    assert_eq!(x, 0.0);
    assert_eq!(y, 0.0);
    assert_eq!(dist, 5.0);
    assert!(dwithin);

    // Test ST_ASTEXT and ST_GEOMFROMTEXT
    let rows = db.query(
        "SELECT ST_ASTEXT(ST_POINT(12.34, 56.78)), ST_X(ST_GEOMFROMTEXT('POINT(12.34 56.78)'));",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let wkt: String = row.get(0)?;
    let parsed_x: f64 = row.get(1)?;
    assert!(wkt.contains("POINT(12.34 56.78)"));
    assert_eq!(parsed_x, 12.34);

    // Test ST_DISTANCE_SPHERE (London to Paris)
    let rows = db.query(
        "SELECT ST_DISTANCE_SPHERE(ST_POINT(-0.1278, 51.5074), ST_POINT(2.3522, 48.8566));",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let dist_m: f64 = row.get(0)?;
    assert!(dist_m > 340_000.0 && dist_m < 350_000.0);

    // Test ST_CONTAINS and ST_AREA for polygons
    let rows = db.query(
        "SELECT ST_CONTAINS('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))', ST_POINT(5, 5)), \
                ST_CONTAINS('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))', ST_POINT(20, 20)), \
                ST_AREA('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let inside: bool = row.get(0)?;
    let outside: bool = row.get(1)?;
    let area: f64 = row.get(2)?;
    assert!(inside);
    assert!(!outside);
    assert_eq!(area, 100.0);

    Ok(())
}

#[test]
fn test_sql_string_regex_hex_and_uuid_functions() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    // Test SUBSTRING_INDEX
    let rows = db.query(
        "SELECT SUBSTRING_INDEX('www.stoolap.io', '.', 2), SUBSTRING_INDEX('www.stoolap.io', '.', -1);",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let prefix: String = row.get(0)?;
    let suffix: String = row.get(1)?;
    assert_eq!(prefix, "www.stoolap");
    assert_eq!(suffix, "io");

    // Test REGEXP_LIKE, REGEXP_REPLACE, REGEXP_SUBSTR
    let rows = db.query(
        "SELECT REGEXP_LIKE('user@stoolap.io', '^[a-z]+@[a-z.]+$'), \
                REGEXP_REPLACE('Price: 100 USD', '[0-9]+', '200'), \
                REGEXP_SUBSTR('Order #12345 completed', '[0-9]+');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let matches: bool = row.get(0)?;
    let replaced: String = row.get(1)?;
    let extracted: String = row.get(2)?;
    assert!(matches);
    assert_eq!(replaced, "Price: 200 USD");
    assert_eq!(extracted, "12345");

    // Test HEX and UNHEX
    let rows = db.query(
        "SELECT HEX('Stoolap'), UNHEX('53746F6F6C6170'), HEX(255);",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let hex_val: String = row.get(0)?;
    let unhex_val: String = row.get(1)?;
    let hex_int: String = row.get(2)?;
    assert_eq!(hex_val, "53746F6F6C6170");
    assert_eq!(unhex_val, "Stoolap");
    assert_eq!(hex_int, "FF");

    // Test GEN_RANDOM_UUID and UUID
    let rows = db.query("SELECT GEN_RANDOM_UUID(), UUID();", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let uuid1: String = row.get(0)?;
    let uuid2: String = row.get(1)?;
    assert_eq!(uuid1.len(), 36);
    assert_eq!(uuid2.len(), 36);
    assert_eq!(&uuid1[14..15], "4"); // UUID v4 version check

    Ok(())
}

#[test]
fn test_sql_vector_norm_and_distance_functions() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3));",
        (),
    )?;

    db.execute(
        "INSERT INTO embeddings VALUES (1, '[3.0, 4.0, 0.0]'), (2, '[0.0, 1.0, 0.0]');",
        (),
    )?;

    // Test VEC_NORM, VECTOR_NORM, INNER_PRODUCT, COSINE_DISTANCE, L2_DISTANCE
    let rows = db.query(
        "SELECT VEC_NORM(vec), VECTOR_NORM(vec) FROM embeddings WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let norm1: f64 = row.get(0)?;
    let norm2: f64 = row.get(1)?;
    assert_eq!(norm1, 5.0);
    assert_eq!(norm2, 5.0);

    let rows = db.query(
        "SELECT INNER_PRODUCT(e1.vec, e2.vec), L2_DISTANCE(e1.vec, e2.vec), COSINE_DISTANCE(e1.vec, e2.vec) \
         FROM embeddings e1, embeddings e2 WHERE e1.id = 1 AND e2.id = 2;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let dot: f64 = row.get(0)?;
    let l2: f64 = row.get(1)?;
    let cos_dist: f64 = row.get(2)?;
    assert!((dot - 4.0).abs() < 1e-5);
    assert!((l2 - 18.0f64.sqrt()).abs() < 1e-5);
    assert!((cos_dist - 0.2).abs() < 1e-5); // 1 - 4/(5*1) = 0.2

    Ok(())
}

#[test]
fn test_sql_convert_tz() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    let rows = db.query(
        "SELECT CONVERT_TZ('2024-06-01 12:00:00', 'UTC', '+05:30');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let ts_str = row.get::<String>(0)?;
    assert!(ts_str.contains("17:30:00"));

    let rows = db.query(
        "SELECT CONVERT_TZ('2024-06-01 12:00:00', '+00:00', '-07:00');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let ts_str = row.get::<String>(0)?;
    assert!(ts_str.contains("05:00:00"));

    Ok(())
}

#[test]
fn test_sql_conditional_upsert_on_conflict_where() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE counters (id INTEGER PRIMARY KEY, version INTEGER, count INTEGER);",
        (),
    )?;

    db.execute("INSERT INTO counters VALUES (1, 10, 100);", ())?;

    // Upsert with WHERE condition that is NOT satisfied (incoming version 5 <= existing version 10)
    // Should NOT update count!
    db.execute(
        "INSERT INTO counters (id, version, count) VALUES (1, 5, 200) \
         ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version, count = EXCLUDED.count \
         WHERE EXCLUDED.version > counters.version;",
        (),
    )?;

    let rows = db.query("SELECT version, count FROM counters WHERE id = 1;", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let ver: i64 = row.get(0)?;
    let count: i64 = row.get(1)?;
    assert_eq!(ver, 10);
    assert_eq!(count, 100); // Unchanged!

    // Upsert with WHERE condition that IS satisfied (incoming version 15 > existing version 10)
    // Should update count!
    db.execute(
        "INSERT INTO counters (id, version, count) VALUES (1, 15, 300) \
         ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version, count = EXCLUDED.count \
         WHERE EXCLUDED.version > counters.version;",
        (),
    )?;

    let rows = db.query("SELECT version, count FROM counters WHERE id = 1;", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let ver: i64 = row.get(0)?;
    let count: i64 = row.get(1)?;
    assert_eq!(ver, 15);
    assert_eq!(count, 300); // Successfully updated!

    Ok(())
}

#[test]
fn test_sql_mysql_values_syntax_in_upsert() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE inventory (item_id INTEGER PRIMARY KEY, quantity INTEGER);",
        (),
    )?;

    db.execute("INSERT INTO inventory VALUES (101, 50);", ())?;

    // MySQL style: ON DUPLICATE KEY UPDATE quantity = inventory.quantity + VALUES(quantity)
    db.execute(
        "INSERT INTO inventory (item_id, quantity) VALUES (101, 25) \
         ON DUPLICATE KEY UPDATE quantity = inventory.quantity + VALUES(quantity);",
        (),
    )?;

    let rows = db.query("SELECT quantity FROM inventory WHERE item_id = 101;", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let qty: i64 = row.get(0)?;
    assert_eq!(qty, 75);

    Ok(())
}

#[test]
fn test_sql_ip_address_network_functions() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    // Test INET_ATON and INET_NTOA
    let rows = db.query(
        "SELECT INET_ATON('192.168.1.1'), INET_NTOA(3232235777);",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let aton: i64 = row.get(0)?;
    let ntoa: String = row.get(1)?;
    assert_eq!(aton, 3232235777);
    assert_eq!(ntoa, "192.168.1.1");

    // Test IS_IPV4 and IS_IPV6
    let rows = db.query(
        "SELECT IS_IPV4('10.0.0.1'), IS_IPV4('invalid_ip'), IS_IPV6('2001:db8::1'), IS_IPV6('10.0.0.1');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let valid_v4: bool = row.get(0)?;
    let invalid_v4: bool = row.get(1)?;
    let valid_v6: bool = row.get(2)?;
    let not_v6: bool = row.get(3)?;
    assert!(valid_v4);
    assert!(!invalid_v4);
    assert!(valid_v6);
    assert!(!not_v6);

    Ok(())
}

#[test]
fn test_sql_datetime_construction_and_age_functions() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    // Test MAKE_DATE, MAKE_TIME, MAKE_TIMESTAMP
    let rows = db.query(
        "SELECT MAKE_DATE(2025, 12, 25), MAKE_TIME(14, 30, 0), MAKE_TIMESTAMP(2025, 12, 25, 14, 30, 45);",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let date_str = row.get::<String>(0)?;
    let time_str = row.get::<String>(1)?;
    let ts_str = row.get::<String>(2)?;
    assert!(date_str.contains("2025-12-25"));
    assert_eq!(time_str, "14:30:00");
    assert!(ts_str.contains("2025-12-25") && ts_str.contains("14:30:45"));

    // Test AGE between two timestamps
    let rows = db.query(
        "SELECT AGE('2025-06-01 00:00:00', '2023-01-01 00:00:00');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let age_str = row.get::<String>(0)?;
    assert!(age_str.contains("year"));

    Ok(())
}

#[test]
fn test_sql_extended_vector_metrics() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE vectors (id INTEGER PRIMARY KEY, v VECTOR(3));",
        (),
    )?;

    db.execute(
        "INSERT INTO vectors VALUES (1, '[1.0, 2.0, 3.0]'), (2, '[4.0, 5.0, 6.0]');",
        (),
    )?;

    // Test COSINE_SIMILARITY, MANHATTAN_DISTANCE, CHEBYSHEV_DISTANCE, HAMMING_DISTANCE
    let rows = db.query(
        "SELECT COSINE_SIMILARITY(v1.v, v2.v), MANHATTAN_DISTANCE(v1.v, v2.v), \
                CHEBYSHEV_DISTANCE(v1.v, v2.v), HAMMING_DISTANCE(v1.v, v2.v) \
         FROM vectors v1, vectors v2 WHERE v1.id = 1 AND v2.id = 2;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let sim: f64 = row.get(0)?;
    let manhattan: f64 = row.get(1)?;
    let chebyshev: f64 = row.get(2)?;
    let hamming: i64 = row.get(3)?;

    // Cosine similarity for [1,2,3] and [4,5,6]: (4+10+18) / (sqrt(14)*sqrt(77)) = 32 / sqrt(1078) ≈ 0.97463
    assert!((sim - 0.97463).abs() < 1e-4);
    // Manhattan: |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
    assert_eq!(manhattan, 9.0);
    // Chebyshev: max(3, 3, 3) = 3
    assert_eq!(chebyshev, 3.0);
    // Hamming: 3 differing dimensions
    assert_eq!(hamming, 3);

    Ok(())
}

#[test]
fn test_sql_spatial_intersects_and_envelope() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    // Test ST_INTERSECTS and ST_ENVELOPE
    let rows = db.query(
        "SELECT ST_INTERSECTS('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))', ST_POINT(5, 5)), \
                ST_INTERSECTS('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))', 'POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))'), \
                ST_ENVELOPE('POLYGON((1 2, 8 4, 6 9, 2 7, 1 2))');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let pt_intersects: bool = row.get(0)?;
    let poly_intersects: bool = row.get(1)?;
    let envelope: String = row.get(2)?;

    assert!(pt_intersects);
    assert!(poly_intersects);
    assert!(envelope.contains("POLYGON((1 2, 8 2, 8 9, 1 9, 1 2))"));

    Ok(())
}

#[test]
fn test_sql_insert_or_ignore() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER);",
        (),
    )?;

    db.execute("INSERT INTO accounts VALUES (1, 100);", ())?;

    // INSERT OR IGNORE with conflicting PK: should silently ignore without error or modification
    db.execute("INSERT OR IGNORE INTO accounts VALUES (1, 500);", ())?;

    let rows = db.query("SELECT balance FROM accounts WHERE id = 1;", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let balance: i64 = row.get(0)?;
    assert_eq!(balance, 100); // Intact!

    // Non-conflicting row should succeed
    db.execute("INSERT OR IGNORE INTO accounts VALUES (2, 250);", ())?;
    let rows = db.query("SELECT balance FROM accounts WHERE id = 2;", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let balance2: i64 = row.get(0)?;
    assert_eq!(balance2, 250);

    Ok(())
}

#[test]
fn test_sql_alter_table_if_exists_and_if_not_exists() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT);",
        (),
    )?;

    // ADD COLUMN IF NOT EXISTS (first time: adds column)
    db.execute(
        "ALTER TABLE employees ADD COLUMN IF NOT EXISTS email TEXT;",
        (),
    )?;

    // ADD COLUMN IF NOT EXISTS (second time: idempotent no-op)
    db.execute(
        "ALTER TABLE employees ADD COLUMN IF NOT EXISTS email TEXT;",
        (),
    )?;

    // DROP COLUMN IF EXISTS on existing column
    db.execute("ALTER TABLE employees DROP COLUMN IF EXISTS email;", ())?;

    // DROP COLUMN IF EXISTS on non-existent column: idempotent no-op
    db.execute(
        "ALTER TABLE employees DROP COLUMN IF EXISTS nonexistent_col;",
        (),
    )?;

    // RENAME COLUMN IF EXISTS on non-existent column: idempotent no-op
    db.execute(
        "ALTER TABLE employees RENAME COLUMN IF EXISTS nonexistent_col TO whatever;",
        (),
    )?;

    // RENAME COLUMN IF EXISTS on existing column
    db.execute(
        "ALTER TABLE employees RENAME COLUMN IF EXISTS name TO full_name;",
        (),
    )?;

    db.execute("INSERT INTO employees VALUES (1, 'Alice');", ())?;
    let rows = db.query("SELECT full_name FROM employees WHERE id = 1;", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let name: String = row.get(0)?;
    assert_eq!(name, "Alice");

    Ok(())
}

#[test]
fn test_sql_string_field_find_in_set_elt_soundex_quote() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    // Test FIELD, FIND_IN_SET, ELT, SOUNDEX, QUOTE
    let rows = db.query(
        "SELECT FIELD('b', 'a', 'b', 'c'), FIND_IN_SET('b', 'a,b,c'), ELT(2, 'first', 'second', 'third'), \
                SOUNDEX('Robert'), QUOTE('Don''t');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let field_val: i64 = row.get(0)?;
    let find_val: i64 = row.get(1)?;
    let elt_val: String = row.get(2)?;
    let soundex_val: String = row.get(3)?;
    let quote_val: String = row.get(4)?;

    assert_eq!(field_val, 2);
    assert_eq!(find_val, 2);
    assert_eq!(elt_val, "second");
    assert_eq!(soundex_val, "R163");
    assert_eq!(quote_val, "'Don\\'t'");

    Ok(())
}

#[test]
fn test_sql_vector_arithmetic_slice_concat() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE vecs (id INTEGER PRIMARY KEY, v1 VECTOR(3), v2 VECTOR(3));",
        (),
    )?;

    db.execute(
        "INSERT INTO vecs VALUES (1, '[1.0, 2.0, 3.0]', '[4.0, 5.0, 6.0]');",
        (),
    )?;

    // Test VEC_ADD, VEC_SUB, VEC_MUL, VEC_SLICE, VEC_CONCAT
    let rows = db.query(
        "SELECT VEC_ADD(v1, v2), VEC_SUB(v2, v1), VEC_MUL(v1, v2), \
                VEC_SLICE(v1, 2, 2), VEC_CONCAT(v1, v2) \
         FROM vecs WHERE id = 1;",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let add_vec = row.get::<stoolap::Value>(0)?;
    let sub_vec = row.get::<stoolap::Value>(1)?;
    let mul_vec = row.get::<stoolap::Value>(2)?;
    let slice_vec = row.get::<stoolap::Value>(3)?;
    let concat_vec = row.get::<stoolap::Value>(4)?;

    assert_eq!(add_vec.to_string(), "[5.0, 7.0, 9.0]");
    assert_eq!(sub_vec.to_string(), "[3.0, 3.0, 3.0]");
    assert_eq!(mul_vec.to_string(), "[4.0, 10.0, 18.0]");
    assert_eq!(slice_vec.to_string(), "[2.0, 3.0]");
    assert_eq!(concat_vec.to_string(), "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]");

    Ok(())
}

#[test]
fn test_sql_extended_gis_metrics_and_srid() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    // Test ST_LENGTH, ST_PERIMETER, ST_NUMPOINTS, ST_SRID, ST_SETSRID
    let rows = db.query(
        "SELECT ST_LENGTH('POLYGON((0 0, 3 0, 3 4, 0 0))'), \
                ST_PERIMETER('POLYGON((0 0, 4 0, 4 3, 0 3, 0 0))'), \
                ST_NUMPOINTS('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'), \
                ST_SRID('POINT(10 20)'), \
                ST_SETSRID('POINT(10 20)', 4326);",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let length: f64 = row.get(0)?;
    let perimeter: f64 = row.get(1)?;
    let num_points: i64 = row.get(2)?;
    let srid: i64 = row.get(3)?;
    let set_srid: String = row.get(4)?;

    // Length of path (0,0)->(3,0) [3] + (3,0)->(3,4) [4] + (3,4)->(0,0) [5] = 12
    assert!((length - 12.0).abs() < 1e-4);
    // Perimeter of 4x3 rect = 4 + 3 + 4 + 3 = 14
    assert!((perimeter - 14.0).abs() < 1e-4);
    assert_eq!(num_points, 5);
    assert_eq!(srid, 4326);
    assert!(set_srid.contains("POINT(10 20)"));

    Ok(())
}

#[test]
fn test_sql_extended_datetime_clock_last_day() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    // Test TIMEOFDAY, CLOCK_TIMESTAMP, STATEMENT_TIMESTAMP, LAST_DAY
    let rows = db.query(
        "SELECT TIMEOFDAY(), CLOCK_TIMESTAMP(), STATEMENT_TIMESTAMP(), LAST_DAY('2024-02-14 10:00:00');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let timeofday: String = row.get(0)?;
    let _clock_ts: stoolap::Value = row.get(1)?;
    let _stmt_ts: stoolap::Value = row.get(2)?;
    let last_day: String = row.get(3)?;

    assert!(!timeofday.is_empty());
    // 2024 is a leap year, last day of Feb is 2024-02-29
    assert!(last_day.contains("2024-02-29"));

    Ok(())
}

#[test]
fn test_sql_inet6_and_json_valid_functions() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    // Test INET6_ATON, INET6_NTOA, IS_VALID_JSON
    let rows = db.query(
        "SELECT INET6_ATON('2001:db8::1'), \
                INET6_NTOA('20010DB8000000000000000000000001'), \
                IS_VALID_JSON('{\"key\": 123}'), \
                IS_VALID_JSON('{invalid json');",
        (),
    )?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let aton6: String = row.get(0)?;
    let ntoa6: String = row.get(1)?;
    let valid_json: bool = row.get(2)?;
    let invalid_json: bool = row.get(3)?;

    assert_eq!(aton6, "20010DB8000000000000000000000001");
    assert_eq!(ntoa6, "2001:db8::1");
    assert!(valid_json);
    assert!(!invalid_json);

    Ok(())
}

#[test]
fn test_sql_replace_into_and_insert_or_replace() -> Result<(), Box<dyn Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);",
        (),
    )?;

    db.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@old.com');", ())?;

    // REPLACE INTO with duplicate PK: should replace the row
    db.execute("REPLACE INTO users VALUES (1, 'Alice Updated', 'alice@new.com');", ())?;

    let rows = db.query("SELECT name, email FROM users WHERE id = 1;", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let name: String = row.get(0)?;
    let email: String = row.get(1)?;
    assert_eq!(name, "Alice Updated");
    assert_eq!(email, "alice@new.com");

    // INSERT OR REPLACE with duplicate PK
    db.execute("INSERT OR REPLACE INTO users VALUES (1, 'Alice Third', 'alice@final.com');", ())?;

    let rows = db.query("SELECT name, email FROM users WHERE id = 1;", ())?;
    let row = rows.into_iter().next().ok_or("expected row")??;
    let name2: String = row.get(0)?;
    let email2: String = row.get(1)?;
    assert_eq!(name2, "Alice Third");
    assert_eq!(email2, "alice@final.com");

    Ok(())
}
