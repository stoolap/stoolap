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

//! Integration tests for hash functions: MD5, SHA1, SHA256, SHA384, SHA512, CRC32

use std::sync::atomic::{AtomicUsize, Ordering};
use stoolap::Database;

static TEST_ID: AtomicUsize = AtomicUsize::new(0);

fn setup_db() -> Database {
    let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
    let db =
        Database::open(&format!("memory://hash_test_{}", id)).expect("Failed to create database");
    db.execute(
        "CREATE TABLE test_data (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO test_data VALUES (1, 'hello', 42), (2, 'world', 100), (3, NULL, NULL)",
        (),
    )
    .expect("Failed to insert");
    db
}

// ---------------------------------------------------------------------------
// MD5
// ---------------------------------------------------------------------------

#[test]
fn test_md5_literal() {
    let db = Database::open("memory://md5_literal").expect("Failed to create database");
    let result: String = db
        .query_one("SELECT MD5('hello')", ())
        .expect("Failed to execute MD5");
    assert_eq!(result, "5d41402abc4b2a76b9719d911017c592");
}

#[test]
fn test_md5_empty_string() {
    let db = Database::open("memory://md5_empty").expect("Failed to create database");
    let result: String = db
        .query_one("SELECT MD5('')", ())
        .expect("Failed to execute MD5");
    assert_eq!(result, "d41d8cd98f00b204e9800998ecf8427e");
}

#[test]
fn test_md5_column() {
    let db = setup_db();
    let result: String = db
        .query_one("SELECT MD5(name) FROM test_data WHERE id = 1", ())
        .expect("Failed to execute MD5 on column");
    assert_eq!(result, "5d41402abc4b2a76b9719d911017c592");
}

#[test]
fn test_md5_null() {
    let db = setup_db();
    let result: Option<String> = db
        .query_one("SELECT MD5(name) FROM test_data WHERE id = 3", ())
        .expect("Failed to execute MD5 on NULL");
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// SHA1
// ---------------------------------------------------------------------------

#[test]
fn test_sha1_literal() {
    let db = Database::open("memory://sha1_literal").expect("Failed to create database");
    let result: String = db
        .query_one("SELECT SHA1('hello')", ())
        .expect("Failed to execute SHA1");
    assert_eq!(result, "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d");
}

#[test]
fn test_sha1_column() {
    let db = setup_db();
    let result: String = db
        .query_one("SELECT SHA1(name) FROM test_data WHERE id = 2", ())
        .expect("Failed to execute SHA1 on column");
    assert_eq!(result, "7c211433f02071597741e6ff5a8ea34789abbf43");
}

// ---------------------------------------------------------------------------
// SHA256
// ---------------------------------------------------------------------------

#[test]
fn test_sha256_literal() {
    let db = Database::open("memory://sha256_literal").expect("Failed to create database");
    let result: String = db
        .query_one("SELECT SHA256('hello')", ())
        .expect("Failed to execute SHA256");
    assert_eq!(
        result,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}

#[test]
fn test_sha256_column() {
    let db = setup_db();
    let result: String = db
        .query_one("SELECT SHA256(name) FROM test_data WHERE id = 1", ())
        .expect("Failed to execute SHA256 on column");
    assert_eq!(
        result,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}

// ---------------------------------------------------------------------------
// SHA384
// ---------------------------------------------------------------------------

#[test]
fn test_sha384_literal() {
    let db = Database::open("memory://sha384_literal").expect("Failed to create database");
    let result: String = db
        .query_one("SELECT SHA384('hello')", ())
        .expect("Failed to execute SHA384");
    assert_eq!(
        result,
        "59e1748777448c69de6b800d7a33bbfb9ff1b463e44354c3553bcdb9c666fa90125a3c79f90397bdf5f6a13de828684f"
    );
}

// ---------------------------------------------------------------------------
// SHA512
// ---------------------------------------------------------------------------

#[test]
fn test_sha512_literal() {
    let db = Database::open("memory://sha512_literal").expect("Failed to create database");
    let result: String = db
        .query_one("SELECT SHA512('hello')", ())
        .expect("Failed to execute SHA512");
    assert_eq!(
        result,
        "9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca72323c3d99ba5c11d7c7acc6e14b8c5da0c4663475c2e5c3adef46f73bcdec043"
    );
}

// ---------------------------------------------------------------------------
// CRC32
// ---------------------------------------------------------------------------

#[test]
fn test_crc32_literal() {
    let db = Database::open("memory://crc32_literal").expect("Failed to create database");
    let result: i64 = db
        .query_one("SELECT CRC32('hello')", ())
        .expect("Failed to execute CRC32");
    assert_eq!(result, 907060870);
}

#[test]
fn test_crc32_column() {
    let db = setup_db();
    let result: i64 = db
        .query_one("SELECT CRC32(name) FROM test_data WHERE id = 1", ())
        .expect("Failed to execute CRC32 on column");
    assert_eq!(result, 907060870);
}

#[test]
fn test_crc32_null() {
    let db = setup_db();
    let result: Option<i64> = db
        .query_one("SELECT CRC32(name) FROM test_data WHERE id = 3", ())
        .expect("Failed to execute CRC32 on NULL");
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// Cross-function and edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_hash_integer_column() {
    let db = setup_db();
    // Hash an integer column
    let result: String = db
        .query_one("SELECT MD5(value) FROM test_data WHERE id = 1", ())
        .expect("Failed to execute MD5 on integer column");
    // MD5 of "42"
    assert_eq!(result, "a1d0c6e83f027327d8461063f4ac58a6");
}

#[test]
fn test_hash_in_where_clause() {
    let db = setup_db();
    // Use hash in WHERE clause
    let result: i64 = db
        .query_one(
            "SELECT id FROM test_data WHERE MD5(name) = '5d41402abc4b2a76b9719d911017c592'",
            (),
        )
        .expect("Failed to use MD5 in WHERE");
    assert_eq!(result, 1);
}

#[test]
fn test_multiple_hash_functions() {
    let db = Database::open("memory://multi_hash").expect("Failed to create database");
    // MD5
    let md5: String = db
        .query_one("SELECT MD5('test')", ())
        .expect("Failed to execute MD5");
    assert_eq!(md5, "098f6bcd4621d373cade4e832627b4f6");
    // SHA256
    let sha256: String = db
        .query_one("SELECT SHA256('test')", ())
        .expect("Failed to execute SHA256");
    assert_eq!(
        sha256,
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    );
    // CRC32
    let crc32: i64 = db
        .query_one("SELECT CRC32('test')", ())
        .expect("Failed to execute CRC32");
    assert_eq!(crc32, 3632233996);
}

#[test]
fn test_nested_hash() {
    let db = Database::open("memory://nested_hash").expect("Failed to create database");
    // Hash of a hash (double hashing)
    let result: String = db
        .query_one("SELECT MD5(MD5('hello'))", ())
        .expect("Failed to execute nested MD5");
    // MD5 of "5d41402abc4b2a76b9719d911017c592"
    assert_eq!(result, "69a329523ce1ec88bf63061863d9cb14");
}

#[test]
fn test_hash_with_concat() {
    let db = Database::open("memory://hash_concat").expect("Failed to create database");
    // Hash of concatenated strings
    let result: String = db
        .query_one("SELECT SHA256(CONCAT('hello', ' ', 'world'))", ())
        .expect("Failed to execute SHA256 with CONCAT");
    // SHA256 of "hello world"
    assert_eq!(
        result,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );
}
