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

//! Regression test: Upsert on tables without PRIMARY KEY (only UNIQUE constraint).
//! Previously failed because apply_on_duplicate_update could not build a WHERE clause
//! when there was no PK column.

use stoolap::Database;

#[test]
fn test_upsert_no_pk_single_unique() {
    let db = Database::open("memory://upsert_no_pk_single").unwrap();

    db.execute(
        "CREATE TABLE users (
            email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            score INTEGER DEFAULT 0
        )",
        (),
    )
    .unwrap();

    // Insert initial row
    db.execute(
        "INSERT INTO users (email, name, score) VALUES ('alice@test.com', 'Alice', 10)",
        (),
    )
    .unwrap();

    // Upsert: same email, update name and score (MySQL style)
    db.execute(
        "INSERT INTO users (email, name, score) VALUES ('alice@test.com', 'Alice Updated', 20)
         ON DUPLICATE KEY UPDATE name = EXCLUDED.name, score = EXCLUDED.score",
        (),
    )
    .unwrap();

    // Verify the update happened
    let results: Vec<_> = db
        .query(
            "SELECT email, name, score FROM users WHERE email = 'alice@test.com'",
            (),
        )
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get::<String>(1).unwrap(), "Alice Updated");
    assert_eq!(results[0].get::<i64>(2).unwrap(), 20);

    // Insert a different email — should succeed
    db.execute(
        "INSERT INTO users (email, name, score) VALUES ('bob@test.com', 'Bob', 30)",
        (),
    )
    .unwrap();

    // Verify total rows
    let count: i64 = db
        .query("SELECT COUNT(*) FROM users", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_upsert_no_pk_composite_unique() {
    let db = Database::open("memory://upsert_no_pk_composite").unwrap();

    db.execute(
        "CREATE TABLE metrics (
            host TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value FLOAT NOT NULL,
            UNIQUE(host, metric_name)
        )",
        (),
    )
    .unwrap();

    // Insert initial rows
    db.execute(
        "INSERT INTO metrics (host, metric_name, value) VALUES ('server1', 'cpu', 45.0)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO metrics (host, metric_name, value) VALUES ('server1', 'memory', 70.0)",
        (),
    )
    .unwrap();

    // Upsert: same host+metric_name, update value (PostgreSQL style)
    db.execute(
        "INSERT INTO metrics (host, metric_name, value) VALUES ('server1', 'cpu', 85.0)
         ON CONFLICT (host, metric_name) DO UPDATE SET value = EXCLUDED.value",
        (),
    )
    .unwrap();

    // Verify update
    let val: f64 = db
        .query(
            "SELECT value FROM metrics WHERE host = 'server1' AND metric_name = 'cpu'",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, 85.0);

    // Memory row should be unchanged
    let val: f64 = db
        .query(
            "SELECT value FROM metrics WHERE host = 'server1' AND metric_name = 'memory'",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, 70.0);

    // New host+metric should insert
    db.execute(
        "INSERT INTO metrics (host, metric_name, value) VALUES ('server2', 'cpu', 30.0)
         ON CONFLICT (host, metric_name) DO UPDATE SET value = EXCLUDED.value",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query("SELECT COUNT(*) FROM metrics", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn test_do_nothing_no_pk() {
    let db = Database::open("memory://do_nothing_no_pk").unwrap();

    db.execute(
        "CREATE TABLE tags (
            name TEXT NOT NULL UNIQUE,
            description TEXT
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO tags (name, description) VALUES ('rust', 'A systems language')",
        (),
    )
    .unwrap();

    // DO NOTHING on conflict — should silently skip
    db.execute(
        "INSERT INTO tags (name, description) VALUES ('rust', 'Updated description')
         ON CONFLICT (name) DO NOTHING",
        (),
    )
    .unwrap();

    // Verify original row is unchanged
    let desc: String = db
        .query("SELECT description FROM tags WHERE name = 'rust'", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(desc, "A systems language");

    // Verify only one row
    let count: i64 = db
        .query("SELECT COUNT(*) FROM tags", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_upsert_no_pk_expression_update() {
    let db = Database::open("memory://upsert_no_pk_expr").unwrap();

    db.execute(
        "CREATE TABLE counters (
            name TEXT NOT NULL UNIQUE,
            count INTEGER NOT NULL DEFAULT 0
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO counters (name, count) VALUES ('page_views', 100)",
        (),
    )
    .unwrap();

    // Upsert with expression: increment count
    db.execute(
        "INSERT INTO counters (name, count) VALUES ('page_views', 1)
         ON CONFLICT (name) DO UPDATE SET count = counters.count + EXCLUDED.count",
        (),
    )
    .unwrap();

    let val: i64 = db
        .query("SELECT count FROM counters WHERE name = 'page_views'", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, 101);

    // Do it again
    db.execute(
        "INSERT INTO counters (name, count) VALUES ('page_views', 5)
         ON CONFLICT (name) DO UPDATE SET count = counters.count + EXCLUDED.count",
        (),
    )
    .unwrap();

    let val: i64 = db
        .query("SELECT count FROM counters WHERE name = 'page_views'", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, 106);
}

#[test]
fn test_upsert_no_pk_insert_select() {
    let db = Database::open("memory://upsert_no_pk_select").unwrap();

    db.execute(
        "CREATE TABLE target (
            k TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL
        )",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE source (
            k TEXT NOT NULL,
            value TEXT NOT NULL
        )",
        (),
    )
    .unwrap();

    // Seed target
    db.execute(
        "INSERT INTO target (k, value) VALUES ('a', 'old_a'), ('b', 'old_b')",
        (),
    )
    .unwrap();

    // Seed source with overlapping + new keys
    db.execute(
        "INSERT INTO source (k, value) VALUES ('a', 'new_a'), ('c', 'new_c')",
        (),
    )
    .unwrap();

    // INSERT...SELECT with upsert
    db.execute(
        "INSERT INTO target (k, value)
         SELECT k, value FROM source
         ON CONFLICT (k) DO UPDATE SET value = EXCLUDED.value",
        (),
    )
    .unwrap();

    // 'a' should be updated
    let val: String = db
        .query("SELECT value FROM target WHERE k = 'a'", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, "new_a");

    // 'b' should be unchanged
    let val: String = db
        .query("SELECT value FROM target WHERE k = 'b'", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, "old_b");

    // 'c' should be inserted
    let val: String = db
        .query("SELECT value FROM target WHERE k = 'c'", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, "new_c");

    // Total 3 rows
    let count: i64 = db
        .query("SELECT COUNT(*) FROM target", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn test_upsert_no_pk_multiple_upserts() {
    let db = Database::open("memory://upsert_no_pk_multi").unwrap();

    db.execute(
        "CREATE TABLE config (
            section TEXT NOT NULL,
            k TEXT NOT NULL,
            value TEXT,
            UNIQUE(section, k)
        )",
        (),
    )
    .unwrap();

    // Batch insert
    db.execute(
        "INSERT INTO config (section, k, value) VALUES
         ('db', 'host', 'localhost'),
         ('db', 'port', '5432'),
         ('app', 'debug', 'false')",
        (),
    )
    .unwrap();

    // Batch upsert — update existing + insert new
    db.execute(
        "INSERT INTO config (section, k, value) VALUES
         ('db', 'host', '10.0.0.1'),
         ('db', 'port', '3306'),
         ('app', 'debug', 'true'),
         ('app', 'log_level', 'info')
         ON DUPLICATE KEY UPDATE value = EXCLUDED.value",
        (),
    )
    .unwrap();

    // Verify updates
    let val: String = db
        .query(
            "SELECT value FROM config WHERE section = 'db' AND k = 'host'",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, "10.0.0.1");

    let val: String = db
        .query(
            "SELECT value FROM config WHERE section = 'db' AND k = 'port'",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, "3306");

    let val: String = db
        .query(
            "SELECT value FROM config WHERE section = 'app' AND k = 'debug'",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, "true");

    // Verify new row
    let val: String = db
        .query(
            "SELECT value FROM config WHERE section = 'app' AND k = 'log_level'",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(val, "info");

    // Total 4 rows
    let count: i64 = db
        .query("SELECT COUNT(*) FROM config", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 4);
}
