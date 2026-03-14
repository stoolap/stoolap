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

//! Regression test for Bug: LIKE/ILIKE with parameterized patterns.
//!
//! Root cause: expression compiler's extract_pattern_string() only accepted
//! StringLiteral, not Parameter expressions. Dynamic LIKE ops were added
//! (LikeDynamic, GlobDynamic, RegexpDynamic) to compile the pattern at runtime.

use stoolap::Database;

fn setup_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).unwrap();
    db.execute(
        "CREATE TABLE logs (id INTEGER PRIMARY KEY, message TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO logs VALUES (1, 'hello world')", ())
        .unwrap();
    db.execute("INSERT INTO logs VALUES (2, 'foo bar')", ())
        .unwrap();
    db.execute("INSERT INTO logs VALUES (3, 'hello again')", ())
        .unwrap();
    db.execute("INSERT INTO logs VALUES (4, 'HELLO UPPER')", ())
        .unwrap();
    db
}

fn count_rows(db: &Database, sql: &str, params: impl stoolap::Params) -> usize {
    let mut rows = db.query(sql, params).unwrap();
    let mut count = 0;
    while let Some(Ok(_)) = rows.next() {
        count += 1;
    }
    count
}

#[test]
fn test_like_with_parameter() {
    let db = setup_db("like_param");
    assert_eq!(
        count_rows(
            &db,
            "SELECT * FROM logs WHERE message LIKE $1",
            ("%hello%",)
        ),
        2
    );
}

#[test]
fn test_like_with_parameter_prefix() {
    let db = setup_db("like_param_prefix");
    assert_eq!(
        count_rows(&db, "SELECT * FROM logs WHERE message LIKE $1", ("hello%",)),
        2
    );
}

#[test]
fn test_like_with_parameter_suffix() {
    let db = setup_db("like_param_suffix");
    assert_eq!(
        count_rows(&db, "SELECT * FROM logs WHERE message LIKE $1", ("%world",)),
        1
    );
}

#[test]
fn test_like_with_parameter_no_match() {
    let db = setup_db("like_param_no_match");
    assert_eq!(
        count_rows(
            &db,
            "SELECT * FROM logs WHERE message LIKE $1",
            ("nonexistent%",)
        ),
        0
    );
}

#[test]
fn test_ilike_with_parameter() {
    let db = setup_db("ilike_param");
    // ILIKE is case-insensitive, so "hello%" should match rows 1, 3, and 4
    assert_eq!(
        count_rows(
            &db,
            "SELECT * FROM logs WHERE message ILIKE $1",
            ("hello%",)
        ),
        3
    );
}

#[test]
fn test_not_like_with_parameter() {
    let db = setup_db("not_like_param");
    // NOT LIKE "%hello%" excludes rows 1 and 3, leaves 2 and 4
    assert_eq!(
        count_rows(
            &db,
            "SELECT * FROM logs WHERE message NOT LIKE $1",
            ("%hello%",)
        ),
        2
    );
}

#[test]
fn test_not_ilike_with_parameter() {
    let db = setup_db("not_ilike_param");
    // NOT ILIKE "hello%" excludes rows 1, 3, and 4, leaves only row 2
    assert_eq!(
        count_rows(
            &db,
            "SELECT * FROM logs WHERE message NOT ILIKE $1",
            ("hello%",)
        ),
        1
    );
}

#[test]
fn test_like_with_parameter_and_order_by() {
    let db = setup_db("like_param_order");
    let mut rows = db
        .query(
            "SELECT id FROM logs WHERE message LIKE $1 ORDER BY id DESC",
            ("%hello%",),
        )
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    assert_eq!(ids, vec![3, 1]);
}

#[test]
fn test_like_with_parameter_exact() {
    let db = setup_db("like_param_exact");
    assert_eq!(
        count_rows(
            &db,
            "SELECT * FROM logs WHERE message LIKE $1",
            ("hello world",)
        ),
        1
    );
}

#[test]
fn test_like_with_parameter_underscore() {
    let db = setup_db("like_param_underscore");
    // _ matches exactly one character
    assert_eq!(
        count_rows(
            &db,
            "SELECT * FROM logs WHERE message LIKE $1",
            ("hello_world",)
        ),
        1
    );
}

#[test]
fn test_like_literal_still_works() {
    let db = setup_db("like_literal");
    // Ensure the static pattern path still works
    assert_eq!(
        count_rows(&db, "SELECT * FROM logs WHERE message LIKE '%hello%'", ()),
        2
    );
}

#[test]
fn test_like_and_glob_dynamic_no_cache_collision() {
    // P1 regression: LIKE and GLOB must use separate caches.
    // '*' in LIKE is a literal char, but '*' in GLOB means "match anything".
    let db = Database::open("memory://like_glob_cache").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'hello')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, '*')", ()).unwrap();

    // LIKE '*' matches only literal '*' (* is not a LIKE wildcard)
    assert_eq!(
        count_rows(&db, "SELECT * FROM t WHERE val LIKE $1", ("*",)),
        1
    );
    // GLOB '*' matches everything (* is a GLOB wildcard)
    assert_eq!(
        count_rows(&db, "SELECT * FROM t WHERE val GLOB $1", ("*",)),
        2
    );
}

#[test]
fn test_like_dynamic_with_escape() {
    // P2 regression: parameterized LIKE with ESCAPE must process the escape character.
    let db = Database::open("memory://like_dyn_escape").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'hello%world')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, 'helloXworld')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, '100% done')", ())
        .unwrap();

    // The escape char '!' makes !% a literal %. Pattern: %!%% = anything + literal% + anything
    let mut rows = db
        .query("SELECT id FROM t WHERE val LIKE $1 ESCAPE '!'", ("%!%%",))
        .unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    ids.sort();
    assert_eq!(
        ids,
        vec![1, 3],
        "Dynamic LIKE with ESCAPE should match literal %"
    );
}

#[test]
fn test_regexp_dynamic_invalid_pattern_errors() {
    // P2 regression: invalid parameterized REGEXP must return an error, not NULL.
    let db = Database::open("memory://regexp_dyn_err").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'hello')", ()).unwrap();

    // Invalid regex pattern should produce an error (may surface during query or row iteration)
    let result = db.query("SELECT * FROM t WHERE val REGEXP $1", ("[invalid",));
    let has_error = match result {
        Err(_) => true,
        Ok(rows) => {
            let mut found_err = false;
            for row_result in rows {
                if row_result.is_err() {
                    found_err = true;
                    break;
                }
            }
            found_err
        }
    };
    assert!(has_error, "Invalid dynamic REGEXP should return an error");
}

#[test]
fn test_not_regexp_dynamic_invalid_pattern_errors() {
    // Verifies NOT REGEXP with invalid param also errors (not false positive)
    let db = Database::open("memory://not_regexp_dyn_err").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'hello')", ()).unwrap();

    let result = db.query("SELECT * FROM t WHERE val NOT REGEXP $1", ("[invalid",));
    let has_error = match result {
        Err(_) => true,
        Ok(rows) => {
            let mut found_err = false;
            for row_result in rows {
                if row_result.is_err() {
                    found_err = true;
                    break;
                }
            }
            found_err
        }
    };
    assert!(
        has_error,
        "Invalid dynamic NOT REGEXP should return an error, not match all rows"
    );
}

#[test]
fn test_regexp_dynamic_error_with_order_by() {
    // Regression: ORDER BY forces materialized path (non-streaming filter).
    // Invalid REGEXP must still surface errors, not silently return zero rows.
    let db = Database::open("memory://regexp_dyn_order_err").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'hello')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, 'world')", ()).unwrap();

    let result = db.query(
        "SELECT * FROM t WHERE val REGEXP $1 ORDER BY id",
        ("[invalid",),
    );
    let has_error = match result {
        Err(_) => true,
        Ok(rows) => {
            let mut found_err = false;
            for row_result in rows {
                if row_result.is_err() {
                    found_err = true;
                    break;
                }
            }
            found_err
        }
    };
    assert!(
        has_error,
        "Invalid dynamic REGEXP with ORDER BY should surface error, not return empty results"
    );
}

#[test]
fn test_like_escape_cache_distinguishes_escape_char() {
    // Regression: LIKE cache key must include escape char semantics.
    // Without it, LIKE $1 and LIKE $1 ESCAPE '!' could reuse wrong compiled pattern.
    let db = Database::open("memory://like_esc_cache").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '100%')", ()).unwrap();
    db.execute("INSERT INTO t VALUES (2, '100X')", ()).unwrap();

    // Without escape: % is a wildcard, so '100%' matches both rows
    assert_eq!(
        count_rows(&db, "SELECT * FROM t WHERE val LIKE $1", ("100%",)),
        2
    );
    // With escape '!': !% is literal %, so '100!%' matches only '100%'
    assert_eq!(
        count_rows(
            &db,
            "SELECT * FROM t WHERE val LIKE $1 ESCAPE '!'",
            ("100!%",)
        ),
        1
    );
}
