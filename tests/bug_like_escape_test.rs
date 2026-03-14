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

//! Regression test for LIKE ESCAPE with backslash-escaped wildcards.
//!
//! The LIKE operator supports backslash escaping: \% matches literal '%',
//! \_ matches literal '_', \\ matches literal '\'.
//! Previously, the pushdown path in LikeExpr::compile_pattern treated
//! backslash as a literal character, not as an escape prefix.

use stoolap::Database;

fn setup_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'hello%world')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, 'helloXworld')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, '100% done')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (4, '100X done')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (5, 'under_score')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (6, 'underXscore')", ())
        .unwrap();
    db.execute(r"INSERT INTO t VALUES (7, 'back\slash')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (8, 'backXslash')", ())
        .unwrap();
    db
}

fn query_ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut rows = db.query(sql, ()).unwrap();
    let mut ids = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        ids.push(row.get::<i64>(0).unwrap());
    }
    ids.sort();
    ids
}

#[test]
fn test_like_escape_literal_percent() {
    let db = setup_db("like_escape_pct");
    // \% should match literal '%', not act as wildcard
    let ids = query_ids(&db, r"SELECT id FROM t WHERE val LIKE '%\%%'");
    assert_eq!(ids, vec![1, 3], "Should match rows with literal '%'");
}

#[test]
fn test_like_escape_literal_underscore() {
    let db = setup_db("like_escape_us");
    // \_ should match literal '_', not act as single-char wildcard
    let ids = query_ids(&db, r"SELECT id FROM t WHERE val LIKE '%\_%'");
    assert_eq!(ids, vec![5], "Should match rows with literal '_'");
}

#[test]
fn test_like_escape_literal_backslash() {
    let db = setup_db("like_escape_bs");
    // To match literal '\' in LIKE: need \\\\ in SQL string (lexer: \\→\, \\→\; parser: \\→\; LIKE: \\→\)
    let ids = query_ids(&db, r"SELECT id FROM t WHERE val LIKE '%\\\\%'");
    assert_eq!(ids, vec![7], "Should match rows with literal backslash");
}

#[test]
fn test_like_escape_no_false_positives() {
    let db = setup_db("like_escape_nfp");
    // Without escape, % is a wildcard matching everything
    let ids = query_ids(&db, "SELECT id FROM t WHERE val LIKE '%world'");
    assert_eq!(ids, vec![1, 2], "Without escape, % matches anything");
}

#[test]
fn test_like_escape_combined() {
    let db = setup_db("like_escape_comb");
    // Pattern: starts with anything, has literal %, then anything
    // Only rows 1 (hello%world) and 3 (100% done) have literal %
    let ids = query_ids(&db, r"SELECT id FROM t WHERE val LIKE '%\%%'");
    assert_eq!(ids, vec![1, 3]);
}

#[test]
fn test_like_escape_with_explicit_escape_clause() {
    let db = setup_db("like_escape_clause");
    // ESCAPE clause with custom escape char
    let ids = query_ids(&db, "SELECT id FROM t WHERE val LIKE '%!%%' ESCAPE '!'");
    assert_eq!(ids, vec![1, 3], "ESCAPE '!' should treat !% as literal '%'");
}

#[test]
fn test_like_escape_underscore_with_escape_clause() {
    let db = setup_db("like_escape_us_clause");
    let ids = query_ids(&db, "SELECT id FROM t WHERE val LIKE '%!_%' ESCAPE '!'");
    assert_eq!(ids, vec![5], "ESCAPE '!' should treat !_ as literal '_'");
}
