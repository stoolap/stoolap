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

//! SUBSTRING opens a window on a string, which may start before it or
//! reach back from its end, and LIKE reads the character after its escape
//! as itself whatever that character is. Every answer here is the one
//! SQLite 3.51 and DuckDB 1.2 both give.

use stoolap::Database;

fn text(db: &Database, sql: &str) -> String {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get::<String>(0)
        .unwrap()
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

#[test]
fn test_substring_window() {
    let db = Database::open("memory://string_boundary_substr").unwrap();
    for (sql, expected) in [
        ("SELECT SUBSTR('abcdef', 2, 3)", "bcd"),
        ("SELECT SUBSTR('abcdef', -3, 2)", "de"),
        ("SELECT SUBSTR('abcdef', -3)", "def"),
        ("SELECT SUBSTR('abcdef', -1, 5)", "f"),
        ("SELECT SUBSTR('abcdef', -10, 3)", ""),
        ("SELECT SUBSTR('abcdef', 0, 2)", "a"),
        ("SELECT SUBSTR('abcdef', 0)", "abcdef"),
        ("SELECT SUBSTR('abcdef', 2, 0)", ""),
        ("SELECT SUBSTR('abcdef', 3, 100)", "cdef"),
        ("SELECT SUBSTR('abcdef', 3, -2)", "ab"),
        ("SELECT SUBSTR('abcdef', 7, 2)", ""),
    ] {
        assert_eq!(text(&db, sql), expected, "{sql}");
    }
}

#[test]
fn test_like_escape_reads_the_next_character_as_itself() {
    let db = Database::open("memory://string_boundary_like").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    for (id, s) in [
        (1i64, "x,y"),
        (2, "a%b"),
        (3, "c_d"),
        (4, "e!f"),
        (5, "plain"),
    ] {
        db.execute("INSERT INTO t VALUES ($1, $2)", (id, s))
            .unwrap();
    }
    for (sql, expected) in [
        ("SELECT COUNT(*) FROM t WHERE s LIKE '%!,%' ESCAPE '!'", 1),
        ("SELECT COUNT(*) FROM t WHERE s LIKE '%!%%' ESCAPE '!'", 1),
        ("SELECT COUNT(*) FROM t WHERE s LIKE '%!_%' ESCAPE '!'", 1),
        ("SELECT COUNT(*) FROM t WHERE s LIKE '%!!%' ESCAPE '!'", 1),
        ("SELECT COUNT(*) FROM t WHERE s LIKE '%_%'", 5),
        ("SELECT COUNT(*) FROM t WHERE s LIKE 'a%b'", 1),
    ] {
        assert_eq!(count(&db, sql), expected, "{sql}");
    }
}

/// A start or a length at the limits of the type runs past an end of the
/// string, which is the same as reaching that end
#[test]
fn test_a_window_at_the_limits_of_the_type() {
    let db = Database::open("memory://string_boundary_limits").unwrap();
    for (sql, expected) in [
        ("SELECT SUBSTR('abc', 1, 9223372036854775807)", "abc"),
        (
            "SELECT '[' || SUBSTR('abc', 9223372036854775807, 2) || ']'",
            "[]",
        ),
        (
            "SELECT '[' || SUBSTR('abc', 3, -9223372036854775808) || ']'",
            "[ab]",
        ),
        ("SELECT SUBSTR('abc', -2, 9223372036854775807)", "bc"),
        (
            "SELECT '[' || SUBSTR('abc', -9223372036854775808, 3) || ']'",
            "[]",
        ),
        (
            "SELECT '[' || SUBSTR('abc', -9223372036854775807, 9223372036854775807) || ']'",
            "[abc]",
        ),
    ] {
        assert_eq!(text(&db, sql), expected, "{sql}");
    }
}

/// There is nothing to find in a string that holds no characters
#[test]
fn test_replacing_a_string_that_holds_nothing() {
    let db = Database::open("memory://string_boundary_replace").unwrap();
    assert_eq!(text(&db, "SELECT REPLACE('abc', '', 'x')"), "abc");
    assert_eq!(text(&db, "SELECT REPLACE('', '', 'x')"), "");
    assert_eq!(text(&db, "SELECT REPLACE('aaa', 'a', '')"), "");
    assert_eq!(text(&db, "SELECT REPLACE('abc', 'b', 'xy')"), "axyc");
}
