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

//! A double-quoted "true" or "false" names a column, not a boolean literal.

use stoolap::Database;

fn strings(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{:?}", row.get::<stoolap::Value>(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

#[test]
fn test_quoted_true_and_false_are_columns() {
    let db = Database::open("memory://quoted_boolean_name").unwrap();
    db.execute("CREATE TABLE t (\"true\" INTEGER, \"false\" INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (8, 8)", ()).unwrap();

    for (sql, expected) in [
        ("SELECT \"true\" + 1, \"false\" + 1 FROM t", "SELECT 9, 9"),
        ("SELECT ABS(\"true\") FROM t", "SELECT 8"),
        ("SELECT \"true\" = 8 FROM t", "SELECT TRUE"),
        ("SELECT \"true\" FROM t WHERE \"true\" + 1 = 9", "SELECT 8"),
        ("SELECT \"true\" AS g FROM t GROUP BY g", "SELECT 8"),
    ] {
        assert_eq!(strings(&db, sql), strings(&db, expected), "{}", sql);
    }
}

#[test]
fn test_quoted_name_text_is_not_shared_through_the_program_cache() {
    // A quoted name that is no column reads as its own text, so programs for
    // "ABC" and "abc" must not be reused for each other
    let db = Database::open("memory://quoted_name_cache").unwrap();
    assert_eq!(strings(&db, "SELECT \"ABC\""), ["Text(\"ABC\")"]);
    assert_eq!(strings(&db, "SELECT \"abc\""), ["Text(\"abc\")"]);
    assert_eq!(strings(&db, "SELECT \"TRUE\""), ["Text(\"TRUE\")"]);
    assert_eq!(strings(&db, "SELECT \"true\""), ["Text(\"true\")"]);
}
