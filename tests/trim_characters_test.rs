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

//! TRIM, LTRIM and RTRIM take a second argument naming the characters to
//! strip; without it they strip whitespace.

use stoolap::Database;

fn one(db: &Database, sql: &str) -> Option<String> {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get::<Option<String>>(0)
        .unwrap()
}

#[test]
fn test_trim_strips_the_named_characters() {
    let db = Database::open("memory://trim_characters").unwrap();
    assert_eq!(one(&db, "SELECT TRIM('xxaxx', 'x')").as_deref(), Some("a"));
    assert_eq!(one(&db, "SELECT TRIM('xyaxy', 'xy')").as_deref(), Some("a"));
    assert_eq!(one(&db, "SELECT LTRIM('xxa', 'x')").as_deref(), Some("a"));
    assert_eq!(one(&db, "SELECT RTRIM('axx', 'x')").as_deref(), Some("a"));
    assert_eq!(one(&db, "SELECT LTRIM('axx', 'x')").as_deref(), Some("axx"));
    assert_eq!(one(&db, "SELECT TRIM('  a  ')").as_deref(), Some("a"));
    assert_eq!(
        one(&db, "SELECT TRIM('  a  ', 'x')").as_deref(),
        Some("  a  ")
    );
    assert_eq!(one(&db, "SELECT TRIM('a', NULL)"), None);
}
