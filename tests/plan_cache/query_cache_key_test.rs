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

//! Queries that differ only in whitespace inside a literal are different
//! queries, and the parsed query cache keeps them apart

use stoolap::Database;

#[test]
fn a_literal_with_more_spaces_is_not_answered_from_the_cache() {
    let db = Database::open("memory://query_cache_key_literal").unwrap();
    let one: String = db.query_one("SELECT 'a b'", ()).unwrap();
    let two: String = db.query_one("SELECT 'a  b'", ()).unwrap();
    assert_eq!(one, "a b");
    assert_eq!(
        two, "a  b",
        "the second literal was read from the first's plan"
    );
}

#[test]
fn a_filter_on_a_literal_with_more_spaces_finds_its_row() {
    let db = Database::open("memory://query_cache_key_filter").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'x  y')", ()).unwrap();
    let narrow: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE s = 'x y'", ())
        .unwrap();
    let wide: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE s = 'x  y'", ())
        .unwrap();
    assert_eq!(narrow, 0);
    assert_eq!(wide, 1, "the row was filtered with the other literal");
}

#[test]
fn surrounding_whitespace_still_shares_a_plan() {
    let db = Database::open("memory://query_cache_key_trim").unwrap();
    let one: i64 = db.query_one("SELECT 7", ()).unwrap();
    let two: i64 = db.query_one("  SELECT 7\n", ()).unwrap();
    assert_eq!((one, two), (7, 7));
}
