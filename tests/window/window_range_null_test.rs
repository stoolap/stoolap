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

//! A RANGE offset is measured against the sort key, and nothing can be
//! measured against a NULL. A row holding one sees the rows that hold one
//! too, and no others.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE w (id INTEGER, k INTEGER, v INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO w VALUES (1,10,1), (2,20,2), (3,NULL,3), (4,NULL,4), (5,30,5)",
        (),
    )
    .unwrap();
    db
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            row.get::<Option<String>>(0)
                .unwrap()
                .unwrap_or_else(|| "NULL".into())
        })
        .collect()
}

#[test]
fn test_a_null_key_sees_only_the_rows_that_hold_one() {
    let db = setup("window_range_null_both");
    assert_eq!(
        rows(
            &db,
            "SELECT SUM(v) OVER (ORDER BY k RANGE BETWEEN 10 PRECEDING AND 10 FOLLOWING) FROM w ORDER BY id"
        ),
        ["3", "8", "7", "7", "7"],
        "the two rows holding a NULL see each other and nothing else"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*) OVER (ORDER BY k RANGE BETWEEN 5 PRECEDING AND 5 FOLLOWING) FROM w ORDER BY id"
        ),
        ["1", "1", "2", "2", "1"]
    );
}

#[test]
fn test_an_offset_paired_with_the_current_row() {
    let db = setup("window_range_null_current");
    assert_eq!(
        rows(
            &db,
            "SELECT SUM(v) OVER (ORDER BY k RANGE BETWEEN 10 PRECEDING AND CURRENT ROW) FROM w ORDER BY id"
        ),
        ["1", "3", "7", "7", "7"]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT SUM(v) OVER (ORDER BY k RANGE BETWEEN CURRENT ROW AND 10 FOLLOWING) FROM w ORDER BY id"
        ),
        ["3", "7", "7", "7", "5"]
    );
}

#[test]
fn test_a_row_offset_still_counts_rows() {
    let db = setup("window_range_null_rows");
    assert_eq!(
        rows(
            &db,
            "SELECT SUM(v) OVER (ORDER BY k ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM w ORDER BY id"
        ),
        ["3", "8", "12", "7", "10"],
        "ROWS counts neighbours, so a NULL key changes nothing"
    );
}
