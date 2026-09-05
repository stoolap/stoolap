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

//! Multiplying a column by zero is NULL for a NULL column value, so the
//! simplifier leaves `x * 0` alone instead of folding it to 0.

use stoolap::Database;

#[test]
fn test_a_null_times_zero_stays_null() {
    let db = Database::open("memory://null_times_zero").unwrap();
    db.execute("CREATE TABLE hx (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO hx VALUES (1, 1), (2, 2), (3, NULL)", ())
        .unwrap();
    let count = |sql: &str| -> i64 {
        db.query(sql, ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap()
    };
    assert_eq!(count("SELECT COUNT(*) FROM hx WHERE a * 0 IS NOT NULL"), 2);
    assert_eq!(count("SELECT COUNT(*) FROM hx WHERE 0 * a IS NULL"), 1);
    assert_eq!(count("SELECT COUNT(a * 0) FROM hx"), 2);
    let values: Vec<Option<i64>> = db
        .query("SELECT a * 0 FROM hx ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get::<Option<i64>>(0).unwrap())
        .collect();
    assert_eq!(values, [Some(0), Some(0), None]);
}
