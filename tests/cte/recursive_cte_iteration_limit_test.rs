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

//! A recursive CTE still producing rows after 10,000 passes fails; one that
//! finishes within them answers in full

use stoolap::Database;

fn count_and_max(db: &Database, last: i64) -> Result<(i64, i64), String> {
    let sql = format!(
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < {last}) \
         SELECT COUNT(*), MAX(n) FROM r"
    );
    let mut rows = db.query(&sql, ()).map_err(|e| e.to_string())?;
    let row = rows.next().unwrap().map_err(|e| e.to_string())?;
    Ok((row.get(0).unwrap(), row.get(1).unwrap()))
}

#[test]
fn a_recursion_past_the_limit_fails_instead_of_returning_a_prefix() {
    let db = Database::open("memory://recursive_cte_past_limit").unwrap();
    for last in [10_002, 20_000] {
        let err = count_and_max(&db, last).unwrap_err();
        assert!(
            err.contains("recursive CTE 'r'") && err.contains("10000 iterations"),
            "n < {last}: {err}"
        );
    }
}

#[test]
fn a_recursion_ending_at_the_limit_answers_in_full() {
    let db = Database::open("memory://recursive_cte_at_limit").unwrap();
    assert_eq!(count_and_max(&db, 10_001), Ok((10_001, 10_001)));
    assert_eq!(count_and_max(&db, 3), Ok((3, 3)));
}
