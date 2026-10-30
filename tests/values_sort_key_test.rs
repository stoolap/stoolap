// Copyright 2026 Stoolap Contributors
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

//! An ORDER BY or DISTINCT ON key over a VALUES list reads a column the
//! SELECT list does not hold, as over any other source

use stoolap::Database;

fn values(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    rows.map(|row| row.map(|row| row.get::<i64>(0).unwrap()))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())
}

#[test]
fn a_sort_key_over_values_reads_a_column_the_select_list_does_not_hold() {
    let db = Database::open("memory://values_sort_key_unselected").unwrap();
    for (sql, want) in [
        (
            "SELECT x FROM (VALUES (5, 2), (6, 1), (7, 3)) AS v(x, y) ORDER BY y",
            vec![6, 5, 7],
        ),
        (
            "SELECT x FROM (VALUES (5, 2), (6, 1), (7, 3)) AS v(x, y) ORDER BY y DESC LIMIT 2",
            vec![7, 5],
        ),
        (
            "SELECT DISTINCT ON (y) x FROM (VALUES (5, 1), (6, 1), (7, 2)) AS v(x, y) ORDER BY y, x",
            vec![5, 7],
        ),
    ] {
        assert_eq!(values(&db, sql), Ok(want.clone()), "{sql}");
        let derived = sql.replace(
            "(VALUES (5, 2), (6, 1), (7, 3)) AS v(x, y)",
            "(SELECT 5 AS x, 2 AS y UNION ALL SELECT 6, 1 UNION ALL SELECT 7, 3) AS v",
        );
        if derived != sql {
            assert_eq!(values(&db, &derived), Ok(want), "{derived}");
        }
    }
}

#[test]
fn a_sort_key_over_values_naming_two_columns_is_ambiguous() {
    let db = Database::open("memory://values_sort_key_ambiguous").unwrap();
    for sql in [
        "SELECT 1 FROM (VALUES (5, 6)) AS v(x, x) ORDER BY x",
        "SELECT DISTINCT ON (x) 1 FROM (VALUES (5, 6)) AS v(x, x)",
    ] {
        let err = values(&db, sql).expect_err(sql);
        assert!(err.contains("ambiguous"), "{sql}: {err}");
    }
}
