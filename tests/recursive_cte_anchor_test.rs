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

//! A branch of a recursive CTE that does not read the CTE runs once, with
//! the first; only a branch that reads it is run again on each pass

use stoolap::Database;

fn values(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

#[test]
fn a_branch_that_does_not_read_the_cte_runs_once() {
    let db = Database::open("memory://recursive_cte_anchor").unwrap();
    db.execute("CREATE TABLE seeds (v INTEGER, s INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO seeds VALUES (50, 100), (60, 200)", ())
        .unwrap();
    for (sql, want) in [
        (
            "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT 100 \
             UNION ALL SELECT n + 1 FROM s WHERE n < 3) SELECT n FROM s ORDER BY n",
            vec![1, 2, 3, 100],
        ),
        (
            "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM s WHERE n < 3 \
             UNION ALL SELECT 100) SELECT n FROM s ORDER BY n",
            vec![1, 2, 3, 100],
        ),
        (
            "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT v FROM seeds \
             UNION ALL SELECT n + 1 FROM s WHERE n < 3 OR n BETWEEN 50 AND 51) \
             SELECT n FROM s ORDER BY n",
            vec![1, 2, 3, 50, 51, 52, 60],
        ),
        (
            "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT 10 \
             UNION ALL SELECT n + 1 FROM s WHERE n < 2 \
             UNION ALL SELECT n + 100 FROM s WHERE n BETWEEN 10 AND 11) \
             SELECT n FROM s ORDER BY n",
            vec![1, 2, 10, 110],
        ),
        (
            "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT 7 \
             UNION ALL SELECT (SELECT MAX(n) FROM s) + 0 FROM seeds WHERE v = 50 AND 0 = 1) \
             SELECT COUNT(*) FROM s",
            vec![2],
        ),
        // A column named like the CTE is not a read of it
        (
            "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT s FROM seeds WHERE v = 50 \
             UNION ALL SELECT n + 1 FROM s WHERE n < 3) SELECT n FROM s ORDER BY n",
            vec![1, 2, 3, 100],
        ),
        (
            "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT MAX(s) - 199 FROM seeds \
             UNION ALL SELECT n + 1 FROM s WHERE n < 3) SELECT n FROM s ORDER BY n",
            vec![1, 1, 2, 2, 3, 3],
        ),
    ] {
        assert_eq!(values(&db, sql), want, "{sql}");
    }
}
