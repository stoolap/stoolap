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

//! Under WITH RECURSIVE only a CTE whose body reads its own name recurses;
//! every other member is an ordinary CTE

use stoolap::Database;

fn values(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    db.query(sql, ())
        .and_then(|rows| rows.map(|row| row?.get(0)).collect())
        .map_err(|e| e.to_string())
}

#[test]
fn a_member_that_does_not_read_itself_runs_as_an_ordinary_cte() {
    let db = Database::open("memory://recursive_cte_list_members").unwrap();
    db.execute("CREATE TABLE seeds (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO seeds VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    for (sql, want) in [
        (
            "WITH RECURSIVE base AS (SELECT 1 AS x), \
             r(n) AS (SELECT x FROM base UNION ALL SELECT n + 1 FROM r WHERE n < 3) \
             SELECT * FROM r",
            vec![1, 2, 3],
        ),
        (
            "WITH RECURSIVE base AS (SELECT v FROM seeds WHERE id < 3), \
             r(n) AS (SELECT v FROM base UNION ALL SELECT n + 1 FROM r WHERE n % 10 < 1) \
             SELECT n FROM r ORDER BY n",
            vec![10, 11, 20, 21],
        ),
        (
            "WITH RECURSIVE t AS (SELECT SUM(v) AS s FROM seeds) SELECT s FROM t",
            vec![60],
        ),
        (
            "WITH RECURSIVE t AS (SELECT v FROM seeds ORDER BY v DESC LIMIT 2) \
             SELECT v FROM t ORDER BY v",
            vec![20, 30],
        ),
        (
            "WITH RECURSIVE t(a) AS (SELECT 1 UNION ALL SELECT 2 ORDER BY 1 DESC LIMIT 1) \
             SELECT a FROM t",
            vec![2],
        ),
        (
            "WITH RECURSIVE t(a) AS (SELECT 1 UNION SELECT 1) SELECT a FROM t",
            vec![1],
        ),
    ] {
        assert_eq!(values(&db, sql), Ok(want), "{sql}");
    }

    let prepared = db
        .prepare(
            "WITH RECURSIVE base AS (SELECT $1 AS x), \
             r(n) AS (SELECT x FROM base UNION ALL SELECT n + 1 FROM r WHERE n < $2) \
             SELECT n FROM r",
        )
        .unwrap();
    let got: Vec<i64> = prepared
        .query((5, 7))
        .and_then(|rows| rows.map(|row| row?.get(0)).collect())
        .unwrap();
    assert_eq!(got, vec![5, 6, 7]);
}

#[test]
fn a_member_that_reads_itself_still_needs_union_all() {
    let db = Database::open("memory://recursive_cte_list_members_refusal").unwrap();
    for sql in [
        "WITH RECURSIVE r(n) AS (SELECT n FROM r) SELECT * FROM r",
        "WITH RECURSIVE base AS (SELECT 1 AS x), r(n) AS (SELECT x FROM base, r) SELECT * FROM r",
        "WITH RECURSIVE r(n) AS (SELECT n FROM (WITH q AS (SELECT n FROM r) SELECT n FROM q) s) \
         SELECT * FROM r",
    ] {
        assert_eq!(
            values(&db, sql),
            Err("invalid argument: Recursive CTE must have UNION ALL between anchor and recursive members".to_string()),
            "{sql}"
        );
    }
}

#[test]
fn a_nested_with_that_redefines_the_name_is_not_a_read_of_it() {
    let db = Database::open("memory://recursive_cte_list_members_shadow").unwrap();
    for (sql, want) in [
        (
            "WITH RECURSIVE r(n) AS (SELECT n FROM (WITH r AS (SELECT 5 AS n) SELECT n FROM r) s) \
             SELECT * FROM r",
            vec![5],
        ),
        (
            "WITH RECURSIVE r(n) AS (SELECT (WITH r AS (SELECT 5 AS n) SELECT n FROM r)) \
             SELECT * FROM r",
            vec![5],
        ),
        (
            "WITH RECURSIVE r(n) AS (SELECT n FROM (WITH RECURSIVE r(n) AS \
             (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT n FROM r) s) \
             SELECT n FROM r ORDER BY n",
            vec![1, 2, 3],
        ),
        (
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r \
             WHERE n < (SELECT n FROM (WITH r AS (SELECT 3 AS n) SELECT n FROM r) s)) \
             SELECT n FROM r ORDER BY n",
            vec![1, 2, 3],
        ),
    ] {
        assert_eq!(values(&db, sql), Ok(want), "{sql}");
    }
}

#[test]
fn a_nested_recursive_cte_s_first_member_reads_the_outer_name() {
    let db = Database::open("memory://recursive_cte_list_members_inner_anchor").unwrap();
    let sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n FROM (\
               WITH RECURSIVE r(n) AS (SELECT n + 1 FROM r WHERE n < 3 \
               UNION ALL SELECT n FROM r WHERE FALSE) SELECT n FROM r) s) \
               SELECT n FROM r ORDER BY n";
    assert_eq!(values(&db, sql), Ok(vec![1, 2, 3]), "{sql}");
}

#[test]
fn a_recursive_cte_with_its_own_order_or_cut_is_refused() {
    let db = Database::open("memory://recursive_cte_list_members_order").unwrap();
    for clause in [
        "ORDER BY n DESC",
        "LIMIT 2",
        "OFFSET 1",
        "ORDER BY n LIMIT 2 OFFSET 1",
    ] {
        let sql = format!(
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5 {clause}) \
             SELECT n FROM r"
        );
        assert_eq!(
            values(&db, &sql),
            Err(
                "invalid argument: Recursive CTE 'r' does not support ORDER BY, LIMIT or OFFSET"
                    .to_string()
            ),
            "{sql}"
        );
    }
    let prepared = db
        .prepare(
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r LIMIT $1) \
             SELECT n FROM r",
        )
        .unwrap();
    let err = prepared
        .query((10,))
        .and_then(|rows| {
            rows.map(|row| row?.get::<i64>(0))
                .collect::<Result<Vec<_>, _>>()
        })
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "invalid argument: Recursive CTE 'r' does not support ORDER BY, LIMIT or OFFSET"
    );
}
