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

//! A grouped CTE with a column list, joined under an ORDER BY LIMIT on its
//! columns, answers the rows the whole sort gives

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect()
}

#[test]
fn a_column_list_names_the_keys_a_join_orders_the_cte_by() {
    let db = Database::open("memory://cte_column_list_order_limit").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t SELECT value, value % 17 FROM generate_series(1, 300)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u SELECT value, value * 3 FROM generate_series(0, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let with = "WITH s(k, total) AS (SELECT a, COUNT(*) FROM t GROUP BY a)";
    for (join, order) in [
        ("LEFT JOIN", "s.total DESC, s.k"),
        ("JOIN", "s.total, s.k DESC"),
    ] {
        let whole = format!(
            "{with} SELECT DISTINCT s.k, s.total FROM s {join} u ON u.id = s.k ORDER BY {order}"
        );
        let want: Vec<_> = rows(&db, &whole).into_iter().skip(2).take(4).collect();
        let sql = format!(
            "{with} SELECT s.k, s.total FROM s {join} u ON u.id = s.k \
             ORDER BY {order} LIMIT 4 OFFSET 2"
        );
        assert_eq!(rows(&db, &sql), want, "{sql}");
    }
}
