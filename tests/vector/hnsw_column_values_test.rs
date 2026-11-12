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

//! A column whose HNSW index cannot list its values answers GROUP BY,
//! DISTINCT and window partitions from its rows

use stoolap::Database;

#[test]
fn an_hnsw_column_groups_and_partitions_by_its_rows() {
    let db = Database::open("memory://an_hnsw_column_groups_and_partitions_by_its_rows").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))",
        "INSERT INTO t VALUES (1, '[1,0]'), (2, '[0,1]'), (3, '[1,0]')",
        "CREATE INDEX t_v ON t(v) USING HNSW",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for (sql, want) in [
        ("SELECT COUNT(*) FROM (SELECT v FROM t GROUP BY v) AS g", 2),
        ("SELECT COUNT(*) FROM (SELECT DISTINCT v FROM t) AS d", 2),
        (
            "SELECT COUNT(*) FROM (SELECT id, COUNT(*) OVER (PARTITION BY v) AS n FROM t) AS w \
             WHERE n = 2",
            2,
        ),
    ] {
        let got: i64 = db.query_one(sql, ()).unwrap();
        if got != want {
            wrong.push(format!("{sql}: {got}, want {want}"));
        }
    }
    let groups = db.query("SELECT v FROM t GROUP BY v", ()).unwrap().count();
    if groups != 2 {
        wrong.push(format!("SELECT v FROM t GROUP BY v: {groups} rows, want 2"));
    }
    let distinct = db.query("SELECT DISTINCT v FROM t", ()).unwrap().count();
    if distinct != 2 {
        wrong.push(format!("SELECT DISTINCT v FROM t: {distinct} rows, want 2"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
