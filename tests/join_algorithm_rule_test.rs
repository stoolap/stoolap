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

//! A join returns every matching pair whatever the order of its inputs: no
//! algorithm is chosen on an order the rows were only sampled for

use stoolap::Database;

const ROWS: i64 = 60_000;

/// Tables a and b hold keys 1..=ROWS in id order, except that a swaps two
/// neighbouring keys in the middle, where a sampled order check never looks
fn nearly_sorted(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for table in ["a", "b"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER)"),
            (),
        )
        .unwrap();
        for start in (1..=ROWS).step_by(1000) {
            let values: Vec<String> = (start..start + 1000)
                .map(|id| {
                    let k = match (table, id) {
                        ("a", 30_001) => 30_002,
                        ("a", 30_002) => 30_001,
                        _ => id,
                    };
                    format!("({id}, {k})")
                })
                .collect();
            db.execute(
                &format!("INSERT INTO {table} VALUES {}", values.join(", ")),
                (),
            )
            .unwrap();
        }
    }
    db
}

#[test]
fn a_join_of_nearly_sorted_inputs_keeps_every_match() {
    let db = nearly_sorted("join_rule_nearly_sorted");
    let mut wrong = Vec::new();
    for sql in [
        "SELECT COUNT(*) FROM a JOIN b ON a.k = b.k",
        "SELECT COUNT(*) FROM a LEFT JOIN b ON a.k = b.k WHERE b.k IS NOT NULL",
        "SELECT COUNT(*) FROM (SELECT k FROM a ORDER BY id) x JOIN (SELECT k FROM b ORDER BY id) y ON x.k = y.k",
        "WITH x(k) AS (SELECT k FROM a ORDER BY id), y(k) AS (SELECT k FROM b ORDER BY id) \
         SELECT COUNT(*) FROM x JOIN y ON x.k = y.k",
    ] {
        let count: i64 = db.query_one(sql, ()).unwrap();
        if count != ROWS {
            wrong.push(format!("{sql}: {count} rows"));
        }
    }
    assert!(wrong.is_empty(), "expected {ROWS}:\n{}", wrong.join("\n"));
}
