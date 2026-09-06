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

//! A volatile function in the WHERE is judged once per row, however the
//! rest of the WHERE is split between storage and memory.

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .next()
        .unwrap()
}

#[test]
fn test_random_beside_a_subquery_rolls_once() {
    let db = Database::open("memory://volatile_filter").unwrap();
    db.execute("CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    let rows: Vec<String> = (1..=20000).map(|i| format!("({i}, {})", i % 2)).collect();
    for chunk in rows.chunks(1000) {
        db.execute(&format!("INSERT INTO x VALUES {}", chunk.join(", ")), ())
            .unwrap();
    }
    // Half the rows pass, give or take a few hundred; rolled twice, a quarter would
    for sql in [
        "SELECT COUNT(*) FROM x WHERE RANDOM() < 0.5 AND EXISTS (SELECT 1)",
        "SELECT COUNT(*) FROM x WHERE RANDOM() < 0.5 AND a >= 0 AND (SELECT 1) = 1",
        "SELECT COUNT(*) FROM x WHERE RANDOM() < 0.5 AND a >= 0",
        "SELECT COUNT(*) FROM x WHERE a = 1 + (CASE WHEN RANDOM() < 0.5 THEN -1 ELSE 0 END) \
         AND EXISTS (SELECT 1)",
        "SELECT COUNT(*) FROM x WHERE RANDOM() BETWEEN 0 AND 0.5 AND EXISTS (SELECT 1)",
    ] {
        let kept = count(&db, sql);
        assert!((8500..=11500).contains(&kept), "{sql} kept {kept}");
    }
}
