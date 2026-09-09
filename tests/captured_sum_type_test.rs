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

//! Result types must survive captured hot/cold grouping and SQL wrappers.

use stoolap::{Database, Value};

#[test]
fn grouped_sum_float_type_is_stable_for_views_derived_tables_and_own_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE totals (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO totals VALUES (1, 'plain', 10), (2, 'plain', 20), (3, 'null', NULL), (4, 'overflow', 9223372036854775807), (5, 'overflow', 1)", ()).unwrap();
    db.execute("CREATE VIEW grouped_totals AS SELECT category, SUM(amount) AS total FROM totals GROUP BY category", ()).unwrap();
    for stage in ["hot", "cold", "own"] {
        match stage {
            "cold" => {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            "own" => {
                db.execute("BEGIN", ()).unwrap();
                db.execute("UPDATE totals SET amount = 11 WHERE id = 1", ())
                    .unwrap();
                db.execute("INSERT INTO totals VALUES (6, 'plain', 1)", ())
                    .unwrap();
            }
            _ => (),
        }
        for sql in [
            "SELECT category, SUM(amount) FROM totals GROUP BY category",
            "SELECT category, total FROM grouped_totals",
            "SELECT category, total FROM (SELECT category, SUM(amount) AS total FROM totals GROUP BY category) AS grouped",
        ] {
            let mut count = 0;
            for row in db.query(sql, ()).unwrap() {
                let row = row.unwrap();
                let category = row.get::<String>(0).unwrap();
                let actual = row.get_value(1).unwrap();
                match category.as_str() {
                    "plain" => assert!(matches!(actual, Value::Float(v) if *v == if stage == "own" { 32.0 } else { 30.0 }), "{stage}: {sql}: {actual:?}"),
                    "overflow" => assert!(matches!(actual, Value::Float(v) if *v == 9_223_372_036_854_775_808.0), "{stage}: {sql}: {actual:?}"),
                    "null" => assert!(matches!(actual, Value::Null(_)), "{stage}: {sql}: {actual:?}"),
                    _ => panic!("unexpected group {category}"),
                }
                count += 1;
            }
            assert_eq!(count, 3, "{stage}: {sql}");
        }
        let mut rows = db
            .query(
                "SELECT SUM(amount) FROM totals WHERE category = 'plain'",
                (),
            )
            .unwrap();
        let row = rows.next().unwrap().unwrap();
        assert!(
            matches!(row.get_value(0), Some(Value::Integer(v)) if *v == if stage == "own" { 32 } else { 30 }),
            "filtered global SUM must keep integer narrowing: {stage}"
        );
    }
    db.execute("ROLLBACK", ()).unwrap();
}
