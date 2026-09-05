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

//! ON CONFLICT ... DO UPDATE SET ... WHERE updates the row met only where
//! the condition holds, read against that row and the EXCLUDED row.

use stoolap::Database;

fn rows(db: &Database) -> Vec<(i64, Option<i64>)> {
    db.query("SELECT k, v FROM u ORDER BY k", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<Option<i64>>(1).unwrap())
        })
        .collect()
}

#[test]
fn test_do_update_where_holds_back_the_rows_it_does_not_cover() {
    let db = Database::open("memory://upsert_where").unwrap();
    db.execute("CREATE TABLE u (k INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (3, 30), (4, 40)", ())
        .unwrap();

    db.execute(
        "INSERT INTO u (k, v) VALUES (3, 1), (4, 1), (5, 1) ON CONFLICT (k) DO UPDATE SET v = excluded.v WHERE u.v > 35",
        (),
    )
    .unwrap();
    assert_eq!(rows(&db), [(3, Some(30)), (4, Some(1)), (5, Some(1))]);

    db.execute(
        "INSERT INTO u (k, v) VALUES (3, 7), (4, 7) ON CONFLICT (k) DO UPDATE SET v = excluded.v WHERE excluded.v > v",
        (),
    )
    .unwrap();
    assert_eq!(rows(&db), [(3, Some(30)), (4, Some(7)), (5, Some(1))]);

    db.execute(
        "INSERT INTO u (k, v) VALUES (3, 9) ON CONFLICT (k) DO UPDATE SET v = u.v + excluded.v WHERE u.v < 100",
        (),
    )
    .unwrap();
    assert_eq!(rows(&db), [(3, Some(39)), (4, Some(7)), (5, Some(1))]);
}
