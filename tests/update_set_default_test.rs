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

//! SET col = DEFAULT in an UPDATE writes the column's default.

use stoolap::Database;

#[test]
fn test_set_default_writes_the_column_default() {
    let db = Database::open("memory://update_set_default").unwrap();
    db.execute(
        "CREATE TABLE dv (id INTEGER PRIMARY KEY, n INTEGER DEFAULT 7, t TEXT DEFAULT 'x', u TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO dv VALUES (1, 1, 'y', 'z')", ())
        .unwrap();
    let changed = db
        .execute(
            "UPDATE dv SET t = DEFAULT, n = DEFAULT, u = DEFAULT WHERE id = 1",
            (),
        )
        .unwrap();
    assert_eq!(changed, 1);
    let row = db
        .query("SELECT n, t, u FROM dv WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 7);
    assert_eq!(row.get::<String>(1).unwrap(), "x");
    assert_eq!(row.get::<Option<String>>(2).unwrap(), None);
}
