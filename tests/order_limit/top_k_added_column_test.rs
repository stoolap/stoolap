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

//! A column added after a volume was sealed reads as its default from that
//! volume on every path, including the filtered ORDER BY + LIMIT scan.

use stoolap::Database;

fn column(db: &Database, sql: &str, idx: usize) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get_value(idx).unwrap().to_string())
        .collect()
}

#[test]
fn test_top_k_reads_the_added_column_default_from_an_older_volume() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/added", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE c (t INTEGER NOT NULL, k TEXT NOT NULL)", ())
        .unwrap();
    db.execute(
        "INSERT INTO c VALUES (1, 'k'), (2, 'k'), (3, 'k'), (4, 'j')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("ALTER TABLE c ADD COLUMN note TEXT DEFAULT 'kept'", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (5, 'k', 'new')", ())
        .unwrap();

    // The filtered top-k over the older volume, ASC and DESC, with the full
    // row and with the added column alone
    assert_eq!(
        column(
            &db,
            "SELECT * FROM c WHERE k = 'k' ORDER BY t ASC LIMIT 2",
            2
        ),
        ["kept", "kept"]
    );
    assert_eq!(
        column(
            &db,
            "SELECT note FROM c WHERE k = 'k' ORDER BY t DESC LIMIT 3",
            0
        ),
        ["new", "kept", "kept"]
    );
    // The other scan paths agree
    assert_eq!(
        column(&db, "SELECT note FROM c WHERE k = 'k' ORDER BY t", 0),
        ["kept", "kept", "kept", "new"]
    );
    assert_eq!(
        column(
            &db,
            "SELECT note, COUNT(*) FROM c GROUP BY note ORDER BY note",
            1
        ),
        ["4", "1"]
    );
}
