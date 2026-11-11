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

//! The primary key of a sealed row is read from the volume's row ids,
//! without decoding its column

use stoolap::Database;

#[cfg(feature = "test-failpoints")]
#[test]
fn a_sealed_row_read_by_its_rows_decodes_no_block_of_its_primary_key() {
    use stoolap::test_failpoints::take_group_column_decodes;
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE x (id INTEGER PRIMARY KEY, k INTEGER, p TEXT)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX x_k ON x(k) USING BTREE", ())
            .unwrap();
        db.execute(
            "INSERT INTO x SELECT value, (value * 7919) % 200000, 'p' || value \
             FROM generate_series(1, 200000)",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    let cases: [(&str, &[(i64, i64)]); 4] = [
        (
            "SELECT id, k FROM x WHERE k > 100 ORDER BY k LIMIT 2",
            &[(185579, 101), (3258, 102)],
        ),
        ("SELECT id, k FROM x WHERE id = 17679", &[(17679, 1)]),
        ("SELECT id, k FROM x WHERE k = 1", &[(17679, 1)]),
        (
            "SELECT id, k FROM x WHERE k >= 101 AND k <= 102 ORDER BY id",
            &[(3258, 102), (185579, 101)],
        ),
    ];
    let mut decoded_key = Vec::new();
    for (sql, want) in cases {
        let db = Database::open(&dsn).unwrap();
        take_group_column_decodes();
        let rows: Vec<(i64, i64)> = db
            .query(sql, ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (row.get(0).unwrap(), row.get(1).unwrap())
            })
            .collect();
        let decodes = take_group_column_decodes();
        db.close().unwrap();
        assert_eq!(rows, want, "{sql}");
        if decodes.iter().any(|(c, _)| *c == 0) {
            decoded_key.push(sql);
        }
    }
    assert!(decoded_key.is_empty(), "decoded the key: {decoded_key:?}");
}

#[test]
fn a_sealed_primary_key_renamed_and_moved_by_column_changes_reads_its_values() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE y (a TEXT, id INTEGER PRIMARY KEY, k INTEGER)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX y_k ON y(k) USING BTREE", ())
            .unwrap();
        db.execute(
            "INSERT INTO y SELECT 'a' || value, value, (value * 7919) % 200000 \
             FROM generate_series(1, 200000)",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("ALTER TABLE y RENAME COLUMN id TO uid", ())
            .unwrap();
        db.execute("ALTER TABLE y ADD COLUMN z INTEGER DEFAULT 5", ())
            .unwrap();
        db.execute("ALTER TABLE y DROP COLUMN a", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    type Rows = &'static [(i64, i64, i64)];
    let cases: [(&str, Rows); 3] = [
        (
            "SELECT uid, k, z FROM y WHERE k > 100 ORDER BY k LIMIT 2",
            &[(185579, 101, 5), (3258, 102, 5)],
        ),
        (
            "SELECT uid, k, z FROM y WHERE uid = 17679",
            &[(17679, 1, 5)],
        ),
        ("SELECT * FROM y WHERE uid = 3258", &[(3258, 102, 5)]),
    ];
    for (sql, want) in cases {
        let rows: Vec<(i64, i64, i64)> = db
            .query(sql, ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (
                    row.get(0).unwrap(),
                    row.get(1).unwrap(),
                    row.get(2).unwrap(),
                )
            })
            .collect();
        assert_eq!(rows, want, "{sql}");
    }
}
