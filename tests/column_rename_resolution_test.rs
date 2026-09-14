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

//! Every read path that names a column of a sealed volume resolves it
//! through the volume's column mapping, so a column renamed since the
//! seal, once or in a cycle, is the column the schema means and not the
//! volume column that used to carry the name.

use stoolap::Database;

/// 70,000 sealed rows with x = 100000 + id and y = 70001 - id, reopened,
/// then x renamed to z and y to x: logical x is the old y
fn swapped(dir: &std::path::Path) -> Database {
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE q (id INTEGER PRIMARY KEY, x INTEGER NOT NULL, y INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO q SELECT g.value, 100000 + g.value, 70001 - g.value FROM generate_series(1, 70000) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN x TO z", ())
        .unwrap();
    db.execute("ALTER TABLE q RENAME COLUMN y TO x", ())
        .unwrap();
    db
}

#[test]
fn an_ordered_limit_takes_its_bounds_from_the_renamed_column() {
    let dir = tempfile::tempdir().unwrap();
    let db = swapped(dir.path());
    let row = db
        .query("SELECT id, x FROM q ORDER BY x ASC LIMIT 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(
        (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()),
        (70_000, 1),
        "the order bound came from the old x column"
    );
    let row = db
        .query("SELECT id, x FROM q ORDER BY x DESC LIMIT 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(
        (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()),
        (1, 70_000)
    );
}

#[test]
fn the_min_max_shortcut_reads_the_renamed_column() {
    let dir = tempfile::tempdir().unwrap();
    let db = swapped(dir.path());
    // Local writes and a LIMIT steer the aggregate onto the index shortcut
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO q (id, z, x) VALUES (70001, 500000, 80000)", ())
        .unwrap();
    let min: i64 = tx.query_one("SELECT MIN(x) FROM q LIMIT 1", ()).unwrap();
    let max: i64 = tx.query_one("SELECT MAX(x) FROM q LIMIT 1", ()).unwrap();
    tx.rollback().unwrap();
    assert_eq!(
        (min, max),
        (1, 80_000),
        "the extrema came from the old x column"
    );
}

#[test]
fn an_anti_join_dictionary_filter_reads_the_renamed_column() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO p VALUES (1)", ()).unwrap();
        db.execute(
            "CREATE TABLE r (id INTEGER PRIMARY KEY, x TEXT NOT NULL, y TEXT NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO r SELECT g.value, 'old', 'needle' FROM generate_series(1, 1000) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.execute("ALTER TABLE r RENAME COLUMN x TO z", ())
        .unwrap();
    db.execute("ALTER TABLE r RENAME COLUMN y TO x", ())
        .unwrap();
    let rows = db
        .query(
            "SELECT p.id FROM p WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.id = p.id AND r.x = 'needle')",
            (),
        )
        .unwrap()
        .count();
    assert_eq!(
        rows, 0,
        "the dictionary of the old x column answered for the new one"
    );
    let rows = db
        .query(
            "SELECT p.id FROM p WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.id = p.id AND r.x = 'old')",
            (),
        )
        .unwrap()
        .count();
    assert_eq!(rows, 1);
}

#[test]
fn a_cycle_of_renames_is_followed_when_the_mapping_is_computed_again() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE c (id INTEGER PRIMARY KEY, x INTEGER NOT NULL, y INTEGER NOT NULL, z INTEGER NOT NULL)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO c SELECT g.value, 1, 2, 3 FROM generate_series(1, 1000) g",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "ALTER TABLE c RENAME COLUMN x TO t",
        "ALTER TABLE c RENAME COLUMN y TO x",
        "ALTER TABLE c RENAME COLUMN z TO y",
        "ALTER TABLE c RENAME COLUMN t TO z",
        // A new column has the mapping computed again from the whole history
        "ALTER TABLE c ADD COLUMN marker INTEGER DEFAULT 0",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let row = db
        .query("SELECT x, y, z, marker FROM c WHERE id = 7", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let values: Vec<Option<i64>> = (0..4).map(|i| row.get::<Option<i64>>(i).unwrap()).collect();
    assert_eq!(values, vec![Some(2), Some(3), Some(1), Some(0)]);
    let rows = db
        .query("SELECT id FROM c WHERE x = 2 AND y = 3 AND z = 1", ())
        .unwrap()
        .count();
    assert_eq!(rows, 1_000);
}

/// The anti-join hands the inner table its non-correlated filter as
/// written; the cold collector must prepare it for the schema, or it
/// rejects every sealed row and the outer rows all pass
#[test]
fn an_anti_join_filter_reaches_the_sealed_rows_of_the_inner_table() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1), (2000)", ()).unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, x TEXT NOT NULL)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO r SELECT g.value, 'old' FROM generate_series(1, 1000) g",
        (),
    )
    .unwrap();
    // A plain projection without ORDER BY or LIMIT: the shape that goes
    // through the anti-join
    let sql =
        "SELECT p.id FROM p WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.id = p.id AND r.x = 'old')";
    let ids = |db: &Database| -> Vec<i64> {
        let mut ids: Vec<i64> = db
            .query(sql, ())
            .unwrap()
            .map(|r| r.unwrap().get::<i64>(0).unwrap())
            .collect();
        ids.sort_unstable();
        ids
    };
    assert_eq!(ids(&db), vec![2000]);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        ids(&db),
        vec![2000],
        "the sealed inner rows were all rejected"
    );
}
