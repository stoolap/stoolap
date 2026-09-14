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

//! CREATE TABLE ... CLUSTER BY: the key is parsed, kept with the schema
//! across a reopen, shown back, and the sealed rows are held in key
//! order while every lookup by id still answers.

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::traits::Engine;
use stoolap::storage::volume::seal::seal_rows;
use stoolap::Database;

const CREATE: &str = "CREATE TABLE ticks (id INTEGER PRIMARY KEY, exchange TEXT NOT NULL, symbol TEXT NOT NULL, time INTEGER NOT NULL, price REAL) CLUSTER BY (exchange, symbol, time)";

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// Rows whose key order differs from their id order: ids ascend while
/// the (exchange, symbol, time) key walks the other way
fn insert_ticks(db: &Database, rows: i64) {
    let insert = db
        .prepare("INSERT INTO ticks VALUES (?, ?, ?, ?, ?)")
        .unwrap();
    for id in 1..=rows {
        let exchange = if id % 2 == 0 { "b" } else { "a" };
        let symbol = if id % 3 == 0 { "y" } else { "x" };
        insert
            .execute((id, exchange, symbol, rows - id, id as f64))
            .unwrap();
    }
}

#[test]
fn a_cluster_by_clause_is_parsed_kept_and_shown() {
    let db = Database::open("memory://cluster_by_decl").unwrap();
    db.execute(CREATE, ()).unwrap();
    let schema = db.engine().get_table_schema("ticks").unwrap();
    assert_eq!(schema.cluster_key, vec![1, 2, 3]);
    let shown: String = db
        .query("SHOW CREATE TABLE ticks", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get::<String>(1)
        .unwrap();
    assert!(
        shown.ends_with("CLUSTER BY (exchange, symbol, time)"),
        "{shown}"
    );
}

#[test]
fn a_cluster_by_column_must_exist_and_appear_once() {
    let db = Database::open("memory://cluster_by_errors").unwrap();
    let missing = db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER) CLUSTER BY (b)",
        (),
    );
    assert!(missing.is_err(), "an unknown column was accepted");
    let twice = db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER) CLUSTER BY (a, a)",
        (),
    );
    assert!(twice.is_err(), "a repeated column was accepted");
    let empty = db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER) CLUSTER BY ()",
        (),
    );
    assert!(empty.is_err(), "an empty key was accepted");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    assert!(db
        .engine()
        .get_table_schema("t")
        .unwrap()
        .cluster_key
        .is_empty());
}

#[test]
fn sealed_rows_of_a_clustered_table_are_held_in_key_order() {
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("k", DataType::Text, false, false)
        .cluster_by(vec![1])
        .build();
    // In key order the ids are 3, 1, 2: the builder keeps that order
    let rows: Vec<(i64, Row)> = [(3, "a"), (1, "b"), (2, "c")]
        .into_iter()
        .map(|(id, k)| {
            (
                id,
                Row::from_values(vec![Value::Integer(id), Value::text(k)]),
            )
        })
        .collect();
    let volume = seal_rows(&schema, &rows).unwrap();
    assert_eq!(volume.row_ids().unwrap(), &[3, 1, 2]);
    for id in 1..=3 {
        let idx = volume.locate(id).unwrap();
        assert_eq!(
            volume.get_row(idx).unwrap().get(0),
            Some(&Value::Integer(id))
        );
    }
}

#[test]
fn a_clustered_table_answers_the_same_after_a_seal_and_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    let rows = 3_000;
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(CREATE, ()).unwrap();
        insert_ticks(&db, rows);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        // A scan without an order walks the volume as sealed: the first row
        // is the first in key order, exchange 'a', symbol 'x', smallest time
        assert_eq!(ids(&db, "SELECT id FROM ticks LIMIT 1"), vec![2999]);
        // Point lookups, an ordered scan, an update and a delete by id all
        // go through the sealed volume, whose rows are in key order now
        assert_eq!(ids(&db, "SELECT id FROM ticks WHERE id = 2"), vec![2]);
        assert_eq!(
            ids(&db, "SELECT id FROM ticks ORDER BY id LIMIT 3"),
            vec![1, 2, 3]
        );
        assert_eq!(
            ids(&db, "SELECT id FROM ticks ORDER BY id DESC LIMIT 2"),
            vec![rows, rows - 1]
        );
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM ticks WHERE exchange = 'a' AND symbol = 'y' ORDER BY time LIMIT 2"
            ),
            vec![rows - 3, rows - 9]
        );
        assert_eq!(
            db.execute("UPDATE ticks SET price = 0 WHERE id = 5", ())
                .unwrap(),
            1
        );
        assert_eq!(db.execute("DELETE FROM ticks WHERE id = 7", ()).unwrap(), 1);
        assert!(ids(&db, "SELECT id FROM ticks WHERE id = 7").is_empty());
        let count: i64 = db.query_one("SELECT COUNT(*) FROM ticks", ()).unwrap();
        assert_eq!(count, rows - 1);
    }
    let db = Database::open(&dsn).unwrap();
    let schema = db.engine().get_table_schema("ticks").unwrap();
    assert_eq!(
        schema.cluster_key,
        vec![1, 2, 3],
        "the key did not survive the reopen"
    );
    assert!(ids(&db, "SELECT id FROM ticks WHERE id = 7").is_empty());
    let price: f64 = db
        .query_one("SELECT price FROM ticks WHERE id = 5", ())
        .unwrap();
    assert_eq!(price, 0.0);
    assert_eq!(
        ids(&db, "SELECT id FROM ticks ORDER BY id LIMIT 3"),
        vec![1, 2, 3]
    );
    let count: i64 = db.query_one("SELECT COUNT(*) FROM ticks", ()).unwrap();
    assert_eq!(count, rows - 1);
}

#[test]
fn compaction_keeps_a_clustered_table_in_key_order() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?compact_threshold=2", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(CREATE, ()).unwrap();
    let insert = db
        .prepare("INSERT INTO ticks VALUES (?, ?, ?, ?, ?)")
        .unwrap();
    // Three sealed volumes, each in its own key order, then a compaction
    // that merges them into one; the merged volume is in key order as a whole
    for round in 0..3i64 {
        for id in round * 1_000 + 1..=(round + 1) * 1_000 {
            let exchange = if id % 2 == 0 { "b" } else { "a" };
            let symbol = if id % 3 == 0 { "y" } else { "x" };
            insert
                .execute((id, exchange, symbol, 3_000 - id, id as f64))
                .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let keys: Vec<(String, String, i64)> = db
        .query("SELECT exchange, symbol, time FROM ticks", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<String>(0).unwrap(),
                r.get::<String>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();
    assert_eq!(keys.len(), 3_000);
    let out_of_order = keys.windows(2).position(|w| w[0] > w[1]);
    assert_eq!(
        out_of_order, None,
        "the merged volume breaks key order at row {out_of_order:?}"
    );
    assert_eq!(ids(&db, "SELECT id FROM ticks WHERE id = 1500"), vec![1500]);
    assert_eq!(
        ids(&db, "SELECT id FROM ticks ORDER BY id DESC LIMIT 1"),
        vec![3_000]
    );
}
