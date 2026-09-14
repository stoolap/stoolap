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
    let count: i64 = db.query_one("SELECT COUNT(*) FROM ticks", ()).unwrap();
    assert_eq!(count, 3_000);
    assert_eq!(ids(&db, "SELECT id FROM ticks WHERE id = 1500"), vec![1500]);
    assert_eq!(
        ids(&db, "SELECT id FROM ticks ORDER BY id DESC LIMIT 1"),
        vec![3_000]
    );
}

/// The sealed volumes of `table` under `dir`, each as (row id, key values)
/// in physical order, read from disk
fn sealed_volumes(
    dir: &std::path::Path,
    table: &str,
    key: &[usize],
) -> Vec<Vec<(i64, Vec<Value>)>> {
    use stoolap::storage::volume::io::read_volume_from_disk;
    let mut paths: Vec<_> = std::fs::read_dir(dir.join("volumes").join(table))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vol"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let volume = read_volume_from_disk(path).unwrap();
            let ids = volume.row_ids().unwrap().to_vec();
            ids.iter()
                .enumerate()
                .map(|(i, &id)| {
                    let row = volume.get_row(i).unwrap();
                    (
                        id,
                        key.iter().map(|&c| row.get(c).cloned().unwrap()).collect(),
                    )
                })
                .collect()
        })
        .collect()
}

fn in_key_order(rows: &[(i64, Vec<Value>)]) -> bool {
    rows.windows(2).all(|w| {
        w[0].1
            .iter()
            .zip(&w[1].1)
            .map(|(a, b)| a.compare(b).unwrap())
            .find(|o| *o != std::cmp::Ordering::Equal)
            .unwrap_or(std::cmp::Ordering::Equal)
            != std::cmp::Ordering::Greater
    })
}

#[test]
fn the_sealed_volume_holds_the_rows_in_key_order_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(CREATE, ()).unwrap();
    insert_ticks(&db, 3_000);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let volumes = sealed_volumes(dir.path(), "ticks", &[1, 2, 3]);
    assert_eq!(volumes.len(), 1);
    let rows = &volumes[0];
    assert_eq!(rows.len(), 3_000);
    assert!(in_key_order(rows), "the sealed volume is not in key order");
    // Exchange 'a', symbol 'x', smallest time first: the largest odd id
    // that is not a multiple of three
    assert_eq!(rows[0].0, 2_999);
    assert_ne!(
        rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        (1..=3_000).collect::<Vec<_>>()
    );
}

#[test]
fn the_merged_volume_of_a_compaction_is_in_key_order_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?compact_threshold=2",
        dir.path().display()
    ))
    .unwrap();
    db.execute(CREATE, ()).unwrap();
    let insert = db
        .prepare("INSERT INTO ticks VALUES (?, ?, ?, ?, ?)")
        .unwrap();
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
    let volumes = sealed_volumes(dir.path(), "ticks", &[1, 2, 3]);
    let total: usize = volumes.iter().map(|v| v.len()).sum();
    assert_eq!(total, 3_000);
    assert_eq!(
        volumes.len(),
        1,
        "compaction did not merge the three volumes"
    );
    assert!(
        in_key_order(&volumes[0]),
        "the merged volume is not in key order"
    );
}

#[test]
fn a_key_column_cannot_be_dropped_and_the_key_follows_a_dropped_column() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, k INTEGER, z INTEGER) CLUSTER BY (k)",
            (),
        )
        .unwrap();
        assert!(
            db.execute("ALTER TABLE t DROP COLUMN k", ()).is_err(),
            "a key column was dropped"
        );
        db.execute("ALTER TABLE t DROP COLUMN a", ()).unwrap();
        let schema = db.engine().get_table_schema("t").unwrap();
        assert_eq!(schema.cluster_key, vec![1]);
        assert_eq!(schema.columns[1].name, "k");
        let shown: String = db
            .query("SHOW CREATE TABLE t", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get::<String>(1)
            .unwrap();
        assert!(shown.ends_with("CLUSTER BY (k)"), "{shown}");
    }
    let db = Database::open(&dsn).unwrap();
    let schema = db.engine().get_table_schema("t").unwrap();
    assert_eq!(schema.cluster_key, vec![1]);
    assert_eq!(schema.columns[1].name, "k");
}

#[test]
fn a_snapshot_restore_keeps_the_key() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER) CLUSTER BY (k)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 5), (2, 3)", ())
        .unwrap();
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();
    db.execute("PRAGMA RESTORE", ()).unwrap();
    assert_eq!(
        db.engine().get_table_schema("t").unwrap().cluster_key,
        vec![1]
    );
}

#[test]
fn dropping_a_referenced_parent_keeps_the_child_key() {
    let db = Database::open("memory://cluster_by_parent_drop").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, p_id INTEGER REFERENCES p(id), k INTEGER) CLUSTER BY (k)",
        (),
    )
    .unwrap();
    db.execute("DROP TABLE p", ()).unwrap();
    assert_eq!(
        db.engine().get_table_schema("c").unwrap().cluster_key,
        vec![2]
    );
}

#[test]
fn a_key_column_must_have_an_order() {
    let db = Database::open("memory://cluster_by_unordered_types").unwrap();
    assert!(db
        .execute(
            "CREATE TABLE j (id INTEGER PRIMARY KEY, doc JSON) CLUSTER BY (doc)",
            ()
        )
        .is_err());
    assert!(db
        .execute(
            "CREATE TABLE v (id INTEGER PRIMARY KEY, e VECTOR(3)) CLUSTER BY (e)",
            ()
        )
        .is_err());
}

#[test]
fn a_key_column_holding_more_than_one_kind_orders_by_kind_then_value() {
    use stoolap::storage::volume::seal::cluster_order;
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("k", DataType::Text, false, false)
        .cluster_by(vec![1])
        .build();
    // A column changed to TEXT after integers were written holds both
    let mut rows: Vec<(i64, Row)> = [
        (1, Value::text("15")),
        (2, Value::Integer(10)),
        (3, Value::text("a")),
        (4, Value::Integer(2)),
        (5, Value::null_unknown()),
    ]
    .into_iter()
    .map(|(id, k)| (id, Row::from_values(vec![Value::Integer(id), k])))
    .collect();
    rows.sort_by(|a, b| cluster_order(&schema, a, b));
    let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids, vec![5, 4, 2, 1, 3]);
}

#[test]
fn a_key_built_outside_sql_is_checked_when_the_table_is_created() {
    let db = Database::open("memory://cluster_by_engine_boundary").unwrap();
    let past_the_end = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .cluster_by(vec![1])
        .build();
    assert!(
        db.engine().create_table(past_the_end).is_err(),
        "a key past the last column was accepted"
    );
    let twice = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("k", DataType::Integer, false, false)
        .cluster_by(vec![1, 1])
        .build();
    assert!(
        db.engine().create_table(twice).is_err(),
        "a repeated key column was accepted"
    );
    let unordered = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("doc", DataType::Json, true, false)
        .cluster_by(vec![1])
        .build();
    assert!(
        db.engine().create_table(unordered).is_err(),
        "a JSON key column was accepted"
    );
    let fine = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("k", DataType::Integer, false, false)
        .cluster_by(vec![1])
        .build();
    db.engine().create_table(fine).unwrap();
    assert_eq!(
        db.engine().get_table_schema("t").unwrap().cluster_key,
        vec![1]
    );
}

#[test]
fn a_key_column_cannot_be_changed_to_a_type_without_an_order() {
    let db = Database::open("memory://cluster_by_modify_column").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER) CLUSTER BY (k)",
        (),
    )
    .unwrap();
    assert!(
        db.execute("ALTER TABLE t MODIFY COLUMN k JSON", ())
            .is_err(),
        "a key column was changed to JSON"
    );
    db.execute("ALTER TABLE t MODIFY COLUMN v JSON", ())
        .unwrap();
    db.execute("ALTER TABLE t MODIFY COLUMN k TEXT", ())
        .unwrap();
    let schema = db.engine().get_table_schema("t").unwrap();
    assert_eq!(schema.cluster_key, vec![1]);
    assert_eq!(schema.columns[1].data_type, DataType::Text);
}

#[test]
fn compaction_of_volumes_whose_keys_are_all_null_does_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?compact_threshold=2",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT) CLUSTER BY (k)",
        (),
    )
    .unwrap();
    // Three single-row volumes with a NULL key: their dictionaries are
    // empty, and the merge compares the NULL cells against each other
    for id in 1..=3 {
        db.execute(&format!("INSERT INTO t VALUES ({id}, NULL)"), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    assert_eq!(ids(&db, "SELECT id FROM t ORDER BY id"), vec![1, 2, 3]);
    let volumes = sealed_volumes(dir.path(), "t", &[1]);
    assert_eq!(volumes.iter().map(|v| v.len()).sum::<usize>(), 3);
}

#[test]
fn a_key_column_cannot_lose_its_order_through_the_table_api() {
    let db = Database::open("memory://cluster_by_table_api").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER) CLUSTER BY (k)",
        (),
    )
    .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut table = tx.get_table("t").unwrap();
    assert!(
        table.modify_column("k", DataType::Json, true).is_err(),
        "the table API turned a key column into JSON"
    );
    table.modify_column("k", DataType::Text, true).unwrap();
    tx.rollback().unwrap();
}

#[test]
fn key_cells_compare_integers_and_floats_exactly() {
    use stoolap::storage::volume::column::ColumnData;
    let ints = ColumnData::Int64 {
        values: vec![9_007_199_254_740_993, 3],
        nulls: vec![false, false],
    };
    let floats = ColumnData::Float64 {
        values: vec![9_007_199_254_740_992.0, 3.0],
        nulls: vec![false, false],
    };
    // 2^53 + 1 is not representable as an f64: a cast would call them equal
    assert_eq!(
        ints.compare_cells(0, &floats, 0),
        std::cmp::Ordering::Greater
    );
    assert_eq!(floats.compare_cells(0, &ints, 0), std::cmp::Ordering::Less);
    assert_eq!(ints.compare_cells(1, &floats, 1), std::cmp::Ordering::Equal);
    assert_eq!(
        ints.compare_cell_with_value(0, &Value::Float(9_007_199_254_740_992.0)),
        std::cmp::Ordering::Greater
    );
    // Two NULL text cells are equal without a dictionary to read
    let nulls = ColumnData::Dictionary {
        ids: vec![0],
        dictionary: std::sync::Arc::from(Vec::<stoolap::common::SmartString>::new()),
        nulls: vec![true],
    };
    assert_eq!(nulls.compare_cells(0, &nulls, 0), std::cmp::Ordering::Equal);
    assert_eq!(
        nulls.compare_cell_with_value(0, &Value::text("a")),
        std::cmp::Ordering::Less
    );
}

#[test]
fn alter_table_cluster_by_sets_the_key_and_orders_the_next_seal() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?compact_threshold=2", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE ticks (id INTEGER PRIMARY KEY, exchange TEXT NOT NULL, symbol TEXT NOT NULL, time INTEGER NOT NULL, price REAL)",
            (),
        )
        .unwrap();
        // The first volume is sealed before the key exists: id order
        insert_ticks(&db, 1_000);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("ALTER TABLE ticks CLUSTER BY (exchange, symbol, time)", ())
            .unwrap();
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
        // The next seal is in key order, and the same checkpoint's compaction
        // rewrites the volume sealed before the key in key order too
        let insert = db
            .prepare("INSERT INTO ticks VALUES (?, ?, ?, ?, ?)")
            .unwrap();
        for id in 1_001..=2_000i64 {
            let exchange = if id % 2 == 0 { "b" } else { "a" };
            let symbol = if id % 3 == 0 { "y" } else { "x" };
            insert
                .execute((id, exchange, symbol, 3_000 - id, id as f64))
                .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let volumes = sealed_volumes(dir.path(), "ticks", &[1, 2, 3]);
        assert_eq!(volumes.iter().map(|v| v.len()).sum::<usize>(), 2_000);
        assert!(
            volumes.iter().all(|v| in_key_order(v)),
            "a volume is still out of key order after the checkpoint"
        );
        // A later compaction keeps everything in key order
        for id in 2_001..=3_000i64 {
            let exchange = if id % 2 == 0 { "b" } else { "a" };
            let symbol = if id % 3 == 0 { "y" } else { "x" };
            insert
                .execute((id, exchange, symbol, 3_000 - id, id as f64))
                .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let volumes = sealed_volumes(dir.path(), "ticks", &[1, 2, 3]);
        assert_eq!(
            volumes.len(),
            1,
            "compaction did not merge the three volumes"
        );
        assert!(in_key_order(&volumes[0]));
        assert_eq!(volumes[0].len(), 3_000);
        assert_eq!(ids(&db, "SELECT id FROM ticks WHERE id = 1500"), vec![1500]);
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        db.engine().get_table_schema("ticks").unwrap().cluster_key,
        vec![1, 2, 3],
        "the key set by ALTER TABLE did not survive the reopen"
    );
    let count: i64 = db.query_one("SELECT COUNT(*) FROM ticks", ()).unwrap();
    assert_eq!(count, 3_000);
}

#[test]
fn alter_table_cluster_by_checks_its_columns() {
    let db = Database::open("memory://alter_cluster_by_errors").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, doc JSON)",
        (),
    )
    .unwrap();
    assert!(db.execute("ALTER TABLE t CLUSTER BY (b)", ()).is_err());
    assert!(db.execute("ALTER TABLE t CLUSTER BY (a, a)", ()).is_err());
    assert!(db.execute("ALTER TABLE t CLUSTER BY (doc)", ()).is_err());
    assert!(db.execute("ALTER TABLE t CLUSTER BY ()", ()).is_err());
    assert!(db
        .engine()
        .get_table_schema("t")
        .unwrap()
        .cluster_key
        .is_empty());
    db.execute("ALTER TABLE t CLUSTER BY (a)", ()).unwrap();
    assert_eq!(
        db.engine().get_table_schema("t").unwrap().cluster_key,
        vec![1]
    );
    // A second ALTER replaces the key
    db.execute("ALTER TABLE t CLUSTER BY (a, id)", ()).unwrap();
    assert_eq!(
        db.engine().get_table_schema("t").unwrap().cluster_key,
        vec![1, 0]
    );
}

#[test]
fn a_retained_table_handle_follows_a_key_change_made_elsewhere() {
    let db = Database::open("memory://cluster_by_retained_handle").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER) CLUSTER BY (a)",
        (),
    )
    .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let mut table = tx.get_table("t").unwrap();
    // The key moves to b while the handle still remembers a
    db.engine().set_cluster_key("t", vec![2]).unwrap();
    // a is free now: the change must go through and both copies agree
    // (the engine's cache is the executor's to refresh, so it is asked to)
    table.modify_column("a", DataType::Json, true).unwrap();
    db.engine().refresh_schema_cache("t").unwrap();
    let live = db.engine().get_table_schema("t").unwrap();
    assert_eq!(live.columns[1].data_type, DataType::Json);
    assert_eq!(table.schema().columns[1].data_type, DataType::Json);
    assert_eq!(table.schema().cluster_key, vec![2]);
    table.drop_column("a").unwrap();
    db.engine().refresh_schema_cache("t").unwrap();
    let live = db.engine().get_table_schema("t").unwrap();
    assert!(live.get_column_index("a").is_none());
    assert_eq!(
        live.cluster_key,
        vec![1],
        "the key did not follow the dropped column"
    );
    assert_eq!(table.schema().cluster_key, vec![1]);
    // b is the key now: the handle refuses to make it JSON
    assert!(table.modify_column("b", DataType::Json, true).is_err());
    tx.rollback().unwrap();
}

#[test]
fn concurrent_alters_replay_in_the_order_they_were_applied() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let expected = {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER) CLUSTER BY (a)",
            (),
        )
        .unwrap();
        let db1 = db.clone();
        let db2 = db.clone();
        let keys = std::thread::spawn(move || {
            for round in 0..40 {
                let key = if round % 2 == 0 { "(b)" } else { "(c)" };
                db1.execute(&format!("ALTER TABLE t CLUSTER BY {key}"), ())
                    .unwrap();
            }
        });
        let columns = std::thread::spawn(move || {
            for round in 0..40 {
                let sql = if round % 2 == 0 {
                    "ALTER TABLE t ADD COLUMN x INTEGER"
                } else {
                    "ALTER TABLE t DROP COLUMN x"
                };
                db2.execute(sql, ()).unwrap();
            }
        });
        keys.join().unwrap();
        columns.join().unwrap();
        let schema = db.engine().get_table_schema("t").unwrap();
        let names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        (names, schema.cluster_key.clone())
    };
    let db = Database::open(&dsn).unwrap();
    let schema = db.engine().get_table_schema("t").unwrap();
    let names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
    assert_eq!(
        (names, schema.cluster_key.clone()),
        expected,
        "the replayed schema differs from the one the statements left"
    );
}

/// A checkpoint re-records every table's CREATE after the point it
/// truncates the log to. An ALTER landing between the checkpoint's read
/// of the catalog and its records would replay before the CREATE that
/// carries the older key and be refused, so the checkpoint takes the DDL
/// guard for that stretch: while a statement holds it, the checkpoint
/// waits
#[test]
fn a_checkpoint_waits_for_the_ddl_guard_before_re_recording_the_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER) CLUSTER BY (a)",
        (),
    )
    .unwrap();
    let held = db.engine().ddl_guard();
    let checkpointer = db.clone();
    let checkpoint = std::thread::spawn(move || {
        checkpointer.execute("PRAGMA CHECKPOINT", ()).unwrap();
    });
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert!(
        !checkpoint.is_finished(),
        "the checkpoint re-recorded the catalog while a DDL statement held the guard"
    );
    drop(held);
    checkpoint.join().unwrap();
}

/// The volume files of `table` under `dir`, by name
fn volume_files(dir: &std::path::Path, table: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.join("volumes").join(table))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".vol"))
        .collect();
    names.sort();
    names
}

const UNCLUSTERED: &str = "CREATE TABLE ticks (id INTEGER PRIMARY KEY, exchange TEXT NOT NULL, symbol TEXT NOT NULL, time INTEGER NOT NULL, price REAL)";

#[test]
fn alter_table_cluster_by_reclusters_a_full_volume_on_the_next_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    // The target floor is 65,536 rows, so this volume is sub-target and
    // alone; no rule but the recluster rewrites a single such volume
    let db = Database::open(&format!(
        "file://{}?target_volume_rows=1000&compact_threshold=100",
        dir.path().display()
    ))
    .unwrap();
    db.execute(UNCLUSTERED, ()).unwrap();
    insert_ticks(&db, 1_000);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = volume_files(dir.path(), "ticks");
    assert_eq!(before.len(), 1);
    assert!(!in_key_order(
        &sealed_volumes(dir.path(), "ticks", &[1, 2, 3])[0]
    ));

    db.execute("ALTER TABLE ticks CLUSTER BY (exchange, symbol, time)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let after = volume_files(dir.path(), "ticks");
    assert_ne!(before, after, "the full volume was not rewritten");
    let volumes = sealed_volumes(dir.path(), "ticks", &[1, 2, 3]);
    assert_eq!(volumes.len(), 1);
    assert_eq!(volumes[0].len(), 1_000);
    assert!(
        in_key_order(&volumes[0]),
        "the rewritten volume is not in key order"
    );
    assert_eq!(ids(&db, "SELECT id FROM ticks WHERE id = 500"), vec![500]);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM ticks", ()).unwrap();
    assert_eq!(count, 1_000);

    // Once in key order, a further checkpoint leaves the volume alone
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(volume_files(dir.path(), "ticks"), after);
}

#[test]
fn an_unclustered_table_keeps_a_full_volume_across_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?target_volume_rows=1000&compact_threshold=100",
        dir.path().display()
    ))
    .unwrap();
    db.execute(UNCLUSTERED, ()).unwrap();
    insert_ticks(&db, 1_000);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = volume_files(dir.path(), "ticks");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(volume_files(dir.path(), "ticks"), before);
}

#[test]
fn the_recluster_of_older_volumes_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?target_volume_rows=1000&compact_threshold=100&checkpoint_on_close=off",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(UNCLUSTERED, ()).unwrap();
        insert_ticks(&db, 1_000);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute("ALTER TABLE ticks CLUSTER BY (exchange, symbol, time)", ())
            .unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert!(!in_key_order(
        &sealed_volumes(dir.path(), "ticks", &[1, 2, 3])[0]
    ));
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let volumes = sealed_volumes(dir.path(), "ticks", &[1, 2, 3]);
    assert_eq!(volumes.len(), 1);
    assert!(
        in_key_order(&volumes[0]),
        "the older volume was not reclustered after the reopen"
    );
    assert_eq!(ids(&db, "SELECT id FROM ticks WHERE id = 7"), vec![7]);
}
