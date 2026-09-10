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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use stoolap::core::{DataType, Error, Result, Row, RowVec, SchemaBuilder, Value};
use stoolap::executor::context::{
    cache_count_counter, cache_exists_fetcher, cache_exists_index, clear_all_thread_local_caches,
    ExecutionContext,
};
use stoolap::executor::expression::clear_program_cache;
use stoolap::executor::Executor;
use stoolap::functions::scalar::SleepFunction;
use stoolap::functions::{global_registry, FunctionInfo, ScalarFunction};
use stoolap::storage::expression::ComparisonExpr;
use stoolap::storage::mvcc::version_store::{AggregateOp, GroupedAggregateResult};
use stoolap::storage::mvcc::{MVCCTable, TransactionVersionStore, VersionStore};
use stoolap::storage::traits::{Engine, Table};
use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};
use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
use stoolap::storage::volume::table::SegmentedTable;
use stoolap::storage::volume::writer::VolumeBuilder;
use stoolap::Database;

fn cold_table(dir: &Path, snapshot: bool) -> (SegmentedTable, PathBuf) {
    cold_table_with_index(dir, snapshot, false)
}

fn cold_table_with_index(dir: &Path, snapshot: bool, indexed: bool) -> (SegmentedTable, PathBuf) {
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("v", DataType::Integer, false, false)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    for id in [1, 2] {
        builder.add_row(
            id,
            &Row::from_values(vec![Value::Integer(id), Value::Integer(id * 10)]),
        );
    }
    let path = write_volume_to_disk(dir, "t", 1, &builder.finish()).unwrap();
    let volume = Arc::new(read_volume_from_disk(&path).unwrap());
    assert!(volume.is_warm());
    volume.mark_accessed();
    let manager = Arc::new(SegmentManager::new("t", Some(dir.to_path_buf())));
    manager.register_segment(
        1,
        volume,
        SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("vol_0000000000000001.vol"),
            row_count: 2,
            min_row_id: 1,
            max_row_id: 2,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        },
        Some(&schema),
    );
    manager.evict_idle_volumes(0);
    manager.evict_idle_volumes(3);
    assert!(manager.segments_raw()[&1].volume.is_cold());
    let store = Arc::new(VersionStore::new(schema.table_name.clone(), schema));
    let mut local = TransactionVersionStore::new(Arc::clone(&store), 1);
    if snapshot {
        local
            .put(
                3,
                Row::from_values(vec![Value::Integer(3), Value::Integer(30)]),
                false,
            )
            .unwrap();
    }
    let hot = Box::new(MVCCTable::new(1, store, local));
    if indexed {
        hot.create_btree_index("v", false, Some("idx_t_v")).unwrap();
    }
    let table = if snapshot {
        SegmentedTable::with_snapshot_seq(hot, manager, 1)
    } else {
        SegmentedTable::new(hot, manager)
    };
    (table, path)
}

#[test]
fn snapshot_count_propagates_reload_failure_and_keeps_local_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (table, path) = cold_table(dir.path(), true);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    for _ in 0..2 {
        let expected = table.segment_manager().ensure_volume(1).map(|_| 0usize);
        assert!(expected
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("failed to reload cold volume seg=1"));
        assert_eq!(
            format!("{:?}", table.collect_all_rows(None).map(|rows| rows.len())),
            format!("{expected:?}")
        );
        assert_eq!(format!("{:?}", table.row_count()), format!("{expected:?}"));
    }
    std::fs::write(&path, bytes).unwrap();
    assert_eq!(table.collect_all_rows(None).unwrap().len(), 3);
}

#[test]
fn distinct_values_propagates_cold_reload_failure() {
    let dir = tempfile::tempdir().unwrap();
    let (table, path) = cold_table(dir.path(), false);
    std::fs::remove_file(&path).unwrap();
    let expected = table
        .segment_manager()
        .get_volumes_newest_first()
        .map(|_| Some(Vec::<Value>::new()));
    assert!(expected
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("cold volume reload failed"));
    assert_eq!(
        format!("{:?}", table.compute_distinct_values(1)),
        format!("{expected:?}")
    );
}

#[test]
fn exact_counts_include_transaction_local_rows() {
    let db = Database::open("memory://exact_counts_include_transaction_local_rows").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    assert_eq!(
        tx.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
            .unwrap(),
        2
    );
    tx.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(
        tx.query_one::<i64, _>("SELECT COUNT(*) + 1 FROM t", ())
            .unwrap(),
        2
    );
    tx.rollback().unwrap();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
            .unwrap(),
        0
    );
}

fn assert_sum_reload_failure(aggregate: impl Fn(&SegmentedTable) -> String) {
    let dir = tempfile::tempdir().unwrap();
    let (table, path) = cold_table(dir.path(), false);
    table.segment_manager().add_tombstones(&[1], 1);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    for _ in 0..2 {
        let expected = table
            .segment_manager()
            .get_volumes_newest_first()
            .map(|_| Some((0.0, 0_usize)));
        assert!(expected
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("cold volume reload failed"));
        assert_eq!(aggregate(&table), format!("{expected:?}"));
    }
    std::fs::write(&path, bytes).unwrap();
    let restored = aggregate(&table);
    assert!(restored == "Some((20.0, 1))" || restored == "Ok(Some((20.0, 1)))");
}

#[test]
fn sum_propagates_cold_reload_failure() {
    assert_sum_reload_failure(|table| format!("{:?}", table.sum_column(1)));
}

#[test]
fn avg_propagates_cold_reload_failure() {
    assert_sum_reload_failure(|table| format!("{:?}", table.avg_column(1)));
}

fn assert_optional_reload_failure(
    read: impl Fn(&SegmentedTable) -> String,
    expected_value: impl std::fmt::Debug,
) {
    let dir = tempfile::tempdir().unwrap();
    let (table, path) = cold_table_with_index(dir.path(), false, true);
    table.segment_manager().add_tombstones(&[1], 1);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    for _ in 0..2 {
        let expected = table
            .segment_manager()
            .get_volumes_newest_first()
            .map(|_| ());
        let error = expected.as_ref().unwrap_err();
        assert!(error.to_string().contains("cold volume reload failed"));
        assert_eq!(read(&table), format!("Err({error:?})"));
    }
    std::fs::write(&path, bytes).unwrap();
    let restored = read(&table);
    let expected = format!("Some({expected_value:?})");
    assert!(
        restored == expected || restored == format!("Ok({expected})"),
        "{restored}"
    );
}

#[test]
fn min_propagates_cold_reload_failure() {
    assert_optional_reload_failure(
        |table| format!("{:?}", table.min_column(1)),
        Some(Value::Integer(20)),
    );
}

#[test]
fn max_propagates_cold_reload_failure() {
    assert_optional_reload_failure(
        |table| format!("{:?}", table.max_column(1)),
        Some(Value::Integer(20)),
    );
}

#[test]
fn partition_count_propagates_cold_reload_failure() {
    assert_optional_reload_failure(|table| format!("{:?}", table.get_partition_count("v")), 1);
}

#[test]
fn partition_values_propagate_cold_reload_failure() {
    assert_optional_reload_failure(
        |table| format!("{:?}", table.get_partition_values("v")),
        vec![Value::Integer(20)],
    );
}

fn surviving_rows() -> RowVec {
    RowVec::from_vec(vec![(
        2,
        Row::from_values(vec![Value::Integer(2), Value::Integer(20)]),
    )])
}

#[test]
fn grouped_partition_rows_propagate_cold_reload_failure() {
    assert_optional_reload_failure(
        |table| format!("{:?}", table.collect_rows_grouped_by_partition("v")),
        vec![(Value::Integer(20), surviving_rows())],
    );
}

#[test]
fn ordered_rows_propagate_cold_reload_failure() {
    assert_optional_reload_failure(
        |table| {
            format!(
                "{:?}",
                table.collect_rows_ordered_by_index("id", true, 2, 0)
            )
        },
        surviving_rows(),
    );
}

#[test]
fn filtered_aggregates_propagate_cold_reload_failure() {
    assert_optional_reload_failure(
        |table| {
            format!(
                "{:?}",
                table.compute_filtered_aggregates(
                    &[(AggregateOp::Sum, 1)],
                    &ComparisonExpr::gte("v", Value::Integer(0)),
                )
            )
        },
        vec![Value::Integer(20)],
    );
}

#[test]
fn grouped_aggregates_propagate_cold_reload_failure() {
    assert_optional_reload_failure(
        |table| {
            format!(
                "{:?}",
                table.compute_grouped_aggregates(&[1], &[(AggregateOp::Sum, 1)])
            )
        },
        vec![GroupedAggregateResult {
            group_values: vec![Value::Integer(20)],
            aggregate_values: vec![Value::Integer(20)],
        }],
    );
}

static WINDOW_HOOK_LOCK: Mutex<()> = Mutex::new(());
static WINDOW_HOOK_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct InvalidFirstPattern;

impl ScalarFunction for InvalidFirstPattern {
    fn name(&self) -> &str {
        "SLEEP"
    }

    fn info(&self) -> FunctionInfo {
        SleepFunction.info()
    }

    fn evaluate(&self, _args: &[Value]) -> Result<Value> {
        Ok(Value::Integer(
            WINDOW_HOOK_CALLS.fetch_add(1, Ordering::SeqCst) as i64,
        ))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Self)
    }
}

struct WindowHook {
    _serial: MutexGuard<'static, ()>,
}

impl Drop for WindowHook {
    fn drop(&mut self) {
        global_registry().register_scalar::<SleepFunction>();
        clear_program_cache();
    }
}

#[test]
fn lazy_window_propagates_first_error_without_retrying() {
    let db =
        Database::open("memory://lazy_window_propagates_first_error_without_retrying").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let _hook = WindowHook {
        _serial: WINDOW_HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner()),
    };
    WINDOW_HOOK_CALLS.store(0, Ordering::SeqCst);
    global_registry().register_scalar::<InvalidFirstPattern>();
    clear_program_cache();
    let result = db.query_one::<bool, _>(
        "SELECT 'x' REGEXP CASE WHEN SLEEP(v) = 0 THEN '[' ELSE 'x' END, \
         ROW_NUMBER() OVER (PARTITION BY v) FROM t LIMIT 1",
        (),
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Invalid regular expression"));
    assert_eq!(WINDOW_HOOK_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn optional_reads_preserve_fallbacks_with_local_changes() {
    let db =
        Database::open("memory://optional_reads_preserve_fallbacks_with_local_changes").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let count = db.prepare("SELECT COUNT(DISTINCT v) FROM t").unwrap();
    for indexed in [false, true] {
        if indexed {
            db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
        }
        assert_eq!(count.query_one::<i64, _>(()).unwrap(), 2);
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
        for (query, expected) in [
            ("SELECT MIN(v) FROM t", 10),
            ("SELECT MAX(v) FROM t", 30),
            ("SELECT COUNT(DISTINCT v) FROM t", 3),
            ("SELECT SUM(v) FROM t WHERE v >= 20", 50),
            (
                "SELECT SUM(v) FROM t GROUP BY v ORDER BY v DESC LIMIT 1",
                30,
            ),
            (
                "SELECT ROW_NUMBER() OVER (PARTITION BY v) FROM t LIMIT 1",
                1,
            ),
            (
                "SELECT ROW_NUMBER() OVER (ORDER BY id DESC) FROM t ORDER BY id DESC LIMIT 1",
                1,
            ),
        ] {
            assert_eq!(
                tx.query_one::<i64, _>(query, ()).unwrap(),
                expected,
                "{query}"
            );
        }
        tx.rollback().unwrap();
        assert_eq!(count.query_one::<i64, _>(()).unwrap(), 2);
    }
}

#[test]
fn sum_and_avg_use_resident_stats_without_reloading() {
    let dir = tempfile::tempdir().unwrap();
    let (table, path) = cold_table(dir.path(), false);
    std::fs::remove_file(&path).unwrap();
    for result in [
        format!("{:?}", table.sum_column(1)),
        format!("{:?}", table.avg_column(1)),
    ] {
        assert!(result == "Some((30.0, 2))" || result == "Ok(Some((30.0, 2)))");
    }
    assert!(table.segment_manager().segments_raw()[&1].volume.is_cold());
}

#[test]
fn sum_and_avg_preserve_local_rows_and_wrapped_expressions() {
    let db =
        Database::open("memory://sum_and_avg_preserve_local_rows_and_wrapped_expressions").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    for (query, expected) in [
        ("SELECT SUM(v) FROM t", 60.0),
        ("SELECT AVG(v) FROM t", 20.0),
        ("SELECT SUM(v) + 1 FROM t", 61.0),
        ("SELECT AVG(v) * 2 FROM t", 40.0),
    ] {
        assert_eq!(tx.query_one::<f64, _>(query, ()).unwrap(), expected);
    }
    tx.rollback().unwrap();
    for (query, expected) in [
        ("SELECT SUM(v) FROM t", 30.0),
        ("SELECT AVG(v) FROM t", 15.0),
        ("SELECT SUM(v) + 1 FROM t", 31.0),
        ("SELECT AVG(v) * 2 FROM t", 30.0),
    ] {
        assert_eq!(db.query_one::<f64, _>(query, ()).unwrap(), expected);
    }
}

#[test]
fn prepared_count_tracks_committed_changes() {
    let db = Database::open("memory://prepared_count_tracks_committed_changes").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    let count = db.prepare("SELECT COUNT(*) FROM t").unwrap();
    for id in 1..=3 {
        assert_eq!(count.query_one::<i64, _>(()).unwrap(), id - 1);
        db.execute("INSERT INTO t VALUES ($1)", (id,)).unwrap();
        assert_eq!(count.query_one::<i64, _>(()).unwrap(), id);
    }
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
    assert_eq!(count.query_one::<i64, _>(()).unwrap(), 2);
}

#[test]
fn cold_distinct_preserves_local_unindexed_values() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (3, 30)", ()).unwrap();
    let values: Vec<i64> = tx
        .query("SELECT DISTINCT v FROM t ORDER BY v", ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(values, [10, 20, 30]);
    tx.rollback().unwrap();
}

fn malformed_identity_table(id_count: usize, snapshot: bool) -> SegmentedTable {
    let schema = SchemaBuilder::new("identity")
        .column("id", DataType::Integer, false, true)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    for id in [1, 2] {
        builder.add_row(id, &Row::from_values(vec![Value::Integer(id)]));
    }
    let mut volume = builder.finish();
    Arc::make_mut(&mut volume.meta).row_ids.resize(id_count, 3);
    let manager = Arc::new(SegmentManager::new("identity", None));
    manager.register_segment(
        1,
        Arc::new(volume),
        SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::new(),
            row_count: 2,
            min_row_id: 1,
            max_row_id: 2,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        },
        Some(&schema),
    );
    let store = Arc::new(VersionStore::new(schema.table_name.clone(), schema));
    let local = TransactionVersionStore::new(Arc::clone(&store), 1);
    let hot = Box::new(MVCCTable::new(1, store, local));
    if snapshot {
        SegmentedTable::with_snapshot_seq(hot, manager, 1)
    } else {
        SegmentedTable::new(hot, manager)
    }
}

fn assert_identity_error(result: impl std::fmt::Debug) {
    let result = format!("{result:?}");
    assert!(result.starts_with("Err("), "{result}");
    assert!(
        result.contains("row ID count does not match volume row count"),
        "{result}"
    );
}

#[test]
fn identity_accessor_reports_invalid_data_and_borrows_valid_ids() {
    for id_count in [0, 1, 3] {
        let table = malformed_identity_table(id_count, false);
        let segments = table.segment_manager().segments_raw();
        assert_eq!(
            segments[&1].volume.row_ids().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }
    let table = malformed_identity_table(2, false);
    let segments = table.segment_manager().segments_raw();
    let volume = &segments[&1].volume;
    let ids = volume.row_ids().unwrap();
    assert_eq!(ids, [1, 2]);
    assert_eq!(ids.as_ptr(), volume.meta.row_ids.as_ptr());
}

#[test]
fn cold_membership_rejects_inconsistent_identity_metadata() {
    for id_count in [0, 1, 3] {
        for snapshot in [false, true] {
            let table = malformed_identity_table(id_count, snapshot);
            assert_identity_error(table.has_row_id(2));
            assert_identity_error(table.segment_manager().row_exists(2));
            assert_identity_error(table.segment_manager().is_row_id_in_volume(2));
        }
    }
}

#[test]
fn cold_fetch_rejects_inconsistent_identity_metadata() {
    for id_count in [0, 1, 3] {
        let table = malformed_identity_table(id_count, false);
        assert_identity_error(table.collect_rows_by_ids(&[2]));
        assert_identity_error(table.segment_manager().get_cold_row(2));
        assert_identity_error(
            table
                .segment_manager()
                .get_cold_row_normalized(2, table.schema()),
        );
    }
}

#[test]
fn exact_counts_do_not_cache_inconsistent_identity_metadata() {
    for id_count in [0, 1, 3] {
        let table = malformed_identity_table(id_count, false);
        for _ in 0..2 {
            assert_identity_error(table.row_count());
            assert_identity_error(table.fast_row_count());
            assert_identity_error(table.segment_manager().deduped_row_count());
        }
    }
}

#[test]
fn cold_update_rejects_inconsistent_identity_before_setter() {
    for id_count in [0, 1, 3] {
        let mut table = malformed_identity_table(id_count, false);
        let mut calls = 0;
        assert_identity_error(table.update_by_row_ids(&[2], &mut |row| {
            calls += 1;
            Ok((row, false))
        }));
        assert_eq!(calls, 0);
        assert!(!table.segment_manager().has_pending_tombstones(1));
    }
}

#[test]
fn cold_delete_rejects_inconsistent_identity_before_tombstone() {
    for id_count in [0, 1, 3] {
        let mut table = malformed_identity_table(id_count, false);
        assert_identity_error(table.delete_by_row_ids(&[2]));
        assert!(!table.segment_manager().has_pending_tombstones(1));
    }
}

#[test]
fn resident_identity_and_counts_do_not_reload_column_files() {
    let dir = tempfile::tempdir().unwrap();
    let (table, path) = cold_table(dir.path(), false);
    std::fs::remove_file(path).unwrap();
    for id in [1, 2, 3] {
        let expected = id != 3;
        let found = format!("{:?}", table.has_row_id(id));
        assert!(found == format!("Ok({expected})") || found == expected.to_string());
    }
    assert_eq!(table.row_count().unwrap(), 2);
    table.segment_manager().add_tombstones(&[1], 1);
    assert_eq!(table.row_count().unwrap(), 1);
    let exists = format!("{:?}", table.segment_manager().row_exists(1));
    assert!(exists == "Ok(false)" || exists == "false");
    let physical = format!("{:?}", table.segment_manager().is_row_id_in_volume(1));
    assert!(physical == "Ok(true)" || physical == "true");
    assert!(table.segment_manager().segments_raw()[&1].volume.is_cold());
}

struct CounterCacheScope;

impl CounterCacheScope {
    fn new() -> Self {
        clear_all_thread_local_caches();
        Self
    }
}

impl Drop for CounterCacheScope {
    fn drop(&mut self) {
        clear_all_thread_local_caches();
    }
}

fn nested_count(executor: &Executor) -> Result<Value> {
    let ctx = ExecutionContext::new().with_outer_row(
        [("outer_t.k".into(), Value::Integer(7))]
            .into_iter()
            .collect(),
        stoolap::common::CompactArc::new(vec!["outer_t.k".to_string()]),
    );
    let mut rows = executor.execute_with_context(
        "SELECT (SELECT COUNT(*) FROM inner_t WHERE inner_t.k = outer_t.k)",
        &ctx,
    )?;
    if let Some(error) = rows.last_error() {
        return Err(error);
    }
    assert!(rows.next());
    let value = rows.row()[0].clone();
    assert!(!rows.next());
    if let Some(error) = rows.last_error() {
        return Err(error);
    }
    Ok(value)
}

fn counter_database(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE inner_t (id INTEGER PRIMARY KEY, k INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_inner_k ON inner_t(k)", ())
        .unwrap();
    db.execute("INSERT INTO inner_t VALUES (1, 7)", ()).unwrap();
    db
}

#[test]
fn counter_factory_failure_does_not_use_cached_fetcher() {
    let db = counter_database("memory://counter_factory_failure_does_not_use_cached_fetcher");
    let engine = db.engine();
    let index = engine
        .get_all_indexes("inner_t")
        .unwrap()
        .into_iter()
        .find(|index| index.column_names() == ["k"])
        .unwrap();
    let mut config = engine.config();
    config.persistence.checkpoint_on_close = false;
    engine.update_engine_config(config).unwrap();
    engine.close_engine().unwrap();
    assert_eq!(&*index.get_row_ids_equal(&[Value::Integer(7)]), &[1]);
    let _caches = CounterCacheScope::new();
    cache_exists_index("inner_t:k".to_string(), index);
    let calls = Arc::new(AtomicUsize::new(0));
    let fallback_calls = Arc::clone(&calls);
    cache_exists_fetcher(
        "inner_t".to_string(),
        Box::new(move |_| {
            fallback_calls.fetch_add(1, Ordering::Relaxed);
            Ok(RowVec::new())
        }),
    );
    let result = nested_count(&Executor::new(Arc::clone(engine)));
    assert!(matches!(result, Err(Error::EngineNotOpen)), "{result:?}");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

trait CounterResult {
    fn from_result(result: Result<usize>) -> Self;
}

impl CounterResult for Result<usize> {
    fn from_result(result: Result<usize>) -> Self {
        result
    }
}

impl CounterResult for usize {
    fn from_result(result: Result<usize>) -> Self {
        result.unwrap_or(0)
    }
}

#[test]
fn counter_callback_failure_reaches_sql_without_fallback() {
    let db = counter_database("memory://counter_callback_failure_reaches_sql_without_fallback");
    let _caches = CounterCacheScope::new();
    let index = db
        .engine()
        .get_all_indexes("inner_t")
        .unwrap()
        .into_iter()
        .find(|index| index.column_names() == ["k"])
        .unwrap();
    cache_exists_index("inner_t:k".to_string(), index);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter_calls = Arc::clone(&calls);
    cache_count_counter(
        "inner_t".to_string(),
        Box::new(move |ids| {
            assert_eq!(ids, [1]);
            counter_calls.fetch_add(1, Ordering::Relaxed);
            CounterResult::from_result(Err(Error::internal("identity read failed")))
        }),
    );
    let result = nested_count(&Executor::new(Arc::clone(db.engine())));
    assert!(
        matches!(&result, Err(Error::Internal { message }) if message == "identity read failed"),
        "{result:?}"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}
