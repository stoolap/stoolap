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

//! TiKV table implementation

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use crate::core::{DataType, Error, Result, Row, RowVec, Schema, Value};
use crate::storage::expression::Expression;
use crate::storage::traits::{QueryResult, Scanner, Table};

use super::encoding;
use super::error::from_tikv_error;
use super::result::TiKVQueryResult;
use super::scanner::TiKVScanner;

/// TiKV table handle
pub struct TiKVTable {
    /// Table ID in TiKV key space
    table_id: u64,
    /// Table name (lowercase)
    name: String,
    /// Table schema
    schema: Schema,
    /// Transaction ID
    txn_id: i64,
    /// Shared TiKV transaction
    txn: Arc<Mutex<Option<tikv_client::Transaction>>>,
    /// Tokio runtime handle
    runtime: tokio::runtime::Handle,
    /// Local row ID counter for this table within this transaction
    next_row_id: AtomicI64,
    /// Upper bound of allocated row ID batch (exclusive)
    row_id_batch_end: AtomicI64,
    /// Whether next_row_id has been initialized from TiKV
    row_id_initialized: std::sync::atomic::AtomicBool,
    /// Optional engine reference for index lookups
    engine: Option<*const super::engine::TiKVEngine>,
}

// SAFETY: TiKVEngine is Send + Sync, and we only use the pointer for read access
unsafe impl Send for TiKVTable {}
unsafe impl Sync for TiKVTable {}

impl TiKVTable {
    pub(crate) fn new(
        table_id: u64,
        name: String,
        schema: Schema,
        txn_id: i64,
        txn: Arc<Mutex<Option<tikv_client::Transaction>>>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self {
            table_id,
            name,
            schema,
            txn_id,
            txn,
            runtime,
            next_row_id: AtomicI64::new(0),
            row_id_batch_end: AtomicI64::new(0),
            row_id_initialized: std::sync::atomic::AtomicBool::new(false),
            engine: None,
        }
    }

    /// Set the engine reference for index lookups
    pub(crate) fn with_engine(mut self, engine: &super::engine::TiKVEngine) -> Self {
        self.engine = Some(engine as *const super::engine::TiKVEngine);
        self
    }

    /// Get the engine reference
    fn engine(&self) -> Option<&super::engine::TiKVEngine> {
        self.engine.map(|p| unsafe { &*p })
    }

    /// Batch size for row ID allocation
    const ROW_ID_BATCH_SIZE: i64 = 100;

    /// Initialize the row ID counter from TiKV (lazy, called on first insert)
    fn ensure_row_id_initialized(&self) -> Result<()> {
        if self
            .row_id_initialized
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Ok(());
        }

        self.allocate_row_id_batch()?;
        self.row_id_initialized
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Allocate a batch of row IDs from TiKV in a single RPC
    fn allocate_row_id_batch(&self) -> Result<()> {
        let key = encoding::make_next_row_id_key(&self.name);
        let mut guard = self.txn.lock();
        let txn = guard
            .as_mut()
            .ok_or_else(|| Error::internal("Transaction consumed"))?;

        let current = self
            .runtime
            .block_on(async { txn.get(key.clone()).await.map_err(from_tikv_error) })?;

        let start_id = current
            .map(|bytes| encoding::decode_i64(&bytes))
            .unwrap_or(1);

        let end_id = start_id + Self::ROW_ID_BATCH_SIZE;

        // Persist the end of the batch to TiKV (reserves the range)
        self.runtime.block_on(async {
            txn.put(key, encoding::encode_i64(end_id).to_vec())
                .await
                .map_err(from_tikv_error)
        })?;

        self.next_row_id.store(start_id, Ordering::Relaxed);
        self.row_id_batch_end.store(end_id, Ordering::Relaxed);

        Ok(())
    }

    /// Allocate the next row ID (uses local batch, refills from TiKV when exhausted)
    fn allocate_row_id(&self) -> Result<i64> {
        self.ensure_row_id_initialized()?;

        let id = self.next_row_id.fetch_add(1, Ordering::Relaxed);
        if id >= self.row_id_batch_end.load(Ordering::Relaxed) {
            // Batch exhausted, allocate a new batch
            self.allocate_row_id_batch()?;
            Ok(self.next_row_id.fetch_add(1, Ordering::Relaxed))
        } else {
            Ok(id)
        }
    }

    /// Execute an operation on the TiKV transaction
    fn with_txn<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut tikv_client::Transaction) -> Result<R>,
    {
        let mut guard = self.txn.lock();
        match guard.as_mut() {
            Some(txn) => f(txn),
            None => Err(Error::internal("Transaction consumed")),
        }
    }

    /// Batch size for paginated TiKV scans
    const SCAN_BATCH_SIZE: u32 = 1000;

    /// Scan all rows of this table from TiKV using paginated scan
    /// This fetches rows in batches to reduce peak memory of a single TiKV response
    fn scan_all_rows(&self) -> Result<Vec<(i64, Vec<Value>)>> {
        let prefix = encoding::make_data_prefix(self.table_id);
        let end = encoding::prefix_end_key(&prefix);
        let mut all_rows = Vec::new();
        let mut start_key = prefix;

        loop {
            let (batch, last_key) = self.with_txn(|txn| {
                let pairs = self.runtime.block_on(async {
                    txn.scan(start_key.clone()..end.clone(), Self::SCAN_BATCH_SIZE)
                        .await
                        .map_err(from_tikv_error)
                })?;

                let pairs: Vec<_> = pairs.collect();
                let mut rows = Vec::with_capacity(pairs.len());
                let mut last = None;
                for pair in &pairs {
                    let key = pair.key().to_owned();
                    let key_bytes: Vec<u8> = key.into();
                    let row_id = encoding::extract_row_id_from_data_key(&key_bytes);
                    let values = encoding::deserialize_row(pair.value())?;
                    last = Some(key_bytes);
                    rows.push((row_id, values));
                }
                Ok((rows, last))
            })?;

            let is_last = batch.len() < Self::SCAN_BATCH_SIZE as usize;
            all_rows.extend(batch);

            if is_last {
                break;
            }

            // Next batch starts after the last key
            if let Some(mut last) = last_key {
                // Append a zero byte to make it exclusive
                last.push(0);
                start_key = last;
            } else {
                break;
            }
        }

        Ok(all_rows)
    }

    /// Build column index mapping: column_name → index in schema
    fn column_index_map(&self) -> FxHashMap<String, usize> {
        self.schema
            .columns
            .iter()
            .enumerate()
            .map(|(i, col)| (col.name.to_lowercase(), i))
            .collect()
    }

    /// Map column names to indices
    fn resolve_column_indices_from_strs(&self, columns: &[&str]) -> Vec<usize> {
        let col_map = self.column_index_map();
        columns
            .iter()
            .filter_map(|name| col_map.get(&name.to_lowercase()).copied())
            .collect()
    }

    /// Add index entries for a row
    fn add_index_entries(&self, row_id: i64, values: &[Value]) -> Result<()> {
        let Some(engine) = self.engine() else {
            return Ok(());
        };
        let indexes = engine.indexes.read();
        let Some(table_indexes) = indexes.get(&self.name) else {
            return Ok(());
        };

        for meta in table_indexes.values() {
            let index_values: Vec<Value> = meta
                .column_ids
                .iter()
                .map(|&id| {
                    let idx = id as usize;
                    if idx < values.len() {
                        values[idx].clone()
                    } else {
                        Value::Null(DataType::Text)
                    }
                })
                .collect();

            if meta.is_unique {
                let mut key = encoding::make_index_prefix(self.table_id, meta.index_id);
                for v in &index_values {
                    key.extend_from_slice(&encoding::encode_value(v));
                }
                self.with_txn(|txn| {
                    self.runtime.block_on(async {
                        txn.put(key, encoding::encode_i64(row_id).to_vec())
                            .await
                            .map_err(from_tikv_error)
                    })
                })?;
            } else {
                let key =
                    encoding::make_index_key(self.table_id, meta.index_id, &index_values, row_id);
                self.with_txn(|txn| {
                    self.runtime
                        .block_on(async { txn.put(key, vec![]).await.map_err(from_tikv_error) })
                })?;
            }
        }
        Ok(())
    }

    /// Remove index entries for a row
    fn remove_index_entries(&self, row_id: i64, values: &[Value]) -> Result<()> {
        let Some(engine) = self.engine() else {
            return Ok(());
        };
        let indexes = engine.indexes.read();
        let Some(table_indexes) = indexes.get(&self.name) else {
            return Ok(());
        };

        for meta in table_indexes.values() {
            let index_values: Vec<Value> = meta
                .column_ids
                .iter()
                .map(|&id| {
                    let idx = id as usize;
                    if idx < values.len() {
                        values[idx].clone()
                    } else {
                        Value::Null(DataType::Text)
                    }
                })
                .collect();

            if meta.is_unique {
                let mut key = encoding::make_index_prefix(self.table_id, meta.index_id);
                for v in &index_values {
                    key.extend_from_slice(&encoding::encode_value(v));
                }
                self.with_txn(|txn| {
                    self.runtime
                        .block_on(async { txn.delete(key).await.map_err(from_tikv_error) })
                })?;
            } else {
                let key =
                    encoding::make_index_key(self.table_id, meta.index_id, &index_values, row_id);
                self.with_txn(|txn| {
                    self.runtime
                        .block_on(async { txn.delete(key).await.map_err(from_tikv_error) })
                })?;
            }
        }
        Ok(())
    }

    /// Get column names
    fn col_names(&self) -> Vec<String> {
        self.schema.columns.iter().map(|c| c.name.clone()).collect()
    }

    /// Apply filter to a row (represented as Vec<Value>)
    fn matches_filter(&self, values: &[Value], expr: Option<&dyn Expression>) -> bool {
        if let Some(e) = expr {
            let row = encoding::values_to_row(values.to_vec());
            e.evaluate(&row).unwrap_or(false)
        } else {
            true
        }
    }
}

impl Table for TiKVTable {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn txn_id(&self) -> i64 {
        self.txn_id
    }

    fn create_column(
        &mut self,
        _name: &str,
        _column_type: DataType,
        _nullable: bool,
    ) -> Result<()> {
        Err(Error::internal("ALTER TABLE not yet supported with TiKV"))
    }

    fn drop_column(&mut self, _name: &str) -> Result<()> {
        Err(Error::internal("ALTER TABLE not yet supported with TiKV"))
    }

    fn insert(&mut self, row: Row) -> Result<Row> {
        let row_id = self.allocate_row_id()?;

        // Handle AUTO_INCREMENT: if PK integer column and value is NULL/0, use row_id
        let mut values = encoding::row_to_values(&row);
        if let Some(pk_col_idx) = self.schema.pk_column_index() {
            if pk_col_idx < values.len() {
                let needs_auto = matches!(&values[pk_col_idx], Value::Null(_) | Value::Integer(0));
                if needs_auto {
                    values[pk_col_idx] = Value::Integer(row_id);
                }
            }
        }

        let key = encoding::make_data_key(self.table_id, row_id);
        let value = encoding::serialize_row(&values)?;

        self.with_txn(|txn| {
            self.runtime
                .block_on(async { txn.put(key, value).await.map_err(from_tikv_error) })
        })?;

        // Maintain indexes
        self.add_index_entries(row_id, &values)?;

        Ok(encoding::values_to_row(values))
    }

    fn insert_batch(&mut self, rows: Vec<Row>) -> Result<()> {
        for row in rows {
            self.insert(row)?;
        }
        Ok(())
    }

    fn update(
        &mut self,
        where_expr: Option<&dyn Expression>,
        setter: &mut dyn FnMut(Row) -> Result<(Row, bool)>,
    ) -> Result<i32> {
        let all_rows = self.scan_all_rows()?;
        let mut updated: i32 = 0;

        for (row_id, values) in &all_rows {
            if !self.matches_filter(values, where_expr) {
                continue;
            }

            let row = encoding::values_to_row(values.clone());
            let (new_row, changed) = setter(row)?;
            if changed {
                let new_values = encoding::row_to_values(&new_row);
                // Update indexes: remove old entries, add new
                self.remove_index_entries(*row_id, values)?;
                self.add_index_entries(*row_id, &new_values)?;

                let key = encoding::make_data_key(self.table_id, *row_id);
                let encoded = encoding::serialize_row(&new_values)?;
                self.with_txn(|txn| {
                    self.runtime
                        .block_on(async { txn.put(key, encoded).await.map_err(from_tikv_error) })
                })?;
                updated += 1;
            }
        }
        Ok(updated)
    }

    fn update_by_row_ids(
        &mut self,
        row_ids: &[i64],
        setter: &mut dyn FnMut(Row) -> Result<(Row, bool)>,
    ) -> Result<i32> {
        let mut updated: i32 = 0;
        for &row_id in row_ids {
            let key = encoding::make_data_key(self.table_id, row_id);
            let existing = self.with_txn(|txn| {
                self.runtime
                    .block_on(async { txn.get(key.clone()).await.map_err(from_tikv_error) })
            })?;

            if let Some(bytes) = existing {
                let old_values = encoding::deserialize_row(&bytes)?;
                let row = encoding::values_to_row(old_values.clone());
                let (new_row, changed) = setter(row)?;
                if changed {
                    let new_values = encoding::row_to_values(&new_row);
                    // Update indexes
                    self.remove_index_entries(row_id, &old_values)?;
                    self.add_index_entries(row_id, &new_values)?;

                    let encoded = encoding::serialize_row(&new_values)?;
                    self.with_txn(|txn| {
                        self.runtime.block_on(async {
                            txn.put(key, encoded).await.map_err(from_tikv_error)
                        })
                    })?;
                    updated += 1;
                }
            }
        }
        Ok(updated)
    }

    fn delete_by_row_ids(&mut self, row_ids: &[i64]) -> Result<i32> {
        let mut deleted: i32 = 0;
        for &row_id in row_ids {
            let key = encoding::make_data_key(self.table_id, row_id);
            // Read old values for index removal
            let old_values = self.with_txn(|txn| {
                self.runtime
                    .block_on(async { txn.get(key.clone()).await.map_err(from_tikv_error) })
            })?;
            if let Some(bytes) = &old_values {
                if let Ok(values) = encoding::deserialize_row(bytes) {
                    let _ = self.remove_index_entries(row_id, &values);
                }
            }

            self.with_txn(|txn| {
                self.runtime
                    .block_on(async { txn.delete(key).await.map_err(from_tikv_error) })
            })?;
            deleted += 1;
        }
        Ok(deleted)
    }

    fn get_active_row_ids(&self) -> Vec<i64> {
        self.scan_all_rows()
            .map(|rows| rows.iter().map(|(id, _)| *id).collect())
            .unwrap_or_default()
    }

    fn delete(&mut self, where_expr: Option<&dyn Expression>) -> Result<i32> {
        let all_rows = self.scan_all_rows()?;
        let mut ids = Vec::new();
        for (row_id, values) in &all_rows {
            if self.matches_filter(values, where_expr) {
                ids.push(*row_id);
            }
        }
        self.delete_by_row_ids(&ids)
    }

    fn scan(
        &self,
        column_indices: &[usize],
        where_expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn Scanner>> {
        let all_rows = self.scan_all_rows()?;
        let mut result_rows = Vec::new();

        for (row_id, values) in all_rows {
            if !self.matches_filter(&values, where_expr) {
                continue;
            }
            let row = encoding::values_to_row(values);
            result_rows.push((row_id, row));
        }

        Ok(Box::new(TiKVScanner::from_rows(
            result_rows,
            column_indices.to_vec(),
        )))
    }

    fn collect_all_rows(&self, where_expr: Option<&dyn Expression>) -> Result<RowVec> {
        let all_rows = self.scan_all_rows()?;
        let mut result = RowVec::new();
        for (row_id, values) in all_rows {
            if !self.matches_filter(&values, where_expr) {
                continue;
            }
            let row = encoding::values_to_row(values);
            result.push((row_id, row));
        }
        Ok(result)
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        // Commit is handled at transaction level
        Ok(())
    }

    fn rollback(&mut self) {
        // Rollback is handled at transaction level
    }

    fn rollback_to_timestamp(&self, _timestamp: i64) {
        // Not supported with TiKV
    }

    fn row_count(&self) -> usize {
        self.scan_all_rows().map(|rows| rows.len()).unwrap_or(0)
    }

    fn row_count_hint(&self) -> usize {
        self.row_count()
    }

    fn fast_row_count(&self) -> Option<usize> {
        // Use scan_keys to avoid deserializing row data
        let prefix = encoding::make_data_prefix(self.table_id);
        let end = encoding::prefix_end_key(&prefix);

        let count = self.with_txn(|txn| {
            let keys = self.runtime.block_on(async {
                txn.scan_keys(prefix..end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)
            })?;
            Ok(keys.count())
        });

        count.ok()
    }

    fn fetch_rows_by_ids_into(
        &self,
        row_ids: &[i64],
        filter: &dyn Expression,
        buffer: &mut RowVec,
    ) {
        for &row_id in row_ids {
            let key = encoding::make_data_key(self.table_id, row_id);
            let result = self.with_txn(|txn| {
                self.runtime
                    .block_on(async { txn.get(key).await.map_err(from_tikv_error) })
            });

            if let Ok(Some(bytes)) = result {
                if let Ok(values) = encoding::deserialize_row(&bytes) {
                    let row = encoding::values_to_row(values);
                    if filter.evaluate(&row).unwrap_or(false) {
                        buffer.push((row_id, row));
                    }
                }
            }
        }
    }

    fn collect_rows_with_limit(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<RowVec> {
        let prefix = encoding::make_data_prefix(self.table_id);
        let end = encoding::prefix_end_key(&prefix);
        let mut result = RowVec::with_capacity(limit);
        let mut skipped = 0;
        let mut start_key = prefix;

        'outer: loop {
            let (batch, last_key) = self.with_txn(|txn| {
                let pairs = self.runtime.block_on(async {
                    txn.scan(start_key.clone()..end.clone(), Self::SCAN_BATCH_SIZE)
                        .await
                        .map_err(from_tikv_error)
                })?;
                let pairs: Vec<_> = pairs.collect();
                let mut rows = Vec::with_capacity(pairs.len());
                let mut last = None;
                for pair in &pairs {
                    let key = pair.key().to_owned();
                    let key_bytes: Vec<u8> = key.into();
                    let row_id = encoding::extract_row_id_from_data_key(&key_bytes);
                    let values = encoding::deserialize_row(pair.value())?;
                    last = Some(key_bytes);
                    rows.push((row_id, values));
                }
                Ok((rows, last))
            })?;

            let is_last = batch.len() < Self::SCAN_BATCH_SIZE as usize;

            for (row_id, values) in batch {
                if !self.matches_filter(&values, where_expr) {
                    continue;
                }
                if skipped < offset {
                    skipped += 1;
                    continue;
                }
                let row = encoding::values_to_row(values);
                result.push((row_id, row));
                if result.len() >= limit {
                    break 'outer;
                }
            }

            if is_last {
                break;
            }

            if let Some(mut last) = last_key {
                last.push(0);
                start_key = last;
            } else {
                break;
            }
        }

        Ok(result)
    }

    fn has_local_changes(&self) -> bool {
        false // TiKV handles change tracking internally
    }

    fn create_index(&self, _name: &str, _columns: &[&str], _is_unique: bool) -> Result<()> {
        // Index creation is handled at transaction level via create_table_index
        Ok(())
    }

    fn drop_index(&self, _name: &str) -> Result<()> {
        // Index deletion is handled at transaction level via drop_table_index
        Ok(())
    }

    fn create_btree_index(
        &self,
        _column_name: &str,
        _is_unique: bool,
        _custom_name: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }

    fn drop_btree_index(&self, _column_name: &str) -> Result<()> {
        Ok(())
    }

    fn has_index_on_column(&self, column_name: &str) -> bool {
        let Some(engine) = self.engine() else {
            return false;
        };
        let col_lower = column_name.to_lowercase();
        let indexes = engine.indexes.read();
        if let Some(table_indexes) = indexes.get(&self.name) {
            table_indexes.values().any(|meta| {
                meta.column_names
                    .iter()
                    .any(|c| c.to_lowercase() == col_lower)
            })
        } else {
            false
        }
    }

    fn get_index_on_column(
        &self,
        column_name: &str,
    ) -> Option<std::sync::Arc<dyn crate::storage::traits::Index>> {
        let engine = self.engine()?;
        let col_lower = column_name.to_lowercase();
        let indexes = engine.indexes.read();
        let table_indexes = indexes.get(&self.name)?;

        // Find an index that covers this column
        let meta = table_indexes.values().find(|meta| {
            meta.column_names
                .iter()
                .any(|c| c.to_lowercase() == col_lower)
        })?;

        // Create a TiKVIndex using the table's own transaction
        Some(std::sync::Arc::new(super::index::TiKVIndex::new(
            meta.clone(),
            Arc::clone(&self.txn),
            self.runtime.clone(),
        )))
    }

    fn collect_rows_ordered_by_index(
        &self,
        column_name: &str,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<RowVec> {
        let col_lower = column_name.to_lowercase();

        // Check if this is the PK column — use data key ordering directly
        if let Some(pk_idx) = self.schema.pk_column_index() {
            let pk_col = &self.schema.columns[pk_idx];
            if pk_col.name.to_lowercase() == col_lower {
                let prefix = encoding::make_data_prefix(self.table_id);
                let end = encoding::prefix_end_key(&prefix);
                let total = (limit + offset) as u32;

                let pairs_result = self.with_txn(|txn| {
                    let kv_pairs: Vec<(Vec<u8>, Vec<u8>)> = if ascending {
                        let pairs = self.runtime.block_on(async {
                            txn.scan(prefix..end, total).await.map_err(from_tikv_error)
                        })?;
                        pairs.into_iter().map(|p| (p.0.into(), p.1)).collect()
                    } else {
                        let pairs = self.runtime.block_on(async {
                            txn.scan_reverse(prefix..end, total)
                                .await
                                .map_err(from_tikv_error)
                        })?;
                        pairs.into_iter().map(|p| (p.0.into(), p.1)).collect()
                    };
                    Ok(kv_pairs)
                });

                if let Ok(kv_pairs) = pairs_result {
                    let mut result = RowVec::with_capacity(limit);
                    for (i, (key, value)) in kv_pairs.into_iter().enumerate() {
                        if i < offset {
                            continue;
                        }
                        if result.len() >= limit {
                            break;
                        }
                        let row_id = encoding::extract_row_id_from_data_key(&key);
                        if let Ok(values) = encoding::deserialize_row(&value) {
                            result.push((row_id, encoding::values_to_row(values)));
                        }
                    }
                    return Some(result);
                }
                return None;
            }
        }

        // Check if column has a secondary index
        let index = self.get_index_on_column(column_name)?;

        // Get ordered row IDs from the index
        // Request extra to handle any potential gaps
        let batch_size = (limit + offset) * 2 + 100;
        let ordered_row_ids = index.get_row_ids_ordered(ascending, batch_size, 0)?;

        let mut result = RowVec::with_capacity(limit);
        let mut skipped = 0;

        for row_id in ordered_row_ids {
            let key = encoding::make_data_key(self.table_id, row_id);
            let fetch_result = self.with_txn(|txn| {
                self.runtime
                    .block_on(async { txn.get(key).await.map_err(from_tikv_error) })
            });

            if let Ok(Some(bytes)) = fetch_result {
                if let Ok(values) = encoding::deserialize_row(&bytes) {
                    if skipped < offset {
                        skipped += 1;
                        continue;
                    }
                    result.push((row_id, encoding::values_to_row(values)));
                    if result.len() >= limit {
                        break;
                    }
                }
            }
        }

        Some(result)
    }

    fn collect_rows_pk_keyset(
        &self,
        start_after: Option<i64>,
        start_from: Option<i64>,
        ascending: bool,
        limit: usize,
    ) -> Option<RowVec> {
        // Only works for tables with integer PK
        self.schema.pk_column_index()?;

        let start_row_id = if let Some(after) = start_after {
            if ascending {
                after + 1
            } else {
                after - 1
            }
        } else if let Some(from) = start_from {
            from
        } else if ascending {
            i64::MIN
        } else {
            i64::MAX
        };

        let prefix = encoding::make_data_prefix(self.table_id);

        let pairs_result = if ascending {
            let start_key = encoding::make_data_key(self.table_id, start_row_id);
            let end = encoding::prefix_end_key(&prefix);
            self.with_txn(|txn| {
                let pairs = self.runtime.block_on(async {
                    txn.scan(start_key..end, limit as u32)
                        .await
                        .map_err(from_tikv_error)
                })?;
                Ok(pairs
                    .into_iter()
                    .map(|p| (p.0.into(), p.1))
                    .collect::<Vec<(Vec<u8>, Vec<u8>)>>())
            })
        } else {
            // For descending, scan backwards from start_row_id
            let end_key = encoding::make_data_key(self.table_id, start_row_id + 1);
            self.with_txn(|txn| {
                let pairs = self.runtime.block_on(async {
                    txn.scan_reverse(prefix..end_key, limit as u32)
                        .await
                        .map_err(from_tikv_error)
                })?;
                Ok(pairs
                    .into_iter()
                    .map(|p| (p.0.into(), p.1))
                    .collect::<Vec<(Vec<u8>, Vec<u8>)>>())
            })
        };

        let kv_pairs = pairs_result.ok()?;
        let mut result = RowVec::with_capacity(limit);

        for (key, value) in kv_pairs {
            let row_id = encoding::extract_row_id_from_data_key(&key);
            if let Ok(values) = encoding::deserialize_row(&value) {
                result.push((row_id, encoding::values_to_row(values)));
                if result.len() >= limit {
                    break;
                }
            }
        }

        Some(result)
    }

    fn rename_column(&mut self, _old_name: &str, _new_name: &str) -> Result<()> {
        Err(Error::internal("ALTER TABLE not yet supported with TiKV"))
    }

    fn modify_column(&mut self, _name: &str, _new_type: DataType, _nullable: bool) -> Result<()> {
        Err(Error::internal("ALTER TABLE not yet supported with TiKV"))
    }

    fn select(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn QueryResult>> {
        let column_indices = self.resolve_column_indices_from_strs(columns);
        let all_rows = self.scan_all_rows()?;
        let mut result_rows = Vec::new();

        for (row_id, values) in all_rows {
            if !self.matches_filter(&values, expr) {
                continue;
            }
            let row = encoding::values_to_row(values);
            result_rows.push((row_id, row));
        }

        let scanner = TiKVScanner::from_rows(result_rows, column_indices);
        let result_columns = if columns.is_empty() {
            self.col_names()
        } else {
            columns.iter().map(|s| s.to_string()).collect()
        };

        Ok(Box::new(TiKVQueryResult::new(result_columns, scanner)))
    }

    fn select_with_aliases(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        _aliases: &FxHashMap<String, String>,
    ) -> Result<Box<dyn QueryResult>> {
        // Aliases are handled at the executor level
        self.select(columns, expr)
    }

    fn select_as_of(
        &self,
        _columns: &[&str],
        _expr: Option<&dyn Expression>,
        _temporal_type: &str,
        _temporal_value: i64,
    ) -> Result<Box<dyn QueryResult>> {
        // Temporal queries are handled at the transaction level (TiKVTransaction::select_as_of)
        // which creates a TiKV snapshot at the requested timestamp.
        Err(Error::internal(
            "Temporal queries should be executed via transaction, not table directly",
        ))
    }
}
