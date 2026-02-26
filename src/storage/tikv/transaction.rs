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

//! TiKV transaction implementation

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::common::CompactArc;
use crate::core::{Error, IsolationLevel, Result, Schema, SchemaColumn};
use crate::storage::expression::Expression;
use crate::storage::traits::{QueryResult, Table, Transaction};

use super::encoding;
use super::engine::TiKVEngine;
use super::error::from_tikv_error;
use super::table::TiKVTable;

/// Savepoint state for DDL rollback emulation
#[derive(Clone)]
struct SavepointState {
    /// Timestamp when savepoint was created
    timestamp: i64,
    /// Number of created_tables entries at savepoint time
    created_tables_len: usize,
    /// Number of dropped_tables entries at savepoint time
    dropped_tables_len: usize,
}

/// TiKV transaction
pub struct TiKVTransaction {
    /// Local transaction ID
    id: i64,
    /// Underlying TiKV transaction (wrapped in Mutex for shared access from Table)
    pub(crate) txn: Arc<Mutex<Option<tikv_client::Transaction>>>,
    /// Isolation level
    isolation_level: IsolationLevel,
    /// Tokio runtime handle
    pub(crate) runtime: tokio::runtime::Handle,
    /// Snapshot of schemas at transaction start
    schemas: FxHashMap<String, (u64, CompactArc<Schema>)>,
    /// Reference to engine (for DDL operations)
    engine: *const TiKVEngine,
    /// Savepoints for DDL rollback emulation
    savepoints: FxHashMap<String, SavepointState>,
    /// Tables created during this transaction (for savepoint DDL rollback)
    created_tables: Vec<String>,
    /// Tables dropped during this transaction (for savepoint DDL rollback)
    dropped_tables: Vec<(String, Schema)>,
}

// SAFETY: TiKVEngine is Send + Sync, and we only use the pointer for read access
unsafe impl Send for TiKVTransaction {}

impl TiKVTransaction {
    pub(crate) fn new(
        id: i64,
        tikv_txn: tikv_client::Transaction,
        isolation_level: IsolationLevel,
        runtime: tokio::runtime::Handle,
        schemas: FxHashMap<String, (u64, CompactArc<Schema>)>,
        engine: &TiKVEngine,
    ) -> Self {
        Self {
            id,
            txn: Arc::new(Mutex::new(Some(tikv_txn))),
            isolation_level,
            runtime,
            schemas,
            engine: engine as *const TiKVEngine,
            savepoints: FxHashMap::default(),
            created_tables: Vec::new(),
            dropped_tables: Vec::new(),
        }
    }

    fn engine(&self) -> &TiKVEngine {
        // SAFETY: engine lifetime is guaranteed to outlive the transaction
        unsafe { &*self.engine }
    }

    /// Get the TiKV transaction, returning an error if already consumed
    fn with_txn<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut tikv_client::Transaction) -> Result<R>,
    {
        let mut guard = self.txn.lock();
        match guard.as_mut() {
            Some(txn) => f(txn),
            None => Err(Error::internal(
                "Transaction already committed or rolled back",
            )),
        }
    }

    /// Get a table using a fresh TiKV snapshot (for Read Committed isolation)
    /// This creates a new optimistic transaction that sees the latest committed data
    fn get_table_fresh_snapshot(&self, name: &str) -> Result<Box<dyn Table>> {
        let table_name = name.to_lowercase();
        let (table_id, schema) = self
            .schemas
            .get(&table_name)
            .ok_or_else(|| Error::TableNotFound(name.to_string()))?;

        let engine = self.engine();
        let client = engine.client()?;
        let fresh_txn = self
            .runtime
            .block_on(async { client.begin_optimistic().await.map_err(from_tikv_error) })?;

        let table = TiKVTable::new(
            *table_id,
            table_name,
            (**schema).clone(),
            self.id,
            Arc::new(parking_lot::Mutex::new(Some(fresh_txn))),
            self.runtime.clone(),
        )
        .with_engine(engine);
        Ok(Box::new(table))
    }

    /// Internal drop table implementation (does not track for savepoint rollback)
    fn drop_table_internal(&mut self, table_name: &str) -> Result<()> {
        // Remove from local cache
        self.schemas.remove(table_name);

        // Update engine cache
        let engine = self.engine();
        engine.schemas.write().remove(table_name);
        engine
            .schema_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        // Delete metadata within this transaction
        self.with_txn(|txn| {
            self.runtime.block_on(async {
                txn.delete(encoding::make_schema_key(table_name))
                    .await
                    .map_err(from_tikv_error)?;
                txn.delete(encoding::make_table_id_key(table_name))
                    .await
                    .map_err(from_tikv_error)?;
                txn.delete(encoding::make_next_row_id_key(table_name))
                    .await
                    .map_err(from_tikv_error)?;
                Ok(())
            })
        })
    }

    /// Persist an updated schema to TiKV and update both local and engine caches
    fn persist_schema(&mut self, table_name: &str, table_id: u64, schema: &Schema) -> Result<()> {
        let schema_bytes = encoding::serialize_schema(schema)?;

        self.with_txn(|txn| {
            self.runtime.block_on(async {
                txn.put(encoding::make_schema_key(table_name), schema_bytes)
                    .await
                    .map_err(from_tikv_error)
            })
        })?;

        let schema_arc = CompactArc::new(schema.clone());

        // Update local cache
        self.schemas
            .insert(table_name.to_string(), (table_id, schema_arc.clone()));

        // Update engine cache
        let engine = self.engine();
        engine
            .schemas
            .write()
            .insert(table_name.to_string(), (table_id, schema_arc));
        engine
            .schema_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        Ok(())
    }
}

impl Drop for TiKVTransaction {
    fn drop(&mut self) {
        // If the transaction hasn't been committed or rolled back, roll it back
        if let Some(mut txn) = self.txn.lock().take() {
            let _ = self.runtime.block_on(async { txn.rollback().await });
        }
    }
}

impl Transaction for TiKVTransaction {
    fn begin(&mut self) -> Result<()> {
        Ok(()) // Already started
    }

    fn commit(&mut self) -> Result<()> {
        let mut txn = self
            .txn
            .lock()
            .take()
            .ok_or_else(|| Error::internal("Transaction already committed or rolled back"))?;

        self.runtime
            .block_on(async { txn.commit().await.map_err(from_tikv_error) })?;
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        let mut txn = self
            .txn
            .lock()
            .take()
            .ok_or_else(|| Error::internal("Transaction already committed or rolled back"))?;

        self.runtime
            .block_on(async { txn.rollback().await.map_err(from_tikv_error) })?;
        Ok(())
    }

    fn create_savepoint(&mut self, name: &str) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;

        self.savepoints.insert(
            name.to_string(),
            SavepointState {
                timestamp,
                created_tables_len: self.created_tables.len(),
                dropped_tables_len: self.dropped_tables.len(),
            },
        );
        Ok(())
    }

    fn release_savepoint(&mut self, name: &str) -> Result<()> {
        if self.savepoints.remove(name).is_none() {
            return Err(Error::invalid_argument(format!(
                "savepoint '{}' does not exist",
                name
            )));
        }
        Ok(())
    }

    fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        let sp_state = self.savepoints.get(name).cloned().ok_or_else(|| {
            Error::invalid_argument(format!("savepoint '{}' does not exist", name))
        })?;

        // Rollback DDL: undo CREATE TABLEs after savepoint
        while self.created_tables.len() > sp_state.created_tables_len {
            if let Some(table_name) = self.created_tables.pop() {
                // Drop the table that was created after the savepoint
                let _ = self.drop_table_internal(&table_name);
            }
        }

        // Rollback DDL: undo DROP TABLEs after savepoint
        while self.dropped_tables.len() > sp_state.dropped_tables_len {
            if let Some((table_name, schema)) = self.dropped_tables.pop() {
                // Recreate the table that was dropped after the savepoint
                let _ = self.create_table(&table_name, schema);
            }
        }

        // Remove this savepoint and all savepoints created after it
        let sp_timestamp = sp_state.timestamp;
        self.savepoints.retain(|_, sp| sp.timestamp <= sp_timestamp);

        Ok(())
    }

    fn get_savepoint_timestamp(&self, name: &str) -> Option<i64> {
        self.savepoints.get(name).map(|sp| sp.timestamp)
    }

    fn id(&self) -> i64 {
        self.id
    }

    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()> {
        self.isolation_level = level;
        Ok(())
    }

    fn create_table(&mut self, name: &str, schema: Schema) -> Result<Box<dyn Table>> {
        let table_name = name.to_lowercase();

        // Check existence
        if self.schemas.contains_key(&table_name) {
            return Err(Error::TableAlreadyExists(table_name));
        }

        // Allocate table ID (access engine via raw pointer to avoid borrow issue)
        let engine = self.engine();
        let table_id = engine
            .next_table_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Persist schema in TiKV within this transaction
        let schema_bytes = encoding::serialize_schema(&schema)?;

        self.with_txn(|txn| {
            self.runtime.block_on(async {
                txn.put(encoding::make_schema_key(&table_name), schema_bytes)
                    .await
                    .map_err(from_tikv_error)?;
                txn.put(
                    encoding::make_table_id_key(&table_name),
                    encoding::encode_u64(table_id).to_vec(),
                )
                .await
                .map_err(from_tikv_error)?;
                txn.put(
                    encoding::META_NEXT_TABLE_ID.to_vec(),
                    encoding::encode_u64(table_id + 1).to_vec(),
                )
                .await
                .map_err(from_tikv_error)?;
                txn.put(
                    encoding::make_next_row_id_key(&table_name),
                    encoding::encode_i64(1).to_vec(),
                )
                .await
                .map_err(from_tikv_error)?;
                Ok(())
            })
        })?;

        // Update local cache
        let schema_arc = CompactArc::new(schema.clone());
        self.schemas
            .insert(table_name.clone(), (table_id, schema_arc.clone()));

        // Update engine cache (access engine via raw pointer again after self borrow is released)
        let engine = self.engine();
        engine
            .schemas
            .write()
            .insert(table_name.clone(), (table_id, schema_arc));
        engine
            .schema_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        // Track for savepoint DDL rollback
        self.created_tables.push(table_name.clone());

        let table = TiKVTable::new(
            table_id,
            table_name,
            schema,
            self.id,
            Arc::clone(&self.txn),
            self.runtime.clone(),
        )
        .with_engine(self.engine());
        Ok(Box::new(table))
    }

    fn drop_table(&mut self, name: &str) -> Result<()> {
        let table_name = name.to_lowercase();

        // Save schema for savepoint DDL rollback before removing
        if let Some((_, schema_arc)) = self.schemas.get(&table_name) {
            self.dropped_tables
                .push((table_name.clone(), (**schema_arc).clone()));
        }

        self.drop_table_internal(&table_name)
    }

    fn get_table(&self, name: &str) -> Result<Box<dyn Table>> {
        let table_name = name.to_lowercase();
        let (table_id, schema) = self
            .schemas
            .get(&table_name)
            .ok_or_else(|| Error::TableNotFound(name.to_string()))?;

        let table = TiKVTable::new(
            *table_id,
            table_name,
            (**schema).clone(),
            self.id,
            Arc::clone(&self.txn),
            self.runtime.clone(),
        )
        .with_engine(self.engine());
        Ok(Box::new(table))
    }

    fn list_tables(&self) -> Result<Vec<String>> {
        Ok(self.schemas.keys().cloned().collect())
    }

    fn rename_table(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        let old_lower = old_name.to_lowercase();
        let new_lower = new_name.to_lowercase();

        // Check old exists and new doesn't
        let (table_id, mut schema) = self
            .schemas
            .remove(&old_lower)
            .ok_or_else(|| Error::TableNotFound(old_name.to_string()))?;
        if self.schemas.contains_key(&new_lower) {
            // Put back the old one
            self.schemas.insert(old_lower, (table_id, schema));
            return Err(Error::TableAlreadyExists(new_name.to_string()));
        }

        // Update schema name
        let schema_mut = CompactArc::make_mut(&mut schema);
        schema_mut.table_name = new_name.to_string();
        schema_mut.table_name_lower = new_lower.clone();

        // Persist: write new keys, delete old keys
        let schema_bytes = encoding::serialize_schema(&schema)?;
        self.with_txn(|txn| {
            self.runtime.block_on(async {
                // Write new schema key
                txn.put(encoding::make_schema_key(&new_lower), schema_bytes)
                    .await
                    .map_err(from_tikv_error)?;
                // Write new table_id key
                txn.put(
                    encoding::make_table_id_key(&new_lower),
                    encoding::encode_u64(table_id).to_vec(),
                )
                .await
                .map_err(from_tikv_error)?;
                // Copy row ID counter
                let rid_key = encoding::make_next_row_id_key(&old_lower);
                if let Some(rid_bytes) = txn.get(rid_key.clone()).await.map_err(from_tikv_error)? {
                    txn.put(encoding::make_next_row_id_key(&new_lower), rid_bytes)
                        .await
                        .map_err(from_tikv_error)?;
                }
                // Delete old keys
                txn.delete(encoding::make_schema_key(&old_lower))
                    .await
                    .map_err(from_tikv_error)?;
                txn.delete(encoding::make_table_id_key(&old_lower))
                    .await
                    .map_err(from_tikv_error)?;
                txn.delete(rid_key).await.map_err(from_tikv_error)?;
                Ok(())
            })
        })?;

        // Update local cache
        self.schemas
            .insert(new_lower.clone(), (table_id, schema.clone()));

        // Update engine cache
        let engine = self.engine();
        let mut eng_schemas = engine.schemas.write();
        eng_schemas.remove(&old_lower);
        eng_schemas.insert(new_lower, (table_id, schema));
        drop(eng_schemas);
        engine
            .schema_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        Ok(())
    }

    fn create_table_index(
        &mut self,
        table_name: &str,
        index_name: &str,
        columns: &[String],
        is_unique: bool,
    ) -> Result<()> {
        let table_name_lower = table_name.to_lowercase();

        // Look up table to get table_id and schema
        let (table_id, schema_arc) = self
            .schemas
            .get(&table_name_lower)
            .ok_or_else(|| Error::TableNotFound(table_name.to_string()))?;
        let table_id = *table_id;
        let schema = (**schema_arc).clone();

        // Resolve column names to IDs and data types
        let mut column_ids = Vec::new();
        let mut data_types = Vec::new();
        for col_name in columns {
            let (idx, col) = schema
                .find_column(col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.clone()))?;
            column_ids.push(idx as i32);
            data_types.push(col.data_type);
        }

        // Allocate index ID
        let engine = self.engine();
        let index_id = engine
            .next_index_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Create index metadata
        let meta = super::index::IndexMetadata {
            name: index_name.to_string(),
            table_name: table_name_lower.clone(),
            table_id,
            index_id,
            column_names: columns.to_vec(),
            column_ids,
            data_types,
            is_unique,
        };

        // Persist index metadata to TiKV
        let meta_bytes = meta.to_bytes()?;
        let meta_key = encoding::make_index_meta_key(&table_name_lower, index_name);

        self.with_txn(|txn| {
            self.runtime
                .block_on(async { txn.put(meta_key, meta_bytes).await.map_err(from_tikv_error) })
        })?;

        // Build index from existing data: scan all rows and add entries
        {
            let prefix = encoding::make_data_prefix(table_id);
            let end = encoding::prefix_end_key(&prefix);
            let col_indices: Vec<usize> = meta.column_ids.iter().map(|id| *id as usize).collect();

            self.with_txn(|txn| {
                let pairs = self.runtime.block_on(async {
                    txn.scan(prefix..end, u32::MAX)
                        .await
                        .map_err(from_tikv_error)
                })?;

                for pair in pairs {
                    let key: Vec<u8> = pair.0.into();
                    let row_id = encoding::extract_row_id_from_data_key(&key);
                    let row_values = encoding::deserialize_row(&pair.1)?;

                    // Extract indexed column values
                    let index_values: Vec<crate::core::Value> = col_indices
                        .iter()
                        .map(|&idx| {
                            if idx < row_values.len() {
                                row_values[idx].clone()
                            } else {
                                crate::core::Value::Null(crate::core::DataType::Text)
                            }
                        })
                        .collect();

                    // Write index entry
                    if is_unique {
                        let idx_key = {
                            let mut k = encoding::make_index_prefix(table_id, index_id);
                            for v in &index_values {
                                k.extend_from_slice(&encoding::encode_value(v));
                            }
                            k
                        };
                        self.runtime.block_on(async {
                            txn.put(idx_key, encoding::encode_i64(row_id).to_vec())
                                .await
                                .map_err(from_tikv_error)
                        })?;
                    } else {
                        let idx_key =
                            encoding::make_index_key(table_id, index_id, &index_values, row_id);
                        self.runtime.block_on(async {
                            txn.put(idx_key, vec![]).await.map_err(from_tikv_error)
                        })?;
                    }
                }
                Ok(())
            })?;
        }

        // Update engine index cache
        engine
            .indexes
            .write()
            .entry(table_name_lower)
            .or_default()
            .insert(index_name.to_string(), meta);

        Ok(())
    }

    fn drop_table_index(&mut self, table_name: &str, index_name: &str) -> Result<()> {
        let table_name_lower = table_name.to_lowercase();

        // Get index metadata
        let engine = self.engine();
        let meta = {
            let indexes = engine.indexes.read();
            indexes
                .get(&table_name_lower)
                .and_then(|t| t.get(index_name))
                .cloned()
        };

        if let Some(meta) = meta {
            // Delete all index entries
            let prefix = encoding::make_index_prefix(meta.table_id, meta.index_id);
            let end = encoding::prefix_end_key(&prefix);

            self.with_txn(|txn| {
                let keys = self.runtime.block_on(async {
                    txn.scan_keys(prefix..end, u32::MAX)
                        .await
                        .map_err(from_tikv_error)
                })?;
                for key in keys {
                    self.runtime
                        .block_on(async { txn.delete(key).await.map_err(from_tikv_error) })?;
                }
                // Delete metadata key
                let meta_key = encoding::make_index_meta_key(&table_name_lower, index_name);
                self.runtime
                    .block_on(async { txn.delete(meta_key).await.map_err(from_tikv_error) })
            })?;

            // Remove from engine cache
            if let Some(table_indexes) = engine.indexes.write().get_mut(&table_name_lower) {
                table_indexes.remove(index_name);
            }
        }

        Ok(())
    }

    fn create_table_btree_index(
        &mut self,
        table_name: &str,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()> {
        let index_name = custom_name.map(|n| n.to_string()).unwrap_or_else(|| {
            format!(
                "idx_{}_{}",
                table_name.to_lowercase(),
                column_name.to_lowercase()
            )
        });
        self.create_table_index(
            table_name,
            &index_name,
            &[column_name.to_string()],
            is_unique,
        )
    }

    fn drop_table_btree_index(&mut self, table_name: &str, column_name: &str) -> Result<()> {
        let index_name = format!(
            "idx_{}_{}",
            table_name.to_lowercase(),
            column_name.to_lowercase()
        );
        self.drop_table_index(table_name, &index_name)
    }

    fn add_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()> {
        let table_name = table_name.to_lowercase();
        let (table_id, schema_arc) = self
            .schemas
            .get(&table_name)
            .ok_or_else(|| Error::TableNotFound(table_name.clone()))?;
        let table_id = *table_id;
        let mut schema = (**schema_arc).clone();

        schema.add_column(column)?;

        self.persist_schema(&table_name, table_id, &schema)
    }

    fn drop_table_column(&mut self, table_name: &str, column_name: &str) -> Result<()> {
        let table_name = table_name.to_lowercase();
        let (table_id, schema_arc) = self
            .schemas
            .get(&table_name)
            .ok_or_else(|| Error::TableNotFound(table_name.clone()))?;
        let table_id = *table_id;
        let mut schema = (**schema_arc).clone();

        schema.remove_column(column_name)?;

        self.persist_schema(&table_name, table_id, &schema)
    }

    fn rename_table_column(
        &mut self,
        table_name: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        let table_name = table_name.to_lowercase();
        let (table_id, schema_arc) = self
            .schemas
            .get(&table_name)
            .ok_or_else(|| Error::TableNotFound(table_name.clone()))?;
        let table_id = *table_id;
        let mut schema = (**schema_arc).clone();

        schema.rename_column(old_name, new_name)?;

        self.persist_schema(&table_name, table_id, &schema)
    }

    fn modify_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()> {
        let table_name = table_name.to_lowercase();
        let (table_id, schema_arc) = self
            .schemas
            .get(&table_name)
            .ok_or_else(|| Error::TableNotFound(table_name.clone()))?;
        let table_id = *table_id;
        let mut schema = (**schema_arc).clone();

        schema.modify_column(&column.name, Some(column.data_type), Some(column.nullable))?;

        self.persist_schema(&table_name, table_id, &schema)
    }

    fn select(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        _original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>> {
        let table = if self.isolation_level == IsolationLevel::ReadCommitted {
            // Read Committed: use a fresh snapshot for each SELECT
            self.get_table_fresh_snapshot(table_name)?
        } else {
            self.get_table(table_name)?
        };
        let col_refs: Vec<&str> = columns_to_fetch.iter().map(|s| s.as_str()).collect();
        table.select(&col_refs, expr)
    }

    fn select_with_aliases(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        aliases: &FxHashMap<String, String>,
        _original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>> {
        let table = if self.isolation_level == IsolationLevel::ReadCommitted {
            self.get_table_fresh_snapshot(table_name)?
        } else {
            self.get_table(table_name)?
        };
        let col_refs: Vec<&str> = columns_to_fetch.iter().map(|s| s.as_str()).collect();
        table.select_with_aliases(&col_refs, expr, aliases)
    }

    fn select_as_of(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        temporal_type: &str,
        temporal_value: i64,
        _original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>> {
        match temporal_type.to_uppercase().as_str() {
            "TIMESTAMP" => {
                // Convert stoolap timestamp (nanoseconds since epoch) to TiKV Timestamp
                // TiKV physical = milliseconds since epoch, logical = 0
                let millis = temporal_value / 1_000_000;
                let tikv_ts = tikv_client::TimestampExt::from_version(
                    (millis << 18) as u64, // physical << 18 + logical(0)
                );

                let table_name_lower = table_name.to_lowercase();
                let (table_id, schema) = self
                    .schemas
                    .get(&table_name_lower)
                    .ok_or_else(|| Error::TableNotFound(table_name.to_string()))?;

                let engine = self.engine();
                let client = engine.client()?;

                // Create a read-only snapshot at the historical timestamp
                let options = tikv_client::TransactionOptions::new_optimistic().read_only();
                let mut snapshot = client.snapshot(tikv_ts, options);

                // Scan all rows at the historical timestamp
                let prefix = encoding::make_data_prefix(*table_id);
                let end = encoding::prefix_end_key(&prefix);

                let pairs = self.runtime.block_on(async {
                    snapshot
                        .scan(prefix..end, u32::MAX)
                        .await
                        .map_err(from_tikv_error)
                })?;

                let col_refs: Vec<&str> = columns_to_fetch.iter().map(|s| s.as_str()).collect();
                let column_indices: Vec<usize> = {
                    let col_map: FxHashMap<String, usize> = schema
                        .columns
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (c.name.to_lowercase(), i))
                        .collect();
                    col_refs
                        .iter()
                        .filter_map(|name| col_map.get(&name.to_lowercase()).copied())
                        .collect()
                };

                let mut result_rows = Vec::new();
                for pair in pairs {
                    let values = encoding::deserialize_row(pair.value())?;
                    // Apply filter
                    let row = encoding::values_to_row(values);
                    if let Some(e) = expr {
                        match e.evaluate(&row) {
                            Ok(true) => {}
                            _ => continue,
                        }
                    }
                    let key_bytes: Vec<u8> = pair.key().to_owned().into();
                    let row_id = encoding::extract_row_id_from_data_key(&key_bytes);
                    result_rows.push((row_id, row));
                }

                let scanner = super::scanner::TiKVScanner::from_rows(result_rows, column_indices);
                let result_columns: Vec<String> = if columns_to_fetch.is_empty() {
                    schema.columns.iter().map(|c| c.name.clone()).collect()
                } else {
                    columns_to_fetch.to_vec()
                };

                Ok(Box::new(super::result::TiKVQueryResult::new(
                    result_columns,
                    scanner,
                )))
            }
            "TRANSACTION" => Err(Error::internal(
                "AS OF TRANSACTION is not supported with TiKV backend \
                 (TiKV transaction IDs are not compatible with stoolap transaction IDs). \
                 Use AS OF TIMESTAMP instead.",
            )),
            _ => Err(Error::internal(format!(
                "Unsupported temporal type: {}",
                temporal_type
            ))),
        }
    }
}
