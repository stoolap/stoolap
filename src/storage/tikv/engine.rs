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

//! TiKV storage engine implementation

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

use crate::common::CompactArc;
use crate::core::{Error, ForeignKeyConstraint, IsolationLevel, Result, Schema};
use crate::storage::config::Config;
use crate::storage::traits::view::ViewDefinition;
use crate::storage::traits::{Engine, Index, Table, Transaction};

use super::encoding;
use super::error::from_tikv_error;
use super::transaction::TiKVTransaction;

/// TiKV storage engine
pub struct TiKVEngine {
    /// Tokio runtime for async TiKV operations
    runtime: tokio::runtime::Runtime,
    /// TiKV transactional client
    client: RwLock<Option<tikv_client::TransactionClient>>,
    /// PD endpoints
    pd_endpoints: Vec<String>,
    /// Default isolation level
    isolation_level: RwLock<IsolationLevel>,
    /// Schema cache: table_name → (table_id, schema)
    pub(crate) schemas: RwLock<FxHashMap<String, (u64, CompactArc<Schema>)>>,
    /// Schema epoch for cache invalidation
    pub(crate) schema_epoch: AtomicU64,
    /// View definitions cache
    views: RwLock<FxHashMap<String, Arc<ViewDefinition>>>,
    /// Next table ID counter
    pub(crate) next_table_id: AtomicU64,
    /// Transaction ID counter (local, not TiKV's)
    next_txn_id: AtomicI64,
    /// Engine configuration
    config: RwLock<Config>,
    /// Whether the engine is open
    is_open: std::sync::atomic::AtomicBool,
    /// Index metadata cache: table_name → {index_name → IndexMetadata}
    pub(crate) indexes: RwLock<FxHashMap<String, FxHashMap<String, super::index::IndexMetadata>>>,
    /// Next index ID counter
    pub(crate) next_index_id: AtomicU64,
}

impl TiKVEngine {
    /// Create a new TiKV engine from a DSN path (comma-separated PD endpoints)
    pub fn new(dsn_path: &str) -> Result<Self> {
        let pd_endpoints: Vec<String> = dsn_path
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if pd_endpoints.is_empty() {
            return Err(Error::internal(
                "TiKV DSN must contain at least one PD endpoint (e.g., tikv://pd1:2379)",
            ));
        }

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::internal(format!("Failed to create tokio runtime: {e}")))?;

        Ok(Self {
            runtime,
            client: RwLock::new(None),
            pd_endpoints,
            isolation_level: RwLock::new(IsolationLevel::SnapshotIsolation),
            schemas: RwLock::new(FxHashMap::default()),
            schema_epoch: AtomicU64::new(0),
            views: RwLock::new(FxHashMap::default()),
            next_table_id: AtomicU64::new(1),
            next_txn_id: AtomicI64::new(1),
            config: RwLock::new(Config::default()),
            is_open: std::sync::atomic::AtomicBool::new(false),
            indexes: RwLock::new(FxHashMap::default()),
            next_index_id: AtomicU64::new(1),
        })
    }

    /// Get a reference to the tokio runtime
    #[allow(dead_code)]
    pub(crate) fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// Get the TiKV client (must be called while engine is open)
    pub(crate) fn client(&self) -> Result<tikv_client::TransactionClient> {
        let guard = self.client.read();
        guard
            .clone()
            .ok_or_else(|| Error::internal("TiKV engine not open"))
    }

    /// Allocate a new transaction ID
    pub(crate) fn next_txn_id(&self) -> i64 {
        self.next_txn_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Get a table's ID from the schema cache
    #[allow(dead_code)]
    pub(crate) fn get_table_id(&self, table_name: &str) -> Option<u64> {
        let schemas = self.schemas.read();
        schemas.get(table_name).map(|(id, _)| *id)
    }

    /// Load schemas from TiKV on startup
    fn load_schemas(&self) -> Result<()> {
        use super::encoding::*;

        let client = self.client()?;
        let mut max_table_id = 0u64;

        let (schemas, views, next_tid, indexes, max_index_id) = self.runtime.block_on(async {
            let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;

            let result: Result<_> = async {
                // Scan all schema keys
                let start = META_SCHEMA_PREFIX.to_vec();
                let end = prefix_end_key(&start);
                let pairs = txn
                    .scan(start..end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)?;

                let mut schemas = FxHashMap::default();
                for pair in pairs {
                    let key = pair.key().to_owned();
                    let key_bytes: Vec<u8> = key.into();
                    let table_name =
                        String::from_utf8_lossy(&key_bytes[META_SCHEMA_PREFIX.len()..]).to_string();
                    let schema = deserialize_schema(pair.value())?;

                    // Get table ID
                    let tid_key = make_table_id_key(&table_name);
                    let tid =
                        if let Some(tid_bytes) = txn.get(tid_key).await.map_err(from_tikv_error)? {
                            decode_u64(&tid_bytes)
                        } else {
                            continue;
                        };

                    if tid > max_table_id {
                        max_table_id = tid;
                    }

                    schemas.insert(table_name, (tid, CompactArc::new(schema)));
                }

                // Load views
                let view_start = META_VIEW_PREFIX.to_vec();
                let view_end = prefix_end_key(&view_start);
                let view_pairs = txn
                    .scan(view_start..view_end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)?;

                let mut views = FxHashMap::default();
                for pair in view_pairs {
                    let key = pair.key().to_owned();
                    let key_bytes: Vec<u8> = key.into();
                    let view_name =
                        String::from_utf8_lossy(&key_bytes[META_VIEW_PREFIX.len()..]).to_string();
                    let query = String::from_utf8_lossy(pair.value()).to_string();
                    views.insert(
                        view_name.clone(),
                        Arc::new(ViewDefinition::new(&view_name, query)),
                    );
                }

                // Load index metadata
                let idx_start = META_INDEX_PREFIX.to_vec();
                let idx_end = prefix_end_key(&idx_start);
                let idx_pairs = txn
                    .scan(idx_start..idx_end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)?;

                let mut indexes: FxHashMap<String, FxHashMap<String, super::index::IndexMetadata>> =
                    FxHashMap::default();
                let mut max_index_id = 0u64;
                for pair in idx_pairs {
                    if let Ok(meta) = super::index::IndexMetadata::from_bytes(pair.value()) {
                        if meta.index_id > max_index_id {
                            max_index_id = meta.index_id;
                        }
                        indexes
                            .entry(meta.table_name.clone())
                            .or_default()
                            .insert(meta.name.clone(), meta);
                    }
                }

                // Load next table ID
                let next_tid = txn
                    .get(META_NEXT_TABLE_ID.to_vec())
                    .await
                    .map_err(from_tikv_error)?
                    .map(|ntid_bytes| decode_u64(&ntid_bytes));

                Ok((schemas, views, next_tid, indexes, max_index_id))
            }
            .await;

            // Always clean up the transaction
            match result {
                Ok(data) => {
                    // Read-only: just rollback (no writes to persist)
                    let _ = txn.rollback().await;
                    Ok(data)
                }
                Err(e) => {
                    let _ = txn.rollback().await;
                    Err(e)
                }
            }
        })?;

        if let Some(ntid) = next_tid {
            self.next_table_id.store(ntid, Ordering::Relaxed);
        } else {
            self.next_table_id
                .store(max_table_id + 1, Ordering::Relaxed);
        }

        *self.schemas.write() = schemas;
        *self.views.write() = views;
        *self.indexes.write() = indexes;
        self.next_index_id
            .store(max_index_id + 1, Ordering::Relaxed);
        Ok(())
    }
}

impl Engine for TiKVEngine {
    fn open(&self) -> Result<()> {
        let client = self.runtime.block_on(async {
            tikv_client::TransactionClient::new(self.pd_endpoints.clone())
                .await
                .map_err(from_tikv_error)
        })?;

        *self.client.write() = Some(client);
        self.is_open
            .store(true, std::sync::atomic::Ordering::Release);

        // Load existing schemas from TiKV
        self.load_schemas()?;

        Ok(())
    }

    fn close(&self) -> Result<()> {
        self.is_open
            .store(false, std::sync::atomic::Ordering::Release);
        *self.client.write() = None;
        Ok(())
    }

    fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        self.begin_transaction_with_level(*self.isolation_level.read())
    }

    fn begin_transaction_with_level(&self, level: IsolationLevel) -> Result<Box<dyn Transaction>> {
        if !self.is_open.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Error::EngineNotOpen);
        }

        let client = self.client()?;
        let tikv_txn = self
            .runtime
            .block_on(async { client.begin_optimistic().await.map_err(from_tikv_error) })?;

        let txn_id = self.next_txn_id();
        let schemas_snapshot = self.schemas.read().clone();

        Ok(Box::new(TiKVTransaction::new(
            txn_id,
            tikv_txn,
            level,
            self.runtime.handle().clone(),
            schemas_snapshot,
            self,
        )))
    }

    fn path(&self) -> Option<&str> {
        None // TiKV is network-based, no local path
    }

    fn table_exists(&self, table_name: &str) -> Result<bool> {
        let name = table_name.to_lowercase();
        Ok(self.schemas.read().contains_key(&name))
    }

    fn index_exists(&self, index_name: &str, table_name: &str) -> Result<bool> {
        let table = table_name.to_lowercase();
        let indexes = self.indexes.read();
        Ok(indexes
            .get(&table)
            .is_some_and(|table_indexes| table_indexes.contains_key(index_name)))
    }

    fn get_index(&self, table_name: &str, index_name: &str) -> Result<Box<dyn Index>> {
        let table = table_name.to_lowercase();
        let indexes = self.indexes.read();
        let meta = indexes
            .get(&table)
            .and_then(|t| t.get(index_name))
            .ok_or_else(|| {
                Error::internal(format!(
                    "Index '{}' not found on table '{}'",
                    index_name, table_name
                ))
            })?
            .clone();

        // Create a read-only transaction for index operations
        let client = self.client()?;
        let tikv_txn = self
            .runtime
            .block_on(async { client.begin_optimistic().await.map_err(from_tikv_error) })?;

        Ok(Box::new(super::index::TiKVIndex::new(
            meta,
            Arc::new(parking_lot::Mutex::new(Some(tikv_txn))),
            self.runtime.handle().clone(),
        )))
    }

    fn get_table_schema(&self, table_name: &str) -> Result<CompactArc<Schema>> {
        let name = table_name.to_lowercase();
        let schemas = self.schemas.read();
        schemas
            .get(&name)
            .map(|(_, schema)| schema.clone())
            .ok_or_else(|| Error::TableNotFound(table_name.to_string()))
    }

    fn schema_epoch(&self) -> u64 {
        self.schema_epoch.load(Ordering::Acquire)
    }

    fn list_table_indexes(&self, table_name: &str) -> Result<FxHashMap<String, String>> {
        let table = table_name.to_lowercase();
        let indexes = self.indexes.read();
        let mut result = FxHashMap::default();
        if let Some(table_indexes) = indexes.get(&table) {
            for (name, meta) in table_indexes {
                result.insert(name.clone(), meta.column_names.join(","));
            }
        }
        Ok(result)
    }

    fn get_all_indexes(&self, table_name: &str) -> Result<Vec<Arc<dyn Index>>> {
        let table = table_name.to_lowercase();
        let indexes = self.indexes.read();
        let mut result: Vec<Arc<dyn Index>> = Vec::new();

        if let Some(table_indexes) = indexes.get(&table) {
            // Create a shared read-only transaction for all indexes
            let client = self.client()?;
            let tikv_txn = self
                .runtime
                .block_on(async { client.begin_optimistic().await.map_err(from_tikv_error) })?;
            let shared_txn = Arc::new(parking_lot::Mutex::new(Some(tikv_txn)));

            for meta in table_indexes.values() {
                result.push(Arc::new(super::index::TiKVIndex::new(
                    meta.clone(),
                    Arc::clone(&shared_txn),
                    self.runtime.handle().clone(),
                )));
            }
        }
        Ok(result)
    }

    fn get_isolation_level(&self) -> IsolationLevel {
        *self.isolation_level.read()
    }

    fn set_isolation_level(&self, level: IsolationLevel) -> Result<()> {
        *self.isolation_level.write() = level;
        Ok(())
    }

    fn get_config(&self) -> Config {
        self.config.read().clone()
    }

    fn update_config(&self, config: Config) -> Result<()> {
        *self.config.write() = config;
        Ok(())
    }

    fn create_snapshot(&self) -> Result<()> {
        Ok(())
    }

    fn get_view_lowercase(&self, name_lower: &str) -> Result<Option<Arc<ViewDefinition>>> {
        Ok(self.views.read().get(name_lower).cloned())
    }

    fn list_views(&self) -> Result<Vec<String>> {
        Ok(self
            .views
            .read()
            .values()
            .map(|v| v.original_name.clone())
            .collect())
    }

    fn create_view(&self, name: &str, query: &str, or_replace: bool) -> Result<()> {
        let name_lower = name.to_lowercase();
        let mut views = self.views.write();
        if views.contains_key(&name_lower) && !or_replace {
            return Err(Error::ViewAlreadyExists(name.to_string()));
        }
        let view_def = Arc::new(ViewDefinition::new(name, query.to_string()));
        views.insert(name_lower.clone(), view_def);

        // Persist to TiKV
        let client = self.client()?;
        let key = encoding::make_view_key(&name_lower);
        let value = query.as_bytes().to_vec();
        self.runtime.block_on(async {
            let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;
            match txn.put(key, value).await.map_err(from_tikv_error) {
                Ok(()) => txn.commit().await.map_err(from_tikv_error),
                Err(e) => {
                    let _ = txn.rollback().await;
                    Err(e)
                }
            }
        })?;
        Ok(())
    }

    fn drop_view(&self, name: &str) -> Result<()> {
        let name_lower = name.to_lowercase();
        self.views.write().remove(&name_lower);

        // Remove from TiKV
        let client = self.client()?;
        let key = encoding::make_view_key(&name_lower);
        self.runtime.block_on(async {
            let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;
            match txn.delete(key).await.map_err(from_tikv_error) {
                Ok(()) => txn.commit().await.map_err(from_tikv_error),
                Err(e) => {
                    let _ = txn.rollback().await;
                    Err(e)
                }
            }
        })?;
        Ok(())
    }

    fn view_exists(&self, name: &str) -> Result<bool> {
        let name_lower = name.to_lowercase();
        Ok(self.views.read().contains_key(&name_lower))
    }

    fn create_table_direct(&self, schema: Schema) -> Result<Schema> {
        let table_name = schema.table_name_lower.clone();

        // Check if table already exists
        if self.schemas.read().contains_key(&table_name) {
            return Err(Error::TableAlreadyExists(table_name));
        }

        // Allocate table ID
        let table_id = self.next_table_id.fetch_add(1, Ordering::Relaxed);

        // Persist to TiKV
        let client = self.client()?;
        let schema_bytes = encoding::serialize_schema(&schema)?;

        let table_name_clone = table_name.clone();
        self.runtime.block_on(async {
            let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;

            let result: Result<()> = async {
                txn.put(encoding::make_schema_key(&table_name_clone), schema_bytes)
                    .await
                    .map_err(from_tikv_error)?;

                txn.put(
                    encoding::make_table_id_key(&table_name_clone),
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
                    encoding::make_next_row_id_key(&table_name_clone),
                    encoding::encode_i64(1).to_vec(),
                )
                .await
                .map_err(from_tikv_error)?;

                Ok(())
            }
            .await;

            match result {
                Ok(()) => txn.commit().await.map_err(from_tikv_error),
                Err(e) => {
                    let _ = txn.rollback().await;
                    Err(e)
                }
            }
        })?;

        // Update cache
        let return_schema = schema.clone();
        self.schemas
            .write()
            .insert(table_name, (table_id, CompactArc::new(schema)));
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(return_schema)
    }

    fn drop_table_direct(&self, name: &str) -> Result<()> {
        let table_name = name.to_lowercase();

        // Get table ID before removing
        let table_id = {
            let schemas = self.schemas.read();
            schemas
                .get(&table_name)
                .map(|(id, _)| *id)
                .ok_or_else(|| Error::TableNotFound(name.to_string()))?
        };

        // Remove all data and metadata from TiKV
        let client = self.client()?;
        let table_name_clone = table_name.clone();
        self.runtime.block_on(async {
            let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;

            let result: Result<()> = async {
                // Delete schema metadata
                txn.delete(encoding::make_schema_key(&table_name_clone))
                    .await
                    .map_err(from_tikv_error)?;
                txn.delete(encoding::make_table_id_key(&table_name_clone))
                    .await
                    .map_err(from_tikv_error)?;
                txn.delete(encoding::make_next_row_id_key(&table_name_clone))
                    .await
                    .map_err(from_tikv_error)?;

                // Delete all data rows
                let data_start = encoding::make_data_prefix(table_id);
                let data_end = encoding::prefix_end_key(&data_start);
                let pairs = txn
                    .scan_keys(data_start..data_end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)?;
                for key in pairs {
                    txn.delete(key).await.map_err(from_tikv_error)?;
                }

                // Delete all index entries
                let idx_start = encoding::make_index_prefix(table_id, 0);
                let idx_end = encoding::prefix_end_key(&{
                    let mut p = Vec::with_capacity(9);
                    p.push(encoding::INDEX_PREFIX);
                    p.extend_from_slice(&encoding::encode_u64(table_id + 1));
                    p
                });
                let idx_keys = txn
                    .scan_keys(idx_start..idx_end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)?;
                for key in idx_keys {
                    txn.delete(key).await.map_err(from_tikv_error)?;
                }

                Ok(())
            }
            .await;

            match result {
                Ok(()) => txn.commit().await.map_err(from_tikv_error),
                Err(e) => {
                    let _ = txn.rollback().await;
                    Err(e)
                }
            }
        })?;

        // Update cache
        self.schemas.write().remove(&table_name);
        self.schema_epoch.fetch_add(1, Ordering::Release);

        Ok(())
    }

    fn refresh_schema_cache(&self, table_name: &str) -> Result<()> {
        let table_name_lower = table_name.to_lowercase();
        let client = self.client()?;

        let result = self.runtime.block_on(async {
            let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;

            let result: Result<Option<(String, u64, _)>> = async {
                let schema_key = encoding::make_schema_key(&table_name_lower);
                if let Some(schema_bytes) = txn.get(schema_key).await.map_err(from_tikv_error)? {
                    let schema = encoding::deserialize_schema(&schema_bytes)?;

                    let tid_key = encoding::make_table_id_key(&table_name_lower);
                    let table_id =
                        if let Some(tid_bytes) = txn.get(tid_key).await.map_err(from_tikv_error)? {
                            encoding::decode_u64(&tid_bytes)
                        } else {
                            return Err(Error::internal("table ID not found"));
                        };

                    Ok(Some((table_name_lower.clone(), table_id, schema)))
                } else {
                    Ok(None)
                }
            }
            .await;

            // Always clean up the transaction (read-only)
            let _ = txn.rollback().await;
            result
        })?;

        if let Some((name, table_id, schema)) = result {
            self.schemas
                .write()
                .insert(name, (table_id, CompactArc::new(schema)));
        }

        self.schema_epoch.fetch_add(1, Ordering::Release);
        Ok(())
    }

    fn get_all_schemas(&self) -> Vec<CompactArc<Schema>> {
        self.schemas
            .read()
            .values()
            .map(|(_, s)| s.clone())
            .collect()
    }

    fn find_referencing_fks(&self, parent_table: &str) -> Arc<Vec<(String, ForeignKeyConstraint)>> {
        let parent_lower = parent_table.to_lowercase();
        let schemas = self.schemas.read();
        let mut result = Vec::new();

        for (table_name, (_, schema)) in schemas.iter() {
            for fk in &schema.foreign_keys {
                if fk.referenced_table == parent_lower {
                    result.push((table_name.clone(), fk.clone()));
                }
            }
        }

        Arc::new(result)
    }

    fn get_table_for_txn(&self, _txn_id: i64, table_name: &str) -> Result<Box<dyn Table>> {
        let name = table_name.to_lowercase();
        let schemas = self.schemas.read();
        let (table_id, schema) = schemas
            .get(&name)
            .ok_or_else(|| Error::TableNotFound(table_name.to_string()))?;

        // Create a new TiKV transaction for this table access
        let client = self.client()?;
        let tikv_txn = self
            .runtime
            .block_on(async { client.begin_optimistic().await.map_err(from_tikv_error) })?;

        let table = super::table::TiKVTable::new(
            *table_id,
            name,
            (**schema).clone(),
            _txn_id,
            Arc::new(parking_lot::Mutex::new(Some(tikv_txn))),
            self.runtime.handle().clone(),
        )
        .with_engine(self);
        Ok(Box::new(table))
    }

    fn fetch_rows_by_ids(&self, table_name: &str, row_ids: &[i64]) -> Result<crate::core::RowVec> {
        let name = table_name.to_lowercase();
        let schemas = self.schemas.read();
        let (table_id, _schema) = schemas
            .get(&name)
            .ok_or_else(|| Error::TableNotFound(table_name.to_string()))?;
        let table_id = *table_id;
        drop(schemas);

        let client = self.client()?;
        self.runtime.block_on(async {
            let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;

            let mut results = crate::core::RowVec::with_capacity(row_ids.len());
            for &row_id in row_ids {
                let key = super::encoding::make_data_key(table_id, row_id);
                if let Some(bytes) = txn.get(key).await.map_err(from_tikv_error)? {
                    let values = super::encoding::deserialize_row(&bytes)?;
                    let row = super::encoding::values_to_row(values);
                    results.push((row_id, row));
                }
            }

            let _ = txn.rollback().await;
            Ok(results)
        })
    }

    #[allow(clippy::type_complexity)]
    fn get_row_fetcher(
        &self,
        table_name: &str,
    ) -> Result<Box<dyn Fn(&[i64]) -> crate::core::RowVec + Send + Sync>> {
        let name = table_name.to_lowercase();
        let schemas = self.schemas.read();
        let (table_id, _schema) = schemas
            .get(&name)
            .ok_or_else(|| Error::TableNotFound(table_name.to_string()))?;
        let table_id = *table_id;
        drop(schemas);

        let client = self.client()?;
        let runtime = self.runtime.handle().clone();

        Ok(Box::new(move |row_ids: &[i64]| -> crate::core::RowVec {
            let mut results = crate::core::RowVec::with_capacity(row_ids.len());
            let fetch_result = runtime.block_on(async {
                let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;
                for &row_id in row_ids {
                    let key = super::encoding::make_data_key(table_id, row_id);
                    if let Some(bytes) = txn.get(key).await.map_err(from_tikv_error)? {
                        if let Ok(values) = super::encoding::deserialize_row(&bytes) {
                            let row = super::encoding::values_to_row(values);
                            results.push((row_id, row));
                        }
                    }
                }
                let _ = txn.rollback().await;
                Ok::<_, Error>(())
            });
            let _ = fetch_result;
            results
        }))
    }

    #[allow(clippy::type_complexity)]
    fn get_row_counter(
        &self,
        table_name: &str,
    ) -> Result<Box<dyn Fn(&[i64]) -> usize + Send + Sync>> {
        let name = table_name.to_lowercase();
        let schemas = self.schemas.read();
        let (table_id, _schema) = schemas
            .get(&name)
            .ok_or_else(|| Error::TableNotFound(table_name.to_string()))?;
        let table_id = *table_id;
        drop(schemas);

        let client = self.client()?;
        let runtime = self.runtime.handle().clone();

        Ok(Box::new(move |row_ids: &[i64]| -> usize {
            let count_result = runtime.block_on(async {
                let mut txn = client.begin_optimistic().await.map_err(from_tikv_error)?;
                let mut count = 0usize;
                for &row_id in row_ids {
                    let key = super::encoding::make_data_key(table_id, row_id);
                    if txn.key_exists(key).await.map_err(from_tikv_error)? {
                        count += 1;
                    }
                }
                let _ = txn.rollback().await;
                Ok::<_, Error>(count)
            });
            count_result.unwrap_or(0)
        }))
    }

    fn record_create_index(
        &self,
        _table_name: &str,
        _index_name: &str,
        _column_names: &[String],
        _is_unique: bool,
        _index_type: crate::core::IndexType,
        _hnsw_m: Option<u16>,
        _hnsw_ef_construction: Option<u16>,
        _hnsw_ef_search: Option<u16>,
        _hnsw_distance_metric: Option<u8>,
    ) -> Result<()> {
        // Handled by TiKVTable::create_index
        Ok(())
    }

    fn record_drop_index(&self, _table_name: &str, _index_name: &str) -> Result<()> {
        // Handled by TiKVTable::drop_index
        Ok(())
    }
}
