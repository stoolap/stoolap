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
use crate::core::{Error, IsolationLevel, Result, Schema};
use crate::storage::config::Config;
use crate::storage::traits::view::ViewDefinition;
use crate::storage::traits::{Engine, Index, Transaction};

use super::error::from_tikv_error;
use super::transaction::TiKVTransaction;

/// Inner state of TiKV storage engine, shared with transactions and tables
pub struct TiKVEngineInner {
    /// TiKV transactional client
    pub(crate) client: RwLock<Option<tikv_client::TransactionClient>>,
    /// PD endpoints
    pub(crate) pd_endpoints: Vec<String>,
    /// Default isolation level
    pub(crate) isolation_level: RwLock<IsolationLevel>,
    /// Schema cache: table_name → (table_id, schema)
    pub(crate) schemas: RwLock<FxHashMap<String, (u64, CompactArc<Schema>)>>,
    /// Schema epoch for cache invalidation
    pub(crate) schema_epoch: AtomicU64,
    /// View definitions cache
    pub(crate) views: RwLock<FxHashMap<String, Arc<ViewDefinition>>>,
    /// Next table ID counter
    pub(crate) next_table_id: AtomicU64,
    /// Transaction ID counter (local, not TiKV's)
    pub(crate) next_txn_id: AtomicI64,
    /// Engine configuration
    pub(crate) config: RwLock<Config>,
    /// Whether the engine is open
    pub(crate) is_open: std::sync::atomic::AtomicBool,
    /// Index metadata cache: table_name → {index_name → IndexMetadata}
    pub(crate) indexes: RwLock<FxHashMap<String, FxHashMap<String, super::index::IndexMetadata>>>,
    /// Next index ID counter
    pub(crate) next_index_id: AtomicU64,
}

/// TiKV storage engine
pub struct TiKVEngine {
    /// Inner shared state
    pub(crate) inner: Arc<TiKVEngineInner>,
    /// Tokio runtime for async TiKV operations (not shared to ensure correct drop order)
    pub(crate) runtime: tokio::runtime::Runtime,
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

        let inner = Arc::new(TiKVEngineInner {
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
        });

        Ok(Self { inner, runtime })
    }

    /// Get a reference to the tokio runtime
    pub(crate) fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// Get the TiKV client (must be called while engine is open)
    pub(crate) fn client(&self) -> Result<tikv_client::TransactionClient> {
        let guard = self.inner.client.read();
        guard
            .clone()
            .ok_or_else(|| Error::internal("TiKV engine not open"))
    }

    /// Allocate a new transaction ID
    pub(crate) fn next_txn_id(&self) -> i64 {
        self.inner.next_txn_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Get a table's ID from the schema cache
    #[allow(dead_code)]
    pub(crate) fn get_table_id(&self, table_name: &str) -> Option<u64> {
        let schemas = self.inner.schemas.read();
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
                    let table_id_key = make_table_id_key(&table_name);
                    let table_id = txn
                        .get(table_id_key)
                        .await
                        .map_err(from_tikv_error)?
                        .map(|id_bytes| decode_u64(&id_bytes))
                        .unwrap_or(0);

                    if table_id > max_table_id {
                        max_table_id = table_id;
                    }

                    schemas.insert(table_name, (table_id, CompactArc::new(schema)));
                }

                // Scan all view keys
                let v_start = META_VIEW_PREFIX.to_vec();
                let v_end = prefix_end_key(&v_start);
                let v_pairs = txn
                    .scan(v_start..v_end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)?;

                let mut views = FxHashMap::default();
                for pair in v_pairs {
                    let key = pair.key().to_owned();
                    let key_bytes: Vec<u8> = key.into();
                    let view_name =
                        String::from_utf8_lossy(&key_bytes[META_VIEW_PREFIX.len()..]).to_string();
                    
                    // Use a placeholder ViewDefinition since we don't have TiKV serialization for it yet
                    // In a real implementation, ViewDefinition would have from_bytes/to_bytes
                    let view_def = ViewDefinition::new(&view_name, String::from_utf8_lossy(pair.value()).to_string());
                    views.insert(view_name, Arc::new(view_def));
                }

                // Scan all index metadata keys
                let i_start = META_INDEX_PREFIX.to_vec();
                let i_end = prefix_end_key(&i_start);
                let i_pairs = txn
                    .scan(i_start..i_end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)?;

                let mut indexes: FxHashMap<String, FxHashMap<String, super::index::IndexMetadata>> =
                    FxHashMap::default();
                let mut max_index_id = 0u64;
                for pair in i_pairs {
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
            self.inner.next_table_id.store(ntid, Ordering::Relaxed);
        } else {
            self.inner
                .next_table_id
                .store(max_table_id + 1, Ordering::Relaxed);
        }

        *self.inner.schemas.write() = schemas;
        *self.inner.views.write() = views;
        *self.inner.indexes.write() = indexes;
        self.inner
            .next_index_id
            .store(max_index_id + 1, Ordering::Relaxed);
        Ok(())
    }
}

impl Engine for TiKVEngine {
    fn open(&self) -> Result<()> {
        let client = self.runtime.block_on(async {
            tikv_client::TransactionClient::new(self.inner.pd_endpoints.clone())
                .await
                .map_err(from_tikv_error)
        })?;

        *self.inner.client.write() = Some(client);
        self.inner
            .is_open
            .store(true, std::sync::atomic::Ordering::Release);

        // Load existing schemas from TiKV
        self.load_schemas()?;

        Ok(())
    }

    fn close(&self) -> Result<()> {
        self.inner
            .is_open
            .store(false, std::sync::atomic::Ordering::Release);
        *self.inner.client.write() = None;
        Ok(())
    }

    fn begin_transaction(&self) -> Result<Box<dyn Transaction>> {
        self.begin_transaction_with_level(*self.inner.isolation_level.read())
    }

    fn begin_transaction_with_level(&self, level: IsolationLevel) -> Result<Box<dyn Transaction>> {
        if !self.inner.is_open.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Error::EngineNotOpen);
        }

        let client = self.client()?;
        let tikv_txn = self
            .runtime
            .block_on(async { client.begin_optimistic().await.map_err(from_tikv_error) })?;

        let txn_id = self.next_txn_id();
        let schemas_snapshot = self.inner.schemas.read().clone();

        Ok(Box::new(TiKVTransaction::new(
            txn_id,
            tikv_txn,
            level,
            self.runtime.handle().clone(),
            schemas_snapshot,
            Arc::clone(&self.inner),
        )))
    }

    fn path(&self) -> Option<&str> {
        None // TiKV is network-based, no local path
    }

    fn table_exists(&self, table_name: &str) -> Result<bool> {
        let name = table_name.to_lowercase();
        Ok(self.inner.schemas.read().contains_key(&name))
    }

    fn index_exists(&self, index_name: &str, table_name: &str) -> Result<bool> {
        let table_name = table_name.to_lowercase();
        let indexes = self.inner.indexes.read();
        if let Some(table_indexes) = indexes.get(&table_name) {
            return Ok(table_indexes.contains_key(index_name));
        }
        Ok(false)
    }

    fn get_index(
        &self,
        table_name: &str,
        index_name: &str,
    ) -> Result<Box<dyn crate::storage::traits::Index>> {
        let table_name_lower = table_name.to_lowercase();
        let indexes = self.inner.indexes.read();
        let meta = indexes
            .get(&table_name_lower)
            .and_then(|t| t.get(index_name))
            .ok_or_else(|| Error::IndexNotFound(index_name.to_string()))?;

        // Engine-level index access is transaction-less (for metadata only)
        let index = super::index::TiKVIndex::new(meta.clone(), None, self.runtime.handle().clone());
        Ok(Box::new(index))
    }

    fn get_table_schema(&self, table_name: &str) -> Result<CompactArc<Schema>> {
        let name = table_name.to_lowercase();
        let schemas = self.inner.schemas.read();
        let (_, schema) = schemas
            .get(&name)
            .ok_or_else(|| Error::TableNotFound(table_name.to_string()))?;
        Ok(schema.clone())
    }

    fn schema_epoch(&self) -> u64 {
        self.inner.schema_epoch.load(Ordering::Acquire)
    }

    fn list_table_indexes(&self, table_name: &str) -> Result<FxHashMap<String, String>> {
        let table_name_lower = table_name.to_lowercase();
        let indexes = self.inner.indexes.read();
        if let Some(table_indexes) = indexes.get(&table_name_lower) {
            Ok(table_indexes
                .iter()
                .map(|(name, _)| (name.clone(), "BTree".to_string()))
                .collect())
        } else {
            Ok(FxHashMap::default())
        }
    }

    fn get_all_indexes(&self, table_name: &str) -> Result<Vec<std::sync::Arc<dyn Index>>> {
        let table_name_lower = table_name.to_lowercase();
        let indexes = self.inner.indexes.read();
        if let Some(table_indexes) = indexes.get(&table_name_lower) {
            let mut result = Vec::new();
            for meta in table_indexes.values() {
                let index = super::index::TiKVIndex::new(meta.clone(), None, self.runtime.handle().clone());
                result.push(std::sync::Arc::new(index) as std::sync::Arc<dyn Index>);
            }
            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }

    fn get_isolation_level(&self) -> IsolationLevel {
        *self.inner.isolation_level.read()
    }

    fn set_isolation_level(&self, level: IsolationLevel) -> Result<()> {
        *self.inner.isolation_level.write() = level;
        Ok(())
    }

    fn get_config(&self) -> Config {
        self.inner.config.read().clone()
    }

    fn update_config(&self, config: Config) -> Result<()> {
        *self.inner.config.write() = config;
        Ok(())
    }

    fn create_snapshot(&self) -> Result<()> {
        // TiKV handles snapshots internally
        Ok(())
    }

    fn get_view_lowercase(&self, name_lower: &str) -> Result<Option<Arc<ViewDefinition>>> {
        let views = self.inner.views.read();
        Ok(views.get(name_lower).cloned())
    }

    fn list_views(&self) -> Result<Vec<String>> {
        let views = self.inner.views.read();
        Ok(views.values().map(|v| v.original_name.clone()).collect())
    }

    fn create_view(&self, name: &str, query: &str, or_replace: bool) -> Result<()> {
        let name_lower = name.to_lowercase();
        let mut views = self.inner.views.write();
        if views.contains_key(&name_lower) && !or_replace {
            return Err(Error::ViewAlreadyExists(name.to_string()));
        }

        let view_def = Arc::new(ViewDefinition::new(name, query.to_string()));
        views.insert(name_lower, view_def);
        Ok(())
    }

    fn drop_view(&self, name: &str) -> Result<()> {
        let name_lower = name.to_lowercase();
        let mut views = self.inner.views.write();
        if views.remove(&name_lower).is_none() {
            return Err(Error::ViewNotFound(name.to_string()));
        }
        Ok(())
    }

    fn view_exists(&self, name: &str) -> Result<bool> {
        let name_lower = name.to_lowercase();
        Ok(self.inner.views.read().contains_key(&name_lower))
    }

    #[allow(clippy::too_many_arguments)]
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

    fn create_table_direct(&self, schema: Schema) -> Result<Schema> {
        let table_name = schema.table_name.clone();
        let table_name_lower = table_name.to_lowercase();

        // Check existence
        if self.inner.schemas.read().contains_key(&table_name_lower) {
            return Err(Error::TableAlreadyExists(table_name));
        }

        // Use a transaction to perform DDL
        let mut txn = self.begin_transaction()?;
        let _ = txn.create_table(&table_name, schema.clone())?;
        txn.commit()?;

        Ok(schema)
    }

    fn drop_table_direct(&self, name: &str) -> Result<()> {
        let mut txn = self.begin_transaction()?;
        txn.drop_table(name)?;
        txn.commit()?;
        Ok(())
    }
}
