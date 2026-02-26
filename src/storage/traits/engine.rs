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

//! Engine trait for the storage engine
//!

use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::common::CompactArc;
use crate::core::{ForeignKeyConstraint, IsolationLevel, Result, RowVec, Schema};
use crate::storage::config::Config;
use crate::storage::traits::view::ViewDefinition;
use crate::storage::traits::{Index, Table, Transaction};

/// Engine represents the storage engine
///
/// This is the main entry point for interacting with the database.
/// It manages transactions, tables, indexes, and persistence.
///
/// # Example
///
/// ```ignore
/// let config = Config::with_path("/tmp/mydb");
/// let mut engine = MvccEngine::new(config);
/// engine.open()?;
///
/// let tx = engine.begin_transaction()?;
/// // ... perform operations ...
/// tx.commit()?;
///
/// engine.close()?;
/// ```
pub trait Engine: Send + Sync {
    /// Opens the storage engine
    ///
    /// This initializes the engine, opens the database path (if any),
    /// recovers from WAL, and loads existing data.
    /// Uses interior mutability so the engine can be wrapped in Arc.
    fn open(&self) -> Result<()>;

    /// Closes the storage engine
    ///
    /// This flushes pending writes, creates a final snapshot if needed,
    /// and releases all resources.
    /// Uses interior mutability so the engine can be wrapped in Arc.
    fn close(&self) -> Result<()>;

    /// Begins a new transaction
    ///
    /// The transaction will use the engine's default isolation level.
    fn begin_transaction(&self) -> Result<Box<dyn Transaction>>;

    /// Begins a new transaction with a specific isolation level
    ///
    /// # Arguments
    /// * `level` - The isolation level for the transaction
    fn begin_transaction_with_level(&self, level: IsolationLevel) -> Result<Box<dyn Transaction>>;

    /// Returns the path to the database directory
    ///
    /// Returns `None` if operating in memory-only mode.
    fn path(&self) -> Option<&str>;

    /// Checks if a table exists
    fn table_exists(&self, table_name: &str) -> Result<bool>;

    /// Checks if an index exists
    fn index_exists(&self, index_name: &str, table_name: &str) -> Result<bool>;

    /// Gets an index by name
    fn get_index(&self, table_name: &str, index_name: &str) -> Result<Box<dyn Index>>;

    /// Gets the schema for a table
    ///
    /// Returns an Arc to avoid cloning the schema on every access.
    /// This is a critical optimization for hot paths like PK lookups.
    fn get_table_schema(&self, table_name: &str) -> Result<CompactArc<Schema>>;

    /// Gets the current schema epoch
    ///
    /// This is a monotonically increasing counter that increments on any
    /// CREATE TABLE, ALTER TABLE, or DROP TABLE operation. Used for fast
    /// cache invalidation without HashMap lookup (~1ns vs ~7ns).
    fn schema_epoch(&self) -> u64;

    /// Lists all indexes for a table
    ///
    /// Returns a map from index name to index type string.
    fn list_table_indexes(&self, table_name: &str) -> Result<FxHashMap<String, String>>;

    /// Gets all index objects for a table
    fn get_all_indexes(&self, table_name: &str) -> Result<Vec<std::sync::Arc<dyn Index>>>;

    /// Gets the current default isolation level
    fn get_isolation_level(&self) -> IsolationLevel;

    /// Sets the default isolation level for new transactions
    /// Uses interior mutability so the engine can be wrapped in Arc.
    fn set_isolation_level(&self, level: IsolationLevel) -> Result<()>;

    /// Gets the current engine configuration
    ///
    /// Returns a clone of the configuration to avoid lifetime issues with internal locks.
    fn get_config(&self) -> Config;

    /// Updates the engine configuration
    ///
    /// Note: Some configuration changes may require a restart to take effect.
    /// Uses interior mutability so the engine can be wrapped in Arc.
    fn update_config(&self, config: Config) -> Result<()>;

    /// Manually triggers snapshot creation for all tables
    ///
    /// This is useful for creating a consistent backup point.
    fn create_snapshot(&self) -> Result<()>;

    /// Start background tasks (e.g., cleanup, compaction)
    ///
    /// Called after the engine is opened and wrapped in Arc.
    fn start_background_tasks(&self) -> Result<()> {
        Ok(())
    }

    /// Record an index creation operation to WAL for persistence
    ///
    /// This should be called by the executor after creating an index to ensure
    /// the index is recreated on recovery.
    #[allow(clippy::too_many_arguments)]
    fn record_create_index(
        &self,
        table_name: &str,
        index_name: &str,
        column_names: &[String],
        is_unique: bool,
        index_type: crate::core::IndexType,
        hnsw_m: Option<u16>,
        hnsw_ef_construction: Option<u16>,
        hnsw_ef_search: Option<u16>,
        hnsw_distance_metric: Option<u8>,
    ) -> Result<()> {
        // Default implementation does nothing (for in-memory engines)
        let _ = (
            table_name,
            index_name,
            column_names,
            is_unique,
            index_type,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
            hnsw_distance_metric,
        );
        Ok(())
    }

    /// Record an index drop operation to WAL for persistence
    fn record_drop_index(&self, table_name: &str, index_name: &str) -> Result<()> {
        // Default implementation does nothing
        let _ = (table_name, index_name);
        Ok(())
    }

    /// Record ALTER TABLE ADD COLUMN operation to WAL for persistence
    fn record_alter_table_add_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: crate::core::DataType,
        nullable: bool,
        default_expr: Option<&str>,
        vector_dimensions: u16,
    ) -> Result<()> {
        // Default implementation does nothing (for in-memory engines)
        let _ = (
            table_name,
            column_name,
            data_type,
            nullable,
            default_expr,
            vector_dimensions,
        );
        Ok(())
    }

    /// Record ALTER TABLE DROP COLUMN operation to WAL for persistence
    fn record_alter_table_drop_column(&self, table_name: &str, column_name: &str) -> Result<()> {
        // Default implementation does nothing
        let _ = (table_name, column_name);
        Ok(())
    }

    /// Record ALTER TABLE RENAME COLUMN operation to WAL for persistence
    fn record_alter_table_rename_column(
        &self,
        table_name: &str,
        old_column_name: &str,
        new_column_name: &str,
    ) -> Result<()> {
        // Default implementation does nothing
        let _ = (table_name, old_column_name, new_column_name);
        Ok(())
    }

    /// Record ALTER TABLE MODIFY COLUMN operation to WAL for persistence
    fn record_alter_table_modify_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: crate::core::DataType,
        nullable: bool,
        vector_dimensions: u16,
    ) -> Result<()> {
        // Default implementation does nothing
        let _ = (
            table_name,
            column_name,
            data_type,
            nullable,
            vector_dimensions,
        );
        Ok(())
    }

    /// Record ALTER TABLE RENAME TO operation to WAL for persistence
    fn record_alter_table_rename(&self, old_table_name: &str, new_table_name: &str) -> Result<()> {
        // Default implementation does nothing
        let _ = (old_table_name, new_table_name);
        Ok(())
    }

    /// Record TRUNCATE TABLE operation to WAL for persistence
    fn record_truncate_table(&self, table_name: &str) -> Result<()> {
        // Default implementation does nothing (for in-memory engines)
        let _ = table_name;
        Ok(())
    }

    /// Fetch rows by IDs directly from storage without creating a full transaction.
    ///
    /// This is an optimization for EXISTS subquery evaluation where we only need
    /// to check if rows exist and evaluate predicates. It avoids the ~2-5μs overhead
    /// of creating a new transaction per EXISTS probe.
    ///
    /// The returned rows represent the latest committed state visible to any reader.
    fn fetch_rows_by_ids(&self, table_name: &str, row_ids: &[i64]) -> Result<RowVec> {
        // Default implementation: fall back to creating a transaction
        // Concrete implementations can override for better performance
        let _ = (table_name, row_ids);
        Err(crate::core::Error::internal(
            "fetch_rows_by_ids not supported by this engine",
        ))
    }

    /// Get a cached row fetcher for a table.
    ///
    /// This returns a function that can be called repeatedly to fetch rows without
    /// the overhead of looking up the table each time. This is useful for EXISTS
    /// subquery evaluation where we probe the same table many times.
    #[allow(clippy::type_complexity)]
    fn get_row_fetcher(
        &self,
        table_name: &str,
    ) -> Result<Box<dyn Fn(&[i64]) -> RowVec + Send + Sync>> {
        // Default implementation: fall back to fetch_rows_by_ids
        let _ = table_name;
        Err(crate::core::Error::internal(
            "get_row_fetcher not supported by this engine",
        ))
    }

    /// Get a count-only function for counting visible rows by their IDs.
    /// This is optimized for COUNT(*) subqueries where we don't need the actual row data.
    #[allow(clippy::type_complexity)]
    fn get_row_counter(
        &self,
        table_name: &str,
    ) -> Result<Box<dyn Fn(&[i64]) -> usize + Send + Sync>> {
        let _ = table_name;
        Err(crate::core::Error::internal(
            "get_row_counter not supported by this engine",
        ))
    }

    // --- View operations ---

    /// Get a view definition by name (case-insensitive, expects lowercase input)
    fn get_view_lowercase(&self, _name_lower: &str) -> Result<Option<Arc<ViewDefinition>>> {
        Ok(None)
    }

    /// Get a view definition by name (handles case conversion)
    fn get_view(&self, name: &str) -> Result<Option<Arc<ViewDefinition>>> {
        self.get_view_lowercase(&name.to_lowercase())
    }

    /// List all view names
    fn list_views(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Create a view
    fn create_view(&self, _name: &str, _query: &str, _or_replace: bool) -> Result<()> {
        Err(crate::core::Error::internal(
            "create_view not supported by this engine",
        ))
    }

    /// Drop a view
    fn drop_view(&self, _name: &str) -> Result<()> {
        Err(crate::core::Error::internal(
            "drop_view not supported by this engine",
        ))
    }

    // --- Foreign key operations ---

    /// Get a table handle for an existing transaction by txn_id.
    /// This allows FK enforcement to participate in the caller's transaction.
    fn get_table_for_txn(
        &self,
        _txn_id: i64,
        _table_name: &str,
    ) -> Result<Box<dyn Table>> {
        Err(crate::core::Error::internal(
            "get_table_for_txn not supported by this engine",
        ))
    }

    /// Find all FK constraints in other tables that reference the given parent table.
    fn find_referencing_fks(
        &self,
        _parent_table: &str,
    ) -> Arc<Vec<(String, ForeignKeyConstraint)>> {
        Arc::new(Vec::new())
    }

    /// Returns all schemas currently in the engine
    fn get_all_schemas(&self) -> Vec<CompactArc<Schema>> {
        Vec::new()
    }

    /// Set the global isolation level (affects all new transactions)
    fn set_global_isolation_level(&self, level: IsolationLevel) {
        let _ = self.set_isolation_level(level);
    }

    // --- DDL operations at engine level ---

    /// Check if a view exists
    fn view_exists(&self, _name: &str) -> Result<bool> {
        Ok(false)
    }

    /// Create a table at the engine level (outside of a transaction)
    fn create_table_direct(&self, _schema: Schema) -> Result<Schema> {
        Err(crate::core::Error::internal(
            "create_table_direct not supported by this engine",
        ))
    }

    /// Drop a table at the engine level (outside of a transaction)
    fn drop_table_direct(&self, _name: &str) -> Result<()> {
        Err(crate::core::Error::internal(
            "drop_table_direct not supported by this engine",
        ))
    }

    /// Refresh the engine's schema cache for a table after DDL operations
    fn refresh_schema_cache(&self, _table_name: &str) -> Result<()> {
        Ok(())
    }

    /// Manually triggers vacuuming of deleted rows and old versions.
    /// Returns (deleted_rows_cleaned, old_versions_cleaned, transactions_cleaned).
    fn vacuum(
        &self,
        _table_name: Option<&str>,
        _retention: std::time::Duration,
    ) -> Result<(i32, i32, i32)> {
        Ok((0, 0, 0))
    }
}

#[cfg(test)]
mod tests {
    // Engine tests will be implemented when we have concrete implementations
    // For now, just verify the trait compiles correctly

    use super::*;

    // Verify trait is object-safe
    fn _assert_object_safe(_: &dyn Engine) {}
}
