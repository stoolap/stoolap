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

use crate::core::{IsolationLevel, Result, Row, Schema};
use crate::storage::config::Config;
use crate::storage::traits::{Index, Transaction};

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
    fn open(&mut self) -> Result<()>;

    /// Closes the storage engine
    ///
    /// This flushes pending writes, creates a final snapshot if needed,
    /// and releases all resources.
    fn close(&mut self) -> Result<()>;

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
    fn get_table_schema(&self, table_name: &str) -> Result<Schema>;

    /// Lists all indexes for a table
    ///
    /// Returns a map from index name to index type string.
    fn list_table_indexes(&self, table_name: &str) -> Result<FxHashMap<String, String>>;

    /// Gets all index objects for a table
    fn get_all_indexes(&self, table_name: &str) -> Result<Vec<std::sync::Arc<dyn Index>>>;

    /// Gets the current default isolation level
    fn get_isolation_level(&self) -> IsolationLevel;

    /// Sets the default isolation level for new transactions
    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()>;

    /// Gets the current engine configuration
    ///
    /// Returns a clone of the configuration to avoid lifetime issues with internal locks.
    fn get_config(&self) -> Config;

    /// Updates the engine configuration
    ///
    /// Note: Some configuration changes may require a restart to take effect.
    fn update_config(&mut self, config: Config) -> Result<()>;

    /// Manually triggers snapshot creation for all tables
    ///
    /// This is useful for creating a consistent backup point.
    fn create_snapshot(&self) -> Result<()>;

    /// Record an index creation operation to WAL for persistence
    ///
    /// This should be called by the executor after creating an index to ensure
    /// the index is recreated on recovery.
    fn record_create_index(
        &self,
        table_name: &str,
        index_name: &str,
        column_names: &[String],
        is_unique: bool,
        index_type: crate::core::IndexType,
    ) {
        // Default implementation does nothing (for in-memory engines)
        let _ = (table_name, index_name, column_names, is_unique, index_type);
    }

    /// Record an index drop operation to WAL for persistence
    fn record_drop_index(&self, table_name: &str, index_name: &str) {
        // Default implementation does nothing
        let _ = (table_name, index_name);
    }

    /// Record ALTER TABLE ADD COLUMN operation to WAL for persistence
    fn record_alter_table_add_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: crate::core::DataType,
        nullable: bool,
        default_expr: Option<&str>,
    ) {
        // Default implementation does nothing (for in-memory engines)
        let _ = (table_name, column_name, data_type, nullable, default_expr);
    }

    /// Record ALTER TABLE DROP COLUMN operation to WAL for persistence
    fn record_alter_table_drop_column(&self, table_name: &str, column_name: &str) {
        // Default implementation does nothing
        let _ = (table_name, column_name);
    }

    /// Record ALTER TABLE RENAME COLUMN operation to WAL for persistence
    fn record_alter_table_rename_column(
        &self,
        table_name: &str,
        old_column_name: &str,
        new_column_name: &str,
    ) {
        // Default implementation does nothing
        let _ = (table_name, old_column_name, new_column_name);
    }

    /// Record ALTER TABLE MODIFY COLUMN operation to WAL for persistence
    fn record_alter_table_modify_column(
        &self,
        table_name: &str,
        column_name: &str,
        data_type: crate::core::DataType,
        nullable: bool,
    ) {
        // Default implementation does nothing
        let _ = (table_name, column_name, data_type, nullable);
    }

    /// Record ALTER TABLE RENAME TO operation to WAL for persistence
    fn record_alter_table_rename(&self, old_table_name: &str, new_table_name: &str) {
        // Default implementation does nothing
        let _ = (old_table_name, new_table_name);
    }

    /// Fetch rows by IDs directly from storage without creating a full transaction.
    ///
    /// This is an optimization for EXISTS subquery evaluation where we only need
    /// to check if rows exist and evaluate predicates. It avoids the ~2-5μs overhead
    /// of creating a new transaction per EXISTS probe.
    ///
    /// The returned rows represent the latest committed state visible to any reader.
    fn fetch_rows_by_ids(&self, table_name: &str, row_ids: &[i64]) -> Result<Vec<(i64, Row)>> {
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
    ) -> Result<Box<dyn Fn(&[i64]) -> Vec<(i64, Row)> + Send + Sync>> {
        // Default implementation: fall back to fetch_rows_by_ids
        let _ = table_name;
        Err(crate::core::Error::internal(
            "get_row_fetcher not supported by this engine",
        ))
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
