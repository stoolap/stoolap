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

//! MVCC Transaction implementation
//!
//! Provides transaction semantics with two-phase commit protocol.
//!

use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::core::{Error, IsolationLevel, Result, Schema, SchemaColumn};
use crate::storage::mvcc::{get_fast_timestamp, MvccError, TransactionRegistry};
use crate::storage::traits::{QueryResult, Table, Transaction};
use crate::storage::Expression;

/// MVCC Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction is active and can perform operations
    Active,
    /// Transaction is being committed (two-phase commit)
    Committing,
    /// Transaction has been committed
    Committed,
    /// Transaction has been rolled back
    RolledBack,
}

/// MVCC Transaction implementation
pub struct MvccTransaction {
    /// Transaction ID
    id: i64,
    /// Transaction start time
    start_time: Instant,
    /// Transaction state
    state: TransactionState,
    /// Tables accessed in this transaction
    tables: FxHashMap<String, Box<dyn Table>>,
    /// Transaction-specific isolation level (if different from engine default)
    isolation_level: Option<IsolationLevel>,
    /// Reference to the transaction registry
    registry: Arc<TransactionRegistry>,
    /// Begin sequence number (for snapshot isolation)
    begin_seq: i64,
    /// Fast path cache for single table operations
    last_table_name: Option<String>,
    /// Engine reference for table operations (will be set by Engine)
    engine_operations: Option<Arc<dyn TransactionEngineOperations>>,
    /// Savepoints: maps savepoint name to timestamp when created
    savepoints: FxHashMap<String, i64>,
    /// Tables created in this transaction (for rollback)
    created_tables: Vec<String>,
    /// Tables dropped in this transaction (for rollback - stores name and schema)
    dropped_tables: Vec<(String, Schema)>,
}

/// Operations that require engine access
///
/// This trait allows the transaction to call back into the engine
/// without creating circular dependencies.
pub trait TransactionEngineOperations: Send + Sync {
    /// Get a table by name, initializing transaction-local version store
    fn get_table_for_transaction(&self, txn_id: i64, table_name: &str) -> Result<Box<dyn Table>>;

    /// Create a new table
    fn create_table(&self, name: &str, schema: Schema) -> Result<Box<dyn Table>>;

    /// Drop a table
    fn drop_table(&self, name: &str) -> Result<()>;

    /// List all tables
    fn list_tables(&self) -> Result<Vec<String>>;

    /// Rename a table
    fn rename_table(&self, old_name: &str, new_name: &str) -> Result<()>;

    /// Commit table changes
    fn commit_table(&self, txn_id: i64, table: &dyn Table) -> Result<()>;

    /// Rollback table changes
    fn rollback_table(&self, txn_id: i64, table: &dyn Table);

    /// Record commit in WAL
    fn record_commit(&self, txn_id: i64) -> Result<()>;

    /// Record rollback in WAL
    fn record_rollback(&self, txn_id: i64) -> Result<()>;

    /// Get all tables with pending changes for a transaction
    fn get_tables_with_pending_changes(&self, txn_id: i64) -> Result<Vec<Box<dyn Table>>>;

    /// Commit all tables for a transaction at once
    fn commit_all_tables(&self, txn_id: i64) -> Result<()>;
}

impl MvccTransaction {
    /// Creates a new MVCC transaction
    pub fn new(id: i64, begin_seq: i64, registry: Arc<TransactionRegistry>) -> Self {
        Self {
            id,
            start_time: Instant::now(),
            state: TransactionState::Active,
            tables: FxHashMap::default(),
            isolation_level: None,
            registry,
            begin_seq,
            last_table_name: None,
            engine_operations: None,
            savepoints: FxHashMap::default(),
            created_tables: Vec::new(),
            dropped_tables: Vec::new(),
        }
    }

    /// Sets the engine operations callback
    pub fn set_engine_operations(&mut self, ops: Arc<dyn TransactionEngineOperations>) {
        self.engine_operations = Some(ops);
    }

    /// Returns the transaction start time
    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    /// Returns the begin sequence number
    pub fn begin_seq(&self) -> i64 {
        self.begin_seq
    }

    /// Returns the current transaction state
    pub fn state(&self) -> TransactionState {
        self.state
    }

    /// Returns the isolation level for this transaction
    pub fn get_isolation_level(&self) -> IsolationLevel {
        self.isolation_level
            .unwrap_or_else(|| self.registry.get_global_isolation_level())
    }

    /// Check if transaction is active
    fn check_active(&self) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(MvccError::TransactionClosed.into());
        }
        Ok(())
    }

    /// Get engine operations, returning error if not set
    fn get_engine_ops(&self) -> Result<&Arc<dyn TransactionEngineOperations>> {
        self.engine_operations
            .as_ref()
            .ok_or_else(|| Error::internal("engine operations not set"))
    }

    /// Clean up transaction resources
    fn cleanup(&mut self) {
        // Clear fast path cache
        self.last_table_name = None;

        // Clear tables
        self.tables.clear();

        // Clear DDL tracking
        self.created_tables.clear();
        self.dropped_tables.clear();

        // Remove transaction isolation level from registry
        self.registry.remove_transaction_isolation_level(self.id);
    }

    /// Check if this is a read-only transaction
    fn is_read_only(&self) -> bool {
        // Check for DDL changes
        if !self.created_tables.is_empty() || !self.dropped_tables.is_empty() {
            return false;
        }
        // Check if any tables have local modifications
        for table in self.tables.values() {
            if table.has_local_changes() {
                return false;
            }
        }
        true
    }

    /// Creates a savepoint with the given name
    ///
    /// Records the current timestamp so we can rollback to this point later.
    /// If a savepoint with this name already exists, it is overwritten.
    pub fn create_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;
        let timestamp = get_fast_timestamp();
        self.savepoints.insert(name.to_string(), timestamp);
        Ok(())
    }

    /// Releases (removes) a savepoint without rolling back
    ///
    /// The changes made after the savepoint remain intact.
    /// Returns an error if the savepoint doesn't exist.
    pub fn release_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;
        if self.savepoints.remove(name).is_none() {
            return Err(Error::invalid_argument(format!(
                "savepoint '{}' does not exist",
                name
            )));
        }
        Ok(())
    }

    /// Rolls back to a savepoint, discarding all changes made after it
    ///
    /// All local changes with timestamps after the savepoint are discarded.
    /// The savepoint itself is also removed (SQL standard behavior).
    pub fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        self.check_active()?;

        let savepoint_ts = self.savepoints.get(name).copied().ok_or_else(|| {
            Error::invalid_argument(format!("savepoint '{}' does not exist", name))
        })?;

        // Rollback each table's local changes that occurred after the savepoint
        for table in self.tables.values() {
            table.rollback_to_timestamp(savepoint_ts);
        }

        // Remove this savepoint and all savepoints created after it
        self.savepoints.retain(|_, &mut ts| ts <= savepoint_ts);

        Ok(())
    }

    /// Check if a savepoint exists
    pub fn has_savepoint(&self, name: &str) -> bool {
        self.savepoints.contains_key(name)
    }

    /// Gets the timestamp associated with a savepoint
    pub fn get_savepoint_ts(&self, name: &str) -> Option<i64> {
        self.savepoints.get(name).copied()
    }
}

impl Transaction for MvccTransaction {
    fn id(&self) -> i64 {
        self.id
    }

    fn begin(&mut self) -> Result<()> {
        // No-op for compatibility - transaction is initialized in new()
        self.check_active()
    }

    fn commit(&mut self) -> Result<()> {
        self.check_active()?;

        // Update state to committing
        self.state = TransactionState::Committing;

        // Get tables with pending changes from the engine
        // This ensures we commit ALL tables accessed by this transaction,
        // not just the ones cached in self.tables
        let tables_with_changes = if let Some(ops) = &self.engine_operations {
            ops.get_tables_with_pending_changes(self.id)?
        } else {
            Vec::new()
        };

        // Check if read-only: no DDL changes and no DML changes
        let is_read_only = self.created_tables.is_empty()
            && self.dropped_tables.is_empty()
            && tables_with_changes.is_empty();

        // Two-phase commit protocol
        if !is_read_only {
            // Phase 1: Start commit - mark transaction as "committing"
            self.registry.start_commit(self.id);

            // Record DML operations to WAL BEFORE committing tables
            // (commit_table needs to see the pending versions before they're moved to global)
            if let Some(ops) = &self.engine_operations {
                for table in tables_with_changes.iter() {
                    if let Err(e) = ops.commit_table(self.id, table.as_ref()) {
                        // WAL write failed - abort the transaction
                        self.registry.abort_transaction(self.id);
                        self.state = TransactionState::RolledBack;
                        self.cleanup();
                        return Err(e);
                    }
                }
            }

            // Phase 2: Commit all tables - apply local changes to global store
            // Use the engine's commit_all_tables which knows about all pending changes
            if let Some(ops) = &self.engine_operations {
                if let Err(e) = ops.commit_all_tables(self.id) {
                    // Abort on failure
                    self.registry.abort_transaction(self.id);
                    self.state = TransactionState::RolledBack;
                    self.cleanup();
                    return Err(e);
                }
            }

            // Phase 3: Complete commit - make changes visible
            self.registry.complete_commit(self.id);

            // Record in WAL
            if let Some(ops) = &self.engine_operations {
                let _ = ops.record_commit(self.id);
            }
        } else {
            // Read-only transaction - just mark as committed in registry
            self.registry.complete_commit(self.id);
        }

        // Mark as committed
        self.state = TransactionState::Committed;
        self.cleanup();
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        self.check_active()?;

        // Check if read-only before rolling back
        let is_read_only = self.is_read_only();

        // Mark transaction as aborted in registry
        self.registry.abort_transaction(self.id);

        // Rollback DDL operations in reverse order
        // We attempt all rollbacks even if some fail, but track errors
        let mut ddl_rollback_errors: Vec<String> = Vec::new();

        if let Some(ops) = &self.engine_operations {
            // Drop tables that were created in this transaction
            for table_name in self.created_tables.iter().rev() {
                if let Err(e) = ops.drop_table(table_name) {
                    ddl_rollback_errors.push(format!(
                        "Failed to drop table '{}' during rollback: {}",
                        table_name, e
                    ));
                }
            }

            // Recreate tables that were dropped in this transaction
            // NOTE: This only recreates the table structure (schema), NOT the data.
            // DROP TABLE within a transaction is a destructive operation - the data
            // cannot be recovered on rollback. This is similar to PostgreSQL's behavior
            // where certain DDL operations are not fully transactional.
            for (table_name, schema) in self.dropped_tables.iter().rev() {
                if let Err(e) = ops.create_table(table_name, schema.clone()) {
                    ddl_rollback_errors.push(format!(
                        "Failed to recreate table '{}' during rollback: {}",
                        table_name, e
                    ));
                } else {
                    // Warn that data was lost
                    eprintln!(
                        "Warning: Table '{}' was recreated during rollback but data was lost. \
                         DROP TABLE is not fully transactional.",
                        table_name
                    );
                }
            }
        }

        // Log DDL rollback errors (but continue with rollback)
        if !ddl_rollback_errors.is_empty() {
            eprintln!(
                "Warning: DDL rollback encountered {} error(s): {:?}",
                ddl_rollback_errors.len(),
                ddl_rollback_errors
            );
        }

        // Rollback all tables - discard local changes
        for (_, table) in self.tables.iter_mut() {
            table.rollback();
        }

        // Notify engine of rollback
        if let Some(ops) = &self.engine_operations {
            for (_, table) in self.tables.iter() {
                ops.rollback_table(self.id, table.as_ref());
            }
        }

        // Record in WAL if not read-only
        if !is_read_only {
            if let Some(ops) = &self.engine_operations {
                let _ = ops.record_rollback(self.id);
            }
        }

        // Mark as rolled back
        self.state = TransactionState::RolledBack;
        self.cleanup();
        Ok(())
    }

    fn create_savepoint(&mut self, name: &str) -> Result<()> {
        // Delegate to the inherent method
        MvccTransaction::create_savepoint(self, name)
    }

    fn release_savepoint(&mut self, name: &str) -> Result<()> {
        // Delegate to the inherent method
        MvccTransaction::release_savepoint(self, name)
    }

    fn rollback_to_savepoint(&mut self, name: &str) -> Result<()> {
        // Delegate to the inherent method
        MvccTransaction::rollback_to_savepoint(self, name)
    }

    fn get_savepoint_timestamp(&self, name: &str) -> Option<i64> {
        // Delegate to the inherent method
        MvccTransaction::get_savepoint_ts(self, name)
    }

    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()> {
        self.check_active()?;

        // Set transaction-specific isolation level
        self.isolation_level = Some(level);

        // Also set in registry
        self.registry
            .set_transaction_isolation_level(self.id, level);

        Ok(())
    }

    fn create_table(&mut self, name: &str, schema: Schema) -> Result<Box<dyn Table>> {
        self.check_active()?;

        let ops = self.get_engine_ops()?;
        let table = ops.create_table(name, schema)?;

        // Track for rollback - store the table name
        self.created_tables.push(name.to_lowercase());

        Ok(table)
    }

    /// Drop a table within this transaction.
    ///
    /// # Warning
    /// DROP TABLE is NOT fully transactional. While the table structure can be
    /// recreated on rollback, the data CANNOT be recovered. This is similar to
    /// PostgreSQL's behavior where certain DDL operations are destructive.
    ///
    /// If you need to safely remove all data with rollback support, use
    /// `DELETE FROM table_name` or `TRUNCATE TABLE table_name` instead.
    fn drop_table(&mut self, name: &str) -> Result<()> {
        self.check_active()?;

        // Before dropping, get the schema so we can recreate on rollback
        // We need to get the table to access its schema
        // Scope the borrow to allow later mutable operations
        let schema = {
            let ops = self.get_engine_ops()?;
            let table = ops.get_table_for_transaction(self.id, name)?;
            table.schema().clone()
        };

        // Save schema for potential rollback (note: data cannot be recovered)
        self.dropped_tables.push((name.to_lowercase(), schema));

        // Now drop the table
        let ops = self.get_engine_ops()?;
        ops.drop_table(name)?;

        // Remove from cache
        self.tables.remove(name);

        // Clear fast path cache if needed
        if let Some(last_name) = &self.last_table_name {
            if last_name == name {
                self.last_table_name = None;
            }
        }

        Ok(())
    }

    fn get_table(&self, name: &str) -> Result<Box<dyn Table>> {
        self.check_active()?;

        // Note: Cached tables would require Clone on Table trait, which isn't object-safe.
        // For now, always get from engine (engine will handle caching internally).
        // The tables HashMap is used for tracking which tables were accessed for commit/rollback.

        // Get from engine
        let ops = self.get_engine_ops()?;
        ops.get_table_for_transaction(self.id, name)
    }

    fn list_tables(&self) -> Result<Vec<String>> {
        self.check_active()?;

        let ops = self.get_engine_ops()?;
        ops.list_tables()
    }

    fn rename_table(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        self.check_active()?;

        let ops = self.get_engine_ops()?;
        ops.rename_table(old_name, new_name)?;

        // Update cache if needed
        if let Some(table) = self.tables.remove(old_name) {
            self.tables.insert(new_name.to_string(), table);
        }

        // Update fast path cache
        if let Some(last_name) = &self.last_table_name {
            if last_name == old_name {
                self.last_table_name = Some(new_name.to_string());
            }
        }

        Ok(())
    }

    fn create_table_index(
        &mut self,
        table_name: &str,
        index_name: &str,
        columns: &[String],
        is_unique: bool,
    ) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        table.create_index(index_name, &col_refs, is_unique)
    }

    fn drop_table_index(&mut self, table_name: &str, index_name: &str) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        table.drop_index(index_name)
    }

    fn create_table_btree_index(
        &mut self,
        table_name: &str,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        table.create_btree_index(column_name, is_unique, custom_name)
    }

    fn drop_table_btree_index(&mut self, table_name: &str, column_name: &str) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        table.drop_btree_index(column_name)
    }

    fn add_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()> {
        self.check_active()?;

        let mut table = self.get_table(table_name)?;
        table.create_column(&column.name, column.data_type, column.nullable)
    }

    fn drop_table_column(&mut self, table_name: &str, column_name: &str) -> Result<()> {
        self.check_active()?;

        let mut table = self.get_table(table_name)?;
        table.drop_column(column_name)
    }

    fn rename_table_column(
        &mut self,
        table_name: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        table.rename_column(old_name, new_name)
    }

    fn modify_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
        table.modify_column(&column.name, column.data_type, column.nullable)
    }

    fn select(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        _original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>> {
        self.check_active()?;

        let table = self.get_table(table_name)?;
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
        self.check_active()?;

        let table = self.get_table(table_name)?;
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
        self.check_active()?;

        let table = self.get_table(table_name)?;
        let col_refs: Vec<&str> = columns_to_fetch.iter().map(|s| s.as_str()).collect();
        table.select_as_of(&col_refs, expr, temporal_type, temporal_value)
    }
}

impl std::fmt::Debug for MvccTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MvccTransaction")
            .field("id", &self.id)
            .field("state", &self.state)
            .field("begin_seq", &self.begin_seq)
            .finish()
    }
}

// Ensure transaction is rolled back on drop if still active
impl Drop for MvccTransaction {
    fn drop(&mut self) {
        if self.state == TransactionState::Active {
            // Silent rollback on drop
            self.registry.abort_transaction(self.id);
            self.cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_creation() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        assert_eq!(txn.id(), txn_id);
        assert_eq!(txn.begin_seq(), begin_seq);
        assert_eq!(txn.state(), TransactionState::Active);
    }

    #[test]
    fn test_transaction_state_transitions() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        assert_eq!(txn.state(), TransactionState::Active);

        // Begin should be no-op
        txn.begin().unwrap();
        assert_eq!(txn.state(), TransactionState::Active);

        // Commit
        txn.commit().unwrap();
        assert_eq!(txn.state(), TransactionState::Committed);

        // Should fail to begin after commit
        assert!(txn.begin().is_err());
    }

    #[test]
    fn test_transaction_rollback() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        assert_eq!(txn.state(), TransactionState::Active);

        // Rollback
        txn.rollback().unwrap();
        assert_eq!(txn.state(), TransactionState::RolledBack);

        // Should fail to begin after rollback
        assert!(txn.begin().is_err());
    }

    #[test]
    fn test_transaction_isolation_level() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        // Default isolation level
        let default_level = txn.get_isolation_level();
        assert_eq!(default_level, IsolationLevel::ReadCommitted);

        // Set transaction-specific level
        txn.set_isolation_level(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(txn.get_isolation_level(), IsolationLevel::SnapshotIsolation);
    }

    #[test]
    fn test_transaction_double_commit() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        // First commit should succeed
        txn.commit().unwrap();

        // Second commit should fail
        assert!(txn.commit().is_err());
    }

    #[test]
    fn test_transaction_commit_after_rollback() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let mut txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        // Rollback first
        txn.rollback().unwrap();

        // Commit should fail
        assert!(txn.commit().is_err());
    }

    #[test]
    fn test_transaction_debug() {
        let registry = Arc::new(TransactionRegistry::new());
        let (txn_id, begin_seq) = registry.begin_transaction();
        let txn = MvccTransaction::new(txn_id, begin_seq, Arc::clone(&registry));

        let debug_str = format!("{:?}", txn);
        assert!(debug_str.contains("MvccTransaction"));
        assert!(debug_str.contains("Active"));
    }
}
