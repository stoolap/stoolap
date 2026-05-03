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

//! Transaction traits for database transactions.
//!
//! Split into [`ReadTransaction`] (non-mutating surface, plus
//! transaction-local mutations like savepoints / commit / rollback that
//! do not write to disk on read-only transactions) and
//! [`WriteTransaction`] (extends `ReadTransaction` with table DDL and
//! the writable `get_table`).
//!
//! Read-only callers receive `Box<dyn ReadTransaction>` and cannot reach
//! `create_table`, `drop_table`, schema DDL, or any path that returns a
//! [`WriteTable`]. The compiler enforces this by construction.

use rustc_hash::FxHashMap;

use crate::core::{IsolationLevel, Result, Schema, SchemaColumn};
use crate::storage::expression::Expression;
use crate::storage::traits::{QueryResult, ReadTable, WriteTable};

/// Read-only surface of a database transaction.
///
/// Includes transaction lifecycle (`begin`, `commit`, `rollback`),
/// savepoint management, isolation-level updates, table listing, and
/// queries — none of which write persistent state on a read-only
/// transaction.
///
/// `commit` is on this trait because it must finalize registry/state
/// cleanup even on read-only transactions; `Drop` aborts otherwise. On
/// read-only transactions, `commit` performs no WAL or data flush.
pub trait ReadTransaction: Send {
    /// Begins the transaction
    fn begin(&mut self) -> Result<()>;

    /// Commits the transaction.
    ///
    /// On read-only transactions: no WAL/data flush, but still finalizes
    /// registry/state cleanup (transitions state to `Committed`).
    /// Cannot be a literal no-op — `Drop` checks `state == Active` and
    /// aborts; if `commit` left the state Active, the transaction would
    /// silently abort on drop.
    fn commit(&mut self) -> Result<()>;

    /// Rolls back the transaction
    fn rollback(&mut self) -> Result<()>;

    /// Creates a savepoint with the given name
    ///
    /// Records the current state so it can be rolled back to later.
    /// If a savepoint with this name already exists, it is overwritten.
    fn create_savepoint(&mut self, name: &str) -> Result<()>;

    /// Releases (removes) a savepoint without rolling back
    ///
    /// The changes made after the savepoint remain intact.
    fn release_savepoint(&mut self, name: &str) -> Result<()>;

    /// Rolls back to a savepoint, discarding all changes made after it
    ///
    /// The savepoint itself is also removed after rollback.
    fn rollback_to_savepoint(&mut self, name: &str) -> Result<()>;

    /// Gets the timestamp associated with a savepoint
    ///
    /// Returns None if the savepoint doesn't exist.
    fn get_savepoint_timestamp(&self, name: &str) -> Option<i64>;

    /// Returns the transaction ID
    fn id(&self) -> i64;

    /// Sets the isolation level for this transaction.
    ///
    /// Mutates transaction-local state only. SQL-level
    /// `SET TRANSACTION ISOLATION LEVEL` is rejected at the parser gate
    /// for read-only databases (`Statement::write_reason()` classifies
    /// it as a writer); the Rust method is safe to leave reachable.
    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()>;

    /// Lists all table names
    fn list_tables(&self) -> Result<Vec<String>>;

    /// Gets a read-only handle to a table by name.
    ///
    /// Returns `Box<dyn ReadTable>`. Never returns a writable handle —
    /// read-only callers cannot reach `WriteTable` through this method.
    fn get_read_table(&self, name: &str) -> Result<Box<dyn ReadTable>>;

    // ---- Query Operations ----

    /// Executes a SELECT query
    ///
    /// # Arguments
    /// * `table_name` - Name of the table to query
    /// * `columns_to_fetch` - Column names to include in the result
    /// * `expr` - Optional filter expression
    /// * `original_columns` - Optional original column names (for aliasing)
    fn select(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>>;

    /// Executes a SELECT query with column aliases
    ///
    /// # Arguments
    /// * `table_name` - Name of the table to query
    /// * `columns_to_fetch` - Column names to include in the result
    /// * `expr` - Optional filter expression
    /// * `aliases` - Map from alias names to original column names
    /// * `original_columns` - Optional original column names
    fn select_with_aliases(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        aliases: &FxHashMap<String, String>,
        original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>>;

    /// Executes a temporal SELECT query as of a specific transaction or timestamp
    ///
    /// # Arguments
    /// * `table_name` - Name of the table to query
    /// * `columns_to_fetch` - Column names to include in the result
    /// * `expr` - Optional filter expression
    /// * `temporal_type` - Either "TRANSACTION" or "TIMESTAMP"
    /// * `temporal_value` - Transaction ID or timestamp in nanoseconds
    /// * `original_columns` - Optional original column names
    fn select_as_of(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        temporal_type: &str,
        temporal_value: i64,
        original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>>;
}

/// Writable surface of a database transaction.
///
/// Extends [`ReadTransaction`] with table DDL (`create_table`,
/// `drop_table`, `rename_table`), index DDL, column DDL, and the
/// writable `get_table` returning `Box<dyn WriteTable>`.
///
/// Read-only callers cannot reach this trait — they hold
/// `Box<dyn ReadTransaction>` instead.
pub trait WriteTransaction: ReadTransaction {
    // ---- Table Operations (write-side) ----

    /// Creates a new table with the given schema
    fn create_table(&mut self, name: &str, schema: Schema) -> Result<Box<dyn WriteTable>>;

    /// Drops a table
    fn drop_table(&mut self, name: &str) -> Result<()>;

    /// Gets a writable handle to a table by name
    fn get_table(&self, name: &str) -> Result<Box<dyn WriteTable>>;

    /// Renames a table
    fn rename_table(&mut self, old_name: &str, new_name: &str) -> Result<()>;

    // ---- Index Operations ----

    /// Creates an index on a table
    ///
    /// # Arguments
    /// * `table_name` - Name of the table
    /// * `index_name` - Name for the new index
    /// * `columns` - Column names to include in the index
    /// * `is_unique` - Whether this is a unique index
    fn create_table_index(
        &mut self,
        table_name: &str,
        index_name: &str,
        columns: &[String],
        is_unique: bool,
    ) -> Result<()>;

    /// Drops an index from a table
    fn drop_table_index(&mut self, table_name: &str, index_name: &str) -> Result<()>;

    /// Creates a btree index on a table column
    ///
    /// # Arguments
    /// * `table_name` - Name of the table
    /// * `column_name` - Name of the column to index
    /// * `is_unique` - Whether this is a unique index
    /// * `custom_name` - Optional custom name for the index
    fn create_table_btree_index(
        &mut self,
        table_name: &str,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()>;

    /// Drops a btree index from a table
    fn drop_table_btree_index(&mut self, table_name: &str, column_name: &str) -> Result<()>;

    /// Stage a CREATE INDEX entry on the txn's deferred-DDL
    /// log. The transactional CREATE TABLE path uses this
    /// (instead of the auto-commit
    /// `MVCCEngine::record_create_index`) so generated
    /// UNIQUE / FK indexes flush as part of the txn's
    /// deferred DDL batch — recovery applies the parent
    /// CreateTable, then this index, then the txn's commit
    /// marker. Default is a no-op for non-MVCC backends.
    #[allow(clippy::too_many_arguments)]
    fn stage_deferred_create_index(
        &mut self,
        _table_name: &str,
        _index_name: &str,
        _columns: &[String],
        _is_unique: bool,
        _index_type: crate::core::IndexType,
        _hnsw_m: Option<u16>,
        _hnsw_ef_construction: Option<u16>,
        _hnsw_ef_search: Option<u16>,
        _hnsw_distance_metric: Option<u8>,
    ) -> Result<()> {
        Ok(())
    }

    // ---- Column Operations (ALTER TABLE) ----

    /// Adds a column to a table
    fn add_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()>;

    /// Drops a column from a table
    fn drop_table_column(&mut self, table_name: &str, column_name: &str) -> Result<()>;

    /// Renames a column in a table
    fn rename_table_column(
        &mut self,
        table_name: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()>;

    /// Modifies a column in a table
    fn modify_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()>;
}

/// Temporal query type for time-travel queries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalType {
    /// Query as of a specific transaction ID
    Transaction,
    /// Query as of a specific timestamp
    Timestamp,
}

impl std::str::FromStr for TemporalType {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "TRANSACTION" => Ok(Self::Transaction),
            "TIMESTAMP" => Ok(Self::Timestamp),
            _ => Err(()),
        }
    }
}

impl TemporalType {
    /// Parses a temporal type from a string
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_type_from_str() {
        assert_eq!(
            TemporalType::parse("TRANSACTION"),
            Some(TemporalType::Transaction)
        );
        assert_eq!(
            TemporalType::parse("transaction"),
            Some(TemporalType::Transaction)
        );
        assert_eq!(
            TemporalType::parse("TIMESTAMP"),
            Some(TemporalType::Timestamp)
        );
        assert_eq!(
            TemporalType::parse("timestamp"),
            Some(TemporalType::Timestamp)
        );
        assert_eq!(TemporalType::parse("INVALID"), None);
    }

    // Verify both traits are object-safe
    fn _assert_read_object_safe(_: &dyn ReadTransaction) {}
    fn _assert_write_object_safe(_: &dyn WriteTransaction) {}
}
