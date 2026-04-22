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

//! Transaction trait for database transactions
//!

use rustc_hash::FxHashMap;

use crate::core::{IsolationLevel, Result, Schema, SchemaColumn};
use crate::storage::expression::Expression;
use crate::storage::traits::{QueryResult, Table};

/// Transaction represents a database transaction
///
/// This trait defines the interface for transaction operations including
/// DDL (table/index management), DML (select/insert/update/delete),
/// and transaction control (begin/commit/rollback).
///
/// # Example
///
/// ```ignore
/// let tx = engine.begin_transaction()?;
///
/// // Create a table
/// let schema = Schema::new("users").with_column("id", DataType::Integer, false);
/// tx.create_table("users", schema)?;
///
/// // Insert data
/// let table = tx.get_table("users")?;
/// table.insert(Row::from_values(vec![Value::Integer(1)]))?;
///
/// // Query data
/// let result = tx.select("users", &["id"], None, None)?;
///
/// tx.commit()?;
/// ```
pub trait Transaction: Send {
    /// Begins the transaction
    fn begin(&mut self) -> Result<()>;

    /// Commits the transaction
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

    /// Sets the isolation level for this transaction
    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()>;

    // ---- Table Operations ----

    /// Creates a new table with the given schema
    fn create_table(&mut self, name: &str, schema: Schema) -> Result<Box<dyn Table>>;

    /// Drops a table
    fn drop_table(&mut self, name: &str) -> Result<()>;

    /// Gets a reference to a table by name
    fn get_table(&self, name: &str) -> Result<Box<dyn Table>>;

    /// Lists all table names
    fn list_tables(&self) -> Result<Vec<String>>;

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

    // Verify trait is object-safe
    fn _assert_object_safe(_: &dyn Transaction) {}
}
