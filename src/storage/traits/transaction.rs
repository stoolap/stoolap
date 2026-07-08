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

//! Transaction traits. [`ReadTransaction`] is the non-mutating surface
//! plus transaction-local lifecycle (savepoints / commit / rollback are
//! no-ops on disk for read-only). [`WriteTransaction`] extends it with
//! table DDL and the writable `get_table`. Read-only callers cannot
//! reach `WriteTransaction` at compile time.

use rustc_hash::FxHashMap;

use crate::core::{IsolationLevel, Result, Schema, SchemaColumn};
use crate::storage::expression::Expression;
use crate::storage::traits::{QueryResult, ReadTable, WriteTable};

/// Read-only surface of a database transaction. Includes lifecycle,
/// savepoints, isolation, table listing, and queries. None of these
/// write persistent state on a read-only transaction.
pub trait ReadTransaction: Send {
    /// Begins the transaction.
    fn begin(&mut self) -> Result<()>;

    /// Commits the transaction. On read-only transactions this still
    /// transitions state to `Committed` (Drop aborts an Active txn).
    fn commit(&mut self) -> Result<()>;

    /// Rolls back the transaction.
    fn rollback(&mut self) -> Result<()>;

    /// Creates (or overwrites) a savepoint.
    fn create_savepoint(&mut self, name: &str) -> Result<()>;

    /// Removes a savepoint, keeping its changes.
    fn release_savepoint(&mut self, name: &str) -> Result<()>;

    /// Rolls back to a savepoint and removes it.
    fn rollback_to_savepoint(&mut self, name: &str) -> Result<()>;

    /// Returns the savepoint's timestamp, or None if it does not exist.
    fn get_savepoint_timestamp(&self, name: &str) -> Option<i64>;

    /// Returns the transaction ID.
    fn id(&self) -> i64;

    /// Sets the transaction-local isolation level. SQL-level
    /// `SET TRANSACTION ISOLATION LEVEL` is rejected at the parser gate
    /// for read-only DBs, so this Rust method is safe to leave reachable.
    fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()>;

    /// Lists all table names.
    fn list_tables(&self) -> Result<Vec<String>>;

    /// Returns a read-only handle to `name`. Never widens to `WriteTable`.
    fn get_read_table(&self, name: &str) -> Result<Box<dyn ReadTable>>;

    // ---- Query Operations ----

    /// Executes a SELECT on `table_name`. `original_columns` carries the
    /// pre-alias names when the executor has rewritten the projection.
    fn select(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>>;

    /// Like [`select`](Self::select) but with output `aliases` (alias -> original).
    fn select_with_aliases(
        &self,
        table_name: &str,
        columns_to_fetch: &[String],
        expr: Option<&dyn Expression>,
        aliases: &FxHashMap<String, String>,
        original_columns: Option<&[String]>,
    ) -> Result<Box<dyn QueryResult>>;

    /// Temporal SELECT. `temporal_type` is `"TRANSACTION"` or `"TIMESTAMP"`;
    /// `temporal_value` is the txn id or nanoseconds since epoch.
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

/// Writable surface of a database transaction. Extends [`ReadTransaction`]
/// with table/index/column DDL and the writable `get_table`.
pub trait WriteTransaction: ReadTransaction {
    // ---- Table Operations ----

    /// Creates a new table with the given schema.
    fn create_table(&mut self, name: &str, schema: Schema) -> Result<Box<dyn WriteTable>>;

    /// Drops a table.
    fn drop_table(&mut self, name: &str) -> Result<()>;

    /// Returns a writable handle to `name`.
    fn get_table(&self, name: &str) -> Result<Box<dyn WriteTable>>;

    /// Renames a table.
    fn rename_table(&mut self, old_name: &str, new_name: &str) -> Result<()>;

    // ---- Index Operations ----

    /// Creates an index named `index_name` on `table_name`.
    fn create_table_index(
        &mut self,
        table_name: &str,
        index_name: &str,
        columns: &[String],
        is_unique: bool,
    ) -> Result<()>;

    /// Drops `index_name` from `table_name`.
    fn drop_table_index(&mut self, table_name: &str, index_name: &str) -> Result<()>;

    /// Creates a btree index on `(table_name, column_name)`.
    fn create_table_btree_index(
        &mut self,
        table_name: &str,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()>;

    /// Drops the btree index covering `(table_name, column_name)`.
    fn drop_table_btree_index(&mut self, table_name: &str, column_name: &str) -> Result<()>;

    /// Stages a CREATE INDEX on the txn's deferred-DDL log so generated
    /// UNIQUE/FK indexes flush atomically with the parent CREATE TABLE.
    /// Default is a no-op for non-MVCC backends.
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

    /// Adds `column` to `table_name`.
    fn add_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()>;

    /// Drops `column_name` from `table_name`.
    fn drop_table_column(&mut self, table_name: &str, column_name: &str) -> Result<()>;

    /// Renames `old_name` to `new_name` in `table_name`.
    fn rename_table_column(
        &mut self,
        table_name: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()>;

    /// Replaces the definition of a column in `table_name`.
    fn modify_table_column(&mut self, table_name: &str, column: SchemaColumn) -> Result<()>;
}

/// Temporal query type for time-travel queries.
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

    fn _assert_read_object_safe(_: &dyn ReadTransaction) {}
    fn _assert_write_object_safe(_: &dyn WriteTransaction) {}
}
