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

//! Table trait for database tables
//!

use rustc_hash::FxHashMap;
use std::fmt;

use crate::core::{DataType, Error, IndexType, Result, Row, Schema, Value};
use crate::storage::expression::Expression;
use crate::storage::traits::{Index, QueryResult, Scanner};

/// Describes the access method that will be used for a table scan
///
/// This is used by EXPLAIN to show users how their queries will be executed.
#[derive(Debug, Clone)]
pub enum ScanPlan {
    /// Sequential scan - reads all rows and applies filter in memory
    SeqScan {
        table: String,
        filter: Option<String>,
    },
    /// Parallel sequential scan - reads all rows and filters in parallel across workers
    ParallelSeqScan {
        table: String,
        filter: Option<String>,
        workers: usize,
    },
    /// Primary key lookup - O(1) direct access by primary key
    PkLookup {
        table: String,
        pk_column: String,
        pk_value: String,
    },
    /// Index scan - uses an index to find matching rows
    IndexScan {
        table: String,
        index_name: String,
        column: String,
        condition: String,
    },
    /// Multi-index scan - uses multiple indexes with AND/OR operations
    MultiIndexScan {
        table: String,
        indexes: Vec<(String, String, String)>, // (index_name, column, condition)
        operation: String,                      // "AND" or "OR"
    },
    /// Composite index scan - uses a multi-column index
    CompositeIndexScan {
        table: String,
        index_name: String,
        columns: Vec<String>,
        conditions: Vec<String>,
    },
}

impl fmt::Display for ScanPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScanPlan::SeqScan { table, filter } => {
                write!(f, "Seq Scan on {}", table)?;
                if let Some(flt) = filter {
                    write!(f, "\n  Filter: {}", flt)?;
                }
                Ok(())
            }
            ScanPlan::ParallelSeqScan {
                table,
                filter,
                workers,
            } => {
                write!(f, "Parallel Seq Scan on {} (workers={})", table, workers)?;
                if let Some(flt) = filter {
                    write!(f, "\n  Filter: {}", flt)?;
                }
                Ok(())
            }
            ScanPlan::PkLookup {
                table,
                pk_column,
                pk_value,
            } => {
                write!(f, "PK Lookup on {}\n  {} = {}", table, pk_column, pk_value)
            }
            ScanPlan::IndexScan {
                table,
                index_name,
                column,
                condition,
            } => {
                write!(
                    f,
                    "Index Scan using {} on {}\n  Index Cond: {} {}",
                    index_name, table, column, condition
                )
            }
            ScanPlan::MultiIndexScan {
                table,
                indexes,
                operation,
            } => {
                write!(f, "Multi-Index Scan on {} ({})", table, operation)?;
                for (idx_name, col, cond) in indexes {
                    write!(f, "\n  -> {} on {}: {}", idx_name, col, cond)?;
                }
                Ok(())
            }
            ScanPlan::CompositeIndexScan {
                table,
                index_name,
                columns,
                conditions,
            } => {
                write!(
                    f,
                    "Composite Index Scan using {} on {}\n  Columns: ({})",
                    index_name,
                    table,
                    columns.join(", ")
                )?;
                for (col, cond) in columns.iter().zip(conditions.iter()) {
                    write!(f, "\n  {} {}", col, cond)?;
                }
                Ok(())
            }
        }
    }
}

/// Table represents a database table
///
/// This trait defines the interface for interacting with a table,
/// including schema management, data manipulation (CRUD), and scanning.
///
/// # Example
///
/// ```ignore
/// let table = transaction.get_table("users")?;
/// println!("Table: {}", table.name());
/// println!("Schema: {:?}", table.schema());
///
/// // Insert a row
/// let row = Row::from_values(vec![Value::Integer(1), Value::text("Alice")]);
/// table.insert(row)?;
///
/// // Scan all rows
/// let scanner = table.scan(&[0, 1], None)?;
/// while scanner.next() {
///     println!("{:?}", scanner.row());
/// }
/// ```
pub trait Table: Send + Sync {
    /// Returns the name of the table
    fn name(&self) -> &str;

    /// Returns the schema of the table
    fn schema(&self) -> &Schema;

    /// Creates a new column in the table
    ///
    /// # Arguments
    /// * `name` - The name of the column
    /// * `column_type` - The data type of the column
    /// * `nullable` - Whether the column can contain NULL values
    fn create_column(&mut self, name: &str, column_type: DataType, nullable: bool) -> Result<()>;

    /// Creates a new column in the table with default expression
    ///
    /// # Arguments
    /// * `name` - The name of the column
    /// * `column_type` - The data type of the column
    /// * `nullable` - Whether the column can contain NULL values
    /// * `default_expr` - Default value expression as string (to be evaluated during INSERT)
    fn create_column_with_default(
        &mut self,
        name: &str,
        column_type: DataType,
        nullable: bool,
        _default_expr: Option<String>,
    ) -> Result<()> {
        // Default implementation ignores default_expr for backwards compatibility
        self.create_column(name, column_type, nullable)
    }

    /// Creates a new column with both expression and pre-computed default value
    ///
    /// The pre-computed default value is used for schema evolution (backfilling existing rows)
    /// while the expression string is used for new inserts.
    ///
    /// # Arguments
    /// * `name` - The name of the column
    /// * `column_type` - The data type of the column
    /// * `nullable` - Whether the column can contain NULL values
    /// * `default_expr` - Default value expression as string (for INSERT)
    /// * `default_value` - Pre-computed default value (for schema evolution)
    fn create_column_with_default_value(
        &mut self,
        name: &str,
        column_type: DataType,
        nullable: bool,
        default_expr: Option<String>,
        _default_value: Option<crate::core::Value>,
    ) -> Result<()> {
        // Default implementation ignores default_value for backwards compatibility
        self.create_column_with_default(name, column_type, nullable, default_expr)
    }

    /// Drops a column from the table
    ///
    /// # Arguments
    /// * `name` - The name of the column to drop
    fn drop_column(&mut self, name: &str) -> Result<()>;

    /// Inserts a single row into the table
    ///
    /// # Arguments
    /// * `row` - The row to insert
    ///
    /// # Returns
    /// The inserted row (with AUTO_INCREMENT values applied)
    fn insert(&mut self, row: Row) -> Result<Row>;

    /// Inserts multiple rows into the table in a single batch operation
    ///
    /// This is more efficient than calling `insert` multiple times.
    ///
    /// # Arguments
    /// * `rows` - The rows to insert
    fn insert_batch(&mut self, rows: Vec<Row>) -> Result<()>;

    /// Updates rows matching the given expression
    ///
    /// # Arguments
    /// * `where_expr` - Expression to filter rows to update (None means all rows)
    /// * `setter` - Function that transforms a row in place, returns true if changed
    ///
    /// # Returns
    /// The number of rows updated
    fn update(
        &mut self,
        where_expr: Option<&dyn Expression>,
        setter: &mut dyn FnMut(Row) -> (Row, bool),
    ) -> Result<i32>;

    /// Deletes rows matching the given expression
    ///
    /// # Arguments
    /// * `where_expr` - Expression to filter rows to delete (None means all rows)
    ///
    /// # Returns
    /// The number of rows deleted
    fn delete(&mut self, where_expr: Option<&dyn Expression>) -> Result<i32>;

    /// Scans the table and returns a scanner over matching rows
    ///
    /// # Arguments
    /// * `column_indices` - Indices of columns to include in the scan
    /// * `where_expr` - Expression to filter rows (None means all rows)
    ///
    /// # Returns
    /// A scanner that iterates over the matching rows
    fn scan(
        &self,
        column_indices: &[usize],
        where_expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn Scanner>>;

    /// Collects all rows matching the expression without intermediate cloning
    ///
    /// This is more efficient than using scan() when you need all rows at once,
    /// as it avoids the double-clone overhead of the scanner interface.
    ///
    /// # Arguments
    /// * `where_expr` - Expression to filter rows (None means all rows)
    ///
    /// # Returns
    /// A vector of all matching rows (ownership transferred, not cloned)
    fn collect_all_rows(&self, where_expr: Option<&dyn Expression>) -> Result<Vec<Row>>;

    /// Collects rows with projection applied directly
    ///
    /// This is more efficient than using scan() for simple column projections,
    /// as it projects rows during collection without intermediate cloning.
    ///
    /// # Arguments
    /// * `column_indices` - Indices of columns to include in the result
    ///
    /// # Returns
    /// A vector of projected rows (ownership transferred, not cloned)
    fn collect_projected_rows(&self, column_indices: &[usize]) -> Result<Vec<Row>> {
        // Default implementation: collect all and project
        let all_rows = self.collect_all_rows(None)?;
        Ok(all_rows
            .into_iter()
            .map(|row| {
                let values: Vec<crate::core::Value> = column_indices
                    .iter()
                    .map(|&idx| {
                        row.get(idx)
                            .cloned()
                            .unwrap_or(crate::core::Value::null_unknown())
                    })
                    .collect();
                Row::from_values(values)
            })
            .collect())
    }

    /// Collects rows with an optional limit (LIMIT pushdown optimization)
    ///
    /// This enables early termination when only a limited number of rows are needed,
    /// avoiding the cost of scanning the entire table.
    ///
    /// # Arguments
    /// * `where_expr` - Optional filter expression
    /// * `limit` - Maximum number of rows to return
    /// * `offset` - Number of rows to skip before returning
    ///
    /// # Returns
    /// A vector of rows up to the limit (after offset)
    fn collect_rows_with_limit(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Row>> {
        // Default implementation: collect all and apply limit/offset
        let all_rows = self.collect_all_rows(where_expr)?;
        Ok(all_rows.into_iter().skip(offset).take(limit).collect())
    }

    /// Collects rows with LIMIT/OFFSET without guaranteeing deterministic order.
    ///
    /// This is an optimization for queries with LIMIT but without ORDER BY.
    /// Since SQL doesn't guarantee order for LIMIT without ORDER BY, we can
    /// skip sorting and return rows in arbitrary order. This provides significant
    /// speedup by enabling true early termination.
    ///
    /// # Arguments
    /// * `where_expr` - Optional filter expression
    /// * `limit` - Maximum number of rows to return
    /// * `offset` - Number of rows to skip before returning
    ///
    /// # Returns
    /// A vector of rows up to the limit (after offset), in arbitrary order
    fn collect_rows_with_limit_unordered(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Row>> {
        // Default implementation: delegate to ordered version
        self.collect_rows_with_limit(where_expr, limit, offset)
    }

    /// Closes the table and releases any resources
    fn close(&mut self) -> Result<()>;

    /// Commits any pending changes in this table's transaction
    ///
    /// This applies the table's local transaction changes to the global store,
    /// making them visible to other transactions.
    fn commit(&mut self) -> Result<()>;

    /// Rolls back any pending changes in this table's transaction
    ///
    /// This discards any local changes that have not been committed.
    fn rollback(&mut self);

    /// Rolls back changes to a specific timestamp (for savepoint support)
    ///
    /// Discards all local changes with timestamps greater than the specified timestamp.
    /// This is used by ROLLBACK TO SAVEPOINT to partially undo transaction changes.
    ///
    /// # Arguments
    /// * `timestamp` - The timestamp to roll back to (in nanoseconds since epoch)
    fn rollback_to_timestamp(&self, timestamp: i64);

    /// Returns true if this table has uncommitted local changes
    ///
    /// This is used by the transaction to determine if the two-phase commit
    /// protocol needs to be executed.
    fn has_local_changes(&self) -> bool;

    /// Returns the pending versions to be committed for WAL logging
    ///
    /// Returns a list of (row_id, row_data, is_deleted, txn_id) tuples representing
    /// all uncommitted changes in this table. This is called before commit()
    /// to capture changes for WAL persistence.
    ///
    /// # Returns
    /// Vec of (row_id, row_data, is_deleted, txn_id) tuples
    fn get_pending_versions(&self) -> Vec<(i64, Row, bool, i64)> {
        Vec::new() // Default implementation returns empty - override in concrete tables
    }

    // ---- Index Operations ----

    /// Creates an index on the table
    ///
    /// # Arguments
    /// * `name` - The name of the index
    /// * `columns` - The column names to include in the index
    /// * `is_unique` - Whether this is a unique index
    fn create_index(&self, name: &str, columns: &[&str], is_unique: bool) -> Result<()>;

    /// Creates an index on the table with a specific index type
    ///
    /// # Arguments
    /// * `name` - The name of the index
    /// * `columns` - The column names to include in the index
    /// * `is_unique` - Whether this is a unique index
    /// * `index_type` - Optional index type (Hash, BTree, Bitmap). If None, auto-selects based on column types.
    ///
    /// # Type-Based Index Selection (when index_type is None):
    /// - TEXT/JSON columns → Hash index (avoids O(strlen) comparisons)
    /// - BOOLEAN columns → Bitmap index (only 2 values, fast AND/OR)
    /// - INTEGER/FLOAT/TIMESTAMP columns → BTree index (supports range queries)
    fn create_index_with_type(
        &self,
        name: &str,
        columns: &[&str],
        is_unique: bool,
        index_type: Option<IndexType>,
    ) -> Result<()> {
        // Default implementation calls create_index (ignores index_type)
        let _ = index_type;
        self.create_index(name, columns, is_unique)
    }

    /// Drops an index from the table
    ///
    /// # Arguments
    /// * `name` - The name of the index to drop
    fn drop_index(&self, name: &str) -> Result<()>;

    /// Creates a btree index on a column
    ///
    /// # Arguments
    /// * `column_name` - The column to index
    /// * `is_unique` - Whether this is a unique index
    /// * `custom_name` - Optional custom name for the index
    fn create_btree_index(
        &self,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()>;

    /// Drops a btree index from the table
    ///
    /// # Arguments
    /// * `column_name` - The column whose index to drop
    fn drop_btree_index(&self, column_name: &str) -> Result<()>;

    /// Creates a multi-column index on the table
    ///
    /// # Arguments
    /// * `name` - The name of the index
    /// * `columns` - The column names to include in the index
    /// * `is_unique` - Whether this is a unique index
    fn create_multi_column_index(
        &self,
        name: &str,
        columns: &[&str],
        is_unique: bool,
    ) -> Result<()> {
        let _ = (name, columns, is_unique);
        Err(Error::NotSupportedMessage(
            "Multi-column indexes not supported by this table type".to_string(),
        ))
    }

    /// Checks if an index exists on a specific column
    ///
    /// # Arguments
    /// * `column_name` - The column to check for an index
    ///
    /// # Returns
    /// true if an index exists on the column, false otherwise
    fn has_index_on_column(&self, column_name: &str) -> bool {
        let _ = column_name;
        false // Default implementation - override in concrete tables
    }

    /// Gets the index on a specific column (if any exists)
    ///
    /// # Arguments
    /// * `column_name` - The column to get the index for
    ///
    /// # Returns
    /// Some(index) if an index exists on the column, None otherwise
    fn get_index_on_column(&self, column_name: &str) -> Option<std::sync::Arc<dyn Index>> {
        let _ = column_name;
        None // Default implementation - override in concrete tables
    }

    /// Gets an index by name
    ///
    /// # Arguments
    /// * `name` - The name of the index
    ///
    /// # Returns
    /// Some(index) if found, None otherwise
    fn get_index(&self, name: &str) -> Option<std::sync::Arc<dyn Index>> {
        let _ = name;
        None // Default implementation - override in concrete tables
    }

    /// Find the best multi-column index that matches a set of predicate columns.
    /// Returns the index if it covers a prefix of the given columns.
    /// For example, an index on (a, b, c) can be used for queries on (a), (a, b), or (a, b, c).
    ///
    /// # Arguments
    /// * `predicate_columns` - The columns used in WHERE clause predicates
    ///
    /// # Returns
    /// Some((index, matched_columns)) if found, None otherwise
    fn get_multi_column_index(
        &self,
        predicate_columns: &[&str],
    ) -> Option<(std::sync::Arc<dyn Index>, usize)> {
        let _ = predicate_columns;
        None // Default implementation - override in concrete tables
    }

    /// Gets the minimum value from an indexed column (O(1) or O(log n) instead of O(n) scan)
    ///
    /// # Arguments
    /// * `column_name` - The column to get the minimum value from
    ///
    /// # Returns
    /// Some(Value) if the column has an index with min/max support, None otherwise
    fn get_index_min_value(&self, column_name: &str) -> Option<Value> {
        let _ = column_name;
        None // Default implementation - override in concrete tables
    }

    /// Gets the maximum value from an indexed column (O(1) or O(log n) instead of O(n) scan)
    ///
    /// # Arguments
    /// * `column_name` - The column to get the maximum value from
    ///
    /// # Returns
    /// Some(Value) if the column has an index with min/max support, None otherwise
    fn get_index_max_value(&self, column_name: &str) -> Option<Value> {
        let _ = column_name;
        None // Default implementation - override in concrete tables
    }

    /// Gets the count of rows in the table (COUNT(*) pushdown optimization)
    ///
    /// This enables O(1) row counting instead of O(n) scan for `SELECT COUNT(*) FROM table`
    /// without WHERE clause.
    ///
    /// # Returns
    /// The number of visible rows in the table
    fn row_count(&self) -> usize {
        0 // Default implementation - override in concrete tables
    }

    /// Collects rows sorted by an indexed column with limit (ORDER BY + LIMIT pushdown)
    ///
    /// For queries like `SELECT * FROM table ORDER BY col LIMIT 10`, this uses the
    /// index to get rows in sorted order, stopping after the limit is reached.
    /// This is O(limit) instead of O(n log n) for full sort.
    ///
    /// # Arguments
    /// * `column_name` - The indexed column to order by
    /// * `ascending` - True for ASC, false for DESC
    /// * `limit` - Maximum number of rows to return
    /// * `offset` - Number of rows to skip
    ///
    /// # Returns
    /// Some(Vec<Row>) if the column has an index, None otherwise
    fn collect_rows_ordered_by_index(
        &self,
        column_name: &str,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<Vec<Row>> {
        let _ = (column_name, ascending, limit, offset);
        None // Default implementation - override in concrete tables
    }

    /// Collects rows grouped by an indexed partition column (PARTITION BY optimization)
    ///
    /// For window functions with `PARTITION BY col` where col is indexed, this uses the
    /// index to iterate through unique values and collect rows already grouped by partition.
    /// This avoids O(n) hash-based grouping in window function execution.
    ///
    /// # Arguments
    /// * `column_name` - The indexed column to partition by
    ///
    /// # Returns
    /// Some(Vec<(Value, Vec<Row>)>) where each tuple is (partition_value, rows_in_partition)
    /// Returns None if the column has no index
    fn collect_rows_grouped_by_partition(
        &self,
        column_name: &str,
    ) -> Option<Vec<(Value, Vec<Row>)>> {
        let _ = column_name;
        None // Default implementation - override in concrete tables
    }

    /// Get distinct partition values from an indexed column.
    /// Used for LIMIT pushdown in window functions - allows fetching partitions one at a time.
    ///
    /// # Arguments
    /// * `column_name` - The indexed column to get partition values from
    ///
    /// # Returns
    /// Some(Vec<Value>) with distinct values, or None if column has no index
    fn get_partition_values(&self, column_name: &str) -> Option<Vec<Value>> {
        let _ = column_name;
        None // Default implementation - override in concrete tables
    }

    /// Get rows for a specific partition value.
    /// Used for LIMIT pushdown in window functions - fetches only one partition at a time.
    ///
    /// # Arguments
    /// * `column_name` - The indexed column
    /// * `partition_value` - The specific partition value to fetch rows for
    ///
    /// # Returns
    /// Some(Vec<Row>) with rows matching the partition value, or None if column has no index
    fn get_rows_for_partition_value(
        &self,
        column_name: &str,
        partition_value: &Value,
    ) -> Option<Vec<Row>> {
        let _ = (column_name, partition_value);
        None // Default implementation - override in concrete tables
    }

    /// Fetch rows by their row IDs with an optional filter.
    ///
    /// This is used for index-based lookups where we have a list of row IDs
    /// and need to fetch the actual rows efficiently.
    ///
    /// # Arguments
    /// * `row_ids` - The row IDs to fetch
    /// * `filter` - Filter expression to apply to fetched rows
    ///
    /// # Returns
    /// Vector of (row_id, Row) pairs for visible, non-deleted rows that pass the filter
    fn fetch_rows_by_ids(&self, row_ids: &[i64], filter: &dyn Expression) -> Vec<(i64, Row)> {
        let _ = (row_ids, filter);
        Vec::new() // Default implementation - override in concrete tables
    }

    // ---- Additional Column Operations ----

    /// Renames a column in the table
    ///
    /// # Arguments
    /// * `old_name` - Current column name
    /// * `new_name` - New column name
    fn rename_column(&self, old_name: &str, new_name: &str) -> Result<()>;

    /// Modifies a column's definition
    ///
    /// # Arguments
    /// * `name` - The column name
    /// * `column_type` - The new data type
    /// * `nullable` - Whether the column can contain NULL values
    fn modify_column(&self, name: &str, column_type: DataType, nullable: bool) -> Result<()>;

    // ---- Query Operations ----

    /// Executes a SELECT query on the table
    ///
    /// # Arguments
    /// * `columns` - Column names to include in the result
    /// * `expr` - Optional filter expression
    ///
    /// # Returns
    /// A QueryResult with the matching rows
    fn select(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn QueryResult>>;

    /// Executes a SELECT query with column aliases
    ///
    /// # Arguments
    /// * `columns` - Column names to include in the result
    /// * `expr` - Optional filter expression
    /// * `aliases` - Map from alias names to original column names
    ///
    /// # Returns
    /// A QueryResult with the matching rows and aliased column names
    fn select_with_aliases(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        aliases: &FxHashMap<String, String>,
    ) -> Result<Box<dyn QueryResult>>;

    /// Executes a temporal SELECT query as of a specific point in time
    ///
    /// # Arguments
    /// * `columns` - Column names to include in the result
    /// * `expr` - Optional filter expression
    /// * `temporal_type` - Either "TRANSACTION" or "TIMESTAMP"
    /// * `temporal_value` - Transaction ID or timestamp in nanoseconds
    ///
    /// # Returns
    /// A QueryResult with rows as they were at the specified point
    fn select_as_of(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        temporal_type: &str,
        temporal_value: i64,
    ) -> Result<Box<dyn QueryResult>>;

    /// Explains what access method would be used for a scan
    ///
    /// This method analyzes the WHERE expression and returns a ScanPlan
    /// describing how the query would be executed (without actually executing it).
    /// Used by EXPLAIN to show users the query execution strategy.
    ///
    /// # Arguments
    /// * `where_expr` - Optional filter expression to analyze
    ///
    /// # Returns
    /// A ScanPlan describing the access method that would be used
    fn explain_scan(&self, where_expr: Option<&dyn Expression>) -> ScanPlan {
        // Default implementation returns SeqScan
        ScanPlan::SeqScan {
            table: self.name().to_string(),
            filter: where_expr.map(|e| format!("{:?}", e)),
        }
    }

    // ---- Zone Map Operations (Statistics for Segment Pruning) ----

    /// Sets the zone maps for this table
    ///
    /// Zone maps contain min/max statistics per segment, enabling the query
    /// executor to skip entire segments when predicates fall outside the range.
    /// This is typically called by ANALYZE.
    ///
    /// # Arguments
    /// * `zone_maps` - The zone map statistics for the table
    fn set_zone_maps(&self, _zone_maps: crate::storage::mvcc::zonemap::TableZoneMap) {
        // Default implementation does nothing - override in concrete tables
    }

    /// Gets the zone maps for this table
    ///
    /// Returns None if zone maps have not been built (ANALYZE not run)
    /// Uses Arc to avoid expensive cloning on high QPS workloads
    fn get_zone_maps(&self) -> Option<std::sync::Arc<crate::storage::mvcc::zonemap::TableZoneMap>> {
        None // Default implementation - override in concrete tables
    }

    /// Gets the segments that need to be scanned for a given predicate
    ///
    /// Uses zone maps to determine which segments can be pruned (skipped)
    /// based on the predicate's column, operator, and value.
    ///
    /// # Arguments
    /// * `column` - The column name in the predicate
    /// * `operator` - The comparison operator
    /// * `value` - The value being compared against
    ///
    /// # Returns
    /// Some(Vec<segment_ids>) if zone maps exist, None otherwise
    fn get_segments_to_scan(
        &self,
        _column: &str,
        _operator: crate::core::Operator,
        _value: &Value,
    ) -> Option<Vec<u32>> {
        None // Default implementation - override in concrete tables
    }
}

#[cfg(test)]
mod tests {
    // Table tests will be implemented when we have concrete implementations
    // For now, just verify the trait compiles correctly

    use super::*;

    // Verify trait is object-safe
    fn _assert_object_safe(_: &dyn Table) {}
}
