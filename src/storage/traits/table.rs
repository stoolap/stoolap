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

//! Table traits. [`ReadTable`] is the non-mutating surface;
//! [`WriteTable`] extends it with mutators. Read-only callers receive
//! `Box<dyn ReadTable>` and cannot reach any mutator at compile time.

use rustc_hash::FxHashMap;
use std::fmt;

use crate::core::{DataType, Error, IndexType, Result, Row, RowVec, Schema, Value};
use crate::storage::expression::Expression;
use crate::storage::mvcc::version_store::{AggregateOp, GroupedAggregateResult};
use crate::storage::traits::{Index, QueryResult, Scanner};

/// Access method used for a table scan (consumed by EXPLAIN).
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
        /// Non-indexed predicates applied as in-memory filter after index lookup
        filter: Option<String>,
    },
    /// Multi-index scan - uses multiple indexes with AND/OR operations
    MultiIndexScan {
        table: String,
        indexes: Vec<(String, String, String)>, // (index_name, column, condition)
        operation: String,                      // "AND" or "OR"
        /// Non-indexed predicates applied as in-memory filter after index lookup
        filter: Option<String>,
    },
    /// Composite index scan - uses a multi-column index
    CompositeIndexScan {
        table: String,
        index_name: String,
        columns: Vec<String>,
        conditions: Vec<String>,
        /// Non-indexed predicates applied as in-memory filter after index lookup
        filter: Option<String>,
    },
    /// HNSW approximate nearest neighbor search
    VectorSearch {
        table: String,
        index_name: String,
        vector_column: String,
        metric: String,
        k: usize,
        ef_search: usize,
        filter: Option<String>,
    },
    /// Brute-force parallel vector distance scan
    VectorBruteForce {
        table: String,
        vector_column: String,
        metric: String,
        k: usize,
        filter: Option<String>,
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
                filter,
            } => {
                write!(
                    f,
                    "Index Scan using {} on {}\n  Index Cond: {} {}",
                    index_name, table, column, condition
                )?;
                if let Some(flt) = filter {
                    write!(f, "\n  Filter: {}", flt)?;
                }
                Ok(())
            }
            ScanPlan::MultiIndexScan {
                table,
                indexes,
                operation,
                filter,
            } => {
                write!(f, "Multi-Index Scan on {} ({})", table, operation)?;
                for (idx_name, col, cond) in indexes {
                    write!(f, "\n  -> {} on {}: {}", idx_name, col, cond)?;
                }
                if let Some(flt) = filter {
                    write!(f, "\n  Filter: {}", flt)?;
                }
                Ok(())
            }
            ScanPlan::CompositeIndexScan {
                table,
                index_name,
                columns,
                conditions,
                filter,
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
                if let Some(flt) = filter {
                    write!(f, "\n  Filter: {}", flt)?;
                }
                Ok(())
            }
            ScanPlan::VectorSearch {
                table,
                index_name,
                vector_column,
                metric,
                k,
                ef_search,
                filter,
            } => {
                write!(
                    f,
                    "HNSW Index Scan using {} on {}\n  Column: {}\n  Metric: {}\n  K: {}\n  EF Search: {}",
                    index_name, table, vector_column, metric, k, ef_search
                )?;
                if let Some(flt) = filter {
                    write!(f, "\n  Filter: {}", flt)?;
                }
                Ok(())
            }
            ScanPlan::VectorBruteForce {
                table,
                vector_column,
                metric,
                k,
                filter,
            } => {
                write!(
                    f,
                    "Vector Scan on {}\n  Column: {}\n  Metric: {}\n  K: {}",
                    table, vector_column, metric, k
                )?;
                if let Some(flt) = filter {
                    write!(f, "\n  Filter: {}", flt)?;
                }
                Ok(())
            }
        }
    }
}

/// Read-only surface of a database table.
///
/// Every method is non-mutating to shared engine state. `&mut self`
/// methods (`close`, `rollback`, `rollback_to_timestamp`) touch only
/// transaction-local state and are safe on read-only handles.
pub trait ReadTable: Send + Sync {
    /// Returns the name of the table.
    fn name(&self) -> &str;

    /// Returns the schema of the table.
    fn schema(&self) -> &Schema;

    /// Returns the transaction ID this table handle belongs to.
    fn txn_id(&self) -> i64;

    /// Returns all active row IDs visible to the current transaction
    /// (used for NOT IN anti-join optimization).
    fn get_active_row_ids(&self) -> Vec<i64>;

    /// Populates `dest` with hot row_ids without an intermediate `Vec`.
    fn collect_hot_row_ids_into(&self, dest: &mut rustc_hash::FxHashSet<i64>) {
        for id in self.get_active_row_ids() {
            dest.insert(id);
        }
    }

    /// Returns true if `row_id` exists in the hot buffer.
    fn has_row_id(&self, _row_id: i64) -> bool {
        false
    }

    /// Returns a scanner over rows matching `where_expr`, projecting `column_indices`.
    fn scan(
        &self,
        column_indices: &[usize],
        where_expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn Scanner>>;

    /// Collects all rows matching `where_expr` (None = all rows).
    fn collect_all_rows(&self, where_expr: Option<&dyn Expression>) -> Result<RowVec>;

    /// Collects rows matching `where_expr` with LIMIT/OFFSET pushdown.
    fn collect_rows_with_limit(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<RowVec> {
        let all_rows = self.collect_all_rows(where_expr)?;
        Ok(all_rows.into_iter().skip(offset).take(limit).collect())
    }

    /// Like [`collect_rows_with_limit`] but order is unspecified
    /// (LIMIT without ORDER BY enables true early termination).
    fn collect_rows_with_limit_unordered(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<RowVec> {
        self.collect_rows_with_limit(where_expr, limit, offset)
    }

    /// Collects all rows in arbitrary order (skips sort for GROUP BY).
    fn collect_all_rows_unsorted(&self) -> Result<RowVec> {
        self.collect_all_rows(None)
    }

    /// Fetches rows for the given `row_ids`, preserving input order
    /// (used by HNSW after ANN lookup).
    fn collect_rows_by_ids(&self, _row_ids: &[i64]) -> Result<RowVec> {
        Err(Error::NotSupported(
            "collect_rows_by_ids not implemented".to_string(),
        ))
    }

    /// Collects top-N rows ordered by `sort_col_idx` (deferred materialization).
    fn collect_rows_sorted_with_limit(
        &self,
        sort_col_idx: usize,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Row>> {
        let mut rows = self.collect_all_rows(None)?;
        rows.sort_by(|(_, a), (_, b)| {
            let va = a.get(sort_col_idx);
            let vb = b.get(sort_col_idx);
            let cmp = match (va, vb) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(va), Some(vb)) => va.compare(vb).unwrap_or(std::cmp::Ordering::Equal),
            };
            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });
        Ok(rows
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(_, row)| row)
            .collect())
    }

    /// Closes the table and releases any resources. Lives on `ReadTable`
    /// because the executor's transaction cleanup calls `close()` on every
    /// cached read-side handle.
    fn close(&mut self) -> Result<()>;

    /// Discards transaction-local state. Lives on `ReadTable` for the same
    /// reason as [`close`](Self::close).
    fn rollback(&mut self);

    /// Discards local changes with timestamps greater than `timestamp`
    /// (nanoseconds since epoch). Used by ROLLBACK TO SAVEPOINT.
    fn rollback_to_timestamp(&self, timestamp: i64);

    /// Returns true if the table has uncommitted local changes
    /// (drives the two-phase commit path).
    fn has_local_changes(&self) -> bool;

    /// Returns `(row_id, row_data, is_deleted, txn_id)` for each pending
    /// version, captured before commit for WAL persistence.
    fn get_pending_versions(&self) -> Vec<(i64, Row, bool, i64)> {
        Vec::new()
    }

    // ---- Index Lookups (read-only) ----

    /// Returns true if `column_name` has an index.
    fn has_index_on_column(&self, column_name: &str) -> bool {
        let _ = column_name;
        false
    }

    /// Returns the index covering `column_name`, if any.
    fn get_index_on_column(&self, column_name: &str) -> Option<std::sync::Arc<dyn Index>> {
        let _ = column_name;
        None
    }

    /// Returns `(index_name, column_names)` for every unique index on the table.
    fn get_unique_indexes(&self) -> Vec<(String, Vec<String>)> {
        Vec::new()
    }

    /// Iterates unique non-PK indexes by reference to avoid String allocations.
    /// Default delegates to [`get_unique_indexes`](Self::get_unique_indexes).
    fn for_each_unique_non_pk_index(
        &self,
        f: &mut dyn FnMut(&str, &[String]) -> Result<()>,
    ) -> Result<()> {
        for (name, cols) in self.get_unique_indexes() {
            f(&name, &cols)?;
        }
        Ok(())
    }

    /// Fast bail-out: returns true if any unique non-PK index exists.
    fn has_unique_non_pk_indexes(&self) -> bool {
        false
    }

    /// Returns the index named `name`, if any.
    fn get_index(&self, name: &str) -> Option<std::sync::Arc<dyn Index>> {
        let _ = name;
        None
    }

    /// Returns the best multi-column index covering a prefix of
    /// `predicate_columns`, plus the matched-prefix length.
    fn get_multi_column_index(
        &self,
        predicate_columns: &[&str],
    ) -> Option<(std::sync::Arc<dyn Index>, usize)> {
        let _ = predicate_columns;
        None
    }

    /// Returns the minimum value of an indexed column (avoids O(n) scan).
    fn get_index_min_value(&self, column_name: &str) -> Option<Value> {
        let _ = column_name;
        None
    }

    /// Returns the maximum value of an indexed column (avoids O(n) scan).
    fn get_index_max_value(&self, column_name: &str) -> Option<Value> {
        let _ = column_name;
        None
    }

    // ---- Counts / Partitions ----

    /// Returns the visible row count (COUNT(*) without WHERE).
    fn row_count(&self) -> usize {
        0
    }

    /// Cheap upper-bound row-count estimate for optimizer decisions.
    fn row_count_hint(&self) -> usize {
        self.row_count()
    }

    /// Returns the exact row count via the COUNT(*) fast path, or None
    /// if the caller must fall back to [`row_count`](Self::row_count).
    fn fast_row_count(&self) -> Option<usize> {
        None
    }

    /// ORDER BY + LIMIT pushdown using an index on `column_name`. Returns
    /// None when the column is unindexed.
    fn collect_rows_ordered_by_index(
        &self,
        column_name: &str,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<RowVec> {
        let _ = (column_name, ascending, limit, offset);
        None
    }

    /// Keyset pagination on an INTEGER PRIMARY KEY. `start_after` is
    /// exclusive (`>`), `start_from` is inclusive (`>=`).
    fn collect_rows_pk_keyset(
        &self,
        start_after: Option<i64>,
        start_from: Option<i64>,
        ascending: bool,
        limit: usize,
    ) -> Option<RowVec> {
        let _ = (start_after, start_from, ascending, limit);
        None
    }

    /// Collects rows grouped by an indexed partition column. Returns None
    /// if the column has no index.
    fn collect_rows_grouped_by_partition(&self, column_name: &str) -> Option<Vec<(Value, RowVec)>> {
        let _ = column_name;
        None
    }

    /// Returns distinct values of an indexed column (LIMIT-pushdown for window funcs).
    fn get_partition_values(&self, column_name: &str) -> Option<Vec<Value>> {
        let _ = column_name;
        None
    }

    /// Returns distinct non-null values of `col_idx` from cold-volume
    /// dictionary metadata when available.
    fn compute_distinct_values(&self, _col_idx: usize) -> Option<Vec<Value>> {
        None
    }

    /// Returns the count of distinct non-null values for an indexed column.
    fn get_partition_count(&self, column_name: &str) -> Option<usize> {
        let _ = column_name;
        None
    }

    /// Returns rows matching `partition_value` on an indexed column.
    fn get_rows_for_partition_value(
        &self,
        column_name: &str,
        partition_value: &Value,
    ) -> Option<RowVec> {
        let _ = (column_name, partition_value);
        None
    }

    /// Fetches rows by id and applies `filter`, returning visible non-deleted rows.
    fn fetch_rows_by_ids(&self, row_ids: &[i64], filter: &dyn Expression) -> RowVec {
        let mut results = RowVec::with_capacity(row_ids.len());
        self.fetch_rows_by_ids_into(row_ids, filter, &mut results);
        results
    }

    /// Like [`fetch_rows_by_ids`](Self::fetch_rows_by_ids) but writes into a reusable buffer.
    fn fetch_rows_by_ids_into(
        &self,
        row_ids: &[i64],
        filter: &dyn Expression,
        buffer: &mut RowVec,
    ) {
        let _ = (row_ids, filter, buffer);
    }

    // ---- Query Operations ----

    /// Executes a SELECT projecting `columns` and filtering by `expr`.
    fn select(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn QueryResult>>;

    /// Like [`select`](Self::select) but with output `aliases` (alias -> original).
    fn select_with_aliases(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        aliases: &FxHashMap<String, String>,
    ) -> Result<Box<dyn QueryResult>>;

    /// Temporal SELECT. `temporal_type` is `"TRANSACTION"` or `"TIMESTAMP"`;
    /// `temporal_value` is the txn id or nanoseconds since epoch.
    fn select_as_of(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        temporal_type: &str,
        temporal_value: i64,
    ) -> Result<Box<dyn QueryResult>>;

    /// Returns a [`ScanPlan`] describing how `where_expr` would be executed
    /// (used by EXPLAIN). Default returns [`ScanPlan::SeqScan`].
    fn explain_scan(&self, where_expr: Option<&dyn Expression>) -> ScanPlan {
        ScanPlan::SeqScan {
            table: self.name().to_string(),
            filter: where_expr.map(|e| format!("{:?}", e)),
        }
    }

    /// Returns the table's zone maps if ANALYZE has been run.
    fn get_zone_maps(&self) -> Option<std::sync::Arc<crate::storage::mvcc::zonemap::TableZoneMap>> {
        None
    }

    /// Returns segment ids that survive zone-map pruning for `(column, operator, value)`,
    /// or None when zone maps are absent.
    fn get_segments_to_scan(
        &self,
        _column: &str,
        _operator: crate::core::Operator,
        _value: &Value,
    ) -> Option<Vec<u32>> {
        None
    }

    // ---- Deferred Aggregation (pushdown) ----

    /// Pushed-down SUM. Returns `(sum, count_non_null)` or None if unavailable.
    fn sum_column(&self, _col_idx: usize) -> Option<(f64, usize)> {
        None
    }

    /// Pushed-down AVG. Returns `(sum, count_non_null)` for caller to divide.
    fn avg_column(&self, _col_idx: usize) -> Option<(f64, usize)> {
        self.sum_column(_col_idx)
    }

    /// Pushed-down MIN. Outer None means unavailable; inner None means no non-NULL rows.
    fn min_column(&self, _col_idx: usize) -> Option<Option<Value>> {
        None
    }

    /// Pushed-down MAX. Outer None means unavailable; inner None means no non-NULL rows.
    fn max_column(&self, _col_idx: usize) -> Option<Option<Value>> {
        None
    }

    /// Pushed-down filtered aggregation: one Value per `(op, col_idx)` pair.
    fn compute_filtered_aggregates(
        &self,
        _aggregates: &[(AggregateOp, usize)],
        _where_expr: &dyn Expression,
    ) -> Option<Vec<crate::core::Value>> {
        None
    }

    /// Pushed-down GROUP BY aggregation directly on arena storage.
    fn compute_grouped_aggregates(
        &self,
        _group_by_indices: &[usize],
        _aggregates: &[(AggregateOp, usize)],
    ) -> Option<Vec<GroupedAggregateResult>> {
        None
    }
}

/// Writable surface of a database table. Extends [`ReadTable`] with
/// DML, DDL, and commit. Read-only callers cannot reach this trait.
pub trait WriteTable: ReadTable {
    // ---- Column DDL ----

    /// Adds column `name` of `column_type`.
    fn create_column(&mut self, name: &str, column_type: DataType, nullable: bool) -> Result<()>;

    /// Adds a column with a `DEFAULT` expression evaluated on each INSERT.
    fn create_column_with_default(
        &mut self,
        name: &str,
        column_type: DataType,
        nullable: bool,
        _default_expr: Option<String>,
    ) -> Result<()> {
        self.create_column(name, column_type, nullable)
    }

    /// Adds a column with both a DEFAULT expression (for new inserts) and a
    /// pre-computed value (for backfilling existing rows during schema evolution).
    fn create_column_with_default_value(
        &mut self,
        name: &str,
        column_type: DataType,
        nullable: bool,
        default_expr: Option<String>,
        _default_value: Option<crate::core::Value>,
    ) -> Result<()> {
        self.create_column_with_default(name, column_type, nullable, default_expr)
    }

    /// Drops the column named `name`.
    fn drop_column(&mut self, name: &str) -> Result<()>;

    /// Renames `old_name` to `new_name`.
    fn rename_column(&mut self, old_name: &str, new_name: &str) -> Result<()>;

    /// Changes the type and nullability of column `name`.
    fn modify_column(&mut self, name: &str, column_type: DataType, nullable: bool) -> Result<()>;

    // ---- DML ----

    /// Inserts `row`, returning the row with AUTO_INCREMENT values applied.
    fn insert(&mut self, row: Row) -> Result<Row>;

    /// Inserts `row` without returning it (skips a Row clone; use when no RETURNING).
    fn insert_discard(&mut self, row: Row) -> Result<()> {
        self.insert(row)?;
        Ok(())
    }

    /// Inserts `rows` as a single batch.
    fn insert_batch(&mut self, rows: Vec<Row>) -> Result<()>;

    /// Updates rows matching `where_expr` (None = all). `setter` returns
    /// `(new_row, changed)`. Returns rows updated.
    fn update(
        &mut self,
        where_expr: Option<&dyn Expression>,
        setter: &mut dyn FnMut(Row) -> Result<(Row, bool)>,
    ) -> Result<i32>;

    /// O(k) UPDATE by row id (used for `UPDATE ... WHERE pk IN (...)`).
    /// Pass `row_ids` sorted for cache locality.
    fn update_by_row_ids(
        &mut self,
        row_ids: &[i64],
        setter: &mut dyn FnMut(Row) -> Result<(Row, bool)>,
    ) -> Result<i32>;

    /// Deletes rows matching `where_expr` (None = all). Returns rows deleted.
    fn delete(&mut self, where_expr: Option<&dyn Expression>) -> Result<i32>;

    /// O(k) DELETE by row id. Pass `row_ids` sorted for cache locality.
    fn delete_by_row_ids(&mut self, row_ids: &[i64]) -> Result<i32>;

    /// Drops storage directly instead of recording delete versions. Default
    /// falls back to `delete(None)`.
    fn truncate(&mut self) -> Result<i32> {
        self.delete(None)
    }

    /// Claims `row_id` in the VersionStore's `uncommitted_writes` map so that
    /// concurrent transactions mirroring the same cold row detect each other.
    /// Takes `&self` but mutates claim state. Required only on `WriteTable`.
    fn try_claim_row(&self, _row_id: i64) -> Result<()> {
        Ok(())
    }

    // ---- Index DDL ----

    /// Creates an index named `name` over `columns`.
    fn create_index(&self, name: &str, columns: &[&str], is_unique: bool) -> Result<()>;

    /// Like [`create_index`](Self::create_index) but with an explicit
    /// `index_type`. None auto-selects: TEXT/JSON -> Hash, BOOLEAN -> Bitmap,
    /// INTEGER/FLOAT/TIMESTAMP -> BTree.
    fn create_index_with_type(
        &self,
        name: &str,
        columns: &[&str],
        is_unique: bool,
        index_type: Option<IndexType>,
    ) -> Result<()> {
        let _ = index_type;
        self.create_index(name, columns, is_unique)
    }

    /// Creates an HNSW vector index. `m` is max connections per node per layer;
    /// `ef_construction`/`ef_search` are build/search beam widths.
    #[allow(clippy::too_many_arguments)]
    fn create_hnsw_index(
        &self,
        name: &str,
        column: &str,
        is_unique: bool,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: crate::storage::index::HnswDistanceMetric,
    ) -> Result<()> {
        let _ = (
            name,
            column,
            is_unique,
            m,
            ef_construction,
            ef_search,
            metric,
        );
        Err(crate::core::Error::internal(
            "HNSW index not supported by this storage engine".to_string(),
        ))
    }

    /// Drops the index named `name`.
    fn drop_index(&self, name: &str) -> Result<()>;

    /// Creates a btree index on `column_name`.
    fn create_btree_index(
        &self,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()>;

    /// Drops the btree index covering `column_name`.
    fn drop_btree_index(&self, column_name: &str) -> Result<()>;

    /// Creates a multi-column index. Default returns NotSupported.
    fn create_multi_column_index(
        &self,
        name: &str,
        columns: &[&str],
        is_unique: bool,
    ) -> Result<()> {
        let _ = (name, columns, is_unique);
        Err(Error::NotSupported(
            "Multi-column indexes not supported by this table type".to_string(),
        ))
    }

    // ---- Unique constraint enforcement ----

    /// Returns the row id that conflicts with the unique key composed of
    /// `row_values` on `(index_name, column_name)`. Volume-backed tables
    /// override to probe cold storage after hot misses.
    fn find_unique_conflict_row_id(
        &self,
        _index_name: &str,
        _column_name: &str,
        _row_values: &[Value],
    ) -> Result<Option<i64>> {
        Ok(None)
    }

    /// Acquires the per-table upsert mutex for ON CONFLICT serialization.
    /// The returned `Box<dyn Any>` erases the guard type only.
    fn acquire_upsert_lock(&self) -> Option<Box<dyn std::any::Any>> {
        None
    }

    // ---- Zone maps (write side) ----

    /// Installs zone-map statistics (typically called by ANALYZE).
    fn set_zone_maps(&self, _zone_maps: crate::storage::mvcc::zonemap::TableZoneMap) {}

    // ---- Lifecycle ----

    /// Applies pending changes to the global store, making them visible.
    fn commit(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_read_object_safe(_: &dyn ReadTable) {}
    fn _assert_write_object_safe(_: &dyn WriteTable) {}

    #[test]
    fn test_seq_scan_display_without_filter() {
        let plan = ScanPlan::SeqScan {
            table: "users".to_string(),
            filter: None,
        };
        assert_eq!(plan.to_string(), "Seq Scan on users");
    }

    #[test]
    fn test_seq_scan_display_with_filter() {
        let plan = ScanPlan::SeqScan {
            table: "users".to_string(),
            filter: Some("age > 18".to_string()),
        };
        assert_eq!(plan.to_string(), "Seq Scan on users\n  Filter: age > 18");
    }

    #[test]
    fn test_parallel_seq_scan_display_without_filter() {
        let plan = ScanPlan::ParallelSeqScan {
            table: "users".to_string(),
            filter: None,
            workers: 4,
        };
        assert_eq!(plan.to_string(), "Parallel Seq Scan on users (workers=4)");
    }

    #[test]
    fn test_parallel_seq_scan_display_with_filter() {
        let plan = ScanPlan::ParallelSeqScan {
            table: "users".to_string(),
            filter: Some("age > 18".to_string()),
            workers: 4,
        };
        assert_eq!(
            plan.to_string(),
            "Parallel Seq Scan on users (workers=4)\n  Filter: age > 18"
        );
    }

    #[test]
    fn test_pk_lookup_display() {
        let plan = ScanPlan::PkLookup {
            table: "users".to_string(),
            pk_column: "id".to_string(),
            pk_value: "42".to_string(),
        };
        assert_eq!(plan.to_string(), "PK Lookup on users\n  id = 42");
    }

    #[test]
    fn test_index_scan_display() {
        let plan = ScanPlan::IndexScan {
            table: "users".to_string(),
            index_name: "idx_email".to_string(),
            column: "email".to_string(),
            condition: "= 'alice@example.com'".to_string(),
            filter: None,
        };
        assert_eq!(
            plan.to_string(),
            "Index Scan using idx_email on users\n  Index Cond: email = 'alice@example.com'"
        );
    }

    #[test]
    fn test_multi_index_scan_display_and() {
        let plan = ScanPlan::MultiIndexScan {
            table: "users".to_string(),
            indexes: vec![
                ("idx_age".to_string(), "age".to_string(), "> 18".to_string()),
                (
                    "idx_status".to_string(),
                    "status".to_string(),
                    "= 'active'".to_string(),
                ),
            ],
            operation: "AND".to_string(),
            filter: None,
        };
        assert_eq!(
            plan.to_string(),
            "Multi-Index Scan on users (AND)\n  -> idx_age on age: > 18\n  -> idx_status on status: = 'active'"
        );
    }

    #[test]
    fn test_multi_index_scan_display_or() {
        let plan = ScanPlan::MultiIndexScan {
            table: "products".to_string(),
            indexes: vec![
                (
                    "idx_name".to_string(),
                    "name".to_string(),
                    "LIKE 'A%'".to_string(),
                ),
                (
                    "idx_sku".to_string(),
                    "sku".to_string(),
                    "= 'XYZ'".to_string(),
                ),
            ],
            operation: "OR".to_string(),
            filter: None,
        };
        assert_eq!(
            plan.to_string(),
            "Multi-Index Scan on products (OR)\n  -> idx_name on name: LIKE 'A%'\n  -> idx_sku on sku: = 'XYZ'"
        );
    }

    #[test]
    fn test_composite_index_scan_display() {
        let plan = ScanPlan::CompositeIndexScan {
            table: "orders".to_string(),
            index_name: "idx_user_date".to_string(),
            columns: vec!["user_id".to_string(), "order_date".to_string()],
            conditions: vec!["= 42".to_string(), "> '2024-01-01'".to_string()],
            filter: None,
        };
        assert_eq!(
            plan.to_string(),
            "Composite Index Scan using idx_user_date on orders\n  Columns: (user_id, order_date)\n  user_id = 42\n  order_date > '2024-01-01'"
        );
    }

    #[test]
    fn test_scan_plan_debug() {
        let plan = ScanPlan::SeqScan {
            table: "users".to_string(),
            filter: None,
        };
        let debug_str = format!("{:?}", plan);
        assert!(debug_str.contains("SeqScan"));
        assert!(debug_str.contains("users"));
    }

    #[test]
    fn test_scan_plan_clone() {
        let plan = ScanPlan::IndexScan {
            table: "users".to_string(),
            index_name: "idx_id".to_string(),
            column: "id".to_string(),
            condition: "= 1".to_string(),
            filter: None,
        };
        let cloned = plan.clone();
        assert_eq!(plan.to_string(), cloned.to_string());
    }

    #[test]
    fn test_multi_index_scan_empty_indexes() {
        let plan = ScanPlan::MultiIndexScan {
            table: "users".to_string(),
            indexes: vec![],
            operation: "AND".to_string(),
            filter: None,
        };
        assert_eq!(plan.to_string(), "Multi-Index Scan on users (AND)");
    }

    #[test]
    fn test_composite_index_scan_single_column() {
        let plan = ScanPlan::CompositeIndexScan {
            table: "orders".to_string(),
            index_name: "idx_user".to_string(),
            columns: vec!["user_id".to_string()],
            conditions: vec!["= 42".to_string()],
            filter: None,
        };
        assert_eq!(
            plan.to_string(),
            "Composite Index Scan using idx_user on orders\n  Columns: (user_id)\n  user_id = 42"
        );
    }

    #[test]
    fn test_parallel_seq_scan_single_worker() {
        let plan = ScanPlan::ParallelSeqScan {
            table: "users".to_string(),
            filter: None,
            workers: 1,
        };
        assert_eq!(plan.to_string(), "Parallel Seq Scan on users (workers=1)");
    }
}
