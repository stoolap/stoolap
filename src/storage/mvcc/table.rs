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

//! MVCC Table implementation
//!
//! Provides MVCC isolation for table operations.
//!

use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

use crate::common::Int64Set;
use crate::core::{DataType, Error, IndexType, Result, Row, Schema, SchemaColumn, Value};
use crate::storage::expression::Expression;
use crate::storage::mvcc::bitmap_index::BitmapIndex;
use crate::storage::mvcc::btree_index::BTreeIndex;
use crate::storage::mvcc::hash_index::HashIndex;
use crate::storage::mvcc::multi_column_index::MultiColumnIndex;
use crate::storage::mvcc::scanner::{LazyMVCCScanner, MVCCScanner};
use crate::storage::mvcc::{TransactionVersionStore, VersionStore};
use crate::storage::traits::{Index, QueryResult, ScanPlan, Scanner, Table};
use crate::storage::MemoryResult;

/// MVCC Table wrapper that provides MVCC isolation for tables
pub struct MVCCTable {
    /// Transaction ID
    txn_id: i64,
    /// Reference to the version store
    version_store: Arc<VersionStore>,
    /// Transaction-local version store (shared between multiple MVCCTable instances for same txn+table)
    txn_versions: Arc<RwLock<TransactionVersionStore>>,
    /// Cached schema for returning references (Arc clone from version_store - O(1) instead of cloning)
    cached_schema: Arc<Schema>,
}

impl MVCCTable {
    /// Creates a new MVCC table with an owned transaction version store
    /// (wraps it in Arc<RwLock> internally)
    pub fn new(
        txn_id: i64,
        version_store: Arc<VersionStore>,
        txn_versions: TransactionVersionStore,
    ) -> Self {
        // Arc clone - O(1) reference count increment, not full schema clone
        let cached_schema = Arc::clone(&version_store.schema());
        Self {
            txn_id,
            version_store,
            txn_versions: Arc::new(RwLock::new(txn_versions)),
            cached_schema,
        }
    }

    /// Creates a new MVCC table with a shared transaction version store
    /// (used by the engine's get_table_for_transaction to share stores)
    pub fn new_with_shared_store(
        txn_id: i64,
        version_store: Arc<VersionStore>,
        txn_versions: Arc<RwLock<TransactionVersionStore>>,
    ) -> Self {
        // Arc clone - O(1) reference count increment, not full schema clone
        let cached_schema = Arc::clone(&version_store.schema());
        Self {
            txn_id,
            version_store,
            txn_versions,
            cached_schema,
        }
    }

    /// Returns the transaction ID
    pub fn txn_id(&self) -> i64 {
        self.txn_id
    }

    /// Returns a reference to the version store
    pub fn version_store(&self) -> &Arc<VersionStore> {
        &self.version_store
    }

    /// Returns a reference to the shared transaction version store
    pub fn txn_versions(&self) -> &Arc<RwLock<TransactionVersionStore>> {
        &self.txn_versions
    }

    /// Auto-selects the optimal index type based on column data types
    ///
    /// # Type-Based Index Selection Rules:
    /// - TEXT/JSON columns → Hash index (avoids O(strlen) comparisons per B-tree node)
    /// - BOOLEAN columns → Bitmap index (only 2 values, fast AND/OR operations)
    /// - INTEGER/FLOAT/TIMESTAMP columns → BTree index (supports range queries)
    /// - Mixed types → BTree as safe default
    ///
    /// For multi-column indexes, the first column's type determines the index type
    /// unless there's a BOOLEAN (which always gets Bitmap for AND/OR optimization).
    fn auto_select_index_type(data_types: &[DataType]) -> IndexType {
        if data_types.is_empty() {
            return IndexType::BTree;
        }

        // Check if any column is BOOLEAN - use Bitmap for fast AND/OR
        let has_boolean = data_types.iter().any(|dt| matches!(dt, DataType::Boolean));
        if has_boolean && data_types.len() == 1 {
            return IndexType::Bitmap;
        }

        // Check the primary (first) column type
        match data_types[0] {
            // TEXT/JSON - use Hash for O(1) lookups, avoid O(strlen) comparisons
            DataType::Text | DataType::Json => IndexType::Hash,

            // BOOLEAN - use Bitmap for fast AND/OR/NOT operations
            DataType::Boolean => IndexType::Bitmap,

            // Numeric types - use BTree for range query support
            DataType::Integer | DataType::Float | DataType::Timestamp => IndexType::BTree,

            // NULL type - use BTree as safe default
            DataType::Null => IndexType::BTree,
        }
    }

    /// Gets the current auto-increment value
    pub fn get_current_auto_increment_value(&self) -> i64 {
        self.version_store.get_auto_increment_counter()
    }

    /// Normalize a row to match the current schema
    ///
    /// This handles schema evolution (ALTER TABLE ADD/DROP COLUMN):
    /// - If row has fewer columns than schema, append default values (or NULLs) for missing columns
    /// - If row has more columns than schema, truncate the row
    fn normalize_row_to_schema(&self, mut row: Row, schema: &Schema) -> Row {
        let schema_cols = schema.columns.len();
        let row_cols = row.len();

        if row_cols < schema_cols {
            // Row has fewer columns - add default values (or NULLs) for new columns
            for i in row_cols..schema_cols {
                let col = &schema.columns[i];
                // Use pre-computed default value if available, otherwise use NULL
                if let Some(ref default_val) = col.default_value {
                    row.push(default_val.clone());
                } else {
                    row.push(Value::null(col.data_type));
                }
            }
        } else if row_cols > schema_cols {
            // Row has more columns - truncate (columns were dropped)
            row.truncate(schema_cols);
        }

        row
    }

    /// Try to extract a primary key lookup from the expression
    ///
    /// Returns Some(row_id) if the expression is a simple equality on the PK column
    fn try_pk_lookup(&self, expr: &dyn Expression, schema: &Schema) -> Option<i64> {
        use crate::core::Operator;

        // Get PK column info
        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() != 1 {
            return None; // Only support single-column PK for now
        }
        let pk_col_idx = pk_indices[0];
        let pk_col = &schema.columns[pk_col_idx];

        // Use the new get_comparison_info method (no downcasting required)
        let (col_name, operator, value) = expr.get_comparison_info()?;

        // Check if it's an equality on the PK column (case-insensitive comparison)
        if !col_name.eq_ignore_ascii_case(&pk_col.name) || operator != Operator::Eq {
            return None;
        }

        // Get the integer value (PKs are always integers in our system)
        match value {
            Value::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to identify a PK range lookup (WHERE id >= X AND id < Y)
    /// Returns Some(Vec<row_id>) if this is a PK range query
    fn try_pk_range_lookup(&self, expr: &dyn Expression, schema: &Schema) -> Option<Vec<i64>> {
        use crate::core::Operator;

        // Get PK column info
        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() != 1 {
            return None;
        }
        let pk_col_idx = pk_indices[0];
        let pk_col_name = &schema.columns[pk_col_idx].name;

        // Check for AND with two comparisons on PK
        let and_operands = expr.get_and_operands()?;
        if and_operands.len() != 2 {
            return None;
        }

        let (info1, info2) = (
            and_operands[0].get_comparison_info(),
            and_operands[1].get_comparison_info(),
        );

        let ((col1, op1, val1), (col2, op2, val2)) = match (info1, info2) {
            (Some(a), Some(b)) => (a, b),
            _ => return None,
        };

        // Both must be on PK column
        if !col1.eq_ignore_ascii_case(pk_col_name) || !col2.eq_ignore_ascii_case(pk_col_name) {
            return None;
        }

        // Extract integer values
        let (v1, v2) = match (val1, val2) {
            (Value::Integer(a), Value::Integer(b)) => (*a, *b),
            _ => return None,
        };

        // Identify lower and upper bounds
        let (min_id, min_inclusive, max_id, max_inclusive) = match (op1, op2) {
            (Operator::Gte, Operator::Lt) => (v1, true, v2, false),
            (Operator::Gte, Operator::Lte) => (v1, true, v2, true),
            (Operator::Gt, Operator::Lt) => (v1, false, v2, false),
            (Operator::Gt, Operator::Lte) => (v1, false, v2, true),
            (Operator::Lt, Operator::Gte) => (v2, true, v1, false),
            (Operator::Lt, Operator::Gt) => (v2, false, v1, false),
            (Operator::Lte, Operator::Gte) => (v2, true, v1, true),
            (Operator::Lte, Operator::Gt) => (v2, false, v1, true),
            _ => return None,
        };

        // Adjust for inclusive/exclusive bounds using saturating arithmetic to prevent overflow
        let start = if min_inclusive {
            min_id
        } else {
            min_id.saturating_add(1)
        };
        let end = if max_inclusive {
            max_id
        } else {
            max_id.saturating_sub(1)
        };

        // Check for invalid range (start > end means empty result)
        if start > end {
            return Some(Vec::new());
        }

        // SAFETY: Limit range size to prevent memory explosion
        // For ranges larger than threshold, return None to fall back to index/full scan
        const MAX_PK_RANGE_SIZE: i64 = 100_000;
        let range_size = end.saturating_sub(start).saturating_add(1);
        if range_size > MAX_PK_RANGE_SIZE {
            return None; // Fall back to index scan or full scan
        }

        // Generate row_ids directly (no index needed - PK IS the row_id)
        Some((start..=end).collect())
    }

    /// Try to use an index to filter row IDs
    ///
    /// Returns Some(row_ids) if an index can be used, None otherwise
    #[allow(clippy::only_used_in_recursion)]
    fn try_index_lookup(&self, expr: &dyn Expression, schema: &Schema) -> Option<Vec<i64>> {
        use crate::core::Operator;
        use crate::storage::mvcc::{intersect_sorted_ids, union_sorted_ids};

        // First, try simple comparison on a single column
        if let Some((col_name, operator, value)) = expr.get_comparison_info() {
            // Skip index for boolean equality - low cardinality (2 values) means ~50% selectivity
            // which makes full scan faster than index lookup + row fetch
            if matches!(value, Value::Boolean(_)) && matches!(operator, Operator::Eq | Operator::Ne)
            {
                return None;
            }

            if let Some(index) = self.version_store.get_index_by_column(col_name) {
                return self.query_index_with_operator(&*index, operator, value);
            }
        }

        // OPTIMIZATION: Handle OR expressions with HYBRID index optimization
        // For (indexed_col = 'a' OR non_indexed_col = 'b'):
        // - Use index for indexed_col operands
        // - Return None only if ALL operands are non-indexed (full scan needed)
        // - If at least one operand uses index but others don't, we still return
        //   the indexed row_ids (the executor will handle memory filtering for others)
        if let Some(or_operands) = expr.get_or_operands() {
            let mut indexed_row_ids: Vec<Vec<i64>> = Vec::with_capacity(or_operands.len());
            let mut has_unindexed_operand = false;
            let mut all_operands_indexed = true;

            for operand in or_operands {
                // Recursively try index lookup for each OR operand
                // Row IDs are already sorted by index (sorted insertion)
                if let Some(row_ids) = self.try_index_lookup(operand.as_ref(), schema) {
                    indexed_row_ids.push(row_ids);
                } else {
                    // This operand can't use an index
                    has_unindexed_operand = true;
                    all_operands_indexed = false;
                }
            }

            // If ALL operands can use indexes, return the union
            if all_operands_indexed && !indexed_row_ids.is_empty() {
                if indexed_row_ids.len() == 1 {
                    return Some(indexed_row_ids.into_iter().next().unwrap());
                }

                let mut result = indexed_row_ids[0].clone();
                for other in &indexed_row_ids[1..] {
                    result = union_sorted_ids(&result, other);
                }
                return Some(result);
            }

            // HYBRID OPTIMIZATION: If some operands use indexes but not all,
            // we can't use pure index lookup (would miss rows from unindexed operands).
            // However, if there are many indexed operands and few unindexed ones,
            // the executor can still benefit from partial index usage.
            // For now, fall back to full scan - the memory filter will handle it.
            // Future: Could return indexed row_ids + flag for partial optimization
            if has_unindexed_operand {
                return None;
            }

            // All indexed - union results
            if indexed_row_ids.is_empty() {
                return None;
            }

            let mut result = indexed_row_ids[0].clone();
            for other in &indexed_row_ids[1..] {
                result = union_sorted_ids(&result, other);
            }
            return Some(result);
        }

        // OPTIMIZATION: Detect BETWEEN/range pattern (col >= X AND col <= Y) for single range scan
        // This avoids two separate index scans + sort + intersection
        if let Some(and_operands) = expr.get_and_operands() {
            // Check for range pattern: exactly 2 comparisons on the same indexed column
            if and_operands.len() == 2 {
                if let (Some((col1, op1, val1)), Some((col2, op2, val2))) = (
                    and_operands[0].get_comparison_info(),
                    and_operands[1].get_comparison_info(),
                ) {
                    // Same column?
                    if col1 == col2 {
                        // Check for range pattern: one lower bound + one upper bound
                        let (lower_op, lower_val, upper_op, upper_val) = match (op1, op2) {
                            (Operator::Gt | Operator::Gte, Operator::Lt | Operator::Lte) => {
                                (op1, val1, op2, val2)
                            }
                            (Operator::Lt | Operator::Lte, Operator::Gt | Operator::Gte) => {
                                (op2, val2, op1, val1)
                            }
                            _ => (Operator::Eq, val1, Operator::Eq, val2), // Not a range
                        };

                        // If we have a valid range pattern, use single range scan
                        if matches!(lower_op, Operator::Gt | Operator::Gte)
                            && matches!(upper_op, Operator::Lt | Operator::Lte)
                        {
                            if let Some(index) = self.version_store.get_index_by_column(col1) {
                                // Only B-tree indexes support range queries
                                // Hash and Bitmap indexes return empty for range queries,
                                // which would incorrectly indicate "no matches" instead of
                                // "can't handle this query"
                                if index.index_type() != IndexType::BTree {
                                    return None; // Fall back to full scan
                                }
                                let min_inclusive = matches!(lower_op, Operator::Gte);
                                let max_inclusive = matches!(upper_op, Operator::Lte);
                                let row_ids = index.get_row_ids_in_range(
                                    std::slice::from_ref(lower_val),
                                    std::slice::from_ref(upper_val),
                                    min_inclusive,
                                    max_inclusive,
                                );
                                // Return result even if empty - we successfully used the index
                                // Empty result is valid (no rows match the range)
                                return Some(row_ids);
                            }
                        }
                    }
                }
            }
        }

        // OPTIMIZATION: Handle AND expressions with recursive index lookup
        // For '(col1 IN (...)) AND (col2 IN (...))', process each operand and intersect results
        // This is critical for semi-join optimization where we combine right_filter with IN clause
        if let Some(and_operands) = expr.get_and_operands() {
            let mut indexed_row_ids: Vec<Vec<i64>> = Vec::with_capacity(and_operands.len());

            for operand in and_operands {
                // Recursively try index lookup for each AND operand
                // OPTIMIZATION: Don't sort here - defer sorting until we know intersection is needed
                if let Some(row_ids) = self.try_index_lookup(operand.as_ref(), schema) {
                    indexed_row_ids.push(row_ids);
                }
                // If an operand can't use an index, we'll still proceed with available indexes
                // and the executor will handle memory filtering for the rest
            }

            // If at least one operand used an index, intersect and return the most restrictive result
            // The executor will handle memory filtering for operands without index support
            if !indexed_row_ids.is_empty() {
                if indexed_row_ids.len() == 1 {
                    // OPTIMIZATION: Single operand - no intersection needed
                    return Some(indexed_row_ids.into_iter().next().unwrap());
                }

                // Row IDs are already sorted by index (sorted insertion)
                // Just sort by length for optimal intersection order
                indexed_row_ids.sort_by_key(|v| v.len());
                let mut result = indexed_row_ids.swap_remove(0); // Take ownership, not clone
                for other in &indexed_row_ids {
                    result = intersect_sorted_ids(&result, other);
                    // Early termination if intersection is empty
                    if result.is_empty() {
                        return Some(Vec::new());
                    }
                }
                return Some(result);
            }

            // No operands could use indexes - fall through to other strategies
            // Note: We don't return None here because the subsequent code might
            // still be able to extract simple comparisons from the AND expression
        }

        // OPTIMIZATION: Handle IN list expressions with direct index lookup
        // For 'col IN (a, b, c)', use get_row_ids_in for efficient multi-value lookup
        if let Some(in_list) = expr
            .as_any()
            .downcast_ref::<crate::storage::expression::InListExpr>()
        {
            // Only handle positive IN (not NOT IN)
            if !in_list.is_not() {
                if let Some(col_name) = in_list.get_column_name() {
                    if let Some(index) = self.version_store.get_index_by_column(col_name) {
                        // Use the efficient get_row_ids_in method
                        let values = in_list.get_values();
                        let row_ids = index.get_row_ids_in(values);
                        return Some(row_ids);
                    }
                }
            }
        }

        // OPTIMIZATION: Handle LIKE prefix patterns with index range scan
        // For 'name LIKE 'John%'', use index range scan from 'John' to 'John\xff'
        if let Some((col_name, prefix, negated)) = expr.get_like_prefix_info() {
            // Don't optimize NOT LIKE (would need complement of range)
            if negated {
                return None;
            }

            if let Some(index) = self.version_store.get_index_by_column(col_name) {
                // Create range from prefix to prefix + '\xff' (highest byte)
                // This captures all strings starting with the prefix
                let min_value = Value::text(&prefix);
                let mut max_prefix = prefix.clone();
                max_prefix.push('\u{FFFF}'); // Highest unicode char
                let max_value = Value::text(&max_prefix);

                // Use index range query
                if let Ok(entries) = index.find_range(
                    &[min_value],
                    &[max_value],
                    true,  // include min
                    false, // exclude max
                ) {
                    let row_ids: Vec<i64> = entries.into_iter().map(|e| e.row_id).collect();
                    return Some(row_ids);
                }
            }
        }

        // Try to extract comparisons from AND expressions
        let comparisons = expr.collect_comparisons();
        if comparisons.is_empty() {
            return None;
        }

        // Group comparisons by column name
        let mut column_comparisons: FxHashMap<&str, Vec<(Operator, &Value)>> = FxHashMap::default();
        for (col_name, op, val) in &comparisons {
            column_comparisons
                .entry(*col_name)
                .or_default()
                .push((*op, *val));
        }

        // OPTIMIZATION: Try multi-column index first
        // Collect columns that have equality predicates (can be used with multi-column index)
        let eq_columns: Vec<&str> = column_comparisons
            .iter()
            .filter_map(|(col_name, ops)| {
                // Check if this column has an equality predicate
                if ops.iter().any(|(op, _)| matches!(op, Operator::Eq)) {
                    Some(*col_name)
                } else {
                    None
                }
            })
            .collect();

        // Try to find a multi-column index that covers these equality columns
        if eq_columns.len() >= 2 {
            if let Some((multi_idx, matched_count)) =
                self.version_store.get_multi_column_index(&eq_columns)
            {
                // Build the values array in index column order
                let index_columns = multi_idx.column_names();
                let mut values: Vec<Value> = Vec::with_capacity(matched_count);
                let mut all_columns_matched = true;

                for idx_col in index_columns.iter().take(matched_count) {
                    if let Some(ops) = column_comparisons.get(idx_col.as_str()) {
                        // Find the equality value for this column
                        if let Some((_, val)) =
                            ops.iter().find(|(op, _)| matches!(op, Operator::Eq))
                        {
                            values.push((*val).clone());
                        } else {
                            all_columns_matched = false;
                            break;
                        }
                    } else {
                        all_columns_matched = false;
                        break;
                    }
                }

                // If we matched all columns in the prefix, use the multi-column index
                if all_columns_matched && values.len() == matched_count {
                    let row_ids = multi_idx.get_row_ids_equal(&values);
                    if !row_ids.is_empty() {
                        // Multi-column index gave us results - check if we need to apply
                        // additional filters for columns not covered by the index
                        let covered_columns: std::collections::HashSet<&str> = index_columns
                            .iter()
                            .take(matched_count)
                            .map(|s| s.as_str())
                            .collect();

                        // Check if there are non-covered columns with predicates
                        let uncovered_columns: Vec<&str> = column_comparisons
                            .keys()
                            .filter(|c| !covered_columns.contains(*c))
                            .copied()
                            .collect();

                        if uncovered_columns.is_empty() {
                            // All predicate columns are covered by multi-column index
                            return Some(row_ids);
                        }
                        // There are uncovered columns - continue to check single-column indexes
                        // and intersect with multi-column index results
                        let mut all_row_ids = vec![row_ids];

                        // Check single-column indexes for uncovered columns
                        for col_name in uncovered_columns {
                            if let Some(single_idx) =
                                self.version_store.get_index_by_column(col_name)
                            {
                                if let Some(ops) = column_comparisons.get(col_name) {
                                    // Try equality first
                                    if let Some((_, val)) =
                                        ops.iter().find(|(op, _)| matches!(op, Operator::Eq))
                                    {
                                        let ids =
                                            single_idx.get_row_ids_equal(std::slice::from_ref(val));
                                        if ids.is_empty() {
                                            return Some(Vec::new());
                                        }
                                        // Don't sort yet - only sort when intersection is needed
                                        all_row_ids.push(ids);
                                    }
                                }
                            }
                        }

                        // Intersect all results
                        if all_row_ids.len() == 1 {
                            return Some(all_row_ids.into_iter().next().unwrap());
                        }
                        // Row IDs are already sorted by index (sorted insertion)
                        let mut result = all_row_ids[0].clone();
                        for other in &all_row_ids[1..] {
                            result = intersect_sorted_ids(&result, other);
                            if result.is_empty() {
                                return Some(Vec::new());
                            }
                        }
                        return Some(result);
                    }
                }
            }
        }

        // Fall back to single-column index strategy
        // Collect row IDs from all indexed columns
        let mut all_row_ids: Vec<Vec<i64>> = Vec::new();

        for (col_name, ops) in &column_comparisons {
            if let Some(index) = self.version_store.get_index_by_column(col_name) {
                // Check for range pattern: col >= min AND col <= max
                let mut min_val: Option<(&Value, bool)> = None; // (value, inclusive)
                let mut max_val: Option<(&Value, bool)> = None;
                let mut eq_val: Option<&Value> = None;

                for (op, val) in ops {
                    match op {
                        Operator::Eq => eq_val = Some(val),
                        Operator::Gt => min_val = Some((val, false)),
                        Operator::Gte => min_val = Some((val, true)),
                        Operator::Lt => max_val = Some((val, false)),
                        Operator::Lte => max_val = Some((val, true)),
                        _ => {}
                    }
                }

                // Equality takes precedence - but skip boolean (low cardinality)
                if let Some(val) = eq_val {
                    // Skip boolean equality - ~50% selectivity makes full scan faster
                    if matches!(val, Value::Boolean(_)) {
                        continue;
                    }
                    // OPTIMIZATION: Use from_ref to avoid clone
                    let row_ids = index.get_row_ids_equal(std::slice::from_ref(val));
                    if row_ids.is_empty() {
                        // If any index returns empty, the AND result is empty
                        return Some(Vec::new());
                    }
                    // Don't sort yet - only sort when intersection is needed
                    all_row_ids.push(row_ids);
                    continue;
                }

                // Range query - but skip Hash indexes (they don't support range queries)
                if min_val.is_some() || max_val.is_some() {
                    // Hash indexes don't support range queries - skip them
                    // and let the query fall back to a full scan
                    if matches!(index.index_type(), IndexType::Hash) {
                        continue;
                    }

                    let row_ids =
                        if let (Some((min, min_inc)), Some((max, max_inc))) = (min_val, max_val) {
                            // OPTIMIZATION: Use from_ref to avoid clone
                            index.get_row_ids_in_range(
                                std::slice::from_ref(min),
                                std::slice::from_ref(max),
                                min_inc,
                                max_inc,
                            )
                        } else if let Some((val, inclusive)) = min_val {
                            let op = if inclusive {
                                Operator::Gte
                            } else {
                                Operator::Gt
                            };
                            self.query_index_with_operator(&*index, op, val)
                                .unwrap_or_default()
                        } else if let Some((val, inclusive)) = max_val {
                            let op = if inclusive {
                                Operator::Lte
                            } else {
                                Operator::Lt
                            };
                            self.query_index_with_operator(&*index, op, val)
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        };

                    if row_ids.is_empty() {
                        // If any index returns empty, the AND result is empty
                        return Some(Vec::new());
                    }
                    // Don't sort yet - only sort when intersection is needed
                    all_row_ids.push(row_ids);
                }
            }
        }

        // If we have no indexed results, return None
        if all_row_ids.is_empty() {
            return None;
        }

        // If we have only one index, return its results
        if all_row_ids.len() == 1 {
            return Some(all_row_ids.into_iter().next().unwrap());
        }

        // Row IDs are already sorted by index (sorted insertion)
        // Intersect all row ID sets for multi-column filtering
        // This is the key optimization - we filter rows using multiple indexes
        let mut result = all_row_ids[0].clone();
        for other in &all_row_ids[1..] {
            result = intersect_sorted_ids(&result, other);
            if result.is_empty() {
                return Some(Vec::new());
            }
        }

        Some(result)
    }

    /// Query an index with a specific operator
    fn query_index_with_operator(
        &self,
        index: &dyn crate::storage::traits::Index,
        operator: crate::core::Operator,
        value: &Value,
    ) -> Option<Vec<i64>> {
        use crate::core::Operator;

        match operator {
            Operator::Eq => {
                let row_ids = index.get_row_ids_equal(std::slice::from_ref(value));
                if row_ids.is_empty() {
                    None
                } else {
                    Some(row_ids)
                }
            }
            Operator::Gt | Operator::Gte | Operator::Lt | Operator::Lte => {
                // Use find_with_operator for range queries
                let entries = index
                    .find_with_operator(operator, std::slice::from_ref(value))
                    .ok()?;
                let row_ids: Vec<i64> = entries.into_iter().map(|e| e.row_id).collect();
                if row_ids.is_empty() {
                    None
                } else {
                    Some(row_ids)
                }
            }
            _ => None,
        }
    }

    /// Fast path for index-based row fetching when there are no local transaction changes.
    /// Returns Some(rows) if we can serve from index, None if we should fall through to full path.
    #[inline]
    fn try_fetch_from_index(
        &self,
        row_ids: Vec<i64>,
        expr: &dyn Expression,
        schema: &Schema,
        limit: usize,
        offset: usize,
    ) -> Option<Vec<Row>> {
        let txn_versions = self.txn_versions.read().unwrap();
        if txn_versions.has_local_changes() {
            return None; // Fall through to full path
        }

        // No local changes - fetch directly from version store by row IDs
        let mut result = Vec::with_capacity(limit.min(row_ids.len()));
        let mut skipped = 0usize;

        for row_id in row_ids {
            if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                if !version.is_deleted() {
                    // Re-apply filter (index may return superset for complex expressions)
                    let row = self.normalize_row_to_schema(version.data.clone(), schema);
                    if expr.evaluate_fast(&row) {
                        if skipped < offset {
                            skipped += 1;
                        } else {
                            result.push(row);
                            if result.len() >= limit {
                                return Some(result);
                            }
                        }
                    }
                }
            }
        }
        Some(result)
    }

    /// Validates and coerces a row against the schema
    /// Returns the coerced row if successful
    fn validate_and_coerce_row(&self, row: &mut Row) -> Result<()> {
        // OPTIMIZATION: Use cached_schema instead of version_store.schema() to avoid clone
        let schema = &self.cached_schema;

        // Check column count
        if row.len() != schema.columns.len() {
            return Err(Error::internal(format!(
                "invalid column count: expected {}, got {}",
                schema.columns.len(),
                row.len()
            )));
        }

        // Validate and coerce each column
        for (i, col) in schema.columns.iter().enumerate() {
            let value = row.get(i).ok_or_else(|| {
                Error::internal(format!("nil value at index {} (column '{}')", i, col.name))
            })?;

            // Check NULL constraint
            if !col.nullable && value.is_null() {
                return Err(Error::internal(format!(
                    "NULL value in non-nullable column '{}'",
                    col.name
                )));
            }

            // Check type compatibility for non-NULL values
            if !value.is_null() {
                let actual_type = value.data_type();
                if actual_type != col.data_type {
                    // Allow Text to Json coercion (JSON strings come in as Text)
                    if actual_type == DataType::Text && col.data_type == DataType::Json {
                        // Coerce Text to Json
                        if let Some(text_val) = value.as_arc_str() {
                            let _ = row.set(i, Value::Json(text_val));
                        }
                    } else {
                        return Err(Error::internal(format!(
                            "type mismatch in column '{}': expected {:?}, got {:?}",
                            col.name, col.data_type, actual_type
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Extracts the primary key value from a row
    /// OPTIMIZATION: Uses cached_schema and pk_column_index to avoid iteration
    fn extract_row_pk(&self, row: &Row) -> i64 {
        // Fast path: use cached PK index if available
        if let Some(pk_idx) = self.cached_schema.pk_column_index() {
            if let Some(value) = row.get(pk_idx) {
                if let Some(pk) = value.as_int64() {
                    return pk;
                }
            }
        }

        // Fallback: If no primary key or not an integer, generate a synthetic row ID
        self.version_store.get_next_auto_increment_id()
    }

    /// Finds the primary key column index
    /// OPTIMIZATION: Uses cached pk_column_index from schema
    #[inline]
    fn find_pk_column_index(&self) -> Option<usize> {
        self.cached_schema.pk_column_index()
    }

    /// Check unique index constraints for a row being inserted
    /// OPTIMIZATION: Uses cached_schema and iterates indexes directly without collecting names
    fn check_unique_constraints(&self, row: &Row, _row_id: i64) -> Result<()> {
        // OPTIMIZATION: Use cached_schema instead of cloning
        let schema = &self.cached_schema;

        // OPTIMIZATION: Iterate indexes directly without collecting names
        // Use iter_unique_indexes to only iterate unique indexes
        self.version_store
            .for_each_unique_index(|index_name, index| {
                // Get ALL columns this index is on
                let column_ids = index.column_ids();
                if column_ids.is_empty() {
                    return Ok(());
                }

                // Collect values for ALL columns in the index
                let values: Vec<Value> = column_ids
                    .iter()
                    .filter_map(|&col_id| row.get(col_id as usize).cloned())
                    .collect();

                // If we didn't get all values, skip this check
                if values.len() != column_ids.len() {
                    return Ok(());
                }

                // NULL values are allowed in unique indexes (multiple NULLs are distinct)
                // For multi-column unique indexes, if ANY column is NULL, it's allowed
                if values.iter().any(|v| v.is_null()) {
                    return Ok(());
                }

                // Check if this value combination already exists in the index
                let entries = index.find(&values)?;
                if !entries.is_empty() {
                    // Value already exists - constraint violation
                    let col_names: Vec<&str> = column_ids
                        .iter()
                        .map(|&col_id| {
                            schema
                                .columns
                                .get(col_id as usize)
                                .map(|c| c.name.as_str())
                                .unwrap_or("unknown")
                        })
                        .collect();
                    return Err(Error::unique_constraint(
                        index_name,
                        col_names.join(", "),
                        format!("{:?}", values),
                    ));
                }
                Ok(())
            })
    }

    /// Prepares a row for insertion: handles auto-increment, validates, and checks constraints.
    /// Returns the row_id on success. The row is modified in-place with auto-increment values.
    #[inline]
    fn prepare_insert(&mut self, row: &mut Row) -> Result<i64> {
        // Handle auto-increment for primary key BEFORE validation
        // This allows inserting without specifying the primary key (only if AUTO_INCREMENT)
        if let Some(pk_idx) = self.find_pk_column_index() {
            let pk_col = &self.cached_schema.columns[pk_idx];
            let is_auto_increment = pk_col.auto_increment;

            if let Some(value) = row.get(pk_idx) {
                if value.is_null() {
                    if is_auto_increment {
                        // Generate new ID for NULL primary key (only with AUTO_INCREMENT)
                        let next_id = self.version_store.get_next_auto_increment_id();
                        let _ = row.set(pk_idx, Value::Integer(next_id));
                    } else {
                        // PRIMARY KEY without AUTO_INCREMENT cannot be NULL
                        return Err(Error::internal(format!(
                            "NULL value not allowed for PRIMARY KEY column '{}'. Use AUTO_INCREMENT for auto-generated IDs.",
                            pk_col.name
                        )));
                    }
                } else if let Some(pk_val) = value.as_int64() {
                    // Track max row_id for ORDER BY contiguous iteration optimization.
                    // Uses the auto-increment counter as a convenient max-row-id tracker.
                    let current = self.version_store.get_auto_increment_counter();
                    if pk_val > current {
                        self.version_store.set_auto_increment_counter(pk_val);
                    }
                }
            }
        }

        // Validate and coerce row AFTER auto-increment has filled in the primary key
        self.validate_and_coerce_row(row)?;

        // Extract row ID
        let row_id = self.extract_row_pk(row);

        // Check if row already exists in local versions
        {
            let txn_versions = self.txn_versions.read().unwrap();
            if txn_versions.has_locally_seen(row_id) && txn_versions.get(row_id).is_some() {
                return Err(Error::primary_key_constraint(row_id));
            }
        }

        // Check if row exists in global store
        if self.version_store.quick_check_row_existence(row_id) {
            if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                if !version.is_deleted() {
                    return Err(Error::primary_key_constraint(row_id));
                }
            }
        }

        // Check unique index constraints
        self.check_unique_constraints(row, row_id)?;

        Ok(row_id)
    }

    /// Commits the transaction's local changes
    ///
    /// This method updates indexes before committing versions to the global store.
    pub fn commit(&mut self) -> Result<()> {
        // Update indexes using already-cached old versions (no extra lookups needed)
        let index_names = self.version_store.list_indexes();

        if !index_names.is_empty() {
            let txn_versions = self.txn_versions.read().unwrap();
            for (row_id, new_version, old_row) in txn_versions.iter_local_with_old() {
                let is_deleted = new_version.is_deleted();
                let new_row = &new_version.data;

                for index_name in &index_names {
                    if let Some(index) = self.version_store.get_index(index_name) {
                        let column_ids = index.column_ids();
                        if column_ids.is_empty() {
                            continue;
                        }

                        // OPTIMIZATION: For UPDATEs, check if any indexed column changed
                        // BEFORE allocating Vecs. This avoids allocation overhead when
                        // updating columns that aren't indexed.
                        if let Some(old_r) = old_row {
                            if !is_deleted {
                                // UPDATE case: check if indexed columns differ (no allocation)
                                let any_changed = column_ids.iter().any(|&col_id| {
                                    let idx = col_id as usize;
                                    let old_val = old_r.get(idx);
                                    let new_val = new_row.get(idx);
                                    old_val != new_val
                                });

                                if !any_changed {
                                    // Indexed columns unchanged - skip this index entirely
                                    continue;
                                }

                                // Values differ - now collect and update
                                let old_values: Vec<Value> = column_ids
                                    .iter()
                                    .map(|&col_id| {
                                        old_r
                                            .get(col_id as usize)
                                            .cloned()
                                            .unwrap_or(Value::Null(DataType::Null))
                                    })
                                    .collect();

                                let new_values: Vec<Value> = column_ids
                                    .iter()
                                    .map(|&col_id| {
                                        new_row
                                            .get(col_id as usize)
                                            .cloned()
                                            .unwrap_or(Value::Null(DataType::Null))
                                    })
                                    .collect();

                                let _ = index.remove(&old_values, row_id, row_id);
                                index.add(&new_values, row_id, row_id)?;
                                continue;
                            }
                        }

                        // INSERT or DELETE case: collect values (allocation needed)
                        let new_values: Vec<Value> = column_ids
                            .iter()
                            .map(|&col_id| {
                                new_row
                                    .get(col_id as usize)
                                    .cloned()
                                    .unwrap_or(Value::Null(DataType::Null))
                            })
                            .collect();

                        if is_deleted {
                            // DELETE: remove from index using old values if available
                            if let Some(old_r) = old_row {
                                let old_values: Vec<Value> = column_ids
                                    .iter()
                                    .map(|&col_id| {
                                        old_r
                                            .get(col_id as usize)
                                            .cloned()
                                            .unwrap_or(Value::Null(DataType::Null))
                                    })
                                    .collect();
                                let _ = index.remove(&old_values, row_id, row_id);
                            } else {
                                let _ = index.remove(&new_values, row_id, row_id);
                            }
                        } else {
                            // INSERT: just add new values
                            index.add(&new_values, row_id, row_id)?;
                        }
                    }
                }
            }
        }

        // Check if there are local changes before committing
        let has_changes = {
            let txn_versions = self.txn_versions.read().unwrap();
            txn_versions.has_local_changes()
        };

        // Now commit the versions to the version store
        self.txn_versions.write().unwrap().commit()?;

        // Mark zone maps as stale if we had any data changes
        // This ensures the optimizer won't use outdated pruning info
        if has_changes {
            self.version_store.mark_zone_maps_stale();
        }

        Ok(())
    }

    /// Rolls back the transaction's local changes
    pub fn rollback(&mut self) {
        self.txn_versions.write().unwrap().rollback();
    }

    /// Returns the row count visible to this transaction
    ///
    /// OPTIMIZATION: Uses single-pass counting instead of per-row visibility checks.
    /// Reduces lock acquisitions from O(N) to O(1) for the global store.
    pub fn row_count(&self) -> usize {
        // Count global visible versions in single pass (O(1) lock instead of O(N))
        let mut count = self.version_store.count_visible_rows(self.txn_id);

        // Adjust for local changes (uncommitted in this transaction)
        let txn_versions = self.txn_versions.read().unwrap();
        for (row_id, version) in txn_versions.iter_local() {
            // Check if this row exists in global store
            let exists_in_global = self.version_store.quick_check_row_existence(row_id);

            if version.is_deleted() {
                // If deleted locally and existed in global, subtract from count
                if exists_in_global {
                    count = count.saturating_sub(1);
                }
            } else {
                // If inserted locally and not in global, add to count
                if !exists_in_global {
                    count += 1;
                }
            }
        }

        count
    }

    /// Collect all visible rows, optionally filtered
    ///
    /// Optimized to use batch fetch even when there are local changes,
    /// then merge the results.
    #[inline]
    fn collect_visible_rows(&self, filter: Option<&dyn Expression>) -> Vec<(i64, Row)> {
        let txn_versions = self.txn_versions.read().unwrap();
        let schema = &self.cached_schema;

        // Check if we have local versions (uncommitted changes in this transaction)
        let has_local = txn_versions.has_local_changes();

        if !has_local {
            // No local versions - use arena-based batch fetch for maximum performance
            // Arena storage provides 50x+ faster scans via contiguous memory access
            let raw_rows = if let Some(expr) = filter {
                // Use filtered version to avoid allocating memory for non-matching rows
                self.version_store
                    .get_all_visible_rows_filtered(self.txn_id, expr)
            } else {
                self.version_store.get_all_visible_rows_arena(self.txn_id)
            };

            // OPTIMIZATION: Skip normalization if first row matches schema column count
            // This is the common case when no ALTER TABLE ADD/DROP COLUMN has occurred
            let schema_cols = schema.columns.len();
            if raw_rows
                .first()
                .is_none_or(|(_, row)| row.len() == schema_cols)
            {
                // Fast path: all rows should have same column count, skip normalization
                return raw_rows;
            }

            // Slow path: need to normalize rows for schema evolution
            return raw_rows
                .into_iter()
                .map(|(row_id, row)| (row_id, self.normalize_row_to_schema(row, schema)))
                .collect();
        }

        // Has local versions - use batch fetch then merge
        // Step 1: Get all global rows in one batch (single lock acquisition)
        let global_rows = self.version_store.get_all_visible_rows_arena(self.txn_id);

        // Step 2: Build set of local row IDs for quick lookup (Int64Set for fast i64 lookups)
        let local_row_ids: Int64Set = txn_versions
            .iter_local()
            .map(|(row_id, _)| row_id)
            .collect();

        // Step 3: Pre-allocate result
        let mut rows = Vec::with_capacity(global_rows.len() + local_row_ids.len());

        // Step 4: Add global rows that don't have local overrides
        for (row_id, row) in global_rows {
            if local_row_ids.contains(&row_id) {
                continue; // Local version takes precedence
            }
            // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
            let row = self.normalize_row_to_schema(row, schema);
            if let Some(expr) = filter {
                if !expr.evaluate_fast(&row) {
                    continue;
                }
            }
            rows.push((row_id, row));
        }

        // Step 5: Add local versions (both updates and inserts)
        for (row_id, version) in txn_versions.iter_local() {
            if version.is_deleted() {
                continue;
            }
            // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
            let row = self.normalize_row_to_schema(version.data.clone(), schema);
            if let Some(expr) = filter {
                if !expr.evaluate_fast(&row) {
                    continue;
                }
            }
            rows.push((row_id, row));
        }

        rows
    }

    /// Collect visible rows WITHOUT sorting for GROUP BY optimization.
    ///
    /// This skips the O(n log n) sort since GROUP BY doesn't care about row order.
    /// Returns rows in DashMap iteration order (non-deterministic).
    #[inline]
    fn collect_visible_rows_unsorted(&self) -> Vec<Row> {
        let txn_versions = self.txn_versions.read().unwrap();
        let schema = &self.cached_schema;

        // Check if we have local versions (uncommitted changes in this transaction)
        let has_local = txn_versions.has_local_changes();

        if !has_local {
            // No local versions - use unsorted arena-based fetch
            let raw_rows = self
                .version_store
                .get_all_visible_rows_unsorted(self.txn_id);
            // Normalize rows and discard row_ids (not needed for GROUP BY)
            return raw_rows
                .into_iter()
                .map(|(_, row)| self.normalize_row_to_schema(row, schema))
                .collect();
        }

        // Has local versions - merge (order doesn't matter)
        let global_rows = self
            .version_store
            .get_all_visible_rows_unsorted(self.txn_id);

        let local_row_ids: Int64Set = txn_versions
            .iter_local()
            .map(|(row_id, _)| row_id)
            .collect();

        let mut rows = Vec::with_capacity(global_rows.len() + local_row_ids.len());

        for (row_id, row) in global_rows {
            if local_row_ids.contains(&row_id) {
                continue;
            }
            rows.push(self.normalize_row_to_schema(row, schema));
        }

        for (_, version) in txn_versions.iter_local() {
            if version.is_deleted() {
                continue;
            }
            rows.push(self.normalize_row_to_schema(version.data.clone(), schema));
        }

        rows
    }

    /// Collect visible rows with early termination when limit is reached
    /// This is the LIMIT pushdown optimization
    fn collect_visible_rows_with_limit(
        &self,
        filter: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Vec<Row> {
        let schema = &self.cached_schema;

        // FAST PATH: Check if this is a primary key equality lookup (WHERE id = X)
        // This is O(1) and skips all scanning
        if let Some(expr) = filter {
            if let Some(pk_id) = self.try_pk_lookup(expr, schema) {
                // Direct O(1) lookup by primary key
                let txn_versions = self.txn_versions.read().unwrap();
                let row = if let Some(row) = txn_versions.get(pk_id) {
                    Some(row)
                } else if let Some(version) =
                    self.version_store.get_visible_version(pk_id, self.txn_id)
                {
                    if !version.is_deleted() {
                        Some(version.data.clone())
                    } else {
                        None
                    }
                } else {
                    None
                };

                return match row {
                    Some(r) if offset == 0 && limit >= 1 => {
                        vec![self.normalize_row_to_schema(r, schema)]
                    }
                    _ => Vec::new(),
                };
            }

            // OPTIMIZATION: Try secondary index lookup for OR/AND expressions on indexed columns
            // This handles queries like: WHERE age = 25 OR age = 50 OR age = 75
            if let Some(row_ids) = self.try_index_lookup(expr, schema) {
                if let Some(result) =
                    self.try_fetch_from_index(row_ids, expr, schema, limit, offset)
                {
                    return result;
                }
                // Has local changes - fall through to full path
            }
        }

        let txn_versions = self.txn_versions.read().unwrap();

        // Check if we have local versions (uncommitted changes in this transaction)
        let has_local = txn_versions.has_local_changes();

        if !has_local {
            // No local versions - use optimized path with true LIMIT pushdown
            let raw_rows = if let Some(expr) = filter {
                // With filter + limit: use filtered limit pushdown
                // This early-terminates when limit is reached after filtering
                self.version_store.get_visible_rows_filtered_with_limit(
                    self.txn_id,
                    expr,
                    limit,
                    offset,
                )
            } else {
                // No filter: use simple LIMIT pushdown
                // This avoids scanning all 10K rows for LIMIT 10 queries - ~30x speedup
                self.version_store
                    .get_visible_rows_with_limit(self.txn_id, limit, offset)
            };

            return raw_rows
                .into_iter()
                .map(|(_, row)| self.normalize_row_to_schema(row, schema))
                .collect();
        }

        // Has local versions - use batch fetch then merge with early termination
        let global_rows = self.version_store.get_all_visible_rows_arena(self.txn_id);

        // Build set of local row IDs for quick lookup
        let local_row_ids: Int64Set = txn_versions
            .iter_local()
            .map(|(row_id, _)| row_id)
            .collect();

        let mut result = Vec::with_capacity(limit);
        let mut count = 0;

        // Add global rows that don't have local overrides
        for (row_id, row) in global_rows {
            if local_row_ids.contains(&row_id) {
                continue; // Local version takes precedence
            }
            let row = self.normalize_row_to_schema(row, schema);
            if let Some(expr) = filter {
                if !expr.evaluate_fast(&row) {
                    continue;
                }
            }
            if count >= offset {
                result.push(row);
                if result.len() >= limit {
                    return result;
                }
            }
            count += 1;
        }

        // Add local versions (both updates and inserts)
        for (_, version) in txn_versions.iter_local() {
            if version.is_deleted() {
                continue;
            }
            let row = self.normalize_row_to_schema(version.data.clone(), schema);
            if let Some(expr) = filter {
                if !expr.evaluate_fast(&row) {
                    continue;
                }
            }
            if count >= offset {
                result.push(row);
                if result.len() >= limit {
                    return result;
                }
            }
            count += 1;
        }

        result
    }

    /// Collect visible rows with LIMIT without guaranteeing deterministic order.
    /// This is an optimization for queries with LIMIT but without ORDER BY.
    /// Since SQL doesn't guarantee order for LIMIT without ORDER BY, we can
    /// skip sorting and return rows in arbitrary order, enabling true early termination.
    fn collect_visible_rows_with_limit_unordered(
        &self,
        filter: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Vec<Row> {
        let schema = &self.cached_schema;

        // FAST PATH: Check if this is a primary key equality lookup (WHERE id = X)
        // This is O(1) and skips all scanning
        if let Some(expr) = filter {
            if let Some(pk_id) = self.try_pk_lookup(expr, schema) {
                // Direct O(1) lookup by primary key
                let txn_versions = self.txn_versions.read().unwrap();
                let row = if let Some(row) = txn_versions.get(pk_id) {
                    Some(row)
                } else if let Some(version) =
                    self.version_store.get_visible_version(pk_id, self.txn_id)
                {
                    if !version.is_deleted() {
                        Some(version.data.clone())
                    } else {
                        None
                    }
                } else {
                    None
                };

                return match row {
                    Some(r) if offset == 0 && limit >= 1 => {
                        vec![self.normalize_row_to_schema(r, schema)]
                    }
                    _ => Vec::new(),
                };
            }

            // OPTIMIZATION: Try secondary index lookup for OR/AND expressions on indexed columns
            // This handles queries like: WHERE age = 25 OR age = 50 OR age = 75
            if let Some(row_ids) = self.try_index_lookup(expr, schema) {
                if let Some(result) =
                    self.try_fetch_from_index(row_ids, expr, schema, limit, offset)
                {
                    return result;
                }
                // Has local changes - fall through to full path
            }
        }

        let txn_versions = self.txn_versions.read().unwrap();

        // Check if we have local versions (uncommitted changes in this transaction)
        let has_local = txn_versions.has_local_changes();

        if !has_local {
            // No local versions - use optimized unordered path with true early termination
            let raw_rows = if let Some(expr) = filter {
                self.version_store
                    .get_visible_rows_filtered_with_limit_unordered(
                        self.txn_id,
                        expr,
                        limit,
                        offset,
                    )
            } else {
                self.version_store
                    .get_visible_rows_with_limit_unordered(self.txn_id, limit, offset)
            };

            return raw_rows
                .into_iter()
                .map(|(_, row)| self.normalize_row_to_schema(row, schema))
                .collect();
        }

        // Has local versions - use same path as ordered (local changes are rare)
        // Early termination is already implemented in collect_visible_rows_with_limit
        // when there are local changes
        drop(txn_versions);
        self.collect_visible_rows_with_limit(filter, limit, offset)
    }
}

impl Table for MVCCTable {
    fn name(&self) -> &str {
        self.version_store.table_name()
    }

    fn schema(&self) -> &Schema {
        &self.cached_schema
    }

    /// Fetch rows by their IDs, applying filter
    ///
    /// Optimized to use batch fetch for global store rows,
    /// reducing lock contention from O(n) to O(1).
    fn fetch_rows_by_ids(&self, row_ids: &[i64], filter: &dyn Expression) -> Vec<(i64, Row)> {
        let mut rows = Vec::with_capacity(row_ids.len());
        self.fetch_rows_by_ids_into(row_ids, filter, &mut rows);
        rows
    }

    /// Fetch rows into a reusable buffer
    fn fetch_rows_by_ids_into(
        &self,
        row_ids: &[i64],
        filter: &dyn Expression,
        rows: &mut Vec<(i64, Row)>,
    ) {
        let txn_versions = self.txn_versions.read().unwrap();
        let schema = &self.cached_schema;

        // Optimize: Reserve space in buffer if empty
        if rows.is_empty() {
            rows.reserve(row_ids.len());
        }

        let mut global_row_ids = Vec::with_capacity(row_ids.len());

        // Step 1: Check local versions first (uncommitted changes in this transaction)
        for &row_id in row_ids {
            if let Some(version) = txn_versions.get_local_version(row_id) {
                if !version.is_deleted() {
                    // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                    let row = self.normalize_row_to_schema(version.data.clone(), schema);
                    if filter.evaluate_fast(&row) {
                        rows.push((row_id, row));
                    }
                }
                // Skip global lookup - local version takes precedence
            } else {
                // Need to fetch from global store
                global_row_ids.push(row_id);
            }
        }

        // Step 2: Batch fetch from global store (single lock acquisition)
        if !global_row_ids.is_empty() {
            let global_rows = self
                .version_store
                .get_visible_versions_batch(&global_row_ids, self.txn_id);

            // Apply filter to fetched rows
            for (row_id, row) in global_rows {
                // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                let row = self.normalize_row_to_schema(row, schema);
                if filter.evaluate_fast(&row) {
                    rows.push((row_id, row));
                }
            }
        }
    }

    fn create_column(&mut self, name: &str, column_type: DataType, nullable: bool) -> Result<()> {
        self.create_column_with_default(name, column_type, nullable, None)
    }

    fn create_column_with_default(
        &mut self,
        name: &str,
        column_type: DataType,
        nullable: bool,
        default_expr: Option<String>,
    ) -> Result<()> {
        self.create_column_with_default_value(name, column_type, nullable, default_expr, None)
    }

    fn create_column_with_default_value(
        &mut self,
        name: &str,
        column_type: DataType,
        nullable: bool,
        default_expr: Option<String>,
        default_value: Option<Value>,
    ) -> Result<()> {
        // Create a SchemaColumn and add to both version store and cached schema
        // Get the next column ID
        let next_id = self.cached_schema.columns.len();
        let column = SchemaColumn::with_default_value(
            next_id,
            name,
            column_type,
            nullable,
            false, // primary_key
            false, // auto_increment
            default_expr,
            default_value,
            None, // check_expr
        );
        {
            let mut schema_guard = self.version_store.schema_mut();
            Arc::make_mut(&mut *schema_guard).add_column(column.clone())?;
        }
        Arc::make_mut(&mut self.cached_schema).add_column(column)?;
        Ok(())
    }

    fn drop_column(&mut self, name: &str) -> Result<()> {
        // Remove column from both version store and cached schema
        {
            let mut schema_guard = self.version_store.schema_mut();
            Arc::make_mut(&mut *schema_guard).remove_column(name)?;
        }
        Arc::make_mut(&mut self.cached_schema).remove_column(name)?;
        Ok(())
    }

    fn insert(&mut self, mut row: Row) -> Result<Row> {
        let row_id = self.prepare_insert(&mut row)?;

        // Clone the row for returning (with AUTO_INCREMENT value applied)
        let inserted_row = row.clone();

        // Add to transaction's local version store
        self.txn_versions.write().unwrap().put(row_id, row, false)?;

        Ok(inserted_row)
    }

    fn insert_discard(&mut self, mut row: Row) -> Result<()> {
        let row_id = self.prepare_insert(&mut row)?;

        // Add to transaction's local version store - NO CLONE!
        self.txn_versions.write().unwrap().put(row_id, row, false)?;

        Ok(())
    }

    fn insert_batch(&mut self, rows: Vec<Row>) -> Result<()> {
        // Use insert_discard since we don't need returned rows
        for row in rows {
            self.insert_discard(row)?;
        }
        Ok(())
    }

    fn update(
        &mut self,
        where_expr: Option<&dyn Expression>,
        setter: &mut dyn FnMut(Row) -> (Row, bool),
    ) -> Result<i32> {
        // OPTIMIZATION: Borrow schema instead of cloning - saves allocation per update
        let schema = &self.cached_schema;

        // Fast path: Check if this is a primary key equality lookup (WHERE id = X)
        if let Some(expr) = where_expr {
            if let Some(pk_id) = self.try_pk_lookup(expr, schema) {
                // Direct O(1) lookup by primary key
                // OPTIMIZATION: Track original version to avoid redundant lookup in put()
                let row_with_original = {
                    let txn_versions = self.txn_versions.read().unwrap();
                    if let Some(row) = txn_versions.get(pk_id) {
                        // Local version - no need to track original (already in write-set)
                        Some((row, None))
                    } else if let Some(version) =
                        self.version_store.get_visible_version(pk_id, self.txn_id)
                    {
                        if !version.is_deleted() {
                            let data = version.data.clone();
                            Some((data, Some(version)))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                if let Some((row, original_version)) = row_with_original {
                    // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                    let row = self.normalize_row_to_schema(row, schema);
                    let (updated_row, _) = setter(row);
                    if let Some(orig) = original_version {
                        // Use optimized put that skips redundant get_visible_version
                        self.txn_versions.write().unwrap().put_with_original(
                            pk_id,
                            updated_row,
                            orig,
                            false,
                        )?;
                    } else {
                        // Local version - use regular put
                        self.txn_versions
                            .write()
                            .unwrap()
                            .put(pk_id, updated_row, false)?;
                    }
                    return Ok(1);
                }
                return Ok(0);
            }

            // Fast path: PK range lookup (WHERE id >= X AND id < Y)
            if let Some(pk_range_ids) = self.try_pk_range_lookup(expr, schema) {
                // OPTIMIZATION: Track original versions to avoid redundant lookups
                let mut local_rows: Vec<(i64, Row)> = Vec::with_capacity(pk_range_ids.len() / 4);
                let mut rows_with_originals: Vec<(i64, Row, crate::storage::mvcc::RowVersion)> =
                    Vec::with_capacity(pk_range_ids.len());

                // Step 1: Check local versions first (single lock acquisition)
                let mut remaining_row_ids: Vec<i64> = Vec::with_capacity(pk_range_ids.len());
                {
                    let txn_versions = self.txn_versions.read().unwrap();
                    for row_id in pk_range_ids {
                        if let Some(row) = txn_versions.get(row_id) {
                            // Local version
                            let row = self.normalize_row_to_schema(row, schema);
                            let (updated_row, _) = setter(row);
                            local_rows.push((row_id, updated_row));
                        } else {
                            remaining_row_ids.push(row_id);
                        }
                    }
                }

                // Step 2: Batch fetch remaining from version store (1 lock for N rows)
                if !remaining_row_ids.is_empty() {
                    let batch_rows = self
                        .version_store
                        .get_visible_versions_for_update(&remaining_row_ids, self.txn_id);
                    for (row_id, row, version) in batch_rows {
                        let row = self.normalize_row_to_schema(row, schema);
                        let (updated_row, _) = setter(row);
                        rows_with_originals.push((row_id, updated_row, version));
                    }
                }

                let update_count = (local_rows.len() + rows_with_originals.len()) as i32;
                if !local_rows.is_empty() || !rows_with_originals.is_empty() {
                    let mut txn_versions = self.txn_versions.write().unwrap();
                    if !local_rows.is_empty() {
                        txn_versions.put_batch_for_update(local_rows)?;
                    }
                    if !rows_with_originals.is_empty() {
                        txn_versions.put_batch_with_originals(rows_with_originals)?;
                    }
                }
                return Ok(update_count);
            }

            // Try index lookup for non-PK columns
            if let Some(filtered_row_ids) = self.try_index_lookup(expr, schema) {
                // Step 1: Check local versions first (these don't need write-set tracking)
                // OPTIMIZATION: Pre-allocate with estimated capacity
                let mut local_rows_to_update: Vec<(i64, Row)> =
                    Vec::with_capacity(filtered_row_ids.len() / 4);
                let mut remaining_row_ids: Vec<i64> = Vec::with_capacity(filtered_row_ids.len());

                {
                    let txn_versions = self.txn_versions.read().unwrap();
                    for &row_id in &filtered_row_ids {
                        if let Some(row) = txn_versions.get(row_id) {
                            // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                            let row = self.normalize_row_to_schema(row, schema);
                            // Re-apply filter
                            if expr.evaluate(&row).unwrap_or(false) {
                                local_rows_to_update.push((row_id, row));
                            }
                        } else {
                            remaining_row_ids.push(row_id);
                        }
                    }
                }

                // Step 2: Batch fetch remaining rows from version store WITH original versions
                // This avoids redundant get_visible_version() calls during put
                // OPTIMIZATION: Pre-allocate with known capacity
                let mut rows_with_originals: Vec<(i64, Row, crate::storage::mvcc::RowVersion)> =
                    Vec::with_capacity(remaining_row_ids.len());
                if !remaining_row_ids.is_empty() {
                    let batch_rows = self
                        .version_store
                        .get_visible_versions_for_update(&remaining_row_ids, self.txn_id);
                    for (row_id, row, original) in batch_rows {
                        // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                        let row = self.normalize_row_to_schema(row, schema);
                        // Re-apply filter (index may be partial match)
                        if expr.evaluate(&row).unwrap_or(false) {
                            rows_with_originals.push((row_id, row, original));
                        }
                    }
                }

                // Step 3: Apply setter to all rows
                let update_count = local_rows_to_update.len() + rows_with_originals.len();

                // OPTIMIZATION: Apply setter in-place, avoiding intermediate Vec allocation
                // Update local rows (these already have write-set tracking)
                for (_, row) in &mut local_rows_to_update {
                    let (updated_row, _) = setter(std::mem::take(row));
                    *row = updated_row;
                }

                // Update rows from version store with pre-fetched originals
                for (_, row, _) in &mut rows_with_originals {
                    let (updated_row, _) = setter(std::mem::take(row));
                    *row = updated_row;
                }

                // Batch put - first the local rows (use regular put)
                {
                    let mut txn_versions = self.txn_versions.write().unwrap();
                    txn_versions.put_batch_for_update(local_rows_to_update)?;
                    // Then the rows with originals (use optimized put)
                    txn_versions.put_batch_with_originals(rows_with_originals)?;
                }
                return Ok(update_count as i32);
            }
        }

        // Fall back to full scan - use batch fetch WITH original versions for O(1) lock
        // OPTIMIZATION: When WHERE clause exists, push filter to storage layer
        // This avoids allocating Row objects for non-matching rows
        let mut rows_with_originals: Vec<(i64, Row, crate::storage::mvcc::RowVersion)> =
            if let Some(expr) = where_expr {
                // Use filtered scan - filter is applied BEFORE cloning rows
                self.version_store
                    .get_all_visible_rows_for_update_filtered(self.txn_id, expr)
                    .into_iter()
                    .map(|(row_id, row, orig)| {
                        // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                        (row_id, self.normalize_row_to_schema(row, schema), orig)
                    })
                    .collect()
            } else {
                // No filter - get all rows
                self.version_store
                    .get_all_visible_rows_for_update(self.txn_id)
                    .into_iter()
                    .map(|(row_id, row, orig)| {
                        // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                        (row_id, self.normalize_row_to_schema(row, schema), orig)
                    })
                    .collect()
            };

        // Also check local inserts that might not be in global store
        // OPTIMIZATION: Filter on reference BEFORE cloning to avoid wasted allocations
        let local_rows_to_update: Vec<(i64, Row)> = {
            let txn_versions = self.txn_versions.read().unwrap();
            txn_versions
                .iter_local()
                .filter_map(|(row_id, version)| {
                    // Skip if already in global store (already processed above)
                    if self.version_store.quick_check_row_existence(row_id) {
                        return None;
                    }
                    if version.is_deleted() {
                        return None;
                    }
                    // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                    let row = self.normalize_row_to_schema(version.data.clone(), schema);
                    // Apply filter on normalized row
                    if let Some(expr) = where_expr {
                        if !expr.evaluate(&row).unwrap_or(false) {
                            return None;
                        }
                    }
                    Some((row_id, row))
                })
                .collect()
        };

        // Apply setter to rows with originals (from version store)
        for (_, row, _) in &mut rows_with_originals {
            let (updated_row, _) = setter(std::mem::take(row));
            *row = updated_row;
        }

        // Apply setter to local rows
        let local_updated: Vec<(i64, Row)> = local_rows_to_update
            .into_iter()
            .map(|(row_id, row)| {
                let (updated_row, _) = setter(row);
                (row_id, updated_row)
            })
            .collect();

        // Batch update all rows at once
        let update_count = rows_with_originals.len() + local_updated.len();
        {
            let mut txn_versions = self.txn_versions.write().unwrap();
            // Use optimized put for rows from version store (avoids O(N) get_visible_version calls)
            txn_versions.put_batch_with_originals(rows_with_originals)?;
            // Use regular put for local rows (already tracked in local store)
            txn_versions.put_batch_for_update(local_updated)?;
        }

        Ok(update_count as i32)
    }

    fn update_by_row_ids(
        &mut self,
        row_ids: &[i64],
        setter: &mut dyn FnMut(Row) -> (Row, bool),
    ) -> Result<i32> {
        let schema = &self.cached_schema;

        // Step 1: Check local versions first (single lock acquisition)
        let mut local_rows: Vec<(i64, Row)> = Vec::with_capacity(row_ids.len() / 4);
        let mut remaining_row_ids: Vec<i64> = Vec::with_capacity(row_ids.len());

        {
            let txn_versions = self.txn_versions.read().unwrap();
            for &row_id in row_ids {
                if let Some(row) = txn_versions.get(row_id) {
                    // Local version - normalize and apply setter
                    let row = self.normalize_row_to_schema(row, schema);
                    let (updated_row, _) = setter(row);
                    local_rows.push((row_id, updated_row));
                } else {
                    remaining_row_ids.push(row_id);
                }
            }
        }

        // Step 2: Batch fetch remaining from version store with original versions
        let mut rows_with_originals: Vec<(i64, Row, crate::storage::mvcc::RowVersion)> =
            Vec::with_capacity(remaining_row_ids.len());
        if !remaining_row_ids.is_empty() {
            let batch_rows = self
                .version_store
                .get_visible_versions_for_update(&remaining_row_ids, self.txn_id);
            for (row_id, row, version) in batch_rows {
                let row = self.normalize_row_to_schema(row, schema);
                let (updated_row, _) = setter(row);
                rows_with_originals.push((row_id, updated_row, version));
            }
        }

        // Step 3: Batch put all updates
        let update_count = (local_rows.len() + rows_with_originals.len()) as i32;
        if !local_rows.is_empty() || !rows_with_originals.is_empty() {
            let mut txn_versions = self.txn_versions.write().unwrap();
            if !local_rows.is_empty() {
                txn_versions.put_batch_for_update(local_rows)?;
            }
            if !rows_with_originals.is_empty() {
                txn_versions.put_batch_with_originals(rows_with_originals)?;
            }
        }

        Ok(update_count)
    }

    fn delete_by_row_ids(&mut self, row_ids: &[i64]) -> Result<i32> {
        let schema = &self.cached_schema;

        // Step 1: Check local versions first
        let mut local_deletes: Vec<(i64, Row)> = Vec::with_capacity(row_ids.len() / 4);
        let mut remaining_row_ids: Vec<i64> = Vec::with_capacity(row_ids.len());

        {
            let txn_versions = self.txn_versions.read().unwrap();
            for &row_id in row_ids {
                if let Some(row) = txn_versions.get(row_id) {
                    // Local version - mark for deletion
                    let row = self.normalize_row_to_schema(row, schema);
                    local_deletes.push((row_id, row));
                } else {
                    remaining_row_ids.push(row_id);
                }
            }
        }

        // Step 2: Batch fetch remaining from version store
        let mut rows_with_originals: Vec<(i64, Row, crate::storage::mvcc::RowVersion)> =
            Vec::with_capacity(remaining_row_ids.len());
        if !remaining_row_ids.is_empty() {
            let batch_rows = self
                .version_store
                .get_visible_versions_for_update(&remaining_row_ids, self.txn_id);
            for (row_id, row, version) in batch_rows {
                let row = self.normalize_row_to_schema(row, schema);
                rows_with_originals.push((row_id, row, version));
            }
        }

        // Step 3: Batch delete all rows
        let delete_count = (local_deletes.len() + rows_with_originals.len()) as i32;
        if !local_deletes.is_empty() || !rows_with_originals.is_empty() {
            let mut txn_versions = self.txn_versions.write().unwrap();
            // Delete local rows
            for (row_id, row) in local_deletes {
                txn_versions.put(row_id, row, true)?;
            }
            // Delete rows from version store with originals
            for (row_id, row, orig) in rows_with_originals {
                txn_versions.put_with_original(row_id, row, orig, true)?;
            }
        }

        Ok(delete_count)
    }

    fn get_active_row_ids(&self) -> Vec<i64> {
        self.version_store.get_all_row_ids()
    }

    fn delete(&mut self, where_expr: Option<&dyn Expression>) -> Result<i32> {
        // OPTIMIZATION: Borrow schema instead of cloning - saves allocation per delete
        let schema = &self.cached_schema;

        // Fast path: Check if this is a primary key equality lookup (WHERE id = X)
        if let Some(expr) = where_expr {
            if let Some(pk_id) = self.try_pk_lookup(expr, schema) {
                // Direct O(1) lookup by primary key
                // OPTIMIZATION: Track original version to avoid redundant lookup in put()
                let row_with_original = {
                    let txn_versions = self.txn_versions.read().unwrap();
                    if let Some(row) = txn_versions.get(pk_id) {
                        // Local version - no need to track original (already in write-set)
                        Some((row, None))
                    } else if let Some(version) =
                        self.version_store.get_visible_version(pk_id, self.txn_id)
                    {
                        if !version.is_deleted() {
                            let data = version.data.clone();
                            Some((data, Some(version)))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                if let Some((row, original_version)) = row_with_original {
                    if let Some(orig) = original_version {
                        // Use optimized put that skips redundant get_visible_version
                        self.txn_versions
                            .write()
                            .unwrap()
                            .put_with_original(pk_id, row, orig, true)?;
                    } else {
                        // Local version - use regular put
                        self.txn_versions.write().unwrap().put(pk_id, row, true)?;
                    }
                    return Ok(1);
                }
                return Ok(0);
            }

            // Fast path: PK range lookup (WHERE id >= X AND id < Y)
            // PK IS the row_id, so we can generate the range directly
            if let Some(pk_range_ids) = self.try_pk_range_lookup(expr, schema) {
                // OPTIMIZATION: Track original versions to avoid redundant lookups
                let mut local_rows: Vec<(i64, Row)> = Vec::with_capacity(pk_range_ids.len() / 4);
                let mut rows_with_originals: Vec<(i64, Row, crate::storage::mvcc::RowVersion)> =
                    Vec::with_capacity(pk_range_ids.len());

                // Step 1: Check local versions first (single lock acquisition)
                let mut remaining_row_ids: Vec<i64> = Vec::with_capacity(pk_range_ids.len());
                {
                    let txn_versions = self.txn_versions.read().unwrap();
                    for row_id in pk_range_ids {
                        if let Some(row) = txn_versions.get(row_id) {
                            // Local version
                            local_rows.push((row_id, row));
                        } else {
                            remaining_row_ids.push(row_id);
                        }
                    }
                }

                // Step 2: Batch fetch remaining from version store (1 lock for N rows)
                if !remaining_row_ids.is_empty() {
                    let batch_rows = self
                        .version_store
                        .get_visible_versions_for_update(&remaining_row_ids, self.txn_id);
                    for (row_id, row, version) in batch_rows {
                        rows_with_originals.push((row_id, row, version));
                    }
                }

                let delete_count = (local_rows.len() + rows_with_originals.len()) as i32;
                if !local_rows.is_empty() || !rows_with_originals.is_empty() {
                    let mut txn_versions = self.txn_versions.write().unwrap();
                    if !local_rows.is_empty() {
                        txn_versions.put_batch_deleted(local_rows)?;
                    }
                    if !rows_with_originals.is_empty() {
                        txn_versions.put_batch_deleted_with_originals(rows_with_originals)?;
                    }
                }
                return Ok(delete_count);
            }

            // Try index lookup for non-PK columns
            if let Some(filtered_row_ids) = self.try_index_lookup(expr, schema) {
                // OPTIMIZATION: Batch collect rows to delete, then batch put
                // This reduces lock contention from 2N locks to 2 locks for N rows
                let mut rows_to_delete: Vec<(i64, Row)> =
                    Vec::with_capacity(filtered_row_ids.len());

                // Step 1: Check local versions first (single lock acquisition)
                let mut remaining_row_ids: Vec<i64> = Vec::with_capacity(filtered_row_ids.len());
                {
                    let txn_versions = self.txn_versions.read().unwrap();
                    for row_id in filtered_row_ids {
                        if let Some(row) = txn_versions.get(row_id) {
                            // Re-apply filter (index may be partial match)
                            if expr.evaluate(&row).unwrap_or(false) {
                                rows_to_delete.push((row_id, row));
                            }
                        } else {
                            remaining_row_ids.push(row_id);
                        }
                    }
                }

                // Step 2: Batch fetch remaining from version store (1 lock for N rows)
                if !remaining_row_ids.is_empty() {
                    let batch_rows = self
                        .version_store
                        .get_visible_versions_batch(&remaining_row_ids, self.txn_id);
                    for (row_id, row) in batch_rows {
                        // Re-apply filter (index may be partial match)
                        if expr.evaluate(&row).unwrap_or(false) {
                            rows_to_delete.push((row_id, row));
                        }
                    }
                }

                // Single batch write for all deletes
                let delete_count = rows_to_delete.len() as i32;
                if !rows_to_delete.is_empty() {
                    self.txn_versions
                        .write()
                        .unwrap()
                        .put_batch_deleted(rows_to_delete)?;
                }
                return Ok(delete_count);
            }
        }

        // Fall back to full scan
        let mut delete_count = 0;

        // Get all visible rows
        let row_ids = self.version_store.get_all_row_ids();

        for row_id in row_ids {
            // OPTIMIZATION: Check filter BEFORE cloning to avoid wasted allocations
            // For DELETE with selective WHERE, this can save 90%+ of clones

            // First, check local versions (already cloned, no extra cost)
            let local_row = {
                let txn_versions = self.txn_versions.read().unwrap();
                txn_versions.get(row_id)
            };

            if let Some(row) = local_row {
                // Apply filter on local row (no clone needed - already have it)
                if let Some(expr) = where_expr {
                    match expr.evaluate(&row) {
                        Ok(true) => {}
                        Ok(false) => continue,
                        Err(_) => continue,
                    }
                }
                // Mark as deleted
                self.txn_versions.write().unwrap().put(row_id, row, true)?;
                delete_count += 1;
            } else if let Some(version) =
                self.version_store.get_visible_version(row_id, self.txn_id)
            {
                if version.is_deleted() {
                    continue;
                }

                // Apply filter BEFORE cloning - evaluate on reference
                if let Some(expr) = where_expr {
                    match expr.evaluate(&version.data) {
                        Ok(true) => {}
                        Ok(false) => continue, // Skip clone entirely!
                        Err(_) => continue,
                    }
                }

                // Only clone AFTER filter passes
                self.txn_versions
                    .write()
                    .unwrap()
                    .put(row_id, version.data.clone(), true)?;
                delete_count += 1;
            }
        }

        // Also check local inserts that might not be in global store
        let local_ids: Vec<i64> = {
            let txn_versions = self.txn_versions.read().unwrap();
            txn_versions.iter_local().map(|(id, _)| id).collect()
        };
        for row_id in local_ids {
            // Skip if already processed
            if self.version_store.quick_check_row_existence(row_id) {
                continue;
            }

            let row = {
                let txn_versions = self.txn_versions.read().unwrap();
                txn_versions.get(row_id)
            };
            if let Some(row) = row {
                // Apply filter
                if let Some(expr) = where_expr {
                    match expr.evaluate(&row) {
                        Ok(true) => {}
                        Ok(false) => continue,
                        Err(_) => continue,
                    }
                }

                // Row is already owned from txn_versions.get(), no extra clone needed
                self.txn_versions.write().unwrap().put(row_id, row, true)?;
                delete_count += 1;
            }
        }

        Ok(delete_count)
    }

    fn scan(
        &self,
        column_indices: &[usize],
        where_expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn Scanner>> {
        // Arc clone - O(1) reference count increment instead of full Schema clone
        let schema = Arc::clone(&self.cached_schema);

        // Fast path: Check if this is a primary key equality lookup (WHERE id = X)
        if let Some(expr) = where_expr {
            if let Some(pk_lookup) = self.try_pk_lookup(expr, &schema) {
                // Direct O(1) lookup by primary key
                // First check local transaction changes, then committed data
                let row = {
                    let txn_versions = self.txn_versions.read().unwrap();
                    if let Some(row) = txn_versions.get(pk_lookup) {
                        Some(row)
                    } else if let Some(version) = self
                        .version_store
                        .get_visible_version(pk_lookup, self.txn_id)
                    {
                        if !version.is_deleted() {
                            Some(version.data.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                if let Some(row) = row {
                    // Normalize row to match current schema (handles ALTER TABLE ADD/DROP COLUMN)
                    let row = self.normalize_row_to_schema(row, &schema);
                    let scanner = MVCCScanner::from_rows(
                        vec![(pk_lookup, row)],
                        schema,
                        column_indices.to_vec(),
                    );
                    return Ok(Box::new(scanner));
                } else {
                    // Row not found - return empty scanner
                    let scanner = MVCCScanner::empty(schema, column_indices.to_vec());
                    return Ok(Box::new(scanner));
                }
            }

            // Try index lookup for non-PK columns
            if let Some(filtered_row_ids) = self.try_index_lookup(expr, &schema) {
                // Use index-based scan - much more efficient
                let rows = self.fetch_rows_by_ids(&filtered_row_ids, expr);
                let scanner = MVCCScanner::from_rows(rows, schema, column_indices.to_vec());
                return Ok(Box::new(scanner));
            }
        }

        // Fall back to full scan
        // Check if we can use lazy scanning (no local uncommitted changes)
        let has_local = self.txn_versions.read().unwrap().has_local_changes();

        if !has_local && where_expr.is_none() {
            // Use lazy scanner for unfiltered full scans with no local changes
            // This enables early termination and reduced memory usage
            let scanner = LazyMVCCScanner::new(
                Arc::clone(&self.version_store),
                self.txn_id,
                &schema,
                column_indices.to_vec(),
            );
            return Ok(Box::new(scanner));
        }

        // Fall back to eager scanning with filtering
        let rows = self.collect_visible_rows(where_expr);
        let scanner = MVCCScanner::from_rows(rows, schema, column_indices.to_vec());
        Ok(Box::new(scanner))
    }

    fn collect_all_rows(&self, where_expr: Option<&dyn Expression>) -> Result<Vec<Row>> {
        // Collect visible rows and extract just the Row values (discard row IDs)
        let rows = self.collect_visible_rows(where_expr);
        Ok(rows.into_iter().map(|(_, row)| row).collect())
    }

    fn collect_all_rows_unsorted(&self) -> Result<Vec<Row>> {
        // Use the unsorted collection for GROUP BY optimization
        Ok(self.collect_visible_rows_unsorted())
    }

    fn collect_projected_rows(&self, column_indices: &[usize]) -> Result<Vec<Row>> {
        // Collect visible rows and project directly during collection
        // This avoids the double-clone overhead of the scanner interface
        let rows = self.collect_visible_rows(None);
        let num_cols = column_indices.len();

        Ok(rows
            .into_iter()
            .map(|(_, row)| {
                let mut values = Vec::with_capacity(num_cols);
                for &idx in column_indices {
                    values.push(row.get(idx).cloned().unwrap_or(Value::null_unknown()));
                }
                Row::from_values(values)
            })
            .collect())
    }

    fn collect_rows_with_limit(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Row>> {
        // Use the optimized version with limit/offset
        Ok(self.collect_visible_rows_with_limit(where_expr, limit, offset))
    }

    fn collect_rows_with_limit_unordered(
        &self,
        where_expr: Option<&dyn Expression>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Row>> {
        // Use the optimized unordered version with true early termination
        Ok(self.collect_visible_rows_with_limit_unordered(where_expr, limit, offset))
    }

    fn collect_rows_sorted_with_limit(
        &self,
        sort_col_idx: usize,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Row>> {
        // DEFERRED MATERIALIZATION OPTIMIZATION
        // Instead of cloning all rows, sorting, and taking limit:
        // 1. Get row indices (no cloning)
        // 2. Load only sort column values
        // 3. Sort indices by values
        // 4. Take top N indices
        // 5. Materialize only N rows
        //
        // Performance: For 100K rows with 20 columns, LIMIT 10:
        // - Old: Clone 2M values, sort, take 10
        // - New: Load 100K sort values, sort indices, clone 200 values

        // Check for local versions - if present, fall back to default implementation
        let has_local = self.txn_versions.read().unwrap().has_local_changes();
        if has_local {
            // Local changes exist - use slower but correct path
            let mut rows = self.collect_visible_rows(None);
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
            return Ok(rows
                .into_iter()
                .skip(offset)
                .take(limit)
                .map(|(_, row)| row)
                .collect());
        }

        // FAST PATH: Use deferred materialization from version_store
        let rows = self.version_store.get_visible_rows_sorted_limit(
            self.txn_id,
            sort_col_idx,
            ascending,
            limit,
            offset,
        );

        // Normalize rows to schema and return
        let schema = &self.cached_schema;
        Ok(rows
            .into_iter()
            .map(|(_, row)| self.normalize_row_to_schema(row, schema))
            .collect())
    }

    fn close(&mut self) -> Result<()> {
        // Rollback any uncommitted changes
        self.txn_versions.write().unwrap().rollback();
        Ok(())
    }

    fn commit(&mut self) -> Result<()> {
        // Call inherent method which handles index updates
        MVCCTable::commit(self)
    }

    fn rollback(&mut self) {
        self.txn_versions.write().unwrap().rollback();
    }

    fn rollback_to_timestamp(&self, timestamp: i64) {
        self.txn_versions
            .write()
            .unwrap()
            .rollback_to_timestamp(timestamp);
    }

    fn has_local_changes(&self) -> bool {
        self.txn_versions.read().unwrap().has_local_changes()
    }

    fn get_pending_versions(&self) -> Vec<(i64, Row, bool, i64)> {
        let txn_versions = self.txn_versions.read().unwrap();
        txn_versions
            .iter_local()
            .map(|(row_id, version)| {
                (
                    row_id,
                    version.data.clone(),
                    version.is_deleted(),
                    version.txn_id,
                )
            })
            .collect()
    }

    fn create_index(&self, name: &str, columns: &[&str], is_unique: bool) -> Result<()> {
        // Delegate to create_index_with_type with auto-selection
        self.create_index_with_type(name, columns, is_unique, None)
    }

    fn create_index_with_type(
        &self,
        name: &str,
        columns: &[&str],
        is_unique: bool,
        index_type: Option<IndexType>,
    ) -> Result<()> {
        if columns.is_empty() {
            return Err(Error::internal("index must have at least one column"));
        }

        let schema = self.version_store.schema();

        // Collect column info
        let mut column_names = Vec::with_capacity(columns.len());
        let mut column_ids = Vec::with_capacity(columns.len());
        let mut data_types = Vec::with_capacity(columns.len());
        let mut col_indices = Vec::with_capacity(columns.len());

        for col_name in columns {
            let (col_idx, col) = schema
                .find_column(col_name)
                .ok_or(Error::ColumnNotFoundNamed(col_name.to_string()))?;
            column_names.push(col.name.clone());
            column_ids.push(col.id as i32);
            data_types.push(col.data_type);
            col_indices.push(col_idx);
        }

        // Determine index type: use explicit type or auto-select based on column types
        // For multi-column indexes, always use MultiColumn (hash+btree hybrid)
        let chosen_type = if columns.len() > 1 {
            IndexType::MultiColumn
        } else {
            index_type.unwrap_or_else(|| Self::auto_select_index_type(&data_types))
        };

        // Check if index with same name already exists
        if self.version_store.index_exists(name) {
            return Err(Error::IndexAlreadyExistsByName(name.to_string()));
        }

        // Check if an index already exists on the same column(s)
        // For single-column indexes, use get_index_by_column
        // For multi-column indexes, check all existing indexes
        if columns.len() == 1 {
            if let Some(existing_idx) = self.version_store.get_index_by_column(columns[0]) {
                if is_unique && !existing_idx.is_unique() {
                    return Err(Error::internal(format!(
                        "cannot create unique index on column '{}': a non-unique index already exists",
                        columns[0]
                    )));
                } else if !is_unique && existing_idx.is_unique() {
                    return Err(Error::internal(format!(
                        "cannot create non-unique index on column '{}': a unique index already exists",
                        columns[0]
                    )));
                }
                // Same type of index on same column - also reject
                return Err(Error::internal(format!(
                    "an index already exists on column '{}'",
                    columns[0]
                )));
            }
        } else {
            // For multi-column indexes, check if any existing index has the exact same columns
            for existing_idx in self.version_store.get_all_indexes() {
                let existing_cols = existing_idx.column_names();
                if existing_cols.len() == columns.len() {
                    let same_cols = existing_cols
                        .iter()
                        .zip(columns.iter())
                        .all(|(a, b)| a == *b);
                    if same_cols {
                        if is_unique && !existing_idx.is_unique() {
                            return Err(Error::internal(format!(
                                "cannot create unique index on columns {:?}: a non-unique index already exists",
                                columns
                            )));
                        } else if !is_unique && existing_idx.is_unique() {
                            return Err(Error::internal(format!(
                                "cannot create non-unique index on columns {:?}: a unique index already exists",
                                columns
                            )));
                        }
                        return Err(Error::internal(format!(
                            "an index already exists on columns {:?}",
                            columns
                        )));
                    }
                }
            }
        }

        // Create the appropriate index type
        let index: Arc<dyn Index> = match chosen_type {
            IndexType::Hash => Arc::new(HashIndex::new(
                name.to_string(),
                self.name().to_string(),
                column_names,
                column_ids,
                data_types,
                is_unique,
            )),
            IndexType::Bitmap => Arc::new(BitmapIndex::new(
                name.to_string(),
                self.name().to_string(),
                column_names,
                column_ids,
                data_types,
                is_unique,
            )),
            IndexType::BTree => {
                // For single-column BTree, use BTreeIndex
                // For multi-column, use MultiColumnIndex
                if columns.len() == 1 {
                    Arc::new(BTreeIndex::new(
                        name.to_string(),
                        self.name().to_string(),
                        column_ids[0],
                        column_names[0].clone(),
                        data_types[0],
                        is_unique,
                    ))
                } else {
                    Arc::new(MultiColumnIndex::new(
                        name.to_string(),
                        self.name().to_string(),
                        column_names,
                        column_ids,
                        data_types,
                        is_unique,
                    ))
                }
            }
            IndexType::MultiColumn => {
                // MultiColumn always uses MultiColumnIndex
                Arc::new(MultiColumnIndex::new(
                    name.to_string(),
                    self.name().to_string(),
                    column_names,
                    column_ids,
                    data_types,
                    is_unique,
                ))
            }
        };

        // Populate the index with existing data
        for row_id in self.version_store.get_all_visible_row_ids(self.txn_id) {
            if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                if !version.is_deleted() {
                    let values: Vec<Value> = col_indices
                        .iter()
                        .map(|&idx| {
                            version
                                .data
                                .get(idx)
                                .cloned()
                                .unwrap_or(Value::Null(DataType::Null))
                        })
                        .collect();
                    index.add(&values, row_id, row_id)?;
                }
            }
        }

        // Also add any local uncommitted data
        let txn_versions = self.txn_versions.read().unwrap();
        for (row_id, version) in txn_versions.iter_local() {
            if !version.is_deleted() && !self.version_store.quick_check_row_existence(row_id) {
                let values: Vec<Value> = col_indices
                    .iter()
                    .map(|&idx| {
                        version
                            .data
                            .get(idx)
                            .cloned()
                            .unwrap_or(Value::Null(DataType::Null))
                    })
                    .collect();
                index.add(&values, row_id, row_id)?;
            }
        }
        drop(txn_versions);

        // Add to version store
        self.version_store.add_index(name.to_string(), index);

        Ok(())
    }

    fn drop_index(&self, name: &str) -> Result<()> {
        // Check if index exists
        if !self.version_store.index_exists(name) {
            return Err(Error::IndexNotFoundByName(name.to_string()));
        }

        // Remove from version store
        self.version_store.remove_index(name);
        Ok(())
    }

    fn has_index_on_column(&self, column_name: &str) -> bool {
        self.version_store
            .get_index_by_column(column_name)
            .is_some()
    }

    fn get_index_on_column(&self, column_name: &str) -> Option<std::sync::Arc<dyn Index>> {
        self.version_store.get_index_by_column(column_name)
    }

    fn get_index(&self, name: &str) -> Option<std::sync::Arc<dyn Index>> {
        self.version_store.get_index(name)
    }

    fn get_multi_column_index(
        &self,
        predicate_columns: &[&str],
    ) -> Option<(std::sync::Arc<dyn Index>, usize)> {
        self.version_store.get_multi_column_index(predicate_columns)
    }

    fn create_btree_index(
        &self,
        column_name: &str,
        is_unique: bool,
        custom_name: Option<&str>,
    ) -> Result<()> {
        // Find the column
        let schema = self.version_store.schema();
        let (col_idx, col) = schema
            .find_column(column_name)
            .ok_or(Error::ColumnNotFound)?;

        // Generate index name
        let index_name = custom_name
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("idx_{}_{}_btree", self.name(), column_name));

        // Check if index with same name already exists
        if self.version_store.index_exists(&index_name) {
            return Err(Error::internal(format!(
                "index already exists: {}",
                index_name
            )));
        }

        // Check if an index already exists on this column
        if let Some(existing_index) = self.version_store.get_index_by_column(column_name) {
            let existing_is_unique = existing_index.is_unique();
            if existing_is_unique && !is_unique {
                return Err(Error::internal(format!(
                    "cannot create non-unique index on column '{}': a unique index already exists",
                    column_name
                )));
            } else if !existing_is_unique && is_unique {
                return Err(Error::internal(format!(
                    "cannot create unique index on column '{}': a non-unique index already exists",
                    column_name
                )));
            } else {
                return Err(Error::internal(format!(
                    "an index already exists on column '{}'",
                    column_name
                )));
            }
        }

        // Create the btree index
        let index = BTreeIndex::new(
            index_name.clone(),
            self.name().to_string(),
            col.id as i32,
            column_name.to_string(),
            col.data_type,
            is_unique,
        );

        // Populate the index with existing data
        // Scan all visible rows
        for row_id in self.version_store.get_all_visible_row_ids(self.txn_id) {
            if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                if !version.is_deleted() {
                    if let Some(value) = version.data.get(col_idx) {
                        index.add(std::slice::from_ref(value), row_id, row_id)?;
                    }
                }
            }
        }

        // Also add any local uncommitted data
        let txn_versions = self.txn_versions.read().unwrap();
        for (row_id, version) in txn_versions.iter_local() {
            if !version.is_deleted() {
                if let Some(value) = version.data.get(col_idx) {
                    // Skip if already added from global store
                    if !self.version_store.quick_check_row_existence(row_id) {
                        index.add(std::slice::from_ref(value), row_id, row_id)?;
                    }
                }
            }
        }
        drop(txn_versions);

        // Add to version store
        self.version_store.add_index(index_name, Arc::new(index));

        Ok(())
    }

    /// Create a multi-column index using MultiColumnIndex
    fn create_multi_column_index(
        &self,
        name: &str,
        columns: &[&str],
        is_unique: bool,
    ) -> Result<()> {
        let schema = self.version_store.schema();

        // Validate and collect column info
        let mut column_names = Vec::with_capacity(columns.len());
        let mut column_ids = Vec::with_capacity(columns.len());
        let mut data_types = Vec::with_capacity(columns.len());
        let mut col_indices = Vec::with_capacity(columns.len());

        for col_name in columns {
            let (col_idx, col) = schema
                .find_column(col_name)
                .ok_or(Error::ColumnNotFoundNamed(col_name.to_string()))?;
            column_names.push(col.name.clone());
            column_ids.push(col.id as i32);
            data_types.push(col.data_type);
            col_indices.push(col_idx);
        }

        // Check if index with same name already exists
        if self.version_store.index_exists(name) {
            return Err(Error::IndexAlreadyExistsByName(name.to_string()));
        }

        // Create the multi-column index
        let index = MultiColumnIndex::new(
            name.to_string(),
            self.name().to_string(),
            column_names,
            column_ids,
            data_types,
            is_unique,
        );

        // Populate the index with existing data
        for row_id in self.version_store.get_all_visible_row_ids(self.txn_id) {
            if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                if !version.is_deleted() {
                    // Collect values for all columns in the index
                    let values: Vec<Value> = col_indices
                        .iter()
                        .map(|&idx| {
                            version
                                .data
                                .get(idx)
                                .cloned()
                                .unwrap_or(Value::Null(DataType::Null))
                        })
                        .collect();
                    index.add(&values, row_id, row_id)?;
                }
            }
        }

        // Also add any local uncommitted data
        let txn_versions = self.txn_versions.read().unwrap();
        for (row_id, version) in txn_versions.iter_local() {
            if !version.is_deleted() {
                // Skip if already added from global store
                if !self.version_store.quick_check_row_existence(row_id) {
                    let values: Vec<Value> = col_indices
                        .iter()
                        .map(|&idx| {
                            version
                                .data
                                .get(idx)
                                .cloned()
                                .unwrap_or(Value::Null(DataType::Null))
                        })
                        .collect();
                    index.add(&values, row_id, row_id)?;
                }
            }
        }
        drop(txn_versions);

        // Add to version store
        self.version_store
            .add_index(name.to_string(), Arc::new(index));

        Ok(())
    }

    fn drop_btree_index(&self, column_name: &str) -> Result<()> {
        // Generate default index name
        let index_name = format!("idx_{}_{}_btree", self.name(), column_name);

        // Check if index exists
        if !self.version_store.index_exists(&index_name) {
            return Err(Error::internal(format!(
                "btree index not found for column: {}",
                column_name
            )));
        }

        // Remove from version store
        self.version_store.remove_index(&index_name);
        Ok(())
    }

    fn get_index_min_value(&self, column_name: &str) -> Option<Value> {
        // Try to find an index on this column and get its minimum value
        if let Some(index) = self.version_store.get_index_by_column(column_name) {
            return index.get_min_value();
        }
        None
    }

    fn get_index_max_value(&self, column_name: &str) -> Option<Value> {
        // Try to find an index on this column and get its maximum value
        if let Some(index) = self.version_store.get_index_by_column(column_name) {
            return index.get_max_value();
        }
        None
    }

    fn row_count(&self) -> usize {
        // Use the optimized row_count method that uses single-pass counting
        MVCCTable::row_count(self)
    }

    fn row_count_hint(&self) -> usize {
        // O(1) upper bound: just return versions.len() without visibility checks
        self.version_store.row_count()
    }

    fn collect_rows_ordered_by_index(
        &self,
        column_name: &str,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<Vec<Row>> {
        // OPTIMIZATION: Handle PRIMARY KEY column specially
        // For INTEGER PRIMARY KEY, the row_id IS the value, so we can iterate
        // the BTreeMap directly in order without any sorting.
        // This uses efficient BTreeMap iteration with skip/take semantics.
        if let Some(pk_idx) = self.cached_schema.pk_column_index() {
            let pk_col = &self.cached_schema.columns[pk_idx];
            if pk_col.name_lower == column_name.to_lowercase() {
                // Use efficient BTreeMap iteration with OFFSET/LIMIT
                // This iterates the BTreeMap directly (already sorted by row_id)
                // and only clones rows that are actually returned (not skipped)
                return self.version_store.collect_rows_pk_ordered(
                    self.txn_id,
                    ascending,
                    limit,
                    offset,
                );
            }
        }

        // Check if column has an index
        let index = self.version_store.get_index_by_column(column_name)?;

        // Try using the efficient ordered iteration method (available in B-tree indexes)
        // We request more row IDs than needed to account for invisible rows
        let batch_size = (limit + offset) * 2 + 100; // Request extra to handle filtered rows

        if let Some(ordered_row_ids) = index.get_row_ids_ordered(ascending, batch_size, 0) {
            // Fast path: B-tree index supports ordered iteration
            let mut rows = Vec::with_capacity(limit.min(100));
            let mut skipped = 0;

            for row_id in ordered_row_ids {
                // Check visibility and get row
                if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                    if version.is_deleted() {
                        continue;
                    }

                    // Handle offset
                    if skipped < offset {
                        skipped += 1;
                        continue;
                    }

                    rows.push(version.data.clone());

                    // Check if we've reached the limit
                    if rows.len() >= limit {
                        return Some(rows);
                    }
                }
            }

            // If we got all needed rows, return them
            // If not, we may need to fetch more (rare case with many invisible rows)
            if !rows.is_empty() {
                return Some(rows);
            }
        }

        // Fallback path: Get all values and sort (for non-B-tree indexes)
        let all_values = index.get_all_values();

        // Sort values using partial_cmp (Value implements PartialOrd)
        let mut sorted_values = all_values;
        sorted_values.sort_by(|a, b| {
            if ascending {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Collect rows by iterating through sorted values
        let mut rows = Vec::with_capacity(limit.min(100));
        let mut skipped = 0;

        for value in sorted_values {
            // Get all row IDs for this value
            let row_ids = index.get_row_ids_equal(&[value]);

            for row_id in row_ids {
                // Check visibility and get row
                if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                    if version.is_deleted() {
                        continue;
                    }

                    // Handle offset
                    if skipped < offset {
                        skipped += 1;
                        continue;
                    }

                    rows.push(version.data.clone());

                    // Check if we've reached the limit
                    if rows.len() >= limit {
                        return Some(rows);
                    }
                }
            }
        }

        Some(rows)
    }

    fn collect_rows_pk_keyset(
        &self,
        start_after: Option<i64>,
        start_from: Option<i64>,
        ascending: bool,
        limit: usize,
    ) -> Option<Vec<Row>> {
        // Only works if table has a single-column INTEGER PRIMARY KEY
        self.cached_schema.pk_column_index()?;

        // Use the efficient keyset iteration from version store
        let rows = self.version_store.collect_rows_keyset(
            self.txn_id,
            start_after,
            start_from,
            ascending,
            limit,
        );

        // Return rows only (discard row_ids)
        Some(rows.into_iter().map(|(_, row)| row).collect())
    }

    fn collect_rows_grouped_by_partition(
        &self,
        column_name: &str,
    ) -> Option<Vec<(Value, Vec<Row>)>> {
        // Check if column has an index
        let index = self.version_store.get_index_by_column(column_name)?;

        // Get all unique values from the index (partition keys)
        let all_values = index.get_all_values();
        if all_values.is_empty() {
            return Some(Vec::new());
        }

        // Collect rows grouped by partition value
        let mut result: Vec<(Value, Vec<Row>)> = Vec::with_capacity(all_values.len());

        for partition_value in all_values {
            // Get all row IDs for this partition value
            let row_ids = index.get_row_ids_equal(std::slice::from_ref(&partition_value));

            // Collect visible rows for this partition
            let mut partition_rows = Vec::with_capacity(row_ids.len());
            for row_id in row_ids {
                if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                    if !version.is_deleted() {
                        partition_rows.push(version.data.clone());
                    }
                }
            }

            if !partition_rows.is_empty() {
                result.push((partition_value, partition_rows));
            }
        }

        Some(result)
    }

    fn get_partition_values(&self, column_name: &str) -> Option<Vec<Value>> {
        // Only use index-based distinct values if no uncommitted local changes
        // (local changes are in txn_versions, not reflected in the index)
        let txn_versions = self.txn_versions.read().unwrap();
        if txn_versions.has_local_changes() {
            return None;
        }
        drop(txn_versions);

        // Get index for the column
        let index = self.version_store.get_index_by_column(column_name)?;
        // Return all distinct values from the index
        Some(index.get_all_values())
    }

    fn get_rows_for_partition_value(
        &self,
        column_name: &str,
        partition_value: &Value,
    ) -> Option<Vec<Row>> {
        // Get index for the column
        let index = self.version_store.get_index_by_column(column_name)?;

        // Get row IDs for this partition value
        let row_ids = index.get_row_ids_equal(std::slice::from_ref(partition_value));

        // Collect visible rows for this partition
        let mut rows = Vec::with_capacity(row_ids.len());
        for row_id in row_ids {
            if let Some(version) = self.version_store.get_visible_version(row_id, self.txn_id) {
                if !version.is_deleted() {
                    rows.push(version.data.clone());
                }
            }
        }

        Some(rows)
    }

    fn rename_column(&self, old_name: &str, new_name: &str) -> Result<()> {
        // This would need mutable access to schema
        // For now, return error - this operation should go through the engine
        Err(Error::internal(format!(
            "rename column not yet implemented: {} -> {}",
            old_name, new_name
        )))
    }

    fn modify_column(&self, name: &str, column_type: DataType, nullable: bool) -> Result<()> {
        // This would need mutable access to schema
        // For now, return error - this operation should go through the engine
        Err(Error::internal(format!(
            "modify column not yet implemented: {} {:?} (nullable: {})",
            name, column_type, nullable
        )))
    }

    fn select(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
    ) -> Result<Box<dyn QueryResult>> {
        // Convert column names to indices
        let column_indices: Vec<usize> = columns
            .iter()
            .filter_map(|name| self.cached_schema.find_column(name).map(|(idx, _)| idx))
            .collect();

        // Scan and collect results
        let mut scanner = self.scan(&column_indices, expr)?;
        let mut rows = Vec::new();

        while scanner.next() {
            rows.push(scanner.take_row());
        }

        scanner.close()?;

        // Create result
        let result_columns: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        let result = MemoryResult::with_rows(result_columns, rows);

        Ok(Box::new(result))
    }

    fn select_with_aliases(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        aliases: &FxHashMap<String, String>,
    ) -> Result<Box<dyn QueryResult>> {
        // Get base result
        let result = self.select(columns, expr)?;

        // Apply aliases
        Ok(result.with_aliases(aliases.clone()))
    }

    fn select_as_of(
        &self,
        columns: &[&str],
        expr: Option<&dyn Expression>,
        temporal_type: &str,
        temporal_value: i64,
    ) -> Result<Box<dyn QueryResult>> {
        // Convert column names to indices
        let column_indices: Vec<usize> = columns
            .iter()
            .filter_map(|name| self.cached_schema.find_column(name).map(|(idx, _)| idx))
            .collect();

        // Get all row IDs
        let row_ids = self.version_store.get_all_row_ids();

        // Collect temporal rows
        let mut rows = Vec::new();
        for row_id in row_ids {
            let version = match temporal_type.to_uppercase().as_str() {
                "TRANSACTION" => self
                    .version_store
                    .get_visible_version_as_of_transaction(row_id, temporal_value),
                "TIMESTAMP" => self
                    .version_store
                    .get_visible_version_as_of_timestamp(row_id, temporal_value),
                _ => {
                    return Err(Error::internal(format!(
                        "unsupported temporal type: {}",
                        temporal_type
                    )))
                }
            };

            if let Some(v) = version {
                if !v.is_deleted() {
                    // Apply filter
                    if let Some(e) = expr {
                        match e.evaluate(&v.data) {
                            Ok(true) => {}
                            Ok(false) => continue,
                            Err(_) => continue,
                        }
                    }

                    // Project columns
                    let projected: Vec<Value> = column_indices
                        .iter()
                        .map(|&idx| v.data.get(idx).cloned().unwrap_or(Value::null_unknown()))
                        .collect();
                    rows.push(Row::from_values(projected));
                }
            }
        }

        // Create result
        let result_columns: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        let result = MemoryResult::with_rows(result_columns, rows);

        Ok(Box::new(result))
    }

    fn explain_scan(&self, where_expr: Option<&dyn Expression>) -> ScanPlan {
        use crate::core::Operator;

        let table_name = self.cached_schema.table_name.clone();
        let schema = &self.cached_schema;

        // No WHERE clause - always Seq Scan
        let Some(expr) = where_expr else {
            return ScanPlan::SeqScan {
                table: table_name,
                filter: None,
            };
        };

        // Check for PK lookup
        let pk_indices = schema.primary_key_indices();
        if pk_indices.len() == 1 {
            let pk_col_idx = pk_indices[0];
            let pk_col = &schema.columns[pk_col_idx];

            if let Some((col_name, operator, value)) = expr.get_comparison_info() {
                if col_name.eq_ignore_ascii_case(&pk_col.name) && operator == Operator::Eq {
                    return ScanPlan::PkLookup {
                        table: table_name,
                        pk_column: pk_col.name.clone(),
                        pk_value: format!("{}", value),
                    };
                }
            }
        }

        // Check for single column index lookup
        if let Some((col_name, operator, value)) = expr.get_comparison_info() {
            // Skip boolean index (low cardinality)
            if !matches!(value, Value::Boolean(_))
                || !matches!(operator, Operator::Eq | Operator::Ne)
            {
                if let Some(index) = self.version_store.get_index_by_column(col_name) {
                    let condition = format!("{} {}", operator_to_string(operator), value);
                    return ScanPlan::IndexScan {
                        table: table_name,
                        index_name: index.name().to_string(),
                        column: col_name.to_string(),
                        condition,
                    };
                }
            }
        }

        // Check for LIKE prefix pattern
        if let Some((col_name, prefix, negated)) = expr.get_like_prefix_info() {
            if !negated {
                if let Some(index) = self.version_store.get_index_by_column(col_name) {
                    return ScanPlan::IndexScan {
                        table: table_name,
                        index_name: index.name().to_string(),
                        column: col_name.to_string(),
                        condition: format!("LIKE '{}%'", prefix),
                    };
                }
            }
        }

        // Check for OR expressions (union of indexes)
        if let Some(or_operands) = expr.get_or_operands() {
            let mut indexed_info: Vec<(String, String, String)> = Vec::new();
            let mut all_indexed = true;

            for operand in or_operands {
                if let Some((col_name, operator, value)) = operand.get_comparison_info() {
                    if let Some(index) = self.version_store.get_index_by_column(col_name) {
                        let condition = format!("{} {}", operator_to_string(operator), value);
                        indexed_info.push((
                            index.name().to_string(),
                            col_name.to_string(),
                            condition,
                        ));
                    } else {
                        all_indexed = false;
                    }
                } else {
                    all_indexed = false;
                }
            }

            if all_indexed && !indexed_info.is_empty() {
                if indexed_info.len() == 1 {
                    let (idx_name, col, cond) = indexed_info.into_iter().next().unwrap();
                    return ScanPlan::IndexScan {
                        table: table_name,
                        index_name: idx_name,
                        column: col,
                        condition: cond,
                    };
                }
                return ScanPlan::MultiIndexScan {
                    table: table_name,
                    indexes: indexed_info,
                    operation: "OR".to_string(),
                };
            }
        }

        // Check for multi-column (composite) index before single-column index intersection
        let comparisons = expr.collect_comparisons();
        if !comparisons.is_empty() {
            // Group by column - needed for both multi-column and single-column index checks
            let mut column_conditions: FxHashMap<&str, Vec<(Operator, &Value)>> =
                FxHashMap::default();
            for (col_name, op, val) in &comparisons {
                column_conditions
                    .entry(*col_name)
                    .or_default()
                    .push((*op, *val));
            }

            // Collect columns with equality predicates (best for multi-column index)
            let eq_columns: Vec<&str> = column_conditions
                .iter()
                .filter_map(|(col, ops)| {
                    if ops.iter().any(|(op, _)| *op == Operator::Eq) {
                        Some(*col)
                    } else {
                        None
                    }
                })
                .collect();

            // Try multi-column index if we have 2+ equality predicates
            if eq_columns.len() >= 2 {
                if let Some((multi_idx, matched_count)) =
                    self.version_store.get_multi_column_index(&eq_columns)
                {
                    let index_columns = multi_idx.column_names();
                    let mut columns = Vec::new();
                    let mut conditions = Vec::new();

                    // Build columns and conditions in index column order
                    for col in index_columns.iter().take(matched_count) {
                        if let Some(ops) = column_conditions.get(col.as_str()) {
                            columns.push(col.clone());
                            // Find the equality condition
                            for (op, val) in ops {
                                if *op == Operator::Eq {
                                    conditions.push(format!("= {}", val));
                                    break;
                                }
                            }
                        }
                    }

                    if !columns.is_empty() {
                        return ScanPlan::CompositeIndexScan {
                            table: table_name,
                            index_name: multi_idx.name().to_string(),
                            columns,
                            conditions,
                        };
                    }
                }
            }
        }

        // Check for AND expressions (intersection of indexes)
        if !comparisons.is_empty() {
            let mut indexed_info: Vec<(String, String, String)> = Vec::new();

            // Re-group by column for single-column index checks
            let mut column_conditions: FxHashMap<&str, Vec<(Operator, &Value)>> =
                FxHashMap::default();
            for (col_name, op, val) in &comparisons {
                column_conditions
                    .entry(*col_name)
                    .or_default()
                    .push((*op, *val));
            }

            for (col_name, ops) in &column_conditions {
                if let Some(index) = self.version_store.get_index_by_column(col_name) {
                    // Simplify to a single condition string
                    let condition = if ops.len() == 1 {
                        let (op, val) = ops[0];
                        format!("{} {}", operator_to_string(op), val)
                    } else {
                        // Multiple conditions on same column (e.g., col >= 5 AND col <= 10)
                        let parts: Vec<String> = ops
                            .iter()
                            .map(|(op, val)| format!("{} {}", operator_to_string(*op), val))
                            .collect();
                        parts.join(" AND ")
                    };
                    indexed_info.push((index.name().to_string(), col_name.to_string(), condition));
                }
            }

            if !indexed_info.is_empty() {
                if indexed_info.len() == 1 {
                    let (idx_name, col, cond) = indexed_info.into_iter().next().unwrap();
                    return ScanPlan::IndexScan {
                        table: table_name,
                        index_name: idx_name,
                        column: col,
                        condition: cond,
                    };
                }
                return ScanPlan::MultiIndexScan {
                    table: table_name,
                    indexes: indexed_info,
                    operation: "AND".to_string(),
                };
            }
        }

        // Default: Seq Scan with filter
        ScanPlan::SeqScan {
            table: table_name,
            filter: Some(format!("{:?}", expr)),
        }
    }

    fn set_zone_maps(&self, zone_maps: crate::storage::mvcc::zonemap::TableZoneMap) {
        self.version_store.set_zone_maps(zone_maps);
    }

    fn get_zone_maps(&self) -> Option<std::sync::Arc<crate::storage::mvcc::zonemap::TableZoneMap>> {
        self.version_store.get_zone_maps()
    }

    fn get_segments_to_scan(
        &self,
        column: &str,
        operator: crate::core::Operator,
        value: &Value,
    ) -> Option<Vec<u32>> {
        self.version_store
            .get_segments_to_scan(column, operator, value)
    }

    fn sum_column(&self, col_idx: usize) -> Option<(f64, usize)> {
        // Only use deferred aggregation if no uncommitted local changes
        // (local changes are in txn_versions, not the main version_store)
        let txn_versions = self.txn_versions.read().unwrap();
        if txn_versions.has_local_changes() {
            return None;
        }
        drop(txn_versions);

        Some(self.version_store.sum_column(self.txn_id, col_idx))
    }

    fn min_column(&self, col_idx: usize) -> Option<Option<Value>> {
        // Only use deferred aggregation if no uncommitted local changes
        let txn_versions = self.txn_versions.read().unwrap();
        if txn_versions.has_local_changes() {
            return None;
        }
        drop(txn_versions);

        Some(self.version_store.min_column(self.txn_id, col_idx))
    }

    fn max_column(&self, col_idx: usize) -> Option<Option<Value>> {
        // Only use deferred aggregation if no uncommitted local changes
        let txn_versions = self.txn_versions.read().unwrap();
        if txn_versions.has_local_changes() {
            return None;
        }
        drop(txn_versions);

        Some(self.version_store.max_column(self.txn_id, col_idx))
    }
}

/// Helper function to convert Operator to string for display
fn operator_to_string(op: crate::core::Operator) -> &'static str {
    use crate::core::Operator;
    match op {
        Operator::Eq => "=",
        Operator::Ne => "!=",
        Operator::Lt => "<",
        Operator::Lte => "<=",
        Operator::Gt => ">",
        Operator::Gte => ">=",
        Operator::Like => "LIKE",
        Operator::In => "IN",
        Operator::NotIn => "NOT IN",
        Operator::IsNull => "IS NULL",
        Operator::IsNotNull => "IS NOT NULL",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SchemaBuilder;

    fn test_schema() -> Schema {
        SchemaBuilder::new("test_table")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build()
    }

    fn simple_schema() -> Schema {
        SchemaBuilder::new("test_table")
            .column("id", DataType::Integer, false, true)
            .build()
    }

    #[test]
    fn test_mvcc_table_creation() {
        let schema = test_schema();

        let version_store = Arc::new(VersionStore::new("test_table".to_string(), schema));
        let txn_id = 1;

        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), txn_id);

        let table = MVCCTable::new(txn_id, version_store, txn_versions);

        assert_eq!(table.name(), "test_table");
        assert_eq!(table.schema().columns.len(), 2);
    }

    #[test]
    fn test_mvcc_table_insert_and_scan() {
        let schema = test_schema();

        let version_store = Arc::new(VersionStore::new("test_table".to_string(), schema));
        let txn_id = 1;

        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), txn_id);

        let mut table = MVCCTable::new(txn_id, version_store, txn_versions);

        // Insert a row
        let row = Row::from_values(vec![Value::Integer(1), Value::text("Alice")]);
        table.insert(row).unwrap();

        // Scan to verify
        let mut scanner = table.scan(&[0, 1], None).unwrap();

        assert!(scanner.next());
        let row = scanner.row();
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(1), Some(&Value::text("Alice")));

        assert!(!scanner.next());
        scanner.close().unwrap();
    }

    #[test]
    fn test_mvcc_table_duplicate_key_error() {
        let schema = simple_schema();

        let version_store = Arc::new(VersionStore::new("test_table".to_string(), schema));
        let txn_id = 1;

        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), txn_id);

        let mut table = MVCCTable::new(txn_id, version_store, txn_versions);

        // Insert first row
        let row1 = Row::from_values(vec![Value::Integer(1)]);
        table.insert(row1).unwrap();

        // Try to insert duplicate
        let row2 = Row::from_values(vec![Value::Integer(1)]);
        let result = table.insert(row2);

        assert!(result.is_err());
    }

    #[test]
    fn test_mvcc_table_delete() {
        let schema = simple_schema();

        let version_store = Arc::new(VersionStore::new("test_table".to_string(), schema));
        let txn_id = 1;

        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), txn_id);

        let mut table = MVCCTable::new(txn_id, version_store, txn_versions);

        // Insert rows
        table
            .insert(Row::from_values(vec![Value::Integer(1)]))
            .unwrap();
        table
            .insert(Row::from_values(vec![Value::Integer(2)]))
            .unwrap();
        table
            .insert(Row::from_values(vec![Value::Integer(3)]))
            .unwrap();

        // Delete all rows (no filter)
        let deleted = table.delete(None).unwrap();
        assert_eq!(deleted, 3);

        // Verify no rows remain
        let mut scanner = table.scan(&[0], None).unwrap();
        assert!(!scanner.next());
        scanner.close().unwrap();
    }

    #[test]
    fn test_mvcc_table_update() {
        let schema = SchemaBuilder::new("test_table")
            .column("id", DataType::Integer, false, true)
            .column("value", DataType::Integer, true, false)
            .build();

        let version_store = Arc::new(VersionStore::new("test_table".to_string(), schema));
        let txn_id = 1;

        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), txn_id);

        let mut table = MVCCTable::new(txn_id, version_store, txn_versions);

        // Insert row
        table
            .insert(Row::from_values(vec![
                Value::Integer(1),
                Value::Integer(10),
            ]))
            .unwrap();

        // Update the row
        let updated = table
            .update(None, &mut |row| {
                let mut new_row = row.clone();
                let _ = new_row.set(1, Value::Integer(20));
                (new_row, false)
            })
            .unwrap();

        assert_eq!(updated, 1);

        // Verify update
        let mut scanner = table.scan(&[0, 1], None).unwrap();
        assert!(scanner.next());
        let row = scanner.row();
        assert_eq!(row.get(1), Some(&Value::Integer(20)));
        scanner.close().unwrap();
    }

    #[test]
    fn test_mvcc_table_validation_error() {
        let schema = SchemaBuilder::new("test_table")
            .column("id", DataType::Integer, false, false)
            .column("name", DataType::Text, false, false) // NOT NULL
            .build();

        let version_store = Arc::new(VersionStore::new("test_table".to_string(), schema));
        let txn_id = 1;

        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), txn_id);

        let mut table = MVCCTable::new(txn_id, version_store, txn_versions);

        // Try to insert with NULL in non-nullable column
        let row = Row::from_values(vec![Value::Integer(1), Value::Null(DataType::Text)]);
        let result = table.insert(row);

        assert!(result.is_err());
    }

    #[test]
    fn test_mvcc_table_row_count() {
        let schema = simple_schema();

        let version_store = Arc::new(VersionStore::new("test_table".to_string(), schema));
        let txn_id = 1;

        let txn_versions = TransactionVersionStore::new(Arc::clone(&version_store), txn_id);

        let mut table = MVCCTable::new(txn_id, version_store, txn_versions);

        assert_eq!(table.row_count(), 0);

        table
            .insert(Row::from_values(vec![Value::Integer(1)]))
            .unwrap();
        table
            .insert(Row::from_values(vec![Value::Integer(2)]))
            .unwrap();

        assert_eq!(table.row_count(), 2);
    }
}
