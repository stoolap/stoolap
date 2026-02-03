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

//! Index trait for database indexes
//!

use crate::common::I64Map;
use crate::core::{DataType, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
use crate::storage::expression::Expression;

/// Index represents an abstract index for a column or set of columns
///
/// This trait defines the interface for index operations including
/// lookups, range queries, and batch operations.
///
/// # Index Types
///
/// - **BTree**: For high-cardinality columns (> 5% unique values)
/// - **Bitmap**: For low-cardinality columns (< 5% unique values)
/// - **BTree**: For range queries and ordered access
pub trait Index: Send + Sync {
    /// Returns the name of the index
    fn name(&self) -> &str;

    /// Returns the name of the table this index belongs to
    fn table_name(&self) -> &str;

    /// Builds the index from existing data
    fn build(&mut self) -> Result<()>;

    /// Adds values to the index with the given row IDs
    ///
    /// # Arguments
    /// * `values` - The column values to index (one per indexed column)
    /// * `row_id` - The row ID in the table
    /// * `ref_id` - The reference ID in the index
    ///
    /// Note: Uses `&self` with interior mutability for thread-safe concurrent access
    fn add(&self, values: &[Value], row_id: i64, ref_id: i64) -> Result<()>;

    /// Adds multiple entries to the index in a single batch operation
    ///
    /// # Arguments
    /// * `entries` - Map from row_id to column values
    fn add_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()>;

    /// Removes values from the index
    ///
    /// # Arguments
    /// * `values` - The column values to remove
    /// * `row_id` - The row ID in the table
    /// * `ref_id` - The reference ID in the index
    ///
    /// Note: Uses `&self` with interior mutability for thread-safe concurrent access
    fn remove(&self, values: &[Value], row_id: i64, ref_id: i64) -> Result<()>;

    /// Removes multiple entries from the index in a single batch operation
    ///
    /// # Arguments
    /// * `entries` - Map from row_id to column values
    fn remove_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()>;

    /// Adds multiple entries from a slice with single lock acquisition
    ///
    /// This is the most efficient batch method - avoids intermediate allocations
    /// and acquires write locks only once for the entire batch.
    ///
    /// # Arguments
    /// * `entries` - Slice of (row_id, values) pairs
    ///
    /// # Performance
    /// - Single lock acquisition for entire batch (vs N acquisitions for N calls to add())
    /// - No intermediate HashMap construction
    /// - Values are borrowed, not cloned (caller owns the data)
    fn add_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        // Default implementation: delegate to add() for backwards compatibility
        // Concrete implementations should override for single-lock optimization
        for &(row_id, values) in entries {
            self.add(values, row_id, row_id)?;
        }
        Ok(())
    }

    /// Removes multiple entries from a slice with single lock acquisition
    ///
    /// This is the most efficient batch removal method - avoids intermediate allocations
    /// and acquires write locks only once for the entire batch.
    ///
    /// # Arguments
    /// * `entries` - Slice of (row_id, values) pairs
    ///
    /// # Performance
    /// - Single lock acquisition for entire batch (vs N acquisitions for N calls to remove())
    /// - No intermediate HashMap construction
    /// - Values are borrowed, not cloned (caller owns the data)
    fn remove_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        // Default implementation: delegate to remove() for backwards compatibility
        // Concrete implementations should override for single-lock optimization
        for &(row_id, values) in entries {
            self.remove(values, row_id, row_id)?;
        }
        Ok(())
    }

    /// Returns the column IDs for this index
    fn column_ids(&self) -> &[i32];

    /// Returns the column names for this index
    fn column_names(&self) -> &[String];

    /// Returns the data types for the indexed columns
    fn data_types(&self) -> &[DataType];

    /// Returns the type of index (BTree, Bitmap, Hash)
    fn index_type(&self) -> IndexType;

    /// Returns true if this is a unique index
    fn is_unique(&self) -> bool;

    /// Finds all entries where the columns equal the given values
    ///
    /// # Arguments
    /// * `values` - The values to search for
    ///
    /// # Returns
    /// Vector of index entries matching the values
    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>>;

    /// Finds all entries where the columns are in the given range
    ///
    /// # Arguments
    /// * `min` - Minimum values (inclusive if min_inclusive is true)
    /// * `max` - Maximum values (inclusive if max_inclusive is true)
    /// * `min_inclusive` - Whether to include the minimum boundary
    /// * `max_inclusive` - Whether to include the maximum boundary
    ///
    /// # Returns
    /// Vector of index entries within the range
    fn find_range(
        &self,
        min: &[Value],
        max: &[Value],
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>>;

    /// Finds all entries matching the given operator and values
    ///
    /// # Arguments
    /// * `op` - The comparison operator
    /// * `values` - The values to compare against
    ///
    /// # Returns
    /// Vector of matching index entries
    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>>;

    /// Returns row IDs with the given values (convenience method)
    ///
    /// This is a simplified version of `find` that returns only row IDs.
    /// Returns a pooled RowIdVec that is automatically recycled on drop.
    fn get_row_ids_equal(&self, values: &[Value]) -> RowIdVec {
        let mut row_ids = RowIdVec::new();
        self.get_row_ids_equal_into(values, &mut row_ids);
        row_ids
    }

    /// Appends row IDs with the given values to the provided buffer
    ///
    /// This enables callers to reuse the vector allocation across multiple calls.
    fn get_row_ids_equal_into(&self, values: &[Value], buffer: &mut Vec<i64>) {
        // Default implementation: delegate to find (allocates)
        // Override in concrete indexes for zero-allocation
        if let Ok(entries) = self.find(values) {
            buffer.reserve(entries.len());
            for entry in entries {
                buffer.push(entry.row_id);
            }
        }
    }

    /// Returns row IDs with values in the given range (convenience method)
    ///
    /// This is a simplified version of `find_range` that returns only row IDs.
    /// Returns a pooled RowIdVec that is automatically recycled on drop.
    fn get_row_ids_in_range(
        &self,
        min_value: &[Value],
        max_value: &[Value],
        include_min: bool,
        include_max: bool,
    ) -> RowIdVec {
        let mut row_ids = RowIdVec::new();
        self.get_row_ids_in_range_into(
            min_value,
            max_value,
            include_min,
            include_max,
            &mut row_ids,
        );
        row_ids
    }

    /// Appends row IDs with values in the given range to the provided buffer
    ///
    /// This enables callers to reuse the vector allocation across multiple calls.
    fn get_row_ids_in_range_into(
        &self,
        min_value: &[Value],
        max_value: &[Value],
        include_min: bool,
        include_max: bool,
        buffer: &mut Vec<i64>,
    ) {
        // Default implementation: delegate to find_range (allocates)
        // Override in concrete indexes for zero-allocation
        if let Ok(entries) = self.find_range(min_value, max_value, include_min, include_max) {
            buffer.reserve(entries.len());
            for entry in entries {
                buffer.push(entry.row_id);
            }
        }
    }

    /// Returns row IDs for values in the given list (IN clause optimization)
    ///
    /// For hash indexes, this does O(k) equality lookups where k is the list size.
    /// For btree indexes, this does O(k * log n) lookups.
    /// Much faster than `get_filtered_row_ids` which may scan the entire index.
    ///
    /// # Arguments
    /// * `value_list` - List of values to match (each is a single column value)
    ///
    /// # Returns
    /// Pooled RowIdVec of row IDs matching any value in the list
    fn get_row_ids_in(&self, value_list: &[Value]) -> RowIdVec {
        let mut results = RowIdVec::new();
        self.get_row_ids_in_into(value_list, &mut results);
        results
    }

    /// Appends row IDs for values in the given list to the provided buffer
    ///
    /// This enables callers to reuse the vector allocation across multiple calls.
    fn get_row_ids_in_into(&self, value_list: &[Value], buffer: &mut Vec<i64>) {
        // Default implementation: do multiple equality lookups
        for value in value_list {
            self.get_row_ids_equal_into(std::slice::from_ref(value), buffer);
        }
    }

    /// Returns row IDs that match the given expression
    ///
    /// # Arguments
    /// * `expr` - The expression to evaluate
    ///
    /// # Returns
    /// Pooled RowIdVec of row IDs matching the expression
    fn get_filtered_row_ids(&self, expr: &dyn Expression) -> RowIdVec;

    /// Returns the minimum value in the index (for MIN aggregate optimization)
    ///
    /// This enables O(1) or O(log n) MIN queries on indexed columns
    /// instead of O(n) full table scans.
    fn get_min_value(&self) -> Option<Value> {
        None // Default implementation - override in concrete indexes
    }

    /// Returns the maximum value in the index (for MAX aggregate optimization)
    ///
    /// This enables O(1) or O(log n) MAX queries on indexed columns
    /// instead of O(n) full table scans.
    fn get_max_value(&self) -> Option<Value> {
        None // Default implementation - override in concrete indexes
    }

    /// Returns all unique values in the index (for ORDER BY optimization)
    ///
    /// This enables sorted iteration through unique values for TOP-N queries.
    fn get_all_values(&self) -> Vec<Value> {
        Vec::new() // Default implementation - override in concrete indexes
    }

    /// Returns the count of distinct non-null values in the index
    ///
    /// This enables O(1) COUNT(DISTINCT col) queries on indexed columns
    /// without cloning all values. Per SQL standard, NULL values are excluded.
    ///
    /// Returns None if the index doesn't support this optimization.
    fn get_distinct_count_excluding_null(&self) -> Option<usize> {
        None // Default implementation - override in concrete indexes
    }

    /// Returns row IDs in sorted order by index value (for ORDER BY optimization)
    ///
    /// This enables efficient ORDER BY queries by iterating through the B-tree
    /// in order, avoiding the need to sort the entire result set.
    ///
    /// # Arguments
    /// * `ascending` - If true, return in ascending order; if false, descending
    /// * `limit` - Maximum number of row IDs to return
    /// * `offset` - Number of row IDs to skip before returning
    ///
    /// # Returns
    /// Vector of row IDs in sorted order, or None if the index doesn't support ordered access
    fn get_row_ids_ordered(
        &self,
        _ascending: bool,
        _limit: usize,
        _offset: usize,
    ) -> Option<Vec<i64>> {
        None // Default implementation - only B-tree indexes support this
    }

    /// Returns grouped row IDs in sorted order by index value (for GROUP BY optimization)
    ///
    /// This enables streaming GROUP BY by iterating through the B-tree in order,
    /// processing one group at a time without needing a hash map.
    ///
    /// # Returns
    /// Vector of (group_value, row_ids) pairs in sorted order,
    /// or None if the index doesn't support ordered group access
    fn get_grouped_row_ids(&self) -> Option<Vec<(Value, Vec<i64>)>> {
        None // Default implementation - only B-tree indexes support this
    }

    /// Iterates through groups in sorted order, calling the callback for each group.
    ///
    /// This is a zero-allocation alternative to `get_grouped_row_ids` that processes
    /// groups one at a time without collecting all groups upfront. This is more
    /// efficient when:
    /// - Early termination is possible (LIMIT)
    /// - Only a few groups pass filtering (HAVING)
    ///
    /// # Arguments
    /// * `callback` - Called for each group with (value, row_ids). Return:
    ///   - `Ok(true)` to continue to next group
    ///   - `Ok(false)` to stop iteration early
    ///   - `Err(e)` to stop and propagate the error
    ///
    /// # Returns
    /// - `Some(Ok(()))` if iteration completed or stopped early with Ok(false)
    /// - `Some(Err(e))` if callback returned an error
    /// - `None` if the index doesn't support ordered group access
    fn for_each_group(
        &self,
        _callback: &mut dyn FnMut(&Value, &[i64]) -> Result<bool>,
    ) -> Option<Result<()>> {
        None // Default implementation - only B-tree indexes support this
    }

    /// Closes the index and releases any resources
    fn close(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    // Index tests will be implemented when we have concrete implementations
    // For now, just verify the trait compiles correctly

    use super::*;

    // Verify trait is object-safe
    fn _assert_object_safe(_: &dyn Index) {}
}
