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

use std::collections::HashMap;

use crate::core::{DataType, IndexEntry, IndexType, Operator, Result, Value};
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
    fn add_batch(&self, entries: &HashMap<i64, Vec<Value>>) -> Result<()>;

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
    fn remove_batch(&self, entries: &HashMap<i64, Vec<Value>>) -> Result<()>;

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
    fn get_row_ids_equal(&self, values: &[Value]) -> Vec<i64>;

    /// Returns row IDs with values in the given range (convenience method)
    ///
    /// This is a simplified version of `find_range` that returns only row IDs.
    fn get_row_ids_in_range(
        &self,
        min_value: &[Value],
        max_value: &[Value],
        include_min: bool,
        include_max: bool,
    ) -> Vec<i64>;

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
    /// Vector of row IDs matching any value in the list
    fn get_row_ids_in(&self, value_list: &[Value]) -> Vec<i64> {
        // Default implementation: do multiple equality lookups
        let mut results = Vec::new();
        for value in value_list {
            results.extend(self.get_row_ids_equal(std::slice::from_ref(value)));
        }
        results
    }

    /// Returns row IDs that match the given expression
    ///
    /// # Arguments
    /// * `expr` - The expression to evaluate
    ///
    /// # Returns
    /// Vector of row IDs matching the expression
    fn get_filtered_row_ids(&self, expr: &dyn Expression) -> Vec<i64>;

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
