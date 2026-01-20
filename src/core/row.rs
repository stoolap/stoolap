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

//! Row type for Stoolap - a collection of column values
//!
//! # Storage Design
//!
//! Row uses a simple 2-variant storage model:
//! - `Shared(CompactArc<[Value]>)`: O(1) clone, for storage reads and sharing
//! - `Owned(CompactVec<Value>)`: Mutable, for intermediate results
//!
//! This design avoids per-value Arc overhead while enabling:
//! - Row-level sharing between arena, version store, transaction store
//! - String-level sharing via SmartString::Shared(Arc<str>)
//! - Zero-copy JOINs via RowRef::Composite/DirectBuildComposite (see operator.rs)

use std::fmt;
use std::ops::Index;

use super::error::{Error, Result};
use super::schema::Schema;
use super::types::DataType;
use super::value::Value;
use crate::common::{CompactArc, CompactVec};

/// Internal storage for Row - simple 2-variant design
///
/// - `Shared`: Arc-wrapped for O(1) clone (storage reads, sharing)
/// - `Owned`: Direct values for mutations and intermediate results
///
/// Both variants are 16 bytes for optimal move performance.
#[derive(Debug, Clone)]
enum RowStorage {
    /// Shared storage - O(1) clone, immutable (checked first for read-heavy workloads)
    Shared(CompactArc<[Value]>),
    /// Owned storage - mutable, for intermediate results
    Owned(CompactVec<Value>),
}

impl Default for RowStorage {
    fn default() -> Self {
        RowStorage::Owned(CompactVec::new())
    }
}

impl PartialEq for RowStorage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RowStorage::Shared(a), RowStorage::Shared(b)) => {
                // Fast path: same Arc pointer
                CompactArc::ptr_eq(a, b) || a.as_ref() == b.as_ref()
            }
            (RowStorage::Owned(a), RowStorage::Owned(b)) => a.as_slice() == b.as_slice(),
            // Mixed storage types - compare by value
            (RowStorage::Shared(a), RowStorage::Owned(b)) => a.as_ref() == b.as_slice(),
            (RowStorage::Owned(a), RowStorage::Shared(b)) => a.as_slice() == b.as_ref(),
        }
    }
}

impl RowStorage {
    /// Get a value by index
    #[inline(always)]
    fn get(&self, index: usize) -> Option<&Value> {
        match self {
            RowStorage::Shared(arc) => arc.get(index),
            RowStorage::Owned(vec) => vec.get(index),
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            RowStorage::Shared(arc) => arc.len(),
            RowStorage::Owned(vec) => vec.len(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get mutable access to owned storage, converting if necessary (copy-on-write)
    #[inline]
    fn make_mut(&mut self) -> &mut CompactVec<Value> {
        match self {
            RowStorage::Owned(vec) => vec,
            RowStorage::Shared(arc) => {
                // Copy-on-write: convert shared to owned
                // Use extend_clone directly instead of .cloned().collect() to avoid
                // the Cloned iterator adapter overhead
                let len = arc.len();
                let mut vec = CompactVec::with_capacity(len);
                vec.extend_clone(arc);
                *self = RowStorage::Owned(vec);
                match self {
                    RowStorage::Owned(vec) => vec,
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Convert to owned Vec<Value>, consuming self
    #[inline]
    fn into_vec(self) -> Vec<Value> {
        match self {
            RowStorage::Owned(vec) => vec.into_vec(),
            RowStorage::Shared(arc) => arc.iter().cloned().collect(),
        }
    }
}

/// A database row containing column values
///
/// Row provides methods for accessing and manipulating column values
/// while maintaining type safety and consistency with the schema.
///
/// # Performance
///
/// - `Shared` storage: O(1) clone (just Arc increment)
/// - `Owned` storage: O(n) clone (copies values)
/// - String values use SmartString with internal Arc<str> for efficient sharing
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Row {
    storage: RowStorage,
}

/// Iterator over row values
pub struct RowIter<'a> {
    inner: std::slice::Iter<'a, Value>,
}

impl<'a> Iterator for RowIter<'a> {
    type Item = &'a Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for RowIter<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a> DoubleEndedIterator for RowIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

/// Mutable iterator over row values
pub struct RowIterMut<'a> {
    inner: std::slice::IterMut<'a, Value>,
}

impl<'a> Iterator for RowIterMut<'a> {
    type Item = &'a mut Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for RowIterMut<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl Row {
    /// Create a new empty row
    #[inline]
    pub fn new() -> Self {
        Self {
            storage: RowStorage::Owned(CompactVec::new()),
        }
    }

    /// Create a row with pre-allocated capacity
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            storage: RowStorage::Owned(CompactVec::with_capacity(capacity)),
        }
    }

    /// Create a row from a vector of values
    #[inline]
    pub fn from_values(values: Vec<Value>) -> Self {
        Self {
            storage: RowStorage::Owned(CompactVec::from_vec(values)),
        }
    }

    /// Create a row from CompactVec<Value>
    #[inline]
    pub fn from_compact_vec(values: CompactVec<Value>) -> Self {
        Self {
            storage: RowStorage::Owned(values),
        }
    }

    /// Create a row from an Arc slice - O(1), no copying
    #[inline]
    pub fn from_arc(values: CompactArc<[Value]>) -> Self {
        Self {
            storage: RowStorage::Shared(values),
        }
    }

    /// Create a row by combining two rows (for JOINs)
    /// Result uses Owned storage (optimal for intermediate results)
    #[inline]
    pub fn from_combined(left: &Row, right: &Row) -> Self {
        let total_len = left.len() + right.len();
        let mut values = CompactVec::with_capacity(total_len);
        // Use extend_clone to avoid Cloned iterator adapter overhead
        values.extend_clone(left.as_slice());
        values.extend_clone(right.as_slice());
        Self {
            storage: RowStorage::Owned(values),
        }
    }

    /// Combine two rows into this row buffer (for JOINs) - reuses allocation
    #[inline]
    pub fn combine_into(&mut self, left: &Row, right: &Row) {
        let total_len = left.len() + right.len();
        let vec = self.storage.make_mut();
        vec.clear();
        vec.reserve(total_len);
        // Use extend_clone to avoid Cloned iterator adapter overhead
        vec.extend_clone(left.as_slice());
        vec.extend_clone(right.as_slice());
    }

    /// Combine rows: clone left, move right (for JOINs) - reuses allocation
    #[inline]
    pub fn combine_into_clone_move(&mut self, left: &Row, right: Row) {
        let total_len = left.len() + right.len();
        let vec = self.storage.make_mut();
        vec.clear();
        vec.reserve(total_len);
        // Use extend_clone to avoid Cloned iterator adapter overhead
        vec.extend_clone(left.as_slice());
        // Move right values
        match right.storage {
            RowStorage::Owned(right_vec) => vec.extend(right_vec),
            RowStorage::Shared(arc) => vec.extend_clone(&arc),
        }
    }

    /// Combine rows: move both (for JOINs) - reuses allocation
    #[inline]
    pub fn combine_into_owned(&mut self, left: Row, right: Row) {
        let total_len = left.len() + right.len();
        let vec = self.storage.make_mut();
        vec.clear();
        vec.reserve(total_len);
        // Move left values - use extend_clone to avoid Cloned iterator overhead
        match left.storage {
            RowStorage::Owned(left_vec) => vec.extend(left_vec),
            RowStorage::Shared(arc) => vec.extend_clone(&arc),
        }
        // Move right values
        match right.storage {
            RowStorage::Owned(right_vec) => vec.extend(right_vec),
            RowStorage::Shared(arc) => vec.extend_clone(&arc),
        }
    }

    /// Combine rows: clone left, move right (for JOINs)
    #[inline]
    pub fn from_combined_clone_move(left: &Row, right: Row) -> Self {
        let total_len = left.len() + right.len();
        let mut values = CompactVec::with_capacity(total_len);
        // Use extend_clone to avoid Cloned iterator adapter overhead
        values.extend_clone(left.as_slice());
        match right.storage {
            RowStorage::Owned(right_vec) => values.extend(right_vec),
            RowStorage::Shared(arc) => values.extend_clone(&arc),
        }
        Self {
            storage: RowStorage::Owned(values),
        }
    }

    /// Combine two owned rows (for JOINs) - moves values without cloning
    #[inline]
    pub fn from_combined_owned(left: Row, right: Row) -> Self {
        // Fast path: both Owned
        match (left.storage, right.storage) {
            (RowStorage::Owned(mut left_vec), RowStorage::Owned(right_vec)) => {
                left_vec.reserve(right_vec.len());
                left_vec.extend(right_vec);
                Self {
                    storage: RowStorage::Owned(left_vec),
                }
            }
            (left_storage, right_storage) => {
                let left_len = left_storage.len();
                let right_len = right_storage.len();
                let mut values = CompactVec::with_capacity(left_len + right_len);
                // Use extend_clone to avoid Cloned iterator adapter overhead
                match left_storage {
                    RowStorage::Owned(v) => values.extend(v),
                    RowStorage::Shared(a) => values.extend_clone(&a),
                }
                match right_storage {
                    RowStorage::Owned(v) => values.extend(v),
                    RowStorage::Shared(a) => values.extend_clone(&a),
                }
                Self {
                    storage: RowStorage::Owned(values),
                }
            }
        }
    }

    /// Create a row with null values for a given schema
    #[inline]
    pub fn null_row(schema: &Schema) -> Self {
        let values: CompactVec<Value> = schema
            .columns
            .iter()
            .map(|col| Value::null(col.data_type))
            .collect();
        Self {
            storage: RowStorage::Owned(values),
        }
    }

    /// Get the number of values in the row
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if the row is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get a value by index
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.storage.get(index)
    }

    /// Get a mutable value by index (triggers copy-on-write if shared)
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.storage.make_mut().get_mut(index)
    }

    /// Set a value at the given index (triggers copy-on-write if shared)
    pub fn set(&mut self, index: usize, value: Value) -> Result<()> {
        let vec = self.storage.make_mut();
        if index >= vec.len() {
            return Err(Error::Internal {
                message: format!("row index {} out of bounds (len={})", index, vec.len()),
            });
        }
        vec[index] = value;
        Ok(())
    }

    /// Push a value to the end of the row
    #[inline]
    pub fn push(&mut self, value: Value) {
        self.storage.make_mut().push(value);
    }

    /// Pop a value from the end of the row
    #[inline]
    pub fn pop(&mut self) -> Option<Value> {
        self.storage.make_mut().pop()
    }

    /// Truncate the row to a specific length
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.storage.make_mut().truncate(len);
    }

    /// Clear the row values while keeping allocated capacity
    #[inline]
    pub fn clear(&mut self) {
        self.storage.make_mut().clear();
    }

    /// Take the values from this row, returning them in a new Row.
    /// The original row is cleared but keeps its allocated capacity.
    #[inline]
    pub fn take_and_clear(&mut self) -> Row {
        match &mut self.storage {
            RowStorage::Owned(vec) => {
                let cap = vec.capacity();
                let values = std::mem::replace(vec, CompactVec::with_capacity(cap));
                Row {
                    storage: RowStorage::Owned(values),
                }
            }
            RowStorage::Shared(arc) => {
                let result = Row {
                    storage: RowStorage::Shared(arc.clone()),
                };
                *self = Row::new();
                result
            }
        }
    }

    /// Reserve capacity for at least `additional` more values
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.storage.make_mut().reserve(additional);
    }

    /// Extend the row with values from a slice
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[Value]) {
        self.storage.make_mut().extend_clone(other);
    }

    /// Extend a CompactVec with this row's values, consuming self.
    ///
    /// OPTIMIZATION: This avoids the intermediate Vec allocation that would occur
    /// with `target.extend(row)` which goes through `Row::into_iter()`.
    /// - Owned storage: directly extends from CompactVec (moves values)
    /// - Shared storage: clones values from Arc slice
    #[inline]
    pub fn extend_into_compact_vec(self, target: &mut CompactVec<Value>) {
        match self.storage {
            RowStorage::Owned(vec) => target.extend(vec),
            RowStorage::Shared(arc) => target.extend_clone(&arc),
        }
    }

    /// Get an iterator over the values
    #[inline(always)]
    pub fn iter(&self) -> RowIter<'_> {
        RowIter {
            inner: match &self.storage {
                RowStorage::Shared(arc) => arc.iter(),
                RowStorage::Owned(vec) => vec.iter(),
            },
        }
    }

    /// Get a mutable iterator over the values (triggers copy-on-write if shared)
    #[inline]
    pub fn iter_mut(&mut self) -> RowIterMut<'_> {
        RowIterMut {
            inner: self.storage.make_mut().iter_mut(),
        }
    }

    /// Get the underlying vector of values, consuming the row
    #[inline]
    pub fn into_values(self) -> Vec<Value> {
        self.storage.into_vec()
    }

    /// Extract the first value, consuming the row
    #[inline]
    pub fn take_first_value(self) -> Option<Value> {
        match self.storage {
            RowStorage::Owned(mut vec) => {
                if vec.is_empty() {
                    None
                } else {
                    Some(vec.swap_remove(0))
                }
            }
            RowStorage::Shared(arc) => arc.first().cloned(),
        }
    }

    /// Check if storage is shared (Arc-wrapped)
    #[inline]
    pub fn is_shared(&self) -> bool {
        matches!(self.storage, RowStorage::Shared(_))
    }

    /// Check if storage is owned
    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self.storage, RowStorage::Owned(_))
    }

    /// Convert Row to CompactArc<[Value]>, consuming self
    /// - Shared: returns the CompactArc directly (O(1))
    /// - Owned: creates new Arc (O(n))
    #[inline]
    pub fn into_arc(self) -> CompactArc<[Value]> {
        match self.storage {
            RowStorage::Shared(arc) => arc,
            // Use into_vec() to avoid cloning each Value - O(1) ptr transfer
            RowStorage::Owned(vec) => CompactArc::from_vec(vec.into_vec()),
        }
    }

    /// Convert to Shared storage for O(1) clone.
    ///
    /// Use this when the row will be cloned multiple times (e.g., hash join build side).
    /// After conversion, `row.clone()` is just an Arc increment.
    #[inline]
    pub fn into_shared(self) -> Self {
        match self.storage {
            RowStorage::Shared(_) => self, // Already shared
            RowStorage::Owned(vec) => Self {
                // Use into_vec() to avoid cloning each Value - O(1) ptr transfer
                storage: RowStorage::Shared(CompactArc::from_vec(vec.into_vec())),
            },
        }
    }

    /// Get CompactArc<[Value]> reference if shared, None if owned
    #[inline]
    pub fn as_arc(&self) -> Option<&CompactArc<[Value]>> {
        match &self.storage {
            RowStorage::Shared(arc) => Some(arc),
            RowStorage::Owned(_) => None,
        }
    }

    /// Get slice of values
    #[inline]
    pub fn as_slice(&self) -> &[Value] {
        match &self.storage {
            RowStorage::Shared(arc) => arc,
            RowStorage::Owned(vec) => vec.as_slice(),
        }
    }

    /// Extract specific columns by their indices
    #[inline]
    pub fn select_columns(&self, indices: &[usize]) -> Result<Row> {
        let len = self.len();
        let mut values = CompactVec::with_capacity(indices.len());
        for &idx in indices {
            match self.storage.get(idx) {
                Some(val) => values.push(val.clone()),
                None => {
                    return Err(Error::Internal {
                        message: format!("column index {} out of bounds (len={})", idx, len),
                    })
                }
            }
        }
        Ok(Row::from_compact_vec(values))
    }

    /// Take specific columns by their indices, consuming the row
    /// Detects prefix projections (0, 1, 2, ..., n-1) and truncates in-place.
    #[inline]
    pub fn take_columns(self, indices: &[usize]) -> Row {
        // Fast path: check if indices form a prefix (0, 1, 2, ..., n-1)
        let is_prefix = !indices.is_empty()
            && indices.len() <= self.len()
            && indices.iter().enumerate().all(|(i, &idx)| i == idx);

        if is_prefix {
            match self.storage {
                RowStorage::Owned(mut vec) => {
                    vec.truncate(indices.len());
                    return Row {
                        storage: RowStorage::Owned(vec),
                    };
                }
                RowStorage::Shared(ref arc) if indices.len() == arc.len() => {
                    return self;
                }
                RowStorage::Shared(arc) => {
                    // Prefix selection from Shared
                    let values: CompactVec<Value> = arc[..indices.len()].iter().cloned().collect();
                    return Row {
                        storage: RowStorage::Owned(values),
                    };
                }
            }
        }

        // General case: select specific columns
        let mut values = CompactVec::with_capacity(indices.len());
        let slice = self.as_slice();
        for &idx in indices {
            if idx < slice.len() {
                values.push(slice[idx].clone());
            } else {
                values.push(Value::null_unknown());
            }
        }
        Row {
            storage: RowStorage::Owned(values),
        }
    }

    /// Validate the row against a schema
    pub fn validate(&self, schema: &Schema) -> Result<()> {
        let len = self.len();

        // Check column count
        if len != schema.columns.len() {
            return Err(Error::table_columns_not_match(schema.columns.len(), len));
        }

        // Check each value
        for (i, (value, col)) in self.iter().zip(schema.columns.iter()).enumerate() {
            // Check nullability
            if value.is_null() && !col.nullable && !col.primary_key {
                return Err(Error::not_null_constraint(&col.name));
            }

            // Check type compatibility (skip for null values)
            if !value.is_null() {
                let value_type = value.data_type();
                if value_type != col.data_type {
                    // Allow some implicit conversions
                    let compatible = matches!(
                        (value_type, col.data_type),
                        (DataType::Integer, DataType::Float) | (DataType::Float, DataType::Integer)
                    );
                    if !compatible {
                        return Err(Error::type_conversion(
                            format!("column {} at index {}: {:?}", col.name, i, value_type),
                            format!("{:?}", col.data_type),
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Clone the row, selecting only the specified column indices
    #[inline]
    pub fn clone_subset(&self, indices: &[usize]) -> Row {
        let mut values = CompactVec::with_capacity(indices.len());
        let slice = self.as_slice();
        for &i in indices {
            if let Some(v) = slice.get(i) {
                values.push(v.clone());
            }
        }
        Row::from_compact_vec(values)
    }

    /// Concatenate two rows
    pub fn concat(&self, other: &Row) -> Row {
        let total_len = self.len() + other.len();
        let mut values = CompactVec::with_capacity(total_len);
        values.extend(self.iter().cloned());
        values.extend(other.iter().cloned());
        Row::from_compact_vec(values)
    }

    /// Create a row by repeating a value
    pub fn repeat(value: Value, count: usize) -> Row {
        let mut values = CompactVec::with_capacity(count);
        for _ in 0..count {
            values.push(value.clone());
        }
        Row::from_compact_vec(values)
    }

    // === Compatibility methods for gradual migration ===
    // These methods maintain API compatibility with code expecting CompactArc

    /// Alias for is_owned (backwards compatibility)
    #[inline]
    pub fn is_inline(&self) -> bool {
        self.is_owned()
    }

    /// Clear and use as inline storage (backwards compatibility)
    #[inline]
    pub fn clear_inline(&mut self) {
        self.clear();
    }

    /// Push to inline storage (backwards compatibility)
    #[inline]
    pub fn push_inline(&mut self, value: Value) {
        self.push(value);
    }

    /// Reserve inline capacity (backwards compatibility)
    #[inline]
    pub fn reserve_inline(&mut self, capacity: usize) {
        let vec = self.storage.make_mut();
        if vec.capacity() < capacity {
            vec.reserve(capacity - vec.len());
        }
    }

    /// Refill with values (backwards compatibility)
    #[inline]
    pub fn refill_inline<I: Iterator<Item = Value>>(&mut self, values: I) {
        let vec = self.storage.make_mut();
        vec.clear();
        vec.extend(values);
    }
}

// Implement Index for convenient access
impl Index<usize> for Row {
    type Output = Value;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.storage.get(index).expect("row index out of bounds")
    }
}

// Implement FromIterator for collecting values into a row
impl FromIterator<Value> for Row {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        Row::from_compact_vec(iter.into_iter().collect())
    }
}

// Implement IntoIterator for consuming iteration
impl IntoIterator for Row {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.storage.into_vec().into_iter()
    }
}

impl<'a> IntoIterator for &'a Row {
    type Item = &'a Value;
    type IntoIter = RowIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl From<Vec<Value>> for Row {
    fn from(values: Vec<Value>) -> Self {
        Row::from_values(values)
    }
}

impl From<CompactArc<[Value]>> for Row {
    fn from(values: CompactArc<[Value]>) -> Self {
        Row::from_arc(values)
    }
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, value) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", value)?;
        }
        write!(f, ")")
    }
}

/// Macro for creating rows conveniently
#[macro_export]
macro_rules! row {
    () => {
        $crate::core::Row::new()
    };
    ($($value:expr),+ $(,)?) => {
        $crate::core::Row::from_values(vec![$($crate::core::Value::from($value)),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::CompactArc;
    use crate::core::schema::SchemaBuilder;

    fn create_test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add_nullable("email", DataType::Text)
            .build()
    }

    #[test]
    fn test_row_creation() {
        let row = Row::new();
        assert!(row.is_empty());
        assert_eq!(row.len(), 0);

        let row = Row::with_capacity(10);
        assert!(row.is_empty());
    }

    #[test]
    fn test_row_from_values() {
        let values = vec![
            Value::integer(1),
            Value::text("hello"),
            Value::null(DataType::Text),
        ];
        let row = Row::from_values(values);
        assert_eq!(row.len(), 3);
        assert!(row.is_owned());
    }

    #[test]
    fn test_row_from_arc() {
        let values: CompactArc<[Value]> =
            CompactArc::from(vec![Value::integer(1), Value::text("hello")]);
        let row = Row::from_arc(values);
        assert_eq!(row.len(), 2);
        assert!(row.is_shared());

        // Clone should be O(1) - just Arc increment
        let row2 = row.clone();
        assert_eq!(row2.len(), 2);
        assert_eq!(row, row2);
        assert!(row2.is_shared());
    }

    #[test]
    fn test_row_push_pop() {
        let mut row = Row::new();
        row.push(Value::integer(1));
        row.push(Value::text("hello"));

        assert_eq!(row.len(), 2);

        let popped = row.pop();
        assert_eq!(popped, Some(Value::text("hello")));
        assert_eq!(row.len(), 1);
    }

    #[test]
    fn test_row_copy_on_write() {
        // Create shared row
        let values: CompactArc<[Value]> =
            CompactArc::from(vec![Value::integer(1), Value::text("hello")]);
        let mut row = Row::from_arc(values);
        assert!(row.is_shared());

        // Mutation should trigger copy-on-write
        row.push(Value::integer(2));
        assert_eq!(row.len(), 3);
        assert!(row.is_owned()); // Now owned after mutation
    }

    #[test]
    fn test_row_get_set() {
        let mut row = Row::from_values(vec![Value::integer(1), Value::text("hello")]);

        assert_eq!(row.get(0), Some(&Value::integer(1)));
        assert_eq!(row.get(1), Some(&Value::text("hello")));
        assert_eq!(row.get(2), None);

        row.set(1, Value::text("world")).unwrap();
        assert_eq!(row.get(1), Some(&Value::text("world")));

        assert!(row.set(10, Value::integer(0)).is_err());
    }

    #[test]
    fn test_row_index() {
        let row = Row::from_values(vec![Value::integer(1), Value::text("hello")]);

        assert_eq!(row[0], Value::integer(1));
        assert_eq!(row[1], Value::text("hello"));
    }

    #[test]
    fn test_row_iteration() {
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::integer(2),
            Value::integer(3),
        ]);

        let sum: i64 = row.iter().filter_map(|v| v.as_int64()).sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_row_select_columns() {
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("hello"),
            Value::float(3.5),
            Value::boolean(true),
        ]);

        let selected = row.select_columns(&[0, 2]).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], Value::integer(1));
        assert_eq!(selected[1], Value::float(3.5));

        assert!(row.select_columns(&[0, 10]).is_err());
    }

    #[test]
    fn test_row_validate() {
        let schema = create_test_schema();

        // Valid row
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::null(DataType::Text),
        ]);
        assert!(row.validate(&schema).is_ok());

        // Wrong column count
        let row = Row::from_values(vec![Value::integer(1)]);
        assert!(row.validate(&schema).is_err());

        // Not null constraint violation
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::null(DataType::Text), // name is not nullable
            Value::null(DataType::Text),
        ]);
        let err = row.validate(&schema).unwrap_err();
        assert!(matches!(err, Error::NotNullConstraint { .. }));
    }

    #[test]
    fn test_row_null_row() {
        let schema = create_test_schema();
        let row = Row::null_row(&schema);

        assert_eq!(row.len(), 3);
        assert!(row[0].is_null());
        assert!(row[1].is_null());
        assert!(row[2].is_null());
    }

    #[test]
    fn test_row_concat() {
        let row1 = Row::from_values(vec![Value::integer(1), Value::integer(2)]);
        let row2 = Row::from_values(vec![Value::integer(3), Value::integer(4)]);

        let combined = row1.concat(&row2);
        assert_eq!(combined.len(), 4);
        assert_eq!(combined[0], Value::integer(1));
        assert_eq!(combined[3], Value::integer(4));
    }

    #[test]
    fn test_row_repeat() {
        let row = Row::repeat(Value::integer(0), 5);
        assert_eq!(row.len(), 5);
        for v in row.iter() {
            assert_eq!(*v, Value::integer(0));
        }
    }

    #[test]
    fn test_row_from_iterator() {
        let row: Row = vec![Value::integer(1), Value::integer(2), Value::integer(3)]
            .into_iter()
            .collect();
        assert_eq!(row.len(), 3);
    }

    #[test]
    fn test_row_display() {
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("hello"),
            Value::null(DataType::Text),
        ]);
        assert_eq!(row.to_string(), "(1, hello, NULL)");

        let empty = Row::new();
        assert_eq!(empty.to_string(), "()");
    }

    #[test]
    fn test_row_clone_subset() {
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("hello"),
            Value::float(3.5),
        ]);

        let subset = row.clone_subset(&[2, 0]);
        assert_eq!(subset.len(), 2);
        assert_eq!(subset[0], Value::float(3.5));
        assert_eq!(subset[1], Value::integer(1));
    }

    #[test]
    fn test_row_equality() {
        let row1 = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let row2 = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let row3 = Row::from_values(vec![Value::integer(1), Value::text("world")]);

        assert_eq!(row1, row2);
        assert_ne!(row1, row3);
    }

    #[test]
    fn test_row_into_values() {
        let row = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let values = row.into_values();

        assert_eq!(values.len(), 2);
        assert_eq!(values[0], Value::integer(1));
    }

    #[test]
    fn test_row_into_arc() {
        let row = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let arc = row.into_arc();
        assert_eq!(arc.len(), 2);

        // From shared - should be O(1)
        let row2 = Row::from_arc(CompactArc::clone(&arc));
        let arc2 = row2.into_arc();
        assert!(CompactArc::ptr_eq(&arc, &arc2));
    }

    #[test]
    fn test_row_combined() {
        let left = Row::from_values(vec![Value::integer(1), Value::integer(2)]);
        let right = Row::from_values(vec![Value::integer(3), Value::integer(4)]);

        let combined = Row::from_combined(&left, &right);
        assert_eq!(combined.len(), 4);
        assert_eq!(combined[0], Value::integer(1));
        assert_eq!(combined[3], Value::integer(4));
    }

    #[test]
    fn test_shared_owned_equality() {
        let owned = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let shared = Row::from_arc(CompactArc::from(vec![
            Value::integer(1),
            Value::text("hello"),
        ]));

        assert_eq!(owned, shared);
    }
}
