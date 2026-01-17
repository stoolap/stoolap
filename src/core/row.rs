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

use std::fmt;
use std::ops::Index;
use std::sync::Arc;

use super::error::{Error, Result};
use super::schema::Schema;
use super::types::DataType;
use super::value::Value;
use super::value_interner::{intern_value, interned_null};
use crate::common::{CompactArc, CompactVec};

/// Internal storage for Row - hybrid approach for memory + performance
///
/// Three variants optimize for different use cases:
/// - `Shared`: Arena storage with O(1) clone (most common for storage reads)
/// - `Inline`: CompactVec<Value> for intermediate results (16 bytes, no Arc overhead)
/// - `Owned`: Arc-wrapped values for storage (shareable with indexes)
///
/// OPTIMIZATION: Shared is first for better branch prediction on read-heavy workloads
/// All three variants are 16 bytes for optimal move performance.
#[derive(Debug, Clone)]
enum RowStorage {
    /// Shared storage - O(1) clone, copy-on-write for mutation (checked first)
    Shared(Arc<[CompactArc<Value>]>),
    /// Inline storage - CompactVec<Value> for intermediate results (16 bytes, no Arc overhead)
    Inline(CompactVec<Value>),
    /// Owned storage - CompactVec<CompactArc<Value>>, supports sharing with indexes (16 bytes)
    Owned(CompactVec<CompactArc<Value>>),
}

impl Default for RowStorage {
    fn default() -> Self {
        RowStorage::Inline(CompactVec::new())
    }
}

impl PartialEq for RowStorage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Match once on storage types, then iterate directly (avoids per-element match)
        match (self, other) {
            (RowStorage::Inline(a), RowStorage::Inline(b)) => a.as_slice() == b.as_slice(),
            (RowStorage::Owned(a), RowStorage::Owned(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x.as_ref() == y.as_ref())
            }
            (RowStorage::Shared(a), RowStorage::Shared(b)) => {
                // Fast path: same Arc pointer
                Arc::ptr_eq(a, b)
                    || (a.len() == b.len()
                        && a.iter()
                            .zip(b.iter())
                            .all(|(x, y)| x.as_ref() == y.as_ref()))
            }
            // Mixed storage types - compare by value
            _ => {
                let a_len = self.len();
                let b_len = other.len();
                if a_len != b_len {
                    return false;
                }
                for i in 0..a_len {
                    match (self.get_value(i), other.get_value(i)) {
                        (Some(a), Some(b)) if a == b => continue,
                        _ => return false,
                    }
                }
                true
            }
        }
    }
}

impl RowStorage {
    /// Get a value by index (works for all storage types)
    /// OPTIMIZATION: Shared checked first (most common for storage reads)
    #[inline(always)]
    fn get_value(&self, index: usize) -> Option<&Value> {
        match self {
            RowStorage::Shared(a) => a.get(index).map(|arc| arc.as_ref()),
            RowStorage::Inline(v) => v.get(index),
            RowStorage::Owned(v) => v.get(index).map(|arc| arc.as_ref()),
        }
    }

    /// Check if this is inline storage
    #[inline]
    fn is_inline(&self) -> bool {
        matches!(self, RowStorage::Inline(_))
    }

    /// OPTIMIZATION: Shared checked first (most common for storage reads)
    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            RowStorage::Shared(a) => a.len(),
            RowStorage::Inline(v) => v.len(),
            RowStorage::Owned(v) => v.len(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get mutable access to inline Vec<Value>
    #[inline]
    fn make_mut_inline(&mut self) -> &mut CompactVec<Value> {
        match self {
            RowStorage::Inline(v) => v,
            RowStorage::Owned(arc_vec) => {
                // Convert CompactArc<Value> to Value - use try_unwrap to avoid cloning when refcount is 1
                let vec = std::mem::take(arc_vec);
                *self = RowStorage::Inline(
                    vec.into_iter()
                        .map(|arc| CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone()))
                        .collect(),
                );
                match self {
                    RowStorage::Inline(v) => v,
                    _ => unreachable!(),
                }
            }
            RowStorage::Shared(arc) => {
                *self = RowStorage::Inline(arc.iter().map(|a| (**a).clone()).collect());
                match self {
                    RowStorage::Inline(v) => v,
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Get mutable access to Arc CompactVec, converting if necessary (copy-on-write)
    #[inline]
    fn make_mut_arc(&mut self) -> &mut CompactVec<CompactArc<Value>> {
        match self {
            RowStorage::Inline(v) => {
                // Convert Value to CompactArc<Value> with interning
                *self =
                    RowStorage::Owned(std::mem::take(v).into_iter().map(intern_value).collect());
                match self {
                    RowStorage::Owned(v) => v,
                    _ => unreachable!(),
                }
            }
            RowStorage::Owned(v) => v,
            RowStorage::Shared(arc) => {
                // Copy-on-write: convert shared to owned (CompactArc::clone is cheap)
                *self = RowStorage::Owned(arc.iter().cloned().collect());
                match self {
                    RowStorage::Owned(v) => v,
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Convert to owned Vec<Value>, consuming self
    #[inline]
    fn into_vec(self) -> Vec<Value> {
        match self {
            RowStorage::Inline(v) => v.into_vec(),
            RowStorage::Owned(v) => {
                // Try to unwrap each Arc - if sole owner, move value; else clone
                v.into_iter()
                    .map(|arc| CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone()))
                    .collect()
            }
            RowStorage::Shared(arc) => arc.iter().map(|a| (**a).clone()).collect(),
        }
    }

    /// Convert to owned Vec<CompactArc<Value>>, consuming self
    #[inline]
    fn into_arc_vec(self) -> Vec<CompactArc<Value>> {
        match self {
            RowStorage::Inline(v) => v.into_iter().map(intern_value).collect(),
            RowStorage::Owned(v) => v.into_vec(),
            RowStorage::Shared(arc) => arc.to_vec(),
        }
    }

    /// Get mutable access to Arc CompactVec (alias for make_mut_arc for backwards compat)
    #[inline]
    fn make_mut(&mut self) -> &mut CompactVec<CompactArc<Value>> {
        self.make_mut_arc()
    }
}

/// A database row containing column values
///
/// Row provides methods for accessing and manipulating column values
/// while maintaining type safety and consistency with the schema.
///
/// Performance: Row uses Arc<[Value]> for O(1) cloning when created from
/// arena storage. Mutations trigger copy-on-write conversion to owned Vec.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Row {
    storage: RowStorage,
}

/// Iterator over row values - handles all storage types transparently
pub enum RowIter<'a> {
    Inline(std::slice::Iter<'a, Value>),
    Arc(std::slice::Iter<'a, CompactArc<Value>>),
}

impl<'a> Iterator for RowIter<'a> {
    type Item = &'a Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RowIter::Inline(iter) => iter.next(),
            RowIter::Arc(iter) => iter.next().map(|arc| arc.as_ref()),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            RowIter::Inline(iter) => iter.size_hint(),
            RowIter::Arc(iter) => iter.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for RowIter<'a> {
    fn len(&self) -> usize {
        match self {
            RowIter::Inline(iter) => iter.len(),
            RowIter::Arc(iter) => iter.len(),
        }
    }
}

impl<'a> DoubleEndedIterator for RowIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            RowIter::Inline(iter) => iter.next_back(),
            RowIter::Arc(iter) => iter.next_back().map(|arc| arc.as_ref()),
        }
    }
}

/// Mutable iterator over row values - converts to Arc storage first
pub struct RowIterMut<'a> {
    inner: std::slice::IterMut<'a, CompactArc<Value>>,
}

impl<'a> Iterator for RowIterMut<'a> {
    type Item = &'a mut Value;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(CompactArc::make_mut)
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
    /// Create a new empty row (uses Inline storage for efficiency)
    #[inline]
    pub fn new() -> Self {
        Self {
            storage: RowStorage::Inline(CompactVec::new()),
        }
    }

    /// Create a row with pre-allocated capacity (uses Inline storage)
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            storage: RowStorage::Inline(CompactVec::with_capacity(capacity)),
        }
    }

    /// Create a row from a vector of values - uses Inline storage (no Arc overhead)
    /// This is optimal for intermediate query results that don't need Arc sharing.
    #[inline]
    pub fn from_values(values: Vec<Value>) -> Self {
        Self {
            storage: RowStorage::Inline(CompactVec::from_vec(values)),
        }
    }

    /// Create a row from CompactVec<Value> - uses Inline storage directly.
    /// More efficient than from_values when you already have a CompactVec.
    #[inline]
    pub fn from_compact_vec(values: CompactVec<Value>) -> Self {
        Self {
            storage: RowStorage::Inline(values),
        }
    }

    /// Create a row from a vector of CompactArc<Value> (no wrapping needed)
    #[inline]
    pub fn from_arc_values(values: Vec<CompactArc<Value>>) -> Self {
        Self {
            storage: RowStorage::Owned(CompactVec::from_vec(values)),
        }
    }

    /// Create an empty row with Arc storage and pre-allocated capacity.
    /// Use this when you'll be calling `clear()` + `push_arc()` repeatedly.
    #[inline]
    pub fn with_arc_capacity(capacity: usize) -> Self {
        Self {
            storage: RowStorage::Owned(CompactVec::with_capacity(capacity)),
        }
    }

    /// Create a row by combining two rows (for JOINs)
    /// Result is always Inline storage (optimal for intermediate results).
    #[inline]
    pub fn from_combined(left: &Row, right: &Row) -> Self {
        let total_len = left.len() + right.len();
        let mut values = CompactVec::with_capacity(total_len);
        values.extend(left.iter().cloned());
        values.extend(right.iter().cloned());
        Self {
            storage: RowStorage::Inline(values),
        }
    }

    /// Combine two rows into this row buffer (for JOINs) - reuses allocation
    /// OPTIMIZATION: Clears and refills existing buffer, avoiding allocation per join result
    #[inline]
    pub fn combine_into(&mut self, left: &Row, right: &Row) {
        let total_len = left.len() + right.len();
        // Use inline storage for intermediate results
        let vec = self.storage.make_mut_inline();
        vec.clear();
        // After clear(), len=0, so reserve(n) ensures capacity >= n
        vec.reserve(total_len);
        vec.extend(left.iter().cloned());
        vec.extend(right.iter().cloned());
    }

    /// Combine rows into buffer: clone left, move right (for JOINs) - reuses allocation
    /// OPTIMIZATION: Reuses buffer allocation AND moves right values instead of cloning
    /// This is the most efficient combine for streaming joins where we own the inner row.
    #[inline]
    pub fn combine_into_clone_move(&mut self, left: &Row, right: Row) {
        let total_len = left.len() + right.len();
        let vec = self.storage.make_mut_inline();
        vec.clear();
        // After clear(), len=0, so reserve(n) ensures capacity >= n
        vec.reserve(total_len);
        // Clone left values
        vec.extend(left.iter().cloned());
        // Move right values - use try_unwrap to avoid cloning when refcount is 1
        match right.storage {
            RowStorage::Inline(right_vec) => vec.extend(right_vec),
            RowStorage::Owned(right_vec) => vec.extend(
                right_vec
                    .into_iter()
                    .map(|arc| CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())),
            ),
            RowStorage::Shared(arc) => vec.extend(arc.iter().map(|a| (**a).clone())),
        }
    }

    /// Combine rows into buffer: move both (for JOINs) - reuses allocation
    /// OPTIMIZATION: Reuses buffer AND moves values from both sides when possible
    /// Use when both outer and inner rows are owned and no longer needed.
    /// NOTE: This converts buffer to Inline storage. If you need Arc storage
    /// (e.g., for `take_from_buffer`), use `combine_into_arc` instead.
    #[inline]
    pub fn combine_into_owned(&mut self, left: Row, right: Row) {
        let total_len = left.len() + right.len();
        let vec = self.storage.make_mut_inline();
        vec.clear();
        // After clear(), len=0, so reserve(n) ensures capacity >= n
        vec.reserve(total_len);
        // Move based on storage type - use try_unwrap to avoid cloning when refcount is 1
        match left.storage {
            RowStorage::Inline(left_vec) => vec.extend(left_vec),
            RowStorage::Owned(left_vec) => vec.extend(
                left_vec
                    .into_iter()
                    .map(|arc| CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())),
            ),
            RowStorage::Shared(arc) => vec.extend(arc.iter().map(|a| (**a).clone())),
        }
        match right.storage {
            RowStorage::Inline(right_vec) => vec.extend(right_vec),
            RowStorage::Owned(right_vec) => vec.extend(
                right_vec
                    .into_iter()
                    .map(|arc| CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())),
            ),
            RowStorage::Shared(arc) => vec.extend(arc.iter().map(|a| (**a).clone())),
        }
    }

    /// Combine two rows into this buffer using Arc storage.
    /// OPTIMIZATION: Keeps buffer in Owned (Arc) storage to avoid type oscillation.
    /// Use this when the buffer will be consumed by `as_mut_arc_vec_with_capacity`.
    #[inline]
    pub fn combine_into_arc(&mut self, left: Row, right: Row) {
        let total_len = left.len() + right.len();
        let vec = self.storage.make_mut_arc();
        vec.clear();
        // After clear(), len=0, so reserve(n) ensures capacity >= n
        vec.reserve(total_len);
        // Extend with Arc values, moving when possible
        match left.storage {
            RowStorage::Inline(left_vec) => {
                vec.extend(left_vec.into_iter().map(intern_value));
            }
            RowStorage::Owned(left_vec) => {
                vec.extend(left_vec);
            }
            RowStorage::Shared(arc) => {
                vec.extend(arc.iter().cloned());
            }
        }
        match right.storage {
            RowStorage::Inline(right_vec) => {
                vec.extend(right_vec.into_iter().map(intern_value));
            }
            RowStorage::Owned(right_vec) => {
                vec.extend(right_vec);
            }
            RowStorage::Shared(arc) => {
                vec.extend(arc.iter().cloned());
            }
        }
    }

    /// Combine rows: clone left values, move right values (for JOINs)
    /// OPTIMIZATION: Moves right values instead of cloning them. Uses Inline for results.
    #[inline]
    pub fn from_combined_clone_move(left: &Row, right: Row) -> Self {
        let total_len = left.len() + right.len();
        let mut values: CompactVec<Value> = CompactVec::with_capacity(total_len);
        // Clone left values
        values.extend(left.iter().cloned());
        // Move right values - use try_unwrap to avoid cloning when refcount is 1
        match right.storage {
            RowStorage::Inline(right_vec) => values.extend(right_vec),
            RowStorage::Owned(right_vec) => values.extend(
                right_vec
                    .into_iter()
                    .map(|arc| CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())),
            ),
            RowStorage::Shared(arc) => values.extend(arc.iter().map(|a| (**a).clone())),
        }
        Self {
            storage: RowStorage::Inline(values),
        }
    }

    /// Create a row by combining two owned rows (for JOINs) - moves values without cloning
    /// OPTIMIZATION: Takes ownership and moves values. Uses Inline for results.
    #[inline]
    pub fn from_combined_owned(left: Row, right: Row) -> Self {
        let total_len = left.len() + right.len();
        // Fast path: both Inline -> keep Inline (optimal)
        match (left.storage, right.storage) {
            (RowStorage::Inline(mut left_vec), RowStorage::Inline(right_vec)) => {
                left_vec.reserve(right_vec.len());
                left_vec.extend(right_vec);
                Self {
                    storage: RowStorage::Inline(left_vec),
                }
            }
            (left_storage, right_storage) => {
                // Mixed storage: convert to values - use try_unwrap to avoid cloning when refcount is 1
                let mut values: CompactVec<Value> = CompactVec::with_capacity(total_len);
                match left_storage {
                    RowStorage::Inline(v) => values.extend(v),
                    RowStorage::Owned(v) => {
                        values.extend(v.into_iter().map(|arc| {
                            CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
                        }))
                    }
                    RowStorage::Shared(a) => values.extend(a.iter().map(|arc| (**arc).clone())),
                }
                match right_storage {
                    RowStorage::Inline(v) => values.extend(v),
                    RowStorage::Owned(v) => {
                        values.extend(v.into_iter().map(|arc| {
                            CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
                        }))
                    }
                    RowStorage::Shared(a) => values.extend(a.iter().map(|arc| (**arc).clone())),
                }
                Self {
                    storage: RowStorage::Inline(values),
                }
            }
        }
    }

    /// Create a row from an Arc slice of values (wraps each in Arc)
    #[inline]
    pub fn from_arc(values: Arc<[Value]>) -> Self {
        Self {
            storage: RowStorage::Owned(values.iter().map(|v| intern_value(v.clone())).collect()),
        }
    }

    /// Create a row from an Arc slice of CompactArc<Value> - O(1) clone
    #[inline]
    pub fn from_arc_slice(values: Arc<[CompactArc<Value>]>) -> Self {
        Self {
            storage: RowStorage::Shared(values),
        }
    }

    /// Convert Inline/Shared storage to Owned in place.
    /// Call this before operations that need Arc access (like as_arc_slice).
    /// - Inline: wraps each value in Arc (with interning for common values)
    /// - Shared: clones Arc references to owned Vec
    /// - No-op if already Owned.
    #[inline]
    pub fn ensure_owned(&mut self) {
        match std::mem::take(&mut self.storage) {
            RowStorage::Inline(v) => {
                self.storage = RowStorage::Owned(v.into_iter().map(intern_value).collect());
            }
            RowStorage::Shared(arc) => {
                self.storage = RowStorage::Owned(arc.iter().cloned().collect());
            }
            owned @ RowStorage::Owned(_) => {
                self.storage = owned; // Put it back
            }
        }
    }

    /// Check if storage is owned (vs shared Arc)
    #[inline]
    pub fn is_owned(&self) -> bool {
        matches!(self.storage, RowStorage::Owned(_))
    }

    /// Create a row with null values for a given schema
    #[inline]
    pub fn null_row(schema: &Schema) -> Self {
        let values: CompactVec<CompactArc<Value>> = schema
            .columns
            .iter()
            .map(|col| interned_null(col.data_type))
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
        self.storage.get_value(index)
    }

    /// Get an CompactArc<Value> by index - for indexes to share references
    /// Note: For Inline storage, this uses value interning for common values
    #[inline]
    pub fn get_arc(&self, index: usize) -> Option<CompactArc<Value>> {
        match &self.storage {
            RowStorage::Inline(v) => v.get(index).map(|val| intern_value(val.clone())),
            RowStorage::Owned(v) => v.get(index).cloned(),
            RowStorage::Shared(a) => a.get(index).cloned(),
        }
    }

    /// Get a mutable value by index (triggers copy-on-write if shared)
    /// Note: This may clone the value if it has multiple Arc references
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.storage
            .make_mut()
            .get_mut(index)
            .map(CompactArc::make_mut)
    }

    /// Set a value at the given index (triggers copy-on-write if shared)
    pub fn set(&mut self, index: usize, value: Value) -> Result<()> {
        let vec = self.storage.make_mut();
        if index >= vec.len() {
            return Err(Error::Internal {
                message: format!("row index {} out of bounds (len={})", index, vec.len()),
            });
        }
        vec[index] = intern_value(value);
        Ok(())
    }

    /// Push a value to the end of the row (triggers copy-on-write if shared)
    #[inline]
    pub fn push(&mut self, value: Value) {
        self.storage.make_mut().push(intern_value(value));
    }

    /// Push an CompactArc<Value> to the end of the row (no wrapping needed)
    #[inline]
    pub fn push_arc(&mut self, value: CompactArc<Value>) {
        self.storage.make_mut().push(value);
    }

    /// Pop a value from the end of the row (triggers copy-on-write if shared)
    #[inline]
    pub fn pop(&mut self) -> Option<Value> {
        self.storage
            .make_mut()
            .pop()
            .map(|arc| CompactArc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
    }

    /// Truncate the row to a specific length (triggers copy-on-write if shared)
    ///
    /// Used for schema evolution when columns are dropped
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.storage.make_mut().truncate(len);
    }

    /// Clear the row values while keeping the allocated capacity
    /// This allows reusing the Vec allocation for streaming operations
    #[inline]
    pub fn clear(&mut self) {
        self.storage.make_mut().clear();
    }

    /// Clear the row's inline storage, keeping allocated capacity.
    /// Uses Inline storage (CompactVec<Value>) without Arc overhead.
    /// This is optimal for streaming operations where rows are reused.
    #[inline]
    pub fn clear_inline(&mut self) {
        self.storage.make_mut_inline().clear();
    }

    /// Push a value to inline storage without Arc wrapping.
    /// More efficient than push() when Arc sharing is not needed.
    #[inline]
    pub fn push_inline(&mut self, value: Value) {
        self.storage.make_mut_inline().push(value);
    }

    /// Clear and refill the row with new values, reusing the buffer.
    /// This is the most efficient way to reuse a Row for streaming results.
    /// Zero allocations if the new data fits in existing capacity.
    #[inline]
    pub fn refill_inline<I: Iterator<Item = Value>>(&mut self, values: I) {
        let vec = self.storage.make_mut_inline();
        vec.clear();
        vec.extend(values);
    }

    /// Reserve capacity in inline storage
    #[inline]
    pub fn reserve_inline(&mut self, capacity: usize) {
        let vec = self.storage.make_mut_inline();
        if vec.capacity() < capacity {
            vec.reserve(capacity - vec.len());
        }
    }

    /// Extend the row with values from a slice (wraps each in Arc)
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[Value]) {
        let vec = self.storage.make_mut();
        vec.extend(other.iter().map(|v| intern_value(v.clone())));
    }

    /// Extend the row with CompactArc<Value> references (no wrapping needed)
    #[inline]
    pub fn extend_from_arc_slice(&mut self, other: &[CompactArc<Value>]) {
        let vec = self.storage.make_mut();
        vec.extend(other.iter().cloned());
    }

    /// Reserve capacity for at least `additional` more values
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.storage.make_mut().reserve(additional);
    }

    /// Get an iterator over the values (handles all storage types)
    /// OPTIMIZATION: Shared checked first (most common for storage reads)
    #[inline(always)]
    pub fn iter(&self) -> RowIter<'_> {
        match &self.storage {
            RowStorage::Shared(a) => RowIter::Arc(a.iter()),
            RowStorage::Inline(v) => RowIter::Inline(v.iter()),
            RowStorage::Owned(v) => RowIter::Arc(v.iter()),
        }
    }

    /// Get an iterator over CompactArc<Value> references (for efficient cloning)
    /// PANICS on Inline storage - use ensure_owned() first if Arc access is needed.
    #[inline]
    pub fn iter_arc(&self) -> std::slice::Iter<'_, CompactArc<Value>> {
        match &self.storage {
            RowStorage::Inline(_) => {
                panic!("iter_arc() called on Inline storage - call ensure_owned() first")
            }
            RowStorage::Owned(v) => v.iter(),
            RowStorage::Shared(a) => a.iter(),
        }
    }

    /// Get a mutable iterator over the values (triggers copy-on-write if shared)
    /// Note: Each dereference may clone if Arc has multiple references
    #[inline]
    pub fn iter_mut(&mut self) -> RowIterMut<'_> {
        RowIterMut {
            inner: self.storage.make_mut().iter_mut(),
        }
    }

    /// Get the underlying vector of values
    #[inline]
    pub fn into_values(self) -> Vec<Value> {
        self.storage.into_vec()
    }

    /// Extract the first value by move, consuming the row.
    /// More efficient than into_values().swap_remove(0) when only first value is needed.
    #[inline]
    pub fn take_first_value(self) -> Option<Value> {
        match self.storage {
            RowStorage::Inline(mut v) => {
                if v.is_empty() {
                    None
                } else {
                    Some(v.swap_remove(0))
                }
            }
            RowStorage::Owned(mut v) => {
                if v.is_empty() {
                    None
                } else {
                    let arc = v.swap_remove(0);
                    Some(CompactArc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone()))
                }
            }
            RowStorage::Shared(arc) => arc.first().map(|a| (**a).clone()),
        }
    }

    /// Get the underlying vector of CompactArc<Value>
    #[inline]
    pub fn into_arc_values(self) -> Vec<CompactArc<Value>> {
        self.storage.into_arc_vec()
    }

    /// Check if storage is inline (raw values, not Arc-wrapped)
    #[inline]
    pub fn is_inline(&self) -> bool {
        self.storage.is_inline()
    }

    /// Try to get Arc slice - returns None for Inline storage
    #[inline]
    pub fn try_as_arc_slice(&self) -> Option<&[CompactArc<Value>]> {
        match &self.storage {
            RowStorage::Inline(_) => None,
            RowStorage::Owned(v) => Some(v),
            RowStorage::Shared(a) => Some(a),
        }
    }

    /// Get mutable CompactVec<CompactArc<Value>> with guaranteed capacity, for buffer reuse patterns.
    /// OPTIMIZATION: Ensures capacity without reallocation if already sufficient.
    #[inline]
    pub fn as_mut_arc_vec_with_capacity(
        &mut self,
        capacity: usize,
    ) -> &mut CompactVec<CompactArc<Value>> {
        let vec = self.storage.make_mut();
        if vec.capacity() < capacity {
            vec.reserve(capacity - vec.len());
        }
        vec
    }

    /// Convert Row to Arc<[CompactArc<Value>]>, consuming self - efficient for arena storage
    /// - Shared: returns the Arc directly (O(1))
    /// - Owned: converts to Arc (O(n) to create the Arc slice)
    /// - Inline: wraps each value in Arc then creates Arc slice
    ///
    /// OPTIMIZATION: Shared checked first (most common path)
    #[inline(always)]
    pub fn into_arc(self) -> Arc<[CompactArc<Value>]> {
        match self.storage {
            RowStorage::Shared(arc) => arc,
            RowStorage::Inline(vec) => {
                let arc_vec: Vec<CompactArc<Value>> = vec.into_iter().map(intern_value).collect();
                Arc::from(arc_vec.into_boxed_slice())
            }
            RowStorage::Owned(vec) => Arc::from(vec.into_boxed_slice()),
        }
    }

    /// Get a mutable reference to the underlying CompactArc<Value> CompactVec (triggers copy-on-write)
    #[inline]
    pub fn as_mut_arc_vec(&mut self) -> &mut CompactVec<CompactArc<Value>> {
        self.storage.make_mut()
    }

    /// Extract specific columns by their indices
    #[inline]
    pub fn select_columns(&self, indices: &[usize]) -> Result<Row> {
        let len = self.len();
        match &self.storage {
            RowStorage::Inline(vec) => {
                // Keep as Inline for intermediate results
                let mut values: CompactVec<Value> = CompactVec::with_capacity(indices.len());
                for &idx in indices {
                    match vec.get(idx) {
                        Some(val) => values.push(val.clone()),
                        None => {
                            return Err(Error::Internal {
                                message: format!(
                                    "column index {} out of bounds (len={})",
                                    idx, len
                                ),
                            })
                        }
                    }
                }
                Ok(Row::from_compact_vec(values))
            }
            RowStorage::Owned(vec) => {
                let mut values = CompactVec::with_capacity(indices.len());
                for &idx in indices {
                    match vec.get(idx) {
                        Some(arc) => values.push(CompactArc::clone(arc)),
                        None => {
                            return Err(Error::Internal {
                                message: format!(
                                    "column index {} out of bounds (len={})",
                                    idx, len
                                ),
                            })
                        }
                    }
                }
                Ok(Row {
                    storage: RowStorage::Owned(values),
                })
            }
            RowStorage::Shared(arc) => {
                let mut values = CompactVec::with_capacity(indices.len());
                for &idx in indices {
                    match arc.get(idx) {
                        Some(a) => values.push(CompactArc::clone(a)),
                        None => {
                            return Err(Error::Internal {
                                message: format!(
                                    "column index {} out of bounds (len={})",
                                    idx, len
                                ),
                            })
                        }
                    }
                }
                Ok(Row {
                    storage: RowStorage::Owned(values),
                })
            }
        }
    }

    /// Take specific columns by their indices, consuming the row
    /// OPTIMIZATION: For Inline/Owned storage, moves values without cloning.
    /// For Shared (Arc) storage, only clones the Arc (cheap).
    /// OPTIMIZATION: Detects prefix projections (0, 1, 2, ..., n-1) and truncates in-place.
    #[inline]
    pub fn take_columns(self, indices: &[usize]) -> Row {
        // Fast path: check if indices form a prefix (0, 1, 2, ..., n-1)
        // This is common for "SELECT col1, col2, col3 FROM ..." when columns are in order
        let is_prefix = !indices.is_empty()
            && indices.len() <= self.len()
            && indices.iter().enumerate().all(|(i, &idx)| i == idx);

        if is_prefix {
            match self.storage {
                RowStorage::Inline(mut vec) => {
                    vec.truncate(indices.len());
                    return Row {
                        storage: RowStorage::Inline(vec),
                    };
                }
                RowStorage::Owned(mut vec) => {
                    vec.truncate(indices.len());
                    return Row {
                        storage: RowStorage::Owned(vec),
                    };
                }
                RowStorage::Shared(ref arc) if indices.len() == arc.len() => {
                    // Selecting all columns from Shared - just return self
                    return self;
                }
                RowStorage::Shared(ref arc) => {
                    // Prefix selection from Shared: clone Arc refs for prefix columns only
                    // This is O(prefix_len) Arc clones (cheap atomic increments), no value cloning
                    let prefix_len = indices.len();
                    let mut values = CompactVec::with_capacity(prefix_len);
                    for i in 0..prefix_len {
                        values.push(CompactArc::clone(&arc[i]));
                    }
                    return Row {
                        storage: RowStorage::Owned(values),
                    };
                }
            }
        }

        // Note: Identity projection (selecting all columns in order) is already
        // handled by the prefix check above, so no separate check needed here.

        match self.storage {
            RowStorage::Inline(vec) => {
                // Inline: clone values we need, vec is dropped at end
                let mut values: CompactVec<Value> = CompactVec::with_capacity(indices.len());
                for &idx in indices {
                    if idx < vec.len() {
                        values.push(vec[idx].clone());
                    } else {
                        values.push(Value::null_unknown());
                    }
                }
                // vec drops here - values we didn't need are freed
                Row::from_compact_vec(values) // Keep as Inline!
            }
            RowStorage::Owned(vec) => {
                // Owned: clone Arc refs we need (O(1) per Arc), drop rest naturally
                // CompactArc::clone is just an atomic increment - much cheaper than allocation.
                let len = vec.len();
                let mut values = CompactVec::with_capacity(indices.len());
                for &idx in indices {
                    if idx < len {
                        values.push(CompactArc::clone(&vec[idx]));
                    } else {
                        values.push(interned_null(DataType::Null));
                    }
                }
                // vec drops here - decrements refcounts for values we cloned,
                // fully drops values we didn't need
                Row {
                    storage: RowStorage::Owned(values),
                }
            }
            RowStorage::Shared(arc) => {
                // Shared: clone Arc references (cheap O(1) per value)
                let mut values = CompactVec::with_capacity(indices.len());
                for &idx in indices {
                    if idx < arc.len() {
                        values.push(CompactArc::clone(&arc[idx]));
                    } else {
                        values.push(interned_null(DataType::Null));
                    }
                }
                Row {
                    storage: RowStorage::Owned(values),
                }
            }
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
        match &self.storage {
            RowStorage::Inline(vec) => {
                // Pre-allocate and collect directly into CompactVec
                let mut values = CompactVec::with_capacity(indices.len());
                for &i in indices {
                    if let Some(v) = vec.get(i) {
                        values.push(v.clone());
                    }
                }
                Row::from_compact_vec(values)
            }
            RowStorage::Owned(vec) => {
                let mut values = CompactVec::with_capacity(indices.len());
                for &i in indices {
                    if let Some(arc) = vec.get(i) {
                        values.push(CompactArc::clone(arc));
                    }
                }
                Row {
                    storage: RowStorage::Owned(values),
                }
            }
            RowStorage::Shared(arc) => {
                let mut values = CompactVec::with_capacity(indices.len());
                for &i in indices {
                    if let Some(a) = arc.get(i) {
                        values.push(CompactArc::clone(a));
                    }
                }
                Row {
                    storage: RowStorage::Owned(values),
                }
            }
        }
    }

    /// Concatenate two rows
    pub fn concat(&self, other: &Row) -> Row {
        // Pre-allocate with exact capacity to avoid reallocation
        let total_len = self.len() + other.len();
        let mut values = CompactVec::with_capacity(total_len);
        values.extend(self.iter().cloned());
        values.extend(other.iter().cloned());
        Row::from_compact_vec(values)
    }

    /// Create a row by repeating a value
    pub fn repeat(value: Value, count: usize) -> Row {
        // Pre-allocate and fill directly
        let mut values = CompactVec::with_capacity(count);
        for _ in 0..count {
            values.push(value.clone());
        }
        Row::from_compact_vec(values)
    }
}

// Note: Deref<Target=[Value]> removed - incompatible with CompactArc<Value> storage
// Use iter() for iteration, get() for indexed access, or as_arc_slice() for Arc references

// Implement Index for convenient access
impl Index<usize> for Row {
    type Output = Value;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.storage
            .get_value(index)
            .expect("row index out of bounds")
    }
}

// Note: IndexMut removed - use get_mut() or set() for mutation

// Implement FromIterator for collecting values into a row
impl FromIterator<Value> for Row {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        // Collect directly into CompactVec to avoid Vec allocation
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

impl From<Arc<[Value]>> for Row {
    fn from(values: Arc<[Value]>) -> Self {
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
    }

    #[test]
    fn test_row_from_arc() {
        let values: Arc<[Value]> = Arc::from(vec![Value::integer(1), Value::text("hello")]);
        let row = Row::from_arc(values);
        assert_eq!(row.len(), 2);

        // Clone should be O(1)
        let row2 = row.clone();
        assert_eq!(row2.len(), 2);
        assert_eq!(row, row2);
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
        let values: Arc<[Value]> = Arc::from(vec![Value::integer(1), Value::text("hello")]);
        let mut row = Row::from_arc(values);

        // Mutation should trigger copy-on-write
        row.push(Value::integer(2));
        assert_eq!(row.len(), 3);
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
}
