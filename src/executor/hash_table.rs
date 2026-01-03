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

//! Optimized hash table for join operations.
//!
//! This module provides a specialized hash table designed for the build phase
//! of hash joins. Key optimizations:
//!
//! 1. **Pre-allocated**: Sized upfront based on build side cardinality
//! 2. **Cache-efficient**: Linear probing within cache lines
//! 3. **Zero-allocation probe**: Iterator returns indices without allocation
//! 4. **Full hash stored**: Quick rejection without row access
//!
//! # Memory Layout
//!
//! ```text
//! JoinHashTable
//! ├── bucket_heads: Vec<i32>    [bucket_count]     // First entry index per bucket
//! ├── entries: Vec<HashEntry>   [row_count]        // One per build row
//! └── bucket_mask: u64                             // For fast modulo
//!
//! HashEntry (16 bytes, cache-aligned)
//! ├── hash: u64     // Full hash for quick rejection
//! ├── row_idx: u32  // Index into build rows
//! └── next: u32     // Next in chain (EMPTY = end)
//! ```

use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

use crate::core::{Row, Value};

/// Sentinel value indicating end of chain or empty bucket.
const EMPTY: u32 = u32::MAX;

/// Minimum number of buckets (must be power of 2).
const MIN_BUCKETS: usize = 16;

/// A hash entry in the join hash table.
///
/// Each entry represents one row from the build side.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HashEntry {
    /// Full 64-bit hash for quick rejection during probe.
    /// Comparing hashes first avoids touching row data for non-matches.
    hash: u64,
    /// Index into the build rows vector.
    row_idx: u32,
    /// Index of next entry in the chain (EMPTY = end of chain).
    next: u32,
}

impl HashEntry {
    #[inline]
    fn new(hash: u64, row_idx: u32, next: u32) -> Self {
        Self {
            hash,
            row_idx,
            next,
        }
    }
}

/// Optimized hash table for join operations.
///
/// This hash table is specifically designed for the build phase of hash joins.
/// It uses chaining with linked entries stored in a flat vector for cache efficiency.
///
/// # Example
///
/// ```ignore
/// // Build phase
/// let mut table = JoinHashTable::with_capacity(build_rows.len());
/// for (idx, row) in build_rows.iter().enumerate() {
///     let hash = hash_row_keys(row, &key_indices);
///     table.insert(hash, idx as u32);
/// }
///
/// // Probe phase
/// for probe_row in probe_rows {
///     let hash = hash_row_keys(probe_row, &probe_key_indices);
///     for build_idx in table.probe(hash) {
///         // Verify actual key equality and produce output
///     }
/// }
/// ```
pub struct JoinHashTable {
    /// First entry index for each bucket (-1 if empty).
    /// Sized to power of 2 for fast modulo via bitwise AND.
    bucket_heads: Vec<i32>,

    /// Flat storage of all entries.
    /// One entry per build row.
    entries: Vec<HashEntry>,

    /// Mask for computing bucket index: bucket = hash & mask
    bucket_mask: u64,

    /// Number of entries inserted.
    len: usize,
}

impl JoinHashTable {
    /// Create a new hash table with capacity for the given number of rows.
    ///
    /// The table is pre-allocated to avoid resizing during build.
    /// Bucket count is sized to achieve ~75% load factor.
    pub fn with_capacity(row_count: usize) -> Self {
        // Choose bucket count as next power of 2 >= row_count * 4/3
        // This gives us ~75% load factor
        let bucket_count = (row_count * 4 / 3).max(MIN_BUCKETS).next_power_of_two();

        let bucket_mask = (bucket_count - 1) as u64;

        Self {
            bucket_heads: vec![-1; bucket_count],
            entries: Vec::with_capacity(row_count),
            bucket_mask,
            len: 0,
        }
    }

    /// Create an empty hash table (for cases where build side is empty).
    pub fn empty() -> Self {
        Self {
            bucket_heads: vec![-1; MIN_BUCKETS],
            entries: Vec::new(),
            bucket_mask: (MIN_BUCKETS - 1) as u64,
            len: 0,
        }
    }

    /// Build a hash table from rows using the specified key indices.
    ///
    /// This is the main entry point for creating a join hash table.
    pub fn build(rows: &[Row], key_indices: &[usize]) -> Self {
        let mut table = Self::with_capacity(rows.len());

        for (idx, row) in rows.iter().enumerate() {
            let hash = hash_row_keys(row, key_indices);
            table.insert(hash, idx as u32);
        }

        table
    }

    /// Build hash table and populate bloom filter in a single pass.
    ///
    /// This is more efficient than building separately because we only
    /// extract and hash key values once for both structures.
    ///
    /// # Arguments
    /// * `rows` - Build side rows
    /// * `key_indices` - Indices of join key columns
    /// * `bloom_builder` - Bloom filter builder to populate
    ///
    /// # Returns
    /// The built hash table (bloom filter is populated in-place)
    pub fn build_with_bloom(
        rows: &[Row],
        key_indices: &[usize],
        bloom_builder: &mut crate::optimizer::bloom::BloomFilterBuilder,
    ) -> Self {
        let mut table = Self::with_capacity(rows.len());

        for (idx, row) in rows.iter().enumerate() {
            // Extract keys and compute hash once
            let hash = hash_row_keys(row, key_indices);

            // Insert into hash table
            table.insert(hash, idx as u32);

            // Insert into bloom filter using the same pre-computed hash
            // This avoids re-hashing the same key values
            bloom_builder.insert_raw_hash(hash);
        }

        table
    }

    /// Insert a row index with its pre-computed hash.
    ///
    /// # Arguments
    /// * `hash` - The hash of the row's key columns
    /// * `row_idx` - The index of the row in the build rows vector
    #[inline]
    pub fn insert(&mut self, hash: u64, row_idx: u32) {
        let bucket = (hash & self.bucket_mask) as usize;

        // Get current head of chain
        let old_head = self.bucket_heads[bucket];

        // Create new entry pointing to old head
        // Use cached len instead of Vec::len() to avoid repeated length checks
        let entry_idx = self.len as u32;
        let next = if old_head >= 0 {
            old_head as u32
        } else {
            EMPTY
        };
        self.entries.push(HashEntry::new(hash, row_idx, next));

        // Update bucket head to point to new entry
        self.bucket_heads[bucket] = entry_idx as i32;
        self.len += 1;
    }

    /// Probe the hash table for matching row indices.
    ///
    /// Returns an iterator that yields row indices for entries
    /// with matching hashes. The caller must verify actual key
    /// equality for each returned index (to handle hash collisions).
    ///
    /// This is a zero-allocation operation - the iterator only
    /// holds a reference to the table.
    #[inline]
    pub fn probe(&self, hash: u64) -> ProbeIter<'_> {
        let bucket = (hash & self.bucket_mask) as usize;
        let first = self.bucket_heads[bucket];

        ProbeIter {
            table: self,
            hash,
            current: first,
        }
    }

    /// Get the number of entries in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the number of buckets.
    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.bucket_heads.len()
    }

    /// Get the load factor (entries / buckets).
    #[inline]
    pub fn load_factor(&self) -> f64 {
        self.len as f64 / self.bucket_heads.len() as f64
    }
}

/// Zero-allocation iterator over probe results.
///
/// Yields row indices for entries whose hash matches the probe hash.
/// The caller must verify actual key equality for each returned index.
pub struct ProbeIter<'a> {
    table: &'a JoinHashTable,
    hash: u64,
    current: i32,
}

impl Iterator for ProbeIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.current >= 0 {
            let entry = &self.table.entries[self.current as usize];
            self.current = if entry.next == EMPTY {
                -1
            } else {
                entry.next as i32
            };

            // Only return if hash matches (quick rejection for non-matches)
            if entry.hash == self.hash {
                return Some(entry.row_idx as usize);
            }
        }
        None
    }
}

// ============================================================================
// Hashing Utilities
// ============================================================================

/// Hash values at given indices using a get function.
///
/// This is a generic version that works with any type that provides indexed access
/// to values (Row, RowRef, etc.). Uses the same FxHash algorithm as hash_row_keys.
#[inline]
pub fn hash_keys_with<'a, F>(key_indices: &[usize], get_value: F) -> u64
where
    F: Fn(usize) -> Option<&'a Value>,
{
    // Fast path for single integer key (most common case: PK joins)
    if key_indices.len() == 1 {
        if let Some(Value::Integer(i)) = get_value(key_indices[0]) {
            // FxHash for single integer - very fast
            return (*i as u64).wrapping_mul(0x517cc1b727220a95);
        }
    }

    let mut hasher = FxHasher::default();

    for &idx in key_indices {
        if let Some(value) = get_value(idx) {
            hash_value(&mut hasher, value);
        } else {
            // NULL marker - use a sentinel that's unlikely to collide
            0xDEADBEEF_u64.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Hash row key columns into a single u64.
///
/// Uses FxHasher which is optimized for trusted keys in embedded database context.
/// This is the same algorithm used in utils.rs but kept here to avoid circular deps.
#[inline]
pub fn hash_row_keys(row: &Row, key_indices: &[usize]) -> u64 {
    // Fast path for single integer key (most common case: PK joins)
    if key_indices.len() == 1 {
        if let Some(Value::Integer(i)) = row.get(key_indices[0]) {
            // FxHash for single integer - very fast
            return (*i as u64).wrapping_mul(0x517cc1b727220a95);
        }
    }

    let mut hasher = FxHasher::default();

    for &idx in key_indices {
        if let Some(value) = row.get(idx) {
            hash_value(&mut hasher, value);
        } else {
            // NULL marker - use a sentinel that's unlikely to collide
            0xDEADBEEF_u64.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Hash a single value into a hasher.
#[inline]
fn hash_value<H: Hasher>(hasher: &mut H, value: &Value) {
    // Type discriminant first for type safety
    match value {
        Value::Integer(i) => {
            1_u8.hash(hasher);
            i.hash(hasher);
        }
        Value::Float(f) => {
            2_u8.hash(hasher);
            // Hash the bits for consistency
            f.to_bits().hash(hasher);
        }
        Value::Text(s) => {
            3_u8.hash(hasher);
            s.hash(hasher);
        }
        Value::Boolean(b) => {
            4_u8.hash(hasher);
            b.hash(hasher);
        }
        Value::Null(_) => {
            5_u8.hash(hasher);
        }
        Value::Timestamp(ts) => {
            6_u8.hash(hasher);
            ts.timestamp_nanos_opt().hash(hasher);
        }
        Value::Json(j) => {
            7_u8.hash(hasher);
            j.hash(hasher);
        }
    }
}

/// Verify that two rows have equal key values.
///
/// Used after hash matching to confirm actual equality (handling hash collisions).
#[inline]
pub fn verify_key_equality(row1: &Row, row2: &Row, indices1: &[usize], indices2: &[usize]) -> bool {
    debug_assert_eq!(indices1.len(), indices2.len());

    // Fast path for single integer key (most common case: PK joins)
    if indices1.len() == 1 {
        let v1 = row1.get(indices1[0]);
        let v2 = row2.get(indices2[0]);
        return match (v1, v2) {
            (Some(Value::Integer(a)), Some(Value::Integer(b))) => a == b,
            (Some(a), Some(b)) => values_equal(a, b),
            _ => false,
        };
    }

    for (&idx1, &idx2) in indices1.iter().zip(indices2.iter()) {
        let v1 = row1.get(idx1);
        let v2 = row2.get(idx2);

        match (v1, v2) {
            (Some(Value::Integer(a)), Some(Value::Integer(b))) => {
                if a != b {
                    return false;
                }
            }
            (Some(a), Some(b)) => {
                if !values_equal(a, b) {
                    return false;
                }
            }
            (None, None) => {
                // NULL = NULL is false in SQL, but for join matching we want false anyway
                return false;
            }
            _ => return false,
        }
    }

    true
}

/// Check if two values are equal (for join key comparison).
///
/// Uses bit-exact comparison for floats to ensure consistent join semantics.
/// Two floats that compare equal with `==` will join correctly.
#[inline]
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x == y,
        // Bit-exact float comparison for consistent join semantics
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::Boolean(x), Value::Boolean(y)) => x == y,
        (Value::Null(_), Value::Null(_)) => false, // NULL != NULL
        (Value::Timestamp(x), Value::Timestamp(y)) => x == y,
        (Value::Json(x), Value::Json(y)) => x == y,
        // Cross-type numeric comparison: convert integer to float and compare bits
        (Value::Integer(x), Value::Float(y)) => (*x as f64).to_bits() == y.to_bits(),
        (Value::Float(x), Value::Integer(y)) => x.to_bits() == (*y as f64).to_bits(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(values: Vec<i64>) -> Row {
        Row::from_values(values.into_iter().map(Value::integer).collect())
    }

    #[test]
    fn test_basic_insert_and_probe() {
        let mut table = JoinHashTable::with_capacity(4);

        // Insert some entries
        table.insert(100, 0);
        table.insert(200, 1);
        table.insert(100, 2); // Same hash as first entry
        table.insert(300, 3);

        assert_eq!(table.len(), 4);

        // Probe for hash 100 should find entries 0 and 2
        let matches: Vec<_> = table.probe(100).collect();
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&0));
        assert!(matches.contains(&2));

        // Probe for hash 200 should find entry 1
        let matches: Vec<_> = table.probe(200).collect();
        assert_eq!(matches, vec![1]);

        // Probe for non-existent hash should find nothing
        let matches: Vec<_> = table.probe(999).collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_build_from_rows() {
        let rows = vec![
            make_row(vec![1, 10]),
            make_row(vec![2, 20]),
            make_row(vec![1, 30]), // Same key as first row
            make_row(vec![3, 40]),
        ];

        let key_indices = vec![0]; // Key on first column
        let table = JoinHashTable::build(&rows, &key_indices);

        assert_eq!(table.len(), 4);

        // Probe for key=1
        let hash = hash_row_keys(&rows[0], &key_indices);
        let matches: Vec<_> = table.probe(hash).collect();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_empty_table() {
        let table = JoinHashTable::empty();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);

        let matches: Vec<_> = table.probe(100).collect();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_load_factor() {
        let mut table = JoinHashTable::with_capacity(100);

        for i in 0..100 {
            table.insert(i as u64, i as u32);
        }

        // With 100 entries, load factor depends on bucket count
        // We target ~75% load factor, but it can vary
        let load = table.load_factor();
        assert!(
            load > 0.3 && load <= 1.0,
            "Load factor {} out of expected range",
            load
        );
        // Verify we have all entries
        assert_eq!(table.len(), 100);
    }

    #[test]
    fn test_verify_key_equality() {
        let row1 = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let row2 = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let row3 = Row::from_values(vec![Value::integer(2), Value::text("hello")]);

        assert!(verify_key_equality(&row1, &row2, &[0, 1], &[0, 1]));
        assert!(!verify_key_equality(&row1, &row3, &[0, 1], &[0, 1]));
    }

    #[test]
    fn test_hash_row_keys() {
        let row1 = make_row(vec![1, 2, 3]);
        let row2 = make_row(vec![1, 2, 3]);
        let row3 = make_row(vec![1, 2, 4]);

        let indices = vec![0, 1];

        // Same keys should produce same hash
        assert_eq!(
            hash_row_keys(&row1, &indices),
            hash_row_keys(&row2, &indices)
        );

        // Different values in non-key column shouldn't affect hash
        let row4 = make_row(vec![1, 2, 999]);
        assert_eq!(
            hash_row_keys(&row1, &indices),
            hash_row_keys(&row4, &indices)
        );

        // Different keys should (usually) produce different hash
        // This isn't guaranteed but is very likely
        assert_ne!(hash_row_keys(&row1, &[0, 2]), hash_row_keys(&row3, &[0, 2]));
    }

    #[test]
    fn test_chain_collision() {
        // Force collisions by using a small bucket count
        let mut table = JoinHashTable {
            bucket_heads: vec![-1; 4], // Only 4 buckets
            entries: Vec::new(),
            bucket_mask: 3,
            len: 0,
        };

        // All these will go to bucket 0 (hash & 3 == 0)
        table.insert(0, 0);
        table.insert(4, 1);
        table.insert(8, 2);
        table.insert(12, 3);

        // Probe for each should find exactly one match
        assert_eq!(table.probe(0).count(), 1);
        assert_eq!(table.probe(4).count(), 1);
        assert_eq!(table.probe(8).count(), 1);
        assert_eq!(table.probe(12).count(), 1);
    }
}
