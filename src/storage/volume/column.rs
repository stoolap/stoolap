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

//! Typed column storage for frozen volumes.
//!
//! Each column is stored as a contiguous typed array with a null bitmap.
//! This avoids the 16-byte Value enum overhead during scans — integer columns
//! are raw `[i64]`, float columns are raw `[f64]`, etc.
//!
//! Text columns use dictionary encoding: values are stored as `[u32]` IDs
//! referencing a deduplicated string table, reducing storage for repeated
//! values (e.g., exchange names, symbols) from N bytes to 4 bytes.

use crate::common::SmartString;
use crate::core::{DataType, Value};

/// Typed column data stored contiguously for cache-friendly access.
///
/// Each variant stores a flat array of the native type plus a null bitmap.
/// The null bitmap uses one byte per row (not bit-packed) for simplicity
/// and fast random access. For 1M rows this is 1MB overhead.
pub enum ColumnData {
    /// 64-bit signed integers with null bitmap.
    /// Used for INTEGER columns and auto-increment IDs.
    Int64 { values: Vec<i64>, nulls: Vec<bool> },

    /// 64-bit floating point with null bitmap.
    /// Used for FLOAT/DOUBLE columns.
    Float64 { values: Vec<f64>, nulls: Vec<bool> },

    /// Timestamps stored as nanoseconds since Unix epoch.
    /// Preserves sub-second precision while enabling integer comparison
    /// and binary search without chrono overhead.
    TimestampNanos { values: Vec<i64>, nulls: Vec<bool> },

    /// Booleans stored as bytes with null bitmap.
    Boolean { values: Vec<bool>, nulls: Vec<bool> },

    /// Dictionary-encoded text.
    /// Each value is a u32 index into the dictionary Vec.
    /// Repeated strings (common in categorical data) share a single
    /// dictionary entry, reducing storage from O(n * avg_len) to O(n * 4 + unique * avg_len).
    Dictionary {
        /// Per-row dictionary IDs
        ids: Vec<u32>,
        /// Deduplicated string table
        dictionary: Vec<SmartString>,
        /// Null bitmap
        nulls: Vec<bool>,
    },

    /// Raw bytes for Extension types (JSON, Vector, etc.)
    /// Falls back to per-value serialization.
    Bytes {
        /// Concatenated byte data for all rows
        data: Vec<u8>,
        /// (offset, length) pairs for each row
        offsets: Vec<(u64, u64)>,
        /// The extension DataType tag (e.g., DataType::Json)
        ext_type: DataType,
        /// Null bitmap
        nulls: Vec<bool>,
    },
}

/// Per-column zone map for segment-level pruning.
///
/// Stores the min and max values seen in the column. A query predicate
/// like `WHERE time >= X` can skip the entire volume if `zone_max < X`.
///
/// For dictionary-encoded columns, min/max reference the dictionary values,
/// not the IDs.
#[derive(Debug, Clone)]
pub struct ZoneMap {
    /// Minimum non-null value in the column
    pub min: Value,
    /// Maximum non-null value in the column
    pub max: Value,
    /// Number of null values
    pub null_count: u32,
    /// Total number of rows
    pub row_count: u32,
}

impl ColumnData {
    /// Get the number of rows in this column.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int64 { values, .. } => values.len(),
            ColumnData::Float64 { values, .. } => values.len(),
            ColumnData::TimestampNanos { values, .. } => values.len(),
            ColumnData::Boolean { values, .. } => values.len(),
            ColumnData::Dictionary { ids, .. } => ids.len(),
            ColumnData::Bytes { offsets, .. } => offsets.len(),
        }
    }

    /// Check if the column is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if a specific row is null.
    #[inline]
    pub fn is_null(&self, idx: usize) -> bool {
        match self {
            ColumnData::Int64 { nulls, .. }
            | ColumnData::Float64 { nulls, .. }
            | ColumnData::TimestampNanos { nulls, .. }
            | ColumnData::Boolean { nulls, .. }
            | ColumnData::Dictionary { nulls, .. }
            | ColumnData::Bytes { nulls, .. } => nulls[idx],
        }
    }

    /// Reconstruct a Value for row `idx`.
    ///
    /// This is the "slow path" used when the executor needs a full Value
    /// (e.g., for projection into result rows). Aggregation and filtering
    /// should use the typed accessors (`get_i64`, `get_f64`) instead.
    pub fn get_value(&self, idx: usize) -> Value {
        match self {
            ColumnData::Int64 { values, nulls } => {
                if nulls[idx] {
                    Value::Null(DataType::Integer)
                } else {
                    Value::Integer(values[idx])
                }
            }
            ColumnData::Float64 { values, nulls } => {
                if nulls[idx] {
                    Value::Null(DataType::Float)
                } else {
                    Value::Float(values[idx])
                }
            }
            ColumnData::TimestampNanos { values, nulls } => {
                if nulls[idx] {
                    Value::Null(DataType::Timestamp)
                } else {
                    let nanos = values[idx];
                    let secs = nanos.div_euclid(1_000_000_000);
                    let sub_nanos = nanos.rem_euclid(1_000_000_000) as u32;
                    match chrono::TimeZone::timestamp_opt(&chrono::Utc, secs, sub_nanos) {
                        chrono::LocalResult::Single(dt) => Value::Timestamp(dt),
                        _ => Value::Null(DataType::Timestamp),
                    }
                }
            }
            ColumnData::Boolean { values, nulls } => {
                if nulls[idx] {
                    Value::Null(DataType::Boolean)
                } else {
                    Value::Boolean(values[idx])
                }
            }
            ColumnData::Dictionary {
                ids,
                dictionary,
                nulls,
            } => {
                if nulls[idx] {
                    Value::Null(DataType::Text)
                } else {
                    Value::Text(dictionary[ids[idx] as usize].clone())
                }
            }
            ColumnData::Bytes {
                data,
                offsets,
                ext_type,
                nulls,
            } => {
                if nulls[idx] {
                    Value::Null(*ext_type)
                } else {
                    let (off, len) = offsets[idx];
                    let bytes = &data[off as usize..(off + len) as usize];
                    // Reconstruct Extension value: prepend type tag
                    let mut tagged = Vec::with_capacity(1 + bytes.len());
                    tagged.push(*ext_type as u8);
                    tagged.extend_from_slice(bytes);
                    Value::Extension(crate::common::CompactArc::from(tagged))
                }
            }
        }
    }

    // =========================================================================
    // Fast typed accessors (no Value construction)
    // =========================================================================

    /// Get raw i64 value. Works for Int64 and TimestampNanos columns.
    /// Returns 0 on type mismatch (callers must guard with type checks).
    #[inline]
    pub fn get_i64(&self, idx: usize) -> i64 {
        match self {
            ColumnData::Int64 { values, .. } | ColumnData::TimestampNanos { values, .. } => {
                values[idx]
            }
            _ => 0,
        }
    }

    /// Get raw f64 value. Returns 0.0 on type mismatch.
    #[inline]
    pub fn get_f64(&self, idx: usize) -> f64 {
        match self {
            ColumnData::Float64 { values, .. } => values[idx],
            _ => 0.0,
        }
    }

    /// Get raw bool value. Returns false on type mismatch.
    #[inline]
    pub fn get_bool(&self, idx: usize) -> bool {
        match self {
            ColumnData::Boolean { values, .. } => values[idx],
            _ => false,
        }
    }

    /// Get dictionary string reference. Returns empty string on type mismatch.
    #[inline]
    pub fn get_str(&self, idx: usize) -> &str {
        match self {
            ColumnData::Dictionary {
                ids, dictionary, ..
            } => &dictionary[ids[idx] as usize],
            _ => "",
        }
    }

    /// Get the dictionary ID for a row. Returns u32::MAX on type mismatch.
    #[inline]
    pub fn get_dict_id(&self, idx: usize) -> u32 {
        match self {
            ColumnData::Dictionary { ids, .. } => ids[idx],
            _ => u32::MAX,
        }
    }

    // =========================================================================
    // Search operations
    // =========================================================================

    /// Binary search on sorted i64 data (timestamps, integer PKs).
    /// Returns the index of the first element >= target.
    pub fn binary_search_ge(&self, target: i64) -> usize {
        match self {
            ColumnData::Int64 { values, .. } | ColumnData::TimestampNanos { values, .. } => {
                values.partition_point(|v| *v < target)
            }
            _ => 0,
        }
    }

    /// Binary search on sorted i64 data.
    /// Returns the index of the first element > target.
    pub fn binary_search_gt(&self, target: i64) -> usize {
        match self {
            ColumnData::Int64 { values, .. } | ColumnData::TimestampNanos { values, .. } => {
                values.partition_point(|v| *v <= target)
            }
            _ => 0,
        }
    }

    /// Look up a string in the dictionary and return its ID.
    /// Returns None if the string is not in the dictionary.
    ///
    /// For hot paths with repeated lookups on large dicts, callers should
    /// build a reverse HashMap externally (done by VolumeScanner's dict
    /// filter cache) instead of calling this per-row.
    pub fn dict_lookup(&self, value: &str) -> Option<u32> {
        match self {
            ColumnData::Dictionary { dictionary, .. } => {
                // Linear scan — cache-friendly and avoids the wasted O(log n)
                // binary search attempt that always fails on unsorted dicts.
                dictionary
                    .iter()
                    .position(|s| s.as_str() == value)
                    .map(|p| p as u32)
            }
            _ => None,
        }
    }
}

/// Simple bloom filter for fast membership testing on column values.
///
/// Used to quickly determine if a value MIGHT exist in a volume column.
/// False positives are possible, false negatives are not.
#[derive(Debug, Clone)]
pub struct ColumnBloomFilter {
    /// Bitset
    bits: Vec<u64>,
    /// Number of bits
    num_bits: usize,
}

impl ColumnBloomFilter {
    /// Create a bloom filter sized for the expected number of elements.
    /// Uses ~10 bits per element with 3 hash functions (~1.7% false positive rate).
    pub fn new(expected_elements: usize) -> Self {
        let num_bits = (expected_elements * 10).max(64);
        let num_words = num_bits.div_ceil(64);
        Self {
            bits: vec![0u64; num_words],
            num_bits,
        }
    }

    /// Add raw bytes with a type tag to the bloom filter (avoids Value allocation).
    fn add_raw(&mut self, tag: u8, bytes: &[u8]) {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut h = FNV_OFFSET;
        h ^= tag as u64;
        h = h.wrapping_mul(FNV_PRIME);
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        self.insert_hash(h);
    }

    /// Add an i64 value (for Integer and TimestampNanos columns).
    pub fn add_i64(&mut self, val: i64) {
        self.add_raw(1, &val.to_le_bytes());
    }

    /// Add an f64 value (for Float columns).
    pub fn add_f64(&mut self, val: f64) {
        self.add_raw(2, &val.to_bits().to_le_bytes());
    }

    /// Add a string value (for Dictionary/Text columns).
    pub fn add_str(&mut self, val: &str) {
        self.add_raw(3, val.as_bytes());
    }

    /// Add a bool value (tag 4, matching hash_value for Value::Boolean).
    pub fn add_bool(&mut self, val: bool) {
        self.add_raw(4, &[val as u8]);
    }

    /// Add a timestamp value as nanoseconds (tag 5, matching hash_value for Value::Timestamp).
    pub fn add_timestamp_nanos(&mut self, nanos: i64) {
        self.add_raw(5, &nanos.to_le_bytes());
    }

    fn insert_hash(&mut self, h: u64) {
        let h1 = h as usize;
        let h2 = (h >> 32) as usize;
        for i in 0..3usize {
            let bit = (h1.wrapping_add(i.wrapping_mul(h2))) % self.num_bits;
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    /// Add a value to the bloom filter.
    pub fn add(&mut self, value: &Value) {
        let h = Self::hash_value(value);
        self.insert_hash(h);
    }

    /// Check if a value MIGHT exist in the filter.
    /// Returns false only if the value is definitely not present.
    pub fn might_contain(&self, value: &Value) -> bool {
        self.might_contain_hash(Self::hash_value(value))
    }

    /// Check bloom filter using a pre-computed hash.
    /// Use `hash_value_static` to compute the hash once, then pass it to
    /// multiple volumes to avoid redundant hashing of the same value.
    #[inline]
    pub fn might_contain_hash(&self, h: u64) -> bool {
        let h1 = h as usize;
        let h2 = (h >> 32) as usize;
        for i in 0..3usize {
            let bit = (h1.wrapping_add(i.wrapping_mul(h2))) % self.num_bits;
            if self.bits[bit / 64] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Compute bloom filter hash for a Value. Can be called once and reused
    /// across multiple volumes via `might_contain_hash`.
    pub fn hash_value_static(value: &Value) -> u64 {
        Self::hash_value(value)
    }

    /// Return the logical number of bits in the filter (needed for serialization).
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Serialize the bitset to bytes in little-endian format.
    pub fn bits_as_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bits.len() * 8);
        for &word in &self.bits {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// Reconstruct a bloom filter from serialized parts.
    ///
    /// `num_bits` is the logical bit count; `bytes` is the raw bitset written
    /// by `bits_as_bytes()` (little-endian u64 words).
    pub fn from_parts(num_bits: usize, bytes: &[u8]) -> Self {
        let num_words = bytes.len() / 8;
        let mut bits = Vec::with_capacity(num_words);
        for i in 0..num_words {
            let offset = i * 8;
            bits.push(u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]));
        }
        // Clamp num_bits to the actual backing storage to prevent OOB in might_contain.
        let max_bits = num_words * 64;
        let safe_num_bits = if num_bits > max_bits || num_bits == 0 {
            max_bits.max(1)
        } else {
            num_bits
        };
        Self {
            bits,
            num_bits: safe_num_bits,
        }
    }

    /// Stable FNV-1a hash that is deterministic across Rust versions.
    ///
    /// Bloom filter bits are persisted to disk, so the hash function MUST
    /// produce identical output regardless of Rust toolchain version.
    /// `std::collections::hash_map::DefaultHasher` does NOT guarantee this.
    fn hash_value(value: &Value) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut h = FNV_OFFSET;

        #[inline(always)]
        fn mix(h: &mut u64, bytes: &[u8]) {
            for &b in bytes {
                *h ^= b as u64;
                *h = h.wrapping_mul(FNV_PRIME);
            }
        }

        match value {
            Value::Integer(i) => {
                mix(&mut h, &[1]);
                mix(&mut h, &i.to_le_bytes());
            }
            Value::Float(f) => {
                mix(&mut h, &[2]);
                mix(&mut h, &f.to_bits().to_le_bytes());
            }
            Value::Text(s) => {
                mix(&mut h, &[3]);
                mix(&mut h, s.as_bytes());
            }
            Value::Boolean(b) => {
                mix(&mut h, &[4]);
                mix(&mut h, &[*b as u8]);
            }
            Value::Timestamp(ts) => {
                mix(&mut h, &[5]);
                let nanos = ts.timestamp_nanos_opt().unwrap_or_else(|| {
                    ts.timestamp()
                        .wrapping_mul(1_000_000_000)
                        .wrapping_add(ts.timestamp_subsec_nanos() as i64)
                });
                mix(&mut h, &nanos.to_le_bytes());
            }
            _ => {
                mix(&mut h, &[0]);
            }
        }
        h
    }
}

impl ZoneMap {
    /// Check if a predicate `column >= value` can possibly match any row.
    /// Returns false if we can definitively skip this volume.
    #[inline]
    pub fn may_contain_gte(&self, value: &Value) -> bool {
        if self.max.is_null() {
            return false; // all nulls
        }
        // max >= value means some rows might match
        self.max
            .compare(value)
            .map(|o| o != std::cmp::Ordering::Less)
            .unwrap_or(true) // on comparison error, don't skip
    }

    /// Check if a predicate `column <= value` can possibly match any row.
    #[inline]
    pub fn may_contain_lte(&self, value: &Value) -> bool {
        if self.min.is_null() {
            return false;
        }
        self.min
            .compare(value)
            .map(|o| o != std::cmp::Ordering::Greater)
            .unwrap_or(true)
    }

    /// Check if a predicate `column = value` can possibly match any row.
    #[inline]
    pub fn may_contain_eq(&self, value: &Value) -> bool {
        if self.min.is_null() {
            return false;
        }
        let above_min = self
            .min
            .compare(value)
            .map(|o| o != std::cmp::Ordering::Greater)
            .unwrap_or(true);
        let below_max = self
            .max
            .compare(value)
            .map(|o| o != std::cmp::Ordering::Less)
            .unwrap_or(true);
        above_min && below_max
    }

    /// Check if a predicate `column BETWEEN low AND high` can possibly match.
    #[inline]
    pub fn may_contain_between(&self, low: &Value, high: &Value) -> bool {
        self.may_contain_gte(low) && self.may_contain_lte(high)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int64_column() {
        let col = ColumnData::Int64 {
            values: vec![10, 20, 30, 0, 50],
            nulls: vec![false, false, false, true, false],
        };
        assert_eq!(col.len(), 5);
        assert_eq!(col.get_i64(0), 10);
        assert_eq!(col.get_i64(2), 30);
        assert!(col.is_null(3));
        assert!(!col.is_null(0));

        // get_value
        assert_eq!(col.get_value(0), Value::Integer(10));
        assert!(col.get_value(3).is_null());
    }

    #[test]
    fn test_float64_column() {
        let col = ColumnData::Float64 {
            values: vec![1.5, 2.5, 0.0],
            nulls: vec![false, false, true],
        };
        assert_eq!(col.get_f64(0), 1.5);
        assert_eq!(col.get_value(2), Value::Null(DataType::Float));
    }

    #[test]
    fn test_dictionary_column() {
        let col = ColumnData::Dictionary {
            ids: vec![0, 1, 0, 1, 0],
            dictionary: vec![SmartString::from("binance"), SmartString::from("coinbase")],
            nulls: vec![false, false, false, false, false],
        };
        assert_eq!(col.get_str(0), "binance");
        assert_eq!(col.get_str(1), "coinbase");
        assert_eq!(col.get_str(2), "binance");
        assert_eq!(col.get_dict_id(0), 0);
        assert_eq!(col.get_dict_id(1), 1);
        assert_eq!(col.dict_lookup("binance"), Some(0));
        assert_eq!(col.dict_lookup("unknown"), None);
    }

    #[test]
    fn test_binary_search() {
        let col = ColumnData::TimestampNanos {
            values: vec![100, 200, 300, 400, 500],
            nulls: vec![false; 5],
        };
        assert_eq!(col.binary_search_ge(250), 2); // first >= 250 is index 2 (300)
        assert_eq!(col.binary_search_ge(300), 2); // first >= 300 is index 2 (300)
        assert_eq!(col.binary_search_ge(100), 0); // first >= 100 is index 0
        assert_eq!(col.binary_search_ge(600), 5); // nothing >= 600
        assert_eq!(col.binary_search_gt(300), 3); // first > 300 is index 3 (400)
    }

    #[test]
    fn test_zone_map_pruning() {
        let zm = ZoneMap {
            min: Value::Integer(10),
            max: Value::Integer(100),
            null_count: 0,
            row_count: 50,
        };
        assert!(zm.may_contain_gte(&Value::Integer(50))); // 100 >= 50
        assert!(zm.may_contain_gte(&Value::Integer(100))); // 100 >= 100
        assert!(!zm.may_contain_gte(&Value::Integer(101))); // 100 < 101
        assert!(zm.may_contain_lte(&Value::Integer(50))); // 10 <= 50
        assert!(zm.may_contain_lte(&Value::Integer(10))); // 10 <= 10
        assert!(!zm.may_contain_lte(&Value::Integer(9))); // 10 > 9
        assert!(zm.may_contain_eq(&Value::Integer(50)));
        assert!(!zm.may_contain_eq(&Value::Integer(5)));
        assert!(!zm.may_contain_eq(&Value::Integer(101)));
    }
}
