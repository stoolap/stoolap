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

//! Runtime Bloom Filter Propagation for Join Optimization
//!
//! This module implements bloom filters that are built during hash join's build phase
//! and can be pushed down to the probe side's scan to filter rows early.
//!
//! ## How It Works
//!
//! 1. During hash join build phase, we build a bloom filter of join keys
//! 2. The bloom filter is propagated ("pushed down") to the probe side
//! 3. Probe side scan uses the bloom filter to skip rows that definitely won't match
//! 4. Only potential matches are sent to the actual join
//!
//! ## Benefits
//!
//! - **I/O Reduction**: Filter rows before reading from storage
//! - **Memory Reduction**: Fewer rows materialized in probe pipeline
//! - **CPU Reduction**: Skip hash probes for non-matching rows
//!
//! ## Example
//!
//! ```sql
//! SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id
//! WHERE c.country = 'US'
//! ```
//!
//! Without bloom filter: Read all orders, probe against customers hash table
//! With bloom filter: Only read orders whose customer_id MIGHT be in the filtered customers
//!
//! ## Stoolap-Specific: Edge Computing Optimization
//!
//! We use a compact bloom filter design optimized for:
//! - Small memory footprint (suitable for edge devices)
//! - Fast hash computation (using FNV-1a)
//! - Configurable false positive rate based on available memory

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::core::Value;

/// Default bits per element for ~1% false positive rate
/// This is used as a reference value; actual bits are calculated from expected elements and FP rate
#[allow(dead_code)]
const DEFAULT_BITS_PER_ELEMENT: usize = 10;

/// Default number of hash functions (optimal for 10 bits/element)
/// Actual hash count is calculated using: k = (m/n) * ln(2)
#[allow(dead_code)]
const DEFAULT_NUM_HASHES: usize = 7;

/// Minimum bloom filter size in bits
const MIN_FILTER_BITS: usize = 64;

/// Maximum bloom filter size in bits (edge computing limit: ~1MB)
const MAX_FILTER_BITS: usize = 8_000_000;

/// A space-efficient probabilistic data structure for set membership testing
///
/// False positives are possible (says "maybe in set" when not),
/// but false negatives are impossible (never says "not in set" when it is).
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit array stored as u64 words
    bits: Vec<u64>,
    /// Number of bits in the filter
    num_bits: usize,
    /// Number of hash functions
    num_hashes: usize,
    /// Number of elements inserted
    element_count: u64,
}

impl BloomFilter {
    /// Create a new bloom filter with expected capacity
    ///
    /// # Arguments
    /// * `expected_elements` - Expected number of elements to insert
    /// * `false_positive_rate` - Desired false positive rate (0.0 to 1.0)
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let fp_rate = false_positive_rate.clamp(0.0001, 0.5);

        // Calculate optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let optimal_bits =
            (-(expected_elements as f64) * fp_rate.ln() / ln2_squared).ceil() as usize;

        // Clamp to reasonable range
        let num_bits = optimal_bits.clamp(MIN_FILTER_BITS, MAX_FILTER_BITS);

        // Round up to multiple of 64 for word alignment
        let num_bits = num_bits.div_ceil(64) * 64;

        // Calculate optimal number of hash functions: k = (m/n) * ln(2)
        let optimal_hashes = ((num_bits as f64 / expected_elements.max(1) as f64)
            * std::f64::consts::LN_2)
            .ceil() as usize;
        let num_hashes = optimal_hashes.clamp(1, 15);

        let num_words = num_bits / 64;

        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            element_count: 0,
        }
    }

    /// Create a bloom filter with default settings for the given capacity
    pub fn with_capacity(expected_elements: usize) -> Self {
        // Use 1% false positive rate by default
        Self::new(expected_elements, 0.01)
    }

    /// Create a small bloom filter for edge computing scenarios
    pub fn for_edge_computing(expected_elements: usize) -> Self {
        // Use higher false positive rate (5%) for smaller memory footprint
        Self::new(expected_elements, 0.05)
    }

    /// Insert a value into the bloom filter
    pub fn insert(&mut self, value: &Value) {
        let hash = self.hash_value(value);
        self.insert_hash(hash);
        self.element_count += 1;
    }

    /// Insert a raw hash value
    fn insert_hash(&mut self, hash: u64) {
        for i in 0..self.num_hashes {
            let bit_idx = self.get_bit_index(hash, i);
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            self.bits[word_idx] |= 1u64 << bit_offset;
        }
    }

    /// Check if a value might be in the set
    ///
    /// Returns:
    /// - `true`: Value MIGHT be in the set (could be false positive)
    /// - `false`: Value is DEFINITELY NOT in the set
    pub fn might_contain(&self, value: &Value) -> bool {
        let hash = self.hash_value(value);
        self.might_contain_hash(hash)
    }

    /// Check using raw hash (internal)
    fn might_contain_hash(&self, hash: u64) -> bool {
        for i in 0..self.num_hashes {
            let bit_idx = self.get_bit_index(hash, i);
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            if (self.bits[word_idx] & (1u64 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Insert using a pre-computed hash value
    ///
    /// This is useful when you want to use the same hash for both
    /// hash table lookup and bloom filter operations.
    pub fn insert_raw_hash(&mut self, hash: u64) {
        self.insert_hash(hash);
        self.element_count += 1;
    }

    /// Check if a pre-computed hash might be in the set
    ///
    /// Returns:
    /// - `true`: Value with this hash MIGHT be in the set (could be false positive)
    /// - `false`: Value with this hash is DEFINITELY NOT in the set
    pub fn might_contain_raw_hash(&self, hash: u64) -> bool {
        self.might_contain_hash(hash)
    }

    /// Get bit index for the i-th hash function
    ///
    /// Uses double hashing: h(i) = h1 + i*h2 + i^2
    /// This provides independent hash functions from a single base hash.
    fn get_bit_index(&self, hash: u64, i: usize) -> usize {
        let h1 = hash as usize;
        let h2 = (hash >> 32) as usize;
        // Double hashing with quadratic probing for better distribution
        let combined = h1.wrapping_add(i.wrapping_mul(h2)).wrapping_add(i * i);
        combined % self.num_bits
    }

    /// Hash a Value using FNV-1a (fast and good distribution)
    fn hash_value(&self, value: &Value) -> u64 {
        let mut hasher = DefaultHasher::new();
        // Use discriminant to ensure different types hash differently
        std::mem::discriminant(value).hash(&mut hasher);
        match value {
            Value::Null(_) => {}
            Value::Boolean(b) => b.hash(&mut hasher),
            Value::Integer(i) => i.hash(&mut hasher),
            Value::Float(f) => f.to_bits().hash(&mut hasher),
            Value::Text(s) => s.hash(&mut hasher),
            Value::Timestamp(t) => t.timestamp_nanos_opt().hash(&mut hasher),
            Value::Json(j) => j.hash(&mut hasher),
        }
        hasher.finish()
    }

    /// Get the estimated false positive rate based on current fill
    pub fn estimated_false_positive_rate(&self) -> f64 {
        if self.element_count == 0 {
            return 0.0;
        }

        // FP rate = (1 - e^(-kn/m))^k
        let k = self.num_hashes as f64;
        let n = self.element_count as f64;
        let m = self.num_bits as f64;

        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// Get number of elements inserted
    pub fn len(&self) -> u64 {
        self.element_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.element_count == 0
    }

    /// Merge another bloom filter into this one (union)
    ///
    /// Both filters must have the same configuration.
    pub fn merge(&mut self, other: &BloomFilter) -> Result<(), &'static str> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err("Cannot merge bloom filters with different configurations");
        }

        for (word, other_word) in self.bits.iter_mut().zip(other.bits.iter()) {
            *word |= *other_word;
        }
        self.element_count += other.element_count;
        Ok(())
    }

    /// Clear the bloom filter
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.element_count = 0;
    }

    /// Get fill ratio (fraction of bits set)
    pub fn fill_ratio(&self) -> f64 {
        let set_bits: usize = self.bits.iter().map(|w| w.count_ones() as usize).sum();
        set_bits as f64 / self.num_bits as f64
    }
}

impl Default for BloomFilter {
    fn default() -> Self {
        Self::with_capacity(1000)
    }
}

/// Builder for creating bloom filters during hash join build phase
pub struct BloomFilterBuilder {
    filter: BloomFilter,
    /// Column name this filter is for
    pub column_name: String,
    /// Table name this filter is from
    pub source_table: String,
}

impl BloomFilterBuilder {
    /// Create a new builder for the given join column
    pub fn new(column_name: String, source_table: String, expected_rows: usize) -> Self {
        Self {
            filter: BloomFilter::with_capacity(expected_rows),
            column_name,
            source_table,
        }
    }

    /// Create a builder optimized for edge computing
    pub fn for_edge(column_name: String, source_table: String, expected_rows: usize) -> Self {
        Self {
            filter: BloomFilter::for_edge_computing(expected_rows),
            column_name,
            source_table,
        }
    }

    /// Add a value to the filter
    pub fn insert(&mut self, value: &Value) {
        self.filter.insert(value);
    }

    /// Finish building and return the filter
    pub fn build(self) -> RuntimeBloomFilter {
        RuntimeBloomFilter {
            filter: self.filter,
            column_name: self.column_name,
            source_table: self.source_table,
        }
    }
}

/// A bloom filter with metadata for runtime propagation
#[derive(Debug, Clone)]
pub struct RuntimeBloomFilter {
    /// The underlying bloom filter
    pub filter: BloomFilter,
    /// Column this filter applies to
    pub column_name: String,
    /// Source table that built this filter
    pub source_table: String,
}

impl RuntimeBloomFilter {
    /// Check if a value might be in the join keys
    pub fn might_match(&self, value: &Value) -> bool {
        self.filter.might_contain(value)
    }

    /// Get estimated selectivity (fraction of rows that pass)
    ///
    /// This is the complement of the filter's effectiveness.
    pub fn estimated_selectivity(&self) -> f64 {
        // Higher fill ratio = more likely to pass = higher selectivity
        // But we also consider the false positive rate
        let fp_rate = self.filter.estimated_false_positive_rate();
        let fill = self.filter.fill_ratio();

        // Selectivity ≈ fill_ratio * (1 + fp_rate)
        // Capped at 1.0
        (fill * (1.0 + fp_rate)).min(1.0)
    }

    /// Check if this filter is worth using
    ///
    /// Returns false if the filter is too full (high false positive rate)
    /// or has too few elements to be useful.
    pub fn is_effective(&self) -> bool {
        // Don't use if empty
        if self.filter.is_empty() {
            return false;
        }

        // Don't use if false positive rate is too high (>50%)
        if self.filter.estimated_false_positive_rate() > 0.5 {
            return false;
        }

        // Don't use if fill ratio is too high (>90%)
        if self.filter.fill_ratio() > 0.9 {
            return false;
        }

        true
    }

    /// Get statistics about this filter
    pub fn stats(&self) -> BloomFilterStats {
        BloomFilterStats {
            column_name: self.column_name.clone(),
            source_table: self.source_table.clone(),
            element_count: self.filter.len(),
            memory_bytes: self.filter.memory_bytes(),
            false_positive_rate: self.filter.estimated_false_positive_rate(),
            fill_ratio: self.filter.fill_ratio(),
            is_effective: self.is_effective(),
        }
    }
}

/// Statistics about a bloom filter
#[derive(Debug, Clone)]
pub struct BloomFilterStats {
    pub column_name: String,
    pub source_table: String,
    pub element_count: u64,
    pub memory_bytes: usize,
    pub false_positive_rate: f64,
    pub fill_ratio: f64,
    pub is_effective: bool,
}

// =============================================================================
// BLOOM FILTER EFFECTIVENESS TRACKING FOR ADAPTIVE OPTIMIZATION
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Global bloom filter effectiveness tracker
static BLOOM_EFFECTIVENESS: OnceLock<BloomEffectivenessTracker> = OnceLock::new();

/// Tracks bloom filter effectiveness for adaptive optimization
///
/// This aggregates statistics across all bloom filter operations to help
/// tune future bloom filters (e.g., adjusting false positive rate).
pub struct BloomEffectivenessTracker {
    /// Total bloom filter checks performed
    total_checks: AtomicU64,
    /// Total true negatives (correctly filtered out)
    true_negatives: AtomicU64,
    /// Total false positives (passed filter but didn't match)
    false_positives: AtomicU64,
    /// Total true positives (passed filter and matched)
    true_positives: AtomicU64,
}

impl BloomEffectivenessTracker {
    /// Create new stats tracker
    fn new() -> Self {
        Self {
            total_checks: AtomicU64::new(0),
            true_negatives: AtomicU64::new(0),
            false_positives: AtomicU64::new(0),
            true_positives: AtomicU64::new(0),
        }
    }

    /// Get the global instance
    pub fn global() -> &'static Self {
        BLOOM_EFFECTIVENESS.get_or_init(Self::new)
    }

    /// Record a bloom filter check with its outcome
    pub fn record_check(&self, passed_filter: bool, actually_matched: bool) {
        self.total_checks.fetch_add(1, Ordering::Relaxed);

        if !passed_filter {
            // Filter said "definitely not present" - true negative
            self.true_negatives.fetch_add(1, Ordering::Relaxed);
        } else if actually_matched {
            // Filter said "maybe present" and it was - true positive
            self.true_positives.fetch_add(1, Ordering::Relaxed);
        } else {
            // Filter said "maybe present" but it wasn't - false positive
            self.false_positives.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a true negative (bloom filter correctly rejected)
    pub fn record_true_negative(&self) {
        self.total_checks.fetch_add(1, Ordering::Relaxed);
        self.true_negatives.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a bloom filter check that passed (may be true or false positive)
    pub fn record_filter_passed(&self) {
        self.total_checks.fetch_add(1, Ordering::Relaxed);
        // We'll assume true positive - false positives are harder to track
        // without expensive secondary verification
        self.true_positives.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the true negative rate (filter effectiveness)
    pub fn true_negative_rate(&self) -> f64 {
        let total = self.total_checks.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let tn = self.true_negatives.load(Ordering::Relaxed);
        tn as f64 / total as f64
    }

    /// Get the estimated false positive rate
    pub fn estimated_false_positive_rate(&self) -> f64 {
        let passed = self.false_positives.load(Ordering::Relaxed)
            + self.true_positives.load(Ordering::Relaxed);
        if passed == 0 {
            return 0.0;
        }
        let fp = self.false_positives.load(Ordering::Relaxed);
        fp as f64 / passed as f64
    }

    /// Get total number of checks
    pub fn total_checks(&self) -> u64 {
        self.total_checks.load(Ordering::Relaxed)
    }

    /// Get total true negatives
    pub fn true_negatives(&self) -> u64 {
        self.true_negatives.load(Ordering::Relaxed)
    }

    /// Recommend optimal false positive rate based on observed statistics
    ///
    /// Returns a suggested FP rate for new bloom filters based on:
    /// - If true negative rate is high (> 50%), bloom filters are effective
    /// - If true negative rate is low, we might want smaller filters or skip them
    pub fn recommend_false_positive_rate(&self) -> f64 {
        let tn_rate = self.true_negative_rate();
        let total = self.total_checks.load(Ordering::Relaxed);

        // Need sufficient samples to make a recommendation
        if total < 1000 {
            return 0.01; // Default 1%
        }

        if tn_rate > 0.7 {
            // Bloom filters are very effective - can use tighter FP rate
            0.005 // 0.5%
        } else if tn_rate > 0.3 {
            // Moderate effectiveness - default rate
            0.01 // 1%
        } else {
            // Low effectiveness - use higher FP rate (smaller filter)
            // or consider skipping bloom filters entirely
            0.05 // 5%
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.total_checks.store(0, Ordering::Relaxed);
        self.true_negatives.store(0, Ordering::Relaxed);
        self.false_positives.store(0, Ordering::Relaxed);
        self.true_positives.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut bf = BloomFilter::with_capacity(100);

        // Insert some values
        bf.insert(&Value::Integer(42));
        bf.insert(&Value::Integer(100));
        bf.insert(&Value::Text("hello".into()));

        // Check membership
        assert!(bf.might_contain(&Value::Integer(42)));
        assert!(bf.might_contain(&Value::Integer(100)));
        assert!(bf.might_contain(&Value::Text("hello".into())));

        // Values not inserted should (usually) not be found
        // Note: false positives are possible
        let mut false_positives = 0;
        for i in 1000..1100 {
            if bf.might_contain(&Value::Integer(i)) {
                false_positives += 1;
            }
        }
        // With 100 checks and 1% FP rate, expect ~1 false positive
        assert!(
            false_positives < 10,
            "Too many false positives: {}",
            false_positives
        );
    }

    #[test]
    fn test_bloom_filter_no_false_negatives() {
        let mut bf = BloomFilter::with_capacity(1000);

        // Insert 1000 values
        for i in 0..1000 {
            bf.insert(&Value::Integer(i));
        }

        // All inserted values MUST be found (no false negatives)
        for i in 0..1000 {
            assert!(
                bf.might_contain(&Value::Integer(i)),
                "False negative for {}",
                i
            );
        }
    }

    #[test]
    fn test_bloom_filter_false_positive_rate() {
        let mut bf = BloomFilter::new(1000, 0.01); // 1% FP rate

        // Insert 1000 values
        for i in 0..1000 {
            bf.insert(&Value::Integer(i));
        }

        // Check 10000 values NOT in the filter
        let mut false_positives = 0;
        for i in 10000..20000 {
            if bf.might_contain(&Value::Integer(i)) {
                false_positives += 1;
            }
        }

        let actual_fp_rate = false_positives as f64 / 10000.0;
        // Allow 5x the target rate due to statistical variance
        assert!(
            actual_fp_rate < 0.05,
            "FP rate {} too high (target: 0.01)",
            actual_fp_rate
        );
    }

    #[test]
    fn test_bloom_filter_different_types() {
        let mut bf = BloomFilter::with_capacity(100);

        bf.insert(&Value::Integer(42));
        bf.insert(&Value::Float(42.0));
        bf.insert(&Value::Text("42".into()));

        // Different types should hash differently
        assert!(bf.might_contain(&Value::Integer(42)));
        assert!(bf.might_contain(&Value::Float(42.0)));
        assert!(bf.might_contain(&Value::Text("42".into())));

        // These are different values
        // (though false positives are possible)
    }

    #[test]
    fn test_bloom_filter_merge() {
        let mut bf1 = BloomFilter::with_capacity(100);
        let mut bf2 = BloomFilter::with_capacity(100);

        bf1.insert(&Value::Integer(1));
        bf1.insert(&Value::Integer(2));
        bf2.insert(&Value::Integer(3));
        bf2.insert(&Value::Integer(4));

        bf1.merge(&bf2).unwrap();

        // Both sets should be present
        assert!(bf1.might_contain(&Value::Integer(1)));
        assert!(bf1.might_contain(&Value::Integer(2)));
        assert!(bf1.might_contain(&Value::Integer(3)));
        assert!(bf1.might_contain(&Value::Integer(4)));
    }

    #[test]
    fn test_runtime_bloom_filter() {
        let mut builder =
            BloomFilterBuilder::new("customer_id".to_string(), "customers".to_string(), 100);

        for i in 0..100 {
            builder.insert(&Value::Integer(i));
        }

        let runtime_filter = builder.build();

        assert!(runtime_filter.might_match(&Value::Integer(50)));
        assert!(runtime_filter.is_effective());

        let stats = runtime_filter.stats();
        assert_eq!(stats.element_count, 100);
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn test_bloom_filter_edge_computing() {
        // Edge computing filter should use less memory
        let standard = BloomFilter::with_capacity(10000);
        let edge = BloomFilter::for_edge_computing(10000);

        assert!(
            edge.memory_bytes() < standard.memory_bytes(),
            "Edge filter should use less memory: {} vs {}",
            edge.memory_bytes(),
            standard.memory_bytes()
        );
    }

    #[test]
    fn test_fill_ratio() {
        let mut bf = BloomFilter::with_capacity(100);

        assert!(bf.fill_ratio() < 0.01, "Empty filter should have low fill");

        for i in 0..100 {
            bf.insert(&Value::Integer(i));
        }

        let fill = bf.fill_ratio();
        assert!(
            fill > 0.1 && fill < 0.9,
            "Fill ratio should be moderate: {}",
            fill
        );
    }

    #[test]
    fn test_effectiveness_check() {
        // Empty filter is not effective
        let builder = BloomFilterBuilder::new("col".to_string(), "table".to_string(), 100);
        let filter = builder.build();
        assert!(!filter.is_effective());

        // Overfilled filter is not effective
        let mut bf = BloomFilter::new(10, 0.5); // Very small
        for i in 0..1000 {
            bf.insert(&Value::Integer(i));
        }
        let runtime = RuntimeBloomFilter {
            filter: bf,
            column_name: "col".to_string(),
            source_table: "table".to_string(),
        };
        assert!(!runtime.is_effective());
    }
}
