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

//! Bloom filter operator for pre-join filtering.
//!
//! This operator wraps a probe-side input and filters out rows that
//! definitely won't match the build side of a hash join. By checking
//! the bloom filter before the hash table probe, we can skip rows
//! early and reduce hash table lookups.
//!
//! # How It Works
//!
//! 1. The bloom filter is built from the build side of a hash join
//! 2. This operator wraps the probe side input
//! 3. For each probe row, we compute a combined hash of all join keys
//! 4. If `might_match_raw_hash` returns false, the row is definitely not in build side
//! 5. If it returns true, the row MIGHT match (false positives possible)
//!
//! # Performance
//!
//! - **Best case**: High selectivity join (few matches) - filters most rows
//! - **Worst case**: Low selectivity (many matches) - small overhead per row
//! - **Typical gain**: 10-40% fewer hash table probes
//!
//! # Multi-Column Join Support
//!
//! This operator uses the same combined hash function as the join hash table,
//! ensuring correct filtering for both single-key and multi-column joins.
//! The hash is computed once and reused for both bloom filter check and
//! hash table probe.

use crate::core::Result;
use crate::executor::hash_table::hash_keys_with;
use crate::executor::operator::{ColumnInfo, Operator, RowRef};
use crate::optimizer::bloom::RuntimeBloomFilter;

/// Operator that filters rows using a bloom filter before hash join probe.
///
/// This is a streaming filter that sits between the probe source and
/// the hash join operator, eliminating rows that definitely won't match.
pub struct BloomFilterOperator {
    /// Child operator (probe source)
    child: Box<dyn Operator>,
    /// Bloom filter built from join build side
    bloom_filter: RuntimeBloomFilter,
    /// Column indices of the join keys in the probe schema
    key_indices: Vec<usize>,
    /// Whether the operator has been opened
    opened: bool,
    /// Statistics: total rows checked
    rows_checked: u64,
    /// Statistics: rows that passed the filter
    rows_passed: u64,
}

impl BloomFilterOperator {
    /// Create a new bloom filter operator.
    ///
    /// # Arguments
    ///
    /// * `child` - The probe side input operator
    /// * `bloom_filter` - Bloom filter built from join build side
    /// * `key_indices` - Indices of the join key columns in probe schema
    pub fn new(
        child: Box<dyn Operator>,
        bloom_filter: RuntimeBloomFilter,
        key_indices: Vec<usize>,
    ) -> Self {
        Self {
            child,
            bloom_filter,
            key_indices,
            opened: false,
            rows_checked: 0,
            rows_passed: 0,
        }
    }

    /// Get the number of rows checked by the filter.
    #[allow(dead_code)]
    pub fn rows_checked(&self) -> u64 {
        self.rows_checked
    }

    /// Get the number of rows that passed the filter.
    #[allow(dead_code)]
    pub fn rows_passed(&self) -> u64 {
        self.rows_passed
    }

    /// Get the filter selectivity (fraction of rows that passed).
    #[allow(dead_code)]
    pub fn selectivity(&self) -> f64 {
        if self.rows_checked == 0 {
            1.0
        } else {
            self.rows_passed as f64 / self.rows_checked as f64
        }
    }
}

impl Operator for BloomFilterOperator {
    fn open(&mut self) -> Result<()> {
        self.child.open()?;
        self.opened = true;
        self.rows_checked = 0;
        self.rows_passed = 0;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if !self.opened {
            return Ok(None);
        }

        // Stream through child, filtering rows that definitely don't match
        while let Some(row_ref) = self.child.next()? {
            self.rows_checked += 1;

            // Check for NULL keys - pass through to let hash join handle NULL semantics
            let has_null = self
                .key_indices
                .iter()
                .any(|&idx| row_ref.get(idx).is_none());

            if has_null {
                self.rows_passed += 1;
                return Ok(Some(row_ref));
            }

            // Compute combined hash using same algorithm as hash table build
            let hash = hash_keys_with(&self.key_indices, |idx| row_ref.get(idx));

            // Check bloom filter - if definitely not in build side, skip
            if self.bloom_filter.might_match_raw_hash(hash) {
                // Might match - pass through to hash join
                self.rows_passed += 1;
                return Ok(Some(row_ref));
            }
            // Definitely no match - skip this row (bloom filter true negative)
        }

        // No more rows from child
        Ok(None)
    }

    fn close(&mut self) -> Result<()> {
        self.opened = false;
        self.child.close()
    }

    fn schema(&self) -> &[ColumnInfo] {
        // Schema passes through unchanged
        self.child.schema()
    }

    fn estimated_rows(&self) -> Option<usize> {
        // Estimate filtered rows using bloom filter's expected selectivity
        self.child.estimated_rows().map(|count| {
            let selectivity = self.bloom_filter.estimated_selectivity();
            (count as f64 * selectivity).ceil() as usize
        })
    }

    fn name(&self) -> &str {
        "BloomFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Row, Value};
    use crate::executor::hash_table::hash_row_keys;
    use crate::executor::operator::MaterializedOperator;
    use crate::optimizer::bloom::BloomFilterBuilder;

    #[test]
    fn test_bloom_filter_operator_basic() {
        // Build rows to insert into bloom filter
        let build_rows = vec![
            Row::from_values(vec![Value::Integer(1)]),
            Row::from_values(vec![Value::Integer(2)]),
            Row::from_values(vec![Value::Integer(3)]),
        ];
        let key_indices = vec![0usize];

        // Build bloom filter using same FxHash as hash table
        let mut builder = BloomFilterBuilder::new("id".to_string(), "test".to_string(), 3);
        for row in &build_rows {
            let hash = hash_row_keys(row, &key_indices);
            builder.insert_raw_hash(hash);
        }
        let bloom = builder.build();

        // Create probe rows: 1, 2, 3, 4, 5
        let probe_rows = vec![
            Row::from_values(vec![Value::Integer(1)]),
            Row::from_values(vec![Value::Integer(2)]),
            Row::from_values(vec![Value::Integer(3)]),
            Row::from_values(vec![Value::Integer(4)]),
            Row::from_values(vec![Value::Integer(5)]),
        ];

        let schema = vec![ColumnInfo::new("id")];
        let child = Box::new(MaterializedOperator::new(probe_rows, schema));

        let mut op = BloomFilterOperator::new(child, bloom, vec![0]);

        // Open and collect results
        op.open().unwrap();
        let mut results = Vec::new();
        while let Some(row_ref) = op.next().unwrap() {
            results.push(row_ref.into_owned());
        }
        op.close().unwrap();

        // Should have 1, 2, 3 for sure (they're in the filter)
        // 4, 5 should be filtered out (they're definitely not in the filter)
        // Note: Due to false positives, 4 or 5 might pass, but usually won't
        assert!(results.len() >= 3, "Should have at least 3 rows that match");
        assert!(results.len() <= 5, "Should have at most 5 rows");

        // Verify statistics
        assert_eq!(op.rows_checked(), 5);
        assert!(op.rows_passed() >= 3);
    }

    #[test]
    fn test_bloom_filter_operator_empty_input() {
        let builder = BloomFilterBuilder::new("id".to_string(), "test".to_string(), 1);
        let bloom = builder.build();

        let probe_rows: Vec<Row> = vec![];
        let schema = vec![ColumnInfo::new("id")];
        let child = Box::new(MaterializedOperator::new(probe_rows, schema));

        let mut op = BloomFilterOperator::new(child, bloom, vec![0]);

        op.open().unwrap();
        let result = op.next().unwrap();
        op.close().unwrap();

        assert!(result.is_none());
        assert_eq!(op.rows_checked(), 0);
    }
}
