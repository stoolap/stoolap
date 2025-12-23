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
//! 3. For each probe row, we check if the join key MIGHT be in the filter
//! 4. If `might_match` returns false, the row is definitely not in build side
//! 5. If `might_match` returns true, the row MIGHT match (false positives possible)
//!
//! # Performance
//!
//! - **Best case**: High selectivity join (few matches) - filters most rows
//! - **Worst case**: Low selectivity (many matches) - small overhead per row
//! - **Typical gain**: 10-40% fewer hash table probes
//!
//! # Limitations
//!
//! **Single-key filtering only**: For multi-column joins (e.g., `ON a.x = b.x AND a.y = b.y`),
//! the bloom filter only checks the first join key. This is a deliberate tradeoff:
//!
//! - **Why single key**: Combining multiple keys into a composite hash would require
//!   allocating a tuple or computing a combined hash, adding overhead per row.
//!   Single-key filtering is simpler and still effective for most joins.
//!
//! - **Impact**: For multi-column joins, the filter may have more false positives
//!   since it only filters on one dimension. Rows that don't match on the first
//!   key are filtered; rows that match the first key but not subsequent keys
//!   will pass through to the hash join (which handles them correctly).
//!
//! - **Future improvement**: Could extend to use a composite bloom filter that
//!   hashes all join keys together, reducing false positives for multi-key joins.

use crate::core::Result;
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
    /// Column index of the join key in the probe schema
    key_column_idx: usize,
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
    /// * `key_column_idx` - Index of the join key column in probe schema
    pub fn new(
        child: Box<dyn Operator>,
        bloom_filter: RuntimeBloomFilter,
        key_column_idx: usize,
    ) -> Self {
        Self {
            child,
            bloom_filter,
            key_column_idx,
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

            // Get the key value from the row
            if let Some(value) = row_ref.get(self.key_column_idx) {
                // Check bloom filter - if definitely not in build side, skip
                if self.bloom_filter.might_match(value) {
                    // Might match - pass through to hash join
                    self.rows_passed += 1;
                    return Ok(Some(row_ref));
                }
                // Definitely no match - skip this row (bloom filter true negative)
            } else {
                // NULL key - pass through (let hash join handle NULL semantics)
                self.rows_passed += 1;
                return Ok(Some(row_ref));
            }
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
    use crate::executor::operator::MaterializedOperator;
    use crate::optimizer::bloom::BloomFilterBuilder;

    #[test]
    fn test_bloom_filter_operator_basic() {
        // Build a bloom filter with values 1, 2, 3
        let mut builder = BloomFilterBuilder::new("id".to_string(), "test".to_string(), 3);
        builder.insert(&Value::Integer(1));
        builder.insert(&Value::Integer(2));
        builder.insert(&Value::Integer(3));
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

        let mut op = BloomFilterOperator::new(child, bloom, 0);

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

        let mut op = BloomFilterOperator::new(child, bloom, 0);

        op.open().unwrap();
        let result = op.next().unwrap();
        op.close().unwrap();

        assert!(result.is_none());
        assert_eq!(op.rows_checked(), 0);
    }
}
