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

//! Pre-computed aggregate statistics per frozen volume.
//!
//! These are computed once during the freeze (seal) operation and stored
//! alongside the volume data. Queries like `SELECT COUNT(*) FROM t` or
//! `SELECT MAX(time) FROM t` can be answered instantly by summing stats
//! across volumes without scanning any row data.

use crate::core::Value;

/// Pre-computed aggregate stats for a single column within a volume.
#[derive(Debug, Clone)]
pub struct ColumnAggregateStats {
    /// Sum of all integer values (using i128 to avoid overflow)
    pub sum_int: i128,
    /// Sum of all float values
    pub sum_float: f64,
    /// Number of non-null numeric values (for AVG computation)
    pub numeric_count: u64,
    /// Minimum non-null value
    pub min: Value,
    /// Maximum non-null value
    pub max: Value,
    /// Number of non-null values
    pub non_null_count: u64,
}

impl Default for ColumnAggregateStats {
    fn default() -> Self {
        Self {
            sum_int: 0,
            sum_float: 0.0,
            numeric_count: 0,
            min: Value::Null(crate::core::DataType::Null),
            max: Value::Null(crate::core::DataType::Null),
            non_null_count: 0,
        }
    }
}

impl ColumnAggregateStats {
    /// Accumulate a value into the stats.
    pub fn accumulate(&mut self, value: &Value) {
        if value.is_null() {
            return;
        }
        self.non_null_count += 1;

        match value {
            Value::Integer(i) => {
                self.sum_int += *i as i128;
                self.numeric_count += 1;
            }
            Value::Float(f) => {
                self.sum_float += *f;
                self.numeric_count += 1;
            }
            _ => {}
        }

        // Update min
        if self.min.is_null() {
            self.min = value.clone();
        } else if let Ok(std::cmp::Ordering::Less) = value.compare(&self.min) {
            self.min = value.clone();
        }

        // Update max
        if self.max.is_null() {
            self.max = value.clone();
        } else if let Ok(std::cmp::Ordering::Greater) = value.compare(&self.max) {
            self.max = value.clone();
        }
    }

    /// Merge stats from another volume (for multi-volume aggregation).
    pub fn merge(&mut self, other: &ColumnAggregateStats) {
        self.sum_int += other.sum_int;
        self.sum_float += other.sum_float;
        self.numeric_count += other.numeric_count;
        self.non_null_count += other.non_null_count;

        if other.min.is_null() {
            return;
        }

        if self.min.is_null() {
            self.min = other.min.clone();
        } else if let Ok(std::cmp::Ordering::Less) = other.min.compare(&self.min) {
            self.min = other.min.clone();
        }

        if self.max.is_null() {
            self.max = other.max.clone();
        } else if let Ok(std::cmp::Ordering::Greater) = other.max.compare(&self.max) {
            self.max = other.max.clone();
        }
    }

    /// Get the sum as f64 (combining integer and float sums).
    /// Note: loses precision for i128 sums beyond 2^53. Prefer `sum_parts()` when possible.
    pub fn sum_as_f64(&self) -> f64 {
        self.sum_int as f64 + self.sum_float
    }

    /// Get the separate integer and float sum components (no precision loss).
    pub fn sum_parts(&self) -> (i128, f64) {
        (self.sum_int, self.sum_float)
    }

    /// Get the average value.
    pub fn avg(&self) -> Option<f64> {
        if self.numeric_count == 0 {
            None
        } else {
            Some(self.sum_as_f64() / self.numeric_count as f64)
        }
    }
}

/// Aggregate stats for the entire volume.
#[derive(Debug, Clone)]
pub struct VolumeAggregateStats {
    /// Total number of rows in the volume (including deleted)
    pub total_rows: u64,
    /// Number of live (non-deleted) rows
    pub live_rows: u64,
    /// Per-column aggregate stats
    pub columns: Vec<ColumnAggregateStats>,
}

impl VolumeAggregateStats {
    /// Create stats for a volume with the given number of columns.
    pub fn new(num_columns: usize) -> Self {
        Self {
            total_rows: 0,
            live_rows: 0,
            columns: (0..num_columns)
                .map(|_| ColumnAggregateStats::default())
                .collect(),
        }
    }

    /// Merge stats from another volume.
    pub fn merge(&mut self, other: &VolumeAggregateStats) {
        self.total_rows += other.total_rows;
        self.live_rows += other.live_rows;
        for (i, col) in self.columns.iter_mut().enumerate() {
            if i < other.columns.len() {
                col.merge(&other.columns[i]);
            }
        }
    }

    /// Get COUNT(*) for this volume.
    #[inline]
    pub fn count_star(&self) -> u64 {
        self.live_rows
    }

    /// Get SUM(col) for a specific column.
    #[inline]
    pub fn sum(&self, col_idx: usize) -> f64 {
        self.columns[col_idx].sum_as_f64()
    }

    /// Get MIN(col) for a specific column.
    #[inline]
    pub fn min(&self, col_idx: usize) -> &Value {
        &self.columns[col_idx].min
    }

    /// Get MAX(col) for a specific column.
    #[inline]
    pub fn max(&self, col_idx: usize) -> &Value {
        &self.columns[col_idx].max
    }

    /// Get AVG(col) for a specific column.
    #[inline]
    pub fn avg(&self, col_idx: usize) -> Option<f64> {
        self.columns[col_idx].avg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_stats_accumulate() {
        let mut stats = ColumnAggregateStats::default();
        stats.accumulate(&Value::Integer(10));
        stats.accumulate(&Value::Integer(20));
        stats.accumulate(&Value::Integer(30));

        assert_eq!(stats.sum_int, 60);
        assert_eq!(stats.numeric_count, 3);
        assert_eq!(stats.non_null_count, 3);
        assert_eq!(stats.min, Value::Integer(10));
        assert_eq!(stats.max, Value::Integer(30));
        assert_eq!(stats.avg(), Some(20.0));
    }

    #[test]
    fn test_column_stats_with_nulls() {
        let mut stats = ColumnAggregateStats::default();
        stats.accumulate(&Value::Float(1.5));
        stats.accumulate(&Value::Null(crate::core::DataType::Float));
        stats.accumulate(&Value::Float(3.5));

        assert_eq!(stats.numeric_count, 2);
        assert_eq!(stats.non_null_count, 2);
        assert_eq!(stats.sum_float, 5.0);
    }

    #[test]
    fn test_volume_stats_merge() {
        let mut s1 = VolumeAggregateStats::new(2);
        s1.live_rows = 100;
        s1.columns[0].accumulate(&Value::Integer(10));
        s1.columns[0].accumulate(&Value::Integer(50));

        let mut s2 = VolumeAggregateStats::new(2);
        s2.live_rows = 200;
        s2.columns[0].accumulate(&Value::Integer(5));
        s2.columns[0].accumulate(&Value::Integer(100));

        s1.merge(&s2);

        assert_eq!(s1.count_star(), 300);
        assert_eq!(s1.min(0), &Value::Integer(5));
        assert_eq!(s1.max(0), &Value::Integer(100));
        assert_eq!(s1.sum(0), 165.0);
    }
}
