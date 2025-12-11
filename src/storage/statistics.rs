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

//! Statistics infrastructure for query optimization
//!
//! This module provides statistics collection and storage for cost-based
//! query optimization. Statistics are stored in system tables and collected
//! via the ANALYZE command.
//!
//! ## System Tables
//!
//! - `_sys_table_stats` - Table-level statistics (row count, page count, etc.)
//! - `_sys_column_stats` - Column-level statistics (distinct count, min/max, histogram)
//!
//! ## Usage
//!
//! Statistics are collected by running `ANALYZE table_name` which scans the table
//! and populates the system tables. The query planner retrieves these statistics
//! to estimate cardinalities and choose optimal access paths.

use crate::core::Value;

/// System table name for table-level statistics
pub const SYS_TABLE_STATS: &str = "_sys_table_stats";

/// System table name for column-level statistics
pub const SYS_COLUMN_STATS: &str = "_sys_column_stats";

/// SQL to create the table statistics system table
/// Note: Stoolap requires INTEGER PRIMARY KEY, so we use an auto-increment id
/// and a unique index on table_name
pub const CREATE_TABLE_STATS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS _sys_table_stats (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    table_name TEXT NOT NULL UNIQUE,
    row_count INTEGER NOT NULL DEFAULT 0,
    page_count INTEGER NOT NULL DEFAULT 0,
    avg_row_size INTEGER NOT NULL DEFAULT 0,
    last_analyzed TIMESTAMP
)
"#;

/// SQL to create the column statistics system table
/// Note: We rely on DELETE before INSERT to maintain uniqueness on (table_name, column_name)
pub const CREATE_COLUMN_STATS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS _sys_column_stats (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    null_count INTEGER NOT NULL DEFAULT 0,
    distinct_count INTEGER NOT NULL DEFAULT 0,
    min_value TEXT,
    max_value TEXT,
    avg_width INTEGER NOT NULL DEFAULT 0,
    histogram TEXT
)
"#;

/// Number of histogram buckets (default)
/// Using a small number for edge computing efficiency
pub const DEFAULT_HISTOGRAM_BUCKETS: usize = 32;

/// Equi-depth histogram for range selectivity estimation
///
/// Each bucket contains approximately the same number of values.
/// This provides better selectivity estimates for skewed data distributions.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bucket boundaries (n+1 values for n buckets)
    /// boundaries[i] is the inclusive lower bound for bucket i
    /// boundaries[i+1] is the exclusive upper bound for bucket i
    pub boundaries: Vec<Value>,
    /// Number of values per bucket (approximately equal for equi-depth)
    pub rows_per_bucket: u64,
    /// Total number of values represented
    pub total_rows: u64,
}

impl Histogram {
    /// Build an equi-depth histogram from sorted values
    ///
    /// The input values must be sorted in ascending order.
    pub fn from_sorted_values(values: &[Value], num_buckets: usize) -> Option<Self> {
        if values.is_empty() || num_buckets == 0 {
            return None;
        }

        // Skip nulls - they're counted separately
        let non_null_values: Vec<_> = values.iter().filter(|v| !v.is_null()).collect();
        if non_null_values.is_empty() {
            return None;
        }

        let total_rows = non_null_values.len() as u64;
        let rows_per_bucket = (total_rows / num_buckets as u64).max(1);

        // Create bucket boundaries for equi-depth histogram
        let mut boundaries = Vec::with_capacity(num_buckets + 1);

        // First boundary is the minimum value
        boundaries.push(non_null_values[0].clone());

        // Add boundaries at regular intervals
        for i in 1..num_buckets {
            let idx = (i * non_null_values.len() / num_buckets).min(non_null_values.len() - 1);
            let boundary = non_null_values[idx].clone();
            // Only add if different from the last boundary
            if boundaries.last() != Some(&boundary) {
                boundaries.push(boundary);
            }
        }

        // Last boundary is the maximum value (exclusive upper bound sentinel)
        let last_value = (*non_null_values.last().unwrap()).clone();
        if boundaries.last() != Some(&last_value) {
            boundaries.push(last_value);
        }

        Some(Self {
            boundaries,
            rows_per_bucket,
            total_rows,
        })
    }

    /// Build an adaptive histogram with bucket count based on data characteristics
    ///
    /// For highly skewed data (low distinct ratio), uses more buckets to capture distribution.
    /// For uniform data (high distinct ratio), uses fewer buckets since distribution is predictable.
    ///
    /// # Arguments
    /// * `values` - Sorted values to build histogram from
    /// * `distinct_count` - Number of distinct values in the data
    /// * `min_buckets` - Minimum number of buckets (default: 10)
    /// * `max_buckets` - Maximum number of buckets (default: 200)
    pub fn adaptive_from_sorted_values(
        values: &[Value],
        distinct_count: u64,
        min_buckets: usize,
        max_buckets: usize,
    ) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let total_rows = values.iter().filter(|v| !v.is_null()).count() as u64;
        if total_rows == 0 {
            return None;
        }

        // Calculate the optimal bucket count based on data characteristics
        //
        // Key insight: The number of buckets should scale with the "spread" of the data:
        // - For low-cardinality data (distinct_count << total_rows), we need more buckets
        //   to accurately capture the distribution of frequently-occurring values
        // - For high-cardinality data (distinct_count ≈ total_rows), fewer buckets suffice
        //   since each value is roughly equally likely
        //
        // Formula: buckets = min(max_buckets, max(min_buckets, sqrt(distinct_count) * skew_factor))
        // where skew_factor = log2(total_rows / distinct_count + 1)

        let distinct = distinct_count.max(1) as f64;
        let rows = total_rows as f64;

        // Skew factor: higher when data is more skewed (many duplicates)
        let skew_factor = (rows / distinct + 1.0).log2();

        // Base bucket count from sqrt of distinct values (information-theoretic bound)
        let base_buckets = distinct.sqrt();

        // Final bucket count with skew adjustment
        let optimal_buckets = (base_buckets * skew_factor).round() as usize;
        let num_buckets = optimal_buckets.clamp(min_buckets, max_buckets);

        Self::from_sorted_values(values, num_buckets)
    }

    /// Estimate selectivity for a range predicate using the histogram
    ///
    /// Returns the fraction of rows that satisfy:
    /// - For Lt/Le: value < bound or value <= bound
    /// - For Gt/Ge: value > bound or value >= bound
    /// - For Eq: value = bound (uses bucket containing the value)
    pub fn estimate_selectivity(&self, value: &Value, operator: HistogramOp) -> f64 {
        if self.boundaries.is_empty() || self.total_rows == 0 {
            return 0.5; // Fallback
        }

        let num_buckets = self.boundaries.len().saturating_sub(1).max(1);

        // Find which bucket contains this value
        let bucket_idx = self.find_bucket(value);

        match operator {
            HistogramOp::Equal => {
                // Equality: estimate as 1/distinct_in_bucket, use 1/rows_per_bucket
                1.0 / self.rows_per_bucket.max(1) as f64
            }
            HistogramOp::LessThan | HistogramOp::LessThanOrEqual => {
                // All buckets before this one + fraction of this bucket
                let full_buckets = bucket_idx as f64;
                let bucket_fraction = self.fraction_in_bucket(value, bucket_idx);
                let effective_buckets = full_buckets + bucket_fraction;
                (effective_buckets / num_buckets as f64).clamp(0.0, 1.0)
            }
            HistogramOp::GreaterThan | HistogramOp::GreaterThanOrEqual => {
                // Remaining buckets after this one + fraction of this bucket
                let remaining_buckets = (num_buckets - bucket_idx - 1) as f64;
                let bucket_fraction = 1.0 - self.fraction_in_bucket(value, bucket_idx);
                let effective_buckets = remaining_buckets + bucket_fraction;
                (effective_buckets / num_buckets as f64).clamp(0.0, 1.0)
            }
        }
    }

    /// Find which bucket contains a value (binary search)
    fn find_bucket(&self, value: &Value) -> usize {
        if self.boundaries.is_empty() {
            return 0;
        }

        // Binary search for the bucket
        let mut low = 0;
        let mut high = self.boundaries.len().saturating_sub(1);

        while low < high {
            let mid = (low + high).div_ceil(2);
            if mid < self.boundaries.len() && &self.boundaries[mid] <= value {
                low = mid;
            } else {
                high = mid.saturating_sub(1);
            }
        }

        // Return bucket index (capped to valid range)
        low.min(self.boundaries.len().saturating_sub(2))
    }

    /// Estimate what fraction of a bucket is below a given value
    fn fraction_in_bucket(&self, value: &Value, bucket_idx: usize) -> f64 {
        if bucket_idx >= self.boundaries.len().saturating_sub(1) {
            return 1.0;
        }

        // Get bucket bounds
        let lower = &self.boundaries[bucket_idx];
        let upper = if bucket_idx + 1 < self.boundaries.len() {
            &self.boundaries[bucket_idx + 1]
        } else {
            return 1.0;
        };

        // Estimate fraction based on value position within bucket
        // For numeric types, use linear interpolation
        match (lower, upper, value) {
            (Value::Integer(lo), Value::Integer(hi), Value::Integer(v)) => {
                if hi == lo {
                    0.5
                } else {
                    ((*v - *lo) as f64 / (*hi - *lo) as f64).clamp(0.0, 1.0)
                }
            }
            (Value::Float(lo), Value::Float(hi), Value::Float(v)) => {
                if (hi - lo).abs() < f64::EPSILON {
                    0.5
                } else {
                    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
                }
            }
            _ => 0.5, // Default to middle of bucket for non-numeric types
        }
    }

    /// Estimate selectivity for a BETWEEN range predicate
    ///
    /// Returns the fraction of rows where low <= value <= high.
    /// Uses bucket walk algorithm for accurate estimation.
    pub fn estimate_range_selectivity(&self, low: &Value, high: &Value) -> f64 {
        if self.boundaries.is_empty() || self.total_rows == 0 {
            return 0.33; // Default range selectivity
        }

        let num_buckets = self.boundaries.len().saturating_sub(1).max(1);

        // Find bucket indices for low and high bounds
        let low_bucket = self.find_bucket(low);
        let high_bucket = self.find_bucket(high);

        // If low > high, return 0
        if low_bucket > high_bucket {
            return 0.0;
        }

        // Same bucket - use fraction within bucket
        if low_bucket == high_bucket {
            let low_frac = self.fraction_in_bucket(low, low_bucket);
            let high_frac = self.fraction_in_bucket(high, high_bucket);
            let bucket_coverage = (high_frac - low_frac).max(0.0);
            return (bucket_coverage / num_buckets as f64).clamp(0.0001, 1.0);
        }

        // Multiple buckets
        let mut total_coverage = 0.0;

        // First bucket: fraction from low to end
        let low_frac = self.fraction_in_bucket(low, low_bucket);
        total_coverage += 1.0 - low_frac;

        // Full buckets in between
        if high_bucket > low_bucket + 1 {
            total_coverage += (high_bucket - low_bucket - 1) as f64;
        }

        // Last bucket: fraction from start to high
        let high_frac = self.fraction_in_bucket(high, high_bucket);
        total_coverage += high_frac;

        (total_coverage / num_buckets as f64).clamp(0.0001, 1.0)
    }

    /// Serialize histogram to JSON string for storage
    pub fn to_json(&self) -> String {
        let boundary_strs: Vec<String> = self.boundaries.iter().map(|v| v.to_string()).collect();
        format!(
            r#"{{"boundaries":[{}],"rows_per_bucket":{},"total_rows":{}}}"#,
            boundary_strs
                .iter()
                .map(|s| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(","),
            self.rows_per_bucket,
            self.total_rows
        )
    }

    /// Parse histogram from JSON string
    pub fn from_json(json: &str) -> Option<Self> {
        // Simple JSON parsing for histogram format
        // Format: {"boundaries":["v1","v2",...],"rows_per_bucket":N,"total_rows":N}
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return None;
        }

        // Extract rows_per_bucket
        let rows_per_bucket = extract_number(json, "rows_per_bucket")?;
        let total_rows = extract_number(json, "total_rows")?;

        // Extract boundaries array
        let boundaries = extract_value_array(json, "boundaries")?;

        Some(Self {
            boundaries,
            rows_per_bucket,
            total_rows,
        })
    }
}

/// Helper function to extract a number from simple JSON
fn extract_number(json: &str, key: &str) -> Option<u64> {
    let key_pattern = format!("\"{}\":", key);
    let start = json.find(&key_pattern)? + key_pattern.len();
    let rest = &json[start..];
    let end = rest.find([',', '}'])?;
    rest[..end].trim().parse().ok()
}

/// Helper function to extract a Value array from simple JSON
fn extract_value_array(json: &str, key: &str) -> Option<Vec<Value>> {
    let key_pattern = format!("\"{}\":[", key);
    let start = json.find(&key_pattern)? + key_pattern.len();
    let rest = &json[start..];
    let end = rest.find(']')?;
    let array_content = &rest[..end];

    let mut values = Vec::new();
    for item in array_content.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        // Remove surrounding quotes
        let item = item.trim_matches('"');
        // Parse as Value (try integer first, then float, then text)
        if let Ok(i) = item.parse::<i64>() {
            values.push(Value::Integer(i));
        } else if let Ok(f) = item.parse::<f64>() {
            values.push(Value::Float(f));
        } else {
            values.push(Value::Text(item.into()));
        }
    }

    Some(values)
}

/// Histogram comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistogramOp {
    Equal,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

/// Maximum number of rows to sample for statistics
/// For large tables, we sample instead of scanning everything
pub const DEFAULT_SAMPLE_SIZE: usize = 10000;

/// Table-level statistics (in-memory representation)
#[derive(Debug, Clone, Default)]
pub struct TableStats {
    /// Table name
    pub table_name: String,
    /// Estimated row count
    pub row_count: u64,
    /// Number of pages/blocks (for I/O cost estimation)
    pub page_count: u64,
    /// Average row size in bytes
    pub avg_row_size: u64,
}

impl TableStats {
    /// Create new empty table statistics
    pub fn new(table_name: String) -> Self {
        Self {
            table_name,
            row_count: 0,
            page_count: 0,
            avg_row_size: 0,
        }
    }

    /// Get the selectivity for an equality predicate
    /// Returns 1/row_count or 0.1 as default
    pub fn equality_selectivity(&self, distinct_count: u64) -> f64 {
        if distinct_count > 0 {
            1.0 / distinct_count as f64
        } else if self.row_count > 0 {
            1.0 / self.row_count as f64
        } else {
            0.1
        }
    }
}

/// Column-level statistics (in-memory representation)
#[derive(Debug, Clone, Default)]
pub struct ColumnStats {
    /// Column name
    pub column_name: String,
    /// Number of NULL values
    pub null_count: u64,
    /// Number of distinct values (approximate)
    pub distinct_count: u64,
    /// Minimum value (for range estimation)
    pub min_value: Option<Value>,
    /// Maximum value (for range estimation)
    pub max_value: Option<Value>,
    /// Average value width in bytes (for memory estimation)
    pub avg_width: u32,
    /// Histogram buckets as JSON string
    pub histogram: Option<String>,
}

impl ColumnStats {
    /// Create new empty column statistics
    pub fn new(column_name: String) -> Self {
        Self {
            column_name,
            null_count: 0,
            distinct_count: 0,
            min_value: None,
            max_value: None,
            avg_width: 0,
            histogram: None,
        }
    }

    /// Check if statistics are empty (never analyzed)
    pub fn is_empty(&self) -> bool {
        self.distinct_count == 0 && self.min_value.is_none() && self.max_value.is_none()
    }

    /// Parse and return the histogram if available
    pub fn parsed_histogram(&self) -> Option<Histogram> {
        self.histogram
            .as_ref()
            .and_then(|json| Histogram::from_json(json))
    }

    /// Set histogram from a Histogram struct
    pub fn set_histogram(&mut self, histogram: &Histogram) {
        self.histogram = Some(histogram.to_json());
    }
}

/// Selectivity estimation utilities
pub struct SelectivityEstimator;

impl SelectivityEstimator {
    /// Estimate selectivity for equality predicate (column = value)
    /// Formula: 1 / distinct_count
    pub fn equality(distinct_count: u64) -> f64 {
        if distinct_count > 0 {
            1.0 / distinct_count as f64
        } else {
            0.1 // default
        }
    }

    /// Advanced histogram-based join cardinality estimation
    ///
    /// This method "walks" both histograms to estimate the overlap between
    /// two columns being joined. This is far more accurate than the naive
    /// `|R| * |S| / max(distinct)` formula, especially for:
    /// - Skewed data distributions
    /// - Non-overlapping ranges
    /// - Partially overlapping ranges
    ///
    /// # Algorithm
    /// For each bucket pair, we estimate:
    /// 1. The overlap ratio between the bucket ranges
    /// 2. The number of matching rows based on key distributions
    ///
    /// # Returns
    /// Estimated number of output rows from the join
    pub fn histogram_join_cardinality(
        left_stats: &ColumnStats,
        right_stats: &ColumnStats,
        left_rows: u64,
        right_rows: u64,
    ) -> u64 {
        // Try to get histograms
        let left_hist = left_stats.parsed_histogram();
        let right_hist = right_stats.parsed_histogram();

        match (left_hist, right_hist) {
            (Some(lh), Some(rh)) => {
                Self::join_cardinality_from_histograms(&lh, &rh, left_rows, right_rows)
            }
            // Fall back to range-based estimation if we have min/max
            _ => Self::join_cardinality_from_ranges(left_stats, right_stats, left_rows, right_rows),
        }
    }

    /// Histogram-based join cardinality using bucket walk algorithm
    ///
    /// This implements a merge-style walk over both histograms to compute
    /// overlap-weighted join cardinality. The key insight is:
    ///
    /// For an equi-join: output ≈ |L| × |R| / max(NDV_L, NDV_R)
    /// With histograms, we can weight this by overlap ratio per bucket.
    fn join_cardinality_from_histograms(
        left_hist: &Histogram,
        right_hist: &Histogram,
        left_rows: u64,
        right_rows: u64,
    ) -> u64 {
        if left_hist.boundaries.len() < 2 || right_hist.boundaries.len() < 2 {
            // Not enough buckets - fall back to naive
            return Self::join_cardinality(
                left_rows,
                right_rows,
                left_hist.total_rows.max(1),
                right_hist.total_rows.max(1),
            );
        }

        // Get overall value ranges
        let left_min = &left_hist.boundaries[0];
        let left_max = &left_hist.boundaries[left_hist.boundaries.len() - 1];
        let right_min = &right_hist.boundaries[0];
        let right_max = &right_hist.boundaries[right_hist.boundaries.len() - 1];

        // Compute overall overlap ratio
        let overall_overlap = Self::bucket_overlap(left_min, left_max, right_min, right_max);

        if overall_overlap < 0.001 {
            // No meaningful overlap
            return 1;
        }

        // For unique keys (NDV ≈ row count), join produces ~min(rows) matching rows
        // For non-unique keys, we use the standard formula weighted by overlap
        let left_ndv = left_hist.total_rows;
        let right_ndv = right_hist.total_rows;

        // Base cardinality: standard formula
        let max_ndv = left_ndv.max(right_ndv).max(1);
        let base_card = (left_rows as u128 * right_rows as u128 / max_ndv as u128) as f64;

        // Weight by overlap
        let weighted_card = base_card * overall_overlap;

        // For 1:1 joins (high selectivity), cap at smaller input
        let min_input = left_rows.min(right_rows) as f64;
        let final_card = weighted_card.min(min_input * overall_overlap * 1.5);

        (final_card.ceil() as u64).max(1)
    }

    /// Compute overlap ratio between two bucket ranges (0.0 to 1.0)
    fn bucket_overlap(left_lo: &Value, left_hi: &Value, right_lo: &Value, right_hi: &Value) -> f64 {
        // Convert to f64 for comparison
        let (llo, lhi) = match (left_lo, left_hi) {
            (Value::Integer(a), Value::Integer(b)) => (*a as f64, *b as f64),
            (Value::Float(a), Value::Float(b)) => (*a, *b),
            _ => return 0.5, // Non-numeric: assume 50% overlap
        };

        let (rlo, rhi) = match (right_lo, right_hi) {
            (Value::Integer(a), Value::Integer(b)) => (*a as f64, *b as f64),
            (Value::Float(a), Value::Float(b)) => (*a, *b),
            _ => return 0.5,
        };

        // No overlap
        if lhi < rlo || rhi < llo {
            return 0.0;
        }

        // Compute overlap range
        let overlap_lo = llo.max(rlo);
        let overlap_hi = lhi.min(rhi);

        if overlap_hi <= overlap_lo {
            return 0.0;
        }

        // Overlap as fraction of the smaller range
        let left_width = (lhi - llo).abs().max(1.0);
        let right_width = (rhi - rlo).abs().max(1.0);
        let overlap_width = overlap_hi - overlap_lo;

        let left_overlap_ratio = overlap_width / left_width;
        let right_overlap_ratio = overlap_width / right_width;

        // Use geometric mean of overlap ratios
        (left_overlap_ratio * right_overlap_ratio).sqrt()
    }

    /// Range-based join cardinality when only min/max are available
    fn join_cardinality_from_ranges(
        left_stats: &ColumnStats,
        right_stats: &ColumnStats,
        left_rows: u64,
        right_rows: u64,
    ) -> u64 {
        // Check for range overlap
        match (
            &left_stats.min_value,
            &left_stats.max_value,
            &right_stats.min_value,
            &right_stats.max_value,
        ) {
            (Some(l_min), Some(l_max), Some(r_min), Some(r_max)) => {
                // Check if ranges overlap at all
                if l_max < r_min || r_max < l_min {
                    // No overlap - join produces no rows
                    return 0;
                }

                // Compute overlap ratio
                let overlap_ratio = Self::range_overlap_ratio(l_min, l_max, r_min, r_max);

                if overlap_ratio < 0.001 {
                    return 0;
                }

                // Apply overlap ratio to naive estimate
                let naive = Self::join_cardinality(
                    left_rows,
                    right_rows,
                    left_stats.distinct_count.max(1),
                    right_stats.distinct_count.max(1),
                );

                ((naive as f64 * overlap_ratio) as u64).max(1)
            }
            _ => {
                // No range info - use naive formula
                Self::join_cardinality(
                    left_rows,
                    right_rows,
                    left_stats.distinct_count.max(1),
                    right_stats.distinct_count.max(1),
                )
            }
        }
    }

    /// Compute overlap ratio between two value ranges
    fn range_overlap_ratio(l_min: &Value, l_max: &Value, r_min: &Value, r_max: &Value) -> f64 {
        // Convert to f64
        let (llo, lhi) = match (l_min, l_max) {
            (Value::Integer(a), Value::Integer(b)) => (*a as f64, *b as f64),
            (Value::Float(a), Value::Float(b)) => (*a, *b),
            _ => return 1.0, // Can't compute, assume full overlap
        };

        let (rlo, rhi) = match (r_min, r_max) {
            (Value::Integer(a), Value::Integer(b)) => (*a as f64, *b as f64),
            (Value::Float(a), Value::Float(b)) => (*a, *b),
            _ => return 1.0,
        };

        // Compute overlap
        let overlap_lo = llo.max(rlo);
        let overlap_hi = lhi.min(rhi);

        if overlap_hi <= overlap_lo {
            return 0.0;
        }

        let total_range = (lhi - llo).abs().max(rhi - rlo).max(1.0);
        let overlap_range = overlap_hi - overlap_lo;

        (overlap_range / total_range).clamp(0.0, 1.0)
    }

    /// Estimate selectivity for range predicate (column > value, column < value)
    /// Using uniform distribution assumption: 1/3 for range predicates
    pub fn range() -> f64 {
        0.33
    }

    /// Estimate selectivity for range predicate using histogram
    ///
    /// If a histogram is available, uses it for accurate estimates.
    /// Otherwise falls back to uniform distribution assumption.
    pub fn range_with_histogram(col_stats: &ColumnStats, value: &Value, op: HistogramOp) -> f64 {
        // Try to use histogram if available
        if let Some(histogram) = col_stats.parsed_histogram() {
            return histogram.estimate_selectivity(value, op);
        }

        // Fall back to min/max based estimation if available
        if let (Some(min_val), Some(max_val)) = (&col_stats.min_value, &col_stats.max_value) {
            // Use linear interpolation between min and max
            let fraction = Self::estimate_position(value, min_val, max_val);

            return match op {
                HistogramOp::Equal => 1.0 / col_stats.distinct_count.max(1) as f64,
                HistogramOp::LessThan | HistogramOp::LessThanOrEqual => fraction,
                HistogramOp::GreaterThan | HistogramOp::GreaterThanOrEqual => 1.0 - fraction,
            };
        }

        // No statistics available - use default
        match op {
            HistogramOp::Equal => 0.1,
            _ => 0.33,
        }
    }

    /// Estimate position of a value between min and max (0.0 to 1.0)
    fn estimate_position(value: &Value, min: &Value, max: &Value) -> f64 {
        match (min, max, value) {
            (Value::Integer(lo), Value::Integer(hi), Value::Integer(v)) => {
                if hi == lo {
                    0.5
                } else {
                    ((*v - *lo) as f64 / (*hi - *lo) as f64).clamp(0.0, 1.0)
                }
            }
            (Value::Float(lo), Value::Float(hi), Value::Float(v)) => {
                if (hi - lo).abs() < f64::EPSILON {
                    0.5
                } else {
                    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
                }
            }
            _ => 0.5, // Default for non-comparable types
        }
    }

    /// Estimate selectivity for LIKE predicate
    /// Prefix patterns are more selective than suffix/infix
    pub fn like(pattern: &str, distinct_count: u64) -> f64 {
        // Prefix-only patterns (e.g., 'abc%') are more selective
        if !pattern.starts_with('%') && pattern.ends_with('%') {
            let prefix_len = pattern.len() - 1;
            if distinct_count > 0 {
                // Estimate based on prefix length
                let prefix_selectivity = (26.0_f64).powi(-(prefix_len as i32));
                return prefix_selectivity.max(1.0 / distinct_count as f64);
            }
            return 0.1;
        }

        // Suffix or infix patterns are less selective
        if pattern.starts_with('%') {
            return 0.25;
        }

        0.15 // Default for mixed patterns
    }

    /// Estimate selectivity for IN list predicate
    /// Formula: list_size / distinct_count
    pub fn in_list(list_size: usize, distinct_count: u64) -> f64 {
        if distinct_count > 0 {
            (list_size as f64 / distinct_count as f64).min(1.0)
        } else {
            (list_size as f64 * 0.1).min(1.0)
        }
    }

    /// Estimate selectivity for IS NULL predicate
    /// Formula: null_count / row_count
    pub fn is_null(null_count: u64, row_count: u64) -> f64 {
        if row_count > 0 {
            null_count as f64 / row_count as f64
        } else {
            0.01
        }
    }

    /// Estimate selectivity for IS NOT NULL predicate
    pub fn is_not_null(null_count: u64, row_count: u64) -> f64 {
        1.0 - Self::is_null(null_count, row_count)
    }

    /// Estimate join cardinality
    /// Formula: |R| * |S| / max(distinct(R.col), distinct(S.col))
    pub fn join_cardinality(
        left_rows: u64,
        right_rows: u64,
        left_distinct: u64,
        right_distinct: u64,
    ) -> u64 {
        let max_distinct = left_distinct.max(right_distinct).max(1);
        (left_rows as u128 * right_rows as u128 / max_distinct as u128) as u64
    }
}

/// Check if a table name is a system statistics table
pub fn is_stats_table(table_name: &str) -> bool {
    let lower = table_name.to_lowercase();
    lower == SYS_TABLE_STATS || lower == SYS_COLUMN_STATS
}

// =============================================================================
// COLUMN CORRELATION TRACKING
// =============================================================================
//
// This is a UNIQUE Stoolap feature that tracks correlations between columns
// for more accurate selectivity estimation.
//
// Problem: Traditional optimizers assume independence:
//   P(A AND B) = P(A) * P(B)
//
// This fails badly for correlated columns:
//   WHERE city = 'NYC' AND state = 'NY'  → Real: 0.02, Naive: 0.0004
//
// Solution: Track correlations and use them in selectivity estimation.

/// Correlation coefficient between two columns (-1.0 to 1.0)
/// - 1.0 = perfect positive correlation (state determines city)
/// - 0.0 = independent (no correlation)
/// - -1.0 = perfect negative correlation
pub type CorrelationCoefficient = f64;

/// Tracks correlations between columns within a table
#[derive(Debug, Clone, Default)]
pub struct ColumnCorrelations {
    /// Correlation coefficients: (col1, col2) -> correlation
    /// Only stores upper triangle (col1 < col2 alphabetically)
    correlations: std::collections::HashMap<(String, String), CorrelationCoefficient>,
    /// Functional dependencies: col1 -> col2 means col1 determines col2
    /// e.g., zip_code -> city (knowing zip tells you the city)
    functional_deps: std::collections::HashMap<String, Vec<String>>,
}

impl ColumnCorrelations {
    /// Create empty correlation tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a correlation between two columns
    pub fn add_correlation(&mut self, col1: &str, col2: &str, coefficient: CorrelationCoefficient) {
        let (c1, c2) = Self::normalize_pair(col1, col2);
        self.correlations.insert(
            (c1.to_string(), c2.to_string()),
            coefficient.clamp(-1.0, 1.0),
        );
    }

    /// Add a functional dependency: determinant -> dependent
    /// e.g., add_functional_dep("zip_code", "city")
    pub fn add_functional_dep(&mut self, determinant: &str, dependent: &str) {
        self.functional_deps
            .entry(determinant.to_string())
            .or_default()
            .push(dependent.to_string());
        // Functional dependency implies correlation = 1.0
        self.add_correlation(determinant, dependent, 1.0);
    }

    /// Get correlation between two columns (0.0 if unknown)
    pub fn get_correlation(&self, col1: &str, col2: &str) -> CorrelationCoefficient {
        let (c1, c2) = Self::normalize_pair(col1, col2);
        self.correlations
            .get(&(c1.to_string(), c2.to_string()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Check if col1 functionally determines col2
    pub fn determines(&self, col1: &str, col2: &str) -> bool {
        self.functional_deps
            .get(col1)
            .map(|deps| deps.iter().any(|d| d == col2))
            .unwrap_or(false)
    }

    /// Get all columns correlated with the given column
    pub fn correlated_columns(&self, col: &str) -> Vec<(&str, CorrelationCoefficient)> {
        self.correlations
            .iter()
            .filter_map(|((c1, c2), coef)| {
                if c1 == col {
                    Some((c2.as_str(), *coef))
                } else if c2 == col {
                    Some((c1.as_str(), *coef))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Normalize column pair for consistent key ordering
    fn normalize_pair<'a>(col1: &'a str, col2: &'a str) -> (&'a str, &'a str) {
        if col1 <= col2 {
            (col1, col2)
        } else {
            (col2, col1)
        }
    }

    /// Compute combined selectivity for multiple predicates considering correlations
    ///
    /// This is the KEY method that makes correlation tracking useful.
    ///
    /// # Arguments
    /// * `selectivities` - List of (column_name, selectivity) pairs
    ///
    /// # Returns
    /// Combined selectivity accounting for correlations
    pub fn combined_selectivity(&self, selectivities: &[(&str, f64)]) -> f64 {
        if selectivities.is_empty() {
            return 1.0;
        }

        if selectivities.len() == 1 {
            return selectivities[0].1;
        }

        // Start with the most selective predicate
        let mut result = selectivities
            .iter()
            .map(|(_, s)| *s)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        // For each additional predicate, account for correlation
        let mut used = vec![false; selectivities.len()];

        // Mark the most selective one as used
        for (i, (_, s)) in selectivities.iter().enumerate() {
            if (*s - result).abs() < f64::EPSILON {
                used[i] = true;
                break;
            }
        }

        // Process remaining predicates
        for (i, (col_i, sel_i)) in selectivities.iter().enumerate() {
            if used[i] {
                continue;
            }

            // Find max correlation with any used column
            let mut max_corr: f64 = 0.0;
            for (j, (col_j, _)) in selectivities.iter().enumerate() {
                if used[j] {
                    let corr = self.get_correlation(col_i, col_j).abs();
                    max_corr = max_corr.max(corr);
                }
            }

            // Adjust selectivity based on correlation:
            // - correlation = 0: multiply (independent) → sel_i
            // - correlation = 1: no additional filtering → 1.0
            // - correlation = 0.5: partial effect → sqrt(sel_i)
            let adjusted_sel = if max_corr >= 0.99 {
                // Highly correlated - minimal additional filtering
                1.0 - (1.0 - *sel_i) * (1.0 - max_corr)
            } else if max_corr < 0.01 {
                // Essentially independent
                *sel_i
            } else {
                // Partial correlation - interpolate
                sel_i.powf(1.0 - max_corr)
            };

            result *= adjusted_sel;
            used[i] = true;
        }

        result.clamp(1e-10, 1.0)
    }

    /// Detect correlations from sample data
    ///
    /// This uses Pearson correlation coefficient for numeric columns
    /// and Cramér's V for categorical columns.
    pub fn detect_correlations_from_sample(
        col1_values: &[Value],
        col2_values: &[Value],
    ) -> Option<CorrelationCoefficient> {
        if col1_values.len() != col2_values.len() || col1_values.is_empty() {
            return None;
        }

        // Try numeric correlation (Pearson)
        let numeric1: Vec<f64> = col1_values.iter().filter_map(|v| v.as_float64()).collect();
        let numeric2: Vec<f64> = col2_values.iter().filter_map(|v| v.as_float64()).collect();

        if numeric1.len() == col1_values.len() && numeric2.len() == col2_values.len() {
            return Some(Self::pearson_correlation(&numeric1, &numeric2));
        }

        // Fall back to categorical correlation (based on value co-occurrence)
        Some(Self::categorical_correlation(col1_values, col2_values))
    }

    /// Pearson correlation coefficient for numeric data
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|a| a * a).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            (numerator / denominator).clamp(-1.0, 1.0)
        }
    }

    /// Categorical correlation based on conditional entropy
    fn categorical_correlation(col1: &[Value], col2: &[Value]) -> f64 {
        use std::collections::HashMap;

        // Count value pair frequencies
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut col1_counts: HashMap<String, usize> = HashMap::new();
        let mut col2_counts: HashMap<String, usize> = HashMap::new();

        for (v1, v2) in col1.iter().zip(col2.iter()) {
            let s1 = v1.to_string();
            let s2 = v2.to_string();
            *pair_counts.entry((s1.clone(), s2.clone())).or_insert(0) += 1;
            *col1_counts.entry(s1).or_insert(0) += 1;
            *col2_counts.entry(s2).or_insert(0) += 1;
        }

        let n = col1.len() as f64;
        let ndv1 = col1_counts.len() as f64;
        let ndv2 = col2_counts.len() as f64;

        if ndv1 <= 1.0 || ndv2 <= 1.0 {
            return 0.0;
        }

        // Compute Cramér's V approximation
        // V = sqrt(χ² / (n * min(k-1, r-1)))
        let mut chi_squared = 0.0;
        for ((v1, v2), observed) in &pair_counts {
            let expected = (col1_counts[v1] as f64 * col2_counts[v2] as f64) / n;
            if expected > 0.0 {
                chi_squared += (*observed as f64 - expected).powi(2) / expected;
            }
        }

        let min_dim = (ndv1 - 1.0).min(ndv2 - 1.0).max(1.0);
        let cramers_v = (chi_squared / (n * min_dim)).sqrt();

        cramers_v.min(1.0)
    }
}

/// Extension to SelectivityEstimator for correlation-aware estimation
impl SelectivityEstimator {
    /// Estimate combined selectivity for multiple predicates with correlation awareness
    ///
    /// This is the method that should be called by the query planner instead of
    /// simply multiplying selectivities.
    pub fn combined_selectivity_with_correlations(
        selectivities: &[(&str, f64)],
        correlations: Option<&ColumnCorrelations>,
    ) -> f64 {
        match correlations {
            Some(corr) => corr.combined_selectivity(selectivities),
            None => {
                // Fall back to independence assumption (multiply)
                selectivities.iter().map(|(_, s)| *s).product()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_stats_new() {
        let stats = TableStats::new("test_table".to_string());
        assert_eq!(stats.table_name, "test_table");
        assert_eq!(stats.row_count, 0);
    }

    #[test]
    fn test_equality_selectivity() {
        // With 100 distinct values, selectivity = 1/100 = 0.01
        let sel = SelectivityEstimator::equality(100);
        assert!((sel - 0.01).abs() < 0.001);

        // With 0 distinct values, use default
        let sel_default = SelectivityEstimator::equality(0);
        assert!((sel_default - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_in_list_selectivity() {
        // IN list with 2 values out of 5 distinct = 0.4
        let sel = SelectivityEstimator::in_list(2, 5);
        assert!((sel - 0.4).abs() < 0.001);

        // Large list should cap at 1.0
        let sel_large = SelectivityEstimator::in_list(10, 5);
        assert!((sel_large - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_null_selectivity() {
        // 100 nulls out of 1000 rows = 0.1
        let sel = SelectivityEstimator::is_null(100, 1000);
        assert!((sel - 0.1).abs() < 0.001);

        // Not null selectivity should be 0.9
        let sel_not_null = SelectivityEstimator::is_not_null(100, 1000);
        assert!((sel_not_null - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_join_cardinality() {
        // 10000 orders, 1000 users, 1000 distinct user_ids
        // Join cardinality = 10000 * 1000 / max(1000, 1000) = 10000
        let join_card = SelectivityEstimator::join_cardinality(10000, 1000, 1000, 1000);
        assert_eq!(join_card, 10000);
    }

    #[test]
    fn test_like_selectivity() {
        // Prefix pattern
        let sel_prefix = SelectivityEstimator::like("abc%", 1000);
        assert!(sel_prefix < 0.1);

        // Suffix pattern (less selective)
        let sel_suffix = SelectivityEstimator::like("%abc", 1000);
        assert!((sel_suffix - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_is_stats_table() {
        assert!(is_stats_table("_sys_table_stats"));
        assert!(is_stats_table("_SYS_TABLE_STATS"));
        assert!(is_stats_table("_sys_column_stats"));
        assert!(!is_stats_table("users"));
        assert!(!is_stats_table("_sys_other"));
    }

    #[test]
    fn test_column_stats_is_empty() {
        let stats = ColumnStats::new("test".to_string());
        assert!(stats.is_empty());

        let mut stats2 = ColumnStats::new("test".to_string());
        stats2.distinct_count = 10;
        assert!(!stats2.is_empty());
    }

    #[test]
    fn test_histogram_from_sorted_values() {
        // Create sorted integer values
        let values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let histogram = Histogram::from_sorted_values(&values, 10).unwrap();

        // Should have boundaries
        assert!(!histogram.boundaries.is_empty());
        assert_eq!(histogram.total_rows, 100);
        assert_eq!(histogram.rows_per_bucket, 10);

        // First boundary should be minimum
        assert_eq!(histogram.boundaries[0], Value::Integer(0));
    }

    #[test]
    fn test_histogram_selectivity_estimation() {
        // Create uniformly distributed values 0-99
        let values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let histogram = Histogram::from_sorted_values(&values, 10).unwrap();

        // LessThan 50 should be approximately 0.5
        let sel_lt_50 = histogram.estimate_selectivity(&Value::Integer(50), HistogramOp::LessThan);
        assert!(
            sel_lt_50 > 0.4 && sel_lt_50 < 0.6,
            "Expected ~0.5, got {}",
            sel_lt_50
        );

        // LessThan 10 should be approximately 0.1
        let sel_lt_10 = histogram.estimate_selectivity(&Value::Integer(10), HistogramOp::LessThan);
        assert!(
            sel_lt_10 > 0.05 && sel_lt_10 < 0.2,
            "Expected ~0.1, got {}",
            sel_lt_10
        );

        // GreaterThan 90 should be approximately 0.1
        let sel_gt_90 =
            histogram.estimate_selectivity(&Value::Integer(90), HistogramOp::GreaterThan);
        assert!(
            sel_gt_90 > 0.0 && sel_gt_90 < 0.2,
            "Expected ~0.1, got {}",
            sel_gt_90
        );
    }

    #[test]
    fn test_histogram_json_round_trip() {
        let values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let histogram = Histogram::from_sorted_values(&values, 10).unwrap();

        // Serialize to JSON
        let json = histogram.to_json();

        // Parse back
        let parsed = Histogram::from_json(&json).expect("Failed to parse histogram JSON");

        // Verify key properties match
        assert_eq!(parsed.total_rows, histogram.total_rows);
        assert_eq!(parsed.rows_per_bucket, histogram.rows_per_bucket);
        assert_eq!(parsed.boundaries.len(), histogram.boundaries.len());
    }

    #[test]
    fn test_range_with_histogram() {
        // Create column stats with histogram
        let values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let histogram = Histogram::from_sorted_values(&values, 10).unwrap();

        let mut col_stats = ColumnStats::new("test".to_string());
        col_stats.set_histogram(&histogram);
        col_stats.min_value = Some(Value::Integer(0));
        col_stats.max_value = Some(Value::Integer(99));
        col_stats.distinct_count = 100;

        // Use histogram-based estimation
        let sel = SelectivityEstimator::range_with_histogram(
            &col_stats,
            &Value::Integer(50),
            HistogramOp::LessThan,
        );
        assert!(sel > 0.4 && sel < 0.6, "Expected ~0.5, got {}", sel);
    }

    #[test]
    fn test_histogram_empty_values() {
        let values: Vec<Value> = vec![];
        let histogram = Histogram::from_sorted_values(&values, 10);
        assert!(histogram.is_none());
    }

    #[test]
    fn test_histogram_with_nulls() {
        use crate::core::DataType;

        // Histogram should skip null values
        let mut values: Vec<Value> = (0..50).map(Value::Integer).collect();
        values.extend((0..10).map(|_| Value::Null(DataType::Integer)));
        values.extend((50..100).map(Value::Integer));

        let histogram = Histogram::from_sorted_values(&values, 10).unwrap();
        assert_eq!(histogram.total_rows, 100); // Only non-null values counted
    }

    #[test]
    fn test_histogram_join_cardinality_full_overlap() {
        // Both columns have values 0-99, should produce high cardinality
        let left_values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let right_values: Vec<Value> = (0..100).map(Value::Integer).collect();

        let left_hist = Histogram::from_sorted_values(&left_values, 10).unwrap();
        let right_hist = Histogram::from_sorted_values(&right_values, 10).unwrap();

        let mut left_stats = ColumnStats::new("left_col".to_string());
        left_stats.set_histogram(&left_hist);
        left_stats.min_value = Some(Value::Integer(0));
        left_stats.max_value = Some(Value::Integer(99));
        left_stats.distinct_count = 100;

        let mut right_stats = ColumnStats::new("right_col".to_string());
        right_stats.set_histogram(&right_hist);
        right_stats.min_value = Some(Value::Integer(0));
        right_stats.max_value = Some(Value::Integer(99));
        right_stats.distinct_count = 100;

        let cardinality =
            SelectivityEstimator::histogram_join_cardinality(&left_stats, &right_stats, 100, 100);

        // Full overlap should produce ~100 rows (1:1 join on primary key)
        assert!(
            (50..=200).contains(&cardinality),
            "Expected ~100 rows for full overlap join, got {}",
            cardinality
        );
    }

    #[test]
    fn test_histogram_join_cardinality_no_overlap() {
        // Left has 0-99, right has 200-299, no overlap
        let left_values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let right_values: Vec<Value> = (200..300).map(Value::Integer).collect();

        let left_hist = Histogram::from_sorted_values(&left_values, 10).unwrap();
        let right_hist = Histogram::from_sorted_values(&right_values, 10).unwrap();

        let mut left_stats = ColumnStats::new("left_col".to_string());
        left_stats.set_histogram(&left_hist);
        left_stats.min_value = Some(Value::Integer(0));
        left_stats.max_value = Some(Value::Integer(99));
        left_stats.distinct_count = 100;

        let mut right_stats = ColumnStats::new("right_col".to_string());
        right_stats.set_histogram(&right_hist);
        right_stats.min_value = Some(Value::Integer(200));
        right_stats.max_value = Some(Value::Integer(299));
        right_stats.distinct_count = 100;

        let cardinality =
            SelectivityEstimator::histogram_join_cardinality(&left_stats, &right_stats, 100, 100);

        // No overlap should produce very few or zero rows
        assert!(
            cardinality <= 10,
            "Expected ~0 rows for non-overlapping join, got {}",
            cardinality
        );
    }

    #[test]
    fn test_histogram_join_cardinality_partial_overlap() {
        // Left has 0-99, right has 50-149, 50% overlap
        let left_values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let right_values: Vec<Value> = (50..150).map(Value::Integer).collect();

        let left_hist = Histogram::from_sorted_values(&left_values, 10).unwrap();
        let right_hist = Histogram::from_sorted_values(&right_values, 10).unwrap();

        let mut left_stats = ColumnStats::new("left_col".to_string());
        left_stats.set_histogram(&left_hist);
        left_stats.min_value = Some(Value::Integer(0));
        left_stats.max_value = Some(Value::Integer(99));
        left_stats.distinct_count = 100;

        let mut right_stats = ColumnStats::new("right_col".to_string());
        right_stats.set_histogram(&right_hist);
        right_stats.min_value = Some(Value::Integer(50));
        right_stats.max_value = Some(Value::Integer(149));
        right_stats.distinct_count = 100;

        let cardinality =
            SelectivityEstimator::histogram_join_cardinality(&left_stats, &right_stats, 100, 100);

        // ~50% overlap should produce ~50 rows
        assert!(
            (20..=100).contains(&cardinality),
            "Expected ~50 rows for 50% overlap join, got {}",
            cardinality
        );
    }

    #[test]
    fn test_range_overlap_ratio() {
        // Full overlap
        let ratio1 = SelectivityEstimator::range_overlap_ratio(
            &Value::Integer(0),
            &Value::Integer(100),
            &Value::Integer(0),
            &Value::Integer(100),
        );
        assert!(
            (ratio1 - 1.0).abs() < 0.01,
            "Full overlap should be 1.0, got {}",
            ratio1
        );

        // No overlap
        let ratio2 = SelectivityEstimator::range_overlap_ratio(
            &Value::Integer(0),
            &Value::Integer(50),
            &Value::Integer(100),
            &Value::Integer(150),
        );
        assert!(ratio2 < 0.01, "No overlap should be 0.0, got {}", ratio2);

        // 50% overlap
        let ratio3 = SelectivityEstimator::range_overlap_ratio(
            &Value::Integer(0),
            &Value::Integer(100),
            &Value::Integer(50),
            &Value::Integer(150),
        );
        assert!(
            ratio3 > 0.3 && ratio3 < 0.7,
            "Partial overlap should be ~0.5, got {}",
            ratio3
        );
    }

    // =========================================================================
    // COLUMN CORRELATION TESTS
    // =========================================================================

    #[test]
    fn test_correlation_basic() {
        let mut corr = ColumnCorrelations::new();
        corr.add_correlation("city", "state", 0.95);

        assert!((corr.get_correlation("city", "state") - 0.95).abs() < 0.001);
        // Should work in either order
        assert!((corr.get_correlation("state", "city") - 0.95).abs() < 0.001);
        // Unknown correlation should be 0
        assert!((corr.get_correlation("city", "zip")).abs() < 0.001);
    }

    #[test]
    fn test_functional_dependency() {
        let mut corr = ColumnCorrelations::new();
        corr.add_functional_dep("zip_code", "city");

        assert!(corr.determines("zip_code", "city"));
        assert!(!corr.determines("city", "zip_code"));
        // FD implies correlation = 1.0
        assert!((corr.get_correlation("zip_code", "city") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_combined_selectivity_independent() {
        let corr = ColumnCorrelations::new();
        // Independent columns: multiply selectivities
        let selectivities = [("col1", 0.1), ("col2", 0.1)];
        let combined = corr.combined_selectivity(&selectivities);
        // Should be close to 0.1 * 0.1 = 0.01
        assert!(
            combined > 0.005 && combined < 0.02,
            "Independent columns: expected ~0.01, got {}",
            combined
        );
    }

    #[test]
    fn test_combined_selectivity_correlated() {
        let mut corr = ColumnCorrelations::new();
        corr.add_correlation("city", "state", 0.99); // Highly correlated

        // city = 'NYC' (0.02) AND state = 'NY' (0.02)
        let selectivities = [("city", 0.02), ("state", 0.02)];
        let combined = corr.combined_selectivity(&selectivities);

        // With high correlation, combined should be closer to 0.02 than 0.0004
        assert!(
            combined > 0.01,
            "Correlated columns: expected > 0.01, got {}",
            combined
        );

        // Without correlation, would be 0.02 * 0.02 = 0.0004
        let naive = selectivities.iter().map(|(_, s)| *s).product::<f64>();
        assert!(
            combined > naive * 10.0,
            "Correlated selectivity should be much higher than naive: {} vs {}",
            combined,
            naive
        );
    }

    #[test]
    fn test_combined_selectivity_partial_correlation() {
        let mut corr = ColumnCorrelations::new();
        corr.add_correlation("age", "income", 0.5); // Moderate correlation

        let selectivities = [("age", 0.1), ("income", 0.1)];
        let combined = corr.combined_selectivity(&selectivities);

        // With 0.5 correlation, should be between 0.01 and 0.1
        let naive = 0.1 * 0.1; // 0.01
        let max = 0.1; // Most selective predicate

        assert!(
            combined >= naive && combined <= max,
            "Partial correlation: expected between {} and {}, got {}",
            naive,
            max,
            combined
        );
    }

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation: y = x
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = ColumnCorrelations::pearson_correlation(&x, &y);
        assert!(
            (corr - 1.0).abs() < 0.001,
            "Perfect correlation should be 1.0, got {}",
            corr
        );

        // Perfect negative correlation: y = -x
        let y_neg = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
        let corr_neg = ColumnCorrelations::pearson_correlation(&x, &y_neg);
        assert!(
            (corr_neg - (-1.0)).abs() < 0.001,
            "Perfect negative correlation should be -1.0, got {}",
            corr_neg
        );

        // No correlation: random-ish
        let y_random = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let corr_random = ColumnCorrelations::pearson_correlation(&x, &y_random);
        assert!(
            corr_random.abs() < 0.8,
            "Low correlation expected, got {}",
            corr_random
        );
    }

    #[test]
    fn test_detect_correlations_numeric() {
        let col1: Vec<Value> = (0..100).map(Value::Integer).collect();
        let col2: Vec<Value> = (0..100).map(Value::Integer).collect();

        let corr = ColumnCorrelations::detect_correlations_from_sample(&col1, &col2);
        assert!(corr.is_some());
        assert!(
            (corr.unwrap() - 1.0).abs() < 0.01,
            "Identical columns should have correlation ~1.0"
        );
    }

    #[test]
    fn test_detect_correlations_categorical() {
        // Simulated city/state correlation
        let cities = vec![
            Value::Text("NYC".into()),
            Value::Text("NYC".into()),
            Value::Text("LA".into()),
            Value::Text("LA".into()),
            Value::Text("Chicago".into()),
        ];
        let states = vec![
            Value::Text("NY".into()),
            Value::Text("NY".into()),
            Value::Text("CA".into()),
            Value::Text("CA".into()),
            Value::Text("IL".into()),
        ];

        let corr = ColumnCorrelations::detect_correlations_from_sample(&cities, &states);
        assert!(corr.is_some());
        // Perfect 1:1 mapping should have high correlation
        assert!(
            corr.unwrap() > 0.8,
            "City/state should be highly correlated"
        );
    }

    #[test]
    fn test_selectivity_with_correlations_api() {
        let mut corr = ColumnCorrelations::new();
        corr.add_correlation("a", "b", 0.9);

        let selectivities = [("a", 0.1), ("b", 0.1)];

        // With correlations
        let with_corr = SelectivityEstimator::combined_selectivity_with_correlations(
            &selectivities,
            Some(&corr),
        );

        // Without correlations (independence)
        let without_corr =
            SelectivityEstimator::combined_selectivity_with_correlations(&selectivities, None);

        assert!(
            with_corr > without_corr,
            "Correlated selectivity should be higher: {} vs {}",
            with_corr,
            without_corr
        );
    }

    // =========================================================================
    // HISTOGRAM BETWEEN RANGE SELECTIVITY TESTS
    // =========================================================================

    #[test]
    fn test_histogram_between_range_selectivity() {
        // Create uniformly distributed values 0-99
        let values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let histogram = Histogram::from_sorted_values(&values, 10).unwrap();

        // BETWEEN 25 AND 75 should be approximately 0.5
        let sel_25_75 =
            histogram.estimate_range_selectivity(&Value::Integer(25), &Value::Integer(75));
        assert!(
            sel_25_75 > 0.4 && sel_25_75 < 0.65,
            "Expected ~0.5 for BETWEEN 25 AND 75, got {}",
            sel_25_75
        );

        // BETWEEN 0 AND 10 should be approximately 0.1
        let sel_0_10 =
            histogram.estimate_range_selectivity(&Value::Integer(0), &Value::Integer(10));
        assert!(
            sel_0_10 > 0.05 && sel_0_10 < 0.2,
            "Expected ~0.1 for BETWEEN 0 AND 10, got {}",
            sel_0_10
        );

        // BETWEEN 90 AND 100 should be approximately 0.1
        let sel_90_100 =
            histogram.estimate_range_selectivity(&Value::Integer(90), &Value::Integer(100));
        assert!(
            sel_90_100 > 0.05 && sel_90_100 < 0.2,
            "Expected ~0.1 for BETWEEN 90 AND 100, got {}",
            sel_90_100
        );

        // BETWEEN 0 AND 100 should be approximately 1.0
        let sel_full =
            histogram.estimate_range_selectivity(&Value::Integer(0), &Value::Integer(100));
        assert!(
            sel_full > 0.9,
            "Expected ~1.0 for full range, got {}",
            sel_full
        );
    }

    #[test]
    fn test_histogram_between_single_bucket() {
        // Create uniformly distributed values 0-99
        let values: Vec<Value> = (0..100).map(Value::Integer).collect();
        let histogram = Histogram::from_sorted_values(&values, 10).unwrap();

        // Small range within single bucket
        let sel_5_8 = histogram.estimate_range_selectivity(&Value::Integer(5), &Value::Integer(8));
        assert!(
            sel_5_8 > 0.0 && sel_5_8 < 0.15,
            "Expected small selectivity for narrow range, got {}",
            sel_5_8
        );
    }

    #[test]
    fn test_histogram_between_float_values() {
        // Create float values 0.0 to 99.0
        let values: Vec<Value> = (0..100).map(|i| Value::Float(i as f64)).collect();
        let histogram = Histogram::from_sorted_values(&values, 10).unwrap();

        // BETWEEN 25.0 AND 75.0 should be approximately 0.5
        let sel = histogram.estimate_range_selectivity(&Value::Float(25.0), &Value::Float(75.0));
        assert!(
            sel > 0.4 && sel < 0.65,
            "Expected ~0.5 for BETWEEN 25.0 AND 75.0, got {}",
            sel
        );
    }

    // =========================================================================
    // ADAPTIVE HISTOGRAM TESTS
    // =========================================================================

    #[test]
    fn test_adaptive_histogram_high_cardinality() {
        // High cardinality: 1000 distinct values in 1000 rows (unique)
        let values: Vec<Value> = (0..1000).map(Value::Integer).collect();
        let histogram = Histogram::adaptive_from_sorted_values(&values, 1000, 10, 200).unwrap();

        // With 1000 distinct values, sqrt(1000) ≈ 32
        // Skew factor = log2(1000/1000 + 1) = log2(2) = 1
        // Expected: ~32 buckets
        assert!(
            histogram.boundaries.len() >= 10 && histogram.boundaries.len() <= 50,
            "High cardinality should use moderate buckets, got {}",
            histogram.boundaries.len()
        );
    }

    #[test]
    fn test_adaptive_histogram_low_cardinality() {
        // Low cardinality: 10 distinct values in 10000 rows (very skewed)
        let mut values: Vec<Value> = Vec::with_capacity(10000);
        for i in 0..10 {
            for _ in 0..1000 {
                values.push(Value::Integer(i));
            }
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let histogram = Histogram::adaptive_from_sorted_values(&values, 10, 10, 200).unwrap();

        // With 10 distinct values and 10000 rows:
        // sqrt(10) ≈ 3.16
        // Skew factor = log2(10000/10 + 1) = log2(1001) ≈ 10
        // Expected: ~32 buckets (3.16 * 10)
        assert!(
            histogram.boundaries.len() >= 10,
            "Low cardinality should use minimum buckets, got {}",
            histogram.boundaries.len()
        );
    }

    #[test]
    fn test_adaptive_histogram_medium_cardinality() {
        // Medium cardinality: 100 distinct values in 10000 rows
        let mut values: Vec<Value> = Vec::with_capacity(10000);
        for i in 0..100 {
            for _ in 0..100 {
                values.push(Value::Integer(i));
            }
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let histogram = Histogram::adaptive_from_sorted_values(&values, 100, 10, 200).unwrap();

        // With 100 distinct values and 10000 rows:
        // sqrt(100) = 10
        // Skew factor = log2(10000/100 + 1) = log2(101) ≈ 6.7
        // Expected: ~67 buckets
        assert!(
            histogram.boundaries.len() >= 10 && histogram.boundaries.len() <= 100,
            "Medium cardinality should use proportional buckets, got {}",
            histogram.boundaries.len()
        );
    }

    #[test]
    fn test_adaptive_histogram_respects_bounds() {
        let values: Vec<Value> = (0..1000000).map(Value::Integer).collect();

        // Test minimum bound
        let histogram_min =
            Histogram::adaptive_from_sorted_values(&values, 1000000, 50, 200).unwrap();
        assert!(
            histogram_min.boundaries.len() >= 50,
            "Should respect min_buckets, got {}",
            histogram_min.boundaries.len()
        );

        // Test maximum bound with skewed data
        let mut skewed: Vec<Value> = Vec::with_capacity(100000);
        for i in 0..10 {
            for _ in 0..10000 {
                skewed.push(Value::Integer(i));
            }
        }
        let histogram_max = Histogram::adaptive_from_sorted_values(&skewed, 10, 10, 50).unwrap();
        assert!(
            histogram_max.boundaries.len() <= 51, // boundaries = buckets + 1
            "Should respect max_buckets, got {}",
            histogram_max.boundaries.len()
        );
    }
}
