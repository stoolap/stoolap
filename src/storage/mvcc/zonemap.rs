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

//! Zone Maps (Min-Max Indexes) for Micro-Partition Pruning
//!
//! Zone maps maintain min/max statistics per data segment, enabling the query
//! executor to skip entire segments when predicates fall outside the range.
//!
//! ## Overview
//!
//! For a table with 1 million rows partitioned into 1000-row segments:
//! - Each column has a `ColumnZoneMap` with 1000 `ZoneMapEntry` values
//! - When `WHERE date > '2024-06-01'`, segments with `max_date <= '2024-06-01'` are skipped
//! - This can reduce I/O by 10-100x for range queries on ordered/clustered data
//!
//! ## Example
//!
//! ```text
//! SELECT * FROM orders WHERE date > '2024-06-01'
//!
//! Zone Map for 'date' column:
//! ┌──────────┬─────────────┬─────────────┐
//! │ Segment  │ Min         │ Max         │
//! ├──────────┼─────────────┼─────────────┤
//! │ 0        │ 2024-01-01  │ 2024-01-31  │ ← PRUNE
//! │ 1        │ 2024-02-01  │ 2024-02-28  │ ← PRUNE
//! │ ...      │ ...         │ ...         │ ← PRUNE
//! │ 5        │ 2024-06-01  │ 2024-06-30  │ ← SCAN (might match)
//! │ 6        │ 2024-07-01  │ 2024-07-31  │ ← SCAN
//! └──────────┴─────────────┴─────────────┘
//! ```

use rustc_hash::FxHashMap;

use crate::core::{Operator, Value};

/// Default segment size for zone maps (number of rows per segment)
pub const DEFAULT_SEGMENT_SIZE: usize = 1000;

/// Statistics for a single segment (micro-partition)
#[derive(Debug, Clone)]
pub struct ZoneMapEntry {
    /// Segment identifier (0-indexed)
    pub segment_id: u32,
    /// Minimum value in segment (None if all NULL)
    pub min_value: Option<Value>,
    /// Maximum value in segment (None if all NULL)
    pub max_value: Option<Value>,
    /// Number of NULL values in segment
    pub null_count: u32,
    /// Number of rows in segment
    pub row_count: u32,
}

impl ZoneMapEntry {
    /// Create a new zone map entry for a segment
    pub fn new(segment_id: u32) -> Self {
        Self {
            segment_id,
            min_value: None,
            max_value: None,
            null_count: 0,
            row_count: 0,
        }
    }

    /// Update entry with a new value
    pub fn update(&mut self, value: &Value) {
        self.row_count += 1;

        if value.is_null() {
            self.null_count += 1;
            return;
        }

        // Update min
        match &self.min_value {
            None => self.min_value = Some(value.clone()),
            Some(current_min) => {
                if value < current_min {
                    self.min_value = Some(value.clone());
                }
            }
        }

        // Update max
        match &self.max_value {
            None => self.max_value = Some(value.clone()),
            Some(current_max) => {
                if value > current_max {
                    self.max_value = Some(value.clone());
                }
            }
        }
    }

    /// Check if segment can be pruned for given predicate
    ///
    /// Returns true if the entire segment can be skipped (no rows will match)
    pub fn can_prune(&self, operator: Operator, value: &Value) -> bool {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::zone_map_entry_checked();
        // If predicate is NULL comparison, check null_count
        if value.is_null() {
            return match operator {
                Operator::Eq => self.null_count == 0, // No NULLs to match
                _ => false,                           // NULL comparisons are complex
            };
        }

        // If segment has no non-null values, can prune any non-null comparison
        let (min, max) = match (&self.min_value, &self.max_value) {
            (Some(min), Some(max)) => (min, max),
            _ => return false, // All NULLs - can't prune
        };

        match operator {
            // col = val: Prune if val < min or val > max
            Operator::Eq => value < min || value > max,

            // col != val: Can only prune if all values are equal AND equal to val
            // (which is rare), so generally don't prune
            Operator::Ne => false,

            // col < val: Prune if min >= val (all values are >= val)
            Operator::Lt => min >= value,

            // col <= val: Prune if min > val (all values are > val)
            Operator::Lte => min > value,

            // col > val: Prune if max <= val (all values are <= val)
            Operator::Gt => max <= value,

            // col >= val: Prune if max < val (all values are < val)
            Operator::Gte => max < value,

            // LIKE, IN, NOT IN, IS NULL, IS NOT NULL cannot be pruned with simple min/max
            Operator::Like
            | Operator::In
            | Operator::NotIn
            | Operator::IsNull
            | Operator::IsNotNull => false,
        }
    }
}

/// Zone map for a single column across all segments
#[derive(Debug, Clone)]
pub struct ColumnZoneMap {
    /// Column name
    pub column_name: String,
    /// Per-segment statistics
    pub segments: Vec<ZoneMapEntry>,
    /// Global minimum across all segments
    pub global_min: Option<Value>,
    /// Global maximum across all segments
    pub global_max: Option<Value>,
}

impl ColumnZoneMap {
    /// Create a new column zone map
    pub fn new(column_name: impl Into<String>) -> Self {
        Self {
            column_name: column_name.into(),
            segments: Vec::new(),
            global_min: None,
            global_max: None,
        }
    }

    /// Get or create segment entry
    pub fn get_or_create_segment(&mut self, segment_id: u32) -> &mut ZoneMapEntry {
        while self.segments.len() <= segment_id as usize {
            self.segments
                .push(ZoneMapEntry::new(self.segments.len() as u32));
        }
        &mut self.segments[segment_id as usize]
    }

    /// Update segment with a value
    pub fn update_segment(&mut self, segment_id: u32, value: &Value) {
        let entry = self.get_or_create_segment(segment_id);
        entry.update(value);

        // Update global min/max
        if !value.is_null() {
            match &self.global_min {
                None => self.global_min = Some(value.clone()),
                Some(current) if value < current => self.global_min = Some(value.clone()),
                _ => {}
            }
            match &self.global_max {
                None => self.global_max = Some(value.clone()),
                Some(current) if value > current => self.global_max = Some(value.clone()),
                _ => {}
            }
        }
    }

    /// Get segments that cannot be pruned for given predicate
    ///
    /// Returns list of segment IDs that might contain matching rows
    pub fn get_unpruned_segments(&self, operator: Operator, value: &Value) -> Vec<u32> {
        self.segments
            .iter()
            .filter(|entry| !entry.can_prune(operator, value))
            .map(|entry| entry.segment_id)
            .collect()
    }

    /// Whether a segment may hold a row the predicate keeps, the segments
    /// asked only until one may. A range bound the column's global minimum
    /// or maximum passes is kept by the segment holding that value, which no
    /// segment check is needed to find
    pub fn may_match(&self, operator: Operator, value: &Value) -> bool {
        let kept = |global: &Option<Value>, keeps: fn(&Value, &Value) -> bool| {
            !value.is_null() && global.as_ref().is_some_and(|bound| keeps(bound, value))
        };
        let by_bound = match operator {
            Operator::Gt => kept(&self.global_max, |max, value| max > value),
            Operator::Gte => kept(&self.global_max, |max, value| max >= value),
            Operator::Lt => kept(&self.global_min, |min, value| min < value),
            Operator::Lte => kept(&self.global_min, |min, value| min <= value),
            _ => false,
        };
        by_bound
            || self
                .segments
                .iter()
                .any(|entry| !entry.can_prune(operator, value))
    }
}

/// Zone map for an entire table
#[derive(Debug, Clone)]
pub struct TableZoneMap {
    /// Zone maps per column
    pub columns: FxHashMap<String, ColumnZoneMap>,
    /// Number of rows per segment
    pub segment_size: usize,
    /// Total number of segments
    pub segment_count: u32,
    /// The publish epoch of the committed state the maps were built from;
    /// None when the build did not read exactly one such state
    pub epoch: Option<u64>,
}

impl TableZoneMap {
    /// Create a new table zone map
    pub fn new(segment_size: usize) -> Self {
        Self {
            columns: FxHashMap::default(),
            segment_size,
            segment_count: 0,
            epoch: None,
        }
    }

    /// Get segment ID for a given row index
    pub fn segment_for_row(&self, row_index: usize) -> u32 {
        (row_index / self.segment_size) as u32
    }

    /// Update zone map with a new row
    pub fn update_row(&mut self, row_index: usize, columns: &[(String, Value)]) {
        let segment_id = self.segment_for_row(row_index);
        if segment_id >= self.segment_count {
            self.segment_count = segment_id + 1;
        }

        for (col_name, value) in columns {
            let col_map = self
                .columns
                .entry(col_name.clone())
                .or_insert_with(|| ColumnZoneMap::new(col_name.clone()));
            col_map.update_segment(segment_id, value);
        }
    }

    /// Get segments to scan for a single-column predicate
    pub fn get_segments_to_scan(
        &self,
        column: &str,
        operator: Operator,
        value: &Value,
    ) -> Option<Vec<u32>> {
        let col_map = self.columns.get(column)?;
        Some(col_map.get_unpruned_segments(operator, value))
    }

    /// Whether a segment of `column` may hold a row the predicate keeps;
    /// None when the column has no zone map
    pub fn may_match(&self, column: &str, operator: Operator, value: &Value) -> Option<bool> {
        Some(self.columns.get(column)?.may_match(operator, value))
    }
}

/// Builder for constructing zone maps during ANALYZE
pub struct ZoneMapBuilder {
    zone_map: TableZoneMap,
    current_row: usize,
}

impl ZoneMapBuilder {
    /// Create a new zone map builder
    pub fn new(segment_size: usize) -> Self {
        Self {
            zone_map: TableZoneMap::new(segment_size),
            current_row: 0,
        }
    }

    /// Add a row to the zone map
    pub fn add_row(&mut self, columns: &[(String, Value)]) {
        self.zone_map.update_row(self.current_row, columns);
        self.current_row += 1;
    }

    /// Finish building and return the zone map
    pub fn build(self) -> TableZoneMap {
        self.zone_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn may_match_answers_as_the_segments_do() {
        use crate::core::DataType;
        // Segments of 10 values from 0, one of NULLs only, one of a single text
        let mut map = ColumnZoneMap::new("c");
        for id in 0..40 {
            map.update_segment(id / 10, &Value::Integer(id as i64));
        }
        map.update_segment(4, &Value::Null(DataType::Integer));
        let mut alone = ColumnZoneMap::new("c");
        alone.update_segment(0, &Value::Null(DataType::Integer));
        let mut text = ColumnZoneMap::new("c");
        text.update_segment(0, &Value::text("m"));
        let values = [
            Value::Integer(-1),
            Value::Integer(0),
            Value::Integer(15),
            Value::Integer(39),
            Value::Integer(40),
            Value::Null(DataType::Integer),
            Value::text("m"),
        ];
        let operators = [
            Operator::Eq,
            Operator::Ne,
            Operator::Lt,
            Operator::Lte,
            Operator::Gt,
            Operator::Gte,
            Operator::IsNull,
        ];
        for map in [&map, &alone, &text, &ColumnZoneMap::new("c")] {
            for value in &values {
                for operator in operators {
                    let segments = !map.get_unpruned_segments(operator, value).is_empty();
                    assert_eq!(
                        map.may_match(operator, value),
                        segments,
                        "{operator:?} {value:?} over {map:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_zone_map_entry_basic() {
        let mut entry = ZoneMapEntry::new(0);
        entry.update(&Value::Integer(10));
        entry.update(&Value::Integer(20));
        entry.update(&Value::Integer(15));

        assert_eq!(entry.row_count, 3);
        assert_eq!(entry.null_count, 0);
        assert_eq!(entry.min_value, Some(Value::Integer(10)));
        assert_eq!(entry.max_value, Some(Value::Integer(20)));
    }

    #[test]
    fn test_zone_map_entry_with_nulls() {
        use crate::core::DataType;

        let mut entry = ZoneMapEntry::new(0);
        entry.update(&Value::Integer(10));
        entry.update(&Value::null(DataType::Integer));
        entry.update(&Value::Integer(20));

        assert_eq!(entry.row_count, 3);
        assert_eq!(entry.null_count, 1);
        assert_eq!(entry.min_value, Some(Value::Integer(10)));
        assert_eq!(entry.max_value, Some(Value::Integer(20)));
    }

    #[test]
    fn test_zone_map_pruning() {
        let mut entry = ZoneMapEntry::new(0);
        for i in 10..=20 {
            entry.update(&Value::Integer(i));
        }

        // col = 5: Can prune (5 < 10)
        assert!(entry.can_prune(Operator::Eq, &Value::Integer(5)));
        // col = 25: Can prune (25 > 20)
        assert!(entry.can_prune(Operator::Eq, &Value::Integer(25)));
        // col = 15: Cannot prune (15 in [10,20])
        assert!(!entry.can_prune(Operator::Eq, &Value::Integer(15)));

        // col < 10: Can prune (min >= 10)
        assert!(entry.can_prune(Operator::Lt, &Value::Integer(10)));
        // col < 15: Cannot prune (some values < 15)
        assert!(!entry.can_prune(Operator::Lt, &Value::Integer(15)));

        // col > 20: Can prune (max <= 20)
        assert!(entry.can_prune(Operator::Gt, &Value::Integer(20)));
        // col > 15: Cannot prune (some values > 15)
        assert!(!entry.can_prune(Operator::Gt, &Value::Integer(15)));
    }

    #[test]
    fn test_column_zone_map() {
        let mut col_map = ColumnZoneMap::new("id");

        // Segment 0: values 1-10
        for i in 1..=10 {
            col_map.update_segment(0, &Value::Integer(i));
        }

        // Segment 1: values 11-20
        for i in 11..=20 {
            col_map.update_segment(1, &Value::Integer(i));
        }

        // Segment 2: values 21-30
        for i in 21..=30 {
            col_map.update_segment(2, &Value::Integer(i));
        }

        assert_eq!(col_map.segments.len(), 3);
        assert_eq!(col_map.global_min, Some(Value::Integer(1)));
        assert_eq!(col_map.global_max, Some(Value::Integer(30)));

        // WHERE id > 25: Should prune segments 0 and 1
        let unpruned = col_map.get_unpruned_segments(Operator::Gt, &Value::Integer(25));
        assert_eq!(unpruned, vec![2]);

        // WHERE id = 15: Should only scan segment 1
        let unpruned = col_map.get_unpruned_segments(Operator::Eq, &Value::Integer(15));
        assert_eq!(unpruned, vec![1]);
    }

    #[test]
    fn test_table_zone_map() {
        let mut table_map = TableZoneMap::new(10);

        // Add 30 rows
        for i in 0..30 {
            table_map.update_row(
                i,
                &[
                    ("id".to_string(), Value::Integer(i as i64)),
                    ("name".to_string(), Value::text(format!("row{}", i))),
                ],
            );
        }

        assert_eq!(table_map.segment_count, 3);
    }

    #[test]
    fn test_zone_map_builder() {
        let mut builder = ZoneMapBuilder::new(5);

        for i in 0..20 {
            builder.add_row(&[("value".to_string(), Value::Integer(i))]);
        }

        let zone_map = builder.build();
        assert_eq!(zone_map.segment_count, 4); // 20 rows / 5 per segment = 4 segments

        let col_map = zone_map.columns.get("value").unwrap();
        assert_eq!(col_map.segments.len(), 4);
        assert_eq!(col_map.segments[0].min_value, Some(Value::Integer(0)));
        assert_eq!(col_map.segments[0].max_value, Some(Value::Integer(4)));
        assert_eq!(col_map.segments[3].min_value, Some(Value::Integer(15)));
        assert_eq!(col_map.segments[3].max_value, Some(Value::Integer(19)));
    }
}
