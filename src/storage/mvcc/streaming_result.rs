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

//! Zero-copy streaming result for full table scans
//!
//! This module provides a streaming iterator that yields references to row data
//! without cloning. It achieves true zero-copy by holding arena locks for the
//! duration of iteration.
//!
//! # Performance
//! - Eliminates ~600µs of cloning overhead per 10K row scan
//! - Memory usage: O(1) instead of O(n)
//! - Works with MVCC visibility by pre-computing visible row indices

use crate::common::CompactArc;
use crate::core::{Row, Value};
use crate::storage::mvcc::arena::ArenaReadGuard;

/// Pre-computed visible row information for zero-copy iteration
#[derive(Clone, Copy)]
pub struct VisibleRowInfo {
    pub row_id: i64,
    /// Arena index for rows in the arena, OR `arena_len + fallback_index` for
    /// chain entries not in the arena (visible under snapshot isolation).
    pub arena_idx: usize,
}

/// Zero-copy streaming result that yields references to arena data
///
/// This struct holds the arena read guard for the duration of iteration,
/// allowing it to yield row references without any cloning.
///
/// For rows whose visible version is a chain entry (not in the arena),
/// the data is stored in `fallback_data` and referenced via indices
/// starting at `arena_len`.
pub struct StreamingResult<'a> {
    /// Unified arena guard (single lock for both data and metadata)
    arena_guard: ArenaReadGuard<'a>,
    /// Pre-sorted list of visible row indices
    visible_indices: Vec<VisibleRowInfo>,
    /// Fallback data for chain entries not in arena (snapshot isolation).
    /// Indexed by `arena_idx - arena_len` when `arena_idx >= arena_len`.
    fallback_data: Vec<CompactArc<[Value]>>,
    /// Length of the arena at construction time (boundary between arena and fallback)
    arena_len: usize,
    /// Current position in visible_indices
    current_pos: usize,
    /// Column names
    columns: Vec<String>,
    /// Temporary row buffer for the current row (to satisfy &Row interface)
    current_row: Row,
    /// Cached Arc for current row (for zero-copy access)
    current_arc: Option<CompactArc<[Value]>>,
}

impl<'a> StreamingResult<'a> {
    /// Create a new streaming result from arena guard and visible indices
    pub fn new(
        arena_guard: ArenaReadGuard<'a>,
        visible_indices: Vec<VisibleRowInfo>,
        columns: Vec<String>,
    ) -> Self {
        let arena_len = arena_guard.len();
        Self {
            arena_guard,
            visible_indices,
            fallback_data: Vec::new(),
            arena_len,
            current_pos: 0,
            columns,
            current_row: Row::new(),
            current_arc: None,
        }
    }

    /// Create a streaming result with fallback data for chain entries not in arena.
    /// Used by the slow path under snapshot isolation where visible versions may be
    /// older chain entries without arena indices.
    pub fn new_with_fallback(
        arena_guard: ArenaReadGuard<'a>,
        visible_indices: Vec<VisibleRowInfo>,
        fallback_data: Vec<CompactArc<[Value]>>,
        columns: Vec<String>,
    ) -> Self {
        let arena_len = arena_guard.len();
        Self {
            arena_guard,
            visible_indices,
            fallback_data,
            arena_len,
            current_pos: 0,
            columns,
            current_row: Row::new(),
            current_arc: None,
        }
    }

    /// Get column names
    #[inline]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Move to next row, returns true if successful
    ///
    /// Note: For best performance, use row_slice() for zero-copy access
    /// instead of row().clone() which allocates.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> bool {
        if self.current_pos < self.visible_indices.len() {
            let info = &self.visible_indices[self.current_pos];

            if info.arena_idx < self.arena_len {
                // Arena path: O(1) Arc clone - no data copying
                if let Some(arc) = self.arena_guard.data().get(info.arena_idx) {
                    self.current_row = Row::from_arc(CompactArc::clone(arc));
                    self.current_arc = Some(CompactArc::clone(arc));
                } else {
                    self.current_arc = None;
                }
            } else {
                // Fallback path: chain entry data stored in fallback_data
                let fb_idx = info.arena_idx - self.arena_len;
                if let Some(arc) = self.fallback_data.get(fb_idx) {
                    self.current_row = Row::from_arc(CompactArc::clone(arc));
                    self.current_arc = Some(CompactArc::clone(arc));
                } else {
                    self.current_arc = None;
                }
            }

            self.current_pos += 1;
            true
        } else {
            false
        }
    }

    /// Get current row as Arc slice (for efficient access)
    #[inline]
    pub fn row_arc_slice(&self) -> Option<&CompactArc<[Value]>> {
        self.current_arc.as_ref()
    }

    /// Get current row ID
    #[inline]
    pub fn row_id(&self) -> i64 {
        if self.current_pos == 0 || self.current_pos > self.visible_indices.len() {
            return 0;
        }
        self.visible_indices[self.current_pos - 1].row_id
    }

    /// Get current row (requires temporary copy for compatibility)
    #[inline]
    pub fn row(&self) -> &Row {
        &self.current_row
    }

    /// Get a specific column value from current row (ZERO-COPY!)
    #[inline]
    pub fn get(&self, col: usize) -> Option<&Value> {
        self.current_arc.as_ref().and_then(|arc| arc.get(col))
    }

    /// Get remaining count
    #[inline]
    pub fn remaining(&self) -> usize {
        self.visible_indices.len().saturating_sub(self.current_pos)
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_pos = 0;
        self.current_arc = None;
    }

    /// Create an AggregationScanner for fast direct aggregations
    ///
    /// This provides the fastest possible aggregation path:
    /// - Zero allocations during computation
    /// - Direct memory access to arena data
    /// - Single pass through visible rows
    pub fn as_aggregation_scanner(&self) -> AggregationScanner<'_> {
        AggregationScanner::new(
            self.arena_guard.data(),
            &self.visible_indices,
            &self.fallback_data,
            self.arena_len,
        )
    }
}

/// Fast aggregation helper that works directly on arena data
///
/// Enables aggregations in a single pass over contiguous memory with
/// zero allocations during the scan. Supports both arena and fallback data.
pub struct AggregationScanner<'a> {
    arena_data: &'a [CompactArc<[Value]>],
    visible_indices: &'a [VisibleRowInfo],
    fallback_data: &'a [CompactArc<[Value]>],
    arena_len: usize,
}

impl<'a> AggregationScanner<'a> {
    pub fn new(
        arena_data: &'a [CompactArc<[Value]>],
        visible_indices: &'a [VisibleRowInfo],
        fallback_data: &'a [CompactArc<[Value]>],
        arena_len: usize,
    ) -> Self {
        Self {
            arena_data,
            visible_indices,
            fallback_data,
            arena_len,
        }
    }

    /// Get the row data for a given VisibleRowInfo (arena or fallback)
    #[inline(always)]
    fn get_row(&self, info: &VisibleRowInfo) -> Option<&CompactArc<[Value]>> {
        if info.arena_idx < self.arena_len {
            self.arena_data.get(info.arena_idx)
        } else {
            self.fallback_data.get(info.arena_idx - self.arena_len)
        }
    }

    /// Sum a column (for INTEGER/FLOAT columns)
    #[inline]
    pub fn sum_column(&self, col_idx: usize) -> f64 {
        let mut sum = 0.0f64;
        for info in self.visible_indices {
            if let Some(row) = self.get_row(info) {
                if let Some(val) = row.get(col_idx) {
                    match val {
                        Value::Integer(i) => sum += *i as f64,
                        Value::Float(f) => sum += *f,
                        _ => {}
                    }
                }
            }
        }
        sum
    }

    /// Count rows
    #[inline]
    pub fn count(&self) -> usize {
        self.visible_indices.len()
    }

    /// Count non-null values in a column
    #[inline]
    pub fn count_column(&self, col_idx: usize) -> usize {
        let mut count = 0;
        for info in self.visible_indices {
            if let Some(row) = self.get_row(info) {
                if let Some(val) = row.get(col_idx) {
                    if !val.is_null() {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// Get min value in a column
    pub fn min_column(&self, col_idx: usize) -> Option<Value> {
        let mut min: Option<Value> = None;
        for info in self.visible_indices {
            if let Some(row) = self.get_row(info) {
                if let Some(val) = row.get(col_idx) {
                    if !val.is_null() {
                        match &min {
                            None => min = Some(val.clone()),
                            Some(current) => {
                                if let Ok(ord) = val.compare(current) {
                                    if ord == std::cmp::Ordering::Less {
                                        min = Some(val.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        min
    }

    /// Get max value in a column
    pub fn max_column(&self, col_idx: usize) -> Option<Value> {
        let mut max: Option<Value> = None;
        for info in self.visible_indices {
            if let Some(row) = self.get_row(info) {
                if let Some(val) = row.get(col_idx) {
                    if !val.is_null() {
                        match &max {
                            None => max = Some(val.clone()),
                            Some(current) => {
                                if let Ok(ord) = val.compare(current) {
                                    if ord == std::cmp::Ordering::Greater {
                                        max = Some(val.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::DataType;
    use crate::storage::mvcc::arena::RowArena;

    #[test]
    fn test_aggregation_scanner() {
        // Create test data using RowArena
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(1), Value::Float(10.0)]);
        arena.insert(2, 1, &[Value::Integer(2), Value::Float(20.0)]);
        arena.insert(3, 1, &[Value::Integer(3), Value::Float(30.0)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        assert_eq!(scanner.count(), 3);
        assert_eq!(scanner.sum_column(0), 6.0); // 1 + 2 + 3
        assert_eq!(scanner.sum_column(1), 60.0); // 10 + 20 + 30
    }

    #[test]
    fn test_streaming_result_empty() {
        let arena = RowArena::new();
        let guard = arena.read_guard();
        let visible: Vec<VisibleRowInfo> = vec![];
        let columns = vec!["id".to_string(), "value".to_string()];

        let mut result = StreamingResult::new(guard, visible, columns);

        // Should return false immediately on empty
        assert!(!result.next());
        assert_eq!(result.remaining(), 0);
        assert!(result.row_arc_slice().is_none());
        assert_eq!(result.row_id(), 0);
        assert!(result.get(0).is_none());
    }

    #[test]
    fn test_streaming_result_iteration() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(100), Value::Text("a".into())]);
        arena.insert(2, 1, &[Value::Integer(200), Value::Text("b".into())]);
        arena.insert(3, 1, &[Value::Integer(300), Value::Text("c".into())]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
        ];
        let columns = vec!["id".to_string(), "name".to_string()];

        let mut result = StreamingResult::new(guard, visible, columns);

        // Check initial state
        assert_eq!(result.remaining(), 3);
        assert_eq!(result.columns(), &["id", "name"]);

        // First row
        assert!(result.next());
        assert_eq!(result.remaining(), 2);
        assert_eq!(result.row_id(), 1);
        assert_eq!(result.get(0), Some(&Value::Integer(100)));
        assert_eq!(result.get(1), Some(&Value::Text("a".into())));
        assert!(result.get(2).is_none()); // Out of bounds

        // Second row
        assert!(result.next());
        assert_eq!(result.remaining(), 1);
        assert_eq!(result.row_id(), 2);
        assert_eq!(result.get(0), Some(&Value::Integer(200)));

        // Third row
        assert!(result.next());
        assert_eq!(result.remaining(), 0);
        assert_eq!(result.row_id(), 3);
        assert_eq!(result.get(0), Some(&Value::Integer(300)));

        // Exhausted
        assert!(!result.next());
        assert_eq!(result.remaining(), 0);
    }

    #[test]
    fn test_streaming_result_reset() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(1)]);
        arena.insert(2, 1, &[Value::Integer(2)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
        ];
        let columns = vec!["id".to_string()];

        let mut result = StreamingResult::new(guard, visible, columns);

        // Iterate to end
        assert!(result.next());
        assert!(result.next());
        assert!(!result.next());
        assert_eq!(result.remaining(), 0);

        // Reset
        result.reset();
        assert_eq!(result.remaining(), 2);
        assert!(result.current_arc.is_none());

        // Can iterate again
        assert!(result.next());
        assert_eq!(result.row_id(), 1);
    }

    #[test]
    fn test_streaming_result_row_id_edge_cases() {
        let arena = RowArena::new();
        arena.insert(100, 1, &[Value::Integer(1)]);

        let guard = arena.read_guard();
        let visible = vec![VisibleRowInfo {
            row_id: 100,
            arena_idx: 0,
        }];
        let columns = vec!["id".to_string()];

        let mut result = StreamingResult::new(guard, visible, columns);

        // Before first next() - current_pos is 0
        assert_eq!(result.row_id(), 0);

        // After next() - current_pos is now 1
        assert!(result.next());
        assert_eq!(result.row_id(), 100);

        // After exhaustion - current_pos remains 1 (== visible_indices.len())
        // next() returns false without incrementing, so row_id() still returns last row's ID
        assert!(!result.next());
        assert_eq!(result.row_id(), 100);
    }

    #[test]
    fn test_streaming_result_row_id_with_many_rows() {
        let arena = RowArena::new();
        arena.insert(10, 1, &[Value::Integer(1)]);
        arena.insert(20, 1, &[Value::Integer(2)]);
        arena.insert(30, 1, &[Value::Integer(3)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 10,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 20,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 30,
                arena_idx: 2,
            },
        ];
        let columns = vec!["id".to_string()];

        let mut result = StreamingResult::new(guard, visible, columns);

        // Before iteration
        assert_eq!(result.row_id(), 0);

        // Iterate through all rows
        assert!(result.next());
        assert_eq!(result.row_id(), 10);

        assert!(result.next());
        assert_eq!(result.row_id(), 20);

        assert!(result.next());
        assert_eq!(result.row_id(), 30);

        // Exhausted
        assert!(!result.next());
        assert_eq!(result.row_id(), 30); // Last valid row
    }

    #[test]
    fn test_streaming_result_row_slice_and_row() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(42), Value::Float(3.5)]);

        let guard = arena.read_guard();
        let visible = vec![VisibleRowInfo {
            row_id: 1,
            arena_idx: 0,
        }];
        let columns = vec!["num".to_string(), "val".to_string()];

        let mut result = StreamingResult::new(guard, visible, columns);

        // Before next(), row_slice should be empty
        assert!(result.row_arc_slice().is_none());

        // After next()
        assert!(result.next());
        let slice = result.row_arc_slice().unwrap();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], Value::Integer(42));
        assert_eq!(slice[1], Value::Float(3.5));

        // row() should also work
        let row = result.row();
        assert_eq!(row.len(), 2);
    }

    #[test]
    fn test_streaming_result_invalid_arena_index() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(1)]);

        let guard = arena.read_guard();
        // Invalid arena_idx (999 doesn't exist)
        let visible = vec![VisibleRowInfo {
            row_id: 1,
            arena_idx: 999,
        }];
        let columns = vec!["id".to_string()];

        let mut result = StreamingResult::new(guard, visible, columns);

        // Should still advance but current_arc will be None
        assert!(result.next());
        assert!(result.current_arc.is_none());
        assert!(result.row_arc_slice().is_none());
    }

    #[test]
    fn test_streaming_result_as_aggregation_scanner() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(10)]);
        arena.insert(2, 1, &[Value::Integer(20)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
        ];
        let columns = vec!["value".to_string()];

        let result = StreamingResult::new(guard, visible, columns);
        let scanner = result.as_aggregation_scanner();

        assert_eq!(scanner.count(), 2);
        assert_eq!(scanner.sum_column(0), 30.0);
    }

    #[test]
    fn test_aggregation_scanner_empty() {
        let arena = RowArena::new();
        let guard = arena.read_guard();
        let visible: Vec<VisibleRowInfo> = vec![];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        assert_eq!(scanner.count(), 0);
        assert_eq!(scanner.sum_column(0), 0.0);
        assert_eq!(scanner.count_column(0), 0);
        assert!(scanner.min_column(0).is_none());
        assert!(scanner.max_column(0).is_none());
    }

    #[test]
    fn test_aggregation_scanner_with_nulls() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(10), Value::Null(DataType::Integer)]);
        arena.insert(2, 1, &[Value::Null(DataType::Integer), Value::Integer(20)]);
        arena.insert(3, 1, &[Value::Integer(30), Value::Integer(30)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        // count() returns all visible rows
        assert_eq!(scanner.count(), 3);

        // count_column() returns non-null count
        assert_eq!(scanner.count_column(0), 2); // 10, 30 (NULL skipped)
        assert_eq!(scanner.count_column(1), 2); // 20, 30 (NULL skipped)

        // sum_column() skips NULLs
        assert_eq!(scanner.sum_column(0), 40.0); // 10 + 30
        assert_eq!(scanner.sum_column(1), 50.0); // 20 + 30
    }

    #[test]
    fn test_aggregation_scanner_all_nulls() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Null(DataType::Integer)]);
        arena.insert(2, 1, &[Value::Null(DataType::Integer)]);
        arena.insert(3, 1, &[Value::Null(DataType::Integer)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        assert_eq!(scanner.count(), 3);
        assert_eq!(scanner.count_column(0), 0);
        assert_eq!(scanner.sum_column(0), 0.0);
        assert!(scanner.min_column(0).is_none());
        assert!(scanner.max_column(0).is_none());
    }

    #[test]
    fn test_aggregation_scanner_min_max() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(50)]);
        arena.insert(2, 1, &[Value::Integer(10)]);
        arena.insert(3, 1, &[Value::Integer(30)]);
        arena.insert(4, 1, &[Value::Null(DataType::Integer)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
            VisibleRowInfo {
                row_id: 4,
                arena_idx: 3,
            },
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        assert_eq!(scanner.min_column(0), Some(Value::Integer(10)));
        assert_eq!(scanner.max_column(0), Some(Value::Integer(50)));
    }

    #[test]
    fn test_aggregation_scanner_min_max_floats() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Float(3.15)]);
        arena.insert(2, 1, &[Value::Float(2.72)]);
        arena.insert(3, 1, &[Value::Float(1.42)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        assert_eq!(scanner.min_column(0), Some(Value::Float(1.42)));
        assert_eq!(scanner.max_column(0), Some(Value::Float(3.15)));
    }

    #[test]
    fn test_aggregation_scanner_min_max_text() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Text("banana".into())]);
        arena.insert(2, 1, &[Value::Text("apple".into())]);
        arena.insert(3, 1, &[Value::Text("cherry".into())]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        assert_eq!(scanner.min_column(0), Some(Value::Text("apple".into())));
        assert_eq!(scanner.max_column(0), Some(Value::Text("cherry".into())));
    }

    #[test]
    fn test_aggregation_scanner_sum_non_numeric() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Text("hello".into())]);
        arena.insert(2, 1, &[Value::Boolean(true)]);
        arena.insert(3, 1, &[Value::Integer(10)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        // sum_column only sums Integer and Float, ignores others
        assert_eq!(scanner.sum_column(0), 10.0);
    }

    #[test]
    fn test_aggregation_scanner_mixed_numeric() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(10)]);
        arena.insert(2, 1, &[Value::Float(20.5)]);
        arena.insert(3, 1, &[Value::Integer(30)]);

        let guard = arena.read_guard();
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 1,
            },
            VisibleRowInfo {
                row_id: 3,
                arena_idx: 2,
            },
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        assert_eq!(scanner.sum_column(0), 60.5);
    }

    #[test]
    fn test_aggregation_scanner_invalid_column() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(10)]);

        let guard = arena.read_guard();
        let visible = vec![VisibleRowInfo {
            row_id: 1,
            arena_idx: 0,
        }];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        // Column 99 doesn't exist
        assert_eq!(scanner.sum_column(99), 0.0);
        assert_eq!(scanner.count_column(99), 0);
        assert!(scanner.min_column(99).is_none());
        assert!(scanner.max_column(99).is_none());
    }

    #[test]
    fn test_aggregation_scanner_invalid_arena_index() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(10)]);

        let guard = arena.read_guard();
        // Include an invalid arena index
        let visible = vec![
            VisibleRowInfo {
                row_id: 1,
                arena_idx: 0,
            },
            VisibleRowInfo {
                row_id: 2,
                arena_idx: 999,
            }, // Invalid
        ];

        let scanner = AggregationScanner::new(guard.data(), &visible, &[], guard.len());

        // Should gracefully skip invalid indices
        assert_eq!(scanner.count(), 2);
        assert_eq!(scanner.sum_column(0), 10.0); // Only valid row counted
        assert_eq!(scanner.count_column(0), 1); // Only valid row counted
    }

    #[test]
    fn test_visible_row_info_clone_copy() {
        let info = VisibleRowInfo {
            row_id: 42,
            arena_idx: 5,
        };

        // Test Copy trait
        let copied = info;
        assert_eq!(copied.row_id, 42);
        assert_eq!(copied.arena_idx, 5);

        // Test Clone trait (use Clone::clone to avoid clone_on_copy warning)
        let cloned = Clone::clone(&info);
        assert_eq!(cloned.row_id, 42);
        assert_eq!(cloned.arena_idx, 5);
    }

    #[test]
    fn test_streaming_result_single_row() {
        let arena = RowArena::new();
        arena.insert(1, 1, &[Value::Integer(42)]);

        let guard = arena.read_guard();
        let visible = vec![VisibleRowInfo {
            row_id: 1,
            arena_idx: 0,
        }];
        let columns = vec!["value".to_string()];

        let mut result = StreamingResult::new(guard, visible, columns);

        assert_eq!(result.remaining(), 1);
        assert!(result.next());
        assert_eq!(result.remaining(), 0);
        assert_eq!(result.get(0), Some(&Value::Integer(42)));
        assert!(!result.next());
    }
}
