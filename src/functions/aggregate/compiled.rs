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

//! Compiled aggregate functions for zero-dispatch hot path execution
//!
//! This module provides `CompiledAggregate`, an enum-based specialization of aggregate
//! functions that eliminates virtual dispatch overhead from `Box<dyn AggregateFunction>`.
//!
//! # Performance
//!
//! Traditional aggregate functions use trait objects:
//! ```text
//! for row in rows {
//!     func.accumulate(value, distinct);  // vtable lookup per row
//! }
//! ```
//!
//! With `CompiledAggregate`, the dispatch becomes a direct enum match:
//! ```text
//! for row in rows {
//!     compiled_agg.accumulate(value);  // inline match, no vtable
//! }
//! ```
//!
//! Expected speedup: 2-5x for aggregation-heavy queries.

use crate::core::Value;
use crate::functions::AggregateFunction;

use super::DistinctTracker;

/// Sum state for tracking integer vs float sums
#[derive(Debug, Clone, Default)]
pub enum SumState {
    #[default]
    Empty,
    Integer(i64),
    Float(f64),
}

/// Compiled aggregate function - enum-based specialization for zero virtual dispatch
///
/// This enum provides specialized implementations for the most common aggregate
/// functions (COUNT, SUM, AVG, MIN, MAX), with a `Dynamic` fallback for complex
/// or rare aggregates (STRING_AGG, ARRAY_AGG, MEDIAN, etc.).
pub enum CompiledAggregate {
    /// COUNT(*) - counts all rows
    CountStar { count: i64 },

    /// COUNT(column) - counts non-NULL values
    Count { count: i64 },

    /// COUNT(DISTINCT column) - counts distinct non-NULL values
    CountDistinct { distinct_tracker: DistinctTracker },

    /// SUM(column) - sums numeric values
    Sum { state: SumState },

    /// SUM(DISTINCT column) - sums distinct numeric values
    SumDistinct {
        state: SumState,
        distinct_tracker: DistinctTracker,
    },

    /// AVG(column) - average of numeric values
    Avg { sum: f64, count: i64 },

    /// AVG(DISTINCT column) - average of distinct numeric values
    AvgDistinct {
        sum: f64,
        count: i64,
        distinct_tracker: DistinctTracker,
    },

    /// MIN(column) - minimum value (type-generic using Value comparison)
    Min { min_value: Option<Value> },

    /// MAX(column) - maximum value (type-generic using Value comparison)
    Max { max_value: Option<Value> },

    /// MIN for integers only (faster path)
    MinInteger { min_value: Option<i64> },

    /// MAX for integers only (faster path)
    MaxInteger { max_value: Option<i64> },

    /// MIN for floats only (faster path)
    MinFloat { min_value: Option<f64> },

    /// MAX for floats only (faster path)
    MaxFloat { max_value: Option<f64> },

    /// Fallback to dynamic dispatch for complex aggregates
    Dynamic(Box<dyn AggregateFunction>),
}

impl std::fmt::Debug for CompiledAggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompiledAggregate::CountStar { count } => {
                f.debug_struct("CountStar").field("count", count).finish()
            }
            CompiledAggregate::Count { count } => {
                f.debug_struct("Count").field("count", count).finish()
            }
            CompiledAggregate::CountDistinct { distinct_tracker } => f
                .debug_struct("CountDistinct")
                .field("distinct_tracker", distinct_tracker)
                .finish(),
            CompiledAggregate::Sum { state } => {
                f.debug_struct("Sum").field("state", state).finish()
            }
            CompiledAggregate::SumDistinct {
                state,
                distinct_tracker,
            } => f
                .debug_struct("SumDistinct")
                .field("state", state)
                .field("distinct_tracker", distinct_tracker)
                .finish(),
            CompiledAggregate::Avg { sum, count } => f
                .debug_struct("Avg")
                .field("sum", sum)
                .field("count", count)
                .finish(),
            CompiledAggregate::AvgDistinct {
                sum,
                count,
                distinct_tracker,
            } => f
                .debug_struct("AvgDistinct")
                .field("sum", sum)
                .field("count", count)
                .field("distinct_tracker", distinct_tracker)
                .finish(),
            CompiledAggregate::Min { min_value } => {
                f.debug_struct("Min").field("min_value", min_value).finish()
            }
            CompiledAggregate::Max { max_value } => {
                f.debug_struct("Max").field("max_value", max_value).finish()
            }
            CompiledAggregate::MinInteger { min_value } => f
                .debug_struct("MinInteger")
                .field("min_value", min_value)
                .finish(),
            CompiledAggregate::MaxInteger { max_value } => f
                .debug_struct("MaxInteger")
                .field("max_value", max_value)
                .finish(),
            CompiledAggregate::MinFloat { min_value } => f
                .debug_struct("MinFloat")
                .field("min_value", min_value)
                .finish(),
            CompiledAggregate::MaxFloat { max_value } => f
                .debug_struct("MaxFloat")
                .field("max_value", max_value)
                .finish(),
            CompiledAggregate::Dynamic(func) => {
                f.debug_tuple("Dynamic").field(&func.name()).finish()
            }
        }
    }
}

impl CompiledAggregate {
    /// Create a compiled COUNT(*) aggregate
    pub fn count_star() -> Self {
        CompiledAggregate::CountStar { count: 0 }
    }

    /// Create a compiled COUNT(column) aggregate
    pub fn count(distinct: bool) -> Self {
        if distinct {
            CompiledAggregate::CountDistinct {
                distinct_tracker: DistinctTracker::default(),
            }
        } else {
            CompiledAggregate::Count { count: 0 }
        }
    }

    /// Create a compiled SUM aggregate
    pub fn sum(distinct: bool) -> Self {
        if distinct {
            CompiledAggregate::SumDistinct {
                state: SumState::Empty,
                distinct_tracker: DistinctTracker::default(),
            }
        } else {
            CompiledAggregate::Sum {
                state: SumState::Empty,
            }
        }
    }

    /// Create a compiled AVG aggregate
    pub fn avg(distinct: bool) -> Self {
        if distinct {
            CompiledAggregate::AvgDistinct {
                sum: 0.0,
                count: 0,
                distinct_tracker: DistinctTracker::default(),
            }
        } else {
            CompiledAggregate::Avg { sum: 0.0, count: 0 }
        }
    }

    /// Create a compiled MIN aggregate (generic)
    pub fn min() -> Self {
        CompiledAggregate::Min { min_value: None }
    }

    /// Create a compiled MIN aggregate for integers
    pub fn min_integer() -> Self {
        CompiledAggregate::MinInteger { min_value: None }
    }

    /// Create a compiled MIN aggregate for floats
    pub fn min_float() -> Self {
        CompiledAggregate::MinFloat { min_value: None }
    }

    /// Create a compiled MAX aggregate (generic)
    pub fn max() -> Self {
        CompiledAggregate::Max { max_value: None }
    }

    /// Create a compiled MAX aggregate for integers
    pub fn max_integer() -> Self {
        CompiledAggregate::MaxInteger { max_value: None }
    }

    /// Create a compiled MAX aggregate for floats
    pub fn max_float() -> Self {
        CompiledAggregate::MaxFloat { max_value: None }
    }

    /// Create from a dynamic aggregate function (fallback)
    pub fn dynamic(func: Box<dyn AggregateFunction>) -> Self {
        CompiledAggregate::Dynamic(func)
    }

    /// Compile from a function name and configuration
    ///
    /// Returns a compiled aggregate for common functions, or wraps
    /// the provided dynamic function for complex/rare aggregates.
    pub fn compile(
        name: &str,
        is_count_star: bool,
        distinct: bool,
        dynamic_fallback: Option<Box<dyn AggregateFunction>>,
    ) -> Option<Self> {
        if name.eq_ignore_ascii_case("COUNT") {
            if is_count_star {
                Some(CompiledAggregate::count_star())
            } else {
                Some(CompiledAggregate::count(distinct))
            }
        } else if name.eq_ignore_ascii_case("SUM") {
            Some(CompiledAggregate::sum(distinct))
        } else if name.eq_ignore_ascii_case("AVG") {
            Some(CompiledAggregate::avg(distinct))
        } else if name.eq_ignore_ascii_case("MIN") {
            Some(CompiledAggregate::min())
        } else if name.eq_ignore_ascii_case("MAX") {
            Some(CompiledAggregate::max())
        } else {
            // Complex aggregates use dynamic fallback
            dynamic_fallback.map(CompiledAggregate::Dynamic)
        }
    }

    /// Accumulate a value into the aggregate
    ///
    /// This is the hot path - all code here should be as fast as possible.
    #[inline(always)]
    pub fn accumulate(&mut self, value: &Value) {
        match self {
            // COUNT(*) - always increment
            CompiledAggregate::CountStar { count } => {
                *count += 1;
            }

            // COUNT(column) - increment for non-NULL
            CompiledAggregate::Count { count } => {
                if !value.is_null() {
                    *count += 1;
                }
            }

            // COUNT(DISTINCT column) - track distinct non-NULL values
            CompiledAggregate::CountDistinct { distinct_tracker } => {
                if !value.is_null() {
                    distinct_tracker.check_and_add(value);
                }
            }

            // SUM(column) - add numeric values
            CompiledAggregate::Sum { state } => {
                if !value.is_null() {
                    Self::accumulate_sum(state, value);
                }
            }

            // SUM(DISTINCT column) - add distinct numeric values
            CompiledAggregate::SumDistinct {
                state,
                distinct_tracker,
            } => {
                if !value.is_null() && distinct_tracker.check_and_add(value) {
                    Self::accumulate_sum(state, value);
                }
            }

            // AVG(column) - track sum and count
            CompiledAggregate::Avg { sum, count } => {
                if let Some(n) = Self::as_f64(value) {
                    *sum += n;
                    *count += 1;
                }
            }

            // AVG(DISTINCT column) - track sum and count for distinct values
            CompiledAggregate::AvgDistinct {
                sum,
                count,
                distinct_tracker,
            } => {
                if !value.is_null() && distinct_tracker.check_and_add(value) {
                    if let Some(n) = Self::as_f64(value) {
                        *sum += n;
                        *count += 1;
                    }
                }
            }

            // MIN(column) - generic comparison
            CompiledAggregate::Min { min_value } => {
                if !value.is_null() {
                    match min_value {
                        None => *min_value = Some(value.clone()),
                        Some(current) => {
                            if Self::is_less_than(value, current) {
                                *min_value = Some(value.clone());
                            }
                        }
                    }
                }
            }

            // MAX(column) - generic comparison
            CompiledAggregate::Max { max_value } => {
                if !value.is_null() {
                    match max_value {
                        None => *max_value = Some(value.clone()),
                        Some(current) => {
                            if Self::is_greater_than(value, current) {
                                *max_value = Some(value.clone());
                            }
                        }
                    }
                }
            }

            // MIN for integers (faster path)
            CompiledAggregate::MinInteger { min_value } => {
                if let Value::Integer(v) = value {
                    match min_value {
                        None => *min_value = Some(*v),
                        Some(current) if *v < *current => *min_value = Some(*v),
                        _ => {}
                    }
                }
            }

            // MAX for integers (faster path)
            CompiledAggregate::MaxInteger { max_value } => {
                if let Value::Integer(v) = value {
                    match max_value {
                        None => *max_value = Some(*v),
                        Some(current) if *v > *current => *max_value = Some(*v),
                        _ => {}
                    }
                }
            }

            // MIN for floats (faster path)
            CompiledAggregate::MinFloat { min_value } => {
                if let Value::Float(v) = value {
                    match min_value {
                        None => *min_value = Some(*v),
                        Some(current) if *v < *current => *min_value = Some(*v),
                        _ => {}
                    }
                }
            }

            // MAX for floats (faster path)
            CompiledAggregate::MaxFloat { max_value } => {
                if let Value::Float(v) = value {
                    match max_value {
                        None => *max_value = Some(*v),
                        Some(current) if *v > *current => *max_value = Some(*v),
                        _ => {}
                    }
                }
            }

            // Dynamic fallback
            CompiledAggregate::Dynamic(func) => {
                func.accumulate(value, false);
            }
        }
    }

    /// Accumulate with DISTINCT flag (for dynamic fallback compatibility)
    #[inline(always)]
    pub fn accumulate_with_distinct(&mut self, value: &Value, distinct: bool) {
        if let CompiledAggregate::Dynamic(func) = self {
            func.accumulate(value, distinct);
        } else {
            // For compiled variants, DISTINCT is handled by the variant type
            self.accumulate(value);
        }
    }

    /// Get the result of the aggregation
    #[inline]
    pub fn result(&self) -> Value {
        match self {
            CompiledAggregate::CountStar { count } => Value::Integer(*count),
            CompiledAggregate::Count { count } => Value::Integer(*count),
            CompiledAggregate::CountDistinct { distinct_tracker } => {
                Value::Integer(distinct_tracker.count() as i64)
            }

            CompiledAggregate::Sum { state } | CompiledAggregate::SumDistinct { state, .. } => {
                match state {
                    SumState::Empty => Value::null_unknown(),
                    SumState::Integer(sum) => Value::Integer(*sum),
                    SumState::Float(sum) => Value::Float(*sum),
                }
            }

            CompiledAggregate::Avg { sum, count }
            | CompiledAggregate::AvgDistinct { sum, count, .. } => {
                if *count == 0 {
                    Value::null_unknown()
                } else {
                    Value::Float(*sum / *count as f64)
                }
            }

            CompiledAggregate::Min { min_value } => {
                min_value.clone().unwrap_or_else(Value::null_unknown)
            }
            CompiledAggregate::Max { max_value } => {
                max_value.clone().unwrap_or_else(Value::null_unknown)
            }

            CompiledAggregate::MinInteger { min_value } => min_value
                .map(Value::Integer)
                .unwrap_or_else(Value::null_unknown),
            CompiledAggregate::MaxInteger { max_value } => max_value
                .map(Value::Integer)
                .unwrap_or_else(Value::null_unknown),

            CompiledAggregate::MinFloat { min_value } => min_value
                .map(Value::Float)
                .unwrap_or_else(Value::null_unknown),
            CompiledAggregate::MaxFloat { max_value } => max_value
                .map(Value::Float)
                .unwrap_or_else(Value::null_unknown),

            CompiledAggregate::Dynamic(func) => func.result(),
        }
    }

    /// Reset the aggregate state
    pub fn reset(&mut self) {
        match self {
            CompiledAggregate::CountStar { count } => *count = 0,
            CompiledAggregate::Count { count } => *count = 0,
            CompiledAggregate::CountDistinct { distinct_tracker } => distinct_tracker.reset(),

            CompiledAggregate::Sum { state } => *state = SumState::Empty,
            CompiledAggregate::SumDistinct {
                state,
                distinct_tracker,
            } => {
                *state = SumState::Empty;
                distinct_tracker.reset();
            }

            CompiledAggregate::Avg { sum, count } => {
                *sum = 0.0;
                *count = 0;
            }
            CompiledAggregate::AvgDistinct {
                sum,
                count,
                distinct_tracker,
            } => {
                *sum = 0.0;
                *count = 0;
                distinct_tracker.reset();
            }

            CompiledAggregate::Min { min_value } => *min_value = None,
            CompiledAggregate::Max { max_value } => *max_value = None,

            CompiledAggregate::MinInteger { min_value } => *min_value = None,
            CompiledAggregate::MaxInteger { max_value } => *max_value = None,

            CompiledAggregate::MinFloat { min_value } => *min_value = None,
            CompiledAggregate::MaxFloat { max_value } => *max_value = None,

            CompiledAggregate::Dynamic(func) => func.reset(),
        }
    }

    /// Helper: accumulate into SumState
    #[inline(always)]
    fn accumulate_sum(state: &mut SumState, value: &Value) {
        match value {
            Value::Integer(i) => match state {
                SumState::Empty => *state = SumState::Integer(*i),
                SumState::Integer(sum) => *sum += i,
                SumState::Float(sum) => *sum += *i as f64,
            },
            Value::Float(f) => match state {
                SumState::Empty => *state = SumState::Float(*f),
                SumState::Integer(sum) => {
                    *state = SumState::Float(*sum as f64 + f);
                }
                SumState::Float(sum) => *sum += f,
            },
            _ => {} // Ignore non-numeric types
        }
    }

    /// Helper: convert Value to f64
    #[inline(always)]
    fn as_f64(value: &Value) -> Option<f64> {
        match value {
            Value::Integer(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Helper: compare values (a < b)
    #[inline(always)]
    fn is_less_than(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null(_), _) | (_, Value::Null(_)) => false,
            (Value::Integer(a), Value::Integer(b)) => a < b,
            (Value::Float(a), Value::Float(b)) => a < b,
            (Value::Integer(a), Value::Float(b)) => (*a as f64) < *b,
            (Value::Float(a), Value::Integer(b)) => *a < (*b as f64),
            (Value::Text(a), Value::Text(b)) => a < b,
            (Value::Boolean(a), Value::Boolean(b)) => !a && *b,
            (Value::Timestamp(a), Value::Timestamp(b)) => a < b,
            _ => false,
        }
    }

    /// Helper: compare values (a > b)
    #[inline(always)]
    fn is_greater_than(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null(_), _) | (_, Value::Null(_)) => false,
            (Value::Integer(a), Value::Integer(b)) => a > b,
            (Value::Float(a), Value::Float(b)) => a > b,
            (Value::Integer(a), Value::Float(b)) => (*a as f64) > *b,
            (Value::Float(a), Value::Integer(b)) => *a > (*b as f64),
            (Value::Text(a), Value::Text(b)) => a > b,
            (Value::Boolean(a), Value::Boolean(b)) => *a && !b,
            (Value::Timestamp(a), Value::Timestamp(b)) => a > b,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_star() {
        let mut agg = CompiledAggregate::count_star();
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::null_unknown());
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result(), Value::Integer(3)); // Counts all rows including NULL
    }

    #[test]
    fn test_count() {
        let mut agg = CompiledAggregate::count(false);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::null_unknown());
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result(), Value::Integer(2)); // Skips NULL
    }

    #[test]
    fn test_count_distinct() {
        let mut agg = CompiledAggregate::count(true);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(1)); // duplicate
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::null_unknown()); // NULL ignored
        assert_eq!(agg.result(), Value::Integer(2));
    }

    #[test]
    fn test_sum_integers() {
        let mut agg = CompiledAggregate::sum(false);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result(), Value::Integer(6));
    }

    #[test]
    fn test_sum_floats() {
        let mut agg = CompiledAggregate::sum(false);
        agg.accumulate(&Value::Float(1.5));
        agg.accumulate(&Value::Float(2.5));
        assert_eq!(agg.result(), Value::Float(4.0));
    }

    #[test]
    fn test_sum_mixed() {
        let mut agg = CompiledAggregate::sum(false);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Float(2.5));
        assert_eq!(agg.result(), Value::Float(3.5));
    }

    #[test]
    fn test_sum_distinct() {
        let mut agg = CompiledAggregate::sum(true);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(1)); // duplicate
        agg.accumulate(&Value::Integer(2));
        assert_eq!(agg.result(), Value::Integer(3)); // 1 + 2
    }

    #[test]
    fn test_avg() {
        let mut agg = CompiledAggregate::avg(false);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result(), Value::Float(2.0));
    }

    #[test]
    fn test_avg_distinct() {
        let mut agg = CompiledAggregate::avg(true);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(1)); // duplicate
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result(), Value::Float(2.0)); // (1 + 3) / 2
    }

    #[test]
    fn test_min_integers() {
        let mut agg = CompiledAggregate::min();
        agg.accumulate(&Value::Integer(5));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(8));
        assert_eq!(agg.result(), Value::Integer(2));
    }

    #[test]
    fn test_max_integers() {
        let mut agg = CompiledAggregate::max();
        agg.accumulate(&Value::Integer(5));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(8));
        assert_eq!(agg.result(), Value::Integer(8));
    }

    #[test]
    fn test_min_integer_fast() {
        let mut agg = CompiledAggregate::min_integer();
        agg.accumulate(&Value::Integer(5));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(8));
        assert_eq!(agg.result(), Value::Integer(2));
    }

    #[test]
    fn test_max_integer_fast() {
        let mut agg = CompiledAggregate::max_integer();
        agg.accumulate(&Value::Integer(5));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(8));
        assert_eq!(agg.result(), Value::Integer(8));
    }

    #[test]
    fn test_min_strings() {
        let mut agg = CompiledAggregate::min();
        agg.accumulate(&Value::text("banana"));
        agg.accumulate(&Value::text("apple"));
        agg.accumulate(&Value::text("cherry"));
        assert_eq!(agg.result(), Value::text("apple"));
    }

    #[test]
    fn test_max_strings() {
        let mut agg = CompiledAggregate::max();
        agg.accumulate(&Value::text("banana"));
        agg.accumulate(&Value::text("apple"));
        agg.accumulate(&Value::text("cherry"));
        assert_eq!(agg.result(), Value::text("cherry"));
    }

    #[test]
    fn test_empty_aggregates() {
        assert!(CompiledAggregate::sum(false).result().is_null());
        assert!(CompiledAggregate::avg(false).result().is_null());
        assert!(CompiledAggregate::min().result().is_null());
        assert!(CompiledAggregate::max().result().is_null());
        assert_eq!(CompiledAggregate::count(false).result(), Value::Integer(0));
        assert_eq!(CompiledAggregate::count_star().result(), Value::Integer(0));
    }

    #[test]
    fn test_reset() {
        let mut agg = CompiledAggregate::sum(false);
        agg.accumulate(&Value::Integer(10));
        agg.reset();
        assert!(agg.result().is_null());

        let mut agg = CompiledAggregate::count(false);
        agg.accumulate(&Value::Integer(10));
        agg.reset();
        assert_eq!(agg.result(), Value::Integer(0));
    }

    #[test]
    fn test_compile() {
        let agg = CompiledAggregate::compile("count", true, false, None);
        assert!(matches!(agg, Some(CompiledAggregate::CountStar { .. })));

        let agg = CompiledAggregate::compile("COUNT", false, false, None);
        assert!(matches!(agg, Some(CompiledAggregate::Count { .. })));

        let agg = CompiledAggregate::compile("sum", false, true, None);
        assert!(matches!(agg, Some(CompiledAggregate::SumDistinct { .. })));

        let agg = CompiledAggregate::compile("avg", false, false, None);
        assert!(matches!(agg, Some(CompiledAggregate::Avg { .. })));

        let agg = CompiledAggregate::compile("min", false, false, None);
        assert!(matches!(agg, Some(CompiledAggregate::Min { .. })));

        let agg = CompiledAggregate::compile("max", false, false, None);
        assert!(matches!(agg, Some(CompiledAggregate::Max { .. })));

        // Unknown without fallback returns None
        let agg = CompiledAggregate::compile("unknown", false, false, None);
        assert!(agg.is_none());
    }
}
