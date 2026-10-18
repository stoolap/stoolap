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

use std::cmp::Ordering;

use crate::core::value::compare_floats;
use crate::core::{Result, SumAccumulator, Value};

use super::{AggregateState, DistinctTracker};

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
    CountDistinct {
        distinct_tracker: Box<DistinctTracker>,
    },

    /// SUM(column) - sums numeric values
    Sum { state: SumAccumulator },

    /// SUM(DISTINCT column) - sums distinct numeric values
    SumDistinct {
        state: SumAccumulator,
        distinct_tracker: Box<DistinctTracker>,
    },

    /// AVG(column) - average of numeric values
    Avg { state: SumAccumulator },

    /// AVG(DISTINCT column) - average of distinct numeric values
    AvgDistinct {
        state: SumAccumulator,
        distinct_tracker: Box<DistinctTracker>,
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

    /// Fallback to dynamic dispatch for complex aggregates, with the call's
    /// modifiers in its state
    Dynamic(AggregateState),
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
            CompiledAggregate::Avg { state } => {
                f.debug_struct("Avg").field("state", state).finish()
            }
            CompiledAggregate::AvgDistinct {
                state,
                distinct_tracker,
            } => f
                .debug_struct("AvgDistinct")
                .field("state", state)
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
            CompiledAggregate::Dynamic(state) => f.debug_tuple("Dynamic").field(state).finish(),
        }
    }
}

/// Puts `value` in place of a MIN or MAX run's `current` extreme when it
/// orders `beyond` it. Two numbers of one type compare as `Value::cmp` has
/// them and copy in place, without a call, a clone or a drop
#[inline(always)]
fn keep_extreme(current: &mut Value, value: &Value, beyond: Ordering) {
    match (&mut *current, value) {
        (Value::Integer(slot), Value::Integer(v)) => {
            if v.cmp(slot) == beyond {
                *slot = *v;
            }
        }
        (Value::Float(slot), Value::Float(v)) => {
            if compare_floats(*v, *slot) == beyond {
                *slot = *v;
            }
        }
        _ => {
            if value.cmp(current) == beyond {
                *current = value.clone();
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
                distinct_tracker: Box::default(),
            }
        } else {
            CompiledAggregate::Count { count: 0 }
        }
    }

    /// Create a compiled SUM aggregate
    pub fn sum(distinct: bool) -> Self {
        if distinct {
            CompiledAggregate::SumDistinct {
                state: SumAccumulator::default(),
                distinct_tracker: Box::default(),
            }
        } else {
            CompiledAggregate::Sum {
                state: SumAccumulator::default(),
            }
        }
    }

    /// Create a compiled AVG aggregate
    pub fn avg(distinct: bool) -> Self {
        if distinct {
            CompiledAggregate::AvgDistinct {
                state: SumAccumulator::default(),
                distinct_tracker: Box::default(),
            }
        } else {
            CompiledAggregate::Avg {
                state: SumAccumulator::default(),
            }
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

    /// Create from an aggregate call's state (fallback)
    pub fn dynamic(state: AggregateState) -> Self {
        CompiledAggregate::Dynamic(state)
    }

    /// Compile from a function name and configuration
    ///
    /// Returns a compiled aggregate for common functions, or wraps the
    /// call's state for complex/rare aggregates.
    pub fn compile(
        name: &str,
        is_count_star: bool,
        distinct: bool,
        dynamic_fallback: Option<AggregateState>,
    ) -> Option<Self> {
        match Self::constructor(name, is_count_star, distinct) {
            Some(make) => Some(make()),
            None => dynamic_fallback.map(CompiledAggregate::Dynamic),
        }
    }

    /// What makes an empty run of a call to `name`, when the call has a
    /// compiled state
    pub fn constructor(
        name: &str,
        is_count_star: bool,
        distinct: bool,
    ) -> Option<fn() -> CompiledAggregate> {
        let make: fn() -> CompiledAggregate = if name.eq_ignore_ascii_case("COUNT") {
            match (is_count_star, distinct) {
                (true, _) => CompiledAggregate::count_star,
                (false, true) => || CompiledAggregate::count(true),
                (false, false) => || CompiledAggregate::count(false),
            }
        } else if name.eq_ignore_ascii_case("SUM") {
            match distinct {
                true => || CompiledAggregate::sum(true),
                false => || CompiledAggregate::sum(false),
            }
        } else if name.eq_ignore_ascii_case("AVG") {
            match distinct {
                true => || CompiledAggregate::avg(true),
                false => || CompiledAggregate::avg(false),
            }
        } else if name.eq_ignore_ascii_case("MIN") {
            CompiledAggregate::min
        } else if name.eq_ignore_ascii_case("MAX") {
            CompiledAggregate::max
        } else {
            return None;
        };
        Some(make)
    }

    /// Whether the call orders its input by its own ORDER BY keys
    pub fn supports_order_by(&self) -> bool {
        match self {
            CompiledAggregate::Dynamic(state) => state.supports_order_by(),
            _ => false,
        }
    }

    /// Accumulates a value with the call's ORDER BY keys for it
    #[inline]
    pub fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>) {
        match self {
            CompiledAggregate::Dynamic(state) => state.accumulate_with_sort_key(value, sort_keys),
            other => other.accumulate(value),
        }
    }

    /// Accumulate a value into the aggregate
    ///
    /// This is the hot path - all code here should be as fast as possible.
    #[inline(always)]
    pub fn accumulate(&mut self, value: &Value) {
        if let CompiledAggregate::CountStar { count } = self {
            *count += 1;
            return;
        }
        if let CompiledAggregate::Sum { state } | CompiledAggregate::Avg { state } = self {
            state.add(value);
            return;
        }
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

            // SUM and AVG add numeric values, a boolean as one or nought
            CompiledAggregate::Sum { state } | CompiledAggregate::Avg { state } => {
                state.add(value);
            }

            // SUM and AVG over DISTINCT add each distinct value once
            CompiledAggregate::SumDistinct {
                state,
                distinct_tracker,
            }
            | CompiledAggregate::AvgDistinct {
                state,
                distinct_tracker,
            } => {
                if !value.is_null() && distinct_tracker.check_and_add(value) {
                    state.add(value);
                }
            }

            // MIN(column) - generic comparison
            CompiledAggregate::Min { min_value } => {
                if !value.is_null() {
                    match min_value {
                        None => *min_value = Some(value.clone()),
                        Some(current) => keep_extreme(current, value, Ordering::Less),
                    }
                }
            }

            // MAX(column) - generic comparison
            CompiledAggregate::Max { max_value } => {
                if !value.is_null() {
                    match max_value {
                        None => *max_value = Some(value.clone()),
                        Some(current) => keep_extreme(current, value, Ordering::Greater),
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
                        Some(current) if compare_floats(*v, *current).is_lt() => {
                            *min_value = Some(*v)
                        }
                        _ => {}
                    }
                }
            }

            // MAX for floats (faster path)
            CompiledAggregate::MaxFloat { max_value } => {
                if let Value::Float(v) = value {
                    match max_value {
                        None => *max_value = Some(*v),
                        Some(current) if compare_floats(*v, *current).is_gt() => {
                            *max_value = Some(*v)
                        }
                        _ => {}
                    }
                }
            }

            // Dynamic fallback
            CompiledAggregate::Dynamic(state) => {
                state.accumulate(value);
            }
        }
    }

    /// Whether runs of this aggregate over disjoint rows merge into the run
    /// over all of them
    pub fn can_merge(&self) -> bool {
        matches!(
            self,
            CompiledAggregate::CountStar { .. }
                | CompiledAggregate::Count { .. }
                | CompiledAggregate::Sum { .. }
                | CompiledAggregate::Avg { .. }
                | CompiledAggregate::Min { .. }
                | CompiledAggregate::Max { .. }
                | CompiledAggregate::MinInteger { .. }
                | CompiledAggregate::MaxInteger { .. }
                | CompiledAggregate::MinFloat { .. }
                | CompiledAggregate::MaxFloat { .. }
        )
    }

    /// Folds a run of the same aggregate over other rows into this one
    pub fn merge(&mut self, other: &CompiledAggregate) {
        debug_assert!(
            self.can_merge() && std::mem::discriminant(self) == std::mem::discriminant(other),
            "merge needs the same mergeable aggregate on both sides"
        );
        let extreme = match other {
            CompiledAggregate::Min { min_value: v } | CompiledAggregate::Max { max_value: v } => {
                v.clone()
            }
            CompiledAggregate::MinInteger { min_value: v }
            | CompiledAggregate::MaxInteger { max_value: v } => v.map(Value::Integer),
            CompiledAggregate::MinFloat { min_value: v }
            | CompiledAggregate::MaxFloat { max_value: v } => v.map(Value::Float),
            _ => None,
        };
        if let Some(value) = extreme {
            self.accumulate(&value);
            return;
        }
        match (self, other) {
            (
                CompiledAggregate::CountStar { count },
                CompiledAggregate::CountStar { count: more },
            )
            | (CompiledAggregate::Count { count }, CompiledAggregate::Count { count: more }) => {
                *count += more
            }
            (CompiledAggregate::Sum { state }, CompiledAggregate::Sum { state: more })
            | (CompiledAggregate::Avg { state }, CompiledAggregate::Avg { state: more }) => {
                state.merge(more)
            }
            _ => {}
        }
    }

    /// Get the result of the aggregation
    #[inline]
    pub fn result(&self) -> Result<Value> {
        Ok(match self {
            CompiledAggregate::CountStar { count } => Value::Integer(*count),
            CompiledAggregate::Count { count } => Value::Integer(*count),
            CompiledAggregate::CountDistinct { distinct_tracker } => {
                Value::Integer(distinct_tracker.count() as i64)
            }

            CompiledAggregate::Sum { state } | CompiledAggregate::SumDistinct { state, .. } => {
                return state.sum();
            }

            CompiledAggregate::Avg { state } | CompiledAggregate::AvgDistinct { state, .. } => {
                state.avg()
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

            CompiledAggregate::Dynamic(state) => return state.result(),
        })
    }

    /// Reset the aggregate state
    pub fn reset(&mut self) {
        match self {
            CompiledAggregate::CountStar { count } => *count = 0,
            CompiledAggregate::Count { count } => *count = 0,
            CompiledAggregate::CountDistinct { distinct_tracker } => distinct_tracker.reset(),

            CompiledAggregate::Sum { state } | CompiledAggregate::Avg { state } => {
                *state = SumAccumulator::default()
            }
            CompiledAggregate::SumDistinct {
                state,
                distinct_tracker,
            }
            | CompiledAggregate::AvgDistinct {
                state,
                distinct_tracker,
            } => {
                *state = SumAccumulator::default();
                distinct_tracker.reset();
            }

            CompiledAggregate::Min { min_value } => *min_value = None,
            CompiledAggregate::Max { max_value } => *max_value = None,

            CompiledAggregate::MinInteger { min_value } => *min_value = None,
            CompiledAggregate::MaxInteger { max_value } => *max_value = None,

            CompiledAggregate::MinFloat { min_value } => *min_value = None,
            CompiledAggregate::MaxFloat { max_value } => *max_value = None,

            CompiledAggregate::Dynamic(state) => state.reset(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_takes_one_cache_line() {
        assert_eq!(std::mem::size_of::<CompiledAggregate>(), 64);
    }

    #[test]
    fn float_min_and_max_order_nan_last() {
        let is_nan = |v: Value| matches!(v, Value::Float(f) if f.is_nan());
        for order in [[f64::NAN, -2.0, 1.5], [-2.0, 1.5, f64::NAN]] {
            let mut min = CompiledAggregate::min_float();
            let mut max = CompiledAggregate::max_float();
            for v in order {
                min.accumulate(&Value::Float(v));
                max.accumulate(&Value::Float(v));
            }
            assert_eq!(min.result().unwrap(), Value::Float(-2.0), "{order:?}");
            assert!(is_nan(max.result().unwrap()), "{order:?}");
        }
        let mut min = CompiledAggregate::min_float();
        min.accumulate(&Value::Float(f64::NAN));
        assert!(is_nan(min.result().unwrap()));
    }

    #[test]
    fn test_count_star() {
        let mut agg = CompiledAggregate::count_star();
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::null_unknown());
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::Integer(3)); // Counts all rows including NULL
    }

    #[test]
    fn test_count() {
        let mut agg = CompiledAggregate::count(false);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::null_unknown());
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::Integer(2)); // Skips NULL
    }

    #[test]
    fn test_count_distinct() {
        let mut agg = CompiledAggregate::count(true);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(1)); // duplicate
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::null_unknown()); // NULL ignored
        assert_eq!(agg.result().unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_sum_integers() {
        let mut agg = CompiledAggregate::sum(false);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::Integer(6));
    }

    #[test]
    fn test_sum_floats() {
        let mut agg = CompiledAggregate::sum(false);
        agg.accumulate(&Value::Float(1.5));
        agg.accumulate(&Value::Float(2.5));
        assert_eq!(agg.result().unwrap(), Value::Float(4.0));
    }

    #[test]
    fn test_sum_mixed() {
        let mut agg = CompiledAggregate::sum(false);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Float(2.5));
        assert_eq!(agg.result().unwrap(), Value::Float(3.5));
    }

    #[test]
    fn test_sum_distinct() {
        let mut agg = CompiledAggregate::sum(true);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(1)); // duplicate
        agg.accumulate(&Value::Integer(2));
        assert_eq!(agg.result().unwrap(), Value::Integer(3)); // 1 + 2
    }

    #[test]
    fn test_avg() {
        let mut agg = CompiledAggregate::avg(false);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::Float(2.0));
    }

    #[test]
    fn test_avg_distinct() {
        let mut agg = CompiledAggregate::avg(true);
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(1)); // duplicate
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::Float(2.0)); // (1 + 3) / 2
    }

    #[test]
    fn test_min_integers() {
        let mut agg = CompiledAggregate::min();
        agg.accumulate(&Value::Integer(5));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(8));
        assert_eq!(agg.result().unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_max_integers() {
        let mut agg = CompiledAggregate::max();
        agg.accumulate(&Value::Integer(5));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(8));
        assert_eq!(agg.result().unwrap(), Value::Integer(8));
    }

    #[test]
    fn test_min_integer_fast() {
        let mut agg = CompiledAggregate::min_integer();
        agg.accumulate(&Value::Integer(5));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(8));
        assert_eq!(agg.result().unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_max_integer_fast() {
        let mut agg = CompiledAggregate::max_integer();
        agg.accumulate(&Value::Integer(5));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(8));
        assert_eq!(agg.result().unwrap(), Value::Integer(8));
    }

    #[test]
    fn test_min_strings() {
        let mut agg = CompiledAggregate::min();
        agg.accumulate(&Value::text("banana"));
        agg.accumulate(&Value::text("apple"));
        agg.accumulate(&Value::text("cherry"));
        assert_eq!(agg.result().unwrap(), Value::text("apple"));
    }

    #[test]
    fn test_max_strings() {
        let mut agg = CompiledAggregate::max();
        agg.accumulate(&Value::text("banana"));
        agg.accumulate(&Value::text("apple"));
        agg.accumulate(&Value::text("cherry"));
        assert_eq!(agg.result().unwrap(), Value::text("cherry"));
    }

    #[test]
    fn test_empty_aggregates() {
        assert!(CompiledAggregate::sum(false).result().unwrap().is_null());
        assert!(CompiledAggregate::avg(false).result().unwrap().is_null());
        assert!(CompiledAggregate::min().result().unwrap().is_null());
        assert!(CompiledAggregate::max().result().unwrap().is_null());
        assert_eq!(
            CompiledAggregate::count(false).result().unwrap(),
            Value::Integer(0)
        );
        assert_eq!(
            CompiledAggregate::count_star().result().unwrap(),
            Value::Integer(0)
        );
    }

    #[test]
    fn test_reset() {
        let mut agg = CompiledAggregate::sum(false);
        agg.accumulate(&Value::Integer(10));
        agg.reset();
        assert!(agg.result().unwrap().is_null());

        let mut agg = CompiledAggregate::count(false);
        agg.accumulate(&Value::Integer(10));
        agg.reset();
        assert_eq!(agg.result().unwrap(), Value::Integer(0));
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
