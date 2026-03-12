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

//! Aggregate Functions
//!
//! This module provides aggregate functions for SQL queries:
//!
//! - [`CountFunction`] - COUNT(*) and COUNT(column)
//! - [`SumFunction`] - SUM(column)
//! - [`AvgFunction`] - AVG(column)
//! - [`MinFunction`] - MIN(column)
//! - [`MaxFunction`] - MAX(column)
//! - [`FirstFunction`] - FIRST(column)
//! - [`LastFunction`] - LAST(column)
//! - [`StringAggFunction`] - STRING_AGG(column, separator)
//! - [`GroupConcatFunction`] - GROUP_CONCAT(column, separator)
//! - [`ArrayAggFunction`] - ARRAY_AGG(column)
//! - [`StddevPopFunction`] - STDDEV_POP(column)
//! - [`StddevFunction`] - STDDEV(column)
//! - [`StddevSampFunction`] - STDDEV_SAMP(column)
//! - [`VarPopFunction`] - VAR_POP(column)
//! - [`VarianceFunction`] - VARIANCE(column)
//! - [`VarSampFunction`] - VAR_SAMP(column)
//! - [`MedianFunction`] - MEDIAN(column)

mod array_agg;
mod avg;
pub mod compiled;
mod count;
mod first;
mod last;
mod max;
mod min;
mod statistics;
mod string_agg;
mod sum;

pub use array_agg::ArrayAggFunction;
pub use avg::AvgFunction;
pub use compiled::CompiledAggregate;
pub use count::CountFunction;
pub use first::FirstFunction;
pub use last::LastFunction;
pub use max::MaxFunction;
pub use min::MinFunction;
pub use statistics::{
    MedianFunction, StddevFunction, StddevPopFunction, StddevSampFunction, VarPopFunction,
    VarSampFunction, VarianceFunction,
};
pub use string_agg::{GroupConcatFunction, StringAggFunction};
pub use sum::SumFunction;

use crate::core::{Value, ValueSet};
use std::cmp::Ordering;

/// Compare two Values for ordering (used by FIRST/LAST with ORDER BY).
/// NULLs sort last (greater than all non-NULL values).
pub(crate) fn compare_sort_keys(a: &[Value], b: &[Value], directions: &[bool]) -> Ordering {
    for (i, (key_a, key_b)) in a.iter().zip(b.iter()).enumerate() {
        let is_asc = directions.get(i).copied().unwrap_or(true);
        let cmp = compare_values_for_sort(key_a, key_b);
        if cmp != Ordering::Equal {
            return if is_asc { cmp } else { cmp.reverse() };
        }
    }
    Ordering::Equal
}

/// Compare two Values for sorting. NULLs sort last.
fn compare_values_for_sort(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Null(_), Value::Null(_)) => Ordering::Equal,
        (Value::Null(_), _) => Ordering::Greater,
        (_, Value::Null(_)) => Ordering::Less,
        (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Integer(a), Value::Float(b)) => {
            (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
        }
        (Value::Float(a), Value::Integer(b)) => {
            a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
        }
        (Value::Text(a), Value::Text(b)) => a.cmp(b),
        (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
        (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// Helper struct for tracking distinct values
///
/// Uses ValueSet directly instead of converting to strings,
/// avoiding allocation overhead for each value.
#[derive(Default, Debug)]
pub struct DistinctTracker {
    seen: ValueSet,
}

impl DistinctTracker {
    /// Check if a value has been seen before (returns true if new)
    /// Note: Caller must ensure value is not NULL before calling
    #[inline]
    pub fn check_and_add(&mut self, value: &Value) -> bool {
        self.seen.insert(value.clone())
    }

    /// Check if a value has been seen before, with null handling (returns true if new)
    #[inline]
    pub fn check_and_add_with_null_check(&mut self, value: &Value) -> bool {
        if value.is_null() {
            return false;
        }
        self.seen.insert(value.clone())
    }

    /// Get the count of distinct values
    #[inline]
    pub fn count(&self) -> usize {
        self.seen.len()
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.seen.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distinct_tracker() {
        let mut tracker = DistinctTracker::default();

        assert!(tracker.check_and_add(&Value::Integer(1)));
        assert!(!tracker.check_and_add(&Value::Integer(1))); // duplicate
        assert!(tracker.check_and_add(&Value::Integer(2)));
        assert!(tracker.check_and_add(&Value::text("hello")));

        assert_eq!(tracker.count(), 3);

        tracker.reset();
        assert_eq!(tracker.count(), 0);
    }

    #[test]
    fn test_null_handling() {
        let mut tracker = DistinctTracker::default();

        // Using the with_null_check variant for null handling
        assert!(!tracker.check_and_add_with_null_check(&Value::null_unknown())); // NULL returns false
        assert!(!tracker.check_and_add_with_null_check(&Value::null_unknown())); // NULL again
        assert_eq!(tracker.count(), 0); // NULLs not counted
    }
}
