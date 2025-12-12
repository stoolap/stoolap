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

use crate::core::Value;
use std::collections::HashSet;

/// Helper to convert Value to a hashable key for distinct tracking
fn value_to_distinct_key(value: &Value) -> Option<String> {
    if value.is_null() {
        return None; // NULL values don't contribute to distinct
    }

    match value {
        Value::Null(_) => None,
        Value::Integer(i) => Some(format!("i:{}", i)),
        Value::Float(f) => Some(format!("f:{}", f)),
        Value::Text(s) => Some(format!("s:{}", s)),
        Value::Boolean(b) => Some(format!("b:{}", b)),
        Value::Timestamp(t) => Some(format!("t:{}", t)),
        Value::Json(j) => Some(format!("j:{}", j)),
    }
}

/// Helper struct for tracking distinct values
#[derive(Default, Debug)]
pub struct DistinctTracker {
    seen: HashSet<String>,
}

impl DistinctTracker {
    /// Check if a value has been seen before (returns true if new)
    pub fn check_and_add(&mut self, value: &Value) -> bool {
        if let Some(key) = value_to_distinct_key(value) {
            self.seen.insert(key)
        } else {
            false // NULL is never considered for distinct
        }
    }

    /// Get the count of distinct values
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

        assert!(!tracker.check_and_add(&Value::null_unknown())); // NULL returns false
        assert!(!tracker.check_and_add(&Value::null_unknown())); // NULL again
        assert_eq!(tracker.count(), 0); // NULLs not counted
    }
}
