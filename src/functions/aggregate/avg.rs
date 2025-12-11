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

//! AVG aggregate function

use crate::core::Value;
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

use super::DistinctTracker;

/// AVG aggregate function
///
/// Returns the average of all non-NULL values in the specified column.
/// Always returns a float64.
#[derive(Default)]
pub struct AvgFunction {
    sum: f64,
    count: i64,
    distinct_tracker: Option<DistinctTracker>,
}

impl AggregateFunction for AvgFunction {
    fn name(&self) -> &str {
        "AVG"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "AVG",
            FunctionType::Aggregate,
            "Returns the average of all non-NULL values in the specified column",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        // Handle NULL values - AVG ignores NULLs
        if value.is_null() {
            return;
        }

        // Handle DISTINCT case
        if distinct {
            if self.distinct_tracker.is_none() {
                self.distinct_tracker = Some(DistinctTracker::default());
            }
            if !self.distinct_tracker.as_mut().unwrap().check_and_add(value) {
                return; // Already seen this value
            }
        }

        // Extract numeric value
        let numeric_value = match value {
            Value::Integer(i) => *i as f64,
            Value::Float(f) => *f,
            _ => return, // Ignore non-numeric types
        };

        self.sum += numeric_value;
        self.count += 1;
    }

    fn result(&self) -> Value {
        if self.count == 0 {
            Value::null_unknown() // Return NULL for empty sets
        } else {
            Value::Float(self.sum / self.count as f64)
        }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
        self.distinct_tracker = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(AvgFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avg_integers() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(1), false);
        avg.accumulate(&Value::Integer(2), false);
        avg.accumulate(&Value::Integer(3), false);
        assert_eq!(avg.result(), Value::Float(2.0));
    }

    #[test]
    fn test_avg_floats() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Float(1.0), false);
        avg.accumulate(&Value::Float(2.0), false);
        avg.accumulate(&Value::Float(3.0), false);
        assert_eq!(avg.result(), Value::Float(2.0));
    }

    #[test]
    fn test_avg_mixed() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(1), false);
        avg.accumulate(&Value::Float(2.0), false);
        avg.accumulate(&Value::Integer(3), false);
        assert_eq!(avg.result(), Value::Float(2.0));
    }

    #[test]
    fn test_avg_ignores_null() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(1), false);
        avg.accumulate(&Value::null_unknown(), false);
        avg.accumulate(&Value::Integer(3), false);
        assert_eq!(avg.result(), Value::Float(2.0)); // (1 + 3) / 2
    }

    #[test]
    fn test_avg_distinct() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(1), true);
        avg.accumulate(&Value::Integer(1), true); // duplicate
        avg.accumulate(&Value::Integer(3), true);
        avg.accumulate(&Value::Integer(3), true); // duplicate
        assert_eq!(avg.result(), Value::Float(2.0)); // (1 + 3) / 2
    }

    #[test]
    fn test_avg_empty() {
        let avg = AvgFunction::default();
        assert!(avg.result().is_null());
    }

    #[test]
    fn test_avg_reset() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(1), false);
        avg.accumulate(&Value::Integer(2), false);
        avg.reset();
        assert!(avg.result().is_null());
    }

    #[test]
    fn test_avg_single_value() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(42), false);
        assert_eq!(avg.result(), Value::Float(42.0));
    }
}
