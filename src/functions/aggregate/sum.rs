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

//! SUM aggregate function

use crate::core::{Result, SumAccumulator, Value};
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// SUM aggregate function
///
/// Returns the sum of all non-NULL values in the specified column: the
/// exact INTEGER for INTEGER inputs, FLOAT once any input is FLOAT.
#[derive(Default)]
pub struct SumFunction {
    state: SumAccumulator,
}

impl AggregateFunction for SumFunction {
    fn name(&self) -> &str {
        "SUM"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SUM",
            FunctionType::Aggregate,
            "Returns the sum of all non-NULL values in the specified column",
            FunctionSignature::new(
                FunctionDataType::Any, // can return either int64 or float64
                vec![FunctionDataType::Any],
                1,
                1,
            ),
        )
    }

    fn accumulate(&mut self, value: &Value) {
        // A boolean counts as one or nought, so SUM(v > 100) counts the
        // rows the condition holds for
        self.state.add(value);
    }

    fn result(&self) -> Result<Value> {
        self.state.sum()
    }

    fn reset(&mut self) {
        self.state = SumAccumulator::default();
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(SumFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_integers() {
        let mut sum = SumFunction::default();
        sum.accumulate(&Value::Integer(1));
        sum.accumulate(&Value::Integer(2));
        sum.accumulate(&Value::Integer(3));
        assert_eq!(sum.result().unwrap(), Value::Integer(6));
    }

    #[test]
    fn test_sum_floats() {
        let mut sum = SumFunction::default();
        sum.accumulate(&Value::Float(1.5));
        sum.accumulate(&Value::Float(2.5));
        sum.accumulate(&Value::Float(3.0));
        assert_eq!(sum.result().unwrap(), Value::Float(7.0));
    }

    #[test]
    fn test_sum_mixed() {
        let mut sum = SumFunction::default();
        sum.accumulate(&Value::Integer(1));
        sum.accumulate(&Value::Float(2.5));
        sum.accumulate(&Value::Integer(3));
        assert_eq!(sum.result().unwrap(), Value::Float(6.5));
    }

    #[test]
    fn test_sum_ignores_null() {
        let mut sum = SumFunction::default();
        sum.accumulate(&Value::Integer(1));
        sum.accumulate(&Value::null_unknown());
        sum.accumulate(&Value::Integer(3));
        assert_eq!(sum.result().unwrap(), Value::Integer(4));
    }

    #[test]
    fn test_sum_distinct() {
        let mut sum =
            super::super::AggregateState::new(Box::new(SumFunction::default()), true, &[], vec![]);
        sum.accumulate(&Value::Integer(1));
        sum.accumulate(&Value::Integer(1)); // duplicate
        sum.accumulate(&Value::Integer(2));
        sum.accumulate(&Value::Integer(2)); // duplicate
        assert_eq!(sum.result().unwrap(), Value::Integer(3)); // 1 + 2
    }

    #[test]
    fn test_sum_empty() {
        let sum = SumFunction::default();
        assert!(sum.result().unwrap().is_null());
    }

    #[test]
    fn test_sum_reset() {
        let mut sum = SumFunction::default();
        sum.accumulate(&Value::Integer(1));
        sum.accumulate(&Value::Integer(2));
        sum.reset();
        assert!(sum.result().unwrap().is_null());
    }

    #[test]
    fn test_sum_negative() {
        let mut sum = SumFunction::default();
        sum.accumulate(&Value::Integer(-5));
        sum.accumulate(&Value::Integer(10));
        sum.accumulate(&Value::Integer(-3));
        assert_eq!(sum.result().unwrap(), Value::Integer(2));
    }
}
