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

use crate::core::{Result, SumAccumulator, Value};
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// AVG aggregate function
///
/// Returns the average of all non-NULL values in the specified column,
/// as FLOAT, from their exact sum.
#[derive(Default)]
pub struct AvgFunction {
    state: SumAccumulator,
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

    fn accumulate(&mut self, value: &Value) {
        // A boolean counts as one or nought, so AVG(v > 100) is the share of
        // rows the condition holds for
        self.state.add(value);
    }

    fn result(&self) -> Result<Value> {
        Ok(self.state.avg())
    }

    fn reset(&mut self) {
        self.state = SumAccumulator::default();
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
        avg.accumulate(&Value::Integer(1));
        avg.accumulate(&Value::Integer(2));
        avg.accumulate(&Value::Integer(3));
        assert_eq!(avg.result().unwrap(), Value::Float(2.0));
    }

    #[test]
    fn test_avg_floats() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Float(1.0));
        avg.accumulate(&Value::Float(2.0));
        avg.accumulate(&Value::Float(3.0));
        assert_eq!(avg.result().unwrap(), Value::Float(2.0));
    }

    #[test]
    fn test_avg_mixed() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(1));
        avg.accumulate(&Value::Float(2.0));
        avg.accumulate(&Value::Integer(3));
        assert_eq!(avg.result().unwrap(), Value::Float(2.0));
    }

    #[test]
    fn test_avg_ignores_null() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(1));
        avg.accumulate(&Value::null_unknown());
        avg.accumulate(&Value::Integer(3));
        assert_eq!(avg.result().unwrap(), Value::Float(2.0)); // (1 + 3) / 2
    }

    #[test]
    fn test_avg_distinct() {
        let mut avg =
            super::super::AggregateState::new(Box::new(AvgFunction::default()), true, &[], vec![]);
        avg.accumulate(&Value::Integer(1));
        avg.accumulate(&Value::Integer(1)); // duplicate
        avg.accumulate(&Value::Integer(3));
        avg.accumulate(&Value::Integer(3)); // duplicate
        assert_eq!(avg.result().unwrap(), Value::Float(2.0)); // (1 + 3) / 2
    }

    #[test]
    fn test_avg_empty() {
        let avg = AvgFunction::default();
        assert!(avg.result().unwrap().is_null());
    }

    #[test]
    fn test_avg_reset() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(1));
        avg.accumulate(&Value::Integer(2));
        avg.reset();
        assert!(avg.result().unwrap().is_null());
    }

    #[test]
    fn test_avg_single_value() {
        let mut avg = AvgFunction::default();
        avg.accumulate(&Value::Integer(42));
        assert_eq!(avg.result().unwrap(), Value::Float(42.0));
    }
}
