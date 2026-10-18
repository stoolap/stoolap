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

//! MAX aggregate function

use crate::core::{Result, Value};
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// MAX aggregate function
///
/// Returns the maximum value of all non-NULL values in the specified column.
/// Works with any comparable type (numbers, strings, timestamps, etc.)
#[derive(Default)]
pub struct MaxFunction {
    max_value: Option<Value>,
}

impl AggregateFunction for MaxFunction {
    fn name(&self) -> &str {
        "MAX"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "MAX",
            FunctionType::Aggregate,
            "Returns the maximum value of all non-NULL values in the specified column",
            FunctionSignature::new(
                FunctionDataType::Any, // MAX returns the same type as input
                vec![FunctionDataType::Any],
                1,
                1,
            ),
        )
    }

    fn accumulate(&mut self, value: &Value) {
        // Handle NULL values - MAX ignores NULLs
        if value.is_null() {
            return;
        }

        // If this is the first non-NULL value, set it as maximum
        if self.max_value.is_none() {
            self.max_value = Some(value.clone());
            return;
        }

        // Compare and update maximum
        if self
            .max_value
            .as_ref()
            .is_some_and(|max| value.cmp(max).is_gt())
        {
            self.max_value = Some(value.clone());
        }
    }

    fn result(&self) -> Result<Value> {
        Ok(self.max_value.clone().unwrap_or_else(Value::null_unknown))
    }

    fn reset(&mut self) {
        self.max_value = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(MaxFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_integers() {
        let mut max = MaxFunction::default();
        max.accumulate(&Value::Integer(5));
        max.accumulate(&Value::Integer(2));
        max.accumulate(&Value::Integer(8));
        assert_eq!(max.result().unwrap(), Value::Integer(8));
    }

    #[test]
    fn test_max_floats() {
        let mut max = MaxFunction::default();
        max.accumulate(&Value::Float(5.5));
        max.accumulate(&Value::Float(2.2));
        max.accumulate(&Value::Float(8.8));
        assert_eq!(max.result().unwrap(), Value::Float(8.8));
    }

    #[test]
    fn test_max_strings() {
        let mut max = MaxFunction::default();
        max.accumulate(&Value::text("banana"));
        max.accumulate(&Value::text("apple"));
        max.accumulate(&Value::text("cherry"));
        assert_eq!(max.result().unwrap(), Value::text("cherry"));
    }

    #[test]
    fn test_max_ignores_null() {
        let mut max = MaxFunction::default();
        max.accumulate(&Value::Integer(5));
        max.accumulate(&Value::null_unknown());
        max.accumulate(&Value::Integer(2));
        assert_eq!(max.result().unwrap(), Value::Integer(5));
    }

    #[test]
    fn test_max_empty() {
        let max = MaxFunction::default();
        assert!(max.result().unwrap().is_null());
    }

    #[test]
    fn test_max_reset() {
        let mut max = MaxFunction::default();
        max.accumulate(&Value::Integer(5));
        max.accumulate(&Value::Integer(8));
        max.reset();
        assert!(max.result().unwrap().is_null());
    }

    #[test]
    fn test_max_single_value() {
        let mut max = MaxFunction::default();
        max.accumulate(&Value::Integer(42));
        assert_eq!(max.result().unwrap(), Value::Integer(42));
    }

    #[test]
    fn test_max_negative() {
        let mut max = MaxFunction::default();
        max.accumulate(&Value::Integer(-5));
        max.accumulate(&Value::Integer(-10));
        max.accumulate(&Value::Integer(-3));
        assert_eq!(max.result().unwrap(), Value::Integer(-3));
    }

    #[test]
    fn test_max_booleans() {
        let mut max = MaxFunction::default();
        max.accumulate(&Value::Boolean(false));
        max.accumulate(&Value::Boolean(true));
        max.accumulate(&Value::Boolean(false));
        assert_eq!(max.result().unwrap(), Value::Boolean(true));
    }
}
