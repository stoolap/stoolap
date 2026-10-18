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

//! MIN aggregate function

use crate::core::{Result, Value};
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// MIN aggregate function
///
/// Returns the minimum value of all non-NULL values in the specified column.
/// Works with any comparable type (numbers, strings, timestamps, etc.)
#[derive(Default)]
pub struct MinFunction {
    min_value: Option<Value>,
}

impl AggregateFunction for MinFunction {
    fn name(&self) -> &str {
        "MIN"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "MIN",
            FunctionType::Aggregate,
            "Returns the minimum value of all non-NULL values in the specified column",
            FunctionSignature::new(
                FunctionDataType::Any, // MIN returns the same type as input
                vec![FunctionDataType::Any],
                1,
                1,
            ),
        )
    }

    fn accumulate(&mut self, value: &Value) {
        // Handle NULL values - MIN ignores NULLs
        if value.is_null() {
            return;
        }

        // If this is the first non-NULL value, set it as minimum
        if self.min_value.is_none() {
            self.min_value = Some(value.clone());
            return;
        }

        // Compare and update minimum
        if self
            .min_value
            .as_ref()
            .is_some_and(|min| value.cmp(min).is_lt())
        {
            self.min_value = Some(value.clone());
        }
    }

    fn result(&self) -> Result<Value> {
        Ok(self.min_value.clone().unwrap_or_else(Value::null_unknown))
    }

    fn reset(&mut self) {
        self.min_value = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(MinFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_integers() {
        let mut min = MinFunction::default();
        min.accumulate(&Value::Integer(5));
        min.accumulate(&Value::Integer(2));
        min.accumulate(&Value::Integer(8));
        assert_eq!(min.result().unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_min_floats() {
        let mut min = MinFunction::default();
        min.accumulate(&Value::Float(5.5));
        min.accumulate(&Value::Float(2.2));
        min.accumulate(&Value::Float(8.8));
        assert_eq!(min.result().unwrap(), Value::Float(2.2));
    }

    #[test]
    fn test_min_strings() {
        let mut min = MinFunction::default();
        min.accumulate(&Value::text("banana"));
        min.accumulate(&Value::text("apple"));
        min.accumulate(&Value::text("cherry"));
        assert_eq!(min.result().unwrap(), Value::text("apple"));
    }

    #[test]
    fn test_min_ignores_null() {
        let mut min = MinFunction::default();
        min.accumulate(&Value::Integer(5));
        min.accumulate(&Value::null_unknown());
        min.accumulate(&Value::Integer(2));
        assert_eq!(min.result().unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_min_empty() {
        let min = MinFunction::default();
        assert!(min.result().unwrap().is_null());
    }

    #[test]
    fn test_min_reset() {
        let mut min = MinFunction::default();
        min.accumulate(&Value::Integer(5));
        min.accumulate(&Value::Integer(2));
        min.reset();
        assert!(min.result().unwrap().is_null());
    }

    #[test]
    fn test_min_single_value() {
        let mut min = MinFunction::default();
        min.accumulate(&Value::Integer(42));
        assert_eq!(min.result().unwrap(), Value::Integer(42));
    }

    #[test]
    fn test_min_negative() {
        let mut min = MinFunction::default();
        min.accumulate(&Value::Integer(-5));
        min.accumulate(&Value::Integer(10));
        min.accumulate(&Value::Integer(-10));
        assert_eq!(min.result().unwrap(), Value::Integer(-10));
    }

    #[test]
    fn test_min_booleans() {
        let mut min = MinFunction::default();
        min.accumulate(&Value::Boolean(true));
        min.accumulate(&Value::Boolean(false));
        min.accumulate(&Value::Boolean(true));
        assert_eq!(min.result().unwrap(), Value::Boolean(false));
    }
}
