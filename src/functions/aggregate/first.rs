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

//! FIRST aggregate function

use crate::core::Value;
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// FIRST aggregate function
///
/// Returns the first non-NULL value in the specified column.
/// The order depends on the query's ORDER BY clause.
#[derive(Default)]
pub struct FirstFunction {
    first_value: Option<Value>,
}

impl AggregateFunction for FirstFunction {
    fn name(&self) -> &str {
        "FIRST"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "FIRST",
            FunctionType::Aggregate,
            "Returns the first non-NULL value in the specified column",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, _distinct: bool) {
        // Handle NULL values - FIRST ignores NULLs
        if value.is_null() {
            return;
        }

        // Only take the first non-NULL value
        if self.first_value.is_none() {
            self.first_value = Some(value.clone());
        }
    }

    fn result(&self) -> Value {
        self.first_value.clone().unwrap_or_else(Value::null_unknown)
    }

    fn reset(&mut self) {
        self.first_value = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(FirstFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_basic() {
        let mut first = FirstFunction::default();
        first.accumulate(&Value::Integer(1), false);
        first.accumulate(&Value::Integer(2), false);
        first.accumulate(&Value::Integer(3), false);
        assert_eq!(first.result(), Value::Integer(1));
    }

    #[test]
    fn test_first_ignores_null() {
        let mut first = FirstFunction::default();
        first.accumulate(&Value::null_unknown(), false);
        first.accumulate(&Value::null_unknown(), false);
        first.accumulate(&Value::Integer(3), false);
        first.accumulate(&Value::Integer(4), false);
        assert_eq!(first.result(), Value::Integer(3));
    }

    #[test]
    fn test_first_empty() {
        let first = FirstFunction::default();
        assert!(first.result().is_null());
    }

    #[test]
    fn test_first_all_null() {
        let mut first = FirstFunction::default();
        first.accumulate(&Value::null_unknown(), false);
        first.accumulate(&Value::null_unknown(), false);
        assert!(first.result().is_null());
    }

    #[test]
    fn test_first_reset() {
        let mut first = FirstFunction::default();
        first.accumulate(&Value::Integer(1), false);
        first.reset();
        first.accumulate(&Value::Integer(5), false);
        assert_eq!(first.result(), Value::Integer(5));
    }

    #[test]
    fn test_first_string() {
        let mut first = FirstFunction::default();
        first.accumulate(&Value::text("hello"), false);
        first.accumulate(&Value::text("world"), false);
        assert_eq!(first.result(), Value::text("hello"));
    }
}
