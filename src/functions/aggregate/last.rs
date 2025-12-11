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

//! LAST aggregate function

use crate::core::Value;
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// LAST aggregate function
///
/// Returns the last non-NULL value in the specified column.
/// The order depends on the query's ORDER BY clause.
#[derive(Default)]
pub struct LastFunction {
    last_value: Option<Value>,
}

impl AggregateFunction for LastFunction {
    fn name(&self) -> &str {
        "LAST"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LAST",
            FunctionType::Aggregate,
            "Returns the last non-NULL value in the specified column",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, _distinct: bool) {
        // Handle NULL values - LAST ignores NULLs
        if value.is_null() {
            return;
        }

        // Always update to the latest non-NULL value
        self.last_value = Some(value.clone());
    }

    fn result(&self) -> Value {
        self.last_value.clone().unwrap_or_else(Value::null_unknown)
    }

    fn reset(&mut self) {
        self.last_value = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(LastFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_basic() {
        let mut last = LastFunction::default();
        last.accumulate(&Value::Integer(1), false);
        last.accumulate(&Value::Integer(2), false);
        last.accumulate(&Value::Integer(3), false);
        assert_eq!(last.result(), Value::Integer(3));
    }

    #[test]
    fn test_last_ignores_null() {
        let mut last = LastFunction::default();
        last.accumulate(&Value::Integer(1), false);
        last.accumulate(&Value::Integer(2), false);
        last.accumulate(&Value::null_unknown(), false);
        last.accumulate(&Value::null_unknown(), false);
        assert_eq!(last.result(), Value::Integer(2));
    }

    #[test]
    fn test_last_empty() {
        let last = LastFunction::default();
        assert!(last.result().is_null());
    }

    #[test]
    fn test_last_all_null() {
        let mut last = LastFunction::default();
        last.accumulate(&Value::null_unknown(), false);
        last.accumulate(&Value::null_unknown(), false);
        assert!(last.result().is_null());
    }

    #[test]
    fn test_last_reset() {
        let mut last = LastFunction::default();
        last.accumulate(&Value::Integer(1), false);
        last.accumulate(&Value::Integer(2), false);
        last.reset();
        last.accumulate(&Value::Integer(5), false);
        assert_eq!(last.result(), Value::Integer(5));
    }

    #[test]
    fn test_last_string() {
        let mut last = LastFunction::default();
        last.accumulate(&Value::text("hello"), false);
        last.accumulate(&Value::text("world"), false);
        assert_eq!(last.result(), Value::text("world"));
    }
}
