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

//! COUNT aggregate function

use crate::core::{Result, Value};
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// COUNT aggregate function
///
/// Returns the number of rows matching the query criteria.
/// - COUNT(*) counts all rows including NULLs
/// - COUNT(column) counts non-NULL values
/// - COUNT(DISTINCT column) counts distinct non-NULL values
#[derive(Default)]
pub struct CountFunction {
    count: i64,
}

impl AggregateFunction for CountFunction {
    fn name(&self) -> &str {
        "COUNT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "COUNT",
            FunctionType::Aggregate,
            "Returns the number of rows matching the query criteria",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Any],
                0, // COUNT(*) has no actual argument
                1, // But can be COUNT(column)
            ),
        )
    }

    fn accumulate(&mut self, value: &Value) {
        // COUNT ignores NULLs; COUNT(*) passes a marker for every row
        if !value.is_null() {
            self.count += 1;
        }
    }

    fn result(&self) -> Result<Value> {
        Ok(Value::Integer(self.count))
    }

    fn reset(&mut self) {
        self.count = 0;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(CountFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_basic() {
        let mut count = CountFunction::default();
        count.accumulate(&Value::Integer(1));
        count.accumulate(&Value::Integer(2));
        count.accumulate(&Value::Integer(3));
        assert_eq!(count.result().unwrap(), Value::Integer(3));
    }

    #[test]
    fn test_count_star() {
        let mut count = CountFunction::default();
        count.accumulate(&Value::text("*"));
        count.accumulate(&Value::text("*"));
        count.accumulate(&Value::text("*"));
        assert_eq!(count.result().unwrap(), Value::Integer(3));
    }

    #[test]
    fn test_count_ignores_null() {
        let mut count = CountFunction::default();
        count.accumulate(&Value::Integer(1));
        count.accumulate(&Value::null_unknown());
        count.accumulate(&Value::Integer(3));
        assert_eq!(count.result().unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_count_distinct() {
        let mut count = super::super::AggregateState::new(
            Box::new(CountFunction::default()),
            true,
            &[],
            vec![],
        );
        count.accumulate(&Value::Integer(1));
        count.accumulate(&Value::Integer(1)); // duplicate
        count.accumulate(&Value::Integer(2));
        count.accumulate(&Value::Integer(2)); // duplicate
        count.accumulate(&Value::Integer(3));
        assert_eq!(count.result().unwrap(), Value::Integer(3));
    }

    #[test]
    fn test_count_reset() {
        let mut count = CountFunction::default();
        count.accumulate(&Value::Integer(1));
        count.accumulate(&Value::Integer(2));
        count.reset();
        assert_eq!(count.result().unwrap(), Value::Integer(0));
    }

    #[test]
    fn test_count_empty() {
        let count = CountFunction::default();
        assert_eq!(count.result().unwrap(), Value::Integer(0));
    }
}
