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

use super::compare_sort_keys;

struct BestEntry {
    value: Value,
    sort_keys: Vec<Value>,
}

/// FIRST aggregate function
///
/// Returns the first non-NULL value in the specified column.
/// When used with ORDER BY, returns the first value according to the specified ordering.
/// Tracks only the current best candidate during accumulation (O(1) memory).
#[derive(Default)]
pub struct FirstFunction {
    first_value: Option<Value>,
    has_order_by: bool,
    order_directions: Vec<bool>,
    best_entry: Option<BestEntry>,
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

    fn set_order_by(&mut self, directions: Vec<bool>) {
        self.order_directions = directions;
        self.has_order_by = true;
    }

    fn supports_order_by(&self) -> bool {
        true
    }

    fn accumulate(&mut self, value: &Value, _distinct: bool) {
        if value.is_null() {
            return;
        }

        // Without ORDER BY, just take the first non-NULL value
        if self.first_value.is_none() {
            self.first_value = Some(value.clone());
        }
    }

    fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>, _distinct: bool) {
        if value.is_null() {
            return;
        }

        // FIRST: keep the entry that would sort first (minimum in sort order).
        // For equal keys, keep the first seen (don't replace).
        let dominated = self.best_entry.as_ref().is_some_and(|best| {
            compare_sort_keys(&sort_keys, &best.sort_keys, &self.order_directions)
                != std::cmp::Ordering::Less
        });
        if !dominated {
            self.best_entry = Some(BestEntry {
                value: value.clone(),
                sort_keys,
            });
        }
    }

    fn result(&self) -> Value {
        if self.has_order_by {
            if let Some(ref entry) = self.best_entry {
                return entry.value.clone();
            }
        }

        self.first_value.clone().unwrap_or_else(Value::null_unknown)
    }

    fn reset(&mut self) {
        self.first_value = None;
        self.best_entry = None;
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

    #[test]
    fn test_first_with_order_by_asc() {
        let mut first = FirstFunction::default();
        first.set_order_by(vec![true]); // ASC
        first.accumulate_with_sort_key(&Value::text("c_val"), vec![Value::Integer(3)], false);
        first.accumulate_with_sort_key(&Value::text("a_val"), vec![Value::Integer(1)], false);
        first.accumulate_with_sort_key(&Value::text("b_val"), vec![Value::Integer(2)], false);
        // ASC order: key 1 is first, so value should be "a_val"
        assert_eq!(first.result(), Value::text("a_val"));
    }

    #[test]
    fn test_first_with_order_by_desc() {
        let mut first = FirstFunction::default();
        first.set_order_by(vec![false]); // DESC
        first.accumulate_with_sort_key(&Value::text("c_val"), vec![Value::Integer(3)], false);
        first.accumulate_with_sort_key(&Value::text("a_val"), vec![Value::Integer(1)], false);
        first.accumulate_with_sort_key(&Value::text("b_val"), vec![Value::Integer(2)], false);
        // DESC order: key 3 is first, so value should be "c_val"
        assert_eq!(first.result(), Value::text("c_val"));
    }
}
