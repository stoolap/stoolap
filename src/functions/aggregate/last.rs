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

use super::compare_sort_keys;

struct BestEntry {
    value: Value,
    sort_keys: Vec<Value>,
}

/// LAST aggregate function
///
/// Returns the last non-NULL value in the specified column.
/// When used with ORDER BY, returns the last value according to the specified ordering.
/// Tracks only the current best candidate during accumulation (O(1) memory).
#[derive(Default)]
pub struct LastFunction {
    last_value: Option<Value>,
    has_order_by: bool,
    order_directions: Vec<bool>,
    best_entry: Option<BestEntry>,
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

        // Without ORDER BY, always update to the latest non-NULL value
        self.last_value = Some(value.clone());
    }

    fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>, _distinct: bool) {
        if value.is_null() {
            return;
        }

        // LAST: keep the entry that would sort last (maximum in sort order).
        // For equal keys, replace (last-seen-wins, matching stable sort behavior).
        let should_replace = self.best_entry.as_ref().is_none_or(|best| {
            compare_sort_keys(&sort_keys, &best.sort_keys, &self.order_directions)
                != std::cmp::Ordering::Less
        });
        if should_replace {
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

        self.last_value.clone().unwrap_or_else(Value::null_unknown)
    }

    fn reset(&mut self) {
        self.last_value = None;
        self.best_entry = None;
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

    #[test]
    fn test_last_with_order_by_asc() {
        let mut last = LastFunction::default();
        last.set_order_by(vec![true]); // ASC
        last.accumulate_with_sort_key(&Value::text("c_val"), vec![Value::Integer(3)], false);
        last.accumulate_with_sort_key(&Value::text("a_val"), vec![Value::Integer(1)], false);
        last.accumulate_with_sort_key(&Value::text("b_val"), vec![Value::Integer(2)], false);
        // ASC order: key 3 is last, so value should be "c_val"
        assert_eq!(last.result(), Value::text("c_val"));
    }

    #[test]
    fn test_last_with_order_by_desc() {
        let mut last = LastFunction::default();
        last.set_order_by(vec![false]); // DESC
        last.accumulate_with_sort_key(&Value::text("c_val"), vec![Value::Integer(3)], false);
        last.accumulate_with_sort_key(&Value::text("a_val"), vec![Value::Integer(1)], false);
        last.accumulate_with_sort_key(&Value::text("b_val"), vec![Value::Integer(2)], false);
        // DESC order: key 1 is last, so value should be "a_val"
        assert_eq!(last.result(), Value::text("a_val"));
    }
}
