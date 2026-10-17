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

//! ARRAY_AGG aggregate function

use crate::core::{Result, Value};
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// Entry for ordered aggregation - stores value with its sort keys
#[derive(Clone)]
struct OrderedEntry {
    value: Value,
    sort_keys: Vec<Value>,
}

/// ARRAY_AGG aggregate function
///
/// Collects all values into a JSON array.
/// Similar to PostgreSQL's ARRAY_AGG.
///
/// Usage:
///   ARRAY_AGG(column)
///   ARRAY_AGG(DISTINCT column)
///   ARRAY_AGG(column ORDER BY expr)
#[derive(Default)]
pub struct ArrayAggFunction {
    /// Values collected (used when no ORDER BY)
    values: Vec<Value>,
    /// Values with sort keys (used when ORDER BY is specified)
    ordered_entries: Vec<OrderedEntry>,
    /// Sort directions: true = ASC, false = DESC
    order_keys: Vec<crate::functions::AggregateOrder>,
    /// Whether ORDER BY is active
    has_order_by: bool,
}

impl AggregateFunction for ArrayAggFunction {
    fn name(&self) -> &str {
        "ARRAY_AGG"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ARRAY_AGG",
            FunctionType::Aggregate,
            "Collects all values into a JSON array",
            FunctionSignature::new(FunctionDataType::Json, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn configure(&mut self, _options: &[Value]) {
        // No configuration options
    }

    fn set_order_by(&mut self, keys: Vec<crate::functions::AggregateOrder>) {
        self.order_keys = keys;
        self.has_order_by = true;
    }

    fn supports_order_by(&self) -> bool {
        true
    }

    fn accumulate(&mut self, value: &Value) {
        // Handle NULL values - ARRAY_AGG includes NULLs by default (standard SQL behavior)
        // but we'll skip them like other aggregates for consistency
        if value.is_null() {
            return;
        }

        self.values.push(value.clone());
    }

    fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>) {
        // Handle NULL values
        if value.is_null() {
            return;
        }

        self.ordered_entries.push(OrderedEntry {
            value: value.clone(),
            sort_keys,
        });
    }

    fn result(&self) -> Result<Value> {
        // Get the values to output, potentially sorted
        let values_to_output: Vec<&Value> = if self.has_order_by && !self.ordered_entries.is_empty()
        {
            // Sort the entries by their sort keys
            let mut entries: Vec<&OrderedEntry> = self.ordered_entries.iter().collect();
            let keys = &self.order_keys;

            entries.sort_by(|a, b| super::compare_sort_keys(&a.sort_keys, &b.sort_keys, keys));

            entries.iter().map(|e| &e.value).collect()
        } else if !self.values.is_empty() {
            self.values.iter().collect()
        } else if !self.ordered_entries.is_empty() {
            // Has ordered entries but no ORDER BY configured - use insertion order
            self.ordered_entries.iter().map(|e| &e.value).collect()
        } else {
            return Ok(Value::null_unknown());
        };

        if values_to_output.is_empty() {
            return Ok(Value::null_unknown());
        }

        // Convert values to JSON array format
        let json_elements: Vec<String> = values_to_output
            .iter()
            .map(|v| match v {
                Value::Text(s) => {
                    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
                }
                Value::Integer(i) => i.to_string(),
                Value::Float(f) => f.to_string(),
                Value::Boolean(b) => b.to_string(),
                Value::Null(_) => "null".to_string(),
                Value::Timestamp(t) => format!("\"{}\"", t),
                Value::Extension(data)
                    if data.first() == Some(&(crate::core::DataType::Json as u8)) =>
                {
                    std::str::from_utf8(&data[1..]).unwrap_or("").to_string()
                }
                Value::Extension(_) => {
                    format!(
                        "\"{}\"",
                        v.to_string().replace('\\', "\\\\").replace('"', "\\\"")
                    )
                }
            })
            .collect();
        Ok(Value::text(format!("[{}]", json_elements.join(","))))
    }

    fn reset(&mut self) {
        self.values.clear();
        self.ordered_entries.clear();
        // Note: order_keys and has_order_by are kept as they're configuration
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(ArrayAggFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_agg_basic() {
        let mut agg = ArrayAggFunction::default();
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::text("[1,2,3]"));
    }

    #[test]
    fn test_array_agg_strings() {
        let mut agg = ArrayAggFunction::default();
        agg.accumulate(&Value::text("a"));
        agg.accumulate(&Value::text("b"));
        agg.accumulate(&Value::text("c"));
        assert_eq!(agg.result().unwrap(), Value::text("[\"a\",\"b\",\"c\"]"));
    }

    #[test]
    fn test_array_agg_ignores_null() {
        let mut agg = ArrayAggFunction::default();
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::null_unknown());
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::text("[1,3]"));
    }

    #[test]
    fn test_array_agg_distinct() {
        let mut agg = super::super::AggregateState::new(
            Box::new(ArrayAggFunction::default()),
            true,
            &[],
            vec![],
        );
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(1)); // duplicate
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::text("[1,2,3]"));
    }

    #[test]
    fn test_array_agg_empty() {
        let agg = ArrayAggFunction::default();
        assert!(agg.result().unwrap().is_null());
    }

    #[test]
    fn test_array_agg_mixed_types() {
        let mut agg = ArrayAggFunction::default();
        agg.accumulate(&Value::text("str"));
        agg.accumulate(&Value::Integer(42));
        agg.accumulate(&Value::Float(3.5));
        agg.accumulate(&Value::Boolean(true));
        assert_eq!(agg.result().unwrap(), Value::text("[\"str\",42,3.5,true]"));
    }

    #[test]
    fn test_array_agg_reset() {
        let mut agg = ArrayAggFunction::default();
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(2));
        agg.reset();
        assert!(agg.result().unwrap().is_null());
    }
}
