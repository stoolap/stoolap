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

//! STRING_AGG aggregate function

use crate::core::Value;
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

use super::DistinctTracker;

/// Entry for ordered aggregation - stores value with its sort keys
#[derive(Clone)]
struct OrderedEntry {
    value: String,
    sort_keys: Vec<Value>,
}

/// STRING_AGG aggregate function
///
/// Concatenates all non-NULL string values with a configurable separator.
/// Similar to GROUP_CONCAT in MySQL or STRING_AGG in PostgreSQL.
///
/// Usage:
///   STRING_AGG(column)           -- Uses comma as default separator
///   STRING_AGG(column, ' | ')    -- Uses custom separator
///   STRING_AGG(column ORDER BY x) -- With ordering
pub struct StringAggFunction {
    values: Vec<String>,
    /// Values with sort keys (used when ORDER BY is specified)
    ordered_entries: Vec<OrderedEntry>,
    /// Sort directions: true = ASC, false = DESC
    order_directions: Vec<bool>,
    /// Whether ORDER BY is active
    has_order_by: bool,
    distinct_tracker: Option<DistinctTracker>,
    /// Separator between values (default: ",")
    separator: String,
}

impl Default for StringAggFunction {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            ordered_entries: Vec::new(),
            order_directions: Vec::new(),
            has_order_by: false,
            distinct_tracker: None,
            separator: ",".to_string(),
        }
    }
}

impl AggregateFunction for StringAggFunction {
    fn name(&self) -> &str {
        "STRING_AGG"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "STRING_AGG",
            FunctionType::Aggregate,
            "Concatenates all non-NULL values with an optional separator (default: comma)",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Any, FunctionDataType::String],
                1,
                2,
            ),
        )
    }

    fn configure(&mut self, options: &[Value]) {
        // First option is the separator
        if let Some(Value::Text(sep)) = options.first() {
            self.separator = sep.to_string();
        }
    }

    fn set_order_by(&mut self, directions: Vec<bool>) {
        self.order_directions = directions;
        self.has_order_by = true;
    }

    fn supports_order_by(&self) -> bool {
        true
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        // Handle NULL values - STRING_AGG ignores NULLs
        if value.is_null() {
            return;
        }

        // Handle DISTINCT case
        if distinct {
            if self.distinct_tracker.is_none() {
                self.distinct_tracker = Some(DistinctTracker::default());
            }
            if !self.distinct_tracker.as_mut().unwrap().check_and_add(value) {
                return; // Already seen this value
            }
        }

        // Convert value to string
        let s = match value {
            Value::Text(s) => s.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::Timestamp(t) => t.to_string(),
            Value::Json(j) => j.to_string(),
            Value::Null(_) => return,
        };

        self.values.push(s);
    }

    fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>, distinct: bool) {
        // Handle NULL values
        if value.is_null() {
            return;
        }

        // Handle DISTINCT case
        if distinct {
            if self.distinct_tracker.is_none() {
                self.distinct_tracker = Some(DistinctTracker::default());
            }
            if !self.distinct_tracker.as_mut().unwrap().check_and_add(value) {
                return; // Already seen this value
            }
        }

        // Convert value to string
        let s = match value {
            Value::Text(s) => s.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::Timestamp(t) => t.to_string(),
            Value::Json(j) => j.to_string(),
            Value::Null(_) => return,
        };

        self.ordered_entries.push(OrderedEntry {
            value: s,
            sort_keys,
        });
    }

    fn result(&self) -> Value {
        // Get the values to output, potentially sorted
        let values_to_output: Vec<&str> = if self.has_order_by && !self.ordered_entries.is_empty() {
            // Sort the entries by their sort keys
            let mut entries: Vec<&OrderedEntry> = self.ordered_entries.iter().collect();
            let directions = &self.order_directions;

            entries.sort_by(|a, b| {
                for (i, (key_a, key_b)) in a.sort_keys.iter().zip(b.sort_keys.iter()).enumerate() {
                    let is_asc = directions.get(i).copied().unwrap_or(true);
                    let cmp = compare_values(key_a, key_b);
                    if cmp != std::cmp::Ordering::Equal {
                        return if is_asc { cmp } else { cmp.reverse() };
                    }
                }
                std::cmp::Ordering::Equal
            });

            entries.iter().map(|e| e.value.as_str()).collect()
        } else if !self.values.is_empty() {
            self.values.iter().map(|s| s.as_str()).collect()
        } else if !self.ordered_entries.is_empty() {
            // Has ordered entries but no ORDER BY configured - use insertion order
            self.ordered_entries
                .iter()
                .map(|e| e.value.as_str())
                .collect()
        } else {
            return Value::null_unknown();
        };

        if values_to_output.is_empty() {
            Value::null_unknown()
        } else {
            Value::text(values_to_output.join(&self.separator))
        }
    }

    fn reset(&mut self) {
        self.values.clear();
        self.ordered_entries.clear();
        self.distinct_tracker = None;
        // Note: order_directions and has_order_by are kept as they're configuration
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(StringAggFunction::default())
    }
}

/// Compare two Values for sorting
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Null(_), Value::Null(_)) => std::cmp::Ordering::Equal,
        (Value::Null(_), _) => std::cmp::Ordering::Greater, // NULLs last
        (_, Value::Null(_)) => std::cmp::Ordering::Less,
        (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (Value::Integer(a), Value::Float(b)) => (*a as f64)
            .partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal),
        (Value::Float(a), Value::Integer(b)) => a
            .partial_cmp(&(*b as f64))
            .unwrap_or(std::cmp::Ordering::Equal),
        (Value::Text(a), Value::Text(b)) => a.cmp(b),
        (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
        (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
        _ => std::cmp::Ordering::Equal, // Different types compare equal
    }
}

/// GROUP_CONCAT function (MySQL-style alias for STRING_AGG)
#[derive(Default)]
pub struct GroupConcatFunction {
    inner: StringAggFunction,
}

impl AggregateFunction for GroupConcatFunction {
    fn name(&self) -> &str {
        "GROUP_CONCAT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "GROUP_CONCAT",
            FunctionType::Aggregate,
            "Concatenates all non-NULL values with an optional separator (MySQL-style alias for STRING_AGG)",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Any, FunctionDataType::String],
                1,
                2,
            ),
        )
    }

    fn configure(&mut self, options: &[Value]) {
        self.inner.configure(options);
    }

    fn set_order_by(&mut self, directions: Vec<bool>) {
        self.inner.set_order_by(directions);
    }

    fn supports_order_by(&self) -> bool {
        true
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        self.inner.accumulate(value, distinct);
    }

    fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>, distinct: bool) {
        self.inner
            .accumulate_with_sort_key(value, sort_keys, distinct);
    }

    fn result(&self) -> Value {
        self.inner.result()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(GroupConcatFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_agg_basic() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("a"), false);
        agg.accumulate(&Value::text("b"), false);
        agg.accumulate(&Value::text("c"), false);
        assert_eq!(agg.result(), Value::text("a,b,c"));
    }

    #[test]
    fn test_string_agg_ignores_null() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("a"), false);
        agg.accumulate(&Value::null_unknown(), false);
        agg.accumulate(&Value::text("c"), false);
        assert_eq!(agg.result(), Value::text("a,c"));
    }

    #[test]
    fn test_string_agg_distinct() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("a"), true);
        agg.accumulate(&Value::text("b"), true);
        agg.accumulate(&Value::text("a"), true); // duplicate
        agg.accumulate(&Value::text("c"), true);
        assert_eq!(agg.result(), Value::text("a,b,c"));
    }

    #[test]
    fn test_string_agg_empty() {
        let agg = StringAggFunction::default();
        assert!(agg.result().is_null());
    }

    #[test]
    fn test_string_agg_with_numbers() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::Integer(1), false);
        agg.accumulate(&Value::Integer(2), false);
        agg.accumulate(&Value::Integer(3), false);
        assert_eq!(agg.result(), Value::text("1,2,3"));
    }

    #[test]
    fn test_string_agg_reset() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("a"), false);
        agg.accumulate(&Value::text("b"), false);
        agg.reset();
        assert!(agg.result().is_null());
    }

    #[test]
    fn test_group_concat() {
        let mut agg = GroupConcatFunction::default();
        agg.accumulate(&Value::text("hello"), false);
        agg.accumulate(&Value::text("world"), false);
        assert_eq!(agg.result(), Value::text("hello,world"));
    }

    #[test]
    fn test_string_agg_single_value() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("only"), false);
        assert_eq!(agg.result(), Value::text("only"));
    }

    #[test]
    fn test_string_agg_mixed_types() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("str"), false);
        agg.accumulate(&Value::Integer(42), false);
        agg.accumulate(&Value::Float(3.5), false);
        agg.accumulate(&Value::Boolean(true), false);
        assert_eq!(agg.result(), Value::text("str,42,3.5,true"));
    }

    #[test]
    fn test_string_agg_custom_separator() {
        let mut agg = StringAggFunction::default();
        agg.configure(&[Value::text(" | ")]);
        agg.accumulate(&Value::text("a"), false);
        agg.accumulate(&Value::text("b"), false);
        agg.accumulate(&Value::text("c"), false);
        assert_eq!(agg.result(), Value::text("a | b | c"));
    }

    #[test]
    fn test_string_agg_dash_separator() {
        let mut agg = StringAggFunction::default();
        agg.configure(&[Value::text("-")]);
        agg.accumulate(&Value::text("1"), false);
        agg.accumulate(&Value::text("2"), false);
        agg.accumulate(&Value::text("3"), false);
        assert_eq!(agg.result(), Value::text("1-2-3"));
    }

    #[test]
    fn test_group_concat_custom_separator() {
        let mut agg = GroupConcatFunction::default();
        agg.configure(&[Value::text("; ")]);
        agg.accumulate(&Value::text("hello"), false);
        agg.accumulate(&Value::text("world"), false);
        assert_eq!(agg.result(), Value::text("hello; world"));
    }
}
