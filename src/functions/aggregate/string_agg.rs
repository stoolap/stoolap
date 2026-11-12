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

use std::borrow::Cow;

use crate::core::{Result, Value};
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

/// A row's separator as its text, read as CONCAT_WS reads its own; None
/// for a NULL one
fn separator_text(separator: &Value) -> Option<Cow<'_, str>> {
    match separator.text_form() {
        None if separator.is_null() => None,
        text => Some(text.unwrap_or_default()),
    }
}

/// Entry for ordered aggregation - stores value with its sort keys
#[derive(Clone)]
struct OrderedEntry {
    value: String,
    sort_keys: Vec<Value>,
}

/// Appends `value` to `out`, after `separator` unless it is the first value
/// or the separator is NULL
fn join(out: &mut Option<String>, value: &str, separator: Option<&str>) {
    match out {
        None => *out = Some(value.to_owned()),
        Some(out) => {
            out.push_str(separator.unwrap_or_default());
            out.push_str(value);
        }
    }
}

/// STRING_AGG aggregate function
///
/// Concatenates all non-NULL string values, each after the first preceded
/// by the separator its row reads, nothing for a NULL one.
///
/// Usage:
///   STRING_AGG(column)           -- Uses comma as default separator
///   STRING_AGG(column, ' | ')    -- Uses custom separator
///   STRING_AGG(column ORDER BY x) -- With ordering
pub struct StringAggFunction {
    /// The values joined so far in arrival order, when none is sorted
    joined: Option<String>,
    /// Values with sort keys (used when ORDER BY is specified)
    ordered_entries: Vec<OrderedEntry>,
    /// Each entry's own separator, by position, when its row reads one
    row_separators: Vec<Value>,
    /// Sort directions: true = ASC, false = DESC
    order_keys: Vec<crate::functions::AggregateOrder>,
    /// Whether ORDER BY is active
    has_order_by: bool,
    /// The call's separator (default: ","), None for a NULL one
    separator: Option<String>,
}

impl StringAggFunction {
    /// Takes `value` with its row's separator `row`, the call's without
    /// one, and its sort keys
    #[inline(always)]
    fn push(&mut self, value: &Value, row: Option<&Value>, sort_keys: Option<Vec<Value>>) {
        // STRING_AGG ignores NULLs; a value is joined as its text form
        let Some(value) = value.text_form() else {
            return;
        };
        match (sort_keys, row) {
            (Some(sort_keys), row) => {
                if let Some(row) = row {
                    self.row_separators.push(row.clone());
                }
                self.ordered_entries.push(OrderedEntry {
                    value: value.into_owned(),
                    sort_keys,
                });
            }
            (None, None) => join(&mut self.joined, &value, self.separator.as_deref()),
            (None, Some(row)) => join(&mut self.joined, &value, separator_text(row).as_deref()),
        }
    }

    /// The separator of the entry at `at`
    fn separator_of(&self, at: usize) -> Option<Cow<'_, str>> {
        match self.row_separators.get(at) {
            Some(separator) => separator_text(separator),
            None => self.separator.as_deref().map(Cow::Borrowed),
        }
    }
}

impl Default for StringAggFunction {
    fn default() -> Self {
        Self {
            joined: None,
            ordered_entries: Vec::new(),
            row_separators: Vec::new(),
            order_keys: Vec::new(),
            has_order_by: false,
            separator: Some(",".to_string()),
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
        if let Some(sep) = options.first() {
            self.separator = separator_text(sep).map(Cow::into_owned);
        }
    }

    fn set_order_by(&mut self, keys: Vec<crate::functions::AggregateOrder>) {
        self.order_keys = keys;
        self.has_order_by = true;
    }

    fn supports_order_by(&self) -> bool {
        true
    }

    fn accumulate(&mut self, value: &Value) {
        self.push(value, None, None);
    }

    fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>) {
        self.push(value, None, Some(sort_keys));
    }

    fn accumulate_with_args(
        &mut self,
        value: &Value,
        args: &[Value],
        sort_keys: Option<Vec<Value>>,
    ) {
        self.push(value, args.first(), sort_keys);
    }

    fn result(&self) -> Result<Value> {
        if !self.has_order_by || self.ordered_entries.is_empty() {
            if let Some(joined) = &self.joined {
                return Ok(Value::text(joined));
            }
        }
        // Sorted by their keys under ORDER BY, else in arrival order
        let mut entries: Vec<&OrderedEntry> = self.ordered_entries.iter().collect();
        if self.has_order_by {
            let keys = &self.order_keys;
            entries.sort_by(|a, b| super::compare_sort_keys(&a.sort_keys, &b.sort_keys, keys));
        }
        let Some((first, rest)) = entries.split_first() else {
            return Ok(Value::null_unknown());
        };
        // An entry's position in arrival order, which its own separator has
        let base = self.ordered_entries.as_ptr() as usize;
        let at = |entry: &OrderedEntry| {
            (entry as *const OrderedEntry as usize - base) / std::mem::size_of::<OrderedEntry>()
        };
        // Room for the values and the call's separators; a row's own may grow it
        let values = entries.iter().map(|e| e.value.len()).sum::<usize>();
        let separators = rest.len() * self.separator.as_deref().map_or(0, str::len);
        let mut joined = String::with_capacity(values + separators);
        joined.push_str(&first.value);
        for entry in rest {
            joined.push_str(self.separator_of(at(entry)).as_deref().unwrap_or_default());
            joined.push_str(&entry.value);
        }
        Ok(Value::text(joined))
    }

    fn reset(&mut self) {
        self.joined = None;
        self.ordered_entries.clear();
        self.row_separators.clear();
        // Note: order_keys and has_order_by are kept as they're configuration
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(StringAggFunction::default())
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

    fn set_order_by(&mut self, keys: Vec<crate::functions::AggregateOrder>) {
        self.inner.set_order_by(keys);
    }

    fn supports_order_by(&self) -> bool {
        true
    }

    fn accumulate(&mut self, value: &Value) {
        self.inner.accumulate(value);
    }

    fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>) {
        self.inner.accumulate_with_sort_key(value, sort_keys);
    }

    fn accumulate_with_args(
        &mut self,
        value: &Value,
        args: &[Value],
        sort_keys: Option<Vec<Value>>,
    ) {
        self.inner.accumulate_with_args(value, args, sort_keys);
    }

    fn result(&self) -> Result<Value> {
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
        agg.accumulate(&Value::text("a"));
        agg.accumulate(&Value::text("b"));
        agg.accumulate(&Value::text("c"));
        assert_eq!(agg.result().unwrap(), Value::text("a,b,c"));
    }

    #[test]
    fn test_string_agg_ignores_null() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("a"));
        agg.accumulate(&Value::null_unknown());
        agg.accumulate(&Value::text("c"));
        assert_eq!(agg.result().unwrap(), Value::text("a,c"));
    }

    #[test]
    fn test_string_agg_distinct() {
        let mut agg = super::super::AggregateState::new(
            Box::new(StringAggFunction::default()),
            true,
            &[],
            vec![],
        );
        agg.accumulate(&Value::text("a"));
        agg.accumulate(&Value::text("b"));
        agg.accumulate(&Value::text("a")); // duplicate
        agg.accumulate(&Value::text("c"));
        assert_eq!(agg.result().unwrap(), Value::text("a,b,c"));
    }

    #[test]
    fn test_string_agg_empty() {
        let agg = StringAggFunction::default();
        assert!(agg.result().unwrap().is_null());
    }

    #[test]
    fn test_string_agg_with_numbers() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::Integer(1));
        agg.accumulate(&Value::Integer(2));
        agg.accumulate(&Value::Integer(3));
        assert_eq!(agg.result().unwrap(), Value::text("1,2,3"));
    }

    #[test]
    fn test_string_agg_reset() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("a"));
        agg.accumulate(&Value::text("b"));
        agg.reset();
        assert!(agg.result().unwrap().is_null());
    }

    #[test]
    fn test_group_concat() {
        let mut agg = GroupConcatFunction::default();
        agg.accumulate(&Value::text("hello"));
        agg.accumulate(&Value::text("world"));
        assert_eq!(agg.result().unwrap(), Value::text("hello,world"));
    }

    #[test]
    fn test_string_agg_single_value() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("only"));
        assert_eq!(agg.result().unwrap(), Value::text("only"));
    }

    #[test]
    fn test_string_agg_mixed_types() {
        let mut agg = StringAggFunction::default();
        agg.accumulate(&Value::text("str"));
        agg.accumulate(&Value::Integer(42));
        agg.accumulate(&Value::Float(3.5));
        agg.accumulate(&Value::Boolean(true));
        assert_eq!(agg.result().unwrap(), Value::text("str,42,3.5,true"));
    }

    #[test]
    fn test_string_agg_custom_separator() {
        let mut agg = StringAggFunction::default();
        agg.configure(&[Value::text(" | ")]);
        agg.accumulate(&Value::text("a"));
        agg.accumulate(&Value::text("b"));
        agg.accumulate(&Value::text("c"));
        assert_eq!(agg.result().unwrap(), Value::text("a | b | c"));
    }

    #[test]
    fn test_string_agg_dash_separator() {
        let mut agg = StringAggFunction::default();
        agg.configure(&[Value::text("-")]);
        agg.accumulate(&Value::text("1"));
        agg.accumulate(&Value::text("2"));
        agg.accumulate(&Value::text("3"));
        assert_eq!(agg.result().unwrap(), Value::text("1-2-3"));
    }

    #[test]
    fn test_group_concat_custom_separator() {
        let mut agg = GroupConcatFunction::default();
        agg.configure(&[Value::text("; ")]);
        agg.accumulate(&Value::text("hello"));
        agg.accumulate(&Value::text("world"));
        assert_eq!(agg.result().unwrap(), Value::text("hello; world"));
    }
}
