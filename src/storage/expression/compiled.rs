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

//! Compiled Filter Expressions for High-Performance Query Evaluation
//!
//! This module provides compile-time specialized filter expressions that eliminate
//! dynamic dispatch overhead in the query hot path.
//!
//! # Performance Benefits
//!
//! | Aspect | `Box<dyn Expression>` | `CompiledFilter` |
//! |--------|----------------------|------------------|
//! | Dispatch | Virtual (vtable) | Direct (enum match) |
//! | Inlining | Impossible | Full inlining |
//! | Branch prediction | Poor | Excellent |
//! | Memory | Heap + vtable | Stack/inline |
//! | Cost per row | ~5-10ns | ~1-2ns |
//!
//! # Usage
//!
//! ```ignore
//! // Compile expression once at query planning time
//! let compiled = CompiledFilter::compile(&expr, &schema);
//!
//! // Fast evaluation in hot loop
//! for row in rows {
//!     if compiled.matches(row) {
//!         // process row
//!     }
//! }
//! ```

use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::core::{Operator, Row, Schema, Value};

use super::between::BetweenExpr;
use super::comparison::ComparisonExpr;
use super::function::{FunctionArg, FunctionExpr};
use super::in_list::InListExpr;
use super::like::LikeExpr;
use super::logical::{AndExpr, ConstBoolExpr, NotExpr, OrExpr};
use super::null_check::NullCheckExpr;
use super::Expression;

/// Compiled pattern for LIKE expressions
#[derive(Debug, Clone)]
pub enum CompiledPattern {
    /// Exact match (no wildcards)
    Exact(Arc<str>),
    /// Prefix match (pattern%)
    Prefix(Arc<str>),
    /// Suffix match (%pattern)
    Suffix(Arc<str>),
    /// Contains match (%pattern%)
    Contains(Arc<str>),
    /// Full regex pattern (complex patterns)
    Regex(regex::Regex),
}

impl CompiledPattern {
    /// Compile a LIKE pattern into an optimized matcher
    pub fn compile(pattern: &str, case_insensitive: bool) -> Self {
        let pattern = if case_insensitive {
            pattern.to_lowercase()
        } else {
            pattern.to_string()
        };

        // Check for simple patterns we can optimize
        let has_leading_percent = pattern.starts_with('%');
        let has_trailing_percent = pattern.ends_with('%');
        let inner = pattern.trim_matches('%');

        // Check if inner part has any wildcards
        let has_inner_wildcards = inner.contains('%') || inner.contains('_');

        if !has_inner_wildcards {
            match (has_leading_percent, has_trailing_percent) {
                (false, false) => CompiledPattern::Exact(Arc::from(inner)),
                (false, true) => CompiledPattern::Prefix(Arc::from(inner)),
                (true, false) => CompiledPattern::Suffix(Arc::from(inner)),
                (true, true) => CompiledPattern::Contains(Arc::from(inner)),
            }
        } else {
            // Complex pattern - fall back to regex
            let regex_pattern = like_to_regex(&pattern);
            let flags = if case_insensitive { "(?i)" } else { "" };
            let full_pattern = format!("^{}{}$", flags, regex_pattern);
            CompiledPattern::Regex(regex::Regex::new(&full_pattern).unwrap_or_else(|_e| {
                // Fallback to match-nothing regex on error
                #[cfg(debug_assertions)]
                eprintln!(
                    "Warning: Failed to compile LIKE pattern '{}' as regex '{}': {}",
                    pattern, full_pattern, _e
                );
                regex::Regex::new("^$").unwrap()
            }))
        }
    }

    /// Check if a string matches this pattern
    #[inline(always)]
    pub fn matches(&self, s: &str, case_insensitive: bool) -> bool {
        // Avoid allocation for case-sensitive matching by using Cow
        use std::borrow::Cow;
        let s: Cow<'_, str> = if case_insensitive {
            Cow::Owned(s.to_lowercase())
        } else {
            Cow::Borrowed(s)
        };

        match self {
            CompiledPattern::Exact(pattern) => s.as_ref() == pattern.as_ref(),
            CompiledPattern::Prefix(prefix) => s.starts_with(prefix.as_ref()),
            CompiledPattern::Suffix(suffix) => s.ends_with(suffix.as_ref()),
            CompiledPattern::Contains(substr) => s.contains(substr.as_ref()),
            CompiledPattern::Regex(re) => re.is_match(&s),
        }
    }
}

/// Convert LIKE pattern to regex pattern
fn like_to_regex(pattern: &str) -> String {
    let mut result = String::with_capacity(pattern.len() * 2);
    let mut chars = pattern.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '%' => result.push_str(".*"),
            '_' => result.push('.'),
            '\\' => {
                // Escape sequence
                if let Some(&next) = chars.peek() {
                    if next == '%' || next == '_' || next == '\\' {
                        result.push(chars.next().unwrap());
                        continue;
                    }
                }
                result.push_str("\\\\");
            }
            // Escape regex metacharacters
            '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' => {
                result.push('\\');
                result.push(c);
            }
            _ => result.push(c),
        }
    }

    result
}

/// Compiled filter expression for high-performance row filtering
///
/// This enum provides compile-time specialized paths for common filter patterns,
/// eliminating virtual dispatch overhead in the query hot path.
#[derive(Debug, Clone)]
pub enum CompiledFilter {
    // =========================================================================
    // Integer comparisons (most common)
    // =========================================================================
    /// col = integer
    IntegerEq { col_idx: usize, value: i64 },
    /// col != integer
    IntegerNe { col_idx: usize, value: i64 },
    /// col > integer
    IntegerGt { col_idx: usize, value: i64 },
    /// col >= integer
    IntegerGte { col_idx: usize, value: i64 },
    /// col < integer
    IntegerLt { col_idx: usize, value: i64 },
    /// col <= integer
    IntegerLte { col_idx: usize, value: i64 },
    /// col BETWEEN min AND max (integers)
    IntegerBetween { col_idx: usize, min: i64, max: i64 },
    /// col IN (v1, v2, ...) for integers
    IntegerIn { col_idx: usize, values: Vec<i64> },

    // =========================================================================
    // Float comparisons
    // =========================================================================
    /// col = float
    FloatEq { col_idx: usize, value: f64 },
    /// col != float
    FloatNe { col_idx: usize, value: f64 },
    /// col > float
    FloatGt { col_idx: usize, value: f64 },
    /// col >= float
    FloatGte { col_idx: usize, value: f64 },
    /// col < float
    FloatLt { col_idx: usize, value: f64 },
    /// col <= float
    FloatLte { col_idx: usize, value: f64 },
    /// col BETWEEN min AND max (floats)
    FloatBetween { col_idx: usize, min: f64, max: f64 },

    // =========================================================================
    // String comparisons
    // =========================================================================
    /// col = string
    StringEq { col_idx: usize, value: Arc<str> },
    /// col != string
    StringNe { col_idx: usize, value: Arc<str> },
    /// col > string
    StringGt { col_idx: usize, value: Arc<str> },
    /// col >= string
    StringGte { col_idx: usize, value: Arc<str> },
    /// col < string
    StringLt { col_idx: usize, value: Arc<str> },
    /// col <= string
    StringLte { col_idx: usize, value: Arc<str> },
    /// col IN (v1, v2, ...) for strings
    StringIn {
        col_idx: usize,
        values: Vec<Arc<str>>,
    },
    /// col LIKE pattern
    StringLike {
        col_idx: usize,
        pattern: CompiledPattern,
        case_insensitive: bool,
        negated: bool,
    },

    // =========================================================================
    // Boolean comparisons
    // =========================================================================
    /// col = boolean
    BooleanEq { col_idx: usize, value: bool },
    /// col != boolean
    BooleanNe { col_idx: usize, value: bool },

    // =========================================================================
    // Timestamp comparisons
    // =========================================================================
    /// col = timestamp
    TimestampEq {
        col_idx: usize,
        value: DateTime<Utc>,
    },
    /// col != timestamp
    TimestampNe {
        col_idx: usize,
        value: DateTime<Utc>,
    },
    /// col > timestamp
    TimestampGt {
        col_idx: usize,
        value: DateTime<Utc>,
    },
    /// col >= timestamp
    TimestampGte {
        col_idx: usize,
        value: DateTime<Utc>,
    },
    /// col < timestamp
    TimestampLt {
        col_idx: usize,
        value: DateTime<Utc>,
    },
    /// col <= timestamp
    TimestampLte {
        col_idx: usize,
        value: DateTime<Utc>,
    },
    /// col BETWEEN min AND max (timestamps)
    TimestampBetween {
        col_idx: usize,
        min: DateTime<Utc>,
        max: DateTime<Utc>,
    },

    // =========================================================================
    // NULL checks
    // =========================================================================
    /// col IS NULL
    IsNull { col_idx: usize },
    /// col IS NOT NULL
    IsNotNull { col_idx: usize },

    // =========================================================================
    // Logical operators
    // =========================================================================
    /// expr AND expr
    And(Box<CompiledFilter>, Box<CompiledFilter>),
    /// expr AND expr AND expr ... (flattened for efficiency)
    AndN(Vec<CompiledFilter>),
    /// expr OR expr
    Or(Box<CompiledFilter>, Box<CompiledFilter>),
    /// expr OR expr OR expr ... (flattened for efficiency)
    OrN(Vec<CompiledFilter>),
    /// NOT expr
    Not(Box<CompiledFilter>),

    // =========================================================================
    // Constants
    // =========================================================================
    /// Always true
    True,
    /// Always false
    False,

    // =========================================================================
    // Scalar function expressions
    // =========================================================================
    /// UPPER(col) = value
    UpperEq { col_idx: usize, value: Arc<str> },
    /// LOWER(col) = value
    LowerEq { col_idx: usize, value: Arc<str> },
    /// TRIM(col) = value
    TrimEq { col_idx: usize, value: Arc<str> },
    /// LENGTH(col) = value
    LengthEq { col_idx: usize, value: i64 },
    /// LENGTH(col) != value
    LengthNe { col_idx: usize, value: i64 },
    /// LENGTH(col) > value
    LengthGt { col_idx: usize, value: i64 },
    /// LENGTH(col) >= value
    LengthGte { col_idx: usize, value: i64 },
    /// LENGTH(col) < value
    LengthLt { col_idx: usize, value: i64 },
    /// LENGTH(col) <= value
    LengthLte { col_idx: usize, value: i64 },

    // =========================================================================
    // Fallback for complex expressions
    // =========================================================================
    /// Dynamic expression (fallback for unsupported patterns)
    Dynamic(Box<dyn Expression>),
}

impl CompiledFilter {
    /// Compile a dynamic expression into an optimized CompiledFilter
    ///
    /// This analyzes the expression structure and creates type-specialized
    /// variants where possible, falling back to Dynamic for complex cases.
    pub fn compile(expr: &dyn Expression, schema: &Schema) -> Self {
        // Try to compile based on expression type using downcasting
        if let Some(comparison) = expr.as_any().downcast_ref::<ComparisonExpr>() {
            return Self::compile_comparison(comparison, schema);
        }

        if let Some(and_expr) = expr.as_any().downcast_ref::<AndExpr>() {
            return Self::compile_and(and_expr, schema);
        }

        if let Some(or_expr) = expr.as_any().downcast_ref::<OrExpr>() {
            return Self::compile_or(or_expr, schema);
        }

        if let Some(not_expr) = expr.as_any().downcast_ref::<NotExpr>() {
            if let Some(inner) = not_expr.get_inner() {
                return CompiledFilter::Not(Box::new(Self::compile(inner, schema)));
            }
        }

        if let Some(null_check) = expr.as_any().downcast_ref::<NullCheckExpr>() {
            return Self::compile_null_check(null_check, schema);
        }

        if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
            return Self::compile_in_list(in_list, schema);
        }

        if let Some(between) = expr.as_any().downcast_ref::<BetweenExpr>() {
            return Self::compile_between(between, schema);
        }

        if let Some(like_expr) = expr.as_any().downcast_ref::<LikeExpr>() {
            return Self::compile_like(like_expr, schema);
        }

        if let Some(const_bool) = expr.as_any().downcast_ref::<ConstBoolExpr>() {
            return if const_bool.value() {
                CompiledFilter::True
            } else {
                CompiledFilter::False
            };
        }

        if let Some(func_expr) = expr.as_any().downcast_ref::<FunctionExpr>() {
            return Self::compile_function_expr(func_expr, schema);
        }

        // Fallback to dynamic dispatch
        CompiledFilter::Dynamic(expr.clone_box())
    }

    /// Compile from a Box<dyn Expression>
    pub fn compile_boxed(expr: &dyn Expression, schema: &Schema) -> Self {
        Self::compile(expr, schema)
    }

    fn compile_comparison(expr: &ComparisonExpr, schema: &Schema) -> Self {
        let col_name = expr.get_column_name().unwrap_or("");
        let col_idx = match super::find_column_index(schema, col_name) {
            Some(idx) => idx,
            None => return CompiledFilter::Dynamic(expr.clone_box()),
        };

        let (_, op, value) = match expr.get_comparison_info() {
            Some(info) => info,
            None => return CompiledFilter::Dynamic(expr.clone_box()),
        };

        match value {
            Value::Integer(i) => {
                let i = *i;
                match op {
                    Operator::Eq => CompiledFilter::IntegerEq { col_idx, value: i },
                    Operator::Ne => CompiledFilter::IntegerNe { col_idx, value: i },
                    Operator::Gt => CompiledFilter::IntegerGt { col_idx, value: i },
                    Operator::Gte => CompiledFilter::IntegerGte { col_idx, value: i },
                    Operator::Lt => CompiledFilter::IntegerLt { col_idx, value: i },
                    Operator::Lte => CompiledFilter::IntegerLte { col_idx, value: i },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            Value::Float(f) => {
                let f = *f;
                match op {
                    Operator::Eq => CompiledFilter::FloatEq { col_idx, value: f },
                    Operator::Ne => CompiledFilter::FloatNe { col_idx, value: f },
                    Operator::Gt => CompiledFilter::FloatGt { col_idx, value: f },
                    Operator::Gte => CompiledFilter::FloatGte { col_idx, value: f },
                    Operator::Lt => CompiledFilter::FloatLt { col_idx, value: f },
                    Operator::Lte => CompiledFilter::FloatLte { col_idx, value: f },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            Value::Text(s) => {
                let s: Arc<str> = Arc::from(s.as_ref());
                match op {
                    Operator::Eq => CompiledFilter::StringEq { col_idx, value: s },
                    Operator::Ne => CompiledFilter::StringNe { col_idx, value: s },
                    Operator::Gt => CompiledFilter::StringGt { col_idx, value: s },
                    Operator::Gte => CompiledFilter::StringGte { col_idx, value: s },
                    Operator::Lt => CompiledFilter::StringLt { col_idx, value: s },
                    Operator::Lte => CompiledFilter::StringLte { col_idx, value: s },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            Value::Boolean(b) => {
                let b = *b;
                match op {
                    Operator::Eq => CompiledFilter::BooleanEq { col_idx, value: b },
                    Operator::Ne => CompiledFilter::BooleanNe { col_idx, value: b },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            Value::Timestamp(t) => {
                let t = *t;
                match op {
                    Operator::Eq => CompiledFilter::TimestampEq { col_idx, value: t },
                    Operator::Ne => CompiledFilter::TimestampNe { col_idx, value: t },
                    Operator::Gt => CompiledFilter::TimestampGt { col_idx, value: t },
                    Operator::Gte => CompiledFilter::TimestampGte { col_idx, value: t },
                    Operator::Lt => CompiledFilter::TimestampLt { col_idx, value: t },
                    Operator::Lte => CompiledFilter::TimestampLte { col_idx, value: t },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            _ => CompiledFilter::Dynamic(expr.clone_box()),
        }
    }

    fn compile_and(expr: &AndExpr, schema: &Schema) -> Self {
        if let Some(operands) = expr.get_and_operands() {
            // Compile all operands and flatten nested ANDs
            let mut flattened = Vec::with_capacity(operands.len());
            for op in operands.iter() {
                let compiled = Self::compile(op.as_ref(), schema);
                Self::flatten_and(&mut flattened, compiled);
            }

            // Short-circuit optimization: AND with False is always False
            if flattened.iter().any(|f| matches!(f, CompiledFilter::False)) {
                return CompiledFilter::False;
            }

            // Remove True constants (A AND True = A)
            flattened.retain(|f| !matches!(f, CompiledFilter::True));

            // Handle resulting cases
            match flattened.len() {
                0 => CompiledFilter::True, // All were True constants
                1 => flattened.pop().unwrap(),
                2 => {
                    let right = flattened.pop().unwrap();
                    let left = flattened.pop().unwrap();
                    CompiledFilter::And(Box::new(left), Box::new(right))
                }
                _ => CompiledFilter::AndN(flattened),
            }
        } else {
            CompiledFilter::Dynamic(expr.clone_box())
        }
    }

    /// Helper to flatten nested AND expressions into a single vector
    fn flatten_and(result: &mut Vec<CompiledFilter>, filter: CompiledFilter) {
        match filter {
            CompiledFilter::And(left, right) => {
                Self::flatten_and(result, *left);
                Self::flatten_and(result, *right);
            }
            CompiledFilter::AndN(filters) => {
                for f in filters {
                    Self::flatten_and(result, f);
                }
            }
            other => result.push(other),
        }
    }

    fn compile_or(expr: &OrExpr, schema: &Schema) -> Self {
        if let Some(operands) = expr.get_or_operands() {
            // Compile all operands and flatten nested ORs
            let mut flattened = Vec::with_capacity(operands.len());
            for op in operands.iter() {
                let compiled = Self::compile(op.as_ref(), schema);
                Self::flatten_or(&mut flattened, compiled);
            }

            // Short-circuit optimization: OR with True is always True
            if flattened.iter().any(|f| matches!(f, CompiledFilter::True)) {
                return CompiledFilter::True;
            }

            // Remove False constants (A OR False = A)
            flattened.retain(|f| !matches!(f, CompiledFilter::False));

            // Handle resulting cases
            match flattened.len() {
                0 => CompiledFilter::False, // All were False constants
                1 => flattened.pop().unwrap(),
                2 => {
                    let right = flattened.pop().unwrap();
                    let left = flattened.pop().unwrap();
                    CompiledFilter::Or(Box::new(left), Box::new(right))
                }
                _ => CompiledFilter::OrN(flattened),
            }
        } else {
            CompiledFilter::Dynamic(expr.clone_box())
        }
    }

    /// Helper to flatten nested OR expressions into a single vector
    fn flatten_or(result: &mut Vec<CompiledFilter>, filter: CompiledFilter) {
        match filter {
            CompiledFilter::Or(left, right) => {
                Self::flatten_or(result, *left);
                Self::flatten_or(result, *right);
            }
            CompiledFilter::OrN(filters) => {
                for f in filters {
                    Self::flatten_or(result, f);
                }
            }
            other => result.push(other),
        }
    }

    fn compile_null_check(expr: &NullCheckExpr, schema: &Schema) -> Self {
        let col_name = expr.get_column_name().unwrap_or("");
        match super::find_column_index(schema, col_name) {
            Some(col_idx) => {
                if expr.is_null_check() {
                    CompiledFilter::IsNull { col_idx }
                } else {
                    CompiledFilter::IsNotNull { col_idx }
                }
            }
            None => CompiledFilter::Dynamic(expr.clone_box()),
        }
    }

    fn compile_in_list(expr: &InListExpr, schema: &Schema) -> Self {
        let col_name = expr.get_column_name().unwrap_or("");
        let col_idx = match super::find_column_index(schema, col_name) {
            Some(idx) => idx,
            None => return CompiledFilter::Dynamic(expr.clone_box()),
        };

        let values = expr.get_values();
        let is_negated = expr.is_not();

        if values.is_empty() {
            // Empty list: IN () is always false, NOT IN () is always true
            return if is_negated {
                CompiledFilter::True
            } else {
                CompiledFilter::False
            };
        }

        // Check if all values are the same type
        let first = &values[0];
        let in_filter = match first {
            Value::Integer(_) => {
                let int_values: Vec<i64> = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Integer(i) => Some(*i),
                        _ => None,
                    })
                    .collect();
                if int_values.len() == values.len() {
                    Some(CompiledFilter::IntegerIn {
                        col_idx,
                        values: int_values,
                    })
                } else {
                    None
                }
            }
            Value::Text(_) => {
                let str_values: Vec<Arc<str>> = values
                    .iter()
                    .filter_map(|v| match v {
                        Value::Text(s) => Some(Arc::from(s.as_ref())),
                        _ => None,
                    })
                    .collect();
                if str_values.len() == values.len() {
                    Some(CompiledFilter::StringIn {
                        col_idx,
                        values: str_values,
                    })
                } else {
                    None
                }
            }
            _ => None,
        };

        match in_filter {
            Some(filter) => {
                // Wrap in NOT if this is a NOT IN expression
                if is_negated {
                    CompiledFilter::Not(Box::new(filter))
                } else {
                    filter
                }
            }
            None => CompiledFilter::Dynamic(expr.clone_box()),
        }
    }

    fn compile_between(expr: &BetweenExpr, schema: &Schema) -> Self {
        let col_name = expr.get_column_name().unwrap_or("");
        let col_idx = match super::find_column_index(schema, col_name) {
            Some(idx) => idx,
            None => return CompiledFilter::Dynamic(expr.clone_box()),
        };

        let (low, high) = expr.get_bounds();
        let is_negated = expr.is_negated();

        let between_filter = match (low, high) {
            (Value::Integer(min), Value::Integer(max)) => CompiledFilter::IntegerBetween {
                col_idx,
                min: *min,
                max: *max,
            },
            (Value::Float(min), Value::Float(max)) => CompiledFilter::FloatBetween {
                col_idx,
                min: *min,
                max: *max,
            },
            (Value::Timestamp(min), Value::Timestamp(max)) => CompiledFilter::TimestampBetween {
                col_idx,
                min: *min,
                max: *max,
            },
            _ => return CompiledFilter::Dynamic(expr.clone_box()),
        };

        // Wrap in NOT if this is a NOT BETWEEN expression
        if is_negated {
            CompiledFilter::Not(Box::new(between_filter))
        } else {
            between_filter
        }
    }

    fn compile_like(expr: &LikeExpr, schema: &Schema) -> Self {
        let col_name = expr.get_column_name().unwrap_or("");
        let col_idx = match super::find_column_index(schema, col_name) {
            Some(idx) => idx,
            None => return CompiledFilter::Dynamic(expr.clone_box()),
        };

        let pattern_str = expr.get_pattern();
        let case_insensitive = expr.is_case_insensitive();
        let negated = expr.is_negated();

        let pattern = CompiledPattern::compile(pattern_str, case_insensitive);

        CompiledFilter::StringLike {
            col_idx,
            pattern,
            case_insensitive,
            negated,
        }
    }

    fn compile_function_expr(expr: &FunctionExpr, schema: &Schema) -> Self {
        let func_name = expr.function_name().to_uppercase();
        let args = expr.get_arguments();
        let op = expr.get_operator();
        let compare_val = expr.get_compare_value();

        // We only support single-column-argument functions for now
        if args.len() != 1 {
            return CompiledFilter::Dynamic(expr.clone_box());
        }

        // Get the column index
        let col_idx = match &args[0] {
            FunctionArg::Column(col_name) => match super::find_column_index(schema, col_name) {
                Some(idx) => idx,
                None => return CompiledFilter::Dynamic(expr.clone_box()),
            },
            _ => return CompiledFilter::Dynamic(expr.clone_box()),
        };

        match func_name.as_str() {
            "UPPER" => {
                // UPPER(col) = 'VALUE' - compare value must be string and operator must be Eq
                if op != Operator::Eq {
                    return CompiledFilter::Dynamic(expr.clone_box());
                }
                match compare_val {
                    Value::Text(s) => CompiledFilter::UpperEq {
                        col_idx,
                        value: Arc::from(s.as_ref()),
                    },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            "LOWER" => {
                // LOWER(col) = 'value' - compare value must be string and operator must be Eq
                if op != Operator::Eq {
                    return CompiledFilter::Dynamic(expr.clone_box());
                }
                match compare_val {
                    Value::Text(s) => CompiledFilter::LowerEq {
                        col_idx,
                        value: Arc::from(s.as_ref()),
                    },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            "TRIM" | "BTRIM" => {
                // TRIM(col) = 'value' - compare value must be string and operator must be Eq
                if op != Operator::Eq {
                    return CompiledFilter::Dynamic(expr.clone_box());
                }
                match compare_val {
                    Value::Text(s) => CompiledFilter::TrimEq {
                        col_idx,
                        value: Arc::from(s.as_ref()),
                    },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            "LENGTH" | "LEN" | "CHAR_LENGTH" | "CHARACTER_LENGTH" => {
                // LENGTH(col) op N - compare value must be integer
                match compare_val {
                    Value::Integer(n) => match op {
                        Operator::Eq => CompiledFilter::LengthEq { col_idx, value: *n },
                        Operator::Ne => CompiledFilter::LengthNe { col_idx, value: *n },
                        Operator::Gt => CompiledFilter::LengthGt { col_idx, value: *n },
                        Operator::Gte => CompiledFilter::LengthGte { col_idx, value: *n },
                        Operator::Lt => CompiledFilter::LengthLt { col_idx, value: *n },
                        Operator::Lte => CompiledFilter::LengthLte { col_idx, value: *n },
                        _ => CompiledFilter::Dynamic(expr.clone_box()),
                    },
                    _ => CompiledFilter::Dynamic(expr.clone_box()),
                }
            }
            _ => CompiledFilter::Dynamic(expr.clone_box()),
        }
    }

    /// Evaluate the filter against a row
    ///
    /// This is the hot path - all type checks are eliminated through enum matching.
    #[inline(always)]
    pub fn matches(&self, row: &Row) -> bool {
        match self {
            // Integer comparisons
            CompiledFilter::IntegerEq { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Integer(i)) if *i == *value)
            }
            CompiledFilter::IntegerNe { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Integer(i)) if *i != *value)
            }
            CompiledFilter::IntegerGt { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Integer(i)) if *i > *value)
            }
            CompiledFilter::IntegerGte { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Integer(i)) if *i >= *value)
            }
            CompiledFilter::IntegerLt { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Integer(i)) if *i < *value)
            }
            CompiledFilter::IntegerLte { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Integer(i)) if *i <= *value)
            }
            CompiledFilter::IntegerBetween { col_idx, min, max } => {
                matches!(row.get(*col_idx), Some(Value::Integer(i)) if *i >= *min && *i <= *max)
            }
            CompiledFilter::IntegerIn { col_idx, values } => {
                if let Some(Value::Integer(i)) = row.get(*col_idx) {
                    values.contains(i)
                } else {
                    false
                }
            }

            // Float comparisons
            CompiledFilter::FloatEq { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Float(f)) if *f == *value)
            }
            CompiledFilter::FloatNe { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Float(f)) if *f != *value)
            }
            CompiledFilter::FloatGt { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Float(f)) if *f > *value)
            }
            CompiledFilter::FloatGte { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Float(f)) if *f >= *value)
            }
            CompiledFilter::FloatLt { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Float(f)) if *f < *value)
            }
            CompiledFilter::FloatLte { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Float(f)) if *f <= *value)
            }
            CompiledFilter::FloatBetween { col_idx, min, max } => {
                matches!(row.get(*col_idx), Some(Value::Float(f)) if *f >= *min && *f <= *max)
            }

            // String comparisons
            CompiledFilter::StringEq { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.as_ref() == value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::StringNe { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.as_ref() != value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::StringGt { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.as_ref() > value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::StringGte { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.as_ref() >= value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::StringLt { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.as_ref() < value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::StringLte { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.as_ref() <= value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::StringIn { col_idx, values } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    values.iter().any(|v| s.as_ref() == v.as_ref())
                } else {
                    false
                }
            }
            CompiledFilter::StringLike {
                col_idx,
                pattern,
                case_insensitive,
                negated,
            } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    let matches = pattern.matches(s.as_ref(), *case_insensitive);
                    if *negated {
                        !matches
                    } else {
                        matches
                    }
                } else {
                    false
                }
            }

            // Boolean comparisons
            CompiledFilter::BooleanEq { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Boolean(b)) if *b == *value)
            }
            CompiledFilter::BooleanNe { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Boolean(b)) if *b != *value)
            }

            // Timestamp comparisons
            CompiledFilter::TimestampEq { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Timestamp(t)) if *t == *value)
            }
            CompiledFilter::TimestampNe { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Timestamp(t)) if *t != *value)
            }
            CompiledFilter::TimestampGt { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Timestamp(t)) if *t > *value)
            }
            CompiledFilter::TimestampGte { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Timestamp(t)) if *t >= *value)
            }
            CompiledFilter::TimestampLt { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Timestamp(t)) if *t < *value)
            }
            CompiledFilter::TimestampLte { col_idx, value } => {
                matches!(row.get(*col_idx), Some(Value::Timestamp(t)) if *t <= *value)
            }
            CompiledFilter::TimestampBetween { col_idx, min, max } => {
                matches!(row.get(*col_idx), Some(Value::Timestamp(t)) if *t >= *min && *t <= *max)
            }

            // NULL checks
            CompiledFilter::IsNull { col_idx } => {
                row.get(*col_idx).map(|v| v.is_null()).unwrap_or(true)
            }
            CompiledFilter::IsNotNull { col_idx } => {
                row.get(*col_idx).map(|v| !v.is_null()).unwrap_or(false)
            }

            // Logical operators
            CompiledFilter::And(left, right) => left.matches(row) && right.matches(row),
            CompiledFilter::AndN(filters) => filters.iter().all(|f| f.matches(row)),
            CompiledFilter::Or(left, right) => left.matches(row) || right.matches(row),
            CompiledFilter::OrN(filters) => filters.iter().any(|f| f.matches(row)),
            CompiledFilter::Not(inner) => {
                // Three-valued logic: NOT(UNKNOWN) = UNKNOWN (false for filtering)
                // Check if inner's false result is due to NULL (UNKNOWN)
                if inner.is_unknown_due_to_null(row) {
                    return false;
                }
                !inner.matches(row)
            }

            // Constants
            CompiledFilter::True => true,
            CompiledFilter::False => false,

            // Scalar function expressions
            CompiledFilter::UpperEq { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    // UPPER(col) = value: convert column to uppercase and compare
                    s.to_uppercase() == value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::LowerEq { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    // LOWER(col) = value: convert column to lowercase and compare
                    s.to_lowercase() == value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::TrimEq { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    // TRIM(col) = value: trim whitespace and compare
                    s.trim() == value.as_ref()
                } else {
                    false
                }
            }
            CompiledFilter::LengthEq { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.chars().count() as i64 == *value
                } else {
                    false
                }
            }
            CompiledFilter::LengthNe { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.chars().count() as i64 != *value
                } else {
                    false
                }
            }
            CompiledFilter::LengthGt { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.chars().count() as i64 > *value
                } else {
                    false
                }
            }
            CompiledFilter::LengthGte { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.chars().count() as i64 >= *value
                } else {
                    false
                }
            }
            CompiledFilter::LengthLt { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    (s.chars().count() as i64) < *value
                } else {
                    false
                }
            }
            CompiledFilter::LengthLte { col_idx, value } => {
                if let Some(Value::Text(s)) = row.get(*col_idx) {
                    s.chars().count() as i64 <= *value
                } else {
                    false
                }
            }

            // Dynamic fallback
            CompiledFilter::Dynamic(expr) => expr.evaluate_fast(row),
        }
    }

    /// Check if this filter is fully compiled (no Dynamic fallbacks)
    pub fn is_fully_compiled(&self) -> bool {
        match self {
            CompiledFilter::Dynamic(_) => false,
            CompiledFilter::And(left, right) => {
                left.is_fully_compiled() && right.is_fully_compiled()
            }
            CompiledFilter::AndN(filters) => filters.iter().all(|f| f.is_fully_compiled()),
            CompiledFilter::Or(left, right) => {
                left.is_fully_compiled() && right.is_fully_compiled()
            }
            CompiledFilter::OrN(filters) => filters.iter().all(|f| f.is_fully_compiled()),
            CompiledFilter::Not(inner) => inner.is_fully_compiled(),
            _ => true,
        }
    }

    /// Check if the filter result would be UNKNOWN (NULL) due to NULL column values
    ///
    /// In SQL's three-valued logic, comparisons with NULL return UNKNOWN.
    /// For filtering purposes, UNKNOWN is treated as false.
    /// However, NOT(UNKNOWN) should remain UNKNOWN, not become true.
    ///
    /// This method detects when a false result is actually UNKNOWN due to NULL.
    #[inline(always)]
    pub fn is_unknown_due_to_null(&self, row: &Row) -> bool {
        match self {
            // Comparison with NULL column produces UNKNOWN
            CompiledFilter::IntegerEq { col_idx, .. }
            | CompiledFilter::IntegerNe { col_idx, .. }
            | CompiledFilter::IntegerGt { col_idx, .. }
            | CompiledFilter::IntegerGte { col_idx, .. }
            | CompiledFilter::IntegerLt { col_idx, .. }
            | CompiledFilter::IntegerLte { col_idx, .. }
            | CompiledFilter::IntegerBetween { col_idx, .. }
            | CompiledFilter::IntegerIn { col_idx, .. }
            | CompiledFilter::FloatEq { col_idx, .. }
            | CompiledFilter::FloatNe { col_idx, .. }
            | CompiledFilter::FloatGt { col_idx, .. }
            | CompiledFilter::FloatGte { col_idx, .. }
            | CompiledFilter::FloatLt { col_idx, .. }
            | CompiledFilter::FloatLte { col_idx, .. }
            | CompiledFilter::FloatBetween { col_idx, .. }
            | CompiledFilter::StringEq { col_idx, .. }
            | CompiledFilter::StringNe { col_idx, .. }
            | CompiledFilter::StringGt { col_idx, .. }
            | CompiledFilter::StringGte { col_idx, .. }
            | CompiledFilter::StringLt { col_idx, .. }
            | CompiledFilter::StringLte { col_idx, .. }
            | CompiledFilter::StringIn { col_idx, .. }
            | CompiledFilter::StringLike { col_idx, .. }
            | CompiledFilter::BooleanEq { col_idx, .. }
            | CompiledFilter::BooleanNe { col_idx, .. }
            | CompiledFilter::TimestampEq { col_idx, .. }
            | CompiledFilter::TimestampNe { col_idx, .. }
            | CompiledFilter::TimestampGt { col_idx, .. }
            | CompiledFilter::TimestampGte { col_idx, .. }
            | CompiledFilter::TimestampLt { col_idx, .. }
            | CompiledFilter::TimestampLte { col_idx, .. }
            | CompiledFilter::TimestampBetween { col_idx, .. }
            // Scalar function expressions also produce UNKNOWN if column is NULL
            | CompiledFilter::UpperEq { col_idx, .. }
            | CompiledFilter::LowerEq { col_idx, .. }
            | CompiledFilter::TrimEq { col_idx, .. }
            | CompiledFilter::LengthEq { col_idx, .. }
            | CompiledFilter::LengthNe { col_idx, .. }
            | CompiledFilter::LengthGt { col_idx, .. }
            | CompiledFilter::LengthGte { col_idx, .. }
            | CompiledFilter::LengthLt { col_idx, .. }
            | CompiledFilter::LengthLte { col_idx, .. } => {
                // Result is UNKNOWN if the column value is NULL
                row.get(*col_idx).map(|v| v.is_null()).unwrap_or(true)
            }

            // IS NULL / IS NOT NULL - never produces UNKNOWN (they check for NULL explicitly)
            CompiledFilter::IsNull { .. } | CompiledFilter::IsNotNull { .. } => false,

            // AND/OR: Match original Expression behavior which returns false by default
            // for is_unknown_due_to_null. This means NOT(AND(...)) won't propagate
            // UNKNOWN from children, matching the existing Expression semantics.
            CompiledFilter::And(_, _) | CompiledFilter::AndN(_) => false,

            // OR: Match original Expression behavior which returns false by default
            CompiledFilter::Or(_, _) | CompiledFilter::OrN(_) => false,

            // NOT(UNKNOWN) = UNKNOWN
            CompiledFilter::Not(inner) => inner.is_unknown_due_to_null(row),

            // Constants never produce UNKNOWN
            CompiledFilter::True | CompiledFilter::False => false,

            // Dynamic expressions have their own is_unknown_due_to_null implementation
            CompiledFilter::Dynamic(expr) => expr.is_unknown_due_to_null(row),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(id: i64, name: &str, age: i64, score: f64, active: bool) -> Row {
        Row::from_values(vec![
            Value::Integer(id),
            Value::text(name),
            Value::Integer(age),
            Value::Float(score),
            Value::Boolean(active),
        ])
    }

    #[test]
    fn test_integer_eq() {
        let filter = CompiledFilter::IntegerEq {
            col_idx: 0,
            value: 42,
        };

        let row1 = make_row(42, "Alice", 30, 95.5, true);
        let row2 = make_row(43, "Bob", 25, 88.0, false);

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
    }

    #[test]
    fn test_integer_between() {
        let filter = CompiledFilter::IntegerBetween {
            col_idx: 2,
            min: 25,
            max: 35,
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true);
        let row2 = make_row(2, "Bob", 40, 88.0, false);
        let row3 = make_row(3, "Charlie", 25, 70.0, true);

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
        assert!(filter.matches(&row3));
    }

    #[test]
    fn test_string_eq() {
        let filter = CompiledFilter::StringEq {
            col_idx: 1,
            value: Arc::from("Alice"),
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true);
        let row2 = make_row(2, "Bob", 25, 88.0, false);

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
    }

    #[test]
    fn test_and_filter() {
        let filter = CompiledFilter::And(
            Box::new(CompiledFilter::IntegerGte {
                col_idx: 2,
                value: 25,
            }),
            Box::new(CompiledFilter::BooleanEq {
                col_idx: 4,
                value: true,
            }),
        );

        let row1 = make_row(1, "Alice", 30, 95.5, true); // age >= 25 AND active = true
        let row2 = make_row(2, "Bob", 30, 88.0, false); // age >= 25 but active = false
        let row3 = make_row(3, "Charlie", 20, 70.0, true); // active = true but age < 25

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
        assert!(!filter.matches(&row3));
    }

    #[test]
    fn test_or_filter() {
        let filter = CompiledFilter::Or(
            Box::new(CompiledFilter::StringEq {
                col_idx: 1,
                value: Arc::from("Alice"),
            }),
            Box::new(CompiledFilter::StringEq {
                col_idx: 1,
                value: Arc::from("Bob"),
            }),
        );

        let row1 = make_row(1, "Alice", 30, 95.5, true);
        let row2 = make_row(2, "Bob", 25, 88.0, false);
        let row3 = make_row(3, "Charlie", 35, 70.0, true);

        assert!(filter.matches(&row1));
        assert!(filter.matches(&row2));
        assert!(!filter.matches(&row3));
    }

    #[test]
    fn test_integer_in() {
        let filter = CompiledFilter::IntegerIn {
            col_idx: 0,
            values: vec![1, 3, 5, 7, 9],
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true);
        let row2 = make_row(2, "Bob", 25, 88.0, false);
        let row3 = make_row(5, "Charlie", 35, 70.0, true);

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
        assert!(filter.matches(&row3));
    }

    #[test]
    fn test_like_prefix() {
        let pattern = CompiledPattern::compile("Al%", false);
        let filter = CompiledFilter::StringLike {
            col_idx: 1,
            pattern,
            case_insensitive: false,
            negated: false,
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true);
        let row2 = make_row(2, "Bob", 25, 88.0, false);
        let row3 = make_row(3, "Albert", 35, 70.0, true);

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
        assert!(filter.matches(&row3));
    }

    #[test]
    fn test_is_fully_compiled() {
        let filter1 = CompiledFilter::IntegerEq {
            col_idx: 0,
            value: 42,
        };
        assert!(filter1.is_fully_compiled());

        let filter2 = CompiledFilter::And(
            Box::new(CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 42,
            }),
            Box::new(CompiledFilter::StringEq {
                col_idx: 1,
                value: Arc::from("test"),
            }),
        );
        assert!(filter2.is_fully_compiled());
    }

    // =========================================================================
    // Scalar function expression tests
    // =========================================================================

    #[test]
    fn test_upper_eq() {
        let filter = CompiledFilter::UpperEq {
            col_idx: 1,
            value: Arc::from("ALICE"),
        };

        let row1 = make_row(1, "alice", 30, 95.5, true);
        let row2 = make_row(2, "Alice", 25, 88.0, false);
        let row3 = make_row(3, "ALICE", 35, 70.0, true);
        let row4 = make_row(4, "bob", 40, 80.0, false);

        assert!(filter.matches(&row1)); // "alice" -> "ALICE"
        assert!(filter.matches(&row2)); // "Alice" -> "ALICE"
        assert!(filter.matches(&row3)); // "ALICE" -> "ALICE"
        assert!(!filter.matches(&row4)); // "bob" -> "BOB" != "ALICE"
    }

    #[test]
    fn test_lower_eq() {
        let filter = CompiledFilter::LowerEq {
            col_idx: 1,
            value: Arc::from("alice"),
        };

        let row1 = make_row(1, "alice", 30, 95.5, true);
        let row2 = make_row(2, "Alice", 25, 88.0, false);
        let row3 = make_row(3, "ALICE", 35, 70.0, true);
        let row4 = make_row(4, "Bob", 40, 80.0, false);

        assert!(filter.matches(&row1)); // "alice" -> "alice"
        assert!(filter.matches(&row2)); // "Alice" -> "alice"
        assert!(filter.matches(&row3)); // "ALICE" -> "alice"
        assert!(!filter.matches(&row4)); // "Bob" -> "bob" != "alice"
    }

    #[test]
    fn test_trim_eq() {
        let filter = CompiledFilter::TrimEq {
            col_idx: 1,
            value: Arc::from("Alice"),
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true);
        let row2 = make_row(2, "  Alice  ", 25, 88.0, false);
        let row3 = make_row(3, "Alice ", 35, 70.0, true);
        let row4 = make_row(4, " Bob ", 40, 80.0, false);

        assert!(filter.matches(&row1)); // "Alice".trim() = "Alice"
        assert!(filter.matches(&row2)); // "  Alice  ".trim() = "Alice"
        assert!(filter.matches(&row3)); // "Alice ".trim() = "Alice"
        assert!(!filter.matches(&row4)); // " Bob ".trim() = "Bob" != "Alice"
    }

    #[test]
    fn test_length_eq() {
        let filter = CompiledFilter::LengthEq {
            col_idx: 1,
            value: 5,
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true); // len = 5
        let row2 = make_row(2, "Bob", 25, 88.0, false); // len = 3

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
    }

    #[test]
    fn test_length_gt() {
        let filter = CompiledFilter::LengthGt {
            col_idx: 1,
            value: 3,
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true); // len = 5 > 3
        let row2 = make_row(2, "Bob", 25, 88.0, false); // len = 3, not > 3
        let row3 = make_row(3, "Al", 35, 70.0, true); // len = 2, not > 3

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
        assert!(!filter.matches(&row3));
    }

    #[test]
    fn test_length_gte() {
        let filter = CompiledFilter::LengthGte {
            col_idx: 1,
            value: 3,
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true); // len = 5 >= 3
        let row2 = make_row(2, "Bob", 25, 88.0, false); // len = 3 >= 3
        let row3 = make_row(3, "Al", 35, 70.0, true); // len = 2, not >= 3

        assert!(filter.matches(&row1));
        assert!(filter.matches(&row2));
        assert!(!filter.matches(&row3));
    }

    #[test]
    fn test_length_lt() {
        let filter = CompiledFilter::LengthLt {
            col_idx: 1,
            value: 4,
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true); // len = 5, not < 4
        let row2 = make_row(2, "Bob", 25, 88.0, false); // len = 3 < 4
        let row3 = make_row(3, "Al", 35, 70.0, true); // len = 2 < 4

        assert!(!filter.matches(&row1));
        assert!(filter.matches(&row2));
        assert!(filter.matches(&row3));
    }

    #[test]
    fn test_length_lte() {
        let filter = CompiledFilter::LengthLte {
            col_idx: 1,
            value: 3,
        };

        let row1 = make_row(1, "Alice", 30, 95.5, true); // len = 5, not <= 3
        let row2 = make_row(2, "Bob", 25, 88.0, false); // len = 3 <= 3
        let row3 = make_row(3, "Al", 35, 70.0, true); // len = 2 <= 3

        assert!(!filter.matches(&row1));
        assert!(filter.matches(&row2));
        assert!(filter.matches(&row3));
    }

    #[test]
    fn test_scalar_func_with_null() {
        let filter = CompiledFilter::UpperEq {
            col_idx: 1,
            value: Arc::from("ALICE"),
        };

        // Row with NULL in name column
        let row = Row::from_values(vec![
            Value::Integer(1),
            Value::null(crate::core::DataType::Text),
            Value::Integer(30),
            Value::Float(95.5),
            Value::Boolean(true),
        ]);

        assert!(!filter.matches(&row));
        assert!(filter.is_unknown_due_to_null(&row));
    }

    // =========================================================================
    // AND/OR chain flattening and optimization tests
    // =========================================================================

    #[test]
    fn test_and_short_circuit_false() {
        // A AND False should simplify to False
        let filter = CompiledFilter::And(
            Box::new(CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 42,
            }),
            Box::new(CompiledFilter::False),
        );

        // When False is present in AND, the entire chain can be simplified
        let row = make_row(42, "Alice", 30, 95.5, true);
        assert!(!filter.matches(&row)); // False AND anything = False
    }

    #[test]
    fn test_or_short_circuit_true() {
        // A OR True should simplify to True
        let filter = CompiledFilter::Or(
            Box::new(CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 999, // Won't match
            }),
            Box::new(CompiledFilter::True),
        );

        let row = make_row(42, "Alice", 30, 95.5, true);
        assert!(filter.matches(&row)); // False OR True = True
    }

    #[test]
    fn test_and_with_true_constant() {
        // A AND True should be equivalent to just A
        let filter = CompiledFilter::And(
            Box::new(CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 42,
            }),
            Box::new(CompiledFilter::True),
        );

        let row1 = make_row(42, "Alice", 30, 95.5, true);
        let row2 = make_row(43, "Bob", 25, 88.0, false);

        assert!(filter.matches(&row1)); // A AND True = A (true when A is true)
        assert!(!filter.matches(&row2)); // A AND True = A (false when A is false)
    }

    #[test]
    fn test_or_with_false_constant() {
        // A OR False should be equivalent to just A
        let filter = CompiledFilter::Or(
            Box::new(CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 42,
            }),
            Box::new(CompiledFilter::False),
        );

        let row1 = make_row(42, "Alice", 30, 95.5, true);
        let row2 = make_row(43, "Bob", 25, 88.0, false);

        assert!(filter.matches(&row1)); // A OR False = A (true when A is true)
        assert!(!filter.matches(&row2)); // A OR False = A (false when A is false)
    }

    #[test]
    fn test_andn_flattened() {
        // Test that AndN properly iterates over all conditions
        let filter = CompiledFilter::AndN(vec![
            CompiledFilter::IntegerGte {
                col_idx: 0,
                value: 1,
            },
            CompiledFilter::IntegerLte {
                col_idx: 0,
                value: 10,
            },
            CompiledFilter::BooleanEq {
                col_idx: 4,
                value: true,
            },
        ]);

        let row1 = make_row(5, "Alice", 30, 95.5, true); // All conditions met
        let row2 = make_row(5, "Bob", 25, 88.0, false); // Third condition fails
        let row3 = make_row(15, "Charlie", 35, 70.0, true); // Second condition fails

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
        assert!(!filter.matches(&row3));
    }

    #[test]
    fn test_orn_flattened() {
        // Test that OrN properly iterates over all conditions
        let filter = CompiledFilter::OrN(vec![
            CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 1,
            },
            CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 5,
            },
            CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 10,
            },
        ]);

        let row1 = make_row(1, "Alice", 30, 95.5, true); // First matches
        let row2 = make_row(5, "Bob", 25, 88.0, false); // Second matches
        let row3 = make_row(10, "Charlie", 35, 70.0, true); // Third matches
        let row4 = make_row(7, "Dave", 40, 80.0, false); // None match

        assert!(filter.matches(&row1));
        assert!(filter.matches(&row2));
        assert!(filter.matches(&row3));
        assert!(!filter.matches(&row4));
    }

    #[test]
    fn test_nested_and_should_flatten() {
        // Nested AND: (A AND B) AND C should be treated equivalently to A AND B AND C
        let inner = CompiledFilter::And(
            Box::new(CompiledFilter::IntegerGte {
                col_idx: 0,
                value: 1,
            }),
            Box::new(CompiledFilter::IntegerLte {
                col_idx: 0,
                value: 10,
            }),
        );
        let filter = CompiledFilter::And(
            Box::new(inner),
            Box::new(CompiledFilter::BooleanEq {
                col_idx: 4,
                value: true,
            }),
        );

        let row1 = make_row(5, "Alice", 30, 95.5, true); // All conditions met
        let row2 = make_row(5, "Bob", 25, 88.0, false); // Outer condition fails
        let row3 = make_row(15, "Charlie", 35, 70.0, true); // Inner condition fails

        assert!(filter.matches(&row1));
        assert!(!filter.matches(&row2));
        assert!(!filter.matches(&row3));
    }

    #[test]
    fn test_nested_or_should_flatten() {
        // Nested OR: (A OR B) OR C should be treated equivalently to A OR B OR C
        let inner = CompiledFilter::Or(
            Box::new(CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 1,
            }),
            Box::new(CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 5,
            }),
        );
        let filter = CompiledFilter::Or(
            Box::new(inner),
            Box::new(CompiledFilter::IntegerEq {
                col_idx: 0,
                value: 10,
            }),
        );

        let row1 = make_row(1, "Alice", 30, 95.5, true); // Inner first matches
        let row2 = make_row(5, "Bob", 25, 88.0, false); // Inner second matches
        let row3 = make_row(10, "Charlie", 35, 70.0, true); // Outer matches
        let row4 = make_row(7, "Dave", 40, 80.0, false); // None match

        assert!(filter.matches(&row1));
        assert!(filter.matches(&row2));
        assert!(filter.matches(&row3));
        assert!(!filter.matches(&row4));
    }
}
