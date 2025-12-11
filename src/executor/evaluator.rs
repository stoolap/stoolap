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

//! Expression Evaluator
//!
//! This module provides expression evaluation for SQL queries.
//! It evaluates parser AST expressions to runtime values.

use chrono::{Datelike, TimeZone, Utc};
use regex::Regex;
use rustc_hash::FxHashMap;
use std::cell::RefCell;

use crate::core::{Error, Result, Row, Value};
use crate::functions::{global_registry, FunctionRegistry};
use crate::parser::ast::*;

use super::context::ExecutionContext;

// Thread-local cache for compiled regex patterns.
// This avoids recompiling the same regex pattern for every row evaluation.
thread_local! {
    static REGEX_CACHE: RefCell<FxHashMap<String, Regex>> = RefCell::new(FxHashMap::default());
}

/// Maximum number of cached regex patterns per thread to prevent memory bloat
const MAX_REGEX_CACHE_SIZE: usize = 100;

/// Get a cached regex or compile and cache it
fn get_or_compile_regex(pattern: &str) -> std::result::Result<Regex, regex::Error> {
    REGEX_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        // Check if already cached
        if let Some(re) = cache.get(pattern) {
            return Ok(re.clone());
        }

        // Compile the regex
        let re = Regex::new(pattern)?;

        // Evict oldest entries if cache is full (simple LRU approximation)
        if cache.len() >= MAX_REGEX_CACHE_SIZE {
            // Remove about half the entries (simple eviction strategy)
            let keys_to_remove: Vec<String> = cache
                .keys()
                .take(MAX_REGEX_CACHE_SIZE / 2)
                .cloned()
                .collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }

        cache.insert(pattern.to_string(), re.clone());
        Ok(re)
    })
}

/// Expression evaluator
///
/// Evaluates parser AST expressions to runtime values using the current
/// row context, parameters, and function registry.
///
/// OPTIMIZATION: Uses FxHashMap (faster hashing) instead of std HashMap.
/// The evaluator is designed to be reusable - call clear() between uses
/// instead of creating new instances to avoid allocation overhead.
pub struct Evaluator<'a> {
    /// Function registry for scalar function calls (borrowed, immutable)
    function_registry: &'a FunctionRegistry,
    /// Current row values (column name -> value) - RARELY USED, prefer row_slice_ptr
    row_map: FxHashMap<String, Value>,
    /// Current row as array (for indexed access) - used for ownership when needed
    row_array: Option<Row>,
    /// OPTIMIZATION: Raw pointer to row slice for zero-copy hot path
    /// This avoids cloning the row on every set_row_array call.
    /// Safety: Caller must ensure the referenced row outlives the evaluator usage.
    row_slice_ptr: *const Value,
    row_slice_len: usize,
    /// OPTIMIZATION: Second row pointer for join operations (zero-copy join evaluation)
    /// This allows evaluating join conditions without combining rows into a new allocation.
    row_slice_ptr2: *const Value,
    row_slice_len2: usize,
    /// Number of columns from the first row (for split lookup in joins)
    first_row_cols: usize,
    /// Column name to index mapping
    column_indices: FxHashMap<String, usize>,
    /// OPTIMIZATION: Pre-computed qualified identifier lookup (table -> column -> index)
    /// This avoids format!() allocation on every row evaluation
    qualified_indices: FxHashMap<String, FxHashMap<String, usize>>,
    /// Table aliases (alias -> table name)
    table_aliases: FxHashMap<String, String>,
    /// Column aliases (alias -> original column)
    column_aliases: FxHashMap<String, String>,
    /// Query parameters (positional)
    params: Vec<Value>,
    /// Query parameters (named)
    named_params: FxHashMap<String, Value>,
    /// Outer row context for correlated subqueries (column name -> value)
    outer_row: FxHashMap<String, Value>,
    /// Qualified outer row context (table_name -> (column_name -> value))
    /// Used for O(1) lookup of qualified identifiers without format!() allocation
    outer_row_qualified: FxHashMap<String, FxHashMap<String, Value>>,
    /// Expression aliases for HAVING clause (expression string -> column index)
    /// This maps expressions like "x + y" to their GROUP BY column indices
    expression_aliases: FxHashMap<String, usize>,
    /// Current transaction ID for CURRENT_TRANSACTION_ID() function
    transaction_id: Option<u64>,
}

impl<'a> Evaluator<'a> {
    /// Create a new evaluator with a function registry reference
    pub fn new(function_registry: &'a FunctionRegistry) -> Self {
        Self {
            function_registry,
            row_map: FxHashMap::default(),
            row_array: None,
            row_slice_ptr: std::ptr::null(),
            row_slice_len: 0,
            row_slice_ptr2: std::ptr::null(),
            row_slice_len2: 0,
            first_row_cols: 0,
            column_indices: FxHashMap::default(),
            qualified_indices: FxHashMap::default(),
            table_aliases: FxHashMap::default(),
            column_aliases: FxHashMap::default(),
            params: Vec::new(),
            named_params: FxHashMap::default(),
            outer_row: FxHashMap::default(),
            outer_row_qualified: FxHashMap::default(),
            expression_aliases: FxHashMap::default(),
            transaction_id: None,
        }
    }

    /// Clear all state for reuse. More efficient than creating a new Evaluator.
    pub fn clear(&mut self) {
        self.row_map.clear();
        self.row_array = None;
        self.row_slice_ptr = std::ptr::null();
        self.row_slice_len = 0;
        self.row_slice_ptr2 = std::ptr::null();
        self.row_slice_len2 = 0;
        self.first_row_cols = 0;
        self.column_indices.clear();
        self.qualified_indices.clear();
        self.table_aliases.clear();
        self.column_aliases.clear();
        self.params.clear();
        self.named_params.clear();
        self.outer_row.clear();
        self.outer_row_qualified.clear();
        self.expression_aliases.clear();
        self.transaction_id = None;
    }

    /// Set the current transaction ID for CURRENT_TRANSACTION_ID() function
    pub fn set_transaction_id(&mut self, txn_id: u64) {
        self.transaction_id = Some(txn_id);
    }

    /// Create an evaluator using the global function registry.
    /// This is efficient as it uses a static reference with no cloning.
    pub fn with_defaults() -> Evaluator<'static> {
        Evaluator::new(global_registry())
    }

    /// Set the current row from a map
    pub fn with_row_map(mut self, row_map: FxHashMap<String, Value>) -> Self {
        self.row_map = row_map;
        self
    }

    /// Set the current row from an array with column names
    ///
    /// OPTIMIZATION: Takes ownership of row instead of cloning internally.
    /// Callers pass `row.clone()` only if they need to keep the original row.
    /// The row_map is NOT populated - use get_row_slice() + column_indices instead.
    pub fn with_row(mut self, row: Row, columns: &[String]) -> Self {
        // Build column index mapping
        self.column_indices.clear();
        self.qualified_indices.clear();
        for (i, col) in columns.iter().enumerate() {
            let lower = col.to_lowercase();
            self.column_indices.insert(lower.clone(), i);

            // Pre-compute qualified identifier lookups
            if let Some(dot_idx) = col.rfind('.') {
                let table_part = col[..dot_idx].to_lowercase();
                let column_part = col[dot_idx + 1..].to_lowercase();
                self.qualified_indices
                    .entry(table_part)
                    .or_default()
                    .insert(column_part, i);
            }
        }

        // Store row by ownership (no clone - caller clones if needed)
        self.row_array = Some(row);
        // Clear raw pointer mode - we're using row_array
        self.row_slice_ptr = std::ptr::null();
        self.row_slice_len = 0;
        // row_map not populated - use get_row_slice() + column_indices instead
        self.row_map.clear();

        self
    }

    /// Set table aliases
    pub fn with_table_aliases(mut self, aliases: FxHashMap<String, String>) -> Self {
        self.table_aliases = aliases;
        self
    }

    /// Set column aliases
    pub fn with_column_aliases(mut self, aliases: FxHashMap<String, String>) -> Self {
        self.column_aliases = aliases;
        self
    }

    /// Set query parameters (positional)
    pub fn with_params(mut self, params: Vec<Value>) -> Self {
        self.params = params;
        self
    }

    /// Set named query parameters
    pub fn with_named_params(mut self, named_params: FxHashMap<String, Value>) -> Self {
        self.named_params = named_params;
        self
    }

    /// Set parameters from execution context
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        self.params = ctx.params().to_vec();
        // Convert std HashMap to FxHashMap
        self.named_params = ctx
            .named_params()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        // Set outer row context for correlated subqueries
        // Uses set_outer_row to also build the qualified index
        self.set_outer_row(ctx.outer_row());
        // Copy transaction ID for CURRENT_TRANSACTION_ID() function
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// Update the current row
    pub fn set_row(&mut self, row: &Row, columns: &[String]) {
        self.row_array = Some(row.clone());
        self.row_map.clear();
        self.column_indices.clear();

        for (i, col) in columns.iter().enumerate() {
            self.column_indices.insert(col.to_lowercase(), i);
            if let Some(value) = row.get(i) {
                self.row_map.insert(col.to_lowercase(), value.clone());
            }
        }
    }

    /// Initialize the column index mapping (call once before set_row_array)
    /// This builds the column name -> index mapping that will be reused for all rows.
    pub fn init_columns(&mut self, columns: &[String]) {
        self.column_indices.clear();
        self.qualified_indices.clear();

        for (i, col) in columns.iter().enumerate() {
            let lower = col.to_lowercase();
            self.column_indices.insert(lower.clone(), i);

            // OPTIMIZATION: Pre-compute qualified identifier lookups to avoid
            // format!() allocation on every row evaluation
            if let Some(dot_idx) = col.rfind('.') {
                let table_part = col[..dot_idx].to_lowercase();
                let column_part = col[dot_idx + 1..].to_lowercase();

                // Add to qualified_indices for O(1) lookup without allocation
                self.qualified_indices
                    .entry(table_part)
                    .or_default()
                    .insert(column_part.clone(), i);

                // Also map unqualified names for qualified columns
                // Don't overwrite if already exists
                self.column_indices.entry(column_part).or_insert(i);
            }
        }
    }

    /// Add aggregate expression aliases for HAVING clause evaluation
    /// This maps expression names like "SUM(price)" to their column indices,
    /// allowing HAVING SUM(price) > 100 to work even when the column is aliased as "total"
    pub fn add_aggregate_aliases(&mut self, aliases: &[(String, usize)]) {
        for (expr_name, idx) in aliases {
            let lower = expr_name.to_lowercase();
            // Only add if not already present (don't override actual column names)
            self.column_indices.entry(lower).or_insert(*idx);
        }
    }

    /// Add expression aliases for HAVING clause evaluation with GROUP BY expressions
    /// This maps expression strings like "x + y" to their GROUP BY column indices,
    /// allowing HAVING x + y > 20 to work when GROUP BY x + y
    pub fn add_expression_aliases(&mut self, aliases: &[(String, usize)]) {
        for (expr_str, idx) in aliases {
            let lower = expr_str.to_lowercase();
            self.expression_aliases.insert(lower, *idx);
        }
    }

    /// Set the row using array-based access (optimized - no map rebuilding)
    /// Call init_columns() once before using this method.
    /// This is the fast path for evaluating expressions over many rows.
    ///
    /// OPTIMIZATION: Uses raw pointer to avoid cloning the row on every call.
    /// This eliminates Vec allocation + Value clones per row.
    ///
    /// # Safety
    /// Caller must ensure the row reference remains valid until the next
    /// set_row_array call or until the evaluator is no longer used.
    #[inline]
    pub fn set_row_array(&mut self, row: &Row) {
        // OPTIMIZATION: Store raw pointer instead of cloning
        // This avoids allocating a new Vec<Value> for every row
        let slice = row.as_slice();
        self.row_slice_ptr = slice.as_ptr();
        self.row_slice_len = slice.len();
        // Clear join mode
        self.row_slice_ptr2 = std::ptr::null();
        self.row_slice_len2 = 0;
        self.first_row_cols = 0;
        // Clear row_array - we're using the raw pointer now
        self.row_array = None;
    }

    /// Set two rows for join condition evaluation (zero-copy)
    ///
    /// This allows evaluating join conditions like `a.id = b.id` without
    /// allocating a combined row. The evaluator treats indices 0..left_len
    /// as coming from left_row, and indices left_len..total as from right_row.
    ///
    /// # Safety
    /// Caller must ensure both row references remain valid until the next
    /// set_row_* call or until the evaluator is no longer used.
    #[inline]
    pub fn set_join_rows(&mut self, left_row: &Row, right_row: &Row) {
        let left_slice = left_row.as_slice();
        let right_slice = right_row.as_slice();

        self.row_slice_ptr = left_slice.as_ptr();
        self.row_slice_len = left_slice.len();
        self.row_slice_ptr2 = right_slice.as_ptr();
        self.row_slice_len2 = right_slice.len();
        self.first_row_cols = left_slice.len();
        self.row_array = None;
    }

    /// Set the outer row context for correlated subqueries.
    /// This is used to pass values from the outer query to the inner subquery.
    /// Also builds a qualified index for O(1) lookup of "table.column" references.
    #[inline]
    pub fn set_outer_row(&mut self, outer_row: Option<&rustc_hash::FxHashMap<String, Value>>) {
        self.outer_row.clear();
        self.outer_row_qualified.clear();
        if let Some(outer) = outer_row {
            for (k, v) in outer.iter() {
                self.outer_row.insert(k.clone(), v.clone());
                // Build qualified index for keys like "table.column"
                if let Some(dot_pos) = k.find('.') {
                    let table = &k[..dot_pos];
                    let column = &k[dot_pos + 1..];
                    self.outer_row_qualified
                        .entry(table.to_string())
                        .or_default()
                        .insert(column.to_string(), v.clone());
                }
            }
        }
    }

    /// Set the outer row context by taking ownership (zero-copy for the flat map).
    /// Also builds a qualified index for O(1) lookup of "table.column" references.
    #[inline]
    pub fn set_outer_row_owned(&mut self, outer_row: FxHashMap<String, Value>) {
        self.outer_row_qualified.clear();
        // Build qualified index before taking ownership
        for (k, v) in outer_row.iter() {
            if let Some(dot_pos) = k.find('.') {
                let table = &k[..dot_pos];
                let column = &k[dot_pos + 1..];
                self.outer_row_qualified
                    .entry(table.to_string())
                    .or_default()
                    .insert(column.to_string(), v.clone());
            }
        }
        self.outer_row = outer_row;
    }

    /// Take ownership of the outer row back (for reuse).
    #[inline]
    pub fn take_outer_row(&mut self) -> FxHashMap<String, Value> {
        self.outer_row_qualified.clear();
        std::mem::take(&mut self.outer_row)
    }

    /// Clear the outer row context
    #[inline]
    pub fn clear_outer_row(&mut self) {
        self.outer_row.clear();
        self.outer_row_qualified.clear();
    }

    /// Get the current row slice (zero-copy access)
    /// For join mode, returns a virtual combined slice view.
    #[inline]
    fn get_row_slice(&self) -> Option<&[Value]> {
        // Non-join mode: single row pointer
        if self.row_slice_ptr2.is_null() {
            if self.row_slice_len > 0 && !self.row_slice_ptr.is_null() {
                // SAFETY: Caller of set_row_array must ensure the row is still valid
                Some(unsafe { std::slice::from_raw_parts(self.row_slice_ptr, self.row_slice_len) })
            } else if let Some(ref row) = self.row_array {
                Some(row.as_slice())
            } else {
                None
            }
        } else {
            // Join mode: return first row slice (for iteration)
            // Individual value access uses get_value_at_index instead
            if !self.row_slice_ptr.is_null() {
                Some(unsafe { std::slice::from_raw_parts(self.row_slice_ptr, self.row_slice_len) })
            } else {
                None
            }
        }
    }

    /// Get a value at a specific index (handles join mode)
    #[inline]
    fn get_value_at_index(&self, idx: usize) -> Option<&Value> {
        // Check if we're in join mode (two rows set)
        if !self.row_slice_ptr2.is_null() && self.first_row_cols > 0 {
            if idx < self.first_row_cols {
                // Index is in the left row
                if idx < self.row_slice_len && !self.row_slice_ptr.is_null() {
                    return Some(unsafe { &*self.row_slice_ptr.add(idx) });
                }
            } else {
                // Index is in the right row (offset by first_row_cols)
                let right_idx = idx - self.first_row_cols;
                if right_idx < self.row_slice_len2 {
                    return Some(unsafe { &*self.row_slice_ptr2.add(right_idx) });
                }
            }
            return None;
        }

        // Non-join mode: single row
        if let Some(slice) = self.get_row_slice() {
            slice.get(idx)
        } else {
            None
        }
    }

    /// Clear the current row
    pub fn clear_row(&mut self) {
        self.row_array = None;
        self.row_slice_ptr = std::ptr::null();
        self.row_slice_len = 0;
        self.row_slice_ptr2 = std::ptr::null();
        self.row_slice_len2 = 0;
        self.first_row_cols = 0;
        self.row_map.clear();
        self.column_indices.clear();
    }

    /// Evaluate an expression to a Value
    pub fn evaluate(&self, expr: &Expression) -> Result<Value> {
        match expr {
            Expression::Identifier(id) => self.evaluate_identifier(id),
            Expression::QualifiedIdentifier(qid) => self.evaluate_qualified_identifier(qid),
            Expression::IntegerLiteral(lit) => Ok(Value::Integer(lit.value)),
            Expression::FloatLiteral(lit) => Ok(Value::Float(lit.value)),
            Expression::StringLiteral(lit) => {
                // Check for type hints like TIMESTAMP '2025-01-01', DATE '2025-01-01', TIME '12:00:00'
                if let Some(ref hint) = lit.type_hint {
                    match hint.to_uppercase().as_str() {
                        "TIMESTAMP" | "DATETIME" => {
                            // Parse as timestamp
                            crate::core::value::parse_timestamp(&lit.value)
                                .map(Value::Timestamp)
                                .map_err(|e| {
                                    Error::Type(format!(
                                        "Invalid TIMESTAMP literal '{}': {}",
                                        lit.value, e
                                    ))
                                })
                        }
                        "DATE" => {
                            // Parse as date (store as timestamp at midnight)
                            crate::core::value::parse_timestamp(&lit.value)
                                .map(Value::Timestamp)
                                .map_err(|e| {
                                    Error::Type(format!(
                                        "Invalid DATE literal '{}': {}",
                                        lit.value, e
                                    ))
                                })
                        }
                        "TIME" => {
                            // Parse as time (store as text for now)
                            Ok(Value::Text(std::sync::Arc::from(lit.value.as_str())))
                        }
                        _ => Ok(Value::Text(std::sync::Arc::from(lit.value.as_str()))),
                    }
                } else {
                    Ok(Value::Text(std::sync::Arc::from(lit.value.as_str())))
                }
            }
            Expression::BooleanLiteral(lit) => Ok(Value::Boolean(lit.value)),
            Expression::NullLiteral(_) => Ok(Value::null_unknown()),
            Expression::IntervalLiteral(interval) => self.evaluate_interval(interval),
            Expression::Parameter(param) => self.evaluate_parameter(param),
            Expression::Infix(infix) => {
                // Check if this infix expression matches a GROUP BY expression alias
                // This allows HAVING x + y > 20 to work when GROUP BY x + y
                if !self.expression_aliases.is_empty() {
                    let expr_str = Self::expression_to_string(expr).to_lowercase();
                    if let Some(&idx) = self.expression_aliases.get(&expr_str) {
                        // Expression matches a GROUP BY alias, return the column value
                        if let Some(val) = self.get_value_at_index(idx) {
                            return Ok(val.clone());
                        }
                    }
                }
                self.evaluate_infix(infix)
            }
            Expression::Prefix(prefix) => self.evaluate_prefix(prefix),
            Expression::List(list) => self.evaluate_list(list),
            Expression::Distinct(distinct) => self.evaluate(&distinct.expr),
            Expression::Exists(_) => Err(Error::NotSupportedMessage(
                "EXISTS subquery should be evaluated in subquery context".to_string(),
            )),
            Expression::AllAny(_) => Err(Error::NotSupportedMessage(
                "ALL/ANY subquery should be evaluated in subquery context".to_string(),
            )),
            Expression::In(in_expr) => self.evaluate_in(in_expr),
            Expression::Between(between) => self.evaluate_between(between),
            Expression::Like(like) => self.evaluate_like_expression(like),
            Expression::ScalarSubquery(_) => Err(Error::NotSupportedMessage(
                "Scalar subquery not yet implemented".to_string(),
            )),
            Expression::ExpressionList(list) => {
                // Evaluate first expression in the list as the result
                if let Some(first) = list.expressions.first() {
                    self.evaluate(first)
                } else {
                    Ok(Value::null_unknown())
                }
            }
            Expression::Case(case) => self.evaluate_case(case),
            Expression::Cast(cast) => self.evaluate_cast(cast),
            Expression::FunctionCall(func) => {
                // Check if this is an aggregate function reference in post-aggregation context
                // Aggregates like SUM(v) become column names like "SUM(v)" in aggregated results
                if crate::functions::registry::global_registry().is_aggregate(&func.function) {
                    // Try to look up the aggregate result as a column
                    let col_name = self.format_aggregate_column_name(func);
                    let col_lower = col_name.to_lowercase();
                    if let Some(&idx) = self.column_indices.get(&col_lower) {
                        // SAFETY: We use raw pointer access for performance
                        // Caller of set_row_array must ensure the row is valid
                        if !self.row_slice_ptr.is_null() && idx < self.row_slice_len {
                            let row = unsafe {
                                std::slice::from_raw_parts(self.row_slice_ptr, self.row_slice_len)
                            };
                            return Ok(row[idx].clone());
                        }
                    }
                    // Fall through to normal function evaluation if not found as column
                }
                self.evaluate_function_call(func)
            }
            Expression::Aliased(aliased) => self.evaluate(&aliased.expression),
            Expression::Window(_) => Err(Error::NotSupportedMessage(
                "Window expression in evaluator not yet implemented".to_string(),
            )),
            Expression::TableSource(_) => Err(Error::InvalidArgumentMessage(
                "TableSource cannot be evaluated as a value".to_string(),
            )),
            Expression::JoinSource(_) => Err(Error::InvalidArgumentMessage(
                "JoinSource cannot be evaluated as a value".to_string(),
            )),
            Expression::SubquerySource(_) => Err(Error::InvalidArgumentMessage(
                "SubquerySource cannot be evaluated as a value".to_string(),
            )),
            Expression::ValuesSource(_) => Err(Error::InvalidArgumentMessage(
                "ValuesSource cannot be evaluated as a value".to_string(),
            )),
            Expression::CteReference(_) => Err(Error::InvalidArgumentMessage(
                "CteReference cannot be evaluated as a value".to_string(),
            )),
            Expression::Star(_) => Err(Error::InvalidArgumentMessage(
                "Star (*) cannot be evaluated as a value".to_string(),
            )),
            Expression::QualifiedStar(_) => Err(Error::InvalidArgumentMessage(
                "Qualified star (table.*) cannot be evaluated as a value".to_string(),
            )),
            Expression::Default(_) => Err(Error::InvalidArgumentMessage(
                "DEFAULT keyword must be handled in INSERT context".to_string(),
            )),
        }
    }

    /// Evaluate an expression to a boolean (for WHERE clauses)
    pub fn evaluate_bool(&self, expr: &Expression) -> Result<bool> {
        let value = self.evaluate(expr)?;
        match value {
            Value::Boolean(b) => Ok(b),
            Value::Null(_) => Ok(false), // NULL is falsy in WHERE
            Value::Integer(i) => Ok(i != 0),
            _ => Err(Error::Type(format!(
                "Expected boolean expression, got {:?}",
                value
            ))),
        }
    }

    /// Evaluate an identifier
    fn evaluate_identifier(&self, id: &Identifier) -> Result<Value> {
        // Use pre-computed lowercase value (no allocation per row!)
        let name = &id.value_lower;

        // Handle SQL-standard date/time keywords
        // These are special identifiers that act like parameterless functions
        match name.as_str() {
            "current_date" => {
                // Return timestamp at midnight UTC (consistent with CurrentDateFunction)
                let now = Utc::now();
                let date = Utc
                    .with_ymd_and_hms(now.year(), now.month(), now.day(), 0, 0, 0)
                    .single()
                    .unwrap_or(now);
                return Ok(Value::Timestamp(date));
            }
            "current_time" => {
                let now = Utc::now();
                return Ok(Value::from(now.format("%H:%M:%S").to_string()));
            }
            "current_timestamp" => {
                return Ok(Value::Timestamp(Utc::now()));
            }
            _ => {}
        }

        // Fast path: use direct index access (works for both single row and join mode)
        // Check column aliases first
        // OPTIMIZATION: column_aliases values are already lowercase (set via with_column_aliases)
        if let Some(original) = self.column_aliases.get(name) {
            if let Some(&idx) = self.column_indices.get(original) {
                if let Some(value) = self.get_value_at_index(idx) {
                    return Ok(value.clone());
                }
            }
        }

        // Direct column lookup via index (handles join mode transparently)
        if let Some(&idx) = self.column_indices.get(name) {
            if let Some(value) = self.get_value_at_index(idx) {
                return Ok(value.clone());
            }
        }

        // Fallback: check row_map (for cases where row_array isn't used)
        // OPTIMIZATION: column_aliases values are already lowercase
        if let Some(original) = self.column_aliases.get(name) {
            if let Some(value) = self.row_map.get(original) {
                return Ok(value.clone());
            }
        }

        if let Some(value) = self.row_map.get(name) {
            return Ok(value.clone());
        }

        // Check outer row context for correlated subqueries
        if let Some(value) = self.outer_row.get(name) {
            return Ok(value.clone());
        }

        // Return NULL for missing columns (common in outer joins)
        Ok(Value::null_unknown())
    }

    /// Evaluate a qualified identifier (table.column)
    fn evaluate_qualified_identifier(&self, qid: &QualifiedIdentifier) -> Result<Value> {
        // Use pre-computed lowercase values (no allocation per row!)
        let table_name = &qid.qualifier.value_lower;
        let column_name = &qid.name.value_lower;

        // For correlated subqueries, check outer_row_qualified FIRST for qualified identifiers
        // Uses nested map to avoid format!() allocation on each lookup
        if let Some(table_map) = self.outer_row_qualified.get(table_name) {
            if let Some(value) = table_map.get(column_name) {
                return Ok(value.clone());
            }
        }

        // Fast path: use direct index access (works for both single row and join mode)
        // OPTIMIZATION: Use pre-computed qualified_indices for O(1) lookup
        // This avoids format!() heap allocation on every row evaluation
        if let Some(table_columns) = self.qualified_indices.get(table_name) {
            if let Some(&idx) = table_columns.get(column_name) {
                if let Some(value) = self.get_value_at_index(idx) {
                    return Ok(value.clone());
                }
            }
        }

        // Also try the full qualified key in column_indices (for "table.column" format)
        // Uses pre-computed key from init_columns, no allocation needed
        if let Some(&idx) = self.column_indices.get(column_name) {
            if let Some(value) = self.get_value_at_index(idx) {
                return Ok(value.clone());
            }
        }

        // Fallback: check row_map (used when row_array not set)
        // Try column name directly in row_map
        if let Some(value) = self.row_map.get(column_name) {
            return Ok(value.clone());
        }

        // Fallback: check outer_row by column name only (if not already found by qualified name)
        if let Some(value) = self.outer_row.get(column_name) {
            return Ok(value.clone());
        }

        // Return NULL for missing columns
        Ok(Value::null_unknown())
    }

    /// Evaluate a parameter
    fn evaluate_parameter(&self, param: &Parameter) -> Result<Value> {
        // Check if it's a named parameter (starts with :)
        if param.name.starts_with(':') {
            // Extract name without the leading ':'
            let name = &param.name[1..];
            if let Some(value) = self.named_params.get(name) {
                Ok(value.clone())
            } else {
                Err(Error::InvalidArgumentMessage(format!(
                    "Named parameter '{}' not found",
                    param.name
                )))
            }
        } else if param.index > 0 && param.index <= self.params.len() {
            // Positional parameter ($1, $2, ?, etc.)
            Ok(self.params[param.index - 1].clone())
        } else {
            Err(Error::InvalidArgumentMessage(format!(
                "Parameter index {} out of range",
                param.index
            )))
        }
    }

    /// Evaluate an infix expression
    /// Uses pre-computed operator enum for zero-allocation, fast dispatch
    #[inline]
    fn evaluate_infix(&self, infix: &InfixExpression) -> Result<Value> {
        use crate::parser::ast::InfixOperator;

        // Handle AND/OR with short-circuit evaluation FIRST
        // This prevents errors from being raised on the right side when
        // the left side already determines the result
        match infix.op_type {
            InfixOperator::And => {
                // SQL AND with short-circuit evaluation:
                // FALSE AND <anything> = FALSE (don't evaluate right side)
                // TRUE AND <right> = evaluate right side
                // NULL AND <right> = need to evaluate right to check for FALSE
                let left = self.evaluate(&infix.left)?;
                let l = self.value_to_tribool(&left)?;

                // Short-circuit: FALSE AND anything = FALSE
                if l == Some(false) {
                    return Ok(Value::Boolean(false));
                }

                // Need to evaluate right side
                let right = self.evaluate(&infix.right)?;
                let r = self.value_to_tribool(&right)?;

                return match (l, r) {
                    (Some(false), _) | (_, Some(false)) => Ok(Value::Boolean(false)),
                    (Some(true), Some(true)) => Ok(Value::Boolean(true)),
                    _ => Ok(Value::null_unknown()), // At least one NULL, neither is FALSE
                };
            }
            InfixOperator::Or => {
                // SQL OR with short-circuit evaluation:
                // TRUE OR <anything> = TRUE (don't evaluate right side)
                // FALSE OR <right> = evaluate right side
                // NULL OR <right> = need to evaluate right to check for TRUE
                let left = self.evaluate(&infix.left)?;
                let l = self.value_to_tribool(&left)?;

                // Short-circuit: TRUE OR anything = TRUE
                if l == Some(true) {
                    return Ok(Value::Boolean(true));
                }

                // Need to evaluate right side
                let right = self.evaluate(&infix.right)?;
                let r = self.value_to_tribool(&right)?;

                return match (l, r) {
                    (Some(true), _) | (_, Some(true)) => Ok(Value::Boolean(true)),
                    (Some(false), Some(false)) => Ok(Value::Boolean(false)),
                    _ => Ok(Value::null_unknown()), // At least one NULL, neither is TRUE
                };
            }
            _ => {}
        }

        // For all other operators, evaluate both sides first
        let left = self.evaluate(&infix.left)?;
        let right = self.evaluate(&infix.right)?;

        // Fast enum-based dispatch (no string allocation or comparison!)
        match infix.op_type {
            // Comparison operators
            InfixOperator::Equal => {
                self.compare_values(&left, &right, |cmp| cmp == std::cmp::Ordering::Equal)
            }
            InfixOperator::NotEqual => {
                self.compare_values(&left, &right, |cmp| cmp != std::cmp::Ordering::Equal)
            }
            InfixOperator::LessThan => {
                self.compare_values(&left, &right, |cmp| cmp == std::cmp::Ordering::Less)
            }
            InfixOperator::LessEqual => {
                self.compare_values(&left, &right, |cmp| cmp != std::cmp::Ordering::Greater)
            }
            InfixOperator::GreaterThan => {
                self.compare_values(&left, &right, |cmp| cmp == std::cmp::Ordering::Greater)
            }
            InfixOperator::GreaterEqual => {
                self.compare_values(&left, &right, |cmp| cmp != std::cmp::Ordering::Less)
            }

            // AND/OR already handled above with short-circuit
            InfixOperator::And | InfixOperator::Or => {
                unreachable!("AND/OR should be handled above with short-circuit")
            }

            InfixOperator::Xor => {
                // XOR with NULL always returns NULL (no short-circuit possible)
                let l = self.value_to_tribool(&left)?;
                let r = self.value_to_tribool(&right)?;
                match (l, r) {
                    (Some(l), Some(r)) => Ok(Value::Boolean(l ^ r)),
                    _ => Ok(Value::null_unknown()),
                }
            }

            // Arithmetic operators (with overflow checking for integers)
            InfixOperator::Add => {
                // Handle timestamp + interval (text)
                if let (Value::Timestamp(ts), Value::Text(interval_str)) = (&left, &right) {
                    return self.add_interval_to_timestamp(*ts, interval_str, true);
                }
                if let (Value::Text(interval_str), Value::Timestamp(ts)) = (&left, &right) {
                    return self.add_interval_to_timestamp(*ts, interval_str, true);
                }
                // Handle timestamp + integer (days)
                if let (Value::Timestamp(ts), Value::Integer(days)) = (&left, &right) {
                    let duration = chrono::Duration::days(*days);
                    return Ok(Value::Timestamp(*ts + duration));
                }
                if let (Value::Integer(days), Value::Timestamp(ts)) = (&left, &right) {
                    let duration = chrono::Duration::days(*days);
                    return Ok(Value::Timestamp(*ts + duration));
                }
                self.arithmetic_op(&left, &right, |a, b| a.checked_add(b), |a, b| a + b)
            }
            InfixOperator::Subtract => {
                // Handle timestamp - interval (text)
                if let (Value::Timestamp(ts), Value::Text(interval_str)) = (&left, &right) {
                    return self.add_interval_to_timestamp(*ts, interval_str, false);
                }
                // Handle timestamp - integer (days)
                if let (Value::Timestamp(ts), Value::Integer(days)) = (&left, &right) {
                    let duration = chrono::Duration::days(*days);
                    return Ok(Value::Timestamp(*ts - duration));
                }
                // Handle timestamp - timestamp (returns interval as text)
                if let (Value::Timestamp(ts1), Value::Timestamp(ts2)) = (&left, &right) {
                    let duration = ts1.signed_duration_since(*ts2);
                    return Ok(Value::text(self.format_duration_as_interval(duration)));
                }
                self.arithmetic_op(&left, &right, |a, b| a.checked_sub(b), |a, b| a - b)
            }
            InfixOperator::Multiply => {
                self.arithmetic_op(&left, &right, |a, b| a.checked_mul(b), |a, b| a * b)
            }
            InfixOperator::Divide => {
                // Check for division by zero
                match &right {
                    Value::Integer(0) => return Err(Error::DivisionByZero),
                    Value::Float(f) if *f == 0.0 => return Err(Error::DivisionByZero),
                    _ => {}
                }
                self.arithmetic_op(&left, &right, |a, b| a.checked_div(b), |a, b| a / b)
            }
            InfixOperator::Modulo => {
                // Modulo - NULL propagation: any operation with NULL returns NULL
                if left.is_null() || right.is_null() {
                    return Ok(Value::null_unknown());
                }
                match (&left, &right) {
                    (Value::Integer(l), Value::Integer(r)) if *r != 0 => Ok(Value::Integer(l % r)),
                    (Value::Integer(_), Value::Integer(0)) => Err(Error::DivisionByZero),
                    _ => Err(Error::Type("Modulo requires integer operands".to_string())),
                }
            }

            // String operators
            InfixOperator::Concat => {
                // SQL standard: || with NULL returns NULL
                if left.is_null() || right.is_null() {
                    return Ok(Value::null_unknown());
                }
                // OPTIMIZATION: Avoid double allocation by building directly into String
                let l = left.to_string();
                let r = right.to_string();
                let mut result = String::with_capacity(l.len() + r.len());
                result.push_str(&l);
                result.push_str(&r);
                Ok(Value::Text(std::sync::Arc::from(result)))
            }

            // LIKE operator
            InfixOperator::Like => self.evaluate_like(&left, &right, false),
            InfixOperator::ILike => self.evaluate_like(&left, &right, true),
            InfixOperator::NotLike => {
                let result = self.evaluate_like(&left, &right, false)?;
                match result {
                    Value::Boolean(b) => Ok(Value::Boolean(!b)),
                    Value::Null(_) => Ok(result), // Propagate NULL
                    _ => Ok(Value::Boolean(false)),
                }
            }
            InfixOperator::NotILike => {
                let result = self.evaluate_like(&left, &right, true)?;
                match result {
                    Value::Boolean(b) => Ok(Value::Boolean(!b)),
                    Value::Null(_) => Ok(result), // Propagate NULL
                    _ => Ok(Value::Boolean(false)),
                }
            }

            // GLOB operator (shell-style wildcards: * ? [...])
            InfixOperator::Glob => self.evaluate_glob(&left, &right),
            InfixOperator::NotGlob => {
                let result = self.evaluate_glob(&left, &right)?;
                match result {
                    Value::Boolean(b) => Ok(Value::Boolean(!b)),
                    Value::Null(_) => Ok(result), // Propagate NULL
                    _ => Ok(Value::Boolean(false)),
                }
            }

            // REGEXP operator (regex matching)
            InfixOperator::Regexp => self.evaluate_regexp(&left, &right),
            InfixOperator::NotRegexp => {
                let result = self.evaluate_regexp(&left, &right)?;
                match result {
                    Value::Boolean(b) => Ok(Value::Boolean(!b)),
                    Value::Null(_) => Ok(result), // Propagate NULL
                    _ => Ok(Value::Boolean(false)),
                }
            }

            // IS operator (handled specially for NULL)
            InfixOperator::Is => {
                if matches!(right, Value::Null(_)) {
                    Ok(Value::Boolean(matches!(left, Value::Null(_))))
                } else {
                    Ok(Value::Boolean(left == right))
                }
            }

            // IS NOT operator (handled specially for NULL)
            InfixOperator::IsNot => {
                if matches!(right, Value::Null(_)) {
                    Ok(Value::Boolean(!matches!(left, Value::Null(_))))
                } else {
                    Ok(Value::Boolean(left != right))
                }
            }

            // IS DISTINCT FROM - NULL-safe not equal
            // Returns TRUE if values are different (treating NULL as a regular value)
            // NULL IS DISTINCT FROM NULL -> FALSE
            // NULL IS DISTINCT FROM 1 -> TRUE
            // 1 IS DISTINCT FROM NULL -> TRUE
            // 1 IS DISTINCT FROM 1 -> FALSE
            // 1 IS DISTINCT FROM 2 -> TRUE
            InfixOperator::IsDistinctFrom => {
                let is_distinct = match (&left, &right) {
                    (Value::Null(_), Value::Null(_)) => false, // NULL and NULL are considered same
                    (Value::Null(_), _) | (_, Value::Null(_)) => true, // NULL and non-NULL are different
                    _ => left != right, // Normal comparison for non-NULL values
                };
                Ok(Value::Boolean(is_distinct))
            }

            // IS NOT DISTINCT FROM - NULL-safe equal
            // Returns TRUE if values are the same (treating NULL as a regular value)
            // NULL IS NOT DISTINCT FROM NULL -> TRUE
            // NULL IS NOT DISTINCT FROM 1 -> FALSE
            // 1 IS NOT DISTINCT FROM NULL -> FALSE
            // 1 IS NOT DISTINCT FROM 1 -> TRUE
            // 1 IS NOT DISTINCT FROM 2 -> FALSE
            InfixOperator::IsNotDistinctFrom => {
                let is_same = match (&left, &right) {
                    (Value::Null(_), Value::Null(_)) => true, // NULL and NULL are considered same
                    (Value::Null(_), _) | (_, Value::Null(_)) => false, // NULL and non-NULL are different
                    _ => left == right, // Normal comparison for non-NULL values
                };
                Ok(Value::Boolean(is_same))
            }

            // JSON operators -> and ->>
            InfixOperator::JsonAccess => {
                // -> returns JSON
                self.evaluate_json_access(&left, &right, false)
            }
            InfixOperator::JsonAccessText => {
                // ->> returns TEXT
                self.evaluate_json_access(&left, &right, true)
            }

            // Bitwise operators
            InfixOperator::BitwiseAnd => {
                if left.is_null() || right.is_null() {
                    return Ok(Value::null_unknown());
                }
                match (&left, &right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l & r)),
                    _ => Err(Error::Type(
                        "Bitwise AND requires integer operands".to_string(),
                    )),
                }
            }
            InfixOperator::BitwiseOr => {
                if left.is_null() || right.is_null() {
                    return Ok(Value::null_unknown());
                }
                match (&left, &right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l | r)),
                    _ => Err(Error::Type(
                        "Bitwise OR requires integer operands".to_string(),
                    )),
                }
            }
            InfixOperator::BitwiseXor => {
                if left.is_null() || right.is_null() {
                    return Ok(Value::null_unknown());
                }
                match (&left, &right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l ^ r)),
                    _ => Err(Error::Type(
                        "Bitwise XOR requires integer operands".to_string(),
                    )),
                }
            }
            InfixOperator::LeftShift => {
                if left.is_null() || right.is_null() {
                    return Ok(Value::null_unknown());
                }
                match (&left, &right) {
                    (Value::Integer(l), Value::Integer(r)) => {
                        if *r < 0 || *r > 63 {
                            return Err(Error::Type(format!(
                                "Shift amount must be between 0 and 63, got {}",
                                r
                            )));
                        }
                        Ok(Value::Integer(l << r))
                    }
                    _ => Err(Error::Type(
                        "Left shift requires integer operands".to_string(),
                    )),
                }
            }
            InfixOperator::RightShift => {
                if left.is_null() || right.is_null() {
                    return Ok(Value::null_unknown());
                }
                match (&left, &right) {
                    (Value::Integer(l), Value::Integer(r)) => {
                        if *r < 0 || *r > 63 {
                            return Err(Error::Type(format!(
                                "Shift amount must be between 0 and 63, got {}",
                                r
                            )));
                        }
                        Ok(Value::Integer(l >> r))
                    }
                    _ => Err(Error::Type(
                        "Right shift requires integer operands".to_string(),
                    )),
                }
            }

            // Array index (fallback to string for error message)
            InfixOperator::Index | InfixOperator::Other => Err(Error::NotSupportedMessage(
                format!("Unsupported infix operator: {}", infix.operator),
            )),
        }
    }

    /// Evaluate a prefix expression
    /// Uses pre-computed operator enum for zero-allocation, fast dispatch
    #[inline]
    fn evaluate_prefix(&self, prefix: &PrefixExpression) -> Result<Value> {
        use crate::parser::ast::PrefixOperator;

        let operand = self.evaluate(&prefix.right)?;

        // Fast enum-based dispatch (no string allocation or comparison!)
        match prefix.op_type {
            PrefixOperator::Negate => match operand {
                Value::Integer(i) => {
                    // Use checked_neg to handle overflow (e.g., -MIN_VALUE)
                    match i.checked_neg() {
                        Some(result) => Ok(Value::Integer(result)),
                        None => Err(Error::Type(format!(
                            "Integer overflow when negating: {}",
                            i
                        ))),
                    }
                }
                Value::Float(f) => Ok(Value::Float(-f)),
                Value::Null(_) => Ok(Value::null_unknown()),
                _ => Err(Error::Type("Cannot negate non-numeric value".to_string())),
            },
            PrefixOperator::Not => {
                // NOT NULL = NULL (three-valued logic)
                match self.value_to_tribool(&operand)? {
                    Some(b) => Ok(Value::Boolean(!b)),
                    None => Ok(Value::null_unknown()),
                }
            }
            PrefixOperator::Plus => Ok(operand), // Unary plus is a no-op
            PrefixOperator::BitwiseNot => {
                // Bitwise NOT (~) - only for integers
                match operand {
                    Value::Integer(i) => Ok(Value::Integer(!i)),
                    Value::Null(_) => Ok(Value::null_unknown()),
                    _ => Err(Error::Type(
                        "Bitwise NOT requires integer operand".to_string(),
                    )),
                }
            }
            PrefixOperator::Other => Err(Error::NotSupportedMessage(format!(
                "Unsupported prefix operator: {}",
                prefix.operator
            ))),
        }
    }

    /// Evaluate a list expression
    fn evaluate_list(&self, list: &ListExpression) -> Result<Value> {
        // For now, return the first element or NULL
        if let Some(first) = list.elements.first() {
            self.evaluate(first)
        } else {
            Ok(Value::null_unknown())
        }
    }

    /// Evaluate an IN expression
    ///
    /// SQL Standard three-valued logic for IN/NOT IN with NULL:
    /// - `x IN (a, b, NULL)`: TRUE if x = a OR x = b, UNKNOWN otherwise (not FALSE)
    /// - `x NOT IN (a, b, NULL)`: FALSE if x = a OR x = b, UNKNOWN otherwise (not TRUE)
    ///
    /// This is because NULL comparisons yield UNKNOWN, so:
    /// - `10 IN (20, NULL)` = `10=20 OR 10=NULL` = `FALSE OR UNKNOWN` = `UNKNOWN`
    /// - `10 NOT IN (20, NULL)` = `NOT(10=20 OR 10=NULL)` = `NOT(FALSE OR UNKNOWN)` = `NOT UNKNOWN` = `UNKNOWN`
    fn evaluate_in(&self, in_expr: &InExpression) -> Result<Value> {
        // Check if left side is a tuple (multi-column IN)
        if let Expression::ExpressionList(left_tuple) = in_expr.left.as_ref() {
            // Multi-column IN: (a, b) IN ((1, 2), (3, 4))
            return self.evaluate_tuple_in(left_tuple, &in_expr.right, in_expr.not);
        }

        let left = self.evaluate(&in_expr.left)?;

        // If left side is NULL, result is always NULL (UNKNOWN)
        if left.is_null() {
            return Ok(Value::null_unknown());
        }

        // Right side should be a list of values
        match in_expr.right.as_ref() {
            Expression::List(list) => self.evaluate_in_list(&left, &list.elements, in_expr.not),
            Expression::ExpressionList(list) => {
                self.evaluate_in_list(&left, &list.expressions, in_expr.not)
            }
            _ => {
                // Single value comparison
                let right = self.evaluate(&in_expr.right)?;
                if right.is_null() {
                    // x = NULL is UNKNOWN
                    return Ok(Value::null_unknown());
                }
                let matches = left == right;
                Ok(Value::Boolean(if in_expr.not { !matches } else { matches }))
            }
        }
    }

    /// Helper for evaluate_in - handles list with proper NULL semantics
    fn evaluate_in_list(&self, left: &Value, items: &[Expression], not: bool) -> Result<Value> {
        let mut has_null = false;

        for item in items {
            let item_val = self.evaluate(item)?;
            if item_val.is_null() {
                // Track that we have a NULL in the list
                has_null = true;
            } else if *left == item_val {
                // Found a match
                return Ok(Value::Boolean(!not));
            }
        }

        // No exact match found
        if has_null {
            // With NULL in list: result is UNKNOWN (NULL)
            // - For IN: FALSE OR UNKNOWN = UNKNOWN
            // - For NOT IN: NOT(FALSE OR UNKNOWN) = NOT UNKNOWN = UNKNOWN
            Ok(Value::null_unknown())
        } else {
            // No NULL in list: definitive answer
            Ok(Value::Boolean(not))
        }
    }

    /// Evaluate a multi-column IN expression: (a, b) IN ((1, 2), (3, 4))
    ///
    /// Follows SQL standard three-valued logic for NULL handling:
    /// - If any left tuple element is NULL, result is UNKNOWN
    /// - If any right tuple contains NULL, that comparison is UNKNOWN
    fn evaluate_tuple_in(
        &self,
        left_tuple: &ExpressionList,
        right: &Expression,
        not: bool,
    ) -> Result<Value> {
        // Evaluate all elements of the left tuple
        let left_values: Vec<Value> = left_tuple
            .expressions
            .iter()
            .map(|e| self.evaluate(e))
            .collect::<Result<Vec<_>>>()?;

        // If any left value is NULL, result is UNKNOWN
        if left_values.iter().any(|v| v.is_null()) {
            return Ok(Value::null_unknown());
        }

        // Right side should be a list of tuples
        let items = match right {
            Expression::ExpressionList(list) => &list.expressions,
            Expression::List(list) => &list.elements,
            _ => {
                return Err(Error::Type(
                    "Multi-column IN requires a list of tuples".to_string(),
                ))
            }
        };

        let mut has_null_comparison = false;

        for item in items {
            // Each item should be a tuple (ExpressionList)
            if let Expression::ExpressionList(right_tuple) = item {
                if right_tuple.expressions.len() != left_values.len() {
                    continue; // Skip tuples with different length
                }

                // Compare element by element
                let mut all_match = true;
                let mut has_null_in_tuple = false;

                for (i, right_expr) in right_tuple.expressions.iter().enumerate() {
                    let right_val = self.evaluate(right_expr)?;
                    if right_val.is_null() {
                        // This tuple has a NULL, comparison is UNKNOWN
                        has_null_in_tuple = true;
                        all_match = false;
                    } else if left_values[i] != right_val {
                        all_match = false;
                        // No need to continue if we found a definite mismatch
                        if !has_null_in_tuple {
                            break;
                        }
                    }
                }

                if all_match {
                    // Found exact match
                    return Ok(Value::Boolean(!not));
                }

                if has_null_in_tuple {
                    has_null_comparison = true;
                }
            }
        }

        // No exact match found
        if has_null_comparison {
            Ok(Value::null_unknown())
        } else {
            Ok(Value::Boolean(not))
        }
    }

    /// Evaluate a BETWEEN expression
    fn evaluate_between(&self, between: &BetweenExpression) -> Result<Value> {
        let value = self.evaluate(&between.expr)?;
        let lower = self.evaluate(&between.lower)?;
        let upper = self.evaluate(&between.upper)?;

        // If any value is NULL, result is NULL (unknown) per SQL standard
        if value.is_null() || lower.is_null() || upper.is_null() {
            return Ok(Value::null_unknown());
        }

        let in_range = value >= lower && value <= upper;
        Ok(Value::Boolean(if between.not {
            !in_range
        } else {
            in_range
        }))
    }

    /// Evaluate a CASE expression
    fn evaluate_case(&self, case: &CaseExpression) -> Result<Value> {
        // Simple CASE: CASE expr WHEN val1 THEN result1 ...
        if let Some(ref value_expr) = case.value {
            let operand_val = self.evaluate(value_expr)?;

            for branch in &case.when_clauses {
                let when_val = self.evaluate(&branch.condition)?;
                // SQL standard: NULL = NULL yields UNKNOWN, not TRUE
                // So we must check for NULL explicitly and skip if either is NULL
                if operand_val.is_null() || when_val.is_null() {
                    // NULL comparison always yields unknown, never matches
                    continue;
                }
                if operand_val == when_val {
                    return self.evaluate(&branch.then_result);
                }
            }
        } else {
            // Searched CASE: CASE WHEN cond1 THEN result1 ...
            for branch in &case.when_clauses {
                if self.evaluate_bool(&branch.condition)? {
                    return self.evaluate(&branch.then_result);
                }
            }
        }

        // ELSE clause
        if let Some(ref else_value) = case.else_value {
            self.evaluate(else_value)
        } else {
            Ok(Value::null_unknown())
        }
    }

    /// Evaluate a CAST expression
    fn evaluate_cast(&self, cast: &CastExpression) -> Result<Value> {
        let value = self.evaluate(&cast.expr)?;
        let target_type = cast.type_name.to_uppercase();

        match target_type.as_str() {
            "INTEGER" | "INT" | "BIGINT" => match value {
                Value::Integer(i) => Ok(Value::Integer(i)),
                Value::Float(f) => Ok(Value::Integer(f as i64)),
                Value::Text(s) => s
                    .parse::<i64>()
                    .map(Value::Integer)
                    .map_err(|_| Error::Type(format!("Cannot cast '{}' to INTEGER", s))),
                Value::Boolean(b) => Ok(Value::Integer(if b { 1 } else { 0 })),
                Value::Null(_) => Ok(Value::null_unknown()),
                _ => Err(Error::Type(format!("Cannot cast {:?} to INTEGER", value))),
            },
            "FLOAT" | "DOUBLE" | "REAL" => match value {
                Value::Integer(i) => Ok(Value::Float(i as f64)),
                Value::Float(f) => Ok(Value::Float(f)),
                Value::Text(s) => s
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| Error::Type(format!("Cannot cast '{}' to FLOAT", s))),
                Value::Null(_) => Ok(Value::null_unknown()),
                _ => Err(Error::Type(format!("Cannot cast {:?} to FLOAT", value))),
            },
            "TEXT" | "VARCHAR" | "CHAR" | "STRING" => Ok(Value::Text(std::sync::Arc::from(
                value.to_string().as_str(),
            ))),
            "BOOLEAN" | "BOOL" => match value {
                Value::Boolean(b) => Ok(Value::Boolean(b)),
                Value::Integer(i) => Ok(Value::Boolean(i != 0)),
                Value::Text(s) => {
                    let s_lower = s.to_lowercase();
                    Ok(Value::Boolean(
                        s_lower == "true" || s_lower == "1" || s_lower == "yes",
                    ))
                }
                Value::Null(_) => Ok(Value::null_unknown()),
                _ => Err(Error::Type(format!("Cannot cast {:?} to BOOLEAN", value))),
            },
            "DATE" => match value {
                Value::Timestamp(t) => {
                    // Truncate time to midnight for DATE cast
                    use chrono::{Datelike, TimeZone, Utc};
                    let truncated = Utc
                        .with_ymd_and_hms(t.year(), t.month(), t.day(), 0, 0, 0)
                        .single()
                        .unwrap_or(t);
                    Ok(Value::Timestamp(truncated))
                }
                Value::Text(s) => match crate::core::parse_timestamp(&s) {
                    Ok(t) => {
                        // Truncate time to midnight for DATE cast
                        use chrono::{Datelike, TimeZone, Utc};
                        let truncated = Utc
                            .with_ymd_and_hms(t.year(), t.month(), t.day(), 0, 0, 0)
                            .single()
                            .unwrap_or(t);
                        Ok(Value::Timestamp(truncated))
                    }
                    Err(_) => Err(Error::Type(format!("Cannot parse '{}' as DATE", s))),
                },
                Value::Integer(i) => {
                    // Interpret as Unix timestamp, then truncate to date
                    use chrono::{Datelike, TimeZone, Utc};
                    match Utc.timestamp_opt(i, 0) {
                        chrono::LocalResult::Single(t) => {
                            let truncated = Utc
                                .with_ymd_and_hms(t.year(), t.month(), t.day(), 0, 0, 0)
                                .single()
                                .unwrap_or(t);
                            Ok(Value::Timestamp(truncated))
                        }
                        _ => Err(Error::Type(format!("Invalid Unix timestamp: {}", i))),
                    }
                }
                Value::Null(_) => Ok(Value::null_unknown()),
                _ => Err(Error::Type(format!("Cannot cast {:?} to DATE", value))),
            },
            "TIMESTAMP" | "DATETIME" | "TIME" => match value {
                Value::Timestamp(t) => Ok(Value::Timestamp(t)),
                Value::Text(s) => match crate::core::parse_timestamp(&s) {
                    Ok(t) => Ok(Value::Timestamp(t)),
                    Err(_) => Err(Error::Type(format!("Cannot parse '{}' as TIMESTAMP", s))),
                },
                Value::Integer(i) => {
                    // Interpret as Unix timestamp
                    use chrono::{TimeZone, Utc};
                    match Utc.timestamp_opt(i, 0) {
                        chrono::LocalResult::Single(t) => Ok(Value::Timestamp(t)),
                        _ => Err(Error::Type(format!("Invalid Unix timestamp: {}", i))),
                    }
                }
                Value::Null(_) => Ok(Value::null_unknown()),
                _ => Err(Error::Type(format!("Cannot cast {:?} to TIMESTAMP", value))),
            },
            "JSON" => match value {
                Value::Json(j) => Ok(Value::Json(j)),
                Value::Text(s) => Ok(Value::Json(s)),
                Value::Integer(i) => Ok(Value::Json(std::sync::Arc::from(i.to_string().as_str()))),
                Value::Float(f) => Ok(Value::Json(std::sync::Arc::from(f.to_string().as_str()))),
                Value::Boolean(b) => Ok(Value::Json(std::sync::Arc::from(b.to_string().as_str()))),
                Value::Null(_) => Ok(Value::Json(std::sync::Arc::from("null"))),
                Value::Timestamp(t) => Ok(Value::Json(std::sync::Arc::from(
                    format!("\"{}\"", t.to_rfc3339()).as_str(),
                ))),
            },
            _ => Err(Error::Type(format!(
                "Unsupported cast target type: {}",
                target_type
            ))),
        }
    }

    /// Evaluate a function call
    fn evaluate_function_call(&self, func: &FunctionCall) -> Result<Value> {
        // OPTIMIZATION: func.function is already uppercase from parsing (see expressions.rs)
        // No need to call to_uppercase() - avoids allocation per row!

        // Special handling for context-dependent functions that need evaluator state
        if func.function == "CURRENT_TRANSACTION_ID" {
            return match self.transaction_id {
                Some(txn_id) => Ok(Value::Integer(txn_id as i64)),
                None => Ok(Value::null_unknown()),
            };
        }

        // Get function from registry
        if let Some(scalar_func) = self.function_registry.get_scalar(&func.function) {
            // Evaluate arguments
            // OPTIMIZATION: Pre-allocate with known capacity
            let mut args = Vec::with_capacity(func.arguments.len());
            for arg in &func.arguments {
                args.push(self.evaluate(arg)?);
            }

            // Call the function using evaluate method
            scalar_func.evaluate(&args)
        } else {
            // Check if this is an aggregate function reference in post-aggregation context
            // (e.g., HAVING SUM(price) > 100 where "SUM(price)" is already a computed column)
            let agg_col_name = self.format_aggregate_column_name(func);
            if let Some(&idx) = self.column_indices.get(&agg_col_name) {
                // Look up the pre-computed aggregate value from the row
                if let Some(val) = self.get_value_at_index(idx) {
                    return Ok(val.clone());
                }
            }
            // Also try lowercase version for case-insensitive lookup
            let agg_col_name_lower = agg_col_name.to_lowercase();
            if let Some(&idx) = self.column_indices.get(&agg_col_name_lower) {
                if let Some(val) = self.get_value_at_index(idx) {
                    return Ok(val.clone());
                }
            }

            Err(Error::NotSupportedMessage(format!(
                "Unknown function: {}",
                func.function
            )))
        }
    }

    /// Format an aggregate function call as a column name (e.g., "SUM(v)")
    fn format_aggregate_column_name(&self, func: &FunctionCall) -> String {
        let args_str: Vec<String> = func
            .arguments
            .iter()
            .map(Self::expression_to_string)
            .collect();
        format!("{}({})", func.function, args_str.join(", "))
    }

    /// Convert an expression to a display string (for column name generation)
    /// NOTE: This must match the format used in aggregation.rs expression_to_string
    fn expression_to_string(expr: &Expression) -> String {
        match expr {
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(qid) => {
                format!("{}.{}", qid.qualifier.value, qid.name.value)
            }
            Expression::Star(_) => "*".to_string(),
            Expression::StringLiteral(lit) => format!("'{}'", lit.value),
            Expression::IntegerLiteral(lit) => lit.value.to_string(),
            Expression::FloatLiteral(lit) => lit.value.to_string(),
            Expression::BooleanLiteral(lit) => lit.value.to_string(),
            Expression::FunctionCall(func) => {
                let args: Vec<String> = func
                    .arguments
                    .iter()
                    .map(Self::expression_to_string)
                    .collect();
                format!("{}({})", func.function, args.join(", "))
            }
            Expression::Infix(infix) => {
                // Format without parentheses to match aggregation.rs format
                format!(
                    "{} {} {}",
                    Self::expression_to_string(&infix.left),
                    infix.operator,
                    Self::expression_to_string(&infix.right)
                )
            }
            Expression::Prefix(prefix) => {
                format!(
                    "{}{}",
                    prefix.operator,
                    Self::expression_to_string(&prefix.right)
                )
            }
            _ => format!("{}", expr),
        }
    }

    /// Evaluate an interval literal
    fn evaluate_interval(&self, interval: &IntervalLiteral) -> Result<Value> {
        // Return interval as string: quantity + unit (e.g., "1 week", not "1 week week")
        Ok(Value::Text(std::sync::Arc::from(
            format!("{} {}", interval.quantity, interval.unit).as_str(),
        )))
    }

    /// Evaluate LIKE pattern matching
    /// Uses the global pattern cache for compiled regex patterns
    /// and fast-path optimizations for simple patterns (prefix%, %suffix, %contains%)
    #[inline]
    fn evaluate_like(
        &self,
        value: &Value,
        pattern: &Value,
        case_insensitive: bool,
    ) -> Result<Value> {
        // Get text to match against
        let text: &str = match value {
            Value::Text(s) => s,
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => return Ok(Value::Boolean(false)),
        };

        // Get pattern string
        let pattern_str: &str = match pattern {
            Value::Text(s) => s,
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => return Ok(Value::Boolean(false)),
        };

        // Use the global pattern cache for compiled patterns
        // This avoids re-compiling regex on every row!
        use super::pattern_cache::global_pattern_cache;

        let cache = global_pattern_cache();
        let compiled = if case_insensitive {
            cache.get_or_compile_insensitive(pattern_str)
        } else {
            cache.get_or_compile(pattern_str)
        };

        // Use optimized matching (no string allocation for simple patterns!)
        let result = if case_insensitive {
            compiled.matches_insensitive(text)
        } else {
            compiled.matches(text)
        };

        Ok(Value::Boolean(result))
    }

    /// Evaluate a LikeExpression (with optional ESCAPE clause)
    fn evaluate_like_expression(&self, like: &LikeExpression) -> Result<Value> {
        let left_value = self.evaluate(&like.left)?;
        let pattern_value = self.evaluate(&like.pattern)?;

        // Determine if this is case-insensitive (ILIKE)
        let case_insensitive = like.operator.contains("ILIKE");
        let is_negated = like.operator.starts_with("NOT ");

        // Handle GLOB and REGEXP separately
        if like.operator.contains("GLOB") {
            let result = self.evaluate_glob(&left_value, &pattern_value)?;
            if is_negated {
                return match result {
                    Value::Boolean(b) => Ok(Value::Boolean(!b)),
                    Value::Null(_) => Ok(result),
                    _ => Ok(Value::Boolean(false)),
                };
            }
            return Ok(result);
        }

        if like.operator.contains("REGEXP") || like.operator.contains("RLIKE") {
            let result = self.evaluate_regexp(&left_value, &pattern_value)?;
            if is_negated {
                return match result {
                    Value::Boolean(b) => Ok(Value::Boolean(!b)),
                    Value::Null(_) => Ok(result),
                    _ => Ok(Value::Boolean(false)),
                };
            }
            return Ok(result);
        }

        // Handle ESCAPE clause for LIKE/ILIKE
        let result = if let Some(ref escape_expr) = like.escape {
            let escape_value = self.evaluate(escape_expr)?;
            self.evaluate_like_with_escape(
                &left_value,
                &pattern_value,
                case_insensitive,
                &escape_value,
            )?
        } else {
            self.evaluate_like(&left_value, &pattern_value, case_insensitive)?
        };

        if is_negated {
            match result {
                Value::Boolean(b) => Ok(Value::Boolean(!b)),
                Value::Null(_) => Ok(result),
                _ => Ok(Value::Boolean(false)),
            }
        } else {
            Ok(result)
        }
    }

    /// Evaluate LIKE pattern matching with ESCAPE character
    fn evaluate_like_with_escape(
        &self,
        value: &Value,
        pattern: &Value,
        case_insensitive: bool,
        escape: &Value,
    ) -> Result<Value> {
        // Get text to match against
        let text: &str = match value {
            Value::Text(s) => s,
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => return Ok(Value::Boolean(false)),
        };

        // Get pattern string
        let pattern_str: &str = match pattern {
            Value::Text(s) => s,
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => return Ok(Value::Boolean(false)),
        };

        // Get escape character
        let escape_char: char = match escape {
            Value::Text(s) => {
                if s.len() != 1 {
                    return Err(Error::Type(format!(
                        "ESCAPE character must be a single character, got '{}'",
                        s
                    )));
                }
                s.chars().next().unwrap()
            }
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => {
                return Err(Error::Type(
                    "ESCAPE must be a single character string".to_string(),
                ))
            }
        };

        // Convert LIKE pattern to regex with custom escape character
        let mut regex_pattern = String::with_capacity(pattern_str.len() * 2);
        regex_pattern.push('^');

        let mut chars = pattern_str.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == escape_char {
                // Next character is literal (escaped)
                if let Some(next) = chars.next() {
                    // Escape regex special characters
                    if regex::escape(&next.to_string()).len() > 1 {
                        regex_pattern.push_str(&regex::escape(&next.to_string()));
                    } else {
                        regex_pattern.push(next);
                    }
                }
            } else {
                match ch {
                    '%' => regex_pattern.push_str(".*"),
                    '_' => regex_pattern.push('.'),
                    // Escape regex special characters
                    '.' | '+' | '*' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|'
                    | '\\' => {
                        regex_pattern.push('\\');
                        regex_pattern.push(ch);
                    }
                    _ => regex_pattern.push(ch),
                }
            }
        }
        regex_pattern.push('$');

        // Compile and match
        let regex_str = if case_insensitive {
            format!("(?i){}", regex_pattern)
        } else {
            regex_pattern
        };

        match get_or_compile_regex(&regex_str) {
            Ok(re) => Ok(Value::Boolean(re.is_match(text))),
            Err(_) => Ok(Value::Boolean(false)),
        }
    }

    /// Evaluate GLOB pattern matching (SQLite-style)
    /// Uses shell-style wildcards: * matches any sequence, ? matches single char, [...] for character classes
    #[inline]
    fn evaluate_glob(&self, value: &Value, pattern: &Value) -> Result<Value> {
        // Get text to match against
        let text: &str = match value {
            Value::Text(s) => s,
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => return Ok(Value::Boolean(false)),
        };

        // Get pattern string
        let pattern_str: &str = match pattern {
            Value::Text(s) => s,
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => return Ok(Value::Boolean(false)),
        };

        // Convert GLOB pattern to regex:
        // * -> .* (match any sequence)
        // ? -> . (match single character)
        // [...] -> [...] (character class)
        // Other special chars are escaped
        let mut regex_pattern = String::with_capacity(pattern_str.len() * 2);
        regex_pattern.push('^');
        let mut chars = pattern_str.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '*' => regex_pattern.push_str(".*"),
                '?' => regex_pattern.push('.'),
                '[' => {
                    // Pass through character class
                    regex_pattern.push('[');
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        regex_pattern.push(next);
                        if next == ']' {
                            break;
                        }
                    }
                }
                // Escape regex special characters
                '.' | '+' | '^' | '$' | '(' | ')' | '{' | '}' | '|' | '\\' => {
                    regex_pattern.push('\\');
                    regex_pattern.push(ch);
                }
                _ => regex_pattern.push(ch),
            }
        }
        regex_pattern.push('$');

        // Use cached regex compilation (GLOB is case-sensitive)
        match get_or_compile_regex(&regex_pattern) {
            Ok(re) => Ok(Value::Boolean(re.is_match(text))),
            Err(_) => Ok(Value::Boolean(false)),
        }
    }

    /// Evaluate REGEXP pattern matching
    /// Uses standard regex syntax
    #[inline]
    fn evaluate_regexp(&self, value: &Value, pattern: &Value) -> Result<Value> {
        // Get text to match against
        let text: &str = match value {
            Value::Text(s) => s,
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => return Ok(Value::Boolean(false)),
        };

        // Get pattern string
        let pattern_str: &str = match pattern {
            Value::Text(s) => s,
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Boolean)),
            _ => return Ok(Value::Boolean(false)),
        };

        // Use cached regex compilation
        match get_or_compile_regex(pattern_str) {
            Ok(re) => Ok(Value::Boolean(re.is_match(text))),
            Err(_) => Err(Error::InvalidArgumentMessage(format!(
                "Invalid regular expression: {}",
                pattern_str
            ))),
        }
    }

    /// Evaluate JSON access operators -> and ->>
    /// -> returns JSON (preserves type)
    /// ->> returns TEXT (extracts as string)
    fn evaluate_json_access(
        &self,
        json_value: &Value,
        key: &Value,
        as_text: bool,
    ) -> Result<Value> {
        // Get the JSON string
        let json_str = match json_value {
            Value::Json(s) => s.as_ref(),
            Value::Text(s) => s.as_ref(),
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Json)),
            _ => {
                return Err(Error::Type(format!(
                    "JSON operator requires JSON value, got {:?}",
                    json_value
                )))
            }
        };

        // Parse the JSON
        let parsed: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(_) => return Ok(Value::Null(crate::core::DataType::Json)),
        };

        // Get the accessed element based on key type
        let result = match key {
            // String key - object field access
            Value::Text(k) => parsed.get(k.as_ref()),
            // Integer key - array index access
            Value::Integer(idx) => {
                if *idx >= 0 {
                    parsed.get(*idx as usize)
                } else {
                    None
                }
            }
            Value::Null(_) => return Ok(Value::Null(crate::core::DataType::Json)),
            _ => {
                return Err(Error::Type(format!(
                    "JSON key must be string or integer, got {:?}",
                    key
                )))
            }
        };

        match result {
            Some(v) => {
                if as_text {
                    // ->> returns TEXT - extract string content without quotes
                    let text = match v {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Null => {
                            return Ok(Value::Null(crate::core::DataType::Text))
                        }
                        other => other.to_string(),
                    };
                    Ok(Value::Text(std::sync::Arc::from(text.as_str())))
                } else {
                    // -> returns JSON
                    let json_out = v.to_string();
                    Ok(Value::Json(std::sync::Arc::from(json_out.as_str())))
                }
            }
            None => {
                if as_text {
                    Ok(Value::Null(crate::core::DataType::Text))
                } else {
                    Ok(Value::Null(crate::core::DataType::Json))
                }
            }
        }
    }

    /// Compare two values and apply a comparison function
    fn compare_values<F>(&self, left: &Value, right: &Value, cmp_fn: F) -> Result<Value>
    where
        F: Fn(std::cmp::Ordering) -> bool,
    {
        // Handle NULL comparison
        if matches!(left, Value::Null(_)) || matches!(right, Value::Null(_)) {
            return Ok(Value::Null(crate::core::DataType::Boolean));
        }

        // Compare values
        match left.partial_cmp(right) {
            Some(ordering) => Ok(Value::Boolean(cmp_fn(ordering))),
            None => Err(Error::Type(format!(
                "Cannot compare {:?} and {:?}",
                left, right
            ))),
        }
    }

    /// Perform arithmetic operation on two values with overflow checking
    fn arithmetic_op<I, F>(
        &self,
        left: &Value,
        right: &Value,
        int_op: I,
        float_op: F,
    ) -> Result<Value>
    where
        I: Fn(i64, i64) -> Option<i64>,
        F: Fn(f64, f64) -> f64,
    {
        // Helper to try converting Text to a numeric value
        fn try_text_to_numeric(s: &str) -> Option<Value> {
            let trimmed = s.trim();
            // Try integer first
            if let Ok(i) = trimmed.parse::<i64>() {
                return Some(Value::Integer(i));
            }
            // Try float
            if let Ok(f) = trimmed.parse::<f64>() {
                return Some(Value::Float(f));
            }
            None
        }

        // Convert operands: try to coerce Text to numeric
        let left_val = match left {
            Value::Text(s) => try_text_to_numeric(s).unwrap_or_else(|| left.clone()),
            _ => left.clone(),
        };
        let right_val = match right {
            Value::Text(s) => try_text_to_numeric(s).unwrap_or_else(|| right.clone()),
            _ => right.clone(),
        };

        match (&left_val, &right_val) {
            (Value::Integer(l), Value::Integer(r)) => match int_op(*l, *r) {
                Some(result) => Ok(Value::Integer(result)),
                None => Err(Error::Type(format!(
                    "Integer overflow in arithmetic operation: {} and {}",
                    l, r
                ))),
            },
            (Value::Float(l), Value::Float(r)) => Ok(Value::Float(float_op(*l, *r))),
            (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(float_op(*l as f64, *r))),
            (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(float_op(*l, *r as f64))),
            (Value::Null(_), _) | (_, Value::Null(_)) => Ok(Value::null_unknown()),
            _ => Err(Error::Type(format!(
                "Cannot perform arithmetic on {:?} and {:?}",
                left, right
            ))),
        }
    }

    /// Add or subtract an interval from a timestamp
    fn add_interval_to_timestamp(
        &self,
        ts: chrono::DateTime<chrono::Utc>,
        interval_str: &str,
        add: bool,
    ) -> Result<Value> {
        // Parse interval string like "24 hours", "1 day", "30 minutes", etc.
        let parts: Vec<&str> = interval_str.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(Error::Type(format!(
                "Invalid interval format: {}",
                interval_str
            )));
        }

        let amount: i64 = parts[0]
            .parse()
            .map_err(|_| Error::Type(format!("Invalid interval amount: {}", parts[0])))?;

        let unit = parts[1].to_lowercase();
        let duration = match unit.trim_end_matches('s') {
            "second" => chrono::Duration::seconds(amount),
            "minute" => chrono::Duration::minutes(amount),
            "hour" => chrono::Duration::hours(amount),
            "day" => chrono::Duration::days(amount),
            "week" => chrono::Duration::weeks(amount),
            "month" => chrono::Duration::days(amount * 30), // Approximate
            "year" => chrono::Duration::days(amount * 365), // Approximate
            _ => return Err(Error::Type(format!("Unknown interval unit: {}", unit))),
        };

        let result = if add { ts + duration } else { ts - duration };

        Ok(Value::Timestamp(result))
    }

    /// Format a chrono Duration as an SQL interval string
    fn format_duration_as_interval(&self, duration: chrono::Duration) -> String {
        let total_secs = duration.num_seconds();
        let is_negative = total_secs < 0;
        let total_secs = total_secs.abs();

        let days = total_secs / 86400;
        let remaining = total_secs % 86400;
        let hours = remaining / 3600;
        let remaining = remaining % 3600;
        let minutes = remaining / 60;
        let seconds = remaining % 60;

        let mut parts = Vec::new();

        if days > 0 {
            parts.push(format!("{} day{}", days, if days == 1 { "" } else { "s" }));
        }
        if hours > 0 {
            parts.push(format!(
                "{} hour{}",
                hours,
                if hours == 1 { "" } else { "s" }
            ));
        }
        if minutes > 0 {
            parts.push(format!(
                "{} minute{}",
                minutes,
                if minutes == 1 { "" } else { "s" }
            ));
        }
        if seconds > 0 || parts.is_empty() {
            parts.push(format!(
                "{} second{}",
                seconds,
                if seconds == 1 { "" } else { "s" }
            ));
        }

        let interval = parts.join(" ");
        if is_negative {
            format!("-{}", interval)
        } else {
            interval
        }
    }

    /// Convert a value to Option<bool> for three-valued logic
    /// Returns None for NULL, Some(true) or Some(false) for boolean values
    fn value_to_tribool(&self, value: &Value) -> Result<Option<bool>> {
        match value {
            Value::Boolean(b) => Ok(Some(*b)),
            Value::Integer(i) => Ok(Some(*i != 0)),
            Value::Null(_) => Ok(None),
            _ => Err(Error::Type(format!(
                "Cannot convert {:?} to boolean",
                value
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::token::{Position, Token, TokenType};

    fn dummy_token(literal: &str) -> Token {
        Token::new(TokenType::Operator, literal, Position::new(0, 1, 1))
    }

    #[test]
    fn test_evaluate_integer_literal() {
        let evaluator = Evaluator::with_defaults();
        let expr = Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token("42"),
            value: 42,
        });
        let result = evaluator.evaluate(&expr).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_evaluate_string_literal() {
        let evaluator = Evaluator::with_defaults();
        let expr = Expression::StringLiteral(StringLiteral {
            token: dummy_token("'hello'"),
            value: "hello".to_string(),
            type_hint: None,
        });
        let result = evaluator.evaluate(&expr).unwrap();
        assert_eq!(result, Value::text("hello"));
    }

    #[test]
    fn test_evaluate_identifier() {
        let mut row_map = FxHashMap::default();
        row_map.insert("name".to_string(), Value::text("Alice"));

        let evaluator = Evaluator::with_defaults().with_row_map(row_map);
        let expr = Expression::Identifier(Identifier::new(dummy_token("name"), "name".to_string()));
        let result = evaluator.evaluate(&expr).unwrap();
        assert_eq!(result, Value::text("Alice"));
    }

    #[test]
    fn test_evaluate_parameter() {
        let evaluator = Evaluator::with_defaults().with_params(vec![Value::Integer(100)]);
        let expr = Expression::Parameter(Parameter {
            token: dummy_token("$1"),
            name: String::new(),
            index: 1,
        });
        let result = evaluator.evaluate(&expr).unwrap();
        assert_eq!(result, Value::Integer(100));
    }

    #[test]
    fn test_evaluate_arithmetic() {
        let evaluator = Evaluator::with_defaults();

        // Test addition
        let expr = Expression::Infix(InfixExpression::new(
            dummy_token("+"),
            Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: dummy_token("10"),
                value: 10,
            })),
            "+".to_string(),
            Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: dummy_token("20"),
                value: 20,
            })),
        ));
        let result = evaluator.evaluate(&expr).unwrap();
        assert_eq!(result, Value::Integer(30));
    }

    #[test]
    fn test_evaluate_comparison() {
        let evaluator = Evaluator::with_defaults();

        // Test equality
        let expr = Expression::Infix(InfixExpression::new(
            dummy_token("="),
            Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: dummy_token("5"),
                value: 5,
            })),
            "=".to_string(),
            Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: dummy_token("5"),
                value: 5,
            })),
        ));
        let result = evaluator.evaluate(&expr).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_evaluate_bool() {
        let evaluator = Evaluator::with_defaults();
        let expr = Expression::BooleanLiteral(BooleanLiteral {
            token: dummy_token("TRUE"),
            value: true,
        });
        assert!(evaluator.evaluate_bool(&expr).unwrap());
    }
}
