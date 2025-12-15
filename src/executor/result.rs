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

//! Execution Result Types
//!
//! This module provides result types for SQL query execution.

use crate::core::{Result, Row, Value};
use crate::parser::ast::Expression;
use crate::storage::traits::QueryResult;
use rustc_hash::{FxHashMap, FxHasher};

use super::expression::RowFilter;

/// Execution result for DML operations (INSERT, UPDATE, DELETE)
///
/// This result type tracks the number of rows affected and the last insert ID
/// for auto-increment columns.
pub struct ExecResult {
    /// Number of rows affected
    affected: i64,
    /// Last insert ID (for auto-increment)
    insert_id: i64,
    /// Column names (empty for DML)
    columns: Vec<String>,
    /// Empty row for row() method
    empty_row: Row,
}

impl ExecResult {
    /// Create a new execution result
    pub fn new(rows_affected: i64, last_insert_id: i64) -> Self {
        Self {
            affected: rows_affected,
            insert_id: last_insert_id,
            columns: Vec::new(),
            empty_row: Row::new(),
        }
    }

    /// Create an empty result (for DDL statements)
    pub fn empty() -> Self {
        Self::new(0, 0)
    }

    /// Create a result with just rows affected
    pub fn with_rows_affected(rows_affected: i64) -> Self {
        Self::new(rows_affected, 0)
    }

    /// Create a result with rows affected and last insert ID
    pub fn with_last_insert_id(rows_affected: i64, last_insert_id: i64) -> Self {
        Self::new(rows_affected, last_insert_id)
    }
}

impl QueryResult for ExecResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next(&mut self) -> bool {
        // DML results have no rows
        false
    }

    fn scan(&self, _dest: &mut [Value]) -> Result<()> {
        Err(crate::core::Error::internal("scan() called on exec result"))
    }

    fn row(&self) -> &Row {
        &self.empty_row
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn rows_affected(&self) -> i64 {
        self.affected
    }

    fn last_insert_id(&self) -> i64 {
        self.insert_id
    }

    fn with_aliases(self: Box<Self>, _aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        self
    }
}

/// Memory-based result for SELECT queries
///
/// This result type stores all rows in memory, suitable for
/// small to medium result sets.
pub struct ExecutorMemoryResult {
    /// Column names
    columns: Vec<String>,
    /// Result rows
    rows: Vec<Row>,
    /// Current row index (None before first next())
    current_index: Option<usize>,
    /// Whether the result is closed
    closed: bool,
    /// Rows affected (0 for SELECT)
    affected: i64,
    /// Last insert ID (0 for SELECT)
    insert_id: i64,
}

impl ExecutorMemoryResult {
    /// Create a new memory result with columns and rows
    pub fn new(columns: Vec<String>, rows: Vec<Row>) -> Self {
        Self {
            columns,
            rows,
            current_index: None,
            closed: false,
            affected: 0,
            insert_id: 0,
        }
    }

    /// Create an empty memory result
    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new())
    }

    /// Create with columns only (no rows yet)
    pub fn with_columns(columns: Vec<String>) -> Self {
        Self::new(columns, Vec::new())
    }

    /// Create with schema
    pub fn with_schema(columns: Vec<String>, rows: Vec<Row>, _schema: crate::core::Schema) -> Self {
        Self::new(columns, rows)
    }

    /// Add a row to the result
    pub fn add_row(&mut self, row: Row) {
        self.rows.push(row);
    }

    /// Get the number of rows
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get all rows
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Take ownership of all rows
    pub fn into_rows(self) -> Vec<Row> {
        self.rows
    }

    /// Reset the cursor to the beginning
    pub fn reset(&mut self) {
        self.current_index = None;
    }

    /// Set rows affected (for modification results)
    pub fn set_rows_affected(&mut self, count: i64) {
        self.affected = count;
    }

    /// Set last insert ID
    pub fn set_last_insert_id(&mut self, id: i64) {
        self.insert_id = id;
    }
}

impl QueryResult for ExecutorMemoryResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next(&mut self) -> bool {
        if self.closed {
            return false;
        }

        let next_index = match self.current_index {
            None => 0,
            Some(i) => i + 1,
        };

        if next_index < self.rows.len() {
            self.current_index = Some(next_index);
            true
        } else {
            false
        }
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        let row = self.row();

        if dest.len() != row.len() {
            return Err(crate::core::Error::internal(format!(
                "scan destination has {} values but row has {} columns",
                dest.len(),
                row.len()
            )));
        }

        for (i, value) in row.iter().enumerate() {
            dest[i] = value.clone();
        }

        Ok(())
    }

    fn row(&self) -> &Row {
        match self.current_index {
            Some(i) if i < self.rows.len() => &self.rows[i],
            _ => panic!("row() called without successful next()"),
        }
    }

    fn take_row(&mut self) -> Row {
        match self.current_index {
            Some(i) if i < self.rows.len() => {
                // Swap out the row with an empty one
                std::mem::take(&mut self.rows[i])
            }
            _ => panic!("take_row() called without successful next()"),
        }
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }

    fn rows_affected(&self) -> i64 {
        self.affected
    }

    fn last_insert_id(&self) -> i64 {
        self.insert_id
    }

    fn with_aliases(
        mut self: Box<Self>,
        aliases: FxHashMap<String, String>,
    ) -> Box<dyn QueryResult> {
        // Apply aliases to column names
        for col in &mut self.columns {
            // Find if this column has an alias (reverse lookup)
            for (alias, original) in &aliases {
                if col == original {
                    *col = alias.clone();
                    break;
                }
            }
        }
        self
    }
}

/// Filtered result that applies a WHERE clause to an underlying result
pub struct FilteredResult {
    /// Underlying result
    inner: Box<dyn QueryResult>,
    /// Filter predicate function
    predicate: Box<dyn Fn(&Row) -> bool + Send + Sync>,
    /// Current row (cached after filter passes)
    current_row: Option<Row>,
    /// Columns cached
    columns: Vec<String>,
}

impl FilteredResult {
    /// Create a new filtered result
    pub fn new(
        inner: Box<dyn QueryResult>,
        predicate: Box<dyn Fn(&Row) -> bool + Send + Sync>,
    ) -> Self {
        let columns = inner.columns().to_vec();
        Self {
            inner,
            predicate,
            current_row: None,
            columns,
        }
    }
}

impl QueryResult for FilteredResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next(&mut self) -> bool {
        // Keep advancing until we find a row that passes the filter
        while self.inner.next() {
            let row = self.inner.row();
            if (self.predicate)(row) {
                // OPTIMIZATION: Use take_row() to avoid clone when possible
                self.current_row = Some(self.inner.take_row());
                return true;
            }
        }
        self.current_row = None;
        false
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        if let Some(ref row) = self.current_row {
            if dest.len() != row.len() {
                return Err(crate::core::Error::internal(format!(
                    "scan destination has {} values but row has {} columns",
                    dest.len(),
                    row.len()
                )));
            }
            for (i, value) in row.iter().enumerate() {
                dest[i] = value.clone();
            }
            Ok(())
        } else {
            Err(crate::core::Error::internal(
                "scan() called without successful next()",
            ))
        }
    }

    fn row(&self) -> &Row {
        self.current_row
            .as_ref()
            .expect("row() called without successful next()")
    }

    fn take_row(&mut self) -> Row {
        self.current_row
            .take()
            .expect("take_row() called without successful next()")
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        self.inner.rows_affected()
    }

    fn last_insert_id(&self) -> i64 {
        self.inner.last_insert_id()
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Mapped result that transforms each row using a mapper function
///
/// Similar to FilteredResult but maps/transforms rows instead of filtering them.
/// Useful for evaluating expressions and projecting computed columns.
pub struct MappedResult {
    /// Underlying result
    inner: Box<dyn QueryResult>,
    /// Mapper function that transforms a row into a new row
    mapper: Box<dyn Fn(&Row) -> Row + Send + Sync>,
    /// Current mapped row
    current_row: Row,
    /// Output column names
    output_columns: Vec<String>,
}

impl MappedResult {
    /// Create a new mapped result
    ///
    /// # Arguments
    /// * `inner` - The source result
    /// * `output_columns` - Names for the output columns
    /// * `mapper` - Function that transforms each source row into an output row
    pub fn new(
        inner: Box<dyn QueryResult>,
        output_columns: Vec<String>,
        mapper: Box<dyn Fn(&Row) -> Row + Send + Sync>,
    ) -> Self {
        Self {
            inner,
            mapper,
            current_row: Row::new(),
            output_columns,
        }
    }
}

impl QueryResult for MappedResult {
    fn columns(&self) -> &[String] {
        &self.output_columns
    }

    fn next(&mut self) -> bool {
        if self.inner.next() {
            let source_row = self.inner.row();
            self.current_row = (self.mapper)(source_row);
            true
        } else {
            false
        }
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        if dest.len() != self.current_row.len() {
            return Err(crate::core::Error::internal(format!(
                "scan destination has {} values but row has {} columns",
                dest.len(),
                self.current_row.len()
            )));
        }
        for (i, value) in self.current_row.iter().enumerate() {
            dest[i] = value.clone();
        }
        Ok(())
    }

    fn row(&self) -> &Row {
        &self.current_row
    }

    fn take_row(&mut self) -> Row {
        std::mem::take(&mut self.current_row)
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        0
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Expression-based filtered result that uses RowFilter for efficient reuse
///
/// Unlike FilteredResult which takes a closure, this struct owns a pre-compiled
/// RowFilter, avoiding per-row compilation. The filter is compiled once during
/// construction and reused for every row.
pub struct ExprFilteredResult {
    /// Underlying result
    inner: Box<dyn QueryResult>,
    /// Pre-compiled row filter (thread-safe, reusable)
    filter: RowFilter,
    /// Current row (cached after filter passes)
    current_row: Option<Row>,
    /// Columns cached
    columns: Vec<String>,
}

// ExprFilteredResult is Send because all fields are Send:
// - Box<dyn QueryResult> is Send (trait bound)
// - RowFilter is Send+Sync (uses Arc internally)
// - Option<Row> and Vec<String> are Send
unsafe impl Send for ExprFilteredResult {}

impl ExprFilteredResult {
    /// Create a new expression-filtered result
    ///
    /// # Arguments
    /// * `inner` - The source result to filter
    /// * `filter_expr` - The WHERE clause expression
    ///
    /// Returns an error if the filter expression cannot be compiled.
    pub fn new(inner: Box<dyn QueryResult>, filter_expr: &Expression) -> Result<Self> {
        let columns = inner.columns().to_vec();
        let filter = RowFilter::new(filter_expr, &columns)?;

        Ok(Self {
            inner,
            filter,
            current_row: None,
            columns,
        })
    }

    /// Create with default function registry (static lifetime)
    ///
    /// This is a convenience method that panics on compilation errors.
    /// For fallible construction, use `new()` instead.
    pub fn with_defaults(inner: Box<dyn QueryResult>, filter_expr: Expression) -> Self {
        let columns = inner.columns().to_vec();
        let filter =
            RowFilter::new(&filter_expr, &columns).expect("Failed to compile filter expression");

        Self {
            inner,
            filter,
            current_row: None,
            columns,
        }
    }
}

impl QueryResult for ExprFilteredResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next(&mut self) -> bool {
        // Keep advancing until we find a row that passes the filter
        while self.inner.next() {
            let row = self.inner.row();
            // Use pre-compiled filter - thread-safe and efficient
            if self.filter.matches(row) {
                self.current_row = Some(self.inner.take_row());
                return true;
            }
        }
        self.current_row = None;
        false
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        if let Some(ref row) = self.current_row {
            if dest.len() != row.len() {
                return Err(crate::core::Error::internal(format!(
                    "scan destination has {} values but row has {} columns",
                    dest.len(),
                    row.len()
                )));
            }
            for (i, value) in row.iter().enumerate() {
                dest[i] = value.clone();
            }
            Ok(())
        } else {
            Err(crate::core::Error::internal(
                "scan() called without successful next()",
            ))
        }
    }

    fn row(&self) -> &Row {
        self.current_row
            .as_ref()
            .expect("row() called without successful next()")
    }

    fn take_row(&mut self) -> Row {
        self.current_row
            .take()
            .expect("take_row() called without successful next()")
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        self.inner.rows_affected()
    }

    fn last_insert_id(&self) -> i64 {
        self.inner.last_insert_id()
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Pre-compiled projection that can be either a Star expansion or a compiled expression.
enum CompiledProjection {
    /// Expand all columns from source (SELECT *)
    Star,
    /// Expand columns for specific table/alias (SELECT t.*)
    QualifiedStar {
        /// Lowercase prefix for matching (e.g., "t.")
        prefix_lower: String,
    },
    /// A pre-compiled expression program
    Compiled(super::expression::SharedProgram),
}

/// Expression-based mapped result with pre-compiled projections
///
/// Unlike MappedResult which takes a closure, this struct pre-compiles all
/// expressions during construction, providing efficient per-row evaluation.
pub struct ExprMappedResult {
    /// Underlying result
    inner: Box<dyn QueryResult>,
    /// Pre-compiled projections (one per output column)
    projections: Vec<CompiledProjection>,
    /// VM instance for expression execution (reused)
    vm: super::expression::ExprVM,
    /// Current mapped row
    current_row: Row,
    /// Output column names
    output_columns: Vec<String>,
    /// Source columns (cached for QualifiedStar matching)
    source_columns: Vec<String>,
}

// ExprMappedResult is Send because all fields are Send:
// - Box<dyn QueryResult> is Send (trait bound)
// - CompiledProjection is Send (SharedProgram uses Arc)
// - ExprVM is Send
// - Row, Vec<String> are Send
unsafe impl Send for ExprMappedResult {}

impl ExprMappedResult {
    /// Create a new expression-mapped result
    ///
    /// # Arguments
    /// * `inner` - The source result to project
    /// * `expressions` - The projection expressions
    /// * `output_columns` - Names for the output columns
    ///
    /// Returns an error if any expression cannot be compiled.
    pub fn new(
        inner: Box<dyn QueryResult>,
        expressions: Vec<Expression>,
        output_columns: Vec<String>,
    ) -> Result<Self> {
        use super::expression::compile_expression;

        let source_columns = inner.columns().to_vec();

        // Pre-compile all expressions
        let mut projections = Vec::with_capacity(expressions.len());
        for expr in &expressions {
            let projection = match expr {
                Expression::Star(_) => CompiledProjection::Star,
                Expression::QualifiedStar(qs) => {
                    let prefix = format!("{}.", qs.qualifier);
                    CompiledProjection::QualifiedStar {
                        prefix_lower: prefix.to_lowercase(),
                    }
                }
                _ => {
                    let program = compile_expression(expr, &source_columns)?;
                    CompiledProjection::Compiled(program)
                }
            };
            projections.push(projection);
        }

        Ok(Self {
            inner,
            projections,
            vm: super::expression::ExprVM::new(),
            current_row: Row::new(),
            output_columns,
            source_columns,
        })
    }

    /// Create with default function registry (static lifetime)
    ///
    /// This is a convenience method that panics on compilation errors.
    /// For fallible construction, use `new()` instead.
    pub fn with_defaults(
        inner: Box<dyn QueryResult>,
        expressions: Vec<Expression>,
        output_columns: Vec<String>,
    ) -> Self {
        Self::new(inner, expressions, output_columns)
            .expect("Failed to compile projection expressions")
    }
}

impl QueryResult for ExprMappedResult {
    fn columns(&self) -> &[String] {
        &self.output_columns
    }

    fn next(&mut self) -> bool {
        use super::expression::ExecuteContext;

        if self.inner.next() {
            let source_row = self.inner.row();
            let row_data = source_row.as_slice();

            let mut result_row = Row::with_capacity(self.projections.len());
            for projection in &self.projections {
                match projection {
                    CompiledProjection::Star => {
                        // Expand all columns from source
                        for value in source_row.iter() {
                            result_row.push(value.clone());
                        }
                    }
                    CompiledProjection::QualifiedStar { prefix_lower, .. } => {
                        // Expand columns for specific table/alias
                        for (idx, col) in self.source_columns.iter().enumerate() {
                            if col.to_lowercase().starts_with(prefix_lower)
                                && idx < source_row.len()
                            {
                                result_row.push(source_row[idx].clone());
                            }
                        }
                    }
                    CompiledProjection::Compiled(program) => {
                        // Execute pre-compiled expression
                        let ctx = ExecuteContext::new(row_data);
                        let value = self
                            .vm
                            .execute(program, &ctx)
                            .unwrap_or(Value::null_unknown());
                        result_row.push(value);
                    }
                }
            }
            self.current_row = result_row;
            true
        } else {
            false
        }
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        if dest.len() != self.current_row.len() {
            return Err(crate::core::Error::internal(format!(
                "scan destination has {} values but row has {} columns",
                dest.len(),
                self.current_row.len()
            )));
        }
        for (i, value) in self.current_row.iter().enumerate() {
            dest[i] = value.clone();
        }
        Ok(())
    }

    fn row(&self) -> &Row {
        &self.current_row
    }

    fn take_row(&mut self) -> Row {
        std::mem::take(&mut self.current_row)
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        0
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Limited result that applies LIMIT and OFFSET to an underlying result
pub struct LimitedResult {
    /// Underlying result
    inner: Box<dyn QueryResult>,
    /// Maximum number of rows to return
    limit: Option<usize>,
    /// Number of rows to skip
    offset: usize,
    /// Number of rows returned so far
    returned_count: usize,
    /// Whether we've skipped the offset rows
    offset_applied: bool,
    /// Columns cached
    columns: Vec<String>,
}

impl LimitedResult {
    /// Create a new limited result
    pub fn new(inner: Box<dyn QueryResult>, limit: Option<usize>, offset: usize) -> Self {
        let columns = inner.columns().to_vec();
        Self {
            inner,
            limit,
            offset,
            returned_count: 0,
            offset_applied: false,
            columns,
        }
    }

    /// Create with just a limit
    pub fn with_limit(inner: Box<dyn QueryResult>, limit: usize) -> Self {
        Self::new(inner, Some(limit), 0)
    }

    /// Create with just an offset
    pub fn with_offset(inner: Box<dyn QueryResult>, offset: usize) -> Self {
        Self::new(inner, None, offset)
    }
}

impl QueryResult for LimitedResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next(&mut self) -> bool {
        // Apply offset first (skip rows)
        if !self.offset_applied {
            for _ in 0..self.offset {
                if !self.inner.next() {
                    self.offset_applied = true;
                    return false;
                }
            }
            self.offset_applied = true;
        }

        // Check limit
        if let Some(limit) = self.limit {
            if self.returned_count >= limit {
                return false;
            }
        }

        // Get next row
        if self.inner.next() {
            self.returned_count += 1;
            true
        } else {
            false
        }
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        self.inner.scan(dest)
    }

    fn row(&self) -> &Row {
        self.inner.row()
    }

    fn take_row(&mut self) -> Row {
        self.inner.take_row()
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        self.inner.rows_affected()
    }

    fn last_insert_id(&self) -> i64 {
        self.inner.last_insert_id()
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Ordered result that sorts rows by ORDER BY expressions
pub struct OrderedResult {
    /// Materialized and sorted rows
    inner: ExecutorMemoryResult,
}

/// Order specification for radix sort
#[derive(Clone, Copy)]
pub struct RadixOrderSpec {
    /// Column index to sort by
    pub col_idx: usize,
    /// Whether to sort ascending
    pub ascending: bool,
    /// NULLS FIRST/LAST specification
    /// None = default (NULLS LAST for ASC, NULLS FIRST for DESC)
    /// Some(true) = NULLS FIRST
    /// Some(false) = NULLS LAST
    pub nulls_first: Option<bool>,
}

impl OrderedResult {
    /// Create a new ordered result by materializing and sorting the inner result
    pub fn new<F>(mut inner: Box<dyn QueryResult>, compare: F) -> Self
    where
        F: Fn(&Row, &Row) -> std::cmp::Ordering,
    {
        let columns = inner.columns().to_vec();

        // Materialize all rows
        let mut rows = Vec::new();
        while inner.next() {
            rows.push(inner.take_row());
        }

        // Sort rows
        rows.sort_by(compare);

        // Create memory result
        let memory_result = ExecutorMemoryResult::new(columns, rows);

        Self {
            inner: memory_result,
        }
    }

    /// Create an ordered result using radix sort for integer columns
    ///
    /// This is O(n) instead of O(n log n) for comparison-based sort.
    /// For 10K rows, this can be 2-5x faster. For 1M rows, 5-20x faster.
    ///
    /// # Arguments
    /// * `inner` - Source result to materialize and sort
    /// * `order_specs` - Column indices and sort directions (must be integer columns)
    /// * `fallback_compare` - Fallback comparison function if radix sort fails
    pub fn new_radix<F>(
        mut inner: Box<dyn QueryResult>,
        order_specs: &[RadixOrderSpec],
        fallback_compare: F,
    ) -> Self
    where
        F: Fn(&Row, &Row) -> std::cmp::Ordering,
    {
        let columns = inner.columns().to_vec();

        // Materialize all rows
        let mut rows = Vec::new();
        while inner.next() {
            rows.push(inner.take_row());
        }

        // Check if any column has explicit NULLS FIRST/LAST setting
        // If so, skip radix sort (which uses fixed NULL ordering) and use comparison sort
        let has_explicit_nulls_ordering = order_specs.iter().any(|s| s.nulls_first.is_some());

        if !has_explicit_nulls_ordering {
            // Try radix sort for single integer column (most common case)
            if order_specs.len() == 1 {
                let spec = &order_specs[0];
                if Self::try_radix_sort_single_int(&mut rows, spec.col_idx, spec.ascending) {
                    return Self {
                        inner: ExecutorMemoryResult::new(columns, rows),
                    };
                }
            }

            // Try radix sort for multiple integer columns
            if order_specs.len() <= 4 && Self::try_radix_sort_multi_int(&mut rows, order_specs) {
                return Self {
                    inner: ExecutorMemoryResult::new(columns, rows),
                };
            }
        }

        // Fallback to comparison sort
        rows.sort_by(fallback_compare);

        Self {
            inner: ExecutorMemoryResult::new(columns, rows),
        }
    }

    /// Try to sort by a single integer column using radix sort
    /// Returns true if successful, false if column is not all integers
    fn try_radix_sort_single_int(rows: &mut [Row], col_idx: usize, ascending: bool) -> bool {
        // Check if all values in this column are integers
        for row in rows.iter() {
            match row.get(col_idx) {
                Some(Value::Integer(_)) => continue,
                Some(Value::Null(_)) => continue, // NULLs are OK, we handle them
                _ => return false,                // Non-integer found
            }
        }

        // All integers - use radix sort
        // We use radsort which handles negative numbers correctly
        if ascending {
            radsort::sort_by_key(rows, |row| {
                match row.get(col_idx) {
                    Some(Value::Integer(i)) => *i,
                    _ => i64::MIN, // NULLs sort first in ascending order
                }
            });
        } else {
            // For descending, we negate the key (radix sort is ascending only)
            // But we need to be careful with i64::MIN
            radsort::sort_by_key(rows, |row| {
                match row.get(col_idx) {
                    Some(Value::Integer(i)) => {
                        // Negate for descending order, handle overflow
                        i.wrapping_neg().wrapping_sub(1)
                    }
                    _ => i64::MAX, // NULLs sort last in descending order
                }
            });
        }

        true
    }

    /// Try to sort by multiple integer columns using radix sort
    /// This uses a composite key approach for up to 4 columns
    fn try_radix_sort_multi_int(rows: &mut [Row], order_specs: &[RadixOrderSpec]) -> bool {
        // First verify all columns are integers
        for row in rows.iter() {
            for spec in order_specs {
                match row.get(spec.col_idx) {
                    Some(Value::Integer(_)) => continue,
                    Some(Value::Null(_)) => continue,
                    _ => return false,
                }
            }
        }

        // For multi-column sort, we need to sort in reverse order of priority
        // (least significant column first, most significant last)
        // This is stable, so later sorts preserve order from earlier ones
        for spec in order_specs.iter().rev() {
            if spec.ascending {
                radsort::sort_by_key(rows, |row| match row.get(spec.col_idx) {
                    Some(Value::Integer(i)) => *i,
                    _ => i64::MIN,
                });
            } else {
                radsort::sort_by_key(rows, |row| match row.get(spec.col_idx) {
                    Some(Value::Integer(i)) => i.wrapping_neg().wrapping_sub(1),
                    _ => i64::MAX,
                });
            }
        }

        true
    }
}

impl QueryResult for OrderedResult {
    fn columns(&self) -> &[String] {
        self.inner.columns()
    }

    fn next(&mut self) -> bool {
        self.inner.next()
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        self.inner.scan(dest)
    }

    fn row(&self) -> &Row {
        self.inner.row()
    }

    fn take_row(&mut self) -> Row {
        self.inner.take_row()
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        0
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Top-N result using a bounded heap for ORDER BY + LIMIT optimization
///
/// This is O(n log k) instead of O(n log n) for full sort, where k = limit.
/// For large datasets with small limits (e.g., 1M rows, LIMIT 10), this can be 5-50x faster.
pub struct TopNResult {
    /// Materialized top-N rows
    inner: ExecutorMemoryResult,
}

impl TopNResult {
    /// Create a new top-N result using a bounded heap
    ///
    /// # Arguments
    /// * `inner` - Source result to process
    /// * `compare` - Comparison function for ordering (returns Less if a should come before b)
    /// * `limit` - Maximum number of rows to return
    /// * `offset` - Number of rows to skip (we need limit + offset rows in heap)
    pub fn new<F>(mut inner: Box<dyn QueryResult>, compare: F, limit: usize, offset: usize) -> Self
    where
        F: Fn(&Row, &Row) -> std::cmp::Ordering + Clone,
    {
        use std::collections::BinaryHeap;

        let columns = inner.columns().to_vec();
        let heap_capacity = limit.saturating_add(offset);

        // If no limit, fall back to full sort
        if heap_capacity == 0 {
            return Self {
                inner: ExecutorMemoryResult::new(columns, Vec::new()),
            };
        }

        // Wrapper for Row with custom ordering (reverse order for max-heap as min-heap)
        struct HeapRow<F: Fn(&Row, &Row) -> std::cmp::Ordering> {
            row: Row,
            compare: F,
        }

        impl<F: Fn(&Row, &Row) -> std::cmp::Ordering> PartialEq for HeapRow<F> {
            fn eq(&self, other: &Self) -> bool {
                (self.compare)(&self.row, &other.row) == std::cmp::Ordering::Equal
            }
        }

        impl<F: Fn(&Row, &Row) -> std::cmp::Ordering> Eq for HeapRow<F> {}

        impl<F: Fn(&Row, &Row) -> std::cmp::Ordering> PartialOrd for HeapRow<F> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl<F: Fn(&Row, &Row) -> std::cmp::Ordering> Ord for HeapRow<F> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // For TOP-N, we want the WORST element at the top of the max-heap
                // so we can efficiently replace it when a better element comes.
                // "Worst" means: should come LATER in the final sorted order.
                // If compare(self, other) = Less, self comes first (self is better).
                // We want worse elements to have HIGHER heap priority (be at top).
                // So: if self is better (Less), self should have LOWER heap priority.
                // Therefore: return the comparison as-is (better = lower priority in heap)
                (self.compare)(&self.row, &other.row)
            }
        }

        // Use max-heap with reversed comparison to simulate min-heap behavior
        let mut heap: BinaryHeap<HeapRow<F>> = BinaryHeap::with_capacity(heap_capacity + 1);

        // Process all rows through the heap
        while inner.next() {
            // OPTIMIZATION: Use take_row() to avoid clone when possible
            let row = inner.take_row();

            if heap.len() < heap_capacity {
                // Heap not full, just push
                // Only clone compare function when actually pushing to heap
                heap.push(HeapRow {
                    row,
                    compare: compare.clone(),
                });
            } else {
                // Heap full, check if new row should replace the worst (largest) one
                if let Some(worst) = heap.peek() {
                    if compare(&row, &worst.row) == std::cmp::Ordering::Less {
                        // New row is better (smaller), replace the worst
                        // Only clone compare function when actually pushing to heap
                        heap.pop();
                        heap.push(HeapRow {
                            row,
                            compare: compare.clone(),
                        });
                    }
                }
            }
        }

        // Extract rows from heap (comes out in reverse order)
        let mut rows: Vec<Row> = heap.into_iter().map(|hr| hr.row).collect();

        // Sort to get correct order (heap gives reverse order)
        rows.sort_by(|a, b| compare(a, b));

        // Apply offset - use drain to avoid extra allocation
        if offset > 0 && offset < rows.len() {
            rows.drain(..offset);
        } else if offset >= rows.len() {
            rows.clear();
        }

        Self {
            inner: ExecutorMemoryResult::new(columns, rows),
        }
    }
}

impl QueryResult for TopNResult {
    fn columns(&self) -> &[String] {
        self.inner.columns()
    }

    fn next(&mut self) -> bool {
        self.inner.next()
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        self.inner.scan(dest)
    }

    fn row(&self) -> &Row {
        self.inner.row()
    }

    fn take_row(&mut self) -> Row {
        self.inner.take_row()
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        0
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Distinct result that removes duplicate rows
pub struct DistinctResult {
    /// Materialized unique rows
    inner: ExecutorMemoryResult,
}

impl DistinctResult {
    /// Create a new distinct result by materializing and deduplicating rows
    ///
    /// OPTIMIZATION: Uses direct value hashing instead of string serialization.
    /// This reduces allocations from O(rows × columns) to O(unique_rows).
    pub fn new(inner: Box<dyn QueryResult>) -> Self {
        Self::with_column_count(inner, None)
    }

    /// Create a distinct result that only considers the first `distinct_columns` columns
    /// for uniqueness comparison. This is used when ORDER BY references columns not in SELECT.
    ///
    /// For example: SELECT DISTINCT a FROM t ORDER BY b
    /// - The result has columns [a, b] for sorting
    /// - But distinctness should only compare column [a]
    /// - After DISTINCT, extra columns are removed
    pub fn with_column_count(
        mut inner: Box<dyn QueryResult>,
        distinct_columns: Option<usize>,
    ) -> Self {
        use std::hash::{Hash, Hasher};

        let columns = inner.columns().to_vec();
        let num_distinct_cols = distinct_columns.unwrap_or(columns.len());

        // OPTIMIZATION: Use hash map with collision handling for correct DISTINCT behavior
        // Maps: hash -> list of indices into result vec (handles hash collisions)
        // Hash values directly instead of converting to strings (600K fewer allocations for 100K rows)
        let mut hash_to_indices: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
        let mut rows = Vec::new();

        while inner.next() {
            // Hash only the columns that matter for distinctness
            let mut hasher = FxHasher::default();
            let row = inner.row();
            for value in row.as_slice().iter().take(num_distinct_cols) {
                value.hash(&mut hasher);
            }
            let hash = hasher.finish();

            // Check if this exact row already exists (handle hash collisions)
            let indices = hash_to_indices.entry(hash).or_default();
            let is_duplicate = indices.iter().any(|&idx| {
                // Compare only the first num_distinct_cols columns
                let existing_row: &Row = &rows[idx];
                for i in 0..num_distinct_cols {
                    match (row.get(i), existing_row.get(i)) {
                        (Some(v1), Some(v2)) if v1 == v2 => continue,
                        (None, None) => continue,
                        _ => return false,
                    }
                }
                true
            });

            if !is_duplicate {
                indices.push(rows.len());
                rows.push(inner.take_row());
            }
        }

        let memory_result = ExecutorMemoryResult::new(columns, rows);

        Self {
            inner: memory_result,
        }
    }
}

impl QueryResult for DistinctResult {
    fn columns(&self) -> &[String] {
        self.inner.columns()
    }

    fn next(&mut self) -> bool {
        self.inner.next()
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        self.inner.scan(dest)
    }

    fn row(&self) -> &Row {
        self.inner.row()
    }

    fn take_row(&mut self) -> Row {
        self.inner.take_row()
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        0
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Aliased result that renames columns
pub struct AliasedResult {
    /// Underlying result
    inner: Box<dyn QueryResult>,
    /// Cached aliased columns
    aliased_columns: Vec<String>,
}

impl AliasedResult {
    /// Create a new aliased result
    pub fn new(inner: Box<dyn QueryResult>, aliases: FxHashMap<String, String>) -> Self {
        let original_columns = inner.columns().to_vec();
        let aliased_columns = original_columns
            .iter()
            .map(|col| {
                // Find if this column has an alias (reverse lookup)
                for (alias, original) in &aliases {
                    if col == original {
                        return alias.clone();
                    }
                }
                col.clone()
            })
            .collect();

        Self {
            inner,
            aliased_columns,
        }
    }
}

impl QueryResult for AliasedResult {
    fn columns(&self) -> &[String] {
        &self.aliased_columns
    }

    fn next(&mut self) -> bool {
        self.inner.next()
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        self.inner.scan(dest)
    }

    fn row(&self) -> &Row {
        self.inner.row()
    }

    fn take_row(&mut self) -> Row {
        self.inner.take_row()
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        self.inner.rows_affected()
    }

    fn last_insert_id(&self) -> i64 {
        self.inner.last_insert_id()
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        // Apply additional aliases
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Projected result that removes extra columns (e.g., ORDER BY columns not in SELECT)
///
/// This result type projects rows to only include the first N columns,
/// removing any extra columns that were added for sorting purposes.
pub struct ProjectedResult {
    inner: Box<dyn QueryResult>,
    /// Number of columns to keep
    keep_columns: usize,
    /// Cached projected columns
    projected_columns: Vec<String>,
    /// Cached projected row
    current_row: Row,
}

impl ProjectedResult {
    /// Create a new projected result
    pub fn new(inner: Box<dyn QueryResult>, keep_columns: usize) -> Self {
        let projected_columns: Vec<String> =
            inner.columns().iter().take(keep_columns).cloned().collect();

        Self {
            inner,
            keep_columns,
            projected_columns,
            current_row: Row::new(),
        }
    }
}

impl QueryResult for ProjectedResult {
    fn columns(&self) -> &[String] {
        &self.projected_columns
    }

    fn next(&mut self) -> bool {
        if self.inner.next() {
            // Project the row to keep only the first N columns
            // OPTIMIZATION: Reuse the current_row's Vec allocation instead of creating new
            let full_row = self.inner.row();
            self.current_row.clear();
            self.current_row.reserve(self.keep_columns);
            for i in 0..self.keep_columns {
                self.current_row
                    .push(full_row.get(i).cloned().unwrap_or(Value::null_unknown()));
            }
            true
        } else {
            false
        }
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        for (i, val) in dest.iter_mut().enumerate().take(self.keep_columns) {
            *val = self
                .current_row
                .get(i)
                .cloned()
                .unwrap_or(Value::null_unknown());
        }
        Ok(())
    }

    fn row(&self) -> &Row {
        &self.current_row
    }

    fn take_row(&mut self) -> Row {
        std::mem::take(&mut self.current_row)
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        0
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Result that wraps a Scanner and implements QueryResult for streaming
///
/// This allows scanners to be used with the streaming filter/limit wrappers
/// without materializing all rows upfront.
pub struct ScannerResult {
    /// The underlying scanner
    scanner: Box<dyn crate::storage::traits::Scanner>,
    /// Column names
    columns: Vec<String>,
    /// Current row (cloned from scanner since scanner.row() returns a reference)
    current_row: Row,
    /// Whether we have a valid current row
    has_current: bool,
}

impl ScannerResult {
    /// Create a new scanner result
    pub fn new(scanner: Box<dyn crate::storage::traits::Scanner>, columns: Vec<String>) -> Self {
        Self {
            scanner,
            columns,
            current_row: Row::new(),
            has_current: false,
        }
    }
}

impl QueryResult for ScannerResult {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next(&mut self) -> bool {
        if self.scanner.next() {
            self.current_row = self.scanner.take_row();
            self.has_current = true;
            true
        } else {
            self.has_current = false;
            false
        }
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        if !self.has_current {
            return Err(crate::core::Error::internal(
                "scan() called without successful next()",
            ));
        }
        if dest.len() != self.current_row.len() {
            return Err(crate::core::Error::internal(format!(
                "scan destination has {} values but row has {} columns",
                dest.len(),
                self.current_row.len()
            )));
        }
        for (i, value) in self.current_row.iter().enumerate() {
            dest[i] = value.clone();
        }
        Ok(())
    }

    fn row(&self) -> &Row {
        if !self.has_current {
            panic!("row() called without successful next()");
        }
        &self.current_row
    }

    fn take_row(&mut self) -> Row {
        if !self.has_current {
            panic!("take_row() called without successful next()");
        }
        // Move out the current row, replace with empty
        std::mem::take(&mut self.current_row)
    }

    fn close(&mut self) -> Result<()> {
        self.scanner.close()
    }

    fn rows_affected(&self) -> i64 {
        0
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

/// Streaming filter result that applies WHERE clause evaluation row-by-row
///
/// This wraps a ScannerResult and applies filtering using the existing FilteredResult
/// mechanics, but the scanner itself streams rows from storage.
pub type StreamingFilterResult = FilteredResult;

/// Streaming projection result that projects columns row-by-row
///
/// This allows streaming projection without materializing all rows into a Vec.
/// Uses simple column index-based projection for performance.
pub struct StreamingProjectionResult {
    /// Underlying result
    inner: Box<dyn QueryResult>,
    /// Column indices to project (from source to output)
    column_indices: Vec<usize>,
    /// Output column names
    output_columns: Vec<String>,
    /// Current projected row
    current_row: Row,
}

impl StreamingProjectionResult {
    /// Create a new streaming projection result
    ///
    /// # Arguments
    /// * `inner` - The source result
    /// * `column_indices` - Indices of columns to keep from the source
    /// * `output_columns` - Names for the output columns
    pub fn new(
        inner: Box<dyn QueryResult>,
        column_indices: Vec<usize>,
        output_columns: Vec<String>,
    ) -> Self {
        Self {
            inner,
            column_indices,
            output_columns,
            current_row: Row::new(),
        }
    }
}

impl QueryResult for StreamingProjectionResult {
    fn columns(&self) -> &[String] {
        &self.output_columns
    }

    fn next(&mut self) -> bool {
        if self.inner.next() {
            // OPTIMIZATION: Reuse the current_row's Vec allocation instead of creating new
            let source_row = self.inner.row();
            self.current_row.clear();
            self.current_row.reserve(self.column_indices.len());
            for &idx in &self.column_indices {
                self.current_row.push(
                    source_row
                        .get(idx)
                        .cloned()
                        .unwrap_or(Value::null_unknown()),
                );
            }
            true
        } else {
            false
        }
    }

    fn scan(&self, dest: &mut [Value]) -> Result<()> {
        if dest.len() != self.current_row.len() {
            return Err(crate::core::Error::internal(format!(
                "scan destination has {} values but row has {} columns",
                dest.len(),
                self.current_row.len()
            )));
        }
        for (i, value) in self.current_row.iter().enumerate() {
            dest[i] = value.clone();
        }
        Ok(())
    }

    fn row(&self) -> &Row {
        &self.current_row
    }

    fn take_row(&mut self) -> Row {
        std::mem::take(&mut self.current_row)
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    fn rows_affected(&self) -> i64 {
        0
    }

    fn last_insert_id(&self) -> i64 {
        0
    }

    fn with_aliases(self: Box<Self>, aliases: FxHashMap<String, String>) -> Box<dyn QueryResult> {
        Box::new(AliasedResult::new(self, aliases))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exec_result() {
        let result = ExecResult::new(5, 10);
        assert_eq!(result.rows_affected(), 5);
        assert_eq!(result.last_insert_id(), 10);
        assert_eq!(result.columns().len(), 0);
    }

    #[test]
    fn test_exec_result_empty() {
        let result = ExecResult::empty();
        assert_eq!(result.rows_affected(), 0);
        assert_eq!(result.last_insert_id(), 0);
    }

    #[test]
    fn test_memory_result() {
        let columns = vec!["id".to_string(), "name".to_string()];
        let rows = vec![
            Row::from_values(vec![Value::Integer(1), Value::text("Alice")]),
            Row::from_values(vec![Value::Integer(2), Value::text("Bob")]),
        ];

        let mut result = ExecutorMemoryResult::new(columns, rows);
        assert_eq!(result.columns().len(), 2);
        assert_eq!(result.row_count(), 2);

        // First row
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(1)));

        // Second row
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(2)));

        // No more rows
        assert!(!result.next());
    }

    #[test]
    fn test_filtered_result() {
        let columns = vec!["id".to_string(), "value".to_string()];
        let rows = vec![
            Row::from_values(vec![Value::Integer(1), Value::Integer(10)]),
            Row::from_values(vec![Value::Integer(2), Value::Integer(20)]),
            Row::from_values(vec![Value::Integer(3), Value::Integer(30)]),
        ];

        let inner = Box::new(ExecutorMemoryResult::new(columns, rows));

        // Filter for value > 15
        let predicate = Box::new(|row: &Row| {
            if let Some(Value::Integer(v)) = row.get(1) {
                *v > 15
            } else {
                false
            }
        });

        let mut result = FilteredResult::new(inner, predicate);

        // Should get rows with value 20 and 30
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(2)));

        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(3)));

        assert!(!result.next());
    }

    #[test]
    fn test_limited_result() {
        let columns = vec!["id".to_string()];
        let rows = vec![
            Row::from_values(vec![Value::Integer(1)]),
            Row::from_values(vec![Value::Integer(2)]),
            Row::from_values(vec![Value::Integer(3)]),
            Row::from_values(vec![Value::Integer(4)]),
            Row::from_values(vec![Value::Integer(5)]),
        ];

        let inner = Box::new(ExecutorMemoryResult::new(columns.clone(), rows.clone()));
        let mut result = LimitedResult::new(inner, Some(2), 1);

        // Skip first row (offset 1), then take 2 rows
        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(2)));

        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(3)));

        assert!(!result.next()); // Limit reached
    }

    #[test]
    fn test_ordered_result() {
        let columns = vec!["id".to_string(), "value".to_string()];
        let rows = vec![
            Row::from_values(vec![Value::Integer(3), Value::Integer(30)]),
            Row::from_values(vec![Value::Integer(1), Value::Integer(10)]),
            Row::from_values(vec![Value::Integer(2), Value::Integer(20)]),
        ];

        let inner = Box::new(ExecutorMemoryResult::new(columns, rows));

        // Sort by id ascending
        let mut result = OrderedResult::new(inner, |a, b| {
            let a_id = a.get(0).and_then(|v| v.as_int64()).unwrap_or(0);
            let b_id = b.get(0).and_then(|v| v.as_int64()).unwrap_or(0);
            a_id.cmp(&b_id)
        });

        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(1)));

        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(2)));

        assert!(result.next());
        assert_eq!(result.row().get(0), Some(&Value::Integer(3)));

        assert!(!result.next());
    }

    #[test]
    fn test_distinct_result() {
        let columns = vec!["name".to_string()];
        let rows = vec![
            Row::from_values(vec![Value::text("Alice")]),
            Row::from_values(vec![Value::text("Bob")]),
            Row::from_values(vec![Value::text("Alice")]), // Duplicate
            Row::from_values(vec![Value::text("Charlie")]),
        ];

        let inner = Box::new(ExecutorMemoryResult::new(columns, rows));
        let mut result = DistinctResult::new(inner);

        let mut names = Vec::new();
        while result.next() {
            if let Some(Value::Text(name)) = result.row().get(0) {
                names.push(name.to_string());
            }
        }

        assert_eq!(names.len(), 3);
        assert!(names.contains(&"Alice".to_string()));
        assert!(names.contains(&"Bob".to_string()));
        assert!(names.contains(&"Charlie".to_string()));
    }

    #[test]
    fn test_aliased_result() {
        let columns = vec!["id".to_string(), "name".to_string()];
        let rows = vec![Row::from_values(vec![
            Value::Integer(1),
            Value::text("Alice"),
        ])];

        let inner = Box::new(ExecutorMemoryResult::new(columns, rows));

        let mut aliases = FxHashMap::default();
        aliases.insert("user_name".to_string(), "name".to_string());

        let mut result = AliasedResult::new(inner, aliases);

        assert_eq!(result.columns(), &["id", "user_name"]);

        assert!(result.next());
        assert_eq!(result.row().get(1), Some(&Value::text("Alice")));
    }
}
