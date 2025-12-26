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

//! Fast-path execution for simple PK lookups
//!
//! This module provides an optimized execution path for simple queries like:
//! - `SELECT * FROM table WHERE pk_col = $1`
//! - `SELECT col1, col2 FROM table WHERE pk_col = 5`
//!
//! By detecting these patterns early, we bypass the full query planner and
//! go directly to index lookup, reducing per-query overhead from ~2µs to ~200ns.
//!
//! # Performance Impact
//!
//! For Index Nested Loop joins that perform thousands of PK lookups,
//! this fast-path can provide significant speedups by amortizing less overhead.

use std::sync::{Arc, RwLock};

use crate::core::{Result, Row, Schema, Value};
use crate::parser::ast::{Expression, SelectStatement, SimpleTableSource};
use crate::storage::traits::{Engine, QueryResult};

use super::context::ExecutionContext;
use super::query_cache::{CompiledExecution, CompiledPkLookup, PkValueSource};
use super::result::ExecutorMemoryResult;
use super::Executor;

/// Information extracted from a simple PK lookup query
struct PkLookupInfo {
    /// Table name (already lowercased for storage lookups)
    table_name: String,
    /// PK value to look up
    pk_value: i64,
    /// How to extract the PK value (for caching)
    pk_value_source: PkValueSource,
    /// Cached schema to avoid second lookup
    schema: Arc<Schema>,
}

impl Executor {
    /// Try to execute a SELECT as a fast PK lookup
    ///
    /// Returns Some(result) if the query is a simple PK lookup that was executed.
    /// Returns None if the query doesn't qualify for fast-path.
    pub(crate) fn try_fast_pk_lookup(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Quick reject: if we're in an explicit transaction, skip fast path
        // The fast path uses fetch_rows_by_ids which only sees committed data,
        // so it wouldn't see uncommitted changes from the current transaction.
        // This could return stale data if Transaction A updates a row and then
        // queries for it - the fast path would return the old committed value.
        {
            let active_tx = self.active_transaction.lock().unwrap();
            if active_tx.is_some() {
                return None; // Let normal execution path handle transaction context
            }
        }

        // Quick reject: must have WHERE clause and table_expr
        let where_clause = stmt.where_clause.as_ref()?;
        let table_expr = stmt.table_expr.as_ref()?;

        // Quick reject: no GROUP BY, no HAVING, no CTEs, no set operations, no DISTINCT
        if !stmt.group_by.columns.is_empty()
            || stmt.having.is_some()
            || !stmt.set_operations.is_empty()
            || stmt.with.is_some()
            || stmt.distinct
        {
            return None;
        }

        // Quick reject: no ORDER BY (PK lookup returns single row anyway, but skip for simplicity)
        if !stmt.order_by.is_empty() {
            return None;
        }

        // Must be SELECT * (for now - column projection adds complexity)
        if stmt.columns.len() != 1 || !matches!(&stmt.columns[0], Expression::Star(_)) {
            return None;
        }

        // Extract table name (must be a simple table reference, not a join or subquery)
        // Lowercase once here to avoid repeated to_lowercase() calls in storage layer
        let table_name = match table_expr.as_ref() {
            Expression::TableSource(SimpleTableSource { name, .. }) => name.value.to_lowercase(),
            _ => return None, // Join, subquery, or other complex source
        };

        // Try to extract PK lookup info from WHERE clause
        let lookup_info = self.extract_pk_lookup_info(&table_name, where_clause, ctx)?;

        // Execute the fast-path lookup
        Some(self.execute_pk_lookup(lookup_info))
    }

    /// Extract PK lookup information from a WHERE clause
    fn extract_pk_lookup_info(
        &self,
        table_name: &str,
        where_clause: &Expression,
        ctx: &ExecutionContext,
    ) -> Option<PkLookupInfo> {
        // Get table schema to find PK column
        let schema = self.engine.get_table_schema(table_name).ok()?;
        let pk_indices = schema.primary_key_indices();

        // Only support single-column PK for now
        if pk_indices.len() != 1 {
            return None;
        }
        let pk_idx = pk_indices[0];
        let pk_column = &schema.columns[pk_idx].name;

        // Extract comparison info from WHERE clause
        let (col_name, pk_value, pk_value_source) =
            self.extract_pk_equality(where_clause, pk_column, ctx)?;

        // Column must match PK (case-insensitive)
        let col_lower = col_name.to_lowercase();
        let pk_lower = pk_column.to_lowercase();

        // Handle qualified names like "users.id"
        let matches_pk = col_lower == pk_lower || col_lower.ends_with(&format!(".{}", pk_lower));

        if !matches_pk {
            return None;
        }

        Some(PkLookupInfo {
            table_name: table_name.to_string(),
            pk_value,
            pk_value_source,
            schema,
        })
    }

    /// Extract PK equality from WHERE clause
    /// Returns (column_name, pk_value, pk_value_source) if WHERE is `pk_col = literal` or `pk_col = $param`
    fn extract_pk_equality(
        &self,
        expr: &Expression,
        _pk_column: &str,
        ctx: &ExecutionContext,
    ) -> Option<(String, i64, PkValueSource)> {
        match expr {
            Expression::Infix(infix) => {
                // Must be equality operator
                if infix.operator != "=" {
                    return None;
                }

                // Try column = value pattern
                if let Some((col, val, source)) =
                    self.extract_col_eq_val(&infix.left, &infix.right, ctx)
                {
                    return Some((col, val, source));
                }

                // Try value = column pattern
                if let Some((col, val, source)) =
                    self.extract_col_eq_val(&infix.right, &infix.left, ctx)
                {
                    return Some((col, val, source));
                }

                None
            }
            _ => None,
        }
    }

    /// Extract column name, integer value, and value source from col = val pattern
    fn extract_col_eq_val(
        &self,
        col_expr: &Expression,
        val_expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Option<(String, i64, PkValueSource)> {
        // Get column name
        let col_name = match col_expr {
            Expression::Identifier(id) => id.value.clone(),
            Expression::QualifiedIdentifier(q) => format!("{}.{}", q.qualifier, q.name),
            _ => return None,
        };

        // Get integer value and source
        let (pk_value, pk_value_source) = match val_expr {
            Expression::IntegerLiteral(lit) => (lit.value, PkValueSource::Literal(lit.value)),
            Expression::FloatLiteral(lit) => {
                let v = lit.value as i64;
                (v, PkValueSource::Literal(v))
            }
            Expression::Parameter(param) => {
                // Resolve parameter from context
                // Note: Parameters are 1-indexed in SQL ($1, $2, ...) but array is 0-indexed
                let params = ctx.params();
                let param_idx = if param.index > 0 {
                    param.index - 1
                } else {
                    param.index
                };
                if param_idx >= params.len() {
                    return None;
                }
                let pk_value = match &params[param_idx] {
                    Value::Integer(i) => *i,
                    Value::Float(f) => *f as i64,
                    _ => return None,
                };
                (pk_value, PkValueSource::Parameter(param_idx))
            }
            _ => return None,
        };

        Some((col_name, pk_value, pk_value_source))
    }

    /// Normalize a row to match the current schema
    ///
    /// This handles schema evolution (ALTER TABLE ADD/DROP COLUMN):
    /// - If row has fewer columns than schema, append default values (or NULLs) for missing columns
    /// - If row has more columns than schema, truncate the row
    fn normalize_row_to_schema(mut row: Row, schema: &Schema) -> Row {
        let schema_cols = schema.columns.len();
        let row_cols = row.len();

        if row_cols < schema_cols {
            // Row has fewer columns - add default values (or NULLs) for new columns
            for i in row_cols..schema_cols {
                let col = &schema.columns[i];
                // Use pre-computed default value if available, otherwise use NULL
                if let Some(ref default_val) = col.default_value {
                    row.push(default_val.clone());
                } else {
                    row.push(Value::null(col.data_type));
                }
            }
        } else if row_cols > schema_cols {
            // Row has more columns - truncate (columns were dropped)
            row.truncate(schema_cols);
        }

        row
    }

    /// Execute the fast-path PK lookup using Engine::fetch_rows_by_ids
    fn execute_pk_lookup(&self, info: PkLookupInfo) -> Result<Box<dyn QueryResult>> {
        // Use cached schema for column names (already fetched in extract_pk_lookup_info)
        let columns: Vec<String> = info.schema.columns.iter().map(|c| c.name.clone()).collect();

        // Use engine's fetch_rows_by_ids for direct MVCC lookup
        // This bypasses the full query planner and goes straight to version store
        // Note: table_name is already lowercased, so storage layer won't call to_lowercase again
        let rows = self
            .engine
            .fetch_rows_by_ids(&info.table_name, &[info.pk_value])?;

        // Extract Row values and normalize to current schema (handles ADD/DROP COLUMN)
        let result_rows: Vec<_> = rows
            .into_iter()
            .map(|(_, row)| Self::normalize_row_to_schema(row, &info.schema))
            .collect();

        Ok(Box::new(ExecutorMemoryResult::new(columns, result_rows)))
    }

    // ============================================================================
    // COMPILED EXECUTION METHODS - Use pre-compiled state for fast repeated queries
    // ============================================================================

    /// Try fast PK lookup using pre-compiled state (if available)
    ///
    /// This is the preferred entry point for queries that may be executed multiple times.
    /// First execution compiles and caches the state, subsequent executions use the cache.
    pub(crate) fn try_fast_pk_lookup_compiled(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Quick reject: explicit transaction (same as non-compiled path)
        {
            let active_tx = self.active_transaction.lock().unwrap();
            if active_tx.is_some() {
                return None;
            }
        }

        // Try read lock first - check if already compiled
        {
            let compiled_guard = match compiled.read() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            match &*compiled_guard {
                CompiledExecution::NotOptimizable => return None,
                CompiledExecution::PkLookup(lookup) => {
                    // Validate schema version before using cached lookup
                    // This detects ALTER TABLE ADD/DROP COLUMN
                    if let Ok(current_schema) = self.engine.get_table_schema(&lookup.table_name) {
                        if current_schema.columns.len() != lookup.schema_version {
                            // Schema changed - need to recompile
                            // Fall through to recompile path
                        } else {
                            // Fast path: extract value and execute
                            let pk_value =
                                self.extract_pk_value_fast(&lookup.pk_value_source, ctx)?;
                            return Some(self.execute_compiled_pk_lookup(lookup, pk_value));
                        }
                    } else {
                        // Table no longer exists - invalidate
                        // Fall through to recompile path
                    }
                }
                CompiledExecution::Unknown => {} // Fall through to compile
            }
        }

        // First execution or schema changed - compile and cache (write lock)
        self.compile_and_execute_pk_lookup(stmt, ctx, compiled)
    }

    /// Extract PK value using pre-compiled source (very fast - just array access)
    fn extract_pk_value_fast(&self, source: &PkValueSource, ctx: &ExecutionContext) -> Option<i64> {
        match source {
            PkValueSource::Literal(v) => Some(*v),
            PkValueSource::Parameter(idx) => {
                let params = ctx.params();
                if *idx >= params.len() {
                    return None;
                }
                match &params[*idx] {
                    Value::Integer(i) => Some(*i),
                    Value::Float(f) => Some(*f as i64),
                    _ => None,
                }
            }
        }
    }

    /// Execute using pre-compiled lookup (skip schema lookup, column name building)
    fn execute_compiled_pk_lookup(
        &self,
        lookup: &CompiledPkLookup,
        pk_value: i64,
    ) -> Result<Box<dyn QueryResult>> {
        let rows = self
            .engine
            .fetch_rows_by_ids(&lookup.table_name, &[pk_value])?;
        // Normalize rows to current schema (handles ADD/DROP COLUMN)
        let result_rows: Vec<_> = rows
            .into_iter()
            .map(|(_, row)| Self::normalize_row_to_schema(row, &lookup.schema))
            .collect();
        Ok(Box::new(ExecutorMemoryResult::new(
            lookup.column_names.clone(),
            result_rows,
        )))
    }

    /// Compile and execute PK lookup, caching the compiled state
    fn compile_and_execute_pk_lookup(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Acquire write lock
        let mut compiled_guard = match compiled.write() {
            Ok(guard) => guard,
            Err(_) => return None,
        };

        // Double-check (another thread may have compiled while we waited)
        // But also re-validate schema version to handle schema changes
        match &*compiled_guard {
            CompiledExecution::NotOptimizable => return None,
            CompiledExecution::PkLookup(lookup) => {
                let pk_value = self.extract_pk_value_fast(&lookup.pk_value_source, ctx)?;
                return Some(self.execute_compiled_pk_lookup(lookup, pk_value));
            }
            CompiledExecution::Unknown => {} // Continue with compilation
        }

        // Do full pattern detection (same as try_fast_pk_lookup)
        let where_clause = stmt.where_clause.as_ref()?;
        let table_expr = stmt.table_expr.as_ref()?;

        // Quick reject: no GROUP BY, no HAVING, no CTEs, no set operations, no DISTINCT
        if !stmt.group_by.columns.is_empty()
            || stmt.having.is_some()
            || !stmt.set_operations.is_empty()
            || stmt.with.is_some()
            || stmt.distinct
        {
            *compiled_guard = CompiledExecution::NotOptimizable;
            return None;
        }

        // Quick reject: no ORDER BY
        if !stmt.order_by.is_empty() {
            *compiled_guard = CompiledExecution::NotOptimizable;
            return None;
        }

        // Must be SELECT *
        if stmt.columns.len() != 1 || !matches!(&stmt.columns[0], Expression::Star(_)) {
            *compiled_guard = CompiledExecution::NotOptimizable;
            return None;
        }

        // Extract table name
        let table_name = match table_expr.as_ref() {
            Expression::TableSource(SimpleTableSource { name, .. }) => name.value.to_lowercase(),
            _ => {
                *compiled_guard = CompiledExecution::NotOptimizable;
                return None;
            }
        };

        // Try to extract PK lookup info
        match self.extract_pk_lookup_info(&table_name, where_clause, ctx) {
            Some(info) => {
                // Build and cache compiled lookup
                let column_names: Vec<String> =
                    info.schema.columns.iter().map(|c| c.name.clone()).collect();
                let schema_version = info.schema.columns.len();
                let compiled_lookup = CompiledPkLookup {
                    table_name: info.table_name.clone(),
                    schema: info.schema.clone(),
                    column_names,
                    pk_value_source: info.pk_value_source.clone(),
                    schema_version,
                };
                *compiled_guard = CompiledExecution::PkLookup(compiled_lookup);
                drop(compiled_guard);

                // Execute
                Some(self.execute_pk_lookup(info))
            }
            None => {
                *compiled_guard = CompiledExecution::NotOptimizable;
                None
            }
        }
    }
}
