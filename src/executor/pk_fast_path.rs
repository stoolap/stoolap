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

use std::sync::RwLock;

use crate::common::{CompactArc, SmartString};
use crate::core::{Result, Row, RowVec, Schema, Value};
use crate::parser::ast::{Expression, SelectStatement};
use crate::storage::traits::{Engine, QueryResult};

use super::context::ExecutionContext;
use super::query_cache::{CompiledExecution, CompiledPkLookup, PkValueSource};
use super::result::ExecutorResult;
use super::Executor;

/// Information extracted from a simple PK lookup query
struct PkLookupInfo {
    /// Table name (already lowercased for storage lookups)
    table_name: String,
    /// PK value to look up
    pk_value: i64,
    /// Cached schema to avoid second lookup
    schema: CompactArc<Schema>,
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
        // Use try_lock for faster rejection under contention.
        {
            let active_tx = match self.active_transaction.try_lock() {
                Ok(guard) => guard,
                Err(_) => return None, // Lock contention - fall back to normal path
            };
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

        // Quick reject: LIMIT 0 or OFFSET would change the result
        if !Self::pk_fast_path_limit_ok(stmt) {
            return None;
        }

        // Must be SELECT * (for now - column projection adds complexity)
        if stmt.columns.len() != 1 || !matches!(&stmt.columns[0], Expression::Star(_)) {
            return None;
        }

        // Extract table name (must be a simple table reference, not a join or subquery)
        // Use pre-computed lowercase from Identifier (avoids allocation and case conversion)
        let table_name: &str = match table_expr.as_ref() {
            Expression::TableSource(ts) => ts.name.value_lower.as_str(),
            _ => return None, // Join, subquery, or other complex source
        };

        // Try to extract PK lookup info from WHERE clause
        let lookup_info = self.extract_pk_lookup_info(table_name, where_clause, ctx)?;

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
        let (pk_value_source, schema) =
            self.extract_pk_lookup_structure(table_name, where_clause)?;
        let pk_value = self.extract_pk_value_fast(&pk_value_source, ctx)?;
        Some(PkLookupInfo {
            table_name: table_name.to_string(),
            pk_value,
            schema,
        })
    }

    /// Structural half of PK-lookup detection: schema/PK/equality shape,
    /// with parameter values left unresolved.
    fn extract_pk_lookup_structure(
        &self,
        table_name: &str,
        where_clause: &Expression,
    ) -> Option<(PkValueSource, CompactArc<Schema>)> {
        // Get table schema to find PK column
        let schema = self.engine.get_table_schema(table_name).ok()?;
        let pk_indices = schema.primary_key_indices();

        // Only support single-column PK for now
        if pk_indices.len() != 1 {
            return None;
        }
        let pk_idx = pk_indices[0];

        // Extract comparison structure from WHERE clause
        let (col_name, pk_value_source) = Self::extract_pk_equality_source(where_clause)?;

        // Column must match PK (case-insensitive)
        // Use schema's pre-computed lowercase for pk_column
        let col_lower = col_name.to_lowercase();
        let pk_lower = &schema.columns[pk_idx].name_lower;

        // Handle qualified names like "users.id"
        let matches_pk = col_lower == *pk_lower || col_lower.ends_with(&format!(".{}", pk_lower));

        if !matches_pk {
            return None;
        }

        Some((pk_value_source, schema))
    }

    /// The PK fast path returns at most one row, so a literal LIMIT >= 1
    /// with no OFFSET cannot change the result. Anything else (LIMIT 0,
    /// any OFFSET, a parameter or expression limit) must take the full
    /// execution path.
    fn pk_fast_path_limit_ok(stmt: &SelectStatement) -> bool {
        if stmt.offset.is_some() {
            return false;
        }
        match stmt.limit.as_deref() {
            None => true,
            Some(Expression::IntegerLiteral(lit)) => lit.value >= 1,
            Some(_) => false,
        }
    }

    /// Extract PK equality structure from WHERE clause without resolving
    /// parameter values. Returns (column_name, pk_value_source) if WHERE is
    /// `pk_col = literal` or `pk_col = $param`.
    fn extract_pk_equality_source(expr: &Expression) -> Option<(String, PkValueSource)> {
        match expr {
            Expression::Infix(infix) => {
                // Must be equality operator
                if infix.operator != "=" {
                    return None;
                }
                Self::extract_col_eq_source(&infix.left, &infix.right)
                    .or_else(|| Self::extract_col_eq_source(&infix.right, &infix.left))
            }
            _ => None,
        }
    }

    /// Extract column name and value source from a col = val pattern.
    /// Purely structural: parameter values are resolved per execution by
    /// `extract_pk_value_fast`, so a wrongly-typed parameter this time must
    /// not make the statement look structurally non-optimizable.
    fn extract_col_eq_source(
        col_expr: &Expression,
        val_expr: &Expression,
    ) -> Option<(String, PkValueSource)> {
        // Get column name
        let col_name = match col_expr {
            Expression::Identifier(id) => id.value.to_string(),
            Expression::QualifiedIdentifier(q) => format!("{}.{}", q.qualifier, q.name),
            _ => return None,
        };

        let source = match val_expr {
            Expression::IntegerLiteral(lit) => PkValueSource::Literal(lit.value),
            Expression::FloatLiteral(lit) => {
                // Only lossless float keys qualify; see lossless_float_key
                PkValueSource::Literal(Self::lossless_float_key(lit.value)?)
            }
            Expression::Parameter(param) => {
                // Named parameters (e.g., :name); positional ($1, $2, ...)
                // are 1-indexed, the array is 0-indexed
                if param.name.starts_with(':') {
                    PkValueSource::NamedParameter(SmartString::new(&param.name[1..]))
                } else if param.index > 0 {
                    PkValueSource::Parameter(param.index - 1)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        Some((col_name, source))
    }

    /// Normalize a row to match the current schema
    ///
    /// This handles schema evolution (ALTER TABLE ADD/DROP COLUMN):
    /// - If row has fewer columns than schema, append default values (or NULLs) for missing columns
    /// - If row has more columns than schema, truncate the row
    #[inline]
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
        // Use cached schema for column names - Arc clone is O(1)
        let columns = info.schema.column_names_arc();

        // Use engine's fetch_rows_by_ids for direct MVCC lookup
        // This bypasses the full query planner and goes straight to version store
        // Note: table_name is already lowercased, so storage layer won't call to_lowercase again
        let rows = self
            .engine
            .fetch_rows_by_ids(&info.table_name, &[info.pk_value])?;

        // Extract Row values and normalize to current schema (handles ADD/DROP COLUMN)
        let result_rows: RowVec = rows
            .into_iter()
            .enumerate()
            .map(|(i, (_, row))| (i as i64, Self::normalize_row_to_schema(row, &info.schema)))
            .collect();

        Ok(Box::new(ExecutorResult::with_arc_columns(
            columns,
            result_rows,
        )))
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
        // Caller (try_compiled_fast_paths) guarantees no explicit transaction
        // is active; this path reads committed state only.

        // Try read lock first - check if already compiled
        {
            let compiled_guard = match compiled.read() {
                Ok(guard) => guard,
                Err(_) => return None,
            };
            match &*compiled_guard {
                CompiledExecution::NotOptimizable(epoch)
                    if self.engine.schema_epoch() == *epoch =>
                {
                    return None
                }
                CompiledExecution::PkLookup(lookup) => {
                    // Fast validation using schema epoch (~1ns vs ~7ns for HashMap lookup)
                    // If epoch matches, no DDL has occurred since compilation
                    if self.engine.schema_epoch() == lookup.cached_epoch {
                        // Fast path: extract value and execute
                        let pk_value = self.extract_pk_value_fast(&lookup.pk_value_source, ctx)?;
                        return Some(self.execute_compiled_pk_lookup(lookup, pk_value));
                    }
                    // Epoch changed - some DDL occurred, need to recompile
                    // Fall through to recompile path
                }
                CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - fall through to recompile
                // These variants are for UPDATE/DELETE/INSERT/COUNT DISTINCT/COUNT(*) - not PK lookups
                CompiledExecution::PkUpdate(_)
                | CompiledExecution::PkDelete(_)
                | CompiledExecution::Insert(_)
                | CompiledExecution::CountDistinct(_)
                | CompiledExecution::CountStar(_) => return None,
            }
        }

        // First execution or schema changed - compile and cache (write lock)
        self.compile_and_execute_pk_lookup(stmt, ctx, compiled)
    }

    /// Extract PK value using pre-compiled source (very fast - just array access)
    fn extract_pk_value_fast(&self, source: &PkValueSource, ctx: &ExecutionContext) -> Option<i64> {
        match source {
            PkValueSource::NamedParameter(name) => match ctx.get_named_param(name)? {
                Value::Integer(i) => Some(*i),
                Value::Float(f) => Self::lossless_float_key(*f),
                _ => None,
            },
            _ => Self::extract_pk_value_from_slice(source, ctx.params()),
        }
    }

    /// A float key qualifies only when it converts to i64 without loss:
    /// truncating 5.5 to 5 would serve row 5 where correct execution
    /// matches nothing. None falls back to the standard path.
    ///
    /// The range check must happen BEFORE the cast: 2^63 saturates to
    /// i64::MAX, and i64::MAX rounds back to 2^63 as f64, so a naive
    /// round-trip equality check passes at the boundary.
    #[inline]
    pub(crate) fn lossless_float_key(f: f64) -> Option<i64> {
        const I64_MIN_F: f64 = -9_223_372_036_854_775_808.0; // -2^63, exact
        const I64_MAX_PLUS_1_F: f64 = 9_223_372_036_854_775_808.0; // 2^63, not representable as i64
        if !(I64_MIN_F..I64_MAX_PLUS_1_F).contains(&f) {
            return None; // Also rejects NaN and infinities
        }
        let v = f as i64;
        (v as f64 == f).then_some(v)
    }

    /// Extract PK value from params slice directly (avoids ExecutionContext overhead)
    #[inline]
    fn extract_pk_value_from_slice(source: &PkValueSource, params: &[Value]) -> Option<i64> {
        match source {
            PkValueSource::Literal(v) => Some(*v),
            PkValueSource::Parameter(idx) => {
                if *idx >= params.len() {
                    return None;
                }
                match &params[*idx] {
                    Value::Integer(i) => Some(*i),
                    Value::Float(f) => Self::lossless_float_key(*f),
                    _ => None,
                }
            }
            PkValueSource::NamedParameter(_) => None, // No ctx available in slice path
        }
    }

    /// Try fast PK lookup with borrowed params slice (avoids Arc allocation)
    pub(crate) fn try_fast_pk_lookup_with_params(
        &self,
        _stmt: &SelectStatement,
        params: &[Value],
        compiled: &RwLock<CompiledExecution>,
    ) -> Option<Result<Box<dyn QueryResult>>> {
        // Try read lock first - check if already compiled
        let compiled_guard = compiled.read().ok()?;
        match &*compiled_guard {
            CompiledExecution::NotOptimizable(_) => None,
            CompiledExecution::PkLookup(lookup) => {
                // Fast validation using schema epoch
                if self.engine.schema_epoch() == lookup.cached_epoch {
                    // Fast path: extract value from slice directly
                    let pk_value =
                        Self::extract_pk_value_from_slice(&lookup.pk_value_source, params)?;
                    Some(self.execute_compiled_pk_lookup(lookup, pk_value))
                } else {
                    // Epoch changed - need recompile, use normal path
                    None
                }
            }
            CompiledExecution::Unknown => None, // Not compiled yet, use normal path
            _ => None,
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
        // Pre-allocate with capacity 1 for single PK lookup (avoids realloc)
        let mut result_rows = RowVec::with_capacity(1);
        for (row_id, (_, row)) in rows.into_iter().enumerate() {
            result_rows.push((
                row_id as i64,
                Self::normalize_row_to_schema(row, &lookup.schema),
            ));
        }
        // Use Arc columns - O(1) clone since column_names is CompactArc<Vec<String>>
        Ok(Box::new(ExecutorResult::with_arc_columns(
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
            CompiledExecution::NotOptimizable(epoch) if self.engine.schema_epoch() == *epoch => {
                return None
            }
            CompiledExecution::PkLookup(lookup) => {
                // Re-validate epoch: another thread may have compiled before DDL
                if self.engine.schema_epoch() == lookup.cached_epoch {
                    let pk_value = self.extract_pk_value_fast(&lookup.pk_value_source, ctx)?;
                    return Some(self.execute_compiled_pk_lookup(lookup, pk_value));
                }
                // Epoch changed since last compilation - fall through to recompile
            }
            CompiledExecution::NotOptimizable(_) | CompiledExecution::Unknown => {} // Epoch changed or first run - recompile
            // These variants are for UPDATE/DELETE/INSERT/COUNT DISTINCT/COUNT(*) - not PK lookups
            CompiledExecution::PkUpdate(_)
            | CompiledExecution::PkDelete(_)
            | CompiledExecution::Insert(_)
            | CompiledExecution::CountDistinct(_)
            | CompiledExecution::CountStar(_) => return None,
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
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Quick reject: no ORDER BY
        if !stmt.order_by.is_empty() {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Quick reject: LIMIT 0 or OFFSET would change the result
        if !Self::pk_fast_path_limit_ok(stmt) {
            *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
            return None;
        }

        // Must be SELECT *
        // Don't set NotOptimizable here - other fast paths (like COUNT DISTINCT) may handle this
        if stmt.columns.len() != 1 || !matches!(&stmt.columns[0], Expression::Star(_)) {
            return None;
        }

        // Extract table name (use pre-computed lowercase)
        let table_name: &str = match table_expr.as_ref() {
            Expression::TableSource(ts) => ts.name.value_lower.as_str(),
            _ => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                return None;
            }
        };

        // Try to extract the PK lookup structure. Only a STRUCTURAL reject
        // may poison the slot: a value-dependent failure (e.g. a text or
        // NULL parameter this execution) must leave the compiled lookup in
        // place so later executions with an integer parameter still
        // fast-path.
        match self.extract_pk_lookup_structure(table_name, where_clause) {
            Some((pk_value_source, schema)) => {
                // Build and cache compiled lookup
                // Use schema's column_names_arc() directly - O(1) Arc clone on execution
                let column_names = schema.column_names_arc();
                let cached_epoch = self.engine.schema_epoch();
                let compiled_lookup = CompiledPkLookup {
                    table_name: SmartString::new(table_name),
                    schema: schema.clone(),
                    column_names,
                    pk_value_source: pk_value_source.clone(),
                    cached_epoch,
                };
                *compiled_guard = CompiledExecution::PkLookup(compiled_lookup);
                drop(compiled_guard);

                // Resolve this execution's value; on failure fall back to
                // the standard path (the stored PkLookup stays valid).
                let pk_value = self.extract_pk_value_fast(&pk_value_source, ctx)?;
                Some(self.execute_pk_lookup(PkLookupInfo {
                    table_name: table_name.to_string(),
                    pk_value,
                    schema,
                }))
            }
            None => {
                *compiled_guard = CompiledExecution::NotOptimizable(self.engine.schema_epoch());
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Executor;

    #[test]
    fn lossless_float_key_boundaries() {
        // Exact conversions pass
        assert_eq!(Executor::lossless_float_key(5.0), Some(5));
        assert_eq!(Executor::lossless_float_key(-5.0), Some(-5));
        assert_eq!(Executor::lossless_float_key(0.0), Some(0));
        // i64::MIN is exactly representable as f64
        assert_eq!(
            Executor::lossless_float_key(-9_223_372_036_854_775_808.0),
            Some(i64::MIN)
        );
        // Fractional values reject
        assert_eq!(Executor::lossless_float_key(5.5), None);
        assert_eq!(Executor::lossless_float_key(-0.5), None);
        // 2^63 saturates to i64::MAX under `as i64` and i64::MAX rounds
        // back to 2^63, so a naive round-trip check passes; the range
        // check must reject it
        assert_eq!(
            Executor::lossless_float_key(9_223_372_036_854_775_808.0),
            None
        );
        assert_eq!(Executor::lossless_float_key(1e19), None);
        assert_eq!(Executor::lossless_float_key(-1e19), None);
        // Non-finite rejects
        assert_eq!(Executor::lossless_float_key(f64::NAN), None);
        assert_eq!(Executor::lossless_float_key(f64::INFINITY), None);
        assert_eq!(Executor::lossless_float_key(f64::NEG_INFINITY), None);
    }
}
