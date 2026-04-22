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

//! Transaction API
//!
//! Provides ACID transaction support with the same ergonomic API as Database.
//!
//! # Examples
//!
//! ```ignore
//! use stoolap::Database;
//!
//! let db = Database::open("memory://")?;
//! db.execute("CREATE TABLE accounts (id INTEGER, balance INTEGER)", ())?;
//! db.execute("INSERT INTO accounts VALUES ($1, $2), ($3, $4)", (1, 1000, 2, 500))?;
//!
//! // Transfer money atomically
//! let mut tx = db.begin()?;
//! tx.execute("UPDATE accounts SET balance = balance - $1 WHERE id = $2", (100, 1))?;
//! tx.execute("UPDATE accounts SET balance = balance + $1 WHERE id = $2", (100, 2))?;
//! tx.commit()?;
//! ```

use std::sync::Arc;

use crate::api::params::{NamedParams, ParamVec};
use crate::core::{Error, Result, Row, Schema, Value};
use crate::executor::context::ExecutionContext;
use crate::executor::expression::ExpressionEval;
use crate::executor::Executor;
use crate::parser::ast::{Expression, Statement};
use crate::parser::Parser;
use crate::storage::expression::Expression as StorageExprTrait;
use crate::storage::traits::{QueryResult, WriteTransaction as StorageTransaction};

use super::database::{EngineEntry, FromValue};
use super::params::Params;
use super::rows::Rows;

/// Transaction represents a database transaction
///
/// Provides ACID guarantees for a series of database operations.
/// Must be explicitly committed or rolled back.
///
/// Holds an `Arc<EngineEntry>` (not just `Arc<MVCCEngine>`) so the
/// engine entry's strong count reflects this transaction. Without
/// that, `Database::close()` — which decides whether to close the
/// engine using `Arc::strong_count(&entry)` — could fire
/// `engine.close_engine()` while a transaction is alive, and the next
/// statement on the transaction would error with `EngineNotOpen`. The
/// engine itself is reachable via `entry.engine`.
pub struct Transaction {
    tx: Option<Box<dyn StorageTransaction>>,
    entry: Arc<EngineEntry>,
    committed: bool,
    rolled_back: bool,
}

impl Transaction {
    /// Create a new transaction wrapper
    pub(crate) fn new(tx: Box<dyn StorageTransaction>, entry: Arc<EngineEntry>) -> Self {
        Self {
            tx: Some(tx),
            entry,
            committed: false,
            rolled_back: false,
        }
    }

    /// Check if the transaction is still active
    fn check_active(&self) -> Result<()> {
        if self.committed {
            return Err(Error::TransactionEnded);
        }
        if self.rolled_back {
            return Err(Error::TransactionEnded);
        }
        if self.tx.is_none() {
            return Err(Error::TransactionNotStarted);
        }
        Ok(())
    }

    /// Get the transaction ID
    pub fn id(&self) -> i64 {
        self.tx.as_ref().map(|tx| tx.id()).unwrap_or(-1)
    }

    /// Execute a SQL statement within the transaction
    ///
    /// # Parameters
    ///
    /// Parameters can be passed using:
    /// - Empty tuple `()` for no parameters
    /// - Tuple syntax `(1, "Alice", 30)` for multiple parameters
    /// - `params!` macro `params![1, "Alice", 30]`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tx = db.begin()?;
    /// tx.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
    /// tx.execute("UPDATE accounts SET balance = balance - $1 WHERE user_id = $2", (100, 1))?;
    /// tx.commit()?;
    /// ```
    pub fn execute<P: Params>(&mut self, sql: &str, params: P) -> Result<i64> {
        self.check_active()?;

        let param_values = params.into_params();
        let result = self.execute_sql(sql, param_values)?;
        Ok(result.rows_affected())
    }

    /// Execute a pre-parsed statement with parameters.
    ///
    /// Avoids re-parsing SQL on every call — ideal for batch operations
    /// where the same statement is executed many times with different params.
    ///
    /// Use `Parser::new(sql).parse_program()` to pre-parse the SQL once.
    pub fn execute_prepared<P: Params>(&mut self, statement: &Statement, params: P) -> Result<i64> {
        self.check_active()?;

        let param_values = params.into_params();
        let ctx = if param_values.is_empty() {
            ExecutionContext::new()
        } else {
            ExecutionContext::with_params(param_values)
        };
        let result = self.execute_statement(statement, &ctx)?;
        Ok(result.rows_affected())
    }

    /// Query using a pre-parsed statement with parameters.
    ///
    /// Avoids re-parsing SQL on every call — ideal for batch read operations
    /// where the same query is executed many times with different params.
    pub fn query_prepared<P: Params>(&mut self, statement: &Statement, params: P) -> Result<Rows> {
        self.check_active()?;

        let param_values = params.into_params();
        let ctx = if param_values.is_empty() {
            ExecutionContext::new()
        } else {
            ExecutionContext::with_params(param_values)
        };
        let result = self.execute_statement(statement, &ctx)?;
        Ok(Rows::new(result))
    }

    /// Execute a query within the transaction
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tx = db.begin()?;
    /// for row in tx.query("SELECT * FROM users WHERE age > $1", (18,))? {
    ///     let row = row?;
    ///     println!("{}", row.get::<String>("name")?);
    /// }
    /// tx.commit()?;
    /// ```
    pub fn query<P: Params>(&mut self, sql: &str, params: P) -> Result<Rows> {
        self.check_active()?;

        let param_values = params.into_params();
        let result = self.execute_sql(sql, param_values)?;
        Ok(Rows::new(result))
    }

    /// Execute a query and return a single value
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tx = db.begin()?;
    /// let count: i64 = tx.query_one("SELECT COUNT(*) FROM users", ())?;
    /// tx.commit()?;
    /// ```
    pub fn query_one<T: FromValue, P: Params>(&mut self, sql: &str, params: P) -> Result<T> {
        let row = self
            .query(sql, params)?
            .next()
            .ok_or(Error::NoRowsReturned)??;
        row.get(0)
    }

    /// Execute a query and return an optional single value
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut tx = db.begin()?;
    /// let name: Option<String> = tx.query_opt("SELECT name FROM users WHERE id = $1", (999,))?;
    /// tx.commit()?;
    /// ```
    pub fn query_opt<T: FromValue, P: Params>(
        &mut self,
        sql: &str,
        params: P,
    ) -> Result<Option<T>> {
        match self.query(sql, params)?.next() {
            Some(row) => Ok(Some(row?.get(0)?)),
            None => Ok(None),
        }
    }

    /// Execute a SQL statement with named parameters within the transaction
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::named_params;
    ///
    /// let mut tx = db.begin()?;
    /// tx.execute_named(
    ///     "INSERT INTO users VALUES (:id, :name)",
    ///     named_params!{ id: 1, name: "Alice" }
    /// )?;
    /// tx.commit()?;
    /// ```
    pub fn execute_named(&mut self, sql: &str, params: NamedParams) -> Result<i64> {
        self.check_active()?;
        let ctx = ExecutionContext::with_named_params(params.into_inner());
        let result = self.execute_sql_with_ctx(sql, ctx)?;
        Ok(result.rows_affected())
    }

    /// Execute a query with named parameters within the transaction
    pub fn query_named(&mut self, sql: &str, params: NamedParams) -> Result<Rows> {
        self.check_active()?;
        let ctx = ExecutionContext::with_named_params(params.into_inner());
        let result = self.execute_sql_with_ctx(sql, ctx)?;
        Ok(Rows::new(result))
    }

    /// Execute a pre-parsed statement with named parameters.
    ///
    /// Combines `execute_prepared` (skip parsing) with `execute_named` (named params).
    pub fn execute_prepared_named(
        &mut self,
        statement: &Statement,
        params: NamedParams,
    ) -> Result<i64> {
        self.check_active()?;
        let ctx = ExecutionContext::with_named_params(params.into_inner());
        let result = self.execute_statement(statement, &ctx)?;
        Ok(result.rows_affected())
    }

    /// Query using a pre-parsed statement with named parameters.
    ///
    /// Combines `query_prepared` (skip parsing) with `query_named` (named params).
    pub fn query_prepared_named(
        &mut self,
        statement: &Statement,
        params: NamedParams,
    ) -> Result<Rows> {
        self.check_active()?;
        let ctx = ExecutionContext::with_named_params(params.into_inner());
        let result = self.execute_statement(statement, &ctx)?;
        Ok(Rows::new(result))
    }

    /// Internal SQL execution
    fn execute_sql(&mut self, sql: &str, params: ParamVec) -> Result<Box<dyn QueryResult>> {
        let ctx = if params.is_empty() {
            ExecutionContext::new()
        } else {
            ExecutionContext::with_params(params)
        };
        self.execute_sql_with_ctx(sql, ctx)
    }

    /// Internal SQL execution with a pre-built execution context
    fn execute_sql_with_ctx(
        &mut self,
        sql: &str,
        ctx: ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let mut parser = Parser::new(sql);
        let program = parser
            .parse_program()
            .map_err(|e| Error::parse(e.to_string()))?;

        let mut last_result: Option<Box<dyn QueryResult>> = None;
        for statement in &program.statements {
            last_result = Some(self.execute_statement(statement, &ctx)?);
        }
        last_result.ok_or(Error::NoStatementsToExecute)
    }

    /// Execute a single statement
    fn execute_statement(
        &mut self,
        statement: &Statement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        use crate::executor::result::ExecResult;

        // For SELECT and INSERT, delegate to the full executor pipeline.
        // INSERT delegation is required for correct handling of partial column lists,
        // default values, type coercion, RETURNING, ON DUPLICATE KEY, and FK validation.
        // Must handle before borrowing self.tx since we need to take ownership.
        if matches!(
            statement,
            Statement::Select(_)
                | Statement::Insert(_)
                | Statement::Update(_)
                | Statement::Delete(_)
        ) {
            let tx = self.tx.take().ok_or(Error::TransactionNotStarted)?;
            // Use the engine entry's shared semantic cache and query
            // planner so DML / ANALYZE inside this transaction
            // invalidate the cache and stats that sibling handles see,
            // not just per-call local state that drops at end-of-method.
            let executor = Executor::with_shared_semantic_cache(
                self.entry.engine.clone(),
                Arc::clone(&self.entry.semantic_cache),
                Arc::clone(&self.entry.query_planner),
            );
            executor.install_transaction(tx);
            let result = executor.execute_statement(statement, ctx);
            // Reclaim the transaction regardless of success/failure
            self.tx = executor.take_transaction();
            return result;
        }

        let tx = self.tx.as_mut().ok_or(Error::TransactionNotStarted)?;

        match statement {
            Statement::Update(stmt) => {
                use crate::executor::expression::{
                    compile_expression, ExecuteContext, ExprVM, RowFilter, SharedProgram,
                };

                let table_name = &stmt.table_name.value;
                let mut table = tx.get_table(table_name)?;
                // Get schema and column names owned to avoid borrow conflict with table.update()
                let schema = table.schema().clone();
                let columns: Vec<String> = schema.column_names_owned().to_vec();

                // Build the setter function that applies updates
                let updates = stmt.updates.clone();

                // OPTIMIZATION: Pre-compile WHERE clause (if present) using RowFilter
                // CRITICAL: Must include context for parameter support
                let where_filter: Option<RowFilter> = stmt
                    .where_clause
                    .as_ref()
                    .map(|expr| RowFilter::new(expr, &columns).map(|f| f.with_context(ctx)))
                    .transpose()?;

                // OPTIMIZATION: Pre-compile update expressions and compute column indices
                // CRITICAL: Return errors instead of silently skipping failed compilations
                let compiled_updates: Vec<(usize, SharedProgram)> = updates
                    .iter()
                    .map(|(col_name, expr)| {
                        let idx = columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(col_name))
                            .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
                        let program = compile_expression(expr, &columns)?;
                        Ok((idx, program))
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Create VM for expression execution (reused for all rows)
                let mut vm = ExprVM::new();

                // Arc-clone params for the setter closure (O(1) refcount bump, no deep copy)
                let positional_params = ctx.params_arc().clone();
                let named_params = ctx.named_params_arc().clone();

                let mut setter = |row: Row| -> Result<(Row, bool)> {
                    // Check WHERE clause if present (uses thread-local VM internally)
                    if let Some(ref filter) = where_filter {
                        if !filter.matches_checked(&row)? {
                            return Ok((row, false));
                        }
                    }

                    // Evaluate all expressions first (while we can still borrow row)
                    // CRITICAL: Include positional and named params for parameter resolution
                    let mut exec_ctx = ExecuteContext::new(&row);
                    if !positional_params.is_empty() {
                        exec_ctx = exec_ctx.with_params(&positional_params);
                    }
                    if !named_params.is_empty() {
                        exec_ctx = exec_ctx.with_named_params(&named_params);
                    }

                    // CRITICAL: Collect evaluated values, propagating any errors
                    let mut updates_to_apply: Vec<(usize, Value)> =
                        Vec::with_capacity(compiled_updates.len());
                    for (idx, program) in compiled_updates.iter() {
                        let v = vm.execute(program, &exec_ctx)?;
                        updates_to_apply.push((*idx, v));
                    }

                    // Now apply updates - take ownership of row
                    let mut new_values = row.into_values();
                    for (idx, value) in updates_to_apply {
                        new_values[idx] = value;
                    }

                    Ok((Row::from_values(new_values), true))
                };

                // Try to convert WHERE clause to storage expression for index optimization
                // (PK lookup, secondary index, etc.). If conversion fails for complex
                // expressions, fall back to None and let the setter handle filtering.
                let storage_where_expr = stmt
                    .where_clause
                    .as_ref()
                    .and_then(|expr| self.convert_to_storage_expression(expr, ctx, &schema).ok());

                let updated_count = table.update(storage_where_expr.as_deref(), &mut setter)?;

                Ok(Box::new(ExecResult::with_rows_affected(
                    updated_count as i64,
                )))
            }
            Statement::Delete(stmt) => {
                let table_name = &stmt.table_name.value;
                let mut table = tx.get_table(table_name)?;
                let schema = table.schema().clone();

                // Convert WHERE clause to Expression if present
                let where_expr = stmt
                    .where_clause
                    .as_ref()
                    .map(|expr| self.convert_to_storage_expression(expr, ctx, &schema))
                    .transpose()?;

                let deleted_count = table.delete(where_expr.as_deref())?;

                Ok(Box::new(ExecResult::with_rows_affected(
                    deleted_count as i64,
                )))
            }
            Statement::Select(_) | Statement::Insert(_) => {
                unreachable!("SELECT and INSERT handled above via executor delegation")
            }
            _ => Err(Error::NotSupported(
                "Only DML statements are supported in transactions".to_string(),
            )),
        }
    }

    /// Convert AST expression to storage expression
    fn convert_to_storage_expression(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
        schema: &Schema,
    ) -> Result<Box<dyn crate::storage::expression::Expression>> {
        use crate::core::Operator;
        use crate::storage::expression::{AndExpr, ComparisonExpr, OrExpr};

        match expr {
            Expression::Infix(infix) => {
                let op_str = infix.operator.as_str();
                match op_str {
                    "AND" => {
                        let left = self.convert_to_storage_expression(&infix.left, ctx, schema)?;
                        let right =
                            self.convert_to_storage_expression(&infix.right, ctx, schema)?;
                        return Ok(Box::new(AndExpr::and(left, right)));
                    }
                    "OR" => {
                        let left = self.convert_to_storage_expression(&infix.left, ctx, schema)?;
                        let right =
                            self.convert_to_storage_expression(&infix.right, ctx, schema)?;
                        return Ok(Box::new(OrExpr::or(left, right)));
                    }
                    _ => {}
                }

                let op = match op_str {
                    "=" | "==" => Operator::Eq,
                    "!=" | "<>" => Operator::Ne,
                    "<" => Operator::Lt,
                    "<=" => Operator::Lte,
                    ">" => Operator::Gt,
                    ">=" => Operator::Gte,
                    _ => {
                        return Err(Error::NotSupported(format!(
                            "Operator {} not supported in transaction WHERE clause",
                            infix.operator
                        )));
                    }
                };

                // Get column name from left side
                let column = match infix.left.as_ref() {
                    Expression::Identifier(id) => id.value.clone(),
                    _ => {
                        return Err(Error::NotSupported(
                            "Only column references supported on left side of comparison"
                                .to_string(),
                        ));
                    }
                };

                // Get value from right side (constant expression, no row context needed)
                let value = ExpressionEval::compile(&infix.right, &[])?
                    .with_context(ctx)
                    .eval_slice(&Row::new())?;

                // Create expression and prepare it for the schema
                let mut storage_expr = ComparisonExpr::new(column, op, value);
                storage_expr.prepare_for_schema(schema);
                Ok(Box::new(storage_expr))
            }
            _ => Err(Error::NotSupported(format!(
                "Expression type {:?} not supported in transaction WHERE clause",
                expr
            ))),
        }
    }

    /// Commit the transaction
    ///
    /// All changes made within the transaction become permanent.
    pub fn commit(&mut self) -> Result<()> {
        self.check_active()?;

        if let Some(mut tx) = self.tx.take() {
            match tx.commit() {
                Ok(()) => {
                    self.committed = true;
                }
                Err(e) => {
                    // Restore the transaction so rollback is still possible
                    self.tx = Some(tx);
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// Roll back the transaction
    ///
    /// All changes made within the transaction are discarded.
    pub fn rollback(&mut self) -> Result<()> {
        if self.committed {
            return Err(Error::TransactionCommitted);
        }

        if self.rolled_back {
            return Ok(()); // Already rolled back
        }

        if let Some(mut tx) = self.tx.take() {
            tx.rollback()?;
            self.rolled_back = true;
        }

        Ok(())
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        // Auto-rollback if not committed
        if !self.committed && !self.rolled_back {
            let _ = self.rollback();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::api::Database;

    #[test]
    fn test_transaction_commit() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test VALUES ($1, $2)", (1, 100))
            .unwrap();

        // Verify data exists
        let value: i64 = db
            .query_one("SELECT value FROM test WHERE id = $1", (1,))
            .unwrap();
        assert_eq!(value, 100);
    }

    #[test]
    fn test_transaction_rollback() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test VALUES ($1, $2)", (1, 100))
            .unwrap();

        let mut tx = db.begin().unwrap();
        tx.execute("UPDATE test SET value = $1 WHERE id = $2", (200, 1))
            .unwrap();
        tx.rollback().unwrap();

        let value: i64 = db
            .query_one("SELECT value FROM test WHERE id = $1", (1,))
            .unwrap();
        assert_eq!(value, 100);
    }

    #[test]
    fn test_transaction_auto_rollback() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test VALUES ($1, $2)", (1, 100))
            .unwrap();

        {
            let mut tx = db.begin().unwrap();
            tx.execute("UPDATE test SET value = $1 WHERE id = $2", (200, 1))
                .unwrap();
            // tx dropped without commit - should auto-rollback
        }

        let value: i64 = db
            .query_one("SELECT value FROM test WHERE id = $1", (1,))
            .unwrap();
        assert_eq!(value, 100);
    }

    #[test]
    fn test_transaction_query() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test VALUES ($1, $2)", (1, 100))
            .unwrap();

        let mut tx = db.begin().unwrap();

        // New API: query with params
        for row in tx.query("SELECT * FROM test", ()).unwrap() {
            let row = row.unwrap();
            assert_eq!(row.get::<i64>(0).unwrap(), 1);
            assert_eq!(row.get::<i64>(1).unwrap(), 100);
        }

        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_query_one() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test VALUES ($1, $2)", (1, 100))
            .unwrap();

        let mut tx = db.begin().unwrap();
        let value: i64 = tx
            .query_one("SELECT value FROM test WHERE id = $1", (1,))
            .unwrap();
        assert_eq!(value, 100);
        tx.commit().unwrap();
    }

    #[test]
    fn test_committed_transaction_error() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();

        let mut tx = db.begin().unwrap();
        tx.commit().unwrap();

        // Should error on further operations
        assert!(tx.execute("INSERT INTO test VALUES ($1)", (1,)).is_err());
        assert!(tx.commit().is_err());
    }

    #[test]
    fn test_transaction_id() {
        let db = Database::open_in_memory().unwrap();
        let tx = db.begin().unwrap();
        assert!(tx.id() > 0);
    }

    #[test]
    fn test_execute_prepared_insert() {
        use crate::parser::Parser;

        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, value FLOAT)",
            (),
        )
        .unwrap();

        // Pre-parse the statement once
        let stmt = Parser::new("INSERT INTO test VALUES ($1, $2, $3)")
            .parse_program()
            .unwrap()
            .statements
            .into_iter()
            .next()
            .unwrap();

        // Execute multiple times with different params
        let mut tx = db.begin().unwrap();
        tx.execute_prepared(&stmt, (1, "Alice", 10.5)).unwrap();
        tx.execute_prepared(&stmt, (2, "Bob", 20.0)).unwrap();
        tx.execute_prepared(&stmt, (3, "Charlie", 30.0)).unwrap();
        tx.commit().unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count, 3);

        let name: String = db
            .query_one("SELECT name FROM test WHERE id = $1", (2,))
            .unwrap();
        assert_eq!(name, "Bob");
    }

    #[test]
    fn test_execute_prepared_no_params() {
        use crate::parser::Parser;

        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER DEFAULT 0)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test VALUES (1, 100)", ()).unwrap();

        let stmt = Parser::new("UPDATE test SET value = 999")
            .parse_program()
            .unwrap()
            .statements
            .into_iter()
            .next()
            .unwrap();

        let mut tx = db.begin().unwrap();
        let affected = tx.execute_prepared(&stmt, ()).unwrap();
        assert_eq!(affected, 1);
        tx.commit().unwrap();

        let value: i64 = db
            .query_one("SELECT value FROM test WHERE id = 1", ())
            .unwrap();
        assert_eq!(value, 999);
    }

    #[test]
    fn test_execute_prepared_on_committed_tx_errors() {
        use crate::parser::Parser;

        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();

        let stmt = Parser::new("INSERT INTO test VALUES ($1)")
            .parse_program()
            .unwrap()
            .statements
            .into_iter()
            .next()
            .unwrap();

        let mut tx = db.begin().unwrap();
        tx.commit().unwrap();
        assert!(tx.execute_prepared(&stmt, (1,)).is_err());
    }

    #[test]
    fn test_transaction_aggregate_count() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT, price FLOAT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO items VALUES (1, 'A', 10.0)", ())
            .unwrap();
        db.execute("INSERT INTO items VALUES (2, 'B', 20.0)", ())
            .unwrap();
        db.execute("INSERT INTO items VALUES (3, 'A', 30.0)", ())
            .unwrap();

        let mut tx = db.begin().unwrap();
        let count: i64 = tx.query_one("SELECT COUNT(*) FROM items", ()).unwrap();
        assert_eq!(count, 3);

        let sum: f64 = tx.query_one("SELECT SUM(price) FROM items", ()).unwrap();
        assert!((sum - 60.0).abs() < f64::EPSILON);

        let avg: f64 = tx.query_one("SELECT AVG(price) FROM items", ()).unwrap();
        assert!((avg - 20.0).abs() < f64::EPSILON);
        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_group_by() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO sales VALUES (1, 'A', 10)", ())
            .unwrap();
        db.execute("INSERT INTO sales VALUES (2, 'B', 20)", ())
            .unwrap();
        db.execute("INSERT INTO sales VALUES (3, 'A', 30)", ())
            .unwrap();

        let mut tx = db.begin().unwrap();
        let rows: Vec<_> = tx
            .query(
                "SELECT category, SUM(amount) as total FROM sales GROUP BY category ORDER BY category",
                (),
            )
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get::<String>(0).unwrap(), "A");
        assert_eq!(rows[0].get::<i64>(1).unwrap(), 40);
        assert_eq!(rows[1].get::<String>(0).unwrap(), "B");
        assert_eq!(rows[1].get::<i64>(1).unwrap(), 20);
        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_select_after_insert() {
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();

        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO test VALUES (1, 100)", ()).unwrap();
        tx.execute("INSERT INTO test VALUES (2, 200)", ()).unwrap();

        // Should see uncommitted inserts within the same transaction
        let count: i64 = tx.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count, 2);

        let sum: i64 = tx.query_one("SELECT SUM(value) FROM test", ()).unwrap();
        assert_eq!(sum, 300);

        // Can still do more DML after SELECT delegation
        tx.execute("INSERT INTO test VALUES (3, 300)", ()).unwrap();
        let count2: i64 = tx.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count2, 3);

        tx.commit().unwrap();

        // Verify committed data
        let final_count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(final_count, 3);
    }
}
