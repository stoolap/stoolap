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

use crate::api::params::ParamVec;
use crate::core::{Error, Result, Row, RowVec, Value};
use crate::executor::context::ExecutionContext;
use crate::executor::expression::ExpressionEval;
use crate::executor::result::ExecutorResult;
use crate::parser::ast::{Expression, Statement};
use crate::parser::Parser;
use crate::storage::traits::{QueryResult, Transaction as StorageTransaction};

use super::database::FromValue;
use super::params::Params;
use super::rows::Rows;

/// Transaction represents a database transaction
///
/// Provides ACID guarantees for a series of database operations.
/// Must be explicitly committed or rolled back.
pub struct Transaction {
    tx: Option<Box<dyn StorageTransaction>>,
    committed: bool,
    rolled_back: bool,
}

impl Transaction {
    /// Create a new transaction wrapper
    pub(crate) fn new(tx: Box<dyn StorageTransaction>) -> Self {
        Self {
            tx: Some(tx),
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

    /// Internal SQL execution
    fn execute_sql(&mut self, sql: &str, params: ParamVec) -> Result<Box<dyn QueryResult>> {
        // Parse the SQL
        let mut parser = Parser::new(sql);
        let program = parser
            .parse_program()
            .map_err(|e| Error::parse(e.to_string()))?;

        // Create execution context with parameters
        let ctx = if params.is_empty() {
            ExecutionContext::new()
        } else {
            ExecutionContext::with_params(params)
        };

        // Execute each statement
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

        let tx = self.tx.as_mut().ok_or(Error::TransactionNotStarted)?;

        match statement {
            Statement::Insert(stmt) => {
                let table_name = &stmt.table_name.value;
                let mut table = tx.get_table(table_name)?;

                let mut total_inserted = 0i64;

                for row_values in &stmt.values {
                    let mut values = Vec::with_capacity(row_values.len());
                    for expr in row_values {
                        // Use ExpressionEval for value expression evaluation
                        let mut eval = ExpressionEval::compile(expr, &[])?.with_context(ctx);
                        values.push(eval.eval_slice(&Row::new())?);
                    }

                    let row = Row::from_values(values);
                    let _ = table.insert(row)?;
                    total_inserted += 1;
                }

                Ok(Box::new(ExecResult::with_rows_affected(total_inserted)))
            }
            Statement::Update(stmt) => {
                use crate::executor::expression::{
                    compile_expression, ExecuteContext, ExprVM, RowFilter, SharedProgram,
                };

                let table_name = &stmt.table_name.value;
                let mut table = tx.get_table(table_name)?;
                // Get column names owned to avoid borrow conflict with table.update()
                let columns: Vec<String> = table.schema().column_names_owned().to_vec();

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
                            .ok_or_else(|| Error::ColumnNotFoundNamed(col_name.to_string()))?;
                        let program = compile_expression(expr, &columns)?;
                        Ok((idx, program))
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Create VM for expression execution (reused for all rows)
                let mut vm = ExprVM::new();

                // CRITICAL: Capture errors from setter since closure can't return Result
                use std::cell::RefCell;
                let update_error: RefCell<Option<Error>> = RefCell::new(None);

                let mut setter = |row: Row| -> (Row, bool) {
                    // If we already have an error, skip processing
                    if update_error.borrow().is_some() {
                        return (row, false);
                    }

                    // Check WHERE clause if present (uses thread-local VM internally)
                    if let Some(ref filter) = where_filter {
                        if !filter.matches(&row) {
                            return (row, false);
                        }
                    }

                    // Evaluate all expressions first (while we can still borrow row)
                    let exec_ctx = ExecuteContext::new(&row);

                    // CRITICAL: Collect evaluated values, capturing any errors
                    let mut updates_to_apply: Vec<(usize, Value)> =
                        Vec::with_capacity(compiled_updates.len());
                    for (idx, program) in compiled_updates.iter() {
                        match vm.execute(program, &exec_ctx) {
                            Ok(v) => updates_to_apply.push((*idx, v)),
                            Err(e) => {
                                *update_error.borrow_mut() = Some(e);
                                return (row, false);
                            }
                        }
                    }

                    // Now apply updates - take ownership of row
                    let mut new_values = row.into_values();
                    for (idx, value) in updates_to_apply {
                        new_values[idx] = value;
                    }

                    (Row::from_values(new_values), true)
                };

                // Use None for where_expr since we handle WHERE in the setter
                let updated_count = table.update(None, &mut setter)?;

                // Check if any errors were captured during update
                if let Some(err) = update_error.into_inner() {
                    return Err(err);
                }

                Ok(Box::new(ExecResult::with_rows_affected(
                    updated_count as i64,
                )))
            }
            Statement::Delete(stmt) => {
                let table_name = &stmt.table_name.value;
                let mut table = tx.get_table(table_name)?;

                // Convert WHERE clause to Expression if present
                let where_expr = stmt
                    .where_clause
                    .as_ref()
                    .map(|expr| self.convert_to_storage_expression(expr, ctx))
                    .transpose()?;

                let deleted_count = table.delete(where_expr.as_deref())?;

                Ok(Box::new(ExecResult::with_rows_affected(
                    deleted_count as i64,
                )))
            }
            Statement::Select(stmt) => {
                // Handle SELECT without FROM
                let table_expr = match &stmt.table_expr {
                    Some(expr) => expr,
                    None => {
                        let mut columns = Vec::new();
                        let mut values = Vec::new();

                        for (i, col_expr) in stmt.columns.iter().enumerate() {
                            let col_name = match col_expr {
                                Expression::Aliased(a) => a.alias.value.to_string(),
                                Expression::Identifier(id) => id.value.to_string(),
                                _ => format!("expr{}", i + 1),
                            };
                            columns.push(col_name);
                            // Use ExpressionEval for constant expression evaluation
                            let mut eval =
                                ExpressionEval::compile(col_expr, &[])?.with_context(ctx);
                            values.push(eval.eval_slice(&Row::new())?);
                        }

                        let mut rows = RowVec::with_capacity(1);
                        rows.push((0, Row::from_values(values)));
                        return Ok(Box::new(ExecutorResult::new(columns, rows)));
                    }
                };

                // Get table name
                let table_name = match table_expr.as_ref() {
                    Expression::TableSource(ts) => &ts.name.value,
                    Expression::Identifier(id) => &id.value,
                    _ => {
                        return Err(Error::NotSupportedMessage(
                            "Complex FROM clauses not supported in transactions".to_string(),
                        ))
                    }
                };

                let table = tx.get_table(table_name)?;
                let schema = table.schema();
                let columns: Vec<String> = schema.column_names_owned().to_vec();

                // Get all column indices for scan
                let column_indices: Vec<usize> = (0..columns.len()).collect();

                // Convert WHERE clause
                let where_expr = stmt
                    .where_clause
                    .as_ref()
                    .map(|expr| self.convert_to_storage_expression(expr, ctx))
                    .transpose()?;

                // Scan table
                let mut scanner = table.scan(&column_indices, where_expr.as_deref())?;
                let mut rows = RowVec::new();
                let mut idx = 0i64;

                while scanner.next() {
                    rows.push((idx, scanner.take_row()));
                    idx += 1;
                }

                // Check for scanner error
                if let Some(err) = scanner.err() {
                    return Err(err.clone());
                }

                // Project columns if needed
                let (result_columns, result_rows) = if stmt.columns.len() == 1 {
                    if let Expression::Star(_) = &stmt.columns[0] {
                        (columns, rows)
                    } else {
                        self.project_columns(stmt, &columns, rows, ctx)?
                    }
                } else {
                    self.project_columns(stmt, &columns, rows, ctx)?
                };

                Ok(Box::new(ExecutorResult::new(result_columns, result_rows)))
            }
            _ => Err(Error::NotSupportedMessage(
                "Only DML statements are supported in transactions".to_string(),
            )),
        }
    }

    /// Convert AST expression to storage expression
    fn convert_to_storage_expression(
        &self,
        expr: &Expression,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn crate::storage::expression::Expression>> {
        use crate::core::Operator;
        use crate::storage::expression::{AndExpr, ComparisonExpr, OrExpr};

        match expr {
            Expression::Infix(infix) => {
                let op_str = infix.operator.as_str();
                match op_str {
                    "AND" => {
                        let left = self.convert_to_storage_expression(&infix.left, ctx)?;
                        let right = self.convert_to_storage_expression(&infix.right, ctx)?;
                        return Ok(Box::new(AndExpr::and(left, right)));
                    }
                    "OR" => {
                        let left = self.convert_to_storage_expression(&infix.left, ctx)?;
                        let right = self.convert_to_storage_expression(&infix.right, ctx)?;
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
                        return Err(Error::NotSupportedMessage(format!(
                            "Operator {} not supported in transaction WHERE clause",
                            infix.operator
                        )));
                    }
                };

                // Get column name from left side
                let column = match infix.left.as_ref() {
                    Expression::Identifier(id) => id.value.clone(),
                    _ => {
                        return Err(Error::NotSupportedMessage(
                            "Only column references supported on left side of comparison"
                                .to_string(),
                        ));
                    }
                };

                // Get value from right side (constant expression, no row context needed)
                let value = ExpressionEval::compile(&infix.right, &[])?
                    .with_context(ctx)
                    .eval_slice(&Row::new())?;

                Ok(Box::new(ComparisonExpr::new(column, op, value)))
            }
            _ => Err(Error::NotSupportedMessage(format!(
                "Expression type {:?} not supported in transaction WHERE clause",
                expr
            ))),
        }
    }

    /// Project columns from rows
    fn project_columns(
        &self,
        stmt: &crate::parser::ast::SelectStatement,
        source_columns: &[String],
        rows: RowVec,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<String>, RowVec)> {
        use crate::executor::expression::compile_expression;

        let mut result_columns = Vec::new();
        let mut result_rows = RowVec::with_capacity(rows.len());

        // Pre-compile expressions and determine output columns
        // Store either None for Star or Some(program) for compiled expressions
        let mut compiled_exprs: Vec<Option<crate::executor::expression::SharedProgram>> =
            Vec::with_capacity(stmt.columns.len());

        for (i, col_expr) in stmt.columns.iter().enumerate() {
            match col_expr {
                Expression::Star(_) => {
                    result_columns.extend(source_columns.iter().cloned());
                    compiled_exprs.push(None); // Star marker
                }
                Expression::Aliased(a) => {
                    result_columns.push(a.alias.value.to_string());
                    compiled_exprs.push(Some(compile_expression(col_expr, source_columns)?));
                }
                Expression::Identifier(id) => {
                    result_columns.push(id.value.to_string());
                    compiled_exprs.push(Some(compile_expression(col_expr, source_columns)?));
                }
                _ => {
                    result_columns.push(format!("expr{}", i + 1));
                    compiled_exprs.push(Some(compile_expression(col_expr, source_columns)?));
                }
            }
        }

        // Create execution context with parameters
        let params = ctx.params();
        // Convert HashMap to FxHashMap for ExecuteContext
        let named_params: rustc_hash::FxHashMap<String, Value> = ctx
            .named_params()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Create VM for execution (reused for all rows and expressions)
        let mut vm = crate::executor::expression::ExprVM::new();
        let num_cols = stmt.columns.len();

        // Project each row
        for (id, row) in rows {
            let mut exec_ctx = crate::executor::expression::ExecuteContext::new(&row);
            if !params.is_empty() {
                exec_ctx = exec_ctx.with_params(params);
            }
            if !named_params.is_empty() {
                exec_ctx = exec_ctx.with_named_params(&named_params);
            }

            let mut values = Vec::with_capacity(num_cols.max(row.len()));

            for compiled in &compiled_exprs {
                match compiled {
                    None => {
                        // Star: expand all columns
                        values.extend(row.iter().cloned());
                    }
                    Some(program) => {
                        // Execute pre-compiled expression
                        values.push(vm.execute(program, &exec_ctx)?);
                    }
                }
            }

            result_rows.push((id, Row::from_values(values)));
        }

        Ok((result_columns, result_rows))
    }

    /// Commit the transaction
    ///
    /// All changes made within the transaction become permanent.
    pub fn commit(&mut self) -> Result<()> {
        self.check_active()?;

        if let Some(mut tx) = self.tx.take() {
            tx.commit()?;
            self.committed = true;
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
}
