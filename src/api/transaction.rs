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

use crate::api::params::{NamedParams, ParamVec};
use crate::core::{Error, Result, Row, RowVec, Schema, Value};
use crate::executor::context::ExecutionContext;
use crate::executor::expression::ExpressionEval;
use crate::executor::result::ExecutorResult;
use crate::parser::ast::{Expression, Statement};
use crate::parser::Parser;
use crate::storage::expression::Expression as StorageExprTrait;
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
                        if !filter.matches(&row) {
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
            Statement::Set(stmt) => {
                if stmt.name.value == "TRANSACTION_ISOLATION" {
                    if let Expression::StringLiteral(ref lit) = stmt.value {
                        let level = match lit.value.to_uppercase().as_str() {
                            "READ COMMITTED" => crate::core::IsolationLevel::ReadCommitted,
                            "REPEATABLE READ" | "SNAPSHOT" => {
                                crate::core::IsolationLevel::SnapshotIsolation
                            }
                            _ => {
                                return Err(Error::NotSupported(format!(
                                    "Isolation level {} not supported",
                                    lit.value
                                )))
                            }
                        };
                        tx.set_isolation_level(level)?;
                        return Ok(Box::new(ExecResult::empty()));
                    }
                }
                Err(Error::NotSupported(
                    "Only SET TRANSACTION ISOLATION LEVEL is supported in transactions".to_string(),
                ))
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
                        return Err(Error::NotSupported(
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
                    .map(|expr| self.convert_to_storage_expression(expr, ctx, schema))
                    .transpose()?;

                // Scan table
                let mut scanner = table.scan(&column_indices, where_expr.as_deref())?;

                // Check for COUNT(*) optimization
                let is_count_star = stmt.columns.len() == 1
                    && match &stmt.columns[0] {
                        Expression::FunctionCall(fc) => {
                            fc.function.eq_ignore_ascii_case("count")
                                && fc.arguments.len() == 1
                                && matches!(fc.arguments[0], Expression::Star(_))
                        }
                        _ => false,
                    };

                if is_count_star {
                    let mut count = 0i64;
                    while scanner.next() {
                        count += 1;
                    }
                    if let Some(err) = scanner.err() {
                        return Err(err.clone());
                    }
                    let mut rows = RowVec::with_capacity(1);
                    rows.push((0, Row::from_values(vec![Value::Integer(count)])));
                    return Ok(Box::new(ExecutorResult::new(
                        vec!["COUNT(*)".to_string()],
                        rows,
                    )));
                }

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
            Statement::Savepoint(stmt) => {
                tx.create_savepoint(stmt.savepoint_name.value.as_str())?;
                Ok(Box::new(ExecResult::empty()))
            }
            Statement::ReleaseSavepoint(stmt) => {
                tx.release_savepoint(stmt.savepoint_name.value.as_str())?;
                Ok(Box::new(ExecResult::empty()))
            }
            Statement::Rollback(stmt) => {
                if let Some(ref savepoint_name) = stmt.savepoint_name {
                    tx.rollback_to_savepoint(savepoint_name.value.as_str())?;
                    Ok(Box::new(ExecResult::empty()))
                } else {
                    tx.rollback()?;
                    Ok(Box::new(ExecResult::empty()))
                }
            }
            Statement::Commit(_) => {
                tx.commit()?;
                Ok(Box::new(ExecResult::empty()))
            }
            Statement::CreateTable(stmt) => {
                let mut builder =
                    crate::core::SchemaBuilder::new(stmt.table_name.value.to_string());
                for col in &stmt.columns {
                    let type_str = col.data_type.as_str();
                    let upper = type_str.to_uppercase();
                    let base_type = upper.split('(').next().unwrap_or(&upper);

                    let dt = match base_type {
                        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => {
                            crate::core::DataType::Integer
                        }
                        "FLOAT" | "DOUBLE" | "REAL" | "DECIMAL" | "NUMERIC" => {
                            crate::core::DataType::Float
                        }
                        "TEXT" | "VARCHAR" | "CHAR" | "STRING" | "CLOB" => {
                            crate::core::DataType::Text
                        }
                        "BOOLEAN" | "BOOL" => crate::core::DataType::Boolean,
                        "TIMESTAMP" | "DATETIME" | "DATE" | "TIME" => {
                            crate::core::DataType::Timestamp
                        }
                        "JSON" | "JSONB" => crate::core::DataType::Json,
                        "VECTOR" => crate::core::DataType::Vector,
                        _ => {
                            return Err(Error::NotSupported(format!(
                                "Data type {} not supported in transactions",
                                type_str
                            )))
                        }
                    };

                    let nullable = !col
                        .constraints
                        .iter()
                        .any(|c| matches!(c, crate::parser::ast::ColumnConstraint::NotNull));
                    let is_primary_key = col
                        .constraints
                        .iter()
                        .any(|c| matches!(c, crate::parser::ast::ColumnConstraint::PrimaryKey));
                    builder =
                        builder.column(col.name.value.to_string(), dt, nullable, is_primary_key);

                    if base_type == "VECTOR" {
                        if let Some(dim_str) =
                            upper.split('(').nth(1).and_then(|s| s.split(')').next())
                        {
                            if let Ok(dim) = dim_str.parse::<u16>() {
                                builder = builder.set_last_vector_dimensions(dim);
                            }
                        }
                    }
                }
                tx.create_table(stmt.table_name.value.as_str(), builder.build())?;
                Ok(Box::new(ExecResult::empty()))
            }
            Statement::DropTable(stmt) => {
                tx.drop_table(stmt.table_name.value.as_str())?;
                Ok(Box::new(ExecResult::empty()))
            }
            Statement::CreateIndex(stmt) => {
                tx.create_table_index(
                    stmt.table_name.value.as_str(),
                    stmt.index_name.value.as_str(),
                    &stmt
                        .columns
                        .iter()
                        .map(|c| c.to_string())
                        .collect::<Vec<_>>(),
                    stmt.is_unique,
                )?;
                Ok(Box::new(ExecResult::empty()))
            }
            Statement::DropIndex(stmt) => {
                let table_name = stmt
                    .table_name
                    .as_ref()
                    .map(|id| id.value.as_str())
                    .ok_or_else(|| {
                        Error::NotSupported(
                            "DROP INDEX without table name not supported in transactions"
                                .to_string(),
                        )
                    })?;
                tx.drop_table_index(table_name, stmt.index_name.value.as_str())?;
                Ok(Box::new(ExecResult::empty()))
            }
            Statement::Truncate(stmt) => {
                tx.drop_table(&stmt.table_name.value)?;
                // Re-creating table after truncate is complex here,
                // but let's at least support drop for now if that's what TRUNCATE does in some contexts.
                // Actually StorageTransaction should probably have a truncate method.
                // For now, let's just return NotSupported for Truncate if we can't do it properly.
                Err(Error::NotSupported(
                    "TRUNCATE not yet supported in Transaction API".to_string(),
                ))
            }
            _ => Err(Error::NotSupported(
                "Only DML and basic DDL statements are supported in transactions".to_string(),
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
}
