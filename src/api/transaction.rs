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
use crate::core::{Error, Result};
use crate::executor::context::ExecutionContext;
use crate::executor::{CachedPlanRef, Executor};
use crate::parser::ast::{RollbackStatement, Statement};
use crate::storage::traits::QueryResult;

use super::database::FromValue;
use super::params::Params;
use super::rows::Rows;

/// Transaction represents a database transaction
///
/// Provides ACID guarantees for a series of database operations.
/// Must be explicitly committed or rolled back.
pub struct Transaction {
    /// Holds the open transaction and runs its statements on the plans and
    /// compiled paths of the session that began it
    executor: Executor,
    committed: bool,
    rolled_back: bool,
}

/// Whether `statement` may run inside a transaction: DML and savepoints
fn runs_in_transaction(statement: &Statement) -> Result<()> {
    match statement {
        Statement::Select(_)
        | Statement::Insert(_)
        | Statement::Update(_)
        | Statement::Delete(_)
        | Statement::Savepoint(_)
        | Statement::ReleaseSavepoint(_)
        | Statement::Rollback(RollbackStatement {
            savepoint_name: Some(_),
            ..
        }) => Ok(()),
        _ => Err(Error::NotSupported(
            "Only DML statements are supported in transactions".to_string(),
        )),
    }
}

impl Transaction {
    /// Create a new transaction wrapper around an executor holding it open
    pub(crate) fn new(executor: Executor) -> Self {
        Self {
            executor,
            committed: false,
            rolled_back: false,
        }
    }

    /// Check if the transaction is still active
    fn check_active(&self) -> Result<()> {
        if self.committed || self.rolled_back {
            return Err(Error::TransactionEnded);
        }
        Ok(())
    }

    /// Get the transaction ID
    pub fn id(&self) -> i64 {
        self.executor.active_txn_id().unwrap_or(-1)
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
        let program = match self.executor.plan_or_program(sql)? {
            Ok(plan) => return self.execute_plan(&plan, ctx),
            Err(program) => program,
        };
        let mut last_result: Option<Box<dyn QueryResult>> = None;
        for statement in &program.statements {
            last_result = Some(self.execute_statement(statement, &ctx)?);
        }
        last_result.ok_or(Error::NoStatementsToExecute)
    }

    /// Runs a cached plan, with its compiled paths, in this transaction
    fn execute_plan(
        &mut self,
        plan: &CachedPlanRef,
        ctx: ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        runs_in_transaction(&plan.statement)?;
        self.executor.execute_with_cached_plan_owned(plan, ctx)
    }

    /// Runs a prepared statement in this transaction: by its plan, or by
    /// its SQL when it holds more than one statement
    pub(crate) fn execute_prepared_statement(
        &mut self,
        statement: &super::Statement,
        ctx: ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        self.check_active()?;
        match statement.plan() {
            Some(plan) => self.execute_plan(plan, ctx),
            None => self.execute_sql_with_ctx(statement.sql(), ctx),
        }
    }

    /// Execute a single statement
    fn execute_statement(
        &mut self,
        statement: &Statement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        runs_in_transaction(statement)?;
        self.executor.execute_statement(statement, ctx)
    }

    /// Commit the transaction
    ///
    /// All changes made within the transaction become permanent.
    pub fn commit(&mut self) -> Result<()> {
        self.check_active()?;

        if let Some(mut tx) = self.executor.take_transaction() {
            if let Err(e) = tx.commit() {
                // Restore the transaction so rollback is still possible
                self.executor.install_transaction(tx);
                return Err(e);
            }
        }
        self.committed = true;
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

        self.rolled_back = true;
        self.executor.rollback_active()
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

    #[test]
    fn transaction_sql_runs_on_the_session_plan() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        let sql = "SELECT v FROM t WHERE id = $1";
        let mut tx = db.begin().unwrap();
        tx.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
        assert_eq!(tx.query_one::<i64, _>(sql, (1,)).unwrap(), 10);
        let plan = db
            .executor()
            .query_cache()
            .get(sql)
            .expect("the transaction did not plan through its session");
        assert_eq!(tx.query_one::<i64, _>(sql, (1,)).unwrap(), 10);
        let again = db.executor().query_cache().get(sql).unwrap();
        assert!(std::sync::Arc::ptr_eq(&plan.statement, &again.statement));
        tx.rollback().unwrap();
        assert!(db.query_one::<i64, _>(sql, (1,)).is_err());
    }

    #[test]
    fn transaction_refuses_statements_other_than_dml() {
        let db = Database::open_in_memory().unwrap();
        let mut tx = db.begin().unwrap();
        for sql in ["CREATE TABLE t (id INTEGER PRIMARY KEY)", "COMMIT", "BEGIN"] {
            assert!(
                matches!(
                    tx.execute(sql, ()),
                    Err(crate::core::Error::NotSupported(_))
                ),
                "{sql} ran inside the transaction"
            );
        }
        tx.commit().unwrap();
    }
}
