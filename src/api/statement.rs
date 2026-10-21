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

//! Prepared statement support
//!
//! # Examples
//!
//! ```ignore
//! use stoolap::Database;
//!
//! let db = Database::open("memory://")?;
//! db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())?;
//!
//! // Prepare a statement for repeated execution
//! let insert = db.prepare("INSERT INTO users VALUES ($1, $2)")?;
//!
//! // Execute multiple times efficiently
//! for (id, name) in [(1, "Alice"), (2, "Bob"), (3, "Charlie")] {
//!     insert.execute((id, name))?;
//! }
//!
//! // Prepare a query
//! let select = db.prepare("SELECT name FROM users WHERE id = $1")?;
//! for id in 1..=3 {
//!     for row in select.query((id,))? {
//!         println!("{}", row?.get::<String>(0)?);
//!     }
//! }
//! ```

use std::sync::{Arc, Weak};

use crate::core::{Error, Result, Value};
use crate::executor::context::ExecutionContext;
use crate::executor::query_cache::CachedPlanRef;
use crate::executor::{bind_named, value_in};
use crate::storage::traits::QueryResult;

use super::database::{Database, DatabaseInnerHandle, FromValue};
use super::params::{NamedParams, Params};
use super::rows::Rows;
use super::transaction::Transaction;

/// A prepared SQL statement
///
/// Prepared statements parse SQL once at prepare time and retain the compiled
/// plan. Subsequent executions use the plan directly, bypassing normalize,
/// hash, and cache-lookup overhead on every call.
///
/// # Thread Safety
///
/// Statement holds a weak reference to the Database and can be used from
/// multiple threads, but each execution is serialized through the
/// database's executor lock.
///
/// # Lifetime
///
/// The Statement becomes invalid when the Database is dropped. Attempting
/// to use a Statement after its Database is dropped will return an error.
#[derive(Clone)]
pub struct Statement {
    /// Weak reference to database - doesn't prevent cleanup
    db_weak: Weak<DatabaseInnerHandle>,
    sql: String,
    /// Pre-compiled plan for single-statement queries (the common case).
    /// `None` for multi-statement SQL, which falls back to the SQL-based path.
    plan: Option<CachedPlanRef>,
}

impl Statement {
    /// Create a new prepared statement.
    ///
    /// Parses the SQL and retains the compiled plan so that subsequent
    /// executions bypass cache lookup entirely. Returns an error if the
    /// SQL is invalid.
    pub(crate) fn new(
        db_weak: Weak<DatabaseInnerHandle>,
        sql: String,
        db: &Database,
    ) -> Result<Self> {
        // A bare expression statement is SQL the parser did not recognise
        // (e.g. "SELECTX INVALID")
        let unrecognised = |statement: &crate::parser::ast::Statement| {
            matches!(statement, crate::parser::ast::Statement::Expression(_))
        };
        let (plan, recognised) = match db.executor().plan_or_program(&sql)? {
            Ok(plan) => {
                let recognised = !unrecognised(&plan.statement);
                (Some(plan), recognised)
            }
            // Multi-statement SQL: validated but not cacheable
            Err(program) => (None, !program.statements.iter().any(unrecognised)),
        };
        if !recognised {
            return Err(Error::parse(format!(
                "invalid SQL: unrecognised statement: {}",
                sql
            )));
        }
        Ok(Self { db_weak, sql, plan })
    }

    /// Get the database, upgrading the weak reference.
    /// Returns an error if the database was dropped.
    #[inline]
    fn get_db(&self) -> Result<Database> {
        self.db_weak
            .upgrade()
            .map(Database::from_inner)
            .ok_or_else(|| Error::internal("Database was dropped"))
    }

    /// Execute the prepared statement
    ///
    /// Returns the number of rows affected for DML statements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stmt = db.prepare("INSERT INTO users VALUES ($1, $2)")?;
    /// stmt.execute((1, "Alice"))?;
    /// stmt.execute((2, "Bob"))?;
    /// ```
    pub fn execute<P: Params>(&self, params: P) -> Result<i64> {
        let db = self.get_db()?;
        if let Some(plan) = &self.plan {
            Ok(db.run_plan(plan, params.into_params())?.rows_affected())
        } else {
            db.execute(&self.sql, params)
        }
    }

    /// Query using the prepared statement
    ///
    /// Returns an iterator over the result rows.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stmt = db.prepare("SELECT * FROM users WHERE age > $1")?;
    ///
    /// for row in stmt.query((18,))? {
    ///     let row = row?;
    ///     println!("{}", row.get::<String>("name")?);
    /// }
    /// ```
    pub fn query<P: Params>(&self, params: P) -> Result<Rows> {
        let db = self.get_db()?;
        if let Some(plan) = &self.plan {
            Ok(Rows::new(db.run_plan(plan, params.into_params())?))
        } else {
            db.query(&self.sql, params)
        }
    }

    /// Execute the prepared statement with named parameters (`:name`)
    pub fn execute_named(&self, params: NamedParams) -> Result<i64> {
        Ok(self
            .run_named(&mut value_in(params.as_map()))?
            .rows_affected())
    }

    /// Query using the prepared statement with named parameters (`:name`)
    pub fn query_named(&self, params: NamedParams) -> Result<Rows> {
        Ok(Rows::new(self.run_named(&mut value_in(params.as_map()))?))
    }

    /// Runs the statement with each value read by its name from `value_of`
    pub(crate) fn run_named(
        &self,
        value_of: &mut impl FnMut(&str) -> Result<Option<Value>>,
    ) -> Result<Box<dyn QueryResult>> {
        let db = self.get_db()?;
        match &self.plan {
            Some(plan) => db.run_plan(plan, bind_named(&plan.params, value_of)?),
            None => db.executor().execute_named_with(&self.sql, value_of),
        }
    }

    /// Query and return a single value
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stmt = db.prepare("SELECT name FROM users WHERE id = $1")?;
    /// let name: String = stmt.query_one((1,))?;
    /// ```
    pub fn query_one<T: FromValue, P: Params>(&self, params: P) -> Result<T> {
        let row = self.query(params)?.next().ok_or(Error::NoRowsReturned)??;
        row.get(0)
    }

    /// Query and return an optional single value
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stmt = db.prepare("SELECT name FROM users WHERE id = $1")?;
    /// let name: Option<String> = stmt.query_opt((999,))?;
    /// ```
    pub fn query_opt<T: FromValue, P: Params>(&self, params: P) -> Result<Option<T>> {
        match self.query(params)?.next() {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(row)) => Ok(Some(row.get(0)?)),
        }
    }

    /// Get the SQL text of this statement
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// The compiled plan when this statement was prepared on `engine`.
    /// Compiled state is checked against its engine's schema epoch only,
    /// so another engine plans the SQL afresh
    pub(crate) fn plan_on(
        &self,
        engine: &Arc<crate::storage::mvcc::engine::MVCCEngine>,
    ) -> Option<&CachedPlanRef> {
        let plan = self.plan.as_ref()?;
        let db = self.db_weak.upgrade()?;
        Arc::ptr_eq(db.executor.engine(), engine).then_some(plan)
    }

    /// Execute the prepared statement inside `tx`
    pub fn execute_in<P: Params>(&self, tx: &mut Transaction, params: P) -> Result<i64> {
        let ctx = ExecutionContext::with_params(params.into_params());
        Ok(tx.execute_prepared_statement(self, ctx)?.rows_affected())
    }

    /// Query using the prepared statement inside `tx`
    pub fn query_in<P: Params>(&self, tx: &mut Transaction, params: P) -> Result<Rows> {
        let ctx = ExecutionContext::with_params(params.into_params());
        Ok(Rows::new(tx.execute_prepared_statement(self, ctx)?))
    }

    /// Execute the prepared statement inside `tx` with named parameters
    pub fn execute_named_in(&self, tx: &mut Transaction, params: NamedParams) -> Result<i64> {
        Ok(tx
            .execute_prepared_statement_named(self, &mut value_in(params.as_map()))?
            .rows_affected())
    }

    /// Query using the prepared statement inside `tx` with named parameters
    pub fn query_named_in(&self, tx: &mut Transaction, params: NamedParams) -> Result<Rows> {
        Ok(Rows::new(tx.execute_prepared_statement_named(
            self,
            &mut value_in(params.as_map()),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepared_statement_execute() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();

        let stmt = db.prepare("INSERT INTO users VALUES ($1, $2)").unwrap();

        stmt.execute((1, "Alice")).unwrap();
        stmt.execute((2, "Bob")).unwrap();
        stmt.execute((3, "Charlie")).unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_prepared_statement_query() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();
        db.execute(
            "INSERT INTO users VALUES ($1, $2), ($3, $4), ($5, $6)",
            (1, "Alice", 2, "Bob", 3, "Charlie"),
        )
        .unwrap();

        let stmt = db.prepare("SELECT name FROM users WHERE id = $1").unwrap();

        let name: String = stmt.query_one((1,)).unwrap();
        assert_eq!(name, "Alice");

        let name: String = stmt.query_one((2,)).unwrap();
        assert_eq!(name, "Bob");

        let name: String = stmt.query_one((3,)).unwrap();
        assert_eq!(name, "Charlie");
    }

    #[test]
    fn test_prepared_statement_query_opt() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))
            .unwrap();

        let stmt = db.prepare("SELECT name FROM users WHERE id = $1").unwrap();

        let name: Option<String> = stmt.query_opt((1,)).unwrap();
        assert_eq!(name, Some("Alice".to_string()));

        let name: Option<String> = stmt.query_opt((999,)).unwrap();
        assert_eq!(name, None);
    }

    #[test]
    fn test_prepared_statement_sql() {
        let db = Database::open_in_memory().unwrap();
        let stmt = db.prepare("SELECT 1").unwrap();
        assert_eq!(stmt.sql(), "SELECT 1");
    }
}
