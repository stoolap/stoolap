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

use crate::core::{Error, Result};
use crate::executor::context::ExecutionContext;
use crate::executor::query_cache::CachedPlanRef;
use crate::parser::Parser;

use super::database::{Database, DatabaseInnerHandle, FromValue};
use super::params::Params;
use super::rows::Rows;

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
        let plan = {
            let executor = db
                .executor()
                .lock()
                .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

            // Check if already cached (e.g. same SQL prepared twice).
            //
            // On a read-only executor, refuse write-intent SQL up front —
            // even on a cache hit produced by an earlier writable caller.
            // Without this, `prepare()` would succeed against a read-only
            // Database and the refusal would only fire later at
            // `execute()` / `query()`, which is harder to debug than the
            // immediate rejection.
            if let Some(cached) = executor.query_cache().get(&sql) {
                if executor.is_read_only() {
                    if let Some(reason) = cached.statement.write_reason() {
                        return Err(Error::read_only_violation_at("parser", reason));
                    }
                }
                Some(cached)
            } else {
                let mut parser = Parser::new(&sql);
                let mut program = parser
                    .parse_program()
                    .map_err(|e| Error::parse(e.to_string()))?;

                if program.statements.len() == 1 {
                    let stmt = program.statements.pop().unwrap();
                    // Reject bare expression statements — they indicate
                    // unrecognised SQL (e.g. "SELECTX INVALID").
                    if matches!(stmt, crate::parser::ast::Statement::Expression(_)) {
                        return Err(Error::parse(format!(
                            "invalid SQL: unrecognised statement: {}",
                            sql
                        )));
                    }
                    // Same read-only refusal for the freshly-parsed path.
                    if executor.is_read_only() {
                        if let Some(reason) = stmt.write_reason() {
                            return Err(Error::read_only_violation_at("parser", reason));
                        }
                    }
                    let (has_params, param_count) = crate::executor::count_parameters(&stmt);
                    let stmt_arc = Arc::new(stmt);
                    let cached =
                        executor
                            .query_cache()
                            .put(&sql, stmt_arc, has_params, param_count);
                    Some(cached)
                } else {
                    // Reject if any statement is a bare expression
                    // (indicates unrecognised SQL tokens).
                    for s in &program.statements {
                        if matches!(s, crate::parser::ast::Statement::Expression(_)) {
                            return Err(Error::parse(format!(
                                "invalid SQL: unrecognised statement: {}",
                                sql
                            )));
                        }
                    }
                    // Multi-statement SQL on a read-only executor: refuse
                    // any program containing at least one write-intent
                    // statement. Single fresh parse — no re-parse needed.
                    if executor.is_read_only() {
                        for s in &program.statements {
                            if let Some(reason) = s.write_reason() {
                                return Err(Error::read_only_violation_at("parser", reason));
                            }
                        }
                    }
                    // Multi-statement SQL: validated but not cacheable
                    None
                }
            }
        };

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
            // Prepared-statement cached-plan path bypasses the
            // `Database::execute` entry point, so the SWMR
            // lease heartbeat / refresh that lives there
            // doesn't fire — call it explicitly here. The
            // multi-statement fallback below already routes
            // through `db.execute`, which heartbeats
            // internally; doing it here too would double the
            // epoch / shm polling on every execution.
            db.heartbeat_and_maybe_refresh()?;
            let executor = db
                .executor()
                .lock()
                .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
            let ctx = ExecutionContext::with_params(params.into_params());
            let result = executor.execute_with_cached_plan(plan, &ctx)?;
            Ok(result.rows_affected())
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
            // See `Statement::execute` for the rationale —
            // heartbeat lives on the cached-plan branch only,
            // since the multi-statement fallback below routes
            // through `db.query` which heartbeats internally.
            db.heartbeat_and_maybe_refresh()?;
            let executor = db
                .executor()
                .lock()
                .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;
            let ctx = ExecutionContext::with_params(params.into_params());
            let result = executor.execute_with_cached_plan(plan, &ctx)?;
            Ok(Rows::new(result))
        } else {
            db.query(&self.sql, params)
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

    /// Get the AST statement from the cached plan (if available).
    ///
    /// Returns `None` for multi-statement SQL which doesn't cache a plan.
    #[cfg(feature = "ffi")]
    pub(crate) fn ast_statement(&self) -> Option<&Arc<crate::parser::ast::Statement>> {
        self.plan.as_ref().map(|p| &p.statement)
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
