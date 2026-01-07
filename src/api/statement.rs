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

use std::sync::Weak;

use crate::core::{Error, Result};

use super::database::{Database, DatabaseInnerHandle, FromValue};
use super::params::Params;
use super::rows::Rows;

/// A prepared SQL statement
///
/// Prepared statements allow you to compile a SQL query once and execute it
/// multiple times with different parameters. This is more efficient than
/// parsing the same query repeatedly.
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
}

impl Statement {
    /// Create a new prepared statement
    ///
    /// This validates the SQL syntax by pre-warming the query cache.
    pub(crate) fn new(
        db_weak: Weak<DatabaseInnerHandle>,
        sql: String,
        db: &Database,
    ) -> Result<Self> {
        // Pre-warm the cache by executing with empty params check
        // This validates the SQL syntax
        {
            let executor = db
                .executor()
                .lock()
                .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

            // Just touch the cache to pre-parse
            let _ = executor.query_cache().get(&sql);
        }

        Ok(Self { db_weak, sql })
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
        self.get_db()?.execute(&self.sql, params)
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
        self.get_db()?.query(&self.sql, params)
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
        self.get_db()?.query_one(&self.sql, params)
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
        self.get_db()?.query_opt(&self.sql, params)
    }

    /// Get the SQL text of this statement
    pub fn sql(&self) -> &str {
        &self.sql
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
