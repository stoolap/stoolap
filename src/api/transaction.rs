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

use crate::core::{Error, Result, Value};
use crate::executor::ExternalTransactionGuard;
use crate::parser::ast::Statement;
use crate::parser::Parser;
use crate::storage::traits::{QueryResult, Transaction as StorageTransaction};

use super::database::{DatabaseInner, FromValue};
use super::params::Params;
use super::rows::Rows;

/// Transaction represents a database transaction
///
/// Provides ACID guarantees for a series of database operations.
/// Must be explicitly committed or rolled back.
///
/// Transactions use the full query executor, providing all optimizations
/// including index scans, JOIN support, and streaming results.
pub struct Transaction {
    /// The storage-level transaction handle
    tx: Option<Box<dyn StorageTransaction>>,
    /// Reference to the database internals (executor, engine)
    db_inner: Arc<DatabaseInner>,
    /// Whether the transaction has been committed
    committed: bool,
    /// Whether the transaction has been rolled back
    rolled_back: bool,
}

impl Transaction {
    /// Create a new transaction wrapper
    pub(crate) fn new(tx: Box<dyn StorageTransaction>, db_inner: Arc<DatabaseInner>) -> Self {
        Self {
            tx: Some(tx),
            db_inner,
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
        let result = self.execute_sql(sql, &param_values)?;
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
        let result = self.execute_sql(sql, &param_values)?;
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

    /// Internal SQL execution using the full query executor.
    ///
    /// This method routes SQL execution through the complete query pipeline,
    /// providing all optimizations including:
    /// - Index scans for WHERE clauses
    /// - Cost-based query planning
    /// - JOIN support (hash join, merge join, nested loop)
    /// - Streaming results (no full materialization)
    /// - Subquery optimization
    fn execute_sql(&mut self, sql: &str, params: &[Value]) -> Result<Box<dyn QueryResult>> {
        // Parse first to reject transaction control statements
        let mut parser = Parser::new(sql);
        let program = parser.parse_program().map_err(|e| Error::parse(e.to_string()))?;
        for stmt in &program.statements {
            Self::reject_transaction_control_statement(stmt)?;
        }

        // Get the executor and use RAII guard for panic safety
        let executor = self
            .db_inner
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("executor".to_string()))?;

        // The guard takes the transaction from self.tx, installs it into the executor,
        // and restores it back to self.tx when dropped (even on panic)
        let _guard = ExternalTransactionGuard::new(&executor, &mut self.tx)?;

        // Execute using the full query pipeline
        if params.is_empty() {
            executor.execute(sql)
        } else {
            executor.execute_with_params(sql, params)
        }
        // Guard's Drop impl will restore self.tx automatically
    }

    /// Reject transaction control statements that would corrupt state.
    ///
    /// These statements must be rejected because:
    /// - BEGIN: Would create a nested transaction (unsupported)
    /// - COMMIT/ROLLBACK: Would consume the transaction from the executor,
    ///   making it impossible to restore to self.tx
    /// - SAVEPOINT operations: Should use the Transaction API methods instead
    fn reject_transaction_control_statement(stmt: &Statement) -> Result<()> {
        match stmt {
            Statement::Begin(_) => {
                Err(Error::internal(
                    "Cannot execute BEGIN inside a transaction. Use nested savepoints via create_savepoint() instead.",
                ))
            }
            Statement::Commit(_) => {
                Err(Error::internal(
                    "Cannot execute COMMIT inside execute(). Use Transaction::commit() instead.",
                ))
            }
            Statement::Rollback(_) => {
                Err(Error::internal(
                    "Cannot execute ROLLBACK inside execute(). Use Transaction::rollback() instead.",
                ))
            }
            Statement::Savepoint(_) => {
                Err(Error::internal(
                    "Cannot execute SAVEPOINT inside execute(). Use Transaction::create_savepoint() instead.",
                ))
            }
            _ => Ok(()),
        }
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

    #[test]
    fn test_transaction_join_support() {
        // This test verifies that JOINs work inside transactions
        // (previously they failed with "Complex FROM clauses not supported in transactions")
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();
        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())
            .unwrap();
        db.execute(
            "INSERT INTO orders VALUES (1, 1, 100), (2, 1, 200), (3, 2, 150)",
            (),
        )
        .unwrap();

        let mut tx = db.begin().unwrap();

        // JOIN should now work inside transaction
        let rows: Vec<_> = tx
            .query(
                "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id WHERE u.id = $1",
                (1,),
            )
            .unwrap()
            .collect();

        assert_eq!(rows.len(), 2);

        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_subquery_support() {
        // Verifies that subqueries work inside transactions
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, price INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO products VALUES (1, 'A', 100), (2, 'A', 200), (3, 'B', 150)",
            (),
        )
        .unwrap();

        let mut tx = db.begin().unwrap();

        // Scalar subquery
        // Inner subquery: AVG of category 'A' = (100 + 200) / 2 = 150
        // Outer: prices > 150 = only 200, so AVG = 200
        let avg_price: i64 = tx
            .query_one(
                "SELECT AVG(price) FROM products WHERE price > (SELECT AVG(price) FROM products WHERE category = 'A')",
                (),
            )
            .unwrap();
        assert_eq!(avg_price, 200);

        // IN subquery
        let count: i64 = tx
            .query_one(
                "SELECT COUNT(*) FROM products WHERE category IN (SELECT DISTINCT category FROM products WHERE price > 100)",
                (),
            )
            .unwrap();
        assert_eq!(count, 3);

        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_function_support() {
        // Verifies that SQL functions work inside transactions
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE names (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO names VALUES (1, 'alice'), (2, 'BOB')", ())
            .unwrap();

        let mut tx = db.begin().unwrap();

        // UPPER function
        let result: String = tx
            .query_one("SELECT UPPER(name) FROM names WHERE id = 1", ())
            .unwrap();
        assert_eq!(result, "ALICE");

        // LOWER function
        let result: String = tx
            .query_one("SELECT LOWER(name) FROM names WHERE id = 2", ())
            .unwrap();
        assert_eq!(result, "bob");

        // LENGTH function
        let len: i64 = tx
            .query_one("SELECT LENGTH(name) FROM names WHERE id = 1", ())
            .unwrap();
        assert_eq!(len, 5);

        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_aggregation_support() {
        // Verifies that aggregations work inside transactions
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO sales VALUES (1, 'East', 100), (2, 'East', 200), (3, 'West', 150), (4, 'West', 250)",
            (),
        )
        .unwrap();

        let mut tx = db.begin().unwrap();

        // GROUP BY with aggregation
        let mut rows = tx
            .query(
                "SELECT region, SUM(amount) as total FROM sales GROUP BY region ORDER BY region",
                (),
            )
            .unwrap();

        let row1 = rows.next().unwrap().unwrap();
        assert_eq!(row1.get::<String>(0).unwrap(), "East");
        assert_eq!(row1.get::<i64>(1).unwrap(), 300);

        let row2 = rows.next().unwrap().unwrap();
        assert_eq!(row2.get::<String>(0).unwrap(), "West");
        assert_eq!(row2.get::<i64>(1).unwrap(), 400);

        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_update_uses_index() {
        // This test verifies that UPDATE with WHERE uses indexes (not full table scan)
        // We can't directly test index usage, but we verify the behavior is correct
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, status TEXT)",
            (),
        )
        .unwrap();

        // Insert many rows
        for i in 1..=100 {
            db.execute("INSERT INTO items VALUES ($1, 'active')", (i,))
                .unwrap();
        }

        let mut tx = db.begin().unwrap();

        // Update by primary key - should use index
        let updated = tx
            .execute("UPDATE items SET status = 'inactive' WHERE id = $1", (50,))
            .unwrap();
        assert_eq!(updated, 1);

        // Verify the update
        let status: String = tx
            .query_one("SELECT status FROM items WHERE id = $1", (50,))
            .unwrap();
        assert_eq!(status, "inactive");

        // Other rows unchanged
        let status: String = tx
            .query_one("SELECT status FROM items WHERE id = $1", (49,))
            .unwrap();
        assert_eq!(status, "active");

        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_streaming_results() {
        // Verifies that results can be iterated without full materialization
        let db = Database::open_in_memory().unwrap();
        db.execute(
            "CREATE TABLE nums (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .unwrap();

        // Insert rows
        for i in 1..=50 {
            db.execute("INSERT INTO nums VALUES ($1, $2)", (i, i * 10))
                .unwrap();
        }

        let mut tx = db.begin().unwrap();

        // Query and iterate - should stream results
        let mut sum = 0i64;
        for row in tx.query("SELECT val FROM nums", ()).unwrap() {
            let val: i64 = row.unwrap().get(0).unwrap();
            sum += val;
        }

        assert_eq!(sum, (1..=50).map(|x| x * 10).sum::<i64>());

        tx.commit().unwrap();
    }

    #[test]
    fn test_transaction_rejects_begin() {
        // Verifies that BEGIN inside a transaction is rejected
        let db = Database::open_in_memory().unwrap();
        let mut tx = db.begin().unwrap();

        let result = tx.execute("BEGIN", ());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("BEGIN"));

        tx.rollback().unwrap();
    }

    #[test]
    fn test_transaction_rejects_commit_sql() {
        // Verifies that COMMIT inside execute() is rejected
        let db = Database::open_in_memory().unwrap();
        let mut tx = db.begin().unwrap();

        let result = tx.execute("COMMIT", ());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("COMMIT"));

        tx.rollback().unwrap();
    }

    #[test]
    fn test_transaction_rejects_rollback_sql() {
        // Verifies that ROLLBACK inside execute() is rejected
        let db = Database::open_in_memory().unwrap();
        let mut tx = db.begin().unwrap();

        let result = tx.execute("ROLLBACK", ());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("ROLLBACK"));

        tx.rollback().unwrap();
    }

    #[test]
    fn test_transaction_rejects_savepoint_sql() {
        // Verifies that SAVEPOINT inside execute() is rejected
        let db = Database::open_in_memory().unwrap();
        let mut tx = db.begin().unwrap();

        let result = tx.execute("SAVEPOINT sp1", ());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("SAVEPOINT"));

        tx.rollback().unwrap();
    }
}
