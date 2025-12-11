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

//! Database struct and operations
//!
//! Provides a modern, ergonomic Rust API for database operations.
//!
//! # Examples
//!
//! ```ignore
//! use stoolap::{Database, params};
//!
//! let db = Database::open("memory://")?;
//!
//! // DDL - no params needed
//! db.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)", ())?;
//!
//! // Insert with params - using tuple syntax
//! db.execute("INSERT INTO users VALUES ($1, $2, $3)", (1, "Alice", 30))?;
//!
//! // Insert with params! macro
//! db.execute("INSERT INTO users VALUES ($1, $2, $3)", params![2, "Bob", 25])?;
//!
//! // Query with iteration
//! for row in db.query("SELECT * FROM users WHERE age > $1", (20,))? {
//!     let row = row?;
//!     let name: String = row.get("name")?;
//!     println!("{}", name);
//! }
//!
//! // Query single value
//! let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::core::{Error, IsolationLevel, Result, Value};
use crate::executor::Executor;
use crate::functions::FunctionRegistry;
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::traits::Engine;
use crate::storage::{Config, SyncMode};

use super::params::{NamedParams, Params};
use super::rows::{FromRow, Rows};
use super::statement::Statement;
use super::transaction::Transaction;

/// Storage scheme constants
pub const MEMORY_SCHEME: &str = "memory";
pub const FILE_SCHEME: &str = "file";

/// Global database registry to ensure single instance per DSN
static DATABASE_REGISTRY: std::sync::LazyLock<RwLock<HashMap<String, Arc<DatabaseInner>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

/// Inner database state (shared between Database instances with same DSN)
struct DatabaseInner {
    engine: Arc<MVCCEngine>,
    executor: Mutex<Executor>,
    dsn: String,
}

impl Drop for DatabaseInner {
    fn drop(&mut self) {
        // Close the engine when the last reference is dropped
        let _ = self.engine.close_engine();
    }
}

/// Database represents a Stoolap database connection
///
/// This is the main entry point for using Stoolap. It wraps the storage engine
/// and executor, providing a simple API for executing SQL queries.
///
/// # Thread Safety
///
/// Database is thread-safe and can be shared across threads via cloning.
/// Each clone shares the same underlying storage engine.
///
/// # Examples
///
/// ```ignore
/// use stoolap::{Database, params};
///
/// // Open in-memory database
/// let db = Database::open("memory://")?;
///
/// // Create table
/// db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())?;
///
/// // Insert with parameters
/// db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
///
/// // Query
/// for row in db.query("SELECT * FROM users", ())? {
///     let row = row?;
///     println!("{}: {}", row.get::<i64>("id")?, row.get::<String>("name")?);
/// }
/// ```
#[derive(Clone)]
pub struct Database {
    inner: Arc<DatabaseInner>,
}

impl Database {
    /// Open a database connection
    ///
    /// The DSN (Data Source Name) specifies the database location:
    /// - `memory://` - In-memory database (data lost when closed)
    /// - `file:///path/to/db` - Persistent database at the specified path
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // In-memory database
    /// let db = Database::open("memory://")?;
    ///
    /// // Persistent database
    /// let db = Database::open("file:///tmp/mydb")?;
    /// ```
    ///
    /// # Engine Reuse
    ///
    /// Opening the same DSN multiple times returns the same engine instance.
    /// This ensures consistency and prevents data corruption.
    pub fn open(dsn: &str) -> Result<Self> {
        // Check if we already have an engine for this DSN
        {
            let registry = DATABASE_REGISTRY
                .read()
                .map_err(|_| Error::internal("Failed to acquire registry read lock"))?;
            if let Some(inner) = registry.get(dsn) {
                return Ok(Database {
                    inner: Arc::clone(inner),
                });
            }
        }

        // Need to create a new engine - acquire write lock
        let mut registry = DATABASE_REGISTRY
            .write()
            .map_err(|_| Error::internal("Failed to acquire registry write lock"))?;

        // Double-check after acquiring write lock
        if let Some(inner) = registry.get(dsn) {
            return Ok(Database {
                inner: Arc::clone(inner),
            });
        }

        // Parse the DSN
        let (scheme, path) = Self::parse_dsn(dsn)?;

        // Create the engine based on scheme
        let engine = match scheme.as_str() {
            MEMORY_SCHEME => {
                let engine = MVCCEngine::in_memory();
                engine.open_engine()?;
                Arc::new(engine)
            }
            FILE_SCHEME => {
                // Parse optional query parameters
                let (_clean_path, config) = Self::parse_file_config(&path)?;

                let engine = MVCCEngine::new(config);
                engine.open_engine()?;
                Arc::new(engine)
            }
            _ => {
                return Err(Error::parse(format!(
                    "Unsupported scheme '{}'. Use 'memory://' or 'file://path'",
                    scheme
                )));
            }
        };

        // Create executor with function registry
        let function_registry = Arc::new(FunctionRegistry::new());
        let executor = Executor::with_function_registry(Arc::clone(&engine), function_registry);

        let inner = Arc::new(DatabaseInner {
            engine,
            executor: Mutex::new(executor),
            dsn: dsn.to_string(),
        });

        // Store in registry
        registry.insert(dsn.to_string(), Arc::clone(&inner));

        Ok(Database { inner })
    }

    /// Open an in-memory database
    ///
    /// This is a convenience method that creates a new in-memory database.
    /// Each call creates a unique instance (unlike `open("memory://")` which
    /// would share the same instance).
    pub fn open_in_memory() -> Result<Self> {
        // Create engine directly without registry (each in_memory call is unique)
        let engine = MVCCEngine::in_memory();
        engine.open_engine()?;
        let engine = Arc::new(engine);

        let function_registry = Arc::new(FunctionRegistry::new());
        let executor = Executor::with_function_registry(Arc::clone(&engine), function_registry);

        let inner = Arc::new(DatabaseInner {
            engine,
            executor: Mutex::new(executor),
            dsn: "memory://".to_string(),
        });

        Ok(Database { inner })
    }

    /// Parse a DSN into scheme and path
    fn parse_dsn(dsn: &str) -> Result<(String, String)> {
        let idx = dsn
            .find("://")
            .ok_or_else(|| Error::parse("Invalid DSN format: expected scheme://path"))?;

        let scheme = dsn[..idx].to_lowercase();
        let path = dsn[idx + 3..].to_string();

        // Validate scheme
        match scheme.as_str() {
            MEMORY_SCHEME | FILE_SCHEME => {}
            _ => {
                return Err(Error::parse(format!(
                    "Unsupported scheme '{}'. Use 'memory://' or 'file://path'",
                    scheme
                )));
            }
        }

        // Validate file path
        if scheme == FILE_SCHEME {
            let clean_path = if path.contains('?') {
                &path[..path.find('?').unwrap()]
            } else {
                &path
            };

            if clean_path.is_empty() {
                return Err(Error::parse("file:// scheme requires a non-empty path"));
            }
        }

        Ok((scheme, path))
    }

    /// Parse file:// config from query parameters
    fn parse_file_config(path: &str) -> Result<(String, Config)> {
        let (clean_path, query) = if let Some(idx) = path.find('?') {
            (path[..idx].to_string(), Some(&path[idx + 1..]))
        } else {
            (path.to_string(), None)
        };

        let mut config = Config::with_path(&clean_path);

        // Parse query parameters
        if let Some(query) = query {
            for param in query.split('&') {
                let mut parts = param.splitn(2, '=');
                let key = parts.next().unwrap_or("");
                let value = parts.next().unwrap_or("");

                match key {
                    // Sync mode: sync=none|normal|full
                    "sync_mode" | "sync" => {
                        config.persistence.sync_mode = match value.to_lowercase().as_str() {
                            "none" | "off" | "0" => SyncMode::None,
                            "normal" | "1" => SyncMode::Normal,
                            "full" | "2" => SyncMode::Full,
                            _ => SyncMode::Normal,
                        };
                    }
                    // Snapshot interval in seconds: snapshot_interval=300
                    "snapshot_interval" => {
                        if let Ok(secs) = value.parse::<u32>() {
                            config.persistence.snapshot_interval = secs;
                        }
                    }
                    // Number of snapshots to keep: keep_snapshots=5
                    "keep_snapshots" => {
                        if let Ok(count) = value.parse::<u32>() {
                            config.persistence.keep_snapshots = count;
                        }
                    }
                    // WAL flush trigger in bytes: wal_flush_trigger=32768
                    "wal_flush_trigger" => {
                        if let Ok(bytes) = value.parse::<usize>() {
                            config.persistence.wal_flush_trigger = bytes;
                        }
                    }
                    // WAL buffer size in bytes: wal_buffer_size=65536
                    "wal_buffer_size" => {
                        if let Ok(bytes) = value.parse::<usize>() {
                            config.persistence.wal_buffer_size = bytes;
                        }
                    }
                    // WAL max size in bytes: wal_max_size=67108864
                    "wal_max_size" => {
                        if let Ok(bytes) = value.parse::<usize>() {
                            config.persistence.wal_max_size = bytes;
                        }
                    }
                    // Commit batch size: commit_batch_size=100
                    "commit_batch_size" => {
                        if let Ok(size) = value.parse::<u32>() {
                            config.persistence.commit_batch_size = size;
                        }
                    }
                    // Sync interval in ms: sync_interval_ms=10
                    "sync_interval_ms" | "sync_interval" => {
                        if let Ok(ms) = value.parse::<u32>() {
                            config.persistence.sync_interval_ms = ms;
                        }
                    }
                    // WAL compression: wal_compression=on|off
                    "wal_compression" => {
                        config.persistence.wal_compression =
                            matches!(value.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                    }
                    // Snapshot compression: snapshot_compression=on|off
                    "snapshot_compression" => {
                        config.persistence.snapshot_compression =
                            matches!(value.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                    }
                    // Both compressions: compression=on|off
                    "compression" => {
                        let enabled =
                            matches!(value.to_lowercase().as_str(), "on" | "true" | "1" | "yes");
                        config.persistence.wal_compression = enabled;
                        config.persistence.snapshot_compression = enabled;
                    }
                    // Compression threshold in bytes: compression_threshold=64
                    "compression_threshold" => {
                        if let Ok(bytes) = value.parse::<usize>() {
                            config.persistence.compression_threshold = bytes;
                        }
                    }
                    _ => {} // Ignore unknown parameters
                }
            }
        }

        Ok((clean_path, config))
    }

    /// Execute a SQL statement
    ///
    /// Use this for DDL (CREATE, DROP, ALTER) and DML (INSERT, UPDATE, DELETE) statements.
    ///
    /// # Parameters
    ///
    /// Parameters can be passed using:
    /// - Empty tuple `()` for no parameters
    /// - Tuple syntax `(1, "Alice", 30)` for multiple parameters
    /// - `params!` macro `params![1, "Alice", 30]`
    ///
    /// # Returns
    ///
    /// Returns the number of rows affected for DML statements, or 0 for DDL.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // DDL - no parameters
    /// db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())?;
    ///
    /// // DML with tuple parameters
    /// db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
    ///
    /// // DML with params! macro
    /// db.execute("INSERT INTO users VALUES ($1, $2)", params![2, "Bob"])?;
    ///
    /// // Update with mixed types
    /// let affected = db.execute(
    ///     "UPDATE users SET name = $1 WHERE id = $2",
    ///     ("Charlie", 1)
    /// )?;
    /// ```
    pub fn execute<P: Params>(&self, sql: &str, params: P) -> Result<i64> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::internal("Failed to acquire executor lock"))?;

        let param_values = params.into_params();
        let result = if param_values.is_empty() {
            executor.execute(sql)?
        } else {
            executor.execute_with_params(sql, &param_values)?
        };
        Ok(result.rows_affected())
    }

    /// Execute a query that returns rows
    ///
    /// # Parameters
    ///
    /// Parameters can be passed using:
    /// - Empty tuple `()` for no parameters
    /// - Tuple syntax `(value,)` for single parameter (note trailing comma)
    /// - Tuple syntax `(1, "Alice")` for multiple parameters
    /// - `params!` macro `params![1, "Alice"]`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Query all rows
    /// for row in db.query("SELECT * FROM users", ())? {
    ///     let row = row?;
    ///     let id: i64 = row.get(0)?;
    ///     let name: String = row.get("name")?;
    /// }
    ///
    /// // Query with parameters
    /// for row in db.query("SELECT * FROM users WHERE age > $1", (18,))? {
    ///     // ...
    /// }
    ///
    /// // Collect into Vec
    /// let users: Vec<_> = db.query("SELECT * FROM users", ())?
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// ```
    pub fn query<P: Params>(&self, sql: &str, params: P) -> Result<Rows> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::internal("Failed to acquire executor lock"))?;

        let param_values = params.into_params();
        let result = if param_values.is_empty() {
            executor.execute(sql)?
        } else {
            executor.execute_with_params(sql, &param_values)?
        };
        Ok(Rows::new(result))
    }

    /// Execute a query and return a single value
    ///
    /// This is a convenience method for queries that return a single row with a single column.
    /// Returns an error if the query returns no rows.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;
    /// let name: String = db.query_one("SELECT name FROM users WHERE id = $1", (1,))?;
    /// ```
    pub fn query_one<T: FromValue, P: Params>(&self, sql: &str, params: P) -> Result<T> {
        let row = self
            .query(sql, params)?
            .next()
            .ok_or_else(|| Error::internal("Query returned no rows"))??;
        row.get(0)
    }

    /// Execute a query and return an optional single value
    ///
    /// Like `query_one`, but returns `None` if no rows are returned.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let name: Option<String> = db.query_opt("SELECT name FROM users WHERE id = $1", (999,))?;
    /// assert!(name.is_none());
    /// ```
    pub fn query_opt<T: FromValue, P: Params>(&self, sql: &str, params: P) -> Result<Option<T>> {
        match self.query(sql, params)?.next() {
            Some(row) => Ok(Some(row?.get(0)?)),
            None => Ok(None),
        }
    }

    /// Prepare a SQL statement for repeated execution
    ///
    /// Prepared statements are more efficient when executing the same query
    /// multiple times with different parameters.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let stmt = db.prepare("SELECT * FROM users WHERE id = $1")?;
    ///
    /// // Execute multiple times with different parameters
    /// for id in 1..=10 {
    ///     for row in stmt.query((id,))? {
    ///         // ...
    ///     }
    /// }
    /// ```
    pub fn prepare(&self, sql: &str) -> Result<Statement> {
        Statement::new(self.clone(), sql.to_string())
    }

    /// Execute a statement with named parameters
    ///
    /// Named parameters use the `:name` syntax in SQL queries.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, named_params};
    ///
    /// let db = Database::open("memory://")?;
    /// db.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)", ())?;
    ///
    /// // Insert with named params
    /// db.execute_named(
    ///     "INSERT INTO users VALUES (:id, :name, :age)",
    ///     named_params!{ id: 1, name: "Alice", age: 30 }
    /// )?;
    ///
    /// // Update with named params
    /// db.execute_named(
    ///     "UPDATE users SET name = :name WHERE id = :id",
    ///     named_params!{ id: 1, name: "Alicia" }
    /// )?;
    /// ```
    pub fn execute_named(&self, sql: &str, params: NamedParams) -> Result<i64> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::internal("Failed to acquire executor lock"))?;

        let result = executor.execute_with_named_params(sql, params.into_inner())?;
        Ok(result.rows_affected())
    }

    /// Execute a query with named parameters
    ///
    /// Named parameters use the `:name` syntax in SQL queries.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, named_params};
    ///
    /// let db = Database::open("memory://")?;
    /// db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())?;
    /// db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())?;
    ///
    /// // Query with named params
    /// for row in db.query_named(
    ///     "SELECT * FROM users WHERE name = :name",
    ///     named_params!{ name: "Alice" }
    /// )? {
    ///     let row = row?;
    ///     println!("Found user: id={}", row.get::<i64>(0)?);
    /// }
    /// ```
    pub fn query_named(&self, sql: &str, params: NamedParams) -> Result<Rows> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::internal("Failed to acquire executor lock"))?;

        let result = executor.execute_with_named_params(sql, params.into_inner())?;
        Ok(Rows::new(result))
    }

    /// Execute a query with named parameters and return a single value
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, named_params};
    ///
    /// let count: i64 = db.query_one_named(
    ///     "SELECT COUNT(*) FROM users WHERE age > :min_age",
    ///     named_params!{ min_age: 18 }
    /// )?;
    /// ```
    pub fn query_one_named<T: FromValue>(&self, sql: &str, params: NamedParams) -> Result<T> {
        let mut rows = self.query_named(sql, params)?;
        match rows.next() {
            Some(Ok(row)) => row.get(0),
            Some(Err(e)) => Err(e),
            None => Err(Error::internal("Query returned no rows")),
        }
    }

    /// Execute a query and map results to structs
    ///
    /// This method executes a query and converts each row to a struct
    /// that implements the `FromRow` trait.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, FromRow, ResultRow, Result};
    ///
    /// struct User {
    ///     id: i64,
    ///     name: String,
    /// }
    ///
    /// impl FromRow for User {
    ///     fn from_row(row: &ResultRow) -> Result<Self> {
    ///         Ok(User {
    ///             id: row.get(0)?,
    ///             name: row.get(1)?,
    ///         })
    ///     }
    /// }
    ///
    /// let db = Database::open("memory://")?;
    /// db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())?;
    /// db.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')", ())?;
    ///
    /// // Query and map to structs
    /// let users: Vec<User> = db.query_as("SELECT id, name FROM users", ())?;
    /// assert_eq!(users.len(), 2);
    /// assert_eq!(users[0].name, "Alice");
    /// ```
    pub fn query_as<T: FromRow, P: Params>(&self, sql: &str, params: P) -> Result<Vec<T>> {
        let rows = self.query(sql, params)?;
        rows.map(|r| r.and_then(|row| T::from_row(&row))).collect()
    }

    /// Execute a query with named parameters and map results to structs
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::{Database, FromRow, ResultRow, Result, named_params};
    ///
    /// struct Product {
    ///     id: i64,
    ///     name: String,
    ///     price: f64,
    /// }
    ///
    /// impl FromRow for Product {
    ///     fn from_row(row: &ResultRow) -> Result<Self> {
    ///         Ok(Product {
    ///             id: row.get(0)?,
    ///             name: row.get(1)?,
    ///             price: row.get(2)?,
    ///         })
    ///     }
    /// }
    ///
    /// let products: Vec<Product> = db.query_as_named(
    ///     "SELECT id, name, price FROM products WHERE price > :min_price",
    ///     named_params!{ min_price: 10.0 }
    /// )?;
    /// ```
    pub fn query_as_named<T: FromRow>(&self, sql: &str, params: NamedParams) -> Result<Vec<T>> {
        let rows = self.query_named(sql, params)?;
        rows.map(|r| r.and_then(|row| T::from_row(&row))).collect()
    }

    /// Begin a new transaction with default isolation level
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let tx = db.begin()?;
    /// tx.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
    /// tx.commit()?;
    /// ```
    pub fn begin(&self) -> Result<Transaction> {
        self.begin_with_isolation(IsolationLevel::ReadCommitted)
    }

    /// Begin a new transaction with a specific isolation level
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use stoolap::IsolationLevel;
    ///
    /// let tx = db.begin_with_isolation(IsolationLevel::Snapshot)?;
    /// // All reads in this transaction see a consistent snapshot
    /// tx.execute("UPDATE users SET balance = balance - 100 WHERE id = $1", (1,))?;
    /// tx.commit()?;
    /// ```
    pub fn begin_with_isolation(&self, isolation: IsolationLevel) -> Result<Transaction> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::internal("Failed to acquire executor lock"))?;

        let tx = executor.begin_transaction_with_isolation(isolation)?;
        Ok(Transaction::new(
            tx,
            Arc::clone(executor.function_registry()),
        ))
    }

    /// Get the underlying storage engine
    ///
    /// This is primarily for advanced use cases and testing.
    pub fn engine(&self) -> &Arc<MVCCEngine> {
        &self.inner.engine
    }

    /// Close the database connection
    ///
    /// This removes the database from the global registry and closes the engine,
    /// releasing the file lock immediately so another process can open the database.
    ///
    /// Note: The engine is also closed automatically when all Database instances
    /// are dropped.
    pub fn close(&self) -> Result<()> {
        // Remove from registry
        let mut registry = DATABASE_REGISTRY
            .write()
            .map_err(|_| Error::internal("Failed to acquire registry write lock"))?;

        registry.remove(&self.inner.dsn);

        // Close the engine immediately to release the file lock
        // This is idempotent - calling close_engine() multiple times is safe
        self.inner.engine.close_engine()?;

        Ok(())
    }

    /// Check if a table exists
    pub fn table_exists(&self, name: &str) -> Result<bool> {
        let engine = &self.inner.engine;
        let tx = engine.begin_transaction()?;
        Ok(tx.get_table(name).is_ok())
    }

    /// Get the DSN this database was opened with
    pub fn dsn(&self) -> &str {
        &self.inner.dsn
    }

    /// Set the default isolation level for new transactions
    pub fn set_default_isolation_level(&self, level: IsolationLevel) -> Result<()> {
        let mut executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::internal("Failed to acquire executor lock"))?;
        executor.set_default_isolation_level(level);
        Ok(())
    }

    /// Create a point-in-time snapshot of the database
    ///
    /// This creates snapshot files for each table that can be used to speed up
    /// database recovery. Instead of replaying the entire WAL, recovery can
    /// load the snapshot and only replay WAL entries after the snapshot.
    ///
    /// Note: This is a no-op for in-memory databases.
    pub fn create_snapshot(&self) -> Result<()> {
        use crate::storage::Engine;
        self.inner.engine.create_snapshot()
    }

    /// Get the internal executor (for Statement use)
    pub(crate) fn executor(&self) -> &Mutex<Executor> {
        &self.inner.executor
    }

    /// Get semantic cache statistics
    ///
    /// Returns statistics about the semantic query cache including hit rates,
    /// exact matches, and subsumption matches.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let db = Database::open("memory://")?;
    /// // ... execute some queries ...
    /// let stats = db.semantic_cache_stats()?;
    /// println!("Cache hits: {}", stats.hits);
    /// println!("Subsumption hits: {}", stats.subsumption_hits);
    /// ```
    pub fn semantic_cache_stats(&self) -> Result<crate::executor::SemanticCacheStatsSnapshot> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::internal("Failed to acquire executor lock"))?;
        Ok(executor.semantic_cache_stats())
    }

    /// Clear the semantic cache
    ///
    /// This clears all cached query results. Useful for testing or when
    /// you want to force queries to re-execute.
    pub fn clear_semantic_cache(&self) -> Result<()> {
        let executor = self
            .inner
            .executor
            .lock()
            .map_err(|_| Error::internal("Failed to acquire executor lock"))?;
        executor.clear_semantic_cache();
        Ok(())
    }
}

/// Trait for converting from Value to a Rust type
pub trait FromValue: Sized {
    /// Convert a Value to Self
    fn from_value(value: &Value) -> Result<Self>;
}

impl FromValue for i64 {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Integer(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            _ => Err(Error::TypeConversion {
                from: format!("{:?}", value),
                to: "Integer".to_string(),
            }),
        }
    }
}

impl FromValue for i32 {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Integer(i) => Ok(*i as i32),
            Value::Float(f) => Ok(*f as i32),
            _ => Err(Error::TypeConversion {
                from: format!("{:?}", value),
                to: "Integer".to_string(),
            }),
        }
    }
}

impl FromValue for f64 {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Float(f) => Ok(*f),
            Value::Integer(i) => Ok(*i as f64),
            _ => Err(Error::TypeConversion {
                from: format!("{:?}", value),
                to: "Float".to_string(),
            }),
        }
    }
}

impl FromValue for String {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Text(s) => Ok(s.to_string()),
            Value::Json(s) => Ok(s.to_string()),
            // Convert other types to string representation
            Value::Integer(i) => Ok(i.to_string()),
            Value::Float(f) => Ok(f.to_string()),
            Value::Boolean(b) => Ok(if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }),
            Value::Timestamp(ts) => Ok(ts.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
            Value::Null(_) => Ok(String::new()),
        }
    }
}

impl FromValue for bool {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Boolean(b) => Ok(*b),
            Value::Integer(i) => Ok(*i != 0),
            _ => Err(Error::TypeConversion {
                from: format!("{:?}", value),
                to: "Boolean".to_string(),
            }),
        }
    }
}

impl FromValue for Value {
    fn from_value(value: &Value) -> Result<Self> {
        Ok(value.clone())
    }
}

impl<T: FromValue> FromValue for Option<T> {
    fn from_value(value: &Value) -> Result<Self> {
        if value.is_null() {
            Ok(None)
        } else {
            Ok(Some(T::from_value(value)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_memory() {
        let db = Database::open("memory://").unwrap();
        assert_eq!(db.dsn(), "memory://");
    }

    #[test]
    fn test_open_in_memory() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO test VALUES ($1)", (1,)).unwrap();

        for row in db.query("SELECT * FROM test", ()).unwrap() {
            let row = row.unwrap();
            let id: i64 = row.get(0).unwrap();
            assert_eq!(id, 1);
        }
    }

    #[test]
    fn test_execute_and_query_new_api() {
        let db = Database::open_in_memory().unwrap();

        // Create table - no params
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
            (),
        )
        .unwrap();

        // Insert with tuple params
        let affected = db
            .execute(
                "INSERT INTO users VALUES ($1, $2, $3), ($4, $5, $6)",
                (1, "Alice", 30, 2, "Bob", 25),
            )
            .unwrap();
        assert_eq!(affected, 2);

        // Query with tuple params
        let rows: Vec<_> = db
            .query("SELECT * FROM users ORDER BY id", ())
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get::<i64>(0).unwrap(), 1);
        assert_eq!(rows[0].get::<String>(1).unwrap(), "Alice");
        assert_eq!(rows[0].get::<i64>(2).unwrap(), 30);
    }

    #[test]
    fn test_query_one() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO test VALUES ($1), ($2), ($3)", (1, 2, 3))
            .unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_query_opt() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO test VALUES ($1)", (1,)).unwrap();

        // Found
        let result: Option<i64> = db
            .query_opt("SELECT id FROM test WHERE id = $1", (1,))
            .unwrap();
        assert_eq!(result, Some(1));

        // Not found
        let result: Option<i64> = db
            .query_opt("SELECT id FROM test WHERE id = $1", (999,))
            .unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_params_macro() {
        let db = Database::open_in_memory().unwrap();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();

        // Use params! macro
        db.execute(
            "INSERT INTO users VALUES ($1, $2)",
            crate::params![1, "Alice"],
        )
        .unwrap();

        let names: Vec<String> = db
            .query("SELECT name FROM users WHERE id = $1", crate::params![1])
            .unwrap()
            .map(|r| r.and_then(|row| row.get(0)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(names, vec!["Alice"]);
    }

    #[test]
    fn test_parse_dsn() {
        // Memory
        let (scheme, path) = Database::parse_dsn("memory://").unwrap();
        assert_eq!(scheme, "memory");
        assert_eq!(path, "");

        // File
        let (scheme, path) = Database::parse_dsn("file:///tmp/test.db").unwrap();
        assert_eq!(scheme, "file");
        assert_eq!(path, "/tmp/test.db");

        // File with params
        let (scheme, path) = Database::parse_dsn("file:///tmp/test.db?sync=full").unwrap();
        assert_eq!(scheme, "file");
        assert_eq!(path, "/tmp/test.db?sync=full");

        // Invalid
        assert!(Database::parse_dsn("invalid").is_err());
        assert!(Database::parse_dsn("unknown://test").is_err());
    }

    #[test]
    fn test_from_value_types() {
        assert_eq!(i64::from_value(&Value::Integer(42)).unwrap(), 42);
        assert_eq!(f64::from_value(&Value::Float(3.5)).unwrap(), 3.5);
        assert_eq!(
            String::from_value(&Value::Text("hello".into())).unwrap(),
            "hello"
        );
        assert!(bool::from_value(&Value::Boolean(true)).unwrap());

        // Optional
        assert_eq!(
            Option::<i64>::from_value(&Value::Integer(42)).unwrap(),
            Some(42)
        );
        assert_eq!(
            Option::<i64>::from_value(&Value::null_unknown()).unwrap(),
            None
        );
    }
}
