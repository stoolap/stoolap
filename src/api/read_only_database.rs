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

//! Read-only database handle.
//!
//! Provides a read-only view over an existing database. Write SQL is
//! rejected at the parser-level gate.
//!
//! ## Current limitations (Step 2b-1)
//!
//! - The handle currently shares the *writable* `MVCCEngine` instance
//!   underneath. There is no shared file lock (LOCK_SH) yet — opening
//!   a `ReadOnlyDatabase` for a not-yet-open DSN takes the same exclusive
//!   file lock as a writable `Database::open`. A standalone
//!   `ReadOnlyEngine` with shared lock is planned (Step 2b-2).
//! - SQL execution still goes through the writable executor path which
//!   internally uses `engine.begin_transaction()` (writable). The
//!   read-only contract is enforced only by the parser-level write gate;
//!   the executor's runtime transaction type is still
//!   `Box<dyn WriteTransaction>`. Migrating the executor to actually
//!   use `Box<dyn ReadTransaction>` for read-only callers is also
//!   planned (Step 2b-2).
//! - The trait split (Step 1, 2a) ensures that *Rust callers* holding
//!   `Box<dyn ReadTable>` or `Box<dyn ReadTransaction>` cannot reach
//!   write methods. SQL strings are gated by the parser.

use std::sync::{Arc, Mutex};

use crate::core::{Error, Result};
use crate::executor::Executor;
use crate::storage::traits::Engine;

use super::database::EngineEntry;
use super::params::Params;
use super::rows::Rows;

/// Read-only handle over a database.
///
/// Constructed via [`crate::api::database::Database::open_read_only`] or
/// [`crate::api::database::Database::as_read_only`]. Rejects all write
/// SQL (INSERT/UPDATE/DELETE/DDL/maintenance PRAGMA/SET TRANSACTION) at
/// query time. Read SQL (SELECT, SHOW, EXPLAIN, BEGIN/COMMIT/ROLLBACK,
/// SAVEPOINT, benign SET no-ops) is allowed.
///
/// Holds an `Arc<EngineEntry>` so the underlying engine cannot be closed
/// while this `ReadOnlyDatabase` handle is alive. The engine stays open
/// as long as any user-visible handle (`Database`, `Database` clone,
/// sibling `Database::open(dsn)`, or `ReadOnlyDatabase`) references it.
///
/// # Transaction visibility
///
/// A `ReadOnlyDatabase` is a *view*, not a connection sharing a session
/// with the writable handle that constructed it. Each handle owns its own
/// executor and therefore its own transaction state:
///
/// - An uncommitted `BEGIN` on the source `Database` is **not** observed
///   by queries through this `ReadOnlyDatabase`. Writes inside the open
///   transaction are not seen until they commit.
/// - A `BEGIN` issued via SQL on this `ReadOnlyDatabase` opens a
///   read-only snapshot transaction local to this handle; it does not
///   interact with any transaction running on the source `Database` or on
///   other `ReadOnlyDatabase` views.
/// - Default isolation level is independent across handles.
///
/// To observe uncommitted writes from a specific transaction, do the read
/// SQL inside that same `Transaction` (read SQL is allowed on transactions
/// regardless of mode).
pub struct ReadOnlyDatabase {
    /// Keeps the engine alive (`EngineEntry::drop` closes the engine
    /// when the last Arc drops).
    entry: Arc<EngineEntry>,
    /// Independent executor with its own transaction state — a BEGIN
    /// on the read-only handle does not affect the writable Database.
    executor: Mutex<Executor>,
}

impl ReadOnlyDatabase {
    /// Construct a `ReadOnlyDatabase` from a shared `EngineEntry`.
    ///
    /// Crate-internal; `Database::open_read_only` and
    /// `Database::as_read_only` are the public entrypoints.
    pub(crate) fn from_entry(entry: Arc<EngineEntry>) -> Self {
        // Read-only executor: DML helper paths refuse to begin writable
        // auto-commit transactions even if the parser-level write gate
        // is bypassed. Shares the engine entry's semantic cache and
        // query planner so DML commits and ANALYZE on a sibling writable
        // handle invalidate this view's cached SELECT results and
        // planner stats.
        let engine = Arc::clone(&entry.engine);
        let semantic_cache = Arc::clone(&entry.semantic_cache);
        let query_planner = Arc::clone(&entry.query_planner);
        let executor =
            Executor::with_shared_semantic_cache_read_only(engine, semantic_cache, query_planner);
        Self {
            entry,
            executor: Mutex::new(executor),
        }
    }

    /// Returns the DSN this handle was opened with.
    pub fn dsn(&self) -> &str {
        &self.entry.dsn
    }

    /// Always returns `true`: a `ReadOnlyDatabase` is read-only by
    /// construction. Symmetric with [`crate::api::database::Database::is_read_only`]
    /// so generic code over both handle types can call the same accessor.
    #[inline]
    pub fn is_read_only(&self) -> bool {
        true
    }

    /// Get the engine as a read-only trait object.
    ///
    /// Returns `Arc<dyn ReadEngine>` so callers get compile-time
    /// enforcement: the trait object exposes only read transactions
    /// (no `Engine::begin_transaction`, no inherent write methods).
    /// Symmetric with [`crate::api::database::Database::read_engine`].
    /// Cheap (one Arc clone).
    pub fn read_engine(&self) -> Arc<dyn crate::storage::traits::ReadEngine> {
        Arc::clone(&self.entry.engine) as Arc<dyn crate::storage::traits::ReadEngine>
    }

    /// Returns `true` if a table with the given name exists.
    pub fn table_exists(&self, name: &str) -> Result<bool> {
        Engine::table_exists(&*self.entry.engine, name)
    }

    /// Execute a read-only SQL query.
    ///
    /// Rejects any statement that mutates persistent state (INSERT, UPDATE,
    /// DELETE, DDL, maintenance PRAGMA, SET TRANSACTION ISOLATION LEVEL).
    /// Read statements (SELECT, SHOW, EXPLAIN, BEGIN/COMMIT/ROLLBACK,
    /// SAVEPOINT, benign SET no-ops) are allowed.
    pub fn query<P: Params>(&self, sql: &str, params: P) -> Result<Rows> {
        // Write rejection happens inside the executor's parse/cache path
        // (Executor::read_only=true), so we don't pre-parse here. This
        // avoids paying for two full parses on every read-only query.
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;

        let param_values = params.into_params();
        let result = if param_values.is_empty() {
            executor.execute(sql)?
        } else if let Some(fast_result) = executor.try_fast_path_with_params(sql, &param_values) {
            fast_result?
        } else {
            executor.execute_with_params(sql, param_values)?
        };
        Ok(Rows::new(result))
    }

    /// Execute a read-only query with named parameters.
    ///
    /// Named parameters use the `:name` syntax in SQL queries.
    pub fn query_named(&self, sql: &str, params: crate::api::NamedParams) -> Result<Rows> {
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;

        let result = executor.execute_with_named_params(sql, params.into_inner())?;
        Ok(Rows::new(result))
    }

    /// Cache a parsed plan for a read-only SQL statement.
    ///
    /// Same shape as [`crate::api::Database::cached_plan`]: parse once,
    /// reuse the [`CachedPlanRef`] across many `query_plan` /
    /// `query_named_plan` calls without re-parsing.
    ///
    /// Rejects write SQL at plan-creation time with `ReadOnlyViolation`.
    /// This is the prepared-statement equivalent on a `ReadOnlyDatabase`
    /// (the `prepare()` path on `Database` requires a `Weak<DatabaseInner>`
    /// that the read-only handle does not have; cached plans give the
    /// same parse-once / execute-many ergonomics without that coupling).
    pub fn cached_plan(&self, sql: &str) -> Result<crate::executor::query_cache::CachedPlanRef> {
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;
        executor.get_or_create_plan(sql)
    }

    /// Query using a pre-cached plan with positional parameters
    /// (no parsing, no cache lookup). Read-only equivalent of
    /// [`crate::api::Database::query_plan`].
    pub fn query_plan<P: Params>(
        &self,
        plan: &crate::executor::query_cache::CachedPlanRef,
        params: P,
    ) -> Result<Rows> {
        use crate::executor::context::ExecutionContext;
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;
        let param_values = params.into_params();
        let ctx = if param_values.is_empty() {
            ExecutionContext::new()
        } else {
            ExecutionContext::with_params(param_values)
        };
        let result = executor.execute_with_cached_plan(plan, &ctx)?;
        Ok(Rows::new(result))
    }

    /// Query using a pre-cached plan with named parameters.
    /// Read-only equivalent of [`crate::api::Database::query_named_plan`].
    pub fn query_named_plan(
        &self,
        plan: &crate::executor::query_cache::CachedPlanRef,
        params: crate::api::NamedParams,
    ) -> Result<Rows> {
        use crate::executor::context::ExecutionContext;
        let executor = self
            .executor
            .lock()
            .map_err(|_| Error::LockAcquisitionFailed("read-only executor".to_string()))?;
        let ctx = ExecutionContext::with_named_params(params.into_inner());
        let result = executor.execute_with_cached_plan(plan, &ctx)?;
        Ok(Rows::new(result))
    }
}

// `ReadOnlyDatabase` has no `Drop` impl: the registry stores `Weak<EngineEntry>`,
// so when the last Arc<EngineEntry> drops the entry's `Drop` closes the
// engine and the registry's Weak silently expires. There is no longer a
// case where a registry-cleanup hook needs to run from this handle.
