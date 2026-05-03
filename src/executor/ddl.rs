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

//! DDL Statement Execution
//!
//! This module implements execution of Data Definition Language (DDL) statements:
//! - CREATE TABLE
//! - DROP TABLE
//! - CREATE INDEX
//! - DROP INDEX
//! - ALTER TABLE
//! - CREATE VIEW
//! - DROP VIEW

use crate::core::{
    DataType, Error, ForeignKeyAction, ForeignKeyConstraint, Result, Row, SchemaBuilder, Value,
};
use crate::parser::ast::*;
use crate::storage::expression::Expression;
use crate::storage::traits::{Engine, QueryResult};

/// Validate a foreign key reference and build a `ForeignKeyConstraint`.
///
/// Checks: parent table exists, referenced column exists and is PK/UNIQUE,
/// FK column exists in the schema being built.
#[allow(clippy::too_many_arguments)]
fn validate_fk_reference(
    engine: &dyn Engine,
    schema_builder: &SchemaBuilder,
    fk_col_name: &str,
    fk_col_display: &str,
    ref_table_lower: &str,
    ref_table_display: &str,
    ref_col_opt: Option<&str>,
    on_delete: ForeignKeyAction,
    on_update: ForeignKeyAction,
) -> Result<ForeignKeyConstraint> {
    // Validate parent table exists
    if !engine.table_exists(ref_table_lower)? {
        return Err(Error::internal(format!(
            "foreign key on column '{}' references non-existent table '{}'",
            fk_col_display, ref_table_display
        )));
    }

    let parent_schema = engine.get_table_schema(ref_table_lower)?;

    // Resolve referenced column (defaults to PK if not specified)
    let ref_col_name = if let Some(rc) = ref_col_opt {
        rc.to_string()
    } else {
        parent_schema
            .pk_column_index()
            .and_then(|idx| parent_schema.columns.get(idx))
            .map(|c| c.name.to_lowercase())
            .ok_or_else(|| {
                Error::internal(format!(
                    "table '{}' has no primary key for FK reference default",
                    ref_table_display
                ))
            })?
    };

    // Validate referenced column exists and is PK or has unique index
    let (ref_col_idx, ref_col_def) = parent_schema.find_column(&ref_col_name).ok_or_else(|| {
        Error::internal(format!(
            "foreign key references non-existent column '{}' in table '{}'",
            ref_col_name, ref_table_display
        ))
    })?;

    if !ref_col_def.primary_key {
        let has_unique = engine
            .get_all_indexes(ref_table_lower)
            .map(|indexes| {
                indexes.iter().any(|idx| {
                    idx.is_unique()
                        && idx.column_ids().len() == 1
                        && idx.column_ids()[0] as usize == ref_col_idx
                })
            })
            .unwrap_or(false);

        if !has_unique {
            return Err(Error::internal(format!(
                "foreign key on '{}' references column '{}' in '{}' which is neither PRIMARY KEY nor UNIQUE",
                fk_col_display, ref_col_name, ref_table_display
            )));
        }
    }

    // Find FK column index in the schema being built
    let fk_col_idx = schema_builder.column_index(fk_col_name).ok_or_else(|| {
        Error::internal(format!(
            "foreign key column '{}' not found in table definition",
            fk_col_display
        ))
    })?;

    // Reject SET NULL action on NOT NULL columns (would always fail at runtime)
    if (matches!(on_delete, ForeignKeyAction::SetNull)
        || matches!(on_update, ForeignKeyAction::SetNull))
        && !schema_builder.is_column_nullable(fk_col_idx)
    {
        return Err(Error::internal(format!(
            "foreign key column '{}' has ON {} SET NULL but is NOT NULL",
            fk_col_display,
            if matches!(on_delete, ForeignKeyAction::SetNull) {
                "DELETE"
            } else {
                "UPDATE"
            }
        )));
    }

    Ok(ForeignKeyConstraint {
        column_index: fk_col_idx,
        column_name: fk_col_name.to_string(),
        referenced_table: ref_table_lower.to_string(),
        referenced_column: ref_col_name,
        on_delete,
        on_update,
    })
}

use super::context::{
    invalidate_in_subquery_cache_for_table, invalidate_scalar_subquery_cache_for_table,
    invalidate_semi_join_cache_for_table, ExecutionContext,
};
use super::expression::ExpressionEval;
use super::result::ExecResult;
use super::Executor;

impl Executor {
    /// Execute a CREATE TABLE statement
    pub(crate) fn execute_create_table(
        &self,
        stmt: &CreateTableStatement,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value;

        // Check if table already exists
        if self.engine.table_exists(table_name)? {
            if stmt.if_not_exists {
                return Ok(Box::new(ExecResult::empty()));
            }
            return Err(Error::TableAlreadyExists(table_name.to_string()));
        }

        // Check if a view with the same name exists
        if self.engine.view_exists(table_name)? {
            return Err(Error::internal(format!(
                "cannot create table '{}': a view with the same name exists",
                table_name
            )));
        }

        // Handle CREATE TABLE ... AS SELECT ...
        if let Some(ref select_stmt) = stmt.as_select {
            return self.execute_create_table_as_select(
                table_name,
                select_stmt,
                stmt.if_not_exists,
                ctx,
            );
        }

        // Build schema from column definitions
        let mut schema_builder = SchemaBuilder::new(table_name.as_str());

        // Collect columns with UNIQUE constraints to create indexes after table creation
        let mut unique_columns: Vec<String> = Vec::new();

        for col_def in &stmt.columns {
            let col_name = &col_def.name.value;
            let data_type = self.parse_data_type(&col_def.data_type)?;
            let nullable = !col_def
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::NotNull));
            let is_primary_key = col_def
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::PrimaryKey));

            // Validate PRIMARY KEY type - only INTEGER is supported
            if is_primary_key && data_type != DataType::Integer {
                return Err(Error::Parse(format!(
                    "PRIMARY KEY column '{}' must be INTEGER type, got {:?}. Only INTEGER PRIMARY KEY is supported.",
                    col_name, data_type
                )));
            }

            let is_unique = col_def
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::Unique));

            let is_auto_increment = col_def
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::AutoIncrement));

            // Extract DEFAULT expression
            let default_expr = col_def.constraints.iter().find_map(|c| {
                if let ColumnConstraint::Default(expr) = c {
                    Some(format!("{}", expr))
                } else {
                    None
                }
            });

            // Extract CHECK expression
            let check_expr = col_def.constraints.iter().find_map(|c| {
                if let ColumnConstraint::Check(expr) = c {
                    Some(format!("{}", expr))
                } else {
                    None
                }
            });

            // Use add_with_constraints to include DEFAULT and CHECK
            schema_builder = schema_builder.add_with_constraints(
                col_name.as_str(),
                data_type,
                nullable && !is_primary_key,
                is_primary_key,
                is_auto_increment,
                default_expr,
                check_expr,
            );

            // Store vector dimension in SchemaColumn if this is a VECTOR type
            if data_type == DataType::Vector {
                let dim = crate::executor::utils::parse_vector_dimension(&col_def.data_type);
                if dim > 0 {
                    schema_builder = schema_builder.set_last_vector_dimensions(dim);
                }
            }

            // Track UNIQUE columns for index creation
            if is_unique && !is_primary_key {
                unique_columns.push(col_name.to_string());
            }
        }

        // Collect foreign key constraints from column-level REFERENCES
        for col_def in &stmt.columns {
            for constraint in &col_def.constraints {
                if let ColumnConstraint::References {
                    table: ref ref_table,
                    column: ref ref_col,
                    on_delete,
                    on_update,
                } = constraint
                {
                    let fk = validate_fk_reference(
                        &*self.engine,
                        &schema_builder,
                        col_def.name.value_lower.as_str(),
                        &col_def.name.value,
                        &ref_table.value_lower,
                        &ref_table.value,
                        ref_col.as_ref().map(|rc| rc.value_lower.as_str()),
                        *on_delete,
                        *on_update,
                    )?;
                    schema_builder = schema_builder.add_foreign_key(fk);
                }
            }
        }

        // Collect table-level UNIQUE and FOREIGN KEY constraints
        let mut table_unique_constraints: Vec<Vec<String>> = Vec::new();
        for constraint in &stmt.table_constraints {
            match constraint {
                TableConstraint::Unique(cols) => {
                    let col_names: Vec<String> = cols.iter().map(|c| c.value.to_string()).collect();
                    table_unique_constraints.push(col_names);
                }
                TableConstraint::ForeignKey(fk) => {
                    let fk_constraint = validate_fk_reference(
                        &*self.engine,
                        &schema_builder,
                        fk.column.value_lower.as_str(),
                        &fk.column.value,
                        &fk.ref_table.value_lower,
                        &fk.ref_table.value,
                        fk.ref_column.as_ref().map(|rc| rc.value_lower.as_str()),
                        fk.on_delete,
                        fk.on_update,
                    )?;
                    schema_builder = schema_builder.add_foreign_key(fk_constraint);
                }
                _ => {}
            }
        }

        let schema = schema_builder.build();

        // Collect FK columns that need auto-created indexes (skip PK and UNIQUE columns)
        let mut fk_index_columns: Vec<String> = Vec::new();
        for fk in &schema.foreign_keys {
            let col = &schema.columns[fk.column_index];
            // Skip if the column is already a PK (has PkIndex) or UNIQUE (gets a unique index above)
            if col.primary_key {
                continue;
            }
            let col_lower = col.name.to_lowercase();
            if unique_columns.iter().any(|u| u.to_lowercase() == col_lower) {
                continue;
            }
            fk_index_columns.push(col.name.clone());
        }

        // Check if there's an active transaction
        let mut active_tx = self.active_transaction.lock().unwrap();

        if let Some(ref mut tx_state) = *active_tx {
            // Use the active transaction for DDL (allows rollback)
            let table = tx_state.transaction.create_table(table_name, schema)?;

            // Create unique indexes and FK indexes. The active-tx
            // CREATE TABLE statement must be atomic from the SQL
            // caller's view: a partial-success outcome (table
            // logged in `ddl_log`, some auto-indexes missing) would
            // leave the open transaction holding a half-formed
            // table that a later COMMIT could promote to durable
            // state. Wrap the auto-index work in a labeled block,
            // capture the first failure, and on any failure undo
            // the just-created table inside the same txn so
            // `ddl_log` records `Create + Drop` (which COMMIT
            // collapses to "no table" and ROLLBACK undoes via the
            // existing reverse-walk).
            let mut autoindex_err: Option<Error> = None;
            'autoindex: {
                for col_name in &unique_columns {
                    let index_name = format!("unique_{}_{}", table_name, col_name);
                    if let Err(e) = table.create_index(&index_name, &[col_name.as_str()], true) {
                        autoindex_err = Some(e);
                        break 'autoindex;
                    }
                    let idx_type = table
                        .get_index(&index_name)
                        .map(|idx| idx.index_type())
                        .unwrap_or(crate::core::IndexType::BTree);
                    if let Err(e) = tx_state.transaction.stage_deferred_create_index(
                        table_name,
                        &index_name,
                        std::slice::from_ref(col_name),
                        true,
                        idx_type,
                        None,
                        None,
                        None,
                        None,
                    ) {
                        let _ = table.drop_index(&index_name);
                        autoindex_err = Some(e);
                        break 'autoindex;
                    }
                }

                // Create multi-column unique indexes from table-level constraints
                for (i, col_names) in table_unique_constraints.iter().enumerate() {
                    let index_name = format!("unique_{}_{}", table_name, i);
                    let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
                    if let Err(e) = table.create_index(&index_name, &col_refs, true) {
                        autoindex_err = Some(e);
                        break 'autoindex;
                    }
                    let idx_type = table
                        .get_index(&index_name)
                        .map(|idx| idx.index_type())
                        .unwrap_or(crate::core::IndexType::BTree);
                    if let Err(e) = tx_state.transaction.stage_deferred_create_index(
                        table_name,
                        &index_name,
                        col_names,
                        true,
                        idx_type,
                        None,
                        None,
                        None,
                        None,
                    ) {
                        let _ = table.drop_index(&index_name);
                        autoindex_err = Some(e);
                        break 'autoindex;
                    }
                }

                // Auto-create indexes on FK columns for efficient referential integrity checks
                for col_name in &fk_index_columns {
                    let index_name = format!("fk_{}_{}", table_name, col_name);
                    if let Err(e) = table.create_index(&index_name, &[col_name.as_str()], false) {
                        autoindex_err = Some(e);
                        break 'autoindex;
                    }
                    let idx_type = table
                        .get_index(&index_name)
                        .map(|idx| idx.index_type())
                        .unwrap_or(crate::core::IndexType::BTree);
                    if let Err(e) = tx_state.transaction.stage_deferred_create_index(
                        table_name,
                        &index_name,
                        std::slice::from_ref(col_name),
                        false,
                        idx_type,
                        None,
                        None,
                        None,
                        None,
                    ) {
                        let _ = table.drop_index(&index_name);
                        autoindex_err = Some(e);
                        break 'autoindex;
                    }
                }
            }
            if let Some(e) = autoindex_err {
                // Drop the index handle FIRST so the in-process
                // `Arc<dyn WriteTable>` doesn't keep the
                // version-store reachable during the undo
                // drop_table (which closes the store after WAL
                // succeeds).
                drop(table);
                // Undo the just-created table inside the same
                // txn. `ddl_log` becomes [Create, Drop]; reverse-
                // walk on rollback or natural reorder on commit
                // both end with "no table". A failure here
                // (latch flipped between, WAL I/O failure in the
                // DROP) latches the engine and forces the
                // entire transaction to be unusable so the
                // failed CREATE TABLE statement can't be
                // promoted to durable state by a subsequent
                // COMMIT.
                if let Err(undo_e) = tx_state.transaction.drop_table(table_name) {
                    eprintln!(
                        "Error: Failed to undo CREATE TABLE '{}' after auto-index \
                         failure: {} — rolling back the entire transaction so the \
                         failed CREATE cannot be committed.",
                        table_name, undo_e
                    );
                    let _ = tx_state.transaction.rollback();
                    *active_tx = None;
                }
                return Err(e);
            }
        } else {
            // No active transaction - use direct engine call (auto-committed)
            self.engine.create_table(schema)?;

            // Create unique indexes and FK indexes. CREATE TABLE
            // is documented as atomic from the SQL caller's view —
            // any failure between the table CREATE and the last
            // generated index must roll the table BACK so an Err
            // doesn't leave the table half-indexed (some
            // generated indexes durable, the failing UNIQUE/FK
            // index missing). Wrap the index work in a labeled
            // block so the table-undo runs once on any failure
            // path.
            let needs_indexes = !unique_columns.is_empty()
                || !table_unique_constraints.is_empty()
                || !fk_index_columns.is_empty();
            if needs_indexes {
                // Hold SH on the engine's
                // `transactional_ddl_fence` across every
                // generated-index `table.create_index` →
                // `record_create_index` window. The fence
                // is scoped to JUST the loops; the
                // table-undo path below calls
                // `drop_table_internal` which itself
                // acquires SH on the same fence, so we MUST
                // release here before the undo (parking_lot
                // RwLock can deadlock on reentrant SH if a
                // checkpoint writer is waiting for EX). The
                // undo is then fenced by `drop_table_internal`'s
                // own SH guard.
                //
                // Without this guard, `engine.create_table`
                // released its own fence above before
                // returning, so a checkpoint could grab EX
                // between any `table.create_index(...)`
                // (in-memory) and its companion
                // `record_create_index(...)` (WAL). A
                // re-record landing in that gap becomes
                // durable; if `record_create_index` then
                // fails and the rollback drops the index,
                // recovery would still rebuild that index
                // (and its parent table) from the
                // checkpoint's re-records — surfacing schema
                // state belonging to a CREATE TABLE
                // statement that ultimately failed.
                let mut autoindex_err: Option<Error> = None;
                {
                    let _ddl_fence_guard = self.engine.ddl_fence().read();
                    let tx = self.engine.begin_writable_transaction_internal()?;
                    let table = tx.get_table(table_name)?;

                    'autoindex: {
                        for col_name in &unique_columns {
                            let index_name = format!("unique_{}_{}", table_name, col_name);
                            if let Err(e) =
                                table.create_index(&index_name, &[col_name.as_str()], true)
                            {
                                autoindex_err = Some(e);
                                break 'autoindex;
                            }
                            let idx_type = table
                                .get_index(&index_name)
                                .map(|idx| idx.index_type())
                                .unwrap_or(crate::core::IndexType::BTree);
                            if let Err(e) = self.engine.record_create_index(
                                table_name,
                                &index_name,
                                std::slice::from_ref(col_name),
                                true,
                                idx_type,
                                None,
                                None,
                                None,
                                None,
                            ) {
                                let _ = table.drop_index(&index_name);
                                autoindex_err = Some(e);
                                break 'autoindex;
                            }
                        }

                        // Create multi-column unique indexes from table-level constraints
                        for (i, col_names) in table_unique_constraints.iter().enumerate() {
                            let index_name = format!("unique_{}_{}", table_name, i);
                            let col_refs: Vec<&str> =
                                col_names.iter().map(|s| s.as_str()).collect();
                            if let Err(e) = table.create_index(&index_name, &col_refs, true) {
                                autoindex_err = Some(e);
                                break 'autoindex;
                            }
                            let idx_type = table
                                .get_index(&index_name)
                                .map(|idx| idx.index_type())
                                .unwrap_or(crate::core::IndexType::BTree);
                            if let Err(e) = self.engine.record_create_index(
                                table_name,
                                &index_name,
                                col_names,
                                true,
                                idx_type,
                                None,
                                None,
                                None,
                                None,
                            ) {
                                let _ = table.drop_index(&index_name);
                                autoindex_err = Some(e);
                                break 'autoindex;
                            }
                        }

                        // Auto-create indexes on FK columns for efficient referential integrity checks
                        for col_name in &fk_index_columns {
                            let index_name = format!("fk_{}_{}", table_name, col_name);
                            if let Err(e) =
                                table.create_index(&index_name, &[col_name.as_str()], false)
                            {
                                autoindex_err = Some(e);
                                break 'autoindex;
                            }
                            let idx_type = table
                                .get_index(&index_name)
                                .map(|idx| idx.index_type())
                                .unwrap_or(crate::core::IndexType::BTree);
                            if let Err(e) = self.engine.record_create_index(
                                table_name,
                                &index_name,
                                std::slice::from_ref(col_name),
                                false,
                                idx_type,
                                None,
                                None,
                                None,
                                None,
                            ) {
                                let _ = table.drop_index(&index_name);
                                autoindex_err = Some(e);
                                break 'autoindex;
                            }
                        }
                    }
                    // Drop the SH guard + tx + table here
                    // so the rollback path below (which
                    // calls `drop_table_internal`, which
                    // takes its OWN SH guard) doesn't
                    // attempt reentrant SH on the same
                    // fence — parking_lot can deadlock if a
                    // checkpoint EX is waiting.
                }
                if let Some(e) = autoindex_err {
                    // Undo the durable CREATE TABLE so the
                    // failed statement leaves no half-formed
                    // table behind. `drop_table_internal`
                    // also writes a durable DropTable WAL +
                    // tears down the version store, so any
                    // earlier successful CreateIndex WAL
                    // entries become orphans cleanable on the
                    // next checkpoint. Recovery sees CREATE
                    // TABLE + index entries + DROP TABLE
                    // and converges to "no table".
                    //
                    // The undo can fail under the same
                    // conditions that produced `autoindex_err`
                    // (latch tripped via
                    // `ensure_writable`, fresh WAL I/O
                    // failure). Discarding that failure would
                    // leave the table durable + partially
                    // indexed and let this process keep
                    // accepting writes against an
                    // inconsistent durable state. Latch the
                    // engine instead so subsequent durability
                    // paths refuse and the user is forced to
                    // restart; WAL recovery converges to
                    // either CREATE+DROP (table gone) or just
                    // CREATE (table present but
                    // partially indexed — visible at the
                    // schema/index level) depending on how
                    // far the original failure got. Surface
                    // both errors to the caller.
                    if let Err(undo_e) = self.engine.drop_table_internal(table_name) {
                        self.engine.enter_catastrophic_failure();
                        return Err(Error::internal(format!(
                            "CREATE TABLE auto-index failed ({}); the compensating \
                             DROP TABLE for '{}' also failed ({}). The engine has \
                             been latched; restart the process so WAL recovery can \
                             converge.",
                            e, table_name, undo_e
                        )));
                    }
                    return Err(e);
                }
            }
        }

        Ok(Box::new(ExecResult::empty()))
    }

    /// Execute CREATE TABLE ... AS SELECT ...
    fn execute_create_table_as_select(
        &self,
        table_name: &str,
        select_stmt: &SelectStatement,
        _if_not_exists: bool,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        use crate::core::Row;

        // Execute the SELECT query to get the result
        // Use execute_select for full query processing (DISTINCT, ORDER BY, etc.)
        let mut result = self.execute_select(select_stmt, ctx)?;
        let columns: Vec<String> = result.columns().to_vec();

        // Materialize the result to get the rows
        let mut rows: Vec<Row> = Vec::new();
        while result.next() {
            rows.push(result.take_row());
        }
        if let Some(err) = result.last_error() {
            return Err(err);
        }

        // Infer schema from the result columns and first row (if available)
        let mut schema_builder = SchemaBuilder::new(table_name);

        for (i, col_name) in columns.iter().enumerate() {
            // Extract base column name (without table prefix)
            let base_name = if let Some(pos) = col_name.rfind('.') {
                &col_name[pos + 1..]
            } else {
                col_name.as_str()
            };

            // Infer data type from first row if available
            let data_type = if let Some(first_row) = rows.first() {
                if let Some(value) = first_row.get(i) {
                    Self::infer_data_type(value)
                } else {
                    DataType::Text // Default to TEXT
                }
            } else {
                DataType::Text // Default to TEXT for empty result
            };

            schema_builder = schema_builder.add_nullable(base_name, data_type);
        }

        let schema = schema_builder.build();

        // Create the table
        self.engine.create_table(schema)?;

        // Insert the rows into the new table
        let rows_count = rows.len();
        if !rows.is_empty() {
            let mut tx = self.engine.begin_writable_transaction_internal()?;
            let mut table = tx.get_table(table_name)?;

            for row in rows {
                let _ = table.insert(row)?;
            }

            // Commit the transaction - it will commit all tables via commit_all_tables()
            tx.commit()?;
        }

        Ok(Box::new(ExecResult::with_rows_affected(rows_count as i64)))
    }

    /// Infer data type from a Value
    fn infer_data_type(value: &crate::core::Value) -> DataType {
        match value {
            crate::core::Value::Integer(_) => DataType::Integer,
            crate::core::Value::Float(_) => DataType::Float,
            crate::core::Value::Text(_) => DataType::Text,
            crate::core::Value::Boolean(_) => DataType::Boolean,
            crate::core::Value::Timestamp(_) => DataType::Timestamp,
            crate::core::Value::Extension(data) => data
                .first()
                .and_then(|&b| DataType::from_u8(b))
                .unwrap_or(DataType::Text),
            crate::core::Value::Null(_) => DataType::Text, // Default nulls to TEXT
        }
    }

    /// Execute a DROP TABLE statement
    pub(crate) fn execute_drop_table(
        &self,
        stmt: &DropTableStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value;

        // Check if table exists
        if !self.engine.table_exists(table_name)? {
            if stmt.if_exists {
                return Ok(Box::new(ExecResult::empty()));
            }
            return Err(Error::TableNotFound(table_name.to_string()));
        }

        // Check if there's an active transaction (peek at txn_id for FK visibility)
        let mut active_tx = self.active_transaction.lock().unwrap();
        let txn_id = active_tx.as_ref().map(|s| s.transaction.id());

        // Check FK constraints: block DROP if child tables reference this table
        // Uses the caller's transaction (if any) so uncommitted child deletes are visible
        super::foreign_key::check_no_referencing_rows(&self.engine, table_name, txn_id)?;

        if let Some(ref mut tx_state) = *active_tx {
            // WARNING: DROP TABLE within a transaction has limited rollback support.
            // On rollback, the table schema will be recreated but DATA WILL BE LOST.
            // This is because table data is immediately deleted and cannot be recovered.
            // For recoverable data deletion, use DELETE FROM or TRUNCATE instead.
            eprintln!(
                "Warning: DROP TABLE '{}' within transaction - data cannot be recovered on rollback",
                table_name
            );
            tx_state.transaction.drop_table(table_name)?;
        } else {
            // No active transaction - use engine method directly (auto-committed with WAL)
            self.engine.drop_table_internal(table_name)?;
        }

        // Invalidate query cache for this table (schema no longer exists)
        self.query_cache.invalidate_table(table_name);
        invalidate_semi_join_cache_for_table(table_name);
        invalidate_scalar_subquery_cache_for_table(table_name);
        invalidate_in_subquery_cache_for_table(table_name);

        Ok(Box::new(ExecResult::empty()))
    }

    /// Execute a CREATE INDEX statement
    pub(crate) fn execute_create_index(
        &self,
        stmt: &CreateIndexStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Hold SH on the engine's `transactional_ddl_fence`
        // across the index mutation-to-WAL window.
        // Checkpoint's `rerecord_ddl_to_wal` takes EX, so
        // this fence blocks any concurrent checkpoint from
        // snapshotting the in-memory index BEFORE
        // `record_create_index` writes the durable
        // CreateIndex record. Without the guard, a
        // checkpoint that landed mid-window would re-record
        // the transient index; if `record_create_index`
        // later fails and the rollback drops the index from
        // the table, recovery would still rebuild it from
        // the checkpoint's CreateIndex re-record.
        let _ddl_fence_guard = self.engine.ddl_fence().read();

        let table_name = &stmt.table_name.value;
        let index_name = &stmt.index_name.value;

        // Check if table exists
        if !self.engine.table_exists(table_name)? {
            return Err(Error::TableNotFound(table_name.to_string()));
        }

        // Check if index already exists
        if self.engine.index_exists(index_name, table_name)? {
            if stmt.if_not_exists {
                return Ok(Box::new(ExecResult::empty()));
            }
            return Err(Error::internal(format!(
                "index already exists: {}",
                index_name
            )));
        }

        // Determine index type
        let is_unique = stmt.is_unique;

        // Get table to validate columns exist
        let tx = self.engine.begin_writable_transaction_internal()?;
        let table = tx.get_table(table_name)?;
        let schema = table.schema();

        // Validate columns
        for col_id in &stmt.columns {
            let col_name = &col_id.value;
            if !schema
                .column_index_map()
                .contains_key(col_id.value_lower.as_str())
            {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }
        }

        // Collect column names
        let column_names: Vec<String> = stmt.columns.iter().map(|c| c.value.to_string()).collect();
        let column_refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();

        // Check if IF NOT EXISTS should suppress errors:
        // 1. Index with same name already exists, OR
        // 2. An index already exists on the column(s) - this prevents errors like
        //    "cannot create non-unique index when unique already exists"
        if stmt.if_not_exists {
            // Check by name
            if table.get_index(index_name).is_some() {
                return Ok(Box::new(ExecResult::empty()));
            }
            // For single-column indexes, also check if column already has an index
            if column_names.len() == 1 && table.has_index_on_column(&column_names[0]) {
                return Ok(Box::new(ExecResult::empty()));
            }
        }

        // Convert USING clause IndexMethod to core IndexType
        let requested_index_type = stmt.index_method.map(|method| match method {
            crate::parser::ast::IndexMethod::BTree => crate::core::IndexType::BTree,
            crate::parser::ast::IndexMethod::Hash => crate::core::IndexType::Hash,
            crate::parser::ast::IndexMethod::Bitmap => crate::core::IndexType::Bitmap,
            crate::parser::ast::IndexMethod::Hnsw => crate::core::IndexType::Hnsw,
        });

        // HNSW only supports single-column indexes — reject multi-column early
        if requested_index_type == Some(crate::core::IndexType::Hnsw) && column_names.len() > 1 {
            return Err(Error::invalid_argument(
                "HNSW index must be on a single vector column; multi-column HNSW indexes are not supported",
            ));
        }

        // Extract HNSW-specific options from WITH clause
        let mut hnsw_m: Option<u16> = None;
        let mut hnsw_ef_construction: Option<u16> = None;
        let mut hnsw_ef_search: Option<u16> = None;
        let mut hnsw_distance_metric: Option<u8> = None;
        let is_hnsw = requested_index_type == Some(crate::core::IndexType::Hnsw);
        for (key, value) in &stmt.options {
            match key.as_str() {
                "m" => {
                    let v = value.parse::<u16>().map_err(|_| {
                        Error::invalid_argument(format!(
                            "invalid value for HNSW option 'm': '{}' (expected integer >= 2)",
                            value
                        ))
                    })?;
                    if v < 2 {
                        return Err(Error::invalid_argument(format!(
                            "HNSW option 'm' must be >= 2, got {}",
                            v
                        )));
                    }
                    hnsw_m = Some(v);
                }
                "ef_construction" => {
                    hnsw_ef_construction = Some(value.parse::<u16>().map_err(|_| {
                        Error::invalid_argument(format!(
                            "invalid value for HNSW option 'ef_construction': '{}' (expected positive integer)",
                            value
                        ))
                    })?);
                }
                "ef_search" => {
                    hnsw_ef_search = Some(value.parse::<u16>().map_err(|_| {
                        Error::invalid_argument(format!(
                            "invalid value for HNSW option 'ef_search': '{}' (expected positive integer)",
                            value
                        ))
                    })?);
                }
                "metric" | "distance" => {
                    let metric =
                        crate::storage::index::HnswDistanceMetric::from_name(&value.to_lowercase())
                            .ok_or_else(|| {
                                Error::invalid_argument(format!(
                            "unknown HNSW distance metric '{}' (expected: l2, cosine, or ip)",
                            value
                        ))
                            })?;
                    hnsw_distance_metric = Some(metric.as_u8());
                }
                other if is_hnsw => {
                    return Err(Error::invalid_argument(format!(
                        "unknown HNSW index option '{}' (valid options: m, ef_construction, ef_search, metric)",
                        other
                    )));
                }
                _ => {}
            }
        }

        // Create the index (supports both single and multi-column)
        let has_hnsw_opts = hnsw_m.is_some()
            || hnsw_ef_construction.is_some()
            || hnsw_ef_search.is_some()
            || hnsw_distance_metric.is_some();
        if has_hnsw_opts && requested_index_type == Some(crate::core::IndexType::Hnsw) {
            // Auto-tune default M based on vector dimensions when not explicitly set
            if hnsw_m.is_none() {
                let dims = schema
                    .find_column(&column_names[0])
                    .map(|(_, col)| col.vector_dimensions as usize)
                    .unwrap_or(0);
                hnsw_m = Some(crate::storage::index::default_m_for_dims(dims) as u16);
            }
            // HNSW with custom params — use dedicated method
            table.create_hnsw_index(
                index_name,
                column_refs[0],
                is_unique,
                hnsw_m.unwrap() as usize,
                hnsw_ef_construction.unwrap_or(crate::storage::index::default_ef_construction(
                    hnsw_m.unwrap() as usize,
                ) as u16) as usize,
                hnsw_ef_search.unwrap_or(crate::storage::index::default_ef_search(
                    hnsw_m.unwrap() as usize
                ) as u16) as usize,
                crate::storage::index::HnswDistanceMetric::from_u8(
                    hnsw_distance_metric.unwrap_or(0),
                )
                .unwrap_or(crate::storage::index::HnswDistanceMetric::L2),
            )?;
        } else {
            // Standard index creation with optional type hint
            table.create_index_with_type(
                index_name,
                &column_refs,
                is_unique,
                requested_index_type,
            )?;
        }

        // Get the created index to determine its actual type for WAL persistence
        let index_type = table
            .get_index(index_name)
            .map(|idx| idx.index_type())
            .unwrap_or(crate::core::IndexType::BTree);

        // Always read back the effective HNSW params from the created index so
        // the WAL persists the exact values used at runtime. This covers both the
        // no-WITH path (all params auto-tuned from dims) and partial-WITH paths
        // (e.g., WITH(metric='cosine') where ef_* are still auto-tuned).
        if index_type == crate::core::IndexType::Hnsw {
            if let Some(idx) = table.get_index(index_name) {
                if let Some(hnsw) = idx
                    .as_any()
                    .downcast_ref::<crate::storage::index::HnswIndex>()
                {
                    let (m, efc, efs, met) = hnsw.params();
                    hnsw_m = Some(m as u16);
                    hnsw_ef_construction = Some(efc as u16);
                    hnsw_ef_search = Some(efs as u16);
                    hnsw_distance_metric = Some(met as u8);
                }
            }
        }

        // Record index creation to WAL for persistence.
        // If WAL write fails, rollback the in-memory index to prevent ghost state
        // where the index exists in memory but won't survive restart.
        if let Err(e) = self.engine.record_create_index(
            table_name,
            index_name,
            &column_names,
            is_unique,
            index_type,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
            hnsw_distance_metric,
        ) {
            // Rollback: remove the index we just created
            let _ = table.drop_index(index_name);
            return Err(e);
        }

        Ok(Box::new(ExecResult::empty()))
    }

    /// Execute a DROP INDEX statement
    pub(crate) fn execute_drop_index(
        &self,
        stmt: &DropIndexStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Hold SH on the engine's `transactional_ddl_fence`
        // across the WAL-to-mutation window — same rationale
        // as `execute_create_index`. Without the guard a
        // checkpoint that landed between `record_drop_index`
        // and `table.drop_index` could re-record the
        // not-yet-removed index, leaving a later CreateIndex
        // entry past the durable DropIndex.
        let _ddl_fence_guard = self.engine.ddl_fence().read();

        let index_name = &stmt.index_name.value;

        // Get table name if specified
        let table_name = match &stmt.table_name {
            Some(t) => t.value.to_string(),
            None => {
                return Err(Error::InvalidArgument(
                    "DROP INDEX requires table name".to_string(),
                ))
            }
        };

        // Check if table exists
        if !self.engine.table_exists(&table_name)? {
            if stmt.if_exists {
                return Ok(Box::new(ExecResult::empty()));
            }
            return Err(Error::TableNotFound(table_name));
        }

        // Check if index exists
        if !self.engine.index_exists(index_name, &table_name)? {
            if stmt.if_exists {
                return Ok(Box::new(ExecResult::empty()));
            }
            return Err(Error::IndexNotFound(index_name.to_string()));
        }

        // Record index drop to WAL BEFORE applying in-memory change.
        // This prevents ghost state where the index is removed from memory
        // but still exists in WAL (would reappear after crash recovery).
        self.engine.record_drop_index(&table_name, index_name)?;

        // Now apply the in-memory change
        let tx = self.engine.begin_writable_transaction_internal()?;
        let table = tx.get_table(&table_name)?;
        table.drop_index(index_name)?;

        Ok(Box::new(ExecResult::empty()))
    }

    /// Execute an ALTER TABLE statement
    pub(crate) fn execute_alter_table(
        &self,
        stmt: &AlterTableStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Hold SH on the engine's `transactional_ddl_fence`
        // across every ALTER variant in this function.
        // ALTER mutates the table schema (and refreshes the
        // engine schema cache) BEFORE the corresponding
        // `record_alter_table_*` WAL write. Without the
        // fence a checkpoint that landed mid-window could
        // snapshot the post-ALTER schema into a CreateTable
        // re-record; if the WAL write later fails and this
        // branch restores the pre-ALTER schema in memory,
        // crash recovery still sees the checkpoint's
        // re-recorded post-ALTER schema and the two views
        // diverge. Same hazard applies to add / drop /
        // rename / modify column and rename table — all
        // inside this same function — so a single
        // function-scoped guard covers every variant.
        let _ddl_fence_guard = self.engine.ddl_fence().read();

        let table_name = &stmt.table_name.value;

        // Check if table exists
        if !self.engine.table_exists(table_name)? {
            return Err(Error::TableNotFound(table_name.to_string()));
        }

        // Get the table for modifications
        let mut tx = self.engine.begin_writable_transaction_internal()?;
        let mut table = tx.get_table(table_name)?;

        // Recheck the catastrophic-failure latch before mutating the
        // table schema. begin_writable_transaction_internal checked
        // it once, but a concurrent marker-failure commit can latch
        // the engine in the gap before we touch
        // create_column_with_default_value / drop_column / etc.
        // Without this check the schema mutation lands in memory,
        // then `record_alter_table_*` (which routes through
        // record_ddl) hits the failed-latch guard and returns Err —
        // leaving the in-memory ALTER live but never recorded to
        // WAL. `MvccTransaction::Drop` only rolls back created /
        // dropped tables, not column-level edits, so the divergence
        // would persist for the lifetime of the process.
        if self.engine.is_failed() {
            return Err(Error::internal(
                "ALTER TABLE refused: engine is in the catastrophic-failure \
                 state from a prior commit's marker write failure. Restart \
                 the process; recovery will discard the markerless transaction.",
            ));
        }

        // Process the single ALTER TABLE operation
        match stmt.operation {
            AlterTableOperation::AddColumn => {
                if let Some(ref col_def) = stmt.column_def {
                    let data_type = self.parse_data_type(&col_def.data_type)?;
                    let nullable = !col_def
                        .constraints
                        .iter()
                        .any(|c| matches!(c, ColumnConstraint::NotNull));

                    // Extract default expression if present
                    let default_expr = col_def.constraints.iter().find_map(|c| {
                        if let ColumnConstraint::Default(expr) = c {
                            Some(expr.to_string())
                        } else {
                            None
                        }
                    });

                    // Pre-compute the default value for schema evolution (backfilling existing rows)
                    // The default_expr string is also stored for new INSERTs
                    let default_value = if let Some(ref expr_str) = default_expr {
                        let val = self.evaluate_default_expression(expr_str, data_type)?;
                        if val.is_null() {
                            None
                        } else {
                            Some(val)
                        }
                    } else {
                        None
                    };

                    // Snapshot pre-mutation schema so the WAL-failure
                    // revert can restore EVERY field verbatim
                    // (column ids, positions, PK / auto-increment /
                    // check / default metadata) instead of
                    // approximating with an inverse op.
                    let pre_schema = table.schema().clone();

                    table.create_column_with_default_value(
                        &col_def.name.value,
                        data_type,
                        nullable,
                        default_expr.clone(),
                        default_value,
                    )?;

                    // Refresh engine's schema cache from version store
                    // The table modified the version_store schema, but engine has a separate cache
                    self.engine.refresh_schema_cache(table_name)?;

                    // Record ALTER TABLE ADD COLUMN to WAL for persistence.
                    // record_ddl has the failed-latch guard. If the
                    // latch was set after our up-front check (between
                    // create_column_with_default_value above and this
                    // call), the WAL record fails and we MUST revert
                    // the in-memory mutation — otherwise the column
                    // stays live in memory but recovery has no record
                    // of it, and `MvccTransaction::Drop` only rolls
                    // back created/dropped tables, not column edits.
                    let vector_dimensions = if data_type == DataType::Vector {
                        crate::executor::utils::parse_vector_dimension(&col_def.data_type)
                    } else {
                        0
                    };
                    if let Err(e) = self.engine.record_alter_table_add_column(
                        table_name,
                        &col_def.name.value,
                        data_type,
                        nullable,
                        default_expr.as_deref(),
                        vector_dimensions,
                    ) {
                        // Restore the entire pre-mutation schema.
                        let _ = self.engine.restore_table_schema(table_name, pre_schema);
                        return Err(e);
                    }
                } else {
                    return Err(Error::InvalidArgument(
                        "ADD COLUMN requires column definition".to_string(),
                    ));
                }
            }
            AlterTableOperation::DropColumn => {
                if let Some(ref col_name) = stmt.column_name {
                    // Snapshot the entire pre-drop schema so revert
                    // restores every column field verbatim — id,
                    // position, PK / auto-increment / check / default
                    // metadata. Reconstructing via
                    // create_column_with_default_value would silently
                    // change those.
                    let pre_schema = table.schema().clone();

                    table.drop_column(&col_name.value)?;

                    // Refresh schema cache FIRST so invalidate_mappings sees the post-drop schema
                    self.engine.refresh_schema_cache(table_name)?;

                    // Record ALTER TABLE DROP COLUMN to WAL BEFORE
                    // propagating to the manifest. propagate_column_drop
                    // mutates segment-manager state that's harder to
                    // revert; doing the WAL call first lets us bail
                    // cleanly with just the table-schema revert if the
                    // latch flipped between the up-front check and now.
                    if let Err(e) = self
                        .engine
                        .record_alter_table_drop_column(table_name, &col_name.value)
                    {
                        // Restore the entire pre-mutation schema.
                        // Indexes the original column had are NOT
                        // re-attached here — restart's WAL replay
                        // will rebuild them from DDL since no DROP
                        // COLUMN landed in WAL.
                        let _ = self.engine.restore_table_schema(table_name, pre_schema);
                        return Err(e);
                    }

                    // Record column drop in manifest and recompute cold volume mappings
                    self.engine
                        .propagate_column_drop(table_name, &col_name.value);
                } else {
                    return Err(Error::InvalidArgument(
                        "DROP COLUMN requires column name".to_string(),
                    ));
                }
            }
            AlterTableOperation::RenameColumn => match (&stmt.column_name, &stmt.new_column_name) {
                (Some(old_name), Some(new_name)) => {
                    let pre_schema = table.schema().clone();

                    table.rename_column(&old_name.value, &new_name.value)?;

                    // Refresh engine's schema cache from version store
                    self.engine.refresh_schema_cache(table_name)?;

                    // Record WAL BEFORE propagating the alias, so a
                    // latch-set failure has only the table-schema
                    // rename to revert via wholesale schema restore.
                    if let Err(e) = self.engine.record_alter_table_rename_column(
                        table_name,
                        &old_name.value,
                        &new_name.value,
                    ) {
                        let _ = self.engine.restore_table_schema(table_name, pre_schema);
                        return Err(e);
                    }

                    // Propagate rename alias to cold volumes
                    self.engine.propagate_column_alias(
                        table_name,
                        &new_name.value,
                        &old_name.value,
                    );
                }
                _ => {
                    return Err(Error::InvalidArgument(
                        "RENAME COLUMN requires old and new column names".to_string(),
                    ));
                }
            },
            AlterTableOperation::ModifyColumn => {
                if let Some(ref col_def) = stmt.column_def {
                    let data_type = self.parse_data_type(&col_def.data_type)?;
                    let nullable = !col_def
                        .constraints
                        .iter()
                        .any(|c| matches!(c, ColumnConstraint::NotNull));

                    // Validate existing data satisfies NOT NULL before applying.
                    // Use IS NULL filter + limit 1 for streaming early-exit scan
                    // instead of materializing the full table.
                    if !nullable {
                        let schema = table.schema();
                        if schema.get_column_index(&col_def.name.value).is_some() {
                            let col_name = col_def.name.value.to_string();
                            let mut filter = crate::storage::expression::ComparisonExpr::new(
                                col_name.clone(),
                                crate::core::Operator::IsNull,
                                Value::Null(data_type),
                            );
                            filter.prepare_for_schema(schema);
                            let nulls =
                                table.collect_rows_with_limit_unordered(Some(&filter), 1, 0)?;
                            if !nulls.is_empty() {
                                return Err(Error::not_null_constraint(col_name));
                            }
                        }
                    }

                    // Snapshot the entire pre-modify schema for the
                    // WAL-failure revert path.
                    let pre_schema = table.schema().clone();

                    table.modify_column(&col_def.name.value, data_type, nullable)?;

                    // Refresh engine's schema cache from version store
                    self.engine.refresh_schema_cache(table_name)?;

                    // Record ALTER TABLE MODIFY COLUMN to WAL.
                    // Revert via wholesale schema restore on
                    // latch-set failure.
                    let vector_dimensions = if data_type == DataType::Vector {
                        crate::executor::utils::parse_vector_dimension(&col_def.data_type)
                    } else {
                        0
                    };
                    if let Err(e) = self.engine.record_alter_table_modify_column(
                        table_name,
                        &col_def.name.value,
                        data_type,
                        nullable,
                        vector_dimensions,
                    ) {
                        let _ = self.engine.restore_table_schema(table_name, pre_schema);
                        return Err(e);
                    }
                } else {
                    return Err(Error::InvalidArgument(
                        "MODIFY COLUMN requires column definition".to_string(),
                    ));
                }
            }
            AlterTableOperation::RenameTable => {
                if let Some(ref new_name) = stmt.new_table_name {
                    tx.rename_table(table_name, &new_name.value)?;

                    // Record ALTER TABLE RENAME TO WAL. On WAL
                    // failure (typically the failed-latch flipping
                    // between tx.rename_table's begin-time check and
                    // now) we MUST undo the in-memory + on-disk
                    // rename. Routing the revert back through
                    // tx.rename_table would be refused by the same
                    // latch — use the engine's
                    // `rename_table_revert` path which bypasses the
                    // latch check (the latch was set BY the failure
                    // we're reverting from, so refusing the revert
                    // would leave durable state diverged with no
                    // path to fix it short of restart-and-replay).
                    if let Err(e) = self
                        .engine
                        .record_alter_table_rename(table_name, &new_name.value)
                    {
                        let _ = self.engine.rename_table_revert(&new_name.value, table_name);
                        return Err(e);
                    }
                } else {
                    return Err(Error::InvalidArgument(
                        "RENAME TABLE requires new table name".to_string(),
                    ));
                }
            }
        }

        tx.commit()?;

        // Invalidate query cache for this table (schema may have changed)
        self.query_cache.invalidate_table(table_name);
        invalidate_semi_join_cache_for_table(table_name);
        invalidate_scalar_subquery_cache_for_table(table_name);
        invalidate_in_subquery_cache_for_table(table_name);

        Ok(Box::new(ExecResult::empty()))
    }

    /// Execute a CREATE VIEW statement
    pub(crate) fn execute_create_view(
        &self,
        stmt: &CreateViewStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let view_name = &stmt.view_name.value;

        // Check if a table with the same name exists
        if self.engine.table_exists(view_name)? {
            return Err(Error::TableAlreadyExists(view_name.to_string()));
        }

        // Convert the query to SQL string
        let query_sql = stmt.query.to_string();

        // Create the view (engine handles if_not_exists logic)
        self.engine
            .create_view(view_name, query_sql, stmt.if_not_exists)?;

        Ok(Box::new(ExecResult::empty()))
    }

    /// Execute a DROP VIEW statement
    pub(crate) fn execute_drop_view(
        &self,
        stmt: &DropViewStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let view_name = &stmt.view_name.value;

        // Drop the view (engine handles if_exists logic)
        self.engine.drop_view(view_name, stmt.if_exists)?;

        // Invalidate subquery caches that may reference this view
        invalidate_semi_join_cache_for_table(view_name);
        invalidate_scalar_subquery_cache_for_table(view_name);
        invalidate_in_subquery_cache_for_table(view_name);

        Ok(Box::new(ExecResult::empty()))
    }

    /// Parse a SQL data type string to DataType enum
    pub(crate) fn parse_data_type(&self, type_str: &str) -> Result<DataType> {
        let upper = type_str.to_uppercase();
        let base_type = upper.split('(').next().unwrap_or(&upper);

        match base_type {
            "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(DataType::Integer),
            "FLOAT" | "DOUBLE" | "REAL" | "DECIMAL" | "NUMERIC" => Ok(DataType::Float),
            "TEXT" | "VARCHAR" | "CHAR" | "STRING" | "CLOB" => Ok(DataType::Text),
            "BOOLEAN" | "BOOL" => Ok(DataType::Boolean),
            // Date and time are all stored as Timestamp
            "TIMESTAMP" | "DATETIME" | "DATE" | "TIME" => Ok(DataType::Timestamp),
            "JSON" | "JSONB" => Ok(DataType::Json),
            // Binary data stored as Text (base64 encoded)
            "BLOB" | "BINARY" | "VARBINARY" => Ok(DataType::Text),
            "VECTOR" => Ok(DataType::Vector),
            _ => Err(Error::Type(format!("Unknown data type: {}", type_str))),
        }
    }

    /// Evaluate a default expression string and return the resulting Value
    fn evaluate_default_expression(
        &self,
        default_expr: &str,
        target_type: DataType,
    ) -> Result<Value> {
        use crate::parser::parse_sql;

        // Parse the default expression as a SELECT expression
        let sql = format!("SELECT {}", default_expr);
        let stmts = match parse_sql(&sql) {
            Ok(s) => s,
            Err(_) => return Ok(Value::null(target_type)),
        };
        if stmts.is_empty() {
            return Ok(Value::null(target_type));
        }

        // Extract the expression from the SELECT statement
        if let Statement::Select(select) = &stmts[0] {
            if let Some(expr) = select.columns.first() {
                let mut eval = ExpressionEval::compile(expr, &[])?;
                let value = eval.eval_slice(&Row::new())?;
                return Ok(value.into_coerce_to_type(target_type));
            }
        }

        Ok(Value::null(target_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mvcc::engine::MVCCEngine;
    use std::sync::Arc;

    fn create_test_executor() -> Executor {
        let engine = MVCCEngine::in_memory();
        engine.open_engine().unwrap();
        Executor::new(Arc::new(engine))
    }

    #[test]
    fn test_create_table() {
        let executor = create_test_executor();

        let result = executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
            .unwrap();
        assert_eq!(result.rows_affected(), 0);

        // Verify table exists
        assert!(executor.engine().table_exists("users").unwrap());
    }

    #[test]
    fn test_create_table_if_not_exists() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            .unwrap();

        // Should not error with IF NOT EXISTS
        let result = executor
            .execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)")
            .unwrap();
        assert_eq!(result.rows_affected(), 0);
    }

    #[test]
    fn test_create_table_already_exists() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            .unwrap();

        // Should error without IF NOT EXISTS
        let result = executor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)");
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_table() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            .unwrap();
        assert!(executor.engine().table_exists("users").unwrap());

        executor.execute("DROP TABLE users").unwrap();
        assert!(!executor.engine().table_exists("users").unwrap());
    }

    #[test]
    fn test_drop_table_if_exists() {
        let executor = create_test_executor();

        // Should not error with IF EXISTS
        let result = executor
            .execute("DROP TABLE IF EXISTS nonexistent")
            .unwrap();
        assert_eq!(result.rows_affected(), 0);
    }

    #[test]
    fn test_drop_table_not_found() {
        let executor = create_test_executor();

        let result = executor.execute("DROP TABLE nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_index() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();

        let result = executor
            .execute("CREATE INDEX idx_name ON users (name)")
            .unwrap();
        assert_eq!(result.rows_affected(), 0);
    }

    #[test]
    fn test_create_unique_index() {
        let executor = create_test_executor();

        executor
            .execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)")
            .unwrap();

        let result = executor
            .execute("CREATE UNIQUE INDEX idx_email ON users (email)")
            .unwrap();
        assert_eq!(result.rows_affected(), 0);
    }

    #[test]
    fn test_parse_data_type() {
        let executor = create_test_executor();

        assert_eq!(
            executor.parse_data_type("INTEGER").unwrap(),
            DataType::Integer
        );
        assert_eq!(executor.parse_data_type("INT").unwrap(), DataType::Integer);
        assert_eq!(
            executor.parse_data_type("BIGINT").unwrap(),
            DataType::Integer
        );
        assert_eq!(executor.parse_data_type("FLOAT").unwrap(), DataType::Float);
        assert_eq!(executor.parse_data_type("DOUBLE").unwrap(), DataType::Float);
        assert_eq!(executor.parse_data_type("TEXT").unwrap(), DataType::Text);
        assert_eq!(
            executor.parse_data_type("VARCHAR(255)").unwrap(),
            DataType::Text
        );
        assert_eq!(
            executor.parse_data_type("BOOLEAN").unwrap(),
            DataType::Boolean
        );
        assert_eq!(
            executor.parse_data_type("TIMESTAMP").unwrap(),
            DataType::Timestamp
        );
        assert_eq!(executor.parse_data_type("JSON").unwrap(), DataType::Json);
    }
}
