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

//! Foreign Key Constraint Enforcement
//!
//! This module provides helpers for checking referential integrity:
//! - On INSERT/UPDATE: verify parent rows exist (index-based O(log n) / O(1))
//! - On DELETE/UPDATE of parent: enforce RESTRICT/CASCADE/SET NULL
//!
//! All operations participate in the caller's transaction (via `txn_id`), ensuring:
//! - CASCADE effects are atomic with the parent operation
//! - FK checks see uncommitted rows from the current transaction
//! - No independent transactions are created (no resource leaks)
//!
//! Performance guarantees:
//! - Zero cost for non-FK tables (all checks short-circuit on empty foreign_keys)
//! - Cached reverse FK mapping (rebuilt only on schema_epoch change)
//! - Index-based parent lookups (no table scans when index exists)

use std::sync::Arc;

use crate::core::{Error, ForeignKeyAction, ForeignKeyConstraint, Result, Schema, Value};
use crate::storage::expression::Expression as StorageExpression;
use crate::storage::traits::Engine;

/// Check that all FK values in a row reference existing parent rows.
/// Called on INSERT and UPDATE (when FK columns change).
///
/// Uses `txn_id` to check within the caller's transaction, so uncommitted
/// parent rows (inserted in the same transaction) are visible.
///
/// Short-circuits immediately if schema has no FKs (zero cost for non-FK tables).
pub(crate) fn check_parent_exists(
    engine: &dyn Engine,
    txn_id: i64,
    schema: &Schema,
    row: &crate::core::Row,
) -> Result<()> {
    for fk in &schema.foreign_keys {
        let fk_value = match row.get(fk.column_index) {
            Some(v) if !v.is_null() => v,
            _ => continue, // NULL FK is allowed (no reference)
        };

        if !parent_row_exists(
            engine,
            txn_id,
            &fk.referenced_table,
            &fk.referenced_column,
            fk_value,
        )? {
            return Err(Error::foreign_key_violation(
                &schema.table_name,
                &fk.column_name,
                &fk.referenced_table,
                &fk.referenced_column,
                format!(
                    "referenced row with {} = {} does not exist",
                    fk.referenced_column, fk_value
                ),
            ));
        }
    }
    Ok(())
}

/// Pre-validate a single FK value against its parent table.
/// Used for early validation of constant SET values in UPDATE statements
/// to prevent dirty state in explicit transactions.
///
/// NULL values are allowed (no reference) and should be skipped by the caller.
pub(crate) fn validate_fk_value(
    engine: &dyn Engine,
    txn_id: i64,
    fk: &ForeignKeyConstraint,
    value: &Value,
    child_table: &str,
) -> Result<()> {
    if !parent_row_exists(
        engine,
        txn_id,
        &fk.referenced_table,
        &fk.referenced_column,
        value,
    )? {
        return Err(Error::foreign_key_violation(
            child_table,
            &fk.column_name,
            &fk.referenced_table,
            &fk.referenced_column,
            format!(
                "referenced row with {} = {} does not exist",
                fk.referenced_column, value
            ),
        ));
    }
    Ok(())
}

/// Check if a value exists in the parent table's referenced column.
///
/// Uses `collect_rows_with_limit_unordered(limit=1)` with a ComparisonExpr filter
/// for both correctness and performance:
/// - O(1) via PK fast path when referenced column is the primary key (common case)
/// - O(log N) via secondary index when available
/// - Falls back to filtered scan with early termination otherwise
/// - Always txn-aware: sees uncommitted INSERTs, respects uncommitted DELETEs
fn parent_row_exists(
    engine: &dyn Engine,
    txn_id: i64,
    parent_table: &str,
    parent_column: &str,
    value: &Value,
) -> Result<bool> {
    let parent_schema = engine.get_table_schema(parent_table).map_err(|_| {
        Error::internal(format!(
            "foreign key references non-existent table '{}'",
            parent_table
        ))
    })?;

    let (_, ref_col) = parent_schema.find_column(parent_column).ok_or_else(|| {
        Error::internal(format!(
            "foreign key references non-existent column '{}' in table '{}'",
            parent_column, parent_table
        ))
    })?;

    // Build a ComparisonExpr for `column = value` — the storage layer will use
    // PK fast path (O(1)) or secondary index (O(log N)) when available
    let mut expr = crate::storage::expression::ComparisonExpr::new(
        ref_col.name.as_str(),
        crate::core::Operator::Eq,
        value.clone(),
    );
    expr.prepare_for_schema(&parent_schema);

    let parent = engine.get_table_for_txn(txn_id, parent_table)?;
    let rows = parent.collect_rows_with_limit_unordered(Some(&expr), 1, 0)?;
    Ok(!rows.is_empty())
}

/// Find all foreign key constraints in other tables that reference the given parent table.
/// Delegates to the engine's cached reverse mapping (rebuilt only on schema_epoch change).
/// Returns Arc-wrapped Vec (ref-count bump only, no cloning).
pub(crate) fn find_referencing_fks(
    engine: &dyn Engine,
    parent_table: &str,
) -> Arc<Vec<(String, ForeignKeyConstraint)>> {
    engine.find_referencing_fks(parent_table)
}

/// Enforce referential actions for DELETE from a parent table.
/// Accepts an iterator of PK values to avoid allocating a separate Vec.
///
/// All CASCADE/SET NULL operations use `txn_id` to participate in the caller's
/// transaction, ensuring atomicity (rollback undoes cascade effects).
///
/// For each deleted PK value, checks all child tables:
/// - RESTRICT/NO ACTION: error if child rows exist
/// - CASCADE: delete matching child rows (batched per child table)
/// - SET NULL: set FK column to NULL in matching child rows (batched per child table)
///
/// Returns the total count of cascaded/affected child rows.
pub(crate) fn enforce_delete_actions_iter<'a>(
    engine: &dyn Engine,
    txn_id: i64,
    parent_table: &str,
    deleted_pk_values: impl Iterator<Item = &'a Value>,
    referencing_fks: &[(String, ForeignKeyConstraint)],
) -> Result<i32> {
    if referencing_fks.is_empty() {
        return Ok(0);
    }

    let mut total_affected = 0i32;

    for pk_value in deleted_pk_values {
        for (child_table_name, fk) in referencing_fks {
            let action = fk.on_delete;

            match action {
                ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {
                    // Check if any child rows reference this PK value
                    if child_rows_exist(engine, txn_id, child_table_name, fk, pk_value)? {
                        return Err(Error::foreign_key_violation(
                            child_table_name,
                            &fk.column_name,
                            parent_table,
                            &fk.referenced_column,
                            format!(
                                "cannot delete row with {} = {} — still referenced by table '{}'",
                                fk.referenced_column, pk_value, child_table_name
                            ),
                        ));
                    }
                }
                ForeignKeyAction::Cascade => {
                    // Delete matching child rows within the caller's transaction
                    let affected = cascade_delete(engine, txn_id, child_table_name, fk, pk_value)?;
                    total_affected = total_affected.saturating_add(affected);
                }
                ForeignKeyAction::SetNull => {
                    // Set FK column to NULL in matching child rows
                    let affected =
                        set_null_on_delete(engine, txn_id, child_table_name, fk, pk_value)?;
                    total_affected = total_affected.saturating_add(affected);
                }
            }
        }
    }

    Ok(total_affected)
}

/// Enforce referential actions for UPDATE of a parent PK.
/// All operations participate in the caller's transaction via `txn_id`.
pub(crate) fn enforce_update_actions(
    engine: &dyn Engine,
    txn_id: i64,
    parent_table: &str,
    old_pk_value: &Value,
    new_pk_value: &Value,
    referencing_fks: &[(String, ForeignKeyConstraint)],
) -> Result<i32> {
    if referencing_fks.is_empty() {
        return Ok(0);
    }

    let mut total_affected = 0i32;

    for (child_table_name, fk) in referencing_fks {
        let action = fk.on_update;

        match action {
            ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {
                if child_rows_exist(engine, txn_id, child_table_name, fk, old_pk_value)? {
                    return Err(Error::foreign_key_violation(
                        child_table_name,
                        &fk.column_name,
                        parent_table,
                        &fk.referenced_column,
                        format!(
                            "cannot update row with {} = {} — still referenced by table '{}'",
                            fk.referenced_column, old_pk_value, child_table_name
                        ),
                    ));
                }
            }
            ForeignKeyAction::Cascade => {
                let affected = cascade_update(
                    engine,
                    txn_id,
                    child_table_name,
                    fk,
                    old_pk_value,
                    new_pk_value,
                )?;
                total_affected = total_affected.saturating_add(affected);
            }
            ForeignKeyAction::SetNull => {
                let affected =
                    set_null_on_delete(engine, txn_id, child_table_name, fk, old_pk_value)?;
                total_affected = total_affected.saturating_add(affected);
            }
        }
    }

    Ok(total_affected)
}

/// Check if any child rows in the child table reference the given parent PK value.
///
/// Uses `collect_rows_with_limit_unordered(limit=1)` with a ComparisonExpr filter
/// for both correctness and performance:
/// - O(log N) via secondary index when an index exists on the FK column
/// - Falls back to filtered scan with early termination otherwise
/// - Always txn-aware: sees uncommitted INSERTs, respects uncommitted DELETEs
fn child_rows_exist(
    engine: &dyn Engine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    parent_pk_value: &Value,
) -> Result<bool> {
    let child = engine.get_table_for_txn(txn_id, child_table)?;
    let child_schema = child.schema();

    // Build a ComparisonExpr for `fk_column = parent_pk_value`
    let col_name = &child_schema.columns[fk.column_index].name;
    let mut expr = crate::storage::expression::ComparisonExpr::new(
        col_name.as_str(),
        crate::core::Operator::Eq,
        parent_pk_value.clone(),
    );
    expr.prepare_for_schema(child_schema);

    let rows = child.collect_rows_with_limit_unordered(Some(&expr), 1, 0)?;
    Ok(!rows.is_empty())
}

/// Maximum CASCADE recursion depth to prevent infinite loops from circular FK references.
const MAX_CASCADE_DEPTH: usize = 16;

/// CASCADE DELETE: delete all child rows referencing the given parent PK value.
/// Operates within the caller's transaction (no independent commit).
/// Recursively cascades to grandchild tables (up to MAX_CASCADE_DEPTH).
fn cascade_delete(
    engine: &dyn Engine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    parent_pk_value: &Value,
) -> Result<i32> {
    cascade_delete_recursive(engine, txn_id, child_table, fk, parent_pk_value, 0)
}

fn cascade_delete_recursive(
    engine: &dyn Engine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    parent_pk_value: &Value,
    depth: usize,
) -> Result<i32> {
    if depth >= MAX_CASCADE_DEPTH {
        return Err(Error::internal(format!(
            "foreign key CASCADE depth limit ({}) exceeded — possible circular reference",
            MAX_CASCADE_DEPTH
        )));
    }

    // Before deleting child rows, collect their PK values for recursive CASCADE.
    // This is needed because the child table may itself be a parent with CASCADE children.
    let grandchild_fks = find_referencing_fks(engine, child_table);
    let mut deleted_child_pks: Vec<Value> = Vec::new();

    if !grandchild_fks.is_empty() {
        // Need to collect PK values of rows about to be deleted
        let child_schema = engine.get_table_schema(child_table)?;
        if let Some(pk_idx) = child_schema.pk_column_index() {
            let child_handle = engine.get_table_for_txn(txn_id, child_table)?;
            // Use filtered scan instead of full table scan
            let col_name = &child_schema.columns[fk.column_index].name;
            let mut filter = crate::storage::expression::ComparisonExpr::new(
                col_name.as_str(),
                crate::core::Operator::Eq,
                parent_pk_value.clone(),
            );
            filter.prepare_for_schema(&child_schema);
            let rows = child_handle.collect_all_rows(Some(&filter))?;
            for (_, row) in rows.iter() {
                if let Some(pk_val) = row.get(pk_idx) {
                    deleted_child_pks.push(pk_val.clone());
                }
            }
        }
    }

    // Pre-check: verify grandchild RESTRICT constraints BEFORE deleting child rows.
    // If we deleted children first and a grandchild RESTRICT check fails, the child
    // deletions would remain in the transaction state (orphaning data in explicit txns).
    if !grandchild_fks.is_empty() && !deleted_child_pks.is_empty() {
        for child_pk in &deleted_child_pks {
            for (grandchild_table, grandchild_fk) in grandchild_fks.iter() {
                if matches!(
                    grandchild_fk.on_delete,
                    ForeignKeyAction::Restrict | ForeignKeyAction::NoAction
                ) && child_rows_exist(engine, txn_id, grandchild_table, grandchild_fk, child_pk)?
                {
                    return Err(Error::foreign_key_violation(
                        grandchild_table,
                        &grandchild_fk.column_name,
                        child_table,
                        &grandchild_fk.referenced_column,
                        format!(
                            "cannot cascade-delete row with {} = {} — still referenced by table '{}'",
                            grandchild_fk.referenced_column, child_pk, grandchild_table
                        ),
                    ));
                }
            }
        }
    }

    // Now delete the matching child rows (safe — RESTRICT checks passed above)
    let mut child = engine.get_table_for_txn(txn_id, child_table)?;
    let child_schema = child.schema();
    let col_name = &child_schema.columns[fk.column_index].name;
    let mut expr = crate::storage::expression::ComparisonExpr::new(
        col_name.as_str(),
        crate::core::Operator::Eq,
        parent_pk_value.clone(),
    );
    expr.prepare_for_schema(child_schema);

    let count = child.delete(Some(&expr))?;
    // Do NOT commit here — changes stay in TransactionVersionStore and are committed
    // atomically when the parent transaction's commit_all_tables() runs.

    let mut total = count;

    // Recursively enforce CASCADE/SET NULL on grandchild tables (RESTRICT already checked above)
    if !grandchild_fks.is_empty() && !deleted_child_pks.is_empty() {
        for child_pk in &deleted_child_pks {
            for (grandchild_table, grandchild_fk) in grandchild_fks.iter() {
                match grandchild_fk.on_delete {
                    ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {
                        // Already checked above — skip
                    }
                    ForeignKeyAction::Cascade => {
                        let affected = cascade_delete_recursive(
                            engine,
                            txn_id,
                            grandchild_table,
                            grandchild_fk,
                            child_pk,
                            depth + 1,
                        )?;
                        total = total.saturating_add(affected);
                    }
                    ForeignKeyAction::SetNull => {
                        let affected = set_null_on_delete(
                            engine,
                            txn_id,
                            grandchild_table,
                            grandchild_fk,
                            child_pk,
                        )?;
                        total = total.saturating_add(affected);
                    }
                }
            }
        }
    }

    Ok(total)
}

/// CASCADE UPDATE: update FK column in all child rows from old to new value.
/// Operates within the caller's transaction (no independent commit).
fn cascade_update(
    engine: &dyn Engine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    old_value: &Value,
    new_value: &Value,
) -> Result<i32> {
    let mut child = engine.get_table_for_txn(txn_id, child_table)?;

    let col_idx = fk.column_index;
    let new_val = new_value.clone();

    // Build a storage expression for FK column = old_value
    let child_schema = child.schema();
    let col_name = &child_schema.columns[col_idx].name;
    let mut expr = crate::storage::expression::ComparisonExpr::new(
        col_name.as_str(),
        crate::core::Operator::Eq,
        old_value.clone(),
    );
    expr.prepare_for_schema(child_schema);

    let count = child.update(Some(&expr), &mut |mut row| {
        let _ = row.set(col_idx, new_val.clone());
        Ok((row, true))
    })?;
    // Do NOT commit here — changes committed atomically with parent transaction.

    Ok(count)
}

/// SET NULL: set FK column to NULL in all child rows referencing the given parent PK value.
/// Operates within the caller's transaction (no independent commit).
fn set_null_on_delete(
    engine: &dyn Engine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    parent_pk_value: &Value,
) -> Result<i32> {
    let mut child = engine.get_table_for_txn(txn_id, child_table)?;

    let col_idx = fk.column_index;

    // Check that the FK column is nullable
    let child_schema = child.schema();
    if !child_schema.columns[col_idx].nullable {
        return Err(Error::foreign_key_violation(
            child_table,
            &fk.column_name,
            &fk.referenced_table,
            &fk.referenced_column,
            format!(
                "cannot SET NULL on non-nullable column '{}'",
                fk.column_name
            ),
        ));
    }

    let null_val = Value::null(child_schema.columns[col_idx].data_type);
    let col_name = &child_schema.columns[col_idx].name;
    let mut expr = crate::storage::expression::ComparisonExpr::new(
        col_name.as_str(),
        crate::core::Operator::Eq,
        parent_pk_value.clone(),
    );
    expr.prepare_for_schema(child_schema);

    let count = child.update(Some(&expr), &mut |mut row| {
        let _ = row.set(col_idx, null_val.clone());
        Ok((row, true))
    })?;
    // Do NOT commit here — changes committed atomically with parent transaction.

    Ok(count)
}

/// Check if any child tables have rows that actually reference the given parent table.
/// Used by DROP TABLE and TRUNCATE to ensure no referencing rows exist.
/// Only counts rows where the FK column is non-NULL (NULL means "no reference").
///
/// Blocks for ALL FK action types (RESTRICT, CASCADE, SET NULL, NO ACTION) because
/// DROP TABLE/TRUNCATE are DDL operations that don't cascade to child rows — they
/// would leave orphaned references. The user must delete child rows first.
///
/// When `txn_id` is provided, uses the caller's transaction for visibility (sees
/// uncommitted deletes within an explicit transaction). Otherwise creates a fresh
/// read-only transaction.
pub(crate) fn check_no_referencing_rows(
    engine: &dyn Engine,
    parent_table: &str,
    txn_id: Option<i64>,
) -> Result<()> {
    let referencing = find_referencing_fks(engine, parent_table);
    if referencing.is_empty() {
        return Ok(());
    }

    for (child_table, fk) in referencing.iter() {
        // Build IS NOT NULL filter on the FK column — pushed down to storage layer
        // so indexes can be used and we stop after the first match (limit=1)
        let child_schema = engine.get_table_schema(child_table)?;
        let col_name = &child_schema.columns[fk.column_index].name;
        let mut not_null_expr =
            crate::storage::expression::NullCheckExpr::is_not_null(col_name.as_str());
        not_null_expr.prepare_for_schema(&child_schema);

        let has_ref = if let Some(tid) = txn_id {
            let child = engine.get_table_for_txn(tid, child_table)?;
            !child
                .collect_rows_with_limit_unordered(Some(&not_null_expr), 1, 0)?
                .is_empty()
        } else {
            let tx = engine.begin_transaction()?;
            let child = tx.get_table(child_table)?;
            !child
                .collect_rows_with_limit_unordered(Some(&not_null_expr), 1, 0)?
                .is_empty()
        };

        if has_ref {
            return Err(Error::foreign_key_violation(
                child_table,
                &fk.column_name,
                parent_table,
                &fk.referenced_column,
                format!(
                    "cannot drop/truncate table '{}' — rows in '{}' still reference it",
                    parent_table, child_table
                ),
            ));
        }
    }

    Ok(())
}
