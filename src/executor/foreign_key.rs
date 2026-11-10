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

use std::cmp::Ordering;
use std::sync::Arc;

use crate::common::I64Set;
use crate::core::{Error, ForeignKeyAction, ForeignKeyConstraint, Result, Schema, Value};
use crate::storage::expression::Expression as StorageExpression;
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::traits::transaction::StatementTable;
use crate::storage::traits::{Engine, ScanAccess};

/// Check that all FK values in a row reference existing parent rows.
/// Called on INSERT and UPDATE (when FK columns change).
///
/// Uses `txn_id` to check within the caller's transaction, so uncommitted
/// parent rows (inserted in the same transaction) are visible.
///
/// Short-circuits immediately if schema has no FKs (zero cost for non-FK tables).
pub(crate) fn check_parent_exists(
    engine: &MVCCEngine,
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
    engine: &MVCCEngine,
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
    engine: &MVCCEngine,
    txn_id: i64,
    parent_table: &str,
    parent_column: &str,
    value: &Value,
) -> Result<bool> {
    let missing = |e| match e {
        Error::TableNotFound(_) => Error::internal(format!(
            "foreign key references non-existent table '{}'",
            parent_table
        )),
        e => e,
    };
    let open = || {
        engine
            .get_table_to_check(txn_id, parent_table)
            .map_err(missing)
    };
    let mut parent = open()?;
    // The filter's positions come from the handle that reads the rows
    let mut parent_schema = parent.schema_arc();
    let mut ref_idx = referenced_position(&parent_schema, parent_table, parent_column)?;

    // Held before the read: a delete of the key then waits for this
    // transaction, or has committed and the read misses the row
    let key = key_value(&parent_schema, ref_idx, value);
    #[cfg(feature = "test-failpoints")]
    crate::test_failpoints::fk_parent_hold_next();
    let column = &parent_schema.columns[ref_idx].name;
    engine.hold_key(txn_id, parent_table, column, &key, false)?;
    // A snapshot's handle took its view before the hold, so it is taken again
    if engine.reads_snapshot(txn_id) {
        parent = open()?;
        parent_schema = parent.schema_arc();
        ref_idx = referenced_position(&parent_schema, parent_table, parent_column)?;
    }

    // The storage layer uses the PK fast path (O(1)) or a secondary index
    // (O(log N)) when available
    let expr = filter_on(&parent_schema, ref_idx, key);

    let plan = parent.index_plan(&expr);
    let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
    let rows = parent.collect_access_rows_with_limit_unordered(access, Some(&expr), 1, 0)?;
    Ok(!rows.is_empty())
}

/// The position in `schema` of the column a foreign key references. A name
/// stored in lower case, as a key naming the primary key by default is, is
/// found without lowering it again
fn referenced_position(schema: &Schema, parent_table: &str, column: &str) -> Result<usize> {
    schema
        .column_index_map()
        .get(column)
        .copied()
        .or_else(|| schema.find_column(column).map(|(at, _)| at))
        .ok_or_else(|| {
            Error::internal(format!(
                "foreign key references non-existent column '{}' in table '{}'",
                column, parent_table
            ))
        })
}

/// Holds each of `values` of the column `fk` references in `parent_table`
/// alone until the transaction ends. Taken before the rows pointing at a
/// key are read: a child check of the key then waits for this transaction,
/// or has committed and the read finds its row
fn hold_alone<'v>(
    engine: &MVCCEngine,
    txn_id: i64,
    parent_table: &str,
    fk: &ForeignKeyConstraint,
    values: impl IntoIterator<Item = &'v Value>,
) -> Result<()> {
    let schema = engine.get_table_schema(parent_table)?;
    let at = referenced_position(&schema, parent_table, &fk.referenced_column)?;
    let column = &schema.columns[at].name;
    for value in values.into_iter().filter(|v| !v.is_null()) {
        engine.hold_key(txn_id, parent_table, column, value, true)?;
    }
    Ok(())
}

/// Find all foreign key constraints in other tables that reference the given parent table.
/// Delegates to the engine's cached reverse mapping (rebuilt only on schema_epoch change).
/// Returns Arc-wrapped Vec (ref-count bump only, no cloning).
pub(crate) fn find_referencing_fks(
    engine: &MVCCEngine,
    parent_table: &str,
) -> Arc<Vec<(String, ForeignKeyConstraint)>> {
    engine.find_referencing_fks(parent_table)
}

/// Ask every key that can refuse the delete before the first change is
/// made, so a refusal leaves the statement having written nothing.
///
/// The walk follows CASCADE and SET NULL down to the keys beneath them: a
/// RESTRICT on a grandchild refuses the delete just as one on a child does,
/// and asking only the children would let the first parent's child go
/// before the second parent's grandchild refused. The walk is skipped when
/// the tree holds nothing that can refuse.
pub(crate) fn pre_check_delete_keys(
    engine: &MVCCEngine,
    txn_id: i64,
    parent_table: &str,
    values: &[&Value],
    referencing_fks: &[(String, ForeignKeyConstraint)],
) -> Result<()> {
    for (_, fk) in referencing_fks {
        hold_alone(engine, txn_id, parent_table, fk, values.iter().copied())?;
    }
    if !fk_tree_can_refuse_delete(engine, referencing_fks, 0) {
        return Ok(());
    }
    for pk_value in values {
        pre_check_delete_recursive(engine, txn_id, parent_table, pk_value, referencing_fks, 0)?;
    }
    Ok(())
}

/// Whether a delete of a row this tree points at can be refused anywhere
/// in it: a key that restricts, at any depth below the keys that cascade,
/// or a cascade deep enough to hit the depth limit. Reads the metadata
/// only, so a tree of nothing but CASCADE and SET NULL skips the row walk.
fn fk_tree_can_refuse_delete(
    engine: &MVCCEngine,
    referencing_fks: &[(String, ForeignKeyConstraint)],
    depth: usize,
) -> bool {
    if depth >= MAX_CASCADE_DEPTH {
        return true;
    }
    for (child_table_name, fk) in referencing_fks {
        match fk.on_delete {
            ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => return true,
            // A cascade removes whole child rows, so every key pointing at
            // the child table is asked, the way the cascade itself asks them
            ForeignKeyAction::Cascade => {
                let grandchild_fks = find_referencing_fks(engine, child_table_name);
                if fk_tree_can_refuse_delete(engine, &grandchild_fks[..], depth + 1) {
                    return true;
                }
            }
            // The child rows stay, so nothing below them changes
            ForeignKeyAction::SetNull => {}
        }
    }
    false
}

/// Walk the keys a delete of `value` would set off, the way the cascade
/// walks them, and refuse at the first key that refuses. Nothing is
/// written, so the caller can ask this of every value before acting.
fn pre_check_delete_recursive(
    engine: &MVCCEngine,
    txn_id: i64,
    parent_table: &str,
    value: &Value,
    referencing_fks: &[(String, ForeignKeyConstraint)],
    depth: usize,
) -> Result<()> {
    for (child_table_name, fk) in referencing_fks {
        match fk.on_delete {
            ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {
                if child_rows_exist(engine, txn_id, child_table_name, fk, value)? {
                    return Err(Error::foreign_key_violation(
                        child_table_name,
                        &fk.column_name,
                        parent_table,
                        &fk.referenced_column,
                        format!(
                            "cannot delete row with {} = {} — still referenced by table '{}'",
                            fk.referenced_column, value, child_table_name
                        ),
                    ));
                }
            }
            ForeignKeyAction::Cascade => {
                if depth >= MAX_CASCADE_DEPTH {
                    return Err(Error::internal(format!(
                        "foreign key CASCADE depth limit ({}) exceeded — possible circular reference",
                        MAX_CASCADE_DEPTH
                    )));
                }
                let grandchild_fks = find_referencing_fks(engine, child_table_name);
                if grandchild_fks.is_empty() {
                    continue;
                }
                // Each key beneath the child names the column it points at,
                // and the child rows that would go are read through that
                // column, whether or not it is the child's primary key
                let child_handle = engine.get_table_to_check(txn_id, child_table_name)?;
                let child_schema = child_handle.schema_arc();
                let filter = key_filter(&child_schema, key_position(&child_schema, fk)?, value);
                let rows = child_handle.collect_all_rows(Some(&filter))?;
                for gfk_entry in grandchild_fks.iter() {
                    let referenced = gfk_entry.1.referenced_column.to_lowercase();
                    let Some(ref_idx) = child_schema
                        .columns
                        .iter()
                        .position(|c| c.name_lower == referenced)
                    else {
                        continue;
                    };
                    for (_, row) in rows.iter() {
                        if let Some(child_value) = row.get(ref_idx) {
                            if child_value.is_null() {
                                continue;
                            }
                            pre_check_delete_recursive(
                                engine,
                                txn_id,
                                child_table_name,
                                child_value,
                                std::slice::from_ref(gfk_entry),
                                depth + 1,
                            )?;
                        }
                    }
                }
            }
            ForeignKeyAction::SetNull => {}
        }
    }
    Ok(())
}

/// Carry out the CASCADE and SET NULL keys for rows already found free to
/// go. Run after `pre_check_delete_keys` for every key of the table.
pub(crate) fn apply_delete_actions(
    engine: &MVCCEngine,
    txn_id: i64,
    _parent_table: &str,
    values: &[&Value],
    referencing_fks: &[(String, ForeignKeyConstraint)],
) -> Result<i32> {
    let mut total_affected = 0i32;

    for pk_value in values {
        for (child_table_name, fk) in referencing_fks {
            match fk.on_delete {
                // Asked above, before anything was written
                ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {}
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

/// Pre-check RESTRICT constraints and CASCADE depth before writing parent rows.
/// Walks the full FK tree (including recursive grandchild RESTRICT behind CASCADE/SET NULL)
/// to detect violations before any rows are modified, preserving statement atomicity.
/// Returns true if the tree has constraints that need row-level pre-checking.
pub(crate) fn pre_check_restrict_for_update(
    engine: &MVCCEngine,
    txn_id: i64,
    parent_table: &str,
    old_value: &Value,
    referencing_fks: &[(String, ForeignKeyConstraint)],
) -> Result<()> {
    for (_, fk) in referencing_fks {
        hold_alone(engine, txn_id, parent_table, fk, [old_value])?;
    }
    pre_check_restrict_recursive(engine, txn_id, parent_table, old_value, referencing_fks, 0)
}

/// Check if the FK tree rooted at these referencing FKs needs row-level pre-checking.
/// Returns true if any path contains RESTRICT/NoAction or exceeds CASCADE depth.
/// This is a metadata-only walk (no row scans) used to skip the expensive pre-scan
/// when the tree is pure CASCADE/SET NULL within depth limits.
pub(crate) fn fk_tree_needs_precheck(
    engine: &MVCCEngine,
    referencing_fks: &[(String, ForeignKeyConstraint)],
) -> bool {
    fk_tree_needs_precheck_recursive(engine, referencing_fks, 0)
}

fn fk_tree_needs_precheck_recursive(
    engine: &MVCCEngine,
    referencing_fks: &[(String, ForeignKeyConstraint)],
    depth: usize,
) -> bool {
    if depth >= MAX_CASCADE_DEPTH {
        return true; // depth limit will be hit → needs pre-check
    }
    for (child_table_name, fk) in referencing_fks {
        match fk.on_update {
            ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {
                return true;
            }
            ForeignKeyAction::Cascade | ForeignKeyAction::SetNull => {
                let grandchild_fks = find_referencing_fks(engine, child_table_name);
                let child_fk_col = &fk.column_name;
                let relevant: Vec<_> = grandchild_fks
                    .iter()
                    .filter(|(_, gfk)| gfk.referenced_column == *child_fk_col)
                    .cloned()
                    .collect();
                if !relevant.is_empty()
                    && fk_tree_needs_precheck_recursive(engine, &relevant, depth + 1)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn pre_check_restrict_recursive(
    engine: &MVCCEngine,
    txn_id: i64,
    parent_table: &str,
    old_value: &Value,
    referencing_fks: &[(String, ForeignKeyConstraint)],
    depth: usize,
) -> Result<()> {
    for (child_table_name, fk) in referencing_fks {
        // Check if child rows actually reference old_value before doing anything.
        // If no matching rows exist, neither RESTRICT nor CASCADE applies.
        if !child_rows_exist(engine, txn_id, child_table_name, fk, old_value)? {
            continue;
        }
        match fk.on_update {
            ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {
                return Err(Error::foreign_key_violation(
                    child_table_name,
                    &fk.column_name,
                    parent_table,
                    &fk.referenced_column,
                    format!(
                        "cannot update row with {} = {} — still referenced by table '{}'",
                        fk.referenced_column, old_value, child_table_name
                    ),
                ));
            }
            ForeignKeyAction::Cascade | ForeignKeyAction::SetNull => {
                // Child rows exist and will be cascaded. Check depth limit
                // now — if exceeded, the actual cascade would fail too.
                if depth >= MAX_CASCADE_DEPTH {
                    return Err(Error::internal(format!(
                        "foreign key CASCADE depth limit ({}) exceeded — possible circular reference",
                        MAX_CASCADE_DEPTH
                    )));
                }
                let grandchild_fks = find_referencing_fks(engine, child_table_name);
                let child_fk_col = &fk.column_name;
                let relevant: Vec<_> = grandchild_fks
                    .iter()
                    .filter(|(_, gfk)| gfk.referenced_column == *child_fk_col)
                    .cloned()
                    .collect();
                if !relevant.is_empty() {
                    pre_check_restrict_recursive(
                        engine,
                        txn_id,
                        child_table_name,
                        old_value,
                        &relevant,
                        depth + 1,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Enforce referential actions for UPDATE of a referenced column.
/// RESTRICT is already handled by pre_check_restrict_for_update before
/// the parent row is written. This function only dispatches CASCADE/SET NULL.
pub(crate) fn enforce_update_actions(
    engine: &MVCCEngine,
    txn_id: i64,
    changes: &[(Value, Value)],
    referencing_fks: &[(String, ForeignKeyConstraint)],
) -> Result<i32> {
    if referencing_fks.is_empty() || changes.is_empty() {
        return Ok(0);
    }

    for (_, fk) in referencing_fks {
        hold_alone(
            engine,
            txn_id,
            &fk.referenced_table,
            fk,
            changes.iter().map(|(old, _)| old),
        )?;
    }

    let mut total_affected = 0i32;

    for (child_table_name, fk) in referencing_fks {
        let action = fk.on_update;

        match action {
            ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {
                // RESTRICT is already enforced by pre_check_restrict_for_update
                // before the parent row is written. Nothing to do here.
            }
            ForeignKeyAction::Cascade => {
                let affected = cascade_update(engine, txn_id, child_table_name, fk, changes)?;
                total_affected = total_affected.saturating_add(affected);
            }
            ForeignKeyAction::SetNull => {
                for (old, _) in changes {
                    let affected = set_null_on_delete(engine, txn_id, child_table_name, fk, old)?;
                    total_affected = total_affected.saturating_add(affected);
                }
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
    engine: &MVCCEngine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    parent_pk_value: &Value,
) -> Result<bool> {
    #[cfg(feature = "test-failpoints")]
    crate::test_failpoints::fk_child_probe_next();
    let child = engine.get_table_to_check(txn_id, child_table)?;
    let child_schema = child.schema();

    let expr = key_filter(
        child_schema,
        key_position(child_schema, fk)?,
        parent_pk_value,
    );

    let plan = child.index_plan(&expr);
    let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
    let rows = child.collect_access_rows_with_limit_unordered(access, Some(&expr), 1, 0)?;
    Ok(!rows.is_empty())
}

/// Refuses with a write conflict a snapshot transaction's action on the
/// children of `parent_value` that would change a row committed after its
/// snapshot, which the snapshot does not see
fn check_children_seen(
    engine: &MVCCEngine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    parent_value: &Value,
) -> Result<()> {
    if !engine.reads_snapshot(txn_id) {
        return Ok(());
    }
    let ids = |child: Box<dyn crate::storage::traits::Table>| -> Result<I64Set> {
        let schema = child.schema();
        let expr = key_filter(schema, key_position(schema, fk)?, parent_value);
        Ok(child
            .collect_all_rows(Some(&expr))?
            .iter()
            .map(|(id, _)| *id)
            .collect())
    };
    let seen = ids(engine.get_table_as_seen(txn_id, child_table)?)?;
    let newest = ids(engine.get_table_to_check(txn_id, child_table)?)?;
    let unseen = newest.iter().find(|&id| !seen.contains(id));
    match unseen {
        Some(id) => Err(Error::internal(format!(
            "write conflict: row {id} of table '{child_table}' was committed after the transaction's snapshot"
        ))),
        None => Ok(()),
    }
}

/// The filter `column = value` over `schema`, with `value` read as
/// [`key_value`] reads it
fn key_filter(
    schema: &Schema,
    column: usize,
    value: &Value,
) -> crate::storage::expression::ComparisonExpr {
    filter_on(schema, column, key_value(schema, column, value))
}

/// `value` as `column` compares it. A number against a numeric column is
/// read as the column's type as a pushed-down equality reads it, so every
/// layer finds the rows that compare equal whichever numeric type the other
/// side has; any other value is compared as it is, both ways
fn key_value(schema: &Schema, column: usize, value: &Value) -> Value {
    let column = &schema.columns[column];
    let numbers =
        column.data_type.is_numeric() && matches!(value, Value::Integer(_) | Value::Float(_));
    numbers
        .then(|| super::pushdown::coerce_constant(Some(column.data_type), value.clone()))
        .flatten()
        .unwrap_or_else(|| value.clone())
}

/// The filter `column = key` over `schema`, `key` as [`key_value`] gives it
fn filter_on(
    schema: &Schema,
    column: usize,
    key: Value,
) -> crate::storage::expression::ComparisonExpr {
    let mut expr = crate::storage::expression::ComparisonExpr::new(
        schema.columns[column].name.as_str(),
        crate::core::Operator::Eq,
        key,
    );
    expr.prepare_for_schema(schema);
    expr
}

/// The position of `fk`'s column in the schema of the child handle that
/// reads the rows: the key is found there by its own column name and
/// parent, since the cached copy's position may be another layout's
fn key_position(child_schema: &Schema, fk: &ForeignKeyConstraint) -> Result<usize> {
    child_schema
        .foreign_keys
        .iter()
        .find(|key| {
            key.column_name == fk.column_name
                && key.referenced_table == fk.referenced_table
                && key.referenced_column == fk.referenced_column
        })
        .map(|key| key.column_index)
        .filter(|&at| at < child_schema.columns.len())
        .ok_or_else(|| Error::SchemaChanged {
            table: child_schema.table_name.clone(),
        })
}

/// Maximum CASCADE recursion depth to prevent infinite loops from circular FK references.
const MAX_CASCADE_DEPTH: usize = 16;

/// CASCADE DELETE: delete all child rows referencing the given parent PK value.
/// Operates within the caller's transaction (no independent commit).
/// Recursively cascades to grandchild tables (up to MAX_CASCADE_DEPTH).
fn cascade_delete(
    engine: &MVCCEngine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    parent_pk_value: &Value,
) -> Result<i32> {
    cascade_delete_recursive(engine, txn_id, child_table, fk, parent_pk_value, 0)
}

fn cascade_delete_recursive(
    engine: &MVCCEngine,
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
    check_children_seen(engine, txn_id, child_table, fk, parent_pk_value)?;

    // Before deleting child rows, read what the keys beneath the child point
    // at. Each such key names a column of the child, its primary key or
    // any other, so the child rows about to go are read through that
    // column, once per key
    let grandchild_fks = find_referencing_fks(engine, child_table);
    let mut deleted_child_keys: Vec<(usize, Vec<Value>)> = Vec::new();

    if !grandchild_fks.is_empty() {
        let child_handle = engine.get_table_to_check(txn_id, child_table)?;
        let child_schema = child_handle.schema_arc();
        // Use filtered scan instead of full table scan
        let filter = key_filter(
            &child_schema,
            key_position(&child_schema, fk)?,
            parent_pk_value,
        );
        let rows = child_handle.collect_all_rows(Some(&filter))?;
        for (gfk_idx, (_, grandchild_fk)) in grandchild_fks.iter().enumerate() {
            let referenced = grandchild_fk.referenced_column.to_lowercase();
            let Some(ref_idx) = child_schema
                .columns
                .iter()
                .position(|c| c.name_lower == referenced)
            else {
                continue;
            };
            let values: Vec<Value> = rows
                .iter()
                .filter_map(|(_, row)| row.get(ref_idx).filter(|v| !v.is_null()).cloned())
                .collect();
            if !values.is_empty() {
                deleted_child_keys.push((gfk_idx, values));
            }
        }
    }

    for (gfk_idx, values) in &deleted_child_keys {
        hold_alone(
            engine,
            txn_id,
            child_table,
            &grandchild_fks[*gfk_idx].1,
            values,
        )?;
    }

    // Pre-check: verify grandchild RESTRICT constraints BEFORE deleting child rows.
    // If we deleted children first and a grandchild RESTRICT check fails, the child
    // deletions would remain in the transaction state (orphaning data in explicit txns).
    for (gfk_idx, values) in &deleted_child_keys {
        let (grandchild_table, grandchild_fk) = &grandchild_fks[*gfk_idx];
        if !matches!(
            grandchild_fk.on_delete,
            ForeignKeyAction::Restrict | ForeignKeyAction::NoAction
        ) {
            continue;
        }
        for child_value in values {
            if child_rows_exist(engine, txn_id, grandchild_table, grandchild_fk, child_value)? {
                return Err(Error::foreign_key_violation(
                    grandchild_table,
                    &grandchild_fk.column_name,
                    child_table,
                    &grandchild_fk.referenced_column,
                    format!(
                        "cannot cascade-delete row with {} = {} — still referenced by table '{}'",
                        grandchild_fk.referenced_column, child_value, grandchild_table
                    ),
                ));
            }
        }
    }

    // Now delete the matching child rows (safe — RESTRICT checks passed above)
    let mut child = engine.get_table_for_txn(txn_id, child_table)?;
    let child_schema = child.schema();
    let expr = key_filter(
        child_schema,
        key_position(child_schema, fk)?,
        parent_pk_value,
    );

    let plan = child.index_plan(&expr);
    let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
    let count = child.delete(access, Some(&expr))?;
    // Do NOT commit here — changes stay in TransactionVersionStore and are committed
    // atomically when the parent transaction's commit_all_tables() runs.

    let mut total = count;

    // Recursively enforce CASCADE/SET NULL on grandchild tables (RESTRICT already checked above)
    for (gfk_idx, values) in &deleted_child_keys {
        let (grandchild_table, grandchild_fk) = &grandchild_fks[*gfk_idx];
        for child_value in values {
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
                        child_value,
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
                        child_value,
                    )?;
                    total = total.saturating_add(affected);
                }
            }
        }
    }

    Ok(total)
}

/// CASCADE UPDATE of every (old, new) value of `changes`, which one
/// statement made to the column `fk` references, into `child_table`.
/// Recursively cascades to grandchild tables (up to MAX_CASCADE_DEPTH).
/// Operates within the caller's transaction (no independent commit).
fn cascade_update(
    engine: &MVCCEngine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    changes: &[(Value, Value)],
) -> Result<i32> {
    cascade_update_recursive(engine, txn_id, child_table, fk, changes, 0)
}

fn cascade_update_recursive(
    engine: &MVCCEngine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    changes: &[(Value, Value)],
    depth: usize,
) -> Result<i32> {
    // The changes whose old value a child row names; the others cascade nothing
    let mut named = Vec::with_capacity(changes.len());
    for change in changes {
        if child_rows_exist(engine, txn_id, child_table, fk, &change.0)? {
            named.push(change.clone());
        }
    }
    if named.is_empty() {
        return Ok(0);
    }

    // Child rows exist. Check depth limit before doing any work.
    if depth >= MAX_CASCADE_DEPTH {
        return Err(Error::internal(format!(
            "foreign key CASCADE depth limit ({}) exceeded — possible circular reference",
            MAX_CASCADE_DEPTH
        )));
    }
    for (old_value, _) in &named {
        check_children_seen(engine, txn_id, child_table, fk, old_value)?;
    }

    // Find grandchild FKs that reference the column being updated
    let grandchild_fks = find_referencing_fks(engine, child_table);
    let child_fk_col = &fk.column_name;

    let relevant_grandchild_fks: Vec<_> = grandchild_fks
        .iter()
        .filter(|(_, gfk)| gfk.referenced_column == *child_fk_col)
        .collect();

    for (_, grandchild_fk) in &relevant_grandchild_fks {
        hold_alone(
            engine,
            txn_id,
            child_table,
            grandchild_fk,
            named.iter().map(|(old, _)| old),
        )?;
    }

    // Pre-check: verify grandchild RESTRICT constraints BEFORE updating child rows
    for (grandchild_table, grandchild_fk) in &relevant_grandchild_fks {
        if !matches!(
            grandchild_fk.on_update,
            ForeignKeyAction::Restrict | ForeignKeyAction::NoAction
        ) {
            continue;
        }
        for (old_value, _) in &named {
            if child_rows_exist(engine, txn_id, grandchild_table, grandchild_fk, old_value)? {
                return Err(Error::foreign_key_violation(
                    grandchild_table,
                    &grandchild_fk.column_name,
                    child_table,
                    &grandchild_fk.referenced_column,
                    format!(
                        "cannot cascade-update row with {} = {} — still referenced by table '{}'",
                        grandchild_fk.referenced_column, old_value, grandchild_table
                    ),
                ));
            }
        }
    }

    // Now update the matching child rows (safe — RESTRICT checks passed above)
    let mut child = engine.get_table_for_txn(txn_id, child_table)?;

    let child_schema = child.schema();
    let col_idx = key_position(child_schema, fk)?;
    let col_name = &child_schema.columns[col_idx].name;
    // The child stores the parent's value in its own column type, checked as stored
    let col_type = child_schema.columns[col_idx].data_type;
    // Each child row a change names and that change's place in `named` and
    // `new_values`, read before any is written: a row one change writes is
    // not met again by the next
    let mut targets: crate::common::I64Map<usize> = crate::common::I64Map::new();
    let mut new_values = Vec::with_capacity(named.len());
    for (at, (old_value, new_value)) in named.iter().enumerate() {
        let new_val = new_value.try_coerce_to_type(col_type)?;
        super::dml::validate_coercion(
            new_value,
            &new_val,
            col_name,
            col_type,
            child_schema.columns[col_idx].vector_dimensions,
        )?;
        if !new_val.is_null() && !matches!(new_val.compare(new_value), Ok(Ordering::Equal)) {
            return Err(Error::foreign_key_violation(
                child_table,
                &fk.column_name,
                &fk.referenced_table,
                &fk.referenced_column,
                format!(
                    "cannot cascade {} = {} into column '{}' of type {:?}",
                    fk.referenced_column, new_value, fk.column_name, col_type
                ),
            ));
        }
        let expr = key_filter(child_schema, col_idx, old_value);
        let plan = child.index_plan(&expr);
        let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
        for (id, _) in child.collect_access_rows(access, Some(&expr))?.iter() {
            targets.insert(*id, at);
        }
        new_values.push(new_val);
    }
    #[cfg(feature = "test-failpoints")]
    crate::test_failpoints::cascade_targets_read();
    let mut ids: Vec<i64> = targets.keys().collect();
    ids.sort_unstable();
    let checks =
        super::dml::checks_reading(super::dml::compile_check_programs(child_schema)?, |c| {
            c == col_idx
        });
    let mut vm = super::expression::ExprVM::new();

    // A child whose primary key is the foreign key moves to its new id, as an
    // UPDATE of its key does, all of the statement's moves at once
    let into_key = child_schema.pk_column_index() == Some(col_idx);
    let mut moves = Vec::new();
    let mut written = vec![false; named.len()];
    let count = child.update_by_row_ids(&ids, &mut |id, mut row| {
        // A row that no longer names the old value since it was read is left
        let Some(&at) = targets.get(id).filter(|&&at| {
            row.get(col_idx)
                .is_some_and(|v| matches!(v.compare(&named[at].0), Ok(Ordering::Equal)))
        }) else {
            return Ok((row, false));
        };
        written[at] = true;
        let original = into_key.then(|| row.clone());
        let _ = row.set(col_idx, new_values[at].clone());
        for check in &checks {
            super::dml::run_check(&mut vm, check, &row)?;
        }
        match original {
            Some(original) => {
                moves.push((id, row));
                Ok((original, true))
            }
            None => Ok((row, true)),
        }
    })?;
    super::dml::move_rows(&mut *child, moves)?;
    // The column's other foreign keys check each value it now holds
    for other in child.schema().foreign_keys.iter() {
        if other.column_index != col_idx || other == fk {
            continue;
        }
        for (value, _) in new_values.iter().zip(&written).filter(|(_, w)| **w) {
            if !value.is_null() {
                validate_fk_value(engine, txn_id, other, value, child_table)?;
            }
        }
    }

    let mut total = count;

    if !relevant_grandchild_fks.is_empty() && count > 0 {
        for (grandchild_table, grandchild_fk) in &relevant_grandchild_fks {
            match grandchild_fk.on_update {
                ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => {
                    // Already checked above
                }
                ForeignKeyAction::Cascade => {
                    let affected = cascade_update_recursive(
                        engine,
                        txn_id,
                        grandchild_table,
                        grandchild_fk,
                        &named,
                        depth + 1,
                    )?;
                    total = total.saturating_add(affected);
                }
                ForeignKeyAction::SetNull => {
                    // SET NULL changes the grandchild's FK column from old_value to NULL.
                    // Deeper descendants that reference the grandchild's column must see
                    // this as an update from old_value → NULL and be cascaded accordingly.
                    let grandchild_schema = engine.get_table_schema(grandchild_table)?;
                    let null_val = Value::null(
                        grandchild_schema.columns[key_position(&grandchild_schema, grandchild_fk)?]
                            .data_type,
                    );
                    let to_null: Vec<(Value, Value)> = named
                        .iter()
                        .map(|(old, _)| (old.clone(), null_val.clone()))
                        .collect();
                    let affected = cascade_update_recursive(
                        engine,
                        txn_id,
                        grandchild_table,
                        grandchild_fk,
                        &to_null,
                        depth + 1,
                    )?;
                    total = total.saturating_add(affected);
                }
            }
        }
    }

    Ok(total)
}

/// SET NULL: set FK column to NULL in all child rows referencing the given parent PK value.
/// Operates within the caller's transaction (no independent commit).
fn set_null_on_delete(
    engine: &MVCCEngine,
    txn_id: i64,
    child_table: &str,
    fk: &ForeignKeyConstraint,
    parent_pk_value: &Value,
) -> Result<i32> {
    check_children_seen(engine, txn_id, child_table, fk, parent_pk_value)?;
    let mut child = engine.get_table_for_txn(txn_id, child_table)?;

    // Check that the FK column is nullable
    let child_schema = child.schema();
    let col_idx = key_position(child_schema, fk)?;
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
    let expr = key_filter(child_schema, col_idx, parent_pk_value);
    let checks =
        super::dml::checks_reading(super::dml::compile_check_programs(child_schema)?, |c| {
            c == col_idx
        });
    let mut vm = super::expression::ExprVM::new();

    let plan = child.index_plan(&expr);
    let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
    let count = child.update(access, Some(&expr), &mut |_, mut row| {
        let _ = row.set(col_idx, null_val.clone());
        for check in &checks {
            super::dml::run_check(&mut vm, check, &row)?;
        }
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
    engine: &MVCCEngine,
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
        // The filter's positions come from the handle that reads the rows
        let references = |child: &dyn crate::storage::traits::Table| -> Result<bool> {
            let child_schema = child.schema();
            let col_name = &child_schema.columns[key_position(child_schema, fk)?].name;
            let mut not_null_expr =
                crate::storage::expression::NullCheckExpr::is_not_null(col_name.as_str());
            not_null_expr.prepare_for_schema(child_schema);
            Ok(!child
                .collect_rows_with_limit_unordered(Some(&not_null_expr), 1, 0)?
                .is_empty())
        };

        let has_ref = if let Some(tid) = txn_id {
            references(&*engine.get_table_to_check(tid, child_table)?)?
        } else {
            let tx = engine.begin_transaction()?;
            references(&*tx.get_statement_table(child_table)?)?
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
