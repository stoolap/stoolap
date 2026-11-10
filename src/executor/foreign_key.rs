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

use std::cell::Cell;
use std::cmp::Ordering;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::common::{I64Map, I64Set};
use crate::core::{Error, ForeignKeyAction, ForeignKeyConstraint, Result, Schema, Value};
use crate::storage::expression::Expression as StorageExpression;
use crate::storage::mvcc::engine::MVCCEngine;
use crate::storage::mvcc::version_store::LayoutToken;
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
    engine.hold_key(txn_id, parent_table, (column, ref_idx), &key, false)?;
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
        engine.hold_key(txn_id, parent_table, (column, at), value, true)?;
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
        .find(|key| same_key(key, fk))
        .map(|key| key.column_index)
        .filter(|&at| at < child_schema.columns.len())
        .ok_or_else(|| Error::SchemaChanged {
            table: child_schema.table_name.clone(),
        })
}

/// Whether `a` and `b` are one key: its column and the column it references
fn same_key(a: &ForeignKeyConstraint, b: &ForeignKeyConstraint) -> bool {
    a.column_name == b.column_name
        && a.referenced_table == b.referenced_table
        && a.referenced_column == b.referenced_column
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

/// Carries out the ON UPDATE actions of the changes one statement made to
/// `table`, each changed column with its (old, new) values. Every child row
/// the actions reach is found by the value it held at the statement's
/// start and written once, with the value all its keys agree on; the keys
/// of each written column are checked once every row is written.
pub(crate) fn enforce_update_actions(
    engine: &MVCCEngine,
    txn_id: i64,
    table: &str,
    changes: Vec<(usize, Vec<(Value, Value)>)>,
) -> Result<()> {
    let roots = changes.len();
    let mut columns: Vec<Changed> = changes
        .into_iter()
        .map(|(column, values)| {
            let mut root = Changed::new(table, column);
            root.done = true;
            root.values = values
                .into_iter()
                .map(|(old, new)| Taken::new(old, new, 0))
                .collect();
            root.mark_changed();
            root
        })
        .collect();
    let actions = find_update_actions(engine, &mut columns)?;
    // Each column's actions run once every action into it has
    let mut ready: Vec<usize> = (0..roots).collect();
    while let Some(at) = ready.pop() {
        if at >= roots {
            columns[at].finish()?;
        }
        for action in actions.iter().filter(|a| a.from == at) {
            run_update_action(engine, txn_id, &mut columns, action, at < roots)?;
            if let Some(to) = action.to {
                let into = &mut columns[to];
                into.waiting -= 1;
                if into.waiting == 0 && !into.done {
                    into.done = true;
                    ready.push(to);
                }
            }
        }
    }
    if columns.iter().any(|c| !c.done && !c.values.is_empty()) {
        return Err(cascade_depth_exceeded());
    }
    #[cfg(feature = "test-failpoints")]
    crate::test_failpoints::cascade_targets_read();
    write_changed(engine, txn_id, &columns[roots..])?;
    check_changed(engine, txn_id, &columns[roots..])
}

/// A column that the ON UPDATE actions of one statement change
struct Changed {
    table: String,
    column: usize,
    /// Each distinct value the column's rows held at the statement's start
    values: Vec<Taken>,
    by_value: FxHashMap<Value, usize>,
    /// Each row an action reached and the place of its start value in `values`
    rows: I64Map<usize>,
    /// The keys whose action writes into the column, each a bit of `Taken::from`
    into: Vec<ForeignKeyConstraint>,
    depth: usize,
    /// The actions into the column that have not run yet
    waiting: usize,
    done: bool,
    /// The child table's store and layout `column` was read from
    token: Option<LayoutToken>,
}

/// What one start value of a column becomes
struct Taken {
    old: Value,
    new: Value,
    /// A bit for each key of `Changed::into` whose action gave `new`
    from: u64,
    /// Whether an action gave `new`; until one does it is `old`
    given: bool,
    /// A value differing from `new` that came in too, refused unless a NULL
    /// takes the place of both
    clash: Option<Value>,
    /// Why a value that came in cannot be stored, refused unless a NULL
    /// takes its place
    refused: Option<Error>,
    changed: bool,
    written: Cell<bool>,
}

/// An ON UPDATE key of a changed column. `to` is the changed column of the
/// child it writes, None for a key that refuses the change
struct UpdateAction {
    from: usize,
    to: Option<usize>,
    /// The key's bit in the `into` of its `to`
    bit: usize,
    child_table: String,
    fk: ForeignKeyConstraint,
}

impl Taken {
    fn new(old: Value, new: Value, from: u64) -> Self {
        Taken {
            old,
            new,
            from,
            given: true,
            clash: None,
            refused: None,
            changed: false,
            written: Cell::new(false),
        }
    }
}

impl Changed {
    fn new(table: &str, column: usize) -> Self {
        Changed {
            table: table.to_string(),
            column,
            values: Vec::new(),
            by_value: FxHashMap::default(),
            rows: I64Map::new(),
            into: Vec::new(),
            depth: 0,
            waiting: 0,
            done: false,
            token: None,
        }
    }

    fn mark_changed(&mut self) {
        for taken in &mut self.values {
            taken.changed = taken.new.is_null() || !same_value(&taken.new, &taken.old);
        }
    }

    /// Settles the column once every action into it has run
    fn finish(&mut self) -> Result<()> {
        if !self.values.is_empty() && self.depth > MAX_CASCADE_DEPTH {
            return Err(cascade_depth_exceeded());
        }
        for taken in &mut self.values {
            if taken.new.is_null() {
                continue;
            }
            if let Some(error) = taken.refused.take() {
                return Err(error);
            }
            if let (Some(clash), Some(fk)) = (&taken.clash, self.into.first()) {
                return Err(Error::foreign_key_violation(
                    &self.table,
                    &fk.column_name,
                    &fk.referenced_table,
                    &fk.referenced_column,
                    format!(
                        "cannot cascade both {} and {} into column '{}'",
                        taken.new, clash, fk.column_name
                    ),
                ));
            }
        }
        self.mark_changed();
        Ok(())
    }

    /// Records that `action` gives `value` to the row `row`, which held
    /// `start`. A NULL, no reference, takes the place of any value; two
    /// values that differ, or one the column cannot store, are refused by
    /// `finish` unless a NULL takes their place
    fn take(&mut self, row: i64, start: &Value, value: &Result<Value>, action: &UpdateAction) {
        let bit = if action.bit < 64 { 1 << action.bit } else { 0 };
        let at = match self.by_value.get(start) {
            Some(&at) => at,
            None => {
                // Until a value comes, the row keeps its own
                let mut taken = Taken::new(start.clone(), start.clone(), 0);
                taken.given = false;
                self.values.push(taken);
                self.by_value.insert(start.clone(), self.values.len() - 1);
                self.values.len() - 1
            }
        };
        let taken = &mut self.values[at];
        match value {
            _ if taken.given && taken.new.is_null() => {}
            Err(error) => {
                if taken.refused.is_none() {
                    taken.refused = Some(error.clone());
                }
            }
            Ok(value) if value.is_null() || !taken.given => {
                taken.new = value.clone();
                taken.given = true;
                taken.from |= bit;
            }
            Ok(value) => {
                if !same_value(&taken.new, value) {
                    taken.clash = Some(value.clone());
                }
                taken.from |= bit;
            }
        }
        self.rows.insert(row, at);
    }
}

fn same_value(a: &Value, b: &Value) -> bool {
    matches!(a.compare(b), Ok(Ordering::Equal))
}

fn cascade_depth_exceeded() -> Error {
    Error::internal(format!(
        "foreign key CASCADE depth limit ({}) exceeded — possible circular reference",
        MAX_CASCADE_DEPTH
    ))
}

/// Every ON UPDATE key beneath the changed columns of `columns`, adding
/// to it each child column a CASCADE or SET NULL key writes
fn find_update_actions(
    engine: &MVCCEngine,
    columns: &mut Vec<Changed>,
) -> Result<Vec<UpdateAction>> {
    let mut actions = Vec::new();
    let mut next = 0;
    while next < columns.len() {
        let table = columns[next].table.clone();
        let column = columns[next].column;
        let schema = engine.get_table_schema(&table)?;
        for (child_table, fk) in find_referencing_fks(engine, &table).iter() {
            if referenced_position(&schema, &table, &fk.referenced_column).ok() != Some(column) {
                continue;
            }
            let (to, bit) = match fk.on_update {
                ForeignKeyAction::Restrict | ForeignKeyAction::NoAction => (None, 0),
                ForeignKeyAction::Cascade | ForeignKeyAction::SetNull => {
                    let child_schema = engine.get_table_schema(child_table)?;
                    let child_column = key_position(&child_schema, fk)?;
                    let at = match columns
                        .iter()
                        .position(|c| c.column == child_column && c.table == *child_table)
                    {
                        Some(at) => at,
                        None => {
                            columns.push(Changed::new(child_table, child_column));
                            columns.len() - 1
                        }
                    };
                    let into = &mut columns[at];
                    into.into.push(fk.clone());
                    into.waiting += 1;
                    (Some(at), into.into.len() - 1)
                }
            };
            actions.push(UpdateAction {
                from: next,
                to,
                bit,
                child_table: child_table.clone(),
                fk: fk.clone(),
            });
        }
        next += 1;
    }
    Ok(actions)
}

/// Runs `action` over the changed values of its column, recording what each
/// child row it reaches takes; a RESTRICT key refuses a change a row names
fn run_update_action(
    engine: &MVCCEngine,
    txn_id: i64,
    columns: &mut [Changed],
    action: &UpdateAction,
    from_root: bool,
) -> Result<()> {
    let fk = &action.fk;
    let child_table = action.child_table.as_str();
    let from = &columns[action.from];
    let changes: Vec<(Value, Value)> = from
        .values
        .iter()
        .filter(|t| t.changed)
        .map(|t| (t.old.clone(), t.new.clone()))
        .collect();
    if changes.is_empty() {
        return Ok(());
    }
    let depth = from.depth + 1;
    hold_alone(
        engine,
        txn_id,
        &fk.referenced_table,
        fk,
        changes.iter().map(|(old, _)| old),
    )?;
    let Some(to) = action.to else {
        // The root's own keys were asked before its rows were written
        if from_root {
            return Ok(());
        }
        for (old_value, _) in &changes {
            if child_rows_exist(engine, txn_id, child_table, fk, old_value)? {
                return Err(Error::foreign_key_violation(
                    child_table,
                    &fk.column_name,
                    &fk.referenced_table,
                    &fk.referenced_column,
                    format!(
                        "cannot cascade-update row with {} = {} — still referenced by table '{}'",
                        fk.referenced_column, old_value, child_table
                    ),
                ));
            }
        }
        return Ok(());
    };

    let child = engine.get_table_for_txn(txn_id, child_table)?;
    let child_schema = child.schema();
    let col_idx = key_position(child_schema, fk)?;
    // Every read of the column and its write are of one layout
    let token = child.layout_token();
    if col_idx != columns[to].column || columns[to].token.is_some_and(|t| t != token) {
        return Err(Error::SchemaChanged {
            table: child_table.to_string(),
        });
    }
    columns[to].token = Some(token);
    let col_name = &child_schema.columns[col_idx].name;
    // The child stores the parent's value in its own column type, checked as stored
    let col_type = child_schema.columns[col_idx].data_type;
    let set_null = fk.on_update == ForeignKeyAction::SetNull;
    let mut reached = false;
    for (old_value, new_value) in &changes {
        if !child_rows_exist(engine, txn_id, child_table, fk, old_value)? {
            continue;
        }
        check_children_seen(engine, txn_id, child_table, fk, old_value)?;
        let new_val = if set_null {
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
            Ok(Value::null(col_type))
        } else {
            // A value the column cannot store is refused only if it is the one kept
            new_value.try_coerce_to_type(col_type).and_then(|new_val| {
                super::dml::validate_coercion(
                    new_value,
                    &new_val,
                    col_name,
                    col_type,
                    child_schema.columns[col_idx].vector_dimensions,
                )?;
                if !new_val.is_null() && !same_value(&new_val, new_value) {
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
                Ok(new_val)
            })
        };
        let expr = key_filter(child_schema, col_idx, old_value);
        let plan = child.index_plan(&expr);
        let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
        for (id, row) in child.collect_access_rows(access, Some(&expr))?.iter() {
            if let Some(start) = row.get(col_idx) {
                columns[to].take(*id, start, &new_val, action);
                reached = true;
            }
        }
    }
    if reached {
        let into = &mut columns[to];
        // A column whose actions all ran already is reached again only round a cycle
        if into.done {
            return Err(cascade_depth_exceeded());
        }
        into.depth = into.depth.max(depth);
    }
    Ok(())
}

/// Writes every child row of `columns` once, with all the values its
/// columns take. A row that no longer holds a column's start value keeps
/// that column; a row whose primary key changes moves to its new id
fn write_changed(engine: &MVCCEngine, txn_id: i64, columns: &[Changed]) -> Result<()> {
    let mut tables: Vec<&str> = Vec::new();
    for column in columns {
        if column.values.iter().any(|t| t.changed) && !tables.contains(&column.table.as_str()) {
            tables.push(&column.table);
        }
    }
    for table in tables {
        let of_table: Vec<&Changed> = columns.iter().filter(|c| c.table == table).collect();
        let mut ids: Vec<i64> = Vec::new();
        for column in &of_table {
            ids.extend(
                column
                    .rows
                    .iter()
                    .filter(|(_, &at)| column.values[at].changed)
                    .map(|(id, _)| id),
            );
        }
        ids.sort_unstable();
        ids.dedup();
        let mut child = engine.get_table_for_txn(txn_id, table)?;
        for column in &of_table {
            if let Some(token) = column.token {
                super::query_cache::check_compiled_token(&*child, token)?;
            }
        }
        let schema = child.schema();
        let checks = super::dml::checks_reading(super::dml::compile_check_programs(schema)?, |c| {
            of_table.iter().any(|column| column.column == c)
        });
        let pk = schema.pk_column_index();
        let mut vm = super::expression::ExprVM::new();
        let mut moves = Vec::new();
        let mut apply: Vec<(usize, &Taken)> = Vec::new();
        child.update_by_row_ids(&ids, &mut |id, mut row| {
            apply.clear();
            for column in &of_table {
                let Some(&at) = column.rows.get(id) else {
                    continue;
                };
                let taken = &column.values[at];
                if taken.changed
                    && row
                        .get(column.column)
                        .is_some_and(|v| same_value(v, &taken.old))
                {
                    apply.push((column.column, taken));
                }
            }
            if apply.is_empty() {
                return Ok((row, false));
            }
            let original = apply
                .iter()
                .any(|(at, _)| Some(*at) == pk)
                .then(|| row.clone());
            for (at, taken) in &apply {
                let _ = row.set(*at, taken.new.clone());
                taken.written.set(true);
            }
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
    }
    Ok(())
}

/// Checks each value written into `columns` against every key of its
/// column whose own action did not give it
fn check_changed(engine: &MVCCEngine, txn_id: i64, columns: &[Changed]) -> Result<()> {
    let checked = |t: &Taken| t.changed && t.written.get() && !t.new.is_null();
    for column in columns {
        if !column.values.iter().any(checked) {
            continue;
        }
        let schema = engine.get_table_schema(&column.table)?;
        for fk in schema
            .foreign_keys
            .iter()
            .filter(|k| k.column_index == column.column)
        {
            let bit = column.into.iter().position(|k| same_key(k, fk));
            for taken in column.values.iter().filter(|t| checked(t)) {
                if bit.is_some_and(|b| b < 64 && taken.from & (1 << b) != 0) {
                    continue;
                }
                validate_fk_value(engine, txn_id, fk, &taken.new, &column.table)?;
            }
        }
    }
    Ok(())
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
