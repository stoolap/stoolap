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

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::common::{I64Map, I64Set};
use crate::core::{Error, ForeignKeyAction, ForeignKeyConstraint, Result, Schema, Value};
use crate::storage::expression::{Expression as StorageExpression, InListExpr, OrExpr};
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
    check_parents(engine, txn_id, schema, row, None)
}

/// The keys into the table a statement inserts into, `table` in lower
/// case, that found no parent row yet: a later row of the statement may be
/// it. Each value waits under its child table and key, kept once in
/// `keys`, and is asked by [`PendingParents::check`] once the rows are written
pub(crate) struct PendingParents<'t> {
    table: &'t str,
    keys: Vec<(String, ForeignKeyConstraint)>,
    values: Vec<(usize, Value)>,
}

impl<'t> PendingParents<'t> {
    pub(crate) fn new(table: &'t str) -> Self {
        PendingParents {
            table,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Checks that `value` of the key `fk` of `child_table` names a parent
    /// row, keeping it for later when the parent is the statement's table
    fn ask(
        &mut self,
        engine: &MVCCEngine,
        txn_id: i64,
        child_table: &str,
        fk: &ForeignKeyConstraint,
        value: &Value,
    ) -> Result<()> {
        if parent_row_exists(
            engine,
            txn_id,
            &fk.referenced_table,
            &fk.referenced_column,
            value,
        )? {
            return Ok(());
        }
        if fk.referenced_table != self.table {
            return Err(missing_parent(child_table, fk, value));
        }
        let known = self
            .keys
            .iter()
            .position(|(table, key)| table == child_table && same_key(key, fk));
        let at = known.unwrap_or_else(|| {
            self.keys.push((child_table.to_string(), fk.clone()));
            self.keys.len() - 1
        });
        self.values.push((at, value.clone()));
        Ok(())
    }

    /// Asks again each key left pending, now that the statement's rows are
    /// written. A value no row holds any more, which a later update of the
    /// statement replaced, names nothing
    pub(crate) fn check(&self, engine: &MVCCEngine, txn_id: i64) -> Result<()> {
        for (at, value) in &self.values {
            let (child_table, fk) = &self.keys[*at];
            if !parent_row_exists(
                engine,
                txn_id,
                &fk.referenced_table,
                &fk.referenced_column,
                value,
            )? && child_rows_exist(engine, txn_id, child_table, fk, value)?
            {
                return Err(missing_parent(child_table, fk, value));
            }
        }
        Ok(())
    }
}

/// [`check_parent_exists`] for a row an INSERT or the update of an upsert
/// writes, leaving a key into the statement's own table to `pending`
pub(crate) fn check_inserted_parents(
    engine: &MVCCEngine,
    txn_id: i64,
    schema: &Schema,
    row: &crate::core::Row,
    pending: &mut PendingParents<'_>,
) -> Result<()> {
    check_parents(engine, txn_id, schema, row, Some(pending))
}

fn check_parents(
    engine: &MVCCEngine,
    txn_id: i64,
    schema: &Schema,
    row: &crate::core::Row,
    mut pending: Option<&mut PendingParents<'_>>,
) -> Result<()> {
    for fk in &schema.foreign_keys {
        let fk_value = match row.get(fk.column_index) {
            Some(v) if !v.is_null() => v,
            _ => continue, // NULL FK is allowed (no reference)
        };
        match pending.as_deref_mut() {
            Some(pending) => pending.ask(engine, txn_id, &schema.table_name, fk, fk_value)?,
            None => validate_fk_value(engine, txn_id, fk, fk_value, &schema.table_name)?,
        }
    }
    Ok(())
}

fn missing_parent(child_table: &str, fk: &ForeignKeyConstraint, value: &Value) -> Error {
    Error::foreign_key_violation(
        child_table,
        &fk.column_name,
        &fk.referenced_table,
        &fk.referenced_column,
        format!(
            "referenced row with {} = {} does not exist",
            fk.referenced_column, value
        ),
    )
}

/// Check a single FK value against its parent table.
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
        return Err(missing_parent(child_table, fk, value));
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
    // A snapshot's handle took its view before the hold, so it is taken again,
    // and a drop meanwhile that moved the column has its new position held
    if engine.reads_snapshot(txn_id) {
        loop {
            parent = open()?;
            parent_schema = parent.schema_arc();
            let at = referenced_position(&parent_schema, parent_table, parent_column)?;
            if at == ref_idx {
                break;
            }
            ref_idx = at;
            let column = &parent_schema.columns[ref_idx].name;
            engine.hold_key(txn_id, parent_table, (column, ref_idx), &key, false)?;
        }
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

/// What the ON DELETE keys of one delete do, worked out from the rows as
/// the statement found them before anything is written. A row goes once
/// whichever keys reach it, and a restricting key or a SET NULL asks only
/// the rows that stay
pub(crate) struct DeletePlan {
    tables: Vec<Reached>,
}

/// A table the keys of a delete reach
struct Reached {
    table: String,
    /// Every row of the table that goes, the statement's own among them
    gone: I64Set,
    /// The rows a cascade removes, beside the statement's own
    cascaded: Vec<i64>,
    /// Each cascading key into the table and the values whose rows go
    cascades: Vec<(ForeignKeyConstraint, Vec<Value>)>,
    /// Each SET NULL key into the table and the values it reached
    null_keys: Vec<(ForeignKeyConstraint, Vec<Value>)>,
    /// Each row a SET NULL key reached: the key and the value, by place
    nulled: Vec<(i64, usize, usize)>,
    /// Each restricting key into the table and the values it names
    restricted: Vec<(ForeignKeyConstraint, Vec<Value>)>,
    /// Whether the cascades' rows are left unread: no key references the
    /// table, so `gone` is needed only by a SET NULL or RESTRICT key of its own
    unread: bool,
}

/// Rows going from `table` whose keys are still to be followed: each key
/// into the table with the values its column held in those rows
struct Going {
    table: String,
    keys: Arc<Vec<(String, ForeignKeyConstraint)>>,
    values: Vec<Vec<Value>>,
}

impl Reached {
    fn new(table: &str) -> Self {
        Reached {
            table: table.to_string(),
            gone: I64Set::new(),
            cascaded: Vec::new(),
            cascades: Vec::new(),
            null_keys: Vec::new(),
            nulled: Vec::new(),
            restricted: Vec::new(),
            unread: false,
        }
    }
}

/// Works out what the ON DELETE keys do to a delete of the rows `ids` of
/// `table`, each of `keys` with the values its column held in those rows.
/// A refusal comes here, before any row goes
pub(crate) fn plan_delete_actions(
    engine: &MVCCEngine,
    txn_id: i64,
    table: &str,
    ids: &[i64],
    keys: Arc<Vec<(String, ForeignKeyConstraint)>>,
    values: Vec<Vec<Value>>,
) -> Result<DeletePlan> {
    let mut root = Reached::new(table);
    for &id in ids {
        root.gone.insert(id);
    }
    let mut plan = DeletePlan { tables: vec![root] };
    let mut work = vec![Going {
        table: table.to_string(),
        keys,
        values,
    }];
    while let Some(going) = work.pop() {
        for ((child, fk), mut values) in going.keys.iter().zip(going.values) {
            values.retain(|v| !v.is_null());
            if values.is_empty() {
                continue;
            }
            hold_alone(engine, txn_id, &going.table, fk, &values)?;
            let at = plan.reached(child);
            if matches!(
                fk.on_delete,
                ForeignKeyAction::Restrict | ForeignKeyAction::NoAction
            ) {
                plan.tables[at].restricted.push((fk.clone(), values));
                continue;
            }
            if let Some(next) = plan.follow(engine, txn_id, at, fk, values)? {
                work.push(next);
            }
        }
    }
    plan.check(engine, txn_id)?;
    Ok(plan)
}

impl DeletePlan {
    fn reached(&mut self, table: &str) -> usize {
        match self.tables.iter().position(|r| r.table == table) {
            Some(at) => at,
            None => {
                self.tables.push(Reached::new(table));
                self.tables.len() - 1
            }
        }
    }

    /// Reads the rows of the table at `at` that the CASCADE or SET NULL key
    /// `fk` finds for each of `values`, recording what the key does to them.
    /// The rows a cascade removes come back, their own keys to be followed
    fn follow(
        &mut self,
        engine: &MVCCEngine,
        txn_id: i64,
        at: usize,
        fk: &ForeignKeyConstraint,
        values: Vec<Value>,
    ) -> Result<Option<Going>> {
        let reached = &mut self.tables[at];
        let child = engine.get_table_to_check(txn_id, &reached.table)?;
        let schema = child.schema_arc();
        let column = key_position(&schema, fk)?;
        let cascade = fk.on_delete == ForeignKeyAction::Cascade;
        let keys = cascade.then(|| find_referencing_fks(engine, &reached.table));
        if keys.as_ref().is_some_and(|keys| keys.is_empty()) {
            for value in &values {
                check_children_seen(engine, txn_id, &reached.table, fk, value)?;
            }
            reached.unread = true;
            reached.cascades.push((fk.clone(), values));
            return Ok(None);
        }
        let positions: Vec<Option<usize>> = keys
            .iter()
            .flat_map(|keys| keys.iter())
            .map(|(_, key)| {
                referenced_position(&schema, &reached.table, &key.referenced_column).ok()
            })
            .collect();
        let mut next: Vec<Vec<Value>> = vec![Vec::new(); positions.len()];
        let mut reaching = Vec::new();
        for value in values {
            check_children_seen(engine, txn_id, &reached.table, fk, &value)?;
            let expr = key_filter(&schema, column, &value);
            let plan = child.index_plan(&expr);
            let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
            let rows = child.collect_access_rows(access, Some(&expr))?;
            if rows.is_empty() {
                continue;
            }
            for (id, row) in rows.iter() {
                if !cascade {
                    reached
                        .nulled
                        .push((*id, reached.null_keys.len(), reaching.len()));
                } else if reached.gone.insert(*id) {
                    reached.cascaded.push(*id);
                    for (values, position) in next.iter_mut().zip(&positions) {
                        let held = position.and_then(|p| row.get(p));
                        if let Some(held) = held.filter(|v| !v.is_null()) {
                            values.push(held.clone());
                        }
                    }
                }
            }
            reaching.push(value);
        }
        if reaching.is_empty() {
            return Ok(None);
        }
        if !cascade {
            reached.null_keys.push((fk.clone(), reaching));
            return Ok(None);
        }
        reached.cascades.push((fk.clone(), reaching));
        Ok(keys
            .filter(|_| next.iter().any(|v| !v.is_empty()))
            .map(|keys| Going {
                table: reached.table.clone(),
                keys,
                values: next,
            }))
    }

    /// Refuses a restricting key that a row staying names, and a SET NULL
    /// into a column that cannot be NULL of a row that stays
    fn check(&mut self, engine: &MVCCEngine, txn_id: i64) -> Result<()> {
        for reached in &mut self.tables {
            if reached.unread && !(reached.nulled.is_empty() && reached.restricted.is_empty()) {
                reached.read_cascaded(engine, txn_id)?;
            }
            let reached = &*reached;
            for (fk, values) in &reached.restricted {
                for value in values {
                    if reached.names_a_staying_row(engine, txn_id, fk, value)? {
                        return Err(Error::foreign_key_violation(
                            &reached.table,
                            &fk.column_name,
                            &fk.referenced_table,
                            &fk.referenced_column,
                            format!(
                                "cannot delete row with {} = {} — still referenced by table '{}'",
                                fk.referenced_column, value, reached.table
                            ),
                        ));
                    }
                }
            }
            let stays = |key: usize| {
                reached
                    .nulled
                    .iter()
                    .any(|&(id, k, _)| k == key && !reached.gone.contains(id))
            };
            for (key, (fk, _)) in reached.null_keys.iter().enumerate() {
                let schema = engine.get_table_schema(&reached.table)?;
                if !schema.columns[key_position(&schema, fk)?].nullable && stays(key) {
                    return Err(Error::foreign_key_violation(
                        &reached.table,
                        &fk.column_name,
                        &fk.referenced_table,
                        &fk.referenced_column,
                        format!(
                            "cannot SET NULL on non-nullable column '{}'",
                            fk.column_name
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Removes the rows the cascades reach and clears the columns the SET
    /// NULL keys reach, once the statement's own rows are gone
    pub(crate) fn write(self, engine: &MVCCEngine, txn_id: i64) -> Result<()> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::cascade_targets_read();
        for reached in &self.tables {
            if !reached.cascades.is_empty() {
                reached.delete_cascaded(engine, txn_id)?;
            }
            reached.clear_nulled(engine, txn_id)?;
        }
        Ok(())
    }
}

impl Reached {
    /// Reads the rows the cascades into the table left unread
    fn read_cascaded(&mut self, engine: &MVCCEngine, txn_id: i64) -> Result<()> {
        let child = engine.get_table_to_check(txn_id, &self.table)?;
        let schema = child.schema();
        for (fk, values) in &self.cascades {
            let column = key_position(schema, fk)?;
            for value in values {
                let expr = key_filter(schema, column, value);
                let plan = child.index_plan(&expr);
                let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
                for (id, _) in child.collect_access_rows(access, Some(&expr))?.iter() {
                    if self.gone.insert(*id) {
                        self.cascaded.push(*id);
                    }
                }
            }
        }
        self.unread = false;
        Ok(())
    }

    /// Whether a row of the table that `fk` finds for `value` stays
    fn names_a_staying_row(
        &self,
        engine: &MVCCEngine,
        txn_id: i64,
        fk: &ForeignKeyConstraint,
        value: &Value,
    ) -> Result<bool> {
        if self.gone.is_empty() {
            return child_rows_exist(engine, txn_id, &self.table, fk, value);
        }
        let child = engine.get_table_to_check(txn_id, &self.table)?;
        let schema = child.schema();
        let expr = key_filter(schema, key_position(schema, fk)?, value);
        let plan = child.index_plan(&expr);
        let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
        // One row past the ones that go is one that stays
        let limit = self.gone.len() + 1;
        let rows = child.collect_access_rows_with_limit_unordered(access, Some(&expr), limit, 0)?;
        let stays = rows.iter().any(|(id, _)| !self.gone.contains(*id));
        Ok(stays)
    }

    /// Deletes the rows every cascade into the table reaches in one call. A
    /// planned row that changed since it was read and stays refuses
    fn delete_cascaded(&self, engine: &MVCCEngine, txn_id: i64) -> Result<()> {
        let mut child = engine.get_table_for_txn(txn_id, &self.table)?;
        let schema = child.schema_arc();
        let mut filters: Vec<Box<dyn StorageExpression>> = Vec::with_capacity(self.cascades.len());
        for (fk, values) in &self.cascades {
            let column = key_position(&schema, fk)?;
            let keys = values
                .iter()
                .map(|v| key_value(&schema, column, v))
                .collect();
            let mut expr = InListExpr::new(schema.columns[column].name.as_str(), keys);
            expr.prepare_for_schema(&schema);
            filters.push(Box::new(expr));
        }
        let expr: Box<dyn StorageExpression> = if filters.len() == 1 {
            filters.remove(0)
        } else {
            let mut expr = OrExpr::new(filters);
            expr.prepare_for_schema(&schema);
            Box::new(expr)
        };
        let plan = child.index_plan(&*expr);
        let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
        let deleted = child.delete(access, Some(&*expr))?;
        // A table no key references has no rows beneath its own to protect
        if self.unread || deleted as usize == self.cascaded.len() {
            return Ok(());
        }
        let seen = engine.get_table_to_check(txn_id, &self.table)?;
        match seen.collect_rows_by_ids(&self.cascaded)?.iter().next() {
            Some((id, _)) => Err(Error::internal(format!(
                "write conflict: row {id} of table '{}' changed while the delete read it",
                self.table
            ))),
            None => Ok(()),
        }
    }

    /// Clears each column a SET NULL key reached in the rows that stay. A
    /// row that no longer holds the value the key found keeps its column
    fn clear_nulled(&self, engine: &MVCCEngine, txn_id: i64) -> Result<()> {
        let mut rows: Vec<(i64, usize, usize)> = self
            .nulled
            .iter()
            .copied()
            .filter(|&(id, ..)| !self.gone.contains(id))
            .collect();
        if rows.is_empty() {
            return Ok(());
        }
        rows.sort_unstable();
        let mut ids: Vec<i64> = rows.iter().map(|r| r.0).collect();
        ids.dedup();
        let mut child = engine.get_table_for_txn(txn_id, &self.table)?;
        let schema = child.schema_arc();
        let columns = self
            .null_keys
            .iter()
            .map(|(fk, _)| key_position(&schema, fk))
            .collect::<Result<Vec<usize>>>()?;
        let checks =
            super::dml::checks_reading(super::dml::compile_check_programs(&schema)?, |c| {
                columns.contains(&c)
            });
        let mut vm = super::expression::ExprVM::new();
        child.update_by_row_ids(&ids, &mut |id, mut row| {
            let from = rows.partition_point(|r| r.0 < id);
            let mut changed = false;
            for &(_, key, value) in rows[from..].iter().take_while(|r| r.0 == id) {
                let column = columns[key];
                let start = &self.null_keys[key].1[value];
                if row.get(column).is_some_and(|v| same_value(v, start)) {
                    let _ = row.set(column, Value::null(schema.columns[column].data_type));
                    changed = true;
                }
            }
            if changed {
                for check in &checks {
                    super::dml::run_check(&mut vm, check, &row)?;
                }
            }
            Ok((row, changed))
        })?;
        Ok(())
    }
}

/// Checks RESTRICT constraints and CASCADE depth for one value the statement
/// changed, after its rows are written and before any child row is. Walks the
/// full FK tree, grandchild RESTRICT behind CASCADE/SET NULL included
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

/// Carries out the ON UPDATE actions of the changes one statement made to
/// `table`, each changed column with its (old, new) values. Every child row
/// the actions reach is found by the value it held at the statement's
/// start and written once, with the value all its keys agree on; the keys
/// of each written column are checked once every row is written.
pub(crate) fn enforce_update_actions(
    engine: &MVCCEngine,
    txn_id: i64,
    table: &str,
    mut changes: Vec<(usize, Vec<(Value, Value)>)>,
    pending: Option<&mut PendingParents<'_>>,
    plans: &mut UpdatePlans,
) -> Result<()> {
    changes.sort_unstable_by_key(|(column, _)| *column);
    let roots = changes.len();
    let plan = plans.find(engine, table, changes.iter().map(|(column, _)| *column))?;
    let mut columns: Vec<Changed> = plan.shapes.iter().map(Changed::new).collect();
    for (root, (_, values)) in columns.iter_mut().zip(changes) {
        root.done = true;
        root.values = values
            .into_iter()
            .map(|(old, new)| Taken::new(old, new, 0))
            .collect();
        root.mark_changed();
    }
    for at in roots..columns.len() {
        let Some(root) = columns[at].shape.shadows else {
            continue;
        };
        if columns[root].values.len() > WRITTEN_INLINE {
            let written = columns[root].values.iter().map(|t| t.new.clone()).collect();
            columns[at].written = written;
        }
    }
    let actions = &plan.actions;
    // Each column's actions run once every action into it has
    let mut ready: Vec<usize> = (0..roots).collect();
    while let Some(at) = ready.pop() {
        if at >= roots {
            columns[at].finish()?;
        }
        for action in actions.iter().filter(|a| a.from == at) {
            run_update_action(engine, txn_id, &mut columns, action, at < roots)?;
            if let Some(to) = action.to.filter(|_| action.counted) {
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
    check_changed(engine, txn_id, &columns[roots..], pending)
}

/// The ON UPDATE keys beneath the columns a statement changes in a table,
/// which depend on the schema alone: each column a key writes, and each key
pub(crate) struct UpdatePlan {
    /// The changed columns it starts from, in order
    roots: SmallVec<[usize; 2]>,
    shapes: Vec<Shape>,
    actions: Vec<UpdateAction>,
}

/// The plans one statement found for its target, by the columns they start
/// from: an upsert finds them once, not for each conflict
#[derive(Default)]
pub(crate) struct UpdatePlans(SmallVec<[UpdatePlan; 1]>);

impl UpdatePlans {
    fn find(
        &mut self,
        engine: &MVCCEngine,
        table: &str,
        roots: impl Iterator<Item = usize> + Clone,
    ) -> Result<&UpdatePlan> {
        let found = self
            .0
            .iter()
            .position(|plan| plan.roots.iter().copied().eq(roots.clone()));
        let at = match found {
            Some(at) => at,
            None => {
                self.0
                    .push(plan_update_actions(engine, table, roots.collect())?);
                self.0.len() - 1
            }
        };
        Ok(&self.0[at])
    }
}

/// A column the keys of a plan start from or write
struct Shape {
    table: String,
    column: usize,
    /// The keys whose action writes into the column, each a bit of `Taken::from`
    into: Vec<ForeignKeyConstraint>,
    /// The root the statement changed this same column in: rows holding
    /// one of its new values are the statement's writes, which no key reaches
    shadows: Option<usize>,
    /// The keys into the column it waits for
    waiting: usize,
}

/// A column that the ON UPDATE actions of one statement change
struct Changed<'p> {
    shape: &'p Shape,
    /// Each distinct value the column's rows held at the statement's start
    values: Vec<Taken>,
    by_value: FxHashMap<Value, usize>,
    /// Each row an action reached and the place of its start value in `values`
    rows: I64Map<usize>,
    depth: usize,
    /// The actions into the column that have not run yet
    waiting: usize,
    done: bool,
    /// The child table's store and layout `column` was read from
    token: Option<LayoutToken>,
    /// The shadowed root's new values when they are too many to look through
    written: FxHashSet<Value>,
}

/// Up to this many values a statement wrote are looked through, not hashed
const WRITTEN_INLINE: usize = 8;

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
    /// Whether `to` waits for it: every key but one round a cycle
    counted: bool,
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

impl<'p> Changed<'p> {
    fn new(shape: &'p Shape) -> Self {
        Changed {
            shape,
            values: Vec::new(),
            by_value: FxHashMap::default(),
            rows: I64Map::new(),
            depth: 0,
            waiting: shape.waiting,
            done: false,
            token: None,
            written: FxHashSet::default(),
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
            if let (Some(clash), Some(fk)) = (&taken.clash, self.shape.into.first()) {
                return Err(Error::foreign_key_violation(
                    &self.shape.table,
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

/// The plan of every ON UPDATE key beneath the changed columns `roots` of
/// `table`, with each child column a CASCADE or SET NULL key writes
fn plan_update_actions(
    engine: &MVCCEngine,
    table: &str,
    roots: SmallVec<[usize; 2]>,
) -> Result<UpdatePlan> {
    let shape = |table: &str, column: usize| Shape {
        table: table.to_string(),
        column,
        into: Vec::new(),
        shadows: None,
        waiting: 0,
    };
    let mut columns: Vec<Shape> = roots.iter().map(|&column| shape(table, column)).collect();
    let roots_len = roots.len();
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
                    let same = |c: &Shape| c.column == child_column && c.table == *child_table;
                    // A column the statement changed has its own values: a key
                    // into it writes a column of its own
                    let at = match columns[roots_len..].iter().position(same) {
                        Some(at) => roots_len + at,
                        None => {
                            let mut into = shape(child_table, child_column);
                            into.shadows = columns[..roots_len].iter().position(same);
                            columns.push(into);
                            columns.len() - 1
                        }
                    };
                    let into = &mut columns[at];
                    into.into.push(fk.clone());
                    (Some(at), into.into.len() - 1)
                }
            };
            actions.push(UpdateAction {
                from: next,
                to,
                bit,
                counted: false,
                child_table: child_table.clone(),
                fk: fk.clone(),
            });
        }
        next += 1;
    }
    count_waits(&mut actions, &mut columns, roots_len);
    Ok(UpdatePlan {
        roots,
        shapes: columns,
        actions,
    })
}

/// Makes each column wait for the keys into it a walk from the roots meets
/// before the column, every key but one leading back round a cycle; such a
/// key is not waited for, and reaching a row through it is refused
fn count_waits(actions: &mut [UpdateAction], columns: &mut [Shape], roots: usize) {
    // Columns are found in order, so each one's keys are one run of
    // `actions`; a graph of up to eight columns walks without the heap
    let mut first: SmallVec<[usize; 9]> = smallvec::smallvec![actions.len(); columns.len() + 1];
    for (at, action) in actions.iter().enumerate().rev() {
        first[action.from] = at;
    }
    for column in (0..columns.len()).rev() {
        first[column] = first[column].min(first[column + 1]);
    }
    const NEW: u8 = 0;
    const OPEN: u8 = 1;
    const DONE: u8 = 2;
    let mut state: SmallVec<[u8; 8]> = smallvec::smallvec![NEW; columns.len()];
    let mut walk: SmallVec<[(usize, usize); 8]> = SmallVec::new();
    for root in 0..roots {
        state[root] = OPEN;
        walk.push((root, first[root]));
        while let Some((column, at)) = walk.last_mut() {
            let column = *column;
            if *at == first[column + 1] {
                state[column] = DONE;
                walk.pop();
                continue;
            }
            let action = *at;
            *at += 1;
            let Some(to) = actions[action].to else {
                continue;
            };
            if state[to] == OPEN {
                continue;
            }
            actions[action].counted = true;
            columns[to].waiting += 1;
            if state[to] == NEW {
                state[to] = OPEN;
                walk.push((to, first[to]));
            }
        }
    }
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
        // The root's own keys were asked before the actions ran
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
    if col_idx != columns[to].shape.column || columns[to].token.is_some_and(|t| t != token) {
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
        // A row holding a value the statement wrote into the column is that write
        let written = columns[to].shape.shadows.is_some_and(|root| {
            match columns[root].values.len() > WRITTEN_INLINE {
                true => {
                    (columns[to].written).contains(&key_value(child_schema, col_idx, old_value))
                }
                false => (columns[root].values.iter()).any(|t| same_value(&t.new, old_value)),
            }
        });
        if written || !child_rows_exist(engine, txn_id, child_table, fk, old_value)? {
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
        let table = column.shape.table.as_str();
        if column.values.iter().any(|t| t.changed) && !tables.contains(&table) {
            tables.push(table);
        }
    }
    for table in tables {
        let of_table: Vec<&Changed> = columns.iter().filter(|c| c.shape.table == table).collect();
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
            of_table.iter().any(|column| column.shape.column == c)
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
                        .get(column.shape.column)
                        .is_some_and(|v| same_value(v, &taken.old))
                {
                    apply.push((column.shape.column, taken));
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
fn check_changed(
    engine: &MVCCEngine,
    txn_id: i64,
    columns: &[Changed],
    mut pending: Option<&mut PendingParents<'_>>,
) -> Result<()> {
    let checked = |t: &Taken| t.changed && t.written.get() && !t.new.is_null();
    for column in columns {
        if !column.values.iter().any(checked) {
            continue;
        }
        let (table, shape) = (&column.shape.table, column.shape);
        let schema = engine.get_table_schema(table)?;
        for fk in schema
            .foreign_keys
            .iter()
            .filter(|k| k.column_index == shape.column)
        {
            let bit = shape.into.iter().position(|k| same_key(k, fk));
            for taken in column.values.iter().filter(|t| checked(t)) {
                if bit.is_some_and(|b| b < 64 && taken.from & (1 << b) != 0) {
                    continue;
                }
                match pending.as_deref_mut() {
                    Some(pending) => pending.ask(engine, txn_id, table, fk, &taken.new)?,
                    None => validate_fk_value(engine, txn_id, fk, &taken.new, table)?,
                }
            }
        }
    }
    Ok(())
}

/// An error when another table's foreign key names `parent_table`, which a
/// DROP TABLE without CASCADE would leave naming nothing. The table's keys
/// into itself go with it
pub(crate) fn check_no_referencing_keys(engine: &MVCCEngine, parent_table: &str) -> Result<()> {
    let referencing = find_referencing_fks(engine, parent_table);
    // The keys store the child's name in lower case
    let own = parent_table.to_lowercase();
    let Some((child_table, fk)) = referencing.iter().find(|(child, _)| *child != own) else {
        return Ok(());
    };
    Err(Error::foreign_key_violation(
        child_table,
        &fk.column_name,
        parent_table,
        &fk.referenced_column,
        format!(
            "cannot drop table '{parent_table}': table '{child_table}' references it, \
             DROP TABLE ... CASCADE drops the reference"
        ),
    ))
}

/// Check if any child tables have rows that actually reference the given parent table.
/// Used by TRUNCATE to ensure no referencing rows exist.
/// Only counts rows where the FK column is non-NULL (NULL means "no reference").
///
/// Blocks for ALL FK action types (RESTRICT, CASCADE, SET NULL, NO ACTION) because
/// TRUNCATE is a DDL operation that doesn't cascade to child rows: it
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

    // A table's rows that reference the table itself go with it; the keys
    // store the child's name in lower case
    let own = parent_table.to_lowercase();
    for (child_table, fk) in referencing.iter().filter(|(child, _)| *child != own) {
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
