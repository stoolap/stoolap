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

//! Volcano-style operator interface for streaming query execution.
//!
//! This module provides the foundation for a streaming execution model where
//! operators pull rows on-demand rather than materializing everything upfront.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐
//! │ Consumer     │ ← Pulls rows via next()
//! └──────┬───────┘
//!        │
//! ┌──────▼───────┐
//! │ Join Op      │ ← Build side materialized, probe side streamed
//! └──────┬───────┘
//!        │
//! ┌──────┴──────┐
//! │             │
//! ▼             ▼
//! ┌─────┐   ┌─────┐
//! │Scan │   │Scan │ ← Stream rows from storage
//! └─────┘   └─────┘
//! ```
//!
//! # Key Benefits
//!
//! 1. **Reduced Memory**: Only materialize what's needed (e.g., hash join build side)
//! 2. **Early Termination**: LIMIT can stop execution without processing all rows
//! 3. **Pipelining**: Multiple operators can work on the same row in sequence
//! 4. **Zero-Copy**: RowRef allows referencing rows without cloning

use std::fmt;
use std::sync::Arc;

use crate::core::{Result, Row, Value};

/// Column information for operator schema.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// Column name
    pub name: String,
    /// Original table alias (if from a table)
    pub table_alias: Option<String>,
}

impl ColumnInfo {
    /// Create a new column info with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            table_alias: None,
        }
    }

    /// Create a column info with table alias.
    pub fn with_table(name: impl Into<String>, table_alias: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            table_alias: Some(table_alias.into()),
        }
    }
}

/// Volcano-style iterator interface for query operators.
///
/// Each operator implements this trait to participate in the streaming
/// execution pipeline. The execution follows the open-next-close pattern:
///
/// 1. `open()` - Initialize the operator (called once)
/// 2. `next()` - Get the next row (called repeatedly until None)
/// 3. `close()` - Release resources (called once at end)
///
/// # Thread Safety
///
/// Operators are `Send` to allow execution on different threads,
/// but individual operators are not `Sync` - they maintain mutable state.
pub trait Operator: Send {
    /// Initialize the operator.
    ///
    /// Called once before the first `next()` call.
    /// This is where child operators should be opened and
    /// any one-time initialization should occur.
    fn open(&mut self) -> Result<()>;

    /// Get the next row from this operator.
    ///
    /// Returns:
    /// - `Ok(Some(row))` - A row is available
    /// - `Ok(None)` - No more rows (exhausted)
    /// - `Err(e)` - An error occurred
    ///
    /// After returning `None`, subsequent calls should continue to return `None`.
    fn next(&mut self) -> Result<Option<RowRef>>;

    /// Close the operator and release resources.
    ///
    /// Called once after all rows have been consumed or when
    /// execution is terminated early. Child operators should
    /// also be closed.
    fn close(&mut self) -> Result<()>;

    /// Get the schema (column information) for this operator's output.
    fn schema(&self) -> &[ColumnInfo];

    /// Get an estimate of the number of rows this operator will produce.
    ///
    /// Returns `None` if the estimate is not available.
    /// Used by the query planner for cost estimation.
    fn estimated_rows(&self) -> Option<usize> {
        None
    }

    /// Get a descriptive name for this operator (for EXPLAIN).
    fn name(&self) -> &str;
}

/// A row reference that can be borrowed, owned, or composite.
///
/// This enum allows operators to return rows without always cloning:
/// - `Borrowed`: Reference to an existing row (zero-copy)
/// - `Owned`: An owned row (when materialization is needed)
/// - `Composite`: Virtual row combining left and right join sides
///
/// # Performance
///
/// The key optimization is that `Composite` allows hash joins to
/// return combined rows without actually copying values from both sides.
/// Values are only copied when the final result is materialized.
#[derive(Debug)]
pub enum RowRef {
    /// Owned row - the row data is owned by this RowRef.
    Owned(Row),

    /// Composite row - combines two rows without copying.
    /// Used by join operators to avoid materializing combined rows.
    Composite(CompositeRow),

    /// Shared build composite - combines probe row with an Arc-referenced build row.
    /// Avoids cloning build rows by keeping a reference to the shared storage.
    SharedBuildComposite(SharedBuildCompositeRow),
}

impl RowRef {
    /// Create an owned RowRef from a Row.
    #[inline]
    pub fn owned(row: Row) -> Self {
        RowRef::Owned(row)
    }

    /// Create a composite RowRef from left and right rows.
    #[inline]
    pub fn composite(left: Row, right: Row) -> Self {
        RowRef::Composite(CompositeRow::new(left, right))
    }

    /// Get the number of columns in this row.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            RowRef::Owned(row) => row.len(),
            RowRef::Composite(comp) => comp.len(),
            RowRef::SharedBuildComposite(shared) => shared.len(),
        }
    }

    /// Check if this row is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a value by index without cloning.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&Value> {
        match self {
            RowRef::Owned(row) => row.get(idx),
            RowRef::Composite(comp) => comp.get(idx),
            RowRef::SharedBuildComposite(shared) => shared.get(idx),
        }
    }

    /// Convert to an owned Row.
    ///
    /// For `Owned`, this is a no-op move.
    /// For `Composite` and `SharedBuildComposite`, this materializes the combined row.
    #[inline]
    pub fn into_owned(self) -> Row {
        match self {
            RowRef::Owned(row) => row,
            RowRef::Composite(comp) => comp.materialize(),
            RowRef::SharedBuildComposite(shared) => shared.materialize(),
        }
    }

    /// Clone to an owned Row.
    ///
    /// Use `into_owned()` when possible to avoid cloning.
    pub fn to_owned(&self) -> Row {
        match self {
            RowRef::Owned(row) => row.clone(),
            RowRef::Composite(comp) => comp.materialize(),
            RowRef::SharedBuildComposite(shared) => shared.materialize(),
        }
    }

    /// Get a reference to the underlying Row if this is Owned.
    #[inline]
    pub fn as_row(&self) -> Option<&Row> {
        match self {
            RowRef::Owned(row) => Some(row),
            RowRef::Composite(_) => None,
            RowRef::SharedBuildComposite(_) => None,
        }
    }

    /// Create a shared-build composite RowRef.
    ///
    /// This avoids cloning build rows by keeping an Arc reference.
    #[inline]
    pub fn shared_build_composite(
        probe: Row,
        build_rows: Arc<Vec<Row>>,
        build_idx: usize,
        probe_is_left: bool,
    ) -> Self {
        RowRef::SharedBuildComposite(SharedBuildCompositeRow::new(
            probe,
            build_rows,
            build_idx,
            probe_is_left,
        ))
    }
}

/// A composite row that references values from two source rows.
///
/// This is the key optimization for joins - instead of cloning all values
/// from both the left and right rows into a new row, we keep references
/// to both and provide a unified view.
///
/// # Memory Layout
///
/// ```text
/// CompositeRow
/// ├── left: Row (owned)
/// ├── right: Row (owned)
/// └── left_cols: usize
///
/// Logical columns: [left_col_0, left_col_1, ..., right_col_0, right_col_1, ...]
///                  |<--- left_cols --->|<--- right cols --->|
/// ```
#[derive(Debug, Clone)]
pub struct CompositeRow {
    /// Left side of the join (probe row in hash join)
    left: Row,
    /// Right side of the join (build row in hash join)
    right: Row,
    /// Number of columns from the left side
    left_cols: usize,
}

impl CompositeRow {
    /// Create a new composite row from left and right parts.
    #[inline]
    pub fn new(left: Row, right: Row) -> Self {
        let left_cols = left.len();
        Self {
            left,
            right,
            left_cols,
        }
    }

    /// Create a composite row with explicit column count.
    ///
    /// Use this when the left row has been modified (e.g., projection).
    #[inline]
    pub fn with_counts(left: Row, right: Row, left_cols: usize) -> Self {
        Self {
            left,
            right,
            left_cols,
        }
    }

    /// Get the total number of columns.
    #[inline]
    pub fn len(&self) -> usize {
        self.left_cols + self.right.len()
    }

    /// Check if this composite row is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.left.is_empty() && self.right.is_empty()
    }

    /// Get a value by index without cloning.
    ///
    /// Indexes 0..left_cols return from the left row.
    /// Indexes left_cols..total return from the right row.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&Value> {
        if idx < self.left_cols {
            self.left.get(idx)
        } else {
            self.right.get(idx - self.left_cols)
        }
    }

    /// Get a reference to the left row.
    #[inline]
    pub fn left(&self) -> &Row {
        &self.left
    }

    /// Get a reference to the right row.
    #[inline]
    pub fn right(&self) -> &Row {
        &self.right
    }

    /// Materialize into an owned Row.
    ///
    /// This creates a single Row by copying all values from both sides.
    /// Only call this when the final result needs to be stored.
    pub fn materialize(&self) -> Row {
        let total = self.len();
        let mut values = Vec::with_capacity(total);

        // Copy left values
        values.extend(self.left.iter().cloned());

        // Copy right values
        values.extend(self.right.iter().cloned());

        Row::from_values(values)
    }

    /// Decompose into the left and right rows.
    #[inline]
    pub fn into_parts(self) -> (Row, Row) {
        (self.left, self.right)
    }
}

impl fmt::Display for CompositeRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for i in 0..self.len() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if let Some(v) = self.get(i) {
                write!(f, "{}", v)?;
            } else {
                write!(f, "NULL")?;
            }
        }
        write!(f, ")")
    }
}

/// A shared-build composite row that references build rows via Arc.
///
/// This optimization avoids cloning build rows by storing:
/// - Owned probe row
/// - Arc reference to shared build row storage
/// - Index into the build rows
///
/// Only the Arc refcount is incremented (O(1)) instead of cloning
/// the entire build row.
#[derive(Debug, Clone)]
pub struct SharedBuildCompositeRow {
    /// Probe row (owned)
    probe: Row,
    /// Shared reference to build rows
    build_rows: Arc<Vec<Row>>,
    /// Index into build_rows
    build_idx: usize,
    /// Number of columns from the probe (left) side
    probe_cols: usize,
    /// Whether probe is left side (true) or right side (false)
    probe_is_left: bool,
}

impl SharedBuildCompositeRow {
    /// Create a new shared-build composite row.
    #[inline]
    pub fn new(
        probe: Row,
        build_rows: Arc<Vec<Row>>,
        build_idx: usize,
        probe_is_left: bool,
    ) -> Self {
        let probe_cols = probe.len();
        Self {
            probe,
            build_rows,
            build_idx,
            probe_cols,
            probe_is_left,
        }
    }

    /// Get the total number of columns.
    #[inline]
    pub fn len(&self) -> usize {
        self.probe_cols + self.build_rows[self.build_idx].len()
    }

    /// Check if this row is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.probe.is_empty() && self.build_rows[self.build_idx].is_empty()
    }

    /// Get a value by index without cloning.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&Value> {
        let build_row = &self.build_rows[self.build_idx];
        if self.probe_is_left {
            // Output: [probe, build]
            if idx < self.probe_cols {
                self.probe.get(idx)
            } else {
                build_row.get(idx - self.probe_cols)
            }
        } else {
            // Output: [build, probe]
            let build_cols = build_row.len();
            if idx < build_cols {
                build_row.get(idx)
            } else {
                self.probe.get(idx - build_cols)
            }
        }
    }

    /// Materialize into an owned Row.
    pub fn materialize(&self) -> Row {
        let build_row = &self.build_rows[self.build_idx];
        let total = self.probe_cols + build_row.len();
        let mut values = Vec::with_capacity(total);

        if self.probe_is_left {
            // Output: [probe, build]
            values.extend(self.probe.iter().cloned());
            values.extend(build_row.iter().cloned());
        } else {
            // Output: [build, probe]
            values.extend(build_row.iter().cloned());
            values.extend(self.probe.iter().cloned());
        }

        Row::from_values(values)
    }
}

impl fmt::Display for SharedBuildCompositeRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for i in 0..self.len() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if let Some(v) = self.get(i) {
                write!(f, "{}", v)?;
            } else {
                write!(f, "NULL")?;
            }
        }
        write!(f, ")")
    }
}

/// A null row with a specified number of columns.
///
/// Used for OUTER JOINs when there's no match on one side.
pub struct NullRow {
    len: usize,
}

impl NullRow {
    /// Create a null row with the specified number of columns.
    pub fn new(len: usize) -> Self {
        Self { len }
    }

    /// Materialize into a Row filled with NULLs.
    pub fn materialize(&self) -> Row {
        Row::from_values(vec![Value::null_unknown(); self.len])
    }
}

// ============================================================================
// Helper Operators
// ============================================================================

/// An empty operator that produces no rows.
///
/// Useful as a placeholder or for empty result sets.
pub struct EmptyOperator {
    schema: Vec<ColumnInfo>,
    opened: bool,
}

impl EmptyOperator {
    /// Create an empty operator with no schema.
    pub fn new() -> Self {
        Self {
            schema: Vec::new(),
            opened: false,
        }
    }

    /// Create an empty operator with a specific schema.
    pub fn with_schema(schema: Vec<ColumnInfo>) -> Self {
        Self {
            schema,
            opened: false,
        }
    }
}

impl Default for EmptyOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for EmptyOperator {
    fn open(&mut self) -> Result<()> {
        self.opened = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        Ok(None)
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn name(&self) -> &str {
        "Empty"
    }
}

/// An operator that yields rows from a pre-materialized vector.
///
/// This is useful for:
/// - Converting existing Vec<Row> results to the operator model
/// - CTEs that have been pre-computed
/// - Subquery results
pub struct MaterializedOperator {
    rows: Vec<Row>,
    schema: Vec<ColumnInfo>,
    current_idx: usize,
    opened: bool,
}

impl MaterializedOperator {
    /// Create an operator from a vector of rows.
    pub fn new(rows: Vec<Row>, schema: Vec<ColumnInfo>) -> Self {
        Self {
            rows,
            schema,
            current_idx: 0,
            opened: false,
        }
    }

    /// Create from an Arc<Vec<Row>>, unwrapping if sole owner or cloning if shared.
    /// This is optimal for CTE results which may have multiple references.
    pub fn from_arc(arc_rows: Arc<Vec<Row>>, schema: Vec<ColumnInfo>) -> Self {
        let rows = Arc::try_unwrap(arc_rows).unwrap_or_else(|arc| (*arc).clone());
        Self::new(rows, schema)
    }

    /// Create from rows with column names as strings.
    pub fn from_rows(rows: Vec<Row>, columns: Vec<String>) -> Self {
        let schema = columns.into_iter().map(ColumnInfo::new).collect();
        Self::new(rows, schema)
    }
}

impl Operator for MaterializedOperator {
    fn open(&mut self) -> Result<()> {
        self.current_idx = 0;
        self.opened = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if self.current_idx >= self.rows.len() {
            return Ok(None);
        }

        // Take ownership of the row, leaving an empty Row in its place.
        // This is O(1) instead of clone() which is O(n) for row width.
        // Safe because we only iterate forward and never revisit rows.
        let row = std::mem::take(&mut self.rows[self.current_idx]);
        self.current_idx += 1;
        Ok(Some(RowRef::Owned(row)))
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn estimated_rows(&self) -> Option<usize> {
        Some(self.rows.len())
    }

    fn name(&self) -> &str {
        "Materialized"
    }
}

// ============================================================================
// QueryResult to Operator Adapter
// ============================================================================

use crate::storage::QueryResult as StorageQueryResult;

/// Operator that streams rows from a QueryResult.
///
/// This adapter allows existing QueryResult (from table scans, etc.)
/// to be used in the streaming operator pipeline. Unlike MaterializedOperator,
/// this does NOT load all rows upfront - it streams them on demand.
///
/// # Benefits
///
/// - **Memory efficient**: Only one row in memory at a time
/// - **Early termination**: LIMIT stops reading immediately
/// - **Streaming pipeline**: Fits into Volcano execution model
pub struct QueryResultOperator {
    result: Box<dyn StorageQueryResult>,
    schema: Vec<ColumnInfo>,
    opened: bool,
}

impl QueryResultOperator {
    /// Create a new streaming operator from a QueryResult.
    pub fn new(result: Box<dyn StorageQueryResult>, columns: Vec<String>) -> Self {
        let schema = columns.into_iter().map(ColumnInfo::new).collect();
        Self {
            result,
            schema,
            opened: false,
        }
    }
}

impl Operator for QueryResultOperator {
    fn open(&mut self) -> Result<()> {
        self.opened = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if !self.opened {
            return Ok(None);
        }

        if self.result.next() {
            // Get the row from the result (row() returns &Row) - single clone
            Ok(Some(RowRef::Owned(self.result.row().clone())))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<()> {
        self.result.close()
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn estimated_rows(&self) -> Option<usize> {
        // QueryResult doesn't expose count, return None
        None
    }

    fn name(&self) -> &str {
        "QueryResultScan"
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_row_basic() {
        let left = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let right = Row::from_values(vec![Value::float(3.14), Value::boolean(true)]);

        let comp = CompositeRow::new(left, right);

        assert_eq!(comp.len(), 4);
        assert_eq!(comp.get(0), Some(&Value::integer(1)));
        assert_eq!(comp.get(1), Some(&Value::text("hello")));
        assert_eq!(comp.get(2), Some(&Value::float(3.14)));
        assert_eq!(comp.get(3), Some(&Value::boolean(true)));
        assert_eq!(comp.get(4), None);
    }

    #[test]
    fn test_composite_row_materialize() {
        let left = Row::from_values(vec![Value::integer(1)]);
        let right = Row::from_values(vec![Value::integer(2)]);

        let comp = CompositeRow::new(left, right);
        let materialized = comp.materialize();

        assert_eq!(materialized.len(), 2);
        assert_eq!(materialized.get(0), Some(&Value::integer(1)));
        assert_eq!(materialized.get(1), Some(&Value::integer(2)));
    }

    #[test]
    fn test_row_ref_owned() {
        let row = Row::from_values(vec![Value::integer(42)]);
        let row_ref = RowRef::owned(row);

        assert_eq!(row_ref.len(), 1);
        assert_eq!(row_ref.get(0), Some(&Value::integer(42)));

        let owned = row_ref.into_owned();
        assert_eq!(owned.get(0), Some(&Value::integer(42)));
    }

    #[test]
    fn test_row_ref_composite() {
        let left = Row::from_values(vec![Value::integer(1)]);
        let right = Row::from_values(vec![Value::integer(2)]);
        let row_ref = RowRef::composite(left, right);

        assert_eq!(row_ref.len(), 2);
        assert_eq!(row_ref.get(0), Some(&Value::integer(1)));
        assert_eq!(row_ref.get(1), Some(&Value::integer(2)));
    }

    #[test]
    fn test_empty_operator() {
        let mut op = EmptyOperator::new();
        op.open().unwrap();

        assert!(op.next().unwrap().is_none());
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn test_materialized_operator() {
        let rows = vec![
            Row::from_values(vec![Value::integer(1)]),
            Row::from_values(vec![Value::integer(2)]),
            Row::from_values(vec![Value::integer(3)]),
        ];
        let schema = vec![ColumnInfo::new("id")];

        let mut op = MaterializedOperator::new(rows, schema);
        op.open().unwrap();

        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get(0), Some(&Value::integer(1)));

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get(0), Some(&Value::integer(2)));

        let row3 = op.next().unwrap().unwrap();
        assert_eq!(row3.get(0), Some(&Value::integer(3)));

        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn test_null_row() {
        let null_row = NullRow::new(3);
        let materialized = null_row.materialize();

        assert_eq!(materialized.len(), 3);
        assert!(materialized.get(0).unwrap().is_null());
        assert!(materialized.get(1).unwrap().is_null());
        assert!(materialized.get(2).unwrap().is_null());
    }
}
