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

//! Window Function Execution
//!
//! This module implements window function execution for SQL queries.
//!
//! Supports:
//! - ROW_NUMBER() - Sequential row numbering
//! - RANK() - Ranking with gaps
//! - DENSE_RANK() - Ranking without gaps
//! - NTILE(n) - Divides rows into n groups
//! - LEAD(col, offset, default) - Access next row's value
//! - LAG(col, offset, default) - Access previous row's value
//!
//! Window clauses:
//! - OVER () - Entire result set as one partition
//! - OVER (PARTITION BY col) - Partition by column values
//! - OVER (ORDER BY col) - Order within partition

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::sync::OnceLock;

use crate::common::{CompactVec, StringMap};
use crate::core::row_vec::RowVec;
use crate::core::value::NULL_VALUE;

/// Stack constant for COUNT(*) accumulation by reference
const COUNT_ONE: Value = Value::Integer(1);

/// The value a window aggregate reads from one row: its argument column,
/// the argument expression evaluated up front, or one per row for COUNT(*)
#[inline(always)]
fn window_argument_value<'a>(
    rows: &'a [(i64, Row)],
    idx: usize,
    arg_col_idx: Option<usize>,
    expression_values: &'a [Value],
) -> &'a Value {
    if let Some(col_idx) = arg_col_idx {
        rows[idx].1.get(col_idx).unwrap_or(&NULL_VALUE)
    } else if !expression_values.is_empty() {
        &expression_values[idx]
    } else {
        &COUNT_ONE
    }
}

/// One frame of an aggregate that orders its own input
fn accumulate_ordered_frame(
    agg_func: &mut crate::functions::aggregate::AggregateState,
    frame: &[usize],
    rows: &[(i64, Row)],
    arg_col_idx: Option<usize>,
    expression_values: &[Value],
    keys: &[Vec<Value>],
) {
    for &idx in frame {
        let value = window_argument_value(rows, idx, arg_col_idx, expression_values);
        agg_func.accumulate_with_sort_key(value, keys[idx].clone());
    }
}

/// Programs over the rows a window statement reads, each run with the
/// statement's parameters, outer row and transaction, and its subqueries
/// answered by the statement's `blocks`
struct RowPrograms<'a> {
    programs: SmallVec<[SharedProgram; 1]>,
    vm: ExprVM,
    ctx: &'a ExecutionContext,
    blocks: Option<&'a dyn SubqueryExecutor>,
}

impl<'a> RowPrograms<'a> {
    /// `exprs` over `columns`, a group item or aggregate read from its slot
    /// by identity when `slots` are given
    fn compile<'e>(
        exprs: impl IntoIterator<Item = &'e Expression>,
        columns: &[String],
        slots: Option<&FxHashMap<ExprKey, usize>>,
        ctx: &'a ExecutionContext,
        blocks: Option<&'a dyn SubqueryExecutor>,
    ) -> Result<Self> {
        // A statement run for an enclosing row reads that row's columns by name
        let outer = super::query::outer_column_names(ctx);
        let programs = exprs
            .into_iter()
            .map(|expr| match (blocks, slots, &outer) {
                (None, None, None) => compile_expression(expr, columns)
                    .map(|program| bind_params(program, ctx.params())),
                _ => {
                    let program = compile_over_slots(
                        expr,
                        columns,
                        global_registry(),
                        slots,
                        outer.as_deref(),
                        blocks.is_some(),
                        Some(slot_params(ctx, blocks)),
                    )?;
                    Ok(blocks
                        .and_then(|blocks| blocks.bind_members(&program))
                        .unwrap_or(program))
                }
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            programs,
            vm: ExprVM::new(),
            ctx,
            blocks,
        })
    }

    /// The first program's value for `row`
    #[inline]
    fn eval(&mut self, row: &Row) -> Result<Value> {
        let exec = with_blocks(slot_context(row, self.ctx), self.blocks);
        self.vm.execute(&self.programs[0], &exec)
    }

    /// Each program's value for `row`, in order, into `out`
    #[inline]
    fn eval_into(&mut self, row: &Row, out: &mut Vec<Value>) -> Result<()> {
        let exec = with_blocks(slot_context(row, self.ctx), self.blocks);
        out.clear();
        for program in &self.programs {
            out.push(self.vm.execute(program, &exec)?);
        }
        Ok(())
    }
}

/// How a window argument is read: a literal by its value, resolved once
/// for the call; a parameter from the statement; anything else by a
/// program compiled the first time a partition reads it
enum ArgSource {
    Literal(Value),
    Parameter,
    Program(OnceLock<Result<Program>>),
}

impl ArgSource {
    fn of(expr: &Expression) -> Self {
        match literal_value(expr) {
            Some(value) => ArgSource::Literal(value),
            None if matches!(expr, Expression::Parameter(_)) => ArgSource::Parameter,
            None => ArgSource::Program(OnceLock::new()),
        }
    }
}

/// A window-function argument that cannot reference row columns, read as
/// its source says and kept for the call
struct ConstArg<'w> {
    expr: &'w Expression,
    deterministic: bool,
    source: ArgSource,
}

impl<'w> ConstArg<'w> {
    fn new(expr: &'w Expression, registry: &FunctionRegistry) -> Self {
        Self {
            expr,
            deterministic: expr_is_deterministic(expr, registry),
            source: ArgSource::of(expr),
        }
    }

    fn evaluate(&self, ctx: &ExecutionContext) -> Result<Value> {
        match &self.source {
            ArgSource::Literal(value) => Ok(value.clone()),
            ArgSource::Parameter => evaluate_once(self.expr, Some(ctx)),
            ArgSource::Program(program) => match program.get_or_init(|| compile_once(self.expr)) {
                Ok(program) => execute_once(program, Some(ctx)),
                Err(e) => Err(e.clone()),
            },
        }
    }
}

/// A constant argument as one partition reads it: a deterministic one
/// evaluates once, a volatile one (RANDOM and friends) on every read
struct PartitionArg<'a, 'w> {
    arg: &'a ConstArg<'w>,
    cached: Option<Value>,
}

impl<'a, 'w> PartitionArg<'a, 'w> {
    fn new(arg: &'a ConstArg<'w>) -> Self {
        Self { arg, cached: None }
    }

    fn value(&mut self, ctx: &ExecutionContext) -> Result<Value> {
        if !self.arg.deterministic {
            return self.arg.evaluate(ctx);
        }
        if let Some(v) = &self.cached {
            return Ok(v.clone());
        }
        let v = self.arg.evaluate(ctx)?;
        self.cached = Some(v.clone());
        Ok(v)
    }
}

/// The arguments of one window function call that are not its input:
/// the frame offsets, LAG and LEAD's offset and default, NTILE's and
/// NTH_VALUE's n. Built once per call; each compiles on first read
struct CallArgs<'w> {
    frame_start: Option<ConstArg<'w>>,
    frame_end: Option<ConstArg<'w>>,
    count: Option<ConstArg<'w>>,
    /// LAG and LEAD's default, which may read the row
    default: Option<(&'w Expression, ArgSource)>,
    /// A value function's first argument for each row, when it is not a column
    input: Vec<Value>,
}

/// How a value function reads its first argument at a row: from its column,
/// else from the values read for each row; None when it has neither
#[derive(Clone, Copy)]
struct ValueInput<'a> {
    column: Option<usize>,
    values: &'a [Value],
}

impl ValueInput<'_> {
    #[inline]
    fn at(self, rows: &[(i64, Row)], row_idx: usize) -> Option<Value> {
        match self.column {
            Some(column) => Some(
                rows[row_idx]
                    .1
                    .get(column)
                    .cloned()
                    .unwrap_or_else(Value::null_unknown),
            ),
            None => self.values.get(row_idx).cloned(),
        }
    }
}

impl<'w> CallArgs<'w> {
    fn of(wf_info: &'w WindowFunctionInfo, registry: &FunctionRegistry) -> Self {
        let offset_of = |bound: &'w WindowFrameBound| match bound {
            WindowFrameBound::Preceding(expr) | WindowFrameBound::Following(expr) => {
                Some(ConstArg::new(expr, registry))
            }
            _ => None,
        };
        let frame = wf_info.frame.as_ref();
        let lead_lag = matches!(wf_info.name.as_str(), "LEAD" | "LAG");
        let count_at = match wf_info.name.as_str() {
            "LEAD" | "LAG" | "NTH_VALUE" => Some(1),
            "NTILE" => Some(0),
            _ => None,
        };
        Self {
            frame_start: frame.and_then(|f| offset_of(&f.start)),
            frame_end: frame.and_then(|f| f.end.as_ref()).and_then(offset_of),
            count: count_at
                .and_then(|at| wf_info.arguments.get(at))
                .map(|expr| ConstArg::new(expr, registry)),
            default: wf_info
                .arguments
                .get(2)
                .filter(|_| lead_lag)
                .map(|expr| (expr, ArgSource::of(expr))),
            input: Vec::new(),
        }
    }

    /// The default's program over the input `columns`, compiled on first
    /// read; None when there is no default or it is read without one
    fn default_program(&self, columns: &[String]) -> Result<Option<&Program>> {
        let Some((expr, ArgSource::Program(program))) = &self.default else {
            return Ok(None);
        };
        let program = program.get_or_init(|| {
            ExprCompiler::new(&CompileContext::with_global_registry(columns))
                .compile(expr)
                .map_err(|e| Error::internal(format!("Compile error: {}", e)))
        });
        program.as_ref().map(Some).map_err(Clone::clone)
    }

    /// LAG and LEAD's default for the current `row`, NULL without one
    fn default_value(
        &self,
        program: Option<&Program>,
        vm: &mut ExprVM,
        row: &Row,
        ctx: &ExecutionContext,
    ) -> Result<Value> {
        match (program, &self.default) {
            (Some(program), _) => vm.execute(program, &execute_context(row, Some(ctx))),
            (None, Some((_, ArgSource::Literal(value)))) => Ok(value.clone()),
            (None, Some((expr, _))) => evaluate_once(expr, Some(ctx)),
            (None, None) => Ok(Value::null_unknown()),
        }
    }
}

/// The error for a call with OVER whose name the registry does not resolve
pub(super) fn unknown_window_function(name: &str) -> Error {
    Error::NotSupported(format!("Unknown window function: {}", name))
}

/// Reads a window argument that counts rows (a ROWS frame offset, the
/// offset of LAG and LEAD, the n of NTILE and NTH_VALUE): an INTEGER of at
/// least `min`, read as every INTEGER argument is
fn window_count(value: &Value, what: &str, min: i64) -> Result<usize> {
    match integer_from(value)? {
        Some(n) if n >= min => Ok(usize::try_from(n).unwrap_or(usize::MAX)),
        _ => Err(Error::invalid_argument(format!(
            "{what} must be an integer of at least {min}, got {value}"
        ))),
    }
}

/// Reads a RANGE frame offset, a distance in the units of the ORDER BY:
/// a number of zero or more
fn range_distance(value: &Value) -> Result<f64> {
    let distance = match value {
        Value::Integer(n) => *n as f64,
        Value::Float(f) => *f,
        _ => f64::NAN,
    };
    if distance >= 0.0 {
        Ok(distance)
    } else {
        Err(Error::invalid_argument(format!(
            "a RANGE frame offset must be a number of at least 0, got {value}"
        )))
    }
}

/// How far a frame edge lies from the current row
#[derive(Clone, Copy)]
enum FrameOffset {
    Rows(usize),
    Range(f64),
}

#[derive(Clone, Copy)]
enum FrameEdge {
    UnboundedPreceding,
    Preceding(FrameOffset),
    CurrentRow,
    Following(FrameOffset),
    UnboundedFollowing,
}

/// The frame of one window function call, its offsets read once and the
/// default filled in; `range` makes CURRENT ROW reach the row's peers
struct Frame {
    start: FrameEdge,
    end: FrameEdge,
    range: bool,
}

impl Frame {
    fn read(wf_info: &WindowFunctionInfo, args: &CallArgs, ctx: &ExecutionContext) -> Result<Self> {
        let Some(frame) = &wf_info.frame else {
            let end = if wf_info.order_by.is_empty() {
                FrameEdge::UnboundedFollowing
            } else {
                FrameEdge::CurrentRow
            };
            return Ok(Frame {
                start: FrameEdge::UnboundedPreceding,
                end,
                range: true,
            });
        };
        let range = frame.unit == WindowFrameUnit::Range;
        let offset = |arg: &Option<ConstArg>| -> Result<FrameOffset> {
            let arg = arg
                .as_ref()
                .ok_or_else(|| Error::internal("a frame offset with no argument"))?;
            let value = arg.evaluate(ctx)?;
            Ok(if range {
                FrameOffset::Range(range_distance(&value)?)
            } else {
                FrameOffset::Rows(window_count(&value, "a ROWS frame offset", 0)?)
            })
        };
        let edge = |bound: &WindowFrameBound, arg: &Option<ConstArg>| -> Result<FrameEdge> {
            Ok(match bound {
                WindowFrameBound::UnboundedPreceding => FrameEdge::UnboundedPreceding,
                WindowFrameBound::Preceding(_) => FrameEdge::Preceding(offset(arg)?),
                WindowFrameBound::CurrentRow => FrameEdge::CurrentRow,
                WindowFrameBound::Following(_) => FrameEdge::Following(offset(arg)?),
                WindowFrameBound::UnboundedFollowing => FrameEdge::UnboundedFollowing,
            })
        };
        Ok(Frame {
            start: edge(&frame.start, &args.frame_start)?,
            end: match &frame.end {
                Some(end) => edge(end, &args.frame_end)?,
                None => FrameEdge::CurrentRow,
            },
            range,
        })
    }

    /// Whether the frame never loses a row as the current row advances,
    /// so an aggregate can accumulate it instead of recomputing
    fn only_grows(&self) -> bool {
        matches!(self.start, FrameEdge::UnboundedPreceding)
            && match self.end {
                FrameEdge::UnboundedPreceding | FrameEdge::Preceding(_) => false,
                // A NULL key frames to its peers, which can end before an earlier row's frame
                FrameEdge::Following(FrameOffset::Range(_)) => false,
                _ => true,
            }
    }
}

/// What the frame bounds of one sorted partition read: its peer groups,
/// and for a RANGE offset each row's first ORDER BY key as a number
/// measured in the sort direction (None for a key that is not a number)
struct PartitionFrames {
    peers: Vec<(usize, usize)>,
    keys: Vec<Option<f64>>,
}

impl PartitionFrames {
    fn new(frame: &Frame, sorted: &[usize], order_by: &ColumnarOrderByValues) -> Self {
        let has_range_offset = |edge: FrameEdge| {
            matches!(
                edge,
                FrameEdge::Preceding(FrameOffset::Range(_))
                    | FrameEdge::Following(FrameOffset::Range(_))
            )
        };
        let keys = if has_range_offset(frame.start) || has_range_offset(frame.end) {
            let sign = if order_by.is_ascending(0) { 1.0 } else { -1.0 };
            sorted
                .iter()
                .map(|&idx| match order_by.get_first(idx) {
                    Some(Value::Integer(i)) => Some(sign * *i as f64),
                    Some(Value::Float(f)) if !f.is_nan() => Some(sign * *f),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        PartitionFrames {
            peers: Executor::precompute_peer_groups(sorted, order_by),
            keys,
        }
    }

    /// The frame of position `i` as a range of positions, end exclusive
    fn bounds(&self, frame: &Frame, i: usize) -> (usize, usize) {
        let len = self.peers.len();
        let start = match frame.start {
            FrameEdge::UnboundedPreceding => 0,
            FrameEdge::CurrentRow if frame.range => self.peers[i].0,
            FrameEdge::CurrentRow => i,
            FrameEdge::Preceding(FrameOffset::Rows(n)) => i.saturating_sub(n),
            FrameEdge::Following(FrameOffset::Rows(n)) => i.saturating_add(n).min(len),
            FrameEdge::Preceding(FrameOffset::Range(d)) => self.first_key_at_least(i, -d),
            FrameEdge::Following(FrameOffset::Range(d)) => self.first_key_at_least(i, d),
            FrameEdge::UnboundedFollowing => len,
        };
        let end = match frame.end {
            FrameEdge::UnboundedPreceding => 0,
            FrameEdge::CurrentRow if frame.range => self.peers[i].1,
            FrameEdge::CurrentRow => i + 1,
            FrameEdge::Preceding(FrameOffset::Rows(n)) => (i + 1).saturating_sub(n),
            FrameEdge::Following(FrameOffset::Rows(n)) => {
                i.saturating_add(n).saturating_add(1).min(len)
            }
            FrameEdge::Preceding(FrameOffset::Range(d)) => self.past_last_key_at_most(i, -d),
            FrameEdge::Following(FrameOffset::Range(d)) => self.past_last_key_at_most(i, d),
            FrameEdge::UnboundedFollowing => len,
        };
        (start, end.max(start))
    }

    /// The first position whose key is at least the current key plus
    /// `shift`; a row without a numeric key starts at its peers. An infinite
    /// key shifted by an infinite distance has every numeric key within it.
    fn first_key_at_least(&self, i: usize, shift: f64) -> usize {
        let Some(current) = self.keys[i] else {
            return self.peers[i].0;
        };
        let bound = current + shift;
        self.keys
            .iter()
            .position(|key| key.is_some_and(|k| k >= bound || bound.is_nan()))
            .unwrap_or(self.keys.len())
    }

    /// One past the last position whose key is at most the current key
    /// plus `shift`; a row without a numeric key ends at its peers
    fn past_last_key_at_most(&self, i: usize, shift: f64) -> usize {
        let Some(current) = self.keys[i] else {
            return self.peers[i].1;
        };
        let bound = current + shift;
        self.keys
            .iter()
            .rposition(|key| key.is_some_and(|k| k <= bound || bound.is_nan()))
            .map_or(0, |j| j + 1)
    }
}

/// Whether this expression yields the same value every time it is evaluated
/// against the same row within the statement
fn expr_is_deterministic(expr: &Expression, registry: &FunctionRegistry) -> bool {
    expression_volatility(expr, registry) != Volatility::PerCall
}
use crate::core::integer::integer_from;
use crate::core::{Error, Result, Row, Value};

/// Type alias for partition keys - stack-allocated for common case (up to 4 columns)
type PartitionKey = SmallVec<[Value; 4]>;

/// Partitions of row indices, each already sorted by its ORDER BY,
/// shared across window functions with the same OVER clause
type SharedSortedPartitions = std::sync::Arc<Vec<Vec<usize>>>;

/// Identity of one ORDER BY item: expression hash, rendered expression,
/// direction and NULLS placement. The hash alone is not identity (two
/// expressions can collide), and rendered SQL alone is not either
/// (Display drops positional parameter indices), so a cache hit
/// requires both to match.
type OrderByItemKey = (u64, String, bool, Option<bool>);
type OrderByKey = SmallVec<[OrderByItemKey; 2]>;

/// Identity of a whole OVER clause: partition items then ORDER BY items
type OverClauseKey = (SmallVec<[(u64, String); 2]>, OrderByKey);

use crate::functions::registry::{global_registry, FunctionRegistry};
use crate::functions::{Volatility, WindowFunction};
use crate::parser::ast::*;
use crate::storage::traits::{QueryResult, Table};

use super::aggregation::{compile_over_slots, slot_context, slot_params, with_blocks};
use super::context::ExecutionContext;
use super::expr_key::ExprKey;
use super::expression::{
    bind_params, compile_expression, compile_once, compute_expression_hash, evaluate_once,
    execute_context, execute_once, literal_value, CompileContext, ExprCompiler, ExprVM, Program,
    SharedProgram, SubqueryExecutor,
};
use super::query::{order_by_inputs, ColumnSource, OutputLayout};
use super::result::{ColumnarResult, ExecutorResult};
use super::utils::{
    build_column_index_map, derived_aggregate_column_name, for_each_qualified_star_column,
    get_table_alias_from_expr, item_name, qualified, qualified_index, qualified_key_index,
    split_qualified,
};
use super::volatility::expression_volatility;
use super::Executor;

/// Wrapper for parallel writes to disjoint indices in a Vec<Value>.
///
/// SAFETY INVARIANT: Each index must be written by at most one thread.
/// In window function computation, each partition writes to its own row indices,
/// which are guaranteed to be disjoint (each row belongs to exactly one partition).
struct ParallelVec {
    ptr: *mut Value,
    len: usize,
}

// SAFETY: ParallelVec is Send because we only write to disjoint indices
// and Value is Send. Each partition writes to its own indices.
unsafe impl Send for ParallelVec {}

// SAFETY: ParallelVec is Sync because parallel writes to disjoint indices
// don't race with each other. Each row index belongs to exactly one partition.
unsafe impl Sync for ParallelVec {}

impl ParallelVec {
    /// Create from a mutable Vec reference.
    /// The Vec must not be accessed through other references while ParallelVec exists.
    #[inline]
    fn new(vec: &mut Vec<Value>) -> Self {
        Self {
            ptr: vec.as_mut_ptr(),
            len: vec.len(),
        }
    }

    /// Write a value at the given index.
    /// SAFETY: No two threads may write to the same index concurrently.
    #[inline]
    fn write(&self, idx: usize, value: Value) {
        debug_assert!(idx < self.len, "index out of bounds");
        // SAFETY: Caller guarantees no concurrent writes to the same index.
        // In window functions, each partition writes to disjoint row indices.
        unsafe {
            *self.ptr.add(idx) = value;
        }
    }
}

/// Information about a window function call in a SELECT list
#[derive(Clone, Debug)]
pub struct WindowFunctionInfo {
    /// The window function name (ROW_NUMBER, RANK, etc.)
    pub name: String,
    /// Arguments to the function (for LEAD, LAG, NTILE)
    pub arguments: Vec<Expression>,
    /// Partition by column names (simple identifiers only, for fast-path lookups)
    pub partition_by: Vec<String>,
    /// Original PARTITION BY expressions (includes function calls, complex exprs)
    pub partition_by_exprs: Vec<Expression>,
    /// Order by expressions
    pub order_by: Vec<OrderByExpression>,
    /// Window frame specification (e.g., ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
    pub frame: Option<WindowFrame>,
    /// Result column name (may include alias)
    pub column_name: String,
    /// Whether DISTINCT was specified (for COUNT(DISTINCT col) OVER())
    pub is_distinct: bool,
    /// The aggregate's FILTER (WHERE ...) clause
    pub filter: Option<Box<Expression>>,
    /// The aggregate's own ORDER BY, for an aggregate that orders its input
    pub call_order_by: Box<[OrderByExpression]>,
    /// An aggregate's arguments after the first, read once for the statement
    pub extra_args: Vec<Value>,
    /// The error one of them gave, reported once there is a row to aggregate
    pub extra_error: Option<Error>,
}

/// A window result, with the layout its ORDER BY reads when it has one
pub(crate) type WindowOutput = (Box<dyn QueryResult>, Option<OutputLayout>);

/// Information about a SELECT list item for window function processing
pub struct SelectItem {
    pub output_name: String,
    pub source: SelectItemSource,
}

/// Source of a SELECT item value
#[allow(clippy::large_enum_variant)]
pub enum SelectItemSource {
    BaseColumn(usize),
    /// Window function name - stored in lowercase for O(1) lookup in window_value_map
    WindowFunction(String),
    Expression(Expression),
    /// Expression containing one or more window functions.
    /// Stores (expression, list_of_window_function_names_lowercase).
    /// Each Window node in the expression is replaced with a placeholder identifier
    /// referencing the corresponding name at the same index.
    ExpressionWithWindow(Expression, Vec<String>),
}

/// Pre-sorted state for window function optimization
/// When rows are pre-sorted by an indexed column, we can skip sorting in window functions
#[derive(Clone, Debug)]
pub struct WindowPreSortedState {
    /// Column name that rows are sorted by (lowercase)
    pub column: String,
    /// Whether sorted in ascending order
    pub ascending: bool,
}

/// Columnar layout for ORDER BY values - optimized for sorting performance
///
/// Instead of `Vec<Vec<(Value, bool)>>` (row-oriented, N allocations for N rows),
/// this uses `Vec<Vec<Value>>` (column-oriented, K allocations for K ORDER BY columns).
///
/// Benefits:
/// - Reduces allocations from O(N) to O(K) where K = number of ORDER BY columns
/// - Stores ascending flags once per column instead of once per value
/// - Better cache locality when accessing sort keys across rows
#[derive(Clone, Debug)]
pub struct ColumnarOrderByValues {
    /// Column values: columns[col_idx][row_idx] = value
    columns: Vec<Vec<Value>>,
    /// Ascending flags: one per ORDER BY column
    ascending: Vec<bool>,
    /// Where NULLs sort, one per ORDER BY column: NULLS FIRST / LAST when
    /// given, otherwise last ascending and first descending
    nulls_first: Vec<bool>,
    /// Number of rows
    num_rows: usize,
}

impl ColumnarOrderByValues {
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0 || self.columns.is_empty()
    }

    /// Get number of columns
    #[inline]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get value at (row, column)
    #[inline]
    pub fn get(&self, row_idx: usize, col_idx: usize) -> Option<&Value> {
        self.columns.get(col_idx).and_then(|col| col.get(row_idx))
    }

    /// Get first ORDER BY value for a row
    #[inline]
    pub fn get_first(&self, row_idx: usize) -> Option<&Value> {
        self.get(row_idx, 0)
    }

    /// Get ascending flag for column
    #[inline]
    pub fn is_ascending(&self, col_idx: usize) -> bool {
        self.ascending.get(col_idx).copied().unwrap_or(true)
    }

    /// Whether NULLs sort before non-NULL values in this column
    #[inline]
    pub fn nulls_first(&self, col_idx: usize) -> bool {
        self.nulls_first
            .get(col_idx)
            .copied()
            .unwrap_or_else(|| !self.is_ascending(col_idx))
    }

    /// Compare ORDER BY values of two rows for equality (without cloning)
    /// Returns true if all ORDER BY column values are equal
    #[inline]
    pub fn rows_equal(&self, row_a: usize, row_b: usize) -> bool {
        for col in &self.columns {
            let val_a = col.get(row_a);
            let val_b = col.get(row_b);
            match (val_a, val_b) {
                (Some(a), Some(b)) if a == b => continue,
                (None, None) => continue,
                _ => return false,
            }
        }
        true
    }
}

/// Pre-grouped state for window function PARTITION BY optimization
/// When rows are fetched grouped by an indexed partition column, we can skip hash-based grouping
#[derive(Clone)]
pub struct WindowPreGroupedState {
    /// Pre-built partition map: partition key -> row indices
    pub partition_map: FxHashMap<PartitionKey, Vec<usize>>,
    /// The column name (lowercase) that the partition map was built from
    pub partition_column: String,
}

impl Executor {
    /// Execute SELECT with window functions
    /// Accepts &[(i64, Row)] to allow RowVec to be passed directly via deref
    /// The window functions ORDER BY reads that no select column is named
    /// for, each as a column named by its identity, so the sort finds it by
    /// that and the projection drops it afterwards
    pub(crate) fn hidden_order_by_windows(stmt: &SelectStatement) -> Vec<Expression> {
        let mut keys: Vec<ExprKey> = Vec::new();
        stmt.order_by
            .iter()
            .map(|ob| &ob.expression)
            .filter(|expr| {
                super::query_classification::QueryClassification::expression_has_window_function(
                    expr,
                )
            })
            .filter_map(|expr| {
                // Named by its identity, which the sort reads, so a visible
                // column that only writes alike is never taken for it
                let key = ExprKey::of(expr);
                let carried = keys.contains(&key)
                    || stmt.columns.iter().any(|c| match c {
                        Expression::Aliased(a) => ExprKey::of(&a.expression) == key,
                        _ => false,
                    });
                if carried {
                    return None;
                }
                keys.push(key);
                let name = key.hidden_column_name();
                let token = crate::parser::token::Token::new(
                    crate::parser::token::TokenType::Identifier,
                    name.clone(),
                    crate::parser::token::Position::default(),
                );
                Some(Expression::Aliased(crate::parser::ast::AliasedExpression {
                    token: token.clone(),
                    expression: Box::new(expr.clone()),
                    alias: crate::parser::ast::Identifier::new(token, name),
                }))
            })
            .collect()
    }

    /// Window functions over the rows, the statement's subqueries answered
    /// by its `blocks`
    pub(crate) fn execute_select_with_window_functions(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<WindowOutput> {
        self.execute_select_with_window_functions_internal(
            stmt,
            ctx,
            base_rows,
            base_columns,
            &FxHashMap::default(),
            None,
            None,
            blocks,
        )
    }

    /// Window functions over aggregated rows, whose group items and aggregates
    /// are found by identity at `group_slots`
    pub(crate) fn execute_select_with_window_functions_over_groups(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        group_slots: &FxHashMap<ExprKey, usize>,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<WindowOutput> {
        self.execute_select_with_window_functions_internal(
            stmt,
            ctx,
            base_rows,
            base_columns,
            group_slots,
            None,
            None,
            blocks,
        )
    }

    /// Execute SELECT with window functions, with optional pre-sorted state
    /// When pre_sorted is Some, rows are already sorted by the specified column,
    /// allowing us to skip sorting for window functions that ORDER BY the same column
    pub(crate) fn execute_select_with_window_functions_presorted(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        pre_sorted: Option<WindowPreSortedState>,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<WindowOutput> {
        self.execute_select_with_window_functions_internal(
            stmt,
            ctx,
            base_rows,
            base_columns,
            &FxHashMap::default(),
            pre_sorted,
            None,
            blocks,
        )
    }

    /// Execute SELECT with window functions, with pre-grouped partitions
    /// When pre_grouped is provided, rows are already grouped by partition column,
    /// allowing us to skip hash-based grouping for window functions
    pub(crate) fn execute_select_with_window_functions_pregrouped(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        pre_grouped: WindowPreGroupedState,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<WindowOutput> {
        self.execute_select_with_window_functions_internal(
            stmt,
            ctx,
            base_rows,
            base_columns,
            &FxHashMap::default(),
            None,
            Some(pre_grouped),
            blocks,
        )
    }

    /// Internal implementation with pre-sorted and pre-grouped state parameters
    #[allow(clippy::too_many_arguments)]
    fn execute_select_with_window_functions_internal(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        group_slots: &FxHashMap<ExprKey, usize>,
        pre_sorted: Option<WindowPreSortedState>,
        pre_grouped: Option<WindowPreGroupedState>,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<WindowOutput> {
        // A window over a row reads its named window's parts
        if let Some(e) = blocks.and_then(|b| b.window_failure()) {
            if !base_rows.is_empty() {
                return Err(e.clone());
            }
        }
        // A window function in ORDER BY that the select list does not carry
        // is computed into a column of its own, named after it, which the
        // sort reads and the projection then drops. It goes in before the
        // subqueries are lifted, so one inside it is lifted with the rest
        let written_items = stmt.columns.len();
        let extended_stmt;
        let stmt = {
            let hidden = Self::hidden_order_by_windows(stmt);
            if hidden.is_empty() {
                stmt
            } else {
                let mut extended = stmt.clone();
                extended.columns.extend(hidden);
                extended_stmt = extended;
                &extended_stmt
            }
        };

        // A subquery inside a window function or its OVER clause is resolved
        // per input row first, unless the blocks answer it; the columns it
        // adds stay out of SELECT *
        let visible_columns = base_columns.len();
        let lifted_stmt;
        let lifted_rows;
        let lifted_columns;
        let (stmt, base_rows, base_columns) = match (blocks.is_none()
            && Self::windows_hold_subquery(stmt))
        .then(|| Self::lift_subqueries_in_windows(stmt, base_columns))
        .flatten()
        {
            Some((rewritten, lifted, names)) => {
                let qualifier = stmt
                    .table_expr
                    .as_deref()
                    .and_then(super::utils::get_table_alias_from_expr);
                let (rows, all_columns) = self.materialize_lifted_subqueries(
                    &lifted,
                    &names,
                    ctx,
                    base_rows,
                    base_columns,
                    qualifier.as_deref(),
                )?;
                lifted_stmt = rewritten;
                lifted_rows = rows;
                lifted_columns = all_columns;
                (
                    &lifted_stmt,
                    lifted_rows.as_slice(),
                    lifted_columns.as_slice(),
                )
            }
            None => (stmt, base_rows, base_columns),
        };

        // Parse window functions from the SELECT list
        let mut window_functions = self.parse_window_functions(stmt, base_columns, ctx)?;
        self.fold_window_control_arguments(&mut window_functions, ctx)?;

        if window_functions.is_empty() {
            // No window functions found, return base result
            let mut rows = RowVec::with_capacity(base_rows.len());
            for (id, row) in base_rows.iter() {
                rows.push((*id, row.clone()));
            }
            let layout = (!stmt.order_by.is_empty() || !stmt.distinct_on.is_empty())
                .then(|| OutputLayout::of_input(base_columns));
            return Ok((
                Box::new(ExecutorResult::new(base_columns.to_vec(), rows)),
                layout,
            ));
        }

        // OPTIMIZATION: LIMIT pushdown for PARTITION BY queries
        // Only safe when there is no top-level ORDER BY (otherwise the sort must see all
        // rows before LIMIT can be applied), when all window functions share the same
        // PARTITION BY so one partition map can serve them all, and when no OVER clause
        // is volatile (RANDOM and friends must draw per window function).
        let all_over_clauses_shareable = window_functions
            .iter()
            .all(|wf| self.over_clause_is_shareable(wf));
        // The streamed rows meet the LIMIT before DISTINCT runs, so DISTINCT keeps it off
        if all_over_clauses_shareable
            && stmt.order_by.is_empty()
            && stmt.offset.is_none()
            && !stmt.distinct
            && stmt.distinct_on.is_empty()
        {
            let has_partition_by = window_functions
                .iter()
                .any(|wf| !wf.partition_by_exprs.is_empty());

            if has_partition_by && Self::all_partitions_match(&window_functions) {
                if let Some(limit_val) = super::row_count::rows_needed(stmt, ctx) {
                    if limit_val > 0 {
                        // Streaming runs without an ORDER BY, which reads no layout
                        let result = self.execute_select_with_window_functions_streaming(
                            stmt,
                            ctx,
                            base_rows,
                            base_columns,
                            group_slots,
                            visible_columns,
                            &window_functions,
                            limit_val,
                            blocks,
                        )?;
                        return Ok((result, None));
                    }
                }
            }
        }

        // Build column index map for base columns
        let col_index_map = build_column_index_map(base_columns);

        // Step 1: Compute all window function values upfront
        // OPTIMIZATION: Use FxHashMap for fastest lookups with trusted keys

        // OPTIMIZATION: Precompute ORDER BY values ONCE for each unique ORDER BY clause
        // This avoids redundant computation when multiple window functions share the same ORDER BY
        // We use a Vec for the cache since the number of unique ORDER BY clauses is typically small
        // NOTE: We use string representation for semantic comparison because PartialEq on expressions
        // compares token positions, making structurally identical expressions from different window
        // functions appear different.
        let mut order_by_cache: Vec<(OrderByKey, ColumnarOrderByValues)> = Vec::new();
        for wf in &window_functions {
            if !wf.order_by.is_empty() {
                // Create a semantic key from the ORDER BY expressions (ignores token positions)
                let cache_key = Self::order_by_cache_key(&wf.order_by);
                // Check if this ORDER BY clause is already in the cache
                let already_cached = order_by_cache.iter().any(|(key, _)| key == &cache_key);
                if !already_cached {
                    let precomputed = self.precompute_order_by_values(
                        &wf.order_by,
                        base_rows,
                        base_columns,
                        &col_index_map,
                        group_slots,
                        ctx,
                        blocks,
                    )?;
                    order_by_cache.push((cache_key, precomputed));
                }
            }
        }

        let mut window_value_map: StringMap<Vec<Value>> = StringMap::new();
        // Sorted partitions shared across window functions with the same
        // OVER clause (K functions pay one map build and one sort)
        let mut partition_cache: Vec<(OverClauseKey, SharedSortedPartitions)> = Vec::new();
        for wf in &window_functions {
            let window_values = self.compute_window_function(
                wf,
                base_rows,
                base_columns,
                &col_index_map,
                group_slots,
                ctx,
                pre_sorted.as_ref(),
                pre_grouped.as_ref(),
                &order_by_cache,
                &mut partition_cache,
                blocks,
            )?;
            window_value_map.insert(wf.column_name.to_lowercase(), window_values);
        }

        // Step 2: Build output columns and rows based on the SELECT list
        // The result should respect the SELECT list order, not just append window functions
        let mut result_columns = Vec::new();

        // Parse the SELECT list to determine output column order
        let laid_out = !stmt.order_by.is_empty() || !stmt.distinct_on.is_empty();
        let mut item_of = Vec::new();
        let select_items = self.parse_select_list_for_window(
            stmt,
            base_columns,
            visible_columns,
            &window_functions,
            group_slots,
            laid_out.then_some(&mut item_of),
        );

        for item in &select_items {
            result_columns.push(item.output_name.clone());
        }

        // Step 3: Build result using COLUMNAR storage
        // OPTIMIZATION: Instead of allocating one Row per result row, we store data column-major
        // and use ColumnarResult which materializes rows lazily with a single reused buffer.
        // This reduces allocations from O(num_rows) to O(num_columns).
        let num_rows = base_rows.len();

        // Pre-transform expressions with window functions by replacing Window expr with Identifier
        // This is done once, not per row
        let transformed_items: Vec<_> = select_items
            .iter()
            .map(|item| match &item.source {
                SelectItemSource::ExpressionWithWindow(expr, wf_names) => {
                    let mut counter = 0;
                    let transformed =
                        Self::replace_windows_with_identifiers(expr, wf_names, &mut counter);
                    (item, Some(transformed), Some(wf_names.clone()))
                }
                _ => (item, None, None),
            })
            .collect();

        // Build extended columns (base_columns + synthetic window columns)
        let mut extended_columns = base_columns.to_vec();
        let mut added_wf_names: Vec<String> = Vec::new();
        for (_, _, wf_names_opt) in &transformed_items {
            if let Some(wf_names) = wf_names_opt {
                for wf_name in wf_names {
                    if !added_wf_names.contains(wf_name) {
                        extended_columns.push(wf_name.clone());
                        added_wf_names.push(wf_name.clone());
                    }
                }
            }
        }

        // Collect Expression items (with base_columns) and their indices
        let base_expr_items: Vec<(usize, &Expression)> = transformed_items
            .iter()
            .enumerate()
            .filter_map(|(i, (item, _, _))| {
                if let SelectItemSource::Expression(expr) = &item.source {
                    Some((i, expr))
                } else {
                    None
                }
            })
            .collect();

        // Collect ExpressionWithWindow transformed expressions and their indices
        let ext_expr_items: Vec<(usize, &Expression)> = transformed_items
            .iter()
            .enumerate()
            .filter_map(|(i, (_, transformed_opt, _))| transformed_opt.as_ref().map(|t| (i, t)))
            .collect();

        // Pre-compile base expressions (Expression items with base_columns)
        // CRITICAL: Propagate compilation errors instead of silently producing NULLs
        let base_exprs: Vec<Expression> =
            base_expr_items.iter().map(|(_, e)| (*e).clone()).collect();
        let mut base_eval = if !base_exprs.is_empty() {
            Some(RowPrograms::compile(
                &base_exprs,
                base_columns,
                Some(group_slots),
                ctx,
                blocks,
            )?)
        } else {
            None
        };

        // Pre-compile extended expressions (ExpressionWithWindow with extended_columns)
        // CRITICAL: Propagate compilation errors instead of silently producing NULLs
        let ext_exprs: Vec<Expression> = ext_expr_items.iter().map(|(_, e)| (*e).clone()).collect();
        let mut ext_eval = if !ext_exprs.is_empty() {
            Some(RowPrograms::compile(
                &ext_exprs,
                &extended_columns,
                Some(group_slots),
                ctx,
                blocks,
            )?)
        } else {
            None
        };

        // OPTIMIZATION: Pre-allocate ext_values buffer for extended expressions
        let ext_values_capacity = if !ext_expr_items.is_empty() {
            base_rows.first().map_or(0, |r| r.1.len()) + added_wf_names.len()
        } else {
            0
        };
        // One row buffer reused for every evaluated row. Handing the buffer to
        // the Row and taking it back with `mem::take` left a zero-capacity
        // vector behind, so each row paid two allocations to rebuild it.
        let mut ext_row = Row::from_compact_vec(CompactVec::with_capacity(ext_values_capacity));
        let mut ext_results: Vec<Value> = Vec::with_capacity(ext_expr_items.len());
        let mut base_results: Vec<Value> = Vec::with_capacity(base_expr_items.len());

        // Number of output columns
        let num_items = select_items.len();

        // COLUMNAR STORAGE OPTIMIZATION: Build columns in column-major order
        // This is more cache-efficient than row-by-row iteration and enables
        // moving window function Vecs directly (zero-copy for window results).
        //
        // Phase 1: Build non-expression columns (WindowFunction, BaseColumn)
        // Phase 2: Fill expression columns row-by-row (requires evaluation)
        let mut column_data: Vec<Vec<Value>> = Vec::with_capacity(num_items);

        // Track which window functions are used in expressions (need to keep them)
        let wf_names_in_exprs: std::collections::HashSet<&str> =
            added_wf_names.iter().map(|s| s.as_str()).collect();

        // Phase 1: Build columns for WindowFunction and BaseColumn items
        // Expression columns get placeholder Vecs (filled in Phase 2)
        for (item, _, _) in &transformed_items {
            match &item.source {
                SelectItemSource::WindowFunction(wf_name_lower) => {
                    // OPTIMIZATION: Move or clone the entire Vec at once
                    // If this window function is also used in an expression, we need to keep it
                    if wf_names_in_exprs.contains(wf_name_lower.as_str()) {
                        // Clone the entire Vec (more cache-efficient than element-by-element)
                        let values = window_value_map
                            .get(wf_name_lower)
                            .cloned()
                            .unwrap_or_else(|| vec![NULL_VALUE; num_rows]);
                        column_data.push(values);
                    } else {
                        // Move the Vec directly (zero-copy)
                        let values = window_value_map
                            .remove(wf_name_lower)
                            .unwrap_or_else(|| vec![NULL_VALUE; num_rows]);
                        column_data.push(values);
                    }
                }
                SelectItemSource::BaseColumn(base_col_idx) => {
                    // OPTIMIZATION: Build base column in one pass (column-wise)
                    let mut values = Vec::with_capacity(num_rows);
                    for (_, base_row) in base_rows {
                        values.push(base_row.get(*base_col_idx).cloned().unwrap_or(NULL_VALUE));
                    }
                    column_data.push(values);
                }
                SelectItemSource::Expression(_) | SelectItemSource::ExpressionWithWindow(_, _) => {
                    // Placeholder - will be filled in Phase 2
                    column_data.push(vec![NULL_VALUE; num_rows]);
                }
            }
        }

        // Phase 2: Fill expression columns row-by-row (requires evaluation)
        let has_expressions = !base_expr_items.is_empty() || !ext_expr_items.is_empty();
        if has_expressions {
            for (row_idx, (_, base_row)) in base_rows.iter().enumerate() {
                // Evaluate base expressions and update their column values
                if let Some(ref mut eval) = base_eval {
                    eval.eval_into(base_row, &mut base_results)?;
                    for (eval_idx, (item_idx, _)) in base_expr_items.iter().enumerate() {
                        if eval_idx < base_results.len() {
                            column_data[*item_idx][row_idx] = base_results[eval_idx].clone();
                        }
                    }
                }

                // Evaluate extended expressions (if any) and update their column values
                if !ext_expr_items.is_empty() {
                    if let Some(ref mut eval) = ext_eval {
                        ext_row.clear();
                        ext_row.extend_from_slice(base_row.as_slice());
                        for wf_name in &added_wf_names {
                            let wf_value = window_value_map
                                .get(wf_name)
                                .and_then(|vals| vals.get(row_idx).cloned())
                                .unwrap_or(NULL_VALUE);
                            ext_row.push(wf_value);
                        }

                        ext_results.clear();
                        eval.eval_into(&ext_row, &mut ext_results)?;
                        for (eval_idx, (item_idx, _)) in ext_expr_items.iter().enumerate() {
                            if eval_idx < ext_results.len() {
                                column_data[*item_idx][row_idx] = ext_results[eval_idx].clone();
                            }
                        }
                    }
                }
            }
        }

        // Every input column ORDER BY reads that no output column copies is
        // carried for the sort, which the expected-column trim then drops
        let (mut result_columns, mut column_data) = (result_columns, column_data);
        let mut sources: super::query::LayoutSources = select_items
            .iter()
            .zip(&item_of)
            .map(|(select_item, &item)| ColumnSource {
                item: (item < written_items).then_some(item),
                input: match select_item.source {
                    SelectItemSource::BaseColumn(idx) => Some(idx),
                    _ => None,
                },
            })
            .collect();
        for idx in order_by_inputs(stmt, base_columns) {
            if sources.iter().any(|source| source.input == Some(idx)) {
                continue;
            }
            result_columns.push(base_columns[idx].clone());
            column_data.push(
                base_rows
                    .iter()
                    .map(|(_, row)| row.get(idx).cloned().unwrap_or_else(Value::null_unknown))
                    .collect(),
            );
            sources.push(ColumnSource {
                item: None,
                input: Some(idx),
            });
        }
        let layout = laid_out.then(|| OutputLayout {
            input_columns: crate::common::CompactArc::new(base_columns.to_vec()),
            sources,
        });

        // Return ColumnarResult which materializes rows lazily with zero per-row allocation
        Ok((
            Box::new(ColumnarResult::new(result_columns, column_data)),
            layout,
        ))
    }

    /// Streaming execution for window functions with LIMIT pushdown
    /// Processes partitions one at a time and stops early when LIMIT is reached
    #[allow(clippy::too_many_arguments)]
    fn execute_select_with_window_functions_streaming(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        base_rows: &[(i64, Row)],
        base_columns: &[String],
        group_slots: &FxHashMap<ExprKey, usize>,
        visible_columns: usize,
        window_functions: &[WindowFunctionInfo],
        limit: usize,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Box<dyn QueryResult>> {
        // Use the first window function with PARTITION BY for partitioning
        let primary_wf = window_functions
            .iter()
            .find(|wf| !wf.partition_by_exprs.is_empty())
            .unwrap(); // Safe: we checked has_partition_by before calling this

        // Build column index map
        let col_index_map = build_column_index_map(base_columns);

        // Build partition map from the primary window function
        let partitions = Self::build_partition_map(
            primary_wf,
            base_rows,
            base_columns,
            &col_index_map,
            group_slots,
            ctx,
            blocks,
        )?;

        // Build result columns from SELECT list
        let select_items = self.parse_select_list_for_window(
            stmt,
            base_columns,
            visible_columns,
            window_functions,
            group_slots,
            None,
        );
        let result_columns: Vec<String> =
            select_items.iter().map(|i| i.output_name.clone()).collect();

        // Precompute ORDER BY values once for each unique ORDER BY clause
        let mut order_by_cache: Vec<(OrderByKey, ColumnarOrderByValues)> = Vec::new();
        for wf in window_functions {
            if !wf.order_by.is_empty() {
                let cache_key = Self::order_by_cache_key(&wf.order_by);
                let already_cached = order_by_cache.iter().any(|(key, _)| key == &cache_key);
                if !already_cached {
                    let precomputed = self.precompute_order_by_values(
                        &wf.order_by,
                        base_rows,
                        base_columns,
                        &col_index_map,
                        group_slots,
                        ctx,
                        blocks,
                    )?;
                    order_by_cache.push((cache_key, precomputed));
                }
            }
        }

        // Resolve window functions from registry
        let resolved_wfs: Vec<_> = window_functions
            .iter()
            .map(|wf| {
                let is_agg = self.function_registry.is_aggregate(&wf.name);
                let func = if is_agg {
                    None
                } else {
                    self.function_registry.get_window(&wf.name)
                };
                let mut args = CallArgs::of(wf, &self.function_registry);
                args.input = self.window_input(
                    wf,
                    base_rows,
                    base_columns,
                    &col_index_map,
                    group_slots,
                    ctx,
                    blocks,
                )?;
                Ok((wf, func, is_agg, args))
            })
            .collect::<Result<_>>()?;

        // Precompute aggregate window functions once over all rows (not per partition)
        let mut precomputed_agg: StringMap<Vec<Value>> = StringMap::new();
        let mut streaming_partition_cache: Vec<(OverClauseKey, SharedSortedPartitions)> =
            Vec::new();
        for (wf, _, is_agg, args) in &resolved_wfs {
            if *is_agg {
                let cache_key = Self::order_by_cache_key(&wf.order_by);
                let precomputed_order_by = order_by_cache
                    .iter()
                    .find(|(key, _)| key == &cache_key)
                    .map(|(_, v)| v);
                let agg_results = self.compute_aggregate_window_function(
                    wf,
                    base_rows,
                    base_columns,
                    &col_index_map,
                    group_slots,
                    ctx,
                    None,
                    precomputed_order_by,
                    &mut streaming_partition_cache,
                    args,
                    blocks,
                )?;
                precomputed_agg.insert(wf.column_name.to_lowercase(), agg_results);
            }
        }

        // Process partitions one at a time, stopping when we have enough rows
        // A user-supplied limit can be i64::MAX; only pre-allocate up
        // to the rows that can actually be produced
        let mut result_rows = RowVec::with_capacity(limit.min(base_rows.len().max(16)));
        let mut result_row_id = 0i64;

        let partitions_vec: Vec<_> = partitions.into_iter().collect();

        for (_partition_key, row_indices) in partitions_vec {
            if result_rows.len() >= limit {
                break;
            }

            // Compute window functions for this partition.
            let mut window_value_map: StringMap<Vec<Value>> = StringMap::new();

            for (wf, win_func_opt, is_agg, args) in &resolved_wfs {
                if *is_agg {
                    // Slice precomputed aggregate results for this partition
                    let key = wf.column_name.to_lowercase();
                    if let Some(all_results) = precomputed_agg.get(&key) {
                        let partition_vals: Vec<Value> = row_indices
                            .iter()
                            .map(|&i| all_results[i].clone())
                            .collect();
                        window_value_map.insert(key, partition_vals);
                    }
                } else if let Some(win_func) = win_func_opt {
                    let cache_key = Self::order_by_cache_key(&wf.order_by);
                    let precomputed_order_by = order_by_cache
                        .iter()
                        .find(|(key, _)| key == &cache_key)
                        .map(|(_, v)| v);
                    let (sorted_values, sorted_indices) = self.compute_window_for_partition(
                        win_func.as_ref(),
                        wf,
                        base_rows,
                        row_indices.clone(),
                        precomputed_order_by,
                        base_columns,
                        &col_index_map,
                        group_slots,
                        ctx,
                        args,
                        false,
                    )?;
                    // Remap: sorted_values[pos] corresponds to row sorted_indices[pos].
                    // Build a vec indexed by position within row_indices using an O(n) index map.
                    let orig_to_local: FxHashMap<usize, usize> = row_indices
                        .iter()
                        .enumerate()
                        .map(|(local, &orig)| (orig, local))
                        .collect();
                    let mut by_orig = vec![NULL_VALUE; row_indices.len()];
                    for (pos, &orig_idx) in sorted_indices.iter().enumerate() {
                        if let Some(&local) = orig_to_local.get(&orig_idx) {
                            by_orig[local] = sorted_values[pos].clone();
                        }
                    }
                    window_value_map.insert(wf.column_name.to_lowercase(), by_orig);
                }
            }

            // Precompile expression evaluators once per partition (not per row)
            enum CompiledItem<'a, 'p> {
                BaseColumn(usize),
                WindowFunction(&'a str),
                Expression(RowPrograms<'p>),
                ExpressionWithWindow(RowPrograms<'p>, Vec<&'a str>),
            }
            let num_items = select_items.len();
            let mut compiled_items: Vec<CompiledItem<'_, '_>> = Vec::with_capacity(num_items);
            for item in &select_items {
                compiled_items.push(match &item.source {
                    SelectItemSource::BaseColumn(idx) => CompiledItem::BaseColumn(*idx),
                    SelectItemSource::WindowFunction(name) => {
                        CompiledItem::WindowFunction(name.as_str())
                    }
                    SelectItemSource::Expression(expr) => CompiledItem::Expression(
                        RowPrograms::compile([expr], base_columns, None, ctx, blocks)?,
                    ),
                    SelectItemSource::ExpressionWithWindow(expr, wf_names) => {
                        let mut counter = 0;
                        let transformed =
                            Self::replace_windows_with_identifiers(expr, wf_names, &mut counter);
                        let mut ext_columns = base_columns.to_vec();
                        for wf_name in wf_names {
                            ext_columns.push(wf_name.clone());
                        }
                        let eval =
                            RowPrograms::compile([&transformed], &ext_columns, None, ctx, blocks)?;
                        let name_refs: Vec<&str> = wf_names.iter().map(|s| s.as_str()).collect();
                        CompiledItem::ExpressionWithWindow(eval, name_refs)
                    }
                });
            }

            let ext_capacity = base_rows.first().map_or(0, |r| r.1.len()) + 1;
            // Reused for every row, see the note on the other window path.
            let mut ext_row = Row::from_compact_vec(CompactVec::with_capacity(ext_capacity));

            // Output rows in partition order (row_indices order)
            for (local_pos, &orig_idx) in row_indices.iter().enumerate() {
                if result_rows.len() >= limit {
                    break;
                }

                let base_row = &base_rows[orig_idx].1;
                let mut values: CompactVec<Value> = CompactVec::with_capacity(num_items);

                for ci in &mut compiled_items {
                    let value = match ci {
                        CompiledItem::BaseColumn(col_idx) => {
                            base_row.get(*col_idx).cloned().unwrap_or(NULL_VALUE)
                        }
                        CompiledItem::WindowFunction(wf_name_lower) => window_value_map
                            .get(*wf_name_lower)
                            .and_then(|vals| vals.get(local_pos).cloned())
                            .unwrap_or(NULL_VALUE),
                        CompiledItem::Expression(eval) => eval.eval(base_row)?,
                        CompiledItem::ExpressionWithWindow(eval, wf_name_refs) => {
                            ext_row.clear();
                            ext_row.extend_from_slice(base_row.as_slice());
                            for wf_name in wf_name_refs.iter() {
                                let wf_value = window_value_map
                                    .get(*wf_name)
                                    .and_then(|vals| vals.get(local_pos).cloned())
                                    .unwrap_or(NULL_VALUE);
                                ext_row.push(wf_value);
                            }
                            eval.eval(&ext_row)?
                        }
                    };
                    values.push(value);
                }
                result_rows.push((result_row_id, Row::from_compact_vec(values)));
                result_row_id += 1;
            }
        }

        Ok(Box::new(ExecutorResult::new(result_columns, result_rows)))
    }

    /// Lazy partition fetching for window functions with LIMIT pushdown
    /// Fetches partitions one at a time from the index and stops when LIMIT is reached
    /// This is the key optimization for PARTITION BY + LIMIT queries
    pub fn execute_select_with_window_functions_lazy_partition(
        &self,
        stmt: &SelectStatement,
        ctx: &ExecutionContext,
        table: &dyn Table,
        base_columns: &[String],
        partition_col: &str,
        limit: usize,
    ) -> Result<Option<Box<dyn QueryResult>>> {
        // Parse window functions from SELECT list
        let mut window_functions = self.parse_window_functions(stmt, base_columns, ctx)?;
        self.fold_window_control_arguments(&mut window_functions, ctx)?;
        if window_functions.is_empty() {
            return Ok(None);
        }

        // Build column index map
        let col_index_map = build_column_index_map(base_columns);
        // Rows read straight from the table: nothing aggregated precedes the window
        let group_slots = FxHashMap::default();

        // Resolve window functions from registry
        let mut resolved_wfs: Vec<_> = window_functions
            .iter()
            .map(|wf| {
                let is_agg = self.function_registry.is_aggregate(&wf.name);
                let func = if is_agg {
                    None
                } else {
                    self.function_registry.get_window(&wf.name)
                };
                (wf, func, is_agg, CallArgs::of(wf, &self.function_registry))
            })
            .collect();

        // Build result columns from SELECT list
        let select_items = self.parse_select_list_for_window(
            stmt,
            base_columns,
            base_columns.len(),
            &window_functions,
            &group_slots,
            None,
        );
        let result_columns: Vec<String> =
            select_items.iter().map(|i| i.output_name.clone()).collect();

        // Get partition values from the index (lazy iteration key!)
        let partition_values = match table.get_partition_values(partition_col)? {
            Some(values) => values,
            None => return Ok(None),
        };

        // Process partitions one at a time, stopping when we have enough rows
        // Process partitions one at a time, stopping when we have enough
        // rows. A user-supplied limit can be i64::MAX, so cap the
        // pre-allocation; rows are fetched lazily per partition
        let mut result_rows = RowVec::with_capacity(limit.min(4096));
        let mut result_row_id = 0i64;

        for partition_value in partition_values {
            if result_rows.len() >= limit {
                break;
            }

            // Fetch rows for this partition only (KEY OPTIMIZATION!); a
            // table that cannot answer for one leaves the query to the scan
            let partition_rows =
                match table.get_rows_for_partition_value(partition_col, &partition_value)? {
                    Some(rows) => rows,
                    None => return Ok(None),
                };

            if partition_rows.is_empty() {
                continue;
            }

            // Precompute ORDER BY values for each unique ORDER BY clause
            let mut order_by_cache: Vec<(OrderByKey, ColumnarOrderByValues)> = Vec::new();
            for wf in &window_functions {
                if !wf.order_by.is_empty() {
                    let cache_key = Self::order_by_cache_key(&wf.order_by);
                    let already_cached = order_by_cache.iter().any(|(key, _)| key == &cache_key);
                    if !already_cached {
                        let precomputed = self.precompute_order_by_values(
                            &wf.order_by,
                            &partition_rows,
                            base_columns,
                            &col_index_map,
                            &group_slots,
                            ctx,
                            None,
                        )?;
                        order_by_cache.push((cache_key, precomputed));
                    }
                }
            }

            // Compute ALL window functions for this partition.
            // Values are stored keyed by local position within row_indices.
            let row_indices: Vec<usize> = (0..partition_rows.len()).collect();
            let mut window_value_map: StringMap<Vec<Value>> = StringMap::new();
            // Per-partition rows: the shared cache would key on the OVER
            // clause while indices differ per partition, so keep it local
            let mut lazy_partition_cache: Vec<(OverClauseKey, SharedSortedPartitions)> = Vec::new();

            for (wf, win_func_opt, is_agg, args) in &mut resolved_wfs {
                let cache_key = Self::order_by_cache_key(&wf.order_by);
                let precomputed_order_by = order_by_cache
                    .iter()
                    .find(|(key, _)| key == &cache_key)
                    .map(|(_, v)| v);
                args.input = self.window_input(
                    wf,
                    &partition_rows,
                    base_columns,
                    &col_index_map,
                    &group_slots,
                    ctx,
                    None,
                )?;

                if *is_agg {
                    let agg_results = self.compute_aggregate_window_function(
                        wf,
                        &partition_rows,
                        base_columns,
                        &col_index_map,
                        &group_slots,
                        ctx,
                        None,
                        precomputed_order_by,
                        &mut lazy_partition_cache,
                        args,
                        None,
                    )?;
                    // agg_results is already indexed by local row index
                    window_value_map.insert(wf.column_name.to_lowercase(), agg_results);
                } else if let Some(win_func) = win_func_opt {
                    let (sorted_values, sorted_indices) = self.compute_window_for_partition(
                        win_func.as_ref(),
                        wf,
                        &partition_rows,
                        row_indices.clone(),
                        precomputed_order_by,
                        base_columns,
                        &col_index_map,
                        &group_slots,
                        ctx,
                        args,
                        false,
                    )?;
                    // Remap from sorted order to local row order
                    let mut by_local = vec![NULL_VALUE; row_indices.len()];
                    for (pos, &local_idx) in sorted_indices.iter().enumerate() {
                        by_local[local_idx] = sorted_values[pos].clone();
                    }
                    window_value_map.insert(wf.column_name.to_lowercase(), by_local);
                }
            }

            // Precompile expression evaluators once per partition (not per row)
            enum CompiledItemLazy<'a, 'p> {
                BaseColumn(usize),
                WindowFunction(&'a str),
                Expression(RowPrograms<'p>),
                ExpressionWithWindow(RowPrograms<'p>, Vec<&'a str>),
            }
            let num_items = select_items.len();
            let mut compiled_items: Vec<CompiledItemLazy<'_, '_>> = Vec::with_capacity(num_items);
            for item in &select_items {
                compiled_items.push(match &item.source {
                    SelectItemSource::BaseColumn(idx) => CompiledItemLazy::BaseColumn(*idx),
                    SelectItemSource::WindowFunction(name) => {
                        CompiledItemLazy::WindowFunction(name.as_str())
                    }
                    SelectItemSource::Expression(expr) => CompiledItemLazy::Expression(
                        RowPrograms::compile([expr], base_columns, None, ctx, None)?,
                    ),
                    SelectItemSource::ExpressionWithWindow(expr, wf_names) => {
                        let mut counter = 0;
                        let transformed =
                            Self::replace_windows_with_identifiers(expr, wf_names, &mut counter);
                        let mut ext_columns = base_columns.to_vec();
                        for wf_name in wf_names {
                            ext_columns.push(wf_name.clone());
                        }
                        let eval =
                            RowPrograms::compile([&transformed], &ext_columns, None, ctx, None)?;
                        let name_refs: Vec<&str> = wf_names.iter().map(|s| s.as_str()).collect();
                        CompiledItemLazy::ExpressionWithWindow(eval, name_refs)
                    }
                });
            }

            // Reused for every row, see the note on the other window path.
            let mut ext_row = Row::from_compact_vec(CompactVec::with_capacity(
                partition_rows.first().map_or(0, |r| r.1.len()) + 1,
            ));

            // Output rows in partition order
            for (local_pos, &row_idx) in row_indices.iter().enumerate() {
                if result_rows.len() >= limit {
                    break;
                }

                let (_, base_row) = &partition_rows[row_idx];
                let mut values: CompactVec<Value> = CompactVec::with_capacity(num_items);

                for ci in &mut compiled_items {
                    let val = match ci {
                        CompiledItemLazy::BaseColumn(col_idx) => {
                            base_row.get(*col_idx).cloned().unwrap_or(NULL_VALUE)
                        }
                        CompiledItemLazy::WindowFunction(wf_name_lower) => window_value_map
                            .get(*wf_name_lower)
                            .and_then(|vals| vals.get(local_pos).cloned())
                            .unwrap_or(NULL_VALUE),
                        CompiledItemLazy::Expression(eval) => eval.eval(base_row)?,
                        CompiledItemLazy::ExpressionWithWindow(eval, wf_name_refs) => {
                            ext_row.clear();
                            ext_row.extend_from_slice(base_row.as_slice());
                            for wf_name in wf_name_refs.iter() {
                                let wf_value = window_value_map
                                    .get(*wf_name)
                                    .and_then(|vals| vals.get(local_pos).cloned())
                                    .unwrap_or(NULL_VALUE);
                                ext_row.push(wf_value);
                            }
                            eval.eval(&ext_row)?
                        }
                    };
                    values.push(val);
                }
                result_rows.push((result_row_id, Row::from_compact_vec(values)));
                result_row_id += 1;
            }
        }

        Ok(Some(Box::new(ExecutorResult::new(
            result_columns,
            result_rows,
        ))))
    }

    /// Parse SELECT list to determine output order and sources, and into
    /// `item_of` the SELECT item each output column comes from
    fn parse_select_list_for_window(
        &self,
        stmt: &SelectStatement,
        base_columns: &[String],
        visible_columns: usize,
        window_functions: &[WindowFunctionInfo],
        group_slots: &FxHashMap<ExprKey, usize>,
        mut item_of: Option<&mut Vec<usize>>,
    ) -> Vec<SelectItem> {
        let col_index_map = build_column_index_map(base_columns);

        let mut items = Vec::new();
        let mut wf_idx = 0;

        for (item, col_expr) in stmt.columns.iter().enumerate() {
            match col_expr {
                Expression::Window(_) => {
                    // Window function - use pre-computed values, named as written
                    let output_name = super::utils::expression_to_string(col_expr);
                    let key = if wf_idx < window_functions.len() {
                        window_functions[wf_idx].column_name.to_lowercase()
                    } else {
                        output_name.to_lowercase()
                    };
                    items.push(SelectItem {
                        output_name,
                        source: SelectItemSource::WindowFunction(key),
                    });
                    wf_idx += 1;
                }
                Expression::Aliased(aliased) => {
                    if let Expression::Window(_) = aliased.expression.as_ref() {
                        // Aliased window function
                        let key = if wf_idx < window_functions.len() {
                            window_functions[wf_idx].column_name.to_lowercase()
                        } else {
                            aliased.alias.value.to_string().to_lowercase()
                        };
                        items.push(SelectItem {
                            output_name: aliased.alias.value.to_string(),
                            source: SelectItemSource::WindowFunction(key),
                        });
                        wf_idx += 1;
                    } else if let Expression::Identifier(id) = aliased.expression.as_ref() {
                        // Aliased column reference - use pre-computed lowercase
                        if let Some(&idx) = col_index_map.get(id.value_lower.as_str()) {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.to_string(),
                                source: SelectItemSource::BaseColumn(idx),
                            });
                        } else {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.to_string(),
                                source: SelectItemSource::Expression(
                                    aliased.expression.as_ref().clone(),
                                ),
                            });
                        }
                    } else {
                        // Collect all embedded window functions
                        let mut embedded_windows = Vec::new();
                        Self::collect_all_windows_in_expression(
                            aliased.expression.as_ref(),
                            &mut embedded_windows,
                        );
                        if !embedded_windows.is_empty() {
                            let mut wf_names = Vec::with_capacity(embedded_windows.len());
                            for _ in &embedded_windows {
                                let name = if wf_idx < window_functions.len() {
                                    window_functions[wf_idx].column_name.to_lowercase()
                                } else {
                                    format!("__wf_{}", wf_idx)
                                };
                                wf_names.push(name);
                                wf_idx += 1;
                            }
                            items.push(SelectItem {
                                output_name: aliased.alias.value.to_string(),
                                source: SelectItemSource::ExpressionWithWindow(
                                    aliased.expression.as_ref().clone(),
                                    wf_names,
                                ),
                            });
                        } else if let Expression::FunctionCall(_) = aliased.expression.as_ref() {
                            // An aggregate computed before the window reads its slot
                            if let Some(&idx) = group_slots.get(&ExprKey::of(&aliased.expression)) {
                                items.push(SelectItem {
                                    output_name: aliased.alias.value.to_string(),
                                    source: SelectItemSource::BaseColumn(idx),
                                });
                            } else {
                                items.push(SelectItem {
                                    output_name: aliased.alias.value.to_string(),
                                    source: SelectItemSource::Expression(
                                        aliased.expression.as_ref().clone(),
                                    ),
                                });
                            }
                        } else {
                            items.push(SelectItem {
                                output_name: aliased.alias.value.to_string(),
                                source: SelectItemSource::Expression(
                                    aliased.expression.as_ref().clone(),
                                ),
                            });
                        }
                    }
                }
                Expression::Identifier(id) => {
                    // Simple column reference - use pre-computed lowercase
                    if let Some(&idx) = col_index_map.get(id.value_lower.as_str()) {
                        items.push(SelectItem {
                            output_name: id.value.to_string(),
                            source: SelectItemSource::BaseColumn(idx),
                        });
                    } else {
                        // Column not found directly - try to match against qualified columns (e.g., "name" matches "e.name")
                        // This handles JOIN cases where columns have table prefixes
                        let mut found = false;
                        for (col_name, &idx) in &col_index_map {
                            if split_qualified(col_name)
                                .is_some_and(|(_, name)| name == id.value_lower.as_str())
                            {
                                items.push(SelectItem {
                                    output_name: id.value.to_string(),
                                    source: SelectItemSource::BaseColumn(idx),
                                });
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            // Still not found - treat as expression and let evaluator handle it
                            items.push(SelectItem {
                                output_name: id.value.to_string(),
                                source: SelectItemSource::Expression(col_expr.clone()),
                            });
                        }
                    }
                }
                Expression::QualifiedIdentifier(qid) => {
                    let (qualifier, name) = (&qid.qualifier.value_lower, &qid.name.value_lower);
                    if let Some(idx) = qualified_index(&col_index_map, qualifier, name) {
                        items.push(SelectItem {
                            output_name: item_name(col_expr),
                            source: SelectItemSource::BaseColumn(idx),
                        });
                    } else {
                        // Column not found - treat as expression and let evaluator handle it
                        items.push(SelectItem {
                            output_name: item_name(col_expr),
                            source: SelectItemSource::Expression(col_expr.clone()),
                        });
                    }
                }
                Expression::Star(_) => {
                    // SELECT * - include all base columns
                    for (idx, col) in base_columns.iter().take(visible_columns).enumerate() {
                        items.push(SelectItem {
                            output_name: col.clone(),
                            source: SelectItemSource::BaseColumn(idx),
                        });
                    }
                }
                Expression::QualifiedStar(qs) => {
                    let table_alias = stmt
                        .table_expr
                        .as_deref()
                        .and_then(get_table_alias_from_expr);
                    let visible = &base_columns[..visible_columns.min(base_columns.len())];
                    for_each_qualified_star_column(
                        visible,
                        &qs.qualifier,
                        table_alias.as_deref(),
                        |idx, start| {
                            items.push(SelectItem {
                                output_name: visible[idx][start..].to_string(),
                                source: SelectItemSource::BaseColumn(idx),
                            })
                        },
                    );
                }
                Expression::FunctionCall(func) => {
                    // First check for embedded window functions (e.g., ABS(ROW_NUMBER() OVER (...)))
                    let mut embedded_windows = Vec::new();
                    Self::collect_all_windows_in_expression(col_expr, &mut embedded_windows);
                    if !embedded_windows.is_empty() {
                        let mut wf_names = Vec::with_capacity(embedded_windows.len());
                        for _ in &embedded_windows {
                            let name = if wf_idx < window_functions.len() {
                                window_functions[wf_idx].column_name.to_lowercase()
                            } else {
                                format!("__wf_{}", wf_idx)
                            };
                            wf_names.push(name);
                            wf_idx += 1;
                        }
                        items.push(SelectItem {
                            output_name: item_name(col_expr),
                            source: SelectItemSource::ExpressionWithWindow(
                                col_expr.clone(),
                                wf_names,
                            ),
                        });
                    } else {
                        // An aggregate computed before the window reads its slot
                        let source = match group_slots.get(&ExprKey::of_call(func)) {
                            Some(&idx) => SelectItemSource::BaseColumn(idx),
                            None => SelectItemSource::Expression(col_expr.clone()),
                        };
                        items.push(SelectItem {
                            output_name: item_name(col_expr),
                            source,
                        });
                    }
                }
                _ => {
                    // Other expressions - collect all embedded window functions
                    let mut embedded_windows = Vec::new();
                    Self::collect_all_windows_in_expression(col_expr, &mut embedded_windows);
                    if !embedded_windows.is_empty() {
                        let mut wf_names = Vec::with_capacity(embedded_windows.len());
                        for _ in &embedded_windows {
                            let name = if wf_idx < window_functions.len() {
                                window_functions[wf_idx].column_name.to_lowercase()
                            } else {
                                format!("__wf_{}", wf_idx)
                            };
                            wf_names.push(name);
                            wf_idx += 1;
                        }
                        items.push(SelectItem {
                            output_name: item_name(col_expr),
                            source: SelectItemSource::ExpressionWithWindow(
                                col_expr.clone(),
                                wf_names,
                            ),
                        });
                    } else {
                        items.push(SelectItem {
                            output_name: item_name(col_expr),
                            source: SelectItemSource::Expression(col_expr.clone()),
                        });
                    }
                }
            }
            if let Some(item_of) = item_of.as_deref_mut() {
                item_of.resize(items.len(), item);
            }
        }

        items
    }

    /// Parse window functions from SELECT list.
    /// Collects ALL window functions, including multiple ones nested in a single expression.
    fn parse_window_functions(
        &self,
        stmt: &SelectStatement,
        _base_columns: &[String],
        ctx: &ExecutionContext,
    ) -> Result<Vec<WindowFunctionInfo>> {
        let mut window_functions = Vec::new();
        let mut col_idx = 0;

        for col_expr in &stmt.columns {
            match col_expr {
                Expression::Window(window_expr) => {
                    let mut wf_info =
                        self.extract_window_function_info(window_expr, &stmt.window_defs, ctx)?;
                    // Two unaliased windows of the same function would share a
                    // name; the internal one is unique, the output keeps the
                    // function's own
                    wf_info.column_name = format!("__wf_{}_0", col_idx);
                    window_functions.push(wf_info);
                    col_idx += 1;
                }
                Expression::Aliased(aliased) => {
                    if let Expression::Window(window_expr) = aliased.expression.as_ref() {
                        let mut wf_info =
                            self.extract_window_function_info(window_expr, &stmt.window_defs, ctx)?;
                        // The alias is only the output name; the internal name
                        // cannot collide with an alias or another window
                        wf_info.column_name = format!("__wf_{}_0", col_idx);
                        window_functions.push(wf_info);
                    } else {
                        // Collect ALL window functions in the expression
                        let mut windows = Vec::new();
                        Self::collect_all_windows_in_expression(
                            aliased.expression.as_ref(),
                            &mut windows,
                        );
                        for (i, window_expr) in windows.into_iter().enumerate() {
                            let mut wf_info = self.extract_window_function_info(
                                window_expr,
                                &stmt.window_defs,
                                ctx,
                            )?;
                            wf_info.column_name = format!("__wf_{}_{}", col_idx, i);
                            window_functions.push(wf_info);
                        }
                    }
                    col_idx += 1;
                }
                _ => {
                    let mut windows = Vec::new();
                    Self::collect_all_windows_in_expression(col_expr, &mut windows);
                    for (i, window_expr) in windows.into_iter().enumerate() {
                        let mut wf_info =
                            self.extract_window_function_info(window_expr, &stmt.window_defs, ctx)?;
                        wf_info.column_name = format!("__wf_{}_{}", col_idx, i);
                        window_functions.push(wf_info);
                    }
                    col_idx += 1;
                }
            }
        }

        Ok(window_functions)
    }

    /// The named windows of `stmt` as its windows read them: the parts an
    /// OVER resolves to, the others emptied, and a window no OVER names left out
    pub(crate) fn used_window_defs(stmt: &SelectStatement) -> Vec<WindowDefinition> {
        if stmt.window_defs.is_empty() {
            return Vec::new();
        }
        let mut windows = Vec::new();
        let read = stmt
            .columns
            .iter()
            .chain(stmt.order_by.iter().map(|ob| &ob.expression))
            .chain(&stmt.distinct_on);
        for expr in read {
            Self::collect_all_windows_in_expression(expr, &mut windows);
        }
        // Whether each definition is named, and its ORDER BY and frame read
        let mut used = vec![(false, false, false); stmt.window_defs.len()];
        for window in windows {
            let (Some((at, def)), Some((_, order_by, frame))) = (
                window.named(&stmt.window_defs),
                window.resolved(&stmt.window_defs),
            ) else {
                continue;
            };
            let read = &mut used[at];
            read.0 = true;
            read.1 |= std::ptr::eq(order_by, def.order_by.as_slice());
            read.2 |= frame.is_some_and(|f| def.frame.as_ref().is_some_and(|d| std::ptr::eq(f, d)));
        }
        stmt.window_defs
            .iter()
            .zip(used)
            .filter(|(_, (named, ..))| *named)
            .map(|(def, (_, order_by, frame))| WindowDefinition {
                name: def.name.clone(),
                partition_by: def.partition_by.clone(),
                order_by: if order_by {
                    def.order_by.clone()
                } else {
                    Vec::new()
                },
                frame: def.frame.clone().filter(|_| frame),
            })
            .collect()
    }

    /// Recursively find a window expression inside an expression
    /// Returns the first Window expression found, if any
    /// Collect all window expressions found anywhere in an expression tree (depth-first order).
    fn collect_all_windows_in_expression<'a>(
        expr: &'a Expression,
        out: &mut Vec<&'a WindowExpression>,
    ) {
        match expr {
            Expression::Window(w) => out.push(w),
            Expression::Infix(infix) => {
                Self::collect_all_windows_in_expression(&infix.left, out);
                Self::collect_all_windows_in_expression(&infix.right, out);
            }
            Expression::Prefix(prefix) => {
                Self::collect_all_windows_in_expression(&prefix.right, out);
            }
            Expression::FunctionCall(f) => {
                for arg in &f.arguments {
                    Self::collect_all_windows_in_expression(arg, out);
                }
            }
            Expression::Aliased(a) => {
                Self::collect_all_windows_in_expression(&a.expression, out);
            }
            Expression::Cast(cast) => {
                Self::collect_all_windows_in_expression(&cast.expr, out);
            }
            Expression::Distinct(d) => {
                Self::collect_all_windows_in_expression(&d.expr, out);
            }
            Expression::Case(c) => {
                if let Some(v) = &c.value {
                    Self::collect_all_windows_in_expression(v, out);
                }
                for clause in &c.when_clauses {
                    Self::collect_all_windows_in_expression(&clause.condition, out);
                    Self::collect_all_windows_in_expression(&clause.then_result, out);
                }
                if let Some(else_val) = &c.else_value {
                    Self::collect_all_windows_in_expression(else_val, out);
                }
            }
            Expression::Between(b) => {
                Self::collect_all_windows_in_expression(&b.expr, out);
                Self::collect_all_windows_in_expression(&b.lower, out);
                Self::collect_all_windows_in_expression(&b.upper, out);
            }
            Expression::In(i) => {
                Self::collect_all_windows_in_expression(&i.left, out);
                Self::collect_all_windows_in_expression(&i.right, out);
            }
            Expression::QuantifiedSet(set) => {
                Self::collect_all_windows_in_expression(&set.left, out);
            }
            Expression::Like(l) => {
                Self::collect_all_windows_in_expression(&l.left, out);
                Self::collect_all_windows_in_expression(&l.pattern, out);
                if let Some(esc) = &l.escape {
                    Self::collect_all_windows_in_expression(esc, out);
                }
            }
            Expression::List(l) => {
                for elem in &l.elements {
                    Self::collect_all_windows_in_expression(elem, out);
                }
            }
            Expression::ExpressionList(l) => {
                for elem in &l.expressions {
                    Self::collect_all_windows_in_expression(elem, out);
                }
            }
            _ => {}
        }
    }

    /// Replace window expressions in an expression tree with identifier references.
    /// Each Window node is replaced with a unique placeholder from `wf_names` in
    /// depth-first order (matching `collect_all_windows_in_expression`).
    fn replace_windows_with_identifiers(
        expr: &Expression,
        wf_names: &[String],
        counter: &mut usize,
    ) -> Expression {
        use crate::parser::token::{Position, Token, TokenType};

        match expr {
            Expression::Window(_) => {
                let name = wf_names
                    .get(*counter)
                    .map_or("__wf_unknown", |s| s.as_str());
                *counter += 1;
                let dummy_token = Token::new(TokenType::Identifier, name, Position::new(0, 0, 0));
                Expression::Identifier(Identifier::new(dummy_token, name.to_string()))
            }
            Expression::Infix(infix) => Expression::Infix(InfixExpression {
                token: infix.token.clone(),
                left: Box::new(Self::replace_windows_with_identifiers(
                    &infix.left,
                    wf_names,
                    counter,
                )),
                operator: infix.operator.clone(),
                op_type: infix.op_type,
                right: Box::new(Self::replace_windows_with_identifiers(
                    &infix.right,
                    wf_names,
                    counter,
                )),
            }),
            Expression::Prefix(prefix) => Expression::Prefix(PrefixExpression {
                token: prefix.token.clone(),
                operator: prefix.operator.clone(),
                op_type: prefix.op_type,
                right: Box::new(Self::replace_windows_with_identifiers(
                    &prefix.right,
                    wf_names,
                    counter,
                )),
            }),
            Expression::FunctionCall(f) => Expression::FunctionCall(Box::new(FunctionCall {
                token: f.token.clone(),
                function: f.function.clone(),
                arguments: f
                    .arguments
                    .iter()
                    .map(|a| Self::replace_windows_with_identifiers(a, wf_names, counter))
                    .collect(),
                is_distinct: f.is_distinct,
                order_by: f.order_by.clone(),
                filter: f.filter.clone(),
            })),
            Expression::Aliased(a) => Expression::Aliased(AliasedExpression {
                token: a.token.clone(),
                expression: Box::new(Self::replace_windows_with_identifiers(
                    &a.expression,
                    wf_names,
                    counter,
                )),
                alias: a.alias.clone(),
            }),
            Expression::Cast(cast) => Expression::Cast(CastExpression {
                token: cast.token.clone(),
                expr: Box::new(Self::replace_windows_with_identifiers(
                    &cast.expr, wf_names, counter,
                )),
                type_name: cast.type_name.clone(),
            }),
            Expression::Distinct(d) => Expression::Distinct(DistinctExpression {
                token: d.token.clone(),
                expr: Box::new(Self::replace_windows_with_identifiers(
                    &d.expr, wf_names, counter,
                )),
            }),
            Expression::Case(case) => {
                let new_value = case.value.as_ref().map(|v| {
                    Box::new(Self::replace_windows_with_identifiers(v, wf_names, counter))
                });
                let new_whens: Vec<WhenClause> = case
                    .when_clauses
                    .iter()
                    .map(|w| WhenClause {
                        token: w.token.clone(),
                        condition: Self::replace_windows_with_identifiers(
                            &w.condition,
                            wf_names,
                            counter,
                        ),
                        then_result: Self::replace_windows_with_identifiers(
                            &w.then_result,
                            wf_names,
                            counter,
                        ),
                    })
                    .collect();
                let new_else = case.else_value.as_ref().map(|e| {
                    Box::new(Self::replace_windows_with_identifiers(e, wf_names, counter))
                });
                Expression::Case(Box::new(CaseExpression {
                    token: case.token.clone(),
                    value: new_value,
                    when_clauses: new_whens,
                    else_value: new_else,
                }))
            }
            Expression::Between(b) => Expression::Between(BetweenExpression {
                token: b.token.clone(),
                expr: Box::new(Self::replace_windows_with_identifiers(
                    &b.expr, wf_names, counter,
                )),
                lower: Box::new(Self::replace_windows_with_identifiers(
                    &b.lower, wf_names, counter,
                )),
                upper: Box::new(Self::replace_windows_with_identifiers(
                    &b.upper, wf_names, counter,
                )),
                not: b.not,
            }),
            Expression::In(i) => Expression::In(InExpression {
                token: i.token.clone(),
                left: Box::new(Self::replace_windows_with_identifiers(
                    &i.left, wf_names, counter,
                )),
                right: Box::new(Self::replace_windows_with_identifiers(
                    &i.right, wf_names, counter,
                )),
                not: i.not,
            }),
            Expression::QuantifiedSet(set) => set.with_left(
                Self::replace_windows_with_identifiers(&set.left, wf_names, counter),
            ),
            Expression::Like(l) => Expression::Like(LikeExpression {
                token: l.token.clone(),
                left: Box::new(Self::replace_windows_with_identifiers(
                    &l.left, wf_names, counter,
                )),
                pattern: Box::new(Self::replace_windows_with_identifiers(
                    &l.pattern, wf_names, counter,
                )),
                operator: l.operator.clone(),
                escape: l.escape.as_ref().map(|e| {
                    Box::new(Self::replace_windows_with_identifiers(e, wf_names, counter))
                }),
            }),
            Expression::List(l) => Expression::List(Box::new(ListExpression {
                token: l.token.clone(),
                elements: l
                    .elements
                    .iter()
                    .map(|e| Self::replace_windows_with_identifiers(e, wf_names, counter))
                    .collect(),
            })),
            Expression::ExpressionList(l) => Expression::ExpressionList(Box::new(ExpressionList {
                token: l.token.clone(),
                expressions: l
                    .expressions
                    .iter()
                    .map(|e| Self::replace_windows_with_identifiers(e, wf_names, counter))
                    .collect(),
            })),
            _ => expr.clone(),
        }
    }

    /// Extract window function info from WindowExpression
    fn extract_window_function_info(
        &self,
        window_expr: &WindowExpression,
        window_defs: &[WindowDefinition],
        ctx: &ExecutionContext,
    ) -> Result<WindowFunctionInfo> {
        let func = &window_expr.function;

        let (partition_by_exprs, order_by, frame) = match window_expr.resolved(window_defs) {
            Some((partition_by, order_by, frame)) => {
                (partition_by.to_vec(), order_by.to_vec(), frame.cloned())
            }
            None => {
                let name = window_expr.window_ref.as_deref().unwrap_or_default();
                return Err(Error::NotSupported(format!(
                    "Unknown window name: {}",
                    name
                )));
            }
        };

        // Extract partition by column names (identifiers only, for fast-path)
        // For qualified identifiers (e.g., l.grp), keep the full qualified name
        // to properly match columns from JOINs
        let partition_by: Vec<String> = partition_by_exprs
            .iter()
            .filter_map(|e| match e {
                Expression::Identifier(id) => Some(id.value.to_string()),
                Expression::QualifiedIdentifier(qid) => {
                    Some(qualified(&qid.qualifier.value, &qid.name.value))
                }
                _ => None,
            })
            .collect();

        // Build column name
        let column_name = format!("{}()", func.function);
        let (extra_args, extra_error) = if self.function_registry.is_aggregate(&func.function) {
            super::aggregation::aggregate_extra_args(&func.arguments, Some(ctx))?
        } else {
            (Vec::new(), None)
        };

        // OPTIMIZATION: func.function is already uppercase from parsing
        Ok(WindowFunctionInfo {
            name: func.function.to_string(),
            arguments: func.arguments.clone(),
            partition_by,
            partition_by_exprs: partition_by_exprs.to_vec(),
            order_by,
            frame,
            column_name,
            is_distinct: func.is_distinct,
            filter: func.filter.clone(),
            call_order_by: func.order_by.clone().into_boxed_slice(),
            extra_args,
            extra_error,
        })
    }

    /// Resolve the sorted partitions for a window function's OVER clause,
    /// sharing the partition map build and the per-partition sort across
    /// window functions with identical PARTITION BY and ORDER BY
    #[allow(clippy::too_many_arguments)]
    fn shared_sorted_partitions(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        pre_grouped: Option<&WindowPreGroupedState>,
        order_by_values: Option<&ColumnarOrderByValues>,
        partition_cache: &mut Vec<(OverClauseKey, SharedSortedPartitions)>,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<SharedSortedPartitions> {
        let shareable = self.over_clause_is_shareable(wf_info);
        let key = Self::over_clause_cache_key(wf_info);
        if shareable {
            if let Some((_, cached)) = partition_cache.iter().find(|(k, _)| k == &key) {
                return Ok(std::sync::Arc::clone(cached));
            }
        }

        let mut partition_vec: Vec<Vec<usize>> = if wf_info.partition_by_exprs.is_empty() {
            vec![(0..rows.len()).collect()]
        } else if let Some(pg) = pre_grouped.filter(|pg| {
            wf_info.partition_by.len() == 1
                && wf_info.partition_by.len() == wf_info.partition_by_exprs.len()
                && wf_info.partition_by[0].to_lowercase() == pg.partition_column
        }) {
            pg.partition_map.values().cloned().collect()
        } else {
            Self::build_partition_map(
                wf_info,
                rows,
                columns,
                col_index_map,
                group_slots,
                ctx,
                blocks,
            )?
            .into_values()
            .collect()
        };

        if let Some(order_by) = order_by_values.filter(|v| !v.is_empty()) {
            for indices in partition_vec.iter_mut() {
                Self::sort_by_order_values(indices, order_by);
            }
        }

        let shared: SharedSortedPartitions = std::sync::Arc::new(partition_vec);
        if shareable {
            partition_cache.push((key, std::sync::Arc::clone(&shared)));
        }
        Ok(shared)
    }

    /// Structural cache key for a (PARTITION BY, ORDER BY) pair; window
    /// functions sharing both reuse one partition map and one sort
    fn over_clause_cache_key(wf: &WindowFunctionInfo) -> OverClauseKey {
        let partition = wf
            .partition_by_exprs
            .iter()
            .map(|e| (compute_expression_hash(e), e.to_string()))
            .collect();
        (partition, Self::order_by_cache_key(&wf.order_by))
    }

    /// Window functions whose OVER clause contains a volatile expression
    /// (RANDOM, UUID, NOW, ...) must not share partitions: each function
    /// evaluates its own draw
    fn over_clause_is_shareable(&self, wf: &WindowFunctionInfo) -> bool {
        wf.partition_by_exprs
            .iter()
            .all(|e| expr_is_deterministic(e, &self.function_registry))
            && wf
                .order_by
                .iter()
                .all(|ob| expr_is_deterministic(&ob.expression, &self.function_registry))
    }

    /// Structural identity of an ORDER BY clause. Rendered SQL cannot
    /// serve here: Display prints bare positional parameters as "?"
    /// without their index, so two clauses binding different values
    /// would share a cache entry.
    #[inline]
    fn order_by_cache_key(order_by: &[OrderByExpression]) -> OrderByKey {
        order_by
            .iter()
            .map(|ob| {
                (
                    compute_expression_hash(&ob.expression),
                    ob.expression.to_string(),
                    ob.ascending,
                    ob.nulls_first,
                )
            })
            .collect()
    }

    /// Check that ALL window functions share the exact same PARTITION BY clause.
    /// Returns false if any window function is unpartitioned (global) or uses a
    /// different partition scheme, because the streaming path builds only one
    /// partition map and computes per-partition subsets.
    fn all_partitions_match(window_functions: &[WindowFunctionInfo]) -> bool {
        let mut canonical: Option<SmallVec<[(u64, String); 2]>> = None;
        for wf in window_functions {
            // A global (unpartitioned) window must see all rows, so
            // per-partition streaming is unsafe.
            if wf.partition_by_exprs.is_empty() {
                return false;
            }
            // Same identity the partition cache uses: hash plus rendered
            // form, per item, so parameters and separators cannot merge
            let key: SmallVec<[(u64, String); 2]> = wf
                .partition_by_exprs
                .iter()
                .map(|e| (compute_expression_hash(e), e.to_string()))
                .collect();
            match &canonical {
                None => canonical = Some(key),
                Some(c) if c != &key => return false,
                _ => {}
            }
        }
        canonical.is_some()
    }

    /// Build partition map from PARTITION BY expressions.
    /// Handles both simple column references (fast path) and complex expressions like
    /// function calls (e.g., TIME_TRUNC('30m', timestamp)) evaluated per row.
    fn build_partition_map(
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<FxHashMap<PartitionKey, Vec<usize>>> {
        let mut partitions: FxHashMap<PartitionKey, Vec<usize>> = FxHashMap::default();

        // Each key a column holds, by its name; none when a key is computed
        // or names no column, which the compiled path reports
        let partition_indices: Option<SmallVec<[usize; 4]>> =
            match wf_info.partition_by.len() == wf_info.partition_by_exprs.len() {
                true => wf_info
                    .partition_by
                    .iter()
                    .map(|part_col| qualified_key_index(col_index_map, &part_col.to_lowercase()))
                    .collect(),
                false => None,
            };

        if let Some(partition_indices) = partition_indices {
            for (i, (_, row)) in rows.iter().enumerate() {
                let key: PartitionKey = partition_indices
                    .iter()
                    .map(|&idx| row.get(idx).cloned().unwrap_or_else(Value::null_unknown))
                    .collect();
                partitions.entry(key).or_default().push(i);
            }
        } else {
            // Complex path: evaluate the expressions per row
            let mut eval = RowPrograms::compile(
                &wf_info.partition_by_exprs,
                columns,
                Some(group_slots),
                ctx,
                blocks,
            )?;
            let mut values = Vec::with_capacity(wf_info.partition_by_exprs.len());
            for (i, (_, row)) in rows.iter().enumerate() {
                eval.eval_into(row, &mut values).map_err(|e| {
                    Error::internal(format!("Failed to evaluate PARTITION BY expression: {}", e))
                })?;
                let key: PartitionKey = values.drain(..).collect();
                partitions.entry(key).or_default().push(i);
            }
        }

        Ok(partitions)
    }

    /// Compute window function values for all rows
    /// pre_sorted: Optional info about whether rows are already sorted by an indexed column
    /// pre_grouped: Optional pre-grouped partitions from index (avoids hash-based grouping)
    /// order_by_cache: Precomputed ORDER BY values cache (keyed by semantic string representation)
    #[allow(clippy::too_many_arguments)]
    fn compute_window_function(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        pre_sorted: Option<&WindowPreSortedState>,
        pre_grouped: Option<&WindowPreGroupedState>,
        order_by_cache: &[(OrderByKey, ColumnarOrderByValues)],
        partition_cache: &mut Vec<(OverClauseKey, SharedSortedPartitions)>,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Vec<Value>> {
        // Check if this is an aggregate function used as window function
        let is_aggregate = self.function_registry.is_aggregate(&wf_info.name);

        if is_aggregate {
            // Handle aggregate functions as window functions (SUM, COUNT, AVG, etc.)
            // Look up precomputed ORDER BY values from cache using semantic key
            let cache_key = Self::order_by_cache_key(&wf_info.order_by);
            let cached_order_by = order_by_cache
                .iter()
                .find(|(key, _)| key == &cache_key)
                .map(|(_, v)| v);
            return self.compute_aggregate_window_function(
                wf_info,
                rows,
                columns,
                col_index_map,
                group_slots,
                ctx,
                pre_sorted,
                cached_order_by,
                partition_cache,
                &CallArgs::of(wf_info, &self.function_registry),
                blocks,
            );
        }

        // Get the window function from registry
        let window_func = self
            .function_registry
            .get_window(&wf_info.name)
            .ok_or_else(|| unknown_window_function(&wf_info.name))?;

        // Look up precomputed ORDER BY values from cache using semantic key
        let precomputed_order_by: Option<&ColumnarOrderByValues> = if !wf_info.order_by.is_empty() {
            let cache_key = Self::order_by_cache_key(&wf_info.order_by);
            order_by_cache
                .iter()
                .find(|(key, _)| key == &cache_key)
                .map(|(_, v)| v)
        } else {
            None
        };

        // Resolve partitions once per distinct OVER clause: the map build
        // and the per-partition sort are shared across window functions
        let partitions = self.shared_sorted_partitions(
            wf_info,
            rows,
            columns,
            col_index_map,
            group_slots,
            ctx,
            pre_grouped,
            precomputed_order_by,
            partition_cache,
            blocks,
        )?;

        // No row, no frame offset evaluates
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut args = CallArgs::of(wf_info, &self.function_registry);
        args.input = self.window_input(
            wf_info,
            rows,
            columns,
            col_index_map,
            group_slots,
            ctx,
            blocks,
        )?;
        let frame = Frame::read(wf_info, &args, ctx)?;

        // If there's no partitioning, all rows form one partition
        if wf_info.partition_by_exprs.is_empty() {
            // Pre-allocate results array and use direct-writing variant
            let mut results: Vec<Value> = vec![NULL_VALUE; rows.len()];

            self.compute_window_for_partition_direct(
                &*window_func,
                wf_info,
                rows,
                partitions[0].clone(),
                true,
                precomputed_order_by,
                columns,
                col_index_map,
                group_slots,
                ctx,
                &frame,
                &args,
                &mut results,
            )?;

            return Ok(results);
        }

        // Compute window function for each partition
        // Use parallel execution for large number of partitions
        let partition_count = partitions.len();
        let use_parallel = partition_count >= 10 && rows.len() >= 1000;

        if use_parallel {
            // Parallel execution: process partitions concurrently
            // MEMORY OPTIMIZATION: Use ParallelVec for direct writes to disjoint indices.
            // This eliminates:
            //   1. Per-partition Vec<Value> allocation from compute_window_for_partition
            //   2. The final .collect() mapping step
            // Each partition writes to its own row indices (guaranteed disjoint).
            let mut results: Vec<Value> = vec![NULL_VALUE; rows.len()];
            let parallel_results = ParallelVec::new(&mut results);
            let lead_lag_default = args.default_program(columns)?;

            // A default holding a REGEXP is copied once per thread, not per job
            #[cfg(feature = "parallel")]
            let min_len = match lead_lag_default.is_some_and(Program::holds_thread_state) {
                true => partitions.len().div_ceil(rayon::current_num_threads()),
                false => 1,
            };
            #[cfg(feature = "parallel")]
            let iter_result = partitions
                .par_iter()
                .with_min_len(min_len)
                .try_for_each_init(
                    || {
                        lead_lag_default
                            .filter(|program| program.holds_thread_state())
                            .map(Program::worker_copy)
                    },
                    |worker, row_indices| -> Result<()> {
                        // Direct writing: each partition writes to its own indices
                        self.compute_window_for_partition_parallel(
                            &*window_func,
                            wf_info,
                            rows,
                            row_indices.clone(),
                            true,
                            precomputed_order_by,
                            col_index_map,
                            group_slots,
                            ctx,
                            &frame,
                            &args,
                            worker.as_ref().or(lead_lag_default),
                            &parallel_results,
                        )
                    },
                );
            #[cfg(not(feature = "parallel"))]
            let iter_result = partitions.iter().try_for_each(|row_indices| -> Result<()> {
                // Direct writing: each partition writes to its own indices
                self.compute_window_for_partition_parallel(
                    &*window_func,
                    wf_info,
                    rows,
                    row_indices.clone(),
                    true,
                    precomputed_order_by,
                    col_index_map,
                    group_slots,
                    ctx,
                    &frame,
                    &args,
                    lead_lag_default,
                    &parallel_results,
                )
            });
            iter_result?;

            // ParallelVec is done, results are written in place
            Ok(results)
        } else {
            // Sequential execution for small partition counts
            // MEMORY OPTIMIZATION: Write directly to final results array instead of
            // creating per-partition Vec<Value> and then mapping back
            let mut results: Vec<Value> = vec![NULL_VALUE; rows.len()];

            for row_indices in partitions.iter() {
                // MEMORY OPTIMIZATION: Direct writing variant - writes to results in-place
                self.compute_window_for_partition_direct(
                    &*window_func,
                    wf_info,
                    rows,
                    row_indices.clone(),
                    true,
                    precomputed_order_by,
                    columns,
                    col_index_map,
                    group_slots,
                    ctx,
                    &frame,
                    &args,
                    &mut results,
                )?;
            }

            Ok(results)
        }
    }

    /// Compute window function for a single partition
    /// Returns (results in sorted order, sorted row indices) to avoid re-sorting in the caller
    /// precomputed_order_by: Optional precomputed ORDER BY values for ALL rows (avoids recomputation)
    /// skip_sorting: If true, skip sorting (rows are already pre-sorted by index)
    ///
    /// MEMORY OPTIMIZATION: This function uses index-based value access instead of cloning
    /// values into intermediate Vec<Value> collections. For LEAD/LAG/FIRST_VALUE/LAST_VALUE,
    /// values are accessed directly from all_rows via indices. For RANK/DENSE_RANK, we use
    /// ColumnarOrderByValues::rows_equal() to compare ORDER BY values without cloning.
    #[allow(clippy::too_many_arguments)]
    fn compute_window_for_partition(
        &self,
        window_func: &dyn WindowFunction,
        wf_info: &WindowFunctionInfo,
        all_rows: &[(i64, Row)],
        mut row_indices: Vec<usize>,
        precomputed_order_by: Option<&ColumnarOrderByValues>,
        columns: &[String],
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        args: &CallArgs,
        skip_sorting: bool,
    ) -> Result<(Vec<Value>, Vec<usize>)> {
        // Empty fallback for when no precomputed values are provided
        let empty_order_by = ColumnarOrderByValues {
            columns: vec![],
            ascending: vec![],
            nulls_first: vec![],
            num_rows: 0,
        };
        let order_by_values = precomputed_order_by.unwrap_or(&empty_order_by);

        // Sort partition by ORDER BY if specified (skip if already pre-sorted by index)
        if !skip_sorting && !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::sort_by_order_values(&mut row_indices, order_by_values);
        }

        // Empty partition: nothing to compute, and no argument evaluates
        if row_indices.is_empty() {
            return Ok((Vec::new(), row_indices));
        }
        let frame = Frame::read(wf_info, args, ctx)?;
        let mut results = Vec::with_capacity(row_indices.len());
        self.compute_sorted_partition(
            window_func,
            wf_info,
            &frame,
            args,
            args.default_program(columns)?,
            all_rows,
            &row_indices,
            order_by_values,
            col_index_map,
            group_slots,
            ctx,
            |_, value| results.push(value),
        )?;
        Ok((results, row_indices))
    }

    /// Computes a window function that is not an aggregate over one sorted,
    /// non-empty partition, handing each row index its value
    #[allow(clippy::too_many_arguments)]
    fn compute_sorted_partition(
        &self,
        window_func: &dyn WindowFunction,
        wf_info: &WindowFunctionInfo,
        frame: &Frame,
        args: &CallArgs,
        lead_lag_default: Option<&Program>,
        all_rows: &[(i64, Row)],
        row_indices: &[usize],
        order_by_values: &ColumnarOrderByValues,
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        mut emit: impl FnMut(usize, Value),
    ) -> Result<()> {
        // OPTIMIZATION: Pre-compute column index for function argument
        let input = ValueInput {
            column: wf_info
                .arguments
                .first()
                .and_then(|arg| self.resolve_column_index(arg, col_index_map, group_slots)),
            values: &args.input,
        };

        // MEMORY OPTIMIZATION: Compute rank info directly from ColumnarOrderByValues
        // instead of cloning ORDER BY values into a Vec<Value>
        // This uses rows_equal() for O(n) comparison without allocating order_values
        let is_rank_function = matches!(
            wf_info.name.as_str(),
            "RANK" | "DENSE_RANK" | "PERCENT_RANK"
        );
        let is_rank = wf_info.name == "RANK";
        let rank_info = if is_rank_function && !order_by_values.is_empty() {
            Self::precompute_rank_info_columnar(row_indices, order_by_values)
        } else {
            vec![]
        };

        let partition_len = row_indices.len();
        let frames = PartitionFrames::new(frame, row_indices, order_by_values);

        // NTH_VALUE's n evaluates only for a frame that is not empty
        let mut count_arg = args.count.as_ref().map(PartitionArg::new);
        let mut default_vm = ExprVM::new();
        let lead_lag_offset = if wf_info.name == "LEAD" {
            "the offset of LEAD"
        } else {
            "the offset of LAG"
        };

        for (i, &row_idx) in row_indices.iter().enumerate() {
            // Handle special functions
            let value = match wf_info.name.as_str() {
                // MEMORY OPTIMIZATION: Access values directly from all_rows via indices
                // instead of cloning into partition_values Vec
                "LEAD" | "LAG" => Self::compute_lead_lag_indexed(
                    wf_info.name == "LEAD",
                    match &mut count_arg {
                        Some(arg) => window_count(&arg.value(ctx)?, lead_lag_offset, 0)?,
                        None => 1,
                    },
                    args.default_value(
                        lead_lag_default,
                        &mut default_vm,
                        &all_rows[row_idx].1,
                        ctx,
                    )?,
                    all_rows,
                    row_indices,
                    input,
                    i,
                )?,
                "NTILE" => Self::compute_ntile(
                    match &mut count_arg {
                        Some(arg) => {
                            window_count(&arg.value(ctx)?, "the bucket count of NTILE", 1)?
                        }
                        None => 1,
                    },
                    partition_len,
                    i,
                ),
                "RANK" | "DENSE_RANK" => Self::compute_rank_fast(is_rank, &rank_info, i),
                "FIRST_VALUE" | "LAST_VALUE" | "NTH_VALUE" => {
                    let (frame_start, frame_end) = frames.bounds(frame, i);

                    // MEMORY OPTIMIZATION: Access values directly via indices
                    match wf_info.name.as_str() {
                        "FIRST_VALUE" => self.compute_first_value_indexed(
                            all_rows,
                            row_indices,
                            input,
                            frame_start,
                            frame_end,
                        )?,
                        "LAST_VALUE" => self.compute_last_value_indexed(
                            all_rows,
                            row_indices,
                            input,
                            frame_start,
                            frame_end,
                        )?,
                        "NTH_VALUE" => match &mut count_arg {
                            Some(arg) if frame_start < frame_end => {
                                let n = window_count(&arg.value(ctx)?, "the n of NTH_VALUE", 1)?;
                                Self::compute_nth_value_indexed(
                                    n,
                                    all_rows,
                                    row_indices,
                                    input,
                                    frame_start,
                                    frame_end,
                                )
                            }
                            _ => Value::null_unknown(),
                        },
                        _ => unreachable!(),
                    }
                }
                "PERCENT_RANK" => {
                    // (rank - 1) / (n - 1) with rank from the O(n) precompute
                    if partition_len <= 1 {
                        Value::Float(0.0)
                    } else {
                        let rank = rank_info.get(i).map(|&(gs, _)| gs + 1).unwrap_or(1);
                        Value::Float((rank - 1) as f64 / (partition_len - 1) as f64)
                    }
                }
                // peer-group end / n from the O(n) precompute
                "CUME_DIST" => Value::Float(frames.peers[i].1 as f64 / partition_len as f64),
                _ => {
                    // ROW_NUMBER and other simple functions - no values needed
                    window_func.process(&[], &[], i)?
                }
            };
            emit(row_idx, value);
        }

        Ok(())
    }

    /// Direct-writing variant of compute_window_for_partition
    /// Writes results directly to the provided results array, avoiding per-partition Vec allocation
    ///
    /// MEMORY OPTIMIZATION: Instead of returning (Vec<Value>, Vec<usize>) and mapping back,
    /// this writes directly to results[orig_idx] for each computed value.
    #[allow(clippy::too_many_arguments)]
    fn compute_window_for_partition_direct(
        &self,
        window_func: &dyn WindowFunction,
        wf_info: &WindowFunctionInfo,
        all_rows: &[(i64, Row)],
        mut row_indices: Vec<usize>,
        already_sorted: bool,
        precomputed_order_by: Option<&ColumnarOrderByValues>,
        columns: &[String],
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        frame: &Frame,
        args: &CallArgs,
        results: &mut [Value],
    ) -> Result<()> {
        // Empty fallback for when no precomputed values are provided
        let empty_order_by = ColumnarOrderByValues {
            columns: vec![],
            ascending: vec![],
            nulls_first: vec![],
            num_rows: 0,
        };
        let order_by_values = precomputed_order_by.unwrap_or(&empty_order_by);

        // Sort partition by ORDER BY if specified (the shared-partition
        // cache hands in pre-sorted indices)
        if !already_sorted && !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::sort_by_order_values(&mut row_indices, order_by_values);
        }

        if row_indices.is_empty() {
            return Ok(());
        }
        // DIRECT WRITE: Write to original row position, avoiding the mapping step
        self.compute_sorted_partition(
            window_func,
            wf_info,
            frame,
            args,
            args.default_program(columns)?,
            all_rows,
            &row_indices,
            order_by_values,
            col_index_map,
            group_slots,
            ctx,
            |row_idx, value| results[row_idx] = value,
        )
    }

    /// Parallel-safe variant of compute_window_for_partition_direct.
    /// Uses ParallelVec for safe writes to disjoint indices from multiple threads.
    ///
    /// MEMORY OPTIMIZATION: Eliminates per-partition Vec<Value> allocation and the
    /// final mapping/collect step in parallel execution path.
    #[allow(clippy::too_many_arguments)]
    fn compute_window_for_partition_parallel(
        &self,
        window_func: &dyn WindowFunction,
        wf_info: &WindowFunctionInfo,
        all_rows: &[(i64, Row)],
        mut row_indices: Vec<usize>,
        already_sorted: bool,
        precomputed_order_by: Option<&ColumnarOrderByValues>,
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        frame: &Frame,
        args: &CallArgs,
        lead_lag_default: Option<&Program>,
        results: &ParallelVec,
    ) -> Result<()> {
        // Empty fallback for when no precomputed values are provided
        let empty_order_by = ColumnarOrderByValues {
            columns: vec![],
            ascending: vec![],
            nulls_first: vec![],
            num_rows: 0,
        };
        let order_by_values = precomputed_order_by.unwrap_or(&empty_order_by);

        // Sort partition by ORDER BY if specified (the shared-partition
        // cache hands in pre-sorted indices)
        if !already_sorted && !wf_info.order_by.is_empty() && !order_by_values.is_empty() {
            Self::sort_by_order_values(&mut row_indices, order_by_values);
        }

        if row_indices.is_empty() {
            return Ok(());
        }
        // DIRECT WRITE: Write to original row position using ParallelVec
        self.compute_sorted_partition(
            window_func,
            wf_info,
            frame,
            args,
            lead_lag_default,
            all_rows,
            &row_indices,
            order_by_values,
            col_index_map,
            group_slots,
            ctx,
            |row_idx, value| results.write(row_idx, value),
        )
    }

    /// Precompute rank information using ColumnarOrderByValues directly (no cloning)
    ///
    /// Returns a vector of (group_start, dense_rank) for each position in sorted_indices.
    /// Uses rows_equal() for O(n) comparison without allocating intermediate Vec<Value>.
    #[inline]
    fn precompute_rank_info_columnar(
        sorted_indices: &[usize],
        order_by: &ColumnarOrderByValues,
    ) -> Vec<(usize, i64)> {
        let n = sorted_indices.len();
        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);

        // First row: group starts at 0, dense_rank = 1
        result.push((0, 1));

        let mut current_group_start = 0;
        let mut current_dense_rank: i64 = 1;

        for i in 1..n {
            // Compare ORDER BY values of adjacent sorted rows without cloning
            if !order_by.rows_equal(sorted_indices[i - 1], sorted_indices[i]) {
                // New group starts
                current_group_start = i;
                current_dense_rank += 1;
            }
            result.push((current_group_start, current_dense_rank));
        }

        result
    }

    /// The peer group of each sorted position as (start, exclusive end):
    /// rows with equal ORDER BY values, every row without an ORDER BY
    fn precompute_peer_groups(
        sorted_indices: &[usize],
        order_by: &ColumnarOrderByValues,
    ) -> Vec<(usize, usize)> {
        let n = sorted_indices.len();
        let mut groups = Vec::with_capacity(n);
        let mut group_start = 0;
        for i in 1..=n {
            if i == n || !order_by.rows_equal(sorted_indices[i - 1], sorted_indices[i]) {
                groups.resize(i, (group_start, i));
                group_start = i;
            }
        }
        groups
    }

    /// Compute LEAD or LAG using index-based access (no cloning).
    /// The offset and the current row's default are read by the caller;
    /// the default evaluates per row because it can reference row columns.
    fn compute_lead_lag_indexed(
        is_lead: bool,
        offset: usize,
        default_value: Value,
        all_rows: &[(i64, Row)],
        sorted_indices: &[usize],
        input: ValueInput<'_>,
        current_pos: usize,
    ) -> Result<Value> {
        // Calculate target position in sorted order
        let target_pos = if is_lead {
            current_pos.checked_add(offset)
        } else {
            // LAG
            current_pos.checked_sub(offset)
        };

        match target_pos {
            Some(pos) if pos < sorted_indices.len() => Ok(input
                .at(all_rows, sorted_indices[pos])
                .unwrap_or(default_value)),
            _ => Ok(default_value),
        }
    }

    /// Compute FIRST_VALUE using index-based access (no cloning)
    fn compute_first_value_indexed(
        &self,
        all_rows: &[(i64, Row)],
        sorted_indices: &[usize],
        input: ValueInput<'_>,
        frame_start: usize,
        frame_end: usize,
    ) -> Result<Value> {
        if frame_start >= frame_end || frame_start >= sorted_indices.len() {
            return Ok(Value::null_unknown());
        }
        Ok(input
            .at(all_rows, sorted_indices[frame_start])
            .unwrap_or_else(Value::null_unknown))
    }

    /// Compute LAST_VALUE using index-based access (no cloning)
    fn compute_last_value_indexed(
        &self,
        all_rows: &[(i64, Row)],
        sorted_indices: &[usize],
        input: ValueInput<'_>,
        frame_start: usize,
        frame_end: usize,
    ) -> Result<Value> {
        if frame_start >= frame_end || frame_end > sorted_indices.len() {
            return Ok(Value::null_unknown());
        }
        Ok(input
            .at(all_rows, sorted_indices[frame_end - 1])
            .unwrap_or_else(Value::null_unknown))
    }

    /// Compute NTH_VALUE using index-based access (no cloning); `n` counts
    /// from 1 within a frame that is not empty
    fn compute_nth_value_indexed(
        n: usize,
        all_rows: &[(i64, Row)],
        sorted_indices: &[usize],
        input: ValueInput<'_>,
        frame_start: usize,
        frame_end: usize,
    ) -> Value {
        if n > frame_end - frame_start {
            return Value::null_unknown();
        }
        input
            .at(all_rows, sorted_indices[frame_start + n - 1])
            .unwrap_or_else(Value::null_unknown)
    }

    /// Compute NTILE function; the bucket count is pre-resolved by the caller
    fn compute_ntile(n: usize, partition_size: usize, current_row: usize) -> Value {
        // NTILE divides rows into n buckets as evenly as possible.
        // If partition_size doesn't divide evenly by n, the first (partition_size % n)
        // buckets get one extra row.
        //
        // For example, NTILE(3) with 7 rows:
        // - base_size = 7 / 3 = 2 (rows per bucket)
        // - remainder = 7 % 3 = 1 (1 bucket gets an extra row)
        // - Bucket 1: rows 0, 1, 2 (3 rows - gets extra)
        // - Bucket 2: rows 3, 4 (2 rows)
        // - Bucket 3: rows 5, 6 (2 rows)

        if n >= partition_size {
            // More buckets than rows - each row gets its own bucket
            return Value::Integer((current_row + 1).min(n) as i64);
        }

        let base_size = partition_size / n;
        let remainder = partition_size % n;

        // Calculate which bucket this row belongs to
        // First 'remainder' buckets have (base_size + 1) rows
        // Remaining buckets have base_size rows
        let bucket = if current_row < remainder * (base_size + 1) {
            // Row is in one of the larger buckets
            current_row / (base_size + 1) + 1
        } else {
            // Row is in one of the smaller buckets
            let rows_in_larger_buckets = remainder * (base_size + 1);
            let row_in_smaller_section = current_row - rows_in_larger_buckets;
            remainder + row_in_smaller_section / base_size + 1
        };

        Value::Integer(bucket as i64)
    }

    /// Compute RANK or DENSE_RANK function using precomputed rank info.
    ///
    /// This is an O(1) lookup using the precomputed (group_start, dense_rank) tuple.
    #[inline]
    fn compute_rank_fast(is_rank: bool, rank_info: &[(usize, i64)], current_row: usize) -> Value {
        if rank_info.is_empty() || current_row >= rank_info.len() {
            return Value::Integer(1);
        }

        let (group_start, dense_rank) = rank_info[current_row];

        if is_rank {
            // RANK: 1-indexed position of first row in group
            Value::Integer((group_start + 1) as i64)
        } else {
            // DENSE_RANK: sequential group number
            Value::Integer(dense_rank)
        }
    }

    /// A value function's first argument for each of `rows` when it is not a
    /// column, read once ahead of the partitions; empty otherwise
    #[allow(clippy::too_many_arguments)]
    fn window_input(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Vec<Value>> {
        let reads_values = matches!(
            wf_info.name.as_str(),
            "LAG" | "LEAD" | "FIRST_VALUE" | "LAST_VALUE" | "NTH_VALUE"
        );
        let Some(arg) = wf_info.arguments.first().filter(|arg| {
            reads_values
                && !matches!(arg, Expression::Star(_))
                && self
                    .resolve_column_index(arg, col_index_map, group_slots)
                    .is_none()
        }) else {
            return Ok(Vec::new());
        };
        let mut eval = RowPrograms::compile([arg], columns, None, ctx, blocks)?;
        rows.iter().map(|(_, row)| eval.eval(row)).collect()
    }

    /// Resolve column index from expression, trying both qualified and unqualified names
    fn resolve_column_index(
        &self,
        expr: &Expression,
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
    ) -> Option<usize> {
        match expr {
            Expression::Identifier(id) => col_index_map.get(id.value_lower.as_str()).copied(),
            Expression::QualifiedIdentifier(qid) => qualified_index(
                col_index_map,
                &qid.qualifier.value_lower,
                &qid.name.value_lower,
            ),
            // A group item or aggregate computed before the window reads its slot,
            // and a derived table's aggregate column the name it carries
            _ => (!group_slots.is_empty())
                .then(|| group_slots.get(&ExprKey::of(expr)).copied())
                .flatten()
                .or_else(|| {
                    derived_aggregate_column_name(expr)
                        .and_then(|name| col_index_map.get(name.to_lowercase().as_str()).copied())
                }),
        }
    }

    /// Precompute ORDER BY values for all rows using columnar layout
    /// Returns a ColumnarOrderByValues structure for cache-efficient sorting
    #[allow(clippy::too_many_arguments)]
    fn precompute_order_by_values(
        &self,
        order_by: &[OrderByExpression],
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<ColumnarOrderByValues> {
        let num_rows = rows.len();
        let num_cols = order_by.len();

        if num_cols == 0 || num_rows == 0 {
            return Ok(ColumnarOrderByValues {
                columns: vec![],
                ascending: vec![],
                nulls_first: vec![],
                num_rows: 0,
            });
        }

        // Extract ascending flags once (not per row!)
        let ascending_flags: Vec<bool> = order_by.iter().map(|ob| ob.ascending).collect();
        let nulls_first_flags: Vec<bool> = order_by
            .iter()
            .map(|ob| crate::core::value::nulls_first(ob.ascending, ob.nulls_first))
            .collect();

        // Each key a column holds, by its name; none when a key is computed
        // or names no column, which the compiled path reports
        let order_by_indices: Option<Vec<usize>> = order_by
            .iter()
            .map(|ob| match &ob.expression {
                Expression::Identifier(id) => col_index_map.get(id.value_lower.as_str()).copied(),
                Expression::QualifiedIdentifier(qid) => qualified_index(
                    col_index_map,
                    &qid.qualifier.value_lower,
                    &qid.name.value_lower,
                ),
                _ => None,
            })
            .collect();

        // Pre-allocate columns with exact capacity
        let mut result_columns: Vec<Vec<Value>> = (0..num_cols)
            .map(|_| Vec::with_capacity(num_rows))
            .collect();

        if let Some(order_by_indices) = order_by_indices {
            // Extract values column by column for better cache locality
            for (col_idx, src_idx) in order_by_indices.iter().enumerate() {
                let col = &mut result_columns[col_idx];
                for (_, row) in rows {
                    col.push(
                        row.get(*src_idx)
                            .cloned()
                            .unwrap_or_else(Value::null_unknown),
                    );
                }
            }
        } else {
            // Extract order_by expressions
            let order_exprs: Vec<Expression> =
                order_by.iter().map(|ob| ob.expression.clone()).collect();

            // A group item or aggregate in ORDER BY reads its slot
            let mut eval =
                RowPrograms::compile(&order_exprs, columns, Some(group_slots), ctx, blocks)?;
            let mut values = Vec::with_capacity(num_cols);
            for (_, row) in rows {
                eval.eval_into(row, &mut values)?;
                for (col_idx, value) in values.drain(..).enumerate() {
                    if col_idx < result_columns.len() {
                        result_columns[col_idx].push(value);
                    }
                }
            }
        }

        Ok(ColumnarOrderByValues {
            columns: result_columns,
            ascending: ascending_flags,
            nulls_first: nulls_first_flags,
            num_rows,
        })
    }

    /// Check if rows are already pre-sorted by the window ORDER BY column
    /// Returns true if we can skip sorting
    fn check_rows_presorted(
        &self,
        wf_info: &WindowFunctionInfo,
        pre_sorted: Option<&WindowPreSortedState>,
    ) -> bool {
        let pre_sorted = match pre_sorted {
            Some(ps) => ps,
            None => return false,
        };

        // Only optimize if there's exactly one ORDER BY column (simple case)
        if wf_info.order_by.len() != 1 {
            return false;
        }

        let order_by = &wf_info.order_by[0];

        // Extract column name from ORDER BY expression
        let order_col = match &order_by.expression {
            Expression::Identifier(id) => id.value_lower.clone(),
            Expression::QualifiedIdentifier(qid) => qid.name.value_lower.clone(),
            _ => return false, // Complex expressions can't be pre-sorted
        };

        // Check if pre-sorted column matches and direction matches
        order_col == pre_sorted.column && order_by.ascending == pre_sorted.ascending
    }

    /// Sort row indices using precomputed ORDER BY values (columnar layout)
    fn sort_by_order_values(row_indices: &mut [usize], order_by_values: &ColumnarOrderByValues) {
        if row_indices.len() < 2 || order_by_values.is_empty() {
            return;
        }

        // Check if single ORDER BY column (most common case) - use fast path
        let is_single_order_by = order_by_values.num_columns() == 1;

        if is_single_order_by {
            // Fast path: single ORDER BY column with type-specific comparison
            let ascending = order_by_values.is_ascending(0);
            let nulls_first = order_by_values.nulls_first(0);

            // Get direct reference to the column for cache-efficient access
            let col = &order_by_values.columns[0];

            // Detect column type from first non-null value (single pass)
            #[derive(Clone, Copy, PartialEq)]
            enum DetectedType {
                Unknown,
                Integer,
                Float,
                Mixed,
            }

            // A 100-row sample picks the candidate fast path; the typed
            // extractors bail to the generic sort when the type drifts
            // past the sample, so this is only a hint
            let mut detected = DetectedType::Unknown;
            for &idx in row_indices.iter().take(100) {
                if let Some(val) = col.get(idx) {
                    match val {
                        Value::Integer(_) => {
                            if detected == DetectedType::Unknown {
                                detected = DetectedType::Integer;
                            } else if detected != DetectedType::Integer {
                                detected = DetectedType::Mixed;
                                break;
                            }
                        }
                        Value::Float(_) => {
                            if detected == DetectedType::Unknown {
                                detected = DetectedType::Float;
                            } else if detected != DetectedType::Float {
                                detected = DetectedType::Mixed;
                                break;
                            }
                        }
                        Value::Null(_) => {
                            // Skip nulls, they're handled by unwrap_or in sort functions
                        }
                        _ => {
                            detected = DetectedType::Mixed;
                            break;
                        }
                    }
                }
            }

            match detected {
                DetectedType::Integer
                    if Self::sort_by_integer_key_columnar(
                        row_indices,
                        col,
                        ascending,
                        nulls_first,
                    ) =>
                {
                    return;
                }
                DetectedType::Float
                    if Self::sort_by_float_key_columnar(
                        row_indices,
                        col,
                        ascending,
                        nulls_first,
                    ) =>
                {
                    return;
                }
                _ => {}
            }
            // Mixed types, unknown, or drift past the sample - generic sort
            Self::sort_single_column_columnar(row_indices, col, ascending, nulls_first);
            return;
        }

        // Multi-column ORDER BY: use parallel sort for large partitions
        const PARALLEL_THRESHOLD: usize = 10_000;

        if row_indices.len() >= PARALLEL_THRESHOLD {
            #[cfg(feature = "parallel")]
            row_indices.par_sort_unstable_by(|&a, &b| {
                Self::compare_order_values_columnar(order_by_values, a, b)
            });
            #[cfg(not(feature = "parallel"))]
            row_indices.sort_unstable_by(|&a, &b| {
                Self::compare_order_values_columnar(order_by_values, a, b)
            });
        } else {
            row_indices.sort_unstable_by(|&a, &b| {
                Self::compare_order_values_columnar(order_by_values, a, b)
            });
        }
    }

    /// Compare two rows by their ORDER BY values (columnar layout)
    #[inline]
    fn compare_order_values_columnar(
        order_by_values: &ColumnarOrderByValues,
        a: usize,
        b: usize,
    ) -> Ordering {
        for col_idx in 0..order_by_values.num_columns() {
            let a_val = order_by_values.get(a, col_idx).filter(|v| !v.is_null());
            let b_val = order_by_values.get(b, col_idx).filter(|v| !v.is_null());
            let cmp = match (a_val, b_val) {
                (Some(av), Some(bv)) => {
                    let cmp = Self::compare_values_fast(av, bv);
                    if order_by_values.is_ascending(col_idx) {
                        cmp
                    } else {
                        cmp.reverse()
                    }
                }
                (None, None) => Ordering::Equal,
                (None, _) if order_by_values.nulls_first(col_idx) => Ordering::Less,
                (None, _) => Ordering::Greater,
                (_, None) if order_by_values.nulls_first(col_idx) => Ordering::Greater,
                (_, None) => Ordering::Less,
            };

            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    /// Value comparison for the window sort, ascending with NULL last
    #[inline]
    fn compare_values_fast(a: &Value, b: &Value) -> Ordering {
        crate::core::value::compare_order_key(Some(a), Some(b), true, None)
    }

    /// Ultra-fast sort for integer ORDER BY column (columnar layout)
    /// Returns false without touching row_indices when a non-Integer,
    /// non-NULL value appears (type drift past the detection sample)
    fn sort_by_integer_key_columnar(
        row_indices: &mut [usize],
        col: &[Value],
        ascending: bool,
        nulls_first: bool,
    ) -> bool {
        const PARALLEL_THRESHOLD: usize = 50_000;

        // OPTIMIZATION: Pre-extract sort keys to avoid bounds checking in the comparison hot path.
        // The sort comparator is called O(n log n) times, and each .get() does bounds checking.
        // By pre-extracting into a Vec<i64>, we pay O(k) allocation cost but get O(1) unchecked
        // access during sorting, which is a significant win for large partitions.
        //
        // We build a parallel array of (original_position, sort_key) and sort by key,
        // then extract the sorted positions back to row_indices.
        let n = row_indices.len();
        if n < 2 {
            return true;
        }

        // Pre-extract sort keys: O(k) where k = partition size.
        // NULL rows are kept out of band instead of using a sentinel key,
        // so a genuine i64::MAX row can never tie with NULLs and the hot
        // sort stays a bare-u64 comparison. The i64 -> u64 map is
        // order-preserving (flip the sign bit).
        let mut keyed: Vec<(usize, u64)> = Vec::with_capacity(n);
        let mut null_indices: Vec<usize> = Vec::new();
        for &idx in row_indices.iter() {
            match col.get(idx) {
                Some(Value::Integer(i)) => keyed.push((idx, (*i as u64) ^ (1u64 << 63))),
                Some(Value::Null(_)) | None => null_indices.push(idx),
                _ => return false, // type drift
            }
        }

        // Sort by pre-extracted key - no bounds checking in comparison!
        if n >= PARALLEL_THRESHOLD {
            if ascending {
                #[cfg(feature = "parallel")]
                keyed.par_sort_unstable_by_key(|&(_, key)| key);
                #[cfg(not(feature = "parallel"))]
                keyed.sort_unstable_by_key(|&(_, key)| key);
            } else {
                #[cfg(feature = "parallel")]
                keyed.par_sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
                #[cfg(not(feature = "parallel"))]
                keyed.sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
            }
        } else if ascending {
            keyed.sort_unstable_by_key(|&(_, key)| key);
        } else {
            keyed.sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
        }

        // Write back: NULLs strictly first or last, as the ORDER BY places them.
        // For floats this also fixes the old sentinel's misplacement of
        // NULLs before +inf and NaN
        let mut pos = 0;
        if nulls_first {
            for &idx in &null_indices {
                row_indices[pos] = idx;
                pos += 1;
            }
        }
        for (idx, _) in keyed {
            row_indices[pos] = idx;
            pos += 1;
        }
        if !nulls_first {
            for &idx in &null_indices {
                row_indices[pos] = idx;
                pos += 1;
            }
        }
        true
    }

    /// Fast sort for float ORDER BY column (columnar layout)
    /// Returns false without touching row_indices when a value that is
    /// neither Float nor NULL appears (type drift past the sample)
    fn sort_by_float_key_columnar(
        row_indices: &mut [usize],
        col: &[Value],
        ascending: bool,
        nulls_first: bool,
    ) -> bool {
        const PARALLEL_THRESHOLD: usize = 50_000;

        // OPTIMIZATION: Pre-extract sort keys to avoid bounds checking in the comparison hot path.
        // The sort comparator is called O(n log n) times, and each .get() does bounds checking.
        // By pre-extracting into a Vec<f64>, we pay O(k) allocation cost but get O(1) unchecked
        // access during sorting. For floats, we also use to_bits() to enable faster integer
        // comparison via sort_by_key instead of sort_by with a closure.
        let n = row_indices.len();
        if n < 2 {
            return true;
        }

        // Pre-extract sort keys as ordered bits for fast integer
        // comparison. The leading class separates NULLs from values, so a
        // genuine f64::MAX row can never tie with NULL rows; NaN keeps a
        // maximal key inside the value class (after all reals, before
        // NULLs, matching the generic comparator)
        let mut keyed: Vec<(usize, u64)> = Vec::with_capacity(n);
        let mut null_indices: Vec<usize> = Vec::new();
        for &idx in row_indices.iter() {
            match col.get(idx) {
                Some(Value::Float(f)) => {
                    let bits = if f.is_nan() {
                        u64::MAX // all NaNs equal, after every real
                    } else {
                        let b = f.to_bits();
                        // Flip sign bit and conditionally flip all bits
                        // for proper ordering
                        if (b & (1u64 << 63)) != 0 {
                            !b // Negative: flip all bits
                        } else {
                            b | (1u64 << 63) // Positive: set sign bit
                        }
                    };
                    keyed.push((idx, bits));
                }
                Some(Value::Null(_)) | None => null_indices.push(idx),
                _ => return false, // type drift
            }
        }

        // Sort by pre-extracted key - no bounds checking, pure integer comparison!
        if n >= PARALLEL_THRESHOLD {
            if ascending {
                #[cfg(feature = "parallel")]
                keyed.par_sort_unstable_by_key(|&(_, key)| key);
                #[cfg(not(feature = "parallel"))]
                keyed.sort_unstable_by_key(|&(_, key)| key);
            } else {
                #[cfg(feature = "parallel")]
                keyed.par_sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
                #[cfg(not(feature = "parallel"))]
                keyed.sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
            }
        } else if ascending {
            keyed.sort_unstable_by_key(|&(_, key)| key);
        } else {
            keyed.sort_unstable_by_key(|&(_, key)| std::cmp::Reverse(key));
        }

        // Write back: NULLs strictly first or last, as the ORDER BY places them.
        // For floats this also fixes the old sentinel's misplacement of
        // NULLs before +inf and NaN
        let mut pos = 0;
        if nulls_first {
            for &idx in &null_indices {
                row_indices[pos] = idx;
                pos += 1;
            }
        }
        for (idx, _) in keyed {
            row_indices[pos] = idx;
            pos += 1;
        }
        if !nulls_first {
            for &idx in &null_indices {
                row_indices[pos] = idx;
                pos += 1;
            }
        }
        true
    }

    /// Single column generic sort (columnar layout)
    fn sort_single_column_columnar(
        row_indices: &mut [usize],
        col: &[Value],
        ascending: bool,
        nulls_first: bool,
    ) {
        const PARALLEL_THRESHOLD: usize = 10_000;
        let compare = |&a: &usize, &b: &usize| -> Ordering {
            let a_val = col.get(a).filter(|v| !v.is_null());
            let b_val = col.get(b).filter(|v| !v.is_null());
            match (a_val, b_val) {
                (Some(a), Some(b)) => {
                    let cmp = Self::compare_values_fast(a, b);
                    if ascending {
                        cmp
                    } else {
                        cmp.reverse()
                    }
                }
                (None, None) => Ordering::Equal,
                (None, _) if nulls_first => Ordering::Less,
                (None, _) => Ordering::Greater,
                (_, None) if nulls_first => Ordering::Greater,
                (_, None) => Ordering::Less,
            }
        };

        if row_indices.len() >= PARALLEL_THRESHOLD {
            #[cfg(feature = "parallel")]
            row_indices.par_sort_unstable_by(compare);
            #[cfg(not(feature = "parallel"))]
            row_indices.sort_unstable_by(compare);
        } else {
            row_indices.sort_unstable_by(compare);
        }
    }

    /// The aggregate's argument per row, NULL where its FILTER rejects the row
    #[allow(clippy::too_many_arguments)]
    fn window_filtered_values(
        &self,
        filter: &Expression,
        rows: &[(i64, Row)],
        columns: &[String],
        ctx: &ExecutionContext,
        arg_col_idx: Option<usize>,
        expression_values: &[Value],
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Vec<Value>> {
        let mut eval = RowPrograms::compile([filter], columns, None, ctx, blocks)?;
        let mut values = Vec::with_capacity(rows.len());
        for (idx, (_, row)) in rows.iter().enumerate() {
            values.push(if matches!(eval.eval(row)?, Value::Boolean(true)) {
                window_argument_value(rows, idx, arg_col_idx, expression_values).clone()
            } else {
                Value::null_unknown()
            });
        }
        Ok(values)
    }

    /// The running state of an aggregate window call, built as every
    /// aggregate path builds it; `ordered` when the call's ORDER BY keys are fed
    fn window_aggregate_state(
        &self,
        wf_info: &WindowFunctionInfo,
        ordered: bool,
    ) -> Result<crate::functions::aggregate::AggregateState> {
        let order_by: &[OrderByExpression] = if ordered { &wf_info.call_order_by } else { &[] };
        Self::aggregate_state(
            &self.function_registry,
            &wf_info.name,
            wf_info.is_distinct,
            &wf_info.extra_args,
            order_by,
        )
        .ok_or_else(|| Error::NotSupported(format!("Unknown aggregate function: {}", wf_info.name)))
    }

    /// The keys the call's own ORDER BY gives each row, empty without one or
    /// for an aggregate that does not order its input
    fn window_call_sort_keys(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        ctx: &ExecutionContext,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Vec<Vec<Value>>> {
        if wf_info.call_order_by.is_empty()
            || !self
                .function_registry
                .get_aggregate(&wf_info.name)
                .is_some_and(|f| f.supports_order_by())
        {
            return Ok(Vec::new());
        }
        let mut evals = RowPrograms::compile(
            wf_info.call_order_by.iter().map(|order| &order.expression),
            columns,
            None,
            ctx,
            blocks,
        )?;
        let mut sort_keys = Vec::with_capacity(rows.len());
        for (_, row) in rows {
            let mut key = Vec::with_capacity(wf_info.call_order_by.len());
            evals.eval_into(row, &mut key)?;
            sort_keys.push(key);
        }
        Ok(sort_keys)
    }

    /// Compute aggregate function as window function (SUM, COUNT, AVG, MIN, MAX)
    /// cached_order_by: Optional precomputed ORDER BY values from cache to avoid redundant computation
    #[allow(clippy::too_many_arguments)]
    fn compute_aggregate_window_function(
        &self,
        wf_info: &WindowFunctionInfo,
        rows: &[(i64, Row)],
        columns: &[String],
        col_index_map: &StringMap<usize>,
        group_slots: &FxHashMap<ExprKey, usize>,
        ctx: &ExecutionContext,
        pre_sorted: Option<&WindowPreSortedState>,
        cached_order_by: Option<&ColumnarOrderByValues>,
        partition_cache: &mut Vec<(OverClauseKey, SharedSortedPartitions)>,
        args: &CallArgs,
        blocks: Option<&dyn SubqueryExecutor>,
    ) -> Result<Vec<Value>> {
        // Check if this is COUNT(*) - Star expression means count all rows
        let is_count_star =
            !wf_info.arguments.is_empty() && matches!(wf_info.arguments[0], Expression::Star(_));

        // Get the column index for the aggregate argument
        let mut arg_col_idx: Option<usize> = if !wf_info.arguments.is_empty() && !is_count_star {
            self.resolve_column_index(&wf_info.arguments[0], col_index_map, group_slots)
        } else {
            // COUNT(*) has no arguments or Star expression
            None
        };

        // Check if the argument is an expression that needs evaluation
        // This handles cases like SUM(val * 2) or SUM(SUM(val)) in grouped results
        // But NOT for COUNT(*) which should just count rows
        let has_expression_arg =
            !wf_info.arguments.is_empty() && !is_count_star && arg_col_idx.is_none();

        // Pre-compute expression values for all rows if needed
        let mut expression_values: Vec<Value> = if has_expression_arg {
            let mut eval =
                RowPrograms::compile(&wf_info.arguments[..1], columns, None, ctx, blocks)?;
            rows.iter()
                .map(|(_, row)| eval.eval(row))
                .collect::<Result<_>>()?
        } else {
            vec![]
        };

        // A FILTER hides a row by handing the aggregate a NULL, which every
        // aggregate skips, so the frames below read one plain value column
        if let Some(filter) = wf_info.filter.as_deref() {
            expression_values = self.window_filtered_values(
                filter,
                rows,
                columns,
                ctx,
                arg_col_idx,
                &expression_values,
                blocks,
            )?;
            arg_col_idx = None;
        }

        // The keys the call's own ORDER BY gives each row, empty without one
        let sort_keys = self.window_call_sort_keys(wf_info, rows, columns, ctx, blocks)?;
        let keys: Option<&[Vec<Value>]> = (!sort_keys.is_empty()).then_some(&sort_keys);

        // Use precomputed ORDER BY values from cache if available
        let empty_order_by = ColumnarOrderByValues {
            columns: vec![],
            ascending: vec![],
            nulls_first: vec![],
            num_rows: 0,
        };
        let order_by_values: &ColumnarOrderByValues = cached_order_by.unwrap_or(&empty_order_by);

        // Check if we can skip sorting (index optimization)
        // Only applies when there's no PARTITION BY (single partition)
        let skip_sorting =
            wf_info.partition_by_exprs.is_empty() && self.check_rows_presorted(wf_info, pre_sorted);

        // Resolve partitions once per distinct OVER clause (map build and
        // per-partition sort shared across window functions). The
        // presorted-by-index path keeps unsorted indices, so it bypasses
        // the shared cache
        let owned_partitions: Option<Vec<Vec<usize>>> = if skip_sorting {
            Some(
                Self::build_partition_map(
                    wf_info,
                    rows,
                    columns,
                    col_index_map,
                    group_slots,
                    ctx,
                    blocks,
                )?
                .into_values()
                .collect(),
            )
        } else {
            None
        };
        let shared_partitions = match &owned_partitions {
            Some(_) => None,
            None => Some(self.shared_sorted_partitions(
                wf_info,
                rows,
                columns,
                col_index_map,
                group_slots,
                ctx,
                None,
                cached_order_by,
                partition_cache,
                blocks,
            )?),
        };
        // Borrow the shared entry instead of cloning the index vectors
        let partitions: &[Vec<usize>] = match (&owned_partitions, &shared_partitions) {
            (Some(owned), _) => owned,
            (None, Some(shared)) => shared,
            (None, None) => &[],
        };

        // Compute aggregate for each partition
        // OPTIMIZATION: Use Option<Value> with vec![None; n] - None is just a discriminant
        // (no data to clone), much faster than vec![NULL_VALUE; n] which clones ~32 byte Values
        // No row, no frame offset evaluates
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(error) = &wf_info.extra_error {
            return Err(error.clone());
        }
        let frame = Frame::read(wf_info, args, ctx)?;
        let mut results: Vec<Option<Value>> = vec![None; rows.len()];

        for row_indices in partitions.iter() {
            let row_indices = row_indices.as_slice();
            // Partitions from the shared cache arrive pre-sorted
            if !wf_info.order_by.is_empty() || wf_info.frame.is_some() {
                let frames = PartitionFrames::new(&frame, row_indices, order_by_values);

                // Get aggregate function ONCE, reuse with reset() for each row
                let mut agg_func = self.window_aggregate_state(wf_info, keys.is_some())?;

                // O(n) incremental accumulation instead of O(n²) reset when
                // the frame only grows; DISTINCT needs the full set
                if frame.only_grows() && !wf_info.is_distinct {
                    // O(n) incremental path: frame start is always 0, only end grows.
                    // Accumulate new values as frame_end advances, never reset.
                    let mut prev_end = 0usize;
                    for (i, &row_idx) in row_indices.iter().enumerate() {
                        let frame_end = frames.bounds(&frame, i).1;

                        // Accumulate only the new values since prev_end
                        if keys.is_none() {
                            for &idx in &row_indices[prev_end..frame_end] {
                                let value = window_argument_value(
                                    rows,
                                    idx,
                                    arg_col_idx,
                                    &expression_values,
                                );
                                agg_func.accumulate(value);
                            }
                        } else {
                            accumulate_ordered_frame(
                                &mut agg_func,
                                &row_indices[prev_end..frame_end],
                                rows,
                                arg_col_idx,
                                &expression_values,
                                sort_keys.as_slice(),
                            );
                        }
                        if frame_end > prev_end {
                            prev_end = frame_end;
                        }
                        results[row_idx] = Some(agg_func.result()?);
                    }
                } else {
                    for (i, &row_idx) in row_indices.iter().enumerate() {
                        // Reset aggregate state for new frame computation
                        agg_func.reset();
                        let (frame_start, frame_end) = frames.bounds(&frame, i);

                        // Accumulate values within the frame
                        if keys.is_none() {
                            for &idx in &row_indices[frame_start..frame_end] {
                                let value = window_argument_value(
                                    rows,
                                    idx,
                                    arg_col_idx,
                                    &expression_values,
                                );
                                agg_func.accumulate(value);
                            }
                        } else {
                            accumulate_ordered_frame(
                                &mut agg_func,
                                &row_indices[frame_start..frame_end],
                                rows,
                                arg_col_idx,
                                &expression_values,
                                sort_keys.as_slice(),
                            );
                        }
                        results[row_idx] = Some(agg_func.result()?);
                    }
                } // end of else (non-incremental path)
            } else {
                // Without ORDER BY or a frame, compute aggregate over entire partition
                let mut agg_func = self.window_aggregate_state(wf_info, keys.is_some())?;

                // Accumulate all values in the partition
                if keys.is_none() {
                    for &idx in row_indices {
                        let value =
                            window_argument_value(rows, idx, arg_col_idx, &expression_values);
                        agg_func.accumulate(value);
                    }
                } else {
                    accumulate_ordered_frame(
                        &mut agg_func,
                        row_indices,
                        rows,
                        arg_col_idx,
                        &expression_values,
                        sort_keys.as_slice(),
                    );
                }
                let aggregate_result = agg_func.result()?;

                // Assign the same aggregate result to all rows in the partition
                for &row_idx in row_indices {
                    results[row_idx] = Some(aggregate_result.clone());
                }
            }
        }

        // Unwrap all values (all indices should have been written)
        Ok(results
            .into_iter()
            .map(|opt| opt.unwrap_or(NULL_VALUE))
            .collect())
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

    fn setup_test_data(executor: &Executor) {
        executor
            .execute(
                "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER)",
            )
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 100000)")
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 90000)")
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (3, 'Carol', 'Sales', 80000)")
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (4, 'Dave', 'Sales', 85000)")
            .unwrap();
        executor
            .execute("INSERT INTO employees VALUES (5, 'Eve', 'Engineering', 95000)")
            .unwrap();
    }

    #[test]
    fn test_row_number_basic() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT name, ROW_NUMBER() OVER () FROM employees")
            .unwrap();

        let columns = result.columns();
        assert!(columns.len() >= 2);

        let mut count = 0;
        while result.next() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_row_number_with_order() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT name, ROW_NUMBER() OVER (ORDER BY salary DESC) FROM employees")
            .unwrap();

        let mut found_rows = false;
        while result.next() {
            found_rows = true;
        }
        assert!(found_rows);
    }

    #[test]
    fn test_row_number_with_partition() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT name, dept, ROW_NUMBER() OVER (PARTITION BY dept) FROM employees")
            .unwrap();

        let mut count = 0;
        while result.next() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[test]
    fn test_has_window_functions() {
        use crate::executor::query_classification::get_classification;

        // Test with window function
        let mut parser = crate::parser::Parser::new("SELECT ROW_NUMBER() OVER () FROM test");
        if let Ok(program) = parser.parse_program() {
            if let crate::parser::ast::Statement::Select(stmt) = &program.statements[0] {
                let classification = get_classification(stmt);
                assert!(classification.has_window_functions);
            }
        }

        // Test without window function
        let mut parser2 = crate::parser::Parser::new("SELECT * FROM test");
        if let Ok(program) = parser2.parse_program() {
            if let crate::parser::ast::Statement::Select(stmt) = &program.statements[0] {
                let classification = get_classification(stmt);
                assert!(!classification.has_window_functions);
            }
        }
    }

    #[test]
    fn test_window_function_info() {
        let info = WindowFunctionInfo {
            name: "ROW_NUMBER".to_string(),
            arguments: vec![],
            partition_by: vec!["dept".to_string()],
            partition_by_exprs: vec![],
            order_by: vec![],
            frame: None,
            column_name: "rn".to_string(),
            is_distinct: false,
            filter: None,
            call_order_by: Box::new([]),
            extra_args: Vec::new(),
            extra_error: None,
        };

        assert_eq!(info.name, "ROW_NUMBER");
        assert_eq!(info.partition_by.len(), 1);
        assert_eq!(info.column_name, "rn");
    }

    #[test]
    fn test_percent_rank_with_order() {
        let executor = create_test_executor();
        setup_test_data(&executor);

        let mut result = executor
            .execute("SELECT salary, PERCENT_RANK() OVER (ORDER BY salary) AS pct FROM employees ORDER BY salary")
            .unwrap();

        let mut pct_ranks = Vec::new();
        let mut row_count = 0;
        while result.next() {
            let row = result.row();
            if let Some(pct) = row.get(1) {
                match pct {
                    crate::core::Value::Float(f) => pct_ranks.push(*f),
                    crate::core::Value::Integer(i) => pct_ranks.push(*i as f64),
                    _ => {}
                }
            }
            row_count += 1;
        }

        eprintln!("DEBUG: pct_ranks = {:?}", pct_ranks);
        eprintln!("DEBUG: row_count = {}", row_count);
        eprintln!("DEBUG: columns = {:?}", result.columns());

        // First row should have pct_rank = 0.0
        assert!(
            (pct_ranks[0] - 0.0).abs() < 0.001,
            "First pct_rank should be 0.0, got {}",
            pct_ranks[0]
        );

        // Verify monotonically non-decreasing
        for i in 1..pct_ranks.len() {
            assert!(
                pct_ranks[i] >= pct_ranks[i - 1],
                "pct_ranks should be non-decreasing"
            );
        }
    }

    /// The SELECT expression of `SELECT {text}`
    fn select_expression(text: &str) -> Expression {
        match &crate::parser::parse_sql(&format!("SELECT {text}")).unwrap()[0] {
            Statement::Select(select) => select.columns[0].clone(),
            other => panic!("not a SELECT: {other:?}"),
        }
    }

    #[test]
    fn a_window_argument_is_compiled_for_its_call_outside_the_program_cache() {
        let executor = create_test_executor();
        setup_test_data(&executor);
        for (sql, argument) in [
            (
                "SELECT NTILE(1 + 700201) OVER (PARTITION BY dept ORDER BY id) FROM employees",
                "1 + 700201",
            ),
            (
                "SELECT LAG(salary, 700203 - 700202) OVER (PARTITION BY dept ORDER BY id) \
                 FROM employees",
                "700203 - 700202",
            ),
            (
                "SELECT NTH_VALUE(salary, 700205 - 700204) OVER (PARTITION BY dept ORDER BY id) \
                 FROM employees",
                "700205 - 700204",
            ),
            (
                "SELECT FIRST_VALUE(salary) OVER (PARTITION BY dept ORDER BY id \
                 ROWS BETWEEN 700207 - 700206 PRECEDING AND CURRENT ROW) FROM employees",
                "700207 - 700206",
            ),
            (
                "SELECT SUM(salary) OVER (PARTITION BY dept ORDER BY id \
                 ROWS BETWEEN 700209 - 700208 PRECEDING AND CURRENT ROW) FROM employees",
                "700209 - 700208",
            ),
            (
                "SELECT NTILE(1 + 700210) OVER (PARTITION BY dept ORDER BY id) FROM employees \
                 LIMIT 3",
                "1 + 700210",
            ),
        ] {
            let _ = executor.execute(sql).unwrap();
            assert!(
                !crate::executor::expression::program_cached(&select_expression(argument), &[]),
                "{sql}: `{argument}` entered the program cache"
            );
        }
    }
}
