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

// Expression Virtual Machine
//
// The VM executes compiled Programs against row data.
// Design goals:
// - Zero allocation in hot path
// - Linear instruction dispatch
// - Reusable across rows (clear() between uses)
// - No recursion

use std::borrow::Cow;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::ops::{row_in, CompareOp, IntervalStep, Op};
use super::program::Program;
use crate::common::{CompactArc, SmartString};
use crate::core::integer::{integer_neg, integer_op, IntegerOp};
use crate::core::interval::{self, Interval, IntervalUnit};
use crate::core::pattern::Pattern;
use crate::core::predicate::{sql_compare, sql_distinct, sql_eq, sql_rows_distinct};
use crate::core::value::{f64_eq, f64_ge, f64_gt, f64_le, f64_lt, f64_ne};
use crate::core::{DataType, Error, Result, Row, Value, NULL_VALUE};
use crate::executor::context::KeptMembers;
use crate::functions::scalar::extreme_position;
use std::cmp::Ordering;

/// The VM's value stack for one run: a value loaded from the row, the
/// program's constants or the parameters is borrowed, a computed one owned
struct Stack<'a>(SmallVec<[Cow<'a, Value>; STACK_INLINE_CAPACITY]>);

impl<'a> Stack<'a> {
    #[inline(always)]
    fn new() -> Self {
        Self(SmallVec::new())
    }

    #[inline]
    fn push(&mut self, value: Value) {
        self.0.push(Cow::Owned(value));
    }

    #[inline]
    fn push_ref(&mut self, value: &'a Value) {
        self.0.push(Cow::Borrowed(value));
    }

    #[inline]
    fn push_cow(&mut self, value: Cow<'a, Value>) {
        self.0.push(value);
    }

    /// The values from `start` to the top replaced by the one at `kept`,
    /// or by NULL when there is none
    #[inline]
    fn take_one(&mut self, start: usize, kept: Option<usize>) {
        let value = match kept {
            Some(at) => self.0.swap_remove(at),
            None => Cow::Borrowed(&NULL_VALUE),
        };
        self.0.truncate(start);
        self.0.push(value);
    }

    /// The top value, NULL when the stack is empty
    #[inline]
    fn pop(&mut self) -> Cow<'a, Value> {
        self.0.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE))
    }

    #[inline]
    fn last(&self) -> Option<&Value> {
        self.0.last().map(|v| &**v)
    }

    #[inline]
    fn last_mut(&mut self) -> Option<&mut Value> {
        self.0.last_mut().map(Cow::to_mut)
    }

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        self.0.swap(a, b);
    }

    /// The top value pushed again; nothing on an empty stack
    #[inline]
    fn dup(&mut self) {
        if let Some(top) = self.0.last().cloned() {
            self.0.push(top);
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.0.clear();
    }

    /// The values from `start` to the top
    #[inline]
    fn from(&self, start: usize) -> &[Cow<'a, Value>] {
        &self.0[start..]
    }

    /// The values from `start` to the top, taken off the stack
    #[inline]
    fn drain(&mut self, start: usize) -> impl Iterator<Item = Value> + use<'_, 'a> {
        self.0.drain(start..).map(Cow::into_owned)
    }
}

/// A predicate's result as a value: None is UNKNOWN, a boolean NULL
#[inline]
fn truth(result: Option<bool>) -> Value {
    result.map_or(Value::Null(DataType::Boolean), Value::Boolean)
}

/// `value LIKE pattern` or `GLOB`, the value read as its text form; NULL
/// for a NULL
#[inline]
fn pattern_match(value: &Value, pattern: &Pattern) -> Value {
    truth(pattern.matches_value(value))
}

/// `value REGEXP regex`, the value read as its text form; NULL for a NULL
#[inline]
fn regexp_match(value: &Value, regex: &regex::Regex) -> Value {
    truth(value.text_form().map(|text| regex.is_match(&text)))
}

/// [`concat_values`] out of line: inlined, it costs the VM's op loop registers
#[inline(never)]
fn concat_text<'v>(parts: impl Iterator<Item = &'v Value> + Clone) -> Value {
    concat_values(parts)
}

/// `parts` joined by `||` in order: NULL of type TEXT when one is NULL,
/// else each part's text, a part of another type written as it displays
#[inline]
pub(crate) fn concat_values<'v>(parts: impl Iterator<Item = &'v Value> + Clone) -> Value {
    let mut total_len = 0usize;
    let mut count = 0usize;
    let mut all_text = true;
    for part in parts.clone() {
        count += 1;
        match part {
            Value::Null(_) => return Value::Null(DataType::Text),
            Value::Text(text) => total_len += text.len(),
            _ => {
                all_text = false;
                total_len += 32;
            }
        }
    }
    if all_text {
        // Up to eight texts are joined straight into the result, any more
        // through a buffer at their exact length
        if count <= 8 {
            let texts: SmallVec<[&str; 8]> = parts
                .filter_map(|part| match part {
                    Value::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            return Value::Text(SmartString::concat_many(&texts));
        }
        if total_len <= 15 {
            let mut buf = [0u8; 15];
            let mut pos = 0;
            for part in parts.clone() {
                if let Value::Text(text) = part {
                    buf[pos..pos + text.len()].copy_from_slice(text.as_bytes());
                    pos += text.len();
                }
            }
            if let Ok(text) = std::str::from_utf8(&buf[..pos]) {
                return Value::Text(SmartString::new(text));
            }
        }
        let mut s = String::with_capacity(total_len);
        for part in parts {
            if let Value::Text(text) = part {
                s.push_str(text);
            }
        }
        return Value::Text(SmartString::from_string(s));
    }
    let mut s = String::with_capacity(total_len);
    for part in parts {
        match part {
            Value::Text(text) => s.push_str(text),
            _ => {
                use std::fmt::Write;
                let _ = write!(s, "{}", part);
            }
        }
    }
    Value::Text(SmartString::from_string_shared(s))
}

/// A parameter tested against a column, read as the column's type
/// `read_as` as the compiler reads a literal there
#[inline]
pub(crate) fn read_param(value: &Value, read_as: Option<DataType>) -> Cow<'_, Value> {
    match read_as {
        Some(column_type) if reads_otherwise(value, column_type) => Cow::Owned(
            crate::executor::pushdown::read_as_column(column_type, value.clone()),
        ),
        _ => Cow::Borrowed(value),
    }
}

/// Whether a parameter read as `column_type` reads as another value than it
/// is: a NULL, or a value of another type
#[inline]
pub(crate) fn reads_otherwise(value: &Value, column_type: DataType) -> bool {
    value.is_null() || value.data_type() != column_type
}

/// `value IN (items)`: `value = item OR value = item ...`, with OR's
/// UNKNOWN
fn in_list<'v>(value: &Value, items: impl Iterator<Item = &'v Value>) -> Option<bool> {
    let mut result = Some(false);
    for item in items {
        match sql_eq(value, item) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => result = None,
        }
    }
    result
}

/// `value BETWEEN low AND high`: `value >= low AND value <= high`, where a
/// false half decides even when the other is UNKNOWN
#[cold]
fn between_cell(value: &Value, low: &Value, high: &Value) -> Option<bool> {
    let above = sql_compare(value, low).map(|o| o.is_ge());
    let below = sql_compare(value, high).map(|o| o.is_le());
    match (above, below) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// Stack capacity for inline storage (avoids heap allocation for simple expressions)
/// Most expressions need 4-8 stack slots; a deeper one, a long concatenation
/// or a call with many arguments, still runs without allocating
const STACK_INLINE_CAPACITY: usize = 32;

/// Execution context for the VM
///
/// Contains all the data needed to evaluate an expression:
/// - Current row
/// - Optional second row (for joins)
/// - Outer row context (for correlated subqueries)
/// - Query parameters
pub struct ExecuteContext<'a> {
    /// Primary row data
    pub row: &'a Row,

    /// Second row for joins (optional)
    pub row2: Option<&'a Row>,

    /// Outer row context for correlated subqueries
    pub outer_row: Option<&'a FxHashMap<CompactArc<str>, Value>>,

    /// Parameter values, by slot
    pub params: &'a [Value],

    /// Subquery executor (for scalar subqueries, EXISTS, etc.)
    /// This is a callback that can execute subqueries
    pub subquery_executor: Option<&'a dyn SubqueryExecutor>,

    /// Current transaction ID (for CURRENT_TRANSACTION_ID())
    pub transaction_id: Option<u64>,

    /// Slots whose value failed to compute, each NULL in `params`; reading
    /// one raises its error
    pub failed: &'a [(usize, Error)],
}

impl<'a> ExecuteContext<'a> {
    /// Create a simple context with just a row
    #[inline]
    pub fn new(row: &'a Row) -> Self {
        Self {
            row,
            row2: None,
            outer_row: None,
            params: &[],
            subquery_executor: None,
            transaction_id: None,
            failed: &[],
        }
    }

    /// Create a context with common parameters that can be reused across rows
    /// Only the row field needs to be updated for each iteration
    #[inline]
    pub fn with_common_params(
        row: &'a Row,
        params: &'a [Value],
        transaction_id: Option<u64>,
    ) -> Self {
        Self {
            row,
            row2: None,
            outer_row: None,
            params,
            subquery_executor: None,
            transaction_id,
            failed: &[],
        }
    }

    /// Create context for join evaluation
    pub fn for_join(row1: &'a Row, row2: &'a Row) -> Self {
        Self {
            row: row1,
            row2: Some(row2),
            outer_row: None,
            params: &[],
            subquery_executor: None,
            transaction_id: None,
            failed: &[],
        }
    }

    /// Add transaction ID
    pub fn with_transaction_id(mut self, transaction_id: Option<u64>) -> Self {
        self.transaction_id = transaction_id;
        self
    }

    /// Add parameters
    pub fn with_params(mut self, params: &'a [Value]) -> Self {
        self.params = params;
        self
    }

    /// Add outer row context
    pub fn with_outer_row(mut self, outer_row: &'a FxHashMap<CompactArc<str>, Value>) -> Self {
        self.outer_row = Some(outer_row);
        self
    }

    /// Add the slots whose value failed to compute
    pub fn with_failed(mut self, failed: &'a [(usize, Error)]) -> Self {
        self.failed = failed;
        self
    }

    /// Add subquery executor, with the slots whose value failed in its run
    pub fn with_subquery_executor(mut self, executor: &'a dyn SubqueryExecutor) -> Self {
        self.subquery_executor = Some(executor);
        self.failed = executor.failed_slots();
        self
    }

    /// Parameter slot `index`, zero-based; a slot without a value, or one
    /// whose value failed to compute, is an error
    #[inline]
    pub fn param_ref(&self, index: usize) -> Result<&'a Value> {
        match self.params.get(index) {
            Some(value) if self.failed.is_empty() => Ok(value),
            Some(value) => read_failed(self.failed, index, value),
            None => Err(missing_parameter(index, self.params.len())),
        }
    }
}

/// `value`, slot `index`'s, or the error the slot failed with; a failed
/// slot holds NULL, so a value that is not NULL never searches `failed`
#[cold]
#[inline(never)]
fn read_failed<'a>(failed: &[(usize, Error)], index: usize, value: &'a Value) -> Result<&'a Value> {
    if !value.is_null() {
        return Ok(value);
    }
    match failed.iter().find(|(slot, _)| *slot == index) {
        Some((_, e)) => Err(e.clone()),
        None => Ok(value),
    }
}

#[cold]
fn missing_parameter(index: usize, provided: usize) -> Error {
    Error::invalid_argument(format!(
        "Query requires {} parameters but only {} provided",
        index + 1,
        provided
    ))
}

/// Runs a prepared plan's subquery blocks for the context being evaluated,
/// whose row or pair of rows a correlated block reads; a `None` answer is
/// UNKNOWN, and an error is the statement's error
pub trait SubqueryExecutor {
    /// The scalar block's value
    fn execute_scalar(&self, subquery_index: u16, ctx: &ExecuteContext) -> Result<Value>;

    /// Whether a row passes the block's WHERE
    fn execute_exists(&self, subquery_index: u16, ctx: &ExecuteContext) -> Result<bool>;

    /// `value IN (block)`
    fn execute_in(
        &self,
        subquery_index: u16,
        value: &Value,
        ctx: &ExecuteContext,
    ) -> Result<Option<bool>>;

    /// `value op ALL (block)` when `all`, else `value op ANY (block)`
    fn execute_quantified(
        &self,
        subquery_index: u16,
        value: &Value,
        op: CompareOp,
        all: bool,
        ctx: &ExecuteContext,
    ) -> Result<Option<bool>>;

    /// `row` tested against a row block's members by the test it answers
    fn execute_row_members(
        &self,
        _subquery_index: u16,
        _row: &[Value],
        _ctx: &ExecuteContext,
    ) -> Result<Option<bool>> {
        Err(no_subquery_executor())
    }

    /// An uncorrelated block's members once this run has read them, which
    /// the VM tests in place of calling the block
    fn kept_members(&self, _subquery_index: u16) -> Option<&KeptMembers> {
        None
    }

    /// The parameters the statement's programs read beside these blocks:
    /// its own, then the values its run computed once; None when the
    /// caller's own parameters are all there are
    fn params(&self) -> Option<&[Value]> {
        None
    }

    /// The slots of [`Self::params`] whose value failed to compute in this
    /// run, each with its error, which a read of the slot raises
    fn failed_slots(&self) -> &[(usize, Error)] {
        &[]
    }

    /// The error of a named window's slot that failed to compute in this
    /// run, which its window raises once it runs over a row
    fn window_failure(&self) -> Option<&Error> {
        None
    }

    /// `program` with every block it tests members of read once and tested
    /// as that set in place; None when it calls a block another way
    fn bind_members(&self, _program: &CompactArc<Program>) -> Option<CompactArc<Program>> {
        None
    }
}

/// A subquery op compiles only into a prepared plan, which runs with its executor
fn no_subquery_executor() -> crate::core::Error {
    crate::core::Error::internal("a subquery ran outside its prepared plan")
}

/// Expression Virtual Machine
///
/// Executes compiled Programs against row data.
/// The VM is reusable - call execute() with different contexts.
/// Capacity for reusable args buffer (most functions have <= 4 args)
const ARGS_BUFFER_CAPACITY: usize = 8;

pub struct ExprVM {
    /// Reusable buffer for function arguments (avoids allocation per call)
    /// Uses SmallVec to avoid heap allocation for functions with <= 8 args
    args_buffer: SmallVec<[Value; ARGS_BUFFER_CAPACITY]>,

    /// Cache for dynamic LIKE patterns (avoids recompilation per row)
    /// Stores (pattern_string, case_insensitive, escape_char, compiled_pattern)
    cached_like: Option<(SmartString, bool, Option<char>, Pattern)>,

    /// Cache for dynamic GLOB patterns (separate from LIKE to avoid cross-contamination)
    /// Stores (pattern_string, compiled_pattern)
    cached_glob: Option<(SmartString, Pattern)>,

    /// Cache for dynamic REGEXP patterns (avoids recompilation per row)
    /// Stores (pattern_string, compiled_regex)
    cached_regexp: Option<(SmartString, regex::Regex)>,

    /// The error `execute_bool` read as false, kept for `take_bool_error`
    bool_error: Option<crate::core::Error>,
}

impl ExprVM {
    /// Create a new VM
    /// Uses inline storage for up to 8 args (no heap allocation)
    pub fn new() -> Self {
        Self {
            args_buffer: SmallVec::new(),
            cached_like: None,
            cached_glob: None,
            cached_regexp: None,
            bool_error: None,
        }
    }

    /// The values from `start` to the top taken off `stack` into the
    /// arguments buffer, for an op that reads them as one slice; the caller
    /// gives the buffer back
    #[inline]
    fn take_values(
        &mut self,
        stack: &mut Stack<'_>,
        start: usize,
    ) -> SmallVec<[Value; ARGS_BUFFER_CAPACITY]> {
        let mut values = std::mem::take(&mut self.args_buffer);
        values.clear();
        values.extend(stack.drain(start));
        values
    }

    /// Drop every value a run left behind, its arguments and cached
    /// patterns, keeping the buffers' capacity
    pub(crate) fn release_values(&mut self) {
        self.args_buffer.clear();
        self.cached_like = None;
        self.cached_glob = None;
        self.cached_regexp = None;
        self.bool_error = None;
    }

    /// Execute a program and return the result
    #[inline]
    /// Match a runtime pattern, caching the compiled form per (pattern, case, escape).
    fn like_dynamic_escape(
        &mut self,
        text_val: &Value,
        pattern_val: &Value,
        ci: bool,
        esc: Option<char>,
    ) -> Value {
        // Both sides are read as their text form; a NULL on either is UNKNOWN
        let (Some(text), Some(pat)) = (text_val.text_form(), pattern_val.text_form()) else {
            return Value::Null(DataType::Boolean);
        };
        if let Some((cached_pat, cached_ci, cached_esc, compiled)) = &self.cached_like {
            if cached_pat.as_str() == pat && *cached_ci == ci && *cached_esc == esc {
                return Value::Boolean(compiled.matches(&text));
            }
        }
        // !% becomes \% so the pattern reads the wildcard as literal
        let processed = match esc {
            Some(esc) => super::compiler::like_escape_pattern(&pat, esc),
            None => pat.to_string(),
        };
        let compiled = Pattern::like(&processed, ci);
        let matched = compiled.matches(&text);
        self.cached_like = Some((SmartString::new(&pat), ci, esc, compiled));
        Value::Boolean(matched)
    }

    /// Match a runtime GLOB pattern, caching the compiled form per pattern
    fn glob_dynamic(&mut self, text_val: &Value, pattern_val: &Value) -> Value {
        let (Some(text), Some(pat)) = (text_val.text_form(), pattern_val.text_form()) else {
            return Value::Null(DataType::Boolean);
        };
        if let Some((cached_pat, compiled)) = &self.cached_glob {
            if cached_pat.as_str() == pat {
                return Value::Boolean(compiled.matches(&text));
            }
        }
        let compiled = Pattern::glob(&pat);
        let matched = compiled.matches(&text);
        self.cached_glob = Some((SmartString::new(&pat), compiled));
        Value::Boolean(matched)
    }

    /// Match a runtime REGEXP pattern, caching the compiled regex per pattern;
    /// an invalid one is an error, as a literal one is when compiled
    fn regexp_dynamic(&mut self, text_val: &Value, pattern_val: &Value) -> Result<Value> {
        let (Some(text), Some(pat)) = (text_val.text_form(), pattern_val.text_form()) else {
            return Ok(Value::Null(DataType::Boolean));
        };
        if let Some((cached_pat, re)) = &self.cached_regexp {
            if cached_pat.as_str() == pat {
                return Ok(Value::Boolean(re.is_match(&text)));
            }
        }
        let re = regex::Regex::new(&pat).map_err(|e| {
            crate::core::Error::invalid_argument(format!(
                "Invalid regular expression '{}': {}",
                pat, e
            ))
        })?;
        let matched = re.is_match(&text);
        self.cached_regexp = Some((SmartString::new(&pat), re));
        Ok(Value::Boolean(matched))
    }

    /// The value `program` computes over `ctx`'s row
    pub fn execute<'a>(
        &mut self,
        program: &'a Program,
        ctx: &'a ExecuteContext<'a>,
    ) -> Result<Value> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::program_ran(program);
        let mut stack = Stack::new();

        let ops = program.ops();
        if ops.is_empty() {
            return Ok(Value::null_unknown());
        }

        let mut pc: usize = 0;

        // Main execution loop
        loop {
            if pc >= ops.len() {
                break;
            }

            match &ops[pc] {
                // =============================================================
                // LOAD OPERATIONS
                // =============================================================
                Op::LoadColumn(idx) => {
                    stack.push_ref(ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE));
                    pc += 1;
                }

                Op::LoadColumn2(idx) => {
                    let value = ctx.row2.and_then(|r| r.get(*idx as usize));
                    stack.push_ref(value.unwrap_or(&NULL_VALUE));
                    pc += 1;
                }

                Op::LoadConst(value) => {
                    stack.push_ref(value);
                    pc += 1;
                }

                Op::LoadParam(idx) => {
                    stack.push_ref(ctx.param_ref(*idx as usize)?);
                    pc += 1;
                }

                Op::LoadNull(dt) => {
                    stack.push(Value::Null(*dt));
                    pc += 1;
                }

                Op::LoadAggregateResult(idx) => {
                    // Aggregate results are stored in the row at specific indices
                    stack.push_ref(ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE));
                    pc += 1;
                }

                // =============================================================
                // COMPARISON OPERATIONS
                // =============================================================
                Op::Eq => {
                    let b = stack.pop();
                    let a = stack.pop();
                    stack.push(truth(sql_eq(&a, &b)));
                    pc += 1;
                }

                Op::Ne => {
                    let b = stack.pop();
                    let a = stack.pop();
                    stack.push(truth(sql_eq(&a, &b).map(|eq| !eq)));
                    pc += 1;
                }

                Op::Lt => {
                    let b = stack.pop();
                    let a = stack.pop();
                    // Fast path for integer/float comparisons
                    let result = match (&*a, &*b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x < *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(f64_lt(*x, *y)),
                        (Value::Integer(x), Value::Float(y)) => {
                            Value::Boolean(crate::core::value::i64_lt_f64(*x, *y))
                        }
                        (Value::Float(x), Value::Integer(y)) => {
                            Value::Boolean(crate::core::value::i64_gt_f64(*y, *x))
                        }
                        _ => self.compare_values(&a, &b, std::cmp::Ordering::Less),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::Le => {
                    let b = stack.pop();
                    let a = stack.pop();
                    // Fast path for integer/float comparisons
                    let result = match (&*a, &*b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x <= *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(f64_le(*x, *y)),
                        (Value::Integer(x), Value::Float(y)) => {
                            Value::Boolean(crate::core::value::i64_le_f64(*x, *y))
                        }
                        (Value::Float(x), Value::Integer(y)) => {
                            Value::Boolean(crate::core::value::i64_ge_f64(*y, *x))
                        }
                        _ => self.compare_values_or_equal(&a, &b, std::cmp::Ordering::Less),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::Gt => {
                    let b = stack.pop();
                    let a = stack.pop();
                    // Fast path for integer/float comparisons
                    let result = match (&*a, &*b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x > *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(f64_gt(*x, *y)),
                        (Value::Integer(x), Value::Float(y)) => {
                            Value::Boolean(crate::core::value::i64_gt_f64(*x, *y))
                        }
                        (Value::Float(x), Value::Integer(y)) => {
                            Value::Boolean(crate::core::value::i64_lt_f64(*y, *x))
                        }
                        _ => self.compare_values(&a, &b, std::cmp::Ordering::Greater),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::Ge => {
                    let b = stack.pop();
                    let a = stack.pop();
                    // Fast path for integer/float comparisons
                    let result = match (&*a, &*b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x >= *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(f64_ge(*x, *y)),
                        (Value::Integer(x), Value::Float(y)) => {
                            Value::Boolean(crate::core::value::i64_ge_f64(*x, *y))
                        }
                        (Value::Float(x), Value::Integer(y)) => {
                            Value::Boolean(crate::core::value::i64_le_f64(*y, *x))
                        }
                        _ => self.compare_values_or_equal(&a, &b, std::cmp::Ordering::Greater),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::IsNull => {
                    let v = stack.pop();
                    stack.push(Value::Boolean(v.is_null()));
                    pc += 1;
                }

                Op::IsNotNull => {
                    let v = stack.pop();
                    stack.push(Value::Boolean(!v.is_null()));
                    pc += 1;
                }

                // =============================================================
                // FUSED COMPARISON OPERATIONS
                // Single instruction for column vs constant (avoids push/pop)
                // =============================================================
                Op::EqColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    // Fast path for integer/float comparisons
                    let result = match val {
                        Value::Integer(v) => Self::eq_int(col_val, *v),
                        Value::Float(v) => Self::eq_float(col_val, *v),
                        _ => truth(sql_eq(col_val, val)),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::NeColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    // Fast path for integer/float comparisons
                    let result = match val {
                        Value::Integer(v) => Self::ne_int(col_val, *v),
                        Value::Float(v) => Self::ne_float(col_val, *v),
                        _ => truth(sql_eq(col_val, val).map(|eq| !eq)),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::LtColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    // Fast path for integer/float comparisons
                    let result = match val {
                        Value::Integer(threshold) => Self::lt_int(col_val, *threshold),
                        Value::Float(threshold) => Self::lt_float(col_val, *threshold),
                        _ => self.compare_values(col_val, val, std::cmp::Ordering::Less),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::LeColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    // Fast path for integer/float comparisons
                    let result = match val {
                        Value::Integer(threshold) => Self::le_int(col_val, *threshold),
                        Value::Float(threshold) => Self::le_float(col_val, *threshold),
                        _ => self.compare_values_or_equal(col_val, val, std::cmp::Ordering::Less),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::GtColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    // Fast path for integer/float comparisons
                    let result = match val {
                        Value::Integer(threshold) => Self::gt_int(col_val, *threshold),
                        Value::Float(threshold) => Self::gt_float(col_val, *threshold),
                        _ => self.compare_values(col_val, val, std::cmp::Ordering::Greater),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::GeColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    // Fast path for integer/float comparisons
                    let result = match val {
                        Value::Integer(threshold) => Self::ge_int(col_val, *threshold),
                        Value::Float(threshold) => Self::ge_float(col_val, *threshold),
                        _ => {
                            self.compare_values_or_equal(col_val, val, std::cmp::Ordering::Greater)
                        }
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::IsNullColumn(idx) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    stack.push(Value::Boolean(col_val.is_null()));
                    pc += 1;
                }

                Op::IsNotNullColumn(idx) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    stack.push(Value::Boolean(!col_val.is_null()));
                    pc += 1;
                }

                Op::LikeColumn(idx, pattern) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    stack.push(pattern_match(col_val, pattern));
                    pc += 1;
                }

                Op::InSetColumn(idx, set) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    stack.push(truth(set.contains(col_val)));
                    pc += 1;
                }

                Op::BetweenColumnConst(idx, low, high) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    // Fast path for integer/float BETWEEN
                    let result = match (low, high) {
                        (Value::Integer(lo), Value::Integer(hi)) => {
                            Self::between_int(col_val, *lo, *hi)
                        }
                        (Value::Float(lo), Value::Float(hi)) => {
                            Self::between_float(col_val, *lo, *hi)
                        }
                        (Value::Integer(lo), Value::Float(hi)) => {
                            Self::between_mixed_int_lo(col_val, *lo, *hi)
                        }
                        (Value::Float(lo), Value::Integer(hi)) => {
                            Self::between_mixed_int_hi(col_val, *lo, *hi)
                        }
                        _ => truth(between_cell(col_val, low, high)),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::CompareColumnParam(idx, slot, compare, read_as) => {
                    let col_val = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                    let param = read_param(ctx.param_ref(*slot as usize)?, *read_as);
                    stack.push(truth(compare.compare(col_val, &param)));
                    pc += 1;
                }

                Op::BetweenColumnParams(idx, low, high, read_as) => {
                    let col_val = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                    let low = read_param(ctx.param_ref(*low as usize)?, *read_as);
                    let high = read_param(ctx.param_ref(*high as usize)?, *read_as);
                    stack.push(truth(between_cell(col_val, &low, &high)));
                    pc += 1;
                }

                Op::InColumnParams(idx, slots, read_as) => {
                    let col_val = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                    let items: SmallVec<[Cow<'_, Value>; 8]> = slots
                        .iter()
                        .map(|slot| Ok(read_param(ctx.param_ref(*slot as usize)?, *read_as)))
                        .collect::<Result<_>>()?;
                    stack.push(truth(in_list(col_val, items.iter().map(|v| &**v))));
                    pc += 1;
                }

                // =============================================================
                // LOGICAL OPERATIONS
                // =============================================================
                Op::And(jump_target) => {
                    // Short-circuit AND: if top is false, jump
                    let top = stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    match top {
                        Value::Boolean(false) => {
                            // Result is false, jump to target
                            pc = *jump_target as usize;
                        }
                        Value::Null(_) => {
                            // Need to evaluate right side to check for false
                            pc += 1;
                        }
                        _ => {
                            // True or truthy, continue to evaluate right side
                            pc += 1;
                        }
                    }
                }

                Op::Or(jump_target) => {
                    // Short-circuit OR: if top is true, jump
                    let top = stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    match top {
                        Value::Boolean(true) => {
                            // Result is true, jump to target
                            pc = *jump_target as usize;
                        }
                        Value::Null(_) => {
                            // Need to evaluate right side to check for true
                            pc += 1;
                        }
                        _ => {
                            // False or falsy, continue to evaluate right side
                            pc += 1;
                        }
                    }
                }

                Op::AndFinalize => {
                    let b = stack.pop();
                    let a = stack.pop();
                    stack.push(Self::and(&a, &b));
                    pc += 1;
                }

                Op::OrFinalize => {
                    let b = stack.pop();
                    let a = stack.pop();
                    let result = match (Self::to_tribool(&a), Self::to_tribool(&b)) {
                        (Some(true), _) | (_, Some(true)) => Value::Boolean(true),
                        (Some(false), Some(false)) => Value::Boolean(false),
                        _ => Value::Null(DataType::Boolean),
                    };
                    stack.push(result);
                    pc += 1;
                }

                Op::Not => {
                    let v = stack.pop();
                    let result = match Self::to_tribool(&v) {
                        Some(b) => Value::Boolean(!b),
                        None => Value::Null(DataType::Boolean),
                    };
                    stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // ARITHMETIC OPERATIONS
                // =============================================================
                Op::Add => {
                    let b = stack.pop();
                    let a = stack.pop();
                    let result = Self::add_op(&a, &b)?;
                    stack.push(result);
                    pc += 1;
                }

                Op::Sub => {
                    let b = stack.pop();
                    let a = stack.pop();
                    let result = self.sub_op(&a, &b)?;
                    stack.push(result);
                    pc += 1;
                }

                Op::Mul => {
                    let b = stack.pop();
                    let a = stack.pop();
                    let result = Self::mul_op(&a, &b)?;
                    stack.push(result);
                    pc += 1;
                }

                Op::Div => {
                    let b = stack.pop();
                    let a = stack.pop();
                    let result = Self::div_op(&a, &b)?;
                    stack.push(result);
                    pc += 1;
                }

                Op::Mod => {
                    let b = stack.pop();
                    let a = stack.pop();
                    let result = Self::mod_op(&a, &b)?;
                    stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // STRING OPERATIONS
                // =============================================================
                Op::Concat => {
                    let b = stack.pop();
                    let a = stack.pop();
                    stack.push(concat_text([&*a, &*b].into_iter()));
                    pc += 1;
                }

                Op::ConcatN(n) => {
                    let start = stack.len().saturating_sub(*n as usize);
                    let result = concat_text(stack.from(start).iter().map(|v| &**v));
                    stack.truncate(start);
                    stack.push(result);
                    pc += 1;
                }

                Op::Like(pattern) | Op::Glob(pattern) | Op::LikeEscape(pattern, _) => {
                    // An ESCAPE is already part of the compiled pattern
                    let v = stack.pop();
                    stack.push(pattern_match(&v, pattern));
                    pc += 1;
                }

                // =============================================================
                // SET OPERATIONS
                // =============================================================
                Op::InSet(set) => {
                    let v = stack.pop();
                    stack.push(truth(set.contains(&v)));
                    pc += 1;
                }

                Op::NotInSet(set) => {
                    let v = stack.pop();
                    stack.push(truth(set.contains(&v).map(|found| !found)));
                    pc += 1;
                }

                // =============================================================
                // FUNCTION CALLS
                // =============================================================
                Op::CallScalar { func, arg_count } => {
                    let arg_count = *arg_count as usize;
                    let start = stack.len().saturating_sub(arg_count);

                    // Reuse args_buffer to avoid allocation
                    self.args_buffer.clear();
                    self.args_buffer.extend(stack.drain(start));

                    let result = func.evaluate(&self.args_buffer)?;
                    stack.push(result);
                    pc += 1;
                }

                Op::Coalesce(n) => {
                    let n = *n as usize;
                    let start = stack.len().saturating_sub(n);

                    // Find first non-null using slice iteration (no intermediate buffer)
                    let result_idx = stack
                        .from(start)
                        .iter()
                        .position(|v| !v.is_null())
                        .map(|i| start + i);
                    stack.take_one(start, result_idx);
                    pc += 1;
                }

                Op::NullIf => {
                    let b = stack.pop();
                    let a = stack.pop();
                    let result = if sql_eq(&a, &b) == Some(true) {
                        Cow::Borrowed(&NULL_VALUE)
                    } else {
                        a
                    };
                    stack.push_cow(result);
                    pc += 1;
                }

                Op::Greatest(n) => {
                    let n = *n as usize;
                    let start = stack.len().saturating_sub(n);
                    let max_idx =
                        extreme_position(stack.from(start), Ordering::Greater).map(|i| start + i);
                    stack.take_one(start, max_idx);
                    pc += 1;
                }

                Op::Least(n) => {
                    let n = *n as usize;
                    let start = stack.len().saturating_sub(n);
                    let min_idx =
                        extreme_position(stack.from(start), Ordering::Less).map(|i| start + i);
                    stack.take_one(start, min_idx);
                    pc += 1;
                }

                // =============================================================
                // NATIVE SCALAR FUNCTIONS (direct function pointer call)
                // In-place mutation - no pop/push overhead
                // =============================================================
                Op::NativeFn1(func) => {
                    if let Some(v) = stack.last_mut() {
                        func(v);
                    }
                    pc += 1;
                }

                // =============================================================
                // TYPE OPERATIONS
                // =============================================================
                Op::Cast(target_type) => {
                    let v = stack.pop();
                    let result = v.try_coerce_to_type(*target_type)?;
                    stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // CASE EXPRESSION
                // =============================================================
                Op::CaseStart => {
                    // Marker only, no operation
                    pc += 1;
                }

                Op::CaseWhen(next_branch) => {
                    let cond = stack.pop();
                    if !Self::to_bool(&cond) {
                        pc = *next_branch as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::CaseThen(end_pos) => {
                    // Result is on stack, jump to end
                    pc = *end_pos as usize;
                }

                Op::CaseElse => {
                    // Marker only
                    pc += 1;
                }

                Op::CaseEnd => {
                    // Marker only
                    pc += 1;
                }

                Op::CaseCompare => {
                    let when_val = stack.pop();
                    // Use reference to avoid clone - case_val stays on stack for next comparison
                    let result = match stack.last() {
                        Some(case_val) => Value::Boolean(sql_eq(case_val, &when_val) == Some(true)),
                        _ => Value::Boolean(false),
                    };
                    stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // CONTROL FLOW
                // =============================================================
                Op::Jump(target) => {
                    pc = *target as usize;
                }

                Op::JumpIfTrue(target) => {
                    let top = stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    if Self::to_bool(top) {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::JumpIfFalse(target) => {
                    let top = stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    if !Self::to_bool(top) {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::JumpIfNull(target) => {
                    let top = stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    if top.is_null() {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::JumpIfNotNull(target) => {
                    let top = stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    if !top.is_null() {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::PopJumpIfTrue(target) => {
                    let v = stack.pop();
                    if Self::to_bool(&v) {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::PopJumpIfFalse(target) => {
                    let v = stack.pop();
                    if !Self::to_bool(&v) {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::Pop => {
                    // Use truncate instead of pop to drop in-place without copying value out
                    let new_len = stack.len().saturating_sub(1);
                    stack.truncate(new_len);
                    pc += 1;
                }

                // =============================================================
                // SPECIAL
                // =============================================================
                Op::Nop => {
                    pc += 1;
                }

                Op::Return => {
                    break;
                }

                Op::ReturnTrue => {
                    stack.clear();
                    stack.push(Value::Boolean(true));
                    break;
                }

                Op::ReturnFalse => {
                    stack.clear();
                    stack.push(Value::Boolean(false));
                    break;
                }

                Op::ReturnNull(dt) => {
                    stack.clear();
                    stack.push(Value::Null(*dt));
                    break;
                }

                op @ (Op::LoadOuterColumn(_)
                | Op::LoadTransactionId
                | Op::IsDistinctFrom
                | Op::IsNotDistinctFrom
                | Op::CompareRows(..)
                | Op::RowsDistinct(..)
                | Op::Xor
                | Op::Neg
                | Op::BitAnd
                | Op::BitOr
                | Op::BitXor
                | Op::BitNot
                | Op::Shl
                | Op::Shr
                | Op::Regexp(_)
                | Op::LikeDynamic(_)
                | Op::LikeDynamicEscape(..)
                | Op::LikeDynamicEscapeExpr(_)
                | Op::GlobDynamic
                | Op::RegexpDynamic
                | Op::JsonAccess
                | Op::JsonAccessText
                | Op::TimestampAddInterval
                | Op::TimestampSubInterval
                | Op::TimestampDiff
                | Op::TimestampAddDays
                | Op::TimestampSubDays
                | Op::ScaleInterval(..)
                | Op::TimestampAddScaledInterval(..)
                | Op::TruncateToDate
                | Op::InList(_)
                | Op::QuantifiedSet(_)
                | Op::Between
                | Op::NotBetween
                | Op::InTupleSet { .. }
                | Op::InRows(..)
                | Op::IsTrue
                | Op::IsNotTrue
                | Op::IsFalse
                | Op::IsNotFalse
                | Op::ReadAsColumn(_)
                | Op::ReadAsVector
                | Op::Dup
                | Op::Swap
                | Op::ExecScalarSubquery(_)
                | Op::ExecExists(_)
                | Op::ExecInSubquery(_)
                | Op::ExecQuantified(..)
                | Op::ExecRowMembers(..)
                | Op::VectorDistanceL2
                | Op::VectorDistanceCosine
                | Op::VectorDistanceIP) => {
                    self.execute_rare(op, &mut stack, ctx)?;
                    pc += 1;
                }
            }
        }

        // Return top of stack or NULL
        Ok(stack.pop().into_owned())
    }

    /// The ops a run seldom meets, kept out of [`Self::execute`]'s op loop
    /// so that loop stays small; none of them jumps
    #[inline(never)]
    fn execute_rare<'a>(
        &mut self,
        op: &'a Op,
        stack: &mut Stack<'a>,
        ctx: &'a ExecuteContext<'a>,
    ) -> Result<()> {
        match op {
            Op::LoadOuterColumn(name) => {
                let value = ctx.outer_row.and_then(|r| r.get(name.as_ref()));
                stack.push_ref(value.unwrap_or(&NULL_VALUE));
            }

            Op::LoadTransactionId => {
                // Load current transaction ID, or NULL if not in a transaction
                let value = match ctx.transaction_id {
                    Some(txn_id) => Value::Integer(txn_id as i64),
                    None => Value::null_unknown(),
                };
                stack.push(value);
            }

            Op::IsDistinctFrom => {
                let b = stack.pop();
                let a = stack.pop();
                stack.push(Value::Boolean(sql_distinct(&a, &b)));
            }

            Op::IsNotDistinctFrom => {
                let b = stack.pop();
                let a = stack.pop();
                stack.push(Value::Boolean(!sql_distinct(&a, &b)));
            }

            Op::CompareRows(op, width) => {
                let start = stack.len().saturating_sub(2 * *width as usize);
                let values = self.take_values(stack, start);
                let (a, b) = values.split_at(*width as usize);
                let result = truth(op.compare_rows(a, b));
                self.args_buffer = values;
                stack.push(result);
            }

            Op::RowsDistinct(negated, width) => {
                let start = stack.len().saturating_sub(2 * *width as usize);
                let values = self.take_values(stack, start);
                let (a, b) = values.split_at(*width as usize);
                let result = Value::Boolean(sql_rows_distinct(a, b) != *negated);
                self.args_buffer = values;
                stack.push(result);
            }

            Op::Xor => {
                let b = stack.pop();
                let a = stack.pop();
                let result = match (Self::to_tribool(&a), Self::to_tribool(&b)) {
                    (Some(a), Some(b)) => Value::Boolean(a ^ b),
                    _ => Value::Null(DataType::Boolean),
                };
                stack.push(result);
            }

            Op::Neg => {
                let result = match &*stack.pop() {
                    Value::Integer(i) => Value::Integer(integer_neg(*i)?),
                    Value::Float(f) => Value::Float(-f),
                    Value::Null(dt) => Value::Null(*dt),
                    _ => Value::Null(DataType::Null),
                };
                stack.push(result);
            }

            // =================================================================
            // BITWISE OPERATIONS
            // =================================================================
            Op::BitAnd => {
                let b = stack.pop();
                let a = stack.pop();
                let result = match (&*a, &*b) {
                    (Value::Integer(x), Value::Integer(y)) => Value::Integer(x & y),
                    _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                    _ => Value::Null(DataType::Null),
                };
                stack.push(result);
            }

            Op::BitOr => {
                let b = stack.pop();
                let a = stack.pop();
                let result = match (&*a, &*b) {
                    (Value::Integer(x), Value::Integer(y)) => Value::Integer(x | y),
                    _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                    _ => Value::Null(DataType::Null),
                };
                stack.push(result);
            }

            Op::BitXor => {
                let b = stack.pop();
                let a = stack.pop();
                let result = match (&*a, &*b) {
                    (Value::Integer(x), Value::Integer(y)) => Value::Integer(x ^ y),
                    _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                    _ => Value::Null(DataType::Null),
                };
                stack.push(result);
            }

            Op::BitNot => {
                let result = match &*stack.pop() {
                    Value::Integer(i) => Value::Integer(!i),
                    Value::Null(dt) => Value::Null(*dt),
                    _ => Value::Null(DataType::Null),
                };
                stack.push(result);
            }

            Op::Shl => {
                let b = stack.pop();
                let a = stack.pop();
                let result = match (&*a, &*b) {
                    (Value::Integer(x), Value::Integer(y)) => {
                        integer_op(IntegerOp::Shl, *x, *y)?.map_or(NULL_VALUE, Value::Integer)
                    }
                    _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                    _ => Value::Null(DataType::Null),
                };
                stack.push(result);
            }

            Op::Shr => {
                let b = stack.pop();
                let a = stack.pop();
                let result = match (&*a, &*b) {
                    (Value::Integer(x), Value::Integer(y)) => {
                        integer_op(IntegerOp::Shr, *x, *y)?.map_or(NULL_VALUE, Value::Integer)
                    }
                    _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                    _ => Value::Null(DataType::Null),
                };
                stack.push(result);
            }

            // =================================================================
            // PATTERNS READ AT RUN TIME
            // =================================================================
            Op::Regexp(regex) => {
                let v = stack.pop();
                stack.push(regexp_match(&v, regex));
            }

            Op::LikeDynamic(case_insensitive) => {
                // Parameters are constant per query, so the compiled
                // pattern is cached rather than rebuilt per row
                let pattern_val = stack.pop();
                let text_val = stack.pop();
                let result =
                    self.like_dynamic_escape(&text_val, &pattern_val, *case_insensitive, None);
                stack.push(result);
            }

            Op::LikeDynamicEscape(case_insensitive, escape_char) => {
                let pattern_val = stack.pop();
                let text_val = stack.pop();
                let escape = Some(*escape_char);
                let result =
                    self.like_dynamic_escape(&text_val, &pattern_val, *case_insensitive, escape);
                stack.push(result);
            }

            Op::LikeDynamicEscapeExpr(case_insensitive) => {
                let escape_val = stack.pop();
                let pattern_val = stack.pop();
                let text_val = stack.pop();
                let result = match &*escape_val {
                    Value::Text(escape) => self.like_dynamic_escape(
                        &text_val,
                        &pattern_val,
                        *case_insensitive,
                        escape.chars().next(),
                    ),
                    Value::Null(_) => Value::Null(DataType::Boolean),
                    _ => {
                        return Err(crate::core::Error::invalid_argument(
                            "ESCAPE must be a single-character string",
                        ))
                    }
                };
                stack.push(result);
            }

            Op::GlobDynamic => {
                let pattern_val = stack.pop();
                let text_val = stack.pop();
                let result = self.glob_dynamic(&text_val, &pattern_val);
                stack.push(result);
            }

            Op::RegexpDynamic => {
                let pattern_val = stack.pop();
                let text_val = stack.pop();
                let result = self.regexp_dynamic(&text_val, &pattern_val)?;
                stack.push(result);
            }

            // =================================================================
            // JSON OPERATIONS
            // =================================================================
            Op::JsonAccess => {
                let key = stack.pop();
                let json_val = stack.pop();
                let result = self.json_access(&json_val, &key, false);
                stack.push(result);
            }

            Op::JsonAccessText => {
                let key = stack.pop();
                let json_val = stack.pop();
                let result = self.json_access(&json_val, &key, true);
                stack.push(result);
            }

            // =================================================================
            // TIMESTAMP OPERATIONS
            // =================================================================
            Op::TimestampAddInterval => {
                let interval = stack.pop();
                let ts = stack.pop();
                let result = Self::timestamp_add_interval(&ts, &interval, false)?;
                stack.push(result);
            }

            Op::TimestampSubInterval => {
                let interval = stack.pop();
                let ts = stack.pop();
                let result = Self::timestamp_add_interval(&ts, &interval, true)?;
                stack.push(result);
            }

            Op::TimestampDiff => {
                let ts2 = stack.pop();
                let ts1 = stack.pop();
                let result = match (&*ts1, &*ts2) {
                    (Value::Timestamp(t1), Value::Timestamp(t2)) => {
                        let duration = t1.signed_duration_since(*t2);
                        Value::Text(SmartString::from_string(
                            self.format_duration_as_interval(duration),
                        ))
                    }
                    _ if ts1.is_null() || ts2.is_null() => Value::Null(DataType::Text),
                    _ => Value::Null(DataType::Text),
                };
                stack.push(result);
            }

            Op::TimestampAddDays => {
                let days = stack.pop();
                let ts = stack.pop();
                let result = match (&*ts, &*days) {
                    (Value::Timestamp(t), Value::Integer(d)) => {
                        Self::timestamp_add_days(*t, *d, false)?
                    }
                    _ if ts.is_null() || days.is_null() => Value::Null(DataType::Timestamp),
                    _ => Value::Null(DataType::Timestamp),
                };
                stack.push(result);
            }

            Op::TimestampSubDays => {
                let days = stack.pop();
                let ts = stack.pop();
                let result = match (&*ts, &*days) {
                    (Value::Timestamp(t), Value::Integer(d)) => {
                        Self::timestamp_add_days(*t, *d, true)?
                    }
                    _ if ts.is_null() || days.is_null() => Value::Null(DataType::Timestamp),
                    _ => Value::Null(DataType::Timestamp),
                };
                stack.push(result);
            }

            Op::ScaleInterval(step, count) => {
                let result = match Self::pop_interval_multiplier(stack, *count)? {
                    Some(k) => Value::Text(SmartString::from_string(Self::interval_text(
                        *step,
                        Self::scale_interval(*step, k)?,
                    ))),
                    None => Value::Null(DataType::Text),
                };
                stack.push(result);
            }

            Op::TimestampAddScaledInterval(step, subtract, count) => {
                let k = Self::pop_interval_multiplier(stack, *count)?;
                let ts = stack.pop();
                let result = match &*ts {
                    Value::Timestamp(t) => match k {
                        Some(k) => {
                            let interval = Self::scale_interval(*step, k)?;
                            Value::Timestamp(interval::shift(*t, interval, *subtract)?)
                        }
                        None => Value::Null(DataType::Timestamp),
                    },
                    _ if ts.is_null() => Value::Null(DataType::Timestamp),
                    _ => {
                        return Err(crate::core::Error::Type(format!(
                            "cannot add an INTERVAL to {:?}",
                            ts.data_type()
                        )))
                    }
                };
                stack.push(result);
            }

            Op::TruncateToDate => {
                let v = stack.pop();
                let result = match &*v {
                    Value::Timestamp(t) => {
                        use chrono::{Datelike, TimeZone, Utc};
                        let truncated = Utc
                            .with_ymd_and_hms(t.year(), t.month(), t.day(), 0, 0, 0)
                            .single()
                            .unwrap_or(*t);
                        Value::Timestamp(truncated)
                    }
                    Value::Text(s) => match crate::core::parse_timestamp(s) {
                        Ok(t) => {
                            use chrono::{Datelike, TimeZone, Utc};
                            let truncated = Utc
                                .with_ymd_and_hms(t.year(), t.month(), t.day(), 0, 0, 0)
                                .single()
                                .unwrap_or(t);
                            Value::Timestamp(truncated)
                        }
                        Err(_) => Value::Null(DataType::Timestamp),
                    },
                    Value::Integer(i) => {
                        use chrono::{Datelike, TimeZone, Utc};
                        match Utc.timestamp_opt(*i, 0) {
                            chrono::LocalResult::Single(t) => {
                                let truncated = Utc
                                    .with_ymd_and_hms(t.year(), t.month(), t.day(), 0, 0, 0)
                                    .single()
                                    .unwrap_or(t);
                                Value::Timestamp(truncated)
                            }
                            _ => Value::Null(DataType::Timestamp),
                        }
                    }
                    Value::Null(_) => Value::Null(DataType::Timestamp),
                    _ => Value::Null(DataType::Timestamp),
                };
                stack.push(result);
            }

            // =================================================================
            // SET AND RANGE OPERATIONS OVER STACK VALUES
            // =================================================================
            Op::InList(count) => {
                let start = stack.len().saturating_sub(*count as usize);
                let items = self.take_values(stack, start);
                let v = stack.pop();
                let result = in_list(&v, items.iter());
                self.args_buffer = items;
                stack.push(truth(result));
            }

            Op::QuantifiedSet(set) => {
                let v = stack.pop();
                stack.push(truth(set.answer(&v)));
            }

            Op::Between => {
                let high = stack.pop();
                let low = stack.pop();
                let val = stack.pop();

                // Fast path for integer/float comparisons
                let result = match (&*val, &*low, &*high) {
                    (Value::Integer(v), Value::Integer(lo), Value::Integer(hi)) => {
                        Value::Boolean(*v >= *lo && *v <= *hi)
                    }
                    (Value::Float(v), Value::Float(lo), Value::Float(hi)) => {
                        Value::Boolean(f64_ge(*v, *lo) && f64_le(*v, *hi))
                    }
                    (Value::Integer(v), Value::Integer(lo), Value::Float(hi)) => {
                        Value::Boolean(*v >= *lo && crate::core::value::i64_le_f64(*v, *hi))
                    }
                    (Value::Integer(v), Value::Float(lo), Value::Integer(hi)) => {
                        Value::Boolean(crate::core::value::i64_ge_f64(*v, *lo) && *v <= *hi)
                    }
                    (Value::Float(v), Value::Integer(lo), Value::Integer(hi)) => Value::Boolean(
                        crate::core::value::i64_le_f64(*lo, *v)
                            && crate::core::value::i64_ge_f64(*hi, *v),
                    ),
                    (Value::Float(v), Value::Integer(lo), Value::Float(hi)) => {
                        Value::Boolean(crate::core::value::i64_le_f64(*lo, *v) && f64_le(*v, *hi))
                    }
                    (Value::Float(v), Value::Float(lo), Value::Integer(hi)) => {
                        Value::Boolean(f64_ge(*v, *lo) && crate::core::value::i64_ge_f64(*hi, *v))
                    }
                    (Value::Integer(v), Value::Float(lo), Value::Float(hi)) => Value::Boolean(
                        crate::core::value::i64_ge_f64(*v, *lo)
                            && crate::core::value::i64_le_f64(*v, *hi),
                    ),
                    _ => truth(between_cell(&val, &low, &high)),
                };
                stack.push(result);
            }

            Op::NotBetween => {
                let high = stack.pop();
                let low = stack.pop();
                let val = stack.pop();

                // Fast path for integer/float comparisons
                let result = match (&*val, &*low, &*high) {
                    (Value::Integer(v), Value::Integer(lo), Value::Integer(hi)) => {
                        Value::Boolean(*v < *lo || *v > *hi)
                    }
                    (Value::Float(v), Value::Float(lo), Value::Float(hi)) => {
                        Value::Boolean(f64_lt(*v, *lo) || f64_gt(*v, *hi))
                    }
                    (Value::Integer(v), Value::Integer(lo), Value::Float(hi)) => {
                        Value::Boolean(*v < *lo || crate::core::value::i64_gt_f64(*v, *hi))
                    }
                    (Value::Integer(v), Value::Float(lo), Value::Integer(hi)) => {
                        Value::Boolean(crate::core::value::i64_lt_f64(*v, *lo) || *v > *hi)
                    }
                    (Value::Float(v), Value::Integer(lo), Value::Integer(hi)) => Value::Boolean(
                        crate::core::value::i64_gt_f64(*lo, *v)
                            || crate::core::value::i64_lt_f64(*hi, *v),
                    ),
                    (Value::Float(v), Value::Integer(lo), Value::Float(hi)) => {
                        Value::Boolean(crate::core::value::i64_gt_f64(*lo, *v) || f64_gt(*v, *hi))
                    }
                    (Value::Float(v), Value::Float(lo), Value::Integer(hi)) => {
                        Value::Boolean(f64_lt(*v, *lo) || crate::core::value::i64_lt_f64(*hi, *v))
                    }
                    (Value::Integer(v), Value::Float(lo), Value::Float(hi)) => Value::Boolean(
                        crate::core::value::i64_lt_f64(*v, *lo)
                            || crate::core::value::i64_gt_f64(*v, *hi),
                    ),
                    _ => truth(between_cell(&val, &low, &high).map(|inside| !inside)),
                };
                stack.push(result);
            }

            Op::InTupleSet {
                tuple_size,
                values,
                negated,
            } => {
                let start = stack.len().saturating_sub(*tuple_size as usize);
                let row = self.take_values(stack, start);
                let found = row_in(&row, values.iter().map(Vec::as_slice));
                self.args_buffer = row;
                let result = if *negated { found.map(|f| !f) } else { found };
                stack.push(truth(result));
            }

            Op::InRows(width, count) => {
                let width = *width as usize;
                let start = stack
                    .len()
                    .saturating_sub(width.saturating_mul(*count as usize + 1));
                let values = self.take_values(stack, start);
                let (row, list) = values.split_at(width);
                let result = truth(row_in(row, list.chunks_exact(width)));
                self.args_buffer = values;
                stack.push(result);
            }

            // =================================================================
            // BOOLEAN CHECKS
            // =================================================================
            Op::IsTrue => {
                let result = matches!(&*stack.pop(), Value::Boolean(true));
                stack.push(Value::Boolean(result));
            }

            Op::IsNotTrue => {
                let result = !matches!(&*stack.pop(), Value::Boolean(true));
                stack.push(Value::Boolean(result));
            }

            Op::IsFalse => {
                let result = matches!(&*stack.pop(), Value::Boolean(false));
                stack.push(Value::Boolean(result));
            }

            Op::IsNotFalse => {
                let result = !matches!(&*stack.pop(), Value::Boolean(false));
                stack.push(Value::Boolean(result));
            }

            Op::ReadAsColumn(column_type) => {
                let v = stack.pop().into_owned();
                stack.push(crate::executor::pushdown::read_as_column(*column_type, v));
            }

            Op::ReadAsVector => {
                let v = stack.pop().into_owned();
                stack.push(crate::functions::scalar::vector::read_as_vector(v));
            }

            Op::Dup => stack.dup(),

            Op::Swap => {
                let len = stack.len();
                if len >= 2 {
                    stack.swap(len - 1, len - 2);
                }
            }

            // =================================================================
            // SUBQUERY OPERATIONS
            // =================================================================
            Op::ExecScalarSubquery(idx) => {
                let executor = ctx.subquery_executor.ok_or_else(no_subquery_executor)?;
                stack.push(executor.execute_scalar(*idx, ctx)?);
            }

            Op::ExecExists(idx) => {
                let executor = ctx.subquery_executor.ok_or_else(no_subquery_executor)?;
                let exists = executor.execute_exists(*idx, ctx)?;
                stack.push(Value::Boolean(exists));
            }

            // The block decides: a NULL value over no rows is FALSE, not UNKNOWN
            Op::ExecInSubquery(idx) => {
                let executor = ctx.subquery_executor.ok_or_else(no_subquery_executor)?;
                let v = stack.pop();
                let answer = match executor.kept_members(*idx) {
                    Some(kept) => kept.answer(&v)?,
                    None => executor.execute_in(*idx, &v, ctx)?,
                };
                stack.push(truth(answer));
            }

            // As IN: a NULL value over no rows is FALSE for ANY and TRUE for ALL
            Op::ExecQuantified(idx, compare_op, all) => {
                let executor = ctx.subquery_executor.ok_or_else(no_subquery_executor)?;
                let v = stack.pop();
                let answer = match executor.kept_members(*idx) {
                    Some(kept) => kept.answer(&v)?,
                    None => executor.execute_quantified(*idx, &v, *compare_op, *all, ctx)?,
                };
                stack.push(truth(answer));
            }

            Op::ExecRowMembers(idx, width) => self.row_members(stack, *idx, *width, ctx)?,

            // =================================================================
            // VECTOR DISTANCE OPERATIONS
            // =================================================================
            Op::VectorDistanceL2 => {
                let b = stack.pop();
                let a = stack.pop();
                let mut buf_a = Vec::new();
                let mut buf_b = Vec::new();
                match (
                    extract_vector_bytes(&a, &mut buf_a),
                    extract_vector_bytes(&b, &mut buf_b),
                ) {
                    (Some(ba), Some(bb)) if ba.len() == bb.len() => {
                        stack.push(Value::Float(
                            crate::functions::scalar::vector::l2_distance_bytes(ba, bb),
                        ));
                    }
                    _ => {
                        stack.push(Value::null_unknown());
                    }
                }
            }

            Op::VectorDistanceCosine => {
                let b = stack.pop();
                let a = stack.pop();
                let mut buf_a = Vec::new();
                let mut buf_b = Vec::new();
                match (
                    extract_vector_bytes(&a, &mut buf_a),
                    extract_vector_bytes(&b, &mut buf_b),
                ) {
                    (Some(ba), Some(bb)) if ba.len() == bb.len() => {
                        stack.push(Value::Float(
                            crate::functions::scalar::vector::cosine_distance_bytes(ba, bb),
                        ));
                    }
                    _ => {
                        stack.push(Value::null_unknown());
                    }
                }
            }

            Op::VectorDistanceIP => {
                let b = stack.pop();
                let a = stack.pop();
                let mut buf_a = Vec::new();
                let mut buf_b = Vec::new();
                match (
                    extract_vector_bytes(&a, &mut buf_a),
                    extract_vector_bytes(&b, &mut buf_b),
                ) {
                    (Some(ba), Some(bb)) if ba.len() == bb.len() => {
                        stack.push(Value::Float(
                            crate::functions::scalar::vector::ip_distance_bytes(ba, bb),
                        ));
                    }
                    _ => {
                        stack.push(Value::null_unknown());
                    }
                }
            }

            // The op loop answers every other op itself
            _ => {}
        }
        Ok(())
    }

    /// Execute and return boolean result (for WHERE clauses)
    ///
    /// This method is optimized for common filter patterns, avoiding
    /// the full VM loop overhead for simple comparisons.
    #[inline]
    pub fn execute_bool(&mut self, program: &Program, ctx: &ExecuteContext) -> bool {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::program_ran(program);
        if Self::runs_chain(program, ctx) {
            return Self::chain(program.ops(), ctx) == Some(true);
        }

        // General path: full VM execution
        match self.execute(program, ctx) {
            Ok(Value::Boolean(b)) => b,
            Ok(Value::Integer(i)) => i != 0,
            Ok(_) => false,
            Err(e) => self.stash_bool_error(e),
        }
    }

    /// Keeps the first error `execute_bool` read as false, for
    /// `take_bool_error`; out of line so the boolean path stays lean
    #[cold]
    #[inline(never)]
    fn stash_bool_error(&mut self, e: crate::core::Error) -> bool {
        self.bool_error.get_or_insert(e);
        false
    }

    /// The error `execute_bool` read as false since the last call, if any
    #[inline]
    pub fn take_bool_error(&mut self) -> Option<crate::core::Error> {
        self.bool_error.take()
    }

    /// Whether `execute_bool` has read an error as false since the last
    /// `take_bool_error`; cheaper than taking it
    #[inline]
    pub fn has_bool_error(&self) -> bool {
        self.bool_error.is_some()
    }

    /// Like execute_bool but returns errors instead of swallowing them.
    ///
    /// Used by RowFilter::matches_checked to propagate VM errors (e.g. invalid
    /// REGEXP patterns) through the query result iterator.
    #[inline]
    pub fn execute_bool_checked(
        &mut self,
        program: &Program,
        ctx: &ExecuteContext,
    ) -> crate::core::Result<bool> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::program_ran(program);
        if Self::runs_chain(program, ctx) {
            return Ok(Self::chain(program.ops(), ctx) == Some(true));
        }

        // General path: full VM execution
        self.execute(program, ctx).map(|v| Self::passes(&v))
    }

    /// Whether a filter's value keeps its row: TRUE or a non-zero integer
    pub(crate) fn passes(v: &Value) -> bool {
        match v {
            Value::Boolean(b) => *b,
            Value::Integer(i) => *i != 0,
            _ => false,
        }
    }

    /// The ops `eval_single_op_tribool` answers as the full VM does. A bare
    /// column load, or a constant that is not a truth value, is a value to be
    /// read as a condition, so it goes to the full VM.
    #[inline]
    fn is_single_op_predicate(op: &Op) -> bool {
        matches!(
            op,
            Op::GtColumnConst(..)
                | Op::LtColumnConst(..)
                | Op::GeColumnConst(..)
                | Op::LeColumnConst(..)
                | Op::EqColumnConst(..)
                | Op::NeColumnConst(..)
                | Op::IsNullColumn(..)
                | Op::IsNotNullColumn(..)
                | Op::BetweenColumnConst(..)
                | Op::InSetColumn(..)
                | Op::LikeColumn(..)
                | Op::LoadConst(Value::Boolean(_) | Value::Integer(_) | Value::Null(_))
                | Op::CompareColumnParam(..)
                | Op::BetweenColumnParams(..)
                | Op::InColumnParams(..)
        )
    }

    /// How many parameter slots a predicate chain of `ops` reads: one past
    /// the highest
    pub(super) fn chain_params(ops: &[Op]) -> usize {
        let past = |slot: &u32| *slot as usize + 1;
        ops.iter()
            .map(|op| match op {
                Op::CompareColumnParam(_, slot, ..) => past(slot),
                Op::BetweenColumnParams(_, low, high, _) => past(low).max(past(high)),
                Op::InColumnParams(_, slots, _) => slots.iter().map(past).max().unwrap_or(0),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }

    /// Whether `program` answers as a predicate chain under `ctx`: one
    /// whose parameters all hold a value, as a chain raises no error
    #[inline]
    fn runs_chain(program: &Program, ctx: &ExecuteContext) -> bool {
        program.is_predicate_chain()
            && (program.chain_params() == 0
                || ctx.failed.is_empty() && ctx.params.len() >= program.chain_params())
    }

    /// Whether `ops` are single-op predicates joined left to right by AND
    /// and OR, each connector jumping just past its finalize
    pub(super) fn is_predicate_chain(ops: &[Op]) -> bool {
        let [first, steps @ .., Op::Return] = ops else {
            return false;
        };
        let (steps, rest) = steps.as_chunks::<3>();
        Self::is_single_op_predicate(first)
            && rest.is_empty()
            && steps.iter().enumerate().all(|(k, step)| {
                matches!(step,
                    [Op::And(to), p, Op::AndFinalize] | [Op::Or(to), p, Op::OrFinalize]
                        if usize::from(*to) == 3 * k + 4 && Self::is_single_op_predicate(p))
            })
    }

    /// A predicate chain's value as a truth value, None for UNKNOWN; a term
    /// the value so far settles is not evaluated
    #[inline]
    fn chain(ops: &[Op], ctx: &ExecuteContext) -> Option<bool> {
        let (first, steps) = ops.split_first()?;
        let mut acc = Self::eval_single_op_tribool(first, ctx);
        for [connector, predicate, _] in steps.as_chunks::<3>().0 {
            acc = match (connector, acc) {
                (Op::And(_), Some(false)) | (Op::Or(_), Some(true)) => acc,
                (Op::And(_), _) => match Self::eval_single_op_tribool(predicate, ctx) {
                    Some(true) => acc,
                    other => other,
                },
                _ => match Self::eval_single_op_tribool(predicate, ctx) {
                    Some(false) => acc,
                    other => other,
                },
            };
        }
        acc
    }

    /// Evaluate a single comparison op and return Option<bool> (tribool)
    /// None = NULL, Some(true) = true, Some(false) = false
    #[inline(always)]
    fn eval_single_op_tribool(op: &Op, ctx: &ExecuteContext) -> Option<bool> {
        match op {
            Op::GtColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v > *threshold),
                Some(Value::Float(v)) => Some(crate::core::value::i64_lt_f64(*threshold, *v)),
                cell => Self::fused_cell(op, cell),
            },
            Op::GtColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(f64_gt(*v, *threshold)),
                Some(Value::Integer(v)) => Some(crate::core::value::i64_gt_f64(*v, *threshold)),
                cell => Self::fused_cell(op, cell),
            },
            Op::LtColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v < *threshold),
                Some(Value::Float(v)) => Some(crate::core::value::i64_gt_f64(*threshold, *v)),
                cell => Self::fused_cell(op, cell),
            },
            Op::LtColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(f64_lt(*v, *threshold)),
                Some(Value::Integer(v)) => Some(crate::core::value::i64_lt_f64(*v, *threshold)),
                cell => Self::fused_cell(op, cell),
            },
            Op::GeColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v >= *threshold),
                Some(Value::Float(v)) => Some(crate::core::value::i64_le_f64(*threshold, *v)),
                cell => Self::fused_cell(op, cell),
            },
            Op::GeColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(f64_ge(*v, *threshold)),
                Some(Value::Integer(v)) => Some(crate::core::value::i64_ge_f64(*v, *threshold)),
                cell => Self::fused_cell(op, cell),
            },
            Op::LeColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v <= *threshold),
                Some(Value::Float(v)) => Some(crate::core::value::i64_ge_f64(*threshold, *v)),
                cell => Self::fused_cell(op, cell),
            },
            Op::LeColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(f64_le(*v, *threshold)),
                Some(Value::Integer(v)) => Some(crate::core::value::i64_le_f64(*v, *threshold)),
                cell => Self::fused_cell(op, cell),
            },
            Op::EqColumnConst(idx, Value::Integer(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v == *val),
                Some(Value::Float(v)) => Some(crate::core::value::i64_eq_f64(*val, *v)),
                cell => Self::fused_cell(op, cell),
            },
            Op::EqColumnConst(idx, Value::Float(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(f64_eq(*v, *val)),
                Some(Value::Integer(v)) => Some(crate::core::value::i64_eq_f64(*v, *val)),
                cell => Self::fused_cell(op, cell),
            },
            Op::NeColumnConst(idx, Value::Integer(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v != *val),
                Some(Value::Float(v)) => Some(crate::core::value::i64_ne_f64(*val, *v)),
                cell => Self::fused_cell(op, cell),
            },
            Op::NeColumnConst(idx, Value::Float(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(f64_ne(*v, *val)),
                Some(Value::Integer(v)) => Some(crate::core::value::i64_ne_f64(*v, *val)),
                cell => Self::fused_cell(op, cell),
            },
            Op::IsNullColumn(idx) => Some(ctx.row.get(*idx as usize).is_none_or(|v| v.is_null())),
            Op::IsNotNullColumn(idx) => {
                Some(ctx.row.get(*idx as usize).is_some_and(|v| !v.is_null()))
            }
            Op::BetweenColumnConst(idx, Value::Integer(lo), Value::Integer(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Integer(v)) => Some(*v >= *lo && *v <= *hi),
                    Some(Value::Float(v)) => Some(
                        crate::core::value::i64_le_f64(*lo, *v)
                            && crate::core::value::i64_ge_f64(*hi, *v),
                    ),
                    cell => Self::fused_cell(op, cell),
                }
            }
            Op::BetweenColumnConst(idx, Value::Float(lo), Value::Float(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => Some(f64_ge(*v, *lo) && f64_le(*v, *hi)),
                    Some(Value::Integer(v)) => Some(
                        crate::core::value::i64_ge_f64(*v, *lo)
                            && crate::core::value::i64_le_f64(*v, *hi),
                    ),
                    cell => Self::fused_cell(op, cell),
                }
            }
            Op::BetweenColumnConst(idx, Value::Integer(lo), Value::Float(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => {
                        Some(crate::core::value::i64_le_f64(*lo, *v) && f64_le(*v, *hi))
                    }
                    Some(Value::Integer(v)) => {
                        Some(*v >= *lo && crate::core::value::i64_le_f64(*v, *hi))
                    }
                    cell => Self::fused_cell(op, cell),
                }
            }
            Op::BetweenColumnConst(idx, Value::Float(lo), Value::Integer(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => {
                        Some(f64_ge(*v, *lo) && crate::core::value::i64_ge_f64(*hi, *v))
                    }
                    Some(Value::Integer(v)) => {
                        Some(crate::core::value::i64_ge_f64(*v, *lo) && *v <= *hi)
                    }
                    cell => Self::fused_cell(op, cell),
                }
            }
            Op::BetweenColumnConst(idx, _, _) => Self::fused_cell(op, ctx.row.get(*idx as usize)),
            Op::InSetColumn(idx, set) => {
                set.contains(ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE))
            }
            // Handle boolean constants (from processed subqueries like NOT EXISTS)
            Op::LoadConst(Value::Boolean(b)) => Some(*b),
            Op::LoadConst(Value::Integer(i)) => Some(*i != 0),
            Op::LoadConst(Value::Null(_)) => None,
            // Handle text comparisons (e.g., country = 'USA')
            Op::EqColumnConst(idx, Value::Text(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Text(v)) => Some(v == val),
                cell => Self::fused_cell(op, cell),
            },
            Op::NeColumnConst(idx, Value::Text(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Text(v)) => Some(v != val),
                cell => Self::fused_cell(op, cell),
            },
            // Handle LIKE pattern matching (e.g., fruit LIKE 'a%')
            Op::LikeColumn(idx, pattern) => ctx
                .row
                .get(*idx as usize)
                .and_then(|value| pattern.matches_value(value)),
            // Any other fused comparison, by the predicate comparison
            Op::LtColumnConst(idx, _)
            | Op::LeColumnConst(idx, _)
            | Op::GtColumnConst(idx, _)
            | Op::GeColumnConst(idx, _)
            | Op::EqColumnConst(idx, _)
            | Op::NeColumnConst(idx, _) => Self::fused_cell(op, ctx.row.get(*idx as usize)),
            // A chain runs only once every parameter it reads holds a value
            Op::CompareColumnParam(idx, slot, compare, read_as) => {
                let cell = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                let param = ctx.params.get(*slot as usize)?;
                match (cell, param) {
                    // An integer read as an integer, or as it is, needs no reading
                    (Value::Integer(v), Value::Integer(p))
                        if read_as.is_none_or(|t| t == DataType::Integer) =>
                    {
                        Some(compare.integers(*v, *p))
                    }
                    _ => compare.compare(cell, &read_param(param, *read_as)),
                }
            }
            Op::BetweenColumnParams(idx, low, high, read_as) => {
                let cell = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                let (low, high) = (
                    ctx.params.get(*low as usize)?,
                    ctx.params.get(*high as usize)?,
                );
                match (cell, low, high) {
                    (Value::Integer(v), Value::Integer(lo), Value::Integer(hi))
                        if read_as.is_none_or(|t| t == DataType::Integer) =>
                    {
                        Some(CompareOp::Ge.integers(*v, *lo) && CompareOp::Le.integers(*v, *hi))
                    }
                    _ => between_cell(
                        cell,
                        &read_param(low, *read_as),
                        &read_param(high, *read_as),
                    ),
                }
            }
            Op::InColumnParams(idx, slots, read_as) => {
                let cell = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                let params = ctx.params;
                if let Value::Integer(v) = cell {
                    if read_as.is_none_or(|t| t == DataType::Integer) {
                        let mut members = slots.iter().map(|slot| params.get(*slot as usize));
                        if let Some(found) = members.try_fold(false, |found, member| match member {
                            Some(Value::Integer(p)) => {
                                Some(found || CompareOp::Eq.integers(*v, *p))
                            }
                            _ => None,
                        }) {
                            return Some(found);
                        }
                    }
                }
                let items = slots.iter().filter_map(|slot| params.get(*slot as usize));
                let items: SmallVec<[Cow<'_, Value>; 8]> =
                    items.map(|v| read_param(v, *read_as)).collect();
                in_list(cell, items.iter().map(|v| &**v))
            }
            // For other comparisons, return None to fall back to full VM
            _ => None,
        }
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    #[inline]
    fn to_bool(v: &Value) -> bool {
        match v {
            Value::Boolean(b) => *b,
            Value::Integer(i) => *i != 0,
            Value::Null(_) => false,
            _ => true, // Non-null values are truthy
        }
    }

    #[inline]
    /// The right operand of an AND met with its left: FALSE wins over UNKNOWN
    pub(crate) fn and(a: &Value, b: &Value) -> Value {
        match (Self::to_tribool(a), Self::to_tribool(b)) {
            (Some(false), _) | (_, Some(false)) => Value::Boolean(false),
            (Some(true), Some(true)) => Value::Boolean(true),
            _ => Value::Null(DataType::Boolean),
        }
    }

    fn to_tribool(v: &Value) -> Option<bool> {
        match v {
            Value::Boolean(b) => Some(*b),
            Value::Integer(i) => Some(*i != 0),
            Value::Null(_) => None,
            _ => Some(true),
        }
    }

    /// The top `width` values tested against block `index`'s members,
    /// replaced by the answer
    #[inline(never)]
    fn row_members(
        &mut self,
        stack: &mut Stack<'_>,
        index: u16,
        width: u16,
        ctx: &ExecuteContext,
    ) -> Result<()> {
        let executor = ctx.subquery_executor.ok_or_else(no_subquery_executor)?;
        let start = stack.len().saturating_sub(width as usize);
        let mut row = std::mem::take(&mut self.args_buffer);
        row.clear();
        row.extend(stack.drain(start));
        let answer = match executor.kept_members(index) {
            Some(kept) => kept.answer_row(&row),
            None => executor.execute_row_members(index, &row, ctx),
        };
        self.args_buffer = row;
        stack.push(truth(answer?));
        Ok(())
    }

    #[inline]
    fn compare_values(&self, a: &Value, b: &Value, expected: std::cmp::Ordering) -> Value {
        // Two NULLs order equal so an index can hold them; asked whether one
        // is below or above the other, SQL answers that it does not know
        truth(sql_compare(a, b).map(|ord| ord == expected))
    }

    /// Answer `<=` or `>=`: `expected` is the strict half of the operator.
    #[inline]
    fn compare_values_or_equal(&self, a: &Value, b: &Value, expected: std::cmp::Ordering) -> Value {
        truth(sql_compare(a, b).map(|ord| ord == expected || ord == std::cmp::Ordering::Equal))
    }

    // =========================================================================
    // INLINE COMPARISON HELPERS
    // These avoid the overhead of partial_cmp → compare → compare_same_type
    // for the common case of integer/float column vs constant comparisons.
    // =========================================================================

    /// A fused column comparison for a cell the numeric fast arms do not
    /// take, by the same rules as the general interpreter
    #[cold]
    pub(crate) fn fused_cell(op: &Op, cell: Option<&Value>) -> Option<bool> {
        let cell = cell.unwrap_or(&NULL_VALUE);
        let value = match op {
            Op::GtColumnConst(_, Value::Integer(t)) => Self::gt_int(cell, *t),
            Op::GtColumnConst(_, Value::Float(t)) => Self::gt_float(cell, *t),
            Op::LtColumnConst(_, Value::Integer(t)) => Self::lt_int(cell, *t),
            Op::LtColumnConst(_, Value::Float(t)) => Self::lt_float(cell, *t),
            Op::GeColumnConst(_, Value::Integer(t)) => Self::ge_int(cell, *t),
            Op::GeColumnConst(_, Value::Float(t)) => Self::ge_float(cell, *t),
            Op::LeColumnConst(_, Value::Integer(t)) => Self::le_int(cell, *t),
            Op::LeColumnConst(_, Value::Float(t)) => Self::le_float(cell, *t),
            Op::EqColumnConst(_, Value::Integer(v)) => Self::eq_int(cell, *v),
            Op::EqColumnConst(_, Value::Float(v)) => Self::eq_float(cell, *v),
            Op::NeColumnConst(_, Value::Integer(v)) => Self::ne_int(cell, *v),
            Op::NeColumnConst(_, Value::Float(v)) => Self::ne_float(cell, *v),
            Op::GtColumnConst(_, v) => truth(sql_compare(cell, v).map(|o| o.is_gt())),
            Op::LtColumnConst(_, v) => truth(sql_compare(cell, v).map(|o| o.is_lt())),
            Op::GeColumnConst(_, v) => truth(sql_compare(cell, v).map(|o| o.is_ge())),
            Op::LeColumnConst(_, v) => truth(sql_compare(cell, v).map(|o| o.is_le())),
            Op::EqColumnConst(_, v) => return sql_eq(cell, v),
            Op::NeColumnConst(_, v) => return sql_eq(cell, v).map(|eq| !eq),
            Op::BetweenColumnConst(_, low, high) => truth(between_cell(cell, low, high)),
            _ => return None,
        };
        Self::to_tribool(&value)
    }

    /// A cell that is not a number or NULL against a numeric constant, by the
    /// predicate comparison
    #[inline]
    fn text_against(
        col_val: &Value,
        constant: Value,
        holds: fn(std::cmp::Ordering) -> bool,
    ) -> Value {
        truth(sql_compare(col_val, &constant).map(holds))
    }

    /// Inline greater-than comparison for integer constant
    #[inline(always)]
    fn gt_int(col_val: &Value, threshold: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v > threshold),
            Value::Float(v) => Value::Boolean(crate::core::value::i64_lt_f64(threshold, *v)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Self::text_against(col_val, Value::Integer(threshold), |o| o.is_gt()),
        }
    }

    /// Inline greater-than comparison for float constant
    #[inline(always)]
    fn gt_float(col_val: &Value, threshold: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(f64_gt(*v, threshold)),
            Value::Integer(v) => Value::Boolean(crate::core::value::i64_gt_f64(*v, threshold)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Self::text_against(col_val, Value::Float(threshold), |o| o.is_gt()),
        }
    }

    /// Inline less-than comparison for integer constant
    #[inline(always)]
    fn lt_int(col_val: &Value, threshold: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v < threshold),
            Value::Float(v) => Value::Boolean(crate::core::value::i64_gt_f64(threshold, *v)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Self::text_against(col_val, Value::Integer(threshold), |o| o.is_lt()),
        }
    }

    /// Inline less-than comparison for float constant
    #[inline(always)]
    fn lt_float(col_val: &Value, threshold: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(f64_lt(*v, threshold)),
            Value::Integer(v) => Value::Boolean(crate::core::value::i64_lt_f64(*v, threshold)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Self::text_against(col_val, Value::Float(threshold), |o| o.is_lt()),
        }
    }

    /// Inline greater-or-equal comparison for integer constant
    #[inline(always)]
    fn ge_int(col_val: &Value, threshold: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v >= threshold),
            Value::Float(v) => Value::Boolean(crate::core::value::i64_le_f64(threshold, *v)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Self::text_against(col_val, Value::Integer(threshold), |o| o.is_ge()),
        }
    }

    /// Inline greater-or-equal comparison for float constant
    #[inline(always)]
    fn ge_float(col_val: &Value, threshold: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(f64_ge(*v, threshold)),
            Value::Integer(v) => Value::Boolean(crate::core::value::i64_ge_f64(*v, threshold)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Self::text_against(col_val, Value::Float(threshold), |o| o.is_ge()),
        }
    }

    /// Inline less-or-equal comparison for integer constant
    #[inline(always)]
    fn le_int(col_val: &Value, threshold: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v <= threshold),
            Value::Float(v) => Value::Boolean(crate::core::value::i64_ge_f64(threshold, *v)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Self::text_against(col_val, Value::Integer(threshold), |o| o.is_le()),
        }
    }

    /// Inline less-or-equal comparison for float constant
    #[inline(always)]
    fn le_float(col_val: &Value, threshold: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(f64_le(*v, threshold)),
            Value::Integer(v) => Value::Boolean(crate::core::value::i64_le_f64(*v, threshold)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Self::text_against(col_val, Value::Float(threshold), |o| o.is_le()),
        }
    }

    /// Inline equality comparison for integer constant
    #[inline(always)]
    fn eq_int(col_val: &Value, val: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v == val),
            Value::Float(v) => Value::Boolean(crate::core::value::i64_eq_f64(val, *v)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => truth(sql_eq(col_val, &Value::Integer(val))),
        }
    }

    /// Inline equality comparison for float constant
    #[inline(always)]
    fn eq_float(col_val: &Value, val: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(f64_eq(*v, val)),
            Value::Integer(v) => Value::Boolean(crate::core::value::i64_eq_f64(*v, val)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => truth(sql_eq(col_val, &Value::Float(val))),
        }
    }

    /// Inline not-equal comparison for integer constant
    #[inline(always)]
    fn ne_int(col_val: &Value, val: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v != val),
            Value::Float(v) => Value::Boolean(crate::core::value::i64_ne_f64(val, *v)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => truth(sql_eq(col_val, &Value::Integer(val)).map(|eq| !eq)),
        }
    }

    /// Inline not-equal comparison for float constant
    #[inline(always)]
    fn ne_float(col_val: &Value, val: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(f64_ne(*v, val)),
            Value::Integer(v) => Value::Boolean(crate::core::value::i64_ne_f64(*v, val)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => truth(sql_eq(col_val, &Value::Float(val)).map(|eq| !eq)),
        }
    }

    /// BETWEEN with an integer lower bound and float upper bound, exact
    /// above 2^53 for both cell types
    #[inline(always)]
    fn between_mixed_int_lo(col_val: &Value, lo: i64, hi: f64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v >= lo && crate::core::value::i64_le_f64(*v, hi)),
            Value::Float(v) => {
                Value::Boolean(crate::core::value::i64_le_f64(lo, *v) && f64_le(*v, hi))
            }
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => truth(between_cell(
                col_val,
                &Value::Integer(lo),
                &Value::Float(hi),
            )),
        }
    }

    /// BETWEEN with a float lower bound and integer upper bound
    #[inline(always)]
    fn between_mixed_int_hi(col_val: &Value, lo: f64, hi: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(crate::core::value::i64_ge_f64(*v, lo) && *v <= hi),
            Value::Float(v) => {
                Value::Boolean(f64_ge(*v, lo) && crate::core::value::i64_ge_f64(hi, *v))
            }
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => truth(between_cell(
                col_val,
                &Value::Float(lo),
                &Value::Integer(hi),
            )),
        }
    }

    /// Inline BETWEEN check for integer bounds
    #[inline(always)]
    fn between_int(col_val: &Value, low: i64, high: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v >= low && *v <= high),
            Value::Float(v) => Value::Boolean(
                crate::core::value::i64_le_f64(low, *v) && crate::core::value::i64_ge_f64(high, *v),
            ),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => truth(between_cell(
                col_val,
                &Value::Integer(low),
                &Value::Integer(high),
            )),
        }
    }

    /// Inline BETWEEN check for float bounds
    #[inline(always)]
    fn between_float(col_val: &Value, low: f64, high: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(f64_ge(*v, low) && f64_le(*v, high)),
            Value::Integer(v) => Value::Boolean(
                crate::core::value::i64_ge_f64(*v, low) && crate::core::value::i64_le_f64(*v, high),
            ),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => truth(between_cell(
                col_val,
                &Value::Float(low),
                &Value::Float(high),
            )),
        }
    }

    #[inline]
    fn arithmetic_op<FF>(a: &Value, b: &Value, int_op: IntegerOp, float_op: FF) -> Result<Value>
    where
        FF: Fn(f64, f64) -> f64,
    {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => {
                Ok(integer_op(int_op, *x, *y)?.map_or(NULL_VALUE, Value::Integer))
            }
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(float_op(*x, *y))),
            (Value::Integer(x), Value::Float(y)) => Ok(Value::Float(float_op(*x as f64, *y))),
            (Value::Float(x), Value::Integer(y)) => Ok(Value::Float(float_op(*x, *y as f64))),
            _ if a.is_null() || b.is_null() => Ok(Value::Null(DataType::Float)),
            _ => Ok(Value::Null(DataType::Null)),
        }
    }

    /// `a + b`, with a timestamp plus days or an interval
    #[inline]
    fn add_op(a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Timestamp(t), Value::Integer(days))
            | (Value::Integer(days), Value::Timestamp(t)) => {
                Self::timestamp_add_days(*t, *days, false)
            }
            (Value::Timestamp(_), Value::Text(_)) => Self::timestamp_add_interval(a, b, false),
            _ => Self::arithmetic_op(a, b, IntegerOp::Add, |x, y| x + y),
        }
    }

    /// `a - b`, with a timestamp minus days or an interval, and the interval
    /// text between two timestamps
    #[inline]
    fn sub_op(&self, a: &Value, b: &Value) -> Result<Value> {
        match (a, b) {
            (Value::Timestamp(t1), Value::Timestamp(t2)) => {
                let duration = t1.signed_duration_since(*t2);
                Ok(Value::Text(SmartString::from_string(
                    self.format_duration_as_interval(duration),
                )))
            }
            (Value::Timestamp(t), Value::Integer(days)) => {
                Self::timestamp_add_days(*t, *days, true)
            }
            (Value::Timestamp(_), Value::Text(_)) => Self::timestamp_add_interval(a, b, true),
            _ => Self::arithmetic_op(a, b, IntegerOp::Sub, |x, y| x - y),
        }
    }

    #[inline]
    fn mul_op(a: &Value, b: &Value) -> Result<Value> {
        Self::arithmetic_op(a, b, IntegerOp::Mul, |x, y| x * y)
    }

    #[inline]
    fn div_op(a: &Value, b: &Value) -> Result<Value> {
        Ok(match (a, b) {
            // Division by zero returns NULL
            (Value::Integer(x), Value::Integer(y)) => {
                integer_op(IntegerOp::Div, *x, *y)?.map_or(NULL_VALUE, Value::Integer)
            }
            (Value::Float(x), Value::Float(y)) if *y != 0.0 => Value::Float(x / y),
            (Value::Integer(x), Value::Float(y)) if *y != 0.0 => Value::Float(*x as f64 / y),
            (Value::Float(x), Value::Integer(y)) if *y != 0 => Value::Float(x / *y as f64),
            _ if a.is_null() || b.is_null() => Value::Null(DataType::Float),
            _ => Value::Null(DataType::Null),
        })
    }

    #[inline]
    fn mod_op(a: &Value, b: &Value) -> Result<Value> {
        Ok(match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => {
                integer_op(IntegerOp::Mod, *x, *y)?.map_or(NULL_VALUE, Value::Integer)
            }
            (Value::Float(x), Value::Float(y)) if *y != 0.0 => Value::Float(x % y),
            (Value::Integer(x), Value::Float(y)) if *y != 0.0 => Value::Float(*x as f64 % y),
            (Value::Float(x), Value::Integer(y)) if *y != 0 => Value::Float(x % *y as f64),
            _ if a.is_null() || b.is_null() => Value::Null(DataType::Float),
            _ => Value::Null(DataType::Null),
        })
    }

    /// JSON access helper
    /// If as_text is true, returns TEXT; otherwise returns JSON
    fn json_access(&self, json_val: &Value, key: &Value, as_text: bool) -> Value {
        use serde_json;

        // Get the JSON string
        let json_str = match json_val {
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                std::str::from_utf8(&data[1..]).unwrap_or("")
            }
            Value::Text(s) => s.as_ref(),
            Value::Null(_) => {
                return Value::Null(if as_text {
                    DataType::Text
                } else {
                    DataType::Json
                })
            }
            _ => {
                return Value::Null(if as_text {
                    DataType::Text
                } else {
                    DataType::Json
                })
            }
        };

        // Parse the JSON
        let parsed: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(_) => {
                return Value::Null(if as_text {
                    DataType::Text
                } else {
                    DataType::Json
                })
            }
        };

        // Access by key or index
        let result = match key {
            Value::Text(k) => parsed.get(k.as_str()),
            Value::Integer(i) => {
                if *i >= 0 {
                    parsed.get(*i as usize)
                } else {
                    None
                }
            }
            _ => None,
        };

        match result {
            Some(v) => {
                if as_text {
                    // ->> returns text
                    match v {
                        serde_json::Value::String(s) => Value::Text(SmartString::new(s)),
                        serde_json::Value::Null => Value::Null(DataType::Text),
                        other => Value::Text(SmartString::from_string(other.to_string())),
                    }
                } else {
                    // -> returns JSON
                    Value::json(v.to_string())
                }
            }
            None => Value::Null(if as_text {
                DataType::Text
            } else {
                DataType::Json
            }),
        }
    }

    /// A timestamp moved by interval text, backwards when `subtract`; NULL
    /// for anything else and for text that is not an interval
    fn timestamp_add_interval(ts: &Value, interval: &Value, subtract: bool) -> Result<Value> {
        let (Value::Timestamp(t), Value::Text(text)) = (ts, interval) else {
            return Ok(Value::Null(DataType::Timestamp));
        };
        Ok(match interval::parse_interval(text)? {
            Some(interval) => Value::Timestamp(interval::shift(*t, interval, subtract)?),
            None => Value::Null(DataType::Timestamp),
        })
    }

    /// A timestamp moved by a number of days, backwards when `subtract`
    fn timestamp_add_days(
        t: chrono::DateTime<chrono::Utc>,
        days: i64,
        subtract: bool,
    ) -> Result<Value> {
        let days = Interval::of(days, IntervalUnit::Day)?;
        Ok(Value::Timestamp(interval::shift(t, days, subtract)?))
    }

    /// An INTERVAL multiplier: Ok for a whole number, Err for a fractional
    /// one, None for NULL; any other type is an error
    fn interval_multiplier(n: &Value) -> Result<Option<std::result::Result<i64, f64>>> {
        match n {
            v if v.is_null() => Ok(None),
            Value::Integer(k) => Ok(Some(Ok(*k))),
            Value::Float(f) => Ok(Some(Err(*f))),
            v => Err(crate::core::Error::Type(format!(
                "cannot multiply an INTERVAL by {:?}",
                v.data_type()
            ))),
        }
    }

    /// Pops `count` INTERVAL multipliers and returns their product, taken in
    /// the order they were pushed; None when one is NULL. Every one is
    /// checked, NULL or not
    fn pop_interval_multiplier(
        stack: &mut Stack<'_>,
        count: u8,
    ) -> Result<Option<std::result::Result<i64, f64>>> {
        let as_f64 = |k: std::result::Result<i64, f64>| k.map_or_else(|f| f, |i| i as f64);
        let start = stack.len().saturating_sub(usize::from(count));
        let mut product = Some(Ok(1i64));
        for n in stack.from(start) {
            product = match (product, Self::interval_multiplier(n)?) {
                (None, _) | (_, None) => None,
                (Some(Ok(a)), Some(Ok(b))) => Some(Ok(a.checked_mul(b).ok_or_else(|| {
                    crate::core::Error::Type(format!(
                        "INTERVAL multiplier {a} times {b} is out of range"
                    ))
                })?)),
                (Some(a), Some(b)) => Some(Err(as_f64(a) * as_f64(b))),
            };
        }
        stack.truncate(start);
        Ok(product)
    }

    /// `step` times `k`. A fixed unit is counted in microseconds, rounded;
    /// a calendar unit takes whole multiples only
    fn scale_interval(step: IntervalStep, k: std::result::Result<i64, f64>) -> Result<Interval> {
        let n = || k.map_or_else(|f| f.to_string(), |i| i.to_string());
        let out_of_range = || {
            crate::core::Error::Type(format!(
                "INTERVAL '{} {}' times {} is out of range",
                step.quantity,
                step.unit.name(),
                n()
            ))
        };
        if let Some(per) = step.unit.months() {
            let k = match k {
                Ok(k) => k,
                Err(f) if f.fract() == 0.0 && f.abs() < 9.2e18 => f as i64,
                Err(_) => {
                    return Err(crate::core::Error::Type(format!(
                        "a {} INTERVAL takes only a whole multiple, not {}",
                        step.unit.name(),
                        n()
                    )))
                }
            };
            let months = step
                .quantity
                .checked_mul(per)
                .and_then(|m| m.checked_mul(k));
            return months.map(Interval::Months).ok_or_else(out_of_range);
        }
        let unit = step
            .quantity
            .checked_mul(step.unit.micros())
            .ok_or_else(out_of_range)?;
        let micros = match k {
            Ok(k) => unit.checked_mul(k).ok_or_else(out_of_range)?,
            Err(f) => {
                let x = (unit as f64 * f).round();
                if !x.is_finite() || x.abs() >= 9.2e18 {
                    return Err(out_of_range());
                }
                x as i64
            }
        };
        Ok(Interval::Duration(chrono::Duration::microseconds(micros)))
    }

    /// `interval` as text `parse_interval` reads back: in `step`'s unit when
    /// it is a whole count of it, else in the largest exact sub-day unit
    fn interval_text(step: IntervalStep, interval: Interval) -> String {
        match interval {
            Interval::Months(m) if step.unit == IntervalUnit::Year && m % 12 == 0 => {
                format!("{} year", m / 12)
            }
            Interval::Months(m) => format!("{m} month"),
            Interval::Duration(d) => {
                let micros = d.num_microseconds().unwrap_or(0);
                let unit = [step.unit, IntervalUnit::Second, IntervalUnit::Millisecond]
                    .into_iter()
                    .find(|u| micros % u.micros() == 0)
                    .unwrap_or(IntervalUnit::Microsecond);
                format!("{} {}", micros / unit.micros(), unit.name())
            }
        }
    }

    /// Format chrono Duration as interval string
    fn format_duration_as_interval(&self, duration: chrono::TimeDelta) -> String {
        let total_seconds = duration.num_seconds();
        let abs_seconds = total_seconds.abs();

        let days = abs_seconds / 86400;
        let hours = (abs_seconds % 86400) / 3600;
        let minutes = (abs_seconds % 3600) / 60;
        let seconds = abs_seconds % 60;

        let sign = if total_seconds < 0 { "-" } else { "" };

        if days > 0 {
            format!(
                "{}{} days {:02}:{:02}:{:02}",
                sign, days, hours, minutes, seconds
            )
        } else {
            format!("{}{:02}:{:02}:{:02}", sign, hours, minutes, seconds)
        }
    }
}

impl Default for ExprVM {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract raw LE f32 bytes from a vector Value, zero-copy for Extension.
/// For Text values, parses and writes into `buf` as fallback.
#[inline]
fn extract_vector_bytes<'a>(v: &'a Value, buf: &'a mut Vec<u8>) -> Option<&'a [u8]> {
    match v {
        Value::Extension(data) if data.first() == Some(&(DataType::Vector as u8)) => {
            Some(&data[1..])
        }
        Value::Text(s) => {
            let floats = crate::core::value::parse_vector_str(s.as_ref())?;
            buf.clear();
            buf.reserve(floats.len() * 4);
            for f in &floats {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            Some(buf.as_slice())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::CompactArc;
    use crate::core::predicate::PredicateSet;
    use crate::core::ValueSet;
    use crate::Row;
    use std::sync::Arc;

    fn pset(set: ValueSet, has_null: bool) -> CompactArc<PredicateSet> {
        CompactArc::new(PredicateSet::new(CompactArc::new(set), has_null))
    }

    #[test]
    fn test_simple_comparison() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(5)),
            Op::Gt,
            Op::Return,
        ]);

        // Test with row where column 0 > 5
        let row = Row::from_values(vec![Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Test with row where column 0 <= 5
        let row = Row::from_values(vec![Value::Integer(3)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_and_short_circuit() {
        let mut vm = ExprVM::new();
        // WHERE col0 > 5 AND col1 < 10
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(5)),
            Op::Gt,
            Op::And(8), // Jump to return false if first condition is false
            Op::LoadColumn(1),
            Op::LoadConst(Value::Integer(10)),
            Op::Lt,
            Op::AndFinalize,
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Integer(10), Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(3), Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_in_set() {
        let mut vm = ExprVM::new();
        let set: ValueSet = [Value::Integer(1), Value::Integer(2), Value::Integer(3)]
            .into_iter()
            .collect();

        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::InSet(pset(set, false)),
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Integer(2)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    // =========================================================================
    // ExecuteContext tests
    // =========================================================================

    #[test]
    fn test_context_new() {
        let row = Row::from_values(vec![Value::Integer(1), Value::Text("test".into())]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(ctx.row.len(), 2);
        assert!(ctx.row2.is_none());
        assert!(ctx.outer_row.is_none());
        assert!(ctx.params.is_empty());
    }

    #[test]
    fn test_context_for_join() {
        let row1 = Row::from_values(vec![Value::Integer(1)]);
        let row2 = Row::from_values(vec![Value::Integer(2)]);
        let ctx = ExecuteContext::for_join(&row1, &row2);
        assert_eq!(ctx.row.len(), 1);
        assert!(ctx.row2.is_some());
        assert_eq!(*ctx.row2.unwrap().get(0).unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_context_with_params() {
        let row = Row::from_values(vec![Value::Integer(1)]);
        let params = vec![Value::Text("param1".into())];
        let ctx = ExecuteContext::new(&row).with_params(&params);
        assert_eq!(ctx.params.len(), 1);
    }

    #[test]
    fn test_context_with_transaction_id() {
        let row = Row::from_values(vec![Value::Integer(1)]);
        let ctx = ExecuteContext::new(&row).with_transaction_id(Some(12345));
        assert_eq!(ctx.transaction_id, Some(12345));
    }

    #[test]
    fn test_context_with_outer_row() {
        let row = Row::from_values(vec![Value::Integer(1)]);
        let mut outer: FxHashMap<CompactArc<str>, Value> = FxHashMap::default();
        outer.insert(CompactArc::from("outer_col"), Value::Integer(42));
        let ctx = ExecuteContext::new(&row).with_outer_row(&outer);
        assert!(ctx.outer_row.is_some());
    }

    // =========================================================================
    // ExprVM creation tests
    // =========================================================================

    #[test]
    fn test_vm_new() {
        let vm = ExprVM::new();
        assert_eq!(vm.args_buffer.len(), 0);
    }

    #[test]
    fn test_vm_default() {
        let vm = ExprVM::default();
        assert_eq!(vm.args_buffer.len(), 0);
    }

    // =========================================================================
    // Load operations tests
    // =========================================================================

    #[test]
    fn test_load_column() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn(1), Op::Return]);
        let row = Row::from_values(vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
        ]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(20));
    }

    #[test]
    fn test_load_column_out_of_bounds() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn(10), Op::Return]);
        let row = Row::from_values(vec![Value::Integer(1)]);
        let ctx = ExecuteContext::new(&row);
        // Out of bounds returns null
        assert!(vm.execute(&program, &ctx).unwrap().is_null());
    }

    #[test]
    fn test_load_column2() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn2(0), Op::Return]);
        let row1 = Row::from_values(vec![Value::Integer(1)]);
        let row2 = Row::from_values(vec![Value::Integer(2)]);
        let ctx = ExecuteContext::for_join(&row1, &row2);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_load_const() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadConst(Value::Float(1.23)), Op::Return]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Float(1.23));
    }

    #[test]
    fn test_load_param() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadParam(0), Op::Return]);
        let row = Row::new();
        let params = vec![Value::Text("hello".into())];
        let ctx = ExecuteContext::new(&row).with_params(&params);
        assert_eq!(
            vm.execute(&program, &ctx).unwrap(),
            Value::Text("hello".into())
        );
    }

    #[test]
    fn test_load_param_without_a_value_is_an_error() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadParam(1), Op::Return]);
        let row = Row::new();
        let params = [Value::Integer(1)];
        let ctx = ExecuteContext::new(&row).with_params(&params);
        assert!(vm.execute(&program, &ctx).is_err());
    }

    #[test]
    fn test_load_null() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadNull(DataType::Integer), Op::Return]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert!(vm.execute(&program, &ctx).unwrap().is_null());
    }

    #[test]
    fn test_load_outer_column() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadOuterColumn(CompactArc::from("outer_val")),
            Op::Return,
        ]);
        let row = Row::new();
        let mut outer: FxHashMap<CompactArc<str>, Value> = FxHashMap::default();
        outer.insert(CompactArc::from("outer_val"), Value::Integer(100));
        let ctx = ExecuteContext::new(&row).with_outer_row(&outer);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(100));
    }

    // =========================================================================
    // Comparison operations tests
    // =========================================================================

    #[test]
    fn test_eq() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(5)),
            Op::LoadConst(Value::Integer(5)),
            Op::Eq,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_ne() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(5)),
            Op::LoadConst(Value::Integer(10)),
            Op::Ne,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_lt() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(3)),
            Op::LoadConst(Value::Integer(5)),
            Op::Lt,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_le() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(5)),
            Op::LoadConst(Value::Integer(5)),
            Op::Le,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_ge() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(5)),
            Op::LoadConst(Value::Integer(5)),
            Op::Ge,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    // =========================================================================
    // Null checks tests
    // =========================================================================

    #[test]
    fn test_is_null() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn(0), Op::IsNull, Op::Return]);

        // Test with null
        let row = Row::from_values(vec![Value::Null(DataType::Integer)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Test with non-null
        let row = Row::from_values(vec![Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_is_not_null() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn(0), Op::IsNotNull, Op::Return]);

        let row = Row::from_values(vec![Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Null(DataType::Integer)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_is_distinct_from() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::LoadColumn(1),
            Op::IsDistinctFrom,
            Op::Return,
        ]);

        // Two nulls are NOT distinct
        let row = Row::from_values(vec![
            Value::Null(DataType::Integer),
            Value::Null(DataType::Integer),
        ]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));

        // Null and non-null ARE distinct
        let row = Row::from_values(vec![Value::Null(DataType::Integer), Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Same values are NOT distinct
        let row = Row::from_values(vec![Value::Integer(5), Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));

        // Different values ARE distinct
        let row = Row::from_values(vec![Value::Integer(5), Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_is_not_distinct_from() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::LoadColumn(1),
            Op::IsNotDistinctFrom,
            Op::Return,
        ]);

        // Two nulls ARE "not distinct"
        let row = Row::from_values(vec![
            Value::Null(DataType::Integer),
            Value::Null(DataType::Integer),
        ]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Same values ARE "not distinct"
        let row = Row::from_values(vec![Value::Integer(5), Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    // =========================================================================
    // Fused comparison operations tests
    // =========================================================================

    #[test]
    fn test_eq_column_const() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::EqColumnConst(0, Value::Integer(42)), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(42)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(100)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_ne_column_const() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::NeColumnConst(0, Value::Integer(42)), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(100)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_lt_column_const() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LtColumnConst(0, Value::Integer(10)), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(15)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_le_column_const() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LeColumnConst(0, Value::Integer(10)), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_gt_column_const() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::GtColumnConst(0, Value::Integer(10)), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(15)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_ge_column_const() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::GeColumnConst(0, Value::Integer(10)), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_is_null_column() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::IsNullColumn(0), Op::Return]);

        let row = Row::from_values(vec![Value::Null(DataType::Integer)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(1)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_is_not_null_column() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::IsNotNullColumn(0), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(1)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_between_column_const() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::BetweenColumnConst(0, Value::Integer(5), Value::Integer(15)),
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(20)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));

        // Edge case: value equals lower bound
        let row = Row::from_values(vec![Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Edge case: value equals upper bound
        let row = Row::from_values(vec![Value::Integer(15)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    // =========================================================================
    // Logical operations tests
    // =========================================================================

    #[test]
    fn test_or_short_circuit() {
        let mut vm = ExprVM::new();
        // WHERE col0 = 1 OR col1 = 2
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(1)),
            Op::Eq,
            Op::Or(8), // Jump if first condition is true
            Op::LoadColumn(1),
            Op::LoadConst(Value::Integer(2)),
            Op::Eq,
            Op::OrFinalize,
            Op::Return,
        ]);

        // First condition true
        let row = Row::from_values(vec![Value::Integer(1), Value::Integer(0)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Second condition true
        let row = Row::from_values(vec![Value::Integer(0), Value::Integer(2)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Both conditions false
        let row = Row::from_values(vec![Value::Integer(0), Value::Integer(0)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_not() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Boolean(true)),
            Op::Not,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_xor() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Boolean(true)),
            Op::LoadConst(Value::Boolean(false)),
            Op::Xor,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Both true = false
        let program = Program::new(vec![
            Op::LoadConst(Value::Boolean(true)),
            Op::LoadConst(Value::Boolean(true)),
            Op::Xor,
            Op::Return,
        ]);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    // =========================================================================
    // Arithmetic operations tests
    // =========================================================================

    #[test]
    fn test_add() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(10)),
            Op::LoadConst(Value::Integer(5)),
            Op::Add,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(15));
    }

    #[test]
    fn test_add_floats() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Float(2.5)),
            Op::LoadConst(Value::Float(3.5)),
            Op::Add,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Float(6.0));
    }

    #[test]
    fn test_sub() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(10)),
            Op::LoadConst(Value::Integer(3)),
            Op::Sub,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(7));
    }

    #[test]
    fn test_mul() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(6)),
            Op::LoadConst(Value::Integer(7)),
            Op::Mul,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(42));
    }

    #[test]
    fn test_div() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(20)),
            Op::LoadConst(Value::Integer(4)),
            Op::Div,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(5));
    }

    #[test]
    fn test_mod() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(17)),
            Op::LoadConst(Value::Integer(5)),
            Op::Mod,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(2));
    }

    #[test]
    fn test_neg() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadConst(Value::Integer(5)), Op::Neg, Op::Return]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(-5));
    }

    // =========================================================================
    // Bitwise operations tests
    // =========================================================================

    #[test]
    fn test_bit_and() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(0b1100)),
            Op::LoadConst(Value::Integer(0b1010)),
            Op::BitAnd,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(0b1000));
    }

    #[test]
    fn test_bit_or() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(0b1100)),
            Op::LoadConst(Value::Integer(0b1010)),
            Op::BitOr,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(0b1110));
    }

    #[test]
    fn test_bit_xor() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(0b1100)),
            Op::LoadConst(Value::Integer(0b1010)),
            Op::BitXor,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(0b0110));
    }

    #[test]
    fn test_bit_not() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(0b0000_0000_0000_0101)),
            Op::BitNot,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        let result = vm.execute(&program, &ctx).unwrap();
        // Bitwise NOT of 5 is -6 in two's complement
        assert_eq!(result, Value::Integer(!5));
    }

    #[test]
    fn test_shl() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(1)),
            Op::LoadConst(Value::Integer(4)),
            Op::Shl,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(16));
    }

    #[test]
    fn test_shr() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(16)),
            Op::LoadConst(Value::Integer(2)),
            Op::Shr,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(4));
    }

    // =========================================================================
    // String operations tests
    // =========================================================================

    #[test]
    fn test_concat() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Text("Hello".into())),
            Op::LoadConst(Value::Text(" World".into())),
            Op::Concat,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(
            vm.execute(&program, &ctx).unwrap(),
            Value::Text("Hello World".into())
        );
    }

    #[test]
    fn test_like_pattern() {
        let mut vm = ExprVM::new();
        let pattern = Pattern::like("%world%", false);
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::Like(Arc::new(pattern)),
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Text("hello world!".into())]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Text("hello there!".into())]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_glob_pattern() {
        let mut vm = ExprVM::new();
        let pattern = Pattern::glob("*.txt");
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::Glob(Arc::new(pattern)),
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Text("document.txt".into())]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Text("document.pdf".into())]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    // =========================================================================
    // Between tests
    // =========================================================================

    #[test]
    fn test_between() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(5)),
            Op::LoadConst(Value::Integer(15)),
            Op::Between,
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(3)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_not_between() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(5)),
            Op::LoadConst(Value::Integer(15)),
            Op::NotBetween,
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Integer(3)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    // =========================================================================
    // Not In Set tests
    // =========================================================================

    #[test]
    fn test_not_in_set() {
        let mut vm = ExprVM::new();
        let set: ValueSet = [Value::Integer(1), Value::Integer(2), Value::Integer(3)]
            .into_iter()
            .collect();

        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::NotInSet(pset(set, false)),
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(2)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_in_set_with_null() {
        let mut vm = ExprVM::new();
        let set: ValueSet = [Value::Integer(1), Value::Integer(2)].into_iter().collect();

        // has_null = true means the set conceptually contains NULL
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::InSet(pset(set, true)),
            Op::Return,
        ]);

        // Value not in set, but set has null -> returns NULL
        let row = Row::from_values(vec![Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        let result = vm.execute(&program, &ctx).unwrap();
        assert!(result.is_null());
    }

    // =========================================================================
    // In Set Column (fused) tests
    // =========================================================================

    #[test]
    fn test_in_set_column() {
        let mut vm = ExprVM::new();
        let set: ValueSet = [Value::Integer(1), Value::Integer(2), Value::Integer(3)]
            .into_iter()
            .collect();

        let program = Program::new(vec![Op::InSetColumn(0, pset(set, false)), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(2)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Integer(5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    // =========================================================================
    // Boolean checks (IS TRUE, IS FALSE, etc.)
    // =========================================================================

    #[test]
    fn test_is_true() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn(0), Op::IsTrue, Op::Return]);

        let row = Row::from_values(vec![Value::Boolean(true)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Boolean(false)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));

        let row = Row::from_values(vec![Value::Null(DataType::Boolean)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_is_not_true() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn(0), Op::IsNotTrue, Op::Return]);

        let row = Row::from_values(vec![Value::Boolean(true)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));

        let row = Row::from_values(vec![Value::Boolean(false)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Null(DataType::Boolean)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_is_false() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn(0), Op::IsFalse, Op::Return]);

        let row = Row::from_values(vec![Value::Boolean(false)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Boolean(true)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_is_not_false() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadColumn(0), Op::IsNotFalse, Op::Return]);

        let row = Row::from_values(vec![Value::Boolean(false)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));

        let row = Row::from_values(vec![Value::Boolean(true)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Null(DataType::Boolean)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    // =========================================================================
    // Coalesce, NullIf, Greatest, Least tests
    // =========================================================================

    #[test]
    fn test_coalesce() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadNull(DataType::Integer),
            Op::LoadNull(DataType::Integer),
            Op::LoadConst(Value::Integer(42)),
            Op::Coalesce(3),
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(42));
    }

    #[test]
    fn test_nullif() {
        let mut vm = ExprVM::new();
        // NULLIF(5, 5) -> NULL
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(5)),
            Op::LoadConst(Value::Integer(5)),
            Op::NullIf,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert!(vm.execute(&program, &ctx).unwrap().is_null());

        // NULLIF(5, 10) -> 5
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(5)),
            Op::LoadConst(Value::Integer(10)),
            Op::NullIf,
            Op::Return,
        ]);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(5));
    }

    #[test]
    fn test_greatest() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(3)),
            Op::LoadConst(Value::Integer(7)),
            Op::LoadConst(Value::Integer(2)),
            Op::Greatest(3),
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(7));
    }

    #[test]
    fn test_least() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(3)),
            Op::LoadConst(Value::Integer(7)),
            Op::LoadConst(Value::Integer(2)),
            Op::Least(3),
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(2));
    }

    // =========================================================================
    // Stack operations tests
    // =========================================================================

    #[test]
    fn test_dup() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(42)),
            Op::Dup,
            Op::Add, // 42 + 42
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(84));
    }

    #[test]
    fn test_swap() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(10)),
            Op::LoadConst(Value::Integer(3)),
            Op::Swap,
            Op::Sub, // 3 - 10
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(-7));
    }

    #[test]
    fn test_pop() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(100)),
            Op::LoadConst(Value::Integer(42)),
            Op::Pop, // Discard 42
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(100));
    }

    // =========================================================================
    // Jump operations tests
    // =========================================================================

    #[test]
    fn test_jump() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(1)),
            Op::Jump(3),
            Op::LoadConst(Value::Integer(2)), // Skipped
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(1));
    }

    #[test]
    fn test_jump_if_true() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Boolean(true)),
            Op::JumpIfTrue(4),
            Op::LoadConst(Value::Integer(0)), // Skipped
            Op::Return,
            Op::LoadConst(Value::Integer(1)), // Executed
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(1));
    }

    #[test]
    fn test_jump_if_false() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Boolean(false)),
            Op::JumpIfFalse(4),
            Op::LoadConst(Value::Integer(0)), // Skipped
            Op::Return,
            Op::LoadConst(Value::Integer(1)), // Executed
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(1));
    }

    #[test]
    fn test_jump_if_null() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadNull(DataType::Integer),
            Op::JumpIfNull(4),
            Op::LoadConst(Value::Integer(0)), // Skipped
            Op::Return,
            Op::LoadConst(Value::Integer(1)), // Executed
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(1));
    }

    #[test]
    fn test_jump_if_not_null() {
        let mut vm = ExprVM::new();
        // Test 1: Non-null value should jump
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(42)),
            Op::JumpIfNotNull(4),
            Op::LoadConst(Value::Integer(0)), // Skipped
            Op::Return,
            Op::LoadConst(Value::Integer(1)), // Executed
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(1));

        // Test 2: Null value should NOT jump
        let program2 = Program::new(vec![
            Op::LoadNull(DataType::Integer),
            Op::JumpIfNotNull(4),
            Op::LoadConst(Value::Integer(99)), // Executed (no jump)
            Op::Return,
            Op::LoadConst(Value::Integer(0)), // Not reached
            Op::Return,
        ]);
        assert_eq!(vm.execute(&program2, &ctx).unwrap(), Value::Integer(99));
    }

    #[test]
    fn test_coalesce_short_circuit() {
        // Test short-circuit COALESCE compilation pattern:
        // COALESCE(NULL, NULL, 42) should return 42 without evaluating further
        let mut vm = ExprVM::new();

        // Simulates: COALESCE(NULL, NULL, 42)
        // Bytecode: LoadNull, JumpIfNotNull(end), Pop, LoadNull, JumpIfNotNull(end), Pop, LoadConst(42)
        let program = Program::new(vec![
            Op::LoadNull(DataType::Integer),   // arg1: NULL
            Op::JumpIfNotNull(8),              // if not null, jump to end (position 8)
            Op::Pop,                           // pop the null
            Op::LoadNull(DataType::Integer),   // arg2: NULL
            Op::JumpIfNotNull(8),              // if not null, jump to end
            Op::Pop,                           // pop the null
            Op::LoadConst(Value::Integer(42)), // arg3: 42
            Op::Return,                        // end
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(42));

        // Test COALESCE(100, NULL, 42) - first non-null wins
        let program2 = Program::new(vec![
            Op::LoadConst(Value::Integer(100)), // arg1: 100 (not null)
            Op::JumpIfNotNull(8),               // jump to end
            Op::Pop,
            Op::LoadNull(DataType::Integer), // arg2: NULL (skipped)
            Op::JumpIfNotNull(8),
            Op::Pop,
            Op::LoadConst(Value::Integer(42)), // arg3: 42 (skipped)
            Op::Return,
        ]);
        assert_eq!(vm.execute(&program2, &ctx).unwrap(), Value::Integer(100));

        // Test COALESCE(NULL, 50, 42) - second wins
        let program3 = Program::new(vec![
            Op::LoadNull(DataType::Integer),   // arg1: NULL
            Op::JumpIfNotNull(8),              // no jump (is null)
            Op::Pop,                           // pop the null
            Op::LoadConst(Value::Integer(50)), // arg2: 50 (not null)
            Op::JumpIfNotNull(8),              // jump to end
            Op::Pop,
            Op::LoadConst(Value::Integer(42)), // arg3: 42 (skipped)
            Op::Return,
        ]);
        assert_eq!(vm.execute(&program3, &ctx).unwrap(), Value::Integer(50));
    }

    // =========================================================================
    // Return variants tests
    // =========================================================================

    #[test]
    fn test_return_true() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::ReturnTrue]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_return_false() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::ReturnFalse]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_return_null() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::ReturnNull(DataType::Text)]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        let result = vm.execute(&program, &ctx).unwrap();
        assert!(result.is_null());
    }

    // =========================================================================
    // Nop tests
    // =========================================================================

    #[test]
    fn test_nop() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(42)),
            Op::Nop,
            Op::Nop,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(42));
    }

    // =========================================================================
    // Empty program tests
    // =========================================================================

    #[test]
    fn test_empty_program() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        let result = vm.execute(&program, &ctx).unwrap();
        assert!(result.is_null());
    }

    // =========================================================================
    // execute_bool tests
    // =========================================================================

    #[test]
    fn test_execute_bool_simple() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::GtColumnConst(0, Value::Integer(5)), Op::Return]);

        let row = Row::from_values(vec![Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert!(vm.execute_bool(&program, &ctx));

        let row = Row::from_values(vec![Value::Integer(3)]);
        let ctx = ExecuteContext::new(&row);
        assert!(!vm.execute_bool(&program, &ctx));
    }

    #[test]
    fn test_execute_bool_with_null() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::GtColumnConst(0, Value::Integer(5)), Op::Return]);

        // Null comparison returns false (not true)
        let row = Row::from_values(vec![Value::Null(DataType::Integer)]);
        let ctx = ExecuteContext::new(&row);
        assert!(!vm.execute_bool(&program, &ctx));
    }

    #[test]
    fn test_execute_bool_between() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::BetweenColumnConst(0, Value::Integer(5), Value::Integer(15)),
            Op::Return,
        ]);

        let row = Row::from_values(vec![Value::Integer(10)]);
        let ctx = ExecuteContext::new(&row);
        assert!(vm.execute_bool(&program, &ctx));

        let row = Row::from_values(vec![Value::Integer(20)]);
        let ctx = ExecuteContext::new(&row);
        assert!(!vm.execute_bool(&program, &ctx));
    }

    #[test]
    fn test_execute_bool_is_null_column() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::IsNullColumn(0), Op::Return]);

        let row = Row::from_values(vec![Value::Null(DataType::Integer)]);
        let ctx = ExecuteContext::new(&row);
        assert!(vm.execute_bool(&program, &ctx));

        let row = Row::from_values(vec![Value::Integer(1)]);
        let ctx = ExecuteContext::new(&row);
        assert!(!vm.execute_bool(&program, &ctx));
    }

    #[test]
    fn test_execute_bool_load_const_true() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadConst(Value::Boolean(true)), Op::Return]);

        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert!(vm.execute_bool(&program, &ctx));
    }

    // =========================================================================
    // Cast tests
    // =========================================================================

    #[test]
    fn test_cast_int_to_float() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(42)),
            Op::Cast(DataType::Float),
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Float(42.0));
    }

    #[test]
    fn test_cast_float_to_int() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Float(42.7)),
            Op::Cast(DataType::Integer),
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(42));
    }

    #[test]
    fn test_cast_text_to_int() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Text("123".into())),
            Op::Cast(DataType::Integer),
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(123));
    }

    // =========================================================================
    // LikeColumn (fused) tests
    // =========================================================================

    #[test]
    fn test_like_column() {
        let mut vm = ExprVM::new();
        let pattern = Pattern::like("test%", false);
        let program = Program::new(vec![Op::LikeColumn(0, Arc::new(pattern)), Op::Return]);

        let row = Row::from_values(vec![Value::Text("testing".into())]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Text("other".into())]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    // =========================================================================
    // TruncateToDate tests
    // =========================================================================

    #[test]
    fn test_truncate_to_date() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Text("2024-01-15 14:30:00".into())),
            Op::TruncateToDate,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        let result = vm.execute(&program, &ctx).unwrap();
        // TruncateToDate returns a Timestamp with time set to 00:00:00
        if let Value::Timestamp(t) = result {
            use chrono::Datelike;
            assert_eq!(t.year(), 2024);
            assert_eq!(t.month(), 1);
            assert_eq!(t.day(), 15);
        } else {
            panic!("Expected Timestamp result, got {:?}", result);
        }
    }

    // =========================================================================
    // Pop/Jump combinations
    // =========================================================================

    #[test]
    fn test_pop_jump_if_true() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Boolean(true)),
            Op::PopJumpIfTrue(4),
            Op::LoadConst(Value::Integer(0)), // Skipped
            Op::Return,
            Op::LoadConst(Value::Integer(1)), // Executed
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(1));
    }

    #[test]
    fn test_pop_jump_if_false() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Boolean(false)),
            Op::PopJumpIfFalse(4),
            Op::LoadConst(Value::Integer(0)), // Skipped
            Op::Return,
            Op::LoadConst(Value::Integer(1)), // Executed
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(1));
    }

    // =========================================================================
    // Mixed type arithmetic
    // =========================================================================

    #[test]
    fn test_add_int_and_float() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Integer(10)),
            Op::LoadConst(Value::Float(2.5)),
            Op::Add,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Float(12.5));
    }

    // =========================================================================
    // String concatenation with non-strings
    // =========================================================================

    #[test]
    fn test_concat_int_to_string() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![
            Op::LoadConst(Value::Text("Value: ".into())),
            Op::LoadConst(Value::Integer(42)),
            Op::Concat,
            Op::Return,
        ]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        assert_eq!(
            vm.execute(&program, &ctx).unwrap(),
            Value::Text("Value: 42".into())
        );
    }

    // =========================================================================
    // Float comparisons
    // =========================================================================

    #[test]
    fn test_gt_column_const_float() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::GtColumnConst(0, Value::Float(2.5)), Op::Return]);

        let row = Row::from_values(vec![Value::Float(3.5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = Row::from_values(vec![Value::Float(1.5)]);
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }

    // =========================================================================
    // Test LoadTransactionId
    // =========================================================================

    #[test]
    fn test_load_transaction_id() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadTransactionId, Op::Return]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row).with_transaction_id(Some(12345));
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(12345));
    }

    #[test]
    fn test_load_transaction_id_none() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadTransactionId, Op::Return]);
        let row = Row::new();
        let ctx = ExecuteContext::new(&row);
        let result = vm.execute(&program, &ctx).unwrap();
        assert!(result.is_null());
    }

    // =========================================================================
    // Subquery ops
    // =========================================================================

    /// A block index, the value asked about, and the quantifier if any
    type Asked = (u16, Value, Option<(CompareOp, bool)>);

    /// Answers every block with `answer`, or fails, recording what it was asked
    struct Blocks {
        answer: Option<bool>,
        fail: bool,
        asked: std::cell::RefCell<Vec<Asked>>,
    }

    impl Blocks {
        fn answering(answer: Option<bool>) -> Self {
            Blocks {
                answer,
                fail: false,
                asked: Default::default(),
            }
        }

        fn failing() -> Self {
            Blocks {
                answer: None,
                fail: true,
                asked: Default::default(),
            }
        }

        fn result<T>(&self, value: T) -> Result<T> {
            if self.fail {
                Err(crate::core::Error::internal("block failed"))
            } else {
                Ok(value)
            }
        }
    }

    impl SubqueryExecutor for Blocks {
        fn execute_scalar(&self, _subquery_index: u16, _ctx: &ExecuteContext) -> Result<Value> {
            self.result(Value::Integer(7))
        }

        fn execute_exists(&self, _subquery_index: u16, _ctx: &ExecuteContext) -> Result<bool> {
            self.result(self.answer.unwrap_or(false))
        }

        fn execute_in(
            &self,
            subquery_index: u16,
            value: &Value,
            _ctx: &ExecuteContext,
        ) -> Result<Option<bool>> {
            self.asked
                .borrow_mut()
                .push((subquery_index, value.clone(), None));
            self.result(self.answer)
        }

        fn execute_quantified(
            &self,
            subquery_index: u16,
            value: &Value,
            op: CompareOp,
            all: bool,
            _ctx: &ExecuteContext,
        ) -> Result<Option<bool>> {
            self.asked
                .borrow_mut()
                .push((subquery_index, value.clone(), Some((op, all))));
            self.result(self.answer)
        }
    }

    fn run_with(program: &Program, row: &Row, blocks: &Blocks) -> Result<Value> {
        let ctx = ExecuteContext::new(row).with_subquery_executor(blocks);
        ExprVM::new().execute(program, &ctx)
    }

    /// A block whose value is the first column of the caller's second row
    struct SecondRow;

    impl SubqueryExecutor for SecondRow {
        fn execute_scalar(&self, _subquery_index: u16, ctx: &ExecuteContext) -> Result<Value> {
            Ok(ctx
                .row2
                .and_then(|row| row.get(0).cloned())
                .unwrap_or_else(Value::null_unknown))
        }

        fn execute_exists(&self, _subquery_index: u16, ctx: &ExecuteContext) -> Result<bool> {
            Ok(ctx.row2.is_some())
        }

        fn execute_in(
            &self,
            _subquery_index: u16,
            _value: &Value,
            _ctx: &ExecuteContext,
        ) -> Result<Option<bool>> {
            Ok(None)
        }

        fn execute_quantified(
            &self,
            _subquery_index: u16,
            _value: &Value,
            _op: CompareOp,
            _all: bool,
            _ctx: &ExecuteContext,
        ) -> Result<Option<bool>> {
            Ok(None)
        }
    }

    #[test]
    fn test_a_block_called_over_a_pair_reads_both_rows() {
        let left = Row::from_values(vec![Value::Integer(1)]);
        let right = Row::from_values(vec![Value::Integer(2)]);
        let ctx = ExecuteContext::for_join(&left, &right).with_subquery_executor(&SecondRow);
        let scalar = Program::new(vec![Op::ExecScalarSubquery(0), Op::Return]);
        assert_eq!(
            ExprVM::new().execute(&scalar, &ctx).unwrap(),
            Value::Integer(2)
        );
        let exists = Program::new(vec![Op::ExecExists(0), Op::Return]);
        assert_eq!(
            ExprVM::new().execute(&exists, &ctx).unwrap(),
            Value::Boolean(true)
        );
    }

    #[test]
    fn test_a_block_answer_reaches_the_stack_three_valued() {
        let row = Row::from_values(vec![Value::Integer(1)]);
        for (op, answer, expected) in [
            (Op::ExecInSubquery(3), Some(true), Value::Boolean(true)),
            (Op::ExecInSubquery(3), Some(false), Value::Boolean(false)),
            (Op::ExecInSubquery(3), None, Value::Null(DataType::Boolean)),
            (
                Op::ExecQuantified(3, CompareOp::Lt, true),
                None,
                Value::Null(DataType::Boolean),
            ),
            (
                Op::ExecQuantified(3, CompareOp::Lt, false),
                Some(true),
                Value::Boolean(true),
            ),
        ] {
            let program = Program::new(vec![Op::LoadColumn(0), op.clone(), Op::Return]);
            let blocks = Blocks::answering(answer);
            assert_eq!(
                run_with(&program, &row, &blocks).unwrap(),
                expected,
                "{op:?} answering {answer:?}"
            );
        }
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::ExecInSubquery(3),
            Op::Not,
            Op::Return,
        ]);
        assert_eq!(
            run_with(&program, &row, &Blocks::answering(None)).unwrap(),
            Value::Null(DataType::Boolean),
            "NOT IN keeps UNKNOWN"
        );
    }

    #[test]
    fn test_a_null_value_is_asked_of_the_block() {
        let row = Row::from_values(vec![Value::Null(DataType::Integer)]);
        let blocks = Blocks::answering(Some(false));
        let program = Program::new(vec![Op::LoadColumn(0), Op::ExecInSubquery(2), Op::Return]);
        assert_eq!(
            run_with(&program, &row, &blocks).unwrap(),
            Value::Boolean(false),
            "NULL IN (a block with no rows) is FALSE"
        );
        let blocks = Blocks::answering(Some(true));
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::ExecQuantified(5, CompareOp::Gt, true),
            Op::Return,
        ]);
        assert_eq!(
            run_with(&program, &row, &blocks).unwrap(),
            Value::Boolean(true),
            "NULL > ALL (a block with no rows) is TRUE"
        );
        assert_eq!(
            blocks.asked.borrow().as_slice(),
            &[(
                5,
                Value::Null(DataType::Integer),
                Some((CompareOp::Gt, true))
            )],
            "the block is asked with the value, its operator and its quantifier"
        );
    }

    #[test]
    fn test_a_block_error_is_the_statement_error() {
        let row = Row::from_values(vec![Value::Integer(1)]);
        for op in [
            Op::ExecScalarSubquery(0),
            Op::ExecExists(0),
            Op::ExecInSubquery(0),
            Op::ExecQuantified(0, CompareOp::Eq, false),
        ] {
            let program = Program::new(vec![Op::LoadColumn(0), op.clone(), Op::Return]);
            assert!(
                run_with(&program, &row, &Blocks::failing()).is_err(),
                "{op:?}: the block's error is not read as FALSE"
            );
            let ctx = ExecuteContext::new(&row);
            assert!(
                ExprVM::new().execute(&program, &ctx).is_err(),
                "{op:?}: a subquery without its prepared plan is an error"
            );
        }
    }

    #[test]
    fn released_values_leave_nothing_of_the_run_behind() {
        let statements =
            crate::parser::parse_sql("SELECT * FROM t WHERE POSITION('x', c) > 0").unwrap();
        let crate::parser::ast::Statement::Select(select) = &statements[0] else {
            panic!("not a SELECT");
        };
        let columns = ["c".to_string()];
        let ctx = super::super::compiler::CompileContext::with_global_registry(&columns);
        let program = super::super::compiler::ExprCompiler::new(&ctx)
            .compile(select.where_clause.as_deref().unwrap())
            .unwrap();
        let row = Row::from_values(vec![Value::text("y".repeat(4096))]);
        let mut vm = ExprVM::new();
        vm.execute(&program, &ExecuteContext::new(&row)).unwrap();
        assert!(!vm.args_buffer.is_empty(), "a call keeps its arguments");
        vm.release_values();
        assert!(vm.args_buffer.is_empty());
    }
}
