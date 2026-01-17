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
use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::ops::{CompareOp, Op};
use super::program::Program;
use crate::common::SmartString;
use crate::core::{DataType, Result, Row, Value, NULL_VALUE};

/// Stack value that can be borrowed (from row/constants) or owned (from operations)
type StackValue<'a> = Cow<'a, Value>;

/// Stack capacity for inline storage (avoids heap allocation for simple expressions)
/// Most expressions need 4-8 stack slots, so 8 covers the common case.
const STACK_INLINE_CAPACITY: usize = 16;

/// Arithmetic operation type (used for safe wrapping operations)
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

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
    pub outer_row: Option<&'a FxHashMap<Arc<str>, Value>>,

    /// Positional parameters
    pub params: &'a [Value],

    /// Named parameters
    pub named_params: Option<&'a FxHashMap<String, Value>>,

    /// Subquery executor (for scalar subqueries, EXISTS, etc.)
    /// This is a callback that can execute subqueries
    pub subquery_executor: Option<&'a dyn SubqueryExecutor>,

    /// Current transaction ID (for CURRENT_TRANSACTION_ID())
    pub transaction_id: Option<u64>,
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
            named_params: None,
            subquery_executor: None,
            transaction_id: None,
        }
    }

    /// Create a context with common parameters that can be reused across rows
    /// Only the row field needs to be updated for each iteration
    #[inline]
    pub fn with_common_params(
        row: &'a Row,
        params: &'a [Value],
        named_params: Option<&'a FxHashMap<String, Value>>,
        transaction_id: Option<u64>,
    ) -> Self {
        Self {
            row,
            row2: None,
            outer_row: None,
            params,
            named_params,
            subquery_executor: None,
            transaction_id,
        }
    }

    /// Create context for join evaluation
    pub fn for_join(row1: &'a Row, row2: &'a Row) -> Self {
        Self {
            row: row1,
            row2: Some(row2),
            outer_row: None,
            params: &[],
            named_params: None,
            subquery_executor: None,
            transaction_id: None,
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

    /// Add named parameters
    pub fn with_named_params(mut self, named_params: &'a FxHashMap<String, Value>) -> Self {
        self.named_params = Some(named_params);
        self
    }

    /// Add outer row context
    pub fn with_outer_row(mut self, outer_row: &'a FxHashMap<Arc<str>, Value>) -> Self {
        self.outer_row = Some(outer_row);
        self
    }

    /// Add subquery executor
    pub fn with_subquery_executor(mut self, executor: &'a dyn SubqueryExecutor) -> Self {
        self.subquery_executor = Some(executor);
        self
    }
}

/// Trait for executing subqueries
pub trait SubqueryExecutor {
    /// Execute a scalar subquery and return the result
    fn execute_scalar(
        &self,
        subquery_index: u16,
        outer_row: Option<&FxHashMap<Arc<str>, Value>>,
    ) -> Result<Value>;

    /// Execute EXISTS subquery
    fn execute_exists(
        &self,
        subquery_index: u16,
        outer_row: Option<&FxHashMap<Arc<str>, Value>>,
    ) -> Result<bool>;

    /// Execute IN subquery and check membership
    fn execute_in(
        &self,
        subquery_index: u16,
        value: &Value,
        outer_row: Option<&FxHashMap<Arc<str>, Value>>,
    ) -> Result<bool>;

    /// Execute ALL subquery
    fn execute_all(
        &self,
        subquery_index: u16,
        value: &Value,
        op: CompareOp,
        outer_row: Option<&FxHashMap<Arc<str>, Value>>,
    ) -> Result<bool>;

    /// Execute ANY subquery
    fn execute_any(
        &self,
        subquery_index: u16,
        value: &Value,
        op: CompareOp,
        outer_row: Option<&FxHashMap<Arc<str>, Value>>,
    ) -> Result<bool>;
}

/// Expression Virtual Machine
///
/// Executes compiled Programs against row data.
/// The VM is reusable - call execute() with different contexts.
/// Capacity for reusable args buffer (most functions have <= 4 args)
const ARGS_BUFFER_CAPACITY: usize = 8;

pub struct ExprVM {
    /// Evaluation stack (reused between executions)
    /// Uses SmallVec to avoid heap allocation for simple expressions (stack depth <= 16)
    stack: SmallVec<[Value; STACK_INLINE_CAPACITY]>,

    /// Reusable buffer for function arguments (avoids allocation per call)
    /// Uses SmallVec to avoid heap allocation for functions with <= 8 args
    args_buffer: SmallVec<[Value; ARGS_BUFFER_CAPACITY]>,
}

impl ExprVM {
    /// Create a new VM with default stack capacity
    /// Uses inline storage for up to 16 stack values and 8 args (no heap allocation)
    pub fn new() -> Self {
        Self {
            stack: SmallVec::new(),
            args_buffer: SmallVec::new(),
        }
    }

    /// Create a VM with specific stack capacity
    /// If capacity > 16, will spill to heap when needed
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            stack: SmallVec::with_capacity(capacity),
            args_buffer: SmallVec::new(),
        }
    }

    /// Execute a program and return the result
    #[inline]
    pub fn execute(&mut self, program: &Program, ctx: &ExecuteContext) -> Result<Value> {
        // Ensure stack has enough capacity
        if self.stack.capacity() < program.max_stack_depth() {
            self.stack
                .reserve(program.max_stack_depth() - self.stack.capacity());
        }
        self.stack.clear();

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
                    let idx = *idx as usize;
                    let value = ctx
                        .row
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(Value::null_unknown);
                    self.stack.push(value);
                    pc += 1;
                }

                Op::LoadColumn2(idx) => {
                    let idx = *idx as usize;
                    let value = ctx
                        .row2
                        .and_then(|r| r.get(idx).cloned())
                        .unwrap_or_else(Value::null_unknown);
                    self.stack.push(value);
                    pc += 1;
                }

                Op::LoadOuterColumn(name) => {
                    let value = ctx
                        .outer_row
                        .and_then(|r| r.get(name.as_ref()).cloned())
                        .unwrap_or_else(Value::null_unknown);
                    self.stack.push(value);
                    pc += 1;
                }

                Op::LoadConst(value) => {
                    self.stack.push(value.clone());
                    pc += 1;
                }

                Op::LoadParam(idx) => {
                    let idx = *idx as usize;
                    let value = ctx
                        .params
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(Value::null_unknown);
                    self.stack.push(value);
                    pc += 1;
                }

                Op::LoadNamedParam(name) => {
                    let value = ctx
                        .named_params
                        .and_then(|p| p.get(name.as_ref()).cloned())
                        .unwrap_or_else(Value::null_unknown);
                    self.stack.push(value);
                    pc += 1;
                }

                Op::LoadNull(dt) => {
                    self.stack.push(Value::Null(*dt));
                    pc += 1;
                }

                Op::LoadAggregateResult(idx) => {
                    // Aggregate results are stored in the row at specific indices
                    let idx = *idx as usize;
                    let value = ctx
                        .row
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(Value::null_unknown);
                    self.stack.push(value);
                    pc += 1;
                }

                Op::LoadTransactionId => {
                    // Load current transaction ID, or NULL if not in a transaction
                    let value = match ctx.transaction_id {
                        Some(txn_id) => Value::Integer(txn_id as i64),
                        None => Value::null_unknown(),
                    };
                    self.stack.push(value);
                    pc += 1;
                }

                // =============================================================
                // COMPARISON OPERATIONS
                // =============================================================
                Op::Eq => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if a.is_null() || b.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(a == b)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Ne => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if a.is_null() || b.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(a != b)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Lt => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    // Fast path for integer/float comparisons
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x < *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(*x < *y),
                        (Value::Integer(x), Value::Float(y)) => Value::Boolean((*x as f64) < *y),
                        (Value::Float(x), Value::Integer(y)) => Value::Boolean(*x < (*y as f64)),
                        _ => self.compare_values(&a, &b, std::cmp::Ordering::Less),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Le => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    // Fast path for integer/float comparisons
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x <= *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(*x <= *y),
                        (Value::Integer(x), Value::Float(y)) => Value::Boolean((*x as f64) <= *y),
                        (Value::Float(x), Value::Integer(y)) => Value::Boolean(*x <= (*y as f64)),
                        _ => match a.partial_cmp(&b) {
                            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => {
                                Value::Boolean(true)
                            }
                            Some(std::cmp::Ordering::Greater) => Value::Boolean(false),
                            None => Value::Null(DataType::Boolean),
                        },
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Gt => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    // Fast path for integer/float comparisons
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x > *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(*x > *y),
                        (Value::Integer(x), Value::Float(y)) => Value::Boolean((*x as f64) > *y),
                        (Value::Float(x), Value::Integer(y)) => Value::Boolean(*x > (*y as f64)),
                        _ => self.compare_values(&a, &b, std::cmp::Ordering::Greater),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Ge => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    // Fast path for integer/float comparisons
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x >= *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(*x >= *y),
                        (Value::Integer(x), Value::Float(y)) => Value::Boolean((*x as f64) >= *y),
                        (Value::Float(x), Value::Integer(y)) => Value::Boolean(*x >= (*y as f64)),
                        _ => match a.partial_cmp(&b) {
                            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal) => {
                                Value::Boolean(true)
                            }
                            Some(std::cmp::Ordering::Less) => Value::Boolean(false),
                            None => Value::Null(DataType::Boolean),
                        },
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::IsNull => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    self.stack.push(Value::Boolean(v.is_null()));
                    pc += 1;
                }

                Op::IsNotNull => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    self.stack.push(Value::Boolean(!v.is_null()));
                    pc += 1;
                }

                Op::IsDistinctFrom => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    // NULL IS DISTINCT FROM NULL = FALSE
                    // NULL IS DISTINCT FROM non-NULL = TRUE
                    // non-NULL IS DISTINCT FROM NULL = TRUE
                    // Otherwise use regular comparison
                    let result = match (a.is_null(), b.is_null()) {
                        (true, true) => false,
                        (true, false) | (false, true) => true,
                        (false, false) => a != b,
                    };
                    self.stack.push(Value::Boolean(result));
                    pc += 1;
                }

                Op::IsNotDistinctFrom => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (a.is_null(), b.is_null()) {
                        (true, true) => true,
                        (true, false) | (false, true) => false,
                        (false, false) => a == b,
                    };
                    self.stack.push(Value::Boolean(result));
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
                        _ => {
                            if col_val.is_null() || val.is_null() {
                                Value::Null(DataType::Boolean)
                            } else {
                                Value::Boolean(col_val == val)
                            }
                        }
                    };
                    self.stack.push(result);
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
                        _ => {
                            if col_val.is_null() || val.is_null() {
                                Value::Null(DataType::Boolean)
                            } else {
                                Value::Boolean(col_val != val)
                            }
                        }
                    };
                    self.stack.push(result);
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
                    self.stack.push(result);
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
                        _ => match col_val.partial_cmp(val) {
                            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => {
                                Value::Boolean(true)
                            }
                            Some(std::cmp::Ordering::Greater) => Value::Boolean(false),
                            None => Value::Null(DataType::Boolean),
                        },
                    };
                    self.stack.push(result);
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
                    self.stack.push(result);
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
                        _ => match col_val.partial_cmp(val) {
                            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal) => {
                                Value::Boolean(true)
                            }
                            Some(std::cmp::Ordering::Less) => Value::Boolean(false),
                            None => Value::Null(DataType::Boolean),
                        },
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::IsNullColumn(idx) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    self.stack.push(Value::Boolean(col_val.is_null()));
                    pc += 1;
                }

                Op::IsNotNullColumn(idx) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    self.stack.push(Value::Boolean(!col_val.is_null()));
                    pc += 1;
                }

                Op::LikeColumn(idx, pattern, case_insensitive) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    let result = match col_val {
                        Value::Text(s) => Value::Boolean(pattern.matches(s, *case_insensitive)),
                        Value::Null(_) => Value::Null(DataType::Boolean),
                        _ => Value::Boolean(false),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::InSetColumn(idx, set, has_null) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    let result = if col_val.is_null() {
                        Value::Null(DataType::Boolean)
                    } else if set.contains(col_val) {
                        Value::Boolean(true)
                    } else if *has_null {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(false)
                    };
                    self.stack.push(result);
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
                            Self::between_float(col_val, *lo as f64, *hi)
                        }
                        (Value::Float(lo), Value::Integer(hi)) => {
                            Self::between_float(col_val, *lo, *hi as f64)
                        }
                        _ if col_val.is_null() || low.is_null() || high.is_null() => {
                            Value::Null(DataType::Boolean)
                        }
                        _ => Value::Boolean(col_val >= low && col_val <= high),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // LOGICAL OPERATIONS
                // =============================================================
                Op::And(jump_target) => {
                    // Short-circuit AND: if top is false, jump
                    let top = self.stack.last().unwrap_or(&Value::Null(DataType::Boolean));
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
                    let top = self.stack.last().unwrap_or(&Value::Null(DataType::Boolean));
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
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (Self::to_tribool(&a), Self::to_tribool(&b)) {
                        (Some(false), _) | (_, Some(false)) => Value::Boolean(false),
                        (Some(true), Some(true)) => Value::Boolean(true),
                        _ => Value::Null(DataType::Boolean),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::OrFinalize => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (Self::to_tribool(&a), Self::to_tribool(&b)) {
                        (Some(true), _) | (_, Some(true)) => Value::Boolean(true),
                        (Some(false), Some(false)) => Value::Boolean(false),
                        _ => Value::Null(DataType::Boolean),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Not => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match Self::to_tribool(&v) {
                        Some(b) => Value::Boolean(!b),
                        None => Value::Null(DataType::Boolean),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Xor => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (Self::to_tribool(&a), Self::to_tribool(&b)) {
                        (Some(a), Some(b)) => Value::Boolean(a ^ b),
                        _ => Value::Null(DataType::Boolean),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // ARITHMETIC OPERATIONS
                // =============================================================
                Op::Add => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    // Handle timestamp + interval or timestamp + integer (days)
                    let result = match (&a, &b) {
                        (Value::Timestamp(t), Value::Integer(days)) => {
                            Value::Timestamp(*t + chrono::Duration::days(*days))
                        }
                        (Value::Integer(days), Value::Timestamp(t)) => {
                            Value::Timestamp(*t + chrono::Duration::days(*days))
                        }
                        (Value::Timestamp(_), Value::Text(_)) => {
                            // Parse interval string - pass references directly to avoid clone
                            self.timestamp_add_interval(&a, &b, true)
                        }
                        _ => Self::arithmetic_op(&a, &b, ArithmeticOp::Add, |x, y| x + y)?,
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Sub => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    // Handle timestamp - interval, timestamp - integer (days), or timestamp - timestamp
                    let result = match (&a, &b) {
                        (Value::Timestamp(t1), Value::Timestamp(t2)) => {
                            // Return interval text
                            let duration = t1.signed_duration_since(*t2);
                            Value::Text(SmartString::from_string(
                                self.format_duration_as_interval(duration),
                            ))
                        }
                        (Value::Timestamp(t), Value::Integer(days)) => {
                            Value::Timestamp(*t - chrono::Duration::days(*days))
                        }
                        (Value::Timestamp(_), Value::Text(_)) => {
                            // Parse interval string - pass references directly to avoid clone
                            self.timestamp_add_interval(&a, &b, false)
                        }
                        _ => Self::arithmetic_op(&a, &b, ArithmeticOp::Sub, |x, y| x - y)?,
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Mul => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = Self::arithmetic_op(&a, &b, ArithmeticOp::Mul, |x, y| x * y)?;
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Div => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = Self::div_op(&a, &b);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Mod => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = Self::mod_op(&a, &b);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Neg => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match v {
                        Value::Integer(i) => Value::Integer(-i),
                        Value::Float(f) => Value::Float(-f),
                        Value::Null(dt) => Value::Null(dt),
                        _ => Value::Null(DataType::Null),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // BITWISE OPERATIONS (inlined for performance)
                // =============================================================
                Op::BitAnd => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Integer(x & y),
                        _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                        _ => Value::Null(DataType::Null),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::BitOr => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Integer(x | y),
                        _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                        _ => Value::Null(DataType::Null),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::BitXor => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Integer(x ^ y),
                        _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                        _ => Value::Null(DataType::Null),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::BitNot => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match v {
                        Value::Integer(i) => Value::Integer(!i),
                        Value::Null(dt) => Value::Null(dt),
                        _ => Value::Null(DataType::Null),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Shl => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => {
                            Value::Integer(x.wrapping_shl(*y as u32))
                        }
                        _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                        _ => Value::Null(DataType::Null),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Shr => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (&a, &b) {
                        (Value::Integer(x), Value::Integer(y)) => {
                            Value::Integer(x.wrapping_shr(*y as u32))
                        }
                        _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
                        _ => Value::Null(DataType::Null),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // STRING OPERATIONS
                // =============================================================
                Op::Concat => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if a.is_null() || b.is_null() {
                        Value::Null(DataType::Text)
                    } else {
                        // Fast path: both are Text
                        match (&a, &b) {
                            (Value::Text(a_str), Value::Text(b_str)) => {
                                // Use optimized concat - handles inline and heap efficiently
                                Value::Text(SmartString::concat(a_str, b_str))
                            }
                            (Value::Text(a_str), _) => {
                                // a is Text, b needs conversion - use Arc to avoid shrink_to_fit
                                use std::fmt::Write;
                                let mut s = String::with_capacity(a_str.len() + 32);
                                s.push_str(a_str);
                                let _ = write!(s, "{}", b);
                                Value::Text(SmartString::from_string_shared(s))
                            }
                            (_, Value::Text(b_str)) => {
                                // a needs conversion, b is Text - use Arc to avoid shrink_to_fit
                                use std::fmt::Write;
                                let mut s = String::with_capacity(32 + b_str.len());
                                let _ = write!(s, "{}", a);
                                s.push_str(b_str);
                                Value::Text(SmartString::from_string_shared(s))
                            }
                            _ => {
                                // Both need conversion - use Arc to avoid shrink_to_fit
                                use std::fmt::Write;
                                let mut s = String::with_capacity(64);
                                let _ = write!(s, "{}{}", a, b);
                                Value::Text(SmartString::from_string_shared(s))
                            }
                        }
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::ConcatN(n) => {
                    let n = *n as usize;
                    let start = self.stack.len().saturating_sub(n);

                    // Single pass: check for NULL and calculate total length
                    let mut total_len = 0usize;
                    let mut has_null = false;
                    let mut all_text = true;

                    for v in &self.stack[start..] {
                        match v {
                            Value::Null(_) => {
                                has_null = true;
                                break;
                            }
                            Value::Text(s) => total_len += s.len(),
                            _ => {
                                all_text = false;
                                total_len += 32;
                            }
                        }
                    }

                    if has_null {
                        self.stack.truncate(start);
                        self.stack.push(Value::Null(DataType::Text));
                        pc += 1;
                        continue;
                    }

                    // Build result - optimize for inline vs heap
                    let result = if all_text && total_len <= 22 {
                        // Fast path: build directly into inline SmartString (no heap allocation)
                        let mut data = [0u8; 22];
                        let mut pos = 0;
                        for v in self.stack.drain(start..) {
                            if let Value::Text(text) = v {
                                let bytes = text.as_bytes();
                                data[pos..pos + bytes.len()].copy_from_slice(bytes);
                                pos += bytes.len();
                            }
                        }
                        // SAFETY: SmartString only stores valid UTF-8
                        SmartString::Inline {
                            len: total_len as u8,
                            data,
                        }
                    } else if all_text {
                        // Heap path: exact capacity, into_boxed_str is O(1) when len == capacity
                        let mut s = String::with_capacity(total_len);
                        for v in self.stack.drain(start..) {
                            if let Value::Text(text) = v {
                                s.push_str(&text);
                            }
                        }
                        // len == capacity, so into_boxed_str is O(1)
                        SmartString::from_string(s)
                    } else {
                        // Mixed types: capacity is estimate, use Arc to avoid shrink_to_fit
                        let mut s = String::with_capacity(total_len);
                        for v in self.stack.drain(start..) {
                            match v {
                                Value::Text(text) => s.push_str(&text),
                                _ => {
                                    use std::fmt::Write;
                                    let _ = write!(s, "{}", v);
                                }
                            }
                        }
                        SmartString::from_string_shared(s)
                    };
                    self.stack.push(Value::Text(result));
                    pc += 1;
                }

                Op::Like(pattern, case_insensitive) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match &v {
                        Value::Text(s) => Value::Boolean(pattern.matches(s, *case_insensitive)),
                        Value::Null(_) => Value::Null(DataType::Boolean),
                        _ => Value::Boolean(false),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Glob(pattern) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match &v {
                        Value::Text(s) => Value::Boolean(pattern.matches(s, false)),
                        Value::Null(_) => Value::Null(DataType::Boolean),
                        _ => Value::Boolean(false),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Regexp(regex) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match &v {
                        Value::Text(s) => Value::Boolean(regex.is_match(s)),
                        Value::Null(_) => Value::Null(DataType::Boolean),
                        _ => Value::Boolean(false),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::LikeEscape(pattern, case_insensitive, _escape) => {
                    // ESCAPE is already incorporated into the compiled pattern
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match &v {
                        Value::Text(s) => Value::Boolean(pattern.matches(s, *case_insensitive)),
                        Value::Null(_) => Value::Null(DataType::Boolean),
                        _ => Value::Boolean(false),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // JSON OPERATIONS
                // =============================================================
                Op::JsonAccess => {
                    let key = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let json_val = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = self.json_access(&json_val, &key, false);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::JsonAccessText => {
                    let key = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let json_val = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = self.json_access(&json_val, &key, true);
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // TIMESTAMP OPERATIONS
                // =============================================================
                Op::TimestampAddInterval => {
                    let interval = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let ts = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = self.timestamp_add_interval(&ts, &interval, true);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::TimestampSubInterval => {
                    let interval = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let ts = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = self.timestamp_add_interval(&ts, &interval, false);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::TimestampDiff => {
                    let ts2 = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let ts1 = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (&ts1, &ts2) {
                        (Value::Timestamp(t1), Value::Timestamp(t2)) => {
                            let duration = t1.signed_duration_since(*t2);
                            Value::Text(SmartString::from_string(
                                self.format_duration_as_interval(duration),
                            ))
                        }
                        _ if ts1.is_null() || ts2.is_null() => Value::Null(DataType::Text),
                        _ => Value::Null(DataType::Text),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::TimestampAddDays => {
                    let days = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let ts = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (&ts, &days) {
                        (Value::Timestamp(t), Value::Integer(d)) => {
                            Value::Timestamp(*t + chrono::Duration::days(*d))
                        }
                        _ if ts.is_null() || days.is_null() => Value::Null(DataType::Timestamp),
                        _ => Value::Null(DataType::Timestamp),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::TimestampSubDays => {
                    let days = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let ts = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match (&ts, &days) {
                        (Value::Timestamp(t), Value::Integer(d)) => {
                            Value::Timestamp(*t - chrono::Duration::days(*d))
                        }
                        _ if ts.is_null() || days.is_null() => Value::Null(DataType::Timestamp),
                        _ => Value::Null(DataType::Timestamp),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // SET OPERATIONS
                // =============================================================
                Op::InSet(set, has_null) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if v.is_null() {
                        Value::Null(DataType::Boolean)
                    } else if set.contains(&v) {
                        Value::Boolean(true)
                    } else if *has_null {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(false)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::NotInSet(set, has_null) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if v.is_null() {
                        Value::Null(DataType::Boolean)
                    } else if set.contains(&v) {
                        Value::Boolean(false)
                    } else if *has_null {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(true)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Between => {
                    let high = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let low = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let val = self.stack.pop().unwrap_or_else(Value::null_unknown);

                    // Fast path for integer/float comparisons
                    let result = match (&val, &low, &high) {
                        (Value::Integer(v), Value::Integer(lo), Value::Integer(hi)) => {
                            Value::Boolean(*v >= *lo && *v <= *hi)
                        }
                        (Value::Float(v), Value::Float(lo), Value::Float(hi)) => {
                            Value::Boolean(*v >= *lo && *v <= *hi)
                        }
                        (Value::Integer(v), Value::Integer(lo), Value::Float(hi)) => {
                            Value::Boolean((*v as f64) >= (*lo as f64) && (*v as f64) <= *hi)
                        }
                        (Value::Integer(v), Value::Float(lo), Value::Integer(hi)) => {
                            Value::Boolean((*v as f64) >= *lo && (*v as f64) <= (*hi as f64))
                        }
                        (Value::Float(v), Value::Integer(lo), Value::Integer(hi)) => {
                            Value::Boolean(*v >= (*lo as f64) && *v <= (*hi as f64))
                        }
                        (Value::Float(v), Value::Integer(lo), Value::Float(hi)) => {
                            Value::Boolean(*v >= (*lo as f64) && *v <= *hi)
                        }
                        (Value::Float(v), Value::Float(lo), Value::Integer(hi)) => {
                            Value::Boolean(*v >= *lo && *v <= (*hi as f64))
                        }
                        (Value::Integer(v), Value::Float(lo), Value::Float(hi)) => {
                            Value::Boolean((*v as f64) >= *lo && (*v as f64) <= *hi)
                        }
                        _ if val.is_null() || low.is_null() || high.is_null() => {
                            Value::Null(DataType::Boolean)
                        }
                        _ => Value::Boolean(val >= low && val <= high),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::NotBetween => {
                    let high = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let low = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let val = self.stack.pop().unwrap_or_else(Value::null_unknown);

                    // Fast path for integer/float comparisons
                    let result = match (&val, &low, &high) {
                        (Value::Integer(v), Value::Integer(lo), Value::Integer(hi)) => {
                            Value::Boolean(*v < *lo || *v > *hi)
                        }
                        (Value::Float(v), Value::Float(lo), Value::Float(hi)) => {
                            Value::Boolean(*v < *lo || *v > *hi)
                        }
                        (Value::Integer(v), Value::Integer(lo), Value::Float(hi)) => {
                            Value::Boolean((*v as f64) < (*lo as f64) || (*v as f64) > *hi)
                        }
                        (Value::Integer(v), Value::Float(lo), Value::Integer(hi)) => {
                            Value::Boolean((*v as f64) < *lo || (*v as f64) > (*hi as f64))
                        }
                        (Value::Float(v), Value::Integer(lo), Value::Integer(hi)) => {
                            Value::Boolean(*v < (*lo as f64) || *v > (*hi as f64))
                        }
                        (Value::Float(v), Value::Integer(lo), Value::Float(hi)) => {
                            Value::Boolean(*v < (*lo as f64) || *v > *hi)
                        }
                        (Value::Float(v), Value::Float(lo), Value::Integer(hi)) => {
                            Value::Boolean(*v < *lo || *v > (*hi as f64))
                        }
                        (Value::Integer(v), Value::Float(lo), Value::Float(hi)) => {
                            Value::Boolean((*v as f64) < *lo || (*v as f64) > *hi)
                        }
                        _ if val.is_null() || low.is_null() || high.is_null() => {
                            Value::Null(DataType::Boolean)
                        }
                        _ => Value::Boolean(val < low || val > high),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::InTupleSet {
                    tuple_size,
                    values,
                    negated,
                } => {
                    let tuple_size = *tuple_size as usize;
                    let start = self.stack.len().saturating_sub(tuple_size);

                    // Reuse args_buffer to avoid allocation
                    self.args_buffer.clear();
                    self.args_buffer.extend(self.stack.drain(start..));

                    // Check if any values are NULL
                    let has_null_in_tuple = self.args_buffer.iter().any(|v| v.is_null());

                    if has_null_in_tuple {
                        // NULL in tuple -> result is NULL
                        self.stack.push(Value::Null(DataType::Boolean));
                    } else {
                        // Check membership
                        let found = values.iter().any(|tuple| {
                            tuple.len() == self.args_buffer.len()
                                && tuple
                                    .iter()
                                    .zip(self.args_buffer.iter())
                                    .all(|(a, b)| a == b)
                        });

                        let result = if *negated { !found } else { found };
                        self.stack.push(Value::Boolean(result));
                    }
                    pc += 1;
                }

                // =============================================================
                // BOOLEAN CHECKS
                // =============================================================
                Op::IsTrue => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match v {
                        Value::Boolean(b) => Value::Boolean(b),
                        Value::Null(_) => Value::Boolean(false),
                        _ => Value::Boolean(false),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::IsNotTrue => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match v {
                        Value::Boolean(b) => Value::Boolean(!b),
                        Value::Null(_) => Value::Boolean(true),
                        _ => Value::Boolean(true),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::IsFalse => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match v {
                        Value::Boolean(b) => Value::Boolean(!b),
                        Value::Null(_) => Value::Boolean(false),
                        _ => Value::Boolean(false),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::IsNotFalse => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match v {
                        Value::Boolean(b) => Value::Boolean(b),
                        Value::Null(_) => Value::Boolean(true),
                        _ => Value::Boolean(true),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // FUNCTION CALLS
                // =============================================================
                Op::CallScalar { func, arg_count } => {
                    let arg_count = *arg_count as usize;
                    let start = self.stack.len().saturating_sub(arg_count);

                    // Reuse args_buffer to avoid allocation
                    self.args_buffer.clear();
                    self.args_buffer.extend(self.stack.drain(start..));

                    let result = func
                        .evaluate(&self.args_buffer)
                        .unwrap_or_else(|_| Value::null_unknown());
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Coalesce(n) => {
                    let n = *n as usize;
                    let start = self.stack.len().saturating_sub(n);

                    // Find first non-null using slice iteration (no intermediate buffer)
                    let result_idx = self.stack[start..]
                        .iter()
                        .position(|v| !v.is_null())
                        .map(|i| start + i);

                    let result = if let Some(idx) = result_idx {
                        // Swap the result to end, pop it, then truncate the rest
                        let last = self.stack.len() - 1;
                        self.stack.swap(idx, last);
                        let result = self.stack.pop().unwrap_or_else(Value::null_unknown);
                        self.stack.truncate(start);
                        result
                    } else {
                        self.stack.truncate(start);
                        Value::null_unknown()
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::NullIf => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if a == b { Value::null_unknown() } else { a };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Greatest(n) => {
                    let n = *n as usize;
                    let start = self.stack.len().saturating_sub(n);

                    // Find max using slice iteration (no intermediate buffer)
                    let mut max_idx: Option<usize> = None;
                    for (i, v) in self.stack[start..].iter().enumerate() {
                        if !v.is_null() {
                            match max_idx {
                                None => max_idx = Some(start + i),
                                Some(mi) => {
                                    if v > &self.stack[mi] {
                                        max_idx = Some(start + i);
                                    }
                                }
                            }
                        }
                    }

                    let result = if let Some(idx) = max_idx {
                        // Swap the result to end, pop it, then truncate the rest
                        let last = self.stack.len() - 1;
                        self.stack.swap(idx, last);
                        let result = self.stack.pop().unwrap_or_else(Value::null_unknown);
                        self.stack.truncate(start);
                        result
                    } else {
                        self.stack.truncate(start);
                        Value::null_unknown()
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Least(n) => {
                    let n = *n as usize;
                    let start = self.stack.len().saturating_sub(n);

                    // Find min using slice iteration (no intermediate buffer)
                    let mut min_idx: Option<usize> = None;
                    for (i, v) in self.stack[start..].iter().enumerate() {
                        if !v.is_null() {
                            match min_idx {
                                None => min_idx = Some(start + i),
                                Some(mi) => {
                                    if v < &self.stack[mi] {
                                        min_idx = Some(start + i);
                                    }
                                }
                            }
                        }
                    }

                    let result = if let Some(idx) = min_idx {
                        // Swap the result to end, pop it, then truncate the rest
                        let last = self.stack.len() - 1;
                        self.stack.swap(idx, last);
                        let result = self.stack.pop().unwrap_or_else(Value::null_unknown);
                        self.stack.truncate(start);
                        result
                    } else {
                        self.stack.truncate(start);
                        Value::null_unknown()
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // NATIVE SCALAR FUNCTIONS (direct function pointer call)
                // In-place mutation - no pop/push overhead
                // =============================================================
                Op::NativeFn1(func) => {
                    if let Some(v) = self.stack.last_mut() {
                        func(v);
                    }
                    pc += 1;
                }

                // =============================================================
                // TYPE OPERATIONS
                // =============================================================
                Op::Cast(target_type) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = v.coerce_to_type(*target_type);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::TruncateToDate => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match &v {
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
                    self.stack.push(result);
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
                    let cond = self.stack.pop().unwrap_or_else(Value::null_unknown);
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
                    let when_val = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    // Use reference to avoid clone - case_val stays on stack for next comparison
                    let result = match self.stack.last() {
                        Some(case_val) if !case_val.is_null() && !when_val.is_null() => {
                            Value::Boolean(case_val == &when_val)
                        }
                        _ => Value::Boolean(false),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                // =============================================================
                // CONTROL FLOW
                // =============================================================
                Op::Jump(target) => {
                    pc = *target as usize;
                }

                Op::JumpIfTrue(target) => {
                    let top = self.stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    if Self::to_bool(top) {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::JumpIfFalse(target) => {
                    let top = self.stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    if !Self::to_bool(top) {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::JumpIfNull(target) => {
                    let top = self.stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    if top.is_null() {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::JumpIfNotNull(target) => {
                    let top = self.stack.last().unwrap_or(&Value::Null(DataType::Boolean));
                    if !top.is_null() {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::PopJumpIfTrue(target) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    if Self::to_bool(&v) {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::PopJumpIfFalse(target) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    if !Self::to_bool(&v) {
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }

                Op::Dup => {
                    if let Some(v) = self.stack.last().cloned() {
                        self.stack.push(v);
                    }
                    pc += 1;
                }

                Op::Pop => {
                    // Use truncate instead of pop to drop in-place without copying value out
                    let new_len = self.stack.len().saturating_sub(1);
                    self.stack.truncate(new_len);
                    pc += 1;
                }

                Op::Swap => {
                    let len = self.stack.len();
                    if len >= 2 {
                        self.stack.swap(len - 1, len - 2);
                    }
                    pc += 1;
                }

                // =============================================================
                // SUBQUERY OPERATIONS
                // =============================================================
                Op::ExecScalarSubquery(idx) => {
                    let result = if let Some(executor) = ctx.subquery_executor {
                        executor
                            .execute_scalar(*idx, ctx.outer_row)
                            .unwrap_or_else(|_| Value::null_unknown())
                    } else {
                        Value::null_unknown()
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::ExecExists(idx) => {
                    let result = if let Some(executor) = ctx.subquery_executor {
                        executor
                            .execute_exists(*idx, ctx.outer_row)
                            .unwrap_or(false)
                    } else {
                        false
                    };
                    self.stack.push(Value::Boolean(result));
                    pc += 1;
                }

                Op::ExecInSubquery(idx) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if v.is_null() {
                        Value::Null(DataType::Boolean)
                    } else if let Some(executor) = ctx.subquery_executor {
                        let in_result = executor
                            .execute_in(*idx, &v, ctx.outer_row)
                            .unwrap_or(false);
                        Value::Boolean(in_result)
                    } else {
                        Value::Boolean(false)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::ExecAll(idx, compare_op) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if v.is_null() {
                        Value::Null(DataType::Boolean)
                    } else if let Some(executor) = ctx.subquery_executor {
                        let all_result = executor
                            .execute_all(*idx, &v, *compare_op, ctx.outer_row)
                            .unwrap_or(false);
                        Value::Boolean(all_result)
                    } else {
                        Value::Boolean(false)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::ExecAny(idx, compare_op) => {
                    let v = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = if v.is_null() {
                        Value::Null(DataType::Boolean)
                    } else if let Some(executor) = ctx.subquery_executor {
                        let any_result = executor
                            .execute_any(*idx, &v, *compare_op, ctx.outer_row)
                            .unwrap_or(false);
                        Value::Boolean(any_result)
                    } else {
                        Value::Boolean(false)
                    };
                    self.stack.push(result);
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
                    self.stack.clear();
                    self.stack.push(Value::Boolean(true));
                    break;
                }

                Op::ReturnFalse => {
                    self.stack.clear();
                    self.stack.push(Value::Boolean(false));
                    break;
                }

                Op::ReturnNull(dt) => {
                    self.stack.clear();
                    self.stack.push(Value::Null(*dt));
                    break;
                }
            }
        }

        // Return top of stack or NULL
        Ok(self.stack.pop().unwrap_or_else(Value::null_unknown))
    }

    /// Execute a program using borrowed values where possible (Cow-based stack)
    ///
    /// This version avoids cloning values from the row when possible.
    /// Values are only cloned when they need to be modified or passed to functions.
    #[inline]
    pub fn execute_cow<'a>(
        &mut self,
        program: &'a Program,
        ctx: &'a ExecuteContext<'a>,
    ) -> Result<Value> {
        // Local stack with borrowed values - lifetime tied to this execution
        let mut stack: SmallVec<[StackValue<'a>; STACK_INLINE_CAPACITY]> = SmallVec::new();

        let ops = program.ops();
        if ops.is_empty() {
            return Ok(NULL_VALUE.clone());
        }

        let mut pc: usize = 0;

        loop {
            if pc >= ops.len() {
                break;
            }

            match &ops[pc] {
                // LOAD OPERATIONS - borrow instead of clone
                Op::LoadColumn(idx) => {
                    let idx = *idx as usize;
                    let value = ctx
                        .row
                        .get(idx)
                        .map(Cow::Borrowed)
                        .unwrap_or_else(|| Cow::Borrowed(&NULL_VALUE));
                    stack.push(value);
                    pc += 1;
                }

                Op::LoadColumn2(idx) => {
                    let idx = *idx as usize;
                    let value = ctx
                        .row2
                        .and_then(|r| r.get(idx))
                        .map(Cow::Borrowed)
                        .unwrap_or_else(|| Cow::Borrowed(&NULL_VALUE));
                    stack.push(value);
                    pc += 1;
                }

                Op::LoadConst(value) => {
                    stack.push(Cow::Borrowed(value));
                    pc += 1;
                }

                Op::LoadParam(idx) => {
                    let idx = *idx as usize;
                    let value = ctx
                        .params
                        .get(idx)
                        .map(Cow::Borrowed)
                        .unwrap_or_else(|| Cow::Borrowed(&NULL_VALUE));
                    stack.push(value);
                    pc += 1;
                }

                Op::LoadNull(dt) => {
                    stack.push(Cow::Owned(Value::Null(*dt)));
                    pc += 1;
                }

                Op::LoadAggregateResult(idx) => {
                    let idx = *idx as usize;
                    let value = ctx
                        .row
                        .get(idx)
                        .map(Cow::Borrowed)
                        .unwrap_or_else(|| Cow::Borrowed(&NULL_VALUE));
                    stack.push(value);
                    pc += 1;
                }

                // COMPARISON OPERATIONS - work with references
                Op::Eq => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = if a.is_null() || b.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(*a == *b)
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::Ne => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = if a.is_null() || b.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(*a != *b)
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::Lt => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = match (&*a, &*b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x < *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(*x < *y),
                        (Value::Integer(x), Value::Float(y)) => Value::Boolean((*x as f64) < *y),
                        (Value::Float(x), Value::Integer(y)) => Value::Boolean(*x < (*y as f64)),
                        _ => self.compare_values(&a, &b, std::cmp::Ordering::Less),
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::Le => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = match (&*a, &*b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x <= *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(*x <= *y),
                        (Value::Integer(x), Value::Float(y)) => Value::Boolean((*x as f64) <= *y),
                        (Value::Float(x), Value::Integer(y)) => Value::Boolean(*x <= (*y as f64)),
                        _ => match (*a).partial_cmp(&*b) {
                            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => {
                                Value::Boolean(true)
                            }
                            Some(std::cmp::Ordering::Greater) => Value::Boolean(false),
                            None => Value::Null(DataType::Boolean),
                        },
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::Gt => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = match (&*a, &*b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x > *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(*x > *y),
                        (Value::Integer(x), Value::Float(y)) => Value::Boolean((*x as f64) > *y),
                        (Value::Float(x), Value::Integer(y)) => Value::Boolean(*x > (*y as f64)),
                        _ => self.compare_values(&a, &b, std::cmp::Ordering::Greater),
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::Ge => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = match (&*a, &*b) {
                        (Value::Integer(x), Value::Integer(y)) => Value::Boolean(*x >= *y),
                        (Value::Float(x), Value::Float(y)) => Value::Boolean(*x >= *y),
                        (Value::Integer(x), Value::Float(y)) => Value::Boolean((*x as f64) >= *y),
                        (Value::Float(x), Value::Integer(y)) => Value::Boolean(*x >= (*y as f64)),
                        _ => match (*a).partial_cmp(&*b) {
                            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal) => {
                                Value::Boolean(true)
                            }
                            Some(std::cmp::Ordering::Less) => Value::Boolean(false),
                            None => Value::Null(DataType::Boolean),
                        },
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::IsNull => {
                    let v = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    stack.push(Cow::Owned(Value::Boolean(v.is_null())));
                    pc += 1;
                }

                Op::IsNotNull => {
                    let v = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    stack.push(Cow::Owned(Value::Boolean(!v.is_null())));
                    pc += 1;
                }

                // LOGICAL OPERATIONS
                Op::And(jump_target) => {
                    let top = stack.last().map(|v| &**v).unwrap_or(&NULL_VALUE);
                    match top {
                        Value::Boolean(false) => pc = *jump_target as usize,
                        Value::Null(_) => pc += 1,
                        _ => pc += 1,
                    }
                }

                Op::Or(jump_target) => {
                    let top = stack.last().map(|v| &**v).unwrap_or(&NULL_VALUE);
                    match top {
                        Value::Boolean(true) => pc = *jump_target as usize,
                        Value::Null(_) => pc += 1,
                        _ => pc += 1,
                    }
                }

                Op::AndFinalize => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = match (Self::to_tribool(&a), Self::to_tribool(&b)) {
                        (Some(false), _) | (_, Some(false)) => Value::Boolean(false),
                        (Some(true), Some(true)) => Value::Boolean(true),
                        _ => Value::Null(DataType::Boolean),
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::OrFinalize => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = match (Self::to_tribool(&a), Self::to_tribool(&b)) {
                        (Some(true), _) | (_, Some(true)) => Value::Boolean(true),
                        (Some(false), Some(false)) => Value::Boolean(false),
                        _ => Value::Null(DataType::Boolean),
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::Not => {
                    let v = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = match Self::to_tribool(&v) {
                        Some(b) => Value::Boolean(!b),
                        None => Value::Null(DataType::Boolean),
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                // FUNCTION CALLS - need to convert to owned
                Op::CallScalar { func, arg_count } => {
                    let arg_count = *arg_count as usize;
                    let start = stack.len().saturating_sub(arg_count);

                    // Convert Cow values to owned for function call
                    self.args_buffer.clear();
                    for cow_val in stack.drain(start..) {
                        self.args_buffer.push(cow_val.into_owned());
                    }

                    let result = func
                        .evaluate(&self.args_buffer)
                        .unwrap_or_else(|_| Value::null_unknown());
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                // FUSED OPERATIONS (already optimized, no stack involvement)
                Op::GtColumnConst(idx, val) => {
                    let col_val = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                    let result = match val {
                        Value::Integer(threshold) => Self::gt_int(col_val, *threshold),
                        Value::Float(threshold) => Self::gt_float(col_val, *threshold),
                        _ => self.compare_values(col_val, val, std::cmp::Ordering::Greater),
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::LtColumnConst(idx, val) => {
                    let col_val = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                    let result = match val {
                        Value::Integer(threshold) => Self::lt_int(col_val, *threshold),
                        Value::Float(threshold) => Self::lt_float(col_val, *threshold),
                        _ => self.compare_values(col_val, val, std::cmp::Ordering::Less),
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::EqColumnConst(idx, val) => {
                    let col_val = ctx.row.get(*idx as usize).unwrap_or(&NULL_VALUE);
                    let result = match val {
                        Value::Integer(v) => Self::eq_int(col_val, *v),
                        Value::Float(v) => Self::eq_float(col_val, *v),
                        _ => {
                            if col_val.is_null() || val.is_null() {
                                Value::Null(DataType::Boolean)
                            } else {
                                Value::Boolean(col_val == val)
                            }
                        }
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                // COALESCE - return first non-null value
                Op::Coalesce(n) => {
                    let n = *n as usize;
                    let start = stack.len().saturating_sub(n);

                    // Find first non-null value index
                    let result_idx = stack[start..]
                        .iter()
                        .position(|v| !v.is_null())
                        .map(|i| start + i);

                    let result = if let Some(idx) = result_idx {
                        // Move the result out, swap to end, pop, then truncate
                        let last = stack.len() - 1;
                        stack.swap(idx, last);
                        // pop() should always succeed since we found an index
                        stack.pop().expect("stack underflow in COALESCE")
                    } else {
                        Cow::Borrowed(&NULL_VALUE)
                    };
                    stack.truncate(start);
                    stack.push(result);
                    pc += 1;
                }

                // NULLIF - return NULL if both args are equal
                Op::NullIf => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = if *a == *b {
                        Cow::Borrowed(&NULL_VALUE)
                    } else {
                        a
                    };
                    stack.push(result);
                    pc += 1;
                }

                // GREATEST - return maximum non-null value
                Op::Greatest(n) => {
                    let n = *n as usize;
                    let start = stack.len().saturating_sub(n);

                    // Find max value index
                    let mut max_idx: Option<usize> = None;
                    for (i, v) in stack[start..].iter().enumerate() {
                        if !v.is_null() {
                            match max_idx {
                                None => max_idx = Some(start + i),
                                Some(mi) => {
                                    if let Some(std::cmp::Ordering::Greater) =
                                        v.partial_cmp(&stack[mi])
                                    {
                                        max_idx = Some(start + i);
                                    }
                                }
                            }
                        }
                    }

                    let result = if let Some(idx) = max_idx {
                        let last = stack.len() - 1;
                        stack.swap(idx, last);
                        stack.pop().expect("stack underflow in GREATEST")
                    } else {
                        Cow::Borrowed(&NULL_VALUE)
                    };
                    stack.truncate(start);
                    stack.push(result);
                    pc += 1;
                }

                // LEAST - return minimum non-null value
                Op::Least(n) => {
                    let n = *n as usize;
                    let start = stack.len().saturating_sub(n);

                    // Find min value index
                    let mut min_idx: Option<usize> = None;
                    for (i, v) in stack[start..].iter().enumerate() {
                        if !v.is_null() {
                            match min_idx {
                                None => min_idx = Some(start + i),
                                Some(mi) => {
                                    if let Some(std::cmp::Ordering::Less) =
                                        v.partial_cmp(&stack[mi])
                                    {
                                        min_idx = Some(start + i);
                                    }
                                }
                            }
                        }
                    }

                    let result = if let Some(idx) = min_idx {
                        let last = stack.len() - 1;
                        stack.swap(idx, last);
                        stack.pop().expect("stack underflow in LEAST")
                    } else {
                        Cow::Borrowed(&NULL_VALUE)
                    };
                    stack.truncate(start);
                    stack.push(result);
                    pc += 1;
                }

                // JUMP/CONTROL FLOW for short-circuit evaluation (used by COALESCE)
                Op::JumpIfNotNull(target) => {
                    if let Some(top) = stack.last() {
                        if !top.is_null() {
                            pc = *target as usize;
                            continue;
                        }
                    }
                    pc += 1;
                }

                Op::Pop => {
                    // Use truncate instead of pop to drop in-place without copying value out
                    let new_len = stack.len().saturating_sub(1);
                    stack.truncate(new_len);
                    pc += 1;
                }

                Op::Jump(target) => {
                    pc = *target as usize;
                }

                Op::JumpIfTrue(target) => {
                    if let Some(top) = stack.last() {
                        if matches!(&**top, Value::Boolean(true)) {
                            pc = *target as usize;
                            continue;
                        }
                    }
                    pc += 1;
                }

                Op::JumpIfFalse(target) => {
                    if let Some(top) = stack.last() {
                        match &**top {
                            Value::Boolean(false) | Value::Null(_) => {
                                pc = *target as usize;
                                continue;
                            }
                            _ => {}
                        }
                    }
                    pc += 1;
                }

                Op::JumpIfNull(target) => {
                    if let Some(top) = stack.last() {
                        if top.is_null() {
                            pc = *target as usize;
                            continue;
                        }
                    }
                    pc += 1;
                }

                Op::PopJumpIfFalse(target) => {
                    let v = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    match &*v {
                        Value::Boolean(false) | Value::Null(_) => {
                            pc = *target as usize;
                        }
                        _ => pc += 1,
                    }
                }

                Op::PopJumpIfTrue(target) => {
                    let v = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    match &*v {
                        Value::Boolean(true) => {
                            pc = *target as usize;
                        }
                        _ => pc += 1,
                    }
                }

                Op::Nop => {
                    pc += 1;
                }

                // STRING CONCATENATION
                Op::Concat => {
                    let b = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let a = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    let result = if a.is_null() || b.is_null() {
                        Value::Null(DataType::Text)
                    } else {
                        // Fast path: both are Text
                        match (&*a, &*b) {
                            (Value::Text(a_str), Value::Text(b_str)) => {
                                // Use optimized concat - handles inline and heap efficiently
                                Value::Text(SmartString::concat(a_str, b_str))
                            }
                            (Value::Text(a_str), _) => {
                                // Use Arc to avoid shrink_to_fit
                                use std::fmt::Write;
                                let mut s = String::with_capacity(a_str.len() + 32);
                                s.push_str(a_str);
                                let _ = write!(s, "{}", *b);
                                Value::Text(SmartString::from_string_shared(s))
                            }
                            (_, Value::Text(b_str)) => {
                                // Use Arc to avoid shrink_to_fit
                                use std::fmt::Write;
                                let mut s = String::with_capacity(32 + b_str.len());
                                let _ = write!(s, "{}", *a);
                                s.push_str(b_str);
                                Value::Text(SmartString::from_string_shared(s))
                            }
                            _ => {
                                // Use Arc to avoid shrink_to_fit
                                use std::fmt::Write;
                                let mut s = String::with_capacity(64);
                                let _ = write!(s, "{}{}", *a, *b);
                                Value::Text(SmartString::from_string_shared(s))
                            }
                        }
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                // Multi-value string concatenation (optimized for chained ||)
                Op::ConcatN(n) => {
                    let n = *n as usize;
                    let start = stack.len().saturating_sub(n);

                    // Single pass: check for NULL and calculate total length
                    let mut total_len = 0usize;
                    let mut has_null = false;
                    let mut all_text = true;

                    for v in &stack[start..] {
                        match &**v {
                            Value::Null(_) => {
                                has_null = true;
                                break;
                            }
                            Value::Text(s) => total_len += s.len(),
                            _ => {
                                all_text = false;
                                total_len += 32;
                            }
                        }
                    }

                    if has_null {
                        stack.truncate(start);
                        stack.push(Cow::Owned(Value::Null(DataType::Text)));
                        pc += 1;
                        continue;
                    }

                    // Build result - optimize for inline vs heap
                    let result = if all_text && total_len <= 22 {
                        // Fast path: build directly into inline SmartString (no heap allocation)
                        let mut data = [0u8; 22];
                        let mut pos = 0;
                        for v in stack.drain(start..) {
                            if let Value::Text(text) = &*v {
                                let bytes = text.as_bytes();
                                data[pos..pos + bytes.len()].copy_from_slice(bytes);
                                pos += bytes.len();
                            }
                        }
                        // SAFETY: SmartString only stores valid UTF-8
                        SmartString::Inline {
                            len: total_len as u8,
                            data,
                        }
                    } else if all_text {
                        // Heap path: exact capacity, into_boxed_str is O(1) when len == capacity
                        let mut s = String::with_capacity(total_len);
                        for v in stack.drain(start..) {
                            if let Value::Text(text) = &*v {
                                s.push_str(text);
                            }
                        }
                        // len == capacity, so into_boxed_str is O(1)
                        SmartString::from_string(s)
                    } else {
                        // Mixed types: capacity is estimate, use Arc to avoid shrink_to_fit
                        let mut s = String::with_capacity(total_len);
                        for v in stack.drain(start..) {
                            match &*v {
                                Value::Text(text) => s.push_str(text),
                                other => {
                                    use std::fmt::Write;
                                    let _ = write!(s, "{}", other);
                                }
                            }
                        }
                        SmartString::from_string_shared(s)
                    };
                    stack.push(Cow::Owned(Value::Text(result)));
                    pc += 1;
                }

                // CASE expression operations
                Op::CaseStart => {
                    // Marker only, no operation
                    pc += 1;
                }

                Op::CaseWhen(next_branch) => {
                    let cond = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
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
                    let when_val = stack.pop().unwrap_or(Cow::Borrowed(&NULL_VALUE));
                    // Use reference to avoid clone - case_val stays on stack for next comparison
                    let result = match stack.last() {
                        Some(case_val) if !case_val.is_null() && !when_val.is_null() => {
                            Value::Boolean(**case_val == *when_val)
                        }
                        _ => Value::Boolean(false),
                    };
                    stack.push(Cow::Owned(result));
                    pc += 1;
                }

                Op::Return => break,
                Op::ReturnTrue => {
                    return Ok(Value::Boolean(true));
                }
                Op::ReturnFalse => {
                    return Ok(Value::Boolean(false));
                }
                Op::ReturnNull(dt) => {
                    return Ok(Value::Null(*dt));
                }

                // For any unhandled operation, fall back to the regular execute
                _ => {
                    return self.execute(program, ctx);
                }
            }
        }

        // Return top of stack or NULL
        Ok(stack
            .pop()
            .map(Cow::into_owned)
            .unwrap_or_else(Value::null_unknown))
    }

    /// Execute and return boolean result (for WHERE clauses)
    ///
    /// This method is optimized for common filter patterns, avoiding
    /// the full VM loop overhead for simple comparisons.
    #[inline]
    pub fn execute_bool(&mut self, program: &Program, ctx: &ExecuteContext) -> bool {
        let ops = program.ops();

        // Fast path: Single comparison + Return (most common filter)
        // Pattern: [GtColumnConst(idx, val), Return]
        if ops.len() == 2 {
            if let Op::Return = &ops[1] {
                return Self::eval_single_op_bool(&ops[0], ctx);
            }
        }

        // Fast path: Two comparisons with AND (range filter)
        // Pattern: [Compare1, And(_), Compare2, AndFinalize, Return]
        if ops.len() == 5 {
            if let (Op::And(_), Op::AndFinalize, Op::Return) = (&ops[1], &ops[3], &ops[4]) {
                let a = Self::eval_single_op_tribool(&ops[0], ctx);
                // Short-circuit: if first is false, result is false
                if a == Some(false) {
                    return false;
                }
                let b = Self::eval_single_op_tribool(&ops[2], ctx);
                // AND logic: true && true = true, anything else = false (for boolean result)
                return a == Some(true) && b == Some(true);
            }
            // Pattern: [Compare1, Or(_), Compare2, OrFinalize, Return]
            if let (Op::Or(_), Op::OrFinalize, Op::Return) = (&ops[1], &ops[3], &ops[4]) {
                let a = Self::eval_single_op_tribool(&ops[0], ctx);
                // Short-circuit: if first is true, result is true
                if a == Some(true) {
                    return true;
                }
                let b = Self::eval_single_op_tribool(&ops[2], ctx);
                // OR logic: true || anything = true, false || true = true
                return a == Some(true) || b == Some(true);
            }
        }

        // General path: full VM execution with Cow-based stack (avoids cloning)
        match self.execute_cow(program, ctx) {
            Ok(Value::Boolean(b)) => b,
            Ok(Value::Integer(i)) => i != 0,
            Ok(Value::Null(_)) => false,
            _ => false,
        }
    }

    /// Evaluate a single comparison op and return bool (for fast path)
    #[inline]
    fn eval_single_op_bool(op: &Op, ctx: &ExecuteContext) -> bool {
        match op {
            Op::GtColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => *v > *threshold,
                Some(Value::Float(v)) => *v > *threshold as f64,
                _ => false,
            },
            Op::GtColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => *v > *threshold,
                Some(Value::Integer(v)) => (*v as f64) > *threshold,
                _ => false,
            },
            Op::LtColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => *v < *threshold,
                Some(Value::Float(v)) => *v < *threshold as f64,
                _ => false,
            },
            Op::LtColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => *v < *threshold,
                Some(Value::Integer(v)) => (*v as f64) < *threshold,
                _ => false,
            },
            Op::GeColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => *v >= *threshold,
                Some(Value::Float(v)) => *v >= *threshold as f64,
                _ => false,
            },
            Op::GeColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => *v >= *threshold,
                Some(Value::Integer(v)) => (*v as f64) >= *threshold,
                _ => false,
            },
            Op::LeColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => *v <= *threshold,
                Some(Value::Float(v)) => *v <= *threshold as f64,
                _ => false,
            },
            Op::LeColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => *v <= *threshold,
                Some(Value::Integer(v)) => (*v as f64) <= *threshold,
                _ => false,
            },
            Op::EqColumnConst(idx, Value::Integer(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => *v == *val,
                Some(Value::Float(v)) => *v == *val as f64,
                _ => false,
            },
            Op::EqColumnConst(idx, Value::Float(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => *v == *val,
                Some(Value::Integer(v)) => (*v as f64) == *val,
                _ => false,
            },
            Op::NeColumnConst(idx, Value::Integer(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => *v != *val,
                Some(Value::Float(v)) => *v != *val as f64,
                Some(Value::Null(_)) => false,
                _ => true,
            },
            Op::NeColumnConst(idx, Value::Float(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => *v != *val,
                Some(Value::Integer(v)) => (*v as f64) != *val,
                Some(Value::Null(_)) => false,
                _ => true,
            },
            Op::IsNullColumn(idx) => ctx.row.get(*idx as usize).is_some_and(|v| v.is_null()),
            Op::IsNotNullColumn(idx) => ctx.row.get(*idx as usize).is_some_and(|v| !v.is_null()),
            Op::BetweenColumnConst(idx, Value::Integer(lo), Value::Integer(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Integer(v)) => *v >= *lo && *v <= *hi,
                    Some(Value::Float(v)) => *v >= *lo as f64 && *v <= *hi as f64,
                    _ => false,
                }
            }
            Op::BetweenColumnConst(idx, Value::Float(lo), Value::Float(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => *v >= *lo && *v <= *hi,
                    Some(Value::Integer(v)) => (*v as f64) >= *lo && (*v as f64) <= *hi,
                    _ => false,
                }
            }
            Op::BetweenColumnConst(idx, Value::Integer(lo), Value::Float(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => *v >= *lo as f64 && *v <= *hi,
                    Some(Value::Integer(v)) => (*v as f64) >= *lo as f64 && (*v as f64) <= *hi,
                    _ => false,
                }
            }
            Op::BetweenColumnConst(idx, Value::Float(lo), Value::Integer(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => *v >= *lo && *v <= *hi as f64,
                    Some(Value::Integer(v)) => (*v as f64) >= *lo && (*v as f64) <= *hi as f64,
                    _ => false,
                }
            }
            Op::InSetColumn(idx, set, has_null) => {
                match ctx.row.get(*idx as usize) {
                    Some(v) if v.is_null() => false, // NULL IN set -> NULL -> false in bool context
                    Some(v) if set.contains(v) => true,
                    Some(_) if *has_null => false, // val not in set, but set has NULL -> NULL -> false
                    Some(_) => false,
                    None => false,
                }
            }
            // Handle LIKE pattern matching (e.g., fruit LIKE 'a%')
            Op::LikeColumn(idx, pattern, case_insensitive) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Text(s)) => pattern.matches(s, *case_insensitive),
                    _ => false, // NULL or non-text -> false
                }
            }
            // For other ops, fall back to tribool and convert
            _ => Self::eval_single_op_tribool(op, ctx) == Some(true),
        }
    }

    /// Evaluate a single comparison op and return Option<bool> (tribool)
    /// None = NULL, Some(true) = true, Some(false) = false
    #[inline]
    fn eval_single_op_tribool(op: &Op, ctx: &ExecuteContext) -> Option<bool> {
        match op {
            Op::GtColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v > *threshold),
                Some(Value::Float(v)) => Some(*v > *threshold as f64),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::GtColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(*v > *threshold),
                Some(Value::Integer(v)) => Some((*v as f64) > *threshold),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::LtColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v < *threshold),
                Some(Value::Float(v)) => Some(*v < *threshold as f64),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::LtColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(*v < *threshold),
                Some(Value::Integer(v)) => Some((*v as f64) < *threshold),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::GeColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v >= *threshold),
                Some(Value::Float(v)) => Some(*v >= *threshold as f64),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::GeColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(*v >= *threshold),
                Some(Value::Integer(v)) => Some((*v as f64) >= *threshold),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::LeColumnConst(idx, Value::Integer(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v <= *threshold),
                Some(Value::Float(v)) => Some(*v <= *threshold as f64),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::LeColumnConst(idx, Value::Float(threshold)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(*v <= *threshold),
                Some(Value::Integer(v)) => Some((*v as f64) <= *threshold),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::EqColumnConst(idx, Value::Integer(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v == *val),
                Some(Value::Float(v)) => Some(*v == *val as f64),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::EqColumnConst(idx, Value::Float(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(*v == *val),
                Some(Value::Integer(v)) => Some((*v as f64) == *val),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::NeColumnConst(idx, Value::Integer(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Integer(v)) => Some(*v != *val),
                Some(Value::Float(v)) => Some(*v != *val as f64),
                Some(Value::Null(_)) | None => None,
                _ => Some(true),
            },
            Op::NeColumnConst(idx, Value::Float(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Float(v)) => Some(*v != *val),
                Some(Value::Integer(v)) => Some((*v as f64) != *val),
                Some(Value::Null(_)) | None => None,
                _ => Some(true),
            },
            Op::IsNullColumn(idx) => Some(ctx.row.get(*idx as usize).is_some_and(|v| v.is_null())),
            Op::IsNotNullColumn(idx) => {
                Some(ctx.row.get(*idx as usize).is_some_and(|v| !v.is_null()))
            }
            Op::BetweenColumnConst(idx, Value::Integer(lo), Value::Integer(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Integer(v)) => Some(*v >= *lo && *v <= *hi),
                    Some(Value::Float(v)) => Some(*v >= *lo as f64 && *v <= *hi as f64),
                    Some(Value::Null(_)) | None => None,
                    _ => Some(false),
                }
            }
            Op::BetweenColumnConst(idx, Value::Float(lo), Value::Float(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => Some(*v >= *lo && *v <= *hi),
                    Some(Value::Integer(v)) => Some((*v as f64) >= *lo && (*v as f64) <= *hi),
                    Some(Value::Null(_)) | None => None,
                    _ => Some(false),
                }
            }
            Op::BetweenColumnConst(idx, Value::Integer(lo), Value::Float(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => Some(*v >= *lo as f64 && *v <= *hi),
                    Some(Value::Integer(v)) => {
                        Some((*v as f64) >= *lo as f64 && (*v as f64) <= *hi)
                    }
                    Some(Value::Null(_)) | None => None,
                    _ => Some(false),
                }
            }
            Op::BetweenColumnConst(idx, Value::Float(lo), Value::Integer(hi)) => {
                match ctx.row.get(*idx as usize) {
                    Some(Value::Float(v)) => Some(*v >= *lo && *v <= *hi as f64),
                    Some(Value::Integer(v)) => {
                        Some((*v as f64) >= *lo && (*v as f64) <= *hi as f64)
                    }
                    Some(Value::Null(_)) | None => None,
                    _ => Some(false),
                }
            }
            Op::InSetColumn(idx, set, has_null) => {
                match ctx.row.get(*idx as usize) {
                    Some(v) if v.is_null() => None, // NULL IN set -> NULL
                    Some(v) if set.contains(v) => Some(true),
                    Some(_) if *has_null => None, // val not in set, but set has NULL -> NULL
                    Some(_) => Some(false),
                    None => None,
                }
            }
            // Handle boolean constants (from processed subqueries like NOT EXISTS)
            Op::LoadConst(Value::Boolean(b)) => Some(*b),
            Op::LoadConst(Value::Integer(i)) => Some(*i != 0),
            Op::LoadConst(Value::Null(_)) => None,
            // Handle text comparisons (e.g., country = 'USA')
            Op::EqColumnConst(idx, Value::Text(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Text(v)) => Some(v == val),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
            Op::NeColumnConst(idx, Value::Text(val)) => match ctx.row.get(*idx as usize) {
                Some(Value::Text(v)) => Some(v != val),
                Some(Value::Null(_)) | None => None,
                _ => Some(true),
            },
            // Handle LIKE pattern matching (e.g., fruit LIKE 'a%')
            Op::LikeColumn(idx, pattern, case_insensitive) => match ctx.row.get(*idx as usize) {
                Some(Value::Text(s)) => Some(pattern.matches(s, *case_insensitive)),
                Some(Value::Null(_)) | None => None,
                _ => Some(false),
            },
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
    fn to_tribool(v: &Value) -> Option<bool> {
        match v {
            Value::Boolean(b) => Some(*b),
            Value::Integer(i) => Some(*i != 0),
            Value::Null(_) => None,
            _ => Some(true),
        }
    }

    #[inline]
    fn compare_values(&self, a: &Value, b: &Value, expected: std::cmp::Ordering) -> Value {
        match a.partial_cmp(b) {
            Some(ord) if ord == expected => Value::Boolean(true),
            Some(_) => Value::Boolean(false),
            None => Value::Null(DataType::Boolean),
        }
    }

    // =========================================================================
    // INLINE COMPARISON HELPERS
    // These avoid the overhead of partial_cmp → compare → compare_same_type
    // for the common case of integer/float column vs constant comparisons.
    // =========================================================================

    /// Inline greater-than comparison for integer constant
    #[inline(always)]
    fn gt_int(col_val: &Value, threshold: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v > threshold),
            Value::Float(v) => Value::Boolean(*v > threshold as f64),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline greater-than comparison for float constant
    #[inline(always)]
    fn gt_float(col_val: &Value, threshold: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(*v > threshold),
            Value::Integer(v) => Value::Boolean((*v as f64) > threshold),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline less-than comparison for integer constant
    #[inline(always)]
    fn lt_int(col_val: &Value, threshold: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v < threshold),
            Value::Float(v) => Value::Boolean(*v < (threshold as f64)),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline less-than comparison for float constant
    #[inline(always)]
    fn lt_float(col_val: &Value, threshold: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(*v < threshold),
            Value::Integer(v) => Value::Boolean((*v as f64) < threshold),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline greater-or-equal comparison for integer constant
    #[inline(always)]
    fn ge_int(col_val: &Value, threshold: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v >= threshold),
            Value::Float(v) => Value::Boolean(*v >= threshold as f64),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline greater-or-equal comparison for float constant
    #[inline(always)]
    fn ge_float(col_val: &Value, threshold: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(*v >= threshold),
            Value::Integer(v) => Value::Boolean((*v as f64) >= threshold),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline less-or-equal comparison for integer constant
    #[inline(always)]
    fn le_int(col_val: &Value, threshold: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v <= threshold),
            Value::Float(v) => Value::Boolean(*v <= threshold as f64),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline less-or-equal comparison for float constant
    #[inline(always)]
    fn le_float(col_val: &Value, threshold: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(*v <= threshold),
            Value::Integer(v) => Value::Boolean((*v as f64) <= threshold),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline equality comparison for integer constant
    #[inline(always)]
    fn eq_int(col_val: &Value, val: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v == val),
            Value::Float(v) => Value::Boolean(*v == val as f64),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline equality comparison for float constant
    #[inline(always)]
    fn eq_float(col_val: &Value, val: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(*v == val),
            Value::Integer(v) => Value::Boolean((*v as f64) == val),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline not-equal comparison for integer constant
    #[inline(always)]
    fn ne_int(col_val: &Value, val: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v != val),
            Value::Float(v) => Value::Boolean(*v != val as f64),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(true),
        }
    }

    /// Inline not-equal comparison for float constant
    #[inline(always)]
    fn ne_float(col_val: &Value, val: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(*v != val),
            Value::Integer(v) => Value::Boolean((*v as f64) != val),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(true),
        }
    }

    /// Inline BETWEEN check for integer bounds
    #[inline(always)]
    fn between_int(col_val: &Value, low: i64, high: i64) -> Value {
        match col_val {
            Value::Integer(v) => Value::Boolean(*v >= low && *v <= high),
            Value::Float(v) => Value::Boolean(*v >= low as f64 && *v <= high as f64),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    /// Inline BETWEEN check for float bounds
    #[inline(always)]
    fn between_float(col_val: &Value, low: f64, high: f64) -> Value {
        match col_val {
            Value::Float(v) => Value::Boolean(*v >= low && *v <= high),
            Value::Integer(v) => Value::Boolean((*v as f64) >= low && (*v as f64) <= high),
            Value::Null(_) => Value::Null(DataType::Boolean),
            _ => Value::Boolean(false),
        }
    }

    #[inline]
    fn arithmetic_op<FF>(a: &Value, b: &Value, int_op: ArithmeticOp, float_op: FF) -> Result<Value>
    where
        FF: Fn(f64, f64) -> f64,
    {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => {
                // Use checked operations to detect overflow and return an error
                let result = match int_op {
                    ArithmeticOp::Add => x.checked_add(*y),
                    ArithmeticOp::Sub => x.checked_sub(*y),
                    ArithmeticOp::Mul => x.checked_mul(*y),
                    ArithmeticOp::Div => {
                        if *y == 0 {
                            return Ok(Value::Null(DataType::Integer));
                        }
                        x.checked_div(*y)
                    }
                    ArithmeticOp::Mod => {
                        if *y == 0 {
                            return Ok(Value::Null(DataType::Integer));
                        }
                        x.checked_rem(*y)
                    }
                };
                match result {
                    Some(r) => Ok(Value::Integer(r)),
                    None => Err(crate::core::Error::Type(format!(
                        "Integer overflow in arithmetic operation: {} and {}",
                        x, y
                    ))),
                }
            }
            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(float_op(*x, *y))),
            (Value::Integer(x), Value::Float(y)) => Ok(Value::Float(float_op(*x as f64, *y))),
            (Value::Float(x), Value::Integer(y)) => Ok(Value::Float(float_op(*x, *y as f64))),
            _ if a.is_null() || b.is_null() => Ok(Value::Null(DataType::Float)),
            _ => Ok(Value::Null(DataType::Null)),
        }
    }

    #[inline]
    fn div_op(a: &Value, b: &Value) -> Value {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) if *y != 0 => Value::Integer(x / y),
            (Value::Float(x), Value::Float(y)) if *y != 0.0 => Value::Float(x / y),
            (Value::Integer(x), Value::Float(y)) if *y != 0.0 => Value::Float(*x as f64 / y),
            (Value::Float(x), Value::Integer(y)) if *y != 0 => Value::Float(x / *y as f64),
            _ if a.is_null() || b.is_null() => Value::Null(DataType::Float),
            _ => Value::Null(DataType::Null), // Division by zero returns NULL
        }
    }

    #[inline]
    fn mod_op(a: &Value, b: &Value) -> Value {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) if *y != 0 => Value::Integer(x % y),
            (Value::Float(x), Value::Float(y)) if *y != 0.0 => Value::Float(x % y),
            (Value::Integer(x), Value::Float(y)) if *y != 0.0 => Value::Float(*x as f64 % y),
            (Value::Float(x), Value::Integer(y)) if *y != 0 => Value::Float(x % *y as f64),
            _ if a.is_null() || b.is_null() => Value::Null(DataType::Float),
            _ => Value::Null(DataType::Null),
        }
    }

    /// JSON access helper
    /// If as_text is true, returns TEXT; otherwise returns JSON
    fn json_access(&self, json_val: &Value, key: &Value, as_text: bool) -> Value {
        use serde_json;

        // Get the JSON string
        let json_str = match json_val {
            Value::Json(s) => s.as_ref(),
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
                    Value::Json(Arc::from(v.to_string().as_str()))
                }
            }
            None => Value::Null(if as_text {
                DataType::Text
            } else {
                DataType::Json
            }),
        }
    }

    /// Add or subtract interval from timestamp
    fn timestamp_add_interval(&self, ts: &Value, interval: &Value, add: bool) -> Value {
        let timestamp = match ts {
            Value::Timestamp(t) => *t,
            Value::Null(_) => return Value::Null(DataType::Timestamp),
            _ => return Value::Null(DataType::Timestamp),
        };

        let interval_str = match interval {
            Value::Text(s) => s.as_ref(),
            Value::Null(_) => return Value::Null(DataType::Timestamp),
            _ => return Value::Null(DataType::Timestamp),
        };

        // Parse interval string
        // Formats: "1 day", "2 hours", "30 minutes", "1 year", "1 month", etc.
        let duration = self.parse_interval(interval_str);

        match duration {
            Some(d) => {
                if add {
                    Value::Timestamp(timestamp + d)
                } else {
                    Value::Timestamp(timestamp - d)
                }
            }
            None => Value::Null(DataType::Timestamp),
        }
    }

    /// Parse interval string into Duration
    fn parse_interval(&self, s: &str) -> Option<chrono::Duration> {
        let s = s.trim();
        let parts: Vec<&str> = s.split_whitespace().collect();

        if parts.len() < 2 {
            // Try parsing as just a number (days)
            if let Ok(n) = s.parse::<i64>() {
                return Some(chrono::Duration::days(n));
            }
            return None;
        }

        let value: i64 = parts[0].parse().ok()?;
        let unit = parts[1];

        // Case-insensitive unit matching without allocation
        // Handle both singular and plural forms
        if unit.eq_ignore_ascii_case("year") || unit.eq_ignore_ascii_case("years") {
            Some(chrono::Duration::days(value * 365))
        } else if unit.eq_ignore_ascii_case("month") || unit.eq_ignore_ascii_case("months") {
            Some(chrono::Duration::days(value * 30))
        } else if unit.eq_ignore_ascii_case("week") || unit.eq_ignore_ascii_case("weeks") {
            Some(chrono::Duration::weeks(value))
        } else if unit.eq_ignore_ascii_case("day") || unit.eq_ignore_ascii_case("days") {
            Some(chrono::Duration::days(value))
        } else if unit.eq_ignore_ascii_case("hour") || unit.eq_ignore_ascii_case("hours") {
            Some(chrono::Duration::hours(value))
        } else if unit.eq_ignore_ascii_case("minute")
            || unit.eq_ignore_ascii_case("minutes")
            || unit.eq_ignore_ascii_case("min")
        {
            Some(chrono::Duration::minutes(value))
        } else if unit.eq_ignore_ascii_case("second")
            || unit.eq_ignore_ascii_case("seconds")
            || unit.eq_ignore_ascii_case("sec")
        {
            Some(chrono::Duration::seconds(value))
        } else if unit.eq_ignore_ascii_case("millisecond")
            || unit.eq_ignore_ascii_case("milliseconds")
            || unit.eq_ignore_ascii_case("ms")
        {
            Some(chrono::Duration::milliseconds(value))
        } else if unit.eq_ignore_ascii_case("microsecond")
            || unit.eq_ignore_ascii_case("microseconds")
            || unit.eq_ignore_ascii_case("us")
        {
            Some(chrono::Duration::microseconds(value))
        } else {
            None
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::CompactArc;
    use crate::Row;
    use ahash::AHashSet;

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
        let set: AHashSet<Value> = [Value::Integer(1), Value::Integer(2), Value::Integer(3)]
            .into_iter()
            .collect();

        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::InSet(CompactArc::new(set), false),
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
    fn test_context_with_named_params() {
        let row = Row::from_values(vec![Value::Integer(1)]);
        let mut named = FxHashMap::default();
        named.insert("name".to_string(), Value::Text("value".into()));
        let ctx = ExecuteContext::new(&row).with_named_params(&named);
        assert!(ctx.named_params.is_some());
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
        let mut outer: FxHashMap<Arc<str>, Value> = FxHashMap::default();
        outer.insert(Arc::from("outer_col"), Value::Integer(42));
        let ctx = ExecuteContext::new(&row).with_outer_row(&outer);
        assert!(ctx.outer_row.is_some());
    }

    // =========================================================================
    // ExprVM creation tests
    // =========================================================================

    #[test]
    fn test_vm_new() {
        let vm = ExprVM::new();
        assert_eq!(vm.stack.len(), 0);
    }

    #[test]
    fn test_vm_with_capacity() {
        let vm = ExprVM::with_capacity(16);
        assert!(vm.stack.capacity() >= 16);
    }

    #[test]
    fn test_vm_default() {
        let vm = ExprVM::default();
        assert_eq!(vm.stack.len(), 0);
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
    fn test_load_named_param() {
        let mut vm = ExprVM::new();
        let program = Program::new(vec![Op::LoadNamedParam(Arc::from("myvar")), Op::Return]);
        let row = Row::new();
        let mut named = FxHashMap::default();
        named.insert("myvar".to_string(), Value::Integer(999));
        let ctx = ExecuteContext::new(&row).with_named_params(&named);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Integer(999));
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
            Op::LoadOuterColumn(Arc::from("outer_val")),
            Op::Return,
        ]);
        let row = Row::new();
        let mut outer: FxHashMap<Arc<str>, Value> = FxHashMap::default();
        outer.insert(Arc::from("outer_val"), Value::Integer(100));
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
        use super::super::ops::CompiledPattern;

        let mut vm = ExprVM::new();
        let pattern = CompiledPattern::compile("%world%", false);
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::Like(Arc::new(pattern), false),
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
        use super::super::ops::CompiledPattern;

        let mut vm = ExprVM::new();
        let pattern = CompiledPattern::compile_glob("*.txt");
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
        let set: AHashSet<Value> = [Value::Integer(1), Value::Integer(2), Value::Integer(3)]
            .into_iter()
            .collect();

        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::NotInSet(CompactArc::new(set), false),
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
        let set: AHashSet<Value> = [Value::Integer(1), Value::Integer(2)].into_iter().collect();

        // has_null = true means the set conceptually contains NULL
        let program = Program::new(vec![
            Op::LoadColumn(0),
            Op::InSet(CompactArc::new(set), true),
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
        let set: AHashSet<Value> = [Value::Integer(1), Value::Integer(2), Value::Integer(3)]
            .into_iter()
            .collect();

        let program = Program::new(vec![
            Op::InSetColumn(0, CompactArc::new(set), false),
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
        use super::super::ops::CompiledPattern;

        let mut vm = ExprVM::new();
        let pattern = CompiledPattern::compile("test%", false);
        let program = Program::new(vec![
            Op::LikeColumn(0, Arc::new(pattern), false),
            Op::Return,
        ]);

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
}
