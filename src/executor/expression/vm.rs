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

use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::ops::{CompareOp, Op};
use super::program::Program;
use crate::core::{DataType, Result, Value};

/// Stack capacity for inline storage (avoids heap allocation for simple expressions)
/// Most expressions need 4-8 stack slots, so 8 covers the common case.
const STACK_INLINE_CAPACITY: usize = 8;

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
    pub row: &'a [Value],

    /// Second row for joins (optional)
    pub row2: Option<&'a [Value]>,

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
    pub fn new(row: &'a [Value]) -> Self {
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

    /// Create context for join evaluation
    pub fn for_join(row1: &'a [Value], row2: &'a [Value]) -> Self {
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
    /// Uses SmallVec to avoid heap allocation for simple expressions (stack depth <= 8)
    stack: SmallVec<[Value; STACK_INLINE_CAPACITY]>,

    /// Reusable buffer for function arguments (avoids allocation per call)
    args_buffer: Vec<Value>,
}

impl ExprVM {
    /// Create a new VM with default stack capacity
    /// Uses inline storage for up to 8 values (no heap allocation for simple expressions)
    pub fn new() -> Self {
        Self {
            stack: SmallVec::new(),
            args_buffer: Vec::with_capacity(ARGS_BUFFER_CAPACITY),
        }
    }

    /// Create a VM with specific stack capacity
    /// If capacity > 8, will spill to heap when needed
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            stack: SmallVec::with_capacity(capacity),
            args_buffer: Vec::with_capacity(ARGS_BUFFER_CAPACITY),
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
                    let result = self.compare_values(&a, &b, std::cmp::Ordering::Less);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Le => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match a.partial_cmp(&b) {
                        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => {
                            Value::Boolean(true)
                        }
                        Some(std::cmp::Ordering::Greater) => Value::Boolean(false),
                        None => Value::Null(DataType::Boolean),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Gt => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = self.compare_values(&a, &b, std::cmp::Ordering::Greater);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Ge => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = match a.partial_cmp(&b) {
                        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal) => {
                            Value::Boolean(true)
                        }
                        Some(std::cmp::Ordering::Less) => Value::Boolean(false),
                        None => Value::Null(DataType::Boolean),
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
                    let result = if col_val.is_null() || val.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(col_val == val)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::NeColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    let result = if col_val.is_null() || val.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(col_val != val)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::LtColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    let result = self.compare_values(col_val, val, std::cmp::Ordering::Less);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::LeColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    let result = match col_val.partial_cmp(val) {
                        Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => {
                            Value::Boolean(true)
                        }
                        Some(std::cmp::Ordering::Greater) => Value::Boolean(false),
                        None => Value::Null(DataType::Boolean),
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::GtColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    let result = self.compare_values(col_val, val, std::cmp::Ordering::Greater);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::GeColumnConst(idx, val) => {
                    let col_val = ctx
                        .row
                        .get(*idx as usize)
                        .unwrap_or(&Value::Null(DataType::Null));
                    let result = match col_val.partial_cmp(val) {
                        Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal) => {
                            Value::Boolean(true)
                        }
                        Some(std::cmp::Ordering::Less) => Value::Boolean(false),
                        None => Value::Null(DataType::Boolean),
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
                    let result = if col_val.is_null() || low.is_null() || high.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(col_val >= low && col_val <= high)
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
                        (Value::Timestamp(t), Value::Text(interval)) => {
                            // Parse interval string
                            self.timestamp_add_interval(
                                &Value::Timestamp(*t),
                                &Value::Text(interval.clone()),
                                true,
                            )
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
                            Value::Text(Arc::from(
                                self.format_duration_as_interval(duration).as_str(),
                            ))
                        }
                        (Value::Timestamp(t), Value::Integer(days)) => {
                            Value::Timestamp(*t - chrono::Duration::days(*days))
                        }
                        (Value::Timestamp(t), Value::Text(interval)) => {
                            // Parse interval string
                            self.timestamp_add_interval(
                                &Value::Timestamp(*t),
                                &Value::Text(interval.clone()),
                                false,
                            )
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
                // BITWISE OPERATIONS
                // =============================================================
                Op::BitAnd => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = Self::bitwise_op(&a, &b, |x, y| x & y);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::BitOr => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = Self::bitwise_op(&a, &b, |x, y| x | y);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::BitXor => {
                    let b = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let a = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let result = Self::bitwise_op(&a, &b, |x, y| x ^ y);
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
                        // Convert values to strings for concatenation
                        let a_str = match &a {
                            Value::Text(s) => s.to_string(),
                            _ => format!("{}", a),
                        };
                        let b_str = match &b {
                            Value::Text(s) => s.to_string(),
                            _ => format!("{}", b),
                        };
                        let s = format!("{}{}", a_str, b_str);
                        Value::Text(Arc::from(s.as_str()))
                    };
                    self.stack.push(result);
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
                            Value::Text(Arc::from(
                                self.format_duration_as_interval(duration).as_str(),
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

                    let result = if val.is_null() || low.is_null() || high.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(val >= low && val <= high)
                    };
                    self.stack.push(result);
                    pc += 1;
                }

                Op::NotBetween => {
                    let high = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let low = self.stack.pop().unwrap_or_else(Value::null_unknown);
                    let val = self.stack.pop().unwrap_or_else(Value::null_unknown);

                    let result = if val.is_null() || low.is_null() || high.is_null() {
                        Value::Null(DataType::Boolean)
                    } else {
                        Value::Boolean(val < low || val > high)
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

                    // Reuse args_buffer to avoid allocation
                    self.args_buffer.clear();
                    self.args_buffer.extend(self.stack.drain(start..));

                    let result = self
                        .args_buffer
                        .drain(..)
                        .find(|v| !v.is_null())
                        .unwrap_or_else(Value::null_unknown);
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

                    // Reuse args_buffer to avoid allocation
                    self.args_buffer.clear();
                    self.args_buffer.extend(self.stack.drain(start..));

                    let result = self
                        .args_buffer
                        .drain(..)
                        .filter(|v| !v.is_null())
                        .max()
                        .unwrap_or_else(Value::null_unknown);
                    self.stack.push(result);
                    pc += 1;
                }

                Op::Least(n) => {
                    let n = *n as usize;
                    let start = self.stack.len().saturating_sub(n);

                    // Reuse args_buffer to avoid allocation
                    self.args_buffer.clear();
                    self.args_buffer.extend(self.stack.drain(start..));

                    let result = self
                        .args_buffer
                        .drain(..)
                        .filter(|v| !v.is_null())
                        .min()
                        .unwrap_or_else(Value::null_unknown);
                    self.stack.push(result);
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
                    let case_val = self
                        .stack
                        .last()
                        .cloned()
                        .unwrap_or_else(Value::null_unknown);
                    let result = if case_val.is_null() || when_val.is_null() {
                        Value::Boolean(false)
                    } else {
                        Value::Boolean(case_val == when_val)
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
                    self.stack.pop();
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

    /// Execute and return boolean result (for WHERE clauses)
    #[inline]
    pub fn execute_bool(&mut self, program: &Program, ctx: &ExecuteContext) -> bool {
        match self.execute(program, ctx) {
            Ok(Value::Boolean(b)) => b,
            Ok(Value::Integer(i)) => i != 0,
            Ok(Value::Null(_)) => false,
            _ => false,
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

    #[inline]
    fn bitwise_op<F>(a: &Value, b: &Value, op: F) -> Value
    where
        F: Fn(i64, i64) -> i64,
    {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => Value::Integer(op(*x, *y)),
            _ if a.is_null() || b.is_null() => Value::Null(DataType::Integer),
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
            Value::Text(k) => parsed.get(k.as_ref()),
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
                        serde_json::Value::String(s) => Value::Text(Arc::from(s.as_str())),
                        serde_json::Value::Null => Value::Null(DataType::Text),
                        other => Value::Text(Arc::from(other.to_string().as_str())),
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
        let s = s.trim().to_lowercase();
        let parts: Vec<&str> = s.split_whitespace().collect();

        if parts.len() < 2 {
            // Try parsing as just a number (days)
            if let Ok(n) = s.parse::<i64>() {
                return Some(chrono::Duration::days(n));
            }
            return None;
        }

        let value: i64 = parts[0].parse().ok()?;
        let unit = parts[1].trim_end_matches('s'); // Remove trailing 's'

        match unit {
            "year" => Some(chrono::Duration::days(value * 365)),
            "month" => Some(chrono::Duration::days(value * 30)),
            "week" => Some(chrono::Duration::weeks(value)),
            "day" => Some(chrono::Duration::days(value)),
            "hour" => Some(chrono::Duration::hours(value)),
            "minute" | "min" => Some(chrono::Duration::minutes(value)),
            "second" | "sec" => Some(chrono::Duration::seconds(value)),
            "millisecond" | "ms" => Some(chrono::Duration::milliseconds(value)),
            "microsecond" | "us" => Some(chrono::Duration::microseconds(value)),
            _ => None,
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
        let row = vec![Value::Integer(10)];
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        // Test with row where column 0 <= 5
        let row = vec![Value::Integer(3)];
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

        let row = vec![Value::Integer(10), Value::Integer(5)];
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = vec![Value::Integer(3), Value::Integer(5)];
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
            Op::InSet(Arc::new(set), false),
            Op::Return,
        ]);

        let row = vec![Value::Integer(2)];
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(true));

        let row = vec![Value::Integer(5)];
        let ctx = ExecuteContext::new(&row);
        assert_eq!(vm.execute(&program, &ctx).unwrap(), Value::Boolean(false));
    }
}
