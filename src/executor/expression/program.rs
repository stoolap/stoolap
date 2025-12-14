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

// Compiled Expression Program
//
// A Program is the compiled form of an AST Expression.
// It contains:
// - A sequence of operations (the "bytecode")
// - Metadata for efficient execution

use std::sync::Arc;

use super::ops::Op;
use crate::core::Value;

/// Constant value stored in the program
#[derive(Debug, Clone)]
pub enum Constant {
    /// A literal value
    Value(Value),
    /// A string (for patterns, column names, etc.)
    String(Arc<str>),
}

/// Compiled expression program
///
/// This is the executable form of an expression. It's designed to be:
/// - Cheap to clone (uses Arc internally for large data)
/// - Fast to execute (linear operation sequence)
/// - Self-contained (no external lookups needed)
#[derive(Clone)]
pub struct Program {
    /// The operation sequence
    ops: Vec<Op>,

    /// Maximum stack depth needed (for pre-allocation)
    max_stack_depth: usize,

    /// Whether this program needs outer row context (correlated subquery)
    needs_outer_context: bool,

    /// Whether this program needs second row (join evaluation)
    needs_second_row: bool,

    /// Whether this program contains subqueries
    has_subqueries: bool,

    /// Source expression string (for debugging)
    #[cfg(debug_assertions)]
    source: Option<String>,
}

impl Program {
    /// Create a new program from operations.
    /// Automatically applies peephole optimizations (instruction fusion).
    pub fn new(ops: Vec<Op>) -> Self {
        // Apply peephole optimizations
        let ops = Self::peephole_optimize(ops);

        let max_stack_depth = Self::compute_stack_depth(&ops);
        let needs_outer_context = ops.iter().any(|op| matches!(op, Op::LoadOuterColumn(_)));
        let needs_second_row = ops.iter().any(|op| matches!(op, Op::LoadColumn2(_)));
        let has_subqueries = ops.iter().any(|op| {
            matches!(
                op,
                Op::ExecScalarSubquery(_)
                    | Op::ExecExists(_)
                    | Op::ExecInSubquery(_)
                    | Op::ExecAll(_, _)
                    | Op::ExecAny(_, _)
            )
        });

        Self {
            ops,
            max_stack_depth,
            needs_outer_context,
            needs_second_row,
            has_subqueries,
            #[cfg(debug_assertions)]
            source: None,
        }
    }

    /// Create a new program without peephole optimization.
    /// Use this for testing or when optimization is not desired.
    pub fn new_unoptimized(ops: Vec<Op>) -> Self {
        let max_stack_depth = Self::compute_stack_depth(&ops);
        let needs_outer_context = ops.iter().any(|op| matches!(op, Op::LoadOuterColumn(_)));
        let needs_second_row = ops.iter().any(|op| matches!(op, Op::LoadColumn2(_)));
        let has_subqueries = ops.iter().any(|op| {
            matches!(
                op,
                Op::ExecScalarSubquery(_)
                    | Op::ExecExists(_)
                    | Op::ExecInSubquery(_)
                    | Op::ExecAll(_, _)
                    | Op::ExecAny(_, _)
            )
        });

        Self {
            ops,
            max_stack_depth,
            needs_outer_context,
            needs_second_row,
            has_subqueries,
            #[cfg(debug_assertions)]
            source: None,
        }
    }

    /// Create an empty program that returns NULL
    pub fn null() -> Self {
        Self::new(vec![Op::LoadNull(crate::core::DataType::Null), Op::Return])
    }

    /// Create a program that returns a constant value
    pub fn constant(value: Value) -> Self {
        Self::new(vec![Op::LoadConst(value), Op::Return])
    }

    /// Create a program that returns true
    pub fn always_true() -> Self {
        Self::new(vec![Op::ReturnTrue])
    }

    /// Create a program that returns false
    pub fn always_false() -> Self {
        Self::new(vec![Op::ReturnFalse])
    }

    /// Get the operations
    #[inline]
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Get the maximum stack depth needed
    #[inline]
    pub fn max_stack_depth(&self) -> usize {
        self.max_stack_depth
    }

    /// Check if this program needs outer row context
    #[inline]
    pub fn needs_outer_context(&self) -> bool {
        self.needs_outer_context
    }

    /// Check if this program needs a second row (for joins)
    #[inline]
    pub fn needs_second_row(&self) -> bool {
        self.needs_second_row
    }

    /// Check if this program contains subqueries
    #[inline]
    pub fn has_subqueries(&self) -> bool {
        self.has_subqueries
    }

    /// Get the number of operations
    #[inline]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if program is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Set source string for debugging
    #[cfg(debug_assertions)]
    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Get source string
    #[cfg(debug_assertions)]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Compute the maximum stack depth needed for a sequence of operations
    fn compute_stack_depth(ops: &[Op]) -> usize {
        let mut depth: i32 = 0;
        let mut max_depth: i32 = 0;

        for op in ops {
            // Calculate stack effect of each operation
            let effect = match op {
                // Push operations (+1)
                Op::LoadColumn(_)
                | Op::LoadColumn2(_)
                | Op::LoadOuterColumn(_)
                | Op::LoadConst(_)
                | Op::LoadParam(_)
                | Op::LoadNamedParam(_)
                | Op::LoadNull(_)
                | Op::LoadAggregateResult(_)
                | Op::LoadTransactionId
                | Op::Dup
                // Fused compare ops: push 1 (load + compare in single op)
                | Op::EqColumnConst(_, _)
                | Op::NeColumnConst(_, _)
                | Op::LtColumnConst(_, _)
                | Op::LeColumnConst(_, _)
                | Op::GtColumnConst(_, _)
                | Op::GeColumnConst(_, _)
                // More fused ops: push 1
                | Op::IsNullColumn(_)
                | Op::IsNotNullColumn(_)
                | Op::LikeColumn(_, _, _)
                | Op::InSetColumn(_, _, _)
                | Op::BetweenColumnConst(_, _, _) => 1,

                // Pop 2, push 1 (-1)
                Op::Eq
                | Op::Ne
                | Op::Lt
                | Op::Le
                | Op::Gt
                | Op::Ge
                | Op::IsDistinctFrom
                | Op::IsNotDistinctFrom
                | Op::AndFinalize
                | Op::OrFinalize
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Mod
                | Op::BitAnd
                | Op::BitOr
                | Op::BitXor
                | Op::Shl
                | Op::Shr
                | Op::Concat
                | Op::Xor
                | Op::NullIf
                | Op::CaseCompare => -1,

                // Pop 3, push 1 (-2)
                Op::Between | Op::NotBetween => -2,

                // JSON/Timestamp binary ops: Pop 2, push 1 (-1)
                Op::JsonAccess
                | Op::JsonAccessText
                | Op::TimestampAddInterval
                | Op::TimestampSubInterval
                | Op::TimestampDiff
                | Op::TimestampAddDays
                | Op::TimestampSubDays => -1,

                // Transform (0) - pop 1, push 1
                Op::IsNull
                | Op::IsNotNull
                | Op::IsTrue
                | Op::IsNotTrue
                | Op::IsFalse
                | Op::IsNotFalse
                | Op::Not
                | Op::Neg
                | Op::BitNot
                | Op::Like(_, _)
                | Op::LikeEscape(_, _, _)
                | Op::Glob(_)
                | Op::Regexp(_)
                | Op::InSet(_, _)
                | Op::NotInSet(_, _)
                | Op::Cast(_)
                | Op::TruncateToDate
                | Op::ExecScalarSubquery(_)
                | Op::ExecExists(_)
                | Op::ExecInSubquery(_)
                | Op::ExecAll(_, _)
                | Op::ExecAny(_, _) => 0,

                // Multi-column IN: pop N, push 1
                Op::InTupleSet { tuple_size, .. } => 1 - (*tuple_size as i32),

                // Pop only (-1)
                Op::Pop => -1,

                // Conditionals (no change to depth calculation)
                Op::And(_) | Op::Or(_) => 0,

                // Function calls: pop N, push 1
                Op::CallScalar { arg_count, .. } => 1 - (*arg_count as i32),
                Op::Coalesce(n) | Op::Greatest(n) | Op::Least(n) => 1 - (*n as i32),

                // Control flow (no stack effect for depth calculation)
                Op::Jump(_)
                | Op::JumpIfTrue(_)
                | Op::JumpIfFalse(_)
                | Op::JumpIfNull(_)
                | Op::PopJumpIfTrue(_)
                | Op::PopJumpIfFalse(_)
                | Op::Swap
                | Op::Nop
                | Op::Return
                | Op::ReturnTrue
                | Op::ReturnFalse
                | Op::ReturnNull(_)
                | Op::CaseStart
                | Op::CaseWhen(_)
                | Op::CaseThen(_)
                | Op::CaseElse
                | Op::CaseEnd => 0,
            };

            depth += effect;
            max_depth = max_depth.max(depth);
        }

        // Ensure at least 1 for safety
        (max_depth as usize).max(1)
    }

    /// Disassemble the program for debugging
    pub fn disassemble(&self) -> String {
        let mut result = String::new();
        for (i, op) in self.ops.iter().enumerate() {
            result.push_str(&format!("{:04}: {:?}\n", i, op));
        }
        result
    }

    /// Apply peephole optimizations to fuse common instruction patterns.
    /// This is called automatically when creating a program via `new()`.
    pub fn optimize(mut self) -> Self {
        self.ops = Self::peephole_optimize(self.ops);
        // Recalculate metadata after optimization
        self.max_stack_depth = Self::compute_stack_depth(&self.ops);
        self
    }

    /// Peephole optimizer: fuse common instruction patterns into single ops
    fn peephole_optimize(mut ops: Vec<Op>) -> Vec<Op> {
        if ops.len() < 2 {
            return ops;
        }

        // Build a set of positions that are jump targets - we can't fuse instructions
        // that are jump targets because that would make the jump land in the middle
        // of what becomes a single instruction.
        let mut jump_targets = std::collections::HashSet::new();
        for op in &ops {
            match op {
                Op::And(t)
                | Op::Or(t)
                | Op::Jump(t)
                | Op::JumpIfTrue(t)
                | Op::JumpIfFalse(t)
                | Op::JumpIfNull(t)
                | Op::PopJumpIfTrue(t)
                | Op::PopJumpIfFalse(t)
                | Op::CaseWhen(t)
                | Op::CaseThen(t) => {
                    jump_targets.insert(*t as usize);
                }
                _ => {}
            }
        }

        let mut result = Vec::with_capacity(ops.len());
        // Map from old instruction position to new instruction position
        let mut position_map: Vec<usize> = Vec::with_capacity(ops.len());
        let mut i = 0;

        while i < ops.len() {
            // Record the new position for this old position
            let new_pos = result.len();

            // Pattern 1: LoadColumn + LoadConst + LoadConst + Between → BetweenColumnConst (4 ops → 1)
            if i + 3 < ops.len() {
                let is_safe = !jump_targets.contains(&i)
                    && !jump_targets.contains(&(i + 1))
                    && !jump_targets.contains(&(i + 2))
                    && !jump_targets.contains(&(i + 3));

                if is_safe {
                    if let (
                        Op::LoadColumn(col_idx),
                        Op::LoadConst(low_val),
                        Op::LoadConst(high_val),
                        Op::Between,
                    ) = (&ops[i], &ops[i + 1], &ops[i + 2], &ops[i + 3])
                    {
                        result.push(Op::BetweenColumnConst(
                            *col_idx,
                            low_val.clone(),
                            high_val.clone(),
                        ));
                        // All 4 positions map to this single new position
                        position_map.push(new_pos);
                        position_map.push(new_pos);
                        position_map.push(new_pos);
                        position_map.push(new_pos);
                        i += 4;
                        continue;
                    }
                }
            }

            // Pattern 2: LoadColumn + LoadConst + Compare → XxColumnConst (3 ops → 1)
            if i + 2 < ops.len() {
                let is_safe = !jump_targets.contains(&i)
                    && !jump_targets.contains(&(i + 1))
                    && !jump_targets.contains(&(i + 2));

                if is_safe {
                    if let (Op::LoadColumn(col_idx), Op::LoadConst(const_val)) =
                        (&ops[i], &ops[i + 1])
                    {
                        let fused = match &ops[i + 2] {
                            Op::Eq => Some(Op::EqColumnConst(*col_idx, const_val.clone())),
                            Op::Ne => Some(Op::NeColumnConst(*col_idx, const_val.clone())),
                            Op::Lt => Some(Op::LtColumnConst(*col_idx, const_val.clone())),
                            Op::Le => Some(Op::LeColumnConst(*col_idx, const_val.clone())),
                            Op::Gt => Some(Op::GtColumnConst(*col_idx, const_val.clone())),
                            Op::Ge => Some(Op::GeColumnConst(*col_idx, const_val.clone())),
                            _ => None,
                        };

                        if let Some(fused_op) = fused {
                            result.push(fused_op);
                            // All 3 positions map to this single new position
                            position_map.push(new_pos);
                            position_map.push(new_pos);
                            position_map.push(new_pos);
                            i += 3;
                            continue;
                        }
                    }
                }
            }

            // Pattern 3: LoadColumn + IsNull/IsNotNull/Like/InSet (2 ops → 1)
            if i + 1 < ops.len() {
                let is_safe = !jump_targets.contains(&i) && !jump_targets.contains(&(i + 1));

                if is_safe {
                    if let Op::LoadColumn(col_idx) = &ops[i] {
                        let fused = match &ops[i + 1] {
                            Op::IsNull => Some(Op::IsNullColumn(*col_idx)),
                            Op::IsNotNull => Some(Op::IsNotNullColumn(*col_idx)),
                            Op::Like(pattern, case_insensitive) => {
                                Some(Op::LikeColumn(*col_idx, pattern.clone(), *case_insensitive))
                            }
                            Op::InSet(set, has_null) => {
                                Some(Op::InSetColumn(*col_idx, set.clone(), *has_null))
                            }
                            _ => None,
                        };

                        if let Some(fused_op) = fused {
                            result.push(fused_op);
                            // Both positions map to this single new position
                            position_map.push(new_pos);
                            position_map.push(new_pos);
                            i += 2;
                            continue;
                        }
                    }
                }
            }

            // No fusion, copy the op
            result.push(std::mem::replace(&mut ops[i], Op::Nop));
            position_map.push(new_pos);
            i += 1;
        }

        // If we fused anything, we need to adjust jump targets
        if result.len() != ops.len() {
            Self::adjust_jump_targets(&mut result, &position_map);
        }

        result
    }

    /// Adjust jump targets after peephole optimization using the position map.
    fn adjust_jump_targets(ops: &mut [Op], position_map: &[usize]) {
        for op in ops.iter_mut() {
            match op {
                Op::And(t)
                | Op::Or(t)
                | Op::Jump(t)
                | Op::JumpIfTrue(t)
                | Op::JumpIfFalse(t)
                | Op::JumpIfNull(t)
                | Op::PopJumpIfTrue(t)
                | Op::PopJumpIfFalse(t)
                | Op::CaseWhen(t)
                | Op::CaseThen(t) => {
                    let old_target = *t as usize;
                    if old_target < position_map.len() {
                        *t = position_map[old_target] as u16;
                    }
                }
                _ => {}
            }
        }
    }
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Program")
            .field("ops_count", &self.ops.len())
            .field("max_stack_depth", &self.max_stack_depth)
            .field("needs_outer_context", &self.needs_outer_context)
            .field("needs_second_row", &self.needs_second_row)
            .field("has_subqueries", &self.has_subqueries)
            .finish()
    }
}

/// Builder for constructing programs
pub struct ProgramBuilder {
    ops: Vec<Op>,
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self {
            ops: Vec::with_capacity(32),
        }
    }

    /// Emit an operation
    #[inline]
    pub fn emit(&mut self, op: Op) {
        self.ops.push(op);
    }

    /// Get current position (for jump targets)
    #[inline]
    pub fn position(&self) -> u16 {
        self.ops.len() as u16
    }

    /// Patch a jump target at a specific position
    pub fn patch_jump(&mut self, pos: usize, target: u16) {
        if pos < self.ops.len() {
            match &mut self.ops[pos] {
                Op::And(t)
                | Op::Or(t)
                | Op::Jump(t)
                | Op::JumpIfTrue(t)
                | Op::JumpIfFalse(t)
                | Op::JumpIfNull(t)
                | Op::PopJumpIfTrue(t)
                | Op::PopJumpIfFalse(t)
                | Op::CaseWhen(t)
                | Op::CaseThen(t) => *t = target,
                _ => {}
            }
        }
    }

    /// Build the final program
    pub fn build(self) -> Program {
        Program::new(self.ops)
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}
