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

use smallvec::SmallVec;

use super::ops::Op;
use crate::common::CompactArc;
use crate::core::Value;

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

    /// Whether every op is handled by ExprVM::execute_cow, so the
    /// borrowed interpreter never bails mid-program into a full owned
    /// re-execution
    cow_supported: bool,

    /// What binding the program to its execution's parameters reads, for a
    /// program that loads one
    binding: Option<Box<Binding>>,

    /// The program this one was bound from, which a later bind binds again
    bound_from: Option<CompactArc<Program>>,

    /// Source expression string (for debugging)
    #[cfg(debug_assertions)]
    source: Option<String>,
}

impl Program {
    /// Create a new program from operations.
    /// Automatically applies peephole optimizations (instruction fusion).
    pub fn new(ops: Vec<Op>) -> Self {
        Self::with_folds(ops, Vec::new())
    }

    /// [`Self::new`] with the ranges of `ops` that fold once the
    /// execution's parameters are known
    pub fn with_folds(ops: Vec<Op>, mut folds: Vec<(u16, u16)>) -> Self {
        // Apply peephole optimizations
        let ops = Self::peephole_optimize(ops, &mut folds);

        let max_stack_depth = Self::compute_stack_depth(&ops);
        let needs_outer_context = ops.iter().any(|op| matches!(op, Op::LoadOuterColumn(_)));
        let needs_second_row = ops.iter().any(|op| matches!(op, Op::LoadColumn2(_)));
        let has_subqueries = ops.iter().any(|op| {
            matches!(
                op,
                Op::ExecScalarSubquery(_)
                    | Op::ExecExists(_)
                    | Op::ExecInSubquery(_)
                    | Op::ExecQuantified(_, _, _)
            )
        });

        let cow_supported = Self::compute_cow_supported(&ops, max_stack_depth);
        let binding = Binding::of(&ops, folds);

        Self {
            ops,
            max_stack_depth,
            needs_outer_context,
            needs_second_row,
            has_subqueries,
            cow_supported,
            binding,
            bound_from: None,
            #[cfg(debug_assertions)]
            source: None,
        }
    }

    /// The program's ops, for a builder to reuse
    pub fn into_ops(self) -> Vec<Op> {
        self.ops
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
                    | Op::ExecQuantified(_, _, _)
            )
        });

        let cow_supported = Self::compute_cow_supported(&ops, max_stack_depth);
        let binding = Binding::of(&ops, Vec::new());

        Self {
            ops,
            max_stack_depth,
            needs_outer_context,
            needs_second_row,
            has_subqueries,
            cow_supported,
            binding,
            bound_from: None,
            #[cfg(debug_assertions)]
            source: None,
        }
    }

    /// Whether a thread running this program beside others should run its
    /// [`Self::worker_copy`]: a REGEXP matched from several threads at once
    /// makes all but one wait on its shared cache
    pub fn holds_thread_state(&self) -> bool {
        self.ops.iter().any(|op| matches!(op, Op::Regexp(_)))
    }

    /// This program for one more thread, each REGEXP with a cache of its own
    pub fn worker_copy(&self) -> Self {
        let ops = self
            .ops
            .iter()
            .map(|op| match op {
                Op::Regexp(regex) => Op::Regexp(std::sync::Arc::new(regex::Regex::clone(regex))),
                op => op.clone(),
            })
            .collect();
        Self::new_unoptimized(ops)
    }

    /// This program with each parameter it loads taken from `params`, as the
    /// compiler builds it from those values written as literals; None when
    /// it loads none, or when binding does not shrink it: a parameter loads
    /// at the cost of its value. A parameter `params` lacks keeps its error
    pub fn bind(&self, params: &[Value]) -> Option<Self> {
        let binding = self.binding.as_deref()?;
        if binding.shrinks.get() == Some(&false) {
            return None;
        }
        let bound = self.bind_with(|op| match op {
            Op::LoadParam(slot) => params.get(*slot as usize).cloned().map(Op::LoadConst),
            _ => None,
        })?;
        let shrinks = bound.ops.len() < self.ops.len();
        // No value changes what fusion makes of a parameter no fold holds
        // and no op takes, so the first bind of such a program settles it
        let settled = binding.folds.is_empty()
            && !self.ops.windows(2).any(|pair| {
                matches!(pair[0], Op::LoadParam(_)) && super::compiler::takes_values(&pair[1])
            });
        if settled {
            let _ = binding.shrinks.set(shrinks);
        }
        shrinks.then_some(bound)
    }

    /// `program` bound to `params`, from the program it was bound from when
    /// it was bound before; that program as it is when binding does not
    /// shrink it
    pub fn bound(program: CompactArc<Program>, params: &[Value]) -> CompactArc<Program> {
        let source = program.bound_from.clone().unwrap_or(program);
        match source.bind(params) {
            Some(mut bound) => {
                bound.bound_from = Some(source);
                CompactArc::new(bound)
            }
            None => source,
        }
    }

    /// Whether this program loads a parameter and a bind found that binding
    /// the parameters never shrinks it
    pub fn bind_never_shrinks(&self) -> bool {
        self.binding
            .as_deref()
            .is_some_and(|binding| binding.shrinks.get() == Some(&false))
    }

    /// This program with each op `known` replaces taken as its replacement,
    /// each range the compiler marked folded once what it reads is known, and
    /// a value so known taken by the op after as the compiler builds that op
    /// from the value written as a literal; None when `known` replaces none
    pub fn bind_with(&self, known: impl Fn(&Op) -> Option<Op>) -> Option<Self> {
        let replaced: SmallVec<[(usize, Op); 4]> = self
            .ops
            .iter()
            .enumerate()
            .filter_map(|(at, op)| known(op).map(|op| (at, op)))
            .collect();
        if replaced.is_empty() {
            return None;
        }
        let mut source = self.ops.clone();
        let targets = JumpTargets::of(&mut source);
        let replaced_at: SmallVec<[usize; 4]> = replaced.iter().map(|(at, _)| *at).collect();
        for (at, op) in replaced {
            source[at] = op;
        }
        let folded = self.folded(&mut source, &replaced_at);
        let mut ops: Vec<Op> = Vec::with_capacity(source.len());
        // Of each op emitted: whether it loads a value known only now, and
        // the first position it stands for
        let mut emitted: SmallVec<[(bool, usize); 16]> = SmallVec::new();
        let mut position_map: SmallVec<[usize; 16]> = SmallVec::new();
        let mut at = 0;
        while at < source.len() {
            let here = ops.len();
            let (op, next, known_now) = match folded.iter().find(|(start, ..)| *start == at) {
                Some((start, end, value)) => {
                    position_map.extend((*start..*end).map(|_| here));
                    (Op::LoadConst(value.clone()), *end, true)
                }
                None => {
                    position_map.push(here);
                    let op = std::mem::replace(&mut source[at], Op::Return);
                    (op, at + 1, replaced_at.contains(&at))
                }
            };
            let after = source.get(next).map(|op| (next, op));
            if let Some((first, takes_next, taken)) =
                Self::taken_by(&op, at, after, &ops, &emitted, &targets)
            {
                ops.truncate(first);
                emitted.truncate(first);
                for position in position_map.iter_mut().rev() {
                    if *position <= first {
                        break;
                    }
                    *position = first;
                }
                ops.push(taken);
                // Built from a value known only now, as a value it is one too
                emitted.push((true, at));
                if takes_next {
                    position_map.push(first);
                    at = next + 1;
                    continue;
                }
            } else {
                ops.push(op);
                emitted.push((known_now, at));
            }
            at = next;
        }
        position_map.push(ops.len());
        Self::adjust_jump_targets(&mut ops, &position_map);
        Some(Self::new(ops))
    }

    /// When `op`, at position `at`, takes the constants the last of `ops`
    /// load, one of them known only now: where the first of them stands,
    /// whether it takes the op after, `next`, too, and the op the compiler
    /// builds in their place. No jump may land on `op`, past the first
    /// value, or on `next` taken
    fn taken_by(
        op: &Op,
        at: usize,
        next: Option<(usize, &Op)>,
        ops: &[Op],
        emitted: &[(bool, usize)],
        targets: &JumpTargets,
    ) -> Option<(usize, bool, Op)> {
        let needed = super::compiler::values_taken(op)?;
        if !matches!(ops.last(), Some(Op::LoadConst(_) | Op::LoadNull(_))) || targets.contains(&at)
        {
            return None;
        }
        let mut values: SmallVec<[Value; 4]> = ops
            .iter()
            .rev()
            .take(needed)
            .map_while(|op| match op {
                Op::LoadConst(value) => Some(value.clone()),
                Op::LoadNull(data_type) => Some(Value::Null(*data_type)),
                _ => None,
            })
            .collect();
        values.reverse();
        let values: SmallVec<[&Value; 4]> = values.iter().collect();
        let next = next.filter(|(position, _)| !targets.contains(position));
        let (count, takes_next, taken) =
            super::compiler::bound_consumer(op, &values, next.map(|(_, op)| op))?;
        let first = ops.len() - count;
        let taken_values = &emitted[first..];
        (taken_values.iter().any(|(known, _)| *known)
            && taken_values[1..]
                .iter()
                .all(|(_, position)| !targets.contains(position)))
        .then_some((first, takes_next, taken))
    }

    /// The marked ranges of `source` that fold now, with their values: each
    /// that holds an op replaced at `replaced`, the outermost first; one
    /// whose fold errors, or that a jump crosses, leaves the ranges inside
    /// it to fold
    fn folded(
        &self,
        source: &mut [Op],
        replaced: &[usize],
    ) -> SmallVec<[(usize, usize, Value); 2]> {
        let marked = self.binding.as_deref().map_or(&[][..], |b| &b.folds[..]);
        let mut folds: SmallVec<[(usize, usize); 4]> = marked
            .iter()
            .map(|&(start, end)| (usize::from(start), usize::from(end)))
            .filter(|(start, end)| replaced.iter().any(|at| (*start..*end).contains(at)))
            .collect();
        folds.sort_unstable_by_key(|&(start, end)| (start, std::cmp::Reverse(end)));
        let mut folded: SmallVec<[(usize, usize, Value); 2]> = SmallVec::new();
        for (start, end) in folds {
            if folded.iter().any(|(s, e, _)| start >= *s && end <= *e) {
                continue;
            }
            let Some(range) = Self::fold_range(source, start, end) else {
                continue;
            };
            if let Some(value) = super::compiler::fold_ops(range) {
                folded.push((start, end, value));
            }
        }
        folded
    }

    /// The ops of `source[start..end]` as a program of their own, their
    /// jumps moved with them; None when a jump crosses the range's edge
    /// other than into its start or out to its end
    fn fold_range(source: &mut [Op], start: usize, end: usize) -> Option<Vec<Op>> {
        let (before, rest) = source.split_at_mut(start);
        let (inside, after) = rest.split_at_mut(end - start);
        let entered = before
            .iter_mut()
            .chain(after.iter_mut())
            .filter_map(jump_target)
            .any(|target| (start + 1..end).contains(&usize::from(*target)));
        if entered {
            return None;
        }
        // Room for the Return the fold ends it with
        let mut range = Vec::with_capacity(inside.len() + 1);
        range.extend_from_slice(inside);
        for target in range.iter_mut().filter_map(jump_target) {
            let at = usize::from(*target);
            if !(start..=end).contains(&at) {
                return None;
            }
            *target = u16::try_from(at - start).ok()?;
        }
        Some(range)
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
                | Op::LikeColumn(_, _)
                | Op::InSetColumn(_, _)
                | Op::BetweenColumnConst(_, _, _)
                // A subquery's value or existence: push 1
                | Op::ExecScalarSubquery(_)
                | Op::ExecExists(_) => 1,

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
                Op::Between | Op::NotBetween | Op::LikeDynamicEscapeExpr(_) => -2,
                // Pop the value and its items, push 1
                Op::InList(count) => -i32::try_from(*count).unwrap_or(i32::MAX),
                // Pop the factors (and the timestamp), push 1
                Op::ScaleInterval(_, count) => 1 - i32::from(*count),
                Op::TimestampAddScaledInterval(_, _, count) => -i32::from(*count),

                // Dynamic pattern ops: Pop 2, push 1 (-1)
                Op::LikeDynamic(_)
                | Op::LikeDynamicEscape(_, _)
                | Op::GlobDynamic
                | Op::RegexpDynamic
                // JSON/Timestamp binary ops: Pop 2, push 1 (-1)
                | Op::JsonAccess
                | Op::JsonAccessText
                | Op::TimestampAddInterval
                | Op::TimestampSubInterval
                | Op::TimestampDiff
                | Op::TimestampAddDays
                | Op::TimestampSubDays
                | Op::VectorDistanceL2
                | Op::VectorDistanceCosine
                | Op::VectorDistanceIP => -1,

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
                | Op::Like(_)
                | Op::LikeEscape(_, _)
                | Op::Glob(_)
                | Op::Regexp(_)
                | Op::InSet(_)
                | Op::NotInSet(_)
                | Op::QuantifiedSet(_)
                | Op::Cast(_)
                | Op::ReadAsColumn(_)
                | Op::TruncateToDate
                | Op::ExecInSubquery(_)
                | Op::ExecQuantified(_, _, _)
                | Op::NativeFn1(_) => 0,

                // Multi-column IN: pop N, push 1
                Op::InTupleSet { tuple_size, .. } => 1 - (*tuple_size as i32),

                // Pop only (-1)
                Op::Pop => -1,

                // Conditionals (no change to depth calculation)
                Op::And(_) | Op::Or(_) => 0,

                // Function calls: pop N, push 1
                Op::CallScalar { arg_count, .. } => 1 - (*arg_count as i32),
                Op::Coalesce(n) | Op::Greatest(n) | Op::Least(n) | Op::ConcatN(n) => {
                    1 - (*n as i32)
                }

                // Control flow (no stack effect for depth calculation)
                Op::Jump(_)
                | Op::JumpIfTrue(_)
                | Op::JumpIfFalse(_)
                | Op::JumpIfNull(_)
                | Op::JumpIfNotNull(_)
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
        let mut folds = self.binding.take().map(|b| b.folds).unwrap_or_default();
        self.ops = Self::peephole_optimize(self.ops, &mut folds);
        self.binding = Binding::of(&self.ops, folds);
        // Recalculate metadata after optimization
        self.max_stack_depth = Self::compute_stack_depth(&self.ops);
        self.cow_supported = Self::compute_cow_supported(&self.ops, self.max_stack_depth);
        self
    }

    /// Whether execute_cow handles every op of this program
    #[inline]
    pub fn cow_supported(&self) -> bool {
        self.cow_supported
    }

    /// Mirrors the op set ExprVM::execute_cow handles; keep the two in
    /// sync (drift only costs performance, the cow catch-all stays). A
    /// program deeper than its inline stack runs owned, which keeps its stack
    fn compute_cow_supported(ops: &[Op], max_stack_depth: usize) -> bool {
        if max_stack_depth > super::vm::STACK_INLINE_CAPACITY {
            return false;
        }
        ops.iter().all(|op| {
            matches!(
                op,
                Op::LoadColumn(_)
                    | Op::LoadColumn2(_)
                    | Op::LoadConst(_)
                    | Op::LoadNull(_)
                    | Op::LoadParam(_)
                    | Op::LoadAggregateResult(_)
                    | Op::Eq
                    | Op::Ne
                    | Op::Lt
                    | Op::Le
                    | Op::Gt
                    | Op::Ge
                    | Op::Add
                    | Op::Sub
                    | Op::Mul
                    | Op::Div
                    | Op::Mod
                    | Op::EqColumnConst(..)
                    | Op::LtColumnConst(..)
                    | Op::GtColumnConst(..)
                    | Op::And(_)
                    | Op::AndFinalize
                    | Op::Or(_)
                    | Op::OrFinalize
                    | Op::Not
                    | Op::IsNull
                    | Op::IsNotNull
                    | Op::Jump(_)
                    | Op::JumpIfFalse(_)
                    | Op::JumpIfTrue(_)
                    | Op::JumpIfNull(_)
                    | Op::JumpIfNotNull(_)
                    | Op::PopJumpIfFalse(_)
                    | Op::PopJumpIfTrue(_)
                    | Op::Pop
                    | Op::Nop
                    | Op::Coalesce(_)
                    | Op::NullIf
                    | Op::Greatest(_)
                    | Op::Least(_)
                    | Op::Concat
                    | Op::ConcatN(_)
                    | Op::CaseStart
                    | Op::CaseWhen(_)
                    | Op::CaseThen(_)
                    | Op::CaseElse
                    | Op::CaseEnd
                    | Op::CaseCompare
                    | Op::CallScalar { .. }
                    | Op::Return
                    | Op::ReturnTrue
                    | Op::ReturnFalse
                    | Op::ReturnNull(_)
            )
        })
    }

    /// Peephole optimizer: fuse common instruction patterns into single ops
    fn peephole_optimize(mut ops: Vec<Op>, folds: &mut Vec<(u16, u16)>) -> Vec<Op> {
        if ops.len() < 2 {
            return ops;
        }

        // Build a set of positions that are jump targets - we can't fuse instructions
        // that are jump targets because that would make the jump land in the middle
        // of what becomes a single instruction.
        let jump_targets = JumpTargets::of(&mut ops);

        let mut result = Vec::with_capacity(ops.len());
        // Map from old instruction position to new instruction position
        let mut position_map: SmallVec<[usize; 32]> = SmallVec::with_capacity(ops.len());
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
                    if let (
                        Op::LoadColumn(col_idx) | Op::LoadAggregateResult(col_idx),
                        Op::LoadConst(const_val),
                    ) = (&ops[i], &ops[i + 1])
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
                            Op::Like(pattern) => Some(Op::LikeColumn(*col_idx, pattern.clone())),
                            Op::InSet(set) => Some(Op::InSetColumn(*col_idx, set.clone())),
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
            Self::adjust_folds(folds, &position_map, result.len());
        }

        result
    }

    /// The fold ranges at their positions after fusion; a range an op fused
    /// across its edge folds no more
    fn adjust_folds(folds: &mut Vec<(u16, u16)>, position_map: &[usize], len: usize) {
        let at = |old: u16| position_map.get(usize::from(old)).copied().unwrap_or(len);
        folds.retain_mut(|(start, end)| {
            let edges_kept =
                (*start == 0 || at(*start - 1) != at(*start)) && at(*end - 1) != at(*end);
            (*start, *end) = (at(*start) as u16, at(*end) as u16);
            edges_kept
        });
    }

    /// Adjust jump targets after peephole optimization using the position map.
    fn adjust_jump_targets(ops: &mut [Op], position_map: &[usize]) {
        for t in ops.iter_mut().filter_map(jump_target) {
            let old_target = *t as usize;
            if old_target < position_map.len() {
                *t = position_map[old_target] as u16;
            }
        }
    }
}

/// What binding a program to its execution's parameters reads beside its ops
#[derive(Clone)]
struct Binding {
    /// The ranges of ops that fold into one value once the parameters are
    /// known, as the compiler marked them
    folds: Vec<(u16, u16)>,
    /// Whether binding the parameters shrinks the program, once a bind
    /// found it out for every value
    shrinks: std::sync::OnceLock<bool>,
}

impl Binding {
    /// The binding of a program of `ops`, when one loads a parameter
    fn of(ops: &[Op], folds: Vec<(u16, u16)>) -> Option<Box<Self>> {
        ops.iter()
            .any(|op| matches!(op, Op::LoadParam(_)))
            .then(|| {
                Box::new(Self {
                    folds,
                    shrinks: std::sync::OnceLock::new(),
                })
            })
    }
}

/// The positions some op of a program jumps to
struct JumpTargets(SmallVec<[bool; 32]>);

impl JumpTargets {
    fn of(ops: &mut [Op]) -> Self {
        let mut targets: SmallVec<[bool; 32]> = smallvec::smallvec![false; ops.len() + 1];
        for t in ops.iter_mut().filter_map(jump_target) {
            if let Some(target) = targets.get_mut(usize::from(*t)) {
                *target = true;
            }
        }
        Self(targets)
    }

    #[inline]
    fn contains(&self, position: &usize) -> bool {
        self.0.get(*position).copied().unwrap_or(false)
    }
}

/// The position `op` jumps to, when it jumps
fn jump_target(op: &mut Op) -> Option<&mut u16> {
    match op {
        Op::And(t)
        | Op::Or(t)
        | Op::Jump(t)
        | Op::JumpIfTrue(t)
        | Op::JumpIfFalse(t)
        | Op::JumpIfNull(t)
        | Op::JumpIfNotNull(t)
        | Op::PopJumpIfTrue(t)
        | Op::PopJumpIfFalse(t)
        | Op::CaseWhen(t)
        | Op::CaseThen(t) => Some(t),
        _ => None,
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
    /// The ranges of ops that fold once the execution's parameters are known
    folds: Vec<(u16, u16)>,
    /// How many parameter loads were emitted
    params_loaded: u32,
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self {
            ops: Vec::with_capacity(32),
            folds: Vec::new(),
            params_loaded: 0,
        }
    }

    /// A builder emitting into `ops`, emptied, to reuse its capacity
    pub fn reusing(mut ops: Vec<Op>) -> Self {
        ops.clear();
        Self {
            ops,
            folds: Vec::new(),
            params_loaded: 0,
        }
    }

    /// How many parameter loads were emitted so far
    #[inline]
    pub fn params_loaded(&self) -> u32 {
        self.params_loaded
    }

    /// How many ops were emitted since `start`, none when the program has
    /// grown past what a position holds
    pub fn emitted_since(&self, start: u16) -> usize {
        match u16::try_from(self.ops.len()) {
            Ok(end) => usize::from(end - start),
            Err(_) => 0,
        }
    }

    /// The ops emitted since `start` fold into one value once the
    /// execution's parameters are known
    pub fn mark_fold(&mut self, start: u16) {
        self.folds.push((start, self.position()));
    }

    /// Emit an operation
    #[inline]
    pub fn emit(&mut self, op: Op) {
        self.params_loaded += u32::from(matches!(op, Op::LoadParam(_)));
        self.ops.push(op);
    }

    /// Get current position (for jump targets)
    #[inline]
    pub fn position(&self) -> u16 {
        self.ops.len() as u16
    }

    /// The value of the one `LoadConst` emitted since `start`, taken back out
    pub fn take_constant_since(&mut self, start: u16) -> Option<Value> {
        if self.ops.len() != usize::from(start) + 1 {
            return None;
        }
        match self.ops.pop() {
            Some(Op::LoadConst(value)) => Some(value),
            Some(op) => {
                self.ops.push(op);
                None
            }
            None => None,
        }
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
                | Op::JumpIfNotNull(t)
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
        Program::with_folds(self.ops, self.folds)
    }

    /// Build without peephole optimization (for tiny fold programs)
    pub fn build_unoptimized(self) -> Program {
        Program::new_unoptimized(self.ops)
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Program factory methods
    // =========================================================================

    #[test]
    fn test_program_null() {
        let prog = Program::null();
        assert!(!prog.is_empty());
        assert!(!prog.needs_outer_context());
        assert!(!prog.needs_second_row());
        assert!(!prog.has_subqueries());
    }

    #[test]
    fn test_program_constant() {
        let prog = Program::constant(Value::Integer(42));
        assert!(!prog.is_empty());
        assert!(!prog.needs_outer_context());
        assert!(!prog.needs_second_row());
    }

    #[test]
    fn test_program_always_true() {
        let prog = Program::always_true();
        assert_eq!(prog.len(), 1);
        assert!(matches!(prog.ops()[0], Op::ReturnTrue));
    }

    #[test]
    fn test_program_always_false() {
        let prog = Program::always_false();
        assert_eq!(prog.len(), 1);
        assert!(matches!(prog.ops()[0], Op::ReturnFalse));
    }

    // =========================================================================
    // Stack depth calculation
    // =========================================================================

    #[test]
    fn test_stack_depth_simple() {
        // LoadConst pushes 1, Return pops and returns
        let prog = Program::new_unoptimized(vec![Op::LoadConst(Value::Integer(1)), Op::Return]);
        assert_eq!(prog.max_stack_depth(), 1);
    }

    #[test]
    fn test_stack_depth_binary_op() {
        // LoadConst (+1), LoadConst (+1), Add (-1) = max 2
        let prog = Program::new_unoptimized(vec![
            Op::LoadConst(Value::Integer(1)),
            Op::LoadConst(Value::Integer(2)),
            Op::Add,
            Op::Return,
        ]);
        assert_eq!(prog.max_stack_depth(), 2);
    }

    #[test]
    fn test_stack_depth_nested() {
        // (1 + 2) * (3 + 4)
        // Load 1, Load 2, Add, Load 3, Load 4, Add, Mul
        let prog = Program::new_unoptimized(vec![
            Op::LoadConst(Value::Integer(1)),
            Op::LoadConst(Value::Integer(2)),
            Op::Add,
            Op::LoadConst(Value::Integer(3)),
            Op::LoadConst(Value::Integer(4)),
            Op::Add,
            Op::Mul,
            Op::Return,
        ]);
        // Stack trace: 1, 2, 1, 2, 3, 2, 1
        // Max is 3 (after loading 3rd value before second add)
        assert!(prog.max_stack_depth() >= 2);
    }

    #[test]
    fn test_stack_depth_fused_ops() {
        // Fused ops push 1
        let prog =
            Program::new_unoptimized(vec![Op::EqColumnConst(0, Value::Integer(5)), Op::Return]);
        assert_eq!(prog.max_stack_depth(), 1);
    }

    #[test]
    fn test_stack_depth_between() {
        // Between pops 3, pushes 1
        let prog = Program::new_unoptimized(vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(1)),
            Op::LoadConst(Value::Integer(10)),
            Op::Between,
            Op::Return,
        ]);
        assert_eq!(prog.max_stack_depth(), 3);
    }

    // =========================================================================
    // Metadata detection
    // =========================================================================

    #[test]
    fn test_needs_outer_context() {
        let prog = Program::new_unoptimized(vec![Op::LoadOuterColumn("col".into()), Op::Return]);
        assert!(prog.needs_outer_context());

        let prog2 = Program::new_unoptimized(vec![Op::LoadColumn(0), Op::Return]);
        assert!(!prog2.needs_outer_context());
    }

    #[test]
    fn test_needs_second_row() {
        let prog = Program::new_unoptimized(vec![
            Op::LoadColumn(0),
            Op::LoadColumn2(1),
            Op::Eq,
            Op::Return,
        ]);
        assert!(prog.needs_second_row());

        let prog2 = Program::new_unoptimized(vec![Op::LoadColumn(0), Op::Return]);
        assert!(!prog2.needs_second_row());
    }

    #[test]
    fn test_has_subqueries() {
        let prog1 = Program::new_unoptimized(vec![Op::ExecExists(0), Op::Return]);
        assert!(prog1.has_subqueries());

        let prog2 = Program::new_unoptimized(vec![Op::ExecScalarSubquery(0), Op::Return]);
        assert!(prog2.has_subqueries());

        let prog3 = Program::new_unoptimized(vec![Op::LoadColumn(0), Op::Return]);
        assert!(!prog3.has_subqueries());
    }

    // =========================================================================
    // Peephole optimization
    // =========================================================================

    #[test]
    fn test_peephole_eq_column_const() {
        // LoadColumn + LoadConst + Eq should fuse to EqColumnConst
        let ops = vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(5)),
            Op::Eq,
            Op::Return,
        ];
        let prog = Program::new(ops);
        // Should be fused to 2 ops: EqColumnConst + Return
        assert_eq!(prog.len(), 2);
        assert!(matches!(prog.ops()[0], Op::EqColumnConst(0, _)));
    }

    #[test]
    fn test_peephole_lt_column_const() {
        let ops = vec![
            Op::LoadColumn(1),
            Op::LoadConst(Value::Integer(10)),
            Op::Lt,
            Op::Return,
        ];
        let prog = Program::new(ops);
        assert_eq!(prog.len(), 2);
        assert!(matches!(prog.ops()[0], Op::LtColumnConst(1, _)));
    }

    #[test]
    fn test_peephole_is_null_column() {
        let ops = vec![Op::LoadColumn(2), Op::IsNull, Op::Return];
        let prog = Program::new(ops);
        assert_eq!(prog.len(), 2);
        assert!(matches!(prog.ops()[0], Op::IsNullColumn(2)));
    }

    #[test]
    fn test_peephole_between_column_const() {
        let ops = vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(1)),
            Op::LoadConst(Value::Integer(100)),
            Op::Between,
            Op::Return,
        ];
        let prog = Program::new(ops);
        // Should fuse 4 ops to 1 BetweenColumnConst
        assert_eq!(prog.len(), 2);
        assert!(matches!(prog.ops()[0], Op::BetweenColumnConst(0, _, _)));
    }

    #[test]
    fn test_peephole_no_fusion_when_not_applicable() {
        // Can't fuse when pattern doesn't match
        let ops = vec![
            Op::LoadConst(Value::Integer(5)),
            Op::LoadColumn(0),
            Op::Eq,
            Op::Return,
        ];
        let prog = Program::new(ops);
        // LoadConst + LoadColumn + Eq doesn't match (order is wrong)
        assert!(prog.len() >= 3);
    }

    #[test]
    fn test_peephole_preserves_jumps() {
        // Ensure jump targets aren't fused over
        let ops = vec![
            Op::LoadColumn(0),
            Op::JumpIfFalse(3), // Jump to position 3
            Op::LoadConst(Value::Integer(5)),
            Op::Eq, // This is position 3 - a jump target
            Op::Return,
        ];
        let prog = Program::new(ops);
        // The fusion should not break jump semantics
        assert!(prog.len() >= 3);
    }

    // =========================================================================
    // ProgramBuilder
    // =========================================================================

    #[test]
    fn test_builder_basic() {
        let mut builder = ProgramBuilder::new();
        builder.emit(Op::LoadConst(Value::Integer(42)));
        builder.emit(Op::Return);

        let prog = builder.build();
        assert!(!prog.is_empty());
    }

    #[test]
    fn test_builder_position() {
        let mut builder = ProgramBuilder::new();
        assert_eq!(builder.position(), 0);

        builder.emit(Op::LoadColumn(0));
        assert_eq!(builder.position(), 1);

        builder.emit(Op::LoadColumn(1));
        assert_eq!(builder.position(), 2);
    }

    #[test]
    fn test_builder_patch_jump() {
        let mut builder = ProgramBuilder::new();
        builder.emit(Op::LoadColumn(0));
        builder.emit(Op::JumpIfFalse(0)); // Placeholder target
        let jump_pos = 1;
        builder.emit(Op::LoadConst(Value::Integer(1)));
        builder.emit(Op::Return);
        let end_pos = builder.position();

        builder.patch_jump(jump_pos, end_pos);

        let prog = builder.build();
        // After optimization, check that jump exists
        let has_jump = prog.ops().iter().any(|op| matches!(op, Op::JumpIfFalse(_)));
        assert!(has_jump || prog.len() < 4); // Either has jump or was optimized away
    }

    #[test]
    fn test_builder_default() {
        let builder: ProgramBuilder = Default::default();
        let prog = builder.build();
        // Empty program still has max_stack_depth of 1
        assert_eq!(prog.max_stack_depth(), 1);
    }

    // =========================================================================
    // Disassemble
    // =========================================================================

    #[test]
    fn test_disassemble() {
        let prog = Program::new_unoptimized(vec![
            Op::LoadColumn(0),
            Op::LoadConst(Value::Integer(5)),
            Op::Eq,
            Op::Return,
        ]);
        let disasm = prog.disassemble();
        assert!(disasm.contains("LoadColumn"));
        assert!(disasm.contains("LoadConst"));
        assert!(disasm.contains("Eq"));
        assert!(disasm.contains("Return"));
        // Check format has line numbers
        assert!(disasm.contains("0000:"));
        assert!(disasm.contains("0001:"));
    }

    // =========================================================================
    // Debug formatting
    // =========================================================================

    #[test]
    fn test_program_debug() {
        let prog = Program::constant(Value::Integer(42));
        let debug_str = format!("{:?}", prog);
        assert!(debug_str.contains("Program"));
        assert!(debug_str.contains("ops_count"));
        assert!(debug_str.contains("max_stack_depth"));
    }

    #[test]
    fn a_worker_copy_matches_by_a_regex_of_its_own() {
        let regex = std::sync::Arc::new(regex::Regex::new("^a").unwrap());
        let prog = Program::new(vec![
            Op::LoadColumn(0),
            Op::Regexp(std::sync::Arc::clone(&regex)),
            Op::Return,
        ]);
        assert!(prog.holds_thread_state());
        let copy = prog.worker_copy();
        assert_eq!(copy.disassemble(), prog.disassemble());
        let Op::Regexp(copied) = &copy.ops()[1] else {
            panic!("the copy keeps the REGEXP at its place");
        };
        assert!(
            !std::sync::Arc::ptr_eq(copied, &regex),
            "the copy owns its regex"
        );
        assert_eq!(copied.as_str(), "^a");
        assert!(!Program::constant(Value::Integer(1)).holds_thread_state());
    }

    #[test]
    fn values_read_as_the_column_bind_into_the_in_list_after_them() {
        let program = Program::new_unoptimized(vec![
            Op::LoadColumn(0),
            Op::LoadParam(0),
            Op::ReadAsColumn(crate::core::DataType::Integer),
            Op::LoadParam(1),
            Op::ReadAsColumn(crate::core::DataType::Integer),
            Op::InList(2),
            Op::Return,
        ]);
        let bound = program
            .bind(&[Value::Integer(6), Value::Float(1.0)])
            .expect("the parameters bind");
        let listing = format!("{:?}", bound.ops());
        assert!(
            !listing.contains("InList"),
            "the list binds to a set test: {listing}"
        );
    }
}
