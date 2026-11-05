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

// Compiled Expression VM - Complete Replacement for AST Evaluator
//
// This module provides a high-performance expression evaluation system that
// completely replaces the recursive AST evaluator with a compiled, stack-based VM.
//
// Design Goals:
// 1. ZERO RECURSION - All expressions compile to linear instruction sequences
// 2. MINIMAL ALLOCATION - Reuse stack, pre-allocate everything possible
// 3. FAST DISPATCH - Direct enum match, no string comparisons
// 4. COMPLETE COVERAGE - Handle ALL expression types including subqueries
//
// Architecture:
//
//   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
//   │ Expression  │ ──► │ ExprCompiler │ ──► │   Program   │
//   │    (AST)    │     │              │     │  (bytecode) │
//   └─────────────┘     └──────────────┘     └─────────────┘
//                                                   │
//                                                   ▼
//   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
//   │   Result    │ ◄── │    ExprVM    │ ◄── │  Row Data   │
//   │   (Value)   │     │              │     │             │
//   └─────────────┘     └──────────────┘     └─────────────┘

mod bound_program;
mod compiler;
mod join_filter;
mod once;
mod ops;
mod program;
mod row_filter;
mod vm;

pub(crate) use compiler::literal_value;
pub(crate) use compiler::parameter_slot;
pub use compiler::{CompileContext, CompileError, ExprCompiler};
pub use join_filter::JoinFilter;
pub(crate) use once::{
    compile_once, evaluate_once, evaluate_once_in, execute_context, execute_once, held_once,
    held_or_compile, try_eval_constant_expr_with_context, HeldOnce,
};
pub(crate) use ops::more_than_one_row;
pub use ops::{CompareOp, Op, QuantifiedMembers, RowComparison, RowMembers, RowTest};
pub(crate) use program::known_params;
pub use program::{Program, SharedProgram};
pub use row_filter::RowFilter;
pub(crate) use vm::{concat_values, run_params};
pub use vm::{ExecuteContext, ExprVM, SubqueryExecutor};

#[cfg(test)]
mod tests;
