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

mod compiler;
mod evaluator_bridge;
#[cfg(test)]
pub(crate) use evaluator_bridge::program_cached;
mod once;
mod ops;
mod program;
mod vm;

pub(crate) use compiler::literal_value;
pub use compiler::{CompileContext, CompileError, ExprCompiler};
pub use evaluator_bridge::{
    clear_program_cache, compile_expression, compile_expression_with_context,
    compute_expression_hash, CompiledEvaluator, ExpressionEval, JoinFilter, RowFilter,
    SharedProgram,
};
pub(crate) use once::{
    compile_once, evaluate_once, execute_context, execute_once, try_eval_constant_expr_with_context,
};
pub use ops::{CompareOp, Op, QuantifiedMembers};
pub use program::Program;
pub use vm::{ExecuteContext, ExprVM, SubqueryExecutor};

#[cfg(test)]
mod tests;
