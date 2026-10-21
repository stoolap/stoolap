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

// CompiledEvaluator Bridge
//
// Provides an Evaluator-compatible API using the Expression VM internally.
// This allows gradual migration from AST-based evaluation to bytecode execution.
//
// Design:
// - Matches Evaluator's public API (new, init_columns, set_row_array, evaluate)
// - Uses per-evaluator local cache for compiled programs
// - Uses ExprVM for execution
//
// Performance Optimization:
// - For closure-based filtering, use `RowFilter` instead of creating evaluators per-row
// - `RowFilter` pre-compiles the expression once and shares `CompactArc<Program>` across threads
// - The VM is lightweight and can be created per-thread without performance penalty

use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::api::params::ParamVec;

// Shared empty defaults: constructors hand these out with an Arc bump
// instead of allocating fresh empty collections per instance
static EMPTY_PARAMS: std::sync::LazyLock<CompactArc<ParamVec>> =
    std::sync::LazyLock::new(|| CompactArc::new(ParamVec::new()));
static EMPTY_NAMED_PARAMS: std::sync::LazyLock<Arc<FxHashMap<String, Value>>> =
    std::sync::LazyLock::new(|| Arc::new(FxHashMap::default()));
use crate::common::{CompactArc, StringMap};
use crate::executor::expr_key::ExprKey;
use lru::LruCache;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHasher};

use super::compiler::{CompileContext, ExprCompiler};
use super::program::Program;
use super::vm::{ExecuteContext, ExprVM};
use crate::core::{Error, Result, Row, Value};
use crate::functions::{global_registry, FunctionRegistry};
use crate::parser::ast::Expression;

use crate::executor::context::ExecutionContext;

// ============================================================================
// PROGRAM CACHE - Global cache for compiled expression programs
// ============================================================================

/// Maximum number of cached programs (LRU eviction)
const PROGRAM_CACHE_SIZE: usize = 256;

/// Global cache for compiled programs using O(1) LRU eviction.
/// Uses parking_lot::Mutex for efficient locking.
static PROGRAM_CACHE: Mutex<Option<LruCache<(u64, u64), SharedProgram>>> = Mutex::new(None);

/// Clear the program cache. Call on database drop to release memory.
pub fn clear_program_cache() {
    let mut guard = PROGRAM_CACHE.lock();
    *guard = None;
}

/// Whether the process-wide cache holds a program for `expr` over `columns`
#[cfg(test)]
pub(crate) fn program_cached(expr: &Expression, columns: &[String]) -> bool {
    let key = compute_cache_key(expr, columns);
    PROGRAM_CACHE
        .lock()
        .as_ref()
        .is_some_and(|cache| cache.contains(&key))
}

/// Compute cache key from expression and columns using efficient recursive hashing.
/// This avoids the overhead of Debug formatting by directly hashing expression structure.
///
/// The key is a PAIR of hashes from independent algorithms (FxHash and
/// ahash with fixed seeds). The cache has no equality check on hit, so a
/// single 64-bit FxHash key would silently execute the wrong compiled
/// program on a collision; FxHash's weak diffusion makes structured
/// collisions realistic. A simultaneous collision in both algorithms is
/// not.
fn compute_cache_key(expr: &Expression, columns: &[String]) -> (u64, u64) {
    let mut h1 = FxHasher::default();
    hash_expression(expr, &mut h1);
    columns.hash(&mut h1);

    use std::hash::BuildHasher;
    // Fixed seeds: the key must be stable across calls within the process.
    let mut h2 = ahash::RandomState::with_seeds(
        0x9e37_79b9_7f4a_7c15,
        0xf39c_c060_5ced_c834,
        0x1082_276b_f3a2_7251,
        0x8f4c_a136_bef1_39c9,
    )
    .build_hasher();
    hash_expression(expr, &mut h2);
    columns.hash(&mut h2);

    (h1.finish(), h2.finish())
}

/// Each aliased slot at the width the program loads it with
fn alias_slots(aliases: &FxHashMap<ExprKey, usize>) -> FxHashMap<ExprKey, u16> {
    aliases
        .iter()
        .map(|(&key, &slot)| (key, slot as u16))
        .collect()
}

/// Compute a u64 hash of an expression without string allocation.
/// This is O(expression_size) and avoids Debug formatting overhead.
/// Use this for cache keys instead of format!("{:?}", expr).
/// Uses FxHasher which is 2-5x faster than SipHash for small keys.
#[inline]
pub fn compute_expression_hash(expr: &Expression) -> u64 {
    let mut hasher = FxHasher::default();
    hash_expression(expr, &mut hasher);
    hasher.finish()
}

/// Recursively hash an expression without string allocation.
/// This is O(expression_size) and avoids Debug formatting overhead.
fn hash_expression<H: std::hash::Hasher>(expr: &Expression, hasher: &mut H) {
    // First hash the discriminant to distinguish variants
    std::mem::discriminant(expr).hash(hasher);

    match expr {
        Expression::Identifier(id) => {
            id.value_lower.hash(hasher);
            // A quoted name compiles differently (no TRUE/FALSE literal, and
            // its exact text when no column has that name)
            id.quoted.hash(hasher);
            if id.quoted {
                id.value.hash(hasher);
            }
        }
        Expression::QualifiedIdentifier(qid) => {
            qid.qualifier.value_lower.hash(hasher);
            qid.name.value_lower.hash(hasher);
        }
        Expression::IntegerLiteral(lit) => {
            lit.value.hash(hasher);
        }
        Expression::FloatLiteral(lit) => {
            lit.value.to_bits().hash(hasher);
        }
        Expression::StringLiteral(lit) => {
            lit.value.hash(hasher);
            lit.type_hint.hash(hasher);
        }
        Expression::BooleanLiteral(lit) => {
            lit.value.hash(hasher);
        }
        Expression::NullLiteral(_) => {
            // Just discriminant is enough
        }
        Expression::IntervalLiteral(lit) => {
            lit.value.hash(hasher);
            lit.unit.hash(hasher);
        }
        Expression::Parameter(param) => {
            param.index.hash(hasher);
            param.name.hash(hasher);
        }
        Expression::Prefix(prefix) => {
            std::mem::discriminant(&prefix.op_type).hash(hasher);
            hash_expression(&prefix.right, hasher);
        }
        Expression::Infix(infix) => {
            std::mem::discriminant(&infix.op_type).hash(hasher);
            hash_expression(&infix.left, hasher);
            hash_expression(&infix.right, hasher);
        }
        Expression::List(list) => {
            list.elements.len().hash(hasher);
            for val in &list.elements {
                hash_expression(val, hasher);
            }
        }
        Expression::Distinct(dist) => {
            hash_expression(&dist.expr, hasher);
        }
        Expression::Exists(exists) => {
            // Use pointer identity for hashing - avoids expensive Debug format allocation
            // The subquery AST is stable during query execution
            (exists.subquery.as_ref() as *const _ as usize).hash(hasher);
        }
        Expression::AllAny(aa) => {
            aa.operator.hash(hasher);
            std::mem::discriminant(&aa.all_any_type).hash(hasher);
            hash_expression(&aa.left, hasher);
            // Use pointer identity for hashing - avoids expensive Debug format allocation
            (aa.subquery.as_ref() as *const _ as usize).hash(hasher);
        }
        Expression::In(in_expr) => {
            in_expr.not.hash(hasher);
            hash_expression(&in_expr.left, hasher);
            hash_expression(&in_expr.right, hasher);
        }
        Expression::InHashSet(in_hash) => {
            in_hash.not.hash(hasher);
            hash_expression(&in_hash.column, hasher);
            in_hash.values.len().hash(hasher);
            // Hash the actual set contents: length alone made `a IN {1}`
            // and `a IN {2}` share a key, and no key derivation can help
            // when its input omits the values. Canonical (sorted) order
            // keeps the hash set-order-independent while every algorithm
            // hashing this expression sees the full material directly.
            let mut sorted: Vec<&crate::core::Value> = in_hash.values.members().iter().collect();
            sorted.sort_by(|a, b| crate::executor::utils::compare_values(a, b));
            for v in sorted {
                v.hash(hasher);
            }
        }
        Expression::QuantifiedSet(quantified) => {
            quantified.operator.hash(hasher);
            quantified.all.hash(hasher);
            quantified.has_null.hash(hasher);
            hash_expression(&quantified.left, hasher);
            quantified.members.len().hash(hasher);
            for v in quantified.members.iter() {
                v.hash(hasher);
            }
        }
        Expression::Between(between) => {
            between.not.hash(hasher);
            hash_expression(&between.expr, hasher);
            hash_expression(&between.lower, hasher);
            hash_expression(&between.upper, hasher);
        }
        Expression::Like(like) => {
            like.operator.hash(hasher);
            hash_expression(&like.left, hasher);
            hash_expression(&like.pattern, hasher);
            if let Some(ref escape) = like.escape {
                true.hash(hasher);
                hash_expression(escape, hasher);
            } else {
                false.hash(hasher);
            }
        }
        Expression::ScalarSubquery(sq) => {
            // Use pointer identity for hashing - avoids expensive Debug format allocation
            (sq.subquery.as_ref() as *const _ as usize).hash(hasher);
        }
        Expression::ExpressionList(list) => {
            list.expressions.len().hash(hasher);
            for e in &list.expressions {
                hash_expression(e, hasher);
            }
        }
        Expression::Case(case) => {
            if let Some(ref val) = case.value {
                true.hash(hasher);
                hash_expression(val, hasher);
            } else {
                false.hash(hasher);
            }
            case.when_clauses.len().hash(hasher);
            for when_clause in &case.when_clauses {
                hash_expression(&when_clause.condition, hasher);
                hash_expression(&when_clause.then_result, hasher);
            }
            if let Some(ref else_val) = case.else_value {
                true.hash(hasher);
                hash_expression(else_val, hasher);
            } else {
                false.hash(hasher);
            }
        }
        Expression::Cast(cast) => {
            hash_expression(&cast.expr, hasher);
            cast.type_name.hash(hasher);
        }
        Expression::FunctionCall(func) => {
            func.function.hash(hasher);
            func.is_distinct.hash(hasher);
            func.arguments.len().hash(hasher);
            for arg in &func.arguments {
                hash_expression(arg, hasher);
            }
            if let Some(ref filter) = func.filter {
                true.hash(hasher);
                hash_expression(filter, hasher);
            } else {
                false.hash(hasher);
            }
        }
        Expression::Aliased(aliased) => {
            aliased.alias.value_lower.hash(hasher);
            hash_expression(&aliased.expression, hasher);
        }
        Expression::Window(window) => {
            window.function.function.hash(hasher);
            window.function.is_distinct.hash(hasher);
            window.function.arguments.len().hash(hasher);
            for arg in &window.function.arguments {
                hash_expression(arg, hasher);
            }
            window.partition_by.len().hash(hasher);
            for e in &window.partition_by {
                hash_expression(e, hasher);
            }
            window.order_by.len().hash(hasher);
            for order in &window.order_by {
                hash_expression(&order.expression, hasher);
                order.ascending.hash(hasher);
                order.nulls_first.hash(hasher);
            }
        }
        Expression::TableSource(ts) => {
            ts.name.value_lower.hash(hasher);
            if let Some(ref alias) = ts.alias {
                true.hash(hasher);
                alias.value_lower.hash(hasher);
            } else {
                false.hash(hasher);
            }
        }
        Expression::JoinSource(js) => {
            // Use pointer identity for hashing - avoids expensive Debug format allocation
            (js.as_ref() as *const _ as usize).hash(hasher);
        }
        Expression::SubquerySource(sq) => {
            if let Some(ref alias) = sq.alias {
                true.hash(hasher);
                alias.value_lower.hash(hasher);
            } else {
                false.hash(hasher);
            }
            // Use pointer identity for hashing - avoids expensive Debug format allocation
            (sq.subquery.as_ref() as *const _ as usize).hash(hasher);
        }
        Expression::ValuesSource(vs) => {
            if let Some(ref alias) = vs.alias {
                true.hash(hasher);
                alias.value_lower.hash(hasher);
            } else {
                false.hash(hasher);
            }
            vs.rows.len().hash(hasher);
            for row in &vs.rows {
                row.len().hash(hasher);
                for e in row {
                    hash_expression(e, hasher);
                }
            }
        }
        Expression::CteReference(cte) => {
            cte.name.value_lower.hash(hasher);
        }
        Expression::FunctionTableSource(fts) => {
            fts.function.value_lower.hash(hasher);
            for arg in &fts.arguments {
                hash_expression(arg, hasher);
            }
        }
        Expression::Star(_) => {
            // Just discriminant
        }
        Expression::QualifiedStar(qs) => {
            qs.qualifier.hash(hasher);
        }
        Expression::Default(_) => {
            // Just discriminant
        }
    }
}

/// Try to get a cached program, or compile and cache it.
/// Uses O(1) LRU cache with parking_lot::Mutex for efficient concurrent access.
fn compile_expression_cached(expr: &Expression, columns: &[String]) -> Result<SharedProgram> {
    #[cfg(feature = "test-failpoints")]
    crate::test_failpoints::bridge_compiled();
    let cache_key = compute_cache_key(expr, columns);

    // Try cache first (O(1) lookup and LRU update)
    {
        let mut guard = PROGRAM_CACHE.lock();
        let cache = guard.get_or_insert_with(|| {
            // SAFETY: PROGRAM_CACHE_SIZE is always > 0
            LruCache::new(NonZeroUsize::new(PROGRAM_CACHE_SIZE).unwrap())
        });
        if let Some(program) = cache.get(&cache_key) {
            return Ok(program.clone());
        }
    }

    // Cache miss - compile the expression (outside lock to avoid blocking)
    let ctx = CompileContext::with_global_registry(columns);
    let compiler = ExprCompiler::new(&ctx);
    let program: SharedProgram = compiler
        .compile(expr)
        .map(CompactArc::new)
        .map_err(|e| Error::internal(format!("Compile error: {}", e)))?;

    // Store in cache (O(1) insertion with automatic LRU eviction)
    {
        let mut guard = PROGRAM_CACHE.lock();
        let cache = guard
            .get_or_insert_with(|| LruCache::new(NonZeroUsize::new(PROGRAM_CACHE_SIZE).unwrap()));
        cache.put(cache_key, program.clone());
    }

    Ok(program)
}

// ============================================================================
// STANDALONE COMPILATION FUNCTIONS
// ============================================================================

/// Compile an expression to a program for a given column schema.
///
/// This is the recommended way to compile expressions for use in closures
/// or parallel execution. The returned `CompactArc<Program>` is `Send + Sync` and
/// can be shared across threads efficiently.
///
/// **Note:** Results are cached globally for performance. Repeated calls
/// with the same expression and columns will return the cached program.
///
/// # Arguments
/// * `expr` - The expression to compile
/// * `columns` - Column names for the schema
///
/// # Returns
/// * `CompactArc<Program>` that can be executed with `RowFilter` or `ExprVM`
pub fn compile_expression(expr: &Expression, columns: &[String]) -> Result<SharedProgram> {
    compile_expression_cached(expr, columns)
}

/// Compile an expression with full context (parameters, outer columns, etc.)
///
/// Use this when you need parameters or correlated subquery support.
pub fn compile_expression_with_context(
    expr: &Expression,
    columns: &[String],
    outer_columns: Option<&[String]>,
    function_registry: &FunctionRegistry,
) -> Result<SharedProgram> {
    let mut ctx = CompileContext::new(columns, function_registry);
    if let Some(outer_cols) = outer_columns {
        ctx = ctx.with_outer_columns(outer_cols);
    }
    let compiler = ExprCompiler::new(&ctx);
    compiler
        .compile(expr)
        .map(CompactArc::new)
        .map_err(|e| Error::internal(format!("Compile error: {}", e)))
}

// ============================================================================
// ROW FILTER - Lightweight, Send+Sync filter for closures
// ============================================================================

/// A lightweight, thread-safe row filter for closure-based filtering.
///
/// `RowFilter` pre-compiles the expression once and can be cloned cheaply
/// (it uses `CompactArc<Program>` internally). Each thread should create its own
/// `ExprVM` for execution.
///
/// # Example
/// ```ignore
/// // Create filter once
/// let filter = RowFilter::new(&where_expr, &columns)?;
///
/// // Use in closure (filter is cloned into closure)
/// let predicate = move |row: &Row| filter.matches(row);
///
/// // Or use with parallel iteration
/// rows.par_iter().filter(|row| filter.matches(row)).collect()
/// ```
#[derive(Clone)]
pub struct RowFilter {
    /// Pre-compiled program (shared across clones)
    program: SharedProgram,
    /// Query parameters (shared) - uses CompactArc<Vec<Value>> to match ExecutionContext
    params: CompactArc<ParamVec>,
    /// Named parameters (shared)
    named_params: Arc<FxHashMap<String, Value>>,
    /// Transaction ID for CURRENT_TRANSACTION_ID()
    transaction_id: Option<u64>,
}

impl RowFilter {
    /// Create a new row filter by compiling the given expression.
    ///
    /// # Arguments
    /// * `expr` - The boolean expression to use as filter
    /// * `columns` - Column names matching the row schema
    pub fn new(expr: &Expression, columns: &[String]) -> Result<Self> {
        Ok(Self::from_program(compile_expression(expr, columns)?))
    }

    /// A filter over a program compiled earlier for the same columns
    pub fn from_program(program: SharedProgram) -> Self {
        Self {
            program,
            params: EMPTY_PARAMS.clone(),
            named_params: EMPTY_NAMED_PARAMS.clone(),
            transaction_id: None,
        }
    }

    /// Create a row filter whose group items and aggregates are read from
    /// the row's slot for each expression's identity (for HAVING)
    pub(crate) fn with_aliases(
        expr: &Expression,
        columns: &[String],
        aliases: &FxHashMap<ExprKey, usize>,
    ) -> Result<Self> {
        let ctx = CompileContext::with_global_registry(columns)
            .with_expression_aliases(alias_slots(aliases));
        let compiler = ExprCompiler::new(&ctx);
        let program = compiler
            .compile(expr)
            .map(CompactArc::new)
            .map_err(|e| Error::internal(format!("Compile error: {}", e)))?;

        Ok(Self {
            program,
            params: EMPTY_PARAMS.clone(),
            named_params: EMPTY_NAMED_PARAMS.clone(),
            transaction_id: None,
        })
    }

    /// Create a filter from execution context.
    ///
    /// PERF: Both `params` and `named_params` share the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed (both use CompactArc<Vec<Value>>)
        self.params = CompactArc::clone(ctx.params_arc());
        // Share named_params Arc - no cloning needed
        self.named_params = Arc::clone(ctx.named_params_arc());
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// Check a row against the filter, returning an evaluation error instead of swallowing it.
    ///
    /// Returns `Err` when the VM encounters a runtime error (e.g. invalid
    /// REGEXP pattern supplied via a parameter). Used by FilteredResult
    /// to surface errors through the Rows iterator.
    #[inline]
    pub fn matches_checked(&self, row: &Row) -> Result<bool> {
        thread_local! {
            static VM: std::cell::RefCell<ExprVM> = std::cell::RefCell::new(ExprVM::new());
        }

        VM.with(|vm| {
            let mut ctx = ExecuteContext::new(row);

            if !self.params.is_empty() {
                ctx = ctx.with_params(&self.params);
            }
            if !self.named_params.is_empty() {
                ctx = ctx.with_named_params(&self.named_params);
            }
            ctx = ctx.with_transaction_id(self.transaction_id);

            if let Ok(mut borrowed_vm) = vm.try_borrow_mut() {
                borrowed_vm.execute_bool_checked(&self.program, &ctx)
            } else {
                let mut temp_vm = ExprVM::new();
                temp_vm.execute_bool_checked(&self.program, &ctx)
            }
        })
    }

    /// Filter a RowVec in-place, removing rows that don't match.
    /// Returns Err if the filter expression produces a runtime error
    /// (e.g. invalid REGEXP pattern supplied via a parameter).
    pub fn retain_checked(&self, rows: &mut crate::core::RowVec) -> Result<()> {
        let mut error: Option<crate::core::Error> = None;
        rows.retain(|(_, row)| {
            if error.is_some() {
                return false;
            }
            match self.matches_checked(row) {
                Ok(b) => b,
                Err(e) => {
                    error = Some(e);
                    false
                }
            }
        });
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

// Static assertions to verify RowFilter implements Send + Sync.
// This is safer than unsafe impl because it will fail at compile time
// if any field doesn't implement Send/Sync, rather than causing UB at runtime.
// All fields are Send + Sync:
// - CompactArc<Program> is Send + Sync (Program is immutable)
// - CompactArc<Value> is Send + Sync
// - Arc<FxHashMap<String, Value>> is Send + Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<RowFilter>;
};

// ============================================================================
// JOIN FILTER - For join condition evaluation
// ============================================================================

/// A filter for join condition evaluation between two rows.
#[derive(Clone)]
pub struct JoinFilter {
    /// Pre-compiled program
    program: SharedProgram,
    /// Query parameters (shared Arc to avoid cloning)
    params: CompactArc<ParamVec>,
    /// Named parameters (shared Arc to avoid cloning)
    named_params: Arc<FxHashMap<String, Value>>,
    /// Transaction ID for CURRENT_TRANSACTION_ID()
    transaction_id: Option<u64>,
}

impl JoinFilter {
    /// Create a join filter by compiling the condition.
    ///
    /// # Arguments
    /// * `expr` - The join condition expression
    /// * `left_columns` - Column names for the left table
    /// * `right_columns` - Column names for the right table
    pub fn new(
        expr: &Expression,
        left_columns: &[String],
        right_columns: &[String],
        function_registry: &FunctionRegistry,
    ) -> Result<Self> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::bridge_compiled();
        let ctx =
            CompileContext::new(left_columns, function_registry).with_second_row(right_columns);
        let compiler = ExprCompiler::new(&ctx);
        let program = compiler
            .compile(expr)
            .map_err(|e| Error::internal(format!("Compile error: {}", e)))?;
        Ok(Self {
            program: CompactArc::new(program),
            params: EMPTY_PARAMS.clone(),
            named_params: EMPTY_NAMED_PARAMS.clone(),
            transaction_id: None,
        })
    }

    /// A filter over a program compiled earlier for the same column lists
    pub fn from_program(program: SharedProgram) -> Self {
        Self {
            program,
            params: EMPTY_PARAMS.clone(),
            named_params: EMPTY_NAMED_PARAMS.clone(),
            transaction_id: None,
        }
    }

    /// Set parameters and the transaction from the execution context.
    /// Required when the join condition holds parameter placeholders
    /// ($1, $2, ...) or reads CURRENT_TRANSACTION_ID().
    #[inline]
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        self.params = CompactArc::clone(ctx.params_arc());
        self.named_params = Arc::clone(ctx.named_params_arc());
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// Check if a pair of rows satisfies the join condition. An error in it
    /// reads as false, sets `failed` and is kept on this thread until
    /// `take_error`
    #[inline]
    pub fn matches(&self, left_row: &Row, right_row: &Row, failed: &mut bool) -> bool {
        JOIN_VM.with(|vm| {
            let mut ctx = ExecuteContext::for_join(left_row, right_row);

            // Apply params if present (required for parameter placeholders like $1, $2)
            if !self.params.is_empty() {
                ctx = ctx.with_params(&self.params);
            }
            if !self.named_params.is_empty() {
                ctx = ctx.with_named_params(&self.named_params);
            }
            ctx = ctx.with_transaction_id(self.transaction_id);

            // Use try_borrow_mut to avoid panic on recursive calls (e.g., nested subqueries).
            // If the VM is already borrowed, create a temporary one for this call.
            if let Ok(mut borrowed_vm) = vm.try_borrow_mut() {
                let matched = borrowed_vm.execute_bool(&self.program, &ctx);
                if borrowed_vm.has_bool_error() {
                    *failed = true;
                }
                matched
            } else {
                // Fallback: create a fresh VM for recursive calls
                let mut temp_vm = ExprVM::new();
                let matched = temp_vm.execute_bool(&self.program, &ctx);
                if let Some(e) = temp_vm.take_bool_error() {
                    JOIN_FALLBACK_ERROR.with(|slot| {
                        slot.borrow_mut().get_or_insert(e);
                    });
                    *failed = true;
                }
                matched
            }
        })
    }

    /// The first error a condition on this thread read as false since the
    /// last call. A join clears it when it opens
    pub fn take_error(&self) -> Option<crate::core::Error> {
        JOIN_FALLBACK_ERROR
            .with(|slot| slot.borrow_mut().take())
            .or_else(|| JOIN_VM.with(|vm| vm.try_borrow_mut().ok()?.take_bool_error()))
    }

    /// The error behind a `failed` flag that `matches` set
    #[cold]
    pub fn failure(&self) -> crate::core::Error {
        self.take_error().unwrap_or_else(|| {
            crate::core::Error::internal("a join condition failed without an error")
        })
    }

    /// Get the underlying program.
    pub fn program(&self) -> &SharedProgram {
        &self.program
    }
}

thread_local! {
    /// The VM join conditions run in on this thread
    static JOIN_VM: std::cell::RefCell<ExprVM> = std::cell::RefCell::new(ExprVM::new());
    /// An error from a join condition run on a temporary VM, for `take_error`
    static JOIN_FALLBACK_ERROR: std::cell::RefCell<Option<crate::core::Error>> =
        const { std::cell::RefCell::new(None) };
}

// Static assertions to verify JoinFilter implements Send + Sync.
// This is safer than unsafe impl because it will fail at compile time
// if any field doesn't implement Send/Sync, rather than causing UB at runtime.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<JoinFilter>;
};

// ============================================================================
// EXPRESSION EVAL - Direct VM usage for maximum performance
// ============================================================================

/// Lightweight expression evaluator using direct VM execution.
///
/// `ExpressionEval` provides the simplest possible API for expression evaluation:
/// 1. Compile the expression once with `new()`
/// 2. Evaluate rows with `eval()` or `eval_bool()`
///
/// This is the recommended replacement for `CompiledEvaluator` when you have
/// a single expression to evaluate repeatedly.
///
/// # Example
/// ```ignore
/// // Compile once
/// let eval = ExpressionEval::compile(&expr, &columns)?;
///
/// // Evaluate many rows
/// for row in rows {
///     let value = eval.eval(&row)?;
///     // or for boolean: let matches = eval.eval_bool(&row);
/// }
/// ```
pub struct ExpressionEval {
    /// Pre-compiled program
    program: SharedProgram,
    /// VM instance (reusable, maintains stack)
    vm: ExprVM,
    /// Query parameters (shared) - uses CompactArc<Vec<Value>> to match ExecutionContext
    params: CompactArc<ParamVec>,
    /// Named parameters (shared) - uses Arc to match ExecutionContext
    named_params: Arc<FxHashMap<String, Value>>,
    /// Outer row context for correlated subqueries
    outer_row: Option<FxHashMap<CompactArc<str>, Value>>,
    /// Transaction ID
    transaction_id: Option<u64>,
}

impl ExpressionEval {
    /// Compile an expression for evaluation.
    pub fn compile(expr: &Expression, columns: &[String]) -> Result<Self> {
        let program = compile_expression(expr, columns)?;
        Ok(Self {
            program,
            vm: ExprVM::new(),
            params: EMPTY_PARAMS.clone(),
            named_params: EMPTY_NAMED_PARAMS.clone(),
            outer_row: None,
            transaction_id: None,
        })
    }

    /// Set context from ExecutionContext.
    ///
    /// PERF: Both `params` and `named_params` share the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed
        self.params = CompactArc::clone(ctx.params_arc());
        // Share named_params Arc - no cloning needed
        self.named_params = Arc::clone(ctx.named_params_arc());
        if let Some(outer) = ctx.outer_row() {
            // Clone the map directly (CompactArc<str> clones are cheap)
            let arc_map = outer.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            self.outer_row = Some(arc_map);
        }
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// Evaluate the expression for a row.
    #[inline]
    pub fn eval(&mut self, row: &Row) -> Result<Value> {
        let mut ctx = ExecuteContext::new(row);

        if !self.params.is_empty() {
            ctx = ctx.with_params(&self.params);
        }
        if !self.named_params.is_empty() {
            ctx = ctx.with_named_params(&self.named_params);
        }
        if let Some(ref outer) = self.outer_row {
            ctx = ctx.with_outer_row(outer);
        }
        ctx = ctx.with_transaction_id(self.transaction_id);

        self.vm.execute_cow(&self.program, &ctx)
    }

    /// Evaluate as boolean (for WHERE/HAVING), returning an evaluation error.
    #[inline]
    pub fn eval_bool_checked(&mut self, row: &Row) -> Result<bool> {
        let mut ctx = ExecuteContext::new(row);

        if !self.params.is_empty() {
            ctx = ctx.with_params(&self.params);
        }
        if !self.named_params.is_empty() {
            ctx = ctx.with_named_params(&self.named_params);
        }
        if let Some(ref outer) = self.outer_row {
            ctx = ctx.with_outer_row(outer);
        }
        ctx = ctx.with_transaction_id(self.transaction_id);

        self.vm.execute_bool_checked(&self.program, &ctx)
    }

    /// Evaluate with a row reference.
    #[inline]
    pub fn eval_slice(&mut self, row: &Row) -> Result<Value> {
        let mut ctx = ExecuteContext::new(row);

        if !self.params.is_empty() {
            ctx = ctx.with_params(&self.params);
        }
        if !self.named_params.is_empty() {
            ctx = ctx.with_named_params(&self.named_params);
        }
        if let Some(ref outer) = self.outer_row {
            ctx = ctx.with_outer_row(outer);
        }
        ctx = ctx.with_transaction_id(self.transaction_id);

        self.vm.execute_cow(&self.program, &ctx)
    }
}

// ============================================================================
// MULTI-EXPRESSION EVALUATOR - For SELECT projections
// ============================================================================

/// Evaluates multiple expressions efficiently (for SELECT projections).
///
/// Pre-compiles all expressions once, then evaluates them together for each row.
pub struct MultiExpressionEval {
    /// Pre-compiled programs for each expression
    programs: Vec<SharedProgram>,
    /// Single VM instance (reused for all expressions)
    vm: ExprVM,
    /// Query parameters (shared) - uses CompactArc<Vec<Value>> to match ExecutionContext
    params: CompactArc<ParamVec>,
    /// Named parameters (shared) - uses Arc to match ExecutionContext
    named_params: Arc<FxHashMap<String, Value>>,
    /// Transaction ID
    transaction_id: Option<u64>,
}

impl MultiExpressionEval {
    /// Compile expressions whose group items and aggregates are read from the
    /// row's slot for each expression's identity (for windows over groups)
    pub(crate) fn compile_with_aliases(
        exprs: &[Expression],
        columns: &[String],
        aliases: &FxHashMap<ExprKey, usize>,
    ) -> Result<Self> {
        let ctx = CompileContext::with_global_registry(columns)
            .with_expression_aliases(alias_slots(aliases));
        let compiler = ExprCompiler::new(&ctx);

        let programs = exprs
            .iter()
            .map(|expr| {
                compiler
                    .compile(expr)
                    .map(CompactArc::new)
                    .map_err(|e| Error::internal(format!("Compile error: {}", e)))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            programs,
            vm: ExprVM::new(),
            params: EMPTY_PARAMS.clone(),
            named_params: EMPTY_NAMED_PARAMS.clone(),
            transaction_id: None,
        })
    }

    /// Set from execution context.
    ///
    /// PERF: Both `params` and `named_params` share the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed
        self.params = CompactArc::clone(ctx.params_arc());
        // Share named_params Arc - no cloning needed
        self.named_params = Arc::clone(ctx.named_params_arc());
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// Evaluate all expressions for a row, returning values in order.
    #[inline]
    pub fn eval_all(&mut self, row: &Row) -> Result<Vec<Value>> {
        let mut ctx = ExecuteContext::new(row);

        if !self.params.is_empty() {
            ctx = ctx.with_params(&self.params);
        }
        if !self.named_params.is_empty() {
            ctx = ctx.with_named_params(&self.named_params);
        }
        ctx = ctx.with_transaction_id(self.transaction_id);

        self.programs
            .iter()
            .map(|prog| self.vm.execute_cow(prog, &ctx))
            .collect()
    }

    /// Evaluate all expressions, writing results into provided buffer.
    #[inline]
    pub fn eval_into(&mut self, row: &Row, output: &mut Vec<Value>) -> Result<()> {
        let mut ctx = ExecuteContext::new(row);

        if !self.params.is_empty() {
            ctx = ctx.with_params(&self.params);
        }
        if !self.named_params.is_empty() {
            ctx = ctx.with_named_params(&self.named_params);
        }
        ctx = ctx.with_transaction_id(self.transaction_id);

        output.clear();
        for prog in &self.programs {
            output.push(self.vm.execute_cow(prog, &ctx)?);
        }
        Ok(())
    }
}

/// Shared program reference for zero-copy caching
pub type SharedProgram = CompactArc<Program>;

// ============================================================================
// COMPILED EVALUATOR - DEPRECATED, use ExpressionEval instead
// ============================================================================

/// Upper bound on programs an evaluator keeps between rows. A statement
/// compiles a few dozen distinct expressions at most; anything past this is
/// row-specific churn from correlated substitution.
const LOCAL_PROGRAM_CACHE_MAX: usize = 128;

/// Compiled expression evaluator using the Expression VM.
///
/// # Deprecated
///
/// **This type is deprecated.** Use the new, more efficient alternatives:
///
/// - [`ExpressionEval`] - For single expression evaluation (most common case)
/// - [`RowFilter`] - For closure-based filtering (Send+Sync safe)
/// - [`JoinFilter`] - For join condition evaluation
/// - [`MultiExpressionEval`] - For SELECT projections (multiple expressions)
///
/// The new APIs pre-compile expressions eagerly rather than lazily, avoiding
/// cache invalidation issues and providing better performance.
///
/// ## Migration Guide
///
/// **Before (CompiledEvaluator):**
/// ```ignore
/// let mut eval = CompiledEvaluator::new(&registry);
/// eval.init_columns(&columns);
/// for row in rows {
///     eval.set_row_array(&row);
///     let value = eval.evaluate(&expr)?;
/// }
/// ```
///
/// **After (ExpressionEval):**
/// ```ignore
/// let mut eval = ExpressionEval::compile(&expr, &columns)?;
/// for row in rows {
///     let value = eval.eval(&row)?;
/// }
/// ```
///
/// # When to use CompiledEvaluator vs new APIs
///
/// **Use the new APIs (recommended for most cases):**
/// - [`ExpressionEval`] - Single expression with static schema
/// - [`RowFilter`] - WHERE clause filtering (thread-safe)
/// - [`MultiExpressionEval`] - SELECT projections (multiple expressions)
///
/// **Use CompiledEvaluator when:**
/// - Expressions change per-row (e.g., after `process_correlated_expression`)
/// - You need dynamic/lazy expression compilation
/// - Complex scenarios with correlated subqueries
pub struct CompiledEvaluator<'a> {
    /// Function registry for compilation
    function_registry: &'a FunctionRegistry,

    /// Column names for compilation context (Arc for zero-copy sharing)
    columns: CompactArc<Vec<String>>,

    /// Outer query columns (for correlated subqueries)
    outer_columns: Option<Vec<String>>,

    /// Query parameters (positional) - uses CompactArc<Vec<Value>> to match ExecutionContext
    params: CompactArc<ParamVec>,

    /// Query parameters (named) - uses Arc to match ExecutionContext
    named_params: Arc<FxHashMap<String, Value>>,

    /// Outer row context for correlated subqueries
    outer_row: Option<FxHashMap<CompactArc<str>, Value>>,

    /// Current transaction ID
    transaction_id: Option<u64>,

    /// Expression aliases for HAVING clause
    expression_aliases: FxHashMap<ExprKey, u16>,

    /// Column aliases
    column_aliases: StringMap<String>,

    /// VM instance (reusable)
    vm: ExprVM,

    /// Local cache: expression hash -> program (fast, no synchronization)
    local_cache: FxHashMap<(u64, u64), SharedProgram>,

    /// Current row values for execution (owned copy for safety)
    current_row: Option<Row>,
}

// CompiledEvaluator is Send + Sync because all fields are Send + Sync:
// - function_registry: &FunctionRegistry is Send + Sync (shared reference to thread-safe registry)
// - All other fields are owned types that are Send + Sync

impl<'a> CompiledEvaluator<'a> {
    /// Create a new compiled evaluator with a function registry reference
    pub fn new(function_registry: &'a FunctionRegistry) -> Self {
        Self {
            function_registry,
            columns: CompactArc::new(Vec::new()),
            outer_columns: None,
            params: EMPTY_PARAMS.clone(),
            named_params: EMPTY_NAMED_PARAMS.clone(),
            outer_row: None,
            transaction_id: None,
            expression_aliases: FxHashMap::default(),
            column_aliases: StringMap::new(),
            vm: ExprVM::new(),
            local_cache: FxHashMap::default(),
            current_row: None,
        }
    }

    /// Create an evaluator using the global function registry.
    pub fn with_defaults() -> CompiledEvaluator<'static> {
        CompiledEvaluator::new(global_registry())
    }

    /// Set parameters from execution context - fluent API
    ///
    /// PERF: Both `params` and `named_params` share the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed
        self.params = CompactArc::clone(ctx.params_arc());
        // Share named_params Arc - no cloning needed
        self.named_params = Arc::clone(ctx.named_params_arc());

        // Set outer row context for correlated subqueries
        if let Some(outer) = ctx.outer_row() {
            // Clone the map directly (CompactArc<str> clones are cheap)
            let arc_map = outer.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            // Convert CompactArc<str> keys to String for outer_columns (needed for compilation)
            let outer_cols: Vec<String> = outer.keys().map(|k| k.to_string()).collect();
            self.outer_row = Some(arc_map);
            // Also set up outer_columns for compilation
            if !outer_cols.is_empty() {
                self.outer_columns = Some(outer_cols);
                // Invalidate local cache since compilation context changed
                self.local_cache.clear();
            }
        }

        self.transaction_id = ctx.transaction_id();
        self
    }

    /// Initialize the column index mapping (call once before set_row_array)
    ///
    /// Performance: when the names are unchanged (e.g. the same table schema
    /// across UPDATE/SELECT operations), the string cloning is skipped. The
    /// comparison is by content: the caller's buffer is not retained, so its
    /// address cannot stand in for its identity.
    pub fn init_columns(&mut self, columns: &[String]) {
        if self.columns.as_slice() == columns {
            return;
        }

        self.columns = CompactArc::new(columns.to_vec());
        // Clear local cache since compilation context changed
        self.local_cache.clear();
    }

    /// Initialize columns from an Arc (zero-copy when schema already has Arc)
    ///
    /// This is the preferred method when the caller already has an CompactArc<Vec<String>>,
    /// such as from `Schema::column_names_arc()`. It avoids all string cloning.
    #[inline]
    pub fn init_columns_arc(&mut self, columns: CompactArc<Vec<String>>) {
        // Both handles are alive here, so pointer equality is exact identity.
        if CompactArc::ptr_eq(&self.columns, &columns) {
            return;
        }

        self.columns = columns;
        // Clear local cache since compilation context changed
        self.local_cache.clear();
    }

    /// Add aggregate expression aliases for HAVING clause evaluation
    pub(crate) fn add_aggregate_aliases(&mut self, aliases: &FxHashMap<ExprKey, usize>) {
        self.expression_aliases.extend(alias_slots(aliases));
        // Invalidate local cache since compilation context changed
        self.local_cache.clear();
    }

    /// Set the row using array-based access (optimized - no map rebuilding)
    /// Call init_columns() once before using this method.
    #[inline]
    pub fn set_row_array(&mut self, row: &Row) {
        self.current_row = Some(row.clone());
    }

    /// Set the outer row context by taking ownership
    /// Accepts CompactArc<str> keys directly to avoid conversion overhead.
    #[inline]
    pub fn set_outer_row_owned(&mut self, outer_row: FxHashMap<CompactArc<str>, Value>) {
        // The compiled programs depend on the outer column *names*, not the
        // values. A correlated loop hands the same key set back every row, so
        // when it matches what was compiled against, the values are swapped
        // in and the cache is kept. Rebuilding the name list and clearing the
        // cache here recompiled every correlated expression on every row.
        let same_columns = match &self.outer_columns {
            Some(cols) => {
                cols.len() == outer_row.len()
                    && outer_row
                        .keys()
                        .all(|k| cols.binary_search_by(|c| c.as_str().cmp(k)).is_ok())
            }
            None => outer_row.is_empty(),
        };
        if !same_columns && !outer_row.is_empty() {
            let mut sorted_cols: Vec<String> = outer_row.keys().map(|k| k.to_string()).collect();
            sorted_cols.sort();
            self.outer_columns = Some(sorted_cols);
            // Invalidate local cache since compilation context changed
            self.local_cache.clear();
        }
        self.outer_row = Some(outer_row);
    }

    /// Take ownership of the outer row back (for reuse)
    /// Returns CompactArc<str> keys directly to avoid conversion overhead.
    #[inline]
    pub fn take_outer_row(&mut self) -> FxHashMap<CompactArc<str>, Value> {
        self.outer_row.take().unwrap_or_default()
    }

    /// Compute hash of expression content for local cache key.
    /// Fast recursive hash that avoids string allocation.
    /// Uses FxHasher which is 2-5x faster than SipHash for small keys.
    #[inline]
    fn expr_hash(&self, expr: &Expression) -> (u64, u64) {
        // Two independent algorithms; see compute_cache_key for why a
        // single unverified 64-bit key is not collision-safe.
        let mut h1 = FxHasher::default();
        Self::hash_expression(expr, &mut h1);

        use std::hash::BuildHasher;
        let mut h2 = ahash::RandomState::with_seeds(
            0x9e37_79b9_7f4a_7c15,
            0xf39c_c060_5ced_c834,
            0x1082_276b_f3a2_7251,
            0x8f4c_a136_bef1_39c9,
        )
        .build_hasher();
        Self::hash_expression(expr, &mut h2);

        (h1.finish(), h2.finish())
    }

    /// Recursively hash an expression without string allocation
    fn hash_expression<H: std::hash::Hasher>(expr: &Expression, hasher: &mut H) {
        // First hash the discriminant to distinguish variants
        std::mem::discriminant(expr).hash(hasher);

        match expr {
            Expression::Identifier(id) => {
                id.value_lower.hash(hasher);
                // A quoted name compiles differently, as in the global key
                id.quoted.hash(hasher);
                if id.quoted {
                    id.value.hash(hasher);
                }
            }
            Expression::QualifiedIdentifier(qid) => {
                qid.qualifier.value_lower.hash(hasher);
                qid.name.value_lower.hash(hasher);
            }
            Expression::IntegerLiteral(lit) => {
                lit.value.hash(hasher);
            }
            Expression::FloatLiteral(lit) => {
                lit.value.to_bits().hash(hasher);
            }
            Expression::StringLiteral(lit) => {
                lit.value.hash(hasher);
                lit.type_hint.hash(hasher);
            }
            Expression::BooleanLiteral(lit) => {
                lit.value.hash(hasher);
            }
            Expression::NullLiteral(_) => {
                // Just discriminant is enough
            }
            Expression::IntervalLiteral(lit) => {
                lit.value.hash(hasher);
                lit.unit.hash(hasher);
            }
            Expression::Parameter(param) => {
                param.index.hash(hasher);
                param.name.hash(hasher);
            }
            Expression::Prefix(prefix) => {
                std::mem::discriminant(&prefix.op_type).hash(hasher);
                Self::hash_expression(&prefix.right, hasher);
            }
            Expression::Infix(infix) => {
                std::mem::discriminant(&infix.op_type).hash(hasher);
                Self::hash_expression(&infix.left, hasher);
                Self::hash_expression(&infix.right, hasher);
            }
            Expression::List(list) => {
                list.elements.len().hash(hasher);
                for val in &list.elements {
                    Self::hash_expression(val, hasher);
                }
            }
            Expression::Distinct(dist) => {
                Self::hash_expression(&dist.expr, hasher);
            }
            Expression::Exists(exists) => {
                // Use pointer identity for hashing - avoids expensive Debug format allocation
                (exists.subquery.as_ref() as *const _ as usize).hash(hasher);
            }
            Expression::AllAny(aa) => {
                aa.operator.hash(hasher);
                std::mem::discriminant(&aa.all_any_type).hash(hasher);
                Self::hash_expression(&aa.left, hasher);
                // Use pointer identity for hashing - avoids expensive Debug format allocation
                (aa.subquery.as_ref() as *const _ as usize).hash(hasher);
            }
            Expression::In(in_expr) => {
                in_expr.not.hash(hasher);
                Self::hash_expression(&in_expr.left, hasher);
                Self::hash_expression(&in_expr.right, hasher);
            }
            Expression::InHashSet(in_hash) => {
                in_hash.not.hash(hasher);
                Self::hash_expression(&in_hash.column, hasher);
                // The set is part of the program (its members are baked into
                // the compiled lookup), so two nodes with different members
                // must not share a cache entry. Hashing only the length let a
                // correlated IN, which builds a fresh one-element set per
                // outer row, hand every row the first row's program. Combine
                // the members commutatively so equal sets hash equal in any
                // iteration order.
                in_hash.values.len().hash(hasher);
                // Written straight into the caller's hasher in a canonical
                // order, so each of the two independent keys sees the members
                // themselves. Folding them into one 64-bit value first would
                // hand both keys the same collision.
                // Sorted through Ord, which puts a NULL first and is total.
                // The plain sort clippy asks for reads PartialOrd, whose
                // NULL is neither below nor above anything, and the sort
                // trips on the broken order once the set is large enough
                #[allow(clippy::unnecessary_sort_by)]
                let members = {
                    let mut members: Vec<&Value> = in_hash.values.members().iter().collect();
                    members.sort_unstable_by(|a, b| a.cmp(b));
                    members
                };
                for value in members {
                    value.hash(hasher);
                }
            }
            Expression::QuantifiedSet(quantified) => {
                quantified.operator.hash(hasher);
                quantified.all.hash(hasher);
                quantified.has_null.hash(hasher);
                Self::hash_expression(&quantified.left, hasher);
                quantified.members.len().hash(hasher);
                for value in quantified.members.iter() {
                    value.hash(hasher);
                }
            }
            Expression::Between(between) => {
                between.not.hash(hasher);
                Self::hash_expression(&between.expr, hasher);
                Self::hash_expression(&between.lower, hasher);
                Self::hash_expression(&between.upper, hasher);
            }
            Expression::Like(like) => {
                like.operator.hash(hasher);
                Self::hash_expression(&like.left, hasher);
                Self::hash_expression(&like.pattern, hasher);
                if let Some(ref escape) = like.escape {
                    true.hash(hasher);
                    Self::hash_expression(escape, hasher);
                } else {
                    false.hash(hasher);
                }
            }
            Expression::ScalarSubquery(sq) => {
                // Use pointer identity for hashing - avoids expensive Debug format allocation
                (sq.subquery.as_ref() as *const _ as usize).hash(hasher);
            }
            Expression::ExpressionList(list) => {
                list.expressions.len().hash(hasher);
                for expr in &list.expressions {
                    Self::hash_expression(expr, hasher);
                }
            }
            Expression::Case(case) => {
                if let Some(ref val) = case.value {
                    true.hash(hasher);
                    Self::hash_expression(val, hasher);
                } else {
                    false.hash(hasher);
                }
                case.when_clauses.len().hash(hasher);
                for when_clause in &case.when_clauses {
                    Self::hash_expression(&when_clause.condition, hasher);
                    Self::hash_expression(&when_clause.then_result, hasher);
                }
                if let Some(ref else_val) = case.else_value {
                    true.hash(hasher);
                    Self::hash_expression(else_val, hasher);
                } else {
                    false.hash(hasher);
                }
            }
            Expression::Cast(cast) => {
                Self::hash_expression(&cast.expr, hasher);
                cast.type_name.hash(hasher);
            }
            Expression::FunctionCall(func) => {
                func.function.hash(hasher);
                func.is_distinct.hash(hasher);
                func.arguments.len().hash(hasher);
                for arg in &func.arguments {
                    Self::hash_expression(arg, hasher);
                }
                if let Some(ref filter) = func.filter {
                    true.hash(hasher);
                    Self::hash_expression(filter, hasher);
                } else {
                    false.hash(hasher);
                }
            }
            Expression::Aliased(aliased) => {
                aliased.alias.value_lower.hash(hasher);
                Self::hash_expression(&aliased.expression, hasher);
            }
            Expression::Window(window) => {
                // Hash the FunctionCall directly (not as Expression)
                window.function.function.hash(hasher);
                window.function.is_distinct.hash(hasher);
                window.function.arguments.len().hash(hasher);
                for arg in &window.function.arguments {
                    Self::hash_expression(arg, hasher);
                }
                window.partition_by.len().hash(hasher);
                for expr in &window.partition_by {
                    Self::hash_expression(expr, hasher);
                }
                window.order_by.len().hash(hasher);
                for order in &window.order_by {
                    Self::hash_expression(&order.expression, hasher);
                    order.ascending.hash(hasher);
                    order.nulls_first.hash(hasher);
                }
            }
            Expression::TableSource(ts) => {
                ts.name.value_lower.hash(hasher);
                if let Some(ref alias) = ts.alias {
                    true.hash(hasher);
                    alias.value_lower.hash(hasher);
                } else {
                    false.hash(hasher);
                }
            }
            Expression::JoinSource(js) => {
                // Use pointer identity for hashing - avoids expensive Debug format allocation
                (js.as_ref() as *const _ as usize).hash(hasher);
            }
            Expression::SubquerySource(sq) => {
                if let Some(ref alias) = sq.alias {
                    true.hash(hasher);
                    alias.value_lower.hash(hasher);
                } else {
                    false.hash(hasher);
                }
                // Use pointer identity for hashing - avoids expensive Debug format allocation
                (sq.subquery.as_ref() as *const _ as usize).hash(hasher);
            }
            Expression::ValuesSource(vs) => {
                if let Some(ref alias) = vs.alias {
                    true.hash(hasher);
                    alias.value_lower.hash(hasher);
                } else {
                    false.hash(hasher);
                }
                vs.rows.len().hash(hasher);
            }
            Expression::CteReference(cte) => {
                cte.name.value_lower.hash(hasher);
            }
            Expression::FunctionTableSource(fts) => {
                fts.function.value_lower.hash(hasher);
                for arg in &fts.arguments {
                    Self::hash_expression(arg, hasher);
                }
            }
            Expression::Star(_) => {
                // Just discriminant
            }
            Expression::QualifiedStar(qs) => {
                qs.qualifier.hash(hasher);
            }
            Expression::Default(_) => {
                // Just discriminant
            }
        }
    }

    /// Get or compile a program for the expression.
    /// Uses local cache for fast lookup within single query evaluation.
    fn get_or_compile(&mut self, expr: &Expression) -> Result<SharedProgram> {
        let expr_key = self.expr_hash(expr);

        // Check local cache (fast path, no synchronization)
        if let Some(program) = self.local_cache.get(&expr_key) {
            return Ok(CompactArc::clone(program));
        }

        // Cache miss: compile the expression
        let program = CompactArc::new(self.compile_expression(expr)?);
        // A correlated WHERE substitutes each outer row's subquery result
        // into the expression, so with the cache kept across rows every
        // distinct result would compile its own program and an IN program
        // would keep its whole set alive. Bounding the cache keeps the win
        // when results repeat and caps memory when they do not.
        if self.local_cache.len() >= LOCAL_PROGRAM_CACHE_MAX {
            self.local_cache.clear();
        }
        self.local_cache
            .insert(expr_key, CompactArc::clone(&program));

        Ok(program)
    }

    /// Compile an expression to a Program
    fn compile_expression(&self, expr: &Expression) -> Result<Program> {
        let mut ctx = CompileContext::new(&self.columns, self.function_registry);

        // Add outer columns if available
        if let Some(ref outer_cols) = self.outer_columns {
            ctx = ctx.with_outer_columns(outer_cols);
        }

        // Add expression aliases
        if !self.expression_aliases.is_empty() {
            ctx = ctx.with_expression_aliases(self.expression_aliases.clone());
        }

        // Add column aliases
        if !self.column_aliases.is_empty() {
            ctx = ctx.with_column_aliases(self.column_aliases.clone());
        }

        let compiler = ExprCompiler::new(&ctx);
        compiler
            .compile(expr)
            .map_err(|e| Error::internal(format!("Compile error: {}", e)))
    }

    /// Evaluate an expression to a Value
    pub fn evaluate(&mut self, expr: &Expression) -> Result<Value> {
        // Compile the expression first
        let program = self.get_or_compile(expr)?;

        // Static empty row for fallback
        static EMPTY_ROW: std::sync::LazyLock<Row> = std::sync::LazyLock::new(Row::new);

        // Get row data from owned copy
        let row = self.current_row.as_ref().unwrap_or(&EMPTY_ROW);

        // Build execution context
        let mut ctx = ExecuteContext::new(row);

        // Add parameters
        if !self.params.is_empty() {
            ctx = ctx.with_params(&self.params);
        }

        // Add named parameters
        if !self.named_params.is_empty() {
            ctx = ctx.with_named_params(&self.named_params);
        }

        // Add outer row
        if let Some(ref outer) = self.outer_row {
            ctx = ctx.with_outer_row(outer);
        }

        // Add transaction ID
        ctx = ctx.with_transaction_id(self.transaction_id);

        // Execute
        self.vm.execute_cow(&program, &ctx)
    }

    /// Compile and cache an expression, returning the shared program for
    /// repeated `evaluate_program` calls in per-row loops
    pub fn compile_cached(&mut self, expr: &Expression) -> Result<SharedProgram> {
        self.get_or_compile(expr)
    }

    /// Evaluate a pre-compiled program against the given row directly,
    /// without storing a per-row copy in the evaluator and without
    /// re-hashing the expression
    pub fn evaluate_program(&mut self, program: &SharedProgram, row: &Row) -> Result<Value> {
        let mut ctx = ExecuteContext::new(row);
        if !self.params.is_empty() {
            ctx = ctx.with_params(&self.params);
        }
        if !self.named_params.is_empty() {
            ctx = ctx.with_named_params(&self.named_params);
        }
        if let Some(ref outer) = self.outer_row {
            ctx = ctx.with_outer_row(outer);
        }
        ctx = ctx.with_transaction_id(self.transaction_id);
        self.vm.execute_cow(program, &ctx)
    }

    /// Evaluate an expression as a boolean (for WHERE/HAVING clauses)
    ///
    /// Returns false for NULL results (SQL three-valued logic).
    pub fn evaluate_bool(&mut self, expr: &Expression) -> Result<bool> {
        // Compile the expression first
        let program = self.get_or_compile(expr)?;

        // Static empty row for fallback
        static EMPTY_ROW: std::sync::LazyLock<Row> = std::sync::LazyLock::new(Row::new);

        // Get row data from owned copy
        let row = self.current_row.as_ref().unwrap_or(&EMPTY_ROW);

        // Build execution context
        let mut ctx = ExecuteContext::new(row);

        // Add parameters
        if !self.params.is_empty() {
            ctx = ctx.with_params(&self.params);
        }

        // Add named parameters
        if !self.named_params.is_empty() {
            ctx = ctx.with_named_params(&self.named_params);
        }

        // Add outer row
        if let Some(ref outer) = self.outer_row {
            ctx = ctx.with_outer_row(outer);
        }

        // Add transaction ID
        ctx = ctx.with_transaction_id(self.transaction_id);

        // Execute and convert to bool
        self.vm.execute_bool_checked(&program, &ctx)
    }
}

impl Default for CompiledEvaluator<'static> {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{
        Expression, Identifier, InfixExpression, InfixOperator, IntegerLiteral,
    };
    use crate::parser::token::{Position, Token, TokenType};

    fn dummy_token() -> Token {
        Token::new(TokenType::Eof, "", Position::default())
    }

    fn make_identifier(name: &str) -> Expression {
        Expression::Identifier(Identifier {
            value: name.into(),
            value_lower: name.to_lowercase().into(),
            position: Position::default(),
            quoted: false,
        })
    }

    fn make_int_literal(value: i64) -> Expression {
        Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token(),
            value,
        })
    }

    fn make_gt(col: &str, val: i64) -> Expression {
        Expression::Infix(InfixExpression {
            token: dummy_token(),
            left: Box::new(make_identifier(col)),
            operator: ">".into(),
            op_type: InfixOperator::GreaterThan,
            right: Box::new(make_int_literal(val)),
        })
    }

    #[test]
    fn test_in_hashset_values_change_the_cache_key() {
        use crate::common::CompactArc;
        use crate::core::{Row, Value, ValueSet};
        use crate::parser::ast::InHashSetExpression;

        fn make_in_hashset(val: i64) -> Expression {
            let mut set = ValueSet::default();
            set.insert(Value::Integer(val));
            Expression::InHashSet(InHashSetExpression {
                token: dummy_token(),
                column: Box::new(make_identifier("a")),
                values: CompactArc::new(crate::core::predicate::PredicateSet::of(CompactArc::new(
                    set,
                ))),
                not: false,
            })
        }

        let cols = vec!["a".to_string()];
        let e1 = make_in_hashset(1);
        let e2 = make_in_hashset(2);

        // Previously only values.len() was hashed, so these two shared one
        // key in BOTH hash algorithms and aliased in the program cache.
        let k1 = compute_cache_key(&e1, &cols);
        let k2 = compute_cache_key(&e2, &cols);
        assert_ne!(k1, k2, "IN-set contents must be part of the cache key");

        // Behavioral repro from review: the second filter must not be
        // served the first filter's compiled program.
        let f1 = RowFilter::new(&e1, &cols).unwrap();
        let f2 = RowFilter::new(&e2, &cols).unwrap();
        let row = Row::from_values(vec![Value::Integer(2)]);
        assert!(
            !f1.matches_checked(&row).unwrap(),
            "a IN {{1}} must reject a = 2"
        );
        assert!(
            f2.matches_checked(&row).unwrap(),
            "a IN {{2}} must accept a = 2"
        );
    }

    #[test]
    fn test_cache_key_pair_stable_and_discriminating() {
        let cols = vec!["a".to_string()];
        let e1 = make_gt("a", 1);
        let e2 = make_gt("a", 2);

        // Stable across calls (fixed seeds).
        assert_eq!(compute_cache_key(&e1, &cols), compute_cache_key(&e1, &cols));

        // Different expressions and different column sets change BOTH
        // components, so a single-algorithm collision cannot alias entries.
        let k1 = compute_cache_key(&e1, &cols);
        let k2 = compute_cache_key(&e2, &cols);
        assert_ne!(k1, k2);
        assert_ne!(k1.0, k2.0);
        assert_ne!(k1.1, k2.1);

        let k3 = compute_cache_key(&e1, &["b".to_string()]);
        assert_ne!(k1, k3);
    }

    fn make_infix(left: Expression, op: InfixOperator, right: Expression) -> Expression {
        let op_str = match op {
            InfixOperator::GreaterThan => ">",
            InfixOperator::LessThan => "<",
            InfixOperator::Equal => "=",
            InfixOperator::Add => "+",
            InfixOperator::Multiply => "*",
            _ => "?",
        };
        Expression::Infix(InfixExpression {
            token: dummy_token(),
            left: Box::new(left),
            operator: op_str.into(),
            op_type: op,
            right: Box::new(right),
        })
    }

    // =========================================================================
    // compute_expression_hash tests
    // =========================================================================

    #[test]
    fn test_compute_expression_hash_same_expr() {
        let expr1 = make_int_literal(42);
        let expr2 = make_int_literal(42);
        assert_eq!(
            compute_expression_hash(&expr1),
            compute_expression_hash(&expr2)
        );
    }

    #[test]
    fn test_compute_expression_hash_different_expr() {
        let expr1 = make_int_literal(42);
        let expr2 = make_int_literal(43);
        assert_ne!(
            compute_expression_hash(&expr1),
            compute_expression_hash(&expr2)
        );
    }

    #[test]
    fn test_compute_expression_hash_complex() {
        // col > 5
        let expr1 = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        // col > 5 (same)
        let expr2 = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        assert_eq!(
            compute_expression_hash(&expr1),
            compute_expression_hash(&expr2)
        );
    }

    // =========================================================================
    // compile_expression tests
    // =========================================================================

    #[test]
    fn test_compile_expression_basic() {
        // col > 5
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        let columns = vec!["col".to_string()];
        let program = compile_expression(&expr, &columns);
        assert!(program.is_ok());
    }

    #[test]
    fn test_compile_expression_unknown_column() {
        // unknown_col > 5
        let expr = make_infix(
            make_identifier("unknown_col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        let columns = vec!["col".to_string()];
        // Unknown columns cause compilation errors
        let program = compile_expression(&expr, &columns);
        assert!(program.is_err());
    }

    // =========================================================================
    // RowFilter tests
    // =========================================================================

    #[test]
    fn test_row_filter_new() {
        // col > 5
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        let columns = vec!["col".to_string()];
        let filter = RowFilter::new(&expr, &columns);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_row_filter_matches_true() {
        // col > 5
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        let columns = vec!["col".to_string()];
        let filter = RowFilter::new(&expr, &columns).unwrap();

        // Row with col = 10 (> 5)
        let row = Row::from(vec![Value::Integer(10)]);
        assert!(filter.matches_checked(&row).unwrap());
    }

    #[test]
    fn test_row_filter_matches_false() {
        // col > 5
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        let columns = vec!["col".to_string()];
        let filter = RowFilter::new(&expr, &columns).unwrap();

        // Row with col = 3 (not > 5)
        let row = Row::from(vec![Value::Integer(3)]);
        assert!(!filter.matches_checked(&row).unwrap());
    }

    #[test]
    fn test_row_filter_clone() {
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        let columns = vec!["col".to_string()];
        let filter = RowFilter::new(&expr, &columns).unwrap();
        let cloned = filter.clone();

        let row = Row::from(vec![Value::Integer(10)]);
        assert!(filter.matches_checked(&row).unwrap());
        assert!(cloned.matches_checked(&row).unwrap());
    }

    // =========================================================================
    // ExpressionEval tests
    // =========================================================================

    #[test]
    fn test_expression_eval_compile() {
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        let columns = vec!["col".to_string()];
        let eval = ExpressionEval::compile(&expr, &columns);
        assert!(eval.is_ok());
    }

    #[test]
    fn test_expression_eval_eval() {
        // col + 10
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::Add,
            make_int_literal(10),
        );
        let columns = vec!["col".to_string()];
        let mut eval = ExpressionEval::compile(&expr, &columns).unwrap();

        let row = Row::from(vec![Value::Integer(5)]);
        let result = eval.eval(&row).unwrap();
        assert_eq!(result, Value::Integer(15));
    }

    #[test]
    fn test_expression_eval_eval_bool() {
        // col > 5
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );
        let columns = vec!["col".to_string()];
        let mut eval = ExpressionEval::compile(&expr, &columns).unwrap();

        let row = Row::from(vec![Value::Integer(10)]);
        assert!(eval.eval_bool_checked(&row).unwrap());

        let row = Row::from(vec![Value::Integer(3)]);
        assert!(!eval.eval_bool_checked(&row).unwrap());
    }

    // =========================================================================
    // MultiExpressionEval tests
    // =========================================================================

    #[test]
    fn test_multi_expression_eval_all() {
        let expr1 = make_infix(
            make_identifier("col"),
            InfixOperator::Add,
            make_int_literal(10),
        );
        let expr2 = make_infix(
            make_identifier("col"),
            InfixOperator::Multiply,
            make_int_literal(2),
        );
        let columns = vec!["col".to_string()];
        let mut eval = MultiExpressionEval::compile_with_aliases(
            &[expr1, expr2],
            &columns,
            &FxHashMap::default(),
        )
        .unwrap();

        let row = Row::from(vec![Value::Integer(5)]);
        let results = eval.eval_all(&row).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], Value::Integer(15)); // 5 + 10
        assert_eq!(results[1], Value::Integer(10)); // 5 * 2
    }

    // =========================================================================
    // CompiledEvaluator tests
    // =========================================================================

    #[test]
    fn test_compiled_evaluator_with_defaults() {
        let eval = CompiledEvaluator::with_defaults();
        assert!(eval.columns.is_empty());
    }

    #[test]
    fn test_compiled_evaluator_init_columns() {
        let mut eval = CompiledEvaluator::with_defaults();
        eval.init_columns(&["col1".to_string(), "col2".to_string()]);
        assert_eq!(eval.columns.len(), 2);
    }

    /// Column names are identified by content, not by the address of the
    /// caller's buffer: a reused buffer holding different names must be picked
    /// up rather than served from a stale copy.
    #[test]
    fn test_compiled_evaluator_init_columns_reused_buffer() {
        let mut eval = CompiledEvaluator::with_defaults();
        let mut names = vec!["a".to_string(), "b".to_string()];
        eval.init_columns(&names);

        // Same buffer address, same length, different names.
        names[0] = "z".to_string();
        eval.init_columns(&names);

        assert_eq!(eval.columns.as_slice(), names.as_slice());

        let row = Row::from(vec![Value::Integer(7), Value::Integer(9)]);
        eval.set_row_array(&row);
        assert_eq!(
            eval.evaluate(&make_identifier("z")).unwrap(),
            Value::Integer(7)
        );
    }

    /// Two IN-set nodes with different members must not share a compiled
    /// program, and the same members in any order must. A correlated IN
    /// builds a fresh one-element set per outer row, so hashing only the
    /// length handed every row the first row's program.
    /// Per-row substituted expressions must not grow the program cache
    /// without bound: a correlated subquery can produce a new literal or a
    /// new IN set on every outer row.
    #[test]
    fn test_local_program_cache_is_bounded() {
        let mut eval = CompiledEvaluator::with_defaults();
        eval.init_columns(&["id".to_string()]);
        let row = Row::from(vec![Value::Integer(1)]);
        eval.set_row_array(&row);
        for i in 0..(LOCAL_PROGRAM_CACHE_MAX as i64 * 4) {
            let expr = make_infix(
                make_identifier("id"),
                InfixOperator::Add,
                make_int_literal(i),
            );
            let _ = eval.evaluate(&expr).unwrap();
            assert!(eval.local_cache.len() <= LOCAL_PROGRAM_CACHE_MAX);
        }
    }

    #[test]
    fn test_expr_hash_distinguishes_in_hash_set_members() {
        use crate::core::ValueSet;
        use crate::executor::utils::dummy_token;
        use crate::parser::ast::InHashSetExpression;

        let node = |members: &[i64]| {
            let set: ValueSet = members.iter().map(|&v| Value::Integer(v)).collect();
            Expression::InHashSet(InHashSetExpression {
                token: dummy_token("IN", TokenType::Keyword),
                column: Box::new(make_identifier("id")),
                values: CompactArc::new(crate::core::predicate::PredicateSet::of(CompactArc::new(
                    set,
                ))),
                not: false,
            })
        };
        let eval = CompiledEvaluator::with_defaults();

        assert_ne!(eval.expr_hash(&node(&[1])), eval.expr_hash(&node(&[2])));
        assert_ne!(
            eval.expr_hash(&node(&[1, 2])),
            eval.expr_hash(&node(&[1, 3]))
        );
        assert_eq!(
            eval.expr_hash(&node(&[1, 2, 3])),
            eval.expr_hash(&node(&[3, 1, 2]))
        );
    }

    #[test]
    fn test_compiled_evaluator_evaluate_bool() {
        let mut eval = CompiledEvaluator::with_defaults();
        eval.init_columns(&["col".to_string()]);
        let row = Row::from(vec![Value::Integer(10)]);
        eval.set_row_array(&row);

        // col > 5
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::GreaterThan,
            make_int_literal(5),
        );

        let result = eval.evaluate_bool(&expr);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_compiled_evaluator_evaluate() {
        let mut eval = CompiledEvaluator::with_defaults();
        eval.init_columns(&["col".to_string()]);
        let row = Row::from(vec![Value::Integer(5)]);
        eval.set_row_array(&row);

        // col + 10
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::Add,
            make_int_literal(10),
        );

        let result = eval.evaluate(&expr);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Integer(15));
    }

    #[test]
    fn test_quoted_identifier_does_not_reuse_the_unquoted_program() {
        let cols = vec!["true".to_string()];
        let mut eval = CompiledEvaluator::default();
        eval.init_columns(&cols);
        let row = Row::from(vec![Value::Integer(8)]);
        eval.set_row_array(&row);

        let quoted = |value: &str| {
            Expression::Identifier(Identifier {
                value: value.into(),
                value_lower: value.to_lowercase().into(),
                position: Position::default(),
                quoted: true,
            })
        };

        // Unquoted true is the boolean literal; a quoted name is the column
        assert_eq!(
            eval.evaluate(&make_identifier("true")).unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(eval.evaluate(&quoted("true")).unwrap(), Value::Integer(8));
        // With no such column a quoted name is its own text
        assert_eq!(eval.evaluate(&quoted("ABC")).unwrap(), Value::text("ABC"));
        assert_eq!(eval.evaluate(&quoted("abc")).unwrap(), Value::text("abc"));
    }

    #[test]
    fn test_compiled_evaluator_default() {
        let eval = CompiledEvaluator::default();
        assert!(eval.columns.is_empty());
    }
}
