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
// - `RowFilter` pre-compiles the expression once and shares `Arc<Program>` across threads
// - The VM is lightweight and can be created per-thread without performance penalty

use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

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
static PROGRAM_CACHE: Mutex<Option<LruCache<u64, SharedProgram>>> = Mutex::new(None);

/// Compute cache key from expression and columns using efficient recursive hashing.
/// This avoids the overhead of Debug formatting by directly hashing expression structure.
/// Uses FxHasher which is 2-5x faster than SipHash for small keys.
fn compute_cache_key(expr: &Expression, columns: &[String]) -> u64 {
    let mut hasher = FxHasher::default();
    // Use efficient recursive hashing (same as CompiledEvaluator::hash_expression)
    hash_expression(expr, &mut hasher);
    // Hash column names
    columns.hash(&mut hasher);
    hasher.finish()
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
fn hash_expression(expr: &Expression, hasher: &mut FxHasher) {
    // First hash the discriminant to distinguish variants
    std::mem::discriminant(expr).hash(hasher);

    match expr {
        Expression::Identifier(id) => {
            id.value_lower.hash(hasher);
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
        }
        Expression::CteReference(cte) => {
            cte.name.value_lower.hash(hasher);
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
        .map(Arc::new)
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
/// or parallel execution. The returned `Arc<Program>` is `Send + Sync` and
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
/// * `Arc<Program>` that can be executed with `RowFilter` or `ExprVM`
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
        .map(Arc::new)
        .map_err(|e| Error::internal(format!("Compile error: {}", e)))
}

// ============================================================================
// ROW FILTER - Lightweight, Send+Sync filter for closures
// ============================================================================

/// A lightweight, thread-safe row filter for closure-based filtering.
///
/// `RowFilter` pre-compiles the expression once and can be cloned cheaply
/// (it uses `Arc<Program>` internally). Each thread should create its own
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
    /// Query parameters (shared) - uses Arc<Vec<Value>> to match ExecutionContext
    params: Arc<Vec<Value>>,
    /// Named parameters (shared)
    named_params: Arc<FxHashMap<String, Value>>,
}

impl RowFilter {
    /// Create a new row filter by compiling the given expression.
    ///
    /// # Arguments
    /// * `expr` - The boolean expression to use as filter
    /// * `columns` - Column names matching the row schema
    pub fn new(expr: &Expression, columns: &[String]) -> Result<Self> {
        let program = compile_expression(expr, columns)?;
        Ok(Self {
            program,
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
        })
    }

    /// Create a row filter with expression aliases for HAVING clause evaluation.
    ///
    /// Expression aliases map expression strings (like "SUM(amount)") to column
    /// indices in the result row. This is used for HAVING clauses where aggregate
    /// expressions need to reference pre-computed aggregate results.
    ///
    /// # Arguments
    /// * `expr` - The boolean expression to use as filter
    /// * `columns` - Column names matching the row schema
    /// * `aliases` - Slice of (expression_name, column_index) pairs
    ///
    /// # Example
    /// ```ignore
    /// // For HAVING SUM(amount) > 100, where SUM(amount) is at column 2
    /// let aliases = vec![("sum(amount)".to_string(), 2)];
    /// let filter = RowFilter::with_aliases(&having_expr, &columns, &aliases)?;
    ///
    /// // Filter rows
    /// for row in rows {
    ///     if filter.matches(&row) {
    ///         // row passes HAVING clause
    ///     }
    /// }
    /// ```
    pub fn with_aliases(
        expr: &Expression,
        columns: &[String],
        aliases: &[(String, usize)],
    ) -> Result<Self> {
        let alias_map: FxHashMap<String, u16> = aliases
            .iter()
            .map(|(name, idx)| (name.to_lowercase(), *idx as u16))
            .collect();

        let ctx = CompileContext::with_global_registry(columns).with_expression_aliases(alias_map);
        let compiler = ExprCompiler::new(&ctx);
        let program = compiler
            .compile(expr)
            .map(Arc::new)
            .map_err(|e| Error::internal(format!("Compile error: {}", e)))?;

        Ok(Self {
            program,
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
        })
    }

    /// Create a filter with query parameters.
    pub fn with_params(mut self, params: Vec<Value>) -> Self {
        self.params = Arc::new(params);
        self
    }

    /// Create a filter with named parameters.
    pub fn with_named_params(mut self, named_params: FxHashMap<String, Value>) -> Self {
        self.named_params = Arc::new(named_params);
        self
    }

    /// Create a filter from execution context.
    ///
    /// PERF: Both `params` and `named_params` share the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed (both use Arc<Vec<Value>>)
        self.params = Arc::clone(ctx.params_arc());
        // Share named_params Arc - no cloning needed
        self.named_params = Arc::clone(ctx.named_params_arc());
        self
    }

    /// Create a filter from a pre-compiled program.
    pub fn from_program(program: SharedProgram) -> Self {
        Self {
            program,
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
        }
    }

    /// Check if a row matches the filter condition.
    ///
    /// This method is thread-safe and can be called from multiple threads.
    /// Each call uses a thread-local VM for execution.
    #[inline]
    pub fn matches(&self, row: &Row) -> bool {
        // Use thread-local VM for zero allocation in hot path
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

            // Use try_borrow_mut to avoid panic on recursive calls (e.g., nested subqueries).
            // If the VM is already borrowed, create a temporary one for this call.
            if let Ok(mut borrowed_vm) = vm.try_borrow_mut() {
                borrowed_vm.execute_bool(&self.program, &ctx)
            } else {
                // Fallback: create a fresh VM for recursive calls
                let mut temp_vm = ExprVM::new();
                temp_vm.execute_bool(&self.program, &ctx)
            }
        })
    }

    /// Evaluate the filter expression and return the value.
    #[inline]
    pub fn evaluate(&self, row: &Row) -> Result<Value> {
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

            // Use try_borrow_mut to avoid panic on recursive calls (e.g., nested subqueries).
            // If the VM is already borrowed, create a temporary one for this call.
            if let Ok(mut borrowed_vm) = vm.try_borrow_mut() {
                borrowed_vm.execute_cow(&self.program, &ctx)
            } else {
                // Fallback: create a fresh VM for recursive calls
                let mut temp_vm = ExprVM::new();
                temp_vm.execute_cow(&self.program, &ctx)
            }
        })
    }

    /// Get the underlying program (for advanced use cases).
    pub fn program(&self) -> &SharedProgram {
        &self.program
    }
}

// Static assertions to verify RowFilter implements Send + Sync.
// This is safer than unsafe impl because it will fail at compile time
// if any field doesn't implement Send/Sync, rather than causing UB at runtime.
// All fields are Send + Sync:
// - Arc<Program> is Send + Sync (Program is immutable)
// - Arc<[Value]> is Send + Sync
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
        let ctx =
            CompileContext::new(left_columns, function_registry).with_second_row(right_columns);
        let compiler = ExprCompiler::new(&ctx);
        let program = compiler
            .compile(expr)
            .map_err(|e| Error::internal(format!("Compile error: {}", e)))?;
        Ok(Self {
            program: Arc::new(program),
        })
    }

    /// Check if a pair of rows satisfies the join condition.
    #[inline]
    pub fn matches(&self, left_row: &Row, right_row: &Row) -> bool {
        thread_local! {
            static VM: std::cell::RefCell<ExprVM> = std::cell::RefCell::new(ExprVM::new());
        }

        VM.with(|vm| {
            let ctx = ExecuteContext::for_join(left_row, right_row);

            // Use try_borrow_mut to avoid panic on recursive calls (e.g., nested subqueries).
            // If the VM is already borrowed, create a temporary one for this call.
            if let Ok(mut borrowed_vm) = vm.try_borrow_mut() {
                borrowed_vm.execute_bool(&self.program, &ctx)
            } else {
                // Fallback: create a fresh VM for recursive calls
                let mut temp_vm = ExprVM::new();
                temp_vm.execute_bool(&self.program, &ctx)
            }
        })
    }

    /// Get the underlying program.
    pub fn program(&self) -> &SharedProgram {
        &self.program
    }
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
    /// Query parameters (shared) - uses Arc<Vec<Value>> to match ExecutionContext
    params: Arc<Vec<Value>>,
    /// Named parameters (shared) - uses Arc to match ExecutionContext
    named_params: Arc<FxHashMap<String, Value>>,
    /// Outer row context for correlated subqueries
    outer_row: Option<FxHashMap<Arc<str>, Value>>,
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
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
            outer_row: None,
            transaction_id: None,
        })
    }

    /// Compile with expression aliases for HAVING clause evaluation.
    ///
    /// Expression aliases map expression strings (like "SUM(amount)") to column
    /// indices in the result row. This is used for HAVING clauses where aggregate
    /// expressions need to reference pre-computed aggregate results.
    ///
    /// # Arguments
    /// * `expr` - The expression to compile
    /// * `columns` - Column names for the result row
    /// * `aliases` - Slice of (expression_name, column_index) pairs
    ///
    /// # Example
    /// ```ignore
    /// // For HAVING SUM(amount) > 100, where SUM(amount) is at column 2
    /// let aliases = vec![("sum(amount)".to_string(), 2)];
    /// let eval = ExpressionEval::compile_with_aliases(&having_expr, &columns, &aliases)?;
    /// ```
    pub fn compile_with_aliases(
        expr: &Expression,
        columns: &[String],
        aliases: &[(String, usize)],
    ) -> Result<Self> {
        let alias_map: FxHashMap<String, u16> = aliases
            .iter()
            .map(|(name, idx)| (name.to_lowercase(), *idx as u16))
            .collect();

        Self::compile_with_options(
            expr,
            columns,
            None,
            None,
            Some(alias_map),
            global_registry(),
        )
    }

    /// Compile with full context options.
    pub fn compile_with_options(
        expr: &Expression,
        columns: &[String],
        columns2: Option<&[String]>,
        outer_columns: Option<&[String]>,
        expression_aliases: Option<FxHashMap<String, u16>>,
        function_registry: &FunctionRegistry,
    ) -> Result<Self> {
        let mut ctx = CompileContext::new(columns, function_registry);
        if let Some(cols2) = columns2 {
            ctx = ctx.with_second_row(cols2);
        }
        if let Some(outer) = outer_columns {
            ctx = ctx.with_outer_columns(outer);
        }
        if let Some(aliases) = expression_aliases {
            ctx = ctx.with_expression_aliases(aliases);
        }
        let compiler = ExprCompiler::new(&ctx);
        let program = compiler
            .compile(expr)
            .map(Arc::new)
            .map_err(|e| Error::internal(format!("Compile error: {}", e)))?;
        Ok(Self {
            program,
            vm: ExprVM::new(),
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
            outer_row: None,
            transaction_id: None,
        })
    }

    /// Create from a pre-compiled program.
    pub fn from_program(program: SharedProgram) -> Self {
        Self {
            program,
            vm: ExprVM::new(),
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
            outer_row: None,
            transaction_id: None,
        }
    }

    /// Set query parameters.
    pub fn with_params(mut self, params: Vec<Value>) -> Self {
        self.params = Arc::new(params);
        self
    }

    /// Set named parameters.
    pub fn with_named_params(mut self, named_params: FxHashMap<String, Value>) -> Self {
        self.named_params = Arc::new(named_params);
        self
    }

    /// Set context from ExecutionContext.
    ///
    /// PERF: Both `params` and `named_params` share the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed
        self.params = Arc::clone(ctx.params_arc());
        // Share named_params Arc - no cloning needed
        self.named_params = Arc::clone(ctx.named_params_arc());
        if let Some(outer) = ctx.outer_row() {
            let mut arc_map = FxHashMap::default();
            for (k, v) in outer.iter() {
                arc_map.insert(Arc::from(k.as_str()), v.clone());
            }
            self.outer_row = Some(arc_map);
        }
        self.transaction_id = ctx.transaction_id();
        self
    }

    /// Set transaction ID.
    pub fn with_transaction_id(mut self, txn_id: Option<u64>) -> Self {
        self.transaction_id = txn_id;
        self
    }

    /// Set outer row for correlated subqueries.
    pub fn set_outer_row(&mut self, outer: &FxHashMap<String, Value>) {
        let mut arc_map = FxHashMap::default();
        for (k, v) in outer.iter() {
            arc_map.insert(Arc::from(k.as_str()), v.clone());
        }
        self.outer_row = Some(arc_map);
    }

    /// Clear outer row.
    pub fn clear_outer_row(&mut self) {
        self.outer_row = None;
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

    /// Evaluate as boolean (for WHERE/HAVING).
    #[inline]
    pub fn eval_bool(&mut self, row: &Row) -> bool {
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

        self.vm.execute_bool(&self.program, &ctx)
    }

    /// Evaluate with two rows (for joins).
    #[inline]
    pub fn eval_join(&mut self, left: &Row, right: &Row) -> Result<Value> {
        let ctx = ExecuteContext::for_join(left, right);
        self.vm.execute_cow(&self.program, &ctx)
    }

    /// Evaluate join as boolean.
    #[inline]
    pub fn eval_join_bool(&mut self, left: &Row, right: &Row) -> bool {
        let ctx = ExecuteContext::for_join(left, right);
        self.vm.execute_bool(&self.program, &ctx)
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

    /// Evaluate as boolean.
    #[inline]
    pub fn eval_slice_bool(&mut self, row: &Row) -> bool {
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

        self.vm.execute_bool(&self.program, &ctx)
    }

    /// Get the underlying program.
    pub fn program(&self) -> &SharedProgram {
        &self.program
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
    /// Query parameters (shared) - uses Arc<Vec<Value>> to match ExecutionContext
    params: Arc<Vec<Value>>,
    /// Named parameters (shared) - uses Arc to match ExecutionContext
    named_params: Arc<FxHashMap<String, Value>>,
    /// Transaction ID
    transaction_id: Option<u64>,
}

impl MultiExpressionEval {
    /// Compile multiple expressions.
    pub fn compile(exprs: &[Expression], columns: &[String]) -> Result<Self> {
        let ctx = CompileContext::with_global_registry(columns);
        let compiler = ExprCompiler::new(&ctx);

        let programs = exprs
            .iter()
            .map(|expr| {
                compiler
                    .compile(expr)
                    .map(Arc::new)
                    .map_err(|e| Error::internal(format!("Compile error: {}", e)))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            programs,
            vm: ExprVM::new(),
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
            transaction_id: None,
        })
    }

    /// Compile multiple expressions with expression aliases.
    ///
    /// Expression aliases map expression strings (like "SUM(amount)") to column
    /// indices in the result row. This is used for window function ORDER BY
    /// clauses where aggregate expressions need to reference pre-computed results.
    ///
    /// # Arguments
    /// * `exprs` - The expressions to compile
    /// * `columns` - Column names for the result row
    /// * `aliases` - Slice of (expression_name, column_index) pairs
    pub fn compile_with_aliases(
        exprs: &[Expression],
        columns: &[String],
        aliases: &[(String, usize)],
    ) -> Result<Self> {
        let alias_map: FxHashMap<String, u16> = aliases
            .iter()
            .map(|(name, idx)| (name.to_lowercase(), *idx as u16))
            .collect();

        let ctx = CompileContext::with_global_registry(columns).with_expression_aliases(alias_map);
        let compiler = ExprCompiler::new(&ctx);

        let programs = exprs
            .iter()
            .map(|expr| {
                compiler
                    .compile(expr)
                    .map(Arc::new)
                    .map_err(|e| Error::internal(format!("Compile error: {}", e)))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            programs,
            vm: ExprVM::new(),
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
            transaction_id: None,
        })
    }

    /// Set query parameters.
    pub fn with_params(mut self, params: Vec<Value>) -> Self {
        self.params = Arc::new(params);
        self
    }

    /// Set from execution context.
    ///
    /// PERF: Both `params` and `named_params` share the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed
        self.params = Arc::clone(ctx.params_arc());
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

    /// Number of expressions.
    pub fn len(&self) -> usize {
        self.programs.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.programs.is_empty()
    }
}

/// Shared program reference for zero-copy caching
pub type SharedProgram = Arc<Program>;

// ============================================================================
// COMPILED EVALUATOR - DEPRECATED, use ExpressionEval instead
// ============================================================================

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
    columns: Arc<Vec<String>>,

    /// Cached Arc pointer for fast equality check (avoids Arc comparison)
    columns_arc_id: usize,

    /// Second row columns (for joins)
    columns2: Option<Vec<String>>,

    /// Outer query columns (for correlated subqueries)
    outer_columns: Option<Vec<String>>,

    /// Query parameters (positional) - uses Arc<Vec<Value>> to match ExecutionContext
    params: Arc<Vec<Value>>,

    /// Query parameters (named) - uses Arc to match ExecutionContext
    named_params: Arc<FxHashMap<String, Value>>,

    /// Outer row context for correlated subqueries
    outer_row: Option<FxHashMap<Arc<str>, Value>>,

    /// Current transaction ID
    transaction_id: Option<u64>,

    /// Expression aliases for HAVING clause
    expression_aliases: FxHashMap<String, u16>,

    /// Column aliases
    column_aliases: FxHashMap<String, String>,

    /// VM instance (reusable)
    vm: ExprVM,

    /// Local cache: expression hash -> program (fast, no synchronization)
    local_cache: FxHashMap<u64, SharedProgram>,

    /// Current row values for execution (owned copy for safety)
    current_row: Option<Row>,

    /// Second row for joins (owned copy for safety)
    current_row2: Option<Row>,
}

// CompiledEvaluator is Send + Sync because all fields are Send + Sync:
// - function_registry: &FunctionRegistry is Send + Sync (shared reference to thread-safe registry)
// - All other fields are owned types that are Send + Sync

impl<'a> CompiledEvaluator<'a> {
    /// Create a new compiled evaluator with a function registry reference
    pub fn new(function_registry: &'a FunctionRegistry) -> Self {
        Self {
            function_registry,
            columns: Arc::new(Vec::new()),
            columns_arc_id: 0,
            columns2: None,
            outer_columns: None,
            params: Arc::new(Vec::new()),
            named_params: Arc::new(FxHashMap::default()),
            outer_row: None,
            transaction_id: None,
            expression_aliases: FxHashMap::default(),
            column_aliases: FxHashMap::default(),
            vm: ExprVM::new(),
            local_cache: FxHashMap::default(),
            current_row: None,
            current_row2: None,
        }
    }

    /// Create an evaluator using the global function registry.
    pub fn with_defaults() -> CompiledEvaluator<'static> {
        CompiledEvaluator::new(global_registry())
    }

    /// Clear all state for reuse.
    pub fn clear(&mut self) {
        self.columns = Arc::new(Vec::new());
        self.columns_arc_id = 0;
        self.columns2 = None;
        self.outer_columns = None;
        self.params = Arc::new(Vec::new());
        self.named_params = Arc::new(FxHashMap::default());
        self.outer_row = None;
        self.transaction_id = None;
        self.expression_aliases.clear();
        self.column_aliases.clear();
        self.local_cache.clear();
        self.current_row = None;
        self.current_row2 = None;
    }

    /// Set the current transaction ID
    pub fn set_transaction_id(&mut self, txn_id: u64) {
        self.transaction_id = Some(txn_id);
    }

    /// Set query parameters (positional) - fluent API
    pub fn with_params(mut self, params: Vec<Value>) -> Self {
        self.params = Arc::new(params);
        self
    }

    /// Set named query parameters - fluent API
    pub fn with_named_params(mut self, named_params: FxHashMap<String, Value>) -> Self {
        self.named_params = Arc::new(named_params);
        self
    }

    /// Set parameters from execution context - fluent API
    ///
    /// PERF: Both `params` and `named_params` share the Arc - zero cloning.
    pub fn with_context(mut self, ctx: &ExecutionContext) -> Self {
        // Share params Arc - no cloning needed
        self.params = Arc::clone(ctx.params_arc());
        // Share named_params Arc - no cloning needed
        self.named_params = Arc::clone(ctx.named_params_arc());

        // Set outer row context for correlated subqueries
        if let Some(outer) = ctx.outer_row() {
            let mut arc_map = FxHashMap::default();
            let mut outer_cols = Vec::new();
            for (k, v) in outer.iter() {
                arc_map.insert(Arc::from(k.as_str()), v.clone());
                outer_cols.push(k.clone());
            }
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

    /// Set the current row from an array with column names - fluent API
    ///
    /// Note: This method stores a pointer to the row. The caller must ensure
    /// the row outlives the evaluator or call set_row_array for each evaluation.
    pub fn with_row(mut self, row: Row, columns: &[String]) -> Self {
        self.init_columns(columns);
        // For fluent API usage, just set the columns.
        // The actual row should be set via set_row_array before evaluate.
        // This matches the Evaluator pattern where with_row takes ownership
        // but set_row_array is the hot path for per-row evaluation.
        let _ = row; // Row will be set via set_row_array
        self
    }

    /// Initialize the column index mapping (call once before set_row_array)
    ///
    /// Performance: This method caches the source slice pointer. When called
    /// repeatedly with the same slice (e.g., same table schema), it skips
    /// the expensive string cloning operation.
    pub fn init_columns(&mut self, columns: &[String]) {
        // Fast path: if same slice by pointer and length, skip clone
        // This handles the common case where init_columns is called repeatedly
        // with the same table schema during UPDATE/SELECT operations
        let new_source_id = columns.as_ptr() as usize;
        if self.columns_arc_id == new_source_id && self.columns.len() == columns.len() {
            return;
        }

        self.columns = Arc::new(columns.to_vec());
        self.columns_arc_id = new_source_id;
        // Clear local cache since compilation context changed
        self.local_cache.clear();
    }

    /// Initialize columns from an Arc (zero-copy when schema already has Arc)
    ///
    /// This is the preferred method when the caller already has an Arc<Vec<String>>,
    /// such as from `Schema::column_names_arc()`. It avoids all string cloning.
    #[inline]
    pub fn init_columns_arc(&mut self, columns: Arc<Vec<String>>) {
        // Use Arc pointer for identity check
        let new_arc_id = Arc::as_ptr(&columns) as usize;
        if self.columns_arc_id == new_arc_id {
            return;
        }

        self.columns = columns;
        self.columns_arc_id = new_arc_id;
        // Clear local cache since compilation context changed
        self.local_cache.clear();
    }

    /// Add aggregate expression aliases for HAVING clause evaluation
    pub fn add_aggregate_aliases(&mut self, aliases: &[(String, usize)]) {
        for (expr_name, idx) in aliases {
            let lower = expr_name.to_lowercase();
            self.expression_aliases.insert(lower, *idx as u16);
        }
        // Invalidate local cache since compilation context changed
        self.local_cache.clear();
    }

    /// Add expression aliases for HAVING clause with GROUP BY expressions
    pub fn add_expression_aliases(&mut self, aliases: &[(String, usize)]) {
        for (expr_str, idx) in aliases {
            let lower = expr_str.to_lowercase();
            self.expression_aliases.insert(lower, *idx as u16);
        }
        // Invalidate local cache since compilation context changed
        self.local_cache.clear();
    }

    /// Set the row using array-based access (optimized - no map rebuilding)
    /// Call init_columns() once before using this method.
    #[inline]
    pub fn set_row_array(&mut self, row: &Row) {
        self.current_row = Some(row.clone());
        // Clear join mode
        self.current_row2 = None;
    }

    /// Set two rows for join condition evaluation
    #[inline]
    pub fn set_join_rows(&mut self, left_row: &Row, right_row: &Row) {
        self.current_row = Some(left_row.clone());
        self.current_row2 = Some(right_row.clone());
    }

    /// Initialize join columns
    pub fn init_join_columns(&mut self, left_columns: &[String], right_columns: &[String]) {
        self.columns = Arc::new(left_columns.to_vec());
        self.columns_arc_id = 0; // Reset since we're creating a new Arc
        self.columns2 = Some(right_columns.to_vec());
        // Invalidate local cache since compilation context changed
        self.local_cache.clear();
    }

    /// Set the outer row context for correlated subqueries
    #[inline]
    pub fn set_outer_row(&mut self, outer_row: Option<&FxHashMap<String, Value>>) {
        if let Some(outer) = outer_row {
            let mut arc_map = FxHashMap::default();
            for (k, v) in outer.iter() {
                arc_map.insert(Arc::from(k.as_str()), v.clone());
            }
            self.outer_row = Some(arc_map);
        } else {
            self.outer_row = None;
        }
    }

    /// Set the outer row context by taking ownership
    #[inline]
    pub fn set_outer_row_owned(&mut self, outer_row: FxHashMap<String, Value>) {
        let mut arc_map = FxHashMap::default();
        let mut outer_cols = Vec::new();
        for (k, v) in outer_row.into_iter() {
            outer_cols.push(k.clone());
            arc_map.insert(Arc::from(k.as_str()), v);
        }
        self.outer_row = Some(arc_map);
        // Also set up outer_columns for compilation so LoadOuterColumn can be emitted
        if !outer_cols.is_empty() {
            // Sort for deterministic order
            outer_cols.sort();
            self.outer_columns = Some(outer_cols);
            // Invalidate local cache since compilation context changed
            self.local_cache.clear();
        }
    }

    /// Compile an expression and return a shared program for parallel use.
    /// The returned Arc<Program> can be cloned cheaply and shared across threads.
    pub fn compile_shared(&mut self, expr: &Expression) -> Result<SharedProgram> {
        self.get_or_compile(expr)
    }

    /// Take ownership of the outer row back (for reuse)
    #[inline]
    pub fn take_outer_row(&mut self) -> FxHashMap<String, Value> {
        if let Some(arc_map) = self.outer_row.take() {
            arc_map
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect()
        } else {
            FxHashMap::default()
        }
    }

    /// Clear the outer row context
    #[inline]
    pub fn clear_outer_row(&mut self) {
        self.outer_row = None;
    }

    /// Initialize outer columns for correlated subquery compilation
    pub fn init_outer_columns(&mut self, outer_columns: &[String]) {
        self.outer_columns = Some(outer_columns.to_vec());
        // Invalidate local cache
        self.local_cache.clear();
    }

    /// Check if outer columns are set (for debugging)
    pub fn has_outer_columns(&self) -> bool {
        self.outer_columns.is_some()
    }

    /// Get outer columns (for debugging)
    pub fn get_outer_columns(&self) -> Option<&Vec<String>> {
        self.outer_columns.as_ref()
    }

    /// Clear the current row
    pub fn clear_row(&mut self) {
        self.current_row = None;
        self.current_row2 = None;
    }

    /// Compute hash of expression content for local cache key.
    /// Fast recursive hash that avoids string allocation.
    /// Uses FxHasher which is 2-5x faster than SipHash for small keys.
    #[inline]
    fn expr_hash(&self, expr: &Expression) -> u64 {
        let mut hasher = FxHasher::default();
        Self::hash_expression(expr, &mut hasher);
        hasher.finish()
    }

    /// Recursively hash an expression without string allocation
    fn hash_expression(expr: &Expression, hasher: &mut FxHasher) {
        // First hash the discriminant to distinguish variants
        std::mem::discriminant(expr).hash(hasher);

        match expr {
            Expression::Identifier(id) => {
                id.value_lower.hash(hasher);
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
                in_hash.values.len().hash(hasher);
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
            return Ok(Arc::clone(program));
        }

        // Cache miss: compile the expression
        let program = Arc::new(self.compile_expression(expr)?);
        self.local_cache.insert(expr_key, Arc::clone(&program));

        Ok(program)
    }

    /// Compile an expression to a Program
    fn compile_expression(&self, expr: &Expression) -> Result<Program> {
        let mut ctx = CompileContext::new(&self.columns, self.function_registry);

        // Add second row columns if available
        if let Some(ref cols2) = self.columns2 {
            ctx = ctx.with_second_row(cols2);
        }

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

        // Get second row if in join mode
        let row2 = self.current_row2.as_ref();

        // Build execution context
        let mut ctx = if let Some(r2) = row2 {
            ExecuteContext::for_join(row, r2)
        } else {
            ExecuteContext::new(row)
        };

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

        // Get second row if in join mode
        let row2 = self.current_row2.as_ref();

        // Build execution context
        let mut ctx = if let Some(r2) = row2 {
            ExecuteContext::for_join(row, r2)
        } else {
            ExecuteContext::new(row)
        };

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
        Ok(self.vm.execute_bool(&program, &ctx))
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
            token: dummy_token(),
            value: name.into(),
            value_lower: name.to_lowercase().into(),
        })
    }

    fn make_int_literal(value: i64) -> Expression {
        Expression::IntegerLiteral(IntegerLiteral {
            token: dummy_token(),
            value,
        })
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
        assert!(filter.matches(&row));
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
        assert!(!filter.matches(&row));
    }

    #[test]
    fn test_row_filter_evaluate() {
        // col + 10
        let expr = make_infix(
            make_identifier("col"),
            InfixOperator::Add,
            make_int_literal(10),
        );
        let columns = vec!["col".to_string()];
        let filter = RowFilter::new(&expr, &columns).unwrap();

        let row = Row::from(vec![Value::Integer(5)]);
        let result = filter.evaluate(&row).unwrap();
        assert_eq!(result, Value::Integer(15));
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
        assert!(filter.matches(&row));
        assert!(cloned.matches(&row));
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
        assert!(eval.eval_bool(&row));

        let row = Row::from(vec![Value::Integer(3)]);
        assert!(!eval.eval_bool(&row));
    }

    // =========================================================================
    // MultiExpressionEval tests
    // =========================================================================

    #[test]
    fn test_multi_expression_eval_compile() {
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

        let eval = MultiExpressionEval::compile(&[expr1, expr2], &columns);
        assert!(eval.is_ok());
        assert_eq!(eval.unwrap().len(), 2);
    }

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
        let mut eval = MultiExpressionEval::compile(&[expr1, expr2], &columns).unwrap();

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
    fn test_compiled_evaluator_default() {
        let eval = CompiledEvaluator::default();
        assert!(eval.columns.is_empty());
    }
}
