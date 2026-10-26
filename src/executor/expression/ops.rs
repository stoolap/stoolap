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

// Compiled Expression Operations
//
// These operations form the instruction set for the expression VM.
// Each operation is designed to be:
// - Self-contained (no external dependencies during execution)
// - Fast to dispatch (small enum, good for branch prediction)
// - Zero allocation (all data pre-computed at compile time)

use std::sync::Arc;

use crate::common::CompactArc;

use crate::core::interval::IntervalUnit;
use crate::core::pattern::Pattern;
use crate::core::predicate::{sql_compare, sql_eq, sql_rows_distinct, PredicateSet};
use crate::core::{DataType, Value};
use crate::functions::{NativeFn1, ScalarFunction};

/// An INTERVAL literal: `quantity` times one `unit`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntervalStep {
    pub quantity: i64,
    pub unit: IntervalUnit,
}

/// Expression VM Operation
///
/// Each operation is self-contained with all data needed for execution.
/// No external lookups during execution - everything resolved at compile time.
#[derive(Clone)]
pub enum Op {
    // =========================================================================
    // LOAD OPERATIONS - Push values onto stack
    // =========================================================================
    /// Load column value by pre-resolved index
    /// Stack: [] -> [value]
    LoadColumn(u16),

    /// Load column from second row (for joins)
    /// Stack: [] -> [value]
    LoadColumn2(u16),

    /// Load from outer row context (for correlated subqueries)
    /// Uses pre-resolved key
    /// Stack: [] -> [value]
    LoadOuterColumn(CompactArc<str>),

    /// Load constant value (pre-cloned at compile time)
    /// Stack: [] -> [value]
    LoadConst(Value),

    /// Load query parameter by index
    /// Stack: [] -> [value]
    LoadParam(u32),

    /// Load NULL with type hint
    /// Stack: [] -> [null]
    LoadNull(DataType),

    // =========================================================================
    // COMPARISON OPERATIONS - Pop 2, push bool
    // =========================================================================
    /// Equal: a == b
    Eq,
    /// Not equal: a != b
    Ne,
    /// Less than: a < b
    Lt,
    /// Less than or equal: a <= b
    Le,
    /// Greater than: a > b
    Gt,
    /// Greater than or equal: a >= b
    Ge,

    /// IS NULL check
    /// Stack: [value] -> [bool]
    IsNull,

    /// IS NOT NULL check
    /// Stack: [value] -> [bool]
    IsNotNull,

    /// IS DISTINCT FROM (NULL-safe not equal)
    /// Stack: [a, b] -> [bool]
    IsDistinctFrom,

    /// IS NOT DISTINCT FROM (NULL-safe equal)
    /// Stack: [a, b] -> [bool]
    IsNotDistinctFrom,

    /// Two rows of N values compared: (a1, .., aN) op (b1, .., bN)
    /// Stack: [a1, .., aN, b1, .., bN] -> [bool]
    CompareRows(CompareOp, u16),

    /// Two rows of N values, IS DISTINCT FROM, or IS NOT DISTINCT FROM when
    /// negated
    /// Stack: [a1, .., aN, b1, .., bN] -> [bool]
    RowsDistinct(bool, u16),

    // =========================================================================
    // FUSED COMPARISON OPERATIONS - Single instruction for column vs constant
    // These avoid push/pop overhead for the most common filter patterns
    // =========================================================================
    /// Fused: column == constant
    /// Stack: [] -> [bool]
    EqColumnConst(u16, Value),

    /// Fused: column != constant
    /// Stack: [] -> [bool]
    NeColumnConst(u16, Value),

    /// Fused: column < constant
    /// Stack: [] -> [bool]
    LtColumnConst(u16, Value),

    /// Fused: column <= constant
    /// Stack: [] -> [bool]
    LeColumnConst(u16, Value),

    /// Fused: column > constant
    /// Stack: [] -> [bool]
    GtColumnConst(u16, Value),

    /// Fused: column >= constant
    /// Stack: [] -> [bool]
    GeColumnConst(u16, Value),

    /// Fused: column IS NULL
    /// Stack: [] -> [bool]
    IsNullColumn(u16),

    /// Fused: column IS NOT NULL
    /// Stack: [] -> [bool]
    IsNotNullColumn(u16),

    /// Fused: column LIKE pattern
    /// Stack: [] -> [bool]
    LikeColumn(u16, Arc<Pattern>), // col_idx, pattern

    /// Fused: column IN (constant set with AHash)
    /// Stack: [] -> [bool]
    InSetColumn(u16, CompactArc<PredicateSet>), // col_idx, set

    /// Fused: column BETWEEN low AND high (constants)
    /// Stack: [] -> [bool]
    BetweenColumnConst(u16, Value, Value), // col_idx, low, high

    // =========================================================================
    // LOGICAL OPERATIONS
    // =========================================================================
    /// Logical AND with short-circuit
    /// If top of stack is false, jump to target
    /// Stack: [bool] -> [bool] (or jump)
    And(u16), // Jump target if false

    /// Logical OR with short-circuit
    /// If top of stack is true, jump to target
    /// Stack: [bool] -> [bool] (or jump)
    Or(u16), // Jump target if true

    /// Logical NOT
    /// Stack: [bool] -> [bool]
    Not,

    /// Logical XOR
    /// Stack: [a, b] -> [bool]
    Xor,

    /// AND finalize - combine left and right results
    /// Stack: [left_bool, right_bool] -> [bool]
    AndFinalize,

    /// OR finalize - combine left and right results
    /// Stack: [left_bool, right_bool] -> [bool]
    OrFinalize,

    // =========================================================================
    // ARITHMETIC OPERATIONS - Pop 2, push result
    // =========================================================================
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    /// Unary negation
    /// Stack: [value] -> [-value]
    Neg,

    // =========================================================================
    // BITWISE OPERATIONS
    // =========================================================================
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

    // =========================================================================
    // STRING OPERATIONS
    // =========================================================================
    /// String concatenation (binary)
    /// Stack: [a, b] -> [a || b]
    Concat,

    /// Multi-value string concatenation (optimized for chained ||)
    /// Stack: [v1, v2, ..., vN] -> [v1 || v2 || ... || vN]
    /// Pre-calculates total length and allocates once
    ConcatN(u8),

    /// LIKE pattern match (pre-compiled pattern)
    /// Stack: [text] -> [bool]
    Like(Arc<Pattern>),

    /// GLOB pattern match
    /// Stack: [text] -> [bool]
    Glob(Arc<Pattern>),

    /// REGEXP match (pre-compiled regex)
    /// Stack: [text] -> [bool]
    Regexp(Arc<regex::Regex>),

    /// LIKE with ESCAPE character
    /// Stack: [text] -> [bool]
    LikeEscape(Arc<Pattern>, char), // pattern, escape_char

    /// Dynamic LIKE: pattern is on the stack (e.g. from a parameter)
    /// Stack: [text, pattern_text] -> [bool]
    LikeDynamic(bool), // case_insensitive

    /// Dynamic LIKE with ESCAPE: pattern is on the stack, escape char is compiled in
    /// Stack: [text, pattern_text] -> [bool]
    LikeDynamicEscape(bool, char), // case_insensitive, escape_char

    /// Dynamic LIKE with ESCAPE: both pattern and escape are on the stack
    /// Stack: [text, pattern_text, escape_text] -> [bool]
    LikeDynamicEscapeExpr(bool), // case_insensitive

    /// Dynamic GLOB: pattern is on the stack
    /// Stack: [text, pattern_text] -> [bool]
    GlobDynamic,

    /// Dynamic REGEXP: pattern is on the stack
    /// Stack: [text, pattern_text] -> [bool]
    RegexpDynamic,

    // =========================================================================
    // JSON OPERATIONS
    // =========================================================================
    /// JSON access: json -> key (returns JSON)
    /// Stack: [json, key] -> [json_value]
    JsonAccess,

    /// JSON access text: json ->> key (returns TEXT)
    /// Stack: [json, key] -> [text_value]
    JsonAccessText,

    // =========================================================================
    // TIMESTAMP OPERATIONS
    // =========================================================================
    /// Add interval to timestamp: timestamp + interval_string
    /// Stack: [timestamp, interval_text] -> [timestamp]
    TimestampAddInterval,

    /// Subtract interval from timestamp: timestamp - interval_string
    /// Stack: [timestamp, interval_text] -> [timestamp]
    TimestampSubInterval,

    /// Subtract timestamps: timestamp - timestamp (returns interval text)
    /// Stack: [timestamp1, timestamp2] -> [interval_text]
    TimestampDiff,

    /// Add days to timestamp: timestamp + integer
    /// Stack: [timestamp, days] -> [timestamp]
    TimestampAddDays,

    /// Subtract days from timestamp: timestamp - integer
    /// Stack: [timestamp, days] -> [timestamp]
    TimestampSubDays,

    /// An INTERVAL literal times the product of `count` numbers, as interval text
    /// Stack: [number; count] -> [interval_text]
    ScaleInterval(IntervalStep, u8),

    /// Add an INTERVAL literal times the product of `count` numbers to a
    /// timestamp, or subtract it when the flag is set
    /// Stack: [timestamp, number; count] -> [timestamp]
    TimestampAddScaledInterval(IntervalStep, bool, u8),

    // =========================================================================
    // VECTOR DISTANCE OPERATIONS
    // =========================================================================
    /// L2 (Euclidean) distance between two vectors
    /// Stack: [vector1, vector2] -> [float]
    VectorDistanceL2,

    /// Cosine distance between two vectors (1 - cosine_similarity)
    /// Stack: [vector1, vector2] -> [float]
    VectorDistanceCosine,

    /// Negative inner product distance between two vectors
    /// Stack: [vector1, vector2] -> [float]
    VectorDistanceIP,

    // =========================================================================
    // SET OPERATIONS
    // =========================================================================
    /// IN set membership (pre-built FxHashSet for fast lookups)
    /// Stack: [value] -> [bool]
    InSet(CompactArc<PredicateSet>),

    /// NOT IN set membership
    /// Stack: [value] -> [bool]
    NotInSet(CompactArc<PredicateSet>),

    /// IN over items evaluated at run time; the value is judged once
    /// Stack: [value, item_1, ..., item_n] -> [bool]
    InList(u32), // item count

    /// `value op ANY` or `value op ALL` over members a subquery yielded
    /// Stack: [value] -> [bool]
    QuantifiedSet(CompactArc<QuantifiedMembers>),

    /// BETWEEN check: value BETWEEN low AND high
    /// Stack: [value, low, high] -> [bool]
    Between,

    /// NOT BETWEEN check
    /// Stack: [value, low, high] -> [bool]
    NotBetween,

    /// Multi-column IN: (a, b) IN ((1, 2), (3, 4))
    /// Stack: [val1, val2, ...valN] -> [bool]
    /// The tuple_values contains pre-evaluated constant tuples
    InTupleSet {
        tuple_size: u8,
        values: Arc<Vec<Vec<Value>>>, // List of tuples
        negated: bool,
    },

    /// A row of N values IN a list of K rows of N values read per row
    /// Stack: [a1, .., aN, r1_1, .., r1_N, .., rK_N] -> [bool]
    InRows(u16, u32),

    // =========================================================================
    // BOOLEAN CHECKS
    // =========================================================================
    /// IS TRUE check
    /// Stack: [value] -> [bool]
    IsTrue,

    /// IS NOT TRUE check
    /// Stack: [value] -> [bool]
    IsNotTrue,

    /// IS FALSE check
    /// Stack: [value] -> [bool]
    IsFalse,

    /// IS NOT FALSE check
    /// Stack: [value] -> [bool]
    IsNotFalse,

    // =========================================================================
    // FUNCTION CALLS
    // =========================================================================
    /// Call scalar function with N arguments
    /// Stack: [arg1, arg2, ..., argN] -> [result]
    CallScalar {
        func: Arc<dyn ScalarFunction>,
        arg_count: u8,
    },

    /// Special: COALESCE - return first non-null
    /// Stack: [arg1, ..., argN] -> [result]
    Coalesce(u8), // arg count

    /// Special: NULLIF(a, b) - return NULL if a = b
    /// Stack: [a, b] -> [a or null]
    NullIf,

    /// Special: GREATEST - return max of args
    /// Stack: [arg1, ..., argN] -> [result]
    Greatest(u8),

    /// Special: LEAST - return min of args
    /// Stack: [arg1, ..., argN] -> [result]
    Least(u8),

    // =========================================================================
    // NATIVE SCALAR FUNCTIONS (function pointer, no dynamic dispatch)
    // =========================================================================
    /// Native scalar function - single argument, direct function pointer call
    /// Stack: [value] -> [result]
    NativeFn1(NativeFn1),

    // =========================================================================
    // TYPE OPERATIONS
    // =========================================================================
    /// Cast value to target type
    /// Stack: [value] -> [casted_value]
    Cast(DataType),

    /// Read a value compared with a column of this type as the scan reads
    /// it; a TEXT that is no number stays TEXT for a numeric column
    /// Stack: [value] -> [value as the column's type]
    ReadAsColumn(DataType),

    /// Truncate timestamp to date (midnight)
    /// Used for CAST(timestamp AS DATE) - truncates time component to 00:00:00
    /// Stack: [value] -> [timestamp_at_midnight]
    TruncateToDate,

    // =========================================================================
    // CASE EXPRESSION
    // =========================================================================
    /// Start of CASE - marks beginning
    CaseStart,

    /// WHEN condition: if top is false, jump to next branch
    /// Stack: [bool] -> [] (condition consumed)
    CaseWhen(u16), // Jump to next WHEN/ELSE/END if false

    /// THEN result: jump to CASE end after pushing result
    /// Stack: [value] -> [value] (then jump)
    CaseThen(u16), // Jump to END

    /// ELSE clause marker
    CaseElse,

    /// End of CASE
    CaseEnd,

    /// Simple CASE: compare value with WHEN value
    /// Stack: [case_value, when_value] -> [bool]
    CaseCompare,

    // =========================================================================
    // CONTROL FLOW
    // =========================================================================
    /// Unconditional jump
    Jump(u16),

    /// Jump if top of stack is true (doesn't pop)
    JumpIfTrue(u16),

    /// Jump if top of stack is false (doesn't pop)
    JumpIfFalse(u16),

    /// Jump if top of stack is NULL (doesn't pop)
    JumpIfNull(u16),

    /// Jump if top of stack is NOT NULL (doesn't pop)
    /// Used for COALESCE short-circuit evaluation
    JumpIfNotNull(u16),

    /// Pop and jump if true
    PopJumpIfTrue(u16),

    /// Pop and jump if false
    PopJumpIfFalse(u16),

    /// Duplicate top of stack
    Dup,

    /// Pop top of stack (discard)
    Pop,

    /// Swap top two stack elements
    Swap,

    // =========================================================================
    // SUBQUERY OPERATIONS (for correlated/scalar subqueries)
    // =========================================================================
    /// Execute scalar subquery and push result
    /// The subquery plan is stored separately and referenced by index
    /// Stack: [] -> [value]
    ExecScalarSubquery(u16), // Subquery plan index

    /// Execute EXISTS subquery
    /// Stack: [] -> [bool]
    ExecExists(u16), // Subquery plan index

    /// Execute IN subquery: TRUE, FALSE or UNKNOWN
    /// Stack: [value] -> [bool or NULL]
    ExecInSubquery(u16), // Subquery plan index

    /// Execute `value op ALL` (true) or `value op ANY` (false): TRUE, FALSE or UNKNOWN
    /// Stack: [value] -> [bool or NULL]
    ExecQuantified(u16, CompareOp, bool), // Subquery index, comparison operator, ALL

    /// A row of N values tested against a row block's members, by the IN,
    /// ANY or ALL test the block answers: TRUE, FALSE or UNKNOWN
    /// Stack: [a1, .., aN] -> [bool or NULL]
    ExecRowMembers(u16, u16), // Subquery index, N

    // =========================================================================
    // AGGREGATE REFERENCES (post-aggregation)
    // =========================================================================
    /// Load pre-computed aggregate result by column index
    /// Used in HAVING clauses where aggregates are already computed
    /// Stack: [] -> [value]
    LoadAggregateResult(u16),

    /// Load current transaction ID
    /// Returns NULL if no transaction is active
    /// Stack: [] -> [value]
    LoadTransactionId,

    // =========================================================================
    // SPECIAL
    // =========================================================================
    /// No operation (placeholder)
    Nop,

    /// Return current top of stack as result
    Return,

    /// Return true immediately
    ReturnTrue,

    /// Return false immediately
    ReturnFalse,

    /// Return NULL immediately
    ReturnNull(DataType),
}

/// The members of `value op ANY` or `value op ALL`, each compared with the
/// value by the comparison contract
#[derive(Debug, Clone)]
pub struct QuantifiedMembers {
    op: CompareOp,
    all: bool,
    members: CompactArc<Vec<Value>>,
    has_null: bool,
    /// The member that decides for a numeric value, when every member is
    /// INTEGER, or every one FLOAT, so their order is the comparison's: the
    /// largest for `< ANY` and `> ALL`, the smallest else
    bound: Option<Value>,
}

impl QuantifiedMembers {
    /// `members` holds no NULL; `has_null` says whether the subquery yielded one
    pub fn new(op: CompareOp, all: bool, members: CompactArc<Vec<Value>>, has_null: bool) -> Self {
        let bound = Self::bound(op, all, &members);
        Self {
            op,
            all,
            members,
            has_null,
            bound,
        }
    }

    /// The comparison, whether it is ALL, the members without NULL, and
    /// whether the subquery yielded a NULL
    pub fn parts(&self) -> (CompareOp, bool, &[Value], bool) {
        (self.op, self.all, &self.members, self.has_null)
    }

    /// The member a numeric value is compared with in place of every
    /// member, when one exists: see the field of that name
    pub fn bound(op: CompareOp, all: bool, members: &[Value]) -> Option<Value> {
        let ordered = members.iter().all(|v| matches!(v, Value::Integer(_)))
            || members.iter().all(|v| matches!(v, Value::Float(_)));
        let largest = matches!(
            (op, all),
            (CompareOp::Lt | CompareOp::Le, false) | (CompareOp::Gt | CompareOp::Ge, true)
        );
        match op {
            CompareOp::Eq | CompareOp::Ne => None,
            _ if !ordered => None,
            _ if largest => members.iter().max_by(|a, b| a.cmp(b)).cloned(),
            _ => members.iter().min_by(|a, b| a.cmp(b)).cloned(),
        }
    }

    /// `value op member` by the comparison contract; None is UNKNOWN
    pub fn compares(op: CompareOp, value: &Value, member: &Value) -> Option<bool> {
        match op {
            CompareOp::Eq => sql_eq(value, member),
            CompareOp::Ne => sql_eq(value, member).map(|eq| !eq),
            CompareOp::Lt => sql_compare(value, member).map(|o| o.is_lt()),
            CompareOp::Le => sql_compare(value, member).map(|o| o.is_le()),
            CompareOp::Gt => sql_compare(value, member).map(|o| o.is_gt()),
            CompareOp::Ge => sql_compare(value, member).map(|o| o.is_ge()),
        }
    }

    /// The operator a quantified comparison is written with
    pub fn operator(written: &str) -> Option<CompareOp> {
        Some(match written {
            "=" => CompareOp::Eq,
            "<>" | "!=" => CompareOp::Ne,
            "<" => CompareOp::Lt,
            "<=" => CompareOp::Le,
            ">" => CompareOp::Gt,
            ">=" => CompareOp::Ge,
            _ => return None,
        })
    }

    /// The three-valued answer for `value`; over no members at all ANY is
    /// FALSE and ALL is TRUE, whatever `value` is
    pub fn answer(&self, value: &Value) -> Option<bool> {
        // A decided member settles it: TRUE for ANY, FALSE for ALL
        let decides = !self.all;
        let mut unknown = self.has_null;
        // Against numeric members a TEXT reads as its number, once, or is UNKNOWN to all
        let parsed;
        let number = match (value, self.bound.is_some()) {
            (Value::Integer(_) | Value::Float(_), true) => Some(value),
            (Value::Text(text), true) => {
                parsed = crate::core::predicate::parse_sql_number(text)?;
                Some(&parsed)
            }
            _ => None,
        };
        let by_contract = |member: &Value, value: &Value| Self::compares(self.op, value, member);
        match self.bound.as_ref().zip(number) {
            Some((bound, number)) => match by_contract(bound, number) {
                Some(result) if result == decides => return Some(decides),
                Some(_) => {}
                None => unknown = true,
            },
            None => {
                for member in self.members.iter() {
                    #[cfg(feature = "test-failpoints")]
                    crate::test_failpoints::quantified_member_compared();
                    match by_contract(member, value) {
                        Some(result) if result == decides => return Some(decides),
                        Some(_) => {}
                        None => unknown = true,
                    }
                }
            }
        }
        if unknown {
            None
        } else {
            Some(!decides)
        }
    }
}

/// Comparison operator for ALL/ANY subqueries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    pub fn compare(&self, a: &Value, b: &Value) -> Option<bool> {
        QuantifiedMembers::compares(*self, a, b)
    }

    /// `(a1, ..) op (b1, ..)` over two rows of one width: `=` holds when
    /// every pair is equal, `<>` when one is not; an ordering is the first
    /// unequal pair's, UNKNOWN when that pair's equality is
    pub fn compare_rows(&self, a: &[Value], b: &[Value]) -> Option<bool> {
        debug_assert_eq!(a.len(), b.len());
        match self {
            CompareOp::Eq => return rows_equal(a, b),
            CompareOp::Ne => return rows_equal(a, b).map(|eq| !eq),
            _ => {}
        }
        for (x, y) in a.iter().zip(b) {
            match QuantifiedMembers::compares(CompareOp::Eq, x, y) {
                Some(true) => {}
                Some(false) => return self.compare(x, y),
                None => return None,
            }
        }
        Some(matches!(self, CompareOp::Le | CompareOp::Ge))
    }
}

/// `=` over two rows: false when a pair is unequal, else UNKNOWN when a
/// pair's equality is, else true
#[inline]
fn rows_equal(a: &[Value], b: &[Value]) -> Option<bool> {
    let mut equal = Some(true);
    for (x, y) in a.iter().zip(b) {
        match QuantifiedMembers::compares(CompareOp::Eq, x, y) {
            Some(true) => {}
            Some(false) => return Some(false),
            None => equal = None,
        }
    }
    equal
}

/// `row IN (list)`: true when a row of the list equals `row`, else UNKNOWN
/// when one's equality is, else false
pub fn row_in<'v>(row: &[Value], list: impl IntoIterator<Item = &'v [Value]>) -> Option<bool> {
    quantified_rows(CompareOp::Eq, false, row, list)
}

/// `row op ALL (members)` when `all`, else `row op ANY (members)`: a member
/// that settles it wins (TRUE for ANY, FALSE for ALL), else UNKNOWN when
/// one's comparison is; over no members ANY is FALSE and ALL TRUE
pub fn quantified_rows<'v>(
    op: CompareOp,
    all: bool,
    row: &[Value],
    members: impl IntoIterator<Item = &'v [Value]>,
) -> Option<bool> {
    // The operator chosen once, so a member costs its comparison alone
    match op {
        CompareOp::Eq => settle_rows(all, members, |member| rows_equal(row, member)),
        CompareOp::Ne => settle_rows(all, members, |member| rows_equal(row, member).map(|eq| !eq)),
        _ => settle_rows(all, members, |member| op.compare_rows(row, member)),
    }
}

/// [`quantified_rows`] over the members' answers
#[inline(always)]
fn settle_rows<'v>(
    all: bool,
    members: impl IntoIterator<Item = &'v [Value]>,
    compare: impl Fn(&[Value]) -> Option<bool>,
) -> Option<bool> {
    let mut unknown = false;
    for member in members {
        match compare(member) {
            Some(answer) if answer != all => return Some(answer),
            Some(_) => {}
            None => unknown = true,
        }
    }
    (!unknown).then_some(all)
}

/// How two rows compare: by an operator, or IS [NOT] DISTINCT FROM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowComparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Distinct,
    NotDistinct,
}

impl RowComparison {
    pub fn answer(self, a: &[Value], b: &[Value]) -> Option<bool> {
        let op = match self {
            RowComparison::Eq => CompareOp::Eq,
            RowComparison::Ne => CompareOp::Ne,
            RowComparison::Lt => CompareOp::Lt,
            RowComparison::Le => CompareOp::Le,
            RowComparison::Gt => CompareOp::Gt,
            RowComparison::Ge => CompareOp::Ge,
            RowComparison::Distinct => return Some(sql_rows_distinct(a, b)),
            RowComparison::NotDistinct => return Some(!sql_rows_distinct(a, b)),
        };
        op.compare_rows(a, b)
    }

    /// The comparison with its sides swapped
    pub fn flipped(self) -> Self {
        match self {
            RowComparison::Lt => RowComparison::Gt,
            RowComparison::Le => RowComparison::Ge,
            RowComparison::Gt => RowComparison::Lt,
            RowComparison::Ge => RowComparison::Le,
            other => other,
        }
    }
}

/// What a row block's members answer for a row
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowTest {
    /// `row op ANY (members)`, or `op ALL` when true; IN is `=` ANY
    Quantified(CompareOp, bool),
    /// `row` compared with the one member, every value NULL for none
    One(RowComparison),
}

/// A row block's members, `width` values each, as `test` asks them
#[derive(Debug, Clone)]
pub struct RowMembers {
    test: RowTest,
    width: usize,
    values: Vec<Value>,
}

impl RowMembers {
    /// `values` holds the members' values in turn, `width` of them each; a
    /// row compared with the one member is compared with NULLs for none
    pub fn new(test: RowTest, width: usize, mut values: Vec<Value>) -> Self {
        debug_assert!(width > 1 && values.len().is_multiple_of(width));
        if matches!(test, RowTest::One(_)) && values.is_empty() {
            values.resize(width, Value::null_unknown());
        }
        Self {
            test,
            width,
            values,
        }
    }

    /// The answer for `row`; more than one member for a row compared with
    /// one is an error
    pub fn answer(&self, row: &[Value]) -> crate::core::Result<Option<bool>> {
        match self.test {
            RowTest::Quantified(op, all) => Ok(quantified_rows(
                op,
                all,
                row,
                self.values.chunks_exact(self.width),
            )),
            RowTest::One(comparison) => match self.values.len() == self.width {
                true => Ok(comparison.answer(row, &self.values)),
                false => Err(more_than_one_row()),
            },
        }
    }

    /// The answer one member settles: TRUE for ANY, FALSE for ALL; none
    /// for a row compared with the one member
    pub fn decides(&self) -> Option<bool> {
        match self.test {
            RowTest::Quantified(_, all) => Some(!all),
            RowTest::One(_) => None,
        }
    }
}

/// A subquery read for one row's value or values gave more
pub(crate) fn more_than_one_row() -> crate::core::Error {
    crate::core::Error::Internal {
        message: "scalar subquery returned more than one row".to_string(),
    }
}

// Make Op Debug-printable (without showing full function pointers)
impl std::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::LoadColumn(idx) => write!(f, "LoadColumn({})", idx),
            Op::LoadColumn2(idx) => write!(f, "LoadColumn2({})", idx),
            Op::LoadOuterColumn(name) => write!(f, "LoadOuterColumn({})", name),
            Op::LoadConst(v) => write!(f, "LoadConst({:?})", v),
            Op::LoadParam(idx) => write!(f, "LoadParam({})", idx),
            Op::LoadNull(dt) => write!(f, "LoadNull({:?})", dt),
            Op::Eq => write!(f, "Eq"),
            Op::Ne => write!(f, "Ne"),
            Op::Lt => write!(f, "Lt"),
            Op::Le => write!(f, "Le"),
            Op::Gt => write!(f, "Gt"),
            Op::Ge => write!(f, "Ge"),
            Op::IsNull => write!(f, "IsNull"),
            Op::IsNotNull => write!(f, "IsNotNull"),
            Op::IsDistinctFrom => write!(f, "IsDistinctFrom"),
            Op::IsNotDistinctFrom => write!(f, "IsNotDistinctFrom"),
            Op::EqColumnConst(col, val) => write!(f, "EqColumnConst({}, {:?})", col, val),
            Op::NeColumnConst(col, val) => write!(f, "NeColumnConst({}, {:?})", col, val),
            Op::LtColumnConst(col, val) => write!(f, "LtColumnConst({}, {:?})", col, val),
            Op::LeColumnConst(col, val) => write!(f, "LeColumnConst({}, {:?})", col, val),
            Op::GtColumnConst(col, val) => write!(f, "GtColumnConst({}, {:?})", col, val),
            Op::GeColumnConst(col, val) => write!(f, "GeColumnConst({}, {:?})", col, val),
            Op::IsNullColumn(col) => write!(f, "IsNullColumn({})", col),
            Op::IsNotNullColumn(col) => write!(f, "IsNotNullColumn({})", col),
            Op::LikeColumn(col, _) => write!(f, "LikeColumn({})", col),
            Op::InSetColumn(col, set) => {
                write!(
                    f,
                    "InSetColumn({}, len={}, has_null={})",
                    col,
                    set.len(),
                    set.has_null()
                )
            }
            Op::BetweenColumnConst(col, low, high) => {
                write!(f, "BetweenColumnConst({}, {:?}, {:?})", col, low, high)
            }
            Op::And(target) => write!(f, "And(jump={})", target),
            Op::Or(target) => write!(f, "Or(jump={})", target),
            Op::Not => write!(f, "Not"),
            Op::Xor => write!(f, "Xor"),
            Op::AndFinalize => write!(f, "AndFinalize"),
            Op::OrFinalize => write!(f, "OrFinalize"),
            Op::Add => write!(f, "Add"),
            Op::Sub => write!(f, "Sub"),
            Op::Mul => write!(f, "Mul"),
            Op::Div => write!(f, "Div"),
            Op::Mod => write!(f, "Mod"),
            Op::Neg => write!(f, "Neg"),
            Op::BitAnd => write!(f, "BitAnd"),
            Op::BitOr => write!(f, "BitOr"),
            Op::BitXor => write!(f, "BitXor"),
            Op::BitNot => write!(f, "BitNot"),
            Op::Shl => write!(f, "Shl"),
            Op::Shr => write!(f, "Shr"),
            Op::Concat => write!(f, "Concat"),
            Op::Like(_) => write!(f, "Like"),
            Op::Glob(_) => write!(f, "Glob"),
            Op::Regexp(_) => write!(f, "Regexp"),
            Op::LikeEscape(_, esc) => write!(f, "LikeEscape(escape='{}')", esc),
            Op::LikeDynamic(ci) => write!(f, "LikeDynamic(case_insensitive={})", ci),
            Op::LikeDynamicEscape(ci, esc) => {
                write!(
                    f,
                    "LikeDynamicEscape(case_insensitive={}, escape='{}')",
                    ci, esc
                )
            }
            Op::LikeDynamicEscapeExpr(ci) => {
                write!(f, "LikeDynamicEscapeExpr(case_insensitive={})", ci)
            }
            Op::GlobDynamic => write!(f, "GlobDynamic"),
            Op::RegexpDynamic => write!(f, "RegexpDynamic"),
            Op::JsonAccess => write!(f, "JsonAccess"),
            Op::JsonAccessText => write!(f, "JsonAccessText"),
            Op::TimestampAddInterval => write!(f, "TimestampAddInterval"),
            Op::TimestampSubInterval => write!(f, "TimestampSubInterval"),
            Op::TimestampDiff => write!(f, "TimestampDiff"),
            Op::TimestampAddDays => write!(f, "TimestampAddDays"),
            Op::TimestampSubDays => write!(f, "TimestampSubDays"),
            Op::ScaleInterval(step, count) => write!(f, "ScaleInterval({:?}, {})", step, count),
            Op::TimestampAddScaledInterval(step, subtract, count) => write!(
                f,
                "TimestampAddScaledInterval({:?}, {}, {})",
                step, subtract, count
            ),
            Op::VectorDistanceL2 => write!(f, "VectorDistanceL2"),
            Op::VectorDistanceCosine => write!(f, "VectorDistanceCosine"),
            Op::VectorDistanceIP => write!(f, "VectorDistanceIP"),
            Op::InSet(set) => {
                write!(f, "InSet(len={}, has_null={})", set.len(), set.has_null())
            }
            Op::NotInSet(set) => {
                write!(
                    f,
                    "NotInSet(len={}, has_null={})",
                    set.len(),
                    set.has_null()
                )
            }
            Op::InList(count) => write!(f, "InList({})", count),
            Op::QuantifiedSet(set) => write!(f, "QuantifiedSet({})", set.members.len()),
            Op::Between => write!(f, "Between"),
            Op::NotBetween => write!(f, "NotBetween"),
            Op::InTupleSet {
                tuple_size,
                values,
                negated,
            } => {
                write!(
                    f,
                    "InTupleSet(size={}, tuples={}, negated={})",
                    tuple_size,
                    values.len(),
                    negated
                )
            }
            Op::InRows(width, count) => write!(f, "InRows({}, {})", width, count),
            Op::CompareRows(op, width) => write!(f, "CompareRows({:?}, {})", op, width),
            Op::RowsDistinct(negated, width) => {
                write!(f, "RowsDistinct(negated={}, {})", negated, width)
            }
            Op::IsTrue => write!(f, "IsTrue"),
            Op::IsNotTrue => write!(f, "IsNotTrue"),
            Op::IsFalse => write!(f, "IsFalse"),
            Op::IsNotFalse => write!(f, "IsNotFalse"),
            Op::CallScalar { arg_count, .. } => write!(f, "CallScalar(args={})", arg_count),
            Op::Coalesce(n) => write!(f, "Coalesce({})", n),
            Op::NullIf => write!(f, "NullIf"),
            Op::Greatest(n) => write!(f, "Greatest({})", n),
            Op::Least(n) => write!(f, "Least({})", n),
            Op::Cast(dt) => write!(f, "Cast({:?})", dt),
            Op::ReadAsColumn(dt) => write!(f, "ReadAsColumn({:?})", dt),
            Op::TruncateToDate => write!(f, "TruncateToDate"),
            Op::CaseStart => write!(f, "CaseStart"),
            Op::CaseWhen(target) => write!(f, "CaseWhen(jump={})", target),
            Op::CaseThen(target) => write!(f, "CaseThen(jump={})", target),
            Op::CaseElse => write!(f, "CaseElse"),
            Op::CaseEnd => write!(f, "CaseEnd"),
            Op::CaseCompare => write!(f, "CaseCompare"),
            Op::Jump(target) => write!(f, "Jump({})", target),
            Op::JumpIfTrue(target) => write!(f, "JumpIfTrue({})", target),
            Op::JumpIfFalse(target) => write!(f, "JumpIfFalse({})", target),
            Op::JumpIfNull(target) => write!(f, "JumpIfNull({})", target),
            Op::JumpIfNotNull(target) => write!(f, "JumpIfNotNull({})", target),
            Op::PopJumpIfTrue(target) => write!(f, "PopJumpIfTrue({})", target),
            Op::PopJumpIfFalse(target) => write!(f, "PopJumpIfFalse({})", target),
            Op::Dup => write!(f, "Dup"),
            Op::Pop => write!(f, "Pop"),
            Op::Swap => write!(f, "Swap"),
            Op::ExecScalarSubquery(idx) => write!(f, "ExecScalarSubquery({})", idx),
            Op::ExecExists(idx) => write!(f, "ExecExists({})", idx),
            Op::ExecInSubquery(idx) => write!(f, "ExecInSubquery({})", idx),
            Op::ExecRowMembers(idx, width) => write!(f, "ExecRowMembers({}, {})", idx, width),
            Op::ExecQuantified(idx, op, all) => {
                write!(f, "ExecQuantified({}, {:?}, {})", idx, op, all)
            }
            Op::LoadAggregateResult(idx) => write!(f, "LoadAggregateResult({})", idx),
            Op::LoadTransactionId => write!(f, "LoadTransactionId"),
            Op::Nop => write!(f, "Nop"),
            Op::Return => write!(f, "Return"),
            Op::ReturnTrue => write!(f, "ReturnTrue"),
            Op::ReturnFalse => write!(f, "ReturnFalse"),
            Op::ReturnNull(dt) => write!(f, "ReturnNull({:?})", dt),
            Op::NativeFn1(_) => write!(f, "NativeFn1(...)"),
            Op::ConcatN(n) => write!(f, "ConcatN({})", n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_op_size() {
        let size = mem::size_of::<Op>();
        println!("Op enum size: {} bytes", size);
        println!("Op alignment: {} bytes", mem::align_of::<Op>());

        // The Op enum contains large variants like:
        // - InTupleSet { tuple_size: u8, values: Arc<Vec<Vec<Value>>>, negated: bool }
        // - CallScalar { func: Arc<dyn ScalarFunction>, arg_count: u8 }
        // - GtColumnConst(u16, Value) where Value is 24 bytes

        // We want to keep Op small for cache efficiency
        // Ideally under 32 bytes, definitely under 64 bytes
        assert!(size <= 64, "Op enum is too large: {} bytes", size);
    }

    // =========================================================================
    // CompareOp tests
    // =========================================================================

    #[test]
    fn test_compare_op_eq() {
        let op = CompareOp::Eq;
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(5)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(6)),
            Some(false)
        );
        assert_eq!(
            op.compare(&Value::Integer(6), &Value::Integer(5)),
            Some(false)
        );
    }

    #[test]
    fn test_compare_op_ne() {
        let op = CompareOp::Ne;
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(5)),
            Some(false)
        );
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(6)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Integer(6), &Value::Integer(5)),
            Some(true)
        );
    }

    #[test]
    fn test_compare_op_lt() {
        let op = CompareOp::Lt;
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(6)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(5)),
            Some(false)
        );
        assert_eq!(
            op.compare(&Value::Integer(6), &Value::Integer(5)),
            Some(false)
        );
    }

    #[test]
    fn test_compare_op_le() {
        let op = CompareOp::Le;
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(6)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(5)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Integer(6), &Value::Integer(5)),
            Some(false)
        );
    }

    #[test]
    fn test_compare_op_gt() {
        let op = CompareOp::Gt;
        assert_eq!(
            op.compare(&Value::Integer(6), &Value::Integer(5)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(5)),
            Some(false)
        );
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(6)),
            Some(false)
        );
    }

    #[test]
    fn test_compare_op_ge() {
        let op = CompareOp::Ge;
        assert_eq!(
            op.compare(&Value::Integer(6), &Value::Integer(5)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(5)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Integer(5), &Value::Integer(6)),
            Some(false)
        );
    }

    #[test]
    fn test_compare_op_with_booleans() {
        // Test boolean comparisons
        let op = CompareOp::Eq;
        assert_eq!(
            op.compare(&Value::Boolean(true), &Value::Boolean(true)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Boolean(true), &Value::Boolean(false)),
            Some(false)
        );
        assert_eq!(
            op.compare(&Value::Boolean(false), &Value::Boolean(false)),
            Some(true)
        );

        let op_ne = CompareOp::Ne;
        assert_eq!(
            op_ne.compare(&Value::Boolean(true), &Value::Boolean(false)),
            Some(true)
        );
    }

    #[test]
    fn test_compare_op_strings() {
        let op = CompareOp::Lt;
        assert_eq!(
            op.compare(&Value::Text("apple".into()), &Value::Text("banana".into())),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Text("banana".into()), &Value::Text("apple".into())),
            Some(false)
        );
    }

    #[test]
    fn test_compare_op_floats() {
        let op = CompareOp::Ge;
        assert_eq!(
            op.compare(&Value::Float(2.5), &Value::Float(2.5)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Float(2.6), &Value::Float(2.5)),
            Some(true)
        );
        assert_eq!(
            op.compare(&Value::Float(2.4), &Value::Float(2.5)),
            Some(false)
        );
    }

    #[test]
    fn test_compare_op_follows_the_predicate_comparison() {
        assert_eq!(
            CompareOp::Eq.compare(&Value::Integer(1), &Value::Text("1".into())),
            Some(true)
        );
        assert_eq!(
            CompareOp::Le.compare(&Value::Boolean(false), &Value::Text("yes".into())),
            Some(true)
        );
        assert_eq!(
            CompareOp::Lt.compare(&Value::Boolean(true), &Value::Integer(2)),
            None
        );
    }

    // =========================================================================
    // Op Debug format tests
    // =========================================================================

    #[test]
    fn test_op_debug_format() {
        // Test that Debug formatting works for various Op variants
        assert_eq!(format!("{:?}", Op::LoadColumn(5)), "LoadColumn(5)");
        assert_eq!(format!("{:?}", Op::LoadColumn2(3)), "LoadColumn2(3)");
        assert_eq!(format!("{:?}", Op::Eq), "Eq");
        assert_eq!(format!("{:?}", Op::Ne), "Ne");
        assert_eq!(format!("{:?}", Op::Lt), "Lt");
        assert_eq!(format!("{:?}", Op::Le), "Le");
        assert_eq!(format!("{:?}", Op::Gt), "Gt");
        assert_eq!(format!("{:?}", Op::Ge), "Ge");
        assert_eq!(format!("{:?}", Op::Add), "Add");
        assert_eq!(format!("{:?}", Op::Sub), "Sub");
        assert_eq!(format!("{:?}", Op::Mul), "Mul");
        assert_eq!(format!("{:?}", Op::Div), "Div");
        assert_eq!(format!("{:?}", Op::Mod), "Mod");
        assert_eq!(format!("{:?}", Op::Not), "Not");
        assert_eq!(format!("{:?}", Op::Neg), "Neg");
        assert_eq!(format!("{:?}", Op::IsNull), "IsNull");
        assert_eq!(format!("{:?}", Op::IsNotNull), "IsNotNull");
        assert_eq!(format!("{:?}", Op::Return), "Return");
        assert_eq!(format!("{:?}", Op::ReturnTrue), "ReturnTrue");
        assert_eq!(format!("{:?}", Op::ReturnFalse), "ReturnFalse");
        assert_eq!(format!("{:?}", Op::Nop), "Nop");
        assert_eq!(format!("{:?}", Op::Dup), "Dup");
        assert_eq!(format!("{:?}", Op::Pop), "Pop");
        assert_eq!(format!("{:?}", Op::Swap), "Swap");
    }

    #[test]
    fn test_op_debug_format_with_values() {
        assert_eq!(
            format!("{:?}", Op::LoadConst(Value::Integer(42))),
            "LoadConst(Integer(42))"
        );
        assert_eq!(
            format!("{:?}", Op::EqColumnConst(0, Value::Integer(10))),
            "EqColumnConst(0, Integer(10))"
        );
        assert_eq!(format!("{:?}", Op::And(5)), "And(jump=5)");
        assert_eq!(format!("{:?}", Op::Or(10)), "Or(jump=10)");
        assert_eq!(format!("{:?}", Op::Jump(15)), "Jump(15)");
        assert_eq!(format!("{:?}", Op::JumpIfTrue(20)), "JumpIfTrue(20)");
        assert_eq!(format!("{:?}", Op::JumpIfFalse(25)), "JumpIfFalse(25)");
    }

    #[test]
    fn test_op_debug_format_special() {
        assert_eq!(format!("{:?}", Op::Coalesce(3)), "Coalesce(3)");
        assert_eq!(format!("{:?}", Op::NullIf), "NullIf");
        assert_eq!(format!("{:?}", Op::Greatest(2)), "Greatest(2)");
        assert_eq!(format!("{:?}", Op::Least(4)), "Least(4)");
        assert_eq!(format!("{:?}", Op::Between), "Between");
        assert_eq!(format!("{:?}", Op::NotBetween), "NotBetween");
        assert_eq!(format!("{:?}", Op::Concat), "Concat");
        assert_eq!(format!("{:?}", Op::JsonAccess), "JsonAccess");
        assert_eq!(format!("{:?}", Op::JsonAccessText), "JsonAccessText");
    }

    #[test]
    fn test_op_debug_bitwise() {
        assert_eq!(format!("{:?}", Op::BitAnd), "BitAnd");
        assert_eq!(format!("{:?}", Op::BitOr), "BitOr");
        assert_eq!(format!("{:?}", Op::BitXor), "BitXor");
        assert_eq!(format!("{:?}", Op::BitNot), "BitNot");
        assert_eq!(format!("{:?}", Op::Shl), "Shl");
        assert_eq!(format!("{:?}", Op::Shr), "Shr");
    }

    #[test]
    fn test_op_debug_case() {
        assert_eq!(format!("{:?}", Op::CaseStart), "CaseStart");
        assert_eq!(format!("{:?}", Op::CaseWhen(5)), "CaseWhen(jump=5)");
        assert_eq!(format!("{:?}", Op::CaseThen(10)), "CaseThen(jump=10)");
        assert_eq!(format!("{:?}", Op::CaseElse), "CaseElse");
        assert_eq!(format!("{:?}", Op::CaseEnd), "CaseEnd");
        assert_eq!(format!("{:?}", Op::CaseCompare), "CaseCompare");
    }

    #[test]
    fn test_op_debug_timestamp() {
        assert_eq!(
            format!("{:?}", Op::TimestampAddInterval),
            "TimestampAddInterval"
        );
        assert_eq!(
            format!("{:?}", Op::TimestampSubInterval),
            "TimestampSubInterval"
        );
        assert_eq!(format!("{:?}", Op::TimestampDiff), "TimestampDiff");
        assert_eq!(format!("{:?}", Op::TimestampAddDays), "TimestampAddDays");
        assert_eq!(format!("{:?}", Op::TimestampSubDays), "TimestampSubDays");
    }

    #[test]
    fn test_op_debug_boolean_checks() {
        assert_eq!(format!("{:?}", Op::IsTrue), "IsTrue");
        assert_eq!(format!("{:?}", Op::IsNotTrue), "IsNotTrue");
        assert_eq!(format!("{:?}", Op::IsFalse), "IsFalse");
        assert_eq!(format!("{:?}", Op::IsNotFalse), "IsNotFalse");
        assert_eq!(format!("{:?}", Op::IsDistinctFrom), "IsDistinctFrom");
        assert_eq!(format!("{:?}", Op::IsNotDistinctFrom), "IsNotDistinctFrom");
    }

    #[test]
    fn test_quantified_members_over_no_rows() {
        let none = CompactArc::new(Vec::new());
        for value in [Value::Integer(1), Value::Null(DataType::Integer)] {
            for op in [CompareOp::Eq, CompareOp::Ne, CompareOp::Lt, CompareOp::Ge] {
                let any = QuantifiedMembers::new(op, false, CompactArc::clone(&none), false);
                let all = QuantifiedMembers::new(op, true, CompactArc::clone(&none), false);
                assert_eq!(
                    any.answer(&value),
                    Some(false),
                    "{value:?} {op:?} ANY of no rows"
                );
                assert_eq!(
                    all.answer(&value),
                    Some(true),
                    "{value:?} {op:?} ALL of no rows"
                );
                let any_null = QuantifiedMembers::new(op, false, CompactArc::clone(&none), true);
                let all_null = QuantifiedMembers::new(op, true, CompactArc::clone(&none), true);
                assert_eq!(
                    any_null.answer(&value),
                    None,
                    "{value:?} {op:?} ANY of one NULL"
                );
                assert_eq!(
                    all_null.answer(&value),
                    None,
                    "{value:?} {op:?} ALL of one NULL"
                );
            }
        }
    }

    #[test]
    fn test_op_debug_subquery() {
        assert_eq!(
            format!("{:?}", Op::ExecScalarSubquery(0)),
            "ExecScalarSubquery(0)"
        );
        assert_eq!(format!("{:?}", Op::ExecExists(1)), "ExecExists(1)");
        assert_eq!(format!("{:?}", Op::ExecInSubquery(2)), "ExecInSubquery(2)");
        assert_eq!(
            format!("{:?}", Op::ExecQuantified(3, CompareOp::Gt, true)),
            "ExecQuantified(3, Gt, true)"
        );
        assert_eq!(
            format!("{:?}", Op::ExecQuantified(4, CompareOp::Lt, false)),
            "ExecQuantified(4, Lt, false)"
        );
    }

    #[test]
    fn test_op_debug_sets() {
        use crate::core::predicate::PredicateSet;
        use crate::core::ValueSet;
        let set = CompactArc::new(PredicateSet::new(
            CompactArc::new(ValueSet::default()),
            false,
        ));
        assert!(format!("{:?}", Op::InSet(set.clone())).contains("InSet"));
        assert!(format!("{:?}", Op::NotInSet(set.clone())).contains("NotInSet"));
        assert!(format!("{:?}", Op::InSetColumn(0, set)).contains("InSetColumn"));
    }

    #[test]
    fn test_op_debug_like_glob() {
        let pattern = Arc::new(Pattern::like("test%", false));
        assert!(format!("{:?}", Op::Like(pattern.clone())).contains("Like"));
        assert!(format!("{:?}", Op::Glob(pattern.clone())).contains("Glob"));
        assert!(format!("{:?}", Op::LikeColumn(0, pattern)).contains("LikeColumn"));
    }
}
