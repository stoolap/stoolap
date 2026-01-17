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
use ahash::AHashSet;
use memchr::memmem;

use crate::core::{DataType, Value};
use crate::functions::{NativeFn1, ScalarFunction};

/// Compiled LIKE/ILIKE pattern for fast matching
#[derive(Debug, Clone)]
pub enum CompiledPattern {
    /// Exact match (no wildcards)
    Exact(String),
    /// Prefix match: "abc%"
    Prefix(String),
    /// Suffix match: "%abc"
    Suffix(String),
    /// Contains match: "%abc%"
    Contains(String),
    /// Prefix + Suffix: "abc%xyz"
    PrefixSuffix(String, String),
    /// Complex pattern requiring regex
    Regex(regex::Regex),
    /// Match all: "%"
    MatchAll,
    /// Single char: "_"
    SingleChar,
}

impl CompiledPattern {
    /// Compile a LIKE pattern into optimized form
    pub fn compile(pattern: &str, case_insensitive: bool) -> Self {
        let pat = if case_insensitive {
            pattern.to_lowercase()
        } else {
            pattern.to_string()
        };

        // Check for simple patterns that don't need regex
        let has_percent = pat.contains('%');
        let has_underscore = pat.contains('_');

        if !has_percent && !has_underscore {
            return CompiledPattern::Exact(pat);
        }

        if pat == "%" {
            return CompiledPattern::MatchAll;
        }

        if pat == "_" {
            return CompiledPattern::SingleChar;
        }

        // Check for prefix pattern: "abc%"
        if pat.ends_with('%') && !pat[..pat.len() - 1].contains('%') && !has_underscore {
            return CompiledPattern::Prefix(pat[..pat.len() - 1].to_string());
        }

        // Check for suffix pattern: "%abc"
        if pat.starts_with('%') && !pat[1..].contains('%') && !has_underscore {
            return CompiledPattern::Suffix(pat[1..].to_string());
        }

        // Check for contains pattern: "%abc%"
        if pat.starts_with('%') && pat.ends_with('%') && pat.len() > 2 {
            let middle = &pat[1..pat.len() - 1];
            if !middle.contains('%') && !middle.contains('_') {
                return CompiledPattern::Contains(middle.to_string());
            }
        }

        // Check for prefix+suffix: "abc%xyz"
        if has_percent && !has_underscore {
            let parts: Vec<&str> = pat.split('%').collect();
            if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                return CompiledPattern::PrefixSuffix(parts[0].to_string(), parts[1].to_string());
            }
        }

        // Fall back to regex for complex patterns
        let regex_pattern = Self::like_to_regex(&pat);
        let regex = if case_insensitive {
            regex::Regex::new(&format!("(?i)^{}$", regex_pattern))
        } else {
            regex::Regex::new(&format!("^{}$", regex_pattern))
        };

        match regex {
            Ok(re) => CompiledPattern::Regex(re),
            Err(_) => CompiledPattern::Exact(pat), // Fallback
        }
    }

    /// Compile a GLOB pattern into optimized form
    /// GLOB uses * for any sequence and ? for single character (case-sensitive)
    pub fn compile_glob(pattern: &str) -> Self {
        let pat = pattern.to_string();

        // Check for simple patterns that don't need regex
        let has_star = pat.contains('*');
        let has_question = pat.contains('?');

        if !has_star && !has_question {
            return CompiledPattern::Exact(pat);
        }

        if pat == "*" {
            return CompiledPattern::MatchAll;
        }

        if pat == "?" {
            return CompiledPattern::SingleChar;
        }

        // Check for prefix pattern: "abc*"
        if pat.ends_with('*') && !pat[..pat.len() - 1].contains('*') && !has_question {
            return CompiledPattern::Prefix(pat[..pat.len() - 1].to_string());
        }

        // Check for suffix pattern: "*abc"
        if pat.starts_with('*') && !pat[1..].contains('*') && !has_question {
            return CompiledPattern::Suffix(pat[1..].to_string());
        }

        // Check for contains pattern: "*abc*"
        if pat.starts_with('*') && pat.ends_with('*') && pat.len() > 2 {
            let middle = &pat[1..pat.len() - 1];
            if !middle.contains('*') && !middle.contains('?') {
                return CompiledPattern::Contains(middle.to_string());
            }
        }

        // Check for prefix+suffix: "abc*xyz"
        if has_star && !has_question {
            let parts: Vec<&str> = pat.split('*').collect();
            if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                return CompiledPattern::PrefixSuffix(parts[0].to_string(), parts[1].to_string());
            }
        }

        // Fall back to regex for complex patterns
        let regex_pattern = Self::glob_to_regex(&pat);
        let regex = regex::Regex::new(&format!("^{}$", regex_pattern));

        match regex {
            Ok(re) => CompiledPattern::Regex(re),
            Err(_) => CompiledPattern::Exact(pat), // Fallback
        }
    }

    /// Convert GLOB pattern to regex (* -> .*, ? -> .)
    fn glob_to_regex(pattern: &str) -> String {
        let mut result = String::with_capacity(pattern.len() * 2);
        let mut chars = pattern.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '*' => result.push_str(".*"),
                '?' => result.push('.'),
                '\\' => {
                    // Escape sequence
                    if let Some(&next) = chars.peek() {
                        if next == '*' || next == '?' || next == '\\' {
                            result.push_str(&regex::escape(&next.to_string()));
                            chars.next();
                        } else {
                            result.push_str(&regex::escape("\\"));
                        }
                    }
                }
                '[' => {
                    // Character class in GLOB - pass through to regex
                    result.push('[');
                    for c in chars.by_ref() {
                        if c == ']' {
                            result.push(']');
                            break;
                        }
                        result.push(c);
                    }
                }
                _ => result.push_str(&regex::escape(&c.to_string())),
            }
        }

        result
    }

    /// Convert LIKE pattern to regex
    fn like_to_regex(pattern: &str) -> String {
        let mut result = String::with_capacity(pattern.len() * 2);
        let mut chars = pattern.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '%' => result.push_str(".*"),
                '_' => result.push('.'),
                '\\' => {
                    // Escape sequence
                    if let Some(&next) = chars.peek() {
                        if next == '%' || next == '_' || next == '\\' {
                            result.push(
                                regex::escape(&next.to_string())
                                    .chars()
                                    .next()
                                    .unwrap_or(next),
                            );
                            chars.next();
                        } else {
                            result.push_str(&regex::escape("\\"));
                        }
                    }
                }
                _ => result.push_str(&regex::escape(&c.to_string())),
            }
        }

        result
    }

    /// Match a string against this pattern
    #[inline]
    pub fn matches(&self, text: &str, case_insensitive: bool) -> bool {
        // Fast path: case-sensitive matching (no allocation)
        if !case_insensitive {
            return match self {
                CompiledPattern::Exact(p) => text == p,
                CompiledPattern::Prefix(p) => text.starts_with(p),
                CompiledPattern::Suffix(p) => text.ends_with(p),
                // Use memmem::find for SIMD-accelerated substring search
                CompiledPattern::Contains(p) => {
                    memmem::find(text.as_bytes(), p.as_bytes()).is_some()
                }
                CompiledPattern::PrefixSuffix(prefix, suffix) => {
                    text.starts_with(prefix)
                        && text.ends_with(suffix)
                        && text.len() >= prefix.len() + suffix.len()
                }
                CompiledPattern::Regex(re) => re.is_match(text),
                CompiledPattern::MatchAll => true,
                CompiledPattern::SingleChar => text.chars().count() == 1,
            };
        }

        // Case-insensitive: use ASCII fast path when possible, fall back to Unicode
        // NOTE: Pattern is already lowercased at compile time (see compile() method),
        // so we only need to lowercase the input text, not the pattern again.
        match self {
            CompiledPattern::Exact(p) => {
                if text.is_ascii() && p.is_ascii() {
                    text.eq_ignore_ascii_case(p)
                } else {
                    // Pattern already lowercased at compile time
                    text.to_lowercase() == *p
                }
            }
            CompiledPattern::Prefix(p) => {
                if text.len() < p.len() {
                    return false;
                }
                if text.is_ascii() && p.is_ascii() {
                    text[..p.len()].eq_ignore_ascii_case(p)
                } else {
                    // Pattern already lowercased at compile time
                    text.to_lowercase().starts_with(p)
                }
            }
            CompiledPattern::Suffix(p) => {
                if text.len() < p.len() {
                    return false;
                }
                if text.is_ascii() && p.is_ascii() {
                    text[text.len() - p.len()..].eq_ignore_ascii_case(p)
                } else {
                    // Pattern already lowercased at compile time
                    text.to_lowercase().ends_with(p)
                }
            }
            CompiledPattern::Contains(p) => {
                if p.len() > text.len() {
                    return false;
                }
                if text.is_ascii() && p.is_ascii() {
                    // Fast path: ASCII-only, use byte-level comparison without allocation
                    text.as_bytes()
                        .windows(p.len())
                        .any(|window| window.eq_ignore_ascii_case(p.as_bytes()))
                } else {
                    // Unicode path: lowercase text only (pattern already lowercased)
                    text.to_lowercase().contains(p)
                }
            }
            CompiledPattern::PrefixSuffix(prefix, suffix) => {
                if text.len() < prefix.len() + suffix.len() {
                    return false;
                }
                if text.is_ascii() && prefix.is_ascii() && suffix.is_ascii() {
                    text[..prefix.len()].eq_ignore_ascii_case(prefix)
                        && text[text.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
                } else {
                    // Patterns already lowercased at compile time
                    let text_lower = text.to_lowercase();
                    text_lower.starts_with(prefix) && text_lower.ends_with(suffix)
                }
            }
            // Regex already has case-insensitivity compiled in
            CompiledPattern::Regex(re) => re.is_match(text),
            CompiledPattern::MatchAll => true,
            CompiledPattern::SingleChar => text.chars().count() == 1,
        }
    }
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
    LoadOuterColumn(Arc<str>),

    /// Load constant value (pre-cloned at compile time)
    /// Stack: [] -> [value]
    LoadConst(Value),

    /// Load query parameter by index
    /// Stack: [] -> [value]
    LoadParam(u16),

    /// Load named parameter
    /// Stack: [] -> [value]
    LoadNamedParam(Arc<str>),

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
    LikeColumn(u16, Arc<CompiledPattern>, bool), // col_idx, pattern, case_insensitive

    /// Fused: column IN (constant set with AHash)
    /// Stack: [] -> [bool]
    InSetColumn(u16, CompactArc<AHashSet<Value>>, bool), // col_idx, set, has_null

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
    Like(Arc<CompiledPattern>, bool), // pattern, case_insensitive

    /// GLOB pattern match
    /// Stack: [text] -> [bool]
    Glob(Arc<CompiledPattern>),

    /// REGEXP match (pre-compiled regex)
    /// Stack: [text] -> [bool]
    Regexp(Arc<regex::Regex>),

    /// LIKE with ESCAPE character
    /// Stack: [text] -> [bool]
    LikeEscape(Arc<CompiledPattern>, bool, char), // pattern, case_insensitive, escape_char

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

    // =========================================================================
    // SET OPERATIONS
    // =========================================================================
    /// IN set membership (pre-built AHashSet for fast lookups)
    /// Stack: [value] -> [bool]
    InSet(CompactArc<AHashSet<Value>>, bool), // set, has_null

    /// NOT IN set membership
    /// Stack: [value] -> [bool]
    NotInSet(CompactArc<AHashSet<Value>>, bool), // set, has_null

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

    /// Execute IN subquery
    /// Stack: [value] -> [bool]
    ExecInSubquery(u16), // Subquery plan index

    /// Execute ALL comparison subquery
    /// Stack: [value] -> [bool]
    ExecAll(u16, CompareOp), // Subquery index, comparison operator

    /// Execute ANY comparison subquery
    /// Stack: [value] -> [bool]
    ExecAny(u16, CompareOp), // Subquery index, comparison operator

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
        match (a.partial_cmp(b), self) {
            (Some(std::cmp::Ordering::Equal), CompareOp::Eq) => Some(true),
            (Some(std::cmp::Ordering::Equal), CompareOp::Ne) => Some(false),
            (Some(std::cmp::Ordering::Equal), CompareOp::Le) => Some(true),
            (Some(std::cmp::Ordering::Equal), CompareOp::Ge) => Some(true),
            (Some(std::cmp::Ordering::Equal), CompareOp::Lt) => Some(false),
            (Some(std::cmp::Ordering::Equal), CompareOp::Gt) => Some(false),

            (Some(std::cmp::Ordering::Less), CompareOp::Lt) => Some(true),
            (Some(std::cmp::Ordering::Less), CompareOp::Le) => Some(true),
            (Some(std::cmp::Ordering::Less), CompareOp::Ne) => Some(true),
            (Some(std::cmp::Ordering::Less), CompareOp::Eq) => Some(false),
            (Some(std::cmp::Ordering::Less), CompareOp::Gt) => Some(false),
            (Some(std::cmp::Ordering::Less), CompareOp::Ge) => Some(false),

            (Some(std::cmp::Ordering::Greater), CompareOp::Gt) => Some(true),
            (Some(std::cmp::Ordering::Greater), CompareOp::Ge) => Some(true),
            (Some(std::cmp::Ordering::Greater), CompareOp::Ne) => Some(true),
            (Some(std::cmp::Ordering::Greater), CompareOp::Eq) => Some(false),
            (Some(std::cmp::Ordering::Greater), CompareOp::Lt) => Some(false),
            (Some(std::cmp::Ordering::Greater), CompareOp::Le) => Some(false),

            (None, _) => None, // NULL comparison
        }
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
            Op::LoadNamedParam(name) => write!(f, "LoadNamedParam({})", name),
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
            Op::LikeColumn(col, _, ci) => write!(f, "LikeColumn({}, case_insensitive={})", col, ci),
            Op::InSetColumn(col, set, has_null) => {
                write!(
                    f,
                    "InSetColumn({}, len={}, has_null={})",
                    col,
                    set.len(),
                    has_null
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
            Op::Like(_, ci) => write!(f, "Like(case_insensitive={})", ci),
            Op::Glob(_) => write!(f, "Glob"),
            Op::Regexp(_) => write!(f, "Regexp"),
            Op::LikeEscape(_, ci, esc) => {
                write!(f, "LikeEscape(case_insensitive={}, escape='{}')", ci, esc)
            }
            Op::JsonAccess => write!(f, "JsonAccess"),
            Op::JsonAccessText => write!(f, "JsonAccessText"),
            Op::TimestampAddInterval => write!(f, "TimestampAddInterval"),
            Op::TimestampSubInterval => write!(f, "TimestampSubInterval"),
            Op::TimestampDiff => write!(f, "TimestampDiff"),
            Op::TimestampAddDays => write!(f, "TimestampAddDays"),
            Op::TimestampSubDays => write!(f, "TimestampSubDays"),
            Op::InSet(set, has_null) => {
                write!(f, "InSet(len={}, has_null={})", set.len(), has_null)
            }
            Op::NotInSet(set, has_null) => {
                write!(f, "NotInSet(len={}, has_null={})", set.len(), has_null)
            }
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
            Op::ExecAll(idx, op) => write!(f, "ExecAll({}, {:?})", idx, op),
            Op::ExecAny(idx, op) => write!(f, "ExecAny({}, {:?})", idx, op),
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
    // CompiledPattern::compile() tests - LIKE patterns
    // =========================================================================

    #[test]
    fn test_pattern_exact() {
        let pattern = CompiledPattern::compile("hello", false);
        assert!(matches!(pattern, CompiledPattern::Exact(ref s) if s == "hello"));
        assert!(pattern.matches("hello", false));
        assert!(!pattern.matches("Hello", false));
        assert!(!pattern.matches("hello world", false));
    }

    #[test]
    fn test_pattern_exact_case_insensitive() {
        let pattern = CompiledPattern::compile("Hello", true);
        assert!(matches!(pattern, CompiledPattern::Exact(ref s) if s == "hello"));
        assert!(pattern.matches("hello", true));
        assert!(pattern.matches("HELLO", true));
        assert!(pattern.matches("HeLLo", true));
    }

    #[test]
    fn test_pattern_match_all() {
        let pattern = CompiledPattern::compile("%", false);
        assert!(matches!(pattern, CompiledPattern::MatchAll));
        assert!(pattern.matches("", false));
        assert!(pattern.matches("anything", false));
        assert!(pattern.matches("with spaces and 123", false));
    }

    #[test]
    fn test_pattern_single_char() {
        let pattern = CompiledPattern::compile("_", false);
        assert!(matches!(pattern, CompiledPattern::SingleChar));
        assert!(pattern.matches("a", false));
        assert!(pattern.matches("Z", false));
        assert!(!pattern.matches("", false));
        assert!(!pattern.matches("ab", false));
    }

    #[test]
    fn test_pattern_prefix() {
        let pattern = CompiledPattern::compile("hello%", false);
        assert!(matches!(pattern, CompiledPattern::Prefix(ref s) if s == "hello"));
        assert!(pattern.matches("hello", false));
        assert!(pattern.matches("hello world", false));
        assert!(pattern.matches("hellooooo", false));
        assert!(!pattern.matches("Hello", false));
        assert!(!pattern.matches("say hello", false));
    }

    #[test]
    fn test_pattern_prefix_case_insensitive() {
        let pattern = CompiledPattern::compile("Hello%", true);
        assert!(pattern.matches("hello world", true));
        assert!(pattern.matches("HELLO WORLD", true));
        assert!(!pattern.matches("say hello", true));
    }

    #[test]
    fn test_pattern_suffix() {
        let pattern = CompiledPattern::compile("%world", false);
        assert!(matches!(pattern, CompiledPattern::Suffix(ref s) if s == "world"));
        assert!(pattern.matches("world", false));
        assert!(pattern.matches("hello world", false));
        assert!(!pattern.matches("World", false));
        assert!(!pattern.matches("world!", false));
    }

    #[test]
    fn test_pattern_suffix_case_insensitive() {
        let pattern = CompiledPattern::compile("%World", true);
        assert!(pattern.matches("hello world", true));
        assert!(pattern.matches("HELLO WORLD", true));
        assert!(!pattern.matches("world!", true));
    }

    #[test]
    fn test_pattern_contains() {
        let pattern = CompiledPattern::compile("%ello%", false);
        assert!(matches!(pattern, CompiledPattern::Contains(ref s) if s == "ello"));
        assert!(pattern.matches("hello", false));
        assert!(pattern.matches("yellow", false));
        assert!(pattern.matches("hello world", false));
        assert!(!pattern.matches("HELLO", false));
    }

    #[test]
    fn test_pattern_contains_case_insensitive() {
        let pattern = CompiledPattern::compile("%ELLO%", true);
        assert!(pattern.matches("hello", true));
        assert!(pattern.matches("YELLOW", true));
        assert!(!pattern.matches("hi", true));
    }

    #[test]
    fn test_pattern_prefix_suffix() {
        let pattern = CompiledPattern::compile("hello%world", false);
        assert!(
            matches!(pattern, CompiledPattern::PrefixSuffix(ref p, ref s) if p == "hello" && s == "world")
        );
        assert!(pattern.matches("helloworld", false));
        assert!(pattern.matches("hello world", false));
        assert!(pattern.matches("hello beautiful world", false));
        assert!(!pattern.matches("hello", false));
        assert!(!pattern.matches("world", false));
    }

    #[test]
    fn test_pattern_prefix_suffix_case_insensitive() {
        let pattern = CompiledPattern::compile("Hello%World", true);
        assert!(pattern.matches("helloworld", true));
        assert!(pattern.matches("HELLO WORLD", true));
        assert!(!pattern.matches("hello", true));
    }

    #[test]
    fn test_pattern_prefix_suffix_too_short() {
        let pattern = CompiledPattern::compile("abc%xyz", false);
        // Text must be at least prefix.len() + suffix.len()
        assert!(!pattern.matches("abcxy", false)); // too short
        assert!(pattern.matches("abcxyz", false)); // exact length
        assert!(pattern.matches("abc123xyz", false));
    }

    #[test]
    fn test_pattern_complex_regex() {
        // Pattern with multiple % or _ that requires regex
        let pattern = CompiledPattern::compile("a%b%c", false);
        assert!(matches!(pattern, CompiledPattern::Regex(_)));
        assert!(pattern.matches("abc", false));
        assert!(pattern.matches("aXbYc", false));
        assert!(pattern.matches("aXXXbYYYc", false));
        assert!(!pattern.matches("ac", false));
    }

    #[test]
    fn test_pattern_underscore_regex() {
        let pattern = CompiledPattern::compile("a_c", false);
        assert!(matches!(pattern, CompiledPattern::Regex(_)));
        assert!(pattern.matches("abc", false));
        assert!(pattern.matches("aXc", false));
        assert!(!pattern.matches("ac", false));
        assert!(!pattern.matches("abbc", false));
    }

    #[test]
    fn test_pattern_mixed_wildcards() {
        let pattern = CompiledPattern::compile("a_%b", false);
        assert!(matches!(pattern, CompiledPattern::Regex(_)));
        assert!(pattern.matches("aXb", false));
        assert!(pattern.matches("aXYZb", false));
        assert!(!pattern.matches("ab", false));
    }

    // =========================================================================
    // CompiledPattern::compile_glob() tests - GLOB patterns
    // =========================================================================

    #[test]
    fn test_glob_exact() {
        let pattern = CompiledPattern::compile_glob("hello");
        assert!(matches!(pattern, CompiledPattern::Exact(ref s) if s == "hello"));
        assert!(pattern.matches("hello", false));
        assert!(!pattern.matches("Hello", false));
    }

    #[test]
    fn test_glob_match_all() {
        let pattern = CompiledPattern::compile_glob("*");
        assert!(matches!(pattern, CompiledPattern::MatchAll));
        assert!(pattern.matches("anything", false));
    }

    #[test]
    fn test_glob_single_char() {
        let pattern = CompiledPattern::compile_glob("?");
        assert!(matches!(pattern, CompiledPattern::SingleChar));
        assert!(pattern.matches("a", false));
        assert!(!pattern.matches("ab", false));
    }

    #[test]
    fn test_glob_prefix() {
        let pattern = CompiledPattern::compile_glob("hello*");
        assert!(matches!(pattern, CompiledPattern::Prefix(ref s) if s == "hello"));
        assert!(pattern.matches("hello", false));
        assert!(pattern.matches("hello world", false));
    }

    #[test]
    fn test_glob_suffix() {
        let pattern = CompiledPattern::compile_glob("*world");
        assert!(matches!(pattern, CompiledPattern::Suffix(ref s) if s == "world"));
        assert!(pattern.matches("world", false));
        assert!(pattern.matches("hello world", false));
    }

    #[test]
    fn test_glob_contains() {
        let pattern = CompiledPattern::compile_glob("*ello*");
        assert!(matches!(pattern, CompiledPattern::Contains(ref s) if s == "ello"));
        assert!(pattern.matches("hello", false));
        assert!(pattern.matches("yellow", false));
    }

    #[test]
    fn test_glob_prefix_suffix() {
        let pattern = CompiledPattern::compile_glob("hello*world");
        assert!(
            matches!(pattern, CompiledPattern::PrefixSuffix(ref p, ref s) if p == "hello" && s == "world")
        );
        assert!(pattern.matches("helloworld", false));
        assert!(pattern.matches("hello beautiful world", false));
    }

    #[test]
    fn test_glob_complex_regex() {
        let pattern = CompiledPattern::compile_glob("a*b*c");
        assert!(matches!(pattern, CompiledPattern::Regex(_)));
        assert!(pattern.matches("abc", false));
        assert!(pattern.matches("aXbYc", false));
    }

    #[test]
    fn test_glob_question_mark() {
        let pattern = CompiledPattern::compile_glob("a?c");
        assert!(matches!(pattern, CompiledPattern::Regex(_)));
        assert!(pattern.matches("abc", false));
        assert!(!pattern.matches("ac", false));
    }

    #[test]
    fn test_glob_character_class() {
        // Character class alone without * or ? is treated as exact match
        let pattern = CompiledPattern::compile_glob("a[bc]d");
        assert!(matches!(pattern, CompiledPattern::Exact(_)));

        // Pattern with * at end is treated as Prefix
        let pattern2 = CompiledPattern::compile_glob("a[bc]*");
        assert!(matches!(pattern2, CompiledPattern::Prefix(ref s) if s == "a[bc]"));

        // *[bc]* is Contains since middle has no wildcards
        let pattern3 = CompiledPattern::compile_glob("*[bc]*");
        assert!(matches!(pattern3, CompiledPattern::Contains(ref s) if s == "[bc]"));
    }

    #[test]
    fn test_glob_escape_edge_cases() {
        // `a\\*b` splits on * giving ["a\\", "b"] - treated as PrefixSuffix
        let pattern1 = CompiledPattern::compile_glob("a\\*b");
        assert!(matches!(pattern1, CompiledPattern::PrefixSuffix(_, _)));

        // Complex escapes with multiple wildcards go to regex
        let pattern2 = CompiledPattern::compile_glob("*a*b*");
        assert!(matches!(pattern2, CompiledPattern::Regex(_)));
        assert!(pattern2.matches("XaYbZ", false));
        assert!(pattern2.matches("ab", false));
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
    fn test_op_debug_subquery() {
        assert_eq!(
            format!("{:?}", Op::ExecScalarSubquery(0)),
            "ExecScalarSubquery(0)"
        );
        assert_eq!(format!("{:?}", Op::ExecExists(1)), "ExecExists(1)");
        assert_eq!(format!("{:?}", Op::ExecInSubquery(2)), "ExecInSubquery(2)");
        assert_eq!(
            format!("{:?}", Op::ExecAll(3, CompareOp::Gt)),
            "ExecAll(3, Gt)"
        );
        assert_eq!(
            format!("{:?}", Op::ExecAny(4, CompareOp::Lt)),
            "ExecAny(4, Lt)"
        );
    }

    #[test]
    fn test_op_debug_sets() {
        let set = CompactArc::new(AHashSet::new());
        assert!(format!("{:?}", Op::InSet(set.clone(), false)).contains("InSet"));
        assert!(format!("{:?}", Op::NotInSet(set.clone(), true)).contains("NotInSet"));
        assert!(format!("{:?}", Op::InSetColumn(0, set, false)).contains("InSetColumn"));
    }

    #[test]
    fn test_op_debug_like_glob() {
        let pattern = Arc::new(CompiledPattern::compile("test%", false));
        assert!(format!("{:?}", Op::Like(pattern.clone(), false)).contains("Like"));
        assert!(format!("{:?}", Op::Glob(pattern.clone())).contains("Glob"));
        assert!(format!("{:?}", Op::LikeColumn(0, pattern, false)).contains("LikeColumn"));
    }
}
