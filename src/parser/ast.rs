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

//! Abstract Syntax Tree (AST) types for SQL parser
//!
//! This module defines the AST node types that represent parsed SQL statements
//! and expressions.

use super::token::{Position, Token};
use ahash::AHashSet;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::core::Value;

// ============================================================================
// Core Traits
// ============================================================================

/// Node trait - base for all AST nodes
pub trait Node: fmt::Display + fmt::Debug {
    /// Returns the literal string of the first token
    fn token_literal(&self) -> &str;
    /// Returns the position of the node in source code
    fn position(&self) -> Position;
}

// ============================================================================
// Expressions
// ============================================================================

/// Expression enum representing all expression types
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Identifier (column name, table name)
    Identifier(Identifier),
    /// Qualified identifier (table.column)
    QualifiedIdentifier(QualifiedIdentifier),
    /// Integer literal
    IntegerLiteral(IntegerLiteral),
    /// Float literal
    FloatLiteral(FloatLiteral),
    /// String literal
    StringLiteral(StringLiteral),
    /// Boolean literal (TRUE/FALSE)
    BooleanLiteral(BooleanLiteral),
    /// NULL literal
    NullLiteral(NullLiteral),
    /// INTERVAL literal
    IntervalLiteral(IntervalLiteral),
    /// Parameter ($1, ?)
    Parameter(Parameter),
    /// Prefix expression (-x, NOT x)
    Prefix(PrefixExpression),
    /// Infix expression (a + b, a = b)
    Infix(InfixExpression),
    /// List of expressions (for IN clause)
    List(ListExpression),
    /// DISTINCT expression
    Distinct(DistinctExpression),
    /// EXISTS subquery
    Exists(ExistsExpression),
    /// ALL/ANY/SOME subquery comparison (e.g., x > ALL (SELECT ...))
    AllAny(AllAnyExpression),
    /// IN expression
    In(InExpression),
    /// Pre-computed IN expression with HashSet (for semi-join optimization)
    /// Uses Arc for cheap cloning in parallel execution
    InHashSet(InHashSetExpression),
    /// BETWEEN expression
    Between(BetweenExpression),
    /// LIKE expression (with optional ESCAPE clause)
    Like(LikeExpression),
    /// Scalar subquery
    ScalarSubquery(ScalarSubquery),
    /// Expression list (for IN values)
    ExpressionList(ExpressionList),
    /// CASE expression
    Case(CaseExpression),
    /// CAST expression
    Cast(CastExpression),
    /// Function call
    FunctionCall(FunctionCall),
    /// Aliased expression (expr AS alias)
    Aliased(AliasedExpression),
    /// Window expression
    Window(WindowExpression),
    /// Simple table source
    TableSource(SimpleTableSource),
    /// Join table source
    JoinSource(Box<JoinTableSource>),
    /// Subquery table source
    SubquerySource(SubqueryTableSource),
    /// VALUES table source (e.g., VALUES (1, 'a'), (2, 'b'))
    ValuesSource(ValuesTableSource),
    /// CTE reference
    CteReference(CteReference),
    /// Star (*) for SELECT *
    Star(StarExpression),
    /// Qualified star (table.*) for SELECT table.*
    QualifiedStar(QualifiedStarExpression),
    /// DEFAULT keyword (for INSERT VALUES)
    Default(DefaultExpression),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Identifier(e) => write!(f, "{}", e),
            Expression::QualifiedIdentifier(e) => write!(f, "{}", e),
            Expression::IntegerLiteral(e) => write!(f, "{}", e),
            Expression::FloatLiteral(e) => write!(f, "{}", e),
            Expression::StringLiteral(e) => write!(f, "{}", e),
            Expression::BooleanLiteral(e) => write!(f, "{}", e),
            Expression::NullLiteral(e) => write!(f, "{}", e),
            Expression::IntervalLiteral(e) => write!(f, "{}", e),
            Expression::Parameter(e) => write!(f, "{}", e),
            Expression::Prefix(e) => write!(f, "{}", e),
            Expression::Infix(e) => write!(f, "{}", e),
            Expression::List(e) => write!(f, "{}", e),
            Expression::Distinct(e) => write!(f, "{}", e),
            Expression::Exists(e) => write!(f, "{}", e),
            Expression::AllAny(e) => write!(f, "{}", e),
            Expression::In(e) => write!(f, "{}", e),
            Expression::InHashSet(e) => write!(f, "{}", e),
            Expression::Between(e) => write!(f, "{}", e),
            Expression::Like(e) => write!(f, "{}", e),
            Expression::ScalarSubquery(e) => write!(f, "{}", e),
            Expression::ExpressionList(e) => write!(f, "{}", e),
            Expression::Case(e) => write!(f, "{}", e),
            Expression::Cast(e) => write!(f, "{}", e),
            Expression::FunctionCall(e) => write!(f, "{}", e),
            Expression::Aliased(e) => write!(f, "{}", e),
            Expression::Window(e) => write!(f, "{}", e),
            Expression::TableSource(e) => write!(f, "{}", e),
            Expression::JoinSource(e) => write!(f, "{}", e),
            Expression::SubquerySource(e) => write!(f, "{}", e),
            Expression::ValuesSource(e) => write!(f, "{}", e),
            Expression::CteReference(e) => write!(f, "{}", e),
            Expression::Star(e) => write!(f, "{}", e),
            Expression::QualifiedStar(e) => write!(f, "{}", e),
            Expression::Default(e) => write!(f, "{}", e),
        }
    }
}

impl Expression {
    /// Get the position of this expression
    pub fn position(&self) -> Position {
        match self {
            Expression::Identifier(e) => e.token.position,
            Expression::QualifiedIdentifier(e) => e.token.position,
            Expression::IntegerLiteral(e) => e.token.position,
            Expression::FloatLiteral(e) => e.token.position,
            Expression::StringLiteral(e) => e.token.position,
            Expression::BooleanLiteral(e) => e.token.position,
            Expression::NullLiteral(e) => e.token.position,
            Expression::IntervalLiteral(e) => e.token.position,
            Expression::Parameter(e) => e.token.position,
            Expression::Prefix(e) => e.token.position,
            Expression::Infix(e) => e.token.position,
            Expression::List(e) => e.token.position,
            Expression::Distinct(e) => e.token.position,
            Expression::Exists(e) => e.token.position,
            Expression::AllAny(e) => e.token.position,
            Expression::In(e) => e.token.position,
            Expression::InHashSet(e) => e.token.position,
            Expression::Between(e) => e.token.position,
            Expression::Like(e) => e.token.position,
            Expression::ScalarSubquery(e) => e.token.position,
            Expression::ExpressionList(e) => e.token.position,
            Expression::Case(e) => e.token.position,
            Expression::Cast(e) => e.token.position,
            Expression::FunctionCall(e) => e.token.position,
            Expression::Aliased(e) => e.token.position,
            Expression::Window(e) => e.token.position,
            Expression::TableSource(e) => e.token.position,
            Expression::JoinSource(e) => e.token.position,
            Expression::SubquerySource(e) => e.token.position,
            Expression::ValuesSource(e) => e.token.position,
            Expression::CteReference(e) => e.token.position,
            Expression::Star(e) => e.token.position,
            Expression::QualifiedStar(e) => e.token.position,
            Expression::Default(e) => e.token.position,
        }
    }
}

// ============================================================================
// Expression Types
// ============================================================================

/// Identifier (column name, table name, etc.)
#[derive(Debug, Clone, PartialEq)]
pub struct Identifier {
    pub token: Token,
    pub value: String,
    /// Pre-computed lowercase value for fast case-insensitive lookups
    pub value_lower: String,
}

impl Identifier {
    /// Create a new identifier with pre-computed lowercase value
    pub fn new(token: Token, value: String) -> Self {
        let value_lower = value.to_lowercase();
        Self {
            token,
            value,
            value_lower,
        }
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// Qualified identifier (table.column)
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedIdentifier {
    pub token: Token,
    pub qualifier: Box<Identifier>,
    pub name: Box<Identifier>,
}

impl fmt::Display for QualifiedIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.qualifier, self.name)
    }
}

/// Integer literal
#[derive(Debug, Clone, PartialEq)]
pub struct IntegerLiteral {
    pub token: Token,
    pub value: i64,
}

impl fmt::Display for IntegerLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// Float literal
#[derive(Debug, Clone, PartialEq)]
pub struct FloatLiteral {
    pub token: Token,
    pub value: f64,
}

impl fmt::Display for FloatLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.token.literal)
    }
}

/// String literal
#[derive(Debug, Clone, PartialEq)]
pub struct StringLiteral {
    pub token: Token,
    pub value: String,
    /// Optional type hint (DATE, TIME, JSON, etc.)
    pub type_hint: Option<String>,
}

impl fmt::Display for StringLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}'", self.value)
    }
}

/// Boolean literal
#[derive(Debug, Clone, PartialEq)]
pub struct BooleanLiteral {
    pub token: Token,
    pub value: bool,
}

impl fmt::Display for BooleanLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", if self.value { "TRUE" } else { "FALSE" })
    }
}

/// NULL literal
#[derive(Debug, Clone, PartialEq)]
pub struct NullLiteral {
    pub token: Token,
}

impl fmt::Display for NullLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NULL")
    }
}

/// INTERVAL literal
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalLiteral {
    pub token: Token,
    pub value: String,
    pub quantity: i64,
    pub unit: String,
}

impl fmt::Display for IntervalLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "INTERVAL '{}'", self.value)
    }
}

/// Parameter ($1, ?)
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub token: Token,
    pub name: String,
    pub index: usize,
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.name.is_empty() {
            write!(f, "?")
        } else {
            write!(f, "{}", self.name)
        }
    }
}

/// Infix operator type (pre-computed at parse time for zero-allocation evaluation)
/// This is a key optimization: instead of string comparison for every row,
/// we match on a small enum which is faster and allocation-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InfixOperator {
    // Comparison operators
    Equal,        // =
    NotEqual,     // <> or !=
    LessThan,     // <
    LessEqual,    // <=
    GreaterThan,  // >
    GreaterEqual, // >=

    // Logical operators
    And,
    Or,
    Xor,

    // Arithmetic operators
    Add,      // +
    Subtract, // -
    Multiply, // *
    Divide,   // /
    Modulo,   // % or MOD

    // String operators
    Concat, // ||

    // Pattern matching
    Like,
    ILike,
    NotLike,
    NotILike,
    Glob,
    NotGlob,
    Regexp,
    NotRegexp,

    // Null check
    Is,                // IS (NULL)
    IsNot,             // IS NOT (NULL)
    IsDistinctFrom,    // IS DISTINCT FROM (NULL-safe not equal)
    IsNotDistinctFrom, // IS NOT DISTINCT FROM (NULL-safe equal)

    // Array index
    Index, // []

    // JSON operators
    JsonAccess,     // -> (returns JSON)
    JsonAccessText, // ->> (returns TEXT)

    // Bitwise operators
    BitwiseAnd, // &
    BitwiseOr,  // |
    BitwiseXor, // ^
    LeftShift,  // <<
    RightShift, // >>

    // Unknown/other (fallback for rare operators)
    Other,
}

impl InfixOperator {
    /// Parse operator string to enum (called once at parse time)
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "=" => InfixOperator::Equal,
            "<>" | "!=" => InfixOperator::NotEqual,
            "<" => InfixOperator::LessThan,
            "<=" => InfixOperator::LessEqual,
            ">" => InfixOperator::GreaterThan,
            ">=" => InfixOperator::GreaterEqual,
            "AND" => InfixOperator::And,
            "OR" => InfixOperator::Or,
            "XOR" => InfixOperator::Xor,
            "+" => InfixOperator::Add,
            "-" => InfixOperator::Subtract,
            "*" => InfixOperator::Multiply,
            "/" => InfixOperator::Divide,
            "%" | "MOD" => InfixOperator::Modulo,
            "||" => InfixOperator::Concat,
            "LIKE" => InfixOperator::Like,
            "ILIKE" => InfixOperator::ILike,
            "NOT LIKE" => InfixOperator::NotLike,
            "NOT ILIKE" => InfixOperator::NotILike,
            "GLOB" => InfixOperator::Glob,
            "NOT GLOB" => InfixOperator::NotGlob,
            "REGEXP" | "RLIKE" => InfixOperator::Regexp,
            "NOT REGEXP" | "NOT RLIKE" => InfixOperator::NotRegexp,
            "IS" => InfixOperator::Is,
            "IS NOT" => InfixOperator::IsNot,
            "IS DISTINCT FROM" => InfixOperator::IsDistinctFrom,
            "IS NOT DISTINCT FROM" => InfixOperator::IsNotDistinctFrom,
            "[]" => InfixOperator::Index,
            "->" => InfixOperator::JsonAccess,
            "->>" => InfixOperator::JsonAccessText,
            "&" => InfixOperator::BitwiseAnd,
            "|" => InfixOperator::BitwiseOr,
            "^" => InfixOperator::BitwiseXor,
            "<<" => InfixOperator::LeftShift,
            ">>" => InfixOperator::RightShift,
            _ => InfixOperator::Other,
        }
    }
}

/// Prefix operator type (pre-computed at parse time)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefixOperator {
    Negate,     // -
    Not,        // NOT
    Plus,       // + (unary plus, no-op)
    BitwiseNot, // ~ (bitwise NOT)
    Other,
}

impl PrefixOperator {
    /// Parse operator string to enum (called once at parse time)
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "-" => PrefixOperator::Negate,
            "NOT" => PrefixOperator::Not,
            "+" => PrefixOperator::Plus,
            "~" => PrefixOperator::BitwiseNot,
            _ => PrefixOperator::Other,
        }
    }
}

/// Star expression (*)
#[derive(Debug, Clone, PartialEq)]
pub struct StarExpression {
    pub token: Token,
}

impl fmt::Display for StarExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "*")
    }
}

/// Qualified star expression (table.*)
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedStarExpression {
    pub token: Token,
    pub qualifier: String,
}

impl fmt::Display for QualifiedStarExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.*", self.qualifier)
    }
}

/// DEFAULT keyword expression (for INSERT VALUES)
#[derive(Debug, Clone, PartialEq)]
pub struct DefaultExpression {
    pub token: Token,
}

impl fmt::Display for DefaultExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DEFAULT")
    }
}

/// Prefix expression (-x, NOT x)
#[derive(Debug, Clone, PartialEq)]
pub struct PrefixExpression {
    pub token: Token,
    pub operator: String,
    /// Pre-computed operator type for fast evaluation (no string comparison)
    pub op_type: PrefixOperator,
    pub right: Box<Expression>,
}

impl PrefixExpression {
    /// Create a new prefix expression with auto-computed op_type
    #[inline]
    pub fn new(token: Token, operator: String, right: Box<Expression>) -> Self {
        let op_type = PrefixOperator::from_str(&operator);
        Self {
            token,
            operator,
            op_type,
            right,
        }
    }
}

impl fmt::Display for PrefixExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.operator == "-" || self.operator == "+" {
            write!(f, "({}{})", self.operator, self.right)
        } else {
            write!(f, "({} {})", self.operator, self.right)
        }
    }
}

/// Infix expression (a + b, a = b)
#[derive(Debug, Clone, PartialEq)]
pub struct InfixExpression {
    pub token: Token,
    pub left: Box<Expression>,
    pub operator: String,
    /// Pre-computed operator type for fast evaluation (no string comparison)
    pub op_type: InfixOperator,
    pub right: Box<Expression>,
}

impl InfixExpression {
    /// Create a new infix expression with auto-computed op_type
    #[inline]
    pub fn new(
        token: Token,
        left: Box<Expression>,
        operator: String,
        right: Box<Expression>,
    ) -> Self {
        let op_type = InfixOperator::from_str(&operator);
        Self {
            token,
            left,
            operator,
            op_type,
            right,
        }
    }
}

impl fmt::Display for InfixExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {} {})", self.left, self.operator, self.right)
    }
}

/// List expression (for IN clause values)
#[derive(Debug, Clone, PartialEq)]
pub struct ListExpression {
    pub token: Token,
    pub elements: Vec<Expression>,
}

impl fmt::Display for ListExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements: Vec<String> = self.elements.iter().map(|e| e.to_string()).collect();
        write!(f, "({})", elements.join(", "))
    }
}

/// DISTINCT expression
#[derive(Debug, Clone, PartialEq)]
pub struct DistinctExpression {
    pub token: Token,
    pub expr: Box<Expression>,
}

impl fmt::Display for DistinctExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DISTINCT {}", self.expr)
    }
}

/// EXISTS expression
#[derive(Debug, Clone, PartialEq)]
pub struct ExistsExpression {
    pub token: Token,
    pub subquery: Box<SelectStatement>,
}

impl fmt::Display for ExistsExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EXISTS ({})", self.subquery)
    }
}

/// ALL/ANY comparison type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllAnyType {
    All,
    Any,
}

impl fmt::Display for AllAnyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllAnyType::All => write!(f, "ALL"),
            AllAnyType::Any => write!(f, "ANY"),
        }
    }
}

/// ALL/ANY/SOME subquery expression (e.g., x > ALL (SELECT ...))
#[derive(Debug, Clone, PartialEq)]
pub struct AllAnyExpression {
    pub token: Token,
    pub left: Box<Expression>,
    pub operator: String,
    pub all_any_type: AllAnyType,
    pub subquery: Box<SelectStatement>,
}

impl fmt::Display for AllAnyExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} ({})",
            self.left, self.operator, self.all_any_type, self.subquery
        )
    }
}

/// IN expression
#[derive(Debug, Clone, PartialEq)]
pub struct InExpression {
    pub token: Token,
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub not: bool,
}

impl fmt::Display for InExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.not {
            write!(f, "{} NOT IN {}", self.left, self.right)
        } else {
            write!(f, "{} IN {}", self.left, self.right)
        }
    }
}

/// Pre-computed IN expression with HashSet for O(1) lookup
///
/// This is used by the semi-join optimization to avoid rebuilding
/// the HashSet on every row during parallel filtering.
/// Arc enables cheap cloning when the expression is cloned for parallel execution.
#[derive(Debug, Clone)]
pub struct InHashSetExpression {
    pub token: Token,
    /// The column/expression to check
    pub column: Box<Expression>,
    /// Pre-computed AHashSet for O(1) lookup - Arc for cheap parallel cloning
    pub values: Arc<AHashSet<Value>>,
    /// Whether this is NOT IN
    pub not: bool,
}

impl PartialEq for InHashSetExpression {
    fn eq(&self, other: &Self) -> bool {
        // Compare by Arc pointer for efficiency (same HashSet = same Arc)
        self.not == other.not
            && Arc::ptr_eq(&self.values, &other.values)
            && self.column == other.column
    }
}

impl fmt::Display for InHashSetExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.not {
            write!(f, "{} NOT IN (<{} values>)", self.column, self.values.len())
        } else {
            write!(f, "{} IN (<{} values>)", self.column, self.values.len())
        }
    }
}

/// BETWEEN expression
#[derive(Debug, Clone, PartialEq)]
pub struct BetweenExpression {
    pub token: Token,
    pub expr: Box<Expression>,
    pub lower: Box<Expression>,
    pub upper: Box<Expression>,
    pub not: bool,
}

impl fmt::Display for BetweenExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.not {
            write!(
                f,
                "{} NOT BETWEEN {} AND {}",
                self.expr, self.lower, self.upper
            )
        } else {
            write!(f, "{} BETWEEN {} AND {}", self.expr, self.lower, self.upper)
        }
    }
}

/// LIKE expression with optional ESCAPE clause
#[derive(Debug, Clone, PartialEq)]
pub struct LikeExpression {
    pub token: Token,
    pub left: Box<Expression>,
    pub pattern: Box<Expression>,
    /// The operator: LIKE, ILIKE, NOT LIKE, NOT ILIKE, GLOB, NOT GLOB, REGEXP, RLIKE, NOT REGEXP, NOT RLIKE
    pub operator: String,
    /// Optional escape character
    pub escape: Option<Box<Expression>>,
}

impl fmt::Display for LikeExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.left, self.operator, self.pattern)?;
        if let Some(ref escape) = self.escape {
            write!(f, " ESCAPE {}", escape)?;
        }
        Ok(())
    }
}

/// Scalar subquery
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarSubquery {
    pub token: Token,
    pub subquery: Box<SelectStatement>,
}

impl fmt::Display for ScalarSubquery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})", self.subquery)
    }
}

/// Expression list (for IN values)
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionList {
    pub token: Token,
    pub expressions: Vec<Expression>,
}

impl fmt::Display for ExpressionList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exprs: Vec<String> = self.expressions.iter().map(|e| e.to_string()).collect();
        write!(f, "({})", exprs.join(", "))
    }
}

/// CASE expression
#[derive(Debug, Clone, PartialEq)]
pub struct CaseExpression {
    pub token: Token,
    pub value: Option<Box<Expression>>,
    pub when_clauses: Vec<WhenClause>,
    pub else_value: Option<Box<Expression>>,
}

impl fmt::Display for CaseExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("CASE");
        if let Some(ref val) = self.value {
            result.push_str(&format!(" {}", val));
        }
        for when in &self.when_clauses {
            result.push_str(&format!(" {}", when));
        }
        if let Some(ref else_val) = self.else_value {
            result.push_str(&format!(" ELSE {}", else_val));
        }
        result.push_str(" END");
        write!(f, "{}", result)
    }
}

/// WHEN clause in CASE expression
#[derive(Debug, Clone, PartialEq)]
pub struct WhenClause {
    pub token: Token,
    pub condition: Expression,
    pub then_result: Expression,
}

impl fmt::Display for WhenClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WHEN {} THEN {}", self.condition, self.then_result)
    }
}

/// CAST expression
#[derive(Debug, Clone, PartialEq)]
pub struct CastExpression {
    pub token: Token,
    pub expr: Box<Expression>,
    pub type_name: String,
}

impl fmt::Display for CastExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CAST({} AS {})", self.expr, self.type_name)
    }
}

/// Function call
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub token: Token,
    pub function: String,
    pub arguments: Vec<Expression>,
    pub is_distinct: bool,
    pub order_by: Vec<OrderByExpression>,
    /// FILTER clause for aggregate functions (e.g., COUNT(*) FILTER (WHERE condition))
    pub filter: Option<Box<Expression>>,
}

impl fmt::Display for FunctionCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut args = String::new();
        if self.is_distinct && !self.arguments.is_empty() {
            args.push_str("DISTINCT ");
            args.push_str(&self.arguments[0].to_string());
            for arg in &self.arguments[1..] {
                args.push_str(", ");
                args.push_str(&arg.to_string());
            }
        } else {
            let arg_strs: Vec<String> = self
                .arguments
                .iter()
                .map(|a| {
                    if matches!(a, Expression::Star(_)) {
                        "*".to_string()
                    } else {
                        a.to_string()
                    }
                })
                .collect();
            args = arg_strs.join(", ");
        }
        if !self.order_by.is_empty() {
            args.push_str(" ORDER BY ");
            let order_strs: Vec<String> = self.order_by.iter().map(|o| o.to_string()).collect();
            args.push_str(&order_strs.join(", "));
        }
        write!(f, "{}({})", self.function, args)?;
        if let Some(filter) = &self.filter {
            write!(f, " FILTER (WHERE {})", filter)?;
        }
        Ok(())
    }
}

/// Aliased expression (expr AS alias)
#[derive(Debug, Clone, PartialEq)]
pub struct AliasedExpression {
    pub token: Token,
    pub expression: Box<Expression>,
    pub alias: Identifier,
}

impl fmt::Display for AliasedExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} AS {}", self.expression, self.alias)
    }
}

/// Window expression
#[derive(Debug, Clone, PartialEq)]
pub struct WindowExpression {
    pub token: Token,
    pub function: Box<FunctionCall>,
    /// Named window reference (e.g., OVER w)
    pub window_ref: Option<String>,
    pub partition_by: Vec<Expression>,
    pub order_by: Vec<OrderByExpression>,
    pub frame: Option<WindowFrame>,
}

impl fmt::Display for WindowExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = self.function.to_string();
        if let Some(ref win_ref) = self.window_ref {
            result.push_str(" OVER ");
            result.push_str(win_ref);
        } else {
            result.push_str(" OVER (");
            if !self.partition_by.is_empty() {
                result.push_str("PARTITION BY ");
                let parts: Vec<String> = self.partition_by.iter().map(|e| e.to_string()).collect();
                result.push_str(&parts.join(", "));
            }
            if !self.order_by.is_empty() {
                if !self.partition_by.is_empty() {
                    result.push(' ');
                }
                result.push_str("ORDER BY ");
                let orders: Vec<String> = self.order_by.iter().map(|o| o.to_string()).collect();
                result.push_str(&orders.join(", "));
            }
            if let Some(ref frame) = self.frame {
                result.push(' ');
                result.push_str(&frame.to_string());
            }
            result.push(')');
        }
        write!(f, "{}", result)
    }
}

/// Window frame specification
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFrame {
    pub unit: WindowFrameUnit,
    pub start: WindowFrameBound,
    pub end: Option<WindowFrameBound>,
}

impl fmt::Display for WindowFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let unit = match self.unit {
            WindowFrameUnit::Rows => "ROWS",
            WindowFrameUnit::Range => "RANGE",
        };
        if let Some(ref end) = self.end {
            write!(f, "{} BETWEEN {} AND {}", unit, self.start, end)
        } else {
            write!(f, "{} {}", unit, self.start)
        }
    }
}

/// Window frame unit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameUnit {
    Rows,
    Range,
}

/// Window frame bound
#[derive(Debug, Clone, PartialEq)]
pub enum WindowFrameBound {
    CurrentRow,
    UnboundedPreceding,
    UnboundedFollowing,
    Preceding(Box<Expression>),
    Following(Box<Expression>),
}

impl fmt::Display for WindowFrameBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WindowFrameBound::CurrentRow => write!(f, "CURRENT ROW"),
            WindowFrameBound::UnboundedPreceding => write!(f, "UNBOUNDED PRECEDING"),
            WindowFrameBound::UnboundedFollowing => write!(f, "UNBOUNDED FOLLOWING"),
            WindowFrameBound::Preceding(e) => write!(f, "{} PRECEDING", e),
            WindowFrameBound::Following(e) => write!(f, "{} FOLLOWING", e),
        }
    }
}

/// Named window definition (WINDOW w AS (...))
#[derive(Debug, Clone, PartialEq)]
pub struct WindowDefinition {
    pub name: String,
    pub partition_by: Vec<Expression>,
    pub order_by: Vec<OrderByExpression>,
    pub frame: Option<WindowFrame>,
}

impl fmt::Display for WindowDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = format!("{} AS (", self.name);
        if !self.partition_by.is_empty() {
            result.push_str("PARTITION BY ");
            let parts: Vec<String> = self.partition_by.iter().map(|e| e.to_string()).collect();
            result.push_str(&parts.join(", "));
        }
        if !self.order_by.is_empty() {
            if !self.partition_by.is_empty() {
                result.push(' ');
            }
            result.push_str("ORDER BY ");
            let orders: Vec<String> = self.order_by.iter().map(|o| o.to_string()).collect();
            result.push_str(&orders.join(", "));
        }
        if let Some(ref frame) = self.frame {
            result.push(' ');
            result.push_str(&frame.to_string());
        }
        result.push(')');
        write!(f, "{}", result)
    }
}

// ============================================================================
// Table Sources
// ============================================================================

/// Simple table source
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleTableSource {
    pub token: Token,
    pub name: Identifier,
    pub alias: Option<Identifier>,
    pub as_of: Option<AsOfClause>,
}

impl fmt::Display for SimpleTableSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = self.name.to_string();
        if let Some(ref as_of) = self.as_of {
            result.push_str(&format!(" {}", as_of));
        }
        if let Some(ref alias) = self.alias {
            result.push_str(&format!(" AS {}", alias));
        }
        write!(f, "{}", result)
    }
}

/// AS OF clause for temporal queries
#[derive(Debug, Clone, PartialEq)]
pub struct AsOfClause {
    pub token: Token,
    pub as_of_type: String, // "TRANSACTION" or "TIMESTAMP"
    pub value: Box<Expression>,
}

impl fmt::Display for AsOfClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AS OF {} {}", self.as_of_type, self.value)
    }
}

/// Join table source
#[derive(Debug, Clone, PartialEq)]
pub struct JoinTableSource {
    pub token: Token,
    pub left: Box<Expression>,
    pub join_type: String,
    pub right: Box<Expression>,
    pub condition: Option<Box<Expression>>,
    /// USING clause columns (e.g., USING(id, name))
    pub using_columns: Vec<Identifier>,
}

impl fmt::Display for JoinTableSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = self.left.to_string();
        result.push_str(&format!(" {} JOIN {}", self.join_type, self.right));
        if let Some(ref cond) = self.condition {
            result.push_str(&format!(" ON {}", cond));
        } else if !self.using_columns.is_empty() {
            let cols: Vec<String> = self.using_columns.iter().map(|c| c.to_string()).collect();
            result.push_str(&format!(" USING ({})", cols.join(", ")));
        }
        write!(f, "{}", result)
    }
}

/// Subquery table source
#[derive(Debug, Clone, PartialEq)]
pub struct SubqueryTableSource {
    pub token: Token,
    pub subquery: Box<SelectStatement>,
    pub alias: Option<Identifier>,
}

impl fmt::Display for SubqueryTableSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = format!("({})", self.subquery);
        if let Some(ref alias) = self.alias {
            result.push_str(&format!(" AS {}", alias));
        }
        write!(f, "{}", result)
    }
}

/// VALUES table source (e.g., VALUES (1, 'a'), (2, 'b') AS t(col1, col2))
#[derive(Debug, Clone, PartialEq)]
pub struct ValuesTableSource {
    pub token: Token,
    /// Each row is a list of expressions
    pub rows: Vec<Vec<Expression>>,
    /// Optional alias for the derived table
    pub alias: Option<Identifier>,
    /// Optional column aliases (e.g., t(col1, col2))
    pub column_aliases: Vec<Identifier>,
}

impl fmt::Display for ValuesTableSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("(VALUES ");
        let rows_str: Vec<String> = self
            .rows
            .iter()
            .map(|row| {
                let values: Vec<String> = row.iter().map(|e| e.to_string()).collect();
                format!("({})", values.join(", "))
            })
            .collect();
        result.push_str(&rows_str.join(", "));
        result.push(')');

        if let Some(ref alias) = self.alias {
            result.push_str(&format!(" AS {}", alias));
            if !self.column_aliases.is_empty() {
                let cols: Vec<String> = self.column_aliases.iter().map(|c| c.to_string()).collect();
                result.push_str(&format!("({})", cols.join(", ")));
            }
        }
        write!(f, "{}", result)
    }
}

/// CTE reference
#[derive(Debug, Clone, PartialEq)]
pub struct CteReference {
    pub token: Token,
    pub name: Identifier,
    pub alias: Option<Identifier>,
}

impl fmt::Display for CteReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = self.name.to_string();
        if let Some(ref alias) = self.alias {
            result.push_str(&format!(" AS {}", alias));
        }
        write!(f, "{}", result)
    }
}

// ============================================================================
// ORDER BY
// ============================================================================

/// ORDER BY expression
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByExpression {
    pub expression: Expression,
    pub ascending: bool,
    /// None = default (NULLS LAST for ASC, NULLS FIRST for DESC in SQL standard)
    /// Some(true) = NULLS FIRST
    /// Some(false) = NULLS LAST
    pub nulls_first: Option<bool>,
}

impl fmt::Display for OrderByExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ascending {
            write!(f, "{} ASC", self.expression)?;
        } else {
            write!(f, "{} DESC", self.expression)?;
        }
        if let Some(nulls_first) = self.nulls_first {
            if nulls_first {
                write!(f, " NULLS FIRST")?;
            } else {
                write!(f, " NULLS LAST")?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Statements
// ============================================================================

/// Statement enum representing all statement types
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Select(SelectStatement),
    Insert(InsertStatement),
    Update(UpdateStatement),
    Delete(DeleteStatement),
    Truncate(TruncateStatement),
    CreateTable(CreateTableStatement),
    DropTable(DropTableStatement),
    AlterTable(AlterTableStatement),
    CreateIndex(CreateIndexStatement),
    DropIndex(DropIndexStatement),
    CreateColumnarIndex(CreateColumnarIndexStatement),
    DropColumnarIndex(DropColumnarIndexStatement),
    CreateView(CreateViewStatement),
    DropView(DropViewStatement),
    Begin(BeginStatement),
    Commit(CommitStatement),
    Rollback(RollbackStatement),
    Savepoint(SavepointStatement),
    Set(SetStatement),
    Pragma(PragmaStatement),
    ShowTables(ShowTablesStatement),
    ShowViews(ShowViewsStatement),
    ShowCreateTable(ShowCreateTableStatement),
    ShowCreateView(ShowCreateViewStatement),
    ShowIndexes(ShowIndexesStatement),
    Describe(DescribeStatement),
    Expression(ExpressionStatement),
    Explain(ExplainStatement),
    Analyze(AnalyzeStatement),
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Select(s) => write!(f, "{}", s),
            Statement::Insert(s) => write!(f, "{}", s),
            Statement::Update(s) => write!(f, "{}", s),
            Statement::Delete(s) => write!(f, "{}", s),
            Statement::Truncate(s) => write!(f, "{}", s),
            Statement::CreateTable(s) => write!(f, "{}", s),
            Statement::DropTable(s) => write!(f, "{}", s),
            Statement::AlterTable(s) => write!(f, "{}", s),
            Statement::CreateIndex(s) => write!(f, "{}", s),
            Statement::DropIndex(s) => write!(f, "{}", s),
            Statement::CreateColumnarIndex(s) => write!(f, "{}", s),
            Statement::DropColumnarIndex(s) => write!(f, "{}", s),
            Statement::CreateView(s) => write!(f, "{}", s),
            Statement::DropView(s) => write!(f, "{}", s),
            Statement::Begin(s) => write!(f, "{}", s),
            Statement::Commit(s) => write!(f, "{}", s),
            Statement::Rollback(s) => write!(f, "{}", s),
            Statement::Savepoint(s) => write!(f, "{}", s),
            Statement::Set(s) => write!(f, "{}", s),
            Statement::Pragma(s) => write!(f, "{}", s),
            Statement::ShowTables(s) => write!(f, "{}", s),
            Statement::ShowViews(s) => write!(f, "{}", s),
            Statement::ShowCreateTable(s) => write!(f, "{}", s),
            Statement::ShowCreateView(s) => write!(f, "{}", s),
            Statement::ShowIndexes(s) => write!(f, "{}", s),
            Statement::Describe(s) => write!(f, "{}", s),
            Statement::Expression(s) => write!(f, "{}", s),
            Statement::Explain(s) => write!(f, "{}", s),
            Statement::Analyze(s) => write!(f, "{}", s),
        }
    }
}

/// Program (collection of statements)
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Statement>,
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for stmt in &self.statements {
            write!(f, "{};", stmt)?;
        }
        Ok(())
    }
}

// ============================================================================
// WITH Clause (CTEs)
// ============================================================================

/// Common Table Expression
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTableExpression {
    pub token: Token,
    pub name: Identifier,
    pub column_names: Vec<Identifier>,
    pub query: Box<SelectStatement>,
    pub is_recursive: bool,
}

impl fmt::Display for CommonTableExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = self.name.to_string();
        if !self.column_names.is_empty() {
            let cols: Vec<String> = self.column_names.iter().map(|c| c.to_string()).collect();
            result.push_str(&format!("({})", cols.join(", ")));
        }
        result.push_str(&format!(" AS ({})", self.query));
        write!(f, "{}", result)
    }
}

/// WITH clause
#[derive(Debug, Clone, PartialEq)]
pub struct WithClause {
    pub token: Token,
    pub ctes: Vec<CommonTableExpression>,
    pub is_recursive: bool,
}

impl fmt::Display for WithClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("WITH ");
        if self.is_recursive {
            result.push_str("RECURSIVE ");
        }
        let cte_strs: Vec<String> = self.ctes.iter().map(|c| c.to_string()).collect();
        result.push_str(&cte_strs.join(", "));
        write!(f, "{}", result)
    }
}

// ============================================================================
// Statement Types
// ============================================================================

/// Set operation type for compound queries
#[derive(Debug, Clone, PartialEq)]
pub enum SetOperationType {
    Union,
    UnionAll,
    Intersect,
    IntersectAll,
    Except,
    ExceptAll,
}

impl fmt::Display for SetOperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SetOperationType::Union => write!(f, "UNION"),
            SetOperationType::UnionAll => write!(f, "UNION ALL"),
            SetOperationType::Intersect => write!(f, "INTERSECT"),
            SetOperationType::IntersectAll => write!(f, "INTERSECT ALL"),
            SetOperationType::Except => write!(f, "EXCEPT"),
            SetOperationType::ExceptAll => write!(f, "EXCEPT ALL"),
        }
    }
}

/// Set operation combining two SELECT statements
#[derive(Debug, Clone, PartialEq)]
pub struct SetOperation {
    pub operation: SetOperationType,
    pub right: Box<SelectStatement>,
}

/// Group by modifier (ROLLUP, CUBE, GROUPING SETS, or none)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GroupByModifier {
    #[default]
    None,
    Rollup,
    Cube,
    /// GROUPING SETS - each inner Vec is one grouping set
    /// e.g., GROUPING SETS ((a, b), (a), ()) has 3 sets
    GroupingSets(Vec<Vec<Expression>>),
}

/// GROUP BY clause with optional ROLLUP/CUBE modifier
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GroupByClause {
    pub columns: Vec<Expression>,
    pub modifier: GroupByModifier,
}

impl fmt::Display for GroupByClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // GROUPING SETS uses its own column lists, not self.columns
        if let GroupByModifier::GroupingSets(sets) = &self.modifier {
            let sets_str: Vec<String> = sets
                .iter()
                .map(|set| {
                    let cols: Vec<String> = set.iter().map(|c| c.to_string()).collect();
                    format!("({})", cols.join(", "))
                })
                .collect();
            return write!(f, "GROUPING SETS ({})", sets_str.join(", "));
        }

        // None, Rollup, Cube all use self.columns
        if self.columns.is_empty() {
            return Ok(());
        }
        let cols: Vec<String> = self.columns.iter().map(|c| c.to_string()).collect();
        match &self.modifier {
            GroupByModifier::None => write!(f, "{}", cols.join(", ")),
            GroupByModifier::Rollup => write!(f, "ROLLUP({})", cols.join(", ")),
            GroupByModifier::Cube => write!(f, "CUBE({})", cols.join(", ")),
            GroupByModifier::GroupingSets(_) => Ok(()), // Already handled above
        }
    }
}

/// SELECT statement
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    pub token: Token,
    pub distinct: bool,
    pub columns: Vec<Expression>,
    pub with: Option<WithClause>,
    pub table_expr: Option<Box<Expression>>,
    pub where_clause: Option<Box<Expression>>,
    pub group_by: GroupByClause,
    pub having: Option<Box<Expression>>,
    /// Named window definitions (WINDOW w AS (...))
    pub window_defs: Vec<WindowDefinition>,
    pub order_by: Vec<OrderByExpression>,
    pub limit: Option<Box<Expression>>,
    pub offset: Option<Box<Expression>>,
    /// Set operations (UNION, INTERSECT, EXCEPT)
    pub set_operations: Vec<SetOperation>,
}

impl fmt::Display for SelectStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::new();
        if let Some(ref with) = self.with {
            result.push_str(&format!("{} ", with));
        }
        result.push_str("SELECT ");
        if self.distinct {
            result.push_str("DISTINCT ");
        }
        let cols: Vec<String> = self.columns.iter().map(|c| c.to_string()).collect();
        result.push_str(&cols.join(", "));
        if let Some(ref table) = self.table_expr {
            result.push_str(&format!(" FROM {}", table));
        }
        if let Some(ref where_clause) = self.where_clause {
            result.push_str(&format!(" WHERE {}", where_clause));
        }
        if !self.group_by.columns.is_empty() {
            result.push_str(&format!(" GROUP BY {}", self.group_by));
        }
        if let Some(ref having) = self.having {
            result.push_str(&format!(" HAVING {}", having));
        }
        if !self.window_defs.is_empty() {
            let wins: Vec<String> = self.window_defs.iter().map(|w| w.to_string()).collect();
            result.push_str(&format!(" WINDOW {}", wins.join(", ")));
        }
        if !self.order_by.is_empty() {
            let orders: Vec<String> = self.order_by.iter().map(|o| o.to_string()).collect();
            result.push_str(&format!(" ORDER BY {}", orders.join(", ")));
        }
        if let Some(ref limit) = self.limit {
            result.push_str(&format!(" LIMIT {}", limit));
        }
        if let Some(ref offset) = self.offset {
            result.push_str(&format!(" OFFSET {}", offset));
        }
        // Add set operations
        for set_op in &self.set_operations {
            result.push_str(&format!(" {} {}", set_op.operation, set_op.right));
        }
        write!(f, "{}", result)
    }
}

/// INSERT statement
#[derive(Debug, Clone, PartialEq)]
pub struct InsertStatement {
    pub token: Token,
    pub table_name: Identifier,
    pub columns: Vec<Identifier>,
    /// VALUES clause rows (None if using SELECT)
    pub values: Vec<Vec<Expression>>,
    /// SELECT statement for INSERT INTO ... SELECT (None if using VALUES)
    pub select: Option<Box<SelectStatement>>,
    pub on_duplicate: bool,
    pub update_columns: Vec<Identifier>,
    pub update_expressions: Vec<Expression>,
    /// RETURNING clause expressions
    pub returning: Vec<Expression>,
}

impl fmt::Display for InsertStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = format!("INSERT INTO {}", self.table_name);
        if !self.columns.is_empty() {
            let cols: Vec<String> = self.columns.iter().map(|c| c.to_string()).collect();
            result.push_str(&format!(" ({})", cols.join(", ")));
        }
        if let Some(ref select) = self.select {
            // INSERT INTO ... SELECT
            result.push_str(&format!(" {}", select));
        } else {
            // INSERT INTO ... VALUES
            result.push_str(" VALUES ");
            let rows: Vec<String> = self
                .values
                .iter()
                .map(|row| {
                    let vals: Vec<String> = row.iter().map(|v| v.to_string()).collect();
                    format!("({})", vals.join(", "))
                })
                .collect();
            result.push_str(&rows.join(", "));
        }
        if self.on_duplicate {
            result.push_str(" ON DUPLICATE KEY UPDATE ");
            let updates: Vec<String> = self
                .update_columns
                .iter()
                .zip(&self.update_expressions)
                .map(|(col, expr)| format!("{} = {}", col, expr))
                .collect();
            result.push_str(&updates.join(", "));
        }
        if !self.returning.is_empty() {
            let returning: Vec<String> = self.returning.iter().map(|e| e.to_string()).collect();
            result.push_str(&format!(" RETURNING {}", returning.join(", ")));
        }
        write!(f, "{}", result)
    }
}

/// UPDATE statement
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateStatement {
    pub token: Token,
    pub table_name: Identifier,
    pub updates: HashMap<String, Expression>,
    pub where_clause: Option<Box<Expression>>,
    /// RETURNING clause expressions
    pub returning: Vec<Expression>,
}

impl fmt::Display for UpdateStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = format!("UPDATE {} SET ", self.table_name);
        let updates: Vec<String> = self
            .updates
            .iter()
            .map(|(col, val)| format!("{} = {}", col, val))
            .collect();
        result.push_str(&updates.join(", "));
        if let Some(ref where_clause) = self.where_clause {
            result.push_str(&format!(" WHERE {}", where_clause));
        }
        if !self.returning.is_empty() {
            let returning: Vec<String> = self.returning.iter().map(|e| e.to_string()).collect();
            result.push_str(&format!(" RETURNING {}", returning.join(", ")));
        }
        write!(f, "{}", result)
    }
}

/// DELETE statement
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStatement {
    pub token: Token,
    pub table_name: Identifier,
    /// Optional table alias (e.g., DELETE FROM users AS u)
    pub alias: Option<Identifier>,
    pub where_clause: Option<Box<Expression>>,
    /// RETURNING clause expressions
    pub returning: Vec<Expression>,
}

impl fmt::Display for DeleteStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = format!("DELETE FROM {}", self.table_name);
        if let Some(ref alias) = self.alias {
            result.push_str(&format!(" AS {}", alias));
        }
        if let Some(ref where_clause) = self.where_clause {
            result.push_str(&format!(" WHERE {}", where_clause));
        }
        if !self.returning.is_empty() {
            let returning: Vec<String> = self.returning.iter().map(|e| e.to_string()).collect();
            result.push_str(&format!(" RETURNING {}", returning.join(", ")));
        }
        write!(f, "{}", result)
    }
}

/// CREATE TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStatement {
    pub token: Token,
    pub table_name: Identifier,
    pub if_not_exists: bool,
    pub columns: Vec<ColumnDefinition>,
    /// Table-level constraints (UNIQUE(cols), CHECK(expr), etc.)
    pub table_constraints: Vec<TableConstraint>,
    /// Optional SELECT statement for CREATE TABLE ... AS SELECT
    pub as_select: Option<Box<SelectStatement>>,
}

impl fmt::Display for CreateTableStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("CREATE TABLE ");
        if self.if_not_exists {
            result.push_str("IF NOT EXISTS ");
        }
        if let Some(ref select) = self.as_select {
            result.push_str(&format!("{} AS {}", self.table_name, select));
            return write!(f, "{}", result);
        }
        result.push_str(&format!("{} (", self.table_name));
        let cols: Vec<String> = self.columns.iter().map(|c| c.to_string()).collect();
        result.push_str(&cols.join(", "));
        if !self.table_constraints.is_empty() {
            let constraints: Vec<String> = self
                .table_constraints
                .iter()
                .map(|c| c.to_string())
                .collect();
            result.push_str(", ");
            result.push_str(&constraints.join(", "));
        }
        result.push(')');
        write!(f, "{}", result)
    }
}

/// Column definition
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDefinition {
    pub name: Identifier,
    pub data_type: String,
    pub constraints: Vec<ColumnConstraint>,
}

impl fmt::Display for ColumnDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = format!("{} {}", self.name, self.data_type);
        for constraint in &self.constraints {
            result.push_str(&format!(" {}", constraint));
        }
        write!(f, "{}", result)
    }
}

/// Column constraint
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnConstraint {
    NotNull,
    PrimaryKey,
    Unique,
    AutoIncrement,
    Default(Expression),
    Check(Expression),
    References(Identifier, Option<Identifier>),
}

impl fmt::Display for ColumnConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnConstraint::NotNull => write!(f, "NOT NULL"),
            ColumnConstraint::PrimaryKey => write!(f, "PRIMARY KEY"),
            ColumnConstraint::Unique => write!(f, "UNIQUE"),
            ColumnConstraint::AutoIncrement => write!(f, "AUTO_INCREMENT"),
            ColumnConstraint::Default(expr) => write!(f, "DEFAULT {}", expr),
            ColumnConstraint::Check(expr) => write!(f, "CHECK ({})", expr),
            ColumnConstraint::References(table, col) => {
                if let Some(col) = col {
                    write!(f, "REFERENCES {}({})", table, col)
                } else {
                    write!(f, "REFERENCES {}", table)
                }
            }
        }
    }
}

/// Table-level constraint (applied to the table rather than a single column)
#[derive(Debug, Clone, PartialEq)]
pub enum TableConstraint {
    /// UNIQUE(col1, col2, ...)
    Unique(Vec<Identifier>),
    /// CHECK(expression) - boxed to reduce enum size
    Check(Box<Expression>),
    /// PRIMARY KEY(col1, col2, ...) - composite primary key (not yet fully supported)
    PrimaryKey(Vec<Identifier>),
}

impl fmt::Display for TableConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableConstraint::Unique(cols) => {
                let col_names: Vec<String> = cols.iter().map(|c| c.value.clone()).collect();
                write!(f, "UNIQUE({})", col_names.join(", "))
            }
            TableConstraint::Check(expr) => write!(f, "CHECK({})", expr),
            TableConstraint::PrimaryKey(cols) => {
                let col_names: Vec<String> = cols.iter().map(|c| c.value.clone()).collect();
                write!(f, "PRIMARY KEY({})", col_names.join(", "))
            }
        }
    }
}

/// Helper enum for parsing - either a column definition or a table constraint
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnOrConstraint {
    Column(ColumnDefinition),
    Constraint(TableConstraint),
}

/// DROP TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct DropTableStatement {
    pub token: Token,
    pub table_name: Identifier,
    pub if_exists: bool,
}

impl fmt::Display for DropTableStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("DROP TABLE ");
        if self.if_exists {
            result.push_str("IF EXISTS ");
        }
        result.push_str(&self.table_name.to_string());
        write!(f, "{}", result)
    }
}

/// TRUNCATE TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct TruncateStatement {
    pub token: Token,
    pub table_name: Identifier,
}

impl fmt::Display for TruncateStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TRUNCATE TABLE {}", self.table_name)
    }
}

/// ALTER TABLE operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlterTableOperation {
    AddColumn,
    DropColumn,
    RenameColumn,
    ModifyColumn,
    RenameTable,
}

/// ALTER TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableStatement {
    pub token: Token,
    pub table_name: Identifier,
    pub operation: AlterTableOperation,
    pub column_def: Option<ColumnDefinition>,
    pub column_name: Option<Identifier>,
    pub new_column_name: Option<Identifier>,
    pub new_table_name: Option<Identifier>,
}

impl fmt::Display for AlterTableStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = format!("ALTER TABLE {} ", self.table_name);
        match self.operation {
            AlterTableOperation::AddColumn => {
                if let Some(ref col) = self.column_def {
                    result.push_str(&format!("ADD COLUMN {}", col));
                }
            }
            AlterTableOperation::DropColumn => {
                if let Some(ref name) = self.column_name {
                    result.push_str(&format!("DROP COLUMN {}", name));
                }
            }
            AlterTableOperation::RenameColumn => {
                if let (Some(ref old), Some(ref new)) = (&self.column_name, &self.new_column_name) {
                    result.push_str(&format!("RENAME COLUMN {} TO {}", old, new));
                }
            }
            AlterTableOperation::ModifyColumn => {
                if let Some(ref col) = self.column_def {
                    result.push_str(&format!("MODIFY COLUMN {}", col));
                }
            }
            AlterTableOperation::RenameTable => {
                if let Some(ref name) = self.new_table_name {
                    result.push_str(&format!("RENAME TO {}", name));
                }
            }
        }
        write!(f, "{}", result)
    }
}

/// Index type for USING clause
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMethod {
    /// B-tree index (default for INTEGER, FLOAT, TIMESTAMP) - good for range queries
    BTree,
    /// Hash index (default for TEXT, JSON) - good for equality lookups
    Hash,
    /// Bitmap index (default for BOOLEAN) - good for low-cardinality columns
    Bitmap,
}

impl fmt::Display for IndexMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexMethod::BTree => write!(f, "BTREE"),
            IndexMethod::Hash => write!(f, "HASH"),
            IndexMethod::Bitmap => write!(f, "BITMAP"),
        }
    }
}

/// CREATE INDEX statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexStatement {
    pub token: Token,
    pub index_name: Identifier,
    pub table_name: Identifier,
    pub columns: Vec<Identifier>,
    pub is_unique: bool,
    pub if_not_exists: bool,
    /// Optional index type from USING clause (None = auto-select based on column type)
    pub index_method: Option<IndexMethod>,
}

impl fmt::Display for CreateIndexStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("CREATE ");
        if self.is_unique {
            result.push_str("UNIQUE ");
        }
        result.push_str("INDEX ");
        if self.if_not_exists {
            result.push_str("IF NOT EXISTS ");
        }
        result.push_str(&format!("{} ON {} (", self.index_name, self.table_name));
        let cols: Vec<String> = self.columns.iter().map(|c| c.to_string()).collect();
        result.push_str(&cols.join(", "));
        result.push(')');
        if let Some(method) = &self.index_method {
            result.push_str(&format!(" USING {}", method));
        }
        write!(f, "{}", result)
    }
}

/// DROP INDEX statement
#[derive(Debug, Clone, PartialEq)]
pub struct DropIndexStatement {
    pub token: Token,
    pub index_name: Identifier,
    pub table_name: Option<Identifier>,
    pub if_exists: bool,
}

impl fmt::Display for DropIndexStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("DROP INDEX ");
        if self.if_exists {
            result.push_str("IF EXISTS ");
        }
        result.push_str(&self.index_name.to_string());
        if let Some(ref table) = self.table_name {
            result.push_str(&format!(" ON {}", table));
        }
        write!(f, "{}", result)
    }
}

/// CREATE COLUMNAR INDEX statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateColumnarIndexStatement {
    pub token: Token,
    pub table_name: Identifier,
    pub column_name: Identifier,
    pub if_not_exists: bool,
    pub is_unique: bool,
}

impl fmt::Display for CreateColumnarIndexStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("CREATE ");
        if self.is_unique {
            result.push_str("UNIQUE ");
        }
        result.push_str("COLUMNAR INDEX ");
        if self.if_not_exists {
            result.push_str("IF NOT EXISTS ");
        }
        result.push_str(&format!("ON {} ({})", self.table_name, self.column_name));
        write!(f, "{}", result)
    }
}

/// DROP COLUMNAR INDEX statement
#[derive(Debug, Clone, PartialEq)]
pub struct DropColumnarIndexStatement {
    pub token: Token,
    pub table_name: Identifier,
    pub column_name: Identifier,
    pub if_exists: bool,
}

impl fmt::Display for DropColumnarIndexStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("DROP COLUMNAR INDEX ");
        if self.if_exists {
            result.push_str("IF EXISTS ");
        }
        result.push_str(&format!("ON {} ({})", self.table_name, self.column_name));
        write!(f, "{}", result)
    }
}

/// CREATE VIEW statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateViewStatement {
    pub token: Token,
    pub view_name: Identifier,
    pub query: Box<SelectStatement>,
    pub if_not_exists: bool,
}

impl fmt::Display for CreateViewStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("CREATE VIEW ");
        if self.if_not_exists {
            result.push_str("IF NOT EXISTS ");
        }
        result.push_str(&format!("{} AS {}", self.view_name, self.query));
        write!(f, "{}", result)
    }
}

/// DROP VIEW statement
#[derive(Debug, Clone, PartialEq)]
pub struct DropViewStatement {
    pub token: Token,
    pub view_name: Identifier,
    pub if_exists: bool,
}

impl fmt::Display for DropViewStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("DROP VIEW ");
        if self.if_exists {
            result.push_str("IF EXISTS ");
        }
        result.push_str(&self.view_name.to_string());
        write!(f, "{}", result)
    }
}

/// BEGIN statement
#[derive(Debug, Clone, PartialEq)]
pub struct BeginStatement {
    pub token: Token,
    pub isolation_level: Option<String>,
}

impl fmt::Display for BeginStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::from("BEGIN TRANSACTION");
        if let Some(ref level) = self.isolation_level {
            result.push_str(&format!(" ISOLATION LEVEL {}", level));
        }
        write!(f, "{}", result)
    }
}

/// COMMIT statement
#[derive(Debug, Clone, PartialEq)]
pub struct CommitStatement {
    pub token: Token,
}

impl fmt::Display for CommitStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "COMMIT")
    }
}

/// ROLLBACK statement
#[derive(Debug, Clone, PartialEq)]
pub struct RollbackStatement {
    pub token: Token,
    pub savepoint_name: Option<Identifier>,
}

impl fmt::Display for RollbackStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.savepoint_name {
            write!(f, "ROLLBACK TO SAVEPOINT {}", name)
        } else {
            write!(f, "ROLLBACK")
        }
    }
}

/// SAVEPOINT statement
#[derive(Debug, Clone, PartialEq)]
pub struct SavepointStatement {
    pub token: Token,
    pub savepoint_name: Identifier,
}

impl fmt::Display for SavepointStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SAVEPOINT {}", self.savepoint_name)
    }
}

/// SET statement
#[derive(Debug, Clone, PartialEq)]
pub struct SetStatement {
    pub token: Token,
    pub name: Identifier,
    pub value: Expression,
}

impl fmt::Display for SetStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SET {} = {}", self.name, self.value)
    }
}

/// PRAGMA statement
#[derive(Debug, Clone, PartialEq)]
pub struct PragmaStatement {
    pub token: Token,
    pub name: Identifier,
    pub value: Option<Expression>,
}

impl fmt::Display for PragmaStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref value) = self.value {
            write!(f, "PRAGMA {} = {}", self.name, value)
        } else {
            write!(f, "PRAGMA {}", self.name)
        }
    }
}

/// SHOW TABLES statement
#[derive(Debug, Clone, PartialEq)]
pub struct ShowTablesStatement {
    pub token: Token,
}

impl fmt::Display for ShowTablesStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SHOW TABLES")
    }
}

/// SHOW VIEWS statement
#[derive(Debug, Clone, PartialEq)]
pub struct ShowViewsStatement {
    pub token: Token,
}

impl fmt::Display for ShowViewsStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SHOW VIEWS")
    }
}

/// SHOW CREATE TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct ShowCreateTableStatement {
    pub token: Token,
    pub table_name: Identifier,
}

impl fmt::Display for ShowCreateTableStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SHOW CREATE TABLE {}", self.table_name)
    }
}

/// SHOW CREATE VIEW statement
#[derive(Debug, Clone, PartialEq)]
pub struct ShowCreateViewStatement {
    pub token: Token,
    pub view_name: Identifier,
}

impl fmt::Display for ShowCreateViewStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SHOW CREATE VIEW {}", self.view_name)
    }
}

/// SHOW INDEXES statement
#[derive(Debug, Clone, PartialEq)]
pub struct ShowIndexesStatement {
    pub token: Token,
    pub table_name: Identifier,
}

impl fmt::Display for ShowIndexesStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SHOW INDEXES FROM {}", self.table_name)
    }
}

/// DESCRIBE statement - shows table structure
#[derive(Debug, Clone, PartialEq)]
pub struct DescribeStatement {
    pub token: Token,
    pub table_name: Identifier,
}

impl fmt::Display for DescribeStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DESCRIBE {}", self.table_name)
    }
}

/// Expression statement (standalone expression)
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionStatement {
    pub token: Token,
    pub expression: Expression,
}

impl fmt::Display for ExpressionStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expression)
    }
}

/// EXPLAIN statement
#[derive(Debug, Clone, PartialEq)]
pub struct ExplainStatement {
    pub token: Token,
    pub statement: Box<Statement>,
    pub analyze: bool,
}

impl fmt::Display for ExplainStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.analyze {
            write!(f, "EXPLAIN ANALYZE {}", self.statement)
        } else {
            write!(f, "EXPLAIN {}", self.statement)
        }
    }
}

/// ANALYZE statement for collecting table statistics
#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzeStatement {
    pub token: Token,
    /// Table name to analyze (None = analyze all tables)
    pub table_name: Option<String>,
}

impl fmt::Display for AnalyzeStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.table_name {
            Some(name) => write!(f, "ANALYZE {}", name),
            None => write!(f, "ANALYZE"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::token::TokenType;

    fn make_token(tt: TokenType, literal: &str) -> Token {
        Token::new(tt, literal, Position::default())
    }

    #[test]
    fn test_identifier_display() {
        let id = Identifier::new(
            make_token(TokenType::Identifier, "users"),
            "users".to_string(),
        );
        assert_eq!(id.to_string(), "users");
    }

    #[test]
    fn test_qualified_identifier_display() {
        let qi = QualifiedIdentifier {
            token: make_token(TokenType::Identifier, "users"),
            qualifier: Box::new(Identifier::new(
                make_token(TokenType::Identifier, "users"),
                "users".to_string(),
            )),
            name: Box::new(Identifier::new(
                make_token(TokenType::Identifier, "id"),
                "id".to_string(),
            )),
        };
        assert_eq!(qi.to_string(), "users.id");
    }

    #[test]
    fn test_integer_literal_display() {
        let lit = IntegerLiteral {
            token: make_token(TokenType::Integer, "42"),
            value: 42,
        };
        assert_eq!(lit.to_string(), "42");
    }

    #[test]
    fn test_string_literal_display() {
        let lit = StringLiteral {
            token: make_token(TokenType::String, "'hello'"),
            value: "hello".to_string(),
            type_hint: None,
        };
        assert_eq!(lit.to_string(), "'hello'");
    }

    #[test]
    fn test_infix_expression_display() {
        let expr = InfixExpression::new(
            make_token(TokenType::Operator, "+"),
            Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: make_token(TokenType::Integer, "1"),
                value: 1,
            })),
            "+".to_string(),
            Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: make_token(TokenType::Integer, "2"),
                value: 2,
            })),
        );
        assert_eq!(expr.to_string(), "(1 + 2)");
    }

    #[test]
    fn test_function_call_display() {
        let fc = FunctionCall {
            token: make_token(TokenType::Identifier, "COUNT"),
            function: "COUNT".to_string(),
            arguments: vec![Expression::Star(StarExpression {
                token: make_token(TokenType::Operator, "*"),
            })],
            is_distinct: false,
            order_by: vec![],
            filter: None,
        };
        assert_eq!(fc.to_string(), "COUNT(*)");
    }

    #[test]
    fn test_select_statement_display() {
        let stmt = SelectStatement {
            token: make_token(TokenType::Keyword, "SELECT"),
            distinct: false,
            columns: vec![Expression::Star(StarExpression {
                token: make_token(TokenType::Operator, "*"),
            })],
            with: None,
            table_expr: Some(Box::new(Expression::TableSource(SimpleTableSource {
                token: make_token(TokenType::Identifier, "users"),
                name: Identifier::new(
                    make_token(TokenType::Identifier, "users"),
                    "users".to_string(),
                ),
                alias: None,
                as_of: None,
            }))),
            where_clause: None,
            group_by: GroupByClause::default(),
            having: None,
            window_defs: vec![],
            order_by: vec![],
            limit: None,
            offset: None,
            set_operations: vec![],
        };
        assert_eq!(stmt.to_string(), "SELECT * FROM users");
    }

    #[test]
    fn test_create_table_display() {
        let stmt = CreateTableStatement {
            token: make_token(TokenType::Keyword, "CREATE"),
            table_name: Identifier::new(
                make_token(TokenType::Identifier, "users"),
                "users".to_string(),
            ),
            if_not_exists: true,
            columns: vec![
                ColumnDefinition {
                    name: Identifier::new(
                        make_token(TokenType::Identifier, "id"),
                        "id".to_string(),
                    ),
                    data_type: "INTEGER".to_string(),
                    constraints: vec![ColumnConstraint::PrimaryKey],
                },
                ColumnDefinition {
                    name: Identifier::new(
                        make_token(TokenType::Identifier, "name"),
                        "name".to_string(),
                    ),
                    data_type: "TEXT".to_string(),
                    constraints: vec![ColumnConstraint::NotNull],
                },
            ],
            table_constraints: vec![],
            as_select: None,
        };
        assert_eq!(
            stmt.to_string(),
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"
        );
    }

    #[test]
    fn test_insert_statement_display() {
        let stmt = InsertStatement {
            token: make_token(TokenType::Keyword, "INSERT"),
            table_name: Identifier::new(
                make_token(TokenType::Identifier, "users"),
                "users".to_string(),
            ),
            columns: vec![
                Identifier::new(make_token(TokenType::Identifier, "id"), "id".to_string()),
                Identifier::new(
                    make_token(TokenType::Identifier, "name"),
                    "name".to_string(),
                ),
            ],
            values: vec![vec![
                Expression::IntegerLiteral(IntegerLiteral {
                    token: make_token(TokenType::Integer, "1"),
                    value: 1,
                }),
                Expression::StringLiteral(StringLiteral {
                    token: make_token(TokenType::String, "'Alice'"),
                    value: "Alice".to_string(),
                    type_hint: None,
                }),
            ]],
            select: None,
            on_duplicate: false,
            update_columns: vec![],
            update_expressions: vec![],
            returning: vec![],
        };
        assert_eq!(
            stmt.to_string(),
            "INSERT INTO users (id, name) VALUES (1, 'Alice')"
        );
    }

    #[test]
    fn test_case_expression_display() {
        let case_expr = CaseExpression {
            token: make_token(TokenType::Keyword, "CASE"),
            value: None,
            when_clauses: vec![WhenClause {
                token: make_token(TokenType::Keyword, "WHEN"),
                condition: Expression::BooleanLiteral(BooleanLiteral {
                    token: make_token(TokenType::Keyword, "TRUE"),
                    value: true,
                }),
                then_result: Expression::IntegerLiteral(IntegerLiteral {
                    token: make_token(TokenType::Integer, "1"),
                    value: 1,
                }),
            }],
            else_value: Some(Box::new(Expression::IntegerLiteral(IntegerLiteral {
                token: make_token(TokenType::Integer, "0"),
                value: 0,
            }))),
        };
        assert_eq!(case_expr.to_string(), "CASE WHEN TRUE THEN 1 ELSE 0 END");
    }
}
