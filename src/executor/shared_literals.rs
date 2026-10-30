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

//! Texts that differ only in literal values share one plan: a literal a
//! WHERE compares with a column is read as a hidden parameter after the
//! statement's own, every other literal stays as written and tells apart
//! the plans of one skeleton

use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::api::params::ParamVec;
use crate::common::SmartString;
use crate::core::{Result, Value};
use crate::parser::ast::{
    Expression, InfixOperator, ParamSlots, Parameter, PrefixOperator, Statement,
};
use crate::parser::token::{Keyword, Token, TokenType};
use crate::parser::{float_literal, integer_literal, string_literal, Lexer};

use super::expression::literal_value;
use super::prepared_select::rebuild;
use super::query_cache::CachedPlanRef;

/// A text with each literal replaced by a mark of its kind, its literals,
/// and every token the lexer read, for the parser
pub(crate) struct Skeleton {
    pub(crate) key: String,
    literals: SmallVec<[Token; 4]>,
    /// Whether each literal follows a minus sign, which the parser folds
    /// into it
    negated: SmallVec<[bool; 4]>,
    pub(crate) tokens: Vec<Token>,
}

/// The kind of literal a hidden parameter is filled from
#[derive(Clone, Copy, PartialEq, Eq)]
enum LiteralKind {
    Integer,
    Float,
    Text,
}

/// How a plan reads one literal of its texts, in text order
enum LiteralUse {
    /// Kept in the plan: a text must write it the same
    Fixed(SmartString),
    /// The next hidden parameter, filled from a literal of this kind
    Shared(LiteralKind),
}

/// A plan the texts of one skeleton share
pub(crate) struct SharedPlan {
    literals: Box<[LiteralUse]>,
    /// The slots the statement's own parameters take
    pub(crate) user: ParamSlots,
    pub(crate) plan: CachedPlanRef,
}

/// The skeleton of a SELECT, UPDATE or DELETE text; None for any other
/// text, one the lexer refuses, one with named parameters or one holding
/// the mark's own character
pub(crate) fn skeleton(sql: &str) -> Option<Skeleton> {
    if sql.contains('\u{1}') {
        return None;
    }
    let mut lexer = Lexer::new(sql);
    let mut key = String::with_capacity(sql.len() + 8);
    let mut literals = SmallVec::new();
    let mut negated = SmallVec::new();
    let mut tokens = Vec::with_capacity(sql.len() / 4 + 2);
    // Where the text after the last literal starts
    let mut rest = 0;
    let mut first = true;
    let mut minus = false;
    loop {
        let token = lexer.next_token();
        match token.token_type {
            TokenType::Eof => {
                key.push_str(sql.get(rest..)?);
                tokens.push(token);
                break;
            }
            TokenType::Comment => {}
            TokenType::Error => return None,
            _ if first
                && !matches!(
                    token.keyword,
                    Some(Keyword::Select | Keyword::Update | Keyword::Delete)
                ) =>
            {
                return None
            }
            TokenType::Integer | TokenType::Float | TokenType::String => {
                key.push_str(sql.get(rest..token.position.offset as usize)?);
                key.push('\u{1}');
                key.push(match token.token_type {
                    TokenType::Integer => '#',
                    TokenType::Float => '.',
                    _ => 's',
                });
                rest = lexer.offset();
                literals.push(token.clone());
                negated.push(minus);
            }
            TokenType::Parameter if token.literal.starts_with(':') => return None,
            _ => {}
        }
        if token.token_type != TokenType::Comment {
            first = false;
            minus = token.token_type == TokenType::Operator && token.literal == "-";
        }
        tokens.push(token);
    }
    Some(Skeleton {
        key,
        literals,
        negated,
        tokens,
    })
}

impl SharedPlan {
    /// The hidden parameters the literals of `skeleton` fill; None when a
    /// fixed literal is written otherwise or a shared one reads as another
    /// kind of literal
    pub(crate) fn bind(&self, skeleton: &Skeleton) -> Option<ParamVec> {
        let mut values = ParamVec::new();
        for (literal, token) in self.literals.iter().zip(&skeleton.literals) {
            match literal {
                LiteralUse::Fixed(text) if token.literal == *text => {}
                LiteralUse::Fixed(_) => return None,
                LiteralUse::Shared(kind) => values.push(shared_value(token, *kind)?),
            }
        }
        Some(values)
    }
}

/// The value of the literal `token`, when the parser reads it as a literal
/// of `kind`
fn shared_value(token: &Token, kind: LiteralKind) -> Option<Value> {
    let literal = literal_of(token)?;
    (kind_of(&literal) == Some(kind))
        .then(|| literal_value(&literal))
        .flatten()
}

/// The literal the parser reads `token` as
fn literal_of(token: &Token) -> Option<Expression> {
    match token.token_type {
        TokenType::Integer => integer_literal(token).ok(),
        TokenType::Float => float_literal(token).ok(),
        TokenType::String => Some(string_literal(token)),
        _ => None,
    }
}

/// The kind of a literal a hidden parameter takes; None for a typed one
fn kind_of(literal: &Expression) -> Option<LiteralKind> {
    match literal {
        Expression::IntegerLiteral(_) => Some(LiteralKind::Integer),
        Expression::FloatLiteral(_) => Some(LiteralKind::Float),
        Expression::StringLiteral(text) if text.type_hint.is_none() => Some(LiteralKind::Text),
        _ => None,
    }
}

/// The token of the number or text literal `expr`
fn literal_token(expr: &Expression) -> Option<&Token> {
    match expr {
        Expression::IntegerLiteral(lit) => Some(&lit.token),
        Expression::FloatLiteral(lit) => Some(&lit.token),
        Expression::StringLiteral(lit) => Some(&lit.token),
        _ => None,
    }
}

/// Where the literal `expr` starts in its text
fn literal_offset(expr: &Expression) -> Option<u32> {
    literal_token(expr).map(|token| token.position.offset)
}

/// The plan of `stmt`, parsed from the text of `skeleton` with the slots
/// `user`, whose WHERE reads each literal it compares with a column as a
/// hidden parameter; None for a statement of no shared form or with no
/// such literal
pub(crate) fn share(
    stmt: &Statement,
    user: &ParamSlots,
    skeleton: &Skeleton,
) -> Option<SharedPlan> {
    if user.names.is_some() {
        return None;
    }
    let where_clause = match stmt {
        Statement::Select(select)
            if select.with.is_none()
                && select.set_operations.is_empty()
                && matches!(
                    select.table_expr.as_deref(),
                    Some(Expression::TableSource(_))
                ) =>
        {
            select.where_clause.as_deref()?
        }
        Statement::Update(update) if update.with.is_none() => update.where_clause.as_deref()?,
        Statement::Delete(delete) if delete.with.is_none() => delete.where_clause.as_deref()?,
        _ => return None,
    };
    let at: FxHashMap<u32, usize> = skeleton
        .literals
        .iter()
        .enumerate()
        .map(|(i, token)| (token.position.offset, i))
        .collect();
    // A literal is shared when it follows no minus sign and the parser read
    // its token alone the same way: a typed one is kept as written
    let mut shared = vec![false; skeleton.literals.len()];
    compared_literals(where_clause, &mut |literal, _place| {
        let Some(&i) = literal_offset(literal).and_then(|offset| at.get(&offset)) else {
            return;
        };
        let alone = literal_of(&skeleton.literals[i]);
        shared[i] = !skeleton.negated[i]
            && alone.as_ref().is_some_and(|alone| {
                kind_of(alone).is_some()
                    && kind_of(alone) == kind_of(literal)
                    && literal_value(alone) == literal_value(literal)
            });
        #[cfg(feature = "test-failpoints")]
        if shared[i] {
            use crate::test_failpoints::SharedLiteral;
            crate::test_failpoints::literal_shared(match _place {
                Place::Comparison => SharedLiteral::Comparison,
                Place::Range => SharedLiteral::Range,
                Place::List => SharedLiteral::List,
            });
        }
    });
    if !shared.contains(&true) {
        return None;
    }
    let mut slots = FxHashMap::default();
    let mut literals = Vec::with_capacity(shared.len());
    for (token, shared) in skeleton.literals.iter().zip(shared) {
        literals.push(match shared {
            true => {
                let slot = user.count.checked_add(slots.len() + 1)?;
                slots.insert(token.position.offset, slot);
                LiteralUse::Shared(literal_of(token).as_ref().and_then(kind_of)?)
            }
            false => LiteralUse::Fixed(token.literal.clone()),
        });
    }
    let where_clause = with_slots(where_clause, &slots).ok()?;
    let mut stmt = stmt.clone();
    match &mut stmt {
        Statement::Select(select) => select.where_clause = Some(Box::new(where_clause)),
        Statement::Update(update) => update.where_clause = Some(Box::new(where_clause)),
        Statement::Delete(delete) => delete.where_clause = Some(Box::new(where_clause)),
        _ => return None,
    }
    let params = ParamSlots {
        count: user.count + slots.len(),
        names: None,
    };
    stmt.set_param_count(params.count);
    Some(SharedPlan {
        literals: literals.into_boxed_slice(),
        user: user.clone(),
        plan: CachedPlanRef::new(Arc::new(stmt), params),
    })
}

/// Where a WHERE tests a column against a literal it may share
#[derive(Clone, Copy)]
enum Place {
    Comparison,
    Range,
    List,
}

/// Hands `each` every literal of `expr` that a comparison, a range or an
/// IN list tests a column against, under its ANDs, ORs and NOTs
fn compared_literals<'e>(expr: &'e Expression, each: &mut impl FnMut(&'e Expression, Place)) {
    let column = |e: &Expression| {
        matches!(
            e,
            Expression::Identifier(_) | Expression::QualifiedIdentifier(_)
        )
    };
    let literal = |e: &Expression| literal_offset(e).is_some();
    match expr {
        Expression::Infix(infix)
            if matches!(infix.op_type, InfixOperator::And | InfixOperator::Or) =>
        {
            compared_literals(&infix.left, each);
            compared_literals(&infix.right, each);
        }
        Expression::Prefix(prefix) if prefix.op_type == PrefixOperator::Not => {
            compared_literals(&prefix.right, each)
        }
        Expression::Infix(infix) if is_comparison(infix.op_type) => {
            if column(&infix.left) && literal(&infix.right) {
                each(&infix.right, Place::Comparison);
            }
            if column(&infix.right) && literal(&infix.left) {
                each(&infix.left, Place::Comparison);
            }
        }
        Expression::Between(between) if column(&between.expr) => {
            for bound in [&between.lower, &between.upper] {
                if literal(bound) {
                    each(bound, Place::Range);
                }
            }
        }
        Expression::In(list) if column(&list.left) => {
            let members = match list.right.as_ref() {
                Expression::List(members) => &members.elements,
                Expression::ExpressionList(members) => &members.expressions,
                _ => return,
            };
            for member in members.iter().filter(|m| literal(m)) {
                each(member, Place::List);
            }
        }
        _ => {}
    }
}

fn is_comparison(op: InfixOperator) -> bool {
    matches!(
        op,
        InfixOperator::Equal
            | InfixOperator::NotEqual
            | InfixOperator::LessThan
            | InfixOperator::LessEqual
            | InfixOperator::GreaterThan
            | InfixOperator::GreaterEqual
    )
}

/// `expr` with each literal starting at an offset of `slots` read as the
/// parameter of its slot
fn with_slots(expr: &Expression, slots: &FxHashMap<u32, usize>) -> Result<Expression> {
    if let Some(token) = literal_token(expr) {
        if let Some(&slot) = slots.get(&token.position.offset) {
            return Ok(Expression::Parameter(Parameter {
                token: token.clone(),
                name: SmartString::from(format!("${slot}")),
                index: slot,
            }));
        }
    }
    rebuild(expr, &mut |child| with_slots(child, slots))
}
