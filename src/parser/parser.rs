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

//! SQL Parser - Main Parser struct and core parsing logic

use super::ast::*;
use super::error::{ParseError, ParseErrors};
use super::lexer::Lexer;
use super::precedence::Precedence;
use super::token::{Keyword, Token, TokenType};
use crate::common::SmartString;

/// Reserved SQL keywords that cannot be used as identifiers
fn is_reserved(keyword: Keyword) -> bool {
    use Keyword::*;
    matches!(
        keyword,
        // Core SQL keywords that should never be identifiers
        Select
            | From
            | Where
            | And
            | Or
            | Not
            | Insert
            | Into
            | Values
            | Update
            | Set
            | Delete
            | Create
            | Drop
            | Table
            | Index
            | View
            | Alter
            | Add
            | Primary
            | Key
            | Foreign
            | References
            | Null
            | True
            | False
            | As
            | On
            | Join
            // LEFT and RIGHT are handled specially - they can be function names
            // or column names when not followed by JOIN
            | Inner
            | Outer
            | Full
            | Cross
            | Group
            | By
            | Order
            | Having
            | Limit
            | Offset
            | Union
            | Intersect
            | Except
            | Case
            | When
            | Then
            | Else
            | End
            | Distinct
            | All
            | Exists
            | In
            | Between
            | Like
            | Glob
            | Regexp
            | Rlike
            | Is
            | Asc
            | Desc
            | Nulls
            // FIRST and LAST are handled specially - they can be function names
            // or column names, or ORDER BY modifiers (NULLS FIRST/LAST)
            | Begin
            | Commit
            | Rollback
            | Savepoint
            | If
            | With
            | Recursive
            | Default
    )
}

/// Where a parser reads its tokens: a lexer, or the tokens a lexer read
/// before, up to its end of input
enum Tokens<'a> {
    Lexer(Lexer<'a>),
    Read(std::vec::IntoIter<Token>, Token),
}

impl Tokens<'_> {
    fn next_token(&mut self) -> Token {
        match self {
            Tokens::Lexer(lexer) => lexer.next_token(),
            Tokens::Read(tokens, eof) => tokens.next().unwrap_or_else(|| eof.clone()),
        }
    }
}

/// The lexer's next token that is not a comment. An unterminated block
/// comment is recorded as an error, so the program is not run.
fn next_significant_token(lexer: &mut Tokens, errors: &mut Vec<ParseError>) -> Token {
    loop {
        let token = lexer.next_token();
        if token.token_type != TokenType::Comment {
            return token;
        }
        let text = token.literal.as_str();
        if text.starts_with("/*") && !(text.len() >= 4 && text.ends_with("*/")) {
            errors.push(ParseError::new(
                "unterminated block comment".to_string(),
                token.position,
            ));
        }
    }
}

/// A view is stored SQL, read inside other statements: it has no values of its own
pub(crate) const VIEW_TAKES_NO_PARAMETERS: &str = "a view cannot take parameters";

/// Whether running `stmt` creates a view, under any EXPLAIN
fn creates_view(stmt: &Statement) -> bool {
    match stmt {
        Statement::CreateView(_) => true,
        Statement::Explain(explain) => creates_view(&explain.statement),
        _ => false,
    }
}

/// SQL Parser using Pratt parsing algorithm
pub struct Parser<'a> {
    /// The lexer providing tokens
    lexer: Tokens<'a>,
    /// Current token being examined
    pub(crate) cur_token: Token,
    /// Next token (peek)
    pub(crate) peek_token: Token,
    /// Collected errors
    errors: Vec<ParseError>,
    /// Current clause context (for error messages and parameter tracking)
    pub(crate) current_clause: &'static str,
    /// Parameter counter within current statement
    parameter_counter: usize,
    /// The highest positional slot of the current statement
    max_positional: usize,
    /// The `:name` slots of the current statement, in slot order
    named_slots: Vec<SmartString>,
    /// Whether each statement's qualifiers are checked against its FROM
    /// entries; off for text the engine wrote itself
    scoped: bool,
    /// Whether the current statement wrote what the scope check can refuse:
    /// a qualifier, a SELECT DISTINCT and its ORDER BY, or an ORDER BY position
    pub(crate) scope_checked: bool,
}

impl<'a> Parser<'a> {
    /// Create a new parser for the given input
    pub fn new(input: &'a str) -> Self {
        Self::reading(Tokens::Lexer(Lexer::new(input)))
    }

    /// A parser of the tokens a lexer read from a text, its end of input
    /// last
    pub(crate) fn from_tokens(tokens: Vec<Token>) -> Parser<'static> {
        let eof = tokens.last().cloned();
        let eof = eof.unwrap_or_else(|| Token::eof(Default::default()));
        Parser::reading(Tokens::Read(tokens.into_iter(), eof))
    }

    fn reading(mut lexer: Tokens<'a>) -> Self {
        let mut errors = Vec::new();
        let cur_token = next_significant_token(&mut lexer, &mut errors);
        let peek_token = next_significant_token(&mut lexer, &mut errors);

        Parser {
            lexer,
            cur_token,
            peek_token,
            errors,
            current_clause: "",
            parameter_counter: 1,
            max_positional: 0,
            named_slots: Vec::new(),
            scoped: true,
            scope_checked: false,
        }
    }

    /// A parser for a stored expression's text, whose qualifiers were
    /// checked when the user wrote it
    pub(crate) fn unscoped(mut self) -> Self {
        self.scoped = false;
        self
    }

    /// Parse the input and return a Program
    pub fn parse_program(&mut self) -> Result<Program, ParseErrors> {
        // Pre-allocate for common case (most queries have 1 statement)
        let mut statements = Vec::with_capacity(1);
        let mut params = smallvec::SmallVec::new();

        while !self.cur_token_is(TokenType::Eof) {
            if let Some((stmt, slots)) = self.parse_statement_with_slots() {
                params.push(slots);
                statements.push(stmt);
            }

            // Skip optional semicolons (handles trailing `;;` etc.)
            while self.peek_token_is_punctuator(";") {
                self.next_token();
            }

            self.next_token();
        }

        if !self.errors.is_empty() {
            return Err(ParseErrors::from_errors(self.errors.clone()));
        }

        Ok(Program { statements, params })
    }

    /// The slots `stmt`, just parsed whole, takes, leaving the counters for
    /// the next statement; an EXPLAIN without ANALYZE reads no value, it
    /// shows each parameter as written
    pub(crate) fn take_param_slots(&mut self, stmt: &Statement) -> ParamSlots {
        let named = std::mem::take(&mut self.named_slots);
        let positional = std::mem::replace(&mut self.max_positional, 0);
        self.parameter_counter = 1;
        if (positional > 0 || !named.is_empty()) && creates_view(stmt) {
            self.add_error(VIEW_TAKES_NO_PARAMETERS.to_string());
        }
        if matches!(stmt, Statement::Explain(explain) if !explain.analyze) {
            return ParamSlots::default();
        }
        if named.is_empty() {
            return ParamSlots {
                count: positional,
                names: None,
            };
        }
        ParamSlots {
            count: named.len(),
            names: Some(named.into()),
        }
    }

    /// The slot of the parameter spelled `name`: `?` takes the next one, `$n`
    /// slot n, and each distinct `:name` the next one on its first appearance.
    /// None, with an error added, for a slot a statement cannot have
    pub(crate) fn parameter_slot(&mut self, name: &str) -> Option<usize> {
        let named = name.starts_with(':');
        if named && self.max_positional > 0 || !named && !self.named_slots.is_empty() {
            self.add_error(
                "a statement takes positional ($1, ?) or named (:name) parameters, not both"
                    .to_string(),
            );
            return None;
        }
        if let Some(name) = name.strip_prefix(':') {
            if let Some(at) = self.named_slots.iter().position(|slot| slot == name) {
                return Some(at + 1);
            }
            self.named_slots.push(SmartString::new(name));
            return Some(self.named_slots.len());
        }
        let index = if name == "?" {
            self.next_parameter_index()
        } else {
            match name.strip_prefix('$').map(str::parse::<usize>) {
                Some(Ok(index)) if index > 0 => index,
                Some(Ok(_)) => {
                    self.add_error("parameter $0: positions start at $1".to_string());
                    return None;
                }
                Some(Err(e)) => {
                    self.add_error(format!("invalid parameter index: {}", e));
                    return None;
                }
                None => {
                    self.add_error(format!("invalid parameter format: {}", name));
                    return None;
                }
            }
        };
        self.max_positional = self.max_positional.max(index);
        Some(index)
    }

    /// Advance to the next token
    pub(crate) fn next_token(&mut self) {
        let next = next_significant_token(&mut self.lexer, &mut self.errors);
        self.cur_token = std::mem::replace(&mut self.peek_token, next);
    }

    /// Check if the current token is of the given type
    pub(crate) fn cur_token_is(&self, t: TokenType) -> bool {
        self.cur_token.token_type == t
    }

    /// Check if the peek token is of the given type
    pub(crate) fn peek_token_is(&self, t: TokenType) -> bool {
        self.peek_token.token_type == t
    }

    /// Check if the current token can be used as an identifier
    /// This allows keywords like TIMESTAMP, DATE, etc. to be used as column/table names
    pub(crate) fn cur_token_is_identifier_like(&self) -> bool {
        match self.cur_token.token_type {
            TokenType::Identifier => true,
            TokenType::Keyword => {
                // Allow non-reserved keywords as identifiers
                // Reserved keywords that cannot be used as identifiers
                !Self::is_reserved_keyword(&self.cur_token)
            }
            _ => false,
        }
    }

    /// Create an Identifier from the current token.
    /// Identifier::new automatically lowercases keyword tokens.
    pub(crate) fn cur_token_as_column_identifier(&self) -> Identifier {
        Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone())
    }

    /// Check if a keyword is truly reserved and cannot be used as an identifier
    /// Note: Some keywords like LEFT, RIGHT, FIRST, LAST are handled specially in
    /// parse_keyword_prefix() where they can be functions or identifiers.
    pub(crate) fn is_reserved_keyword(token: &Token) -> bool {
        token.keyword.is_some_and(is_reserved)
    }

    /// Check if the current token is a specific keyword
    pub(crate) fn cur_token_is_keyword(&self, keyword: Keyword) -> bool {
        self.cur_token.keyword == Some(keyword)
    }

    /// Check if the peek token is a specific keyword
    pub(crate) fn peek_token_is_keyword(&self, keyword: Keyword) -> bool {
        self.peek_token.keyword == Some(keyword)
    }

    /// Check if the current token is a specific punctuator
    pub(crate) fn cur_token_is_punctuator(&self, punc: &str) -> bool {
        self.cur_token.token_type == TokenType::Punctuator && self.cur_token.literal == punc
    }

    /// Check if the peek token is a specific punctuator
    pub(crate) fn peek_token_is_punctuator(&self, punc: &str) -> bool {
        self.peek_token.token_type == TokenType::Punctuator && self.peek_token.literal == punc
    }

    /// Check if the peek token is a specific operator
    pub(crate) fn peek_token_is_operator(&self, op: &str) -> bool {
        self.peek_token.token_type == TokenType::Operator && self.peek_token.literal == op
    }

    /// Check if the peek token can be used as an identifier (true identifier or non-reserved keyword)
    pub(crate) fn peek_token_is_identifier_like(&self) -> bool {
        match self.peek_token.token_type {
            TokenType::Identifier => true,
            TokenType::Keyword => !Self::is_reserved_keyword(&self.peek_token),
            _ => false,
        }
    }

    /// Expect the peek token to be an identifier (or non-reserved keyword) and advance
    pub(crate) fn expect_peek_identifier_like(&mut self) -> bool {
        if self.peek_token_is_identifier_like() {
            self.next_token();
            true
        } else {
            self.peek_error(TokenType::Identifier);
            false
        }
    }

    /// Expect the peek token to be of a specific type and advance
    pub(crate) fn expect_peek(&mut self, t: TokenType) -> bool {
        if self.peek_token_is(t) {
            self.next_token();
            true
        } else {
            self.peek_error(t);
            false
        }
    }

    /// Expect the peek token to be a specific keyword and advance
    pub(crate) fn expect_keyword(&mut self, keyword: Keyword) -> bool {
        if self.peek_token_is_keyword(keyword) {
            self.next_token();
            true
        } else {
            self.add_error(format!(
                "expected {} after {}, got {}",
                keyword.as_str(),
                self.cur_token.literal,
                Self::format_token_for_error(&self.peek_token)
            ));
            false
        }
    }

    /// Get the precedence of the peek token
    pub(crate) fn peek_precedence(&self) -> Precedence {
        match self.peek_token.token_type {
            TokenType::Operator => Precedence::for_operator(&self.peek_token.literal),
            TokenType::Keyword => self
                .peek_token
                .keyword
                .map_or(Precedence::Lowest, Precedence::for_keyword),
            TokenType::Punctuator => {
                if self.peek_token.literal == "." {
                    Precedence::Dot
                } else if self.peek_token.literal == "(" {
                    Precedence::Call
                } else if self.peek_token.literal == "[" {
                    Precedence::Index
                } else {
                    Precedence::Lowest
                }
            }
            _ => Precedence::Lowest,
        }
    }

    /// Get the precedence of the current token
    pub(crate) fn cur_precedence(&self) -> Precedence {
        match self.cur_token.token_type {
            TokenType::Operator => Precedence::for_operator(&self.cur_token.literal),
            TokenType::Keyword => self
                .cur_token
                .keyword
                .map_or(Precedence::Lowest, Precedence::for_keyword),
            TokenType::Punctuator => {
                if self.cur_token.literal == "." {
                    Precedence::Dot
                } else if self.cur_token.literal == "(" {
                    Precedence::Call
                } else if self.cur_token.literal == "[" {
                    Precedence::Index
                } else {
                    Precedence::Lowest
                }
            }
            _ => Precedence::Lowest,
        }
    }

    /// Add an error for unexpected peek token type
    pub(crate) fn peek_error(&mut self, expected: TokenType) {
        let expected_desc = match expected {
            TokenType::Identifier => "identifier (name)",
            TokenType::Keyword => "keyword",
            TokenType::Punctuator => "'(' or ')'",
            TokenType::String => "string literal",
            TokenType::Integer => "integer",
            TokenType::Float => "number",
            _ => "token",
        };

        if self.peek_token.token_type == TokenType::Eof {
            if !self.current_clause.is_empty() {
                self.add_error(format!(
                    "expected {} after {}",
                    expected_desc, self.current_clause
                ));
            } else {
                self.add_error(format!(
                    "unexpected end of input, expected {}",
                    expected_desc
                ));
            }
        } else if expected == TokenType::Identifier
            && self.peek_token.token_type == TokenType::Keyword
            && Self::is_reserved_keyword(&self.peek_token)
        {
            self.add_error(format!(
                "'{}' is a reserved keyword and cannot be used as an identifier. \
                 Use double quotes to escape it: \"{}\"",
                self.peek_token.literal.to_uppercase(),
                self.peek_token.literal
            ));
        } else {
            self.add_error(format!(
                "expected {}, got {}",
                expected_desc,
                Self::format_token_for_error(&self.peek_token)
            ));
        }
    }

    /// Format a token for display in error messages (shows "end of input" for EOF)
    pub(crate) fn format_token_for_error(token: &Token) -> String {
        if token.token_type == TokenType::Eof {
            "end of input".to_string()
        } else {
            format!("'{}'", token.literal)
        }
    }

    /// Add an error message
    pub(crate) fn add_error(&mut self, msg: String) {
        self.errors
            .push(ParseError::new(msg, self.cur_token.position));
    }

    /// Refuses a qualifier `stmt`, parsed whole without error, writes where
    /// no FROM entry in scope has that name
    pub(crate) fn check_scopes(&mut self, stmt: &Statement) {
        if !std::mem::take(&mut self.scope_checked) || !self.scoped || !self.errors.is_empty() {
            return;
        }
        if let Err(error) = super::scope::check(stmt) {
            self.errors
                .push(ParseError::new(error.message, error.position));
        }
    }

    /// Get collected errors
    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    /// Get the next parameter index
    pub(crate) fn next_parameter_index(&mut self) -> usize {
        let idx = self.parameter_counter;
        self.parameter_counter += 1;
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = Parser::new("SELECT * FROM users");
        assert!(parser.cur_token_is_keyword(Keyword::Select));
    }

    #[test]
    fn test_next_token() {
        let mut parser = Parser::new("SELECT * FROM users");
        assert!(parser.cur_token_is_keyword(Keyword::Select));
        parser.next_token();
        assert!(parser.cur_token_is(TokenType::Operator));
        assert_eq!(parser.cur_token.literal, "*");
    }

    #[test]
    fn test_peek_token() {
        let parser = Parser::new("SELECT * FROM users");
        assert!(parser.cur_token_is_keyword(Keyword::Select));
        assert!(parser.peek_token_is_operator("*"));
    }
}
