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

use std::collections::HashSet;
use std::sync::LazyLock;

use super::ast::*;
use super::error::{ParseError, ParseErrors};
use super::lexer::Lexer;
use super::precedence::Precedence;
use super::token::{Token, TokenType};

/// Reserved SQL keywords that cannot be used as identifiers (O(1) lookup)
static RESERVED_KEYWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // Core SQL keywords that should never be identifiers
        "SELECT",
        "FROM",
        "WHERE",
        "AND",
        "OR",
        "NOT",
        "INSERT",
        "INTO",
        "VALUES",
        "UPDATE",
        "SET",
        "DELETE",
        "CREATE",
        "DROP",
        "TABLE",
        "INDEX",
        "VIEW",
        "ALTER",
        "ADD",
        "PRIMARY",
        "KEY",
        "FOREIGN",
        "REFERENCES",
        "NULL",
        "TRUE",
        "FALSE",
        "AS",
        "ON",
        "JOIN",
        // LEFT and RIGHT are handled specially - they can be function names
        // or column names when not followed by JOIN
        "INNER",
        "OUTER",
        "FULL",
        "CROSS",
        "GROUP",
        "BY",
        "ORDER",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "DISTINCT",
        "ALL",
        "EXISTS",
        "IN",
        "BETWEEN",
        "LIKE",
        "GLOB",
        "REGEXP",
        "RLIKE",
        "IS",
        "ASC",
        "DESC",
        "NULLS",
        // FIRST and LAST are handled specially - they can be function names
        // or column names, or ORDER BY modifiers (NULLS FIRST/LAST)
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
        "IF",
        "WITH",
        "RECURSIVE",
    ]
    .into_iter()
    .collect()
});

/// SQL Parser using Pratt parsing algorithm
pub struct Parser {
    /// The lexer providing tokens
    lexer: Lexer,
    /// Current token being examined
    pub(crate) cur_token: Token,
    /// Next token (peek)
    pub(crate) peek_token: Token,
    /// Collected errors
    errors: Vec<ParseError>,
    /// Current clause context (for error messages and parameter tracking)
    pub(crate) current_clause: String,
    /// Current statement ID (for multi-statement queries)
    current_statement_id: usize,
    /// Parameter counter within current statement
    parameter_counter: usize,
}

impl Parser {
    /// Create a new parser for the given input
    pub fn new(input: &str) -> Self {
        let mut lexer = Lexer::new(input);
        let cur_token = lexer.next_token();
        let peek_token = lexer.next_token();

        Parser {
            lexer,
            cur_token,
            peek_token,
            errors: Vec::new(),
            current_clause: String::new(),
            current_statement_id: 0,
            parameter_counter: 1,
        }
    }

    /// Parse the input and return a Program
    pub fn parse_program(&mut self) -> Result<Program, ParseErrors> {
        let mut statements = Vec::new();

        while !self.cur_token_is(TokenType::Eof) {
            // Skip comments
            if self.cur_token_is(TokenType::Comment) {
                self.next_token();
                continue;
            }

            if let Some(stmt) = self.parse_statement() {
                statements.push(stmt);
            }

            // Skip optional semicolon
            if self.peek_token_is_punctuator(";") {
                self.next_token();
            }

            self.next_token();
            self.current_statement_id += 1;
            self.parameter_counter = 1;
        }

        if !self.errors.is_empty() {
            return Err(ParseErrors::from_errors(self.errors.clone()));
        }

        Ok(Program { statements })
    }

    /// Advance to the next token
    pub(crate) fn next_token(&mut self) {
        self.cur_token = std::mem::replace(&mut self.peek_token, self.lexer.next_token());
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
                !Self::is_reserved_keyword(&self.cur_token.literal)
            }
            _ => false,
        }
    }

    /// Check if a keyword is truly reserved and cannot be used as an identifier
    /// Note: Some keywords like LEFT, RIGHT, FIRST, LAST are handled specially in
    /// parse_keyword_prefix() where they can be functions or identifiers.
    /// Uses O(1) HashSet lookup instead of O(n) match chain.
    pub(crate) fn is_reserved_keyword(keyword: &str) -> bool {
        // Use uppercase for case-insensitive comparison
        // Note: Keywords are typically already uppercase from the lexer
        RESERVED_KEYWORDS.contains(keyword.to_uppercase().as_str())
    }

    /// Check if the current token is a specific keyword
    pub(crate) fn cur_token_is_keyword(&self, keyword: &str) -> bool {
        self.cur_token.token_type == TokenType::Keyword
            && self.cur_token.literal.eq_ignore_ascii_case(keyword)
    }

    /// Check if the peek token is a specific keyword
    pub(crate) fn peek_token_is_keyword(&self, keyword: &str) -> bool {
        self.peek_token.token_type == TokenType::Keyword
            && self.peek_token.literal.eq_ignore_ascii_case(keyword)
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
    pub(crate) fn expect_keyword(&mut self, keyword: &str) -> bool {
        if self.peek_token_is_keyword(keyword) {
            self.next_token();
            true
        } else {
            self.add_error(format!(
                "expected keyword {}, got {} at {}",
                keyword, self.peek_token.literal, self.peek_token.position
            ));
            false
        }
    }

    /// Get the precedence of the peek token
    pub(crate) fn peek_precedence(&self) -> Precedence {
        match self.peek_token.token_type {
            TokenType::Operator => Precedence::for_operator(&self.peek_token.literal),
            TokenType::Keyword => Precedence::for_operator(&self.peek_token.literal),
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
            TokenType::Keyword => Precedence::for_operator(&self.cur_token.literal),
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
        // Give a better error message when expecting an identifier but got a reserved keyword
        if expected == TokenType::Identifier
            && self.peek_token.token_type == TokenType::Keyword
            && Self::is_reserved_keyword(&self.peek_token.literal)
        {
            self.add_error(format!(
                "'{}' is a reserved keyword and cannot be used as an identifier. Use double quotes to escape it: \"{}\"",
                self.peek_token.literal.to_uppercase(),
                self.peek_token.literal
            ));
        } else {
            self.add_error(format!(
                "expected {:?}, got {:?} at {}",
                expected, self.peek_token.token_type, self.peek_token.position
            ));
        }
    }

    /// Add an error message
    pub(crate) fn add_error(&mut self, msg: String) {
        self.errors
            .push(ParseError::new(msg, self.cur_token.position));
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

    /// Get current statement ID
    #[allow(dead_code)]
    pub(crate) fn current_statement_id(&self) -> usize {
        self.current_statement_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = Parser::new("SELECT * FROM users");
        assert!(parser.cur_token_is_keyword("SELECT"));
    }

    #[test]
    fn test_next_token() {
        let mut parser = Parser::new("SELECT * FROM users");
        assert!(parser.cur_token_is_keyword("SELECT"));
        parser.next_token();
        assert!(parser.cur_token_is(TokenType::Operator));
        assert_eq!(parser.cur_token.literal, "*");
    }

    #[test]
    fn test_peek_token() {
        let parser = Parser::new("SELECT * FROM users");
        assert!(parser.cur_token_is_keyword("SELECT"));
        assert!(parser.peek_token_is_operator("*"));
    }
}
