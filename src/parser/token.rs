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

//! Token types for SQL lexer
//!
//! This module defines the token types used by the SQL lexer and parser.

use rustc_hash::FxHashSet;
use std::fmt;
use std::sync::LazyLock;

/// Position represents a position in the input source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    /// Byte offset, starting at 0
    pub offset: usize,
    /// Line number, starting at 1
    pub line: usize,
    /// Column number, starting at 1
    pub column: usize,
}

impl Position {
    /// Create a new position
    pub fn new(offset: usize, line: usize, column: usize) -> Self {
        Self {
            offset,
            line,
            column,
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// TokenType represents the type of a token
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    /// Error token
    Error,
    /// End of file
    Eof,
    /// Identifier (table name, column name, etc.)
    Identifier,
    /// SQL keyword (SELECT, FROM, WHERE, etc.)
    Keyword,
    /// String literal ('hello')
    String,
    /// Integer number (123)
    Integer,
    /// Floating point number (123.45)
    Float,
    /// Operator (=, <, >, +, -, etc.)
    Operator,
    /// Punctuator (comma, semicolon, parentheses, etc.)
    Punctuator,
    /// Comment (-- or /* */)
    Comment,
    /// Date literal
    Date,
    /// Time literal
    Time,
    /// Timestamp literal
    Timestamp,
    /// Parameter ($1, ?)
    Parameter,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::Error => write!(f, "ERROR"),
            TokenType::Eof => write!(f, "EOF"),
            TokenType::Identifier => write!(f, "IDENTIFIER"),
            TokenType::Keyword => write!(f, "KEYWORD"),
            TokenType::String => write!(f, "STRING"),
            TokenType::Integer => write!(f, "INTEGER"),
            TokenType::Float => write!(f, "FLOAT"),
            TokenType::Operator => write!(f, "OPERATOR"),
            TokenType::Punctuator => write!(f, "PUNCTUATOR"),
            TokenType::Comment => write!(f, "COMMENT"),
            TokenType::Date => write!(f, "DATE"),
            TokenType::Time => write!(f, "TIME"),
            TokenType::Timestamp => write!(f, "TIMESTAMP"),
            TokenType::Parameter => write!(f, "PARAMETER"),
        }
    }
}

/// Token represents a lexical token
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The type of the token
    pub token_type: TokenType,
    /// The literal string value
    pub literal: String,
    /// The position in the source
    pub position: Position,
    /// Error message (if token_type is Error)
    pub error: Option<String>,
}

impl Token {
    /// Create a new token
    pub fn new(token_type: TokenType, literal: impl Into<String>, position: Position) -> Self {
        Self {
            token_type,
            literal: literal.into(),
            position,
            error: None,
        }
    }

    /// Create an error token
    pub fn error(
        message: impl Into<String>,
        literal: impl Into<String>,
        position: Position,
    ) -> Self {
        Self {
            token_type: TokenType::Error,
            literal: literal.into(),
            position,
            error: Some(message.into()),
        }
    }

    /// Create an EOF token
    pub fn eof(position: Position) -> Self {
        Self {
            token_type: TokenType::Eof,
            literal: String::new(),
            position,
            error: None,
        }
    }

    /// Check if this is an EOF token
    pub fn is_eof(&self) -> bool {
        self.token_type == TokenType::Eof
    }

    /// Check if this is an error token
    pub fn is_error(&self) -> bool {
        self.token_type == TokenType::Error
    }

    /// Check if this is a keyword with the given value (case-insensitive)
    pub fn is_keyword(&self, keyword: &str) -> bool {
        self.token_type == TokenType::Keyword && self.literal.eq_ignore_ascii_case(keyword)
    }

    /// Check if this is an operator with the given value
    pub fn is_operator(&self, op: &str) -> bool {
        self.token_type == TokenType::Operator && self.literal == op
    }

    /// Check if this is a punctuator with the given value
    pub fn is_punctuator(&self, punct: &str) -> bool {
        self.token_type == TokenType::Punctuator && self.literal == punct
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.token_type == TokenType::Error {
            write!(
                f,
                "{}: {} at {}",
                self.token_type,
                self.error.as_deref().unwrap_or("unknown error"),
                self.position
            )
        } else if self.token_type == TokenType::Keyword {
            write!(
                f,
                "{}: {} at {}",
                self.token_type, self.literal, self.position
            )
        } else {
            write!(
                f,
                "{}: '{}' at {}",
                self.token_type, self.literal, self.position
            )
        }
    }
}

/// SQL keywords (case-insensitive)
pub static KEYWORDS: &[&str] = &[
    "SELECT",
    "FROM",
    "WHERE",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "CREATE",
    "TABLE",
    "DROP",
    "ALTER",
    "ADD",
    "COLUMN",
    "AND",
    "OR",
    "XOR",
    "NOT",
    "NULL",
    "PRIMARY",
    "PRAGMA",
    "KEY",
    "AUTO_INCREMENT",
    "AUTOINCREMENT",
    "DEFAULT",
    "AS",
    "OF",
    "DISTINCT",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "LIMIT",
    "OFFSET",
    "GROUP",
    "HAVING",
    "JOIN",
    "INNER",
    "OUTER",
    "LEFT",
    "RIGHT",
    "FULL",
    "ON",
    "DUPLICATE",
    "USING",
    "CROSS",
    "NATURAL",
    "TRUE",
    "FALSE",
    "INTEGER",
    "FLOAT",
    "TEXT",
    "BOOLEAN",
    "BOOL",
    "TIMESTAMP",
    "TIMESTAMPTZ",
    "DATETIME",
    "DATE",
    "TIME",
    "JSON",
    "CASE",
    "CAST",
    "EXTRACT",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "BETWEEN",
    "IN",
    "IS",
    "LIKE",
    "ILIKE",
    "ESCAPE",
    "GLOB",
    "REGEXP",
    "RLIKE",
    "EXISTS",
    "ALL",
    "ANY",
    "SOME",
    "IF",
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "WITH",
    "UNIQUE",
    "CHECK",
    "FOREIGN",
    "REFERENCES",
    "SHOW",
    "DESCRIBE",
    "DESC",
    "TABLES",
    "VIEWS",
    "INDEXES",
    "CASCADE",
    "RESTRICT",
    "INDEX",
    "COLUMNAR",
    "VIEW",
    "TRIGGER",
    "PROCEDURE",
    "FUNCTION",
    "RETURNING",
    "OVER",
    "PARTITION",
    "RANGE",
    "ROWS",
    "WINDOW",
    "UNBOUNDED",
    "BEGIN",
    "TRANSACTION",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "PRECEDING",
    "FOLLOWING",
    "CURRENT",
    "ROW",
    "MODIFY",
    "RENAME",
    "TO",
    "VARCHAR",
    "CHAR",
    "STRING",
    "BIGINT",
    "TINYINT",
    "SMALLINT",
    "REAL",
    "DOUBLE",
    "DECIMAL",
    "NUMERIC",
    "INT",
    "ISOLATION",
    "LEVEL",
    "READ",
    "COMMITTED",
    "UNCOMMITTED",
    "INTERVAL",
    "RECURSIVE",
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "NULLS",
    "FIRST",
    "LAST",
    "TRUNCATE",
    "SOME",
    "FILTER",
    "RETURNING",
    "EXPLAIN",
    "ANALYZE",
    "FETCH",
    "NEXT",
    "ONLY",
];

/// Compiled keyword set for O(1) lookups
/// Uses FxHashSet for fast hashing of short strings
static KEYWORD_SET: LazyLock<FxHashSet<&'static str>> = LazyLock::new(|| {
    let mut set = FxHashSet::with_capacity_and_hasher(KEYWORDS.len(), Default::default());
    for kw in KEYWORDS {
        set.insert(*kw);
    }
    set
});

/// Check if a string is an SQL keyword (case-insensitive)
/// Uses a compiled HashSet for O(1) lookups instead of O(n) linear search
#[inline]
pub fn is_keyword(s: &str) -> bool {
    // Fast path: check if already uppercase and in set
    if KEYWORD_SET.contains(s) {
        return true;
    }
    // Slow path: uppercase and check (only for non-uppercase input)
    // Use a stack buffer for small strings to avoid allocation
    if s.len() <= 32 {
        let mut buf = [0u8; 32];
        let bytes = s.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            buf[i] = b.to_ascii_uppercase();
        }
        // SAFETY: We only uppercased ASCII bytes, result is valid UTF-8
        let upper = unsafe { std::str::from_utf8_unchecked(&buf[..s.len()]) };
        KEYWORD_SET.contains(upper)
    } else {
        // Very long identifiers (rare) - fall back to allocation
        let upper = s.to_uppercase();
        KEYWORD_SET.contains(upper.as_str())
    }
}

/// SQL operators
pub static OPERATORS: &[&str] = &[
    "=", ">", "<", ">=", "<=", "<>", "!=", "+", "-", "*", "/", "%",
    "||", // String concatenation
    "->", "->>", // JSON operators
    "#>", "#>>", // JSON path operators
    "@>", "<@", // JSON contains
    "?", "?|", "?&", // JSON exists
    "&", "|", "^", "~", "<<", ">>", // Bitwise operators
];

/// Compiled operator set for O(1) lookups
static OPERATOR_SET: LazyLock<FxHashSet<&'static str>> = LazyLock::new(|| {
    let mut set = FxHashSet::with_capacity_and_hasher(OPERATORS.len(), Default::default());
    for op in OPERATORS {
        set.insert(*op);
    }
    set
});

/// Check if a string is an SQL operator
#[inline]
pub fn is_operator(s: &str) -> bool {
    OPERATOR_SET.contains(s)
}

/// SQL punctuators
pub static PUNCTUATORS: &[char] = &[',', ';', '(', ')', '.', ':', '[', ']'];

/// Check if a character is an SQL punctuator
pub fn is_punctuator(c: char) -> bool {
    PUNCTUATORS.contains(&c)
}

/// Characters that can be part of an operator
pub fn is_operator_char(c: char) -> bool {
    // Don't include # as an operator character since it's used for comments
    matches!(
        c,
        '=' | '<'
            | '>'
            | '!'
            | '+'
            | '-'
            | '*'
            | '/'
            | '%'
            | '|'
            | '&'
            | '^'
            | '~'
            | '?'
            | '@'
            | ':'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_display() {
        let pos = Position::new(10, 2, 5);
        assert_eq!(pos.to_string(), "line 2, column 5");
    }

    #[test]
    fn test_token_type_display() {
        assert_eq!(TokenType::Keyword.to_string(), "KEYWORD");
        assert_eq!(TokenType::Identifier.to_string(), "IDENTIFIER");
        assert_eq!(TokenType::String.to_string(), "STRING");
        assert_eq!(TokenType::Eof.to_string(), "EOF");
    }

    #[test]
    fn test_token_creation() {
        let token = Token::new(TokenType::Keyword, "SELECT", Position::new(0, 1, 1));
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "SELECT");
        assert!(token.is_keyword("SELECT"));
        assert!(token.is_keyword("select"));
        assert!(!token.is_keyword("FROM"));
    }

    #[test]
    fn test_error_token() {
        let token = Token::error("unexpected character", "x", Position::new(5, 1, 6));
        assert!(token.is_error());
        assert_eq!(token.error, Some("unexpected character".to_string()));
    }

    #[test]
    fn test_eof_token() {
        let token = Token::eof(Position::new(100, 5, 10));
        assert!(token.is_eof());
        assert_eq!(token.literal, "");
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("SELECT"));
        assert!(is_keyword("select"));
        assert!(is_keyword("Select"));
        assert!(!is_keyword("SELEC"));
        assert!(!is_keyword("mycolumn"));
    }

    #[test]
    fn test_is_operator() {
        assert!(is_operator("="));
        assert!(is_operator(">="));
        assert!(is_operator("->"));
        assert!(is_operator("->>"));
        assert!(!is_operator("==="));
    }

    #[test]
    fn test_is_punctuator() {
        assert!(is_punctuator(','));
        assert!(is_punctuator(';'));
        assert!(is_punctuator('('));
        assert!(is_punctuator(')'));
        assert!(!is_punctuator('x'));
    }

    #[test]
    fn test_is_operator_char() {
        assert!(is_operator_char('='));
        assert!(is_operator_char('+'));
        assert!(is_operator_char('-'));
        assert!(is_operator_char('|'));
        assert!(!is_operator_char('a'));
        assert!(!is_operator_char('#')); // # is for comments, not operators
    }

    #[test]
    fn test_token_display() {
        let keyword = Token::new(TokenType::Keyword, "SELECT", Position::new(0, 1, 1));
        assert!(keyword.to_string().contains("KEYWORD: SELECT"));

        let string = Token::new(TokenType::String, "hello", Position::new(7, 1, 8));
        assert!(string.to_string().contains("STRING: 'hello'"));

        let error = Token::error("bad token", "x", Position::new(0, 1, 1));
        assert!(error.to_string().contains("ERROR: bad token"));
    }
}
