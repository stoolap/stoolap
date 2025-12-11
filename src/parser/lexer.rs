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

//! SQL Lexer (Tokenizer)
//!
//! This module provides the lexer for tokenizing SQL input strings.

use super::token::{
    is_keyword, is_operator, is_operator_char, is_punctuator, Position, Token, TokenType,
};

/// SQL Lexer for tokenizing input
pub struct Lexer {
    /// Input string
    input: Vec<char>,
    /// Current position in input (points to current char)
    position: usize,
    /// Current reading position in input (after current char)
    read_position: usize,
    /// Current character under examination
    ch: char,
    /// Current position tracking
    pos: Position,
    /// Last error encountered
    last_error: Option<String>,
}

impl Lexer {
    /// Create a new lexer for the given input
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let mut lexer = Self {
            input: chars,
            position: 0,
            read_position: 0,
            ch: '\0',
            pos: Position::new(0, 1, 1),
            last_error: None,
        };
        lexer.read_char();
        lexer
    }

    /// Read the next character
    fn read_char(&mut self) {
        // Update position before changing character
        if self.ch == '\n' {
            self.pos.line += 1;
            self.pos.column = 1;
        } else if self.ch != '\0' {
            self.pos.column += 1;
        }

        if self.read_position >= self.input.len() {
            self.ch = '\0'; // EOF
        } else {
            self.ch = self.input[self.read_position];
            self.position = self.read_position;
            self.read_position += 1;
        }

        self.pos.offset = self.position;
    }

    /// Peek at the next character without advancing
    fn peek_char(&self) -> char {
        if self.read_position >= self.input.len() {
            '\0'
        } else {
            self.input[self.read_position]
        }
    }

    /// Peek at a character N positions ahead without advancing
    fn peek_char_n(&self, n: usize) -> char {
        let pos = self.read_position + n - 1;
        if pos >= self.input.len() {
            '\0'
        } else {
            self.input[pos]
        }
    }

    /// Check if a character marks the start of a line comment after --
    /// Returns true only if the character is whitespace, newline, or end of input
    fn is_comment_start_after_dashes(&self) -> bool {
        let char_after_second_dash = self.peek_char_n(2);
        // -- followed by whitespace, newline, or EOF is a comment
        // -- followed by digit, letter, or underscore is double negation
        char_after_second_dash == '\0'
            || char_after_second_dash == ' '
            || char_after_second_dash == '\t'
            || char_after_second_dash == '\n'
            || char_after_second_dash == '\r'
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let pos = self.pos;

        match self.ch {
            '\0' => Token::eof(pos),

            // String literal (single quotes)
            '\'' => {
                let literal = self.read_string_literal();
                Token::new(TokenType::String, literal, pos)
            }

            // Double-quoted identifier
            '"' => {
                let literal = self.read_quoted_identifier('"');
                Token::new(TokenType::Identifier, literal, pos)
            }

            // Backtick-quoted identifier (MySQL style)
            '`' => {
                let literal = self.read_quoted_identifier('`');
                Token::new(TokenType::Identifier, literal, pos)
            }

            // Negative number (but not if we're looking at --digit which is double negation)
            // This is handled by falling through to the operator case for the first -,
            // then treating the second - as an operator, then reading the digit as a number.
            // We don't handle negative numbers at lexer level - parser handles unary minus.

            // Number literal
            c if c.is_ascii_digit() => {
                let literal = self.read_number();
                if literal.contains('.') || literal.contains('e') || literal.contains('E') {
                    Token::new(TokenType::Float, literal, pos)
                } else {
                    Token::new(TokenType::Integer, literal, pos)
                }
            }

            // Single line comment (#)
            '#' => {
                let literal = self.read_line_comment();
                Token::new(TokenType::Comment, literal, pos)
            }

            // Single line comment (--)
            // Only treat as comment if followed by whitespace/newline/EOF
            // Otherwise, treat as two minus operators (for double negation like --val)
            '-' if self.peek_char() == '-' && self.is_comment_start_after_dashes() => {
                let literal = self.read_line_comment();
                Token::new(TokenType::Comment, literal, pos)
            }

            // Multi-line comment
            '/' if self.peek_char() == '*' => {
                let literal = self.read_block_comment();
                Token::new(TokenType::Comment, literal, pos)
            }

            // Parameter ($1, $2, etc.)
            '$' if self.peek_char().is_ascii_digit() => {
                let literal = self.read_parameter();
                Token::new(TokenType::Parameter, literal, pos)
            }

            // Parameter (?)
            '?' => {
                self.read_char();
                Token::new(TokenType::Parameter, "?", pos)
            }

            // Named parameter (:name)
            ':' if self.peek_char().is_alphabetic() || self.peek_char() == '_' => {
                let literal = self.read_named_parameter();
                Token::new(TokenType::Parameter, literal, pos)
            }

            // Star is always an operator (SELECT * handled by parser)
            '*' => {
                self.read_char();
                Token::new(TokenType::Operator, "*", pos)
            }

            // Regular punctuator
            c if is_punctuator(c) => {
                self.read_char();
                Token::new(TokenType::Punctuator, c.to_string(), pos)
            }

            // Operator
            c if is_operator_char(c) => {
                let literal = self.read_operator();
                Token::new(TokenType::Operator, literal, pos)
            }

            // Identifier or keyword
            c if c.is_alphabetic() || c == '_' => {
                let literal = self.read_identifier();
                if is_keyword(&literal) {
                    Token::new(TokenType::Keyword, literal.to_uppercase(), pos)
                } else {
                    Token::new(TokenType::Identifier, literal, pos)
                }
            }

            // Unrecognized character
            c => {
                self.read_char();
                Token::error(
                    format!("unrecognized character: {:?}", c),
                    c.to_string(),
                    pos,
                )
            }
        }
    }

    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while self.ch.is_whitespace() {
            // Note: read_char() handles line/column tracking when it encounters '\n'
            // So we don't need to update pos.line here
            if self.ch == '\r' && self.peek_char() == '\n' {
                // Skip \r in \r\n sequences (Windows line endings)
                self.read_char(); // consume '\r'
            }
            self.read_char();
        }
    }

    /// Read an identifier
    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        result.push(self.ch);
        self.read_char();

        while self.ch.is_alphanumeric() || self.ch == '_' || self.ch == '$' {
            result.push(self.ch);
            self.read_char();
        }

        result
    }

    /// Read a number (integer or float)
    fn read_number(&mut self) -> String {
        let mut result = String::new();
        result.push(self.ch);
        self.read_char();

        // Read all digits before decimal point
        while self.ch.is_ascii_digit() {
            result.push(self.ch);
            self.read_char();
        }

        // Check for decimal point
        if self.ch == '.' && self.peek_char().is_ascii_digit() {
            result.push(self.ch);
            self.read_char();

            // Read all digits after decimal point
            while self.ch.is_ascii_digit() {
                result.push(self.ch);
                self.read_char();
            }
        }

        // Check for exponent (E or e)
        if self.ch == 'e' || self.ch == 'E' {
            result.push(self.ch);
            self.read_char();

            // Check for sign after exponent
            if self.ch == '+' || self.ch == '-' {
                result.push(self.ch);
                self.read_char();
            }

            // Must have at least one digit after exponent
            if !self.ch.is_ascii_digit() {
                self.last_error = Some("invalid number format: exponent has no digits".to_string());
                return result;
            }

            // Read all digits in exponent
            while self.ch.is_ascii_digit() {
                result.push(self.ch);
                self.read_char();
            }
        }

        result
    }

    /// Read a string literal (single-quoted)
    fn read_string_literal(&mut self) -> String {
        let mut result = String::new();
        let quote = self.ch;
        result.push(quote);
        self.read_char(); // consume opening quote

        loop {
            if self.ch == '\0' {
                self.last_error = Some("unterminated string literal".to_string());
                result.push(quote); // Add closing quote for consistency
                break;
            } else if self.ch == quote {
                // Check for SQL-style escape (doubled quote)
                if self.peek_char() == quote {
                    // It's an escaped quote - SQL standard: '' becomes '
                    result.push(self.ch); // Add single quote (not both)
                    self.read_char(); // consume first quote
                    self.read_char(); // consume second quote
                } else {
                    // End of string
                    result.push(quote);
                    self.read_char();
                    break;
                }
            } else if self.ch == '\\' {
                // Handle backslash escape sequences
                result.push(self.ch);
                self.read_char();
                if self.ch != '\0' {
                    result.push(self.ch);
                    self.read_char();
                }
            } else {
                result.push(self.ch);
                self.read_char();
            }
        }

        result
    }

    /// Read a quoted identifier (double quotes or backticks)
    fn read_quoted_identifier(&mut self, quote: char) -> String {
        let mut result = String::new();
        self.read_char(); // consume opening quote

        while self.ch != '\0' {
            // Handle doubled quotes as escape (e.g., "abc""def" -> abc"def)
            if self.ch == quote && self.peek_char() == quote {
                result.push(self.ch);
                self.read_char(); // consume first quote
                self.read_char(); // consume second quote
            } else if self.ch == quote {
                // Found closing quote
                break;
            } else {
                result.push(self.ch);
                self.read_char();
            }
        }

        // Consume closing quote if not EOF
        if self.ch == quote {
            self.read_char();
        } else {
            self.last_error = Some(format!(
                "unterminated quoted identifier starting with {}",
                quote
            ));
        }

        result
    }

    /// Read a single-line comment (-- or #)
    fn read_line_comment(&mut self) -> String {
        let mut result = String::new();
        result.push(self.ch);

        // Skip the start of comment (-- or #)
        if self.ch == '-' && self.peek_char() == '-' {
            self.read_char(); // first -
            result.push(self.ch); // second -
            self.read_char(); // move past second -
        } else if self.ch == '#' {
            self.read_char(); // move past #
        }

        // Read until end of line or EOF
        while self.ch != '\n' && self.ch != '\0' {
            result.push(self.ch);
            self.read_char();
        }

        result
    }

    /// Read a block comment (/* ... */)
    fn read_block_comment(&mut self) -> String {
        let mut result = String::new();

        // Start with the opening /* sequence
        result.push(self.ch); // /
        self.read_char();
        result.push(self.ch); // *
        self.read_char();

        // Read until */ or EOF
        while !(self.ch == '*' && self.peek_char() == '/') && self.ch != '\0' {
            result.push(self.ch);
            self.read_char();
        }

        // Handle closing */
        if self.ch != '\0' {
            result.push(self.ch); // *
            self.read_char();
            result.push(self.ch); // /
            self.read_char();
        } else {
            self.last_error = Some("unterminated block comment".to_string());
        }

        result
    }

    /// Read an operator
    fn read_operator(&mut self) -> String {
        let mut result = String::new();
        let first_char = self.ch;
        result.push(first_char);
        self.read_char();

        // Check for multi-character operators
        if self.ch != '\0' {
            let two_chars: String = [first_char, self.ch].iter().collect();
            if is_operator(&two_chars) {
                result.push(self.ch);
                self.read_char();

                // Check for three-character operators
                if self.ch != '\0' {
                    let three_chars = format!("{}{}", two_chars, self.ch);
                    if is_operator(&three_chars) {
                        result.push(self.ch);
                        self.read_char();
                    }
                }
            }
        }

        result
    }

    /// Read a parameter ($1, $2, etc.)
    fn read_parameter(&mut self) -> String {
        let mut result = String::new();
        result.push(self.ch); // $
        self.read_char();

        // Read all digits
        while self.ch.is_ascii_digit() {
            result.push(self.ch);
            self.read_char();
        }

        // Validate parameter has digits
        if result.len() == 1 {
            self.last_error = Some("parameter number expected after $".to_string());
        }

        result
    }

    /// Read a named parameter (:name)
    fn read_named_parameter(&mut self) -> String {
        let mut result = String::new();
        result.push(self.ch); // :
        self.read_char();

        // Read identifier part (alphanumeric + underscore)
        while self.ch.is_alphanumeric() || self.ch == '_' {
            result.push(self.ch);
            self.read_char();
        }

        result
    }

    /// Get the last error encountered
    pub fn get_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Peek at the next token without advancing
    pub fn peek_token(&mut self) -> Token {
        // Save current state
        let saved_position = self.position;
        let saved_read_position = self.read_position;
        let saved_ch = self.ch;
        let saved_pos = self.pos;

        // Get the next token
        let token = self.next_token();

        // Restore state
        self.position = saved_position;
        self.read_position = saved_read_position;
        self.ch = saved_ch;
        self.pos = saved_pos;

        token
    }

    /// Peek at the next n tokens without advancing
    pub fn peek_tokens(&mut self, n: usize) -> Vec<Token> {
        if n == 0 {
            return Vec::new();
        }

        // Save current state
        let saved_position = self.position;
        let saved_read_position = self.read_position;
        let saved_ch = self.ch;
        let saved_pos = self.pos;

        // Get the next n tokens
        let mut tokens = Vec::with_capacity(n);
        for _ in 0..n {
            let token = self.next_token();
            if token.is_eof() {
                tokens.push(token);
                break;
            }
            tokens.push(token);
        }

        // Restore state
        self.position = saved_position;
        self.read_position = saved_read_position;
        self.ch = saved_ch;
        self.pos = saved_pos;

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let mut lexer = Lexer::new("SELECT * FROM users");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "SELECT");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Operator);
        assert_eq!(token.literal, "*");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "FROM");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Identifier);
        assert_eq!(token.literal, "users");

        let token = lexer.next_token();
        assert!(token.is_eof());
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("123 45.67 -89 3.14e10 1.5E-3");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Integer);
        assert_eq!(token.literal, "123");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Float);
        assert_eq!(token.literal, "45.67");

        // Note: Negative numbers are now tokenized as operator + number
        // (to support double negation like --val)
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Operator);
        assert_eq!(token.literal, "-");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Integer);
        assert_eq!(token.literal, "89");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Float);
        assert_eq!(token.literal, "3.14e10");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Float);
        assert_eq!(token.literal, "1.5E-3");
    }

    #[test]
    fn test_string_literals() {
        let mut lexer = Lexer::new("'hello' 'world''s' 'escaped\\ntext'");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::String);
        assert_eq!(token.literal, "'hello'");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::String);
        assert_eq!(token.literal, "'world's'");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::String);
        assert_eq!(token.literal, "'escaped\\ntext'");
    }

    #[test]
    fn test_quoted_identifiers() {
        let mut lexer = Lexer::new("\"table name\" `column`");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Identifier);
        assert_eq!(token.literal, "table name");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Identifier);
        assert_eq!(token.literal, "column");
    }

    #[test]
    fn test_operators() {
        let mut lexer = Lexer::new("= <> >= <= != + - * / || -> ->>");

        let expected = vec![
            "=", "<>", ">=", "<=", "!=", "+", "-", "*", "/", "||", "->", "->>",
        ];

        for exp in expected {
            let token = lexer.next_token();
            assert_eq!(token.token_type, TokenType::Operator);
            assert_eq!(token.literal, exp);
        }
    }

    #[test]
    fn test_punctuators() {
        let mut lexer = Lexer::new("( ) , ; . [ ]");

        let expected = vec!["(", ")", ",", ";", ".", "[", "]"];

        for exp in expected {
            let token = lexer.next_token();
            assert_eq!(token.token_type, TokenType::Punctuator);
            assert_eq!(token.literal, exp);
        }
    }

    #[test]
    fn test_comments() {
        let mut lexer = Lexer::new("-- line comment\nSELECT /* block */ 1");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);
        assert!(token.literal.contains("line comment"));

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "SELECT");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);
        assert!(token.literal.contains("block"));

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Integer);
        assert_eq!(token.literal, "1");
    }

    #[test]
    fn test_double_negation() {
        // --5 should tokenize as two minus operators followed by integer 5
        let mut lexer = Lexer::new("SELECT --5");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "SELECT");

        let token = lexer.next_token();
        assert_eq!(
            token.token_type,
            TokenType::Operator,
            "Expected Operator, got {:?} with literal '{}'",
            token.token_type,
            token.literal
        );
        assert_eq!(token.literal, "-");

        let token = lexer.next_token();
        assert_eq!(
            token.token_type,
            TokenType::Operator,
            "Expected Operator, got {:?} with literal '{}'",
            token.token_type,
            token.literal
        );
        assert_eq!(token.literal, "-");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Integer);
        assert_eq!(token.literal, "5");

        // --val should tokenize as two minus operators followed by identifier val
        let mut lexer = Lexer::new("SELECT --val");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "SELECT");

        let token = lexer.next_token();
        assert_eq!(
            token.token_type,
            TokenType::Operator,
            "Expected Operator for first -, got {:?} with literal '{}'",
            token.token_type,
            token.literal
        );
        assert_eq!(token.literal, "-");

        let token = lexer.next_token();
        assert_eq!(
            token.token_type,
            TokenType::Operator,
            "Expected Operator for second -, got {:?} with literal '{}'",
            token.token_type,
            token.literal
        );
        assert_eq!(token.literal, "-");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Identifier);
        assert_eq!(token.literal, "val");

        // -- with space should still be comment
        let mut lexer = Lexer::new("SELECT -- comment");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);
    }

    #[test]
    fn test_parameters() {
        let mut lexer = Lexer::new("$1 $23 ? :name :user_id :_private");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Parameter);
        assert_eq!(token.literal, "$1");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Parameter);
        assert_eq!(token.literal, "$23");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Parameter);
        assert_eq!(token.literal, "?");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Parameter);
        assert_eq!(token.literal, ":name");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Parameter);
        assert_eq!(token.literal, ":user_id");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Parameter);
        assert_eq!(token.literal, ":_private");
    }

    #[test]
    fn test_keywords_case_insensitive() {
        let mut lexer = Lexer::new("select SELECT Select");

        for _ in 0..3 {
            let token = lexer.next_token();
            assert_eq!(token.token_type, TokenType::Keyword);
            assert_eq!(token.literal, "SELECT");
        }
    }

    #[test]
    fn test_position_tracking() {
        let mut lexer = Lexer::new("SELECT\nFROM");

        let token = lexer.next_token();
        assert_eq!(token.position.line, 1);
        assert_eq!(token.position.column, 1);

        let token = lexer.next_token();
        assert_eq!(token.position.line, 2);
        assert_eq!(token.position.column, 1);
    }

    #[test]
    fn test_peek_token() {
        let mut lexer = Lexer::new("SELECT FROM");

        let peek1 = lexer.peek_token();
        assert_eq!(peek1.literal, "SELECT");

        let peek2 = lexer.peek_token();
        assert_eq!(peek2.literal, "SELECT"); // Same token

        let actual = lexer.next_token();
        assert_eq!(actual.literal, "SELECT");

        let next = lexer.next_token();
        assert_eq!(next.literal, "FROM");
    }

    #[test]
    fn test_peek_tokens() {
        let mut lexer = Lexer::new("SELECT * FROM users");

        let peeked = lexer.peek_tokens(3);
        assert_eq!(peeked.len(), 3);
        assert_eq!(peeked[0].literal, "SELECT");
        assert_eq!(peeked[1].literal, "*");
        assert_eq!(peeked[2].literal, "FROM");

        // Position should not have changed
        let actual = lexer.next_token();
        assert_eq!(actual.literal, "SELECT");
    }

    #[test]
    fn test_complex_query() {
        let query = r#"
            SELECT u.id, u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.active = TRUE AND o.amount >= 100.50
            GROUP BY u.id, u.name
            HAVING COUNT(o.id) > 0
            ORDER BY order_count DESC
            LIMIT 10
        "#;

        let mut lexer = Lexer::new(query);
        let mut tokens = Vec::new();

        loop {
            let token = lexer.next_token();
            if token.is_eof() {
                break;
            }
            tokens.push(token);
        }

        // Verify we got reasonable tokens
        assert!(tokens.len() > 30);
        assert!(tokens.iter().any(|t| t.is_keyword("SELECT")));
        assert!(tokens.iter().any(|t| t.is_keyword("FROM")));
        assert!(tokens.iter().any(|t| t.is_keyword("JOIN")));
        assert!(tokens.iter().any(|t| t.is_keyword("WHERE")));
        assert!(tokens.iter().any(|t| t.is_keyword("GROUP")));
        assert!(tokens.iter().any(|t| t.is_keyword("HAVING")));
        assert!(tokens.iter().any(|t| t.is_keyword("ORDER")));
        assert!(tokens.iter().any(|t| t.is_keyword("LIMIT")));
    }

    #[test]
    fn test_error_token() {
        let mut lexer = Lexer::new("SELECT © FROM");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Error);
        assert!(token.error.is_some());
    }
}
