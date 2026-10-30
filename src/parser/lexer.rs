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
    is_operator_char, lookup_keyword, operator_len, punctuator_str, Position, Token, TokenType,
};
use crate::common::SmartString;

/// SQL Lexer for tokenizing input
///
/// Reads the input as bytes: every byte SQL gives a meaning is ASCII, so a
/// character is decoded only where a non-ASCII byte is met
pub struct Lexer<'a> {
    /// Input string as bytes, borrowed from the caller
    input: &'a [u8],
    /// The byte the next token or character starts at
    at: usize,
    /// The line `at` is on, starting at 1
    line: u32,
    /// The byte the line starts at
    line_start: usize,
    /// The bytes between `line_start` and `at` that continue a character
    continuing: usize,
    /// Last error encountered
    last_error: Option<String>,
}

/// A byte of an identifier: an ASCII letter, digit, `_` or `$`
const WORD: u8 = 1;
/// An ASCII digit
const DIGIT: u8 = 2;

/// The classes of each byte
static BYTE_CLASS: [u8; 256] = {
    let mut classes = [0; 256];
    let mut b = 0;
    while b < 256 {
        let byte = b as u8;
        if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$' {
            classes[b] |= WORD;
        }
        if byte.is_ascii_digit() {
            classes[b] |= DIGIT;
        }
        b += 1;
    }
    classes
};

/// Lexer state saved for a peek
#[derive(Clone, Copy)]
struct Mark {
    at: usize,
    line: u32,
    line_start: usize,
    continuing: usize,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given input
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            at: 0,
            line: 1,
            line_start: 0,
            continuing: 0,
            last_error: None,
        }
    }

    /// The byte at `i`, NUL past the end
    #[inline]
    fn byte(&self, i: usize) -> u8 {
        self.input.get(i).copied().unwrap_or(0)
    }

    /// The character at byte `i` and its length, NUL past the end; read
    /// only where a byte is not ASCII, so kept out of the hot paths
    #[inline(never)]
    fn char_at(&self, i: usize) -> (char, usize) {
        match self.input.get(i) {
            None => ('\0', 0),
            Some(&b) if b < 0x80 => (char::from(b), 1),
            Some(_) => {
                let rest = self.text(i, self.input.len());
                rest.chars().next().map_or(('\0', 0), |c| (c, c.len_utf8()))
            }
        }
    }

    /// The position of the byte `at`
    #[inline]
    fn position(&self) -> Position {
        let column = self.at - self.line_start - self.continuing + 1;
        Position::new(
            u32::try_from(self.at).unwrap_or(u32::MAX),
            self.line,
            u32::try_from(column).unwrap_or(u32::MAX),
        )
    }

    /// Step past `len` bytes of one character that is not a newline
    #[inline]
    fn step(&mut self, len: usize) {
        self.at += len;
        self.continuing += len - 1;
    }

    /// Step past one byte of any kind, counting a newline or a byte that
    /// continues a character
    #[inline]
    fn step_byte(&mut self) {
        match self.input[self.at] {
            b'\n' => {
                self.line = self.line.saturating_add(1);
                self.line_start = self.at + 1;
                self.continuing = 0;
            }
            b if b & 0xC0 == 0x80 => self.continuing += 1,
            _ => {}
        }
        self.at += 1;
    }

    /// Step to byte `end` past every byte before it, see [`Self::step_byte`]
    fn step_to(&mut self, end: usize) {
        while self.at < end {
            self.step_byte();
        }
    }

    fn mark(&self) -> Mark {
        Mark {
            at: self.at,
            line: self.line,
            line_start: self.line_start,
            continuing: self.continuing,
        }
    }

    fn reset(&mut self, mark: Mark) {
        (self.at, self.line, self.line_start, self.continuing) =
            (mark.at, mark.line, mark.line_start, mark.continuing);
    }

    /// The byte offset just past the last token read
    pub(crate) fn offset(&self) -> usize {
        self.at
    }

    /// The token text from byte `start` to `at`, copied once
    fn token_since(&self, start: usize) -> SmartString {
        SmartString::new(self.text(start, self.at))
    }

    /// The input from byte `start` to byte `end`, both the first byte of a char
    fn text(&self, start: usize, end: usize) -> &'a str {
        // SAFETY: `input` is the byte form of a `&str` and both bounds are
        // the first byte of a decoded char, so both are char boundaries.
        unsafe { std::str::from_utf8_unchecked(&self.input[start..end]) }
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let pos = self.position();
        let next = self.byte(self.at + 1);

        match self.byte(self.at) {
            // The end, or a NUL, which ends the input wherever it stands
            0 => Token::eof(pos),

            // String literal (single quotes)
            b'\'' => {
                let literal = self.read_string_literal();
                if let Some(err) = self.last_error.take() {
                    return Token::error(err, "", pos);
                }
                Token::new(TokenType::String, literal, pos)
            }

            // Double-quoted identifier (identifier with string fallback)
            b'"' => {
                let literal = self.read_quoted_identifier(b'"');
                if let Some(err) = self.last_error.take() {
                    return Token::error(err, "", pos);
                }
                Token::new_quoted(TokenType::Identifier, literal, pos)
            }

            // Backtick-quoted identifier (MySQL style)
            b'`' => {
                let literal = self.read_quoted_identifier(b'`');
                if let Some(err) = self.last_error.take() {
                    return Token::error(err, "", pos);
                }
                Token::new(TokenType::Identifier, literal, pos)
            }

            // Negative numbers: parser handles unary minus, not the lexer.

            // Number literal
            b'0'..=b'9' => match self.read_number() {
                (literal, true) => Token::new(TokenType::Float, literal, pos),
                (literal, false) => Token::new(TokenType::Integer, literal, pos),
            },

            // Single line comment (#)
            b'#' => {
                let literal = self.read_line_comment(1);
                Token::new(TokenType::Comment, literal, pos)
            }

            // Single line comment (--) per SQL standard (SQL:2023 section 5.2)
            // Double negation should be written as `- -val` or `- (-val)`
            b'-' if next == b'-' => {
                let literal = self.read_line_comment(2);
                Token::new(TokenType::Comment, literal, pos)
            }

            // Multi-line comment
            b'/' if next == b'*' => {
                let literal = self.read_block_comment();
                Token::new(TokenType::Comment, literal, pos)
            }

            // Parameter ($1, $2, etc.)
            b'$' if next.is_ascii_digit() => {
                let literal = self.read_parameter();
                Token::new(TokenType::Parameter, literal, pos)
            }

            // Parameter (?)
            b'?' => {
                self.at += 1;
                Token::new(TokenType::Parameter, "?", pos)
            }

            // Named parameter (:name)
            b':' if {
                let (c, _) = self.char_at(self.at + 1);
                c.is_alphabetic() || c == '_'
            } =>
            {
                let literal = self.read_named_parameter();
                Token::new(TokenType::Parameter, literal, pos)
            }

            // Star is always an operator (SELECT * handled by parser)
            b'*' => {
                self.at += 1;
                Token::new(TokenType::Operator, "*", pos)
            }

            b if b < 0x80 => {
                let c = char::from(b);
                // Regular punctuator - use static string to avoid allocation
                if let Some(punctuator) = punctuator_str(c) {
                    self.at += 1;
                    return Token::new(TokenType::Punctuator, punctuator, pos);
                }
                if is_operator_char(c) {
                    let start = self.at;
                    self.at += operator_len(&self.input[start..]).max(1);
                    return Token::new(TokenType::Operator, self.text(start, self.at), pos);
                }
                if c.is_ascii_alphabetic() || c == '_' {
                    return self.word(pos);
                }
                self.at += 1;
                Self::unrecognized(c, pos)
            }

            // A character of more than one byte: a letter starts a word
            _ => match self.char_at(self.at) {
                (c, _) if c.is_alphabetic() => self.word(pos),
                (c, len) => {
                    self.step(len);
                    Self::unrecognized(c, pos)
                }
            },
        }
    }

    /// The identifier or keyword starting at the current byte
    fn word(&mut self, pos: Position) -> Token {
        let word = self.read_identifier();
        match lookup_keyword(word) {
            Some(keyword) => Token::from_keyword(keyword, pos),
            None => Token::new(TokenType::Identifier, word, pos),
        }
    }

    #[inline(never)]
    fn unrecognized(c: char, pos: Position) -> Token {
        Token::error(
            format!("unrecognized character: {:?}", c),
            c.to_string(),
            pos,
        )
    }

    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        loop {
            match self.byte(self.at) {
                b'\n' => self.step_to(self.at + 1),
                b' ' | b'\t' | b'\r' | 0x0B | 0x0C => self.at += 1,
                b if b < 0x80 => return,
                _ => match self.char_at(self.at) {
                    (c, len) if c.is_whitespace() => self.step(len),
                    _ => return,
                },
            }
        }
    }

    /// Step past the bytes from `at` that `class` holds
    #[inline]
    fn skip_class(&mut self, class: u8) {
        let rest = &self.input[self.at..];
        self.at += rest
            .iter()
            .position(|&b| BYTE_CLASS[usize::from(b)] & class == 0)
            .unwrap_or(rest.len());
    }

    /// Read an identifier: letters, digits, `_` and `$`
    fn read_identifier(&mut self) -> &'a str {
        let start = self.at;
        loop {
            self.skip_class(WORD);
            if self.byte(self.at) < 0x80 {
                break;
            }
            match self.char_at(self.at) {
                (c, len) if c.is_alphanumeric() => self.step(len),
                _ => break,
            }
        }
        self.text(start, self.at)
    }

    /// Read a number, and whether it has a decimal point or an exponent
    fn read_number(&mut self) -> (SmartString, bool) {
        let start = self.at;
        self.skip_class(DIGIT);
        let mut float = false;

        // A decimal point followed by a digit
        if self.byte(self.at) == b'.' && self.byte(self.at + 1).is_ascii_digit() {
            self.at += 1;
            self.skip_class(DIGIT);
            float = true;
        }

        // Check for exponent (E or e)
        if matches!(self.byte(self.at), b'e' | b'E') {
            self.at += 1;
            float = true;

            // Check for sign after exponent
            if matches!(self.byte(self.at), b'+' | b'-') {
                self.at += 1;
            }

            // Must have at least one digit after exponent
            if !self.byte(self.at).is_ascii_digit() {
                self.last_error = Some("invalid number format: exponent has no digits".to_string());
                return (self.token_since(start), float);
            }
            self.skip_class(DIGIT);
        }

        (self.token_since(start), float)
    }

    /// Read a string literal (single-quoted)
    fn read_string_literal(&mut self) -> SmartString {
        let start = self.at;

        // Fast path: no doubled quote, backslash or NUL before the closing
        // quote, so the token is the input's own text, quotes included
        let mut end = start + 1;
        loop {
            match self.byte(end) {
                b'\'' if self.byte(end + 1) != b'\'' => {
                    self.step_to(end + 1);
                    return SmartString::new(self.text(start, end + 1));
                }
                b'\'' | b'\\' | 0 => return self.read_escaped_string(start, end),
                _ => end += 1,
            }
        }
    }

    /// The string literal from byte `start`, whose fast path stopped at byte
    /// `end`: escapes or edge cases, each run between them copied at once
    #[inline(never)]
    fn read_escaped_string(&mut self, start: usize, end: usize) -> SmartString {
        // Room for the text before the first escape and the quotes, or for a
        // short literal when the escape opens it
        let mut result = String::with_capacity(match end > start + 1 {
            true => end - start + 2,
            false => 32,
        });
        result.push('\'');
        self.step_to(start + 1);
        loop {
            let run = self.at;
            let mut end = run;
            while !matches!(self.byte(end), b'\'' | b'\\' | 0) {
                end += 1;
            }
            result.push_str(self.text(run, end));
            self.step_to(end);
            match self.byte(self.at) {
                0 => {
                    self.last_error = Some(match self.at >= self.input.len() {
                        true => "unterminated string literal".to_string(),
                        false => "NULL byte (0x00) is not allowed in string literals".to_string(),
                    });
                    result.push('\'');
                    break;
                }
                b'\'' if self.byte(self.at + 1) == b'\'' => {
                    result.push('\'');
                    self.at += 2;
                }
                b'\'' => {
                    result.push('\'');
                    self.at += 1;
                    break;
                }
                // A backslash keeps the character after it, a quote too
                _ => {
                    result.push('\\');
                    self.at += 1;
                    let (c, len) = self.char_at(self.at);
                    if c != '\0' {
                        result.push(c);
                        self.step_to(self.at + len);
                    }
                }
            }
        }

        SmartString::from_string(result)
    }

    /// Read a quoted identifier (double quotes or backticks)
    #[inline(never)]
    fn read_quoted_identifier(&mut self, quote: u8) -> SmartString {
        self.at += 1; // consume opening quote
        let start = self.at;
        // Only a doubled quote changes the text, so the body is sliced from the
        // input unless one turns up; then each run between them is copied once
        let mut unescaped: Option<String> = None;
        let mut run = start;

        loop {
            match self.byte(self.at) {
                0 => break,
                // Handle doubled quotes as escape (e.g., "abc""def" -> abc"def)
                b if b == quote && self.byte(self.at + 1) == quote => {
                    let buf = unescaped.get_or_insert_with(String::new);
                    buf.push_str(self.text(run, self.at));
                    buf.push(char::from(quote));
                    self.at += 2;
                    run = self.at;
                }
                b if b == quote => break,
                _ => self.step_byte(),
            }
        }

        let result = match unescaped {
            Some(mut buf) => {
                buf.push_str(self.text(run, self.at));
                SmartString::from_string(buf)
            }
            None => self.token_since(start),
        };

        // Consume closing quote if not EOF
        if self.byte(self.at) == quote {
            self.at += 1;
        } else if self.at >= self.input.len() {
            self.last_error = Some(format!(
                "unterminated quoted identifier starting with {}",
                char::from(quote)
            ));
        } else {
            // NULL byte encountered
            self.last_error =
                Some("NULL byte (0x00) is not allowed in quoted identifiers".to_string());
        }
        // It separates a qualifier from a name inside the engine
        if result.contains(crate::executor::utils::QUALIFIER_SEP) {
            self.last_error = Some(
                "the control character U+001F is not allowed in quoted identifiers".to_string(),
            );
        }

        result
    }

    /// Read a single-line comment (-- or #)
    /// Read a single-line comment (-- or #) whose start is `opening` bytes,
    /// up to the end of its line
    #[inline(never)]
    fn read_line_comment(&mut self, opening: usize) -> SmartString {
        let start = self.at;
        self.at += opening;
        while !matches!(self.byte(self.at), b'\n' | 0) {
            self.step_byte();
        }
        self.token_since(start)
    }

    /// Read a block comment (/* ... */)
    #[inline(never)]
    fn read_block_comment(&mut self) -> SmartString {
        let start = self.at;
        self.at += 2;
        loop {
            match self.byte(self.at) {
                0 => {
                    self.last_error = Some("unterminated block comment".to_string());
                    break;
                }
                b'*' if self.byte(self.at + 1) == b'/' => {
                    self.at += 2;
                    break;
                }
                _ => self.step_byte(),
            }
        }
        self.token_since(start)
    }

    /// Read a parameter ($1, $2, etc.)
    fn read_parameter(&mut self) -> SmartString {
        let start = self.at;
        self.at += 1;
        self.skip_class(DIGIT);
        self.token_since(start)
    }

    /// Read a named parameter (:name): letters, digits and `_` after the colon
    #[inline(never)]
    fn read_named_parameter(&mut self) -> SmartString {
        let start = self.at;
        self.at += 1;
        loop {
            match self.byte(self.at) {
                b if b.is_ascii_alphanumeric() || b == b'_' => self.at += 1,
                b if b < 0x80 => break,
                _ => match self.char_at(self.at) {
                    (c, len) if c.is_alphanumeric() => self.step(len),
                    _ => break,
                },
            }
        }
        self.token_since(start)
    }

    /// Get the last error encountered
    pub fn get_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Peek at the next token without advancing
    pub fn peek_token(&mut self) -> Token {
        let mark = self.mark();
        let token = self.next_token();
        self.reset(mark);
        token
    }

    /// Peek at the next n tokens without advancing
    pub fn peek_tokens(&mut self, n: usize) -> Vec<Token> {
        if n == 0 {
            return Vec::new();
        }

        let mark = self.mark();
        let mut tokens = Vec::with_capacity(n);
        for _ in 0..n {
            let token = self.next_token();
            if token.is_eof() {
                tokens.push(token);
                break;
            }
            tokens.push(token);
        }
        self.reset(mark);
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::token::Keyword;

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

        // Negative numbers are tokenized as operator + number (parser handles unary minus)
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
    fn test_double_dash_is_always_comment() {
        // Per SQL standard (SQL:2023 5.2), -- always starts a line comment
        // regardless of what follows the dashes.

        // --5 is a comment, not double negation
        let mut lexer = Lexer::new("SELECT --5");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "SELECT");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);

        // --val is a comment, not double negation
        let mut lexer = Lexer::new("SELECT --val");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);

        // --comment at start of input is a comment
        let mut lexer = Lexer::new("--comment\nSELECT 1");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "SELECT");

        // -- with space is still a comment
        let mut lexer = Lexer::new("SELECT -- comment");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);

        // Double negation must use `- -` with a space
        let mut lexer = Lexer::new("SELECT - -5");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Operator);
        assert_eq!(token.literal, "-");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Operator);
        assert_eq!(token.literal, "-");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Integer);
        assert_eq!(token.literal, "5");
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
    fn test_position_after_string_literal() {
        for (sql, line, column) in [
            ("'ab', x", 1, 7),
            ("'ğü' x", 1, 6),
            ("'a\nb' x", 2, 4),
            ("'a''b' x", 1, 8),
        ] {
            let mut lexer = Lexer::new(sql);
            let mut token = lexer.next_token();
            while token.literal != "x" {
                token = lexer.next_token();
            }
            assert_eq!(
                (token.position.line, token.position.column),
                (line, column),
                "position of x in {sql:?}"
            );
        }
    }

    #[test]
    fn test_position_after_identifier() {
        for (sql, line, column) in [
            ("ab x", 1, 4),
            ("SeLeCt x", 1, 8),
            ("tablo_adı x", 1, 11),
            ("çağ_ü x", 1, 7),
            ("a_1$b\ncd x", 2, 4),
        ] {
            let mut lexer = Lexer::new(sql);
            let mut token = lexer.next_token();
            while token.literal != "x" {
                token = lexer.next_token();
            }
            assert_eq!(
                (token.position.line, token.position.column),
                (line, column),
                "position of x in {sql:?}"
            );
        }
    }

    #[test]
    fn test_position_saturates_at_the_counter_limit() {
        for sql in ["ab", "a\nb", "'ab'", "'a\nb'"] {
            let mut lexer = Lexer::new(sql);
            lexer.line = u32::MAX;
            let mut token = lexer.next_token();
            while !token.is_eof() {
                token = lexer.next_token();
            }
            assert_eq!(token.position.line, u32::MAX, "line at the end of {sql:?}");
        }
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
        for keyword in [
            Keyword::Select,
            Keyword::From,
            Keyword::Join,
            Keyword::Where,
            Keyword::Group,
            Keyword::Having,
            Keyword::Order,
            Keyword::Limit,
        ] {
            assert!(tokens.iter().any(|t| t.is_keyword(keyword)), "{keyword:?}");
        }
    }

    /// The readers slice the input instead of pushing char by char, so the
    /// byte bounds have to land on character boundaries for multi-byte input
    /// and the literals have to come out identical for every token shape.
    #[test]
    fn test_multibyte_identifiers_are_sliced_whole() {
        let mut lexer = Lexer::new("SELECT çağ_ü, 日本語 FROM tablo_adı");

        assert_eq!(lexer.next_token().literal, "SELECT");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Identifier);
        assert_eq!(token.literal, "çağ_ü");
        assert_eq!(lexer.next_token().literal, ",");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Identifier);
        assert_eq!(token.literal, "日本語");
        assert_eq!(lexer.next_token().literal, "FROM");
        assert_eq!(lexer.next_token().literal, "tablo_adı");
        assert!(lexer.next_token().is_eof());
    }

    #[test]
    fn test_long_tokens_keep_their_full_text() {
        let ident = "a_column_name_well_past_the_inline_buffer";
        let comment = format!("/* {} */", "x".repeat(300));
        let sql = format!(
            "SELECT {} {} FROM t -- trailing until end of input",
            ident, comment
        );
        let mut lexer = Lexer::new(&sql);

        assert_eq!(lexer.next_token().literal, "SELECT");
        assert_eq!(lexer.next_token().literal, ident);
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);
        assert_eq!(token.literal, comment);
        assert_eq!(lexer.next_token().literal, "FROM");
        assert_eq!(lexer.next_token().literal, "t");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);
        assert_eq!(token.literal, "-- trailing until end of input");
        assert!(lexer.next_token().is_eof());
    }

    #[test]
    fn test_hash_comment_keeps_marker_and_stops_at_newline() {
        let mut lexer = Lexer::new("# note here\n42");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);
        assert_eq!(token.literal, "# note here");
        assert_eq!(lexer.next_token().literal, "42");
    }

    #[test]
    fn test_quoted_identifier_slices_unless_escaped() {
        let mut lexer = Lexer::new("\"plain name\" \"say \"\"hi\"\" now\" `tick``ed`");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Identifier);
        assert_eq!(token.literal, "plain name");
        let token = lexer.next_token();
        assert_eq!(token.literal, "say \"hi\" now");
        let token = lexer.next_token();
        assert_eq!(token.literal, "tick`ed");
        assert!(lexer.next_token().is_eof());
    }

    #[test]
    fn test_parameters_are_sliced() {
        let mut lexer = Lexer::new("$1 $12345 :name :long_named_parameter_x");
        assert_eq!(lexer.next_token().literal, "$1");
        assert_eq!(lexer.next_token().literal, "$12345");
        assert_eq!(lexer.next_token().literal, ":name");
        assert_eq!(lexer.next_token().literal, ":long_named_parameter_x");
    }

    /// A malformed number or comment keeps its token type and leaves the
    /// message in `get_error`; a bad parameter or quoted identifier becomes an
    /// Error token. Slicing must preserve the partial literal and the report.
    #[test]
    fn test_reader_error_paths_still_report() {
        // exponent without digits: the partial literal is kept
        let mut lexer = Lexer::new("1e+");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Float);
        assert_eq!(token.literal, "1e+");
        assert!(lexer.get_error().is_some());

        // a bare $ is a parameter without a number: an error token
        let mut lexer = Lexer::new("$ ");
        assert_eq!(lexer.next_token().token_type, TokenType::Error);

        // a block comment that never closes keeps everything it saw
        let mut lexer = Lexer::new("/* never closed");
        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Comment);
        assert_eq!(token.literal, "/* never closed");
        assert!(lexer.get_error().is_some());

        // a quoted identifier that never closes is an error token
        let mut lexer = Lexer::new("\"never closed");
        assert_eq!(lexer.next_token().token_type, TokenType::Error);
    }

    #[test]
    fn test_error_token() {
        let mut lexer = Lexer::new("SELECT © FROM");

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Keyword);

        let token = lexer.next_token();
        assert_eq!(token.token_type, TokenType::Error);
        // Error message is stored in literal field
        assert!(!token.literal.is_empty());
    }
}
