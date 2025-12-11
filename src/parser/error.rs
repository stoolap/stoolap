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

//! Parser error types
//!
//! This module provides error types for SQL parsing.

use super::token::Position;
use std::fmt;

/// A single parse error
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// Error message
    pub message: String,
    /// Position in source
    pub position: Position,
    /// SQL context where error occurred
    pub context: String,
}

impl ParseError {
    /// Create a new parse error
    pub fn new(message: impl Into<String>, position: Position) -> Self {
        Self {
            message: message.into(),
            position,
            context: String::new(),
        }
    }

    /// Create a parse error with context
    pub fn with_context(
        message: impl Into<String>,
        position: Position,
        context: impl Into<String>,
    ) -> Self {
        Self {
            message: message.into(),
            position,
            context: context.into(),
        }
    }

    /// Format the error with context for display
    pub fn format_error(&self) -> String {
        if self.context.is_empty() {
            return self.to_string();
        }

        let lines: Vec<&str> = self.context.lines().collect();
        if self.position.line == 0 || self.position.line > lines.len() {
            return self.to_string();
        }

        let line = lines[self.position.line - 1];
        let pointer = " ".repeat(self.position.column.saturating_sub(1)) + "^";

        format!("{}\n{}\n{}", self, line, pointer)
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at position {}", self.message, self.position)
    }
}

impl std::error::Error for ParseError {}

/// Collection of parse errors
#[derive(Debug, Clone)]
pub struct ParseErrors {
    /// List of errors
    pub errors: Vec<ParseError>,
    /// Original SQL string
    pub sql: String,
}

impl ParseErrors {
    /// Create a new empty error collection
    pub fn new(sql: impl Into<String>) -> Self {
        Self {
            errors: Vec::new(),
            sql: sql.into(),
        }
    }

    /// Create from a vector of errors
    pub fn from_errors(errors: Vec<ParseError>) -> Self {
        Self {
            errors,
            sql: String::new(),
        }
    }

    /// Add an error
    pub fn push(&mut self, error: ParseError) {
        self.errors.push(error);
    }

    /// Check if there are any errors
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the number of errors
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Format all errors for display
    pub fn format_errors(&self) -> String {
        if self.errors.is_empty() {
            return String::new();
        }

        let mut result = format!(
            "SQL parsing failed with {} error(s):\n\n",
            self.errors.len()
        );

        for (i, err) in self.errors.iter().enumerate() {
            result.push_str(&format!("Error {}: {}\n", i + 1, err.message));

            // Add context from SQL
            let lines: Vec<&str> = self.sql.lines().collect();
            if err.position.line > 0 && err.position.line <= lines.len() {
                let line = lines[err.position.line - 1];
                result.push_str(&format!("Line {}: {}\n", err.position.line, line));
                let pointer = " ".repeat(err.position.column + 7); // +7 for "Line X: "
                result.push_str(&format!("{}^\n", pointer));
            }

            // Add suggestion
            if let Some(suggestion) = get_suggestion(err) {
                result.push_str(&format!("Suggestion: {}\n", suggestion));
            }

            result.push('\n');
        }

        result
    }
}

impl fmt::Display for ParseErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.errors.is_empty() {
            write!(f, "SQL parse error")
        } else {
            write!(f, "{}", self.errors[0])
        }
    }
}

impl std::error::Error for ParseErrors {}

/// Get a helpful suggestion for a parse error
fn get_suggestion(err: &ParseError) -> Option<String> {
    let msg = &err.message;
    let ctx = &err.context;

    // Expected token errors
    if msg.contains("expected table name or subquery") {
        return Some("You might be missing a column or table name, or using a reserved keyword without proper quoting. Try enclosing names in double quotes if they're reserved words.".to_string());
    }

    if ctx.contains("SELET") {
        return Some("Did you mean 'SELECT'?".to_string());
    }

    if msg.contains("expected ')' or ','") {
        return Some("You're missing a closing parenthesis. Make sure all opening parentheses are matched with closing ones.".to_string());
    }

    if msg.contains("expected next token to be PUNCTUATOR") {
        return Some("A punctuation character like '(', ')', ',', ';' is expected here. Check for missing parentheses or commas in lists.".to_string());
    }

    if ctx.contains("LEFTJOIN") {
        return Some(
            "Did you mean 'LEFT JOIN'? LEFT JOIN needs a space between the words.".to_string(),
        );
    }

    if msg.contains("expected next token to be IDENTIFIER") {
        return Some("You might be missing a column or table name, or using a reserved keyword without proper quoting.".to_string());
    }

    if msg.contains("expected next token to be KEYWORD") {
        return Some(
            "A SQL keyword (like SELECT, FROM, WHERE, GROUP BY, etc.) is expected here."
                .to_string(),
        );
    }

    if msg.contains("expected next token to be OPERATOR") {
        return Some("An operator such as =, <, >, <=, >=, <>, != is expected here.".to_string());
    }

    if msg.contains("expected next token to be NUMBER") {
        return Some("A numeric value is expected here. Make sure you're providing a valid number without quotes.".to_string());
    }

    if msg.contains("expected next token to be STRING") {
        return Some(
            "A string value is expected here. String literals should be enclosed in single quotes."
                .to_string(),
        );
    }

    // Unexpected token errors
    if msg.contains("unexpected token OPERATOR") {
        return Some("You have an unexpected operator here. Check if you're missing a value or have an extra operator.".to_string());
    }

    if msg.contains("unexpected token PUNCTUATOR") {
        return Some("There's an unexpected punctuation character here. Check for mismatched parentheses or extra commas.".to_string());
    }

    if msg.contains("unexpected token EOF") {
        return Some("Your SQL statement is incomplete. You might be missing a closing parenthesis, quote, or the end of a clause.".to_string());
    }

    // Common typos
    if msg.contains("SELET") || ctx.contains("SELET") {
        return Some("Did you mean 'SELECT'?".to_string());
    }

    if msg.contains("UPDAT") || ctx.contains("UPDAT") {
        return Some("Did you mean 'UPDATE'?".to_string());
    }

    if msg.contains("DELET") || ctx.contains("DELET") {
        return Some("Did you mean 'DELETE'?".to_string());
    }

    if msg.contains("GROUPBY") || ctx.contains("GROUPBY") {
        return Some(
            "Did you mean 'GROUP BY'? GROUP BY needs a space between the words.".to_string(),
        );
    }

    if msg.contains("ORDERBY") || ctx.contains("ORDERBY") {
        return Some(
            "Did you mean 'ORDER BY'? ORDER BY needs a space between the words.".to_string(),
        );
    }

    // JOIN issues
    if ctx.contains("JOIN") && !ctx.contains("ON") {
        return Some(
            "Your JOIN clause is missing the ON condition that specifies how tables are related."
                .to_string(),
        );
    }

    // Missing parentheses
    if msg.contains("missing ')'") {
        return Some("You're missing a closing parenthesis.".to_string());
    }

    if msg.contains("missing '('") {
        return Some("You're missing an opening parenthesis.".to_string());
    }

    // Default suggestion
    Some("Check syntax near this location. Common issues include missing keywords, misplaced clauses, unclosed parentheses, or incorrect identifiers.".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::new("unexpected token", Position::new(10, 1, 11));
        assert_eq!(
            err.to_string(),
            "unexpected token at position line 1, column 11"
        );
    }

    #[test]
    fn test_parse_error_with_context() {
        let err = ParseError::with_context(
            "unexpected token",
            Position::new(7, 1, 8),
            "SELECT * FORM users",
        );
        let formatted = err.format_error();
        assert!(formatted.contains("SELECT * FORM users"));
        assert!(formatted.contains("^"));
    }

    #[test]
    fn test_parse_errors_collection() {
        let mut errors = ParseErrors::new("SELECT SELET FROM");
        assert!(errors.is_empty());

        errors.push(ParseError::new("unexpected token", Position::new(7, 1, 8)));
        assert_eq!(errors.len(), 1);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_suggestion_for_typo() {
        let err = ParseError::with_context(
            "unexpected identifier",
            Position::new(0, 1, 1),
            "SELET * FROM users",
        );
        let suggestion = get_suggestion(&err);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("SELECT"));
    }

    #[test]
    fn test_suggestion_for_missing_identifier() {
        let err = ParseError::new(
            "expected next token to be IDENTIFIER",
            Position::new(0, 1, 1),
        );
        let suggestion = get_suggestion(&err);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("column or table name"));
    }
}
