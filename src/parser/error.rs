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
}

impl ParseError {
    /// Create a new parse error
    pub fn new(message: impl Into<String>, position: Position) -> Self {
        Self {
            message: message.into(),
            position,
        }
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
}

impl ParseErrors {
    /// Create from a vector of errors
    pub fn from_errors(errors: Vec<ParseError>) -> Self {
        Self { errors }
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
}
