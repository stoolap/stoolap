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

//! String Quotes Tests
//!
//! Tests proper handling of single quotes (strings) vs double quotes (identifiers)

use stoolap::parser::{Lexer, TokenType};

/// Test single quotes create string tokens
#[test]
fn test_single_quotes_string() {
    let input = "'simple string'";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(
        token.token_type,
        TokenType::String,
        "Single quotes should create String token"
    );
    // The literal includes the quotes
    assert_eq!(token.literal, "'simple string'");
}

/// Test double quotes create identifier tokens
#[test]
fn test_double_quotes_identifier() {
    let input = "\"double quoted\"";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(
        token.token_type,
        TokenType::Identifier,
        "Double quotes should create Identifier token"
    );
    // Identifier tokens don't include quotes
    assert_eq!(token.literal, "double quoted");
}

/// Test backticks create identifier tokens
#[test]
fn test_backticks_identifier() {
    let input = "`backtick id`";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(
        token.token_type,
        TokenType::Identifier,
        "Backticks should create Identifier token"
    );
    // Identifier tokens don't include quotes
    assert_eq!(token.literal, "backtick id");
}

/// Test date string in single quotes
#[test]
fn test_date_string() {
    let input = "'2023-05-15'";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(token.token_type, TokenType::String);
    // Extract content without quotes
    let content = &token.literal[1..token.literal.len() - 1];
    assert_eq!(content, "2023-05-15");
}

/// Test time string in single quotes
#[test]
fn test_time_string() {
    let input = "'14:30:00'";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(token.token_type, TokenType::String);
    let content = &token.literal[1..token.literal.len() - 1];
    assert_eq!(content, "14:30:00");
}

/// Test empty string
#[test]
fn test_empty_string() {
    let input = "''";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(token.token_type, TokenType::String);
    assert_eq!(token.literal, "''");
}

/// Test string with spaces
#[test]
fn test_string_with_spaces() {
    let input = "'hello world'";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(token.token_type, TokenType::String);
    let content = &token.literal[1..token.literal.len() - 1];
    assert_eq!(content, "hello world");
}

/// Test identifier with spaces (double quotes)
#[test]
fn test_identifier_with_spaces() {
    let input = "\"column name\"";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(token.token_type, TokenType::Identifier);
    assert_eq!(token.literal, "column name");
}

/// Test SQL query with mixed quotes
#[test]
fn test_mixed_quotes_in_query() {
    // Double quotes for identifier, single quotes for value
    let input = "SELECT \"column\" FROM mytable WHERE name = 'value'";
    let mut lexer = Lexer::new(input);

    // SELECT
    let token = lexer.next_token();
    assert_eq!(token.token_type, TokenType::Keyword);
    assert_eq!(token.literal.to_uppercase(), "SELECT");

    // "column" - identifier
    let token = lexer.next_token();
    assert_eq!(token.token_type, TokenType::Identifier);
    assert_eq!(token.literal, "column");

    // FROM
    let token = lexer.next_token();
    assert_eq!(token.token_type, TokenType::Keyword);

    // mytable - identifier (not a keyword)
    let token = lexer.next_token();
    assert_eq!(token.token_type, TokenType::Identifier);
    assert_eq!(token.literal, "mytable");

    // WHERE
    let token = lexer.next_token();
    assert_eq!(token.token_type, TokenType::Keyword);

    // name
    let token = lexer.next_token();
    assert_eq!(token.token_type, TokenType::Identifier);

    // =
    let token = lexer.next_token();
    assert_eq!(token.token_type, TokenType::Operator);

    // 'value' - string
    let token = lexer.next_token();
    assert_eq!(token.token_type, TokenType::String);
    let content = &token.literal[1..token.literal.len() - 1];
    assert_eq!(content, "value");
}

/// Test string with numbers
#[test]
fn test_string_with_numbers() {
    let input = "'abc123'";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    assert_eq!(token.token_type, TokenType::String);
    let content = &token.literal[1..token.literal.len() - 1];
    assert_eq!(content, "abc123");
}

/// Test quoted keyword as identifier
#[test]
fn test_quoted_keyword_identifier() {
    let input = "\"select\"";
    let mut lexer = Lexer::new(input);
    let token = lexer.next_token();

    // When quoted, keywords become identifiers
    assert_eq!(token.token_type, TokenType::Identifier);
    assert_eq!(token.literal, "select");
}
