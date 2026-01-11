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

//! SQL Parser
//!
//! This module provides a complete SQL parser for Stoolap, including:
//!
//! - [`Lexer`] - Tokenizer for SQL input
//! - [`Parser`] - Parser that builds AST from tokens
//! - [`ast`] - Abstract Syntax Tree types
//! - [`token`] - Token types
//! - [`error`] - Parser error types
//!
//! # Example
//!
//! ```
//! use stoolap::parser::{parse_sql, Statement};
//!
//! let sql = "SELECT * FROM users WHERE id = 1";
//! let statements = parse_sql(sql).unwrap();
//! assert_eq!(statements.len(), 1);
//! ```

pub mod ast;
pub mod error;
pub mod lexer;
#[allow(clippy::module_inception)]
pub mod parser;
pub mod precedence;
pub mod token;

// Expression and statement parsing are implemented as impl blocks on Parser
mod expressions;
mod statements;

// Re-export main types
pub use ast::{
    // Expressions
    AliasedExpression,
    AllAnyExpression,
    AllAnyType,
    AlterTableOperation,
    AlterTableStatement,
    // Table sources
    AsOfClause,
    BeginStatement,
    BetweenExpression,
    BooleanLiteral,
    CaseExpression,
    CastExpression,
    ColumnConstraint,
    ColumnDefinition,
    CommitStatement,
    CommonTableExpression,
    CreateIndexStatement,
    CreateTableStatement,
    CreateViewStatement,
    CteReference,
    DeleteStatement,
    DistinctExpression,
    DropIndexStatement,
    DropTableStatement,
    DropViewStatement,
    ExistsExpression,
    ExplainStatement,
    Expression,
    ExpressionList,
    ExpressionStatement,
    FloatLiteral,
    FunctionCall,
    Identifier,
    InExpression,
    InfixExpression,
    InsertStatement,
    IntegerLiteral,
    IntervalLiteral,
    JoinTableSource,
    ListExpression,
    NullLiteral,
    // ORDER BY
    OrderByExpression,
    Parameter,
    PragmaStatement,
    PrefixExpression,
    Program,
    QualifiedIdentifier,
    RollbackStatement,
    SavepointStatement,
    ScalarSubquery,
    SelectStatement,
    SetOperation,
    SetOperationType,
    SetStatement,
    ShowCreateTableStatement,
    ShowIndexesStatement,
    ShowTablesStatement,
    SimpleTableSource,
    StarExpression,
    // Statements
    Statement,
    StringLiteral,
    SubqueryTableSource,
    UpdateStatement,
    WhenClause,
    WindowExpression,
    // Window
    WindowFrame,
    WindowFrameBound,
    WindowFrameUnit,
    WithClause,
};

pub use error::{ParseError, ParseErrors};
pub use lexer::Lexer;
pub use parser::Parser;
pub use precedence::Precedence;
pub use token::{
    is_keyword, is_operator, is_punctuator, Position, Token, TokenType, KEYWORDS, OPERATORS,
    PUNCTUATORS,
};

/// Parse SQL and return statements
///
/// This is the main entry point for parsing SQL strings.
///
/// # Arguments
///
/// * `sql` - The SQL string to parse
///
/// # Returns
///
/// * `Ok(Vec<Statement>)` - Successfully parsed statements
/// * `Err(ParseErrors)` - Parse errors encountered
///
/// # Example
///
/// ```
/// use stoolap::parser::parse_sql;
///
/// let statements = parse_sql("SELECT 1").unwrap();
/// assert_eq!(statements.len(), 1);
/// ```
pub fn parse_sql(sql: &str) -> Result<Vec<Statement>, ParseErrors> {
    let sql = sql.trim();
    if sql.is_empty() {
        return Err(ParseErrors::from_errors(vec![ParseError::new(
            "No statements found in query".to_string(),
            Position::new(0, 1, 1),
        )]));
    }

    let mut parser = Parser::new(sql);
    let program = parser.parse_program()?;

    if program.statements.is_empty() {
        return Err(ParseErrors::from_errors(vec![ParseError::new(
            "No statements found in query".to_string(),
            Position::new(0, 1, 1),
        )]));
    }

    Ok(program.statements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_select() {
        let result = parse_sql("SELECT 1");
        assert!(result.is_ok());
        let statements = result.unwrap();
        assert_eq!(statements.len(), 1);
    }

    #[test]
    fn test_parse_select_from() {
        let result = parse_sql("SELECT * FROM users");
        assert!(result.is_ok());
        let statements = result.unwrap();
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Select(s) => {
                assert!(!s.distinct);
                assert_eq!(s.columns.len(), 1);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_empty_string() {
        let result = parse_sql("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_whitespace_only() {
        let result = parse_sql("   \n\t  ");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_select_with_where() {
        let result = parse_sql("SELECT id, name FROM users WHERE id = 1");
        assert!(result.is_ok());
        let statements = result.unwrap();
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 2);
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_insert() {
        let result = parse_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')");
        assert!(result.is_ok());
        let statements = result.unwrap();
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Insert(s) => {
                assert_eq!(s.table_name.value, "users");
                assert_eq!(s.columns.len(), 2);
                assert_eq!(s.values.len(), 1);
            }
            _ => panic!("Expected INSERT statement"),
        }
    }

    #[test]
    fn test_parse_update() {
        let result = parse_sql("UPDATE users SET name = 'Bob' WHERE id = 1");
        assert!(result.is_ok());
        let statements = result.unwrap();
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Update(s) => {
                assert_eq!(s.table_name.value, "users");
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected UPDATE statement"),
        }
    }

    #[test]
    fn test_parse_delete() {
        let result = parse_sql("DELETE FROM users WHERE id = 1");
        assert!(result.is_ok());
        let statements = result.unwrap();
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::Delete(s) => {
                assert_eq!(s.table_name.value, "users");
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected DELETE statement"),
        }
    }

    #[test]
    fn test_parse_create_table() {
        let result = parse_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)");
        assert!(result.is_ok());
        let statements = result.unwrap();
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::CreateTable(s) => {
                assert_eq!(s.table_name.value, "users");
                assert_eq!(s.columns.len(), 2);
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parse_complex_query() {
        let result = parse_sql(
            r#"
            SELECT u.id, u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.active = TRUE
            GROUP BY u.id, u.name
            HAVING COUNT(o.id) > 0
            ORDER BY order_count DESC
            LIMIT 10
        "#,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_cte() {
        let result = parse_sql("WITH temp AS (SELECT * FROM users) SELECT * FROM temp");
        assert!(result.is_ok());
        let statements = result.unwrap();
        match &statements[0] {
            Statement::Select(s) => {
                assert!(s.with.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_transaction() {
        let begin = parse_sql("BEGIN TRANSACTION").unwrap();
        assert!(matches!(begin[0], Statement::Begin(_)));

        let commit = parse_sql("COMMIT").unwrap();
        assert!(matches!(commit[0], Statement::Commit(_)));

        let rollback = parse_sql("ROLLBACK").unwrap();
        assert!(matches!(rollback[0], Statement::Rollback(_)));
    }
}
