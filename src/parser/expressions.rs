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

//! Expression parsing methods for the SQL Parser

use super::ast::*;
use super::parser::Parser;
use super::precedence::Precedence;
use super::token::{Token, TokenType};

impl Parser {
    /// Parse an expression with the given precedence
    pub fn parse_expression(&mut self, precedence: Precedence) -> Option<Expression> {
        // Parse prefix expression
        let mut left = self.parse_prefix_expression()?;

        // Parse infix expressions while precedence allows
        while !self.peek_token_is(TokenType::Eof) && precedence < self.peek_precedence() {
            // Check if this is an infix operator we can handle
            if !self.is_infix_token() {
                return Some(left);
            }

            self.next_token();
            left = self.parse_infix_expression(left)?;
        }

        Some(left)
    }

    /// Check if the peek token is an infix operator
    fn is_infix_token(&self) -> bool {
        match self.peek_token.token_type {
            TokenType::Operator => true,
            TokenType::Keyword => {
                let kw = self.peek_token.literal.to_uppercase();
                matches!(
                    kw.as_str(),
                    "AND"
                        | "OR"
                        | "XOR"
                        | "LIKE"
                        | "ILIKE"
                        | "GLOB"
                        | "REGEXP"
                        | "RLIKE"
                        | "IS"
                        | "IN"
                        | "BETWEEN"
                        | "AS"
                        | "NOT"
                )
            }
            TokenType::Punctuator => {
                matches!(self.peek_token.literal.as_str(), "." | "(" | "[")
            }
            _ => false,
        }
    }

    /// Parse a prefix expression (literals, identifiers, unary operators, etc.)
    fn parse_prefix_expression(&mut self) -> Option<Expression> {
        match self.cur_token.token_type {
            TokenType::Identifier => Some(self.parse_identifier()),
            TokenType::Integer => self.parse_integer_literal(),
            TokenType::Float => self.parse_float_literal(),
            TokenType::String => Some(self.parse_string_literal()),
            TokenType::Parameter => self.parse_parameter(),
            TokenType::Keyword => self.parse_keyword_expression(),
            TokenType::Operator => self.parse_unary_expression(),
            TokenType::Punctuator => self.parse_punctuator_expression(),
            _ => {
                self.add_error(format!(
                    "no prefix parse function for {:?} at {}",
                    self.cur_token.token_type, self.cur_token.position
                ));
                None
            }
        }
    }

    /// Parse an identifier
    fn parse_identifier(&self) -> Expression {
        Expression::Identifier(Identifier::new(
            self.cur_token.clone(),
            self.cur_token.literal.clone(),
        ))
    }

    /// Parse an integer literal
    fn parse_integer_literal(&mut self) -> Option<Expression> {
        match self.cur_token.literal.parse::<i64>() {
            Ok(value) => Some(Expression::IntegerLiteral(IntegerLiteral {
                token: self.cur_token.clone(),
                value,
            })),
            Err(e) => {
                self.add_error(format!(
                    "could not parse {} as integer: {}",
                    self.cur_token.literal, e
                ));
                None
            }
        }
    }

    /// Parse a float literal
    fn parse_float_literal(&mut self) -> Option<Expression> {
        match self.cur_token.literal.parse::<f64>() {
            Ok(value) => Some(Expression::FloatLiteral(FloatLiteral {
                token: self.cur_token.clone(),
                value,
            })),
            Err(e) => {
                self.add_error(format!(
                    "could not parse {} as float: {}",
                    self.cur_token.literal, e
                ));
                None
            }
        }
    }

    /// Parse a string literal
    fn parse_string_literal(&self) -> Expression {
        let literal = &self.cur_token.literal;
        // Remove surrounding quotes
        let value = if literal.len() >= 2 {
            let inner = &literal[1..literal.len() - 1];
            // Handle escape sequences
            // Note: We don't convert \" to " because:
            // 1. In SQL, double quotes don't need escaping inside single-quoted strings
            // 2. Converting \" to " breaks JSON content like {"key":"value with \"quotes\""}
            inner
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\r")
                .replace("\\'", "'")
                .replace("\\\\", "\\")
        } else {
            literal.clone()
        };

        // Check for JSON type hint
        let type_hint = if value.starts_with('{') && value.ends_with('}') {
            Some("JSON".to_string())
        } else {
            None
        };

        Expression::StringLiteral(StringLiteral {
            token: self.cur_token.clone(),
            value,
            type_hint,
        })
    }

    /// Parse a parameter ($1, ?, :name)
    fn parse_parameter(&mut self) -> Option<Expression> {
        let name = self.cur_token.literal.clone();
        let index = if name == "?" {
            self.next_parameter_index()
        } else if let Some(stripped) = name.strip_prefix('$') {
            match stripped.parse::<usize>() {
                Ok(idx) => idx,
                Err(e) => {
                    self.add_error(format!("invalid parameter index: {}", e));
                    return None;
                }
            }
        } else if name.starts_with(':') {
            // Named parameter - use 0 as index, will be resolved by name
            0
        } else {
            self.add_error(format!("invalid parameter format: {}", name));
            return None;
        };

        Some(Expression::Parameter(Parameter {
            token: self.cur_token.clone(),
            name,
            index,
        }))
    }

    /// Parse a keyword expression (TRUE, FALSE, NULL, CASE, CAST, etc.)
    fn parse_keyword_expression(&mut self) -> Option<Expression> {
        let keyword = self.cur_token.literal.to_uppercase();
        match keyword.as_str() {
            "TRUE" => Some(Expression::BooleanLiteral(BooleanLiteral {
                token: self.cur_token.clone(),
                value: true,
            })),
            "FALSE" => Some(Expression::BooleanLiteral(BooleanLiteral {
                token: self.cur_token.clone(),
                value: false,
            })),
            "NULL" => Some(Expression::NullLiteral(NullLiteral {
                token: self.cur_token.clone(),
            })),
            "CASE" => self.parse_case_expression(),
            "CAST" => self.parse_cast_expression(),
            "EXTRACT" => self.parse_extract_expression(),
            "EXISTS" => self.parse_exists_expression(),
            "NOT" => self.parse_not_expression(),
            "INTERVAL" => self.parse_interval_literal(),
            "DEFAULT" => Some(Expression::Default(DefaultExpression {
                token: self.cur_token.clone(),
            })),
            "TIMESTAMP" | "DATE" | "TIME" => {
                // These can be either typed literals (TIMESTAMP 'value') or column names
                if self.peek_token_is(TokenType::String) {
                    self.parse_typed_literal()
                } else {
                    // Treat as identifier (column name)
                    Some(Expression::Identifier(Identifier::new(
                        self.cur_token.clone(),
                        self.cur_token.literal.clone(),
                    )))
                }
            }
            // Keywords that can also be function names when followed by (
            "LEFT" | "RIGHT" | "CHAR" | "FIRST" | "LAST" | "TRUNCATE" => {
                if self.peek_token_is_punctuator("(") {
                    // Treat as function call
                    let ident = Expression::Identifier(Identifier::new(
                        self.cur_token.clone(),
                        self.cur_token.literal.clone(),
                    ));
                    self.next_token(); // move to (
                    self.parse_function_call(ident)
                } else {
                    // Treat as identifier if not followed by (
                    Some(Expression::Identifier(Identifier::new(
                        self.cur_token.clone(),
                        self.cur_token.literal.clone(),
                    )))
                }
            }
            // Non-reserved keywords can be used as column names
            _ if !Self::is_reserved_keyword(&keyword) => Some(Expression::Identifier(
                Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone()),
            )),
            _ => {
                self.add_error(format!(
                    "unexpected keyword: {} at {}",
                    keyword, self.cur_token.position
                ));
                None
            }
        }
    }

    /// Parse a unary/prefix operator expression
    fn parse_unary_expression(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone();
        let operator = self.cur_token.literal.clone();

        // Special case: * as standalone (for SELECT *, RETURNING *, etc.)
        // When * appears as a prefix and the next token is not something that
        // could continue a multiplication, treat it as Star expression
        if operator == "*" {
            // Check if this is standalone * (not multiplication)
            // Peek at next token to see if it could be a multiplication operand
            if self.peek_token.is_eof()
                || self.peek_token.is_punctuator(",")
                || self.peek_token.is_punctuator(";")
                || self.peek_token.is_punctuator(")")
                || self.peek_token_is_keyword("FROM")
                || self.peek_token_is_keyword("WHERE")
                || self.peek_token_is_keyword("ORDER")
                || self.peek_token_is_keyword("GROUP")
                || self.peek_token_is_keyword("HAVING")
                || self.peek_token_is_keyword("LIMIT")
                || self.peek_token_is_keyword("UNION")
                || self.peek_token_is_keyword("INTERSECT")
                || self.peek_token_is_keyword("EXCEPT")
            {
                return Some(Expression::Star(StarExpression { token }));
            }
        }

        self.next_token();

        // Special case: negative numbers
        if operator == "-"
            && (self.cur_token_is(TokenType::Integer) || self.cur_token_is(TokenType::Float))
        {
            if self.cur_token_is(TokenType::Integer) {
                // Handle i64::MIN special case: -9223372036854775808
                // The literal "9223372036854775808" is too large for i64, but when negated
                // it equals i64::MIN. We must check this before trying parse::<i64>().
                if self.cur_token.literal == "9223372036854775808" {
                    return Some(Expression::IntegerLiteral(IntegerLiteral {
                        token: self.cur_token.clone(),
                        value: i64::MIN,
                    }));
                }
                if let Ok(value) = self.cur_token.literal.parse::<i64>() {
                    return Some(Expression::IntegerLiteral(IntegerLiteral {
                        token: self.cur_token.clone(),
                        value: -value,
                    }));
                }
            } else if let Ok(value) = self.cur_token.literal.parse::<f64>() {
                return Some(Expression::FloatLiteral(FloatLiteral {
                    token: self.cur_token.clone(),
                    value: -value,
                }));
            }
        }

        let right = self.parse_expression(Precedence::Prefix)?;

        Some(Expression::Prefix(PrefixExpression::new(
            token,
            operator,
            Box::new(right),
        )))
    }

    /// Parse a punctuator expression (parentheses, star)
    fn parse_punctuator_expression(&mut self) -> Option<Expression> {
        match self.cur_token.literal.as_str() {
            "(" => self.parse_grouped_expression(),
            "*" => Some(Expression::Star(StarExpression {
                token: self.cur_token.clone(),
            })),
            _ => {
                self.add_error(format!(
                    "unexpected punctuator: {} at {}",
                    self.cur_token.literal, self.cur_token.position
                ));
                None
            }
        }
    }

    /// Parse a grouped expression, tuple, or scalar subquery
    fn parse_grouped_expression(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone();

        // Check for subquery
        if self.peek_token_is_keyword("SELECT") {
            self.next_token(); // Move to SELECT
            let subquery = self.parse_select_statement()?;

            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                self.add_error(format!(
                    "expected ')' after scalar subquery at {}",
                    self.cur_token.position
                ));
                return None;
            }

            return Some(Expression::ScalarSubquery(ScalarSubquery {
                token,
                subquery: Box::new(subquery),
            }));
        }

        self.next_token(); // Move past (

        // Empty parentheses
        if self.cur_token_is_punctuator(")") {
            return Some(Expression::Identifier(Identifier::new(
                self.cur_token.clone(),
                "()".to_string(),
            )));
        }

        let first_expr = self.parse_expression(Precedence::Lowest)?;

        // Check if this is a tuple (comma-separated list)
        if self.peek_token_is_punctuator(",") {
            let mut expressions = vec![first_expr];

            while self.peek_token_is_punctuator(",") {
                self.next_token(); // consume comma
                self.next_token(); // move to next expression

                if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                    expressions.push(expr);
                }
            }

            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                self.add_error(format!("expected ')' at {}", self.cur_token.position));
                return None;
            }

            // Return as ExpressionList (tuple)
            return Some(Expression::ExpressionList(ExpressionList {
                token,
                expressions,
            }));
        }

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!("expected ')' at {}", self.cur_token.position));
            return None;
        }

        Some(first_expr)
    }

    /// Parse an infix expression
    fn parse_infix_expression(&mut self, left: Expression) -> Option<Expression> {
        match self.cur_token.token_type {
            TokenType::Operator => self.parse_binary_expression(left),
            TokenType::Keyword => self.parse_keyword_infix(left),
            TokenType::Punctuator => self.parse_punctuator_infix(left),
            _ => {
                self.add_error(format!(
                    "unexpected infix token: {:?} at {}",
                    self.cur_token.token_type, self.cur_token.position
                ));
                None
            }
        }
    }

    /// Parse a binary operator expression
    fn parse_binary_expression(&mut self, left: Expression) -> Option<Expression> {
        let token = self.cur_token.clone();
        let operator = self.cur_token.literal.clone();
        let precedence = self.cur_precedence();

        self.next_token();

        // Check for ALL/ANY/SOME subquery comparison
        if self.cur_token_is_keyword("ALL")
            || self.cur_token_is_keyword("ANY")
            || self.cur_token_is_keyword("SOME")
        {
            return self.parse_all_any_expression(left, token, operator);
        }

        let right = self.parse_expression(precedence)?;

        Some(Expression::Infix(InfixExpression::new(
            token,
            Box::new(left),
            operator,
            Box::new(right),
        )))
    }

    /// Parse ALL/ANY/SOME subquery comparison (e.g., x > ALL (SELECT ...))
    fn parse_all_any_expression(
        &mut self,
        left: Expression,
        token: Token,
        operator: String,
    ) -> Option<Expression> {
        use super::ast::{AllAnyExpression, AllAnyType};

        let all_any_type = if self.cur_token_is_keyword("ALL") {
            AllAnyType::All
        } else {
            AllAnyType::Any // SOME is an alias for ANY
        };

        // Move past ALL/ANY/SOME
        self.next_token();

        // Expect opening parenthesis
        if !self.cur_token_is_punctuator("(") {
            self.add_error(format!(
                "expected '(' after {} at {}",
                all_any_type, self.cur_token.position
            ));
            return None;
        }

        // Move past (
        self.next_token();

        // Parse subquery
        if !self.cur_token_is_keyword("SELECT") {
            self.add_error(format!(
                "expected SELECT in {} subquery at {}",
                all_any_type, self.cur_token.position
            ));
            return None;
        }

        let subquery = self.parse_select_statement()?;

        // Expect closing parenthesis
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after {} subquery at {}",
                all_any_type, self.cur_token.position
            ));
            return None;
        }

        Some(Expression::AllAny(AllAnyExpression {
            token,
            left: Box::new(left),
            operator,
            all_any_type,
            subquery: Box::new(subquery),
        }))
    }

    /// Parse a keyword infix expression (AND, OR, LIKE, IS, IN, BETWEEN, AS)
    fn parse_keyword_infix(&mut self, left: Expression) -> Option<Expression> {
        let keyword = self.cur_token.literal.to_uppercase();
        match keyword.as_str() {
            "AND" | "OR" | "XOR" => {
                let token = self.cur_token.clone();
                let operator = keyword.clone();
                let precedence = self.cur_precedence();

                self.next_token();
                let right = self.parse_expression(precedence)?;

                Some(Expression::Infix(InfixExpression::new(
                    token,
                    Box::new(left),
                    operator,
                    Box::new(right),
                )))
            }
            "LIKE" | "ILIKE" | "GLOB" | "REGEXP" | "RLIKE" => {
                self.parse_like_expression(left, keyword.clone(), false)
            }
            "IS" => self.parse_is_expression(left),
            "IN" => self.parse_in_expression(left, false),
            "BETWEEN" => self.parse_between_expression(left, false),
            "AS" => self.parse_alias_expression(left),
            "NOT" => {
                // Handle NOT IN, NOT BETWEEN, NOT LIKE, NOT ILIKE, NOT GLOB, NOT REGEXP, NOT RLIKE
                if self.peek_token_is_keyword("IN") {
                    self.next_token(); // consume IN
                    self.parse_in_expression(left, true)
                } else if self.peek_token_is_keyword("BETWEEN") {
                    self.next_token(); // consume BETWEEN
                    self.parse_between_expression(left, true)
                } else if self.peek_token_is_keyword("LIKE") {
                    self.next_token(); // consume LIKE
                    self.parse_like_expression(left, "LIKE".to_string(), true)
                } else if self.peek_token_is_keyword("ILIKE") {
                    self.next_token(); // consume ILIKE
                    self.parse_like_expression(left, "ILIKE".to_string(), true)
                } else if self.peek_token_is_keyword("GLOB") {
                    self.next_token(); // consume GLOB
                    self.parse_like_expression(left, "GLOB".to_string(), true)
                } else if self.peek_token_is_keyword("REGEXP") {
                    self.next_token(); // consume REGEXP
                    self.parse_like_expression(left, "REGEXP".to_string(), true)
                } else if self.peek_token_is_keyword("RLIKE") {
                    self.next_token(); // consume RLIKE
                    self.parse_like_expression(left, "RLIKE".to_string(), true)
                } else {
                    let token = self.cur_token.clone();
                    self.next_token();
                    let right = self.parse_expression(Precedence::Not)?;
                    Some(Expression::Infix(InfixExpression::new(
                        token,
                        Box::new(left),
                        "NOT".to_string(),
                        Box::new(right),
                    )))
                }
            }
            _ => {
                self.add_error(format!(
                    "unexpected infix keyword: {} at {}",
                    keyword, self.cur_token.position
                ));
                None
            }
        }
    }

    /// Parse a punctuator infix expression (., (, [)
    fn parse_punctuator_infix(&mut self, left: Expression) -> Option<Expression> {
        match self.cur_token.literal.as_str() {
            "." => self.parse_qualified_identifier(left),
            "(" => self.parse_function_call(left),
            "[" => self.parse_index_expression(left),
            "*" => {
                // Multiplication
                let token = self.cur_token.clone();
                let precedence = Precedence::Product;

                self.next_token();
                let right = self.parse_expression(precedence)?;

                Some(Expression::Infix(InfixExpression::new(
                    token,
                    Box::new(left),
                    "*".to_string(),
                    Box::new(right),
                )))
            }
            _ => {
                self.add_error(format!(
                    "unexpected infix punctuator: {} at {}",
                    self.cur_token.literal, self.cur_token.position
                ));
                None
            }
        }
    }

    /// Parse a qualified identifier (table.column) or qualified star (table.*)
    fn parse_qualified_identifier(&mut self, left: Expression) -> Option<Expression> {
        let left_ident = match left {
            Expression::Identifier(id) => id,
            _ => {
                self.add_error(format!(
                    "left side of '.' must be an identifier at {}",
                    self.cur_token.position
                ));
                return None;
            }
        };

        // Check for qualified star (table.*)
        if self.peek_token_is_operator("*") {
            self.next_token(); // consume *
            return Some(Expression::QualifiedStar(QualifiedStarExpression {
                token: left_ident.token.clone(),
                qualifier: left_ident.value,
            }));
        }

        // Allow both identifiers and keywords as column names (e.g., t.level, t.type)
        // Keywords like LEVEL can be valid column names when used after a dot
        if !self.peek_token_is(TokenType::Identifier) && !self.peek_token_is(TokenType::Keyword) {
            self.peek_error(TokenType::Identifier);
            return None;
        }
        self.next_token();

        Some(Expression::QualifiedIdentifier(QualifiedIdentifier {
            token: left_ident.token.clone(),
            qualifier: Box::new(left_ident),
            name: Box::new(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            )),
        }))
    }

    /// Parse a function call
    fn parse_function_call(&mut self, left: Expression) -> Option<Expression> {
        let left_ident = match left {
            Expression::Identifier(id) => id,
            _ => {
                self.add_error(format!(
                    "left side of '(' must be an identifier at {}",
                    self.cur_token.position
                ));
                return None;
            }
        };

        let mut call = FunctionCall {
            token: left_ident.token.clone(),
            function: left_ident.value.to_uppercase(),
            arguments: Vec::new(),
            is_distinct: false,
            order_by: Vec::new(),
            filter: None,
        };

        // Check for empty function call
        if self.peek_token_is_punctuator(")") {
            self.next_token();
            // Check for FILTER clause
            if self.peek_token_is_keyword("FILTER") {
                call.filter = self.parse_filter_clause();
            }
            // Check for OVER clause (window function with no arguments)
            if self.peek_token_is_keyword("OVER") {
                return self.parse_window_expression(call);
            }
            return Some(Expression::FunctionCall(call));
        }

        // Check for * argument (COUNT(*))
        if self.peek_token_is_operator("*") {
            self.next_token();
            call.arguments.push(Expression::Star(StarExpression {
                token: self.cur_token.clone(),
            }));

            // Check for ORDER BY inside function
            if self.peek_token_is_keyword("ORDER") {
                self.parse_function_order_by(&mut call);
            }

            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                self.add_error(format!("expected ')' at {}", self.cur_token.position));
                return None;
            }

            // Check for FILTER clause
            if self.peek_token_is_keyword("FILTER") {
                call.filter = self.parse_filter_clause();
            }

            // Check for OVER clause (window function with * argument)
            if self.peek_token_is_keyword("OVER") {
                return self.parse_window_expression(call);
            }

            return Some(Expression::FunctionCall(call));
        }

        // Check for DISTINCT
        if self.peek_token_is_keyword("DISTINCT") {
            self.next_token();
            call.is_distinct = true;
        }

        self.next_token();

        // For POSITION(x IN y) syntax, parse first argument with higher precedence
        // to prevent IN from being consumed as an infix operator
        let first_arg_precedence = if call.function == "POSITION" {
            Precedence::LessGreater // Higher than Equals, so IN won't be consumed
        } else {
            Precedence::Lowest
        };

        // Parse first argument
        if let Some(arg) = self.parse_expression(first_arg_precedence) {
            call.arguments.push(arg);
        }

        // Special handling for POSITION(substring IN string) syntax
        // SQL standard: POSITION(substring IN string) returns position of substring in string
        if call.function == "POSITION" && self.peek_token_is_keyword("IN") {
            self.next_token(); // consume IN
            self.next_token(); // move to string expression

            if let Some(string_arg) = self.parse_expression(Precedence::Lowest) {
                // For POSITION(x IN y), the result is POSITION(x, y) = position of x in y
                call.arguments.push(string_arg);
            } else {
                self.add_error(format!(
                    "expected expression after IN in POSITION at {}",
                    self.cur_token.position
                ));
                return None;
            }
        }

        // Parse additional arguments
        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token(); // move to next arg

            if let Some(arg) = self.parse_expression(Precedence::Lowest) {
                call.arguments.push(arg);
            } else {
                self.add_error(format!(
                    "expected expression after ',' at {}",
                    self.cur_token.position
                ));
                return None;
            }
        }

        // Check for ORDER BY inside function
        if self.peek_token_is_keyword("ORDER") {
            self.parse_function_order_by(&mut call);
        }

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!("expected ')' at {}", self.cur_token.position));
            return None;
        }

        // Check for FILTER clause
        if self.peek_token_is_keyword("FILTER") {
            call.filter = self.parse_filter_clause();
        }

        // Check for OVER clause (window function)
        if self.peek_token_is_keyword("OVER") {
            return self.parse_window_expression(call);
        }

        Some(Expression::FunctionCall(call))
    }

    /// Parse FILTER clause for aggregate functions (FILTER (WHERE condition))
    fn parse_filter_clause(&mut self) -> Option<Box<Expression>> {
        self.next_token(); // consume FILTER

        // Expect opening parenthesis
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!(
                "expected '(' after FILTER at {}",
                self.cur_token.position
            ));
            return None;
        }

        // Expect WHERE keyword
        if !self.expect_keyword("WHERE") {
            self.add_error(format!(
                "expected WHERE after FILTER( at {}",
                self.cur_token.position
            ));
            return None;
        }

        // Parse the condition
        self.next_token();
        let condition = self.parse_expression(Precedence::Lowest)?;

        // Expect closing parenthesis
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after FILTER condition at {}",
                self.cur_token.position
            ));
            return None;
        }

        Some(Box::new(condition))
    }

    /// Parse ORDER BY inside a function call
    fn parse_function_order_by(&mut self, call: &mut FunctionCall) {
        self.next_token(); // consume ORDER

        if !self.expect_keyword("BY") {
            return;
        }

        self.next_token(); // move to first expression

        if let Some(order_expr) = self.parse_order_by_expression() {
            call.order_by.push(order_expr);
        }

        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token(); // move to next expression

            if let Some(order_expr) = self.parse_order_by_expression() {
                call.order_by.push(order_expr);
            }
        }
    }

    /// Parse a window expression (function OVER (...) or function OVER window_name)
    fn parse_window_expression(&mut self, function: FunctionCall) -> Option<Expression> {
        self.next_token(); // consume OVER
        let token = self.cur_token.clone();

        // Check if this is a named window reference (OVER w) instead of inline spec (OVER (...))
        if self.peek_token_is(TokenType::Identifier) {
            self.next_token(); // consume window name
            let window_ref = self.cur_token.literal.clone();
            return Some(Expression::Window(WindowExpression {
                token,
                function: Box::new(function),
                window_ref: Some(window_ref),
                partition_by: Vec::new(),
                order_by: Vec::new(),
                frame: None,
            }));
        }

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!(
                "expected '(' or window name after OVER at {}",
                self.cur_token.position
            ));
            return None;
        }

        let mut partition_by = Vec::new();
        let mut order_by = Vec::new();
        let mut frame = None;

        // Parse PARTITION BY
        if self.peek_token_is_keyword("PARTITION") {
            self.next_token(); // consume PARTITION
            if !self.expect_keyword("BY") {
                return None;
            }

            self.next_token();
            if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                partition_by.push(expr);
            }

            while self.peek_token_is_punctuator(",") {
                self.next_token(); // consume comma
                self.next_token();
                if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                    partition_by.push(expr);
                }
            }
        }

        // Parse ORDER BY
        if self.peek_token_is_keyword("ORDER") {
            self.next_token(); // consume ORDER
            if !self.expect_keyword("BY") {
                return None;
            }

            self.next_token();
            if let Some(order_expr) = self.parse_order_by_expression() {
                order_by.push(order_expr);
            }

            while self.peek_token_is_punctuator(",") {
                self.next_token(); // consume comma
                self.next_token();
                if let Some(order_expr) = self.parse_order_by_expression() {
                    order_by.push(order_expr);
                }
            }
        }

        // Parse window frame (ROWS/RANGE)
        if self.peek_token_is_keyword("ROWS") || self.peek_token_is_keyword("RANGE") {
            frame = self.parse_window_frame();
        }

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after window specification at {}",
                self.cur_token.position
            ));
            return None;
        }

        Some(Expression::Window(WindowExpression {
            token,
            function: Box::new(function),
            window_ref: None,
            partition_by,
            order_by,
            frame,
        }))
    }

    /// Parse a window frame (ROWS/RANGE BETWEEN ... AND ...)
    fn parse_window_frame(&mut self) -> Option<WindowFrame> {
        self.next_token();
        let unit = if self.cur_token_is_keyword("ROWS") {
            WindowFrameUnit::Rows
        } else {
            WindowFrameUnit::Range
        };

        self.next_token();

        // Check for BETWEEN
        let (start, end) = if self.cur_token_is_keyword("BETWEEN") {
            self.next_token();
            let start = self.parse_window_frame_bound()?;

            if !self.expect_keyword("AND") {
                return None;
            }
            self.next_token();
            let end = self.parse_window_frame_bound()?;

            (start, Some(end))
        } else {
            let start = self.parse_window_frame_bound()?;
            (start, None)
        };

        Some(WindowFrame { unit, start, end })
    }

    /// Parse a window frame bound
    fn parse_window_frame_bound(&mut self) -> Option<WindowFrameBound> {
        if self.cur_token_is_keyword("CURRENT") {
            if !self.expect_keyword("ROW") {
                return None;
            }
            Some(WindowFrameBound::CurrentRow)
        } else if self.cur_token_is_keyword("UNBOUNDED") {
            self.next_token();
            if self.cur_token_is_keyword("PRECEDING") {
                Some(WindowFrameBound::UnboundedPreceding)
            } else if self.cur_token_is_keyword("FOLLOWING") {
                Some(WindowFrameBound::UnboundedFollowing)
            } else {
                self.add_error(format!(
                    "expected PRECEDING or FOLLOWING after UNBOUNDED at {}",
                    self.cur_token.position
                ));
                None
            }
        } else {
            // N PRECEDING or N FOLLOWING
            let expr = self.parse_expression(Precedence::Lowest)?;
            self.next_token();
            if self.cur_token_is_keyword("PRECEDING") {
                Some(WindowFrameBound::Preceding(Box::new(expr)))
            } else if self.cur_token_is_keyword("FOLLOWING") {
                Some(WindowFrameBound::Following(Box::new(expr)))
            } else {
                self.add_error(format!(
                    "expected PRECEDING or FOLLOWING at {}",
                    self.cur_token.position
                ));
                None
            }
        }
    }

    /// Parse an index expression (arr[idx])
    fn parse_index_expression(&mut self, left: Expression) -> Option<Expression> {
        let token = self.cur_token.clone();

        self.next_token();
        let index = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "]" {
            self.add_error(format!("expected ']' at {}", self.cur_token.position));
            return None;
        }

        Some(Expression::Infix(InfixExpression::new(
            token,
            Box::new(left),
            "[]".to_string(),
            Box::new(index),
        )))
    }

    /// Parse an IS expression (IS NULL, IS NOT NULL, IS TRUE, IS FALSE, IS DISTINCT FROM, IS NOT DISTINCT FROM)
    fn parse_is_expression(&mut self, left: Expression) -> Option<Expression> {
        let token = self.cur_token.clone();
        let mut operator = "IS".to_string();

        // Check for IS NOT
        if self.peek_token_is_keyword("NOT") {
            self.next_token();
            operator = "IS NOT".to_string();
        }

        self.next_token();

        // Handle IS NULL / IS NOT NULL
        if self.cur_token_is_keyword("NULL") {
            return Some(Expression::Infix(InfixExpression::new(
                token,
                Box::new(left),
                operator,
                Box::new(Expression::NullLiteral(NullLiteral {
                    token: self.cur_token.clone(),
                })),
            )));
        }

        // Handle IS TRUE / IS NOT TRUE
        if self.cur_token_is_keyword("TRUE") {
            return Some(Expression::Infix(InfixExpression::new(
                token,
                Box::new(left),
                operator,
                Box::new(Expression::BooleanLiteral(BooleanLiteral {
                    token: self.cur_token.clone(),
                    value: true,
                })),
            )));
        }

        // Handle IS FALSE / IS NOT FALSE
        if self.cur_token_is_keyword("FALSE") {
            return Some(Expression::Infix(InfixExpression::new(
                token,
                Box::new(left),
                operator,
                Box::new(Expression::BooleanLiteral(BooleanLiteral {
                    token: self.cur_token.clone(),
                    value: false,
                })),
            )));
        }

        // Handle IS DISTINCT FROM / IS NOT DISTINCT FROM
        if self.cur_token_is_keyword("DISTINCT") {
            if !self.expect_keyword("FROM") {
                return None;
            }
            self.next_token();
            let right = self.parse_expression(Precedence::Equals)?;
            let distinct_op = if operator == "IS" {
                "IS DISTINCT FROM".to_string()
            } else {
                "IS NOT DISTINCT FROM".to_string()
            };
            return Some(Expression::Infix(InfixExpression::new(
                token,
                Box::new(left),
                distinct_op,
                Box::new(right),
            )));
        }

        self.add_error(format!(
            "expected NULL, TRUE, FALSE, or DISTINCT FROM after IS at {}",
            self.cur_token.position
        ));
        None
    }

    /// Parse an IN expression
    fn parse_in_expression(&mut self, left: Expression, not: bool) -> Option<Expression> {
        let token = self.cur_token.clone();

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!(
                "expected '(' after IN at {}",
                self.cur_token.position
            ));
            return None;
        }

        // Check for subquery
        if self.peek_token_is_keyword("SELECT") {
            self.next_token();
            let subquery = self.parse_select_statement()?;

            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                self.add_error(format!(
                    "expected ')' after IN subquery at {}",
                    self.cur_token.position
                ));
                return None;
            }

            return Some(Expression::In(InExpression {
                token,
                left: Box::new(left),
                right: Box::new(Expression::ScalarSubquery(ScalarSubquery {
                    token: self.cur_token.clone(),
                    subquery: Box::new(subquery),
                })),
                not,
            }));
        }

        // Parse value list
        let list_token = self.cur_token.clone();
        self.next_token();

        let mut expressions = Vec::new();

        // Handle empty list
        if self.cur_token_is_punctuator(")") {
            return Some(Expression::In(InExpression {
                token,
                left: Box::new(left),
                right: Box::new(Expression::ExpressionList(ExpressionList {
                    token: list_token,
                    expressions,
                })),
                not,
            }));
        }

        // Parse first expression
        if let Some(expr) = self.parse_expression(Precedence::Lowest) {
            expressions.push(expr);
        }

        // Parse additional expressions
        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token(); // move to next expression

            if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                expressions.push(expr);
            }
        }

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' in IN expression at {}",
                self.cur_token.position
            ));
            return None;
        }

        Some(Expression::In(InExpression {
            token,
            left: Box::new(left),
            right: Box::new(Expression::ExpressionList(ExpressionList {
                token: list_token,
                expressions,
            })),
            not,
        }))
    }

    /// Parse a BETWEEN expression
    fn parse_between_expression(&mut self, left: Expression, not: bool) -> Option<Expression> {
        let token = self.cur_token.clone();

        self.next_token();
        let lower = self.parse_expression(Precedence::Equals)?;

        if !self.expect_keyword("AND") {
            return None;
        }

        self.next_token();
        let upper = self.parse_expression(Precedence::Equals)?;

        Some(Expression::Between(BetweenExpression {
            token,
            expr: Box::new(left),
            lower: Box::new(lower),
            upper: Box::new(upper),
            not,
        }))
    }

    /// Parse a LIKE expression with optional ESCAPE clause
    fn parse_like_expression(
        &mut self,
        left: Expression,
        op: String,
        not: bool,
    ) -> Option<Expression> {
        let token = self.cur_token.clone();
        let operator = if not { format!("NOT {}", op) } else { op };
        let precedence = self.cur_precedence();

        self.next_token();
        let pattern = self.parse_expression(precedence)?;

        // Check for optional ESCAPE clause
        let escape = if self.peek_token_is_keyword("ESCAPE") {
            self.next_token(); // consume ESCAPE
            self.next_token(); // move to escape character
            Some(Box::new(self.parse_expression(Precedence::Lowest)?))
        } else {
            None
        };

        Some(Expression::Like(LikeExpression {
            token,
            left: Box::new(left),
            pattern: Box::new(pattern),
            operator,
            escape,
        }))
    }

    /// Parse an alias expression (expr AS alias)
    fn parse_alias_expression(&mut self, left: Expression) -> Option<Expression> {
        let token = self.cur_token.clone();

        // Allow both identifiers and keywords as aliases (e.g., AS level, AS type)
        if !self.peek_token_is(TokenType::Identifier) && !self.peek_token_is(TokenType::Keyword) {
            self.peek_error(TokenType::Identifier);
            return None;
        }
        self.next_token();

        Some(Expression::Aliased(AliasedExpression {
            token,
            expression: Box::new(left),
            alias: Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone()),
        }))
    }

    /// Parse a CASE expression
    fn parse_case_expression(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone();
        let mut value = None;
        let mut when_clauses = Vec::new();
        let mut else_value = None;

        // Check if this is a simple CASE (with value) or searched CASE
        if !self.peek_token_is_keyword("WHEN") {
            self.next_token();
            value = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
        }

        // Parse WHEN clauses
        while self.peek_token_is_keyword("WHEN") {
            self.next_token(); // consume WHEN
            let when_token = self.cur_token.clone();

            self.next_token();
            let condition = self.parse_expression(Precedence::Lowest)?;

            if !self.expect_keyword("THEN") {
                return None;
            }

            self.next_token();
            let then_result = self.parse_expression(Precedence::Lowest)?;

            when_clauses.push(WhenClause {
                token: when_token,
                condition,
                then_result,
            });
        }

        if when_clauses.is_empty() {
            self.add_error(format!(
                "expected at least one WHEN clause in CASE at {}",
                self.cur_token.position
            ));
            return None;
        }

        // Parse ELSE clause
        if self.peek_token_is_keyword("ELSE") {
            self.next_token(); // consume ELSE
            self.next_token();
            else_value = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
        }

        if !self.expect_keyword("END") {
            return None;
        }

        Some(Expression::Case(CaseExpression {
            token,
            value,
            when_clauses,
            else_value,
        }))
    }

    /// Parse a CAST expression
    fn parse_cast_expression(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone();

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!(
                "expected '(' after CAST at {}",
                self.cur_token.position
            ));
            return None;
        }

        self.next_token();
        // Use Equals precedence to prevent AS from being consumed as an alias
        let expr = self.parse_expression(Precedence::Equals)?;

        if !self.expect_keyword("AS") {
            return None;
        }

        // Parse type name
        if !self.peek_token_is(TokenType::Keyword) && !self.peek_token_is(TokenType::Identifier) {
            self.add_error(format!(
                "expected type name after AS in CAST at {}",
                self.peek_token.position
            ));
            return None;
        }
        self.next_token();
        let type_name = self.cur_token.literal.clone();

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after type name in CAST at {}",
                self.cur_token.position
            ));
            return None;
        }

        Some(Expression::Cast(CastExpression {
            token,
            expr: Box::new(expr),
            type_name,
        }))
    }

    /// Parse an EXTRACT expression: EXTRACT(field FROM source)
    /// Converts to equivalent function call: YEAR(source), MONTH(source), etc.
    fn parse_extract_expression(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone();

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!(
                "expected '(' after EXTRACT at {}",
                self.cur_token.position
            ));
            return None;
        }

        // Parse field name (YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, etc.)
        if !self.peek_token_is(TokenType::Keyword) && !self.peek_token_is(TokenType::Identifier) {
            self.add_error(format!(
                "expected field name (YEAR, MONTH, DAY, etc.) in EXTRACT at {}",
                self.peek_token.position
            ));
            return None;
        }
        self.next_token();
        let field = self.cur_token.literal.to_uppercase();

        // Validate field name - all supported fields
        let valid_fields = [
            "YEAR",
            "MONTH",
            "DAY",
            "HOUR",
            "MINUTE",
            "SECOND",
            "DOW",
            "DAYOFWEEK",
            "ISODOW",
            "DOY",
            "DAYOFYEAR",
            "WEEK",
            "ISOWEEK",
            "QUARTER",
            "EPOCH",
            "MILLISECOND",
            "MILLISECONDS",
            "MICROSECOND",
            "MICROSECONDS",
        ];
        if !valid_fields.contains(&field.as_str()) {
            self.add_error(format!(
                "invalid EXTRACT field '{}'. Valid fields: YEAR, MONTH, DAY, HOUR, MINUTE, SECOND, DOW, DOY, WEEK, QUARTER, EPOCH, etc.",
                field
            ));
            return None;
        }

        // Expect FROM keyword
        if !self.expect_keyword("FROM") {
            return None;
        }

        // Parse source expression
        self.next_token();
        let source = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after source in EXTRACT at {}",
                self.cur_token.position
            ));
            return None;
        }

        // For basic fields (YEAR, MONTH, DAY, HOUR, MINUTE, SECOND), convert to direct function call
        // For other fields (DOW, DOY, WEEK, etc.), call EXTRACT(field_str, source)
        let basic_fields = ["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"];
        if basic_fields.contains(&field.as_str()) {
            Some(Expression::FunctionCall(FunctionCall {
                token,
                function: field,
                arguments: vec![source],
                is_distinct: false,
                order_by: Vec::new(),
                filter: None,
            }))
        } else {
            // Call EXTRACT with field name as string argument
            let field_literal = Expression::StringLiteral(StringLiteral {
                token: Token {
                    token_type: TokenType::String,
                    literal: field.to_lowercase(),
                    position: token.position,
                    error: None,
                },
                value: field.to_lowercase(),
                type_hint: None,
            });
            Some(Expression::FunctionCall(FunctionCall {
                token,
                function: "EXTRACT".to_string(),
                arguments: vec![field_literal, source],
                is_distinct: false,
                order_by: Vec::new(),
                filter: None,
            }))
        }
    }

    /// Parse an EXISTS expression
    fn parse_exists_expression(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone();

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!(
                "expected '(' after EXISTS at {}",
                self.cur_token.position
            ));
            return None;
        }

        self.next_token();
        if !self.cur_token_is_keyword("SELECT") {
            self.add_error(format!(
                "expected SELECT in EXISTS subquery at {}",
                self.cur_token.position
            ));
            return None;
        }

        let subquery = self.parse_select_statement()?;

        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after EXISTS subquery at {}",
                self.cur_token.position
            ));
            return None;
        }

        Some(Expression::Exists(ExistsExpression {
            token,
            subquery: Box::new(subquery),
        }))
    }

    /// Parse a NOT expression
    fn parse_not_expression(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone();

        // Check for NOT EXISTS
        if self.peek_token_is_keyword("EXISTS") {
            self.next_token();
            let exists_expr = self.parse_exists_expression()?;
            return Some(Expression::Prefix(PrefixExpression::new(
                token,
                "NOT".to_string(),
                Box::new(exists_expr),
            )));
        }

        self.next_token();
        // Use Precedence::Not (lower than comparison) so NOT captures the full comparison
        // This makes "NOT category = 'Electronics'" parse as "NOT (category = 'Electronics')"
        let right = self.parse_expression(Precedence::Not)?;

        Some(Expression::Prefix(PrefixExpression::new(
            token,
            "NOT".to_string(),
            Box::new(right),
        )))
    }

    /// Parse an INTERVAL literal
    /// Supports two syntaxes:
    /// - INTERVAL '1 month' (quoted string)
    /// - INTERVAL 1 MONTH (integer followed by unit keyword)
    fn parse_interval_literal(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone();

        // Check if the next token is an integer (unquoted syntax) or string (quoted syntax)
        if self.peek_token_is(TokenType::Integer) {
            // INTERVAL 1 MONTH syntax
            self.next_token();
            let quantity = match self.cur_token.literal.parse::<i64>() {
                Ok(q) => q,
                Err(_) => {
                    self.add_error(format!(
                        "invalid interval quantity: {} at {}",
                        self.cur_token.literal, self.cur_token.position
                    ));
                    return None;
                }
            };

            // Expect a unit keyword (SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, YEAR)
            self.next_token();
            let unit_raw = self.cur_token.literal.to_uppercase();
            let unit = match unit_raw.as_str() {
                "SECOND" | "SECONDS" => "second",
                "MINUTE" | "MINUTES" => "minute",
                "HOUR" | "HOURS" => "hour",
                "DAY" | "DAYS" => "day",
                "WEEK" | "WEEKS" => "week",
                "MONTH" | "MONTHS" => "month",
                "YEAR" | "YEARS" => "year",
                _ => {
                    self.add_error(format!(
                        "invalid interval unit: {} at {}. Expected SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, or YEAR",
                        self.cur_token.literal, self.cur_token.position
                    ));
                    return None;
                }
            };

            let value = format!("{} {}", quantity, unit);
            Some(Expression::IntervalLiteral(IntervalLiteral {
                token,
                value,
                quantity,
                unit: unit.to_string(),
            }))
        } else if self.expect_peek(TokenType::String) {
            // INTERVAL '1 month' syntax (original)
            let literal = &self.cur_token.literal;
            let value = if literal.len() >= 2 {
                literal[1..literal.len() - 1].to_string()
            } else {
                literal.clone()
            };

            // Parse the interval value
            let parts: Vec<&str> = value.split_whitespace().collect();
            if parts.len() < 2 {
                self.add_error(format!(
                    "invalid interval format: {} at {}",
                    value, self.cur_token.position
                ));
                return None;
            }

            let quantity = match parts[0].parse::<i64>() {
                Ok(q) => q,
                Err(_) => {
                    self.add_error(format!(
                        "invalid interval quantity: {} at {}",
                        parts[0], self.cur_token.position
                    ));
                    return None;
                }
            };

            let unit = parts[1].to_lowercase().trim_end_matches('s').to_string();

            Some(Expression::IntervalLiteral(IntervalLiteral {
                token,
                value,
                quantity,
                unit,
            }))
        } else {
            None
        }
    }

    /// Parse a typed literal (TIMESTAMP '...', DATE '...', TIME '...')
    fn parse_typed_literal(&mut self) -> Option<Expression> {
        let type_hint = self.cur_token.literal.clone();

        if !self.peek_token_is(TokenType::String) {
            self.add_error(format!(
                "expected string literal after {} at {}",
                type_hint, self.cur_token.position
            ));
            return None;
        }

        self.next_token();

        let literal = &self.cur_token.literal;
        let value = if literal.len() >= 2 && literal.starts_with('\'') && literal.ends_with('\'') {
            literal[1..literal.len() - 1].to_string()
        } else {
            literal.clone()
        };

        Some(Expression::StringLiteral(StringLiteral {
            token: self.cur_token.clone(),
            value,
            type_hint: Some(type_hint),
        }))
    }

    /// Parse an expression list (comma-separated)
    pub fn parse_expression_list(&mut self) -> Vec<Expression> {
        let mut list = Vec::new();

        self.next_token();
        if let Some(expr) = self.parse_expression(Precedence::Lowest) {
            list.push(expr);
        }

        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token();
            if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                list.push(expr);
            }
        }

        list
    }

    /// Parse a GROUP BY clause with optional ROLLUP/CUBE/GROUPING SETS modifier
    pub fn parse_group_by_clause(&mut self) -> GroupByClause {
        use crate::parser::ast::{GroupByClause, GroupByModifier};

        self.next_token();

        // Check for ROLLUP, CUBE, or GROUPING SETS
        let modifier = if self.cur_token.token_type == TokenType::Identifier
            || self.cur_token.token_type == TokenType::Keyword
        {
            let upper = self.cur_token.literal.to_uppercase();
            if upper == "ROLLUP" {
                self.next_token(); // consume ROLLUP
                                   // Expect opening paren
                if self.cur_token.literal == "(" {
                    self.next_token(); // consume (
                    let columns = self.parse_group_by_columns();
                    // cur_token should now be at ), don't advance past it
                    // to maintain consistency with regular GROUP BY parsing
                    // where peek_token points to next clause keyword (ORDER, HAVING, etc.)
                    return GroupByClause {
                        columns,
                        modifier: GroupByModifier::Rollup,
                    };
                }
                GroupByModifier::None
            } else if upper == "CUBE" {
                self.next_token(); // consume CUBE
                                   // Expect opening paren
                if self.cur_token.literal == "(" {
                    self.next_token(); // consume (
                    let columns = self.parse_group_by_columns();
                    // cur_token should now be at ), don't advance past it
                    // to maintain consistency with regular GROUP BY parsing
                    return GroupByClause {
                        columns,
                        modifier: GroupByModifier::Cube,
                    };
                }
                GroupByModifier::None
            } else if upper == "GROUPING" {
                // Check for GROUPING SETS
                if self.peek_token.literal.to_uppercase() == "SETS" {
                    self.next_token(); // consume GROUPING
                    self.next_token(); // consume SETS
                                       // Expect opening paren for the outer grouping sets list
                    if self.cur_token.literal == "(" {
                        self.next_token(); // consume (
                        let sets = self.parse_grouping_sets();
                        // cur_token should now be at )
                        return GroupByClause {
                            columns: Vec::new(),
                            modifier: GroupByModifier::GroupingSets(sets),
                        };
                    }
                }
                GroupByModifier::None
            } else {
                GroupByModifier::None
            }
        } else {
            GroupByModifier::None
        };

        // Regular GROUP BY (no ROLLUP/CUBE/GROUPING SETS)
        let mut columns = Vec::new();
        if let Some(expr) = self.parse_expression(Precedence::Lowest) {
            columns.push(expr);
        }

        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token();
            if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                columns.push(expr);
            }
        }

        GroupByClause { columns, modifier }
    }

    /// Parse columns inside ROLLUP() or CUBE()
    fn parse_group_by_columns(&mut self) -> Vec<Expression> {
        let mut columns = Vec::new();

        if let Some(expr) = self.parse_expression(Precedence::Lowest) {
            columns.push(expr);
        }

        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token();
            if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                columns.push(expr);
            }
        }

        // Move past the last expression to the )
        if self.peek_token_is_punctuator(")") {
            self.next_token();
        }

        columns
    }

    /// Parse GROUPING SETS content: ((col1, col2), (col1), ())
    /// Returns a Vec of Vec<Expression>, where each inner Vec is one grouping set
    fn parse_grouping_sets(&mut self) -> Vec<Vec<Expression>> {
        let mut sets = Vec::new();

        // Parse first set (must be there)
        if self.cur_token.literal == "(" {
            sets.push(self.parse_single_grouping_set());
        }

        // Parse additional sets separated by comma
        while self.cur_token.literal == "," {
            self.next_token(); // consume comma
            if self.cur_token.literal == "(" {
                sets.push(self.parse_single_grouping_set());
            }
        }

        // cur_token should be at the closing ) of GROUPING SETS
        sets
    }

    /// Parse a single grouping set: (col1, col2) or ()
    fn parse_single_grouping_set(&mut self) -> Vec<Expression> {
        let mut columns = Vec::new();

        self.next_token(); // consume opening (

        // Check for empty set ()
        if self.cur_token.literal == ")" {
            self.next_token(); // consume )
            return columns;
        }

        // Parse first column
        if let Some(expr) = self.parse_expression(Precedence::Lowest) {
            columns.push(expr);
        }

        // Parse additional columns separated by comma within this set
        while self.peek_token_is_punctuator(",") {
            self.next_token(); // move to comma
            self.next_token(); // move past comma to next expression
            if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                columns.push(expr);
            }
        }

        // Advance to the closing ) if not already there, then move past it
        if self.peek_token_is_punctuator(")") {
            self.next_token(); // move to )
        }
        self.next_token(); // move past )

        columns
    }

    /// Parse WINDOW clause definitions (WINDOW w AS (...), w2 AS (...))
    pub fn parse_window_definitions(&mut self) -> Vec<WindowDefinition> {
        let mut defs = Vec::new();

        // First window definition
        if let Some(def) = self.parse_single_window_definition() {
            defs.push(def);
        }

        // Additional window definitions separated by comma
        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token(); // move to next window name
            if let Some(def) = self.parse_single_window_definition() {
                defs.push(def);
            }
        }

        defs
    }

    /// Parse a single window definition: name AS (partition/order/frame)
    fn parse_single_window_definition(&mut self) -> Option<WindowDefinition> {
        self.next_token(); // move to window name

        // Get window name
        let name = if self.cur_token_is(TokenType::Identifier) {
            self.cur_token.literal.clone()
        } else if self.cur_token_is(TokenType::Keyword) {
            // Allow keywords as window names
            self.cur_token.literal.clone()
        } else {
            self.add_error(format!(
                "expected window name at {}",
                self.cur_token.position
            ));
            return None;
        };

        // Expect AS keyword
        if !self.expect_keyword("AS") {
            return None;
        }

        // Expect opening parenthesis
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!(
                "expected '(' after AS in window definition at {}",
                self.cur_token.position
            ));
            return None;
        }

        let mut partition_by = Vec::new();
        let mut order_by = Vec::new();
        let mut frame = None;

        // Parse PARTITION BY
        if self.peek_token_is_keyword("PARTITION") {
            self.next_token(); // consume PARTITION
            if !self.expect_keyword("BY") {
                return None;
            }

            self.next_token();
            if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                partition_by.push(expr);
            }

            while self.peek_token_is_punctuator(",") {
                self.next_token(); // consume comma
                self.next_token();
                if let Some(expr) = self.parse_expression(Precedence::Lowest) {
                    partition_by.push(expr);
                }
            }
        }

        // Parse ORDER BY
        if self.peek_token_is_keyword("ORDER") {
            self.next_token(); // consume ORDER
            if !self.expect_keyword("BY") {
                return None;
            }

            self.next_token();
            if let Some(order_expr) = self.parse_order_by_expression() {
                order_by.push(order_expr);
            }

            while self.peek_token_is_punctuator(",") {
                self.next_token(); // consume comma
                self.next_token();
                if let Some(order_expr) = self.parse_order_by_expression() {
                    order_by.push(order_expr);
                }
            }
        }

        // Parse window frame (ROWS/RANGE)
        if self.peek_token_is_keyword("ROWS") || self.peek_token_is_keyword("RANGE") {
            frame = self.parse_window_frame();
        }

        // Expect closing parenthesis
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after window specification at {}",
                self.cur_token.position
            ));
            return None;
        }

        Some(WindowDefinition {
            name,
            partition_by,
            order_by,
            frame,
        })
    }

    /// Parse an ORDER BY expression
    pub fn parse_order_by_expression(&mut self) -> Option<OrderByExpression> {
        let expression = self.parse_expression(Precedence::Lowest)?;

        let mut ascending = true;
        if self.peek_token_is_keyword("ASC") {
            self.next_token();
            ascending = true;
        } else if self.peek_token_is_keyword("DESC") {
            self.next_token();
            ascending = false;
        }

        // Parse optional NULLS FIRST / NULLS LAST
        let nulls_first = if self.peek_token_is_keyword("NULLS") {
            self.next_token(); // consume NULLS
            if self.peek_token_is_keyword("FIRST") {
                self.next_token(); // consume FIRST
                Some(true)
            } else if self.peek_token_is_keyword("LAST") {
                self.next_token(); // consume LAST
                Some(false)
            } else {
                // NULLS without FIRST/LAST - invalid, but treat as default
                None
            }
        } else {
            None
        };

        Some(OrderByExpression {
            expression,
            ascending,
            nulls_first,
        })
    }

    /// Parse ORDER BY expressions (comma-separated)
    pub fn parse_order_by_expressions(&mut self) -> Vec<OrderByExpression> {
        let mut list = Vec::new();

        self.next_token();
        if let Some(expr) = self.parse_order_by_expression() {
            list.push(expr);
        }

        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token();
            if let Some(expr) = self.parse_order_by_expression() {
                list.push(expr);
            }
        }

        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_expr(input: &str) -> Option<Expression> {
        let mut parser = Parser::new(input);
        parser.parse_expression(Precedence::Lowest)
    }

    #[test]
    fn test_parse_identifier() {
        let expr = parse_expr("users").unwrap();
        match expr {
            Expression::Identifier(id) => assert_eq!(id.value, "users"),
            _ => panic!("expected Identifier"),
        }
    }

    #[test]
    fn test_parse_integer() {
        let expr = parse_expr("42").unwrap();
        match expr {
            Expression::IntegerLiteral(lit) => assert_eq!(lit.value, 42),
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_parse_float() {
        let expr = parse_expr("3.5").unwrap();
        match expr {
            Expression::FloatLiteral(lit) => assert!((lit.value - 3.5).abs() < 0.001),
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_parse_string() {
        let expr = parse_expr("'hello'").unwrap();
        match expr {
            Expression::StringLiteral(lit) => assert_eq!(lit.value, "hello"),
            _ => panic!("expected StringLiteral"),
        }
    }

    #[test]
    fn test_parse_boolean() {
        let expr = parse_expr("TRUE").unwrap();
        match expr {
            Expression::BooleanLiteral(lit) => assert!(lit.value),
            _ => panic!("expected BooleanLiteral"),
        }

        let expr = parse_expr("FALSE").unwrap();
        match expr {
            Expression::BooleanLiteral(lit) => assert!(!lit.value),
            _ => panic!("expected BooleanLiteral"),
        }
    }

    #[test]
    fn test_parse_null() {
        let expr = parse_expr("NULL").unwrap();
        match expr {
            Expression::NullLiteral(_) => {}
            _ => panic!("expected NullLiteral"),
        }
    }

    #[test]
    fn test_parse_infix() {
        let expr = parse_expr("1 + 2").unwrap();
        match expr {
            Expression::Infix(infix) => {
                assert_eq!(infix.operator, "+");
            }
            _ => panic!("expected InfixExpression"),
        }
    }

    #[test]
    fn test_parse_precedence() {
        let expr = parse_expr("1 + 2 * 3").unwrap();
        // Should be parsed as 1 + (2 * 3)
        match expr {
            Expression::Infix(infix) => {
                assert_eq!(infix.operator, "+");
                match infix.right.as_ref() {
                    Expression::Infix(right) => {
                        assert_eq!(right.operator, "*");
                    }
                    _ => panic!("expected nested InfixExpression"),
                }
            }
            _ => panic!("expected InfixExpression"),
        }
    }

    #[test]
    fn test_parse_qualified_identifier() {
        let expr = parse_expr("users.id").unwrap();
        match expr {
            Expression::QualifiedIdentifier(qi) => {
                assert_eq!(qi.qualifier.value, "users");
                assert_eq!(qi.name.value, "id");
            }
            _ => panic!("expected QualifiedIdentifier"),
        }
    }

    #[test]
    fn test_parse_function_call() {
        let expr = parse_expr("COUNT(*)").unwrap();
        match expr {
            Expression::FunctionCall(fc) => {
                assert_eq!(fc.function, "COUNT");
                assert_eq!(fc.arguments.len(), 1);
            }
            _ => panic!("expected FunctionCall"),
        }
    }

    #[test]
    fn test_parse_count_star_with_filter() {
        let expr = parse_expr("COUNT(*) FILTER (WHERE x = 1)").unwrap();
        match expr {
            Expression::FunctionCall(fc) => {
                assert_eq!(fc.function, "COUNT");
                assert_eq!(fc.arguments.len(), 1);
                assert!(
                    fc.filter.is_some(),
                    "FILTER clause should be parsed for COUNT(*)"
                );
            }
            _ => panic!("expected FunctionCall"),
        }
    }

    #[test]
    fn test_parse_function_call_with_multiple_args() {
        let expr = parse_expr("STRING_AGG(name, '; ')").unwrap();
        match expr {
            Expression::FunctionCall(fc) => {
                assert_eq!(fc.function, "STRING_AGG");
                assert_eq!(
                    fc.arguments.len(),
                    2,
                    "Expected 2 arguments, got: {:?}",
                    fc.arguments
                );
                // First argument should be an identifier
                match &fc.arguments[0] {
                    Expression::Identifier(id) => assert_eq!(id.value, "name"),
                    other => panic!("Expected Identifier, got: {:?}", other),
                }
                // Second argument should be a string literal
                match &fc.arguments[1] {
                    Expression::StringLiteral(lit) => assert_eq!(lit.value, "; "),
                    other => panic!("Expected StringLiteral, got: {:?}", other),
                }
            }
            _ => panic!("expected FunctionCall"),
        }
    }

    #[test]
    fn test_parse_is_null() {
        let expr = parse_expr("x IS NULL").unwrap();
        match expr {
            Expression::Infix(infix) => {
                assert_eq!(infix.operator, "IS");
            }
            _ => panic!("expected InfixExpression"),
        }
    }

    #[test]
    fn test_parse_is_not_null() {
        let expr = parse_expr("x IS NOT NULL").unwrap();
        match expr {
            Expression::Infix(infix) => {
                assert_eq!(infix.operator, "IS NOT");
            }
            _ => panic!("expected InfixExpression"),
        }
    }

    #[test]
    fn test_parse_in() {
        let expr = parse_expr("x IN (1, 2, 3)").unwrap();
        match expr {
            Expression::In(in_expr) => {
                assert!(!in_expr.not);
            }
            _ => panic!("expected InExpression"),
        }
    }

    #[test]
    fn test_parse_not_in() {
        let expr = parse_expr("x NOT IN (1, 2, 3)").unwrap();
        match expr {
            Expression::In(in_expr) => {
                assert!(in_expr.not);
            }
            _ => panic!("expected InExpression"),
        }
    }

    #[test]
    fn test_parse_between() {
        let expr = parse_expr("x BETWEEN 1 AND 10").unwrap();
        match expr {
            Expression::Between(between) => {
                assert!(!between.not);
            }
            _ => panic!("expected BetweenExpression"),
        }
    }

    #[test]
    fn test_parse_case() {
        let expr = parse_expr("CASE WHEN x = 1 THEN 'one' ELSE 'other' END").unwrap();
        match expr {
            Expression::Case(case) => {
                assert!(case.value.is_none());
                assert_eq!(case.when_clauses.len(), 1);
                assert!(case.else_value.is_some());
            }
            _ => panic!("expected CaseExpression"),
        }
    }

    #[test]
    fn test_parse_cast() {
        let expr = parse_expr("CAST(x AS INTEGER)").unwrap();
        match expr {
            Expression::Cast(cast) => {
                assert_eq!(cast.type_name, "INTEGER");
            }
            _ => panic!("expected CastExpression"),
        }
    }
}
