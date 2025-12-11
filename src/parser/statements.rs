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

//! Statement parsing methods for the SQL Parser

use std::collections::HashMap;

use super::ast::*;
use super::parser::Parser;
use super::precedence::Precedence;
use super::token::TokenType;

impl Parser {
    /// Parse a statement
    pub fn parse_statement(&mut self) -> Option<Statement> {
        // Skip comments
        while self.cur_token_is(TokenType::Comment) {
            self.next_token();
        }

        if self.cur_token_is(TokenType::Eof) {
            return None;
        }

        if self.cur_token_is(TokenType::Keyword) {
            let keyword = self.cur_token.literal.to_uppercase();
            match keyword.as_str() {
                "SELECT" => self.parse_select_statement().map(Statement::Select),
                "WITH" => self.parse_with_statement(),
                "INSERT" => self.parse_insert_statement().map(Statement::Insert),
                "UPDATE" => self.parse_update_statement().map(Statement::Update),
                "DELETE" => self.parse_delete_statement().map(Statement::Delete),
                "TRUNCATE" => self.parse_truncate_statement().map(Statement::Truncate),
                "CREATE" => self.parse_create_statement(),
                "DROP" => self.parse_drop_statement(),
                "ALTER" => self.parse_alter_statement().map(Statement::AlterTable),
                "BEGIN" => self.parse_begin_statement().map(Statement::Begin),
                "COMMIT" => self.parse_commit_statement().map(Statement::Commit),
                "ROLLBACK" => self.parse_rollback_statement().map(Statement::Rollback),
                "SAVEPOINT" => self.parse_savepoint_statement().map(Statement::Savepoint),
                "SET" => self.parse_set_statement().map(Statement::Set),
                "PRAGMA" => self.parse_pragma_statement().map(Statement::Pragma),
                "SHOW" => self.parse_show_statement(),
                "DESCRIBE" | "DESC" => self.parse_describe_statement().map(Statement::Describe),
                "EXPLAIN" => self.parse_explain_statement().map(Statement::Explain),
                "ANALYZE" => self.parse_analyze_statement().map(Statement::Analyze),
                _ => {
                    // Try to parse as expression statement
                    self.parse_expression_statement().map(Statement::Expression)
                }
            }
        } else {
            // Try to parse as expression statement
            self.parse_expression_statement().map(Statement::Expression)
        }
    }

    /// Parse a SELECT statement
    pub fn parse_select_statement(&mut self) -> Option<SelectStatement> {
        let token = self.cur_token.clone();

        let mut stmt = SelectStatement {
            token,
            distinct: false,
            columns: Vec::new(),
            with: None,
            table_expr: None,
            where_clause: None,
            group_by: GroupByClause::default(),
            having: None,
            window_defs: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            set_operations: Vec::new(),
        };

        // Check for DISTINCT
        if self.peek_token_is_keyword("DISTINCT") {
            self.next_token();
            stmt.distinct = true;
        }

        // Parse column list
        self.next_token();
        stmt.columns = self.parse_select_columns();

        // Parse FROM clause
        if self.peek_token_is_keyword("FROM") {
            self.next_token(); // consume FROM
            self.next_token(); // move to table expression
            stmt.table_expr = Some(Box::new(self.parse_table_expression()?));
        }

        // Parse WHERE clause
        if self.peek_token_is_keyword("WHERE") {
            self.next_token(); // consume WHERE
            self.current_clause = "WHERE".to_string();
            self.next_token();
            stmt.where_clause = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
        }

        // Parse GROUP BY clause
        if self.peek_token_is_keyword("GROUP") {
            self.next_token(); // consume GROUP
            if !self.expect_keyword("BY") {
                return None;
            }
            self.current_clause = "GROUP BY".to_string();
            stmt.group_by = self.parse_group_by_clause();
        }

        // Parse HAVING clause
        if self.peek_token_is_keyword("HAVING") {
            self.next_token(); // consume HAVING
            self.current_clause = "HAVING".to_string();
            self.next_token();
            stmt.having = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
        }

        // Parse WINDOW clause (named window definitions)
        if self.peek_token_is_keyword("WINDOW") {
            self.next_token(); // consume WINDOW
            self.current_clause = "WINDOW".to_string();
            stmt.window_defs = self.parse_window_definitions();
        }

        // Parse UNION, INTERSECT, EXCEPT set operations
        while self.peek_token_is_keyword("UNION")
            || self.peek_token_is_keyword("INTERSECT")
            || self.peek_token_is_keyword("EXCEPT")
        {
            if let Some(set_op) = self.parse_set_operation() {
                stmt.set_operations.push(set_op);
            } else {
                break;
            }
        }

        // Parse ORDER BY clause (applies to entire compound query)
        if self.peek_token_is_keyword("ORDER") {
            self.next_token(); // consume ORDER
            if !self.expect_keyword("BY") {
                return None;
            }
            self.current_clause = "ORDER BY".to_string();
            stmt.order_by = self.parse_order_by_expressions();
        }

        // Parse LIMIT clause
        if self.peek_token_is_keyword("LIMIT") {
            self.next_token(); // consume LIMIT
            self.current_clause = "LIMIT".to_string();
            self.next_token();
            stmt.limit = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
        }

        // Parse OFFSET clause
        if self.peek_token_is_keyword("OFFSET") {
            self.next_token(); // consume OFFSET
            self.current_clause = "OFFSET".to_string();
            self.next_token();
            stmt.offset = Some(Box::new(self.parse_expression(Precedence::Lowest)?));

            // Optional ROWS/ROW keyword after OFFSET value
            if self.peek_token_is_keyword("ROWS") || self.peek_token_is_keyword("ROW") {
                self.next_token();
            }
        }

        // Parse FETCH FIRST/NEXT n ROWS ONLY clause (alternative to LIMIT)
        if self.peek_token_is_keyword("FETCH") {
            self.next_token(); // consume FETCH

            // FIRST or NEXT (both are equivalent)
            if !self.peek_token_is_keyword("FIRST") && !self.peek_token_is_keyword("NEXT") {
                self.add_error(format!(
                    "expected FIRST or NEXT after FETCH at {}",
                    self.peek_token.position
                ));
                return None;
            }
            self.next_token(); // consume FIRST/NEXT

            self.current_clause = "FETCH".to_string();
            self.next_token();
            stmt.limit = Some(Box::new(self.parse_expression(Precedence::Lowest)?));

            // Optional ROWS/ROW keyword
            if self.peek_token_is_keyword("ROWS") || self.peek_token_is_keyword("ROW") {
                self.next_token();
            }

            // Optional ONLY keyword
            if self.peek_token_is_keyword("ONLY") {
                self.next_token();
            }
        }

        self.current_clause.clear();
        Some(stmt)
    }

    /// Parse a set operation (UNION, INTERSECT, EXCEPT)
    /// Handles SQL standard precedence: INTERSECT/EXCEPT bind tighter than UNION
    fn parse_set_operation(&mut self) -> Option<SetOperation> {
        self.next_token(); // consume UNION/INTERSECT/EXCEPT

        let keyword = self.cur_token.literal.to_uppercase();
        let is_union = keyword == "UNION";
        let operation = if keyword == "UNION" {
            if self.peek_token_is_keyword("ALL") {
                self.next_token();
                SetOperationType::UnionAll
            } else {
                SetOperationType::Union
            }
        } else if keyword == "INTERSECT" {
            if self.peek_token_is_keyword("ALL") {
                self.next_token();
                SetOperationType::IntersectAll
            } else {
                SetOperationType::Intersect
            }
        } else if keyword == "EXCEPT" {
            if self.peek_token_is_keyword("ALL") {
                self.next_token();
                SetOperationType::ExceptAll
            } else {
                SetOperationType::Except
            }
        } else {
            return None;
        };

        // Expect SELECT
        if !self.expect_keyword("SELECT") {
            return None;
        }

        // Parse the right side SELECT
        let mut right = self.parse_simple_select()?;

        // If this is UNION, the right side should consume any INTERSECT/EXCEPT operations
        // because INTERSECT/EXCEPT have higher precedence than UNION
        if is_union {
            while self.peek_token_is_keyword("INTERSECT") || self.peek_token_is_keyword("EXCEPT") {
                if let Some(set_op) = self.parse_set_operation() {
                    right.set_operations.push(set_op);
                } else {
                    break;
                }
            }
        }

        Some(SetOperation {
            operation,
            right: Box::new(right),
        })
    }

    /// Parse a simple SELECT (without set operations, used for right side of UNION etc)
    fn parse_simple_select(&mut self) -> Option<SelectStatement> {
        let token = self.cur_token.clone();

        let mut stmt = SelectStatement {
            token,
            distinct: false,
            columns: Vec::new(),
            with: None,
            table_expr: None,
            where_clause: None,
            group_by: GroupByClause::default(),
            having: None,
            window_defs: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            offset: None,
            set_operations: Vec::new(),
        };

        // Check for DISTINCT
        if self.peek_token_is_keyword("DISTINCT") {
            self.next_token();
            stmt.distinct = true;
        }

        // Parse column list
        self.next_token();
        stmt.columns = self.parse_select_columns();

        // Parse FROM clause
        if self.peek_token_is_keyword("FROM") {
            self.next_token(); // consume FROM
            self.next_token(); // move to table expression
            stmt.table_expr = Some(Box::new(self.parse_table_expression()?));
        }

        // Parse WHERE clause
        if self.peek_token_is_keyword("WHERE") {
            self.next_token(); // consume WHERE
            self.current_clause = "WHERE".to_string();
            self.next_token();
            stmt.where_clause = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
        }

        // Parse GROUP BY clause
        if self.peek_token_is_keyword("GROUP") {
            self.next_token(); // consume GROUP
            if !self.expect_keyword("BY") {
                return None;
            }
            self.current_clause = "GROUP BY".to_string();
            stmt.group_by = self.parse_group_by_clause();
        }

        // Parse HAVING clause
        if self.peek_token_is_keyword("HAVING") {
            self.next_token(); // consume HAVING
            self.current_clause = "HAVING".to_string();
            self.next_token();
            stmt.having = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
        }

        self.current_clause.clear();
        Some(stmt)
    }

    /// Parse SELECT columns
    fn parse_select_columns(&mut self) -> Vec<Expression> {
        let mut columns = Vec::new();

        // Parse first column
        if let Some(col) = self.parse_select_column() {
            columns.push(col);
        }

        // Parse additional columns
        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token(); // move to next column
            if let Some(col) = self.parse_select_column() {
                columns.push(col);
            }
        }

        columns
    }

    /// Parse a single SELECT column
    fn parse_select_column(&mut self) -> Option<Expression> {
        // Check for * (all columns)
        if self.cur_token_is(TokenType::Operator) && self.cur_token.literal == "*" {
            return Some(Expression::Star(StarExpression {
                token: self.cur_token.clone(),
            }));
        }

        // Parse expression
        let expr = self.parse_expression(Precedence::Lowest)?;

        // Check for alias with AS keyword
        if self.peek_token_is_keyword("AS") {
            self.next_token(); // consume AS
                               // Allow both identifiers and keywords as aliases (e.g., AS level, AS type)
            if !self.peek_token_is(TokenType::Identifier) && !self.peek_token_is(TokenType::Keyword)
            {
                self.peek_error(TokenType::Identifier);
                return None;
            }
            self.next_token();
            return Some(Expression::Aliased(AliasedExpression {
                token: self.cur_token.clone(),
                expression: Box::new(expr),
                alias: Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone()),
            }));
        }

        // Check for implicit alias (identifier without AS)
        // Must be an identifier that's not a reserved keyword like FROM, WHERE, etc.
        if self.peek_token_is(TokenType::Identifier) {
            let alias_candidate = self.peek_token.literal.to_uppercase();
            // List of keywords that cannot be implicit aliases (they end the column list or start clauses)
            let reserved = [
                "FROM",
                "WHERE",
                "GROUP",
                "HAVING",
                "ORDER",
                "LIMIT",
                "OFFSET",
                "UNION",
                "INTERSECT",
                "EXCEPT",
                "INTO",
                "FOR",
                "WINDOW",
                "FETCH",
                "ON",
                "USING",
                "NATURAL",
                "LEFT",
                "RIGHT",
                "INNER",
                "OUTER",
                "CROSS",
                "FULL",
                "JOIN",
            ];
            if !reserved.contains(&alias_candidate.as_str()) {
                self.next_token();
                return Some(Expression::Aliased(AliasedExpression {
                    token: self.cur_token.clone(),
                    expression: Box::new(expr),
                    alias: Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone()),
                }));
            }
        }

        Some(expr)
    }

    /// Parse a table expression (for FROM clause)
    fn parse_table_expression(&mut self) -> Option<Expression> {
        let left = self.parse_simple_table_expression()?;
        self.parse_join_table_expression(left)
    }

    /// Parse a simple table expression (table name, subquery, VALUES, or CTE reference)
    fn parse_simple_table_expression(&mut self) -> Option<Expression> {
        // Check for subquery or VALUES
        if self.cur_token_is_punctuator("(") {
            self.next_token();
            if self.cur_token_is_keyword("SELECT") {
                let subquery = self.parse_select_statement()?;

                if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                    self.add_error(format!(
                        "expected ')' after subquery at {}",
                        self.cur_token.position
                    ));
                    return None;
                }

                let mut alias = None;
                if self.peek_token_is_keyword("AS") {
                    self.next_token();
                    self.next_token();
                    alias = Some(Identifier::new(
                        self.cur_token.clone(),
                        self.cur_token.literal.clone(),
                    ));
                } else if self.peek_token_is(TokenType::Identifier) {
                    self.next_token();
                    alias = Some(Identifier::new(
                        self.cur_token.clone(),
                        self.cur_token.literal.clone(),
                    ));
                }

                return Some(Expression::SubquerySource(SubqueryTableSource {
                    token: self.cur_token.clone(),
                    subquery: Box::new(subquery),
                    alias,
                }));
            } else if self.cur_token_is_keyword("VALUES") {
                // Parse VALUES clause as table source
                return self.parse_values_table_source();
            }
        }

        // Parse table name - accept both identifiers and keywords (for CTE references like 'first')
        if !self.cur_token_is(TokenType::Identifier) && !self.cur_token_is(TokenType::Keyword) {
            self.add_error(format!(
                "expected table name at {}",
                self.cur_token.position
            ));
            return None;
        }

        let token = self.cur_token.clone();
        let name = Identifier::new(token.clone(), self.cur_token.literal.clone());

        // Check for AS OF clause (temporal queries)
        let as_of = if self.peek_token_is_keyword("AS") {
            self.next_token(); // consume AS
            if self.peek_token_is_keyword("OF") {
                self.next_token(); // consume OF
                self.next_token(); // move to TRANSACTION or TIMESTAMP

                let as_of_type = self.cur_token.literal.to_uppercase();
                if as_of_type != "TRANSACTION" && as_of_type != "TIMESTAMP" {
                    self.add_error(format!(
                        "expected TRANSACTION or TIMESTAMP after AS OF at {}",
                        self.cur_token.position
                    ));
                    return None;
                }

                self.next_token();
                let value = self.parse_expression(Precedence::Lowest)?;

                Some(AsOfClause {
                    token: self.cur_token.clone(),
                    as_of_type,
                    value: Box::new(value),
                })
            } else {
                // This is an alias starting with AS
                None
            }
        } else {
            None
        };

        // Check for alias (can occur after AS OF or after table name)
        let mut alias = None;
        if self.peek_token_is_keyword("AS") {
            self.next_token(); // consume AS
            if !self.expect_peek(TokenType::Identifier) {
                return None;
            }
            alias = Some(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ));
        } else if self.peek_token_is(TokenType::Identifier) && as_of.is_none() {
            // Check if this might be a join keyword or clause keyword
            let peek_upper = self.peek_token.literal.to_uppercase();
            if !matches!(
                peek_upper.as_str(),
                "JOIN"
                    | "LEFT"
                    | "RIGHT"
                    | "INNER"
                    | "OUTER"
                    | "CROSS"
                    | "NATURAL"
                    | "ON"
                    | "WHERE"
                    | "GROUP"
                    | "HAVING"
                    | "ORDER"
                    | "LIMIT"
                    | "OFFSET"
                    | "FETCH"
                    | "UNION"
                    | "INTERSECT"
                    | "EXCEPT"
            ) {
                self.next_token();
                alias = Some(Identifier::new(
                    self.cur_token.clone(),
                    self.cur_token.literal.clone(),
                ));
            }
        }

        Some(Expression::TableSource(SimpleTableSource {
            token,
            name,
            alias,
            as_of,
        }))
    }

    /// Parse a JOIN table expression
    fn parse_join_table_expression(&mut self, mut left: Expression) -> Option<Expression> {
        loop {
            // Check for JOIN keywords
            let join_type = if self.peek_token_is_keyword("JOIN") {
                self.next_token();
                "INNER".to_string()
            } else if self.peek_token_is_keyword("INNER") {
                self.next_token();
                if !self.expect_keyword("JOIN") {
                    return None;
                }
                "INNER".to_string()
            } else if self.peek_token_is_keyword("LEFT") {
                self.next_token();
                if self.peek_token_is_keyword("OUTER") {
                    self.next_token();
                }
                if !self.expect_keyword("JOIN") {
                    return None;
                }
                "LEFT".to_string()
            } else if self.peek_token_is_keyword("RIGHT") {
                self.next_token();
                if self.peek_token_is_keyword("OUTER") {
                    self.next_token();
                }
                if !self.expect_keyword("JOIN") {
                    return None;
                }
                "RIGHT".to_string()
            } else if self.peek_token_is_keyword("FULL") {
                self.next_token();
                if self.peek_token_is_keyword("OUTER") {
                    self.next_token();
                }
                if !self.expect_keyword("JOIN") {
                    return None;
                }
                "FULL".to_string()
            } else if self.peek_token_is_keyword("CROSS") {
                self.next_token();
                if !self.expect_keyword("JOIN") {
                    return None;
                }
                "CROSS".to_string()
            } else if self.peek_token_is_keyword("NATURAL") {
                self.next_token();
                let natural_type = if self.peek_token_is_keyword("LEFT") {
                    self.next_token();
                    if self.peek_token_is_keyword("OUTER") {
                        self.next_token();
                    }
                    "NATURAL LEFT"
                } else if self.peek_token_is_keyword("RIGHT") {
                    self.next_token();
                    if self.peek_token_is_keyword("OUTER") {
                        self.next_token();
                    }
                    "NATURAL RIGHT"
                } else {
                    "NATURAL"
                };
                if !self.expect_keyword("JOIN") {
                    return None;
                }
                natural_type.to_string()
            } else if self.peek_token_is_punctuator(",") {
                // Implicit CROSS JOIN with comma syntax: FROM t1, t2
                self.next_token(); // consume comma
                "CROSS".to_string()
            } else {
                // No more joins
                break;
            };

            let token = self.cur_token.clone();
            self.next_token();
            let right = self.parse_simple_table_expression()?;

            // Parse ON or USING clause (not for CROSS JOIN or NATURAL JOIN)
            let mut condition = None;
            let mut using_columns = Vec::new();

            if !join_type.starts_with("CROSS") && !join_type.starts_with("NATURAL") {
                if self.peek_token_is_keyword("ON") {
                    self.next_token(); // consume ON
                    self.next_token();
                    condition = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
                } else if self.peek_token_is_keyword("USING") {
                    self.next_token(); // consume USING
                    if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
                        self.add_error(format!(
                            "expected '(' after USING at {}",
                            self.cur_token.position
                        ));
                        return None;
                    }
                    using_columns = self.parse_identifier_list();
                    if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                        self.add_error(format!(
                            "expected ')' after USING columns at {}",
                            self.cur_token.position
                        ));
                        return None;
                    }
                }
            }

            left = Expression::JoinSource(Box::new(JoinTableSource {
                token,
                left: Box::new(left),
                join_type,
                right: Box::new(right),
                condition,
                using_columns,
            }));
        }

        Some(left)
    }

    /// Parse a WITH statement (CTE)
    fn parse_with_statement(&mut self) -> Option<Statement> {
        let with_clause = self.parse_with_clause()?;

        // After WITH, expect SELECT or INSERT
        self.next_token();
        if self.cur_token_is_keyword("SELECT") {
            let mut select = self.parse_select_statement()?;
            select.with = Some(with_clause);
            Some(Statement::Select(select))
        } else if self.cur_token_is_keyword("INSERT") {
            // WITH ... INSERT INTO ... SELECT
            let mut insert = self.parse_insert_statement()?;
            // The INSERT must use SELECT (not VALUES) for CTE to make sense
            if let Some(ref mut select) = insert.select {
                select.with = Some(with_clause);
            } else {
                self.add_error(
                    "WITH clause requires INSERT ... SELECT, not INSERT ... VALUES".to_string(),
                );
                return None;
            }
            Some(Statement::Insert(insert))
        } else {
            self.add_error(format!(
                "expected SELECT or INSERT after WITH clause at {}",
                self.cur_token.position
            ));
            None
        }
    }

    /// Parse a WITH clause
    fn parse_with_clause(&mut self) -> Option<WithClause> {
        let token = self.cur_token.clone();
        let mut is_recursive = false;

        // Check for RECURSIVE
        if self.peek_token_is_keyword("RECURSIVE") {
            self.next_token();
            is_recursive = true;
        }

        let mut ctes = Vec::new();

        // Parse first CTE
        self.next_token();
        if let Some(cte) = self.parse_common_table_expression(is_recursive) {
            ctes.push(cte);
        }

        // Parse additional CTEs
        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token(); // move to CTE name
            if let Some(cte) = self.parse_common_table_expression(is_recursive) {
                ctes.push(cte);
            }
        }

        Some(WithClause {
            token,
            ctes,
            is_recursive,
        })
    }

    /// Parse a Common Table Expression
    fn parse_common_table_expression(
        &mut self,
        is_recursive: bool,
    ) -> Option<CommonTableExpression> {
        // Accept both identifiers and keywords as CTE names (context-dependent identifiers)
        // Keywords like FIRST, LAST, VALUE, etc. are valid CTE names in SQL
        if !self.cur_token_is(TokenType::Identifier) && !self.cur_token_is(TokenType::Keyword) {
            self.add_error(format!("expected CTE name at {}", self.cur_token.position));
            return None;
        }

        let token = self.cur_token.clone();
        let name = Identifier::new(token.clone(), self.cur_token.literal.clone());

        // Optional column list
        let mut column_names = Vec::new();
        if self.peek_token_is_punctuator("(") {
            self.next_token(); // consume (
            column_names = self.parse_identifier_list();
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                self.add_error(format!(
                    "expected ')' after column list at {}",
                    self.cur_token.position
                ));
                return None;
            }
        }

        // Expect AS
        if !self.expect_keyword("AS") {
            return None;
        }

        // Expect (
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!(
                "expected '(' after AS at {}",
                self.cur_token.position
            ));
            return None;
        }

        // Parse the CTE query
        self.next_token();
        if !self.cur_token_is_keyword("SELECT") {
            self.add_error(format!(
                "expected SELECT in CTE at {}",
                self.cur_token.position
            ));
            return None;
        }

        let query = self.parse_select_statement()?;

        // Expect )
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after CTE query at {}",
                self.cur_token.position
            ));
            return None;
        }

        Some(CommonTableExpression {
            token,
            name,
            column_names,
            query: Box::new(query),
            is_recursive,
        })
    }

    /// Parse an INSERT statement
    fn parse_insert_statement(&mut self) -> Option<InsertStatement> {
        let token = self.cur_token.clone();

        // Expect INTO
        if !self.expect_keyword("INTO") {
            return None;
        }

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Parse optional column list
        let mut columns = Vec::new();
        if self.peek_token_is_punctuator("(") {
            self.next_token(); // consume (
            columns = self.parse_identifier_list();
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                self.add_error(format!("expected ')' at {}", self.cur_token.position));
                return None;
            }
        }

        // Check if next is VALUES, SELECT, or WITH (CTE)
        if self.peek_token_is_keyword("SELECT") {
            // INSERT INTO ... SELECT
            self.next_token(); // consume SELECT (we're now on SELECT)
            let select_stmt = self.parse_select_statement()?;

            // Parse optional RETURNING clause
            let returning = self.parse_returning_clause();

            return Some(InsertStatement {
                token,
                table_name,
                columns,
                values: Vec::new(),
                select: Some(Box::new(select_stmt)),
                on_duplicate: false,
                update_columns: Vec::new(),
                update_expressions: Vec::new(),
                returning,
            });
        }

        if self.peek_token_is_keyword("WITH") {
            // INSERT INTO ... WITH ... SELECT
            self.next_token(); // consume WITH
            let with_clause = self.parse_with_clause()?;

            // Expect SELECT after WITH clause
            if !self.expect_keyword("SELECT") {
                return None;
            }

            // Parse the SELECT statement
            let mut select_stmt = self.parse_select_statement()?;

            // Attach the WITH clause to the SELECT
            select_stmt.with = Some(with_clause);

            // Parse optional RETURNING clause
            let returning = self.parse_returning_clause();

            return Some(InsertStatement {
                token,
                table_name,
                columns,
                values: Vec::new(),
                select: Some(Box::new(select_stmt)),
                on_duplicate: false,
                update_columns: Vec::new(),
                update_expressions: Vec::new(),
                returning,
            });
        }

        // Expect VALUES
        if !self.expect_keyword("VALUES") {
            return None;
        }

        // Parse value lists
        let values = self.parse_value_lists()?;

        // Check for ON DUPLICATE KEY UPDATE
        let mut on_duplicate = false;
        let mut update_columns = Vec::new();
        let mut update_expressions = Vec::new();

        if self.peek_token_is_keyword("ON") {
            self.next_token(); // consume ON
            if !self.expect_keyword("DUPLICATE") {
                return None;
            }
            if !self.expect_keyword("KEY") {
                return None;
            }
            if !self.expect_keyword("UPDATE") {
                return None;
            }

            on_duplicate = true;

            loop {
                if !self.expect_peek(TokenType::Identifier) {
                    return None;
                }
                let col = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                update_columns.push(col);

                if !self.expect_peek(TokenType::Operator) || self.cur_token.literal != "=" {
                    self.add_error(format!("expected '=' at {}", self.cur_token.position));
                    return None;
                }

                self.next_token();
                let expr = self.parse_expression(Precedence::Lowest)?;
                update_expressions.push(expr);

                if !self.peek_token_is_punctuator(",") {
                    break;
                }
                self.next_token(); // consume comma
            }
        }

        // Parse optional RETURNING clause
        let returning = self.parse_returning_clause();

        Some(InsertStatement {
            token,
            table_name,
            columns,
            values,
            select: None,
            on_duplicate,
            update_columns,
            update_expressions,
            returning,
        })
    }

    /// Parse RETURNING clause for INSERT/UPDATE/DELETE statements
    fn parse_returning_clause(&mut self) -> Vec<Expression> {
        if !self.peek_token_is_keyword("RETURNING") {
            return Vec::new();
        }

        self.next_token(); // consume RETURNING

        // Parse the expression list
        self.parse_expression_list()
    }

    /// Parse value lists for INSERT
    fn parse_value_lists(&mut self) -> Option<Vec<Vec<Expression>>> {
        let mut value_lists = Vec::new();

        // Expect (
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!("expected '(' at {}", self.cur_token.position));
            return None;
        }

        // Parse first value list
        let values = self.parse_expression_list();
        value_lists.push(values);

        // Expect )
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!("expected ')' at {}", self.cur_token.position));
            return None;
        }

        // Parse additional value lists
        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma

            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
                self.add_error(format!("expected '(' at {}", self.cur_token.position));
                return None;
            }

            let values = self.parse_expression_list();
            value_lists.push(values);

            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                self.add_error(format!("expected ')' at {}", self.cur_token.position));
                return None;
            }
        }

        Some(value_lists)
    }

    /// Parse VALUES clause as a table source (e.g., (VALUES (1, 'a'), (2, 'b')) AS t(col1, col2))
    fn parse_values_table_source(&mut self) -> Option<Expression> {
        let token = self.cur_token.clone(); // VALUES token

        // Parse value lists - we're already on VALUES, call parse_value_lists which expects (
        let rows = self.parse_value_lists()?;

        // Expect ) to close the outer parenthesis
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!(
                "expected ')' after VALUES at {}",
                self.cur_token.position
            ));
            return None;
        }

        // Parse optional alias
        let mut alias = None;
        let mut column_aliases = Vec::new();

        if self.peek_token_is_keyword("AS") {
            self.next_token(); // consume AS
            if !self.expect_peek(TokenType::Identifier) {
                self.add_error(format!(
                    "expected alias after AS at {}",
                    self.cur_token.position
                ));
                return None;
            }
            alias = Some(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ));

            // Parse optional column aliases: AS t(col1, col2)
            if self.peek_token_is_punctuator("(") {
                self.next_token(); // consume (
                column_aliases = self.parse_identifier_list();
                if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                    self.add_error(format!(
                        "expected ')' after column aliases at {}",
                        self.cur_token.position
                    ));
                    return None;
                }
            }
        } else if self.peek_token_is(TokenType::Identifier) {
            // Implicit alias without AS
            self.next_token();
            alias = Some(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ));

            // Parse optional column aliases
            if self.peek_token_is_punctuator("(") {
                self.next_token(); // consume (
                column_aliases = self.parse_identifier_list();
                if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                    self.add_error(format!(
                        "expected ')' after column aliases at {}",
                        self.cur_token.position
                    ));
                    return None;
                }
            }
        }

        Some(Expression::ValuesSource(ValuesTableSource {
            token,
            rows,
            alias,
            column_aliases,
        }))
    }

    /// Parse an UPDATE statement
    fn parse_update_statement(&mut self) -> Option<UpdateStatement> {
        let token = self.cur_token.clone();

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Expect SET
        if !self.expect_keyword("SET") {
            return None;
        }

        // Parse column-value pairs
        let mut updates = HashMap::new();
        loop {
            if !self.expect_peek(TokenType::Identifier) {
                return None;
            }
            let column_name = self.cur_token.literal.clone();

            if !self.expect_peek(TokenType::Operator) || self.cur_token.literal != "=" {
                self.add_error(format!("expected '=' at {}", self.cur_token.position));
                return None;
            }

            self.next_token();
            let value_expr = self.parse_expression(Precedence::Lowest)?;
            updates.insert(column_name, value_expr);

            if !self.peek_token_is_punctuator(",") {
                break;
            }
            self.next_token(); // consume comma
        }

        // Parse WHERE clause
        let where_clause = if self.peek_token_is_keyword("WHERE") {
            self.next_token(); // consume WHERE
            self.current_clause = "WHERE".to_string();
            self.next_token();
            Some(Box::new(self.parse_expression(Precedence::Lowest)?))
        } else {
            None
        };

        self.current_clause.clear();

        // Parse optional RETURNING clause
        let returning = self.parse_returning_clause();

        Some(UpdateStatement {
            token,
            table_name,
            updates,
            where_clause,
            returning,
        })
    }

    /// Parse a DELETE statement
    /// DELETE FROM table [AS alias] [WHERE condition] [RETURNING ...]
    fn parse_delete_statement(&mut self) -> Option<DeleteStatement> {
        let token = self.cur_token.clone();

        // Expect FROM
        if !self.expect_keyword("FROM") {
            return None;
        }

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Parse optional alias (AS alias or just alias)
        let alias = if self.peek_token_is_keyword("AS") {
            self.next_token(); // consume AS
            if !self.expect_peek(TokenType::Identifier) {
                return None;
            }
            Some(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ))
        } else if self.peek_token_is(TokenType::Identifier)
            && !self.peek_token_is_keyword("WHERE")
            && !self.peek_token_is_keyword("RETURNING")
        {
            // Alias without AS keyword (e.g., DELETE FROM users u WHERE ...)
            self.next_token();
            Some(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ))
        } else {
            None
        };

        // Parse WHERE clause
        let where_clause = if self.peek_token_is_keyword("WHERE") {
            self.next_token(); // consume WHERE
            self.current_clause = "WHERE".to_string();
            self.next_token();
            Some(Box::new(self.parse_expression(Precedence::Lowest)?))
        } else {
            None
        };

        self.current_clause.clear();

        // Parse optional RETURNING clause
        let returning = self.parse_returning_clause();

        Some(DeleteStatement {
            token,
            table_name,
            alias,
            where_clause,
            returning,
        })
    }

    /// Parse a TRUNCATE statement
    /// TRUNCATE TABLE table_name or TRUNCATE table_name
    fn parse_truncate_statement(&mut self) -> Option<TruncateStatement> {
        let token = self.cur_token.clone();

        // Optional TABLE keyword
        if self.peek_token_is_keyword("TABLE") {
            self.next_token(); // consume TABLE
        }

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        Some(TruncateStatement { token, table_name })
    }

    /// Parse a CREATE statement
    fn parse_create_statement(&mut self) -> Option<Statement> {
        if self.peek_token_is_keyword("TABLE") {
            self.next_token();
            self.parse_create_table_statement()
                .map(Statement::CreateTable)
        } else if self.peek_token_is_keyword("UNIQUE") {
            self.next_token();
            if !self.expect_keyword("INDEX") {
                return None;
            }
            self.parse_create_index_statement(true)
                .map(Statement::CreateIndex)
        } else if self.peek_token_is_keyword("INDEX") {
            self.next_token();
            self.parse_create_index_statement(false)
                .map(Statement::CreateIndex)
        } else if self.peek_token_is_keyword("COLUMNAR") {
            self.next_token();
            if !self.expect_keyword("INDEX") {
                return None;
            }
            self.parse_create_columnar_index_statement()
                .map(Statement::CreateColumnarIndex)
        } else if self.peek_token_is_keyword("VIEW") {
            self.next_token();
            self.parse_create_view_statement()
                .map(Statement::CreateView)
        } else {
            self.add_error(format!(
                "expected TABLE, INDEX, COLUMNAR INDEX, or VIEW after CREATE at {}",
                self.cur_token.position
            ));
            None
        }
    }

    /// Parse a CREATE TABLE statement
    fn parse_create_table_statement(&mut self) -> Option<CreateTableStatement> {
        let token = self.cur_token.clone();

        // Check for IF NOT EXISTS
        let if_not_exists = if self.peek_token_is_keyword("IF") {
            self.next_token();
            if !self.expect_keyword("NOT") {
                return None;
            }
            if !self.expect_keyword("EXISTS") {
                return None;
            }
            true
        } else {
            false
        };

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Check for AS SELECT (CREATE TABLE ... AS SELECT ...)
        if self.peek_token_is_keyword("AS") {
            self.next_token(); // consume AS
            if !self.expect_keyword("SELECT") {
                return None;
            }
            let select_stmt = self.parse_select_statement()?;
            return Some(CreateTableStatement {
                token,
                table_name,
                if_not_exists,
                columns: Vec::new(),
                table_constraints: Vec::new(),
                as_select: Some(Box::new(select_stmt)),
            });
        }

        // Expect (
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!("expected '(' at {}", self.cur_token.position));
            return None;
        }

        // Parse column definitions and table-level constraints
        let (columns, table_constraints) = self.parse_column_definitions_and_constraints();

        // Expect )
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!("expected ')' at {}", self.cur_token.position));
            return None;
        }

        Some(CreateTableStatement {
            token,
            table_name,
            if_not_exists,
            columns,
            table_constraints,
            as_select: None,
        })
    }

    /// Parse column definitions and table-level constraints
    fn parse_column_definitions_and_constraints(
        &mut self,
    ) -> (Vec<ColumnDefinition>, Vec<TableConstraint>) {
        let mut columns = Vec::new();
        let mut table_constraints = Vec::new();

        self.next_token();

        // First item could be a column or a table constraint
        if let Some(item) = self.parse_column_or_constraint() {
            match item {
                ColumnOrConstraint::Column(col) => columns.push(col),
                ColumnOrConstraint::Constraint(tc) => table_constraints.push(tc),
            }
        }

        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token();
            if let Some(item) = self.parse_column_or_constraint() {
                match item {
                    ColumnOrConstraint::Column(col) => columns.push(col),
                    ColumnOrConstraint::Constraint(tc) => table_constraints.push(tc),
                }
            }
        }

        (columns, table_constraints)
    }

    /// Parse either a column definition or a table-level constraint
    fn parse_column_or_constraint(&mut self) -> Option<ColumnOrConstraint> {
        // Check if this is a table-level constraint (UNIQUE, CHECK, PRIMARY KEY)
        if self.cur_token_is_keyword("UNIQUE") {
            // UNIQUE(col1, col2, ...)
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
                return None;
            }
            let columns = self.parse_constraint_column_list()?;
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                return None;
            }
            return Some(ColumnOrConstraint::Constraint(TableConstraint::Unique(
                columns,
            )));
        }

        if self.cur_token_is_keyword("CHECK") {
            // CHECK(expression)
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
                return None;
            }
            self.next_token();
            let expr = self.parse_expression(Precedence::Lowest)?;
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                return None;
            }
            return Some(ColumnOrConstraint::Constraint(TableConstraint::Check(
                Box::new(expr),
            )));
        }

        if self.cur_token_is_keyword("PRIMARY") {
            // PRIMARY KEY(col1, col2, ...)
            if !self.expect_keyword("KEY") {
                return None;
            }
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
                return None;
            }
            let columns = self.parse_constraint_column_list()?;
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                return None;
            }
            return Some(ColumnOrConstraint::Constraint(TableConstraint::PrimaryKey(
                columns,
            )));
        }

        // Otherwise, parse as a column definition
        self.parse_column_definition()
            .map(ColumnOrConstraint::Column)
    }

    /// Parse a comma-separated list of column identifiers (for UNIQUE(col1, col2) etc.)
    fn parse_constraint_column_list(&mut self) -> Option<Vec<Identifier>> {
        let mut identifiers = Vec::new();

        self.next_token();
        if !self.cur_token_is_identifier_like() {
            self.add_error(format!(
                "expected column name at {}",
                self.cur_token.position
            ));
            return None;
        }
        identifiers.push(Identifier::new(
            self.cur_token.clone(),
            self.cur_token.literal.clone(),
        ));

        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token();
            if !self.cur_token_is_identifier_like() {
                self.add_error(format!(
                    "expected column name at {}",
                    self.cur_token.position
                ));
                return None;
            }
            identifiers.push(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ));
        }

        Some(identifiers)
    }

    /// Parse a single column definition
    fn parse_column_definition(&mut self) -> Option<ColumnDefinition> {
        // Allow both identifiers and non-reserved keywords as column names
        if !self.cur_token_is_identifier_like() {
            // Check if it's a reserved keyword and give a better error message
            if self.cur_token.token_type == TokenType::Keyword
                && Self::is_reserved_keyword(&self.cur_token.literal)
            {
                self.add_error(format!(
                    "'{}' is a reserved keyword and cannot be used as a column name. Use double quotes to escape it: \"{}\"",
                    self.cur_token.literal.to_uppercase(),
                    self.cur_token.literal
                ));
            } else {
                self.add_error(format!(
                    "expected column name at {}",
                    self.cur_token.position
                ));
            }
            return None;
        }

        let name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Parse data type
        if !self.expect_peek(TokenType::Keyword) {
            return None;
        }
        let data_type = self.cur_token.literal.to_uppercase();

        // Handle DECIMAL(precision, scale) and NUMERIC(precision, scale) syntax
        // We parse and ignore the precision/scale, mapping to FLOAT internally
        if (data_type == "DECIMAL" || data_type == "NUMERIC") && self.peek_token_is_punctuator("(")
        {
            self.next_token(); // consume (
                               // Skip precision
            if self.peek_token.token_type == TokenType::Integer {
                self.next_token();
            }
            // Skip comma and scale if present
            if self.peek_token_is_punctuator(",") {
                self.next_token(); // consume ,
                if self.peek_token.token_type == TokenType::Integer {
                    self.next_token(); // consume scale
                }
            }
            // Consume closing )
            if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                self.add_error("expected ) after DECIMAL precision/scale".to_string());
                return None;
            }
        }

        // Parse constraints
        let mut constraints = Vec::new();
        while self.peek_token_is(TokenType::Keyword) {
            let constraint_keyword = self.peek_token.literal.to_uppercase();
            match constraint_keyword.as_str() {
                "PRIMARY" => {
                    self.next_token(); // consume PRIMARY
                    if !self.expect_keyword("KEY") {
                        return None;
                    }
                    constraints.push(ColumnConstraint::PrimaryKey);
                }
                "NOT" => {
                    self.next_token(); // consume NOT
                    if !self.expect_keyword("NULL") {
                        return None;
                    }
                    constraints.push(ColumnConstraint::NotNull);
                }
                "UNIQUE" => {
                    self.next_token();
                    constraints.push(ColumnConstraint::Unique);
                }
                "DEFAULT" => {
                    self.next_token(); // consume DEFAULT
                    self.next_token();
                    let expr = self.parse_expression(Precedence::Lowest)?;
                    constraints.push(ColumnConstraint::Default(expr));
                }
                "CHECK" => {
                    self.next_token(); // consume CHECK
                    if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
                        return None;
                    }
                    self.next_token();
                    let expr = self.parse_expression(Precedence::Lowest)?;
                    if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
                        return None;
                    }
                    constraints.push(ColumnConstraint::Check(expr));
                }
                "REFERENCES" => {
                    self.next_token(); // consume REFERENCES
                    if !self.expect_peek(TokenType::Identifier) {
                        return None;
                    }
                    let ref_table =
                        Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                    let ref_column = if self.peek_token_is_punctuator("(") {
                        self.next_token();
                        if !self.expect_peek(TokenType::Identifier) {
                            return None;
                        }
                        let col =
                            Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")"
                        {
                            return None;
                        }
                        Some(col)
                    } else {
                        None
                    };
                    constraints.push(ColumnConstraint::References(ref_table, ref_column));
                }
                "AUTO_INCREMENT" | "AUTOINCREMENT" => {
                    constraints.push(ColumnConstraint::AutoIncrement);
                    self.next_token();
                }
                _ => break,
            }
        }

        Some(ColumnDefinition {
            name,
            data_type,
            constraints,
        })
    }

    /// Parse a CREATE INDEX statement
    fn parse_create_index_statement(&mut self, is_unique: bool) -> Option<CreateIndexStatement> {
        let token = self.cur_token.clone();

        // Check for IF NOT EXISTS
        let if_not_exists = if self.peek_token_is_keyword("IF") {
            self.next_token();
            if !self.expect_keyword("NOT") {
                return None;
            }
            if !self.expect_keyword("EXISTS") {
                return None;
            }
            true
        } else {
            false
        };

        // Parse index name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let index_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Expect ON
        if !self.expect_keyword("ON") {
            return None;
        }

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Expect (
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!("expected '(' at {}", self.cur_token.position));
            return None;
        }

        // Parse column list
        let columns = self.parse_identifier_list();

        // Expect )
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!("expected ')' at {}", self.cur_token.position));
            return None;
        }

        // Parse optional USING clause
        let index_method = if self.peek_token_is_keyword("USING") {
            self.next_token(); // consume USING
            self.next_token(); // move to method name

            let method_name = self.cur_token.literal.to_uppercase();
            match method_name.as_str() {
                "BTREE" | "B_TREE" => Some(IndexMethod::BTree),
                "HASH" => Some(IndexMethod::Hash),
                "BITMAP" => Some(IndexMethod::Bitmap),
                _ => {
                    self.add_error(format!(
                        "unknown index method '{}'. Supported methods: BTREE, HASH, BITMAP",
                        self.cur_token.literal
                    ));
                    return None;
                }
            }
        } else {
            None
        };

        Some(CreateIndexStatement {
            token,
            index_name,
            table_name,
            columns,
            is_unique,
            if_not_exists,
            index_method,
        })
    }

    /// Parse a CREATE COLUMNAR INDEX statement
    fn parse_create_columnar_index_statement(&mut self) -> Option<CreateColumnarIndexStatement> {
        let token = self.cur_token.clone();

        // Check for IF NOT EXISTS
        let if_not_exists = if self.peek_token_is_keyword("IF") {
            self.next_token();
            if !self.expect_keyword("NOT") {
                return None;
            }
            if !self.expect_keyword("EXISTS") {
                return None;
            }
            true
        } else {
            false
        };

        // Expect ON
        if !self.expect_keyword("ON") {
            return None;
        }

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Expect (
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!("expected '(' at {}", self.cur_token.position));
            return None;
        }

        // Parse column name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let column_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Expect )
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!("expected ')' at {}", self.cur_token.position));
            return None;
        }

        Some(CreateColumnarIndexStatement {
            token,
            table_name,
            column_name,
            if_not_exists,
            is_unique: false,
        })
    }

    /// Parse a CREATE VIEW statement
    fn parse_create_view_statement(&mut self) -> Option<CreateViewStatement> {
        let token = self.cur_token.clone();

        // Check for IF NOT EXISTS
        let if_not_exists = if self.peek_token_is_keyword("IF") {
            self.next_token();
            if !self.expect_keyword("NOT") {
                return None;
            }
            if !self.expect_keyword("EXISTS") {
                return None;
            }
            true
        } else {
            false
        };

        // Parse view name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let view_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Expect AS
        if !self.expect_keyword("AS") {
            return None;
        }

        // Expect SELECT or WITH (for CTEs)
        self.next_token();
        let query = if self.cur_token_is_keyword("SELECT") {
            self.parse_select_statement()?
        } else if self.cur_token_is_keyword("WITH") {
            // Parse CTE and attach to SELECT
            let with_clause = self.parse_with_clause()?;
            self.next_token();
            if !self.cur_token_is_keyword("SELECT") {
                self.add_error(format!(
                    "expected SELECT after WITH clause in CREATE VIEW at {}",
                    self.cur_token.position
                ));
                return None;
            }
            let mut select = self.parse_select_statement()?;
            select.with = Some(with_clause);
            select
        } else {
            self.add_error(format!(
                "expected SELECT or WITH after AS in CREATE VIEW at {}",
                self.cur_token.position
            ));
            return None;
        };

        Some(CreateViewStatement {
            token,
            view_name,
            query: Box::new(query),
            if_not_exists,
        })
    }

    /// Parse a DROP statement
    fn parse_drop_statement(&mut self) -> Option<Statement> {
        if self.peek_token_is_keyword("TABLE") {
            self.next_token();
            self.parse_drop_table_statement().map(Statement::DropTable)
        } else if self.peek_token_is_keyword("INDEX") {
            self.next_token();
            self.parse_drop_index_statement().map(Statement::DropIndex)
        } else if self.peek_token_is_keyword("COLUMNAR") {
            self.next_token();
            if !self.expect_keyword("INDEX") {
                return None;
            }
            self.parse_drop_columnar_index_statement()
                .map(Statement::DropColumnarIndex)
        } else if self.peek_token_is_keyword("VIEW") {
            self.next_token();
            self.parse_drop_view_statement().map(Statement::DropView)
        } else {
            self.add_error(format!(
                "expected TABLE, INDEX, COLUMNAR INDEX, or VIEW after DROP at {}",
                self.cur_token.position
            ));
            None
        }
    }

    /// Parse a DROP TABLE statement
    fn parse_drop_table_statement(&mut self) -> Option<DropTableStatement> {
        let token = self.cur_token.clone();

        // Check for IF EXISTS
        let if_exists = if self.peek_token_is_keyword("IF") {
            self.next_token();
            if !self.expect_keyword("EXISTS") {
                return None;
            }
            true
        } else {
            false
        };

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        Some(DropTableStatement {
            token,
            table_name,
            if_exists,
        })
    }

    /// Parse a DROP INDEX statement
    fn parse_drop_index_statement(&mut self) -> Option<DropIndexStatement> {
        let token = self.cur_token.clone();

        // Check for IF EXISTS
        let if_exists = if self.peek_token_is_keyword("IF") {
            self.next_token();
            if !self.expect_keyword("EXISTS") {
                return None;
            }
            true
        } else {
            false
        };

        // Parse index name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let index_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Check for optional ON clause
        let table_name = if self.peek_token_is_keyword("ON") {
            self.next_token();
            if !self.expect_peek(TokenType::Identifier) {
                return None;
            }
            Some(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ))
        } else {
            None
        };

        Some(DropIndexStatement {
            token,
            index_name,
            table_name,
            if_exists,
        })
    }

    /// Parse a DROP COLUMNAR INDEX statement
    fn parse_drop_columnar_index_statement(&mut self) -> Option<DropColumnarIndexStatement> {
        let token = self.cur_token.clone();

        // Check for IF EXISTS
        let if_exists = if self.peek_token_is_keyword("IF") {
            self.next_token();
            if !self.expect_keyword("EXISTS") {
                return None;
            }
            true
        } else {
            false
        };

        // Expect ON
        if !self.expect_keyword("ON") {
            return None;
        }

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Expect (
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != "(" {
            self.add_error(format!("expected '(' at {}", self.cur_token.position));
            return None;
        }

        // Parse column name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let column_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Expect )
        if !self.expect_peek(TokenType::Punctuator) || self.cur_token.literal != ")" {
            self.add_error(format!("expected ')' at {}", self.cur_token.position));
            return None;
        }

        Some(DropColumnarIndexStatement {
            token,
            table_name,
            column_name,
            if_exists,
        })
    }

    /// Parse a DROP VIEW statement
    fn parse_drop_view_statement(&mut self) -> Option<DropViewStatement> {
        let token = self.cur_token.clone();

        // Check for IF EXISTS
        let if_exists = if self.peek_token_is_keyword("IF") {
            self.next_token();
            if !self.expect_keyword("EXISTS") {
                return None;
            }
            true
        } else {
            false
        };

        // Parse view name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let view_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        Some(DropViewStatement {
            token,
            view_name,
            if_exists,
        })
    }

    /// Parse an ALTER statement
    fn parse_alter_statement(&mut self) -> Option<AlterTableStatement> {
        let token = self.cur_token.clone();

        // Expect TABLE
        if !self.expect_keyword("TABLE") {
            return None;
        }

        // Parse table name
        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }
        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        // Parse operation
        if !self.expect_peek(TokenType::Keyword) {
            return None;
        }

        let operation_keyword = self.cur_token.literal.to_uppercase();
        let (operation, column_def, column_name, new_column_name, new_table_name) =
            match operation_keyword.as_str() {
                "ADD" => {
                    // Check for optional COLUMN keyword
                    if self.peek_token_is_keyword("COLUMN") {
                        self.next_token();
                    }
                    self.next_token();
                    let col_def = self.parse_column_definition()?;
                    (
                        AlterTableOperation::AddColumn,
                        Some(col_def),
                        None,
                        None,
                        None,
                    )
                }
                "DROP" => {
                    // Check for optional COLUMN keyword
                    if self.peek_token_is_keyword("COLUMN") {
                        self.next_token();
                    }
                    if !self.expect_peek(TokenType::Identifier) {
                        return None;
                    }
                    let col_name =
                        Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                    (
                        AlterTableOperation::DropColumn,
                        None,
                        Some(col_name),
                        None,
                        None,
                    )
                }
                "RENAME" => {
                    if !self.expect_peek(TokenType::Keyword) {
                        return None;
                    }
                    let rename_keyword = self.cur_token.literal.to_uppercase();
                    if rename_keyword == "COLUMN" {
                        if !self.expect_peek(TokenType::Identifier) {
                            return None;
                        }
                        let col_name =
                            Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                        if !self.expect_keyword("TO") {
                            return None;
                        }
                        if !self.expect_peek(TokenType::Identifier) {
                            return None;
                        }
                        let new_col_name =
                            Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                        (
                            AlterTableOperation::RenameColumn,
                            None,
                            Some(col_name),
                            Some(new_col_name),
                            None,
                        )
                    } else if rename_keyword == "TO" {
                        if !self.expect_peek(TokenType::Identifier) {
                            return None;
                        }
                        let new_tbl_name =
                            Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                        (
                            AlterTableOperation::RenameTable,
                            None,
                            None,
                            None,
                            Some(new_tbl_name),
                        )
                    } else {
                        self.add_error(format!(
                            "expected COLUMN or TO after RENAME at {}",
                            self.cur_token.position
                        ));
                        return None;
                    }
                }
                "MODIFY" => {
                    // Check for optional COLUMN keyword
                    if self.peek_token_is_keyword("COLUMN") {
                        self.next_token();
                    }
                    self.next_token();
                    let col_def = self.parse_column_definition()?;
                    (
                        AlterTableOperation::ModifyColumn,
                        Some(col_def),
                        None,
                        None,
                        None,
                    )
                }
                _ => {
                    self.add_error(format!(
                        "expected ADD, DROP, RENAME, or MODIFY at {}",
                        self.cur_token.position
                    ));
                    return None;
                }
            };

        Some(AlterTableStatement {
            token,
            table_name,
            operation,
            column_def,
            column_name,
            new_column_name,
            new_table_name,
        })
    }

    /// Parse a BEGIN statement
    fn parse_begin_statement(&mut self) -> Option<BeginStatement> {
        let token = self.cur_token.clone();

        // Check for optional TRANSACTION keyword
        if self.peek_token_is_keyword("TRANSACTION") {
            self.next_token();
        }

        // Check for ISOLATION LEVEL
        let isolation_level = if self.peek_token_is_keyword("ISOLATION") {
            self.next_token();
            if !self.expect_keyword("LEVEL") {
                return None;
            }
            self.next_token();

            let level = self.cur_token.literal.to_uppercase();
            let isolation = match level.as_str() {
                "SNAPSHOT" | "SERIALIZABLE" => level,
                "REPEATABLE" => {
                    if self.peek_token_is_keyword("READ") {
                        self.next_token();
                    }
                    "REPEATABLE READ".to_string()
                }
                "READ" => {
                    if self.peek_token_is_keyword("UNCOMMITTED") {
                        self.next_token();
                        "READ UNCOMMITTED".to_string()
                    } else if self.peek_token_is_keyword("COMMITTED") {
                        self.next_token();
                        "READ COMMITTED".to_string()
                    } else {
                        self.add_error(format!(
                            "expected UNCOMMITTED or COMMITTED after READ at {}",
                            self.cur_token.position
                        ));
                        return None;
                    }
                }
                _ => {
                    self.add_error(format!(
                        "invalid isolation level: {} at {}",
                        level, self.cur_token.position
                    ));
                    return None;
                }
            };
            Some(isolation)
        } else {
            None
        };

        Some(BeginStatement {
            token,
            isolation_level,
        })
    }

    /// Parse a COMMIT statement
    fn parse_commit_statement(&mut self) -> Option<CommitStatement> {
        let token = self.cur_token.clone();

        // Check for optional TRANSACTION keyword
        if self.peek_token_is_keyword("TRANSACTION") {
            self.next_token();
        }

        Some(CommitStatement { token })
    }

    /// Parse a ROLLBACK statement
    fn parse_rollback_statement(&mut self) -> Option<RollbackStatement> {
        let token = self.cur_token.clone();

        // Check for optional TRANSACTION keyword
        if self.peek_token_is_keyword("TRANSACTION") {
            self.next_token();
        }

        // Check for TO SAVEPOINT clause
        let savepoint_name = if self.peek_token_is_keyword("TO") {
            self.next_token();
            if self.peek_token_is_keyword("SAVEPOINT") {
                self.next_token();
            }
            if !self.expect_peek(TokenType::Identifier) {
                return None;
            }
            Some(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ))
        } else {
            None
        };

        Some(RollbackStatement {
            token,
            savepoint_name,
        })
    }

    /// Parse a SAVEPOINT statement
    fn parse_savepoint_statement(&mut self) -> Option<SavepointStatement> {
        let token = self.cur_token.clone();

        if !self.expect_peek(TokenType::Identifier) {
            return None;
        }

        let savepoint_name =
            Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        Some(SavepointStatement {
            token,
            savepoint_name,
        })
    }

    /// Parse a SET statement
    fn parse_set_statement(&mut self) -> Option<SetStatement> {
        let token = self.cur_token.clone();

        self.next_token();
        if !self.cur_token_is(TokenType::Identifier) {
            self.add_error(format!(
                "expected variable name at {}",
                self.cur_token.position
            ));
            return None;
        }

        let name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        self.next_token();
        // Expect '=' or 'TO'
        let is_equals = self.cur_token_is(TokenType::Operator) && self.cur_token.literal == "=";
        if !is_equals && !self.cur_token_is_keyword("TO") {
            self.add_error(format!(
                "expected '=' or 'TO' after variable name at {}",
                self.cur_token.position
            ));
            return None;
        }

        self.next_token();
        let value = self.parse_expression(Precedence::Lowest)?;

        Some(SetStatement { token, name, value })
    }

    /// Parse a PRAGMA statement
    fn parse_pragma_statement(&mut self) -> Option<PragmaStatement> {
        let token = self.cur_token.clone();

        self.next_token();
        if !self.cur_token_is(TokenType::Identifier) {
            self.add_error(format!(
                "expected pragma name at {}",
                self.cur_token.position
            ));
            return None;
        }

        let name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        self.next_token();

        // Check for optional value
        let value = if self.cur_token_is(TokenType::Operator) && self.cur_token.literal == "=" {
            self.next_token();
            Some(self.parse_expression(Precedence::Lowest)?)
        } else {
            None
        };

        Some(PragmaStatement { token, name, value })
    }

    /// Parse a SHOW statement
    fn parse_show_statement(&mut self) -> Option<Statement> {
        let token = self.cur_token.clone();

        if self.peek_token_is_keyword("TABLES") {
            self.next_token();
            Some(Statement::ShowTables(ShowTablesStatement { token }))
        } else if self.peek_token_is_keyword("VIEWS") {
            self.next_token();
            Some(Statement::ShowViews(ShowViewsStatement { token }))
        } else if self.peek_token_is_keyword("CREATE") {
            self.next_token();
            // Check for TABLE or VIEW
            if self.peek_token_is_keyword("TABLE") {
                self.next_token();
                if !self.expect_peek(TokenType::Identifier) {
                    return None;
                }
                let table_name =
                    Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                Some(Statement::ShowCreateTable(ShowCreateTableStatement {
                    token,
                    table_name,
                }))
            } else if self.peek_token_is_keyword("VIEW") {
                self.next_token();
                if !self.expect_peek(TokenType::Identifier) {
                    return None;
                }
                let view_name =
                    Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
                Some(Statement::ShowCreateView(ShowCreateViewStatement {
                    token,
                    view_name,
                }))
            } else {
                self.add_error(format!(
                    "expected TABLE or VIEW after SHOW CREATE at {}",
                    self.cur_token.position
                ));
                None
            }
        } else if self.peek_token_is_keyword("INDEXES") || self.peek_token_is_keyword("INDEX") {
            self.next_token();
            if !self.expect_keyword("FROM") {
                return None;
            }
            if !self.expect_peek(TokenType::Identifier) {
                return None;
            }
            let table_name =
                Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());
            Some(Statement::ShowIndexes(ShowIndexesStatement {
                token,
                table_name,
            }))
        } else {
            self.add_error(format!(
                "unsupported SHOW statement at {}",
                self.cur_token.position
            ));
            None
        }
    }

    /// Parse a DESCRIBE statement
    fn parse_describe_statement(&mut self) -> Option<DescribeStatement> {
        let token = self.cur_token.clone();

        // Move past DESCRIBE/DESC keyword
        self.next_token();

        // Optional TABLE keyword (DESCRIBE TABLE t or just DESCRIBE t)
        if self.cur_token_is(TokenType::Keyword) && self.cur_token.literal.to_uppercase() == "TABLE"
        {
            self.next_token();
        }

        // Expect table name
        if !self.cur_token_is(TokenType::Identifier) && !self.cur_token_is(TokenType::Keyword) {
            self.add_error(format!(
                "expected table name after DESCRIBE at {}",
                self.cur_token.position
            ));
            return None;
        }

        let table_name = Identifier::new(self.cur_token.clone(), self.cur_token.literal.clone());

        Some(DescribeStatement { token, table_name })
    }

    /// Parse an EXPLAIN statement
    fn parse_explain_statement(&mut self) -> Option<ExplainStatement> {
        let token = self.cur_token.clone();

        // Check for ANALYZE option
        let analyze = if self.peek_token_is_keyword("ANALYZE") {
            self.next_token();
            true
        } else {
            false
        };

        // Move to the statement to explain
        self.next_token();

        // Parse the inner statement (SELECT, INSERT, UPDATE, DELETE)
        let statement = self.parse_statement()?;

        Some(ExplainStatement {
            token,
            statement: Box::new(statement),
            analyze,
        })
    }

    /// Parse an ANALYZE statement
    /// Syntax: ANALYZE [table_name]
    fn parse_analyze_statement(&mut self) -> Option<AnalyzeStatement> {
        let token = self.cur_token.clone();

        // Move past ANALYZE keyword
        self.next_token();

        // Optional table name
        let table_name = if self.cur_token_is(TokenType::Identifier)
            || (self.cur_token_is(TokenType::Keyword)
                && !self.cur_token.literal.eq_ignore_ascii_case("TABLE"))
        {
            let name = self.cur_token.literal.clone();
            Some(name)
        } else if self.cur_token_is(TokenType::Keyword)
            && self.cur_token.literal.eq_ignore_ascii_case("TABLE")
        {
            // ANALYZE TABLE table_name syntax
            self.next_token();
            if self.cur_token_is(TokenType::Identifier) || self.cur_token_is(TokenType::Keyword) {
                let name = self.cur_token.literal.clone();
                Some(name)
            } else {
                None
            }
        } else {
            None
        };

        Some(AnalyzeStatement { token, table_name })
    }

    /// Parse an expression statement
    fn parse_expression_statement(&mut self) -> Option<ExpressionStatement> {
        let token = self.cur_token.clone();
        let expression = self.parse_expression(Precedence::Lowest)?;

        Some(ExpressionStatement { token, expression })
    }

    /// Parse an identifier list (allows keywords as identifiers for CTE column aliases)
    pub fn parse_identifier_list(&mut self) -> Vec<Identifier> {
        let mut list = Vec::new();

        self.next_token();
        // Accept both identifiers and keywords as column names
        if self.cur_token_is(TokenType::Identifier) || self.cur_token_is(TokenType::Keyword) {
            list.push(Identifier::new(
                self.cur_token.clone(),
                self.cur_token.literal.clone(),
            ));
        }

        while self.peek_token_is_punctuator(",") {
            self.next_token(); // consume comma
            self.next_token(); // move to identifier/keyword
                               // Accept both identifiers and keywords as column names
            if self.cur_token_is(TokenType::Identifier) || self.cur_token_is(TokenType::Keyword) {
                list.push(Identifier::new(
                    self.cur_token.clone(),
                    self.cur_token.literal.clone(),
                ));
            } else {
                self.add_error(format!(
                    "expected Identifier, got {:?} at {}",
                    self.cur_token.token_type, self.cur_token.position
                ));
                return list;
            }
        }

        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_stmt(input: &str) -> Option<Statement> {
        let mut parser = Parser::new(input);
        parser.parse_statement()
    }

    #[test]
    fn test_parse_simple_select() {
        let stmt = parse_stmt("SELECT * FROM users").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.columns.len(), 1);
                assert!(matches!(select.columns[0], Expression::Star(_)));
            }
            _ => panic!("expected SelectStatement"),
        }
    }

    #[test]
    fn test_parse_select_with_where() {
        let stmt = parse_stmt("SELECT id, name FROM users WHERE id = 1").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.columns.len(), 2);
                assert!(select.where_clause.is_some());
            }
            _ => panic!("expected SelectStatement"),
        }
    }

    #[test]
    fn test_parse_count_star_with_filter_in_select() {
        let stmt = parse_stmt("SELECT COUNT(*) FILTER (WHERE category = 'Z') FROM data").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert_eq!(select.columns.len(), 1);
                match &select.columns[0] {
                    Expression::FunctionCall(fc) => {
                        assert_eq!(fc.function.to_uppercase(), "COUNT");
                        assert!(
                            fc.filter.is_some(),
                            "FILTER clause should be parsed for COUNT(*) in SELECT"
                        );
                    }
                    _ => panic!("expected FunctionCall for COUNT(*)"),
                }
            }
            _ => panic!("expected SelectStatement"),
        }
    }

    #[test]
    fn test_parse_select_with_join() {
        let stmt =
            parse_stmt("SELECT u.id FROM users u LEFT JOIN orders o ON u.id = o.user_id").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.table_expr.is_some());
                match select.table_expr.as_ref().unwrap().as_ref() {
                    Expression::JoinSource(_) => {}
                    _ => panic!("expected JoinSource"),
                }
            }
            _ => panic!("expected SelectStatement"),
        }
    }

    #[test]
    fn test_parse_insert() {
        let stmt = parse_stmt("INSERT INTO users (id, name) VALUES (1, 'Alice')").unwrap();
        match stmt {
            Statement::Insert(insert) => {
                assert_eq!(insert.table_name.value, "users");
                assert_eq!(insert.columns.len(), 2);
                assert_eq!(insert.values.len(), 1);
            }
            _ => panic!("expected InsertStatement"),
        }
    }

    #[test]
    fn test_parse_update() {
        let stmt = parse_stmt("UPDATE users SET name = 'Bob' WHERE id = 1").unwrap();
        match stmt {
            Statement::Update(update) => {
                assert_eq!(update.table_name.value, "users");
                assert_eq!(update.updates.len(), 1);
                assert!(update.where_clause.is_some());
            }
            _ => panic!("expected UpdateStatement"),
        }
    }

    #[test]
    fn test_parse_delete() {
        let stmt = parse_stmt("DELETE FROM users WHERE id = 1").unwrap();
        match stmt {
            Statement::Delete(delete) => {
                assert_eq!(delete.table_name.value, "users");
                assert!(delete.where_clause.is_some());
            }
            _ => panic!("expected DeleteStatement"),
        }
    }

    #[test]
    fn test_parse_create_table() {
        let stmt =
            parse_stmt("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)").unwrap();
        match stmt {
            Statement::CreateTable(create) => {
                assert_eq!(create.table_name.value, "users");
                assert_eq!(create.columns.len(), 2);
            }
            _ => panic!("expected CreateTableStatement"),
        }
    }

    #[test]
    fn test_parse_drop_table() {
        let stmt = parse_stmt("DROP TABLE IF EXISTS users").unwrap();
        match stmt {
            Statement::DropTable(drop) => {
                assert_eq!(drop.table_name.value, "users");
                assert!(drop.if_exists);
            }
            _ => panic!("expected DropTableStatement"),
        }
    }

    #[test]
    fn test_parse_begin_commit() {
        let stmt = parse_stmt("BEGIN TRANSACTION").unwrap();
        match stmt {
            Statement::Begin(_) => {}
            _ => panic!("expected BeginStatement"),
        }

        let stmt = parse_stmt("COMMIT").unwrap();
        match stmt {
            Statement::Commit(_) => {}
            _ => panic!("expected CommitStatement"),
        }
    }

    #[test]
    fn test_parse_with_cte() {
        let stmt = parse_stmt("WITH temp AS (SELECT * FROM users) SELECT * FROM temp").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.with.is_some());
                let with = select.with.as_ref().unwrap();
                assert_eq!(with.ctes.len(), 1);
                assert_eq!(with.ctes[0].name.value, "temp");
            }
            _ => panic!("expected SelectStatement"),
        }
    }

    #[test]
    fn test_parse_fetch_first() {
        // FETCH FIRST n ROWS ONLY
        let stmt = parse_stmt("SELECT * FROM users FETCH FIRST 10 ROWS ONLY").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.limit.is_some());
            }
            _ => panic!("expected SelectStatement"),
        }

        // FETCH FIRST n ROW ONLY (singular)
        let stmt = parse_stmt("SELECT * FROM users FETCH FIRST 1 ROW ONLY").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.limit.is_some());
            }
            _ => panic!("expected SelectStatement"),
        }

        // FETCH NEXT n ROWS ONLY
        let stmt = parse_stmt("SELECT * FROM users FETCH NEXT 5 ROWS ONLY").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.limit.is_some());
            }
            _ => panic!("expected SelectStatement"),
        }

        // OFFSET with FETCH
        let stmt =
            parse_stmt("SELECT * FROM users OFFSET 10 ROWS FETCH FIRST 5 ROWS ONLY").unwrap();
        match stmt {
            Statement::Select(select) => {
                assert!(select.limit.is_some());
                assert!(select.offset.is_some());
            }
            _ => panic!("expected SelectStatement"),
        }
    }
}
