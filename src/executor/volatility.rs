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

//! How volatile an expression or a statement is: the most volatile function
//! anywhere in it, by the registry that runs it, and the rows of the running
//! statement's own CTEs. Every match is exhaustive, so a new form of the tree
//! decides here before it compiles.

use crate::functions::{FunctionRegistry, Volatility};
use crate::parser::ast::{
    Expression, FunctionCall, GroupByModifier, OrderByExpression, SelectStatement, WindowFrame,
    WindowFrameBound,
};

/// The volatility of `expr`. A subquery, derived table or CTE counts as one
/// per statement at most, as it runs once for each binding of the rows it
/// reads
pub(crate) fn expression_volatility(expr: &Expression, registry: &FunctionRegistry) -> Volatility {
    Walk {
        registry,
        statement_rows: &|_| false,
        constant: false,
        whole_blocks: false,
    }
    .expression(expr)
}

/// Whether anything `stmt` evaluates, at any depth of the blocks nested in
/// it, draws a new value each time it runs: a block run once for each row
/// of `stmt` draws again for each
pub(crate) fn statement_draws_per_call(
    stmt: &SelectStatement,
    registry: &FunctionRegistry,
) -> bool {
    Walk {
        registry,
        statement_rows: &|_| false,
        constant: false,
        whole_blocks: true,
    }
    .statement(stmt)
        == Volatility::PerCall
}

/// Whether `expr` is one value for the whole statement, so it may be
/// evaluated once with no row: it reads no column, holds no subquery,
/// aggregate or window, and draws nothing per call. A name counts as a
/// column even where none is bound, as a quoted one compiles to its text
pub(crate) fn statement_constant(expr: &Expression, registry: &FunctionRegistry) -> bool {
    Walk {
        registry,
        statement_rows: &|_| false,
        constant: true,
        whole_blocks: false,
    }
    .expression(expr)
        != Volatility::PerCall
}

/// The volatility of everything `stmt` evaluates, its CTEs, set operations
/// and the blocks nested in it included, inside a running statement where a
/// table `name` for which `statement_rows(name)` holds is one of its CTEs:
/// rows that are the statement's own, so one per statement
pub(crate) fn statement_volatility_in(
    stmt: &SelectStatement,
    registry: &FunctionRegistry,
    statement_rows: &dyn Fn(&str) -> bool,
) -> Volatility {
    Walk {
        registry,
        statement_rows,
        constant: false,
        whole_blocks: false,
    }
    .statement(stmt)
}

/// [`expression_volatility`] inside a running statement, as for [`statement_volatility_in`]
pub(crate) fn expression_volatility_in(
    expr: &Expression,
    registry: &FunctionRegistry,
    statement_rows: &dyn Fn(&str) -> bool,
) -> Volatility {
    Walk {
        registry,
        statement_rows,
        constant: false,
        whole_blocks: false,
    }
    .expression(expr)
}

struct Walk<'a> {
    registry: &'a FunctionRegistry,
    /// Whether a table, by its lowercase name, is a CTE of the running statement
    statement_rows: &'a dyn Fn(&str) -> bool,
    /// Asked by [`statement_constant`]: what is not one value for the
    /// statement answers PerCall
    constant: bool,
    /// A nested block keeps its own volatility instead of counting once per
    /// statement at most
    whole_blocks: bool,
}

impl Walk<'_> {
    fn expression(&self, expr: &Expression) -> Volatility {
        let one = |e: &Expression| self.expression(e);
        let optional = |e: Option<&Expression>| e.map_or(Volatility::Fixed, one);
        let varies = if self.constant {
            Volatility::PerCall
        } else {
            Volatility::Fixed
        };
        match expr {
            // An unquoted TRUE or FALSE compiles to the literal
            Expression::Identifier(id)
                if !id.quoted && matches!(id.value_lower.as_str(), "true" | "false") =>
            {
                Volatility::Fixed
            }
            Expression::Identifier(_)
            | Expression::QualifiedIdentifier(_)
            | Expression::Star(_)
            | Expression::QualifiedStar(_)
            | Expression::Default(_) => varies,
            Expression::IntegerLiteral(_)
            | Expression::FloatLiteral(_)
            | Expression::StringLiteral(_)
            | Expression::BooleanLiteral(_)
            | Expression::NullLiteral(_)
            | Expression::IntervalLiteral(_)
            | Expression::Parameter(_) => Volatility::Fixed,
            Expression::CteReference(_) => Volatility::PerStatement,
            Expression::Prefix(prefix) => one(&prefix.right),
            Expression::Infix(infix) => one(&infix.left).max(one(&infix.right)),
            Expression::List(list) => self.all(&list.elements),
            Expression::ExpressionList(list) => self.all(&list.expressions),
            Expression::Distinct(distinct) => one(&distinct.expr),
            Expression::Exists(exists) => self.block(&exists.subquery),
            Expression::AllAny(all_any) => one(&all_any.left).max(self.block(&all_any.subquery)),
            Expression::In(in_expr) => one(&in_expr.left).max(one(&in_expr.right)),
            Expression::InHashSet(in_set) => one(&in_set.column),
            Expression::QuantifiedSet(set) => one(&set.left),
            Expression::Between(between) => one(&between.expr)
                .max(one(&between.lower))
                .max(one(&between.upper)),
            Expression::Like(like) => one(&like.left)
                .max(one(&like.pattern))
                .max(optional(like.escape.as_deref())),
            Expression::ScalarSubquery(subquery) => self.block(&subquery.subquery),
            Expression::Case(case) => optional(case.value.as_deref())
                .max(most(case.when_clauses.iter().map(|when| {
                    one(&when.condition).max(one(&when.then_result))
                })))
                .max(optional(case.else_value.as_deref())),
            Expression::Cast(cast) => one(&cast.expr),
            Expression::FunctionCall(call) => self.call(call),
            Expression::Aliased(aliased) => one(&aliased.expression),
            Expression::Window(window) => self
                .call(&window.function)
                .max(varies)
                .max(self.all(&window.partition_by))
                .max(self.ordered(&window.order_by))
                .max(self.frame(window.frame.as_ref())),
            Expression::TableSource(table) => {
                let own = if (self.statement_rows)(&table.name.value_lower) {
                    Volatility::PerStatement
                } else {
                    Volatility::Fixed
                };
                own.max(optional(table.as_of.as_ref().map(|as_of| &*as_of.value)))
            }
            Expression::JoinSource(join) => one(&join.left)
                .max(one(&join.right))
                .max(optional(join.condition.as_deref())),
            Expression::SubquerySource(source) => self.block(&source.subquery),
            Expression::ValuesSource(values) => most(values.rows.iter().map(|row| self.all(row))),
            Expression::FunctionTableSource(source) => self
                .registry
                .volatility(&source.function.value)
                .max(self.all(&source.arguments)),
        }
    }

    fn statement(&self, stmt: &SelectStatement) -> Volatility {
        let boxed = |e: &Option<Box<Expression>>| {
            e.as_deref()
                .map_or(Volatility::Fixed, |e| self.expression(e))
        };
        let SelectStatement {
            token: _,
            param_count: _,
            distinct: _,
            distinct_on,
            columns,
            with,
            table_expr,
            where_clause,
            group_by,
            having,
            window_defs,
            order_by,
            limit,
            offset,
            set_operations,
            prepared: _,
        } = stmt;
        let grouping_sets = match &group_by.modifier {
            GroupByModifier::GroupingSets(sets) => most(sets.iter().map(|set| self.all(set))),
            GroupByModifier::None | GroupByModifier::Rollup | GroupByModifier::Cube => {
                Volatility::Fixed
            }
        };
        self.all(distinct_on)
            .max(self.all(columns))
            .max(with.as_ref().map_or(Volatility::Fixed, |with| {
                most(with.ctes.iter().map(|cte| self.block(&cte.query)))
            }))
            .max(boxed(table_expr))
            .max(boxed(where_clause))
            .max(self.all(&group_by.columns))
            .max(grouping_sets)
            .max(boxed(having))
            .max(most(window_defs.iter().map(|window| {
                self.all(&window.partition_by)
                    .max(self.ordered(&window.order_by))
                    .max(self.frame(window.frame.as_ref()))
            })))
            .max(self.ordered(order_by))
            .max(boxed(limit))
            .max(boxed(offset))
            .max(most(
                set_operations.iter().map(|op| self.statement(&op.right)),
            ))
    }

    /// A block runs once per binding of what it reads: its own calls draw
    /// once per statement at most, seen from outside it
    fn block(&self, stmt: &SelectStatement) -> Volatility {
        if self.constant {
            return Volatility::PerCall;
        }
        if self.whole_blocks {
            return self.statement(stmt);
        }
        self.statement(stmt).min(Volatility::PerStatement)
    }

    fn call(&self, call: &FunctionCall) -> Volatility {
        if self.constant && self.registry.is_aggregate(&call.function) {
            return Volatility::PerCall;
        }
        self.registry
            .volatility(&call.function)
            .max(self.all(&call.arguments))
            .max(self.ordered(&call.order_by))
            .max(
                call.filter
                    .as_deref()
                    .map_or(Volatility::Fixed, |e| self.expression(e)),
            )
    }

    fn all(&self, exprs: &[Expression]) -> Volatility {
        most(exprs.iter().map(|e| self.expression(e)))
    }

    fn ordered(&self, order: &[OrderByExpression]) -> Volatility {
        most(order.iter().map(|o| self.expression(&o.expression)))
    }

    fn frame(&self, frame: Option<&WindowFrame>) -> Volatility {
        let bound = |bound: &WindowFrameBound| match bound {
            WindowFrameBound::Preceding(e) | WindowFrameBound::Following(e) => self.expression(e),
            WindowFrameBound::CurrentRow
            | WindowFrameBound::UnboundedPreceding
            | WindowFrameBound::UnboundedFollowing => Volatility::Fixed,
        };
        frame.map_or(Volatility::Fixed, |frame| {
            bound(&frame.start).max(frame.end.as_ref().map_or(Volatility::Fixed, bound))
        })
    }
}

fn most(levels: impl Iterator<Item = Volatility>) -> Volatility {
    levels.max().unwrap_or(Volatility::Fixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::Statement;
    use crate::parser::Parser;

    fn statement_volatility(stmt: &SelectStatement, registry: &FunctionRegistry) -> Volatility {
        statement_volatility_in(stmt, registry, &|_| false)
    }

    fn select(sql: &str) -> SelectStatement {
        let mut parser = Parser::new(sql);
        let program = parser.parse_program().expect("parses");
        match program.statements.into_iter().next() {
            Some(Statement::Select(stmt)) => stmt,
            other => panic!("not a SELECT: {other:?}"),
        }
    }

    #[test]
    fn test_every_position_answers_with_its_most_volatile_call() {
        let registry = FunctionRegistry::new();
        let cases = [
            ("SELECT a + 1 FROM t", Volatility::Fixed),
            (
                "SELECT ABS(-5), UPPER('x'), COALESCE(a, 1) FROM t",
                Volatility::Fixed,
            ),
            ("SELECT NOW() FROM t", Volatility::PerStatement),
            (
                "SELECT CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP FROM t",
                Volatility::PerStatement,
            ),
            (
                "SELECT CURRENT_TRANSACTION_ID() FROM t",
                Volatility::PerStatement,
            ),
            ("SELECT RANDOM() FROM t", Volatility::PerCall),
            ("SELECT NO_SUCH_FN(a) FROM t", Volatility::PerCall),
            (
                "SELECT * FROM t WHERE a < RANDOM() * 20",
                Volatility::PerCall,
            ),
            (
                "SELECT * FROM t WHERE a > NOW() AND b < RANDOM()",
                Volatility::PerCall,
            ),
            (
                "SELECT CASE WHEN a = 1 THEN RANDOM() END FROM t",
                Volatility::PerCall,
            ),
            (
                "SELECT CAST(RANDOM() AS INTEGER) FROM t",
                Volatility::PerCall,
            ),
            (
                "SELECT * FROM t WHERE a IN (1, RANDOM())",
                Volatility::PerCall,
            ),
            (
                "SELECT * FROM t WHERE a BETWEEN 0 AND RANDOM()",
                Volatility::PerCall,
            ),
            (
                "SELECT * FROM t WHERE s LIKE 'x%' ESCAPE CHR(RANDOM())",
                Volatility::PerCall,
            ),
            ("SELECT (a, RANDOM()) = (1, 2) FROM t", Volatility::PerCall),
            (
                "SELECT LAG(a, CAST(RANDOM() AS INTEGER)) OVER (ORDER BY a) FROM t",
                Volatility::PerCall,
            ),
            (
                "SELECT ROW_NUMBER() OVER (PARTITION BY RANDOM()) FROM t",
                Volatility::PerCall,
            ),
            (
                "SELECT SUM(a) FILTER (WHERE RANDOM() < 0.5) FROM t",
                Volatility::PerCall,
            ),
            (
                "SELECT a FROM t GROUP BY a HAVING SUM(a) > RANDOM()",
                Volatility::PerCall,
            ),
            ("SELECT a FROM t ORDER BY RANDOM()", Volatility::PerCall),
            (
                "SELECT a FROM t LIMIT CAST(RANDOM() * 10 AS INTEGER)",
                Volatility::PerCall,
            ),
            ("SELECT a FROM t UNION SELECT RANDOM()", Volatility::PerCall),
            (
                "WITH c AS (SELECT RANDOM() AS r) SELECT * FROM c",
                Volatility::PerStatement,
            ),
            (
                "SELECT * FROM (SELECT RANDOM() AS r) AS d",
                Volatility::PerStatement,
            ),
            (
                "SELECT * FROM t WHERE a = (SELECT RANDOM())",
                Volatility::PerStatement,
            ),
            (
                "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.v < RANDOM())",
                Volatility::PerStatement,
            ),
            (
                "SELECT * FROM t WHERE a < ALL (SELECT RANDOM())",
                Volatility::PerStatement,
            ),
            (
                "SELECT * FROM t WHERE a = (SELECT MAX(v) FROM u)",
                Volatility::Fixed,
            ),
        ];
        for (sql, expected) in cases {
            assert_eq!(
                statement_volatility(&select(sql), &registry),
                expected,
                "{sql}"
            );
        }
    }

    #[test]
    fn test_a_block_answers_for_its_own_calls() {
        let registry = FunctionRegistry::new();
        let stmt = select("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.v < RANDOM())");
        let Some(Expression::Exists(exists)) = stmt.where_clause.as_deref() else {
            panic!("EXISTS in WHERE");
        };
        assert_eq!(
            statement_volatility(&exists.subquery, &registry),
            Volatility::PerCall,
            "the block itself draws per call, though the statement runs it once"
        );
    }

    #[test]
    fn test_the_rows_of_a_statement_cte_are_one_per_statement() {
        let registry = FunctionRegistry::new();
        let stmt = select("SELECT * FROM t WHERE a IN (SELECT x FROM c)");
        let ctes = |name: &str| name == "c";
        assert_eq!(statement_volatility(&stmt, &registry), Volatility::Fixed);
        assert_eq!(
            statement_volatility_in(&stmt, &registry, &ctes),
            Volatility::PerStatement,
            "c is the running statement's CTE"
        );
    }

    #[test]
    fn test_a_statement_constant_reads_no_row() {
        let registry = FunctionRegistry::new();
        let value = |sql: &str| {
            let stmt = select(&format!("SELECT {sql} FROM t"));
            stmt.columns.into_iter().next().expect("one column")
        };
        for (sql, constant) in [
            ("1 + 2", true),
            ("$1 * 2", true),
            ("CAST('5' AS INTEGER)", true),
            ("NOW() - INTERVAL '1 day'", true),
            ("TRUE", true),
            ("a", false),
            ("\"a\"", false),
            ("CAST(\"1\" AS INTEGER)", false),
            ("t.a + 1", false),
            ("\"true\"", false),
            ("RANDOM()", false),
            ("SUM(1)", false),
            ("ROW_NUMBER() OVER ()", false),
            ("(SELECT 1)", false),
            ("EXISTS (SELECT 1)", false),
        ] {
            assert_eq!(
                statement_constant(&value(sql), &registry),
                constant,
                "{sql}"
            );
        }
    }
}
