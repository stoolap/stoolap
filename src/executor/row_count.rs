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

//! The row counts a LIMIT and an OFFSET ask for: every path that applies
//! them, or decides a shortcut by them, reads them here.

use crate::core::{Error, Result, Row, RowVec, Value};
use crate::parser::ast::{Expression, SelectStatement};

use super::context::ExecutionContext;
use super::expression::ExpressionEval;

/// The LIMIT and OFFSET of one statement, read
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowCounts {
    /// None when the statement has no LIMIT
    pub limit: Option<usize>,
    pub offset: usize,
}

impl RowCounts {
    /// Keep the rows the counts ask for, of rows already in order
    pub fn apply(self, rows: &mut RowVec) {
        let skipped = self.offset.min(rows.len());
        rows.drain(..skipped);
        if let Some(limit) = self.limit {
            rows.truncate(limit);
        }
    }
}

/// The LIMIT and OFFSET of `stmt`. A count is an INTEGER of zero or more,
/// or a FLOAT holding one; NULL, any other value and an evaluation error
/// are the statement's error
pub(crate) fn row_counts(stmt: &SelectStatement, ctx: &ExecutionContext) -> Result<RowCounts> {
    Ok(RowCounts {
        limit: stmt
            .limit
            .as_deref()
            .map(|expr| row_count(expr, ctx, "LIMIT"))
            .transpose()?,
        offset: stmt
            .offset
            .as_deref()
            .map_or(Ok(0), |expr| row_count(expr, ctx, "OFFSET"))?,
    })
}

/// LIMIT + OFFSET, the rows a read that stops early needs; None without a
/// LIMIT or when either is not a count, which the path applying them reports
pub(crate) fn rows_needed(stmt: &SelectStatement, ctx: &ExecutionContext) -> Option<usize> {
    let counts = row_counts(stmt, ctx).ok()?;
    Some(counts.limit?.saturating_add(counts.offset))
}

/// The row count `expr` asks for, as [`row_counts`] reads it
pub(crate) fn row_count(
    expr: &Expression,
    ctx: &ExecutionContext,
    clause: &'static str,
) -> Result<usize> {
    let value = match expr {
        Expression::IntegerLiteral(lit) => Value::Integer(lit.value),
        _ => ExpressionEval::compile(expr, &[])?
            .with_context(ctx)
            .eval_slice(&Row::new())?,
    };
    // 2^63, the first FLOAT above every INTEGER
    const ABOVE_INTEGERS: f64 = 9_223_372_036_854_775_808.0;
    let count = match value {
        Value::Integer(count) => count,
        Value::Float(f) if f.fract() == 0.0 && f.abs() < ABOVE_INTEGERS => f as i64,
        other => {
            return Err(Error::Parse(format!(
                "{clause} must be an integer, got {other:?}"
            )))
        }
    };
    usize::try_from(count)
        .map_err(|_| Error::Parse(format!("{clause} must be non-negative, got {count}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::Statement;
    use crate::parser::Parser;

    fn counts(sql: &str) -> Result<RowCounts> {
        let mut parser = Parser::new(sql);
        let program = parser.parse_program().expect("parses");
        let Some(Statement::Select(stmt)) = program.statements.into_iter().next() else {
            panic!("not a SELECT: {sql}");
        };
        row_counts(&stmt, &ExecutionContext::new())
    }

    #[test]
    fn test_a_count_is_a_whole_number_of_zero_or_more() {
        let ok = |sql: &str, limit: Option<usize>, offset: usize| {
            assert_eq!(counts(sql).unwrap(), RowCounts { limit, offset }, "{sql}");
        };
        ok("SELECT 1", None, 0);
        ok("SELECT 1 LIMIT 0", Some(0), 0);
        ok("SELECT 1 LIMIT 3 OFFSET 2", Some(3), 2);
        ok("SELECT 1 LIMIT 1 + 1", Some(2), 0);
        ok("SELECT 1 LIMIT 2.0 OFFSET 1.0", Some(2), 1);
        ok("SELECT 1 LIMIT -0.0", Some(0), 0);
        ok(
            "SELECT 1 LIMIT 9223372036854775807",
            Some(usize::MAX >> 1),
            0,
        );
        for sql in [
            "SELECT 1 LIMIT 2.5",
            "SELECT 1 LIMIT 2 OFFSET 1.5",
            "SELECT 1 LIMIT -1",
            "SELECT 1 LIMIT 2 OFFSET -1",
            "SELECT 1 LIMIT -0.5",
            "SELECT 1 LIMIT NULL",
            "SELECT 1 LIMIT 'x'",
            "SELECT 1 LIMIT 1e30",
            "SELECT 1 LIMIT 0.0 / 0.0",
        ] {
            assert!(counts(sql).is_err(), "{sql} is an error");
        }
    }
}
