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

//! Nested Loop Join Operator.
//!
//! This operator implements the classic nested loop join with O(N*M) complexity.
//! It's the fallback algorithm used when:
//! - No equality join keys exist (non-equi joins)
//! - Join condition uses complex expressions
//! - CROSS JOIN is requested
//!
//! Despite its higher complexity, it supports all join conditions and types.

use crate::core::value::NULL_VALUE;
use crate::core::{Result, Row, Value};
use crate::executor::expression::JoinFilter;
use crate::executor::operator::{ColumnInfo, Operator, RowRef};
use crate::executor::utils::combine_rows_with_nulls;
use crate::functions::registry::global_registry;
use crate::parser::ast::Expression;

use super::hash_join::JoinType;

/// Nested Loop Join Operator.
///
/// For each row in the outer (left) input, scans all rows in the inner (right)
/// input and emits matches based on the join condition.
pub struct NestedLoopJoinOperator {
    // Input operators
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,

    // Join configuration
    join_type: JoinType,
    condition: Option<Expression>,

    // Compiled filter (created in open())
    filter: Option<JoinFilter>,

    // Output schema
    schema: Vec<ColumnInfo>,
    left_col_count: usize,
    right_col_count: usize,

    // Materialized right side (inner loop)
    right_rows: Vec<Row>,

    // Current state
    current_left_row: Option<Row>,
    current_right_idx: usize,
    left_had_match: bool,

    // Track matched right rows for RIGHT/FULL OUTER
    right_matched: Vec<bool>,

    // Phase for returning unmatched right rows
    returning_unmatched_right: bool,
    unmatched_right_idx: usize,

    // Cached null rows for OUTER joins (avoid repeated allocation)
    cached_null_right: Vec<Value>,
    cached_null_left: Vec<Value>,

    // State tracking
    opened: bool,
    left_exhausted: bool,
}

impl NestedLoopJoinOperator {
    /// Create a new nested loop join operator.
    ///
    /// # Arguments
    /// * `left` - Left (outer) input operator
    /// * `right` - Right (inner) input operator
    /// * `join_type` - Type of join (INNER, LEFT, RIGHT, FULL, CROSS)
    /// * `condition` - Join condition (None for CROSS JOIN)
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        join_type: JoinType,
        condition: Option<Expression>,
    ) -> Self {
        // Build combined schema
        let mut schema = Vec::new();
        schema.extend(left.schema().iter().cloned());
        schema.extend(right.schema().iter().cloned());

        let left_col_count = left.schema().len();
        let right_col_count = right.schema().len();

        Self {
            left,
            right,
            join_type,
            condition,
            filter: None,
            schema,
            left_col_count,
            right_col_count,
            right_rows: Vec::new(),
            current_left_row: None,
            current_right_idx: 0,
            left_had_match: false,
            right_matched: Vec::new(),
            returning_unmatched_right: false,
            unmatched_right_idx: 0,
            cached_null_right: Vec::new(), // Initialized in open()
            cached_null_left: Vec::new(),  // Initialized in open()
            opened: false,
            left_exhausted: false,
        }
    }

    /// Create a NULL row for the right side (uses cached values).
    #[inline]
    fn null_right_row(&self) -> Row {
        Row::from_values(self.cached_null_right.clone())
    }

    /// Combine left and right rows into output row.
    #[inline]
    fn combine(&self, left: &Row, right: &Row) -> Row {
        Row::from_combined(left, right)
    }

    /// Get the next left row from the outer input.
    #[inline]
    fn advance_left(&mut self) -> Result<bool> {
        match self.left.next()? {
            Some(row_ref) => {
                self.current_left_row = Some(row_ref.into_owned());
                self.current_right_idx = 0;
                self.left_had_match = false;
                Ok(true)
            }
            None => {
                self.left_exhausted = true;
                Ok(false)
            }
        }
    }
}

impl Operator for NestedLoopJoinOperator {
    fn open(&mut self) -> Result<()> {
        // Open both inputs
        self.left.open()?;
        self.right.open()?;

        // Pre-cache null rows for OUTER joins (avoids repeated allocation)
        // NULL_VALUE is a static constant, cloning Vec is just memcpy
        if matches!(
            self.join_type,
            JoinType::Left | JoinType::Right | JoinType::Full
        ) {
            self.cached_null_right = vec![NULL_VALUE; self.right_col_count];
            self.cached_null_left = vec![NULL_VALUE; self.left_col_count];
        }

        // Build column names for filter compilation
        let left_cols: Vec<String> = self.left.schema().iter().map(|c| c.name.clone()).collect();
        let right_cols: Vec<String> = self.right.schema().iter().map(|c| c.name.clone()).collect();

        // Compile join filter if condition exists
        if let Some(ref cond) = self.condition {
            self.filter = Some(JoinFilter::new(
                cond,
                &left_cols,
                &right_cols,
                global_registry(),
            )?);
        }

        // Materialize right side (inner loop must be restarted for each left row)
        while let Some(row_ref) = self.right.next()? {
            self.right_rows.push(row_ref.into_owned());
        }

        // Initialize matched tracking for RIGHT/FULL OUTER
        if matches!(self.join_type, JoinType::Right | JoinType::Full) {
            self.right_matched = vec![false; self.right_rows.len()];
        }

        // Get first left row
        self.advance_left()?;

        self.opened = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if !self.opened {
            return Err(crate::core::Error::internal(
                "NestedLoopJoinOperator::next called before open",
            ));
        }

        let is_left_outer = matches!(self.join_type, JoinType::Left | JoinType::Full);
        let is_right_outer = matches!(self.join_type, JoinType::Right | JoinType::Full);
        let is_cross = matches!(self.join_type, JoinType::Cross);

        // Phase 2: Return unmatched right rows (for RIGHT/FULL OUTER)
        if self.returning_unmatched_right {
            while self.unmatched_right_idx < self.right_rows.len() {
                let idx = self.unmatched_right_idx;
                self.unmatched_right_idx += 1;

                if !self.right_matched[idx] {
                    let right_row = &self.right_rows[idx];
                    let combined = Row::from_values(combine_rows_with_nulls(
                        right_row,
                        self.right_col_count,
                        self.left_col_count,
                        false,
                    ));
                    return Ok(Some(RowRef::Owned(combined)));
                }
            }
            return Ok(None);
        }

        // Phase 1: Nested loop join
        loop {
            // Check if left is exhausted
            if self.left_exhausted {
                // Switch to returning unmatched right rows if needed
                if is_right_outer {
                    self.returning_unmatched_right = true;
                    self.unmatched_right_idx = 0;
                    return self.next();
                }
                return Ok(None);
            }

            let left_row = match &self.current_left_row {
                Some(row) => row,
                None => {
                    // Try to get next left row
                    if !self.advance_left()? {
                        if is_right_outer {
                            self.returning_unmatched_right = true;
                            self.unmatched_right_idx = 0;
                            return self.next();
                        }
                        return Ok(None);
                    }
                    self.current_left_row.as_ref().unwrap()
                }
            };

            // Try to find a match in right rows
            while self.current_right_idx < self.right_rows.len() {
                let right_idx = self.current_right_idx;
                self.current_right_idx += 1;

                let right_row = &self.right_rows[right_idx];

                // Check join condition
                let matches = if let Some(ref filter) = self.filter {
                    filter.matches(left_row, right_row)
                } else {
                    // CROSS JOIN or no condition
                    is_cross || self.condition.is_none()
                };

                if matches {
                    self.left_had_match = true;

                    // Mark right row as matched for OUTER joins
                    if is_right_outer {
                        self.right_matched[right_idx] = true;
                    }

                    let combined = self.combine(left_row, right_row);
                    return Ok(Some(RowRef::Owned(combined)));
                }
            }

            // Exhausted right side for current left row
            // Handle LEFT/FULL OUTER: emit left row with NULLs if no match
            if is_left_outer && !self.left_had_match {
                let left_row = self.current_left_row.take().unwrap();
                self.advance_left()?;
                let null_right = self.null_right_row();
                // Use owned variant - both rows are owned and won't be used again
                let combined = Row::from_combined_owned(left_row, null_right);
                return Ok(Some(RowRef::Owned(combined)));
            }

            // Move to next left row
            if !self.advance_left()? {
                // Left exhausted - handle unmatched right rows
                if is_right_outer {
                    self.returning_unmatched_right = true;
                    self.unmatched_right_idx = 0;
                    return self.next();
                }
                return Ok(None);
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        self.left.close()?;
        self.right.close()?;
        Ok(())
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn estimated_rows(&self) -> Option<usize> {
        let left_est = self.left.estimated_rows()?;
        let right_est = self.right.estimated_rows()?;

        Some(match self.join_type {
            JoinType::Inner => (left_est * right_est) / 10, // Assume 10% selectivity
            JoinType::Left => left_est,
            JoinType::Right => right_est,
            JoinType::Full => left_est + right_est,
            JoinType::Cross => left_est * right_est,
            JoinType::Semi => left_est.min(right_est),
            JoinType::Anti => left_est,
        })
    }

    fn name(&self) -> &str {
        match self.join_type {
            JoinType::Inner => "NestedLoop (INNER)",
            JoinType::Left => "NestedLoop (LEFT)",
            JoinType::Right => "NestedLoop (RIGHT)",
            JoinType::Full => "NestedLoop (FULL)",
            JoinType::Cross => "NestedLoop (CROSS)",
            JoinType::Semi => "NestedLoop (SEMI)",
            JoinType::Anti => "NestedLoop (ANTI)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::operator::MaterializedOperator;
    use crate::parser::ast::{Identifier, InfixExpression};
    use crate::parser::token::{Position, Token, TokenType};

    fn make_rows(data: Vec<Vec<i64>>) -> Vec<Row> {
        data.into_iter()
            .map(|vals| Row::from_values(vals.into_iter().map(Value::integer).collect()))
            .collect()
    }

    fn make_operator(data: Vec<Vec<i64>>, cols: Vec<&str>) -> Box<dyn Operator> {
        let rows = make_rows(data);
        let schema = cols.into_iter().map(ColumnInfo::new).collect();
        Box::new(MaterializedOperator::new(rows, schema))
    }

    fn collect_results(op: &mut dyn Operator) -> Result<Vec<Row>> {
        let mut results = Vec::new();
        op.open()?;
        while let Some(row_ref) = op.next()? {
            results.push(row_ref.into_owned());
        }
        op.close()?;
        Ok(results)
    }

    fn make_eq_condition(left_col: &str, right_col: &str) -> Expression {
        Expression::Infix(InfixExpression::new(
            Token::new(TokenType::Operator, "=", Position::default()),
            Box::new(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, left_col, Position::default()),
                left_col.to_string(),
            ))),
            "=".to_string(),
            Box::new(Expression::Identifier(Identifier::new(
                Token::new(TokenType::Identifier, right_col, Position::default()),
                right_col.to_string(),
            ))),
        ))
    }

    #[test]
    fn test_inner_nested_loop() {
        let left = make_operator(
            vec![vec![1, 10], vec![2, 20], vec![3, 30]],
            vec!["left_id", "value"],
        );
        let right = make_operator(vec![vec![1, 100], vec![3, 300]], vec!["right_id", "data"]);

        let condition = make_eq_condition("left_id", "right_id");

        let mut join = NestedLoopJoinOperator::new(left, right, JoinType::Inner, Some(condition));

        let results = collect_results(&mut join).unwrap();

        // Should have 2 matches: id=1 and id=3
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_cross_join() {
        let left = make_operator(vec![vec![1], vec![2]], vec!["a"]);
        let right = make_operator(vec![vec![10], vec![20]], vec!["b"]);

        let mut join = NestedLoopJoinOperator::new(left, right, JoinType::Cross, None);

        let results = collect_results(&mut join).unwrap();

        // 2 x 2 = 4 rows
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_left_nested_loop() {
        let left = make_operator(
            vec![vec![1, 10], vec![2, 20], vec![3, 30]],
            vec!["left_id", "value"],
        );
        let right = make_operator(vec![vec![1, 100]], vec!["right_id", "data"]);

        let condition = make_eq_condition("left_id", "right_id");

        let mut join = NestedLoopJoinOperator::new(left, right, JoinType::Left, Some(condition));

        let results = collect_results(&mut join).unwrap();

        // All 3 left rows preserved
        assert_eq!(results.len(), 3);

        // id=2 and id=3 should have NULLs
        let row2 = results
            .iter()
            .find(|r| r.get(0) == Some(&Value::integer(2)))
            .unwrap();
        assert!(row2.get(2).unwrap().is_null());
    }

    #[test]
    fn test_right_nested_loop() {
        let left = make_operator(vec![vec![1, 10]], vec!["left_id", "value"]);
        let right = make_operator(
            vec![vec![1, 100], vec![2, 200], vec![3, 300]],
            vec!["right_id", "data"],
        );

        let condition = make_eq_condition("left_id", "right_id");

        let mut join = NestedLoopJoinOperator::new(left, right, JoinType::Right, Some(condition));

        let results = collect_results(&mut join).unwrap();

        // All 3 right rows preserved
        assert_eq!(results.len(), 3);
    }
}
