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

//! A comparison of two columns of one row, such as `b > a`, answered by
//! `sql_compare_op`, the rule the expression VM compares by

use std::any::Any;

use super::{find_column_index, resolve_alias, Expression};
use crate::common::{KeyMap, SmartString};
use crate::core::predicate::sql_compare_op;
use crate::core::{Operator, Result, Row, Schema};

/// `left operator right`, both columns of the row
#[derive(Debug, Clone)]
pub struct ColumnPairExpr {
    left: SmartString,
    operator: Operator,
    right: SmartString,
    /// The two columns' places, once prepared
    indices: Option<(usize, usize)>,
}

impl ColumnPairExpr {
    /// `left operator right`; None for an operator other than a comparison
    pub fn new(left: &str, operator: Operator, right: &str) -> Option<Self> {
        matches!(
            operator,
            Operator::Eq
                | Operator::Ne
                | Operator::Lt
                | Operator::Lte
                | Operator::Gt
                | Operator::Gte
        )
        .then(|| Self {
            left: SmartString::from(left),
            operator,
            right: SmartString::from(right),
            indices: None,
        })
    }

    /// The two column names and the operator between them
    pub fn parts(&self) -> (&str, Operator, &str) {
        (&self.left, self.operator, &self.right)
    }

    /// The answer for `row`: None when it is UNKNOWN or unprepared
    #[inline]
    fn answer(&self, row: &Row) -> Option<bool> {
        let (left, right) = self.indices?;
        sql_compare_op(row.get(left)?, self.operator, row.get(right)?)
    }
}

impl Expression for ColumnPairExpr {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        Ok(self.answer(row).unwrap_or(false))
    }

    fn evaluate_fast(&self, row: &Row) -> bool {
        self.answer(row).unwrap_or(false)
    }

    fn can_fail(&self) -> bool {
        false
    }

    fn with_aliases(&self, aliases: &KeyMap<String, String>) -> Box<dyn Expression> {
        Box::new(Self {
            left: SmartString::from(resolve_alias(&self.left, aliases)),
            operator: self.operator,
            right: SmartString::from(resolve_alias(&self.right, aliases)),
            indices: None,
        })
    }

    fn prepare_for_schema(&mut self, schema: &Schema) {
        if self.indices.is_none() {
            self.indices =
                find_column_index(schema, &self.left).zip(find_column_index(schema, &self.right));
        }
    }

    fn is_prepared(&self) -> bool {
        self.indices.is_some()
    }

    fn collect_column_indices(&self, out: &mut Vec<usize>) -> bool {
        match self.indices {
            Some((left, right)) => {
                out.extend([left, right]);
                true
            }
            None => false,
        }
    }

    fn column_pair(&self) -> Option<(&str, Operator, &str)> {
        Some(self.parts())
    }

    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }

    fn is_unknown_due_to_null(&self, row: &Row) -> bool {
        self.indices.is_some() && self.answer(row).is_none()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
