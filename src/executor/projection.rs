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

//! One rule for a SELECT list's output row: an item that is a column of the
//! input is copied, one whose value is known before the first row is a
//! constant, and every other item is the compiler's program, run per row.

use crate::common::CompactArc;
use crate::core::{Result, Row, Value, NULL_VALUE};
use crate::parser::ast::Expression;

use super::expression::SharedProgram;
use super::expression::{
    bind_params, compile_expression, concat_values, ExecuteContext, ExprVM, Op,
};
use super::utils::for_each_qualified_star_column;

/// One output item, as its compiled program reads
pub(crate) enum Item {
    /// The input column at this index
    Column(usize),
    /// A value known before the first row
    Constant(Value),
    /// The execution's parameter at this slot
    Param(usize),
    /// Columns, constants and parameters joined by `||`
    Concat(Concat),
    /// The compiler's program for the item, over the input row
    Program(SharedProgram),
}

impl Item {
    /// The item `program` computes: a column, a constant, a parameter or a
    /// concatenation of them when that is all it does
    fn of(program: SharedProgram) -> Self {
        match program.ops() {
            [Op::LoadColumn(index), Op::Return] => Item::Column(usize::from(*index)),
            [Op::LoadConst(value), Op::Return] => Item::Constant(value.clone()),
            [Op::LoadNull(data_type), Op::Return] => Item::Constant(Value::Null(*data_type)),
            [Op::LoadParam(slot), Op::Return] => Item::Param(*slot as usize),
            _ => match Concat::of(&program) {
                Some(concat) => Item::Concat(concat),
                None => Item::Program(program),
            },
        }
    }
}

/// A value a concatenation reads: an input column, a constant or a
/// parameter
#[derive(Debug, Clone)]
enum Part {
    Column(usize),
    Constant(Value),
    Param(usize),
}

/// A program that only loads columns, constants and parameters and joins
/// them by `||`, run without the VM by the rule the VM's `||` runs
#[derive(Debug, Clone)]
pub(crate) struct Concat {
    parts: Box<[Part]>,
    /// The program, when a part is a parameter: binding it to an
    /// execution's values may fold it whole
    source: Option<SharedProgram>,
}

impl Concat {
    /// `program` as a concatenation, when that is all it does
    pub(crate) fn of(program: &SharedProgram) -> Option<Self> {
        let (Op::Return, rest) = program.ops().split_last()? else {
            return None;
        };
        let (join, loads) = rest.split_last()?;
        let count = match join {
            Op::Concat => 2,
            Op::ConcatN(n) => usize::from(*n),
            _ => return None,
        };
        if loads.len() != count {
            return None;
        }
        let parts: Box<[Part]> = loads
            .iter()
            .map(|op| match op {
                Op::LoadColumn(index) => Some(Part::Column(usize::from(*index))),
                Op::LoadConst(value) => Some(Part::Constant(value.clone())),
                Op::LoadNull(data_type) => Some(Part::Constant(Value::Null(*data_type))),
                Op::LoadParam(slot) => Some(Part::Param(*slot as usize)),
                _ => None,
            })
            .collect::<Option<_>>()?;
        let source = parts
            .iter()
            .any(|part| matches!(part, Part::Param(_)))
            .then(|| CompactArc::clone(program));
        Some(Self { parts, source })
    }

    /// The program to bind to an execution's parameters, when a part is one
    pub(crate) fn source(&self) -> Option<&SharedProgram> {
        self.source.as_ref()
    }

    /// The parts of `exec`'s row and parameters joined; a missing parameter
    /// is the error the VM's load of it raises
    #[inline]
    pub(crate) fn value(&self, exec: &ExecuteContext) -> Result<Value> {
        if self.source.is_none() {
            let row = exec.row;
            return Ok(concat_values(self.parts.iter().map(|part| match part {
                Part::Column(index) => row.get(*index).unwrap_or(&NULL_VALUE),
                Part::Constant(value) => value,
                Part::Param(_) => &NULL_VALUE,
            })));
        }
        for part in self.parts.iter() {
            if let Part::Param(slot) = part {
                exec.param_ref(*slot)?;
            }
        }
        Ok(concat_values(self.parts.iter().map(|part| match part {
            Part::Column(index) => exec.row.get(*index).unwrap_or(&NULL_VALUE),
            Part::Constant(value) => value,
            Part::Param(slot) => exec.param_ref(*slot).unwrap_or(&NULL_VALUE),
        })))
    }
}

/// A SELECT list over one input layout
pub(crate) struct Projection {
    items: Vec<Item>,
}

impl Projection {
    /// The items of `expressions` over the input `columns`, a qualified star
    /// naming `table_alias`'s columns
    pub(crate) fn compile(
        expressions: &[Expression],
        columns: &[String],
        table_alias: Option<&str>,
    ) -> Result<Self> {
        let mut items = Vec::with_capacity(expressions.len());
        for expression in expressions {
            match expression {
                Expression::Star(_) => items.extend((0..columns.len()).map(Item::Column)),
                Expression::QualifiedStar(star) => {
                    for_each_qualified_star_column(columns, &star.qualifier, table_alias, |i, _| {
                        items.push(Item::Column(i))
                    })
                }
                expression => items.push(Item::of(compile_expression(expression, columns)?)),
            }
        }
        Ok(Self { items })
    }

    /// The input columns at `indices`, in that order
    pub(crate) fn columns(indices: impl IntoIterator<Item = usize>) -> Self {
        Self {
            items: indices.into_iter().map(Item::Column).collect(),
        }
    }

    /// The programs bound to `params`; one that binds to a value is a
    /// constant for the execution
    pub(crate) fn bind(&mut self, params: &[Value]) {
        if params.is_empty() {
            return;
        }
        for item in &mut self.items {
            let program = match item {
                Item::Program(program) => program,
                Item::Concat(concat) => match concat.source() {
                    Some(source) => source,
                    None => continue,
                },
                _ => continue,
            };
            *item = Item::of(bind_params(program.clone(), params));
        }
    }

    /// Whether the output is the input row of `width` columns as it is
    pub(crate) fn is_columns_in_order(&self, width: usize) -> bool {
        self.items.len() == width
            && self
                .items
                .iter()
                .enumerate()
                .all(|(i, item)| matches!(item, Item::Column(index) if *index == i))
    }

    /// How many values an output row holds
    pub(crate) fn width(&self) -> usize {
        self.items.len()
    }

    /// Push the output values for `exec`'s row onto `out`
    #[inline]
    pub(crate) fn project_into(
        &self,
        vm: &mut ExprVM,
        exec: &ExecuteContext,
        out: &mut Row,
    ) -> Result<()> {
        for item in &self.items {
            let value = match item {
                Item::Column(index) => exec
                    .row
                    .get(*index)
                    .cloned()
                    .unwrap_or_else(Value::null_unknown),
                Item::Constant(value) => value.clone(),
                Item::Param(slot) => exec.param_ref(*slot)?.clone(),
                Item::Concat(concat) => concat.value(exec)?,
                Item::Program(program) => vm.execute_cow(program, exec)?,
            };
            out.push_inline(value);
        }
        Ok(())
    }
}
